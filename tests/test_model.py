from __future__ import annotations

import copy
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from delivery_eta import artifact
from delivery_eta.batch import predict_file
from delivery_eta.cli import main
from delivery_eta.data import load_orders
from delivery_eta.features import FEATURES, OPTIONAL, encode, haversine_km
from delivery_eta.metrics import error_metrics, range_metrics
from delivery_eta.predictor import Predictor
from delivery_eta.training import (
    TrainConfig,
    gate_failures,
    split_by_time,
    time_folds,
    train,
    with_missing_inputs,
)

from .conftest import an_order, raw_row


@pytest.fixture(scope="module")
def predictor(model_dir: Path) -> Predictor:
    return Predictor.load(model_dir)


# --- Inputs -----------------------------------------------------------------------------


def test_distance_matches_a_known_one() -> None:
    # Delhi to Mumbai is about 1,150 km in a straight line.
    assert haversine_km(28.6139, 77.2090, 19.0760, 72.8777) == pytest.approx(1150, rel=0.01)


def test_unknown_values_become_nan() -> None:
    X = encode({name: [None] if name in OPTIONAL else [1.0] for name in FEATURES})
    assert X.shape == (1, len(FEATURES))
    assert np.isnan(X[0, [FEATURES.index(n) for n in OPTIONAL]]).all()


def test_training_and_the_api_see_the_same_numbers(tmp_path: Path) -> None:
    """The bug this project was restructured to prevent."""
    path = tmp_path / "one.csv"
    pd.DataFrame([raw_row()]).to_csv(path, index=False)
    cleaned, _ = load_orders(path, with_target=True)
    distance = float(cleaned["distance_km"].iloc[0])
    order = an_order(distance_km=distance, order_hour=19)
    from_api = encode({name: [order[name]] for name in FEATURES})  # type: ignore[literal-required]
    np.testing.assert_array_equal(encode(cleaned), from_api)


# --- Training ---------------------------------------------------------------------------


def test_folds_never_learn_from_the_future() -> None:
    dates = pd.Series(pd.date_range("2022-01-01", periods=30)).repeat(3).reset_index(drop=True)
    folds = list(time_folds(dates, folds=4, days=5))
    assert len(folds) == 4
    for learn, check in folds:
        assert dates[learn].max() < dates[check].min()
        assert dates[check].nunique() == 5
    with pytest.raises(ValueError, match="too few"):
        list(time_folds(dates[:30], folds=4, days=5))


def test_the_newest_days_are_held_out() -> None:
    orders = pd.DataFrame({"order_date": pd.date_range("2022-01-01", periods=20)})
    earlier, newest = split_by_time(orders, 5)
    assert len(earlier) == 15
    assert len(newest) == 5
    assert earlier["order_date"].max() < newest["order_date"].min()


def test_rows_with_missing_inputs_are_added_not_swapped(small_config: TrainConfig) -> None:
    X, y = np.ones((100, len(FEATURES))), np.arange(100.0)
    Xa, ya = with_missing_inputs(X, y, small_config)
    assert len(Xa) == len(ya) == 200
    np.testing.assert_array_equal(Xa[:100], X)
    required = [FEATURES.index(n) for n in FEATURES if n not in OPTIONAL]
    assert not np.isnan(Xa[:, required]).any()
    assert np.isnan(Xa[100:]).any(axis=1).all()


def test_training_twice_gives_the_same_model(
    small_data: Path, small_config: TrainConfig, model_dir: Path, tmp_path: Path
) -> None:
    report, _ = train(small_data, small_config, tmp_path / "again")
    assert report["model_version"] == artifact.load(model_dir).version


def test_a_bad_model_is_never_saved(
    small_data: Path, small_config: TrainConfig, tmp_path: Path
) -> None:
    strict = replace(small_config, gate={**small_config.gate, "max_mae": 0.1})
    _, failures = train(small_data, strict, tmp_path / "model")
    assert any("too high" in f for f in failures)
    assert not (tmp_path / "model").exists()


@pytest.mark.parametrize(
    ("break_it", "reason"),
    [
        (lambda r: r["overall"].update(mae=9.0), "too high"),
        (lambda r: r.update(improvement_over_average=0.1), "guessing the average"),
        (lambda r: r["overall"].update(coverage=0.5), "ranges contain"),
        (lambda r: r["when_unknown"]["rider_not_assigned"].update(mae=9.0), "no rider details"),
        (
            lambda r: r["when_unknown"]["everything_optional"].update(mae=99.0),
            "every optional input",
        ),
        (lambda r: r["groups"]["traffic"]["jam"].update(mae=9.0, orders=500), "traffic=jam"),
    ],
)
def test_the_gate_catches_each_kind_of_failure(
    model_dir: Path, small_config: TrainConfig, break_it, reason: str
) -> None:
    report = copy.deepcopy(artifact.load(model_dir).info["report"]["test"])
    assert gate_failures(report, small_config.gate) == []
    break_it(report)
    failures = gate_failures(report, small_config.gate)
    assert len(failures) == 1
    assert reason in failures[0]


def test_metrics_on_a_worked_example() -> None:
    actual, predicted = np.array([10.0, 20.0, 30.0]), np.array([12.0, 20.0, 24.0])
    m = error_metrics(actual, predicted)
    assert m["mae"] == pytest.approx(8 / 3)
    assert m["bias"] == pytest.approx(-4 / 3)
    assert m["within_5_min"] == pytest.approx(2 / 3)
    r = range_metrics(actual, np.array([9.0, 21.0, 20.0]), np.array([11.0, 22.0, 30.0]))
    assert r["coverage"] == pytest.approx(2 / 3)


# --- The saved model --------------------------------------------------------------------


@pytest.fixture
def copied_model(model_dir: Path, tmp_path: Path) -> Path:
    return Path(shutil.copytree(model_dir, tmp_path / "model"))


def test_a_changed_model_file_is_refused(copied_model: Path) -> None:
    path = copied_model / "forest.npz"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(artifact.ArtifactError, match="fingerprint"):
        artifact.load(copied_model)


def test_a_model_for_other_inputs_is_refused(copied_model: Path) -> None:
    path = copied_model / "model.json"
    info = json.loads(path.read_text())
    info["features"] = info["features"][1:]
    path.write_text(json.dumps(info))
    with pytest.raises(artifact.ArtifactError, match="different inputs"):
        artifact.load(copied_model)


def test_a_missing_or_unreadable_model_is_explained(copied_model: Path, tmp_path: Path) -> None:
    with pytest.raises(artifact.ArtifactError, match="delivery-eta train"):
        artifact.load(tmp_path / "nowhere")
    (copied_model / "model.json").write_text("{")
    with pytest.raises(artifact.ArtifactError, match="can't be read"):
        artifact.load(copied_model)


# --- Estimates --------------------------------------------------------------------------


def test_estimates_follow_what_the_data_says(predictor: Predictor) -> None:
    near, far, calm, older = (
        e.minutes
        for e in predictor.predict(
            [
                an_order(distance_km=2),
                an_order(distance_km=12),
                an_order(traffic="low", order_hour=9),
                an_order(rider_age=35),
            ]
        )
    )
    usual = predictor.predict([an_order()])[0].minutes
    assert near < usual < far
    assert calm < usual
    assert older > usual


def test_the_range_contains_the_estimate_and_widens_with_less_known(predictor: Predictor) -> None:
    known, unknown = predictor.predict([an_order(), an_order(**dict.fromkeys(OPTIONAL))])
    for e in (known, unknown):
        assert 1 <= e.low <= e.minutes <= e.high
    assert unknown.high - unknown.low > known.high - known.low
    assert known.missing == []
    assert unknown.missing == list(OPTIONAL)


def test_one_at_a_time_matches_all_together(predictor: Predictor) -> None:
    orders = [an_order(distance_km=d) for d in (2, 5, 9)] + [an_order(weather=None)]
    assert predictor.predict(orders) == [predictor.predict([o])[0] for o in orders]
    assert predictor.predict([]) == []


@pytest.mark.parametrize(
    ("change", "words"),
    [
        ({"distance_km": 45.0}, "distance_km 45 is outside"),
        ({"rider_age": 60.0}, "rider_age 60 is outside"),
        ({"order_hour": 4}, "no orders between 4:00"),
        ({"order_hour": 9}, "hardly seen jam traffic between 9:00"),
    ],
)
def test_unusual_orders_get_a_warning(predictor: Predictor, change: dict, words: str) -> None:
    assert predictor.predict([an_order()])[0].warnings == []
    (warning,) = predictor.predict([an_order(**change)])[0].warnings
    assert words in warning


def test_factors_add_up_to_the_estimate(predictor: Predictor) -> None:
    average = predictor.predict([an_order()], explain=True)[0]
    other = predictor.predict([an_order(distance_km=12)], explain=True)[0]
    assert average.factors is not None
    assert other.factors is not None
    assert {f.input for f in average.factors} == set(FEATURES)
    difference = sum(f.minutes for f in other.factors) - sum(f.minutes for f in average.factors)
    assert difference == pytest.approx(other.minutes - average.minutes, abs=1.0)
    sizes = [abs(f.minutes) for f in other.factors]
    assert sizes == sorted(sizes, reverse=True)


# --- Files and the command line ---------------------------------------------------------


def test_predicting_a_file_keeps_every_row(model_dir: Path, tmp_path: Path) -> None:
    source, target = tmp_path / "in.csv", tmp_path / "out" / "out.csv"
    rows = [raw_row(), raw_row(ID="0x2", Restaurant_latitude="0")]
    pd.DataFrame(rows).drop(columns="Time_taken(min)").to_csv(source, index=False)
    assert predict_file(source, target, model_dir) == (1, 1)
    out = pd.read_csv(target)
    assert out["order_id"].tolist() == ["0x1", "0x2"]
    assert out["minutes"].notna().tolist() == [True, False]
    assert out["problem"].tolist()[1] == "no location recorded"


def test_files_and_the_api_agree(model_dir: Path, predictor: Predictor, tmp_path: Path) -> None:
    source, target = tmp_path / "in.csv", tmp_path / "out.csv"
    pd.DataFrame([raw_row()]).to_csv(source, index=False)
    predict_file(source, target, model_dir)
    cleaned, _ = load_orders(source, with_target=False)
    order = an_order(distance_km=float(cleaned["distance_km"].iloc[0]), order_hour=19)
    assert pd.read_csv(target)["minutes"].iloc[0] == predictor.predict([order])[0].minutes


def test_command_line(
    small_data: Path, model_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(["predict", str(small_data), str(tmp_path / "o.csv"), "--model", str(model_dir)]) == 0
    )
    assert "Predicted 4,000 orders" in capsys.readouterr().out
    assert main(["predict", "none.csv", "o.csv", "--model", str(model_dir)]) == 1
    assert "No file at" in capsys.readouterr().err


def test_command_line_trains_and_reports(
    small_data: Path, small_config: TrainConfig, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, report = tmp_path / "config.json", tmp_path / "reports" / "metrics.json"
    config.write_text(json.dumps(asdict(small_config)))
    command = ["train", "--data", str(small_data), "--config", str(config)]
    command += ["--model", str(tmp_path / "m"), "--report", str(report)]
    assert main(command) == 0
    version = artifact.load(tmp_path / "m").version
    assert version in capsys.readouterr().out
    assert json.loads(report.read_text())["model_version"] == version


def test_tuning_only_changes_the_config_for_something_better(
    small_data: Path, small_config: TrainConfig, tmp_path: Path
) -> None:
    pytest.importorskip("optuna")
    from delivery_eta.experiments import tune

    config = tmp_path / "config.json"
    poor = replace(small_config, rounds=3, params={"num_leaves": 4, "learning_rate": 0.01})
    config.write_text(json.dumps(asdict(poor)))
    before, after = tune(small_data, config, trials=2)
    assert after < before
    better = TrainConfig.load(config)
    assert better.rounds != poor.rounds
    assert better.gate == poor.gate


def test_comparing_models_ranks_them(
    small_data: Path, small_config: TrainConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("xgboost")
    from delivery_eta import experiments

    every = experiments._candidates(1)
    assert len(every) == 10
    quick = {name: every[name] for name in ("Guess the average", "Linear regression")}
    monkeypatch.setattr(experiments, "_candidates", lambda seed: quick)
    result = experiments.compare(small_data, small_config, tmp_path / "r" / "c.csv")
    assert (tmp_path / "r" / "c.csv").is_file()
    assert result["mae"].is_monotonic_increasing
    assert result["model"].iloc[-1] == "Guess the average"
    assert len(result) == 4


def test_command_line_refuses_a_bad_model(
    small_data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.json"
    real = json.loads((Path(__file__).parent.parent / "configs" / "train.json").read_text())
    real.update(rounds=5, cv_folds=1, params={"num_leaves": 4})
    config.write_text(json.dumps(real))
    command = ["train", "--data", str(small_data), "--config", str(config)]
    assert main([*command, "--model", str(tmp_path / "m")]) == 1
    assert "quality gate" in capsys.readouterr().err
    assert not (tmp_path / "m").exists()
