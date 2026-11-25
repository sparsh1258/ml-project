"""The trees the service runs must answer exactly as LightGBM does."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from delivery_eta import artifact
from delivery_eta.data import load_training_orders
from delivery_eta.features import FEATURES, MISSING_TOGETHER, REQUIRED, encode
from delivery_eta.forest import CHUNK, PARTS, Forest, with_range
from delivery_eta.training import TrainConfig, blank, fit, to_forest

from .conftest import REAL_DATA

lgb = pytest.importorskip("lightgbm")


@pytest.fixture(scope="module")
def trained(small_config: TrainConfig):
    """Trained on real orders, so every kind of split appears."""
    orders, _ = load_training_orders(REAL_DATA)
    X = encode(orders)[::4]
    return fit(X, orders["minutes"].to_numpy(float)[::4], small_config), X


@pytest.fixture(scope="module")
def forest(trained) -> Forest:
    return Forest.from_boosters(trained[0])


def assert_same(forest: Forest, boosters, X: np.ndarray) -> None:
    for part, ours in zip(PARTS, forest.predict(X), strict=True):
        np.testing.assert_allclose(ours, boosters[part].predict(X), rtol=0, atol=1e-9)


def test_same_answers_on_real_orders(trained, forest: Forest) -> None:
    boosters, X = trained
    assert_same(forest, boosters, X[::5])


@pytest.mark.parametrize("names", MISSING_TOGETHER.values(), ids=MISSING_TOGETHER.keys())
def test_same_answers_when_inputs_are_unknown(trained, forest: Forest, names) -> None:
    boosters, X = trained
    assert_same(forest, boosters, blank(X[::50], names))


@pytest.mark.parametrize("names", [*[(n,) for n in FEATURES], REQUIRED, FEATURES], ids=str)
def test_same_answers_when_any_input_is_unknown(trained, forest: Forest, names) -> None:
    # The API insists on distance and hour, but the trees must not depend on that.
    boosters, X = trained
    assert_same(forest, boosters, blank(X[::100], names))


def test_same_answers_at_the_edges(trained, forest: Forest) -> None:
    boosters, X = trained
    rng = np.random.default_rng(3)
    odd = X[:400].copy()
    odd[:, 0] = rng.uniform(0, 60, 400)  # far outside the data
    odd[:, 2] = rng.uniform(0, 120, 400)
    assert_same(forest, boosters, odd)
    # Exactly on, just below and just above the values where trees split.
    for name in ("distance_km", "rider_age", "rider_rating"):
        column = FEATURES.index(name)
        splits = np.unique(forest.threshold[forest.feature == column])[:100]
        splits = splits[splits < 1e300]  # LightGBM's way of writing infinity
        assert len(splits) > 5
        for value in (splits, np.nextafter(splits, -np.inf), np.nextafter(splits, np.inf)):
            rows = X[: len(splits)].copy()
            rows[:, column] = value
            assert_same(forest, boosters, rows)


def test_every_input_and_kind_of_split_was_tried(forest: Forest) -> None:
    splits = forest.feature >= 0
    assert set(forest.feature[splits].tolist()) == set(range(len(FEATURES)))
    for kind in (forest.is_category, forest.unknown_left):
        assert kind[splits].any()
        assert not kind[splits].all()


def test_batches_match_one_at_a_time(trained, forest: Forest) -> None:
    X = trained[1][: CHUNK + 3]
    together = np.column_stack(forest.predict(X))
    alone = np.vstack([np.column_stack(forest.predict(X[i : i + 1])) for i in range(len(X))])
    np.testing.assert_array_equal(together, alone)


def test_explanations_add_up_exactly(trained, forest: Forest) -> None:
    X = trained[1][:200]
    point, _, _ = forest.predict(X)
    moved = forest.explain(X)
    assert moved.shape == (200, len(FEATURES))
    np.testing.assert_allclose(moved.sum(axis=1) + forest.average, point, atol=1e-9)


def test_saving_and_loading_changes_nothing(
    trained, forest: Forest, model_dir: Path, tmp_path: Path
) -> None:
    info = artifact.load(model_dir).info
    kept = {k: info[k] for k in ("range_adjustment", "seen", "hours_seen")}
    artifact.save(forest, kept, tmp_path / "m")
    loaded = artifact.load(tmp_path / "m").forest
    assert loaded.fingerprint() == forest.fingerprint()
    X = trained[1][:50]
    np.testing.assert_array_equal(
        np.column_stack(loaded.predict(X)), np.column_stack(forest.predict(X))
    )


def test_training_refuses_trees_that_differ(trained, monkeypatch: pytest.MonkeyPatch) -> None:
    boosters, X = trained
    assert to_forest(boosters, X).fingerprint()
    monkeypatch.setattr(Forest, "predict", lambda self, X: tuple(np.zeros(len(X)) for _ in PARTS))
    with pytest.raises(RuntimeError, match="differ from LightGBM"):
        to_forest(boosters, X)


@pytest.mark.parametrize(
    ("change", "words"),
    [
        ({"missing_type": "Zero"}, "missing type Zero"),
        ({"decision_type": "<"}, "decision type <"),
    ],
)
def test_splits_it_cannot_run_are_refused(change: dict[str, str], words: str) -> None:
    leaf = {"leaf_value": 1.0}
    split = {
        "split_feature": 0,
        "threshold": 1.0,
        "decision_type": "<=",
        "missing_type": "NaN",
        "default_left": True,
        "internal_value": 1.0,
        "left_child": leaf,
        "right_child": leaf,
    }

    class Fake:
        def dump_model(self):
            return {"tree_info": [{"tree_structure": {**split, **change}}]}

    with pytest.raises(ValueError, match=words):
        Forest.from_boosters(dict.fromkeys(PARTS, Fake()))


def test_the_range_always_contains_the_estimate() -> None:
    point, low, high = with_range(
        np.array([20.0, 20.0, 2.0]),
        np.array([15.0, 25.0, 0.5]),
        np.array([30.0, 10.0, 4.0]),
        1.0,
    )
    assert point.tolist() == [20, 20, 2]
    assert low.tolist() == [14, 20, 1]
    assert high.tolist() == [31, 20, 5]
