"""Trains, evaluates and checks the model.

Evaluation follows time: the newest days are held out, the model learns only from the days
before them, and settings are chosen on folds inside those earlier days. A model that fails
the quality gate is never saved.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from delivery_eta import artifact
from delivery_eta.data import TARGET, file_sha256, load_training_orders
from delivery_eta.features import (
    CATEGORIES,
    CATEGORY_POSITIONS,
    FEATURES,
    MISSING_TOGETHER,
    NUMERIC,
    Floats,
    encode,
)
from delivery_eta.metrics import error_metrics, range_metrics

Boosters = dict[str, lgb.Booster]


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    test_days: int
    cv_folds: int
    cv_days: int
    mask_share: float
    coverage: float
    rounds: int
    params: dict[str, Any]
    gate: dict[str, float]

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def blank(X: Floats, names: tuple[str, ...]) -> Floats:
    """A copy with these inputs unknown."""
    X = X.copy()
    X[:, [FEATURES.index(n) for n in names]] = np.nan
    return X


def with_missing_inputs(X: Floats, y: Floats, cfg: TrainConfig) -> tuple[Floats, Floats]:
    """The data plus copies of some rows with inputs blanked, so the model learns to cope."""
    rng = np.random.default_rng(cfg.seed)
    count = int(cfg.mask_share * len(X))
    Xs, ys = [X], [y]
    for names in MISSING_TOGETHER.values():
        rows = rng.choice(len(X), size=count, replace=False)
        Xs.append(blank(X[rows], names))
        ys.append(y[rows])
    return np.vstack(Xs), np.concatenate(ys)


def fit(
    X: Floats, y: Floats, cfg: TrainConfig, parts: tuple[str, ...] = artifact.PARTS
) -> Boosters:
    """The estimate (`point`) and the two ends of its likely range (`low`, `high`)."""
    tail = (1 - cfg.coverage) / 2
    objectives: dict[str, dict[str, Any]] = {
        "point": {"objective": "regression"},
        "low": {"objective": "quantile", "alpha": tail},
        "high": {"objective": "quantile", "alpha": 1 - tail},
    }
    fixed = {
        "verbosity": -1,
        "seed": cfg.seed,
        "deterministic": True,
        "force_row_wise": True,
    }
    Xa, ya = with_missing_inputs(X, y, cfg)
    boosters = {}
    for part in parts:
        data = lgb.Dataset(
            Xa, ya, feature_name=list(FEATURES), categorical_feature=CATEGORY_POSITIONS
        )
        params = {**cfg.params, **fixed, **objectives[part]}
        boosters[part] = lgb.train(params, data, num_boost_round=cfg.rounds)
    return boosters


def predict(boosters: Boosters, X: Floats, adjustment: float) -> tuple[Floats, Floats, Floats]:
    """Estimate, low and high. The range always contains the estimate."""
    point = np.asarray(boosters["point"].predict(X))
    low = np.minimum(np.asarray(boosters["low"].predict(X)) - adjustment, point)
    high = np.maximum(np.asarray(boosters["high"].predict(X)) + adjustment, point)
    return point, np.maximum(low, 1.0), high


def time_folds(dates: pd.Series, folds: int, days: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expanding window: each fold predicts the `days` days right after the ones it learns from."""
    unique = np.sort(dates.unique())
    if len(unique) <= folds * days:
        raise ValueError(f"{len(unique)} days of data is too few for {folds} folds of {days} days.")
    for k in range(folds, 0, -1):
        start = len(unique) - k * days
        window = unique[start : start + days]
        yield (
            np.flatnonzero((dates < window[0]).to_numpy()),
            np.flatnonzero(dates.isin(window).to_numpy()),
        )


def split_by_time(orders: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_test_day = np.sort(orders["order_date"].unique())[-test_days]
    newest = orders["order_date"] >= first_test_day
    return orders[~newest].reset_index(drop=True), orders[newest].reset_index(drop=True)


def cross_validate(orders: pd.DataFrame, cfg: TrainConfig) -> tuple[dict[str, float], float]:
    """Average error over the folds, and how much to widen ranges to reach the target coverage."""
    X, y = encode(orders), orders[TARGET].to_numpy(dtype=float)
    maes, misses = [], []
    for learn, check in time_folds(orders["order_date"], cfg.cv_folds, cfg.cv_days):
        boosters = fit(X[learn], y[learn], cfg)
        point, low, high = predict(boosters, X[check], 0.0)
        maes.append(error_metrics(y[check], point)["mae"])
        misses.append(np.maximum(low - y[check], y[check] - high))
    missed = np.concatenate(misses)
    level = min(1.0, np.ceil((len(missed) + 1) * cfg.coverage) / len(missed))
    adjustment = float(np.quantile(missed, level, method="higher"))
    return {"mae": float(np.mean(maes)), "mae_std": float(np.std(maes))}, adjustment


def evaluate(
    boosters: Boosters, test: pd.DataFrame, adjustment: float, average: float
) -> dict[str, Any]:
    """Everything reported about the model, all on days it never saw."""
    X, y = encode(test), test[TARGET].to_numpy(dtype=float)
    point, low, high = predict(boosters, X, adjustment)
    overall = {**error_metrics(y, point), **range_metrics(y, low, high)}
    guess = error_metrics(y, np.full_like(y, average))["mae"]

    distance = pd.cut(
        test["distance_km"], [0, 5, 10, 15, 100], labels=["0-5", "5-10", "10-15", "15+"]
    )
    groupings = {
        "traffic": test["traffic"],
        "weather": test["weather"],
        "city": test["city"],
        "festival": test["festival"],
        "vehicle": test["vehicle"],
        "distance_km": distance.astype(object),
        "rider_age": test["rider_age"].map(
            lambda a: "under 30" if a < 30 else "30 and over", na_action="ignore"
        ),
    }
    groups: dict[str, dict[str, Any]] = {}
    for name, values in groupings.items():
        groups[name] = {}
        for level in sorted(values.dropna().unique()):
            rows = (values == level).to_numpy()
            groups[name][str(level)] = {
                "orders": int(rows.sum()),
                **error_metrics(y[rows], point[rows]),
                **range_metrics(y[rows], low[rows], high[rows]),
            }

    unknown = {}
    for name, columns in MISSING_TOGETHER.items():
        p, lo, hi = predict(boosters, blank(X, columns), adjustment)
        unknown[name] = {"mae": error_metrics(y, p)["mae"], **range_metrics(y, lo, hi)}

    return {
        "orders": len(test),
        "overall": overall,
        "guess_the_average_mae": guess,
        "improvement_over_average": 1 - overall["mae"] / guess,
        "groups": groups,
        "when_unknown": unknown,
    }


def gate_failures(report: dict[str, Any], gate: dict[str, float]) -> list[str]:
    """Why this model must not be released. Empty when it may be."""
    overall = report["overall"]
    worst_name, worst = max(
        (
            (f"{group}={level}", m["mae"])
            for group, levels in report["groups"].items()
            for level, m in levels.items()
            if m["orders"] >= 200
        ),
        key=lambda item: item[1],
    )
    unassigned = report["when_unknown"]["rider_not_assigned"]["mae"]
    nothing = report["when_unknown"]["everything_optional"]["mae"]
    checks = [
        (overall["mae"] <= gate["max_mae"], f"average error {overall['mae']:.2f} min is too high"),
        (
            report["improvement_over_average"] >= gate["min_improvement_over_average"],
            f"only {report['improvement_over_average']:.0%} better than guessing the average",
        ),
        (
            worst <= gate["max_worst_group_mae"],
            f"average error for {worst_name} is {worst:.2f} min",
        ),
        (
            unassigned <= gate["max_mae_rider_not_assigned"],
            f"average error with no rider details is {unassigned:.2f} min",
        ),
        (
            nothing < report["guess_the_average_mae"],
            "worse than guessing the average when every optional input is unknown",
        ),
        (
            gate["min_coverage"] <= overall["coverage"] <= gate["max_coverage"],
            f"ranges contain the real time in {overall['coverage']:.0%} of orders",
        ),
    ]
    return [message for ok, message in checks if not ok]


def traffic_hours(orders: pd.DataFrame, at_least: int = 20) -> dict[str, list[int]]:
    """The hours each traffic level was really seen in. In this data the two go together."""
    seen = {}
    for level in CATEGORIES["traffic"]:
        hours = orders.loc[orders["traffic"] == level, "order_hour"].value_counts()
        seen[level] = sorted(int(h) for h in hours[hours >= at_least].index)
    return seen


def _rounded(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    return round(value, 4) if isinstance(value, float) else value


def train(data: Path, cfg: TrainConfig, folder: Path) -> tuple[dict[str, Any], list[str]]:
    """Evaluate on the newest days; if good enough, refit on everything and save.

    Returns the report and the reasons the model was refused (empty when it was saved).
    """
    orders, cleaning = load_training_orders(data)
    earlier, newest = split_by_time(orders, cfg.test_days)
    cv, adjustment = cross_validate(earlier, cfg)
    y = earlier[TARGET].to_numpy(dtype=float)
    candidate = fit(encode(earlier), y, cfg)
    report = evaluate(candidate, newest, adjustment, float(y.mean()))
    report = _rounded(
        {
            "data": {
                "file_sha256": file_sha256(data),
                "first_day": str(orders["order_date"].min().date()),
                "last_day": str(orders["order_date"].max().date()),
                "test_starts": str(newest["order_date"].min().date()),
                "cleaning": cleaning.as_dict(),
            },
            "cross_validation": cv,
            "test": report,
        }
    )
    failures = gate_failures(report["test"], cfg.gate)
    if failures:
        return report, failures

    final = fit(encode(orders), orders[TARGET].to_numpy(dtype=float), cfg)
    info = artifact.save(
        final,
        {
            "range_adjustment": adjustment,
            "coverage": cfg.coverage,
            "seen": {
                name: [float(orders[name].min()), float(orders[name].max())] for name in NUMERIC
            },
            "hours_seen": sorted(int(h) for h in orders["order_hour"].unique()),
            "traffic_hours_seen": traffic_hours(orders),
            "trained_on_orders": len(orders),
            "rounds": cfg.rounds,
            "params": cfg.params,
            "libraries": {
                "python": platform.python_version(),
                "lightgbm": version("lightgbm"),
                "numpy": version("numpy"),
            },
            "report": report,
        },
        folder,
    )
    report["model_version"] = info["version"]
    return report, []
