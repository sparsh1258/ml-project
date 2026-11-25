"""Choosing the model: compare model types, and tune the one we ship.

Only the earlier days are used here, through the same time-ordered folds as training. The
held-out test days play no part in any choice.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from delivery_eta.data import TARGET, load_training_orders
from delivery_eta.features import CATEGORIES, NUMERIC, encode
from delivery_eta.metrics import error_metrics
from delivery_eta.training import TrainConfig, fit, split_by_time, time_folds


def _earlier_days(data: Path, cfg: TrainConfig) -> pd.DataFrame:
    orders, _ = load_training_orders(data)
    return split_by_time(orders, cfg.test_days)[0]


def _candidates(seed: int) -> dict[str, Any]:
    """Each model type with the preparation it needs, as scikit-learn pipelines."""
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import (
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor

    def prepared(model: Any) -> Any:
        return make_pipeline(
            ColumnTransformer(
                [
                    (
                        "numbers",
                        make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                        list(NUMERIC),
                    ),
                    (
                        "categories",
                        make_pipeline(
                            SimpleImputer(strategy="most_frequent"),
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                        list(CATEGORIES),
                    ),
                ]
            ),
            model,
        )

    return {
        "Guess the average": prepared(DummyRegressor()),
        "Linear regression": prepared(LinearRegression()),
        "Ridge": prepared(Ridge(alpha=1.0)),
        "Lasso": prepared(Lasso(alpha=0.1, max_iter=5000)),
        "KNN (k=15, Manhattan)": prepared(
            KNeighborsRegressor(n_neighbors=15, weights="distance", metric="manhattan", n_jobs=-1)
        ),
        "Decision tree": prepared(DecisionTreeRegressor(max_depth=12, random_state=seed)),
        "Random forest": prepared(
            RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_leaf=3, random_state=seed, n_jobs=-1
            )
        ),
        "Gradient boosting": prepared(
            GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=seed)
        ),
        "Histogram gradient boosting": prepared(
            HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=seed)
        ),
        "XGBoost": prepared(
            XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=-1,
            )
        ),
    }


def compare(data: Path, cfg: TrainConfig, out: Path) -> pd.DataFrame:
    """Average error of each model type over the time-ordered folds, best first."""
    orders = _earlier_days(data, cfg)
    table = orders.astype({name: object for name in CATEGORIES})
    X, y = encode(orders), orders[TARGET].to_numpy(dtype=float)
    folds = list(time_folds(orders["order_date"], cfg.cv_folds, cfg.cv_days))
    rows = []

    def record(name: str, scores: list[dict[str, float]], seconds: float) -> None:
        means = {k: float(np.mean([s[k] for s in scores])) for k in ("mae", "rmse", "r2")}
        rows.append({"model": name, **means, "seconds": seconds})
        print(f"  {name:38s} {means['mae']:.3f} min", flush=True)

    for name, model in _candidates(cfg.seed).items():
        start, scores = time.perf_counter(), []
        for learn, check in folds:
            model.fit(table.iloc[learn], y[learn])
            scores.append(error_metrics(y[check], np.asarray(model.predict(table.iloc[check]))))
        record(name, scores, time.perf_counter() - start)

    for name, config in (
        ("LightGBM", replace(cfg, mask_share=0.0)),
        ("LightGBM, trained for missing inputs", cfg),
    ):
        start, scores = time.perf_counter(), []
        for learn, check in folds:
            point = fit(X[learn], y[learn], config, parts=("point",))["point"]
            scores.append(error_metrics(y[check], np.asarray(point.predict(X[check]))))
        record(name, scores, time.perf_counter() - start)

    result = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.round({"mae": 3, "rmse": 3, "r2": 3, "seconds": 1}).to_csv(
        out, index=False, lineterminator="\n"
    )
    return result


def tune(data: Path, config_path: Path, trials: int) -> tuple[float, float]:
    """Search LightGBM settings and write them to the config if they beat the current ones.

    Returns the average error before and after.
    """
    import optuna

    cfg = TrainConfig.load(config_path)
    orders = _earlier_days(data, cfg)
    X, y = encode(orders), orders[TARGET].to_numpy(dtype=float)
    folds = list(time_folds(orders["order_date"], cfg.cv_folds, cfg.cv_days))

    def score(config: TrainConfig) -> float:
        maes = []
        for learn, check in folds:
            point = fit(X[learn], y[learn], config, parts=("point",))["point"]
            maes.append(error_metrics(y[check], np.asarray(point.predict(X[check])))["mae"])
        return float(np.mean(maes))

    def proposal(trial: Any) -> TrainConfig:
        return replace(
            cfg,
            rounds=trial.suggest_int("rounds", 200, 900, step=50),
            params={
                "learning_rate": round(
                    trial.suggest_float("learning_rate", 0.02, 0.12, log=True), 4
                ),
                "num_leaves": trial.suggest_int("num_leaves", 24, 160, log=True),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 120, log=True),
                "feature_fraction": round(trial.suggest_float("feature_fraction", 0.5, 1.0), 2),
                "bagging_fraction": round(trial.suggest_float("bagging_fraction", 0.6, 1.0), 2),
                "bagging_freq": 1,
                "lambda_l2": round(trial.suggest_float("lambda_l2", 0.01, 20.0, log=True), 3),
                "cat_smooth": round(trial.suggest_float("cat_smooth", 1.0, 60.0, log=True), 1),
                "cat_l2": round(trial.suggest_float("cat_l2", 0.5, 40.0, log=True), 1),
            },
        )

    before = score(cfg)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=cfg.seed))
    study.optimize(
        lambda trial: score(proposal(trial)),
        n_trials=trials,
        callbacks=[
            lambda s, t: print(
                f"  trial {t.number + 1}: {t.value:.4f}  best {s.best_value:.4f}", flush=True
            )
        ],
    )
    best = proposal(study.best_trial)
    after = min(before, float(study.best_value))
    if study.best_value < before:
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        saved.update(rounds=best.rounds, params=best.params)
        config_path.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8", newline="\n")
    return before, after
