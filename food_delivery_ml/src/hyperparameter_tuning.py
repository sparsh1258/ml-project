"""
Hyperparameter Tuning Module
==============================
Uses RandomizedSearchCV to tune the strongest models:
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - KNN (multiple K values, uniform vs distance weighting)

All tuning uses cross-validation on training data only.
"""

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

RANDOM_STATE = 42


def tune_knn(preprocessor, X_train, y_train, cv=3):
    """
    Tune KNN with Manhattan distance:
      - K ∈ {3, 5, 7, 9, 11, 15, 21}
      - weights ∈ {uniform, distance}
    """
    print("  Tuning KNN (Manhattan distance)...")
    param_grid = {
        "regressor__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "regressor__weights": ["uniform", "distance"],
    }
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", KNeighborsRegressor(metric="manhattan", p=1, n_jobs=-1)),
    ])
    search = RandomizedSearchCV(
        pipeline, param_grid,
        n_iter=14,  # exhaustive for 7×2=14 combos
        scoring="neg_mean_absolute_error",
        cv=cv, random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    reg = best.named_steps["regressor"]
    print(f"    Best KNN: k={reg.n_neighbors}, weights={reg.weights}, "
          f"MAE={-search.best_score_:.4f}")
    return best, search


def tune_random_forest(preprocessor, X_train, y_train, cv=3, n_iter=40):
    """Tune Random Forest."""
    print("  Tuning Random Forest...")
    param_dist = {
        "regressor__n_estimators": randint(100, 500),
        "regressor__max_depth": [8, 10, 12, 15, 20, None],
        "regressor__min_samples_split": randint(2, 20),
        "regressor__min_samples_leaf": randint(1, 10),
        "regressor__max_features": ["sqrt", "log2", 0.5, 0.7, 1.0],
    }
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    search = RandomizedSearchCV(
        pipeline, param_dist,
        n_iter=n_iter, scoring="neg_mean_absolute_error",
        cv=cv, random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"    Best RF MAE: {-search.best_score_:.4f}")
    return search.best_estimator_, search


def tune_gradient_boosting(preprocessor, X_train, y_train, cv=3, n_iter=40):
    """Tune Gradient Boosting."""
    print("  Tuning Gradient Boosting...")
    param_dist = {
        "regressor__n_estimators": randint(100, 500),
        "regressor__learning_rate": uniform(0.01, 0.19),
        "regressor__max_depth": randint(3, 10),
        "regressor__min_samples_split": randint(2, 20),
        "regressor__min_samples_leaf": randint(1, 10),
        "regressor__subsample": uniform(0.6, 0.4),
    }
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(random_state=RANDOM_STATE)),
    ])
    search = RandomizedSearchCV(
        pipeline, param_dist,
        n_iter=n_iter, scoring="neg_mean_absolute_error",
        cv=cv, random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"    Best GB MAE: {-search.best_score_:.4f}")
    return search.best_estimator_, search


def tune_xgboost(preprocessor, X_train, y_train, cv=3, n_iter=50):
    """Tune XGBoost."""
    if not HAS_XGB:
        print("  Skipping XGBoost tuning (not installed).")
        return None, None

    print("  Tuning XGBoost...")
    param_dist = {
        "regressor__n_estimators": randint(100, 600),
        "regressor__learning_rate": uniform(0.01, 0.19),
        "regressor__max_depth": randint(3, 12),
        "regressor__min_child_weight": randint(1, 10),
        "regressor__subsample": uniform(0.6, 0.4),
        "regressor__colsample_bytree": uniform(0.5, 0.5),
        "regressor__reg_alpha": uniform(0, 1.0),
        "regressor__reg_lambda": uniform(0.5, 2.0),
    }
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        )),
    ])
    search = RandomizedSearchCV(
        pipeline, param_dist,
        n_iter=n_iter, scoring="neg_mean_absolute_error",
        cv=cv, random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"    Best XGB MAE: {-search.best_score_:.4f}")
    return search.best_estimator_, search


def tune_lightgbm(preprocessor, X_train, y_train, cv=3, n_iter=50):
    """Tune LightGBM."""
    if not HAS_LGBM:
        print("  Skipping LightGBM tuning (not installed).")
        return None, None

    print("  Tuning LightGBM...")
    param_dist = {
        "regressor__n_estimators": randint(100, 600),
        "regressor__learning_rate": uniform(0.01, 0.19),
        "regressor__num_leaves": randint(15, 127),
        "regressor__max_depth": [-1, 5, 8, 10, 15, 20],
        "regressor__min_child_samples": randint(5, 50),
        "regressor__subsample": uniform(0.6, 0.4),
        "regressor__colsample_bytree": uniform(0.5, 0.5),
        "regressor__reg_alpha": uniform(0, 1.0),
        "regressor__reg_lambda": uniform(0.5, 2.0),
    }
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LGBMRegressor(
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )),
    ])
    search = RandomizedSearchCV(
        pipeline, param_dist,
        n_iter=n_iter, scoring="neg_mean_absolute_error",
        cv=cv, random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"    Best LGBM MAE: {-search.best_score_:.4f}")
    return search.best_estimator_, search


def run_all_tuning(preprocessor, X_train, y_train, cv=3):
    """
    Run hyperparameter tuning for all supported models.
    Returns dict of {name: fitted_pipeline}.
    """
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING")
    print("=" * 60)

    tuned_models = {}

    # KNN
    best_knn, _ = tune_knn(preprocessor, X_train, y_train, cv=cv)
    tuned_models["Tuned KNN (Manhattan)"] = best_knn

    # Random Forest
    best_rf, _ = tune_random_forest(preprocessor, X_train, y_train, cv=cv)
    tuned_models["Tuned Random Forest"] = best_rf

    # Gradient Boosting
    best_gb, _ = tune_gradient_boosting(preprocessor, X_train, y_train, cv=cv)
    tuned_models["Tuned Gradient Boosting"] = best_gb

    # XGBoost
    best_xgb, _ = tune_xgboost(preprocessor, X_train, y_train, cv=cv)
    if best_xgb is not None:
        tuned_models["Tuned XGBoost"] = best_xgb

    # LightGBM
    best_lgbm, _ = tune_lightgbm(preprocessor, X_train, y_train, cv=cv)
    if best_lgbm is not None:
        tuned_models["Tuned LightGBM"] = best_lgbm

    print(f"\n  {len(tuned_models)} tuned models ready.")
    print("=" * 60)
    return tuned_models


if __name__ == "__main__":
    print("Run main.py to perform hyperparameter tuning.")
