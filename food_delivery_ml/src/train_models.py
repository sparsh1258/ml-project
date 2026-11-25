"""
Model Training Module
======================
Trains all baseline regression models and returns a dictionary of
{model_name: fitted_model}.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

# Optional: tree-boosting libraries
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed — XGBoost models will be skipped.")

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("WARNING: lightgbm not installed — LightGBM models will be skipped.")

RANDOM_STATE = 42


def get_baseline_models(preprocessor) -> dict:
    """
    Return a dictionary of {name: sklearn Pipeline} for all baseline models.
    Each pipeline wraps the shared preprocessor + a regressor.
    
    KNN specifically uses Manhattan distance (p=1, metric='manhattan').
    """
    models = {}

    # 0. Dummy baseline (mean strategy)
    models["Dummy (Mean)"] = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", DummyRegressor(strategy="mean")),
    ])

    # 1. Linear Regression
    models["Linear Regression"] = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])

    # 2. Ridge Regression
    models["Ridge Regression"] = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ])

    # 3. Lasso Regression
    models["Lasso Regression"] = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Lasso(alpha=0.1, random_state=RANDOM_STATE, max_iter=5000)),
    ])

    # 4. KNN — Manhattan distance (p=1)
    #    We test K=7 as a reasonable default; tuning explores more K values.
    models["KNN (k=7, Manhattan)"] = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", KNeighborsRegressor(
            n_neighbors=7,
            weights="distance",
            metric="manhattan",
            p=1,
            n_jobs=-1,
        )),
    ])

    # 5. Decision Tree
    models["Decision Tree"] = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", DecisionTreeRegressor(
            max_depth=15,
            random_state=RANDOM_STATE,
        )),
    ])

    # 6. Random Forest
    models["Random Forest"] = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    # 7. Gradient Boosting
    models["Gradient Boosting"] = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
        )),
    ])

    # 8. XGBoost
    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            )),
        ])

    # 9. LightGBM
    if HAS_LGBM:
        models["LightGBM"] = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", LGBMRegressor(
                n_estimators=200,
                max_depth=-1,
                learning_rate=0.1,
                num_leaves=31,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )),
        ])

    return models


def train_all_models(models: dict, X_train, y_train) -> dict:
    """
    Fit every model on the training data.
    Returns the same dict with fitted pipelines.
    """
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    fitted = {}
    for name, pipeline in models.items():
        print(f"  Training {name}...")
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        print(f"    ✓ {name} trained.")

    print(f"\n  {len(fitted)} models trained successfully.")
    print("=" * 60)
    return fitted


if __name__ == "__main__":
    print("Run main.py to train all models.")
