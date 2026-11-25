"""
Preprocessing Module
=====================
Builds a scikit-learn ColumnTransformer + Pipeline so that all
transformations are fitted on training data only (no data leakage).
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

# Numerical features that go through imputation + scaling
NUMERICAL_FEATURES = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries",
    "haversine_distance_km",
    "manhattan_distance_km",
    "latitude_difference",
    "longitude_difference",
    "order_hour",
    "order_minute",
    "pickup_hour",
    "pickup_minute",
    "day_of_week",
    "is_weekend",
    "month",
    "preparation_time_min",
]

# Categorical features — one-hot encoded (low cardinality)
CATEGORICAL_FEATURES = [
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
    "order_time_period",
]

# Ordinal: Road_traffic_density has natural ordering
ORDINAL_TRAFFIC = ["Low", "Medium", "High", "Jam"]

# Target column
TARGET = "Time_taken(min)"

# Columns to drop (IDs, raw lat/lon already captured by distance features, raw date/time cols, flags)
DROP_COLUMNS = [
    "ID", "Delivery_person_ID",
    "Restaurant_latitude", "Restaurant_longitude",
    "Delivery_location_latitude", "Delivery_location_longitude",
    "Order_Date", "Time_Orderd", "Time_Order_picked",
    "invalid_coordinates",
]


def get_feature_lists():
    """Return the numerical and categorical feature lists."""
    return NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer:
      - Numerical: median imputation → standard scaling
      - Categorical: most-frequent imputation → one-hot encoding
    """
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # drop IDs, raw dates, etc.
    )
    return preprocessor


def prepare_features(df: pd.DataFrame):
    """
    Split DataFrame into X (features) and y (target).
    Drops columns that should not be used as features.
    Returns X, y DataFrames.
    """
    # Ensure target exists
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in DataFrame.")

    y = df[TARGET].copy()

    # Drop target and other non-feature columns
    cols_to_drop = [c for c in DROP_COLUMNS + [TARGET] if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    # Keep only the defined feature columns that exist
    keep_cols = [c for c in NUMERICAL_FEATURES + CATEGORICAL_FEATURES if c in X.columns]
    X = X[keep_cols]

    return X, y


def get_feature_names(preprocessor, X_sample) -> list:
    """
    Get feature names after preprocessing.
    Must be called after the preprocessor has been fitted.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback
        num_names = NUMERICAL_FEATURES
        cat_pipeline = preprocessor.named_transformers_["cat"]
        encoder = cat_pipeline.named_steps["encoder"]
        cat_names = list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        return num_names + cat_names


if __name__ == "__main__":
    from data_cleaning import clean_data
    from feature_engineering import engineer_features

    df = clean_data()
    df = engineer_features(df)
    X, y = prepare_features(df)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Features: {list(X.columns)}")

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    print(f"Transformed shape: {X_transformed.shape}")
    print(f"Feature names: {get_feature_names(preprocessor, X)}")
