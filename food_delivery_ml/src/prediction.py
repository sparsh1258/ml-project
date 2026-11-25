"""
Prediction Module
==================
Provides a clean prediction function that:
  - Accepts a dictionary or DataFrame of input features
  - Applies the same preprocessing used during training
  - Returns predicted delivery duration in minutes
  - Handles invalid or missing inputs
  - Never requires the actual delivery duration as input
"""

import os
import numpy as np
import pandas as pd
import joblib
from .feature_engineering import haversine_distance, manhattan_distance_geo

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_delivery_model.pkl")


def _ensure_dataframe(input_data) -> pd.DataFrame:
    """Convert dict or Series to single-row DataFrame."""
    if isinstance(input_data, dict):
        return pd.DataFrame([input_data])
    elif isinstance(input_data, pd.Series):
        return pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        return input_data.copy()
    else:
        raise TypeError(f"Expected dict, Series, or DataFrame, got {type(input_data)}")


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the same engineered features used during training.
    This replicates the feature engineering done in the training pipeline
    so the prediction function is self-contained.
    """
    # Strip whitespace from string columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Clean weather conditions prefix
    if "Weatherconditions" in df.columns:
        df["Weatherconditions"] = (
            df["Weatherconditions"]
            .astype(str)
            .str.replace("conditions ", "", regex=False)
            .str.strip()
        )

    # Numeric conversions
    for col in ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Distance features
    if all(c in df.columns for c in [
        "Restaurant_latitude", "Restaurant_longitude",
        "Delivery_location_latitude", "Delivery_location_longitude"
    ]):
        df["haversine_distance_km"] = haversine_distance(
            df["Restaurant_latitude"], df["Restaurant_longitude"],
            df["Delivery_location_latitude"], df["Delivery_location_longitude"],
        )
        df["manhattan_distance_km"] = manhattan_distance_geo(
            df["Restaurant_latitude"], df["Restaurant_longitude"],
            df["Delivery_location_latitude"], df["Delivery_location_longitude"],
        )
        df["latitude_difference"] = (
            df["Delivery_location_latitude"] - df["Restaurant_latitude"]
        )
        df["longitude_difference"] = (
            df["Delivery_location_longitude"] - df["Restaurant_longitude"]
        )

    # Time features
    if "Order_Date" in df.columns:
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], dayfirst=True, errors="coerce")
        df["day_of_week"] = df["Order_Date"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["month"] = df["Order_Date"].dt.month

    if "Time_Orderd" in df.columns:
        time_ord = pd.to_datetime(df["Time_Orderd"], format="%H:%M:%S", errors="coerce")
        df["order_hour"] = time_ord.dt.hour
        df["order_minute"] = time_ord.dt.minute
    else:
        df["order_hour"] = np.nan
        df["order_minute"] = np.nan

    if "Time_Order_picked" in df.columns:
        time_pick = pd.to_datetime(df["Time_Order_picked"], format="%H:%M:%S", errors="coerce")
        df["pickup_hour"] = time_pick.dt.hour
        df["pickup_minute"] = time_pick.dt.minute
    else:
        df["pickup_hour"] = np.nan
        df["pickup_minute"] = np.nan

    # Preparation time
    if "order_hour" in df.columns and "pickup_hour" in df.columns:
        order_min = df["order_hour"] * 60 + df["order_minute"]
        pickup_min = df["pickup_hour"] * 60 + df["pickup_minute"]
        prep = pickup_min - order_min
        prep = prep.where(prep >= 0, prep + 1440)
        prep = prep.where(prep <= 120, np.nan)
        df["preparation_time_min"] = prep

    # Time period
    if "order_hour" in df.columns:
        df["order_time_period"] = pd.cut(
            df["order_hour"],
            bins=[-1, 6, 12, 17, 21, 24],
            labels=["night", "morning", "afternoon", "evening", "night_late"],
            ordered=False,
        )
        df["order_time_period"] = df["order_time_period"].astype(str).replace("night_late", "night")

    # Replace 'NaN'/'nan' strings with actual NaN
    df.replace({"NaN": np.nan, "nan": np.nan}, inplace=True)

    # Festival: convert string
    if "Festival" in df.columns:
        df["Festival"] = df["Festival"].astype(str).str.strip()
        df["Festival"].replace({"NaN": np.nan, "nan": np.nan}, inplace=True)

    return df


def predict_delivery_time(input_data, model_path: str = MODEL_PATH) -> np.ndarray:
    """
    Predict delivery time in minutes.

    Parameters
    ----------
    input_data : dict, pd.Series, or pd.DataFrame
        Input features. Required columns:
          - Restaurant_latitude, Restaurant_longitude
          - Delivery_location_latitude, Delivery_location_longitude
          - Order_Date (dd-mm-yyyy), Time_Orderd (HH:MM:SS), Time_Order_picked (HH:MM:SS)
          - Delivery_person_Age, Delivery_person_Ratings
          - Weatherconditions, Road_traffic_density
          - Vehicle_condition, Type_of_order, Type_of_vehicle
          - multiple_deliveries, Festival, City

    model_path : str
        Path to the saved model (joblib .pkl).

    Returns
    -------
    np.ndarray
        Predicted delivery times in minutes.
    """
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Run main.py first to train and save the model."
        )
    model = joblib.load(model_path)

    # Prepare features
    df = _ensure_dataframe(input_data)
    df = _add_engineered_features(df)

    # Predict
    predictions = model.predict(df)
    return predictions


if __name__ == "__main__":
    # Example usage
    sample = {
        "Delivery_person_Age": 30,
        "Delivery_person_Ratings": 4.5,
        "Restaurant_latitude": 22.745,
        "Restaurant_longitude": 75.892,
        "Delivery_location_latitude": 22.765,
        "Delivery_location_longitude": 75.912,
        "Order_Date": "19-03-2022",
        "Time_Orderd": "11:30:00",
        "Time_Order_picked": "11:45:00",
        "Weatherconditions": "Sunny",
        "Road_traffic_density": "High",
        "Vehicle_condition": 2,
        "Type_of_order": "Snack",
        "Type_of_vehicle": "motorcycle",
        "multiple_deliveries": 0,
        "Festival": "No",
        "City": "Urban",
    }
    try:
        result = predict_delivery_time(sample)
        print(f"Predicted delivery time: {result[0]:.1f} minutes")
    except FileNotFoundError as e:
        print(e)
