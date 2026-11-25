"""
Feature Engineering Module
===========================
Creates meaningful features from the cleaned dataset:
  - Geographical distance (Haversine and Manhattan)
  - Date/time features (hour, day of week, weekend, time period)
  - Preparation time between order and pickup
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# A. Geographical Distance Features
# ---------------------------------------------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle (Haversine) distance between two points
    on Earth given their latitude and longitude in decimal degrees.
    
    Returns distance in kilometres.
    """
    R = 6371.0  # Earth radius in km

    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def manhattan_distance_geo(lat1, lon1, lat2, lon2):
    """
    Approximate Manhattan / city-block distance using geographic coordinates.
    
    This is NOT actual road-network distance — it is a coordinate-based
    approximation that accounts for the fact that 1° longitude varies with
    latitude (shrinks toward the poles).
    
    Uses the midpoint latitude to scale the longitude difference.
    
    Returns distance in kilometres.
    """
    R = 6371.0  # km
    # Midpoint latitude for longitude scaling
    mid_lat = np.radians((lat1 + lat2) / 2)

    # 1 degree of latitude ≈ R × π/180 km
    # 1 degree of longitude ≈ R × π/180 × cos(mid_lat) km
    km_per_deg_lat = R * np.pi / 180
    km_per_deg_lon = R * np.pi / 180 * np.cos(mid_lat)

    lat_dist = np.abs(lat2 - lat1) * km_per_deg_lat
    lon_dist = np.abs(lon2 - lon1) * km_per_deg_lon

    return lat_dist + lon_dist


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add haversine and Manhattan distance features."""

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
    return df


# ---------------------------------------------------------------------------
# B. Date and Time Features
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features:
      - order_hour, order_minute
      - pickup_hour, pickup_minute
      - day_of_week (0=Mon, 6=Sun)
      - is_weekend
      - month
      - order_time_period (morning/afternoon/evening/night)
      - preparation_time_min (minutes between order and pickup)
    """
    # Order time features
    df["order_hour"] = df["Time_Orderd"].dt.hour
    df["order_minute"] = df["Time_Orderd"].dt.minute

    # Pickup time features
    df["pickup_hour"] = df["Time_Order_picked"].dt.hour
    df["pickup_minute"] = df["Time_Order_picked"].dt.minute

    # Date features
    df["day_of_week"] = df["Order_Date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["Order_Date"].dt.month

    # Time period
    df["order_time_period"] = pd.cut(
        df["order_hour"],
        bins=[-1, 6, 12, 17, 21, 24],
        labels=["night", "morning", "afternoon", "evening", "night_late"],
        ordered=False,
    )
    # Merge night_late into night
    df["order_time_period"] = df["order_time_period"].astype(str).replace("night_late", "night")

    # Preparation time (minutes between order placed and order picked up)
    # Handle midnight crossover: if pickup time < order time, add 24 hours
    if df["Time_Orderd"].notna().any() and df["Time_Order_picked"].notna().any():
        order_minutes = df["Time_Orderd"].dt.hour * 60 + df["Time_Orderd"].dt.minute
        pickup_minutes = df["Time_Order_picked"].dt.hour * 60 + df["Time_Order_picked"].dt.minute

        prep_time = pickup_minutes - order_minutes
        # Handle midnight crossover (e.g., ordered at 23:50, picked at 00:05)
        prep_time = prep_time.where(prep_time >= 0, prep_time + 1440)  # +24h in minutes

        # Cap unreasonable preparation times (> 120 min) as NaN
        prep_time = prep_time.where(prep_time <= 120, np.nan)

        df["preparation_time_min"] = prep_time

    return df


# ---------------------------------------------------------------------------
# C. Full Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps.
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    print("\n1. Adding distance features...")
    df = add_distance_features(df)
    print(f"   haversine_distance_km: mean={df['haversine_distance_km'].mean():.2f} km")
    print(f"   manhattan_distance_km: mean={df['manhattan_distance_km'].mean():.2f} km")

    print("2. Adding time features...")
    df = add_time_features(df)

    # For rows with invalid coordinates, set distance features to NaN
    if "invalid_coordinates" in df.columns:
        invalid_mask = df["invalid_coordinates"]
        dist_cols = [
            "haversine_distance_km", "manhattan_distance_km",
            "latitude_difference", "longitude_difference",
        ]
        for col in dist_cols:
            df.loc[invalid_mask, col] = np.nan
        print(f"   Set distance features to NaN for {invalid_mask.sum()} invalid-coordinate rows")

    print(f"\n   Final shape after feature engineering: {df.shape}")
    print("=" * 60)
    return df


if __name__ == "__main__":
    from data_cleaning import clean_data
    df = clean_data()
    df = engineer_features(df)
    print(df[["haversine_distance_km", "manhattan_distance_km",
              "latitude_difference", "longitude_difference"]].describe())
