"""
Data Cleaning Module
====================
Loads the raw dataset, strips whitespace, parses encoded NaN values,
converts types, and flags invalid geographic coordinates.
"""

import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "delivery_data.csv")


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and return as-is."""
    return pd.read_csv(path)


def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all string columns and column names."""
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def replace_nan_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Many categorical columns encode missing values as the literal string 'NaN'.
    Replace these with np.nan so they are handled uniformly.
    """
    df.replace({"NaN": np.nan, "nan": np.nan, "": np.nan}, inplace=True)
    return df


def parse_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target column 'Time_taken(min)' has format '(min) 24'.
    Extract the integer and store as float.
    """
    df["Time_taken(min)"] = (
        df["Time_taken(min)"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(float)
    )
    return df


def parse_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weatherconditions column has format 'conditions Sunny'.
    Strip the 'conditions ' prefix.
    """
    df["Weatherconditions"] = (
        df["Weatherconditions"]
        .astype(str)
        .str.replace("conditions ", "", regex=False)
        .str.strip()
    )
    df["Weatherconditions"].replace({"nan": np.nan, "NaN": np.nan}, inplace=True)
    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Delivery_person_Age, Delivery_person_Ratings, multiple_deliveries
    are stored as strings — convert to numeric.
    """
    for col in ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_dates_and_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse Order_Date (dd-mm-yyyy) and time columns.
    Handles inconsistent date formats gracefully with dayfirst=True.
    """
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], dayfirst=True, errors="coerce")

    # Parse times — these are HH:MM:SS strings
    for col in ["Time_Orderd", "Time_Order_picked"]:
        df[col] = pd.to_datetime(df[col], format="%H:%M:%S", errors="coerce")
    return df


def flag_invalid_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag invalid geographic coordinates:
    - Zero or near-zero lat/lon (likely missing)
    - Negative latitudes that are geographically impossible for India
    - Values outside valid ranges (lat: -90 to 90, lon: -180 to 180)
    
    Indian data should roughly be: lat ∈ [6, 36], lon ∈ [68, 98]
    """
    lat_cols = ["Restaurant_latitude", "Delivery_location_latitude"]
    lon_cols = ["Restaurant_longitude", "Delivery_location_longitude"]

    # Flag rows where any coordinate is zero or near-zero (< 1 absolute)
    zero_mask = pd.Series(False, index=df.index)
    for col in lat_cols + lon_cols:
        zero_mask |= df[col].abs() < 1.0

    # Flag negative latitudes (Indian coords should be positive)
    neg_lat_mask = pd.Series(False, index=df.index)
    for col in lat_cols:
        neg_lat_mask |= df[col] < 0

    df["invalid_coordinates"] = zero_mask | neg_lat_mask
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove fully duplicate rows (if any)."""
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"  Removed {n_removed} duplicate rows.")
    else:
        print("  No duplicate rows found.")
    return df


def clean_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Full cleaning pipeline — returns a cleaned DataFrame.
    """
    print("=" * 60)
    print("DATA CLEANING")
    print("=" * 60)

    print("\n1. Loading raw data...")
    df = load_raw_data(path)
    print(f"   Shape: {df.shape}")

    print("2. Stripping whitespace...")
    df = strip_whitespace(df)

    print("3. Replacing literal 'NaN' strings...")
    df = replace_nan_strings(df)

    print("4. Parsing target variable...")
    df = parse_target(df)
    print(f"   Target range: {df['Time_taken(min)'].min():.0f} – {df['Time_taken(min)'].max():.0f} min")

    print("5. Parsing weather conditions...")
    df = parse_weather(df)

    print("6. Converting numeric columns...")
    df = convert_numeric_columns(df)

    print("7. Parsing dates and times...")
    df = parse_dates_and_times(df)

    print("8. Flagging invalid coordinates...")
    df = flag_invalid_coordinates(df)
    n_invalid = df["invalid_coordinates"].sum()
    print(f"   {n_invalid} rows with invalid coordinates ({n_invalid/len(df)*100:.1f}%)")

    print("9. Removing duplicate rows...")
    df = remove_duplicates(df)

    # Summary of missing values after cleaning
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print("\n   Missing values after cleaning:")
        for col, count in missing.items():
            print(f"     {col}: {count} ({count/len(df)*100:.1f}%)")

    print(f"\n   Final shape: {df.shape}")
    print("=" * 60)
    return df


if __name__ == "__main__":
    df = clean_data()
    print(df.dtypes)
    print(df.head())
