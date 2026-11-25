"""Reads raw order CSVs and cleans them into the form the model uses.

Rows are never dropped silently: each row gets a `problem` (empty when it is usable),
and the report counts everything that was fixed or rejected.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from delivery_eta.features import CATEGORIES, MAX_DISTANCE_KM, haversine_km

TARGET = "minutes"

RAW_COORDINATES = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
]
RAW_CATEGORIES = {
    "vehicle": "Type_of_vehicle",
    "weather": "Weatherconditions",
    "traffic": "Road_traffic_density",
    "festival": "Festival",
    "city": "City",
    "order_type": "Type_of_order",
}
RAW_TARGET = "Time_taken(min)"
RAW_COLUMNS = [
    "ID",
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    *RAW_COORDINATES,
    "Order_Date",
    "Time_Orderd",
    "Time_Order_picked",
    "Vehicle_condition",
    "multiple_deliveries",
    *RAW_CATEGORIES.values(),
]
SPELLING = {"metropolitian": "metropolitan"}
BLANKS = ["", "NaN", "nan", "conditions NaN"]

# When the order time is missing it is estimated from the pickup time. Preparation takes
# 5, 10 or 15 minutes in every row that has both.
TYPICAL_PREP_MINUTES = 10


class DataError(ValueError):
    """The data can't be used. The message says why, in plain words."""


@dataclass
class CleaningReport:
    rows_read: int = 0
    rows_usable: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    fixed: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def file_sha256(path: Path) -> str:
    """Fingerprint of a text file, the same whether it has Windows or Unix line endings."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _minute_of_day(text: pd.Series) -> pd.Series:
    time = pd.to_datetime(text, format="%H:%M:%S", errors="coerce")
    return time.dt.hour * 60 + time.dt.minute


def load_orders(path: Path, *, with_target: bool) -> tuple[pd.DataFrame, CleaningReport]:
    """Every row of the file, cleaned. Rows that can't be used have a non-empty `problem`."""
    path = Path(path)
    if not path.is_file():
        raise DataError(f"No file at {path}.")
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw.columns = raw.columns.str.strip()
    wanted = [*RAW_COLUMNS, RAW_TARGET] if with_target else RAW_COLUMNS
    absent = [c for c in wanted if c not in raw.columns]
    if absent:
        raise DataError(f"{path.name} is missing these columns: {', '.join(absent)}.")
    if raw.empty:
        raise DataError(f"{path.name} has no rows.")

    raw = raw[wanted].apply(lambda column: column.str.strip())
    raw = raw.mask(raw.isin(BLANKS))
    report = CleaningReport(rows_read=len(raw))
    out = pd.DataFrame(index=raw.index)
    out["order_id"] = raw["ID"]
    out["order_date"] = pd.to_datetime(raw["Order_Date"], format="%d-%m-%Y", errors="coerce")

    # Coordinates: India is north of the equator and east of Greenwich, so a minus sign is a
    # recording error. (0, 0) means the location was never recorded.
    coordinates = raw[RAW_COORDINATES].apply(pd.to_numeric, errors="coerce")
    report.fixed["flipped coordinate signs"] = int((coordinates < 0).any(axis=1).sum())
    coordinates = coordinates.abs()
    no_location = coordinates.isna().any(axis=1) | (coordinates < 1).any(axis=1)
    out["distance_km"] = haversine_km(*(coordinates[c] for c in RAW_COORDINATES))
    out.loc[no_location, "distance_km"] = np.nan

    ordered = _minute_of_day(raw["Time_Orderd"])
    estimated = (_minute_of_day(raw["Time_Order_picked"]) - TYPICAL_PREP_MINUTES) % 1440
    report.fixed["order time estimated from pickup"] = int(
        (ordered.isna() & estimated.notna()).sum()
    )
    out["order_hour"] = ordered.fillna(estimated) // 60

    # Rider details: rows with an impossible age or rating have neither trusted.
    age = pd.to_numeric(raw["Delivery_person_Age"], errors="coerce")
    rating = pd.to_numeric(raw["Delivery_person_Ratings"], errors="coerce")
    impossible = (age.notna() & ~age.between(18, 70)) | (rating.notna() & ~rating.between(1, 5))
    report.fixed["impossible rider details blanked"] = int(impossible.sum())
    out["rider_age"] = age.mask(impossible)
    out["rider_rating"] = rating.mask(impossible)

    # Condition 3 only ever appears on rows whose rider details are blank or impossible.
    condition = pd.to_numeric(raw["Vehicle_condition"], errors="coerce")
    odd = condition.notna() & ~condition.isin([0, 1, 2])
    report.fixed["unknown vehicle condition blanked"] = int(odd.sum())
    out["vehicle_condition"] = condition.mask(odd)
    others = pd.to_numeric(raw["multiple_deliveries"], errors="coerce")
    out["other_deliveries"] = others.where(others.isin([0, 1, 2, 3]))

    for name, column in RAW_CATEGORIES.items():
        text = raw[column].str.removeprefix("conditions ").str.lower()
        text = text.str.replace(r"[ -]", "_", regex=True).replace(SPELLING)
        unknown = text.notna() & ~text.isin(CATEGORIES[name])
        if unknown.any():
            report.fixed[f"unknown {name} blanked"] = int(unknown.sum())
        out[name] = text.mask(unknown).astype(object)

    problems: list[tuple[str, pd.Series]] = [
        ("no location recorded", no_location),
        (f"farther than {MAX_DISTANCE_KM:g} km", out["distance_km"] > MAX_DISTANCE_KM),
        ("no order time", out["order_hour"].isna()),
    ]
    if with_target:
        out[TARGET] = pd.to_numeric(
            raw[RAW_TARGET].str.extract(r"(\d+(?:\.\d+)?)", expand=False), errors="coerce"
        )
        problems += [
            ("no usable delivery time", ~out[TARGET].between(1, 180)),
            ("no order date", out["order_date"].isna()),
            ("repeated order", out["order_id"].notna() & out["order_id"].duplicated()),
        ]
    out["problem"] = ""
    for reason, rows in problems:
        fresh = rows & (out["problem"] == "")
        if fresh.any():
            out.loc[fresh, "problem"] = reason
            report.rejected[reason] = int(fresh.sum())
    report.rows_usable = int((out["problem"] == "").sum())
    return out, report


def load_training_orders(path: Path, min_rows: int = 1000) -> tuple[pd.DataFrame, CleaningReport]:
    """Usable rows only, oldest first, after checking the file is fit to train on."""
    orders, report = load_orders(path, with_target=True)
    usable = orders[orders["problem"] == ""].drop(columns="problem")
    if len(usable) < min_rows:
        raise DataError(f"Only {len(usable)} usable rows; training needs at least {min_rows}.")
    if len(usable) < 0.8 * report.rows_read:
        raise DataError(f"Too many rows rejected: {report.rejected}.")
    for column in (*CATEGORIES, "rider_age", "rider_rating"):
        share = float(usable[column].isna().mean())
        if share > 0.3:
            raise DataError(f"{column} is missing in {share:.0%} of rows; the limit is 30%.")
    return usable.sort_values("order_date", kind="stable").reset_index(drop=True), report
