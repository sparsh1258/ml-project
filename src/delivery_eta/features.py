"""Turns orders into the numbers the model sees.

Training and the live API both go through this module, so the two can't drift apart.
It needs only numpy, which keeps the running service small.
"""

from __future__ import annotations

from datetime import timedelta, timezone
from typing import Any, Final

import numpy as np
import numpy.typing as npt

Floats = npt.NDArray[np.float64]

# Orders are in India, which has one time zone and no daylight saving.
LOCAL_TIME: Final = timezone(timedelta(hours=5, minutes=30), "IST")

# Longest straight-line trip we accept. The training data ends at 21 km.
MAX_DISTANCE_KM: Final = 50.0

NUMERIC: Final[tuple[str, ...]] = (
    "distance_km",
    "order_hour",
    "rider_age",
    "rider_rating",
    "vehicle_condition",
    "other_deliveries",
)
CATEGORIES: Final[dict[str, tuple[str, ...]]] = {
    "vehicle": ("motorcycle", "scooter", "electric_scooter", "bicycle"),
    "weather": ("sunny", "cloudy", "fog", "windy", "stormy", "sandstorms"),
    "traffic": ("low", "medium", "high", "jam"),
    "festival": ("no", "yes"),
    "city": ("metropolitan", "urban", "semi_urban"),
    "order_type": ("snack", "meal", "drinks", "buffet"),
}
FEATURES: Final[tuple[str, ...]] = NUMERIC + tuple(CATEGORIES)
CATEGORY_POSITIONS: Final[list[int]] = [FEATURES.index(name) for name in CATEGORIES]

# An order must have these; everything else may be unknown.
REQUIRED: Final[tuple[str, ...]] = ("distance_km", "order_hour")
OPTIONAL: Final[tuple[str, ...]] = tuple(f for f in FEATURES if f not in REQUIRED)

# Inputs that go missing together in practice. The model is trained with each of these
# blanked out in a share of rows, so it copes when they are unknown.
MISSING_TOGETHER: Final[dict[str, tuple[str, ...]]] = {
    "rider_not_assigned": (
        "rider_age",
        "rider_rating",
        "vehicle_condition",
        "other_deliveries",
        "vehicle",
    ),
    "rider_profile": ("rider_age", "rider_rating"),
    "rider_age": ("rider_age",),
    "rider_rating": ("rider_rating",),
    "weather": ("weather",),
    "traffic": ("traffic",),
    "weather_and_traffic": ("weather", "traffic"),
    "festival": ("festival",),
    "city": ("city",),
    "everything_optional": OPTIONAL,
}

_CODES: Final = {
    name: {level: float(code) for code, level in enumerate(levels)}
    for name, levels in CATEGORIES.items()
}


def haversine_km(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> Any:
    """Straight-line distance over the Earth's surface, in kilometres."""
    lat1, lon1, lat2, lon2 = (np.radians(v) for v in (lat1, lon1, lat2, lon2))
    h = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


def encode(orders: Any) -> Floats:
    """The model's input matrix: one row per order, columns in FEATURES order.

    `orders` is anything that gives a column by name (a dict of lists or a DataFrame).
    Unknown values become NaN, which the model treats as missing.
    """
    columns: list[Floats] = [np.asarray(orders[name], dtype=np.float64) for name in NUMERIC]
    for name, codes in _CODES.items():
        values = [codes.get(v, np.nan) if isinstance(v, str) else np.nan for v in orders[name]]
        columns.append(np.asarray(values, dtype=np.float64))
    return np.column_stack(columns)
