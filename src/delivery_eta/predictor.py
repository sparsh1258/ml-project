"""Turns a loaded model into estimates people can use."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np

from delivery_eta import artifact
from delivery_eta.features import FEATURES, OPTIONAL, Floats, encode


class Order(TypedDict):
    """One order in the model's own terms. None means unknown."""

    distance_km: float
    order_hour: int
    rider_age: float | None
    rider_rating: float | None
    vehicle_condition: float | None
    other_deliveries: float | None
    vehicle: str | None
    weather: str | None
    traffic: str | None
    festival: str | None
    city: str | None
    order_type: str | None


@dataclass(frozen=True)
class Factor:
    """How much one input moved this estimate, in minutes."""

    input: str
    minutes: float


@dataclass(frozen=True)
class Estimate:
    minutes: float
    low: float
    high: float
    missing: list[str]
    warnings: list[str]
    factors: list[Factor] | None


def _number(value: float) -> str:
    return f"{value:.1f}".removesuffix(".0")


class Predictor:
    def __init__(self, model: artifact.Model) -> None:
        self.model = model
        self._adjustment = float(model.info["range_adjustment"])
        self._seen: dict[str, list[float]] = model.info["seen"]
        self._hours = set(model.info["hours_seen"])
        self._traffic_hours: dict[str, list[int]] = model.info["traffic_hours_seen"]

    @classmethod
    def load(cls, folder: Path) -> Predictor:
        return cls(artifact.load(folder))

    def _run(self, part: str, X: Floats) -> Floats:
        # One thread each: requests are small, and the server handles several at once.
        return np.asarray(self.model.boosters[part].predict(X, num_threads=1))

    def predict(self, orders: Sequence[Order], *, explain: bool = False) -> list[Estimate]:
        if not orders:
            return []
        X = encode({name: [order[name] for order in orders] for name in FEATURES})  # type: ignore[literal-required]
        point = self._run("point", X)
        low = np.maximum(np.minimum(self._run("low", X) - self._adjustment, point), 1.0)
        high = np.maximum(self._run("high", X) + self._adjustment, point)
        parts: Floats | None = None
        if explain:
            parts = np.asarray(
                self.model.boosters["point"].predict(X, pred_contrib=True, num_threads=1)
            )
        return [
            Estimate(
                minutes=round(float(point[i]), 1),
                low=round(float(low[i]), 1),
                high=round(float(high[i]), 1),
                missing=[name for name in OPTIONAL if order[name] is None],  # type: ignore[literal-required]
                warnings=self._warnings(order),
                factors=None if parts is None else self._factors(parts[i]),
            )
            for i, order in enumerate(orders)
        ]

    def _warnings(self, order: Order) -> list[str]:
        warnings = []
        for name, (smallest, largest) in self._seen.items():
            value = order[name]  # type: ignore[literal-required]
            if name == "order_hour" or value is None:
                continue
            if not smallest - 0.05 <= value <= largest + 0.05:
                warnings.append(
                    f"{name} {_number(value)} is outside what the model learned from "
                    f"({_number(smallest)} to {_number(largest)}); treat this estimate with care."
                )
        hour, traffic = order["order_hour"], order["traffic"]
        if hour not in self._hours:
            warnings.append(
                f"The model has seen no orders between {hour}:00 and {hour}:59; "
                "treat this estimate with care."
            )
        elif traffic is not None and hour not in self._traffic_hours[traffic]:
            warnings.append(
                f"The model has hardly seen {traffic} traffic between {hour}:00 and {hour}:59; "
                "treat this estimate with care."
            )
        return warnings

    @staticmethod
    def _factors(contributions: Floats) -> list[Factor]:
        # The last column is the starting point (the average), not an input.
        pairs = zip(FEATURES, contributions[:-1], strict=True)
        ranked = sorted(pairs, key=lambda pair: -abs(pair[1]))
        return [Factor(name, round(float(value), 1) + 0.0) for name, value in ranked]
