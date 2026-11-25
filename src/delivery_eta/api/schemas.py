"""What the API accepts and returns."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from delivery_eta.features import LOCAL_TIME, MAX_DISTANCE_KM, haversine_km
from delivery_eta.predictor import Estimate, Order


class Point(BaseModel):
    """A place in India."""

    model_config = ConfigDict(extra="forbid")

    lat: Annotated[float, Field(ge=6, le=38, examples=[22.745])]
    lon: Annotated[float, Field(ge=68, le=98, examples=[75.892])]


class OrderIn(BaseModel):
    """One order. Give `distance_km`, or both `restaurant` and `customer`.

    Everything else is optional: leave out what isn't known yet, and the range widens.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "restaurant": {"lat": 22.745, "lon": 75.892},
                    "customer": {"lat": 22.765, "lon": 75.912},
                    "ordered_at": "2026-11-25T19:30:00+05:30",
                    "rider_age": 28,
                    "rider_rating": 4.7,
                    "vehicle": "motorcycle",
                    "vehicle_condition": 2,
                    "other_deliveries": 0,
                    "weather": "sunny",
                    "traffic": "jam",
                    "festival": False,
                    "city": "metropolitan",
                    "order_type": "meal",
                }
            ]
        },
    )

    restaurant: Point | None = None
    customer: Point | None = None
    distance_km: Annotated[
        float | None,
        Field(gt=0, le=MAX_DISTANCE_KM, description="Straight-line distance, not by road."),
    ] = None
    ordered_at: Annotated[
        datetime | None,
        Field(description="Now, when left out. Without a UTC offset it is Indian time."),
    ] = None
    rider_age: Annotated[int | None, Field(ge=18, le=70)] = None
    rider_rating: Annotated[float | None, Field(ge=1, le=5)] = None
    vehicle: Literal["motorcycle", "scooter", "electric_scooter", "bicycle"] | None = None
    vehicle_condition: Annotated[
        int | None, Field(ge=0, le=2, description="As in the training data; 0 is the slowest.")
    ] = None
    other_deliveries: Annotated[
        int | None, Field(ge=0, le=3, description="Other orders the rider is carrying.")
    ] = None
    weather: Literal["sunny", "cloudy", "fog", "windy", "stormy", "sandstorms"] | None = None
    traffic: Literal["low", "medium", "high", "jam"] | None = None
    festival: bool | None = None
    city: Literal["metropolitan", "urban", "semi_urban"] | None = None
    order_type: Literal["snack", "meal", "drinks", "buffet"] | None = None

    @model_validator(mode="after")
    def _one_way_to_give_distance(self) -> Self:
        points = (self.restaurant is not None) + (self.customer is not None)
        if self.distance_km is not None and points:
            raise ValueError("give distance_km or the two places, not both")
        if self.distance_km is None and points < 2:
            raise ValueError("give distance_km, or both restaurant and customer")
        if self.distance_km is None and self.kilometres() > MAX_DISTANCE_KM:
            apart = f"{self.kilometres():.0f} km apart"
            raise ValueError(f"the places are {apart}; the limit is {MAX_DISTANCE_KM:g} km")
        return self

    def kilometres(self) -> float:
        if self.distance_km is not None:
            return self.distance_km
        if self.restaurant is None or self.customer is None:
            raise ValueError("no distance")
        return float(
            haversine_km(
                self.restaurant.lat, self.restaurant.lon, self.customer.lat, self.customer.lon
            )
        )

    def to_order(self, now: datetime) -> Order:
        when = self.ordered_at or now
        local = when if when.tzinfo is None else when.astimezone(LOCAL_TIME)
        return Order(
            distance_km=self.kilometres(),
            order_hour=local.hour,
            rider_age=self.rider_age,
            rider_rating=self.rider_rating,
            vehicle_condition=self.vehicle_condition,
            other_deliveries=self.other_deliveries,
            vehicle=self.vehicle,
            weather=self.weather,
            traffic=self.traffic,
            festival=None if self.festival is None else ("yes" if self.festival else "no"),
            city=self.city,
            order_type=self.order_type,
        )


class FactorOut(BaseModel):
    input: str
    minutes: Annotated[float, Field(description="Minutes added (or saved, when negative).")]


class EstimateOut(BaseModel):
    minutes: Annotated[float, Field(description="Expected delivery time.")]
    low: Annotated[float, Field(description="Start of the likely range.")]
    high: Annotated[float, Field(description="End of the likely range.")]
    distance_km: float
    missing: Annotated[list[str], Field(description="Inputs that were not given.")]
    warnings: Annotated[list[str], Field(description="Reasons to treat this one with care.")]
    factors: Annotated[
        list[FactorOut] | None, Field(description="With ?explain=true, biggest effect first.")
    ] = None
    model_version: str

    @classmethod
    def build(cls, estimate: Estimate, order: Order, version: str) -> EstimateOut:
        return cls(
            minutes=estimate.minutes,
            low=estimate.low,
            high=estimate.high,
            distance_km=round(order["distance_km"], 2),
            missing=estimate.missing,
            warnings=estimate.warnings,
            factors=None
            if estimate.factors is None
            else [FactorOut(input=f.input, minutes=f.minutes) for f in estimate.factors],
            model_version=version,
        )


class BatchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    orders: Annotated[list[OrderIn], Field(min_length=1)]


class BatchOut(BaseModel):
    estimates: list[EstimateOut]


class ModelOut(BaseModel):
    version: str
    trained_on_orders: int
    first_day: str
    last_day: str
    test_orders: int
    average_error_minutes: float
    guess_the_average_error_minutes: float
    within_5_minutes: float
    range_coverage: float
    seen: dict[str, list[float]]
    key_required: bool
