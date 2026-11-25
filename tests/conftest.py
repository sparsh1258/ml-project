"""Shared test setup: small made-up orders, and a real model trained on them."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from delivery_eta.api.app import create_app
from delivery_eta.api.settings import Settings
from delivery_eta.predictor import Order
from delivery_eta.training import TrainConfig, train

ROOT = Path(__file__).parent.parent
REAL_DATA = ROOT / "data" / "raw" / "train.csv"
CONFIG = ROOT / "configs" / "train.json"


def raw_row(**changes: str) -> dict[str, str]:
    """One order as it appears in the raw file, stray spaces included."""
    row = {
        "ID": "0x1 ",
        "Delivery_person_ID": "INDORES13DEL02 ",
        "Delivery_person_Age": "28",
        "Delivery_person_Ratings": "4.7",
        "Restaurant_latitude": "22.745049",
        "Restaurant_longitude": "75.892471",
        "Delivery_location_latitude": "22.765049",
        "Delivery_location_longitude": "75.912471",
        "Order_Date": "19-03-2022",
        "Time_Orderd": "19:30:00",
        "Time_Order_picked": "19:45:00",
        "Weatherconditions": "conditions Sunny",
        "Road_traffic_density": "Jam ",
        "Vehicle_condition": "2",
        "Type_of_order": "Snack ",
        "Type_of_vehicle": "electric_scooter ",
        "multiple_deliveries": "0",
        "Festival": "No ",
        "City": "Metropolitian ",
        "Time_taken(min)": "(min) 24",
    }
    return {**row, **changes}


@pytest.fixture
def write_csv(tmp_path: Path) -> Callable[..., Path]:
    def write(*rows: dict[str, str], name: str = "orders.csv") -> Path:
        path = tmp_path / name
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    return write


def made_up_orders(count: int = 4000) -> pd.DataFrame:
    """Orders whose time depends on distance, traffic and the rider in a known way."""
    rng = np.random.default_rng(7)
    hour = rng.choice([9, 13, 18, 20], count)
    traffic = np.array(["Low ", "High ", "Medium ", "Jam "])[np.searchsorted([9, 13, 18, 20], hour)]
    north = rng.uniform(0.01, 0.12, count)
    age = rng.integers(20, 40, count)
    extra = {"Low ": 0, "Medium ": 4, "High ": 6, "Jam ": 12}
    minutes = 12 + 80 * north + [extra[t] for t in traffic] + 5 * (age >= 30)
    minutes = np.clip(np.round(minutes + rng.normal(0, 1.5, count)), 5, 90).astype(int)
    days = pd.Timestamp("2022-02-01") + pd.to_timedelta(rng.integers(0, 40, count), unit="D")
    rows = [
        raw_row(
            ID=f"0x{i:x}",
            Delivery_person_Age=str(age[i]),
            Delivery_location_latitude=str(22.745049 + north[i]),
            Delivery_location_longitude="75.892471",
            Order_Date=days[i].strftime("%d-%m-%Y"),
            Time_Orderd=f"{hour[i]:02d}:30:00",
            Time_Order_picked=f"{hour[i]:02d}:40:00",
            Road_traffic_density=traffic[i],
            **{"Time_taken(min)": f"(min) {minutes[i]}"},
        )
        for i in range(count)
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def small_config() -> TrainConfig:
    return replace(
        TrainConfig.load(CONFIG),
        rounds=60,
        cv_folds=2,
        params={"learning_rate": 0.1, "num_leaves": 15},
        gate={
            "max_mae": 3.0,
            "min_improvement_over_average": 0.4,
            "max_worst_group_mae": 5.0,
            "max_mae_rider_not_assigned": 5.0,
            "min_coverage": 0.6,
            "max_coverage": 0.97,
        },
    )


@pytest.fixture(scope="session")
def small_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "train.csv"
    made_up_orders().to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def model_dir(
    tmp_path_factory: pytest.TempPathFactory, small_data: Path, small_config: TrainConfig
) -> Path:
    folder = tmp_path_factory.mktemp("artifacts") / "model"
    _, failures = train(small_data, small_config, folder)
    assert failures == []
    return folder


def an_order(**changes: Any) -> Order:
    order: dict[str, Any] = {
        "distance_km": 6.0,
        "order_hour": 20,
        "rider_age": 28.0,
        "rider_rating": 4.7,
        "vehicle_condition": 2.0,
        "other_deliveries": 0.0,
        "vehicle": "electric_scooter",
        "weather": "sunny",
        "traffic": "jam",
        "festival": "no",
        "city": "metropolitan",
        "order_type": "snack",
    }
    return {**order, **changes}  # type: ignore[typeddict-item, return-value]


def a_request(**changes: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "distance_km": 6,
        "ordered_at": "2026-11-25T20:30:00",
        "rider_age": 28,
        "rider_rating": 4.7,
        "vehicle": "electric_scooter",
        "vehicle_condition": 2,
        "other_deliveries": 0,
        "weather": "sunny",
        "traffic": "jam",
        "festival": False,
        "city": "metropolitan",
        "order_type": "snack",
    }
    return {k: v for k, v in {**body, **changes}.items() if v is not None}


@pytest.fixture
def make_client(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[..., TestClient]]:
    # Settings on the developer's machine must not change what the tests see.
    for name in Settings.model_fields:
        monkeypatch.delenv(f"ETA_{name.upper()}", raising=False)
    opened: list[TestClient] = []

    def make(**settings: Any) -> TestClient:
        app = create_app(Settings(**{"model_dir": model_dir, **settings}))
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()
        opened.append(client)
        return client

    yield make
    for client in opened:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()
