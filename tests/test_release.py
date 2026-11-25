"""Checks on the real data, and on the real model when one has been built."""

from __future__ import annotations

import json

import numpy as np
import pytest

from delivery_eta import artifact
from delivery_eta.data import TARGET, file_sha256, load_training_orders
from delivery_eta.features import FEATURES, encode
from delivery_eta.predictor import Predictor
from delivery_eta.training import TrainConfig, gate_failures

from .conftest import CONFIG, REAL_DATA, ROOT

MODEL = ROOT / "artifacts" / "model"
needs_model = pytest.mark.skipif(not MODEL.is_dir(), reason="run `delivery-eta train` first")


def test_the_real_data_cleans_as_documented() -> None:
    orders, report = load_training_orders(REAL_DATA)
    assert report.rows_read == 45593
    assert report.rows_usable == len(orders) == 41953
    assert report.rejected == {"no location recorded": 3640}
    assert not np.isnan(encode(orders)[:, [FEATURES.index("distance_km")]]).any()
    assert orders[TARGET].between(10, 54).all()


def test_the_published_numbers_pass_the_gate_and_match_the_data() -> None:
    report = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert report["data"]["file_sha256"] == file_sha256(REAL_DATA)
    assert gate_failures(report["test"], TrainConfig.load(CONFIG).gate) == []


@needs_model
def test_the_built_model_matches_the_published_numbers() -> None:
    info = artifact.load(MODEL).info
    published = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    built, stated = info["report"]["test"]["overall"], published["test"]["overall"]
    # Different processors round differently, so close is enough.
    for name in ("mae", "coverage", "within_5_min"):
        assert built[name] == pytest.approx(stated[name], abs=0.05)


@needs_model
def test_the_built_model_agrees_with_real_groups_of_orders() -> None:
    orders, _ = load_training_orders(REAL_DATA)
    point = np.asarray(artifact.load(MODEL).boosters["point"].predict(encode(orders)))
    for name in ("traffic", "weather", "city", "festival"):
        for level, rows in orders.groupby(name).groups.items():
            real, predicted = orders.loc[rows, TARGET].mean(), point[rows].mean()
            assert abs(real - predicted) < 2, (name, level, real, predicted)


@needs_model
def test_the_built_model_is_fast_enough() -> None:
    import time

    predictor = Predictor.load(MODEL)
    order = dict.fromkeys(FEATURES)
    order.update(distance_km=8.0, order_hour=20)  # type: ignore[arg-type]
    predictor.predict([order])  # type: ignore[list-item]
    start = time.perf_counter()
    for _ in range(50):
        predictor.predict([order])  # type: ignore[list-item]
    assert (time.perf_counter() - start) / 50 < 0.05
