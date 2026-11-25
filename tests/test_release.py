"""Checks on the real data and on the released model in model/."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import pytest

from delivery_eta import artifact
from delivery_eta.data import TARGET, file_sha256, load_training_orders
from delivery_eta.features import FEATURES, encode
from delivery_eta.predictor import Predictor
from delivery_eta.training import TrainConfig, gate_failures

from .conftest import CONFIG, REAL_DATA, ROOT, an_order

MODEL = ROOT / "model"


def published() -> dict:
    return json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))


def test_the_real_data_cleans_as_documented() -> None:
    orders, report = load_training_orders(REAL_DATA)
    assert report.rows_read == 45593
    assert report.rows_usable == len(orders) == 41953
    assert report.rejected == {"no location recorded": 3640}
    assert not np.isnan(encode(orders)[:, [FEATURES.index("distance_km")]]).any()
    assert orders[TARGET].between(10, 54).all()


def test_the_released_model_is_up_to_date() -> None:
    """After changing the data or settings, run `train --report reports/metrics.json`."""
    info = artifact.load(MODEL).info
    config = TrainConfig.load(CONFIG)
    assert info["report"]["data"]["file_sha256"] == file_sha256(REAL_DATA)
    assert info["params"] == config.params
    assert info["rounds"] == config.rounds
    assert info["coverage"] == config.coverage
    assert published()["model_version"] == info["version"]
    assert published()["test"] == info["report"]["test"]


def test_the_released_model_passes_the_gate() -> None:
    assert gate_failures(published()["test"], TrainConfig.load(CONFIG).gate) == []


def test_the_released_model_agrees_with_real_groups_of_orders() -> None:
    orders, _ = load_training_orders(REAL_DATA)
    orders = orders.iloc[::10]
    point, _, _ = artifact.load(MODEL).forest.predict(encode(orders))
    for name in ("traffic", "weather", "festival"):
        for level, rows in orders.reset_index(drop=True).groupby(name).groups.items():
            real, predicted = orders[TARGET].to_numpy()[rows].mean(), point[rows].mean()
            assert abs(real - predicted) < 2.5, (name, level, real, predicted)


def test_the_released_model_is_fast_enough() -> None:
    predictor = Predictor.load(MODEL)
    predictor.predict([an_order()])
    start = time.perf_counter()
    for _ in range(50):
        predictor.predict([an_order()], explain=True)
    assert (time.perf_counter() - start) / 50 < 0.1


TRY_IT = """
import sys
for name in ("lightgbm", "pandas", "scipy", "sklearn", "xgboost"):
    sys.modules[name] = None  # importing any of these now fails, as it would on the host
sys.path.insert(0, sys.argv[1])
from app import app
from fastapi import FastAPI
from fastapi.testclient import TestClient
assert isinstance(app, FastAPI)
with TestClient(app) as client:
    reply = client.post("/api/v1/predict?explain=true", json={"distance_km": 8})
    assert reply.status_code == 200, reply.text
    assert client.get("/").status_code == 200
    print(reply.json()["model_version"])
"""


def test_the_hosted_app_needs_nothing_from_training(tmp_path: Path) -> None:
    """Started the way Vercel does: from app.py, in another folder, with only what it installs."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ETA_", "PYTHONPATH"))}
    run = subprocess.run(
        [sys.executable, "-c", TRY_IT, str(ROOT)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode == 0, run.stderr
    assert artifact.load(MODEL).version in run.stdout


def test_vercel_is_told_where_the_app_is() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    module, name = project["tool"]["vercel"]["entrypoint"].split(":")
    assert f"\n{name} = create_app()" in (ROOT / f"{module}.py").read_text(encoding="utf-8")
    needed = " ".join(project["project"]["dependencies"])
    for heavy in ("lightgbm", "pandas", "scipy", "scikit-learn"):
        assert heavy not in needed


@pytest.mark.parametrize(
    "path", ["src", "model", "app.py", "pyproject.toml", "uv.lock", "README.md"]
)
def test_vercel_is_sent_what_it_needs(path: str) -> None:
    ignored = (ROOT / ".vercelignore").read_text(encoding="utf-8").split()
    assert f"/{path}" not in ignored
