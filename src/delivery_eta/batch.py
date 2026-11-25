"""Predicts every order in a raw CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from delivery_eta import artifact
from delivery_eta.data import load_orders
from delivery_eta.features import encode
from delivery_eta.training import predict


def predict_file(source: Path, target: Path, model_folder: Path) -> tuple[int, int]:
    """Write one line per order. Orders that can't be predicted say why in `problem`.

    Returns how many were predicted and how many were not.
    """
    model = artifact.load(model_folder)
    orders, _ = load_orders(source, with_target=False)
    usable = (orders["problem"] == "").to_numpy()
    out = orders[["order_id", "problem"]].copy()
    for name in ("minutes", "low", "high"):
        out[name] = np.nan
    if usable.any():
        adjustment = float(model.info["range_adjustment"])
        point, low, high = predict(model.boosters, encode(orders[usable]), adjustment)
        out.loc[usable, ["minutes", "low", "high"]] = np.column_stack([point, low, high]).round(1)
    out["model_version"] = model.version
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = ["order_id", "minutes", "low", "high", "problem", "model_version"]
    out[columns].to_csv(target, index=False, lineterminator="\n")
    return int(usable.sum()), int((~usable).sum())
