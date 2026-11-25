"""How good predictions are, in minutes."""

from __future__ import annotations

import numpy as np

from delivery_eta.features import Floats


def error_metrics(actual: Floats, predicted: Floats) -> dict[str, float]:
    error = predicted - actual
    size = np.abs(error)
    spread = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "mae": float(size.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1 - np.sum(error**2) / spread) if spread else 0.0,
        "within_5_min": float((size <= 5).mean()),
        "p90_error": float(np.quantile(size, 0.9)),
        "bias": float(error.mean()),
    }


def range_metrics(actual: Floats, low: Floats, high: Floats) -> dict[str, float]:
    return {
        "coverage": float(((actual >= low) & (actual <= high)).mean()),
        "mean_width": float((high - low).mean()),
    }
