"""The saved model: the trees as plain numbers, and a JSON description.

Nothing here is pickled, so loading a model can't run code. The file's fingerprint is checked
on load, and a model built for different inputs is refused.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from delivery_eta.features import CATEGORIES, FEATURES
from delivery_eta.forest import Forest

FORMAT = 2
TREES = "forest.npz"


class ArtifactError(RuntimeError):
    """The saved model is missing, damaged, or built for different inputs."""


@dataclass(frozen=True)
class Model:
    forest: Forest
    info: dict[str, Any]

    @property
    def version(self) -> str:
        return str(self.info["version"])


def save(forest: Forest, info: dict[str, Any], folder: Path) -> dict[str, Any]:
    """Write the model to `folder`, replacing what was there only once everything is written."""
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **forest.arrays())  # type: ignore[arg-type]
    data = buffer.getvalue()
    identity = f"{forest.fingerprint()} {info['range_adjustment']!r}"
    info = {
        "format": FORMAT,
        "version": hashlib.sha256(identity.encode()).hexdigest()[:12],
        "features": list(FEATURES),
        "categories": {k: list(v) for k, v in CATEGORIES.items()},
        "files": {TREES: hashlib.sha256(data).hexdigest()},
        **info,
    }
    folder = Path(folder)
    staging = folder.with_name(folder.name + ".new")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    (staging / TREES).write_bytes(data)
    (staging / "model.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    shutil.rmtree(folder, ignore_errors=True)
    staging.rename(folder)
    return info


def load(folder: Path) -> Model:
    folder = Path(folder)
    try:
        info = json.loads((folder / "model.json").read_text(encoding="utf-8"))
        data = (folder / TREES).read_bytes()
    except FileNotFoundError as e:
        raise ArtifactError(
            f"No trained model in {folder.absolute()}. Build one with: delivery-eta train"
        ) from e
    except ValueError as e:
        raise ArtifactError(f"The model in {folder} can't be read: {e}") from e
    if info.get("format") != FORMAT:
        raise ArtifactError(f"Model format {info.get('format')} isn't supported; retrain it.")
    if info["features"] != list(FEATURES) or info["categories"] != {
        k: list(v) for k, v in CATEGORIES.items()
    }:
        raise ArtifactError("This model was built for different inputs; retrain it.")
    if hashlib.sha256(data).hexdigest() != info["files"].get(TREES):
        raise ArtifactError(f"{TREES} doesn't match its recorded fingerprint.")
    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        return Model(Forest(**{name: arrays[name] for name in arrays.files}), info)
