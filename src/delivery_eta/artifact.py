"""The saved model: three LightGBM text files and a JSON description.

Nothing here is pickled, so loading a model can't run code. Every file's fingerprint is
checked on load, and a model built for different inputs is refused.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb

from delivery_eta.features import CATEGORIES, FEATURES

FORMAT = 1
PARTS = ("point", "low", "high")


class ArtifactError(RuntimeError):
    """The saved model is missing, damaged, or built for different inputs."""


@dataclass(frozen=True)
class Model:
    boosters: dict[str, lgb.Booster]
    info: dict[str, Any]

    @property
    def version(self) -> str:
        return str(self.info["version"])


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def save(boosters: dict[str, lgb.Booster], info: dict[str, Any], folder: Path) -> dict[str, Any]:
    """Write the model to `folder`, replacing what was there only once everything is written."""
    texts = {part: boosters[part].model_to_string() for part in PARTS}
    files = {f"{part}.txt": _sha256(text) for part, text in texts.items()}
    identity = json.dumps([files, info["range_adjustment"]], sort_keys=True)
    info = {
        "format": FORMAT,
        "version": _sha256(identity)[:12],
        "features": list(FEATURES),
        "categories": {k: list(v) for k, v in CATEGORIES.items()},
        "files": files,
        **info,
    }
    folder = Path(folder)
    staging = folder.with_name(folder.name + ".new")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for part, text in texts.items():
        (staging / f"{part}.txt").write_bytes(text.encode())
    (staging / "model.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(folder, ignore_errors=True)
    staging.rename(folder)
    return info


def load(folder: Path) -> Model:
    folder = Path(folder)
    try:
        info = json.loads((folder / "model.json").read_text(encoding="utf-8"))
        texts = {part: (folder / f"{part}.txt").read_bytes().decode() for part in PARTS}
    except FileNotFoundError as e:
        raise ArtifactError(
            f"No trained model in {folder}. Build one with: delivery-eta train"
        ) from e
    except ValueError as e:
        raise ArtifactError(f"The model in {folder} can't be read: {e}") from e
    if info.get("format") != FORMAT:
        raise ArtifactError(f"Model format {info.get('format')} isn't supported; retrain it.")
    if info["features"] != list(FEATURES) or info["categories"] != {
        k: list(v) for k, v in CATEGORIES.items()
    }:
        raise ArtifactError("This model was built for different inputs; retrain it.")
    for part, text in texts.items():
        if _sha256(text) != info["files"].get(f"{part}.txt"):
            raise ArtifactError(f"{part}.txt doesn't match its recorded fingerprint.")
    return Model({part: lgb.Booster(model_str=text) for part, text in texts.items()}, info)
