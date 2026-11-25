"""Runs the trained trees with numpy alone.

LightGBM trains the model, but to run it needs a system library (libgomp) that many hosts
don't have and don't let you install. The trees themselves are simple, so the service walks
them directly. Training checks that the answers match LightGBM's before saving anything.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import numpy.typing as npt

from delivery_eta.features import FEATURES, Floats

# The estimate, and the two ends of its likely range.
PARTS = ("point", "low", "high")

# Rows handled at once, which keeps memory small for big batches.
CHUNK = 128


@dataclass(frozen=True)
class Forest:
    """Every tree of every part, as flat arrays with one entry per node.

    A leaf has feature -1. `value` is a leaf's answer, or the average answer below a split.
    """

    roots: npt.NDArray[np.int32]
    first_tree: npt.NDArray[np.int32]  # where each part's trees start in `roots`
    feature: npt.NDArray[np.int8]
    threshold: Floats
    categories: npt.NDArray[np.uint64]  # for a category split: which codes go left, as bits
    is_category: npt.NDArray[np.bool_]
    unknown_left: npt.NDArray[np.bool_]  # where an unknown value goes
    left: npt.NDArray[np.int32]
    right: npt.NDArray[np.int32]
    value: Floats

    @classmethod
    def from_boosters(cls, boosters: Mapping[str, Any]) -> Forest:
        """Copy the trees out of trained LightGBM models."""
        roots: list[int] = []
        first_tree: list[int] = []
        nodes: list[list[Any]] = []

        def add(node: Mapping[str, Any]) -> int:
            at = len(nodes)
            nodes.append([-1, 0.0, 0, False, False, 0, 0, 0.0])
            if "split_feature" not in node:
                nodes[at][7] = float(node["leaf_value"])
                return at
            is_category = node["decision_type"] == "=="
            threshold, bits = 0.0, 0
            if is_category:
                for code in str(node["threshold"]).split("||"):
                    bits |= 1 << int(code)
                unknown_left = False
            elif node["decision_type"] == "<=":
                threshold = float(node["threshold"])
                if node["missing_type"] == "NaN":
                    unknown_left = bool(node["default_left"])
                elif node["missing_type"] == "None":
                    unknown_left = threshold >= 0  # LightGBM reads unknown as 0 here
                else:
                    raise ValueError(f"can't run missing type {node['missing_type']}")
            else:
                raise ValueError(f"can't run decision type {node['decision_type']}")
            left, right = add(node["left_child"]), add(node["right_child"])
            nodes[at] = [
                int(node["split_feature"]),
                threshold,
                bits,
                is_category,
                unknown_left,
                left,
                right,
                float(node["internal_value"]),
            ]
            return at

        for part in PARTS:
            first_tree.append(len(roots))
            for tree in boosters[part].dump_model()["tree_info"]:
                roots.append(add(tree["tree_structure"]))
        columns = list(zip(*nodes, strict=True))
        return cls(
            roots=np.array(roots, np.int32),
            first_tree=np.array(first_tree, np.int32),
            feature=np.array(columns[0], np.int8),
            threshold=np.array(columns[1], np.float64),
            categories=np.array(columns[2], np.uint64),
            is_category=np.array(columns[3], np.bool_),
            unknown_left=np.array(columns[4], np.bool_),
            left=np.array(columns[5], np.int32),
            right=np.array(columns[6], np.int32),
            value=np.array(columns[7], np.float64),
        )

    def arrays(self) -> dict[str, npt.NDArray[Any]]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, array in self.arrays().items():
            digest.update(f"{name}:{array.dtype.str}:{array.shape}".encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def _goes_left(self, at: npt.NDArray[np.int32], x: Floats) -> npt.NDArray[np.bool_]:
        unknown = np.isnan(x)
        is_category = self.is_category[at]
        by_number = x <= self.threshold[at]  # False when unknown
        codes = np.where(is_category & ~unknown, x, 0).astype(np.uint64)
        by_category = ((self.categories[at] >> codes) & np.uint64(1)).astype(np.bool_)
        known = np.where(is_category, by_category, by_number)
        return np.where(unknown, self.unknown_left[at], known)

    def predict(self, X: Floats) -> tuple[Floats, Floats, Floats]:
        """The raw answer of each part, for every row."""
        chunks = [self._predict(X[i : i + CHUNK]) for i in range(0, len(X), CHUNK)]
        point, low, high = np.vstack(chunks).T
        return point, low, high

    def _predict(self, X: Floats) -> Floats:
        trees = len(self.roots)
        node = np.tile(self.roots, len(X))
        row_start = np.repeat(np.arange(len(X)) * X.shape[1], trees)
        values = X.ravel()
        active = np.arange(len(node))
        while len(active):
            at = node[active]
            x = values[row_start[active] + self.feature[at]]
            below = np.where(self._goes_left(at, x), self.left[at], self.right[at])
            node[active] = below
            active = active[self.feature[below] >= 0]
        answers = self.value[node].reshape(len(X), trees)
        return np.add.reduceat(answers, self.first_tree, axis=1)

    def explain(self, X: Floats) -> Floats:
        """How far each input moved the estimate from the average, for every row.

        Follows each tree's path and credits every step to the input it asked about. The
        steps add up exactly to the estimate minus the average.
        """
        roots = self.roots[: self.first_tree[1]]
        width = len(FEATURES)
        node = np.tile(roots, len(X))
        row_start = np.repeat(np.arange(len(X)) * width, len(roots))
        values = X.ravel()
        moved = np.zeros(len(X) * width)
        active = np.arange(len(node))
        while len(active):
            at = node[active]
            cell = row_start[active] + self.feature[at]
            below = np.where(self._goes_left(at, values[cell]), self.left[at], self.right[at])
            moved += np.bincount(cell, self.value[below] - self.value[at], len(moved))
            node[active] = below
            active = active[self.feature[below] >= 0]
        return moved.reshape(len(X), width)

    @property
    def average(self) -> float:
        """The estimate before any input is considered."""
        return float(self.value[self.roots[: self.first_tree[1]]].sum())


def with_range(
    point: Floats, low: Floats, high: Floats, adjustment: float
) -> tuple[Floats, Floats, Floats]:
    """The final estimate and range. The range always contains the estimate."""
    return (
        point,
        np.maximum(np.minimum(low - adjustment, point), 1.0),
        np.maximum(high + adjustment, point),
    )
