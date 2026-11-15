"""Trial protocol: deterministic enrol/test split and target/impostor trials.

The split is per identity, so every trial has exactly one enrolled model and one
test image, and the label is derived from identity equality rather than stored
separately. This matches the verification decision made by the live gate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass(frozen=True)
class Split:
    enroll: dict[int, list[int]]
    test: dict[int, list[int]]

    @property
    def identities(self) -> list[int]:
        return sorted(self.enroll)

    def counts(self) -> dict[str, int]:
        return {"identities": len(self.enroll),
                "enroll_images": sum(len(v) for v in self.enroll.values()),
                "test_images": sum(len(v) for v in self.test.values())}


def split(labels: np.ndarray, k_enroll: int, seed: int = config.SEED) -> Split:
    """Give each identity exactly k_enroll images; the rest are test.

    Identities with fewer than k_enroll + 1 images cannot contribute a target
    trial and are dropped explicitly rather than silently.
    """
    rng = np.random.default_rng(seed)
    enroll: dict[int, list[int]] = {}
    test: dict[int, list[int]] = {}
    for ident in sorted({int(v) for v in labels}):
        idx = np.flatnonzero(labels == ident)
        if len(idx) < k_enroll + 1:
            continue
        perm = rng.permutation(len(idx))
        enroll[ident] = sorted(int(idx[i]) for i in perm[:k_enroll])
        test[ident] = sorted(int(idx[i]) for i in perm[k_enroll:])
    return Split(enroll, test)


def plan(sp: Split, max_impostor: int | None = None,
         seed: int = config.SEED) -> tuple[list[tuple[int, int]],
                                           list[tuple[int, int]]]:
    """Target trials (model, own test image) and impostor trials (model, other)."""
    targets: list[tuple[int, int]] = []
    impostors: list[tuple[int, int]] = []
    for model_id, own in sp.test.items():
        targets.extend((model_id, i) for i in own)
        for other_id, other_imgs in sp.test.items():
            if other_id == model_id:
                continue
            impostors.extend((model_id, i) for i in other_imgs)
    if max_impostor is not None and len(impostors) > max_impostor:
        rng = np.random.default_rng(seed + 1)
        keep = np.sort(rng.choice(len(impostors), max_impostor, replace=False))
        impostors = [impostors[i] for i in keep]
    return targets, impostors
