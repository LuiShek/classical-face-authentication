"""LBPH wrapper: fit a per-identity model, expose its chi-square distance.

The recogniser is trained with a single label because this project does
verification, not identification: one model per enrolled identity, and the
distance to it is the score.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import config


class LbphModel:
    def __init__(self, *, radius: int = config.LBPH_RADIUS,
                 neighbors: int = config.LBPH_NEIGHBORS,
                 grid_x: int = config.LBPH_GRID_X,
                 grid_y: int = config.LBPH_GRID_Y,
                 threshold: float = 1e9) -> None:
        self.rec = cv2.face.LBPHFaceRecognizer_create(
            radius=radius, neighbors=neighbors,
            grid_x=grid_x, grid_y=grid_y, threshold=threshold)

    def fit(self, images) -> "LbphModel":
        imgs = [np.ascontiguousarray(i, dtype=np.uint8) for i in images]
        if not imgs:
            raise ValueError("cannot fit on zero images")
        self.rec.train(imgs, np.zeros(len(imgs), dtype=np.int32))
        return self

    def distance(self, image: np.ndarray) -> float:
        """Chi-square distance; lower means more similar. Not a probability."""
        _, dist = self.rec.predict(np.ascontiguousarray(image, dtype=np.uint8))
        return float(dist)

    def histograms(self) -> np.ndarray:
        return np.stack(self.rec.getHistograms())

    def save(self, path: Path) -> None:
        """OpenCV recognisers are not reliably picklable; use their own format."""
        self.rec.write(str(path))

    @classmethod
    def load(cls, path: Path) -> "LbphModel":
        m = cls()
        m.rec.read(str(path))
        return m
