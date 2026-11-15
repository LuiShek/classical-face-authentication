"""Face detection: YuNet wrapper.

OpenCV 5.0 ships no Haar cascades, so YuNet is the primary detector. It also
returns five landmarks, which is what makes alignment possible at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config


@dataclass(frozen=True)
class Face:
    box: tuple[float, float, float, float]   # x, y, w, h
    pts: np.ndarray                          # 5x2 landmarks
    score: float

    @property
    def area(self) -> float:
        return self.box[2] * self.box[3]


class Detector:
    """Thin wrapper so callers do not juggle input-size state themselves."""

    def __init__(self, model_path=None, score_threshold: float = 0.6,
                 nms: float = 0.3) -> None:
        path = str(model_path or config.YUNET)
        if not config.YUNET.exists():
            raise SystemExit(f"error: missing {path}; run scripts/fetch_assets.py")
        self._d = cv2.FaceDetectorYN.create(path, "", (320, 320),
                                            score_threshold, nms, 5000)
        self._size: tuple[int, int] | None = None

    def detect(self, frame: np.ndarray) -> list[Face]:
        h, w = frame.shape[:2]
        if self._size != (w, h):
            self._d.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self._d.detect(frame)
        if faces is None:
            return []
        return [Face(box=tuple(float(v) for v in f[:4]),
                     pts=np.asarray(f[4:14], dtype=np.float32).reshape(5, 2),
                     score=float(f[14])) for f in faces]

    def largest(self, frame: np.ndarray) -> Face | None:
        faces = self.detect(frame)
        return max(faces, key=lambda f: f.area) if faces else None
