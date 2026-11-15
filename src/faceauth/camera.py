"""Frame sources: the webcam, or a directory of images for offline runs.

Keeping capture behind a tiny interface is what makes the rest of the gate
testable without a camera, and it is also how ``--from`` enrolment works.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".pgm"}


class FrameSource(Protocol):
    description: str
    live: bool

    def read(self) -> np.ndarray | None: ...
    def close(self) -> None: ...


class Camera:
    """Webcam capture. DirectShow first: MSMF is the OpenCV default and flaky."""

    def __init__(self, index: int = 0, warmup: int = 5) -> None:
        self.live = True
        self.index = index
        self._cap: cv2.VideoCapture | None = None
        self.description = f"camera {index}"
        for backend, name in ((cv2.CAP_DSHOW, "DSHOW"),
                              (cv2.CAP_MSMF, "MSMF"),
                              (cv2.CAP_ANY, "ANY")):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened() and cap.read()[0]:
                # Best effort: keep only the newest frame while enrolment waits
                # between takes. Some backends ignore this, so capture.collect
                # also drains frames during the interval.
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._cap = cap
                self.description = f"camera {index} via {name}"
                break
            cap.release()
        if self._cap is None:
            raise SystemExit(f"error: no camera at index {index} "
                             f"(tried DSHOW, MSMF, ANY)")
        for _ in range(warmup):          # let auto-exposure settle
            self._cap.read()

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class ImageDir:
    """Yields each image in a directory once, in name order."""

    def __init__(self, path: Path) -> None:
        self.live = False
        path = Path(path)
        if not path.exists():
            raise SystemExit(f"error: {path} does not exist")
        if path.is_file():
            self._paths = [path]
        else:
            self._paths = sorted(p for p in path.rglob("*")
                                 if p.suffix.lower() in IMG_EXT)
        if not self._paths:
            raise SystemExit(f"error: no images in {path}")
        self.description = (f"{self._paths[0]}" if len(self._paths) == 1
                            else f"{len(self._paths)} images from {path}")
        self._i = 0

    def read(self) -> np.ndarray | None:
        while self._i < len(self._paths):
            p = self._paths[self._i]
            self._i += 1
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                return img
            print(f"warning: could not read {p}", file=sys.stderr)
        return None

    def close(self) -> None:
        self._i = len(self._paths)


def open_source(index: int | None, from_path: Path | None) -> FrameSource:
    if from_path is not None:
        return ImageDir(from_path)
    return Camera(index if index is not None else 0)
