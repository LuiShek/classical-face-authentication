"""Turning frames into usable crops, with the quality gate made explicit.

Enrolment and verification must apply the *same* transforms, otherwise the
threshold measured at enrolment means nothing at verification time. Both paths
therefore go through :func:`assess`, which is the only place that decides what a
usable frame is.

Rejections and warnings are deliberately separate. A frame containing a
background face is usable - the largest face wins, deterministically - but the
operator still gets told, because silently scoring "whichever face was biggest"
is the kind of behaviour that hides a bad capture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable

import cv2
import numpy as np

from . import config, detect, preprocess


@dataclass(frozen=True)
class Assessment:
    crop: np.ndarray | None
    reasons: tuple[str, ...] = ()        # non-empty means unusable
    warnings: tuple[str, ...] = ()       # usable, but worth saying out loud
    face_area: float = 0.0
    blur: float = 0.0

    @property
    def ok(self) -> bool:
        return self.crop is not None and not self.reasons


@dataclass
class CaptureResult:
    crops: list[np.ndarray]
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.crops)


def blur_score(gray: np.ndarray) -> float:
    """Laplacian variance: low means soft focus or motion blur."""
    return float(cv2.Laplacian(np.ascontiguousarray(gray), cv2.CV_64F).var())


def assess(frame: np.ndarray, detector: detect.Detector,
           clahe: bool = False) -> Assessment:
    """Detect the largest face and crop it, or explain why it was rejected."""
    faces = detector.detect(frame)
    if not faces:
        return Assessment(None, ("no face detected",))
    face = max(faces, key=lambda f: f.area)
    warnings: list[str] = []
    if len(faces) > 1:
        # Not fatal: the largest face is a deterministic choice and background
        # faces are common in photographs. Refusing outright would throw away
        # usable probes; staying silent would hide an ambiguous capture.
        warnings.append(f"{len(faces)} faces in frame; using the largest")
    if face.box[2] < config.MIN_FACE_PX or face.box[3] < config.MIN_FACE_PX:
        return Assessment(None, (f"face too small ({face.box[2]:.0f}x"
                                 f"{face.box[3]:.0f} < {config.MIN_FACE_PX})",),
                          tuple(warnings), face_area=face.area)

    gray = preprocess.to_gray(frame)
    try:
        crop = preprocess.aligned_crop_raw(gray, face, config.IMG_SIZE)
    except ValueError as exc:
        return Assessment(None, (f"alignment failed: {exc}",), tuple(warnings),
                          face_area=face.area)

    blur = blur_score(crop)
    reasons: list[str] = []
    if blur < config.BLUR_MIN_VAR:
        reasons.append(f"too blurred (laplacian var {blur:.0f} < "
                       f"{config.BLUR_MIN_VAR:.0f})")

    usable = preprocess.equalise(crop) if clahe else crop
    return Assessment(usable, tuple(reasons), tuple(warnings),
                      face_area=face.area, blur=blur)


def collect(source, detector: detect.Detector, takes: int, *, clahe: bool = False,
            max_attempts: int | None = None,
            on_reject: Callable[[int, Assessment], None] | None = None,
            on_warn: Callable[[str], None] | None = None,
            on_accept: Callable[[int, Assessment], None] | None = None,
            live_interval: float = 0.0) -> CaptureResult:
    """Pull usable crops from a source, skipping frames the gate rejects.

    ``max_attempts`` exists so a camera pointing at a wall fails with a reason
    instead of looping forever; a directory source runs out of frames on its own.
    """
    result = CaptureResult(crops=[])
    attempts = 0
    next_live_take = 0.0
    limit = max_attempts if max_attempts is not None else max(30, takes * 30)
    while len(result.crops) < takes and attempts < limit:
        frame = source.read()
        if frame is None:
            break
        # Drain the live stream instead of sleeping. Sleeping can let a camera
        # backend queue old frames, after which the next "take" is still the
        # same instant. Directory/image sources are never discarded.
        now = time.monotonic()
        if (live_interval > 0 and getattr(source, "live", False)
                and now < next_live_take):
            continue
        attempts += 1
        a = assess(frame, detector, clahe=clahe)
        for w in a.warnings:
            if w not in result.warnings:
                result.warnings.append(w)
                if on_warn is not None:
                    on_warn(w)
        if a.ok:
            result.crops.append(a.crop)
            if on_accept is not None:
                on_accept(len(result.crops), a)
            if (live_interval > 0 and getattr(source, "live", False)
                    and len(result.crops) < takes):
                next_live_take = now + live_interval
            continue
        result.problems.append("; ".join(a.reasons))
        if on_reject is not None:
            on_reject(attempts, a)
    return result
