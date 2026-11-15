"""Preprocessing: grayscale, CLAHE, box crop, 5-point alignment.

Every path (enrolment, verification, evaluation) must go through these exact
functions. If the evaluation crops differently from the live gate, the measured
threshold means nothing.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config

# Canonical 5-point template for a 112x112 crop, in YuNet's landmark order
# (subject-right eye, subject-left eye, nose tip, mouth right, mouth left).
CANONICAL_5PT = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalise(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP,
                            tileGridSize=config.CLAHE_GRID)
    return clahe.apply(np.ascontiguousarray(gray, dtype=np.uint8))


def prepare(img: np.ndarray, size: int | None = None) -> np.ndarray:
    """Grayscale -> optional square resize -> CLAHE. Returns uint8."""
    g = to_gray(img)
    if size is not None and g.shape[:2] != (size, size):
        g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    return equalise(g)


def crop_box(img: np.ndarray, box, size: int = config.IMG_SIZE) -> np.ndarray:
    """Axis-aligned crop of the detector box, resized. No rotation.

    This is the "unaligned" control in the ablation: same detector, same
    output size, only missing the landmark-based transform.
    """
    x, y, w, h = (int(round(v)) for v in box)
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        raise ValueError("degenerate detection box")
    return cv2.resize(img[y0:y1, x0:x1], (size, size),
                      interpolation=cv2.INTER_AREA)


def align(img: np.ndarray, pts: np.ndarray,
          size: int = config.IMG_SIZE) -> np.ndarray:
    """Similarity transform (rotation + scale + translation) onto the template.

    Partial affine: 4 degrees of freedom, so it cannot shear or distort the face
    to fit, which matters because LBPH compares texture at fixed positions.
    """
    src = np.asarray(pts, dtype=np.float32).reshape(5, 2)
    dst = CANONICAL_5PT * (size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        raise ValueError("alignment failed: degenerate landmarks")
    return cv2.warpAffine(np.ascontiguousarray(img), M, (size, size),
                          flags=cv2.INTER_LINEAR, borderValue=0)


def aligned_crop_raw(frame_or_gray: np.ndarray, face, size: int = config.IMG_SIZE) -> np.ndarray:
    """Align and crop, returning raw greyscale (no CLAHE).

    Photometric normalisation is deliberately *not* baked into the cache: it is
    a model choice, so it belongs at scoring time where it can be ablated
    without re-running detection over 3,000 images.
    """
    gray = to_gray(frame_or_gray)
    return np.ascontiguousarray(align(gray, face.pts, size), dtype=np.uint8)


def boxed_crop_raw(frame_or_gray: np.ndarray, face, size: int = config.IMG_SIZE) -> np.ndarray:
    """Unaligned control path, raw greyscale."""
    gray = to_gray(frame_or_gray)
    return np.ascontiguousarray(crop_box(gray, face.box, size), dtype=np.uint8)


def refine_alignment(detector, gray: np.ndarray, pts: np.ndarray,
                     size: int = config.IMG_SIZE, iters: int = 1) -> np.ndarray:
    """Align, re-detect on the warped crop, align again.

    The first warp leaves a residual of a few pixels (landmark noise plus the
    similarity constraint). Re-detecting on the already-frontal crop and
    composing a second transform shrinks that residual, which matters because
    LBPH compares texture at fixed grid positions.
    """
    img = align(gray, pts, size)
    for _ in range(max(0, iters)):
        f = detector.largest(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        if f is None:
            break
        img = align(img, f.pts, size)
    return np.ascontiguousarray(img, dtype=np.uint8)


def aligned_crop(frame: np.ndarray, face, size: int = config.IMG_SIZE) -> np.ndarray:
    """Aligned and photometrically normalised (live-path default)."""
    return prepare(aligned_crop_raw(frame, face, size), size=None)


def boxed_crop(frame: np.ndarray, face, size: int = config.IMG_SIZE) -> np.ndarray:
    """Unaligned control path, with the same photometric step."""
    return prepare(boxed_crop_raw(frame, face, size), size=None)
