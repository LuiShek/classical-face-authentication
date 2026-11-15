"""Preprocessing invariants: geometry, dtype, and "raw really means raw"."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from faceauth import preprocess


@dataclass
class FakeFace:
    box: tuple[float, float, float, float]
    pts: np.ndarray


def synthetic(size: int = 160) -> np.ndarray:
    """A grey ramp plus a bright blob, so histogram operations show up."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    img = (x / size * 120 + 40).astype(np.uint8)
    blob = ((x - size / 2) ** 2 + (y - size / 2) ** 2) < (size / 6) ** 2
    img[blob] = 240
    return img


def face_at(size: int = 160) -> FakeFace:
    """Landmarks placed exactly on the canonical template, scaled to `size`."""
    scale = size / 112.0
    return FakeFace(box=(size * 0.25, size * 0.2, size * 0.5, size * 0.5),
                    pts=preprocess.CANONICAL_5PT * scale)


def test_to_gray_accepts_both_depths():
    gray = synthetic()
    assert preprocess.to_gray(gray).shape == gray.shape
    bgr = np.stack([gray, gray, gray], axis=-1)
    out = preprocess.to_gray(bgr)
    assert out.ndim == 2
    assert out.shape == gray.shape


def test_equalise_preserves_shape_and_dtype():
    img = synthetic()
    out = preprocess.equalise(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_align_with_canonical_points_is_near_identity():
    """Warping a face whose landmarks already sit on the template must not move it."""
    img = synthetic(112)
    pts = preprocess.CANONICAL_5PT.copy()
    warped = preprocess.align(img, pts, 112)
    assert warped.shape == (112, 112)
    # allow interpolation error, but nowhere near a visible shift
    assert np.abs(warped.astype(np.float32) - img.astype(np.float32)).mean() < 3.0


def test_align_rejects_degenerate_landmarks():
    img = synthetic(112)
    with pytest.raises(ValueError):
        preprocess.align(img, np.zeros((5, 2), dtype=np.float32), 112)


def test_crop_box_clamps_to_image_and_resizes():
    img = synthetic(160)
    # box deliberately runs past the right and bottom edges
    out = preprocess.crop_box(img, (140, 140, 60, 60), 112)
    assert out.shape == (112, 112)
    assert out.dtype == np.uint8


def test_crop_box_rejects_tiny_box():
    img = synthetic(160)
    with pytest.raises(ValueError):
        preprocess.crop_box(img, (10, 10, 2, 2), 112)


def test_raw_crop_is_not_equalised():
    """The cache contract depends on this: raw must stay untouched.

    If CLAHE were ever baked in here, every evaluation would silently score
    equalised pixels while claiming to compare raw against CLAHE.
    """
    img = synthetic(112)
    raw = preprocess.aligned_crop_raw(img, face_at(112))
    assert raw.dtype == np.uint8
    assert raw.shape == (112, 112)
    assert not np.array_equal(raw, preprocess.equalise(raw))
    # and the normalised convenience path is exactly equalise(raw)
    assert np.array_equal(preprocess.aligned_crop(img, face_at(112)),
                          preprocess.equalise(raw))


def test_boxed_crop_raw_matches_box_geometry():
    img = synthetic(160)
    raw = preprocess.boxed_crop_raw(img, face_at(160))
    expected = preprocess.crop_box(preprocess.to_gray(img),
                                   face_at(160).box, 112)
    assert np.array_equal(raw, expected)


class _NoDetection:
    def largest(self, frame):
        return None


def test_refine_alignment_without_a_redetection_returns_first_warp():
    img = synthetic(112)
    once = preprocess.align(img, preprocess.CANONICAL_5PT.copy(), 112)
    refined = preprocess.refine_alignment(_NoDetection(), img,
                                          preprocess.CANONICAL_5PT.copy(),
                                          size=112, iters=2)
    assert np.array_equal(refined, once)


def test_refine_alignment_is_idempotent_under_canonical_landmarks():
    """A crop already on the template must survive refinement unchanged."""
    img = synthetic(112)

    class Perfect:
        def largest(self, frame):
            return FakeFace(box=(0, 0, 112, 112),
                            pts=preprocess.CANONICAL_5PT.copy())

    refined = preprocess.refine_alignment(Perfect(), img,
                                          preprocess.CANONICAL_5PT.copy(),
                                          size=112, iters=1)
    assert np.abs(refined.astype(np.float32) - img.astype(np.float32)).mean() < 3.0
