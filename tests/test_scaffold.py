"""Core installation checks: LBPH, asset availability, and safe data paths."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from faceauth import config


def test_lbph_is_available():
    """The entire approach depends on this symbol surviving OpenCV upgrades."""
    assert hasattr(cv2, "face"), "opencv-contrib-python required, not opencv-python"
    rec = cv2.face.LBPHFaceRecognizer_create(
        radius=config.LBPH_RADIUS, neighbors=config.LBPH_NEIGHBORS,
        grid_x=config.LBPH_GRID_X, grid_y=config.LBPH_GRID_Y)
    imgs = [np.random.randint(0, 255, (80, 80), dtype=np.uint8) for _ in range(4)]
    rec.train(imgs, np.array([0, 0, 1, 1]))
    label, dist = rec.predict(imgs[0])
    assert label == 0
    assert dist >= 0.0
    assert rec.getHistograms()[0].size == 8 * 8 * 256


def test_data_and_biometrics_live_outside_one_drive():
    for p in (config.DATA, config.FACES):
        assert "OneDrive" not in str(p), f"{p} would be synced"
        assert config.ROOT not in p.parents, f"{p} is inside the repo"


def test_preprocessing_constants_are_sane():
    assert config.IMG_SIZE >= config.MIN_FACE_PX
    assert config.LBPH_GRID_X * config.LBPH_GRID_Y * 256 == 16384


@pytest.mark.parametrize("asset", ["YUNET", "HAAR_FRONTAL", "HAAR_EYE"])
def test_assets_present(asset):
    p = getattr(config, asset)
    if not p.exists():
        pytest.skip(f"{p.name} not fetched; run scripts/fetch_assets.py")
    assert p.stat().st_size > 10_000
