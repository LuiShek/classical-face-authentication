"""Cache guard: a stale or mislabelled cache must fail, not score silently."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from faceauth import cache, config


def _write(tmp_path: Path, name: str, **arrays) -> None:
    (tmp_path / "prepared").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tmp_path / "prepared" / name, **arrays)


def test_raw_cache_loads(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA", tmp_path)
    imgs = np.zeros((4, 112, 92), dtype=np.uint8)
    _write(tmp_path, "orl.npz", images=imgs,
           labels=np.array([0, 0, 1, 1], dtype=np.int32), stage="raw")
    images, labels = cache.load_cache("orl")
    assert images.shape == imgs.shape
    assert labels.tolist() == [0, 0, 1, 1]


def test_non_raw_stage_is_refused(tmp_path, monkeypatch):
    """This is the exact failure the guard exists for: CLAHE baked into a cache."""
    monkeypatch.setattr(config, "DATA", tmp_path)
    _write(tmp_path, "orl.npz", images=np.zeros((2, 112, 92), np.uint8),
           labels=np.zeros(2, np.int32), stage="clahe-baked")
    with pytest.raises(SystemExit, match="stage=clahe-baked"):
        cache.load_cache("orl")


def test_missing_stage_is_refused(tmp_path, monkeypatch):
    """Caches predating the marker are unknown provenance, not implicitly raw."""
    monkeypatch.setattr(config, "DATA", tmp_path)
    _write(tmp_path, "orl.npz", images=np.zeros((2, 112, 92), np.uint8),
           labels=np.zeros(2, np.int32))
    with pytest.raises(SystemExit, match="stage=unknown"):
        cache.load_cache("orl")


def test_unknown_lfw_variant_names_the_available_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA", tmp_path)
    _write(tmp_path, "lfw.npz",
           aligned=np.zeros((2, 112, 112), np.uint8),
           boxed=np.zeros((2, 112, 112), np.uint8),
           labels=np.zeros(2, np.int32), stage="raw")
    with pytest.raises(SystemExit, match="aligned_refined"):
        cache.load_cache("lfw", variant="aligned_refined")


def test_missing_cache_points_at_the_builder(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA", tmp_path)
    with pytest.raises(SystemExit, match="prepare_data"):
        cache.load_cache("orl")
