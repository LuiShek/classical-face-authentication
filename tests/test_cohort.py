"""Background cohort: labeled sampling, DPAPI storage and reproducibility."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from faceauth import cache, cohort, config


@pytest.fixture
def fake_lfw(tmp_path, monkeypatch):
    rng = np.random.default_rng(123)
    labels = np.repeat(np.arange(12), 4).astype(np.int32)
    images = rng.integers(0, 256, (len(labels), 112, 112), dtype=np.uint8)
    source = tmp_path / "lfw.npz"
    source.write_bytes(b"immutable fake cache")
    monkeypatch.setattr(config, "COHORT", tmp_path / "cohort")
    monkeypatch.setattr(cache, "load_cache",
                        lambda dataset, variant: (images, labels))
    monkeypatch.setattr(cache, "path_for", lambda dataset: source)
    return images, labels, source


def test_build_is_encrypted_and_keeps_identity_labels(fake_lfw):
    _, _, source = fake_lfw
    bg = cohort.build(identities=10, per_identity=3, seed=7)
    assert bg.crops.shape == (30, 112, 112)
    assert bg.identities == 10
    assert all(np.sum(bg.labels == i) == 3 for i in range(10))
    encrypted = config.COHORT / cohort.DATA_FILE
    assert encrypted.exists()
    assert not encrypted.read_bytes().startswith(b"PK")
    assert bg.meta["source_cache_sha256"] == hashlib.sha256(
        source.read_bytes()).hexdigest()


def test_load_round_trips_the_encrypted_cohort(fake_lfw):
    expected = cohort.build(identities=10, per_identity=3, seed=8)
    actual = cohort.load(required=True)
    assert np.array_equal(actual.crops, expected.crops)
    assert np.array_equal(actual.labels, expected.labels)
    assert actual.meta["storage"] == "dpapi-v1"


def test_same_seed_is_reproducible(fake_lfw):
    first = cohort.build(identities=10, per_identity=3, seed=9)
    second = cohort.build(identities=10, per_identity=3, seed=9, force=True)
    assert np.array_equal(first.crops, second.crops)
    assert np.array_equal(first.labels, second.labels)


def test_too_few_identities_is_refused(fake_lfw):
    with pytest.raises(SystemExit, match="at least"):
        cohort.build(identities=config.MIN_COHORT_IDENTITIES - 1,
                     per_identity=3)


def test_existing_cohort_requires_force(fake_lfw):
    cohort.build(identities=10, per_identity=3)
    with pytest.raises(SystemExit, match="--force"):
        cohort.build(identities=10, per_identity=3)


def test_optional_load_returns_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "COHORT", tmp_path / "cohort")
    assert cohort.load() is None
    with pytest.raises(SystemExit, match="cohort build"):
        cohort.load(required=True)
