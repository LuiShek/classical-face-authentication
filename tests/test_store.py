"""Enrolment storage: names cannot become paths, and models round-trip."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from faceauth import config, gate, secure, store


def crops(seed: int = 0, n: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (n, config.IMG_SIZE, config.IMG_SIZE),
                        dtype=np.uint8)


@pytest.fixture(autouse=True)
def isolated_faces(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOCAL", tmp_path)
    monkeypatch.setattr(config, "FACES", tmp_path / "faces")
    return tmp_path / "faces"


def enrol(name: str, seed: int = 0) -> store.Stored:
    c = crops(seed)
    cal = gate.calibrate(c, crops(seed + 100))
    model = gate.Template(c).model
    store.save(name, model, c, cal.as_meta())
    return store.load(name)


def test_slugify_keeps_names_readable():
    assert store.slugify("Alice Example") == "alice-example"
    assert store.slugify("  Test.User  ") == "test-user"


@pytest.mark.parametrize("name", ["..", "/", "", "   ", "CON", "a/b", "a\\b"])
def test_names_that_could_escape_the_store_are_rejected(name):
    """A name is never allowed to become part of a path."""
    with pytest.raises(SystemExit):
        store.model_dir(name)


def test_traversal_attempt_is_rejected_rather_than_rewritten():
    with pytest.raises(SystemExit, match="path separator"):
        store.model_dir("../../etc/passwd")


def test_save_and_load_round_trip():
    s = enrol("alice", seed=1)
    assert s.name == "alice"
    assert s.crops.shape == (5, config.IMG_SIZE, config.IMG_SIZE)
    assert np.isfinite(s.threshold)
    assert s.meta["calibration"]["method"] == "loo+far0"


def test_save_leaves_no_temporary_file():
    enrol("temp-check", seed=2)
    d = store.model_dir("temp-check")
    assert (d / "meta.json").exists()
    assert not (d / "meta.json.tmp").exists()
    assert list(config.LOCAL.glob("faceauth-model-*.yml")) == []


def test_crash_remnant_is_removed_before_next_secure_save():
    config.LOCAL.mkdir(parents=True, exist_ok=True)
    stale = config.LOCAL / "faceauth-model-stale.yml"
    stale.write_text("sensitive stale model", encoding="utf-8")
    import os, time
    os.utime(stale, (time.time() - 600, time.time() - 600))
    enrol("cleanup", seed=32)
    assert not stale.exists()


def test_every_profile_has_the_three_files():
    enrol("layout", seed=3)
    d = store.model_dir("layout")
    assert sorted(p.name for p in d.iterdir()) == ["crops.npz.dpapi", "meta.json",
                                                   "model.yml.dpapi"]
    assert not (d / "crops.npz").exists()
    assert not (d / "model.yml").exists()


def test_encrypted_files_really_decrypt_to_the_expected_formats():
    s = enrol("encrypted", seed=33)
    d = store.model_dir("encrypted")
    crop_bytes = secure.protected_read(d / store.CROPS_ENC)
    model_bytes = secure.protected_read(d / store.MODEL_ENC)
    assert crop_bytes.startswith(b"PK")       # compressed npz/zip
    assert b"opencv_lbphfaces" in model_bytes
    assert s.meta["storage"] == "dpapi-v1"


def test_load_of_unknown_identity_explains_the_next_step():
    with pytest.raises(SystemExit, match="enroll"):
        store.load("nobody")


def test_others_excludes_the_named_identity():
    enrol("first", seed=4)
    enrol("second", seed=5)
    assert [o.name for o in store.others("first")] == ["second"]
    assert [o.name for o in store.others("second")] == ["first"]


def test_a_broken_profile_is_skipped_not_fatal():
    enrol("good", seed=6)
    broken = store.model_dir("broken")
    broken.mkdir(parents=True)
    (broken / "meta.json").write_text("{not json", encoding="utf-8")
    names = [o.name for o in store.others("good")]
    assert names == []
    rows = store.profiles()
    assert any(r.get("broken") for r in rows)


def test_delete_removes_the_directory():
    enrol("doomed", seed=7)
    d = store.delete("doomed")
    assert not d.exists()
    assert not store.exists("doomed")


def test_delete_of_unknown_identity_raises():
    with pytest.raises(SystemExit):
        store.delete("ghost")


def test_meta_is_json_with_the_evidence_needed_to_justify_the_threshold():
    enrol("evidence", seed=8)
    meta = json.loads((store.model_dir("evidence") / "meta.json")
                      .read_text(encoding="utf-8"))
    for key in ("threshold", "scale", "takes", "created", "calibration"):
        assert key in meta
    assert meta["calibration"]["n_genuine"] == 5


def test_legacy_plaintext_can_be_encrypted_before_it_is_removed():
    enrol("legacy", seed=40)
    d = store.model_dir("legacy")
    # Recreate the old layout from the verified encrypted bytes.
    (d / "model.yml").write_bytes(secure.protected_read(d / store.MODEL_ENC))
    (d / "crops.npz").write_bytes(secure.protected_read(d / store.CROPS_ENC))
    (d / store.MODEL_ENC).unlink()
    (d / store.CROPS_ENC).unlink()

    plan = store.plaintext_files("legacy")
    assert sorted(p.name for p in plan) == ["crops.npz", "model.yml"]
    store.migrate("legacy", remove_plaintext=False)
    assert store.load("legacy").crops.shape[0] == 5
    assert all(p.exists() for p in plan), "preview migration retains plaintext"

    store.migrate("legacy", remove_plaintext=True)
    assert not any(p.exists() for p in plan)
    assert store.load("legacy").crops.shape[0] == 5


def test_encrypted_profile_can_be_restored_without_losing_ciphertext():
    enrol("rollback", seed=41)
    d = store.model_dir("rollback")
    restored = store.restore_plaintext("rollback")
    assert sorted(p.name for p in restored) == ["crops.npz", "model.yml"]
    assert (d / store.MODEL_ENC).exists() and (d / store.CROPS_ENC).exists()
    assert store.load("rollback").crops.shape[0] == 5
