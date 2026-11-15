"""Persistence for enrolled faces.

Layout per identity, under ``config.FACES`` (outside the repository, because an
enrolled model is biometric data)::

    <slug>/model.yml.dpapi   DPAPI-protected OpenCV LBPH bytes
    <slug>/crops.npz.dpapi   DPAPI-protected enrolment crops
    <slug>/meta.json         non-biometric threshold/calibration evidence

Keeping the crops means ``calibrate`` never has to ask the user to sit in front
of the camera again, and keeps the LBPH model and its recorded evidence from
drifting apart.
"""
from __future__ import annotations

import json
import io
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config, lbph, secure

MODEL_ENC = "model.yml.dpapi"
CROPS_ENC = "crops.npz.dpapi"

SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
RESERVED = {".", "..", "con", "prn", "aux", "nul"}


def slugify(name: str) -> str:
    """Filesystem-safe identity key.

    Spaces, case and punctuation collapse to dashes, but path separators are
    *rejected* rather than rewritten: silently turning ``a/b`` into ``a-b`` would
    let a mistyped name land on a different person's enrolment.
    """
    if any(ch in name for ch in ("/", "\\")):
        raise SystemExit(f"error: {name!r} contains a path separator")
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    if not slug or slug in RESERVED or not SLUG.match(slug):
        raise SystemExit(f"error: {name!r} is not a usable identity name "
                         f"(use letters, digits, spaces or '-')")
    return slug


def model_dir(name: str) -> Path:
    return config.FACES / slugify(name)


@dataclass(frozen=True)
class Stored:
    name: str
    model: lbph.LbphModel
    crops: np.ndarray
    meta: dict

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"])

    @property
    def scale(self) -> str:
        return str(self.meta.get("scale", "raw"))


def exists(name: str) -> bool:
    d = model_dir(name)
    return (d / MODEL_ENC).exists() or (d / "model.yml").exists()


def _npz_bytes(crops: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, crops=crops)
    return buf.getvalue()


def _cleanup_model_temps() -> None:
    """Remove crash remnants before creating another plaintext OpenCV temp."""
    if not config.LOCAL.exists():
        return
    cutoff = time.time() - 300              # do not race another live process
    for path in config.LOCAL.glob("faceauth-model-*.yml"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except FileNotFoundError:
            pass


def _model_bytes(model: lbph.LbphModel) -> bytes:
    config.LOCAL.mkdir(parents=True, exist_ok=True)
    _cleanup_model_temps()
    fd, raw_path = tempfile.mkstemp(prefix="faceauth-model-", suffix=".yml",
                                    dir=config.LOCAL)
    os.close(fd)
    Path(raw_path).unlink(missing_ok=True)  # OpenCV creates the file itself
    try:
        model.save(Path(raw_path))
        return Path(raw_path).read_bytes()
    finally:
        Path(raw_path).unlink(missing_ok=True)


def _model_from_bytes(data: bytes) -> lbph.LbphModel:
    config.LOCAL.mkdir(parents=True, exist_ok=True)
    _cleanup_model_temps()
    fd, raw_path = tempfile.mkstemp(prefix="faceauth-model-", suffix=".yml",
                                    dir=config.LOCAL)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return lbph.LbphModel.load(Path(raw_path))
    finally:
        Path(raw_path).unlink(missing_ok=True)


def save(name: str, model: lbph.LbphModel, crops: np.ndarray,
         calibration: dict) -> Path:
    """Write the model, its crops and its calibration evidence together."""
    d = model_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    crops = np.ascontiguousarray(np.asarray(crops, dtype=np.uint8))
    if crops.ndim != 3:
        raise ValueError(f"crops must be [n, h, w], got {crops.shape}")
    secure.atomic_protected_write(d / MODEL_ENC, _model_bytes(model))
    secure.atomic_protected_write(d / CROPS_ENC, _npz_bytes(crops))
    meta = {
        "name": name,
        "slug": d.name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "takes": int(len(crops)),
        "size": int(crops.shape[1]),
        "storage": "dpapi-v1",
        **calibration,
    }
    # Write-then-replace: a half-written meta.json would leave a model whose
    # threshold cannot be read, which is worse than no model at all.
    tmp = d / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(d / "meta.json")
    return d


def load(name: str) -> Stored:
    d = model_dir(name)
    model_enc, model_plain = d / MODEL_ENC, d / "model.yml"
    if not model_enc.exists() and not model_plain.exists():
        raise SystemExit(f"error: no enrolled identity {name!r}; "
                         f"run 'enroll' first or see 'list'")
    meta_path = d / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"error: {d} has a model but no meta.json (threshold "
                         f"unknown); re-enrol to rebuild it")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    crops_enc, crops_plain = d / CROPS_ENC, d / "crops.npz"
    if crops_enc.exists():
        crops = np.load(io.BytesIO(secure.protected_read(crops_enc)),
                        allow_pickle=False)["crops"]
    elif crops_plain.exists():
        crops = np.load(crops_plain, allow_pickle=False)["crops"]
    else:
        raise SystemExit(f"error: {d} has no enrolment crops")
    model = (_model_from_bytes(secure.protected_read(model_enc))
             if model_enc.exists() else lbph.LbphModel.load(model_plain))
    return Stored(name=meta.get("name", name), model=model,
                  crops=crops, meta=meta)


def plaintext_files(name: str) -> list[Path]:
    d = model_dir(name)
    return [p for p in (d / "model.yml", d / "crops.npz") if p.exists()]


def migrate(name: str, *, remove_plaintext: bool = False) -> list[Path]:
    """Create verified DPAPI copies of a legacy profile; optionally remove raw.

    Removal is deliberately separate from encryption so callers can preview and
    verify the recoverable encrypted copies before approving plaintext removal.
    """
    d = model_dir(name)
    plain = plaintext_files(name)
    if not plain:
        return []
    model_plain, crops_plain = d / "model.yml", d / "crops.npz"
    if model_plain.exists():
        secure.atomic_protected_write(d / MODEL_ENC, model_plain.read_bytes())
    if crops_plain.exists():
        secure.atomic_protected_write(d / CROPS_ENC, crops_plain.read_bytes())
    # Full application-level verification, not just a DPAPI byte round-trip.
    loaded = load(name)
    if loaded.crops.ndim != 3 or not np.isfinite(loaded.threshold):
        raise RuntimeError(f"migrated profile {name!r} failed verification")
    meta_path = d / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["storage"] = "dpapi-v1"
    tmp = d / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(meta_path)
    if remove_plaintext:
        for p in plain:
            p.unlink()
    return plain


def restore_plaintext(name: str) -> list[Path]:
    """Restore exact plaintext files while retaining encrypted rollback copies."""
    d = model_dir(name)
    pairs = [(d / MODEL_ENC, d / "model.yml"),
             (d / CROPS_ENC, d / "crops.npz")]
    if not all(enc.exists() for enc, _ in pairs):
        raise SystemExit(f"error: {name!r} has no complete encrypted profile")
    restored = []
    for enc, raw in pairs:
        data = secure.protected_read(enc)
        tmp = raw.with_name(raw.name + ".tmp")
        tmp.write_bytes(data)
        if tmp.read_bytes() != data:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"plaintext restore verification failed for {raw}")
        tmp.replace(raw)
        restored.append(raw)
    # Verify both formats through the normal loader (which still prefers DPAPI).
    loaded = load(name)
    if loaded.crops.ndim != 3:
        raise RuntimeError(f"restored profile {name!r} failed verification")
    return restored


def profiles() -> list[dict]:
    if not config.FACES.exists():
        return []
    out = []
    for d in sorted(p for p in config.FACES.iterdir() if p.is_dir()):
        meta_path = d / "meta.json"
        if not meta_path.exists():
            out.append({"name": d.name, "slug": d.name, "broken": True})
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            out.append({"name": d.name, "slug": d.name, "broken": True})
            continue
        out.append(meta)
    return out


def others(name: str) -> list[Stored]:
    """Every other enrolled identity that loads cleanly.

    A broken profile is skipped rather than fatal: one corrupt folder must not
    make calibration impossible for everybody else.
    """
    me = slugify(name)
    out = []
    for p in profiles():
        if p.get("broken") or p.get("slug") == me:
            continue
        try:
            out.append(load(p["name"]))
        except SystemExit:
            continue
    return out


def delete(name: str) -> Path:
    d = model_dir(name)
    if not d.exists():
        raise SystemExit(f"error: no enrolled identity {name!r}")
    shutil.rmtree(d)
    return d
