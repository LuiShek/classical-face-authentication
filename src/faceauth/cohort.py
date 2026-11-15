"""Encrypted, labeled background cohort for per-user calibration.

Registered users are not a practical cohort for a two-person installation. This
module samples distinct identities from the already prepared LFW cache and
stores only their aligned raw crops under Windows DPAPI. Labels are retained so
the calibration gate counts identities, not merely photographs.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import cache, config, secure

DATA_FILE = "background.npz.dpapi"
META_FILE = "meta.json"


@dataclass(frozen=True)
class Background:
    crops: np.ndarray
    labels: np.ndarray
    meta: dict

    @property
    def identities(self) -> int:
        return int(len(np.unique(self.labels)))


def exists() -> bool:
    return (config.COHORT / DATA_FILE).exists()


def _payload(crops: np.ndarray, labels: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, crops=np.asarray(crops, dtype=np.uint8),
                        labels=np.asarray(labels, dtype=np.int32))
    return buf.getvalue()


def build(*, identities: int = 30, per_identity: int = 3,
          seed: int = config.SEED, force: bool = False) -> Background:
    if exists() and not force:
        raise SystemExit("error: a background cohort already exists; use --force "
                         "to replace it")
    images, labels = cache.load_cache("lfw", "aligned")
    available = np.array(sorted(int(v) for v in np.unique(labels)), dtype=np.int32)
    eligible = np.array([v for v in available
                         if int(np.sum(labels == v)) >= per_identity], dtype=np.int32)
    if identities < config.MIN_COHORT_IDENTITIES:
        raise SystemExit(f"error: cohort needs at least {config.MIN_COHORT_IDENTITIES} "
                         f"identities, got {identities}")
    if identities > len(eligible):
        raise SystemExit(f"error: requested {identities} identities but only "
                         f"{len(eligible)} have {per_identity} images")

    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(eligible, size=identities, replace=False))
    crops, out_labels = [], []
    for new_label, original in enumerate(selected):
        idx = np.flatnonzero(labels == original)
        chosen = np.sort(rng.choice(idx, size=per_identity, replace=False))
        crops.extend(images[chosen])
        out_labels.extend([new_label] * per_identity)
    crop_array = np.stack(crops).astype(np.uint8)
    label_array = np.asarray(out_labels, dtype=np.int32)

    source = cache.path_for("lfw")
    meta = {
        "storage": "dpapi-v1",
        "source": "Kaggle jessicali9530/lfw-dataset (deep-funnelled subset)",
        "source_cache_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "variant": "aligned",
        "stage": "raw",
        "identities": identities,
        "per_identity": per_identity,
        "crops": int(len(crop_array)),
        "seed": seed,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    config.COHORT.mkdir(parents=True, exist_ok=True)
    secure.atomic_protected_write(config.COHORT / DATA_FILE,
                                  _payload(crop_array, label_array))
    tmp = config.COHORT / (META_FILE + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(config.COHORT / META_FILE)
    return Background(crop_array, label_array, meta)


def load(required: bool = False) -> Background | None:
    path = config.COHORT / DATA_FILE
    if not path.exists():
        if required:
            raise SystemExit("error: no background cohort; run "
                             "'faceauth cohort build'")
        return None
    raw = secure.protected_read(path)
    z = np.load(io.BytesIO(raw), allow_pickle=False)
    meta_path = config.COHORT / META_FILE
    meta = (json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.exists() else {})
    bg = Background(z["crops"], z["labels"], meta)
    if bg.crops.ndim != 3 or len(bg.crops) != len(bg.labels):
        raise SystemExit(f"error: malformed background cohort in {path}")
    return bg
