"""Build the evaluation caches: ORL (pre-cropped) and LFW (needs detection).

Why this does not use sklearn's fetchers
----------------------------------------
``fetch_olivetti_faces`` and ``fetch_lfw_people`` download from figshare, which
answers this machine with HTTP 403 (the download runs through ``urlretrieve``
with no User-Agent). Both datasets are therefore fetched from sources that were
verified reachable here:

  ORL  https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip
  LFW  kaggle jessicali9530/lfw-dataset  (the UMass host does not resolve)

ORL images are already cropped to the face by hand, at their native 92x112.
Running a detector on them would measure the detector on images it did not have
to find, and resampling them to square would distort the texture LBPH reads. So
ORL bypasses detection and keeps its native geometry.

LFW images are in the wild, so they carry the detection and alignment question.
Both crops are cached at once - landmark-aligned, and a detector-box crop with
no rotation - so the alignment ablation needs no re-run.

Everything cached here is *raw* greyscale: no CLAHE, no histogram work. Lighting
normalisation is a model choice, so it happens at scoring time where it can be
ablated without re-running detection over 3,000 images. The ``stage`` marker in
each cache is checked by the evaluator, so a cache built by an older version
fails loudly instead of silently reporting CLAHE numbers as raw ones.

Usage
-----
  py scripts/prepare_data.py --dataset orl
  py scripts/prepare_data.py --dataset lfw
  py scripts/prepare_data.py --dataset lfw --refine   # extra cache variant
  py scripts/prepare_data.py --dataset all
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from faceauth import config, detect, preprocess

ORL_URL = "https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip"
LFW_KAGGLE = "jessicali9530/lfw-dataset"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) faceauth/0.1"}
LFW_MIN_PER_IDENTITY = 20          # matches the usual "LFW people" subset
IMG_EXT = {".jpg", ".jpeg", ".png", ".pgm"}


def out_dir() -> Path:
    d = config.DATA / "prepared"
    d.mkdir(parents=True, exist_ok=True)
    return d


def raw_dir() -> Path:
    d = config.DATA / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())


# ---------------------------------------------------------------- ORL

def ensure_orl() -> Path:
    root = raw_dir() / "orl"
    if len(list(root.rglob("*.pgm"))) >= 400 if root.exists() else False:
        return root
    zip_path = raw_dir() / "att_faces.zip"
    if not zip_path.exists():
        print(f"downloading ORL from {ORL_URL} ...", flush=True)
        download(ORL_URL, zip_path)
    if zip_path.stat().st_size < 3_000_000:
        raise SystemExit(f"error: {zip_path} is only {zip_path.stat().st_size} B")
    print(f"extracting {zip_path.name} ...", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(root)
    return root


def _natural(p: Path) -> tuple:
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name))


def prepare_orl() -> None:
    t0 = time.time()
    root = ensure_orl()
    subject_dirs = sorted([d for d in root.iterdir() if d.is_dir()], key=_natural)
    if len(subject_dirs) != config.ORL_IDENTITIES:
        raise SystemExit(f"error: expected {config.ORL_IDENTITIES} subjects, "
                         f"found {len(subject_dirs)}")

    images, labels = [], []
    for label, d in enumerate(subject_dirs):
        files = sorted(d.glob("*.pgm"), key=_natural)
        if len(files) != config.ORL_IMAGES_PER_ID:
            raise SystemExit(f"error: {d.name} has {len(files)} images")
        for f in files:
            # PGM is already single-channel, so this is a straight copy: raw,
            # native geometry, no detector, no equalisation.
            images.append(np.ascontiguousarray(np.array(Image.open(f)),
                                               dtype=np.uint8))
            labels.append(label)

    arr = np.stack(images)
    lab = np.asarray(labels, dtype=np.int32)
    path = out_dir() / "orl.npz"
    np.savez_compressed(path, images=arr, labels=lab, dataset="orl",
                        stage="raw", native_size=f"{arr.shape[2]}x{arr.shape[1]}")
    print(f"ORL  : {arr.shape[0]} images, {len(np.unique(lab))} identities, "
          f"native {arr.shape[2]}x{arr.shape[1]} (no resize, no detector)")
    print(f"       -> {path}  ({path.stat().st_size / 1e6:.2f} MB, "
          f"{time.time() - t0:.1f}s)")


# ---------------------------------------------------------------- LFW

def ensure_lfw() -> Path:
    root = raw_dir() / "lfw_dl"
    root.mkdir(parents=True, exist_ok=True)
    found = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT]
    if found:
        return root
    print(f"downloading LFW via kaggle ({LFW_KAGGLE}) ...", flush=True)
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", LFW_KAGGLE,
                        "-p", str(root), "--unzip"], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            f"error: kaggle download failed ({exc}). Install and authenticate "
            f"the kaggle CLI, or drop any LFW layout into {root}.") from exc
    return root


def _group_by_identity(paths: list[Path]) -> dict[str, list[Path]]:
    """Handle both lfw/<name>/<name>_0001.jpg and a flat <name>_0001.jpg layout."""
    by_parent: dict[str, list[Path]] = {}
    for p in paths:
        by_parent.setdefault(p.parent.name, []).append(p)
    if max(len(v) for v in by_parent.values()) > 1:
        return by_parent
    by_stem: dict[str, list[Path]] = {}
    for p in paths:
        m = re.match(r"^(.+?)_\d+$", p.stem)
        by_stem.setdefault(m.group(1) if m else p.stem, []).append(p)
    return by_stem


def prepare_lfw(refine: bool = False) -> None:
    t0 = time.time()
    root = ensure_lfw()
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)
    groups = _group_by_identity(paths)
    identities = sorted(k for k, v in groups.items()
                        if len(v) >= LFW_MIN_PER_IDENTITY)
    if not identities:
        raise SystemExit(f"error: no identity has >= {LFW_MIN_PER_IDENTITY} "
                         f"images in {root}")
    print(f"LFW  : {len(paths)} images, {len(groups)} raw identities, "
          f"{len(identities)} with >= {LFW_MIN_PER_IDENTITY} images", flush=True)

    det = detect.Detector()
    aligned, boxed, refined, lab, names = [], [], [], [], []
    failed = 0
    total = sum(len(groups[k]) for k in identities)
    done = 0
    for label, name in enumerate(identities):
        for f in sorted(groups[name]):
            done += 1
            bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if bgr is None:
                failed += 1
                continue
            face = det.largest(bgr)
            if face is None:
                failed += 1
                continue
            gray = preprocess.to_gray(bgr)
            # Compute every crop before appending any of them: a failure part
            # way through used to leave the parallel arrays at different
            # lengths, which would misalign labels against images.
            try:
                a = preprocess.aligned_crop_raw(gray, face)
                b = preprocess.boxed_crop_raw(gray, face)
                r = (preprocess.refine_alignment(det, gray, face.pts, iters=1)
                     if refine else None)
            except ValueError:
                failed += 1
                continue
            aligned.append(a)
            boxed.append(b)
            if refine:
                refined.append(r)
            lab.append(label)
            names.append(name)
            if done % 400 == 0:
                print(f"       {done}/{total} processed, {failed} failed",
                      flush=True)

    a = np.stack(aligned)
    b = np.stack(boxed)
    labels = np.asarray(lab, dtype=np.int32)
    path = out_dir() / "lfw.npz"
    payload = dict(aligned=a, boxed=b, labels=labels,
                   names=np.asarray(names), dataset="lfw", stage="raw",
                   size=a.shape[1])
    if refine:
        payload["aligned_refined"] = np.stack(refined)
    np.savez_compressed(path, **payload)

    rate = len(a) / total
    print(f"       detection: {len(a)}/{total} ({rate * 100:.1f}%), "
          f"{failed} failed, {time.time() - t0:.0f}s")
    print(f"       -> {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    if rate < 0.95:
        print(f"warning: detection rate {rate * 100:.1f}% is low for 250px "
              f"faces; check the detector threshold before trusting the numbers")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["orl", "lfw", "all"], default="all")
    ap.add_argument("--refine", action="store_true",
                    help="LFW only: also cache a re-detected, re-warped crop")
    args = ap.parse_args()
    if args.dataset in ("orl", "all"):
        prepare_orl()
    if args.dataset in ("lfw", "all"):
        prepare_lfw(refine=args.refine)
    return 0


if __name__ == "__main__":
    sys.exit(main())
