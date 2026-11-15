"""Fetch detector assets that OpenCV 5 no longer bundles.

OpenCV 5.0.0 ships an empty ``cv2.data.haarcascades`` directory and no YuNet
model, so both are downloaded here once. They are committed to the repository
because they are small and not derived from anybody's face.

Usage
-----
  py scripts/fetch_assets.py
  py scripts/fetch_assets.py --force
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from faceauth import config

OPENCV_4X = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades"
OPENCV_ZOO = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
              "face_detection_yunet")

ASSETS = {
    config.YUNET.name: f"{OPENCV_ZOO}/{config.YUNET.name}",
    config.HAAR_FRONTAL.name: f"{OPENCV_4X}/{config.HAAR_FRONTAL.name}",
    config.HAAR_EYE.name: f"{OPENCV_4X}/{config.HAAR_EYE.name}",
}

MIN_BYTES = {
    config.YUNET.name: 100_000,
    config.HAAR_FRONTAL.name: 500_000,
    config.HAAR_EYE.name: 200_000,
}


def fetch(url: str, dest: Path, min_bytes: int, force: bool = False) -> str:
    if dest.exists() and not force and dest.stat().st_size >= min_bytes:
        return f"present  {dest.name}  ({dest.stat().st_size:,} B)"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    size = tmp.stat().st_size
    if size < min_bytes:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"error: {dest.name} downloaded only {size:,} B "
                         f"(expected >= {min_bytes:,}); URL moved?")
    tmp.replace(dest)
    return f"fetched  {dest.name}  ({size:,} B)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    failed = 0
    for name, url in ASSETS.items():
        try:
            print(fetch(url, config.ASSETS / name, MIN_BYTES[name], args.force))
        except Exception as exc:                     # network, 404, disk
            print(f"FAILED   {name}: {exc}")
            failed += 1

    print(f"\nassets dir: {config.ASSETS}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
