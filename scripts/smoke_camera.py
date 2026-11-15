"""Headless camera smoke test: open the webcam, detect a face, report timing.

Deliberately writes no image to disk. Running this must not create a facial
biometric artefact, so it prints statistics only.

Usage
-----
  py scripts/smoke_camera.py
  py scripts/smoke_camera.py --index 1 --frames 40
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from faceauth import config


def open_camera(index: int) -> tuple[cv2.VideoCapture, str]:
    """Try DSHOW first: MSMF is the OpenCV default and is flaky on laptops."""
    for backend, name in ((cv2.CAP_DSHOW, "DSHOW"),
                          (cv2.CAP_MSMF, "MSMF"),
                          (cv2.CAP_ANY, "ANY")):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened() and cap.read()[0]:
            return cap, name
        cap.release()
    raise SystemExit(f"error: no camera at index {index} "
                     f"(tried DSHOW, MSMF, ANY)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--frames", type=int, default=30)
    args = ap.parse_args()

    if not config.YUNET.exists():
        raise SystemExit(f"error: missing {config.YUNET}; "
                         f"run scripts/fetch_assets.py first")

    cap, backend = open_camera(args.index)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"camera   : index {args.index} via {backend}, {w}x{h}")

    det = cv2.FaceDetectorYN.create(str(config.YUNET), "", (w, h),
                                    0.6, 0.3, 5000)
    det.setInputSize((w, h))

    for _ in range(5):                       # let auto-exposure settle
        cap.read()

    detections, times = 0, []
    for _ in range(args.frames):
        ok, frame = cap.read()
        if not ok:
            continue
        t0 = time.perf_counter()
        _, faces = det.detect(frame)
        times.append(time.perf_counter() - t0)
        if faces is not None and len(faces):
            detections += 1
            if detections == 1:
                x, y, fw, fh = faces[0][:4]
                print(f"face     : {fw:.0f}x{fh:.0f} px at ({x:.0f},{y:.0f}), "
                      f"score {faces[0][14]:.3f}")
    cap.release()

    if not times:
        raise SystemExit("error: camera opened but returned no frames")
    ms = 1000 * float(np.mean(times))
    print(f"detector : {ms:.1f} ms/frame ({1000 / ms:.1f} fps) over "
          f"{len(times)} frames")
    print(f"faces    : {detections}/{len(times)} frames")
    if detections == 0:
        print("warning  : no face detected - check lighting, distance, "
              "and that the camera is not covered")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
