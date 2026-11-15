"""Basic active liveness from YuNet's five facial landmarks.

The challenge is deliberately classical and fail-closed: establish a neutral
pose, request a random left/right head turn, then require a return to center. A
stationary print cannot satisfy the landmark trajectory. This is deterrence,
not certified presentation-attack detection; an adaptive video or articulated
mask remains outside its guarantees.
"""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import config, detect


@dataclass(frozen=True)
class Result:
    passed: bool
    reason: str
    direction: str
    baseline: float | None = None
    peak_delta: float = 0.0
    frames: int = 0


def yaw_proxy(face: detect.Face) -> float:
    """Nose displacement from eye midpoint, normalised by inter-eye distance."""
    pts = np.asarray(face.pts, dtype=np.float64).reshape(5, 2)
    eyes = pts[:2]
    eye_distance = float(np.linalg.norm(eyes[1] - eyes[0]))
    if eye_distance < 1e-6:
        raise ValueError("eye landmarks are degenerate")
    midpoint_x = float(eyes[:, 0].mean())
    return float((pts[2, 0] - midpoint_x) / eye_distance)


def challenge(source, detector: detect.Detector, *, direction: str | None = None,
              timeout: float = config.LIVENESS_TIMEOUT_SECONDS,
              clock: Callable[[], float] = time.monotonic,
              prompt: Callable[[str], None] = print) -> Result:
    """Run neutral -> random turn -> neutral, returning evidence and a reason."""
    direction = direction or secrets.choice(("left", "right"))
    if direction not in {"left", "right"}:
        raise ValueError(f"unknown liveness direction {direction!r}")
    # In an unmirrored camera frame, turning toward the subject's left shoulder
    # moves the nose toward increasing image x relative to the eyes.
    sign = 1.0 if direction == "left" else -1.0
    start = clock()
    state = "neutral"
    streak = 0
    frames = 0
    neutral_values: list[float] = []
    baseline: float | None = None
    baseline_area: float | None = None
    peak = 0.0
    prompted = False

    prompt("Liveness: look straight at the camera.")
    while clock() - start <= timeout:
        frame = source.read()
        if frame is None:
            return Result(False, "camera stopped during liveness", direction,
                          baseline, peak, frames)
        frames += 1
        faces = detector.detect(frame)
        if len(faces) > 1:
            return Result(False, "multiple faces during liveness", direction,
                          baseline, peak, frames)
        if not faces:
            streak = 0
            continue
        face = faces[0]
        try:
            yaw = yaw_proxy(face)
        except ValueError:
            streak = 0
            continue

        if state == "neutral":
            if abs(yaw) <= config.LIVENESS_NEUTRAL_MAX:
                neutral_values.append(yaw)
                window = neutral_values[-config.LIVENESS_CONSECUTIVE_FRAMES:]
                stable = (len(window) >= config.LIVENESS_CONSECUTIVE_FRAMES
                          and max(window) - min(window)
                          <= config.LIVENESS_NEUTRAL_STABILITY)
                streak = len(window) if stable else 0
            else:
                streak = 0
                neutral_values.clear()
            if streak >= config.LIVENESS_CONSECUTIVE_FRAMES:
                baseline = float(np.mean(neutral_values[-streak:]))
                baseline_area = face.area
                state, streak = "turn", 0
        elif state == "turn":
            if not prompted:
                prompt(f"Liveness: turn your head toward your {direction} shoulder.")
                prompted = True
            delta = sign * (yaw - baseline)
            peak = max(peak, delta)
            area_ratio = face.area / max(float(baseline_area), 1.0)
            # Keep roughly the same distance; moving a flat photograph toward
            # the lens is not accepted as a head turn.
            if delta >= config.LIVENESS_TURN_DELTA and 0.70 <= area_ratio <= 1.30:
                streak += 1
            else:
                streak = 0
            if streak >= config.LIVENESS_CONSECUTIVE_FRAMES:
                state, streak = "return", 0
                prompt("Liveness: return to center.")
        else:  # return
            if abs(yaw - baseline) <= config.LIVENESS_RETURN_DELTA:
                streak += 1
            else:
                streak = 0
            if streak >= config.LIVENESS_CONSECUTIVE_FRAMES:
                return Result(True, "challenge completed", direction,
                              baseline, peak, frames)
    return Result(False, f"liveness timed out before {state} completed",
                  direction, baseline, peak, frames)
