"""Active liveness state machine over deterministic landmark trajectories."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from faceauth import config, detect, liveness, preprocess


FRAME = np.zeros((240, 320, 3), dtype=np.uint8)


def face(yaw: float = 0.0, area_scale: float = 1.0) -> detect.Face:
    pts = preprocess.CANONICAL_5PT.copy()
    eye_distance = float(np.linalg.norm(pts[1] - pts[0]))
    midpoint = float(pts[:2, 0].mean())
    pts[2, 0] = midpoint + yaw * eye_distance
    side = 100.0 * np.sqrt(area_scale)
    return detect.Face((50.0, 50.0, side, side), pts, 0.99)


class Source:
    live = True

    def __init__(self, n: int):
        self.n = n

    def read(self):
        if self.n <= 0:
            return None
        self.n -= 1
        return FRAME


class SequenceDetector:
    def __init__(self, sequence):
        self.sequence = list(sequence)

    def detect(self, frame):
        return self.sequence.pop(0) if self.sequence else []


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.01
        return self.t


def test_left_turn_and_return_passes():
    seq = [[face(0.0)], [face(0.01)],
           [face(0.14)], [face(0.15)],
           [face(0.03)], [face(0.01)]]
    prompts = []
    result = liveness.challenge(Source(len(seq)), SequenceDetector(seq),
                                direction="left", clock=Clock(),
                                prompt=prompts.append)
    assert result.passed
    assert result.direction == "left"
    assert result.peak_delta >= config.LIVENESS_TURN_DELTA
    assert len(prompts) == 3


def test_right_turn_and_return_passes():
    seq = [[face(0.0)], [face(-0.01)],
           [face(-0.14)], [face(-0.15)],
           [face(-0.03)], [face(0.0)]]
    result = liveness.challenge(Source(len(seq)), SequenceDetector(seq),
                                direction="right", clock=Clock(), prompt=lambda x: None)
    assert result.passed


def test_stationary_print_fails():
    seq = [[face(0.0)] for _ in range(12)]
    result = liveness.challenge(Source(len(seq)), SequenceDetector(seq),
                                direction="left", clock=Clock(), prompt=lambda x: None)
    assert not result.passed
    assert "camera stopped" in result.reason or "timed out" in result.reason


def test_wrong_direction_fails():
    seq = [[face(0.0)], [face(0.0)]] + [[face(-0.2)] for _ in range(6)]
    result = liveness.challenge(Source(len(seq)), SequenceDetector(seq),
                                direction="left", clock=Clock(), prompt=lambda x: None)
    assert not result.passed


def test_moving_flat_photo_closer_does_not_count_as_turn():
    seq = [[face(0.0)], [face(0.0)],
           [face(0.2, 1.6)], [face(0.2, 1.6)],
           [face(0.0)], [face(0.0)]]
    result = liveness.challenge(Source(len(seq)), SequenceDetector(seq),
                                direction="left", clock=Clock(), prompt=lambda x: None)
    assert not result.passed


def test_multiple_faces_fail_closed():
    seq = [[face(0.0), face(0.0)]]
    result = liveness.challenge(Source(1), SequenceDetector(seq),
                                direction="left", clock=Clock(), prompt=lambda x: None)
    assert not result.passed
    assert "multiple faces" in result.reason
