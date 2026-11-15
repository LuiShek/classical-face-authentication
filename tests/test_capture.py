"""Capture: the quality gate rejects for the right reasons and warns for the rest."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from faceauth import capture, config, detect, preprocess


def frame(size: int = 320) -> np.ndarray:
    """High-frequency noise, so the blur gate passes on texture alone."""
    return np.random.default_rng(0).integers(
        0, 256, (size, size), dtype=np.uint8)


def points(offset: int = 0) -> np.ndarray:
    return (preprocess.CANONICAL_5PT + offset).astype(np.float32)


class StubDetector:
    def __init__(self, faces) -> None:
        self._faces = faces

    def detect(self, _frame):
        return list(self._faces)


def face(box, offset: int = 0, score: float = 0.9) -> detect.Face:
    return detect.Face(box=box, pts=points(offset), score=score)


class Source:
    live = False

    def __init__(self, frames) -> None:
        self.description = "stub"
        self._frames = list(frames)
        self.reads = 0

    def read(self):
        self.reads += 1
        return self._frames.pop(0) if self._frames else None

    def close(self) -> None:
        pass


def test_no_face_is_a_rejection_with_a_reason():
    a = capture.assess(frame(), StubDetector([]))
    assert not a.ok
    assert "no face" in a.reasons[0]


def test_a_small_face_is_rejected():
    tiny = config.MIN_FACE_PX - 10
    a = capture.assess(frame(), StubDetector([face((10, 10, tiny, tiny))]))
    assert not a.ok
    assert "too small" in a.reasons[0]


def test_two_faces_warns_but_still_produces_a_crop():
    """A background face must not make an otherwise good probe unusable."""
    a = capture.assess(frame(), StubDetector([
        face((40, 40, 100, 100)), face((220, 220, 60, 60), offset=180, score=0.7)]))
    assert a.ok, a.reasons
    assert a.crop is not None
    assert a.warnings and "2 faces" in a.warnings[0]


def test_the_largest_face_wins():
    seen = {}

    def on_reject(attempt, assessment):
        seen.setdefault("area", assessment.face_area)

    # Two faces, the right-hand one larger; the crop must come from it.
    small = face((10, 10, 40, 40))
    large = face((60, 60, 120, 120), offset=40)
    a = capture.assess(frame(), StubDetector([small, large]))
    assert a.ok
    assert a.face_area == large.area


def test_a_blurred_face_is_rejected_by_laplacian_variance():
    flat = np.full((200, 200), 128, dtype=np.uint8)

    class Detector:
        def detect(self, _f):
            return [face((40, 40, 120, 120))]

    a = capture.assess(flat, Detector())
    assert not a.ok
    assert "blurred" in a.reasons[0]


def test_collect_stops_at_the_requested_take_count():
    src = Source([frame() for _ in range(10)])
    res = capture.collect(src, StubDetector([face((10, 10, 120, 120))]), takes=3)
    assert len(res.crops) == 3
    assert src.reads == 3


def test_collect_skips_rejected_frames_and_keeps_going():
    class Alternating(StubDetector):
        def __init__(self):
            super().__init__(None)
            self.i = 0

        def detect(self, _f):
            self.i += 1
            return [face((10, 10, 120, 120))] if self.i % 2 else []

    src = Source([frame() for _ in range(10)])
    res = capture.collect(src, Alternating(), takes=2)
    assert len(res.crops) == 2
    assert src.reads == 3          # accept, reject, accept


def test_collect_reports_why_it_failed():
    src = Source([frame() for _ in range(3)])
    res = capture.collect(src, StubDetector([]), takes=3)
    assert res.crops == []
    assert res.problems and "no face" in res.problems[0]


def test_collect_stops_when_the_source_is_exhausted():
    src = Source([frame()])
    res = capture.collect(src, StubDetector([face((10, 10, 120, 120))]), takes=5)
    assert len(res.crops) == 1
    assert res.problems == []          # not a rejection, just no more frames


def test_collect_deduplicates_warnings():
    src = Source([frame() for _ in range(3)])
    two = [face((10, 10, 120, 120)), face((200, 200, 50, 50), offset=150)]
    res = capture.collect(src, StubDetector(two), takes=3)
    assert res.warnings == ["2 faces in frame; using the largest"]


def test_collect_reports_each_accepted_take():
    src = Source([frame() for _ in range(3)])
    seen = []
    res = capture.collect(
        src, StubDetector([face((10, 10, 120, 120))]), takes=3,
        on_accept=lambda n, assessment: seen.append((n, assessment.ok)))
    assert len(res.crops) == 3
    assert seen == [(1, True), (2, True), (3, True)]


def test_live_interval_drains_frames_between_takes(monkeypatch):
    src = Source([frame() for _ in range(4)])
    src.live = True
    times = iter([0.0, 0.1, 0.2, 0.8])
    monkeypatch.setattr(capture.time, "monotonic", lambda: next(times))
    res = capture.collect(
        src, StubDetector([face((10, 10, 120, 120))]), takes=2,
        live_interval=0.7)
    assert len(res.crops) == 2
    assert src.reads == 4          # two buffered frames were drained, not used
