"""Multi-frame trial grouping never crosses source identities."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_pooling
from faceauth import protocol


def test_grouped_targets_have_exactly_n_frames():
    labels = np.repeat(np.arange(4), 8)
    sp = protocol.split(labels, 2)
    targets, _ = evaluate_pooling.grouped_trials(sp, 3, None)
    assert len(targets) == 4 * 2       # six test images -> two groups each
    assert all(len(group) == 3 for _, group in targets)


def test_impostor_group_never_mixes_source_identities():
    labels = np.repeat(np.arange(4), 8)
    sp = protocol.split(labels, 2)
    _, impostors = evaluate_pooling.grouped_trials(sp, 3, None)
    for model_id, group in impostors:
        assert len(set(int(labels[i]) for i in group)) == 1
        assert int(labels[group[0]]) != model_id


def test_leftover_frames_are_dropped_not_short_pooled():
    labels = np.repeat(np.arange(3), 7)   # k=2 -> five test images
    sp = protocol.split(labels, 2)
    targets, _ = evaluate_pooling.grouped_trials(sp, 3, None)
    assert len(targets) == 3              # one full triple per identity
    assert all(len(group) == 3 for _, group in targets)


def test_impostor_subsample_is_reproducible():
    labels = np.repeat(np.arange(8), 10)
    sp = protocol.split(labels, 2)
    a = evaluate_pooling.grouped_trials(sp, 3, 25)
    b = evaluate_pooling.grouped_trials(sp, 3, 25)
    assert a == b
    assert len(a[1]) == 25
