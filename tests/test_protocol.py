"""Protocol invariants: deterministic splits, correct trial labelling."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from faceauth import protocol


def test_split_is_deterministic_and_sized():
    labels = np.repeat(np.arange(5), 10)
    a = protocol.split(labels, 3, seed=0)
    b = protocol.split(labels, 3, seed=0)
    assert a.enroll == b.enroll and a.test == b.test
    assert all(len(v) == 3 for v in a.enroll.values())
    assert all(len(v) == 7 for v in a.test.values())
    assert a.counts() == {"identities": 5, "enroll_images": 15,
                          "test_images": 35}
    # enroll and test never share an image
    for ident in a.enroll:
        assert not set(a.enroll[ident]) & set(a.test[ident])


def test_split_drops_identities_without_a_target_trial():
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1])   # identity 1 has only 3
    sp = protocol.split(labels, 3)
    assert sp.identities == [0]


def test_plan_sizes_and_labels():
    labels = np.repeat(np.arange(5), 10)
    sp = protocol.split(labels, 3)
    targets, impostors = protocol.plan(sp)
    assert len(targets) == 5 * 7
    assert len(impostors) == 5 * 4 * 7
    for model_id, img in targets:
        assert labels[img] == model_id
    for model_id, img in impostors:
        assert labels[img] != model_id
    assert len({(m, i) for m, i in targets + impostors}) == len(targets) + len(impostors)


def test_impostor_subsample_is_reproducible():
    labels = np.repeat(np.arange(5), 10)
    sp = protocol.split(labels, 3)
    a = protocol.plan(sp, max_impostor=50)
    b = protocol.plan(sp, max_impostor=50)
    assert a[1] == b[1]
    assert len(a[1]) == 50
