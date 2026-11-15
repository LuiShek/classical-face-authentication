"""Classical baseline feature invariants."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import baselines


def test_pair_feature_is_absolute_difference_and_symmetric():
    a = np.array([1.0, 4.0, -2.0], dtype=np.float32)
    b = np.array([3.0, 1.0, -5.0], dtype=np.float32)
    expected = np.array([2.0, 3.0, 3.0], dtype=np.float32)
    assert np.array_equal(baselines.pair_feature(a, b), expected)
    assert np.array_equal(baselines.pair_feature(a, b),
                          baselines.pair_feature(b, a))
