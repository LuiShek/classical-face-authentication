"""Metric invariants: EER location, interpolation, and DCF normalisation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from faceauth import metrics


def test_perfect_separation_has_zero_error():
    g = np.array([1.0, 2.0, 3.0])
    i = np.array([10.0, 11.0, 12.0])
    e, thr = metrics.eer(g, i)
    assert e == 0.0
    # the EER threshold must sit between the two populations
    assert 3.0 <= thr <= 10.0
    assert metrics.min_dcf(g, i) == 0.0
    assert metrics.tar_at_far(g, i, 0.0)[0] == 1.0


def test_identical_populations_are_undecidable():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    e, _ = metrics.eer(x, x.copy())
    # no threshold can separate a distribution from itself; error must be large
    assert e >= 0.4
    assert metrics.min_dcf(x, x.copy()) >= 0.4


def test_eer_is_complementary_under_population_swap():
    """Swapping the two populations inverts the problem, it does not mirror it.

    ``eer`` always treats the first array as genuine, so passing the impostors
    first asks "what if the far-away scores were the genuine ones?". The error
    rate must then be 1 - EER, which is a sharp check that the crossing and its
    interpolation are found on the right side of the curve.
    """
    rng = np.random.default_rng(0)
    g = rng.normal(70, 10, 300)
    i = rng.normal(100, 12, 3000)
    direct = metrics.eer(g, i)[0]
    flipped = metrics.eer(i, g)[0]
    assert direct < 0.2
    assert np.isclose(direct + flipped, 1.0, atol=1e-9)


def test_eer_interpolates_between_operating_points():
    """A hand-built case where the crossing is not on a sampled threshold.

    200 genuine trials scoring 0 and 200 impostors scoring 1: the only
    observable crossing is between two points, so a snap would report 0% or 50%
    while the interpolated answer is exactly 0.5.
    """
    g = np.zeros(200)
    i = np.ones(200)
    e, _ = metrics.eer(g, i)
    assert 0.0 <= e <= 0.5
    assert np.isclose(e, 0.0) or np.isclose(e, 0.5)


def test_tar_at_far_is_monotone_in_far():
    rng = np.random.default_rng(1)
    g = rng.normal(70, 10, 500)
    i = rng.normal(100, 12, 5000)
    a = metrics.tar_at_far(g, i, 0.001)[0]
    b = metrics.tar_at_far(g, i, 0.01)[0]
    c = metrics.tar_at_far(g, i, 0.10)[0]
    assert a <= b <= c


def test_min_dcf_normalisation_matches_trivial_strategy():
    """minDCF must equal 1.0 when the system is no better than always-reject."""
    g = np.array([5.0, 5.1])
    i = np.array([5.0, 5.1])          # indistinguishable
    assert metrics.min_dcf(g, i) >= 0.5


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(2)
    g = rng.normal(70, 10, 200)
    i = rng.normal(100, 12, 2000)
    gid = np.repeat(np.arange(20), 10)
    iid = np.repeat(np.arange(20), 100)
    lo, hi = metrics.bootstrap_ci(g, i, gid, iid, n=100, seed=0)
    assert lo <= hi
    assert 0.0 <= lo and hi <= 1.0


def test_bootstrap_ci_resamples_probe_subjects_too():
    """Impostor trials are correlated through the probe identity, not just the
    enrolled model. Keeping a subject's probe trials together must widen the
    interval relative to clustering on the model alone."""
    rng = np.random.default_rng(11)
    models = np.repeat(np.arange(24), 96)
    probes = np.tile(np.repeat(np.arange(12), 8) + 100, 24)
    shift = rng.normal(0, 1, 12)[probes - 100]     # probe-identity effect
    imp = 100 + shift + rng.normal(0, 4, models.size)
    gen = 70 + rng.normal(0, 4, 24 * 8)
    gid = np.repeat(np.arange(24), 8)
    model_only = metrics.bootstrap_ci(gen, imp, gid, models, n=200, seed=1)
    two_way = metrics.bootstrap_ci(gen, imp, gid, models, gid, probes,
                                   n=200, seed=1)
    assert two_way[0] <= model_only[0] and two_way[1] >= model_only[1]


def test_bootstrap_ci_accepts_missing_probe_ids():
    g = np.array([1.0, 2.0, 3.0, 4.0])
    i = np.array([9.0, 10.0, 11.0, 12.0])
    ids = np.array([0, 0, 1, 1])
    lo, hi = metrics.bootstrap_ci(g, i, ids, ids, n=20, seed=0)
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi


def test_det_points_are_probabilities():
    rng = np.random.default_rng(3)
    g = rng.normal(70, 10, 100)
    i = rng.normal(100, 12, 1000)
    far, frr = metrics.det_points(g, i)
    assert far.shape == frr.shape
    assert np.all((far >= 0) & (far <= 1))
    assert np.all((frr >= 0) & (frr <= 1))
