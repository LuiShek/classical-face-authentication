"""Decision rule and per-user calibration behaviour."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from faceauth import config, gate


def crops(seed: int, n: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (n, config.IMG_SIZE, config.IMG_SIZE),
                        dtype=np.uint8)


def test_decide_is_inclusive_at_the_threshold():
    assert gate.decide(10.0, 10.0)
    assert gate.decide(9.999, 10.0)
    assert not gate.decide(10.001, 10.0)


def test_leave_one_out_needs_at_least_three_crops():
    with pytest.raises(ValueError, match=">= 3"):
        gate.loo_scores(crops(0, 2))


def test_leave_one_out_scores_every_crop():
    a = crops(1)
    scores = gate.loo_scores(a)
    assert len(scores) == len(a)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0)


def test_calibration_without_a_cohort_controls_only_false_rejects():
    """No cohort means no measured FAR - the calibration must say so."""
    cal = gate.calibrate(crops(2), None, target_frr=0.2)
    assert cal.method == "loo-only"
    assert np.isfinite(cal.threshold)
    assert cal.far is None
    assert cal.eer is None
    assert cal.n_impostor == 0
    assert cal.frr is not None and cal.frr <= 0.2 + 1e-9


@pytest.mark.parametrize("field,value", [("target_frr", -0.1),
                                          ("target_frr", 1.0),
                                          ("target_far", -0.1),
                                          ("target_far", 1.0)])
def test_invalid_live_error_targets_are_rejected(field, value):
    kwargs = {field: value}
    with pytest.raises(ValueError, match=field):
        gate.calibrate(crops(2), crops(3), **kwargs)


def test_calibration_with_a_cohort_reports_both_error_rates():
    cal = gate.calibrate(crops(3), crops(4), target_frr=0.05,
                         cohort_identities=20)
    assert cal.method == "loo+far0"
    assert np.isfinite(cal.threshold)
    assert cal.eer is not None and 0.0 <= cal.eer <= 1.0
    assert cal.far == 0.0
    assert cal.frr is not None and 0.0 <= cal.frr <= 1.0
    assert cal.n_impostor == 6


def test_a_cohort_of_one_identity_is_refused():
    """One other person is not a cohort; an EER from 5+5 scores is a coin flip."""
    cal = gate.calibrate(crops(3), crops(4), cohort_identities=1)
    assert cal.eer is None
    assert cal.far is None, "an unmeasured FAR must not be reported as a number"
    assert "cohort 1" in cal.method
    assert np.isfinite(cal.threshold)
    assert cal.frr is not None


def test_cohort_size_decides_which_calibration_path_runs():
    few = gate.calibrate(crops(3), crops(4),
                         cohort_identities=config.MIN_COHORT_IDENTITIES - 1)
    enough = gate.calibrate(crops(3), crops(4),
                            cohort_identities=config.MIN_COHORT_IDENTITIES)
    assert few.method != "loo+far0"
    assert enough.method == "loo+far0"


def test_meta_payload_round_trips_through_json():
    import json
    cal = gate.calibrate(crops(5), crops(6))
    payload = cal.as_meta()
    again = json.loads(json.dumps(payload))
    assert again["threshold"] == payload["threshold"]
    assert again["calibration"]["method"] == "loo+far0"
    assert again["calibration"]["target_far"] == 0.0
    assert again["scale"] in {"raw", "z"}


def test_raw_scale_is_a_passthrough():
    assert gate.apply_scale(12.5, "raw", None) == 12.5
    assert gate.apply_scale(12.5, "raw", {"mean": 50, "std": 2}) == 12.5


def test_z_scale_without_a_cohort_falls_back_to_raw():
    """Better an unnormalised score than a division by a cohort that is not there."""
    assert gate.apply_scale(12.5, "z", None) == 12.5


def test_z_scale_uses_the_cohort():
    assert gate.apply_scale(14.0, "z", {"mean": 12.0, "std": 2.0}) == pytest.approx(1.0)


def test_z_calibration_records_the_cohort_it_used():
    cal = gate.calibrate(crops(7), crops(8), scale="z")
    assert cal.cohort_mean is not None and cal.cohort_std is not None
    assert cal.as_meta()["cohort"]["std"] == pytest.approx(round(cal.cohort_std, 6))


def test_same_crop_scores_near_zero_against_its_own_template():
    a = crops(9)
    t = gate.Template(a)
    assert t.distance(a[0]) == pytest.approx(0.0)


def test_a_different_identity_scores_further_than_its_own_crop():
    a, b = crops(10), crops(11)
    t = gate.Template(a)
    assert t.distance(a[0]) < t.distance(b[0])


def test_clahe_template_equalises_the_probe():
    """The wrapper, not the caller, must apply the stored photometric setting."""
    a = crops(12)
    equalised = gate.Template(a, clahe=True)
    raw = gate.Template(a, clahe=False)
    assert equalised.clahe is True and raw.clahe is False
    # Both wrappers match their own crop exactly, because each transforms the
    # probe with the setting it was trained under.
    assert equalised.distance(a[0]) == pytest.approx(0.0)
    assert raw.distance(a[0]) == pytest.approx(0.0)
    # Bypassing the wrapper feeds a raw probe to an equalised model, which does
    # not match - that is the mistake the wrapper exists to prevent.
    assert equalised.model.distance(a[0]) > 0.0


def test_distance_to_matches_template_distance():
    a = crops(13)
    t = gate.Template(a)
    assert gate.distance_to(t.model, a[1], False) == pytest.approx(t.distance(a[1]))
