"""Verification metrics on a distance scale (lower distance = more genuine)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve

from . import config


def _similarity(genuine: np.ndarray, impostor: np.ndarray):
    """Flip distances into similarities so ROC conventions hold."""
    s = -np.concatenate([genuine, impostor]).astype(np.float64)
    y = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    return s, y


def eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """Equal error rate and the distance threshold that achieves it.

    Returns (eer, threshold_on_distance). Interpolates between the two
    neighbouring operating points instead of snapping to one, because with a few
    hundred target trials the snap can be off by several tenths of a percent.
    """
    s, y = _similarity(genuine, impostor)
    fpr, tpr, thr = roc_curve(y, s)
    frr = 1.0 - tpr
    gap = fpr - frr
    sign_change = np.flatnonzero(np.sign(gap[:-1]) != np.sign(gap[1:]))
    if len(sign_change):
        i = int(sign_change[0])
        # Linear interpolation of the crossing on the index axis.
        g0, g1 = gap[i], gap[i + 1]
        w = abs(g0) / (abs(g0) + abs(g1)) if (abs(g0) + abs(g1)) else 0.0
        err = fpr[i] + w * (fpr[i + 1] - fpr[i])
        d_thr = -(thr[i] + w * (thr[i + 1] - thr[i]))
    else:
        i = int(np.argmin(np.abs(gap)))
        err = 0.5 * (fpr[i] + frr[i])
        d_thr = -thr[i]
    return float(err), float(d_thr)


def tar_at_far(genuine: np.ndarray, impostor: np.ndarray,
               far: float = 0.01) -> tuple[float, float]:
    """True accept rate at a fixed false accept rate, plus its threshold."""
    s, y = _similarity(genuine, impostor)
    fpr, tpr, thr = roc_curve(y, s)
    ok = np.flatnonzero(fpr <= far)
    if not len(ok):
        return 0.0, float("nan")
    i = int(ok[-1])
    return float(tpr[i]), float(-thr[i])


def det_points(genuine: np.ndarray, impostor: np.ndarray):
    """(FAR, FRR) operating points, for a DET curve."""
    s, y = _similarity(genuine, impostor)
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, 1.0 - tpr


def min_dcf(genuine: np.ndarray, impostor: np.ndarray,
            p_target: float = 0.5, c_miss: float = 1.0,
            c_fa: float = 1.0) -> float:
    """Normalised minimum detection cost, minimised over all thresholds.

    Unlike EER, this weights the two error types by what they cost. Normalised
    by the cost of the trivial always-accept / always-reject strategies, so 1.0
    means "no better than guessing" and 0.0 means "no errors".
    """
    fpr, frr = det_points(genuine, impostor)
    cost = c_miss * p_target * frr + c_fa * (1.0 - p_target) * fpr
    norm = min(c_miss * p_target, c_fa * (1.0 - p_target))
    return float(cost.min() / norm)


def bootstrap_ci(genuine: np.ndarray, impostor: np.ndarray,
                 genuine_model_ids: np.ndarray, impostor_model_ids: np.ndarray,
                 genuine_probe_ids: np.ndarray | None = None,
                 impostor_probe_ids: np.ndarray | None = None,
                 n: int = 500, seed: int = config.SEED) -> tuple[float, float]:
    """Identity-clustered CI: resample subjects, not trials.

    A subject appears in many trials, both as the enrolled model and as the
    source of a probe, so its trials are correlated. Resampling individual
    trials would treat those as independent and report an interval that is far
    too narrow.

    Each replicate draws a set of subjects with replacement and recomputes the
    metric from every planned trial whose enrolled model and probe source both
    fall inside that draw. Conditioning both sides on the same draw keeps the
    correlation that a model-only cluster misses, and it is the conservative
    choice: on the LFW benchmark it widens the interval relative to clustering
    on the enrolled model alone.
    """
    g_model = np.asarray(genuine_model_ids)
    i_model = np.asarray(impostor_model_ids)
    g_probe = g_model if genuine_probe_ids is None else np.asarray(genuine_probe_ids)
    i_probe = i_model if impostor_probe_ids is None else np.asarray(impostor_probe_ids)

    subjects = np.unique(np.concatenate([g_model, g_probe, i_model, i_probe]))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        draw = np.unique(rng.choice(subjects, size=len(subjects), replace=True))
        g_idx = np.flatnonzero(np.isin(g_model, draw) & np.isin(g_probe, draw))
        i_idx = np.flatnonzero(np.isin(i_model, draw) & np.isin(i_probe, draw))
        if len(g_idx) == 0 or len(i_idx) == 0:
            continue
        vals.append(eer(genuine[g_idx], impostor[i_idx])[0])
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
