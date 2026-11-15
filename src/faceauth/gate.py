"""The decision rule, and per-user threshold calibration.

The threshold cannot be copied from a dataset. ORL and LFW differ by 20 points
of EER, and a webcam differs from both, so each enrolled identity carries its own
calibrated threshold plus the evidence behind it.

Genuine scores come from leave-one-out over the enrolment crops: fit on k-1, score
the held-out one. Impostor scores come from scoring *other enrolled identities'*
crops against this model, which is the same direction as an impostor trial in the
evaluation protocol. With no other enrolled identity there are no impostor scores,
and the fallback says so instead of inventing a FAR.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config, lbph, metrics, preprocess


def as_images(crops, clahe: bool) -> list[np.ndarray]:
    arr = np.asarray(crops)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if clahe:
        return [preprocess.equalise(c) for c in arr]
    return [np.ascontiguousarray(c, dtype=np.uint8) for c in arr]


def distance_to(model, crop: np.ndarray, clahe: bool = False) -> float:
    """Score one probe crop against an already-fitted model.

    The photometric flag must be the one the model was trained with; it is
    stored in each identity's meta for exactly this reason.
    """
    img = (preprocess.equalise(np.ascontiguousarray(crop, dtype=np.uint8))
           if clahe else np.ascontiguousarray(crop, dtype=np.uint8))
    return float(model.distance(img))


class Template:
    """An LBPH model bound to the photometric setting it was trained with.

    Keeping the flag next to the model is what stops the enrolment path and the
    verification path from silently disagreeing about CLAHE.
    """

    def __init__(self, crops, clahe: bool = False) -> None:
        self.clahe = bool(clahe)
        self.images = as_images(crops, self.clahe)
        if not self.images:
            raise ValueError("cannot build a template from zero crops")
        self.model = lbph.LbphModel().fit(self.images)

    def distance(self, crop: np.ndarray) -> float:
        return distance_to(self.model, crop, self.clahe)


def decide(distance: float, threshold: float) -> bool:
    """Accept when the distance is at or below the calibrated threshold."""
    return bool(distance <= threshold)


@dataclass
class Calibration:
    scale: str = "raw"
    threshold: float = float("nan")
    method: str = "none"
    target_frr: float = 0.05
    target_far: float = 0.0
    eer: float | None = None
    far: float | None = None
    frr: float | None = None
    n_genuine: int = 0
    n_impostor: int = 0
    cohort_mean: float | None = None
    cohort_std: float | None = None
    genuine: list[float] = field(default_factory=list)

    def as_meta(self) -> dict:
        payload = {
            "scale": self.scale,
            "threshold": round(float(self.threshold), 6),
            "calibration": {
                "method": self.method,
                "target_frr": self.target_frr,
                "target_far": self.target_far,
                "eer": None if self.eer is None else round(self.eer, 6),
                "far": None if self.far is None else round(self.far, 6),
                "frr": None if self.frr is None else round(self.frr, 6),
                "n_genuine": self.n_genuine,
                "n_impostor": self.n_impostor,
                "genuine_mean": round(float(np.mean(self.genuine)), 3)
                if self.genuine else None,
            },
        }
        if self.cohort_mean is not None:
            payload["cohort"] = {"mean": round(float(self.cohort_mean), 6),
                                 "std": round(float(self.cohort_std), 6)}
        return payload


def loo_scores(crops, clahe: bool = False) -> np.ndarray:
    """Leave-one-out genuine distances. Needs at least three crops."""
    images = as_images(crops, clahe)
    if len(images) < 3:
        raise ValueError(f"leave-one-out needs >= 3 crops, got {len(images)}")
    out = []
    for i in range(len(images)):
        model = lbph.LbphModel().fit([images[j] for j in range(len(images))
                                      if j != i])
        out.append(float(model.distance(images[i])))
    return np.array(out, dtype=np.float64)


def impostor_scores(template: Template, other_crops, clahe: bool = False) -> np.ndarray:
    """Distances from this identity's model to other identities' enrolled crops."""
    # Keep these raw: Template.distance applies the profile's photometric
    # setting. Equalising here as well would apply CLAHE twice.
    imgs = as_images(other_crops, False) if other_crops is not None else []
    return np.array([template.distance(c) for c in imgs], dtype=np.float64)


def apply_scale(distance: float, scale: str, cohort: dict | None) -> float:
    if scale != "z":
        return float(distance)
    if not cohort:
        return float(distance)
    return float((distance - cohort["mean"]) / max(cohort["std"], 1e-9))


def calibrate(crops, other_crops=None, *, clahe: bool = False,
              scale: str = "raw", target_frr: float = 0.05,
              target_far: float = 0.0,
              cohort_identities: int | None = None) -> Calibration:
    """Threshold from this identity's own held-out crops, plus a cohort if any.

    ``cohort_identities`` is how many *distinct* identities contributed the
    impostor scores. It is not the same as the number of impostor scores: five
    identities with fifty photos each estimate the impostor distribution far
    better than one identity with fifty photos. When the caller reports fewer
    than ``config.MIN_COHORT_IDENTITIES``, the EER path is refused and the
    threshold falls back to controlling the false-reject rate only, because an
    EER from one other person is a coin flip dressed up as a measurement.
    """
    gen = loo_scores(crops, clahe)
    template = Template(crops, clahe)
    imp = impostor_scores(template, other_crops, clahe) if other_crops is not None \
        else np.array([], dtype=np.float64)
    too_small = (cohort_identities is not None
                 and cohort_identities < config.MIN_COHORT_IDENTITIES)

    if not 0.0 <= target_frr < 1.0:
        raise ValueError(f"target_frr must be in [0, 1), got {target_frr}")
    if not 0.0 <= target_far < 1.0:
        raise ValueError(f"target_far must be in [0, 1), got {target_far}")
    cal = Calibration(scale=scale, target_frr=target_frr,
                      target_far=target_far,
                      n_genuine=int(len(gen)), n_impostor=int(len(imp)),
                      genuine=[float(v) for v in gen])

    if scale == "z" and len(imp) > 1 and not too_small:
        cal.cohort_mean = float(imp.mean())
        cal.cohort_std = float(imp.std(ddof=1))
        cohort = {"mean": cal.cohort_mean, "std": cal.cohort_std}
        gen = np.array([apply_scale(g, scale, cohort) for g in gen])
        imp = np.array([apply_scale(i, scale, cohort) for i in imp])

    if len(imp) and not too_small:
        # The user explicitly prefers false rejects over false accepts. Bound
        # the observed cohort FAR first, then take the stricter of that bound
        # and the requested genuine quantile. EER remains diagnostic only.
        ordered = np.sort(imp)
        allowed = int(np.floor(target_far * len(ordered)))
        if allowed == 0:
            impostor_limit = float(np.nextafter(ordered[0], -np.inf))
        else:
            impostor_limit = float(np.nextafter(ordered[allowed], -np.inf))
        genuine_limit = float(np.quantile(gen, 1.0 - target_frr))
        thr = min(genuine_limit, impostor_limit)
        err, _ = metrics.eer(gen, imp)
        cal.method = f"loo+far{target_far:g}"
        cal.eer = float(err)
        cal.threshold = float(thr)
        cal.far = float(np.mean(imp <= cal.threshold))
        cal.frr = float(np.mean(gen > cal.threshold))
    else:
        # No usable cohort: control the false-reject rate and admit that the
        # false-accept rate is unmeasured. Pretending otherwise is how "it works
        # on my face" projects get written.
        cal.method = ("loo-only" if cohort_identities is None
                      else f"loo-only (cohort {cohort_identities}"
                           f"<{config.MIN_COHORT_IDENTITIES})")
        cal.cohort_mean = cal.cohort_std = None
        cal.threshold = float(np.quantile(gen, 1.0 - target_frr))
        cal.frr = float(np.mean(gen > cal.threshold))
        cal.far = None
    return cal
