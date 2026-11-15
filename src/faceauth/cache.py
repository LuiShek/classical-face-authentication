"""Reading the prepared caches, with the staleness guard applied.

The guard matters more than it looks. A cache built by an older version of
``prepare_data.py`` holds CLAHE'd pixels; scoring it as if it were raw would
attribute the equalisation to the model and quietly produce a wrong comparison
between arms. The cache therefore carries a ``stage`` marker and anything other
than ``raw`` is refused here, once, so no caller can forget to check.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config

STAGE = "raw"


def path_for(dataset: str) -> Path:
    return config.DATA / "prepared" / f"{dataset}.npz"


def load_cache(dataset: str, variant: str = "aligned",
               size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (images, labels) for a dataset, optionally resized in memory.

    ``variant`` selects the LFW crop (aligned | boxed | aligned_refined). ORL is
    pre-cropped by hand, so it ignores the variant and keeps native geometry.
    """
    path = path_for(dataset)
    if not path.exists():
        raise SystemExit(f"error: missing {path}; run scripts/prepare_data.py")
    z = np.load(path, allow_pickle=False)
    stage = str(z["stage"]) if "stage" in z.files else "unknown"
    if stage != STAGE:
        raise SystemExit(
            f"error: {path.name} has stage={stage}, expected {STAGE}. Rebuild "
            f"it with scripts/prepare_data.py.")
    if dataset == "lfw":
        if variant not in z.files:
            raise SystemExit(
                f"error: variant {variant!r} not in {path.name} "
                f"(available: {[k for k in z.files if k not in ('labels', 'names')]}); "
                f"rebuild LFW with --refine for aligned_refined")
        images = z[variant]
    else:
        images = z["images"]
    labels = z["labels"]
    if size is not None and images.shape[1] != size:
        import cv2
        images = np.stack([cv2.resize(i, (size, size),
                                      interpolation=cv2.INTER_AREA)
                           for i in images])
    return images, labels
