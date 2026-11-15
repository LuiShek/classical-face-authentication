"""Non-LBPH baselines on the identical trial protocol.

The point is not to win. LBPH is only meaningful if a reader can see what a
simple alternative does on the same splits, so three cheap classical baselines
are measured with the same enrolment images, the same test images, and the same
impostor subsample:

  pixel       per-image mean-subtracted, L2-normalised greyscale, cosine to the
              mean enrolment vector
  hog         cv2 HOG descriptors, cosine to the mean enrolment vector
  eigenfaces  PCA fit on the enrolment images, cosine in the projected space
  hogsvm      LinearSVC on HOG pair differences, trained only on enrolment
              pairs (same-identity vs different-identity)

Everything each method learns comes from enrolment images of the evaluated
identities; no test image is used for fitting, only for scoring.

Usage
-----
  py scripts/baselines.py --dataset orl --k 5 --method all
  py scripts/baselines.py --dataset lfw --variant aligned --k 10 --method hogsvm
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from faceauth import cache, config, metrics, preprocess, protocol

METHODS = ("pixel", "hog", "eigenfaces", "hogsvm")


# ---------------------------------------------------------------- features

def pixel_features(images: np.ndarray) -> np.ndarray:
    """Row per image: mean-subtracted, unit-norm. Makes cosine a plain dot."""
    x = images.reshape(len(images), -1).astype(np.float32)
    x -= x.mean(axis=1, keepdims=True)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-8)


def hog_features(images: np.ndarray) -> np.ndarray:
    """Dense HOG at 8x8 cells, 16x16 blocks.

    OpenCV 5 asserts that (winSize - blockSize) is a multiple of blockStride, so
    a 92-px-wide crop (ORL) is rejected outright. Padding the border by 2 px
    satisfies the arithmetic without rescaling the face, which would change the
    geometry the other methods are measured on.
    """
    h, w = images.shape[1:3]
    pw = (-(w - 16)) % 8
    ph = (-(h - 16)) % 8
    if pw or ph:
        images = np.stack([
            cv2.copyMakeBorder(np.ascontiguousarray(i), 0, ph, 0, pw,
                               cv2.BORDER_REPLICATE) for i in images])
        h, w = images.shape[1:3]
    hog = cv2.HOGDescriptor((w, h), (16, 16), (8, 8), (8, 8), 9)
    return np.stack([hog.compute(np.ascontiguousarray(i)).ravel()
                     for i in images]).astype(np.float32)


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-8)


def pair_feature(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Symmetric pair descriptor: ``|a - b|``.

    Symmetry is the whole point. A signed ``a - b`` labelled "same identity" in
    both orders asks for ``wÂ·d >= 1`` and ``-wÂ·d >= 1`` simultaneously, which
    collapses the classifier into a non-converging degenerate fit.
    """
    return np.abs(a - b)


def _cosine_pairs(xf: np.ndarray, templates: np.ndarray, order: list[int],
                  pairs) -> np.ndarray:
    col = {m: i for i, m in enumerate(order)}
    sim = _unit(xf) @ _unit(templates).T
    return np.array([sim[i, col[m]] for m, i in pairs], dtype=np.float64)


# ---------------------------------------------------------------- methods

def run_cosine(method: str, images: np.ndarray, sp: protocol.Split,
               targets, impostors) -> np.ndarray:
    """Shared path for pixel / hog / eigenfaces: mean enrolment template."""
    t0 = time.time()
    if method == "pixel":
        xf = pixel_features(images)
        proj = xf
    elif method == "hog":
        xf = hog_features(images)
        proj = xf
    elif method == "eigenfaces":
        from sklearn.decomposition import PCA
        xf = images.reshape(len(images), -1).astype(np.float32)
        n_enroll = sum(len(v) for v in sp.enroll.values())
        n_comp = int(min(50, n_enroll - 1, xf.shape[1]))
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=config.SEED)
        pca.fit(xf[np.concatenate([sp.enroll[m] for m in sp.identities])])
        proj = pca.transform(xf).astype(np.float32)
        print(f"  eigenfaces: {n_comp} components, "
              f"{100 * pca.explained_variance_ratio_.sum():.1f}% variance")
    else:
        raise ValueError(method)

    templates = np.stack([proj[sp.enroll[m]].mean(axis=0) for m in sp.identities])
    order = list(sp.identities)
    g = _cosine_pairs(proj, templates, order, targets)
    i = _cosine_pairs(proj, templates, order, impostors)
    print(f"  {method} scored {len(g) + len(i)} pairs in {time.time() - t0:.1f}s")
    return np.concatenate([g, i])


def run_hogsvm(images: np.ndarray, sp: protocol.Split,
               targets, impostors) -> np.ndarray:
    """LinearSVC on symmetric HOG differences; trained on enrolment pairs only.

    The feature is ``|a - b|``, not ``a - b``. Feeding both orders of a signed
    difference with the same label is a contradictory objective: it demands
    ``w.d >= 1`` and ``-w.d >= 1`` for the same pair, which collapses the
    classifier. The absolute difference is symmetric by construction, so one
    sample per pair is enough and the model cannot exploit image order.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    t0 = time.time()
    xf = hog_features(images)
    rng = np.random.default_rng(config.SEED)

    xs, ys = [], []
    for m in sp.identities:
        idx = sp.enroll[m]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                xs.append(pair_feature(xf[idx[a]], xf[idx[b]]))
                ys.append(1)
    n_pos = len(xs)
    ids = sp.identities
    for _ in range(n_pos):
        m1, m2 = rng.choice(ids, 2, replace=False)
        a = xf[int(rng.choice(sp.enroll[m1]))]
        b = xf[int(rng.choice(sp.enroll[m2]))]
        xs.append(pair_feature(a, b))
        ys.append(0)

    feats = np.stack(xs)
    scaler = StandardScaler().fit(feats)
    clf = LinearSVC(C=0.1, dual="auto", max_iter=3000, random_state=config.SEED)
    clf.fit(scaler.transform(feats), np.array(ys))
    print(f"  hogsvm: {len(ys)} pair samples ({n_pos} same-identity), "
          f"{feats.shape[1]}-dim, fit in {time.time() - t0:.1f}s")

    # Scoring keeps every test image against the full enrolment set: the
    # template is the mean absolute difference to those images.
    w = clf.coef_.ravel() / scaler.scale_
    b0 = clf.intercept_[0] - float((clf.coef_.ravel() * scaler.mean_ / scaler.scale_).sum())
    by_model: dict[int, set[int]] = {}
    for m, i in targets + impostors:
        by_model.setdefault(m, set()).add(i)

    scores: dict[tuple[int, int], float] = {}
    for m, idxs in by_model.items():
        enr = xf[sp.enroll[m]]
        rows = sorted(idxs)
        d = np.abs(xf[rows][:, None, :] - enr[None, :, :]).mean(axis=1)
        s = d @ w + b0
        for j, i in enumerate(rows):
            scores[(m, i)] = float(s[j])

    g = np.array([scores[(m, i)] for m, i in targets])
    imp = np.array([scores[(m, i)] for m, i in impostors])
    return np.concatenate([g, imp])


# ---------------------------------------------------------------- driver

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["orl", "lfw"], default="orl")
    ap.add_argument("--variant", default="aligned")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--method", default="all", choices=(*METHODS, "all"))
    ap.add_argument("--clahe", choices=["on", "off"], default="off")
    ap.add_argument("--max-impostor", type=int, default=20000)
    ap.add_argument("--bootstrap", type=int, default=500)
    args = ap.parse_args()

    k = args.k if args.k is not None else config.UTTS_ENROLL
    images, labels = cache.load_cache(args.dataset, args.variant)
    if args.clahe == "on":
        images = np.stack([preprocess.equalise(i) for i in images])

    sp = protocol.split(labels, k)
    targets, impostors = protocol.plan(sp, max_impostor=args.max_impostor)
    gid = np.array([m for m, _ in targets])
    g_probe = np.array([labels[i] for _, i in targets])
    iid = np.array([m for m, _ in impostors])
    i_probe = np.array([labels[i] for _, i in impostors])
    variant = args.variant if args.dataset == "lfw" else "native"
    methods = METHODS if args.method == "all" else (args.method,)
    print(f"{args.dataset.upper()}: {len(images)} images, "
          f"{len(sp.identities)} identities, k={k}, clahe={args.clahe}",
          flush=True)
    print(f"trials: {len(targets)} target / {len(impostors)} impostor", flush=True)
    print("score is a similarity: higher means more likely genuine\n")

    config.OUTPUTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in methods:
        print(f"[{method}]")
        sim = run_cosine(method, images, sp, targets, impostors) \
            if method != "hogsvm" else run_hogsvm(images, sp, targets, impostors)
        g, imp = -sim[:len(targets)], -sim[len(targets):]   # distance scale
        e, thr = metrics.eer(g, imp)
        tar, _ = metrics.tar_at_far(g, imp, 0.01)
        lo, hi = metrics.bootstrap_ci(g, imp, gid, iid, g_probe, i_probe,
                                      n=args.bootstrap) \
            if args.bootstrap else (float("nan"), float("nan"))
        rows.append((method, e, lo, hi, tar))
        tag = f"{args.dataset}_{variant}_k{k}_clahe-{args.clahe}_base-{method}"
        np.savez_compressed(config.OUTPUTS / f"scores_{tag}.npz",
                            genuine=g, impostor=imp, genuine_ids=gid,
                            impostor_ids=iid, genuine_probe=g_probe,
                            impostor_probe=i_probe, eer=e, ci_low=lo,
                            ci_high=hi, bootstrap=args.bootstrap,
                            seed=config.SEED, k=k, method=method)
        (config.OUTPUTS / f"report_{tag}.txt").write_text(
            f"dataset={args.dataset}\nvariant={variant}\nk_enroll={k}\n"
            f"clahe={args.clahe}\nmethod={method}\n"
            f"EER={e * 100:.3f}%\nEER_CI_LOW={lo * 100:.3f}%\n"
            f"EER_CI_HIGH={hi * 100:.3f}%\nTAR_at_FAR1={tar * 100:.3f}%\n"
            f"targets={len(g)}\nimpostors={len(imp)}\n", encoding="utf-8")
        print(f"  EER {e * 100:6.2f}%  (95% CI {lo * 100:5.2f}-{hi * 100:5.2f}%)"
              f"  TAR@FAR=1% {tar * 100:5.2f}%\n", flush=True)

    print("=" * 62)
    print(f"  {'method':12}  {'EER':>7}  {'95% CI':>16}  {'TAR@FAR=1%':>10}")
    for method, e, lo, hi, tar in rows:
        print(f"  {method:12}  {e * 100:6.2f}%  ({lo * 100:5.2f}-{hi * 100:5.2f}%)"
              f"  {tar * 100:9.2f}%")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
