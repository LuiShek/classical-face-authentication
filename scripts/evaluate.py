"""Measure LBPH on the same trial protocol for ORL and LFW.

Score is the chi-square distance from a test image to a per-identity model;
lower means more similar. Two score scales are reported in one pass:

  raw   the distance as LBPH emits it
  z     (d - cohort_mean) / cohort_std, where the cohort statistics come from
        one *enrolment* image of every other identity

The cohort is built from enrolment images only, never from test images, so the
normalisation cannot peek at the trials it is being scored on.

EER is reported with an identity-clustered bootstrap CI, because trials sharing
an identity are not independent.

Usage
-----
  py scripts/evaluate.py --dataset orl --k 5
  py scripts/evaluate.py --dataset lfw --variant aligned --k 10
  py scripts/evaluate.py --dataset lfw --variant boxed --k 10 --clahe on
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from faceauth import cache, config, lbph, metrics, preprocess, protocol


def load(dataset: str, variant: str, size: int | None):
    return cache.load_cache(dataset, variant, size)


def cohort_stats(models, sp, dist,
                 pool: str = "one") -> tuple[dict[int, float], dict[int, float]]:
    """Per-model mean/std of distances to the other identities' enrolment images.

    Enrolment images are used deliberately: they are available before any test
    trial, which is what makes the normalisation deployable rather than
    evaluation-only. ``pool="one"`` takes a single image per other identity so
    the statistics are not dominated by whichever identity happens to be
    photographed most; ``pool="all"`` uses the whole enrolment set for a lower
    variance estimate.
    """
    means, stds = {}, {}
    for m in sp.identities:
        others = [o for o in sp.identities if o != m]
        ds = np.array([dist(m, j)
                       for o in others
                       for j in (sp.enroll[o] if pool == "all" else sp.enroll[o][:1])],
                      dtype=np.float64)
        means[m] = float(ds.mean())
        stds[m] = float(ds.std(ddof=1)) if len(ds) > 1 and ds.std() > 0 else 1.0
    return means, stds


def summarise(g, imp, gid, iid, g_probe, i_probe, bootstrap: int) -> dict:
    e, thr = metrics.eer(g, imp)
    tar, tar_thr = metrics.tar_at_far(g, imp, 0.01)
    lo, hi = (metrics.bootstrap_ci(g, imp, gid, iid, g_probe, i_probe, n=bootstrap)
              if bootstrap else (float("nan"), float("nan")))
    return {"eer": e, "thr": thr, "tar": tar, "tar_thr": tar_thr,
            "ci_low": lo, "ci_high": hi}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["orl", "lfw"], default="orl")
    ap.add_argument("--variant", default="aligned",
                    help="LFW only: aligned | boxed | aligned_refined")
    ap.add_argument("--k", type=int, default=None, help="enrol images per identity")
    ap.add_argument("--clahe", choices=["on", "off"], default="off",
                    help="photometric normalisation at scoring time")
    ap.add_argument("--max-impostor", type=int, default=20000)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--cohort", choices=["one", "all"], default="all",
                    help="enrolment images per cohort identity for normalisation")
    ap.add_argument("--size", type=int, default=None,
                    help="resize cached images before LBPH (ablation only)")
    args = ap.parse_args()

    k = args.k if args.k is not None else config.UTTS_ENROLL
    images, labels = load(args.dataset, args.variant, args.size)
    if args.clahe == "on":
        t_c = time.time()
        images = np.stack([preprocess.equalise(i) for i in images])
        print(f"CLAHE applied to {len(images)} images in {time.time() - t_c:.1f}s")
    t0 = time.time()

    sp = protocol.split(labels, k)
    targets, impostors = protocol.plan(sp, max_impostor=args.max_impostor)
    counts = sp.counts()
    variant = args.variant if args.dataset == "lfw" else "native"
    print(f"{args.dataset.upper()}: {len(images)} images, "
          f"{counts['identities']} identities, k={k} enrol, clahe={args.clahe}",
          flush=True)
    print(f"trials: {len(targets)} target / {len(impostors)} impostor", flush=True)

    models = {m: lbph.LbphModel().fit(images[sp.enroll[m]]) for m in sp.identities}
    cache: dict[tuple[int, int], float] = {}

    def dist(model_id: int, img_idx: int) -> float:
        key = (model_id, img_idx)
        v = cache.get(key)
        if v is None:
            v = models[model_id].distance(images[img_idx])
            cache[key] = v
        return v

    g = np.array([dist(m, i) for m, i in targets], dtype=np.float64)
    imp = np.array([dist(m, i) for m, i in impostors], dtype=np.float64)
    # Track both the enrolled model identity and the probe source identity so
    # the bootstrap can cluster on subjects instead of individual trials.
    gid = np.array([m for m, _ in targets])
    g_probe = np.array([labels[i] for _, i in targets])
    iid = np.array([m for m, _ in impostors])
    i_probe = np.array([labels[i] for _, i in impostors])

    means, stds = cohort_stats(models, sp, dist, pool=args.cohort)
    # Counted after scoring, otherwise it reads 0 even though the pairs ran.
    n_cohort = len(cache) - len(g) - len(imp)
    mu_g = np.array([means[m] for m in gid])
    sd_g = np.array([stds[m] for m in gid])
    mu_i = np.array([means[m] for m in iid])
    sd_i = np.array([stds[m] for m in iid])
    # Mean-only and full z are kept apart because dividing by a per-model std
    # can hurt models whose spread is already small: if mean-only wins, the
    # variance term is doing damage and should be dropped.
    mg, mi = g - mu_g, imp - mu_i
    zg, zi = mg / sd_g, mi / sd_i

    scales = {
        "raw": (g, imp),
        "mean": (mg, mi),
        "z-norm": (zg, zi),
    }
    results = {n: summarise(a, b, gid, iid, g_probe, i_probe, args.bootstrap)
               for n, (a, b) in scales.items()}

    print()
    print("=" * 74)
    print(f"  {'scale':7}  {'EER':>7}  {'95% CI':>16}  {'TAR@FAR=1%':>10}  "
          f"{'thr(EER)':>9}")
    for name, s in results.items():
        print(f"  {name:7}  {s['eer'] * 100:6.2f}%  "
              f"({s['ci_low'] * 100:5.2f}-{s['ci_high'] * 100:5.2f}%)  "
              f"{s['tar'] * 100:9.2f}%  {s['thr']:9.3f}")
    print("-" * 74)
    print(f"  variant {args.dataset}/{variant}  k={k}  clahe={args.clahe}  "
          f"cohort={args.cohort}  trials {len(g)}T/{len(imp)}I  "
          f"cohort {n_cohort} pairs  total {len(cache)}")
    print("=" * 74)
    print(f"DONE in {time.time() - t0:.0f}s")

    config.OUTPUTS.mkdir(parents=True, exist_ok=True)
    # Every axis that changes the number belongs in the name: a k=10 run once
    # silently overwrote the k=3 report, and CLAHE would do the same here.
    tag = (f"{args.dataset}_{variant}_k{k}"
           f"_clahe-{args.clahe}_coh-{args.cohort}")
    np.savez_compressed(
        config.OUTPUTS / f"scores_{tag}.npz",
        genuine=g, impostor=imp, genuine_ids=gid, impostor_ids=iid,
        genuine_probe=g_probe, impostor_probe=i_probe,
        genuine_mean=mg, impostor_mean=mi, genuine_z=zg, impostor_z=zi,
        cohort_mean=np.array([means[m] for m in sp.identities]),
        cohort_std=np.array([stds[m] for m in sp.identities]),
        eer=results["raw"]["eer"], eer_mean=results["mean"]["eer"],
        eer_z=results["z-norm"]["eer"],
        threshold=results["raw"]["thr"], threshold_z=results["z-norm"]["thr"],
        ci_low=results["raw"]["ci_low"], ci_high=results["raw"]["ci_high"],
        ci_low_mean=results["mean"]["ci_low"], ci_high_mean=results["mean"]["ci_high"],
        ci_low_z=results["z-norm"]["ci_low"], ci_high_z=results["z-norm"]["ci_high"],
        bootstrap=args.bootstrap, seed=config.SEED,
        k=k, clahe=args.clahe, cohort=args.cohort)

    def block(name: str, s: dict) -> str:
        return (f"[{name}]\nEER={s['eer'] * 100:.3f}%\n"
                f"EER_CI_LOW={s['ci_low'] * 100:.3f}%\n"
                f"EER_CI_HIGH={s['ci_high'] * 100:.3f}%\n"
                f"TAR_at_FAR1={s['tar'] * 100:.3f}%\n"
                f"threshold_EER={s['thr']:.5f}\n"
                f"threshold_FAR1={s['tar_thr']:.5f}\n")

    (config.OUTPUTS / f"report_{tag}.txt").write_text(
        f"dataset={args.dataset}\nvariant={variant}\nk_enroll={k}\n"
        f"clahe={args.clahe}\ncohort={args.cohort}\n"
        f"targets={len(g)}\nimpostors={len(imp)}\n"
        f"cohort_pairs={n_cohort}\npairs_scored={len(cache)}\n"
        f"bootstrap_iters={args.bootstrap}\n"
        + "".join(block(n, s) for n, s in results.items()),
        encoding="utf-8")
    print(f"wrote {config.OUTPUTS / f'report_{tag}.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
