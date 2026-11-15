"""Compare single-frame LBPH with median pooling over N probe frames.

Models and enrolment splits are identical between arms. Pooled impostor groups
contain frames from one source identity only; mixing identities inside a median
would create a trial that no real login can produce.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from faceauth import cache, config, lbph, metrics, protocol


def grouped_trials(sp: protocol.Split, takes: int, max_impostor: int | None,
                   seed: int = config.SEED):
    if takes < 1:
        raise ValueError("takes must be >= 1")
    targets, impostors = [], []
    groups = {}
    for ident in sp.identities:
        idx = np.asarray(sp.test[ident], dtype=np.int32)
        # Avoid grouping adjacent filenames/recording bursts. The per-identity
        # permutation is deterministic and shared by paired reducers.
        rng = np.random.default_rng(seed + 1009 * ident + takes)
        idx = idx[rng.permutation(len(idx))]
        groups[ident] = [tuple(int(v) for v in idx[i:i + takes])
                         for i in range(0, len(idx) - takes + 1, takes)]
    for model_id in sp.identities:
        targets.extend((model_id, group) for group in groups[model_id])
        for source_id in sp.identities:
            if source_id == model_id:
                continue
            impostors.extend((model_id, group) for group in groups[source_id])
    if max_impostor is not None and len(impostors) > max_impostor:
        rng = np.random.default_rng(seed + takes)
        keep = np.sort(rng.choice(len(impostors), max_impostor, replace=False))
        impostors = [impostors[i] for i in keep]
    return targets, impostors


def evaluate(images, labels, *, k: int, takes: int, max_impostor: int,
             bootstrap: int, reducer: str = "median"):
    sp = protocol.split(labels, k)
    targets, impostors = grouped_trials(sp, takes, max_impostor)
    models = {m: lbph.LbphModel().fit(images[sp.enroll[m]]) for m in sp.identities}
    cache_dist: dict[tuple[int, int], float] = {}

    def distance(model_id: int, idx: int) -> float:
        key = model_id, idx
        if key not in cache_dist:
            cache_dist[key] = models[model_id].distance(images[idx])
        return cache_dist[key]

    def pooled(model_id: int, group: tuple[int, ...]) -> float:
        values = [distance(model_id, i) for i in group]
        if reducer == "first":
            return float(values[0])
        if reducer == "median":
            return float(np.median(values))
        raise ValueError(f"unknown reducer {reducer!r}")

    g = np.array([pooled(m, grp) for m, grp in targets])
    imp = np.array([pooled(m, grp) for m, grp in impostors])
    gid = np.array([m for m, _ in targets])
    g_probe = np.array([labels[grp[0]] for _, grp in targets])
    iid = np.array([m for m, _ in impostors])
    i_probe = np.array([labels[grp[0]] for _, grp in impostors])
    e, threshold = metrics.eer(g, imp)
    tar, _ = metrics.tar_at_far(g, imp, 0.01)
    ci = (metrics.bootstrap_ci(g, imp, gid, iid, g_probe, i_probe, n=bootstrap)
          if bootstrap else (float("nan"), float("nan")))
    return {"genuine": g, "impostor": imp, "genuine_ids": gid,
            "genuine_probe": g_probe, "impostor_ids": iid,
            "impostor_probe": i_probe, "eer": e, "threshold": threshold,
            "tar": tar, "ci": ci, "pairs": len(cache_dist)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["orl", "lfw"], required=True)
    ap.add_argument("--variant", default="aligned")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--takes", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--max-impostor", type=int, default=20000)
    ap.add_argument("--bootstrap", type=int, default=500)
    args = ap.parse_args()

    images, labels = cache.load_cache(args.dataset, args.variant)
    config.OUTPUTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for takes in args.takes:
        reducers = ("median",) if takes == 1 else ("first", "median")
        for reducer in reducers:
            t0 = time.time()
            result = evaluate(images, labels, k=args.k, takes=takes,
                              max_impostor=args.max_impostor,
                              bootstrap=args.bootstrap, reducer=reducer)
            label = (f"single-paired-n{takes}" if reducer == "first"
                     else f"median-n{takes}")
            rows.append((label, takes, reducer, result))
            artifact = (f"n1-paired-n{takes}" if reducer == "first"
                        else f"n{takes}")
            tag = f"pool_{args.dataset}_{args.variant}_k{args.k}_{artifact}"
            np.savez_compressed(config.OUTPUTS / f"scores_{tag}.npz",
                                **{k: v for k, v in result.items()
                                   if k not in {"pairs"}},
                                ci_low=result["ci"][0], ci_high=result["ci"][1],
                                bootstrap=args.bootstrap, seed=config.SEED,
                                takes=takes, reducer=reducer, k=args.k)
            print(f"{label}: EER {result['eer'] * 100:.2f}% "
                  f"(95% CI {result['ci'][0] * 100:.2f}-"
                  f"{result['ci'][1] * 100:.2f}), "
                  f"TAR@FAR=1% {result['tar'] * 100:.2f}%, "
                  f"{len(result['genuine'])}T/{len(result['impostor'])}I, "
                  f"{time.time() - t0:.0f}s", flush=True)

    report = [f"dataset={args.dataset}", f"variant={args.variant}",
              f"k_enroll={args.k}"]
    for label, takes, reducer, r in rows:
        report.extend([f"[{label}]", f"takes={takes}", f"reducer={reducer}",
                       f"EER={r['eer'] * 100:.3f}%",
                       f"EER_CI_LOW={r['ci'][0] * 100:.3f}%",
                       f"EER_CI_HIGH={r['ci'][1] * 100:.3f}%",
                       f"TAR_at_FAR1={r['tar'] * 100:.3f}%",
                       f"targets={len(r['genuine'])}",
                       f"impostors={len(r['impostor'])}"])
    out = config.OUTPUTS / f"report_pooling_{args.dataset}_{args.variant}_k{args.k}.txt"
    out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
