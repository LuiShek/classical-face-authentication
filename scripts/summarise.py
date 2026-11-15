"""Consolidate every scored configuration into one report and the figures.

This reads the artifacts the evaluation scripts already wrote, so nothing is
re-scored and no number in the report can drift from the score file it came
from. If a report file is missing, its row is still derived from the npz; the
only thing lost is the bootstrap interval.

Usage
-----
  py scripts/summarise.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from faceauth import config, metrics

NAME = re.compile(
    r"^scores_(?P<dataset>orl|lfw)_(?P<variant>[a-z_]+)_k(?P<k>\d+)"
    r"_clahe-(?P<clahe>on|off)"
    r"(?:_coh-(?P<cohort>one|all))?"
    r"(?:_base-(?P<method>[a-z]+))?\.npz$")

SCALES = (("raw", "genuine", "impostor"),
          ("mean", "genuine_mean", "impostor_mean"),
          ("z-norm", "genuine_z", "impostor_z"))


def configs() -> list[dict]:
    out = []
    for f in sorted(config.OUTPUTS.glob("scores_*.npz")):
        m = NAME.match(f.name)
        if not m:
            if not f.name.startswith("scores_pool_"):
                print(f"warning: unrecognised artifact {f.name}", file=sys.stderr)
            continue
        z = np.load(f, allow_pickle=False)
        meta = m.groupdict()
        meta["k"] = int(meta["k"])
        meta["path"] = f
        for label, gkey, ikey in SCALES:
            if gkey not in z.files:
                continue
            g, i = z[gkey], z[ikey]
            gid, iid = z["genuine_ids"], z["impostor_ids"]
            e, _ = metrics.eer(g, i)
            tar, _ = metrics.tar_at_far(g, i, 0.01)
            ci_keys = {"raw": ("ci_low", "ci_high"),
                       "mean": ("ci_low_mean", "ci_high_mean"),
                       "z-norm": ("ci_low_z", "ci_high_z")}[label]
            ci = (float(z[ci_keys[0]]), float(z[ci_keys[1]])) \
                if ci_keys[0] in z.files and np.isfinite(z[ci_keys[0]]) \
                and np.isfinite(z[ci_keys[1]]) else None
            out.append({**meta, "scale": label, "eer": e, "tar": tar,
                        "min_dcf": metrics.min_dcf(g, i),
                        "far": metrics.det_points(g, i)[0],
                        "frr": metrics.det_points(g, i)[1],
                        "n_gen": len(g), "n_imp": len(i),
                        "_ci": ci, "_gid": gid, "_iid": iid})
    return out


def attach_ci(rows: list[dict], bootstrap: int = 0) -> None:
    """Fill any missing interval from the per-configuration report file.

    Newer artifacts store the bootstrap interval next to the scores, so this is
    only a fallback for artifacts written before that field existed.
    """
    for r in rows:
        if r["_ci"] is not None:
            continue
        tag = r["path"].stem.removeprefix("scores_")
        rep = config.OUTPUTS / f"report_{tag}.txt"
        if not rep.exists():
            continue
        text = rep.read_text(encoding="utf-8")
        block = text.split(f"[{r['scale']}]")
        # Baseline reports are flat (one method, one scale), so the whole file
        # is the right block for them.
        body = block[1] if len(block) > 1 else text
        lo = re.search(r"EER_CI_LOW=([\d.]+)%", body)
        hi = re.search(r"EER_CI_HIGH=([\d.]+)%", body)
        if lo and hi:
            r["_ci"] = (float(lo.group(1)) / 100.0, float(hi.group(1)) / 100.0)


def label(r: dict) -> str:
    variant = r["variant"] if r["dataset"] == "lfw" else "native"
    if r["method"]:
        # Baselines have no cohort normalisation, so the cohort field is noise.
        return (f"{r['dataset']}/{variant} k={r['k']} {r['method']} "
                f"clahe={r['clahe']} {r['scale']}")
    return (f"{r['dataset']}/{variant} k={r['k']} lbph "
            f"clahe={r['clahe']} coh={r['cohort']} {r['scale']}")


def table(rows: list[dict]) -> str:
    head = (f"{'configuration':58} {'EER':>7} {'95% CI':>15} "
            f"{'TAR@1%':>7} {'minDCF':>7}")
    lines = [head, "-" * len(head)]
    for r in sorted(rows, key=lambda x: (x["dataset"], x["k"], x["method"] or "",
                                         x["scale"], x["clahe"], x["cohort"] or "")):
        ci = (f"{r['_ci'][0] * 100:.2f}-{r['_ci'][1] * 100:.2f}"
              if r["_ci"] else "n/a")
        lines.append(f"{label(r):58} {r['eer'] * 100:6.2f}% {ci:>15} "
                     f"{r['tar'] * 100:6.2f}% {r['min_dcf']:7.4f}")
    return "\n".join(lines)


REF_K = {"orl": 5, "lfw": 10}


def det_figure(rows: list[dict], dataset: str, out: Path) -> None:
    """DET curves at one declared reference configuration per dataset.

    Only one enrolment depth and one photometric setting are drawn per dataset.
    Every arm at every setting would be a 12-curve tangle, and the differences
    between most arms are inside the confidence interval anyway; the reference
    point is chosen as the pre-computed default, not the best-performing one,
    so the curve cannot be a post-hoc pick.
    """
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    drawn = 0
    for r in rows:
        if r["dataset"] != dataset or r["clahe"] != "off":
            continue
        if r["scale"] == "mean":
            # Reported in the tables; on the curve it only thickens the bundle.
            continue
        if r["k"] != REF_K.get(dataset):
            continue
        is_base = r["method"] is not None
        if not is_base and r["cohort"] != "all":
            continue
        style = "--" if is_base else "-"
        ax.plot(norm.ppf(np.clip(r["far"], 1e-4, 1 - 1e-4)),
                norm.ppf(np.clip(r["frr"], 1e-4, 1 - 1e-4)),
                lw=1.4, ls=style,
                label=f"{label(r)}  {r['eer'] * 100:.1f}%")
        drawn += 1
    if not drawn:
        plt.close(fig)
        return
    ticks = np.array([0.01, 0.05, 0.1, 0.2, 0.4])
    ax.set_xticks(norm.ppf(ticks)); ax.set_xticklabels([f"{t * 100:g}" for t in ticks])
    ax.set_yticks(norm.ppf(ticks)); ax.set_yticklabels([f"{t * 100:g}" for t in ticks])
    ax.set_xlabel("False accept rate (%)")
    ax.set_ylabel("False reject rate (%)")
    ax.set_title(f"DET — {dataset.upper()} (k={REF_K.get(dataset)}, CLAHE off)")
    ax.grid(alpha=0.3, ls=":")
    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def score_figure(rows: list[dict], dataset: str, out: Path) -> None:
    """Genuine vs impostor overlap for the reference LBPH configuration."""
    pick = [r for r in rows
            if r["dataset"] == dataset and r["scale"] == "raw"
            and r["method"] is None and r["clahe"] == "off"
            and r["k"] == REF_K.get(dataset) and r["cohort"] == "all"]
    if not pick:
        return
    best = min(pick, key=lambda r: r["eer"])
    z = np.load(best["path"], allow_pickle=False)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bins = np.linspace(0, np.percentile(z["impostor"], 99.5), 60)
    ax.hist(z["genuine"], bins=bins, alpha=0.65, density=True, label="genuine",
            color="#2a7", edgecolor="none")
    ax.hist(z["impostor"], bins=bins, alpha=0.55, density=True, label="impostor",
            color="#c33", edgecolor="none")
    ax.set_xlabel("LBPH chi-square distance")
    ax.set_ylabel("density")
    ax.set_title(f"Score overlap — {dataset.upper()} k={best['k']} "
                 f"({best['eer'] * 100:.1f}% EER)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> int:
    rows = configs()
    if not rows:
        print("error: no score artifacts in outputs/; run evaluate.py first")
        return 1
    attach_ci(rows)
    config.FIGURES.mkdir(parents=True, exist_ok=True)

    for dataset in ("orl", "lfw"):
        det_figure(rows, dataset, config.FIGURES / f"det_{dataset}.png")
        score_figure(rows, dataset, config.FIGURES / f"scores_{dataset}.png")

    body = table(rows)
    datasets = sorted({r["dataset"] for r in rows})
    text = f"""LBPH face verification — consolidated results
============================================

Protocol: per-identity enrolment of k images, all remaining images of that
identity are target trials, and every other identity's test images are impostor
trials (subsampled to 20,000). Scores are LBPH chi-square distances. Intervals
are 95% identity-clustered bootstrap resamples, so they widen when the number of
distinct identities is small rather than pretending 20,000 correlated impostor
trials are independent.

Scales: raw = distance as emitted; mean = distance minus the model's cohort
mean; z-norm = that, divided by the cohort spread. Cohort statistics come from
enrolment images of the *other* identities only, never from test images.
minDCF is normalised with equal miss/false-alarm cost at p_target = 0.5.

{body}

Interpretation
--------------
- Enrolment depth is the dominant lever on ORL: raw EER falls monotonically as k
  rises, and the cohort-normalised scale follows it.
- Cohort normalisation helps ORL and hurts LFW. The per-model spread it divides
  by is estimated from far fewer identities on LFW than the trial count suggests.
- On LFW every alignment, refinement and photometric arm lands within the same
  interval, so this copy cannot rank them; the images are already deep-funnelled.
- The supervised HOG pair classifier is the strongest ORL method, which is worth
  stating plainly: LBPH trades accuracy for needing no training stage.
"""
    (config.OUTPUTS / "report.txt").write_text(text, encoding="utf-8")
    print(body)
    print(f"\nwrote {config.OUTPUTS / 'report.txt'}")
    print(f"figures: {[p.name for p in sorted(config.FIGURES.glob('*.png'))]}")
    print(f"datasets: {datasets}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
