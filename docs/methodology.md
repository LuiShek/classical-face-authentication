# Methodology

What is measured, how, and what the measurement cannot tell you. If a number in
the README disagrees with this document, this document is the one to trust.

## 1. What the system is

A verification system, not an identification system. Given an enrolled identity
and a single probe frame it answers "is this the enrolled person?", scoring with
LBPH (Local Binary Patterns, Histograms) — a texture descriptor, no neural
network, no training corpus.

LBPH labels every pixel by comparing it with its 8 neighbours, then builds a
histogram of those labels per grid cell. Comparing neighbourhoods rather than
intensities makes the descriptor largely invariant to monotone lighting change.
Distances between histograms are chi-square.

## 2. Data

| Dataset | Images | Identities | Geometry | Role |
|---|---:|---:|---|---|
| ORL / AT&T | 400 | 40 | native 92×112, hand-cropped | controlled, no detector |
| LFW (deep-funnelled) | 3,023 | 62 (≥20 images) | 112×112 after detection+alignment | in the wild, detector required |

Both sources are fetched from hosts verified reachable from this machine;
`sklearn.datasets.fetch_olivetti_faces` and `fetch_lfw_people` are **not** used
because figshare answers them with HTTP 403 (sklearn downloads via
`urlretrieve`, which sends no User-Agent) and `vis-www.cs.umass.edu` does not
resolve here.

**ORL bypasses detection deliberately.** The images are already cropped to the
face by hand. Running a detector on them would score the detector on a task it
never had to solve, and resampling 92×112 to a square would distort the texture
LBPH reads.

**The LFW copy is deep-funnelled, which limits one claim.** These images are
already face-centred, so the aligned and unaligned arms land in nearly the same
place. The alignment ablation can show that alignment does not *hurt*; it cannot
show what alignment buys on raw in-the-wild photographs. Un-funnelled LFW would
be needed for that, and is listed as an open item rather than claimed as done.

## 3. Trial protocol

For each identity with at least `k + 1` images, `k` images are enrolled and all
remaining images are probes. The split is seeded (seed 0) and identical across
every arm, so all comparisons are paired.

- **Target trials**: (model of identity *i*, probe of identity *i*).
- **Impostor trials**: (model of identity *i*, probe of identity *j ≠ i*),
  subsampled to 20,000 with a fixed seed.

Impostor trials vastly outnumber target trials because that is what a real gate
sees. It also means an EER computed without clustering is over-confident: trials
sharing an identity are correlated. All intervals here are 95%
**identity-clustered** bootstrap intervals. Each replicate draws a set of
identities with replacement and recomputes the metric from every trial whose
enrolled model and probe source both lie inside that draw. Clustering on the
enrolled model alone would leave impostor trials that share a probe identity
split across clusters and understate the uncertainty; on LFW that model-only
variant gives a visibly narrower interval. With 40 identities the interval is
wide, and that width is honest. The default is 500 replicates with a fixed seed;
the count and seed are stored in each artifact.

## 4. Caches: raw pixels, not normalised pixels

`scripts/prepare_data.py` caches **raw greyscale** crops and marks them
`stage="raw"`. Photometric normalisation is applied at scoring time
(`evaluate.py --clahe on|off`), because it is a model choice: baking it into the
cache means every comparison requires re-detecting 3,000 images.

`faceauth.cache.load_cache` refuses a cache whose `stage` is not `raw`. This is
not defensive decoration: an earlier build baked CLAHE into the cache, and
scoring that as if it were raw would have attributed equalisation to the model.
The guard was added after the mistake, and it fails loudly.

LFW is cached three ways so ablations need no re-detection:

| Key | Crop |
|---|---|
| `aligned` | 5-point similarity transform onto a canonical template |
| `boxed` | detector box, axis-aligned, no rotation |
| `aligned_refined` | `aligned`, re-detected and re-warped once (needs `--refine`) |

## 5. Detection and alignment

Detection uses YuNet (`cv2.FaceDetectorYN`), because OpenCV 5 no longer ships
Haar cascades at all — `cv2.data.haarcascades` contains only `__init__.py`. YuNet
also returns five landmarks, which is what makes alignment possible.

Alignment is `cv2.estimateAffinePartial2D`: 4 degrees of freedom (rotation,
uniform scale, translation) onto the canonical 5-point template. A full affine
would fit 6 degrees and could shear a face into place, which changes the texture
positions LBPH depends on.

Measured registration residual, by re-detecting landmarks on the warped crop:

| Variant | Mean residual | p95 |
|---|---:|---:|
| one warp | 4.29 px | 7.29 |
| re-detected and re-warped once | 3.91 px | 8.04 |
| re-warped twice | 3.81 px | 8.29 |

The refinement buys 0.4 px on average and makes the tail slightly worse, which
is why it stays an ablation arm rather than becoming the default.

## 6. Scores and normalisation

The score is the chi-square distance to the per-identity model (lower = more
similar). Three scales are reported:

| Scale | Definition |
|---|---|
| `raw` | the distance as emitted |
| `mean` | `d - μ_m`, where `μ_m` is the model's cohort mean |
| `z-norm` | `(d - μ_m) / σ_m` |

Cohort statistics come from distances between model *m* and the **enrolment**
images of every other identity — never test images. That is what keeps the
normalisation deployable: those images exist before any probe arrives.

`minDCF` is the minimum detection cost over all thresholds, with equal
miss/false-alarm cost at `p_target = 0.5`, normalised so 1.0 means "no better
than the trivial strategy".

## 7. Baselines

Four non-LBPH methods run on the identical splits:

| Method | What it is |
|---|---|
| `pixel` | mean-subtracted, L2-normalised greyscale, cosine to mean enrolment vector |
| `hog` | `cv2.HOGDescriptor` at 8×8 cells, cosine to mean enrolment vector |
| `eigenfaces` | PCA (50 components) fit on enrolment images, cosine in the projected space |
| `hogsvm` | LinearSVC on `\|a - b\|` HOG pair differences, trained on enrolment pairs |

Two design notes worth keeping:

- HOG needs `(winSize - blockSize) % blockStride == 0`. ORL's 92-px width
  violates it, so the border is padded by 2 px with `BORDER_REPLICATE` rather
  than rescaling — rescaling would change the geometry every other method sees.
- `hogsvm` uses `|a - b|`, not `a - b`. Labelling both orders of a signed
  difference as positive asks for `w·d ≥ 1` and `-w·d ≥ 1` simultaneously, which
  collapses the classifier; an early version did exactly that and produced a
  50.67% EER with a 450-second non-converging fit. The absolute difference is
  symmetric by construction.

**`hogsvm` is an optimistic comparison, and the report says so.** It is trained
on enrolment pairs of the very identities it is then tested on. LBPH uses the
same enrolment images as templates, but it does not fit weights to them. Measured
anyway: it reaches 1.97% EER on ORL and 22.14% on LFW, so a learned pair metric
does beat LBPH here — with the training-set caveat attached. A protocol that
trains on disjoint background identities would be stricter and is an open item.

## 8. What this measurement does not cover

- **Anti-spoofing ceiling.** Login now requires a randomized active head-turn
  trajectory, which rejects a stationary print. It is not certified PAD: an
  adaptive replay, articulated print or mask may still pass.
- **Demographic and acquisition bias.** ORL is 40 people under controlled
  conditions; LFW is celebrities from news photos. Neither is a deployment
  sample.
- **Threshold transfer.** An EER threshold from these datasets is not a live
  operating point — which is why the live gate calibrates per identity (§9)
  rather than reading a number off the tables above.
- **Alignment on raw images.** See §2: the available LFW copy is already
  aligned.

## 9. The live gate

The evaluation measures the method. The gate decides for one enrolled person, and
it takes its threshold from data that exists before any probe arrives:

| Quantity | Source |
|---|---|
| genuine scores | leave-one-out over this identity's enrolment frames |
| impostor scores | this identity's model scored against registered users plus a labeled 30-identity LFW background cohort |
| threshold | stricter of the genuine 95th percentile and the zero-observed-FAR cohort boundary |

Two rules the measurements forced:

- **A cohort of one identity is not a cohort.** Below
  `MIN_COHORT_IDENTITIES = 10` other identities, the EER path is refused and the
  threshold controls only the false-reject rate, with the false-accept rate
  recorded as unmeasured. Measured: with a single other identity the EER path
  reported 40–60%, which is noise with a number attached. The gate therefore
  refuses to treat a tiny cohort as false-accept evidence.
- **Thresholds are re-derived, not inherited.** `calibrate --all` recomputes
  every identity against the current cohort, because whoever enrolled first would
  otherwise stay calibrated against a cohort of one.
- **False accepts dominate the live objective.** Dataset tables still report EER
  for comparison, but live calibration targets observed cohort FAR=0. The EER
  remains metadata; it no longer chooses the gate threshold.

The dedicated cohort is deterministically sampled (seed 0, 30 identities × 3
aligned raw crops) from the existing LFW cache. Labels are retained, so the gate
counts identities rather than photographs. Its source-cache SHA-256 and sampling
settings are stored beside the DPAPI-encrypted payload. LFW is domain-mismatched
to a webcam, so zero observed cohort false accepts is evidence about this cohort,
not a guarantee about the world.

Storage, per identity, under `%LOCALAPPDATA%\faceauth\faces\<slug>\`:
`model.yml.dpapi` (OpenCV model bytes), `crops.npz.dpapi` (enrolment frames, so
recalibration needs no camera), and `meta.json` (non-biometric threshold, scale,
and counts). DPAPI binds ciphertext to the current Windows account. Migration
first creates and application-verifies encrypted copies, then removes plaintext
only with `--yes`; `secure restore` is the rollback. A name containing a path
separator is rejected rather than rewritten, and `delete` also requires `--yes`.

Decision quality on real photographs is reported in the README. It is a working
gate, not an accurate one at five frames, and the per-identity EERs (18–60%
across 13 identities) are coarse because each rests on five genuine scores.

The primary CLI is deliberately an authentication workflow:

1. `faceauth` on an empty store asks for a name and captures five spaced frames.
2. It offers to log in immediately.
3. Later runs show Login / Register / List / Exit.
4. With multiple profiles, Login asks which name to authenticate. It does not
   infer a name from the lowest LBPH distance: that would turn a verifier with a
   measured ~26% LFW EER into an unsafe open-set identifier.

Webcam enrolment waits 0.7 seconds between accepted frames and prints small pose
hints. Without spacing, a 17-fps camera records five near-identical crops in
about a quarter-second; calling that five-frame enrolment would overstate the
amount of evidence in the model.

Login then performs two additional controls:

- **Active challenge:** stable neutral landmark geometry → random left/right
  turn (two frames) → return to neutral (two frames), with face area constrained
  to prevent "move a flat photo closer" from counting as a turn.
- **Rate limit:** three failed Login attempts lock the selected identity for 30
  seconds; each later failure doubles the delay to a 15-minute cap. A successful
  Login clears the state. Low-level `verify` is an evaluation primitive and is
  intentionally not an authentication boundary.

### Multi-frame experiment

Models and enrolment splits are identical between the single- and three-frame
arms. Each pooled trial contains three probes from one source identity; leftovers
are dropped and impostor groups never mix identities. The score is the median of
the three LBPH distances.

| Dataset | One frame | Median of 3 | TAR@FAR=1% (1 → 3) |
|---|---:|---:|---:|
| ORL, k=5 (paired) | 5.00% EER | **0.06%** (40 groups) | 87.50% → **100.00%** |
| LFW aligned, k=10 (paired) | 26.83% | **18.74%** (779 groups) | 27.47% → **38.13%** |

The ORL result is directionally positive but has only 40 pooled target groups.
The paired LFW intervals (24.23–28.42% vs 16.50–20.62%) do not overlap, which is
why three frames became the live default. The all-single-frame reference is
26.30% on 2,403 target trials; it is reported separately rather than substituted
for the paired single-frame arm.

## 10. Reproducing

```powershell
$py = "$env:LOCALAPPDATA\faceauth\venv\Scripts\python.exe"
& $py scripts\prepare_data.py --dataset all      # --refine adds aligned_refined
& $py scripts\evaluate.py --dataset orl --k 5
& $py scripts\evaluate.py --dataset lfw --variant aligned --k 10
& $py scripts\baselines.py --dataset orl --k 7 --method all
& $py scripts\evaluate_pooling.py --dataset lfw --variant aligned --k 10 --takes 1 3
& $py scripts\summarise.py
```

`summarise.py` reads the artifacts and writes `outputs/report.txt` plus the
figures, so no number in the report can drift from the score file it came from.

The live gate:

```powershell
& $py -m pip install -e .
& "$env:LOCALAPPDATA\faceauth\venv\Scripts\Activate.ps1"
faceauth                             # first-run Register, then Login
faceauth register                    # prompt for a name
faceauth login                       # choose/auto-select a registered name
faceauth cohort build                # 30 identities × 3 encrypted crops
faceauth calibrate --all             # apply FAR=0 per-user thresholds
faceauth secure status               # inspect legacy plaintext before migration
faceauth selftest                    # assets + decision path, no camera
faceauth verify <name> --json        # low-level/offline, no liveness or lockout
```
