# Roadmap

The authentication system, benchmark suite, encrypted storage, active liveness,
background cohort, and interactive CLI are complete. The remaining items are
research extensions rather than blockers for the current release.

## Completed

- Deterministic ORL and LFW evaluation protocols
- Identity-clustered confidence intervals, DET curves, TAR@FAR, and minDCF
- LBPH authentication with raw, mean-normalized, and z-normalized score studies
- Raw-pixel, HOG, eigenfaces, and supervised HOG-pair baselines
- Three-frame median pooling experiment and live default
- Interactive registration and named-profile Login
- Active randomized head-turn liveness
- Conservative zero-observed-FAR cohort calibration
- DPAPI-encrypted models, enrolment crops, and background cohort
- Persistent per-profile exponential lockout
- Reversible biometric migration and rollback commands

## Research extensions

### Stronger presentation-attack detection

Evaluate print, display-replay, and mask attacks using a dedicated
presentation-attack dataset. Replace the current landmark challenge only if a
new method improves attack detection without an unacceptable genuine-user
failure rate.

### Disjoint-identity metric learning

Train the HOG pair classifier on background identities and evaluate it on
strictly disjoint identities. The current supervised baseline uses enrolment
pairs from evaluated identities and is therefore optimistic.

### Un-funnelled LFW alignment study

Repeat the aligned-versus-boxed experiment on un-funnelled LFW. The current
deep-funnelled copy is already centered and cannot isolate the value of
landmark alignment.

### LBPH parameter study

Evaluate a pre-declared grid over radius, neighbors, and spatial grid size using
the existing deterministic splits. Report all runs, not only the best setting.

### Detector comparison

Compare YuNet with the committed Haar frontal-face and eye cascades under the
same detection-rate, alignment, and latency protocol.

### Cross-platform protected storage

DPAPI intentionally makes the current secure store Windows-specific. A future
cross-platform backend should use the operating system keychain without adding
application-managed encryption keys to the repository.
