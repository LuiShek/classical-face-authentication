"""Central paths and constants.

Large data, biometric payloads, and the virtual environment live outside the
repository and outside synced project directories. Nothing under ``LOCAL`` is
ever committed.

Enrolled face models are biometric data, so they live under ``FACES`` outside
the repository as well - a stray ``git add -A`` cannot leak them.
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]

ASSETS = PACKAGE / "assets"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
DOCS = ROOT / "docs"

LOCAL = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "faceauth"
DATA = LOCAL / "data"          # datasets and cached crops (large)
FACES = LOCAL / "faces"        # enrolled models (biometric)
COHORT = LOCAL / "cohort"      # encrypted background-impostor crops
AUTH_STATE = LOCAL / "auth-state.json"  # failures / temporary lockouts

# Detector assets. OpenCV 5.0 no longer bundles Haar cascades and ships no
# YuNet model, so scripts/fetch_assets.py downloads them once and they are
# committed (they are not biometric data).
YUNET = ASSETS / "face_detection_yunet_2023mar.onnx"
HAAR_FRONTAL = ASSETS / "haarcascade_frontalface_default.xml"
HAAR_EYE = ASSETS / "haarcascade_eye.xml"

# Preprocessing contract: every image path (enrol and verify) must use these
# exact values, otherwise distances are not comparable.
IMG_SIZE = 112                 # canonical square after alignment
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

# LBPH parameters. 8x8 grid x 256 bins = 16,384-dimensional histogram, which
# matches the verified OpenCV 5 default configuration.
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8

# Trial protocol
UTTS_ENROLL = 3                # ORL: 3 enrol / 7 test per identity
ORL_IDENTITIES = 40
ORL_IMAGES_PER_ID = 10

# Enrolment depth for the live gate. ORL EER falls 12.14% -> 6.36% -> 5.83% (raw)
# across k = 3, 5, 7, so 5 is where the gain flattens while a webcam capture is
# still quick.
ENROLL_TAKES = 5
LOGIN_TAKES = 3                # paired LFW EER 26.83% -> 18.74%
LOGIN_FRAME_INTERVAL = 0.35    # drain near-duplicate webcam frames between takes

# Distinct identities needed before cohort calibration is trusted. A tiny cohort
# produces unstable false-accept estimates, so below this the gate falls back to
# controlling the false-reject rate and reports FAR as unmeasured.
MIN_COHORT_IDENTITIES = 10

# Score scale for the live gate. Cohort z-norm wins on ORL (5.83% -> 2.99% at
# k=7) and loses on LFW (26.09% -> 27.42%), so the default follows the
# in-the-wild benchmark and z-norm stays available as an option.
SCORE_SCALE = "raw"            # "raw" | "z"
LIVE_TARGET_FAR = 0.0           # user preference: reject rather than false-accept
COHORT_IDENTITIES = 30
COHORT_IMAGES_PER_ID = 3

# User-facing Login rate limit. The low-level Verify command remains available
# for offline evaluation and automation; interactive/scripted Login enforces it.
MAX_FAILED_ATTEMPTS = 3
LOCKOUT_BASE_SECONDS = 30
LOCKOUT_MAX_SECONDS = 15 * 60

# Basic active liveness. This rejects a stationary print, but it is not a
# certified presentation-attack detector and a sufficiently adaptive replay may
# still pass.
LIVENESS_TIMEOUT_SECONDS = 12.0
LIVENESS_NEUTRAL_MAX = 0.25
LIVENESS_NEUTRAL_STABILITY = 0.03
LIVENESS_TURN_DELTA = 0.12
LIVENESS_RETURN_DELTA = 0.08
LIVENESS_CONSECUTIVE_FRAMES = 2

MIN_FACE_PX = 64               # reject crops smaller than this
BLUR_MIN_VAR = 40.0            # Laplacian variance floor for a usable frame

SEED = 0
