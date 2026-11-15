"""Command-line gate: interactive register/login plus scriptable subcommands.

Exit codes are part of the interface, because this is meant to sit behind
something else: 0 accept / success, 1 reject / selftest failure, 2 error.

Usage
-----
  faceauth                         # interactive Register / Login menu
  faceauth register               # asks for a name, then captures a face
  faceauth login                  # asks/selects a name, then authenticates
  py -m faceauth.cli enroll alice
  py -m faceauth.cli verify alice
  py -m faceauth.cli verify alice --image probe.jpg --json
  py -m faceauth.cli list
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import (camera, capture, cohort, config, detect, gate, liveness,
               ratelimit, secure, store)

OK, REJECT, ERROR = 0, 1, 2


def _detector() -> detect.Detector:
    if not config.YUNET.exists():
        raise SystemExit(f"error: missing {config.YUNET}; "
                         f"run scripts/fetch_assets.py first")
    return detect.Detector()


def _calibration_line(meta: dict) -> str:
    c = meta.get("calibration", {})
    bits = [f"threshold {meta['threshold']:.2f} ({meta.get('scale', 'raw')})",
            f"method {c.get('method', '?')}"]
    if c.get("eer") is not None:
        bits.append(f"EER {c['eer'] * 100:.1f}%")
    if c.get("far") is not None:
        bits.append(f"FAR {c['far'] * 100:.1f}%")
    if c.get("frr") is not None:
        bits.append(f"FRR {c['frr'] * 100:.1f}%")
    return ", ".join(bits)


def _calibration_material(name: str) -> tuple[np.ndarray | None, int, int]:
    """All known impostor crops, count of identities, background count."""
    others = store.others(name)
    chunks = [o.crops for o in others]
    identity_count = len(others)
    background_count = 0
    background = cohort.load()
    if background is not None:
        chunks.append(background.crops)
        background_count = background.identities
        identity_count += background_count
    material = np.concatenate(chunks) if chunks else None
    return material, identity_count, background_count


# ---------------------------------------------------------------- commands

def cmd_enroll(args) -> int:
    if store.exists(args.name) and not args.force:
        raise SystemExit(f"error: {args.name!r} is already enrolled; "
                         f"re-run with --force to replace it")
    det = _detector()
    source = capture_source(args)
    t0 = time.time()

    def show(attempt: int, assessment) -> None:
        if attempt % 5 == 1:
            print(f"  ... {attempt} frames, rejected: "
                  f"{'; '.join(assessment.reasons)}", flush=True)

    def accepted(n: int, assessment) -> None:
        hints = ("look straight", "turn slightly left", "turn slightly right",
                 "raise your chin slightly", "look straight again")
        next_hint = hints[n] if n < len(hints) else "change expression slightly"
        print(f"  captured {n}/{args.takes}"
              + (f" - next: {next_hint}"
                 if n < args.takes and getattr(source, "live", False) else ""),
              flush=True)

    # Capture always returns raw crops. The model wrapper applies CLAHE exactly
    # once when that profile was enrolled with it; passing it here as well would
    # equalise every enrolment and probe twice.
    try:
        res = capture.collect(source, det, args.takes, clahe=False,
                              on_reject=show,
                              on_warn=lambda w: print(f"  ! {w}", file=sys.stderr),
                              on_accept=accepted, live_interval=0.7)
    finally:
        source.close()
    if len(res.crops) < args.takes:
        detail = res.problems[0] if res.problems else "source ran out of frames"
        raise SystemExit(f"error: only {len(res.crops)}/{args.takes} usable "
                         f"frames ({detail})")
    crops = res.crops

    other_crops, cohort_identities, background_identities = \
        _calibration_material(args.name)
    cal = gate.calibrate(crops, other_crops, clahe=args.clahe,
                         scale=args.scale, target_frr=args.target_frr,
                         target_far=args.target_far,
                         cohort_identities=(cohort_identities
                                            if cohort_identities else None))
    template = gate.Template(crops, clahe=args.clahe)
    meta = {**cal.as_meta(), "clahe": bool(args.clahe),
            "source": source.description}
    d = store.save(args.name, template.model, np.stack(crops), meta)

    print(f"enrolled : {args.name}  ({len(crops)} frames, {time.time() - t0:.1f}s)")
    print(f"stored   : {d}")
    print(f"calib    : {_calibration_line(meta)}")
    _cohort_note(cohort_identities, cal, background_identities)
    print(f"next     : login with 'faceauth login {args.name}'", flush=True)
    return OK


def _valid_profiles() -> list[dict]:
    return [p for p in store.profiles() if not p.get("broken")]


def _prompt_name(prompt: str = "Name: ") -> str:
    while True:
        name = input(prompt).strip()
        if not name:
            print("Name cannot be empty. Please try again.")
            continue
        try:
            store.slugify(name)          # validate before the camera opens
        except SystemExit as exc:
            print(exc, file=sys.stderr)
            continue
        return name


def cmd_register(args) -> int:
    """Friendly name for enrolment; prompt when the name was omitted."""
    if not args.name:
        args.name = _prompt_name("Your name: ")
    print("\nRegistration")
    print("Keep your face in view. I will capture several spaced frames.")
    return cmd_enroll(args)


def _choose_login_name(given: str | None = None) -> str:
    if given:
        return given
    rows = _valid_profiles()
    if not rows:
        raise SystemExit("error: no registered faces; run 'faceauth register' first")
    if len(rows) == 1:
        return str(rows[0].get("name", rows[0]["slug"]))
    print("Registered users:")
    for i, row in enumerate(rows, 1):
        print(f"  {i}. {row.get('name', row['slug'])}")
    choice = input("Choose name or #number: ").strip()
    # Exact names win, including a valid numeric name such as "2". A list index
    # must be explicit (#2), so name syntax and menu syntax cannot collide.
    for row in rows:
        name = str(row.get("name", row["slug"]))
        if choice.casefold() == name.casefold():
            return name
    if choice.startswith("#") and choice[1:].isdigit() \
            and 1 <= int(choice[1:]) <= len(rows):
        row = rows[int(choice[1:]) - 1]
        return str(row.get("name", row["slug"]))
    if choice:
        return choice
    raise SystemExit("error: no login name selected")


def cmd_login(args) -> int:
    """Interactive authentication: choose a registered name, then verify it."""
    args.name = _choose_login_name(args.name)
    key = store.slugify(args.name)
    limit = ratelimit.status(key)
    if limit.locked:
        seconds = int(np.ceil(limit.remaining))
        if args.json:
            print(json.dumps({"identity": args.name, "accepted": False,
                              "locked": True, "retry_after": seconds}))
        else:
            print(f"REJECT: too many failed logins; retry in {seconds}s")
        return REJECT
    if not args.json:
        print(f"\nLogin as {args.name}")
        print("Look at the camera...")
    args.require_liveness = True
    result = cmd_verify(args)
    if result == OK:
        ratelimit.record_success(key)
    elif result == REJECT:
        state = ratelimit.record_failure(key)
        if state.locked and not args.json:
            print(f"locked   : retry in {int(np.ceil(state.remaining))}s")
    return result


def _cohort_note(n_others: int, cal: gate.Calibration,
                 background: int = 0) -> None:
    """Say plainly whether the false-accept rate is estimated or unknown."""
    if n_others == 0:
        print("warning  : no other identity is enrolled, so the false-accept "
              "rate is unmeasured - this threshold controls false rejects only")
    elif n_others < config.MIN_COHORT_IDENTITIES:
        print(f"warning  : cohort has {n_others} identit"
              f"{'y' if n_others == 1 else 'ies'} but "
              f"{config.MIN_COHORT_IDENTITIES} are needed to estimate a false-"
              f"accept rate; threshold controls false rejects only")
    else:
        print(f"cohort   : {n_others} identities "
              f"({cal.n_impostor} impostor scores"
              f"{f', {background} background identities' if background else ''})")


def cmd_verify(args) -> int:
    stored = store.load(args.name)
    clahe = bool(stored.meta.get("clahe", False))
    det = _detector()
    source = capture_source(args)
    try:
        if getattr(args, "require_liveness", False):
            if not getattr(source, "live", False):
                print("error: Login requires a live camera; use the low-level "
                      "Verify command for offline images", file=sys.stderr)
                return ERROR
            live = liveness.challenge(
                source, det,
                prompt=lambda text: print(text, file=sys.stderr
                                          if args.json else sys.stdout))
            if not live.passed:
                if args.json:
                    print(json.dumps({"identity": stored.name,
                                      "accepted": False, "liveness": False,
                                      "reason": live.reason}))
                else:
                    print(f"liveness: FAIL ({live.reason})")
                    print("decision : REJECT")
                return REJECT
            if not args.json:
                print(f"liveness: PASS ({live.direction} challenge, "
                      f"{live.frames} frames)")
        on_accept = None
        live_interval = 0.0
        if getattr(source, "live", False) and args.takes > 1:
            stream = sys.stderr if args.json else sys.stdout
            on_accept = lambda n, assessment: print(
                f"  login frame {n}/{args.takes}", file=stream, flush=True)
            live_interval = config.LOGIN_FRAME_INTERVAL
        res = capture.collect(
            source, det, args.takes, clahe=False,
            on_warn=lambda w: print(f"  ! {w}", file=sys.stderr),
            on_accept=on_accept, live_interval=live_interval)
    finally:
        source.close()
    if not res.crops:
        detail = res.problems[0] if res.problems else "source ran out of frames"
        raise SystemExit(f"error: no usable frame ({detail})")
    crops = res.crops

    raw = np.array([gate.distance_to(stored.model, c, clahe) for c in crops])
    pooled = float(np.median(raw))
    score = gate.apply_scale(pooled, stored.scale, stored.meta.get("cohort"))
    accepted = gate.decide(score, stored.threshold)
    margin = stored.threshold - score

    if args.json:
        print(json.dumps({"identity": stored.name, "accepted": accepted,
                          "score": round(score, 4), "threshold": stored.threshold,
                          "scale": stored.scale, "margin": round(margin, 4),
                          "distances": [round(float(v), 4) for v in raw],
                          "takes": len(crops)}))
    else:
        print(f"identity : {stored.name}")
        print(f"frames   : {len(crops)}  distances "
              f"{', '.join(f'{v:.1f}' for v in raw)}")
        print(f"score    : {score:.2f} {stored.scale}  (median of "
              f"{len(crops)})")
        print(f"threshold: {stored.threshold:.2f}  "
              f"({_calibration_line(stored.meta)})")
        print(f"margin   : {'+' if margin >= 0 else ''}{margin:.2f}")
        print(f"decision : {'ACCEPT' if accepted else 'REJECT'}")
    return OK if accepted else REJECT


def cmd_calibrate(args) -> int:
    if args.all_identities or args.name is None:
        return _calibrate_all(args)
    stored = store.load(args.name)
    clahe = bool(stored.meta.get("clahe", False))
    other_crops, cohort_identities, background_identities = \
        _calibration_material(stored.name)
    before = stored.threshold
    cal = gate.calibrate(stored.crops, other_crops, clahe=clahe,
                         scale=args.scale or stored.scale,
                         target_frr=args.target_frr,
                         target_far=args.target_far,
                         cohort_identities=(cohort_identities
                                            if cohort_identities else None))
    meta = {**stored.meta, **cal.as_meta()}
    store.save(stored.name, stored.model, stored.crops, meta)
    print(f"identity : {stored.name}")
    print(f"before   : {before:.2f}")
    print(f"after    : {cal.threshold:.2f}")
    print(f"calib    : {_calibration_line(meta)}")
    _cohort_note(cohort_identities, cal, background_identities)
    return OK


def _calibrate_all(args) -> int:
    """Recalibrate every identity against the cohort as it stands now.

    Necessary because a threshold is only as good as the cohort that existed
    when it was set: without this, whoever was enrolled first stays calibrated
    against a cohort of one forever, which is exactly the trap the minimum
    cohort size exists to prevent.
    """
    profiles = [p for p in store.profiles() if not p.get("broken")]
    if not profiles:
        print(f"no enrolled identities in {config.FACES}")
        return OK
    loaded: dict[str, store.Stored] = {}
    for p in profiles:
        try:
            loaded[p["slug"]] = store.load(p.get("name", p["slug"]))
        except SystemExit as exc:
            print(f"skipping {p.get('name')}: {exc}", file=sys.stderr)

    background = cohort.load()
    print(f"{'identity':22} {'before':>7} {'after':>7}  {'method':22} {'EER':>6}")
    for slug, s in loaded.items():
        others = [o.crops for k, o in loaded.items() if k != slug]
        if background is not None:
            others.append(background.crops)
        other_crops = np.concatenate(others) if others else None
        n_others = len(loaded) - 1 + (background.identities
                                      if background is not None else 0)
        clahe = bool(s.meta.get("clahe", False))
        cal = gate.calibrate(s.crops, other_crops, clahe=clahe,
                             scale=args.scale or s.scale,
                             target_frr=args.target_frr,
                             target_far=args.target_far,
                             cohort_identities=n_others if n_others else None)
        meta = {**s.meta, **cal.as_meta()}
        store.save(s.name, s.model, s.crops, meta)
        eer = "-" if cal.eer is None else f"{cal.eer * 100:5.1f}%"
        print(f"{s.name:22} {s.threshold:7.2f} {cal.threshold:7.2f}  "
              f"{cal.method:22} {eer:>6}")
    available_impostor_ids = max(0, len(loaded) - 1) + (
        background.identities if background is not None else 0)
    if available_impostor_ids < config.MIN_COHORT_IDENTITIES:
        print(f"\nnote: only {available_impostor_ids} distinct impostor "
              f"identities are available; {config.MIN_COHORT_IDENTITIES} are "
              f"needed before FAR calibration")
    return OK


def cmd_cohort_build(args) -> int:
    bg = cohort.build(identities=args.identities,
                      per_identity=args.per_identity,
                      seed=args.seed, force=args.force)
    print(f"cohort   : {bg.identities} identities / {len(bg.crops)} encrypted crops")
    print(f"stored   : {config.COHORT / cohort.DATA_FILE}")
    print("next     : run 'faceauth calibrate --all'")
    return OK


def cmd_cohort_status(args) -> int:
    bg = cohort.load()
    if bg is None:
        print("no background cohort; run 'faceauth cohort build'")
        return OK
    print(f"identities : {bg.identities}")
    print(f"crops      : {len(bg.crops)}")
    print(f"source     : {bg.meta.get('source', '?')}")
    print(f"storage    : {bg.meta.get('storage', '?')}")
    print(f"created    : {bg.meta.get('created', '?')}")
    return OK


def cmd_secure_status(args) -> int:
    rows = _valid_profiles()
    if not rows:
        print(f"no enrolled identities in {config.FACES}")
        return OK
    print(f"{'identity':22} {'storage':12} plaintext")
    for row in rows:
        plain = store.plaintext_files(row.get("name", row["slug"]))
        print(f"{row.get('name', row['slug']):22} "
              f"{row.get('storage', 'plaintext'):12} "
              f"{', '.join(p.name for p in plain) if plain else '-'}")
    return OK


def cmd_secure_migrate(args) -> int:
    if args.all_identities:
        names = [p.get("name", p["slug"]) for p in _valid_profiles()]
    elif args.name:
        names = [args.name]
    else:
        raise SystemExit("error: name an identity or pass --all")
    plans = [(name, store.plaintext_files(name)) for name in names]
    plans = [(name, files) for name, files in plans if files]
    if not plans:
        print("no plaintext biometric files need migration")
        return OK
    print("DPAPI migration plan:")
    for name, files in plans:
        print(f"  {name}: {', '.join(str(p) for p in files)}")
    if not args.yes:
        print("preview only: verified encrypted copies will be created first; "
              "rerun with --yes to remove the plaintext files")
        return OK
    for name, _ in plans:
        removed = store.migrate(name, remove_plaintext=True)
        print(f"migrated : {name} ({len(removed)} plaintext files removed after "
              f"decrypt/load verification)")
    return OK


def cmd_secure_restore(args) -> int:
    if not args.yes:
        raise SystemExit(
            "error: restoring creates plaintext biometric files; rerun with --yes")
    restored = store.restore_plaintext(args.name)
    print(f"restored : {args.name} ({', '.join(p.name for p in restored)})")
    print("warning  : encrypted copies were retained; run secure migrate --yes "
          "to remove plaintext again")
    return OK


def cmd_list(args) -> int:
    rows = store.profiles()
    if not rows:
        print(f"no enrolled identities in {config.FACES}")
        return OK
    print(f"{'name':22} {'takes':>5} {'threshold':>9}  {'method':10} {'EER':>6}  created")
    for m in rows:
        if m.get("broken"):
            print(f"{m['name']:22} {'':>5} {'':>9}  {'BROKEN':10} {'':>6}  "
                  f"(unreadable meta.json)")
            continue
        c = m.get("calibration", {})
        eer = c.get("eer")
        eer_txt = "    -" if eer is None else f"{eer * 100:5.1f}%"
        print(f"{m.get('name', '?'):22} {m.get('takes', 0):5d} "
              f"{m.get('threshold', float('nan')):9.2f}  "
              f"{c.get('method', '?'):10} {eer_txt:>6}  "
              f"{m.get('created', '?')}")
    return OK


def cmd_delete(args) -> int:
    if not args.yes:
        raise SystemExit(f"error: deleting {args.name!r} removes its biometric "
                         f"model permanently; re-run with --yes if that is intended")
    d = store.delete(args.name)
    print(f"deleted  : {d}")
    return OK


def _synthetic_crops(seed: int, n: int) -> np.ndarray:
    """Distinct structured textures, one ring position per identity.

    Pure noise is useless here: every noise image has nearly the same LBP
    histogram, so genuine and impostor scores overlap and any "reject the
    impostor" check becomes a coin flip. The ring position gives each synthetic
    identity real texture structure while staying deterministic.
    """
    s = config.IMG_SIZE
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:s, 0:s]
    cy = s / 2 + 14 * np.sin(seed * 1.7)
    cx = s / 2 + 14 * np.cos(seed * 1.7)
    out = np.empty((n, s, s), dtype=np.uint8)
    for i in range(n):
        img = rng.integers(60, 110, (s, s)).astype(np.uint8)
        r = 20 + i % 3
        img[((xx - cx) ** 2 + (yy - cy) ** 2) < r * r] = 235
        out[i] = img
    return out


def cmd_selftest(args) -> int:
    """Exercise the store and decision path with synthetic crops.

    This is a logic check, not a face check: it proves persistence, calibration
    and the decision rule agree with each other on data where the answer is
    known. Camera and detector health is what scripts/smoke_camera.py is for.
    """
    checks: list[tuple[str, bool, str]] = []

    assets = config.YUNET.exists()
    checks.append(("detector asset present", assets, str(config.YUNET.name)))

    try:
        det = _detector()
        checks.append(("detector constructs", det is not None, ""))
    except SystemExit as exc:
        checks.append(("detector constructs", False, str(exc)))

    try:
        sample = b"faceauth selftest biometric bytes"
        checks.append(("DPAPI round-trip",
                       secure.unprotect(secure.protect(sample)) == sample,
                       "current Windows user"))
    except Exception as exc:                       # noqa: BLE001
        checks.append(("DPAPI round-trip", False,
                       f"{type(exc).__name__}: {exc}"))

    try:
        pts = np.array([[38.0, 52.0], [74.0, 52.0], [56.0, 72.0],
                        [42.0, 92.0], [71.0, 92.0]], dtype=np.float32)
        synthetic_face = detect.Face((0, 0, 100, 100), pts, 1.0)
        checks.append(("liveness geometry",
                       abs(liveness.yaw_proxy(synthetic_face)) < 0.02,
                       "neutral yaw proxy"))
    except Exception as exc:                       # noqa: BLE001
        checks.append(("liveness geometry", False,
                       f"{type(exc).__name__}: {exc}"))

    a = _synthetic_crops(1, 6)
    b = _synthetic_crops(9, 6)

    try:
        loo = gate.loo_scores(a)
        checks.append(("leave-one-out runs", len(loo) == len(a),
                       f"{len(loo)} scores"))
        cal = gate.calibrate(a, b)
        checks.append(("calibration produces a threshold",
                       np.isfinite(cal.threshold), f"{cal.threshold:.2f} "
                       f"({cal.method}, EER {cal.eer * 100:.1f}%)"
                       if cal.eer is not None else f"{cal.threshold:.2f}"))
        own = gate.Template(a).distance(a[0])
        checks.append(("own crop accepted", gate.decide(own, cal.threshold),
                       f"{own:.2f} <= {cal.threshold:.2f}"))
        alien = gate.Template(a).distance(b[0])
        checks.append(("other crop rejected", not gate.decide(alien, cal.threshold),
                       f"{alien:.2f} > {cal.threshold:.2f}"))
        checks.append(("genuine is closer than impostor", own < alien,
                       f"{own:.2f} < {alien:.2f}"))
    except Exception as exc:                       # noqa: BLE001 - report, don't crash
        checks.append(("gate logic", False, f"{type(exc).__name__}: {exc}"))

    failed = [name for name, ok, _ in checks if not ok]
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:32} {detail}")
    print(f"\n{'FAILED' if failed else 'OK'}: {len(checks) - len(failed)}/"
          f"{len(checks)} checks")
    return ERROR if failed else OK


# ---------------------------------------------------------------- wiring

def capture_source(args):
    """Single probe image wins over a directory; both bypass the camera."""
    if getattr(args, "image", None) is not None:
        return camera.ImageDir(args.image)
    return camera.open_source(getattr(args, "index", None),
                              getattr(args, "from_path", None))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="faceauth",
        description="Classical face authentication (LBPH) - enrol, verify, calibrate.")
    sub = ap.add_subparsers(dest="command", required=True)

    def add_source(p):
        p.add_argument("--index", type=int, default=None, help="camera index")
        p.add_argument("--from", dest="from_path", type=Path, default=None,
                       help="read frames from a file or directory")
        p.add_argument("--image", type=Path, default=None,
                       help="use a single image instead of the camera")

    e = sub.add_parser("enroll", help="capture frames and store an identity")
    e.add_argument("name")
    e.add_argument("--takes", type=int, default=config.ENROLL_TAKES)
    e.add_argument("--clahe", action="store_true",
                   help="apply CLAHE (off by default: it measured worse)")
    e.add_argument("--scale", choices=["raw", "z"], default=config.SCORE_SCALE)
    e.add_argument("--target-frr", type=float, default=0.05)
    e.add_argument("--target-far", type=float, default=config.LIVE_TARGET_FAR)
    e.add_argument("--force", action="store_true", help="replace an existing enrolment")
    add_source(e)
    e.set_defaults(func=cmd_enroll)

    r = sub.add_parser("register", help="ask for a name and register a face")
    r.add_argument("name", nargs="?", default=None)
    r.add_argument("--takes", type=int, default=config.ENROLL_TAKES)
    r.add_argument("--clahe", action="store_true",
                   help="apply CLAHE (off by default: it measured worse)")
    r.add_argument("--scale", choices=["raw", "z"], default=config.SCORE_SCALE)
    r.add_argument("--target-frr", type=float, default=0.05)
    r.add_argument("--target-far", type=float, default=config.LIVE_TARGET_FAR)
    r.add_argument("--force", action="store_true", help="replace an existing registration")
    add_source(r)
    r.set_defaults(func=cmd_register)

    v = sub.add_parser("verify", help="score a probe against an enrolled identity")
    v.add_argument("name")
    v.add_argument("--takes", type=int, default=1,
                   help="frames to pool (median) before deciding")
    v.add_argument("--json", action="store_true")
    add_source(v)
    v.set_defaults(func=cmd_verify)

    login = sub.add_parser("login", help="choose a registered name and authenticate")
    login.add_argument("name", nargs="?", default=None)
    login.add_argument("--takes", type=int, default=config.LOGIN_TAKES,
                       help="frames to pool (median) before deciding")
    login.add_argument("--json", action="store_true")
    login.add_argument("--index", type=int, default=None, help="camera index")
    login.set_defaults(func=cmd_login)

    c = sub.add_parser("calibrate",
                       help="recompute thresholds from stored crops")
    c.add_argument("name", nargs="?", default=None,
                   help="omit, or use --all, to recalibrate every identity")
    c.add_argument("--all", action="store_true", dest="all_identities",
                   help="recalibrate every enrolled identity against the "
                        "current cohort")
    c.add_argument("--scale", choices=["raw", "z"], default=None)
    c.add_argument("--target-frr", type=float, default=0.05,
                   help="used only when the cohort is too small for an EER")
    c.add_argument("--target-far", type=float, default=config.LIVE_TARGET_FAR,
                   help="maximum observed cohort FAR (default: 0)")
    c.set_defaults(func=cmd_calibrate)

    cohort_parser = sub.add_parser("cohort",
                                   help="build or inspect the background cohort")
    cohort_sub = cohort_parser.add_subparsers(dest="cohort_command", required=True)
    cb = cohort_sub.add_parser("build", help="sample an encrypted LFW cohort")
    cb.add_argument("--identities", type=int, default=config.COHORT_IDENTITIES)
    cb.add_argument("--per-identity", type=int,
                    default=config.COHORT_IMAGES_PER_ID)
    cb.add_argument("--seed", type=int, default=config.SEED)
    cb.add_argument("--force", action="store_true")
    cb.set_defaults(func=cmd_cohort_build)
    cohort_sub.add_parser("status", help="show cohort metadata").set_defaults(
        func=cmd_cohort_status)

    secure_parser = sub.add_parser("secure",
                                   help="inspect or migrate biometric encryption")
    secure_sub = secure_parser.add_subparsers(dest="secure_command", required=True)
    secure_sub.add_parser("status", help="show profile storage formats").set_defaults(
        func=cmd_secure_status)
    sm = secure_sub.add_parser("migrate", help="encrypt legacy plaintext profiles")
    sm.add_argument("name", nargs="?", default=None)
    sm.add_argument("--all", action="store_true", dest="all_identities")
    sm.add_argument("--yes", action="store_true",
                    help="remove plaintext only after encrypted copies verify")
    sm.set_defaults(func=cmd_secure_migrate)
    sr = secure_sub.add_parser("restore",
                               help="restore plaintext while retaining DPAPI copies")
    sr.add_argument("name")
    sr.add_argument("--yes", action="store_true")
    sr.set_defaults(func=cmd_secure_restore)

    sub.add_parser("list", help="show enrolled identities").set_defaults(
        func=cmd_list)

    d = sub.add_parser("delete", help="remove an enrolled identity")
    d.add_argument("name")
    d.add_argument("--yes", action="store_true", help="confirm the deletion")
    d.set_defaults(func=cmd_delete)

    sub.add_parser("selftest", help="check assets and the decision path").set_defaults(
        func=cmd_selftest)
    return ap


def _invoke_interactive(argv: list[str]) -> int:
    """Run one menu action without letting a user error kill the whole menu."""
    try:
        args = build_parser().parse_args(argv)
        return args.func(args)
    except SystemExit as exc:
        text = str(exc)
        if text and text not in {"0", "1", "2"}:
            print(text, file=sys.stderr)
        return ERROR


def interactive() -> int:
    """No-argument UX: first-run registration, then a persistent login menu."""
    print("FaceAuth - classical face authentication")
    print("No face images are written to the project directory.\n")

    if not _valid_profiles():
        print("No registered face found. Let's register you first.")
        name = _prompt_name("Your name: ")
        # ``--`` stops a valid name such as "--help" being reparsed as CLI
        # syntax after it came from an interactive prompt.
        code = _invoke_interactive(["register", "--", name])
        if code != OK:
            return code
        answer = input("\nRegistration complete. Login now? [Y/n]: ").strip().lower()
        if answer not in {"n", "no"}:
            return _invoke_interactive(["login", "--", name])
        return OK

    while True:
        print("\n1. Login\n2. Register another face\n3. List users\n4. Exit")
        choice = input("Choose [1-4]: ").strip().lower()
        if choice in {"1", "login", "l"}:
            # Authentication is the terminal action. Propagating this result is
            # what preserves 0=accept / 1=reject for the no-argument CLI too.
            return _invoke_interactive(["login"])
        elif choice in {"2", "register", "r"}:
            _invoke_interactive(["register"])
        elif choice in {"3", "list"}:
            _invoke_interactive(["list"])
        elif choice in {"4", "exit", "quit", "q"}:
            print("Goodbye.")
            return OK
        else:
            print("Please choose 1, 2, 3 or 4.")


def main(argv: list[str] | None = None) -> int:
    try:
        actual = sys.argv[1:] if argv is None else argv
        if not actual:
            return interactive()
        args = build_parser().parse_args(actual)
        return args.func(args)
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.", file=sys.stderr)
        return ERROR
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            return ERROR
        return int(exc.code or 0)


if __name__ == "__main__":
    sys.exit(main())
