"""CLI wiring: defaults, exit codes, and the confirmation gate on deletion."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from faceauth import capture, cli, cohort, config, gate, ratelimit, store


def crops(seed: int = 0, n: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (n, config.IMG_SIZE, config.IMG_SIZE),
                        dtype=np.uint8)


@pytest.fixture(autouse=True)
def isolated_faces(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOCAL", tmp_path)
    monkeypatch.setattr(config, "FACES", tmp_path / "faces")
    monkeypatch.setattr(config, "COHORT", tmp_path / "cohort")
    monkeypatch.setattr(config, "AUTH_STATE", tmp_path / "auth-state.json")
    return tmp_path / "faces"


def enrol(name: str, seed: int = 0) -> np.ndarray:
    c = crops(seed)
    cal = gate.calibrate(c, crops(seed + 100))
    store.save(name, gate.Template(c).model, c, cal.as_meta())
    return c


class _Source:
    description = "stub"
    live = False

    def close(self) -> None:
        pass


def stub_capture(monkeypatch, crop: np.ndarray, *, live: bool = False) -> None:
    """Replace the camera + detector with a fixed crop, keeping the rest real."""
    source = _Source()
    source.live = live
    monkeypatch.setattr(cli, "capture_source", lambda args: source)
    monkeypatch.setattr(cli.capture, "collect",
                        lambda *a, **k: capture.CaptureResult(crops=[crop]))
    monkeypatch.setattr(cli, "_detector", lambda: object())
    if live:
        monkeypatch.setattr(
            cli.liveness, "challenge",
            lambda *a, **k: cli.liveness.Result(
                True, "challenge completed", "left", 0.0, 0.2, 8))


def test_enroll_defaults_to_the_measured_enrolment_depth():
    args = cli.build_parser().parse_args(["enroll", "someone"])
    assert args.takes == config.ENROLL_TAKES == 5
    assert args.scale == config.SCORE_SCALE == "raw"
    assert args.clahe is False


def test_verify_defaults_to_a_single_frame():
    args = cli.build_parser().parse_args(["verify", "someone"])
    assert args.takes == 1


def test_register_and_login_are_friendly_aliases():
    register = cli.build_parser().parse_args(["register"])
    login = cli.build_parser().parse_args(["login"])
    assert register.name is None
    assert register.takes == config.ENROLL_TAKES
    assert login.name is None
    assert login.takes == config.LOGIN_TAKES == 3


def test_double_dash_preserves_a_name_that_looks_like_an_option():
    register = cli.build_parser().parse_args(["register", "--", "--help"])
    login = cli.build_parser().parse_args(["login", "--", "--help"])
    assert register.name == "--help"
    assert login.name == "--help"


def test_list_on_an_empty_store_explains_itself(capsys):
    assert cli.main(["list"]) == cli.OK
    assert "no enrolled identities" in capsys.readouterr().out


def test_list_shows_a_stored_identity(capsys):
    enrol("alice", seed=1)
    assert cli.main(["list"]) == cli.OK
    out = capsys.readouterr().out
    assert "alice" in out
    assert "loo+far0" in out


def test_verify_accepts_the_enrolled_crop(monkeypatch, capsys):
    own = enrol("me", seed=2)
    stub_capture(monkeypatch, own[0])
    assert cli.main(["verify", "me", "--json"]) == cli.OK
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["accepted"] is True
    assert payload["score"] <= payload["threshold"]


def test_verify_rejects_an_impostor_crop(monkeypatch, capsys):
    enrol("me", seed=3)
    impostor = crops(3 + 100)[0]          # by construction, not me
    stub_capture(monkeypatch, impostor)
    assert cli.main(["verify", "me", "--json"]) == cli.REJECT
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["accepted"] is False
    assert payload["score"] > payload["threshold"]


def test_verify_pools_multiple_frames_by_median(monkeypatch, capsys):
    own = enrol("pooled", seed=4)
    monkeypatch.setattr(cli, "capture_source", lambda args: _Source())
    monkeypatch.setattr(cli.capture, "collect",
                        lambda *a, **k: capture.CaptureResult(crops=list(own[:3])))
    monkeypatch.setattr(cli, "_detector", lambda: object())
    assert cli.main(["verify", "pooled", "--takes", "3", "--json"]) == cli.OK
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["takes"] == 3
    # The pooled score must be the median of the individual distances.
    assert payload["score"] == pytest.approx(
        float(np.median(payload["distances"])), abs=1e-3)


def test_enroll_refuses_to_overwrite_without_force(monkeypatch, capsys):
    enrol("existing", seed=5)
    assert cli.main(["enroll", "existing", "--from", "ignored"]) == cli.ERROR
    assert "--force" in capsys.readouterr().err


def test_delete_requires_explicit_confirmation(capsys):
    c = enrol("keep-me", seed=6)
    assert c is not None
    assert cli.main(["delete", "keep-me"]) == cli.ERROR
    assert "--yes" in capsys.readouterr().err
    assert store.exists("keep-me"), "a refused delete must not remove anything"


def test_delete_with_confirmation_removes_the_identity():
    enrol("remove-me", seed=7)
    assert cli.main(["delete", "remove-me", "--yes"]) == cli.OK
    assert not store.exists("remove-me")


def test_secure_migration_preview_keeps_plaintext(monkeypatch, capsys):
    enrol("legacy", seed=70)
    d = store.model_dir("legacy")
    from faceauth import secure
    (d / "model.yml").write_bytes(secure.protected_read(d / store.MODEL_ENC))
    (d / "crops.npz").write_bytes(secure.protected_read(d / store.CROPS_ENC))
    (d / store.MODEL_ENC).unlink()
    (d / store.CROPS_ENC).unlink()
    assert cli.main(["secure", "migrate", "legacy"]) == cli.OK
    assert (d / "model.yml").exists() and (d / "crops.npz").exists()
    assert "preview only" in capsys.readouterr().out


def test_secure_migration_yes_removes_only_verified_plaintext(capsys):
    enrol("legacy", seed=71)
    d = store.model_dir("legacy")
    from faceauth import secure
    (d / "model.yml").write_bytes(secure.protected_read(d / store.MODEL_ENC))
    (d / "crops.npz").write_bytes(secure.protected_read(d / store.CROPS_ENC))
    (d / store.MODEL_ENC).unlink()
    (d / store.CROPS_ENC).unlink()
    assert cli.main(["secure", "migrate", "legacy", "--yes"]) == cli.OK
    assert not (d / "model.yml").exists()
    assert not (d / "crops.npz").exists()
    assert store.load("legacy").crops.shape[0] == 5
    assert "migrated" in capsys.readouterr().out


def test_secure_restore_requires_confirmation_and_retains_encrypted_copy(capsys):
    enrol("rollback", seed=72)
    assert cli.main(["secure", "restore", "rollback"]) == cli.ERROR
    assert "--yes" in capsys.readouterr().err
    assert cli.main(["secure", "restore", "rollback", "--yes"]) == cli.OK
    d = store.model_dir("rollback")
    assert (d / "model.yml").exists() and (d / "crops.npz").exists()
    assert (d / store.MODEL_ENC).exists() and (d / store.CROPS_ENC).exists()
    assert "restored" in capsys.readouterr().out


def test_verify_without_an_enrolment_names_the_next_step(capsys):
    assert cli.main(["verify", "ghost"]) == cli.ERROR
    assert "enroll" in capsys.readouterr().err


def test_scriptable_operational_error_exits_two(tmp_path):
    env = os.environ.copy()
    env["LOCALAPPDATA"] = str(tmp_path)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-m", "faceauth.cli", "verify", "ghost"],
        text=True, capture_output=True, env=env, check=False)
    assert result.returncode == cli.ERROR
    assert "run 'enroll' first" in result.stderr


def test_calibrate_all_rewrites_every_threshold(capsys):
    for i, name in enumerate(["a", "b", "c"]):
        enrol(name, seed=20 + i)
    before = {p["name"]: p["threshold"] for p in store.profiles()}
    assert cli.main(["calibrate", "--all"]) == cli.OK
    out = capsys.readouterr().out
    for name in before:
        assert name in out
    after = {p["name"]: p["threshold"] for p in store.profiles()}
    assert set(after) == set(before)


def test_calibrate_all_with_no_identities_explains_itself(capsys):
    assert cli.main(["calibrate", "--all"]) == cli.OK
    assert "no enrolled identities" in capsys.readouterr().out


def test_background_cohort_counts_distinct_identities(monkeypatch):
    enrol("target", seed=25)
    enrol("other", seed=26)
    bg_crops = crops(90, n=20)
    bg_labels = np.repeat(np.arange(10), 2).astype(np.int32)
    monkeypatch.setattr(
        cli.cohort, "load",
        lambda: cohort.Background(bg_crops, bg_labels, {"storage": "dpapi-v1"}))
    material, identities, background = cli._calibration_material("target")
    assert identities == 11             # ten background + one registered user
    assert background == 10
    assert len(material) == 25           # 20 background + five registered crops


def test_register_prompts_for_a_name(monkeypatch):
    args = cli.build_parser().parse_args(["register"])
    monkeypatch.setattr("builtins.input", lambda prompt: "Alice")
    seen = {}

    def fake_enroll(enroll_args):
        seen["name"] = enroll_args.name
        return cli.OK

    monkeypatch.setattr(cli, "cmd_enroll", fake_enroll)
    assert cli.cmd_register(args) == cli.OK
    assert seen["name"] == "Alice"


def test_login_auto_selects_the_only_registered_name(monkeypatch):
    enrol("only-user", seed=30)
    args = cli.build_parser().parse_args(["login"])
    seen = {}

    def fake_verify(verify_args):
        seen["name"] = verify_args.name
        return cli.OK

    monkeypatch.setattr(cli, "cmd_verify", fake_verify)
    assert cli.cmd_login(args) == cli.OK
    assert seen["name"] == "only-user"


def test_login_json_emits_only_the_json_result(monkeypatch, capsys):
    own = enrol("json-user", seed=33)
    stub_capture(monkeypatch, own[0])
    stub_capture(monkeypatch, own[0], live=True)
    assert cli.main(["login", "json-user", "--json"]) == cli.OK
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["accepted"] is True


def test_login_refuses_a_non_live_source(monkeypatch, capsys):
    own = enrol("live-required", seed=34)
    stub_capture(monkeypatch, own[0])
    assert cli.main(["login", "live-required", "--json"]) == cli.ERROR
    assert "requires a live camera" in capsys.readouterr().err


def test_login_parser_does_not_offer_image_or_directory_bypass():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["login", "person", "--image", "photo.jpg"])
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["login", "person", "--from", "photos"])


def test_failed_logins_trigger_persistent_lockout(monkeypatch, capsys):
    enrol("locked-user", seed=35)
    impostor = crops(135)[0]
    stub_capture(monkeypatch, impostor, live=True)
    for _ in range(config.MAX_FAILED_ATTEMPTS):
        assert cli.main(["login", "locked-user"]) == cli.REJECT
    state = ratelimit.status("locked-user")
    assert state.locked
    capsys.readouterr()
    assert cli.main(["login", "locked-user"]) == cli.REJECT
    assert "too many failed logins" in capsys.readouterr().out


def test_successful_login_resets_previous_failures(monkeypatch):
    own = enrol("reset-user", seed=36)
    ratelimit.record_failure("reset-user", now=1.0)
    ratelimit.record_failure("reset-user", now=2.0)
    stub_capture(monkeypatch, own[0], live=True)
    assert cli.main(["login", "reset-user"]) == cli.OK
    assert ratelimit.status("reset-user").failures == 0


def test_login_asks_which_name_when_multiple_are_registered(monkeypatch):
    enrol("first", seed=31)
    enrol("second", seed=32)
    args = cli.build_parser().parse_args(["login"])
    monkeypatch.setattr("builtins.input", lambda prompt: "#2")
    seen = {}

    def fake_verify(verify_args):
        seen["name"] = verify_args.name
        return cli.OK

    monkeypatch.setattr(cli, "cmd_verify", fake_verify)
    assert cli.cmd_login(args) == cli.OK
    assert seen["name"] == "second"


def test_first_run_interactive_registers_then_logs_in(monkeypatch):
    answers = iter(["Alice", ""])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    calls = []
    monkeypatch.setattr(cli, "_valid_profiles", lambda: [])
    monkeypatch.setattr(
        cli, "_invoke_interactive",
        lambda argv: calls.append(argv) or cli.OK)

    assert cli.interactive() == cli.OK
    assert calls == [["register", "--", "Alice"],
                     ["login", "--", "Alice"]]


def test_existing_user_menu_propagates_login_result(monkeypatch):
    answers = iter(["1"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    monkeypatch.setattr(cli, "_valid_profiles", lambda: [{"name": "Alice"}])
    calls = []
    monkeypatch.setattr(
        cli, "_invoke_interactive",
        lambda argv: calls.append(argv) or cli.REJECT)

    assert cli.interactive() == cli.REJECT
    assert calls == [["login"]]


def test_prompt_name_retries_blank_and_validates_before_capture(monkeypatch, capsys):
    answers = iter(["", "a/b", "Valid Name"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert cli._prompt_name() == "Valid Name"
    captured = capsys.readouterr()
    assert "cannot be empty" in captured.out
    assert "path separator" in captured.err


def test_first_run_name_that_looks_like_an_option_is_not_reparsed(monkeypatch):
    answers = iter(["--help", ""])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    monkeypatch.setattr(cli, "_valid_profiles", lambda: [])
    calls = []
    monkeypatch.setattr(
        cli, "_invoke_interactive",
        lambda argv: calls.append(argv) or cli.OK)
    assert cli.interactive() == cli.OK
    assert calls == [["register", "--", "--help"],
                     ["login", "--", "--help"]]


def test_numeric_name_does_not_collide_with_number_selection(monkeypatch):
    monkeypatch.setattr(
        cli, "_valid_profiles",
        lambda: [{"name": "first", "slug": "first"},
                 {"name": "2", "slug": "2"}])
    monkeypatch.setattr("builtins.input", lambda prompt: "2")
    assert cli._choose_login_name() == "2"


def test_interrupt_is_not_reported_as_success(monkeypatch):
    monkeypatch.setattr(cli, "interactive", lambda: (_ for _ in ()).throw(
        KeyboardInterrupt()))
    assert cli.main([]) == cli.ERROR


def test_no_argument_main_routes_to_interactive(monkeypatch):
    monkeypatch.setattr(cli, "interactive", lambda: 17)
    assert cli.main([]) == 17


def test_selftest_passes_on_this_installation(capsys):
    """Runs the real detector asset and the real decision path."""
    assert cli.main(["selftest"]) == cli.OK
    out = capsys.readouterr().out
    assert "FAIL" not in out
    assert "OK:" in out
