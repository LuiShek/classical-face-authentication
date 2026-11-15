"""Persistent lockout state and exponential backoff."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from faceauth import config, ratelimit


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTH_STATE", tmp_path / "auth-state.json")


def test_first_failures_do_not_lock():
    for i in range(config.MAX_FAILED_ATTEMPTS - 1):
        state = ratelimit.record_failure("person", now=100.0 + i)
        assert not state.locked
        assert state.failures == i + 1


def test_threshold_failure_locks_for_the_base_period():
    for i in range(config.MAX_FAILED_ATTEMPTS):
        state = ratelimit.record_failure("person", now=100.0)
    assert state.locked
    assert state.remaining == pytest.approx(config.LOCKOUT_BASE_SECONDS)


def test_lock_expires_but_failure_count_remains():
    for _ in range(config.MAX_FAILED_ATTEMPTS):
        ratelimit.record_failure("person", now=100.0)
    state = ratelimit.status("person", now=100.0 + config.LOCKOUT_BASE_SECONDS + 1)
    assert not state.locked
    assert state.failures == config.MAX_FAILED_ATTEMPTS


def test_next_failure_doubles_the_lockout():
    for _ in range(config.MAX_FAILED_ATTEMPTS):
        ratelimit.record_failure("person", now=100.0)
    state = ratelimit.record_failure("person", now=200.0)
    assert state.remaining == pytest.approx(config.LOCKOUT_BASE_SECONDS * 2)


def test_success_resets_failures_and_lock():
    for _ in range(config.MAX_FAILED_ATTEMPTS):
        ratelimit.record_failure("person", now=100.0)
    ratelimit.record_success("person")
    assert ratelimit.status("person", now=100.0).failures == 0
    assert not ratelimit.status("person", now=100.0).locked


def test_state_survives_a_fresh_read():
    ratelimit.record_failure("person", now=100.0)
    assert config.AUTH_STATE.exists()
    assert ratelimit.status("person", now=100.0).failures == 1


def test_corrupt_state_fails_closed(tmp_path):
    config.AUTH_STATE.write_text("{broken", encoding="utf-8")
    with pytest.raises(SystemExit, match="unreadable"):
        ratelimit.status("person", now=100.0)
