"""Persistent per-identity failed-attempt lockouts."""
from __future__ import annotations

import json
import msvcrt
import time
from contextlib import contextmanager
from dataclasses import dataclass

from . import config


@dataclass(frozen=True)
class Status:
    failures: int
    remaining: float

    @property
    def locked(self) -> bool:
        return self.remaining > 0


def _load() -> dict:
    if not config.AUTH_STATE.exists():
        return {"version": 1, "profiles": {}}
    try:
        data = json.loads(config.AUTH_STATE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A corrupt security state must not silently unlock everyone.
        raise SystemExit(f"error: unreadable rate-limit state "
                         f"{config.AUTH_STATE}; inspect or remove it manually")
    if not isinstance(data.get("profiles"), dict):
        raise SystemExit(f"error: malformed rate-limit state {config.AUTH_STATE}")
    return data


@contextmanager
def _locked():
    """Serialize state updates across concurrent Windows processes."""
    config.AUTH_STATE.parent.mkdir(parents=True, exist_ok=True)
    lock_path = config.AUTH_STATE.with_name(config.AUTH_STATE.name + ".lock")
    with open(lock_path, "a+b") as lock:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def _save(data: dict) -> None:
    config.AUTH_STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = config.AUTH_STATE.with_name(config.AUTH_STATE.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(config.AUTH_STATE)


def status(key: str, *, now: float | None = None) -> Status:
    now = time.time() if now is None else now
    with _locked():
        entry = _load()["profiles"].get(key, {})
    remaining = max(0.0, float(entry.get("locked_until", 0.0)) - now)
    return Status(int(entry.get("failures", 0)), remaining)


def record_failure(key: str, *, now: float | None = None) -> Status:
    now = time.time() if now is None else now
    with _locked():
        data = _load()
        entry = data["profiles"].setdefault(key, {})
        failures = int(entry.get("failures", 0)) + 1
        entry["failures"] = failures
        if failures >= config.MAX_FAILED_ATTEMPTS:
            exponent = failures - config.MAX_FAILED_ATTEMPTS
            seconds = min(config.LOCKOUT_BASE_SECONDS * (2 ** exponent),
                          config.LOCKOUT_MAX_SECONDS)
            entry["locked_until"] = now + seconds
        else:
            entry["locked_until"] = 0.0
        _save(data)
        remaining = max(0.0, float(entry["locked_until"]) - now)
        return Status(failures, remaining)


def record_success(key: str) -> None:
    with _locked():
        data = _load()
        if key in data["profiles"]:
            del data["profiles"][key]
            _save(data)


def clear(key: str) -> None:
    """Administrative reset; equivalent to a successful login."""
    record_success(key)
