"""DPAPI encryption: round-trip, non-determinism and tamper failure."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from faceauth import secure


def test_dpapi_round_trip():
    plain = b"biometric-test\x00" + bytes(range(64))
    cipher = secure.protect(plain)
    assert cipher != plain
    assert secure.unprotect(cipher) == plain


def test_dpapi_ciphertext_is_randomised():
    plain = b"same biometric bytes"
    a, b = secure.protect(plain), secure.protect(plain)
    assert a != b
    assert secure.unprotect(a) == secure.unprotect(b) == plain


def test_atomic_protected_write_leaves_no_tempfile(tmp_path):
    path = tmp_path / "profile.dpapi"
    secure.atomic_protected_write(path, b"secret face bytes")
    assert secure.protected_read(path) == b"secret face bytes"
    assert not (tmp_path / "profile.dpapi.tmp").exists()


def test_tampering_is_rejected():
    cipher = bytearray(secure.protect(b"do not change"))
    cipher[len(cipher) // 2] ^= 0x40
    with pytest.raises(OSError):
        secure.unprotect(bytes(cipher))
