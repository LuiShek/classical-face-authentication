"""Windows DPAPI protection for biometric bytes.

DPAPI binds ciphertext to the current Windows user account. No application key
is stored beside the data, and copying a profile directory to another account is
not enough to decrypt it. This is local-at-rest protection, not a defence
against malware already running as the signed-in user.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
from ctypes import wintypes

CRYPTPROTECT_UI_FORBIDDEN = 0x01
ENTROPY = b"faceauth/dpapi/v1"


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD),
                ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _input_blob(data: bytes) -> tuple[_Blob, object]:
    buf = ctypes.create_string_buffer(data)
    blob = _Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte)))
    return blob, buf                         # keep the buffer alive for the call


def _win_api():
    if os.name != "nt":
        raise RuntimeError("faceauth biometric encryption requires Windows DPAPI")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_Blob), wintypes.LPCWSTR, ctypes.POINTER(_Blob),
        wintypes.LPVOID, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(_Blob)]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_Blob), ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(_Blob), wintypes.LPVOID, wintypes.LPVOID,
        wintypes.DWORD, ctypes.POINTER(_Blob)]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL
    return crypt32, kernel32


def _call(data: bytes, *, decrypt: bool) -> bytes:
    if not isinstance(data, bytes):
        raise TypeError("DPAPI input must be bytes")
    crypt32, kernel32 = _win_api()
    src, src_buf = _input_blob(data)
    entropy, entropy_buf = _input_blob(ENTROPY)
    dst = _Blob()
    description = wintypes.LPWSTR()
    if decrypt:
        ok = crypt32.CryptUnprotectData(
            ctypes.byref(src), ctypes.byref(description), ctypes.byref(entropy),
            None, None, CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(dst))
    else:
        ok = crypt32.CryptProtectData(
            ctypes.byref(src), "faceauth biometric", ctypes.byref(entropy),
            None, None, CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(dst))
    # Keep both input buffers alive until the native function has returned.
    _ = src_buf, entropy_buf
    if not ok:
        code = ctypes.get_last_error()
        raise OSError(code, ctypes.FormatError(code))
    try:
        return ctypes.string_at(dst.pbData, dst.cbData)
    finally:
        if dst.pbData:
            kernel32.LocalFree(ctypes.cast(dst.pbData, wintypes.HLOCAL))
        if description:
            kernel32.LocalFree(ctypes.cast(description, wintypes.HLOCAL))


def protect(data: bytes) -> bytes:
    return _call(data, decrypt=False)


def unprotect(data: bytes) -> bytes:
    return _call(data, decrypt=True)


def atomic_protected_write(path: Path, plaintext: bytes) -> None:
    """Protect, independently verify, then atomically replace the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ciphertext = protect(plaintext)
    recovered = unprotect(ciphertext)
    if hashlib.sha256(recovered).digest() != hashlib.sha256(plaintext).digest():
        raise RuntimeError(f"DPAPI verification failed for {path}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(ciphertext)
    tmp.replace(path)


def protected_read(path: Path) -> bytes:
    try:
        return unprotect(Path(path).read_bytes())
    except OSError as exc:
        raise SystemExit(
            f"error: cannot decrypt {path}; it must be opened by the Windows "
            f"account that protected it ({exc})") from exc
