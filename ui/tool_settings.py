"""Model-independent UI search settings; credentials never leave this process."""
from __future__ import annotations

import base64
import ctypes
import json
import os
from pathlib import Path
import threading
import uuid

from ui.web_search import DEFAULT_SETTINGS, public_settings, validate_settings


def _protect(value: str) -> dict:
    if not value:
        return {"encoding": "empty", "value": ""}
    if os.name != "nt":
        # The containing directory and atomic file are private to this OS user.
        return {"encoding": "user-file", "value": value}
    return {"encoding": "dpapi", "value": base64.b64encode(_dpapi(value.encode(), False)).decode("ascii")}


def _unprotect(value: dict) -> str:
    if not isinstance(value, dict) or not isinstance(value.get("encoding"), str) or not isinstance(value.get("value"), str):
        raise ValueError("Invalid saved search credential")
    if value.get("encoding") == "empty":
        return ""
    if value.get("encoding") == "user-file" and os.name != "nt":
        return str(value["value"])
    if value.get("encoding") == "dpapi" and os.name == "nt":
        return _dpapi(base64.b64decode(value["value"], validate=True), True).decode("utf-8")
    raise ValueError("The saved search credential cannot be opened by this OS user")


def _dpapi(data: bytes, decrypt: bool) -> bytes:
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]

    buffer = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    result = Blob()
    crypt = ctypes.WinDLL("crypt32", use_last_error=True)
    free = ctypes.WinDLL("kernel32", use_last_error=True).LocalFree
    free.argtypes, free.restype = [ctypes.c_void_p], ctypes.c_void_p
    if decrypt:
        function = crypt.CryptUnprotectData
        function.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
        arguments = [ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result)]
    else:
        function = crypt.CryptProtectData
        function.argtypes = [ctypes.POINTER(Blob), wintypes.LPCWSTR, ctypes.c_void_p,
                             ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
        arguments = [ctypes.byref(source), "TinyStrata search", None, None, None, 1, ctypes.byref(result)]
    function.restype = wintypes.BOOL
    if not function(*arguments):
        raise OSError("Windows could not protect the search credential")
    try:
        return ctypes.string_at(result.data, result.size)
    finally:
        free(result.data)


class ToolSettings:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock = threading.RLock()

    def _read(self):
        if not self.path.exists():
            return dict(DEFAULT_SETTINGS), 0
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(saved, dict) or saved.get("version") != 1:
                raise ValueError("Invalid saved settings version")
            config = validate_settings({"provider": saved["provider"], "max_concurrency": saved["max_concurrency"],
                                        "brave_api_key": _unprotect(saved["credential"])})
            revision = saved["revision"]
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
                raise ValueError()
            return config, revision
        except (OSError, ValueError, KeyError, TypeError):
            raise ValueError("Saved search settings are unreadable; the original file has been preserved") from None

    def read(self):
        with self.lock:
            return self._read()[0]

    def public(self):
        with self.lock:
            config, revision = self._read()
            return {**public_settings(config), "revision": revision}

    def update(self, changes: dict):
        if not isinstance(changes, dict) or set(changes) - {"provider", "max_concurrency", "brave_api_key", "revision"}:
            raise ValueError("Unknown search setting")
        with self.lock:
            old, revision = self._read()
            expected = changes.get("revision", revision)
            if isinstance(expected, bool) or not isinstance(expected, int) or expected != revision:
                raise ValueError("Search settings changed in another page; reopen Chat settings")
            config = validate_settings({**old, **{k: v for k, v in changes.items() if k != "revision"}})
            saved = {"version": 1, "revision": revision + 1, "provider": config["provider"],
                     "max_concurrency": config["max_concurrency"], "credential": _protect(config["brave_api_key"])}
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            temporary = self.path.with_name(self.path.name + "." + uuid.uuid4().hex + ".tmp")
            try:
                with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as output:
                    json.dump(saved, output, ensure_ascii=False)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)
            return {**public_settings(config), "revision": revision + 1}
