"""Download only the pinned Python/WASM runtime into the user's UI cache."""
from __future__ import annotations

import hashlib
from pathlib import Path
import os
import threading
import time
import urllib.request
from urllib.parse import urlsplit
import uuid

VERSION = "314.0.7"
BASE = f"https://cdn.jsdelivr.net/pyodide/v{VERSION}/full/"
FILES = {
    "pyodide.js": (18912, "3141b814715a72e59b51b1b18b9ceae5bf19f7c852417e431bb0a34feadf825c"),
    "pyodide.asm.mjs": (1250344, "f7cdc8ece80678ceb712f8e65ebe6d3a83203a180c399865f49612a051693635"),
    "pyodide.asm.wasm": (9598218, "cc36e3cab04fdfc9a63ff13eb52eae2b911bf46c025cc7b281f394bd3de1d5e6"),
    "python_stdlib.zip": (2545637, "fa1957e5777068fc4f7437f96d860ae2fbe9c19732ba06c84e004ec16dd7dd7a"),
    "pyodide-lock.json": (119077, "5dc2fc119108bc148c7457dc86e7675b5c87e1cafd420b9c34c1eaef7b36c010"),
}
NOTICES = {
    "LICENSE": ("python-runtime-LICENSE.txt", "1f256ecad192880510e84ad60474eab7589218784b9a50bc7ceee34c2b91f1d5"),
    "THIRD-PARTY-NOTICES.txt": ("python-runtime-NOTICES.txt", "4f5c326da2f662db6c9dc1a448bf2d6f9df396de67d74c406b147e1d54273a71"),
}


def file_hash(path):
    actual = hashlib.sha256()
    with path.open("rb") as source:
        for data in iter(lambda: source.read(131072), b""):
            actual.update(data)
    return actual.hexdigest()


class FixedRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        target = urlsplit(newurl)
        if target.scheme != "https" or target.hostname != "cdn.jsdelivr.net":
            raise ValueError("Unexpected runtime download redirect")
        return super().redirect_request(request, fp, code, message, headers, newurl)


class RuntimeCache:
    def __init__(self, path: str | Path):
        self.path = Path(path) / VERSION
        self.lock = threading.Lock()

    def _target(self, name):
        if name not in FILES:
            return None
        target = (self.path / name).resolve()
        return target if target.is_relative_to(self.path.resolve()) else None

    def _valid(self, name):
        target = self._target(name)
        size, digest = FILES[name]
        try:
            if target is None or not target.is_file() or target.stat().st_size != size:
                return False
            return file_hash(target) == digest
        except OSError:
            return False

    def status(self):
        notices = all((self.path / name).is_file() and file_hash(self.path / name) == value[1] for name, value in NOTICES.items())
        return {"version": VERSION, "ready": all(self._valid(name) for name in FILES) and notices,
                "bytes": sum(value[0] for value in FILES.values())}

    def asset(self, version, name):
        if version != VERSION or name not in FILES or not self._valid(name):
            return None
        return self._target(name)

    def prepare(self, cancel: threading.Event | None = None):
        deadline = time.monotonic() + 150
        while not self.lock.acquire(timeout=0.1):
            if (cancel and cancel.is_set()) or time.monotonic() > deadline:
                raise TimeoutError("Python runtime preparation was cancelled or timed out")
        try:
            self.path.mkdir(parents=True, exist_ok=True, mode=0o700)
            for name, (source_name, digest) in NOTICES.items():
                source = Path(__file__).with_name(source_name)
                if file_hash(source) != digest:
                    raise ValueError("Python runtime notice verification failed")
                target = self.path / name
                if target.resolve().is_relative_to(self.path.resolve()):
                    target.write_bytes(source.read_bytes())
                else:
                    raise ValueError("Invalid runtime notice path")
            opener = urllib.request.build_opener(FixedRedirects())
            for name, (size, digest) in FILES.items():
                if cancel and cancel.is_set():
                    raise TimeoutError("Python runtime preparation was cancelled")
                if self._valid(name):
                    continue
                target = self._target(name)
                if target is None:
                    raise ValueError("Invalid runtime cache path")
                temporary = self.path / (name + "." + uuid.uuid4().hex + ".part")
                try:
                    request = urllib.request.Request(BASE + name, headers={"User-Agent": "TinyStrata runtime setup"})
                    with opener.open(request, timeout=20) as response, temporary.open("xb") as output:
                        if response.status != 200:
                            raise OSError("Runtime download failed")
                        total, actual = 0, hashlib.sha256()
                        while data := response.read(65536):
                            if (cancel and cancel.is_set()) or time.monotonic() > deadline:
                                raise TimeoutError("Python runtime preparation was cancelled or timed out")
                            total += len(data)
                            if total > size:
                                raise ValueError("Runtime download size mismatch")
                            output.write(data)
                            actual.update(data)
                        output.flush()
                        os.fsync(output.fileno())
                    if total != size or actual.hexdigest() != digest:
                        raise ValueError("Runtime download verification failed")
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
            return self.status()
        finally:
            self.lock.release()
