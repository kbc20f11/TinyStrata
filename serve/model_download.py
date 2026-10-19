"""Download the pinned Unsloth MTP UD-IQ3_XXS model without an inference process.

This module accepts no model URL, repository, filename or destination from an
HTTP request.  The manifest is deliberately fixed; its private override exists
only so the transfer protocol can be exercised with a local HTTP fixture.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import shutil
import threading
from typing import Callable, Iterable
import urllib.request


@dataclass(frozen=True)
class _Manifest:
    repo: str
    revision: str
    filename: str
    size: int
    sha256: str
    url: str


MODEL_REPO = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
MODEL_REVISION = "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d"
MODEL_FILENAME = "Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
MODEL_SIZE = 14_069_266_720
# Pinned Hugging Face LFS SHA-256 from the repository manifest.
MODEL_SHA256 = "36f9ec0e4c775f6efd3a61c1ea76f0875128469c3d012e3d9e2495a90e7a7150"
FIXED_MODEL = _Manifest(
    MODEL_REPO, MODEL_REVISION, MODEL_FILENAME, MODEL_SIZE, MODEL_SHA256,
    f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_FILENAME}?download=true",
)
MMPROJ_FILENAME = "mmproj-F16.gguf"
MMPROJ_SIZE = 899_283_584
MMPROJ_SHA256 = "71f3cbc1f7cc0f30d09d41cfa924c0060827ebc33bf15ace7e86661e856f0160"
FIXED_MMPROJ = _Manifest(
    MODEL_REPO, MODEL_REVISION, MMPROJ_FILENAME, MMPROJ_SIZE, MMPROJ_SHA256,
    f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MMPROJ_FILENAME}?download=true",
)

LFM2_MODEL_REPO = "LiquidAI/LFM2.5-24B-A2B-GGUF"
LFM2_MODEL_REVISION = "0912d33141f77e10705345a6732bfd83f0252fdc"
LFM2_MODEL_FILENAME = "LFM2.5-24B-A2B-Q4_K_M.gguf"
LFM2_MODEL_SIZE = 14_526_258_464
LFM2_MODEL_SHA256 = "d3334405b96228eb9908cdebb5c6194908f2da64282bda852afc3a3910c64e7d"
FIXED_LFM2_MODEL = _Manifest(
    LFM2_MODEL_REPO, LFM2_MODEL_REVISION, LFM2_MODEL_FILENAME, LFM2_MODEL_SIZE, LFM2_MODEL_SHA256,
    f"https://huggingface.co/{LFM2_MODEL_REPO}/resolve/{LFM2_MODEL_REVISION}/{LFM2_MODEL_FILENAME}?download=true",
)


def manifest_for_arch(arch: str = "qwen35moe") -> _Manifest:
    """The server-selected fixed artifact, never selected by HTTP request data."""
    if arch == "qwen35moe":
        return FIXED_MODEL
    if arch == "lfm2moe":
        return FIXED_LFM2_MODEL
    raise ValueError(f"Unsupported TinyStrata model architecture: {arch}")


class _Cancelled(Exception):
    pass


class FixedModelDownloader:
    """One resumable background transfer and verified-model activation at a time.

    ``on_ready`` runs on the download worker after SHA-256 verification.  It can
    install tokenizer/configuration and start the engine.  If activation fails,
    another ``start()`` retries it using the verified file without a download.
    Existing configured files are reused only after the same size/hash checks.
    ``cancel()`` applies to transfer/hash verification; loading has already handed
    the verified file to the engine and cannot be cancelled through that endpoint.
    ``close()`` permanently stops the manager.  Activation callbacks must check
    ``closed`` before publishing a newly loaded engine during server shutdown.
    """

    _ACTIVE = frozenset(("downloading", "verifying", "loading"))

    def __init__(self, directory: Path, on_ready: Callable[[Path], None] | None = None,
                 existing_paths: Iterable[Path] = (), *,
                 _manifest: _Manifest | None = None, _chunk_size: int = 1024 * 1024,
                 _timeout: float = 15.0, _reserve_bytes: int = 64 * 1024 * 1024):
        self.directory = Path(directory).resolve()
        self.manifest = _manifest or FIXED_MODEL
        self.target = self.directory / self.manifest.filename
        self.partial = self.target.with_name(self.target.name + ".part")
        self._existing = tuple(Path(path).resolve() for path in existing_paths)
        self._on_ready = on_ready
        self._chunk_size = _chunk_size
        self._timeout = _timeout
        self._reserve_bytes = _reserve_bytes
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._closed = False
        self._thread: threading.Thread | None = None
        self._state = "idle"
        self._bytes = 0
        self._error: str | None = None
        self._ready_path: Path | None = None
        self._verified_path: Path | None = None
        self._restart_partial = False

    def _snapshot(self) -> dict:
        model = self.manifest
        return {
            "state": self._state,
            "bytes_downloaded": self._bytes,
            "total_bytes": model.size,
            "progress": min(1.0, self._bytes / model.size) if model.size else 0.0,
            "error": self._error,
            "can_cancel": self._state in ("downloading", "verifying") and not self._closed,
            "closed": self._closed,
            "model": {"repo": model.repo, "revision": model.revision,
                      "filename": model.filename, "sha256": model.sha256},
            "path": str(self._ready_path) if self._ready_path is not None else None,
        }

    def status(self) -> dict:
        with self._lock:
            return self._snapshot()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def start(self) -> dict:
        with self._lock:
            if self._closed or self._state == "ready" or self._state in self._ACTIVE:
                return self._snapshot()
            self._cancel.clear()
            self._error = None
            self._ready_path = None
            self._state = "verifying" if self._verified_path is not None else "downloading"
            self._thread = threading.Thread(target=self._run, name="model-download", daemon=True)
            self._thread.start()
            return self._snapshot()

    def cancel(self) -> dict:
        with self._lock:
            if self._state in ("downloading", "verifying"):
                self._cancel.set()
            return self._snapshot()

    def close(self, timeout: float = 5.0) -> None:
        with self._lock:
            self._closed = True
            self._cancel.set()
            worker = self._thread
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout)

    def _update(self, *, state: str | None = None, count: int | None = None) -> None:
        with self._lock:
            if state is not None:
                self._state = state
            if count is not None:
                self._bytes = count

    def _check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise _Cancelled()

    def _matches(self, path: Path) -> bool:
        if not path.is_file() or path.stat().st_size != self.manifest.size:
            return False
        self._update(state="verifying", count=self.manifest.size)
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while True:
                self._check_cancelled()
                block = source.read(self._chunk_size)
                if not block:
                    break
                digest.update(block)
        self._check_cancelled()
        return digest.hexdigest() == self.manifest.sha256

    def _download(self) -> Path:
        model = self.manifest
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.target.exists():
            if self._matches(self.target):
                return self.target
            raise ValueError("The existing target file does not match the pinned model. "
                             "Move it aside before downloading; it was not overwritten.")
        for candidate in self._existing:
            if candidate != self.target and self._matches(candidate):
                return candidate
        self._check_cancelled()
        if self._restart_partial:
            # A failed checksum is retained for inspection until an explicit retry.
            with self.partial.open("wb"):
                pass
            self._restart_partial = False
        offset = self.partial.stat().st_size if self.partial.exists() else 0
        if offset > model.size:
            raise ValueError("The partial file is larger than the pinned model; move it aside to retry.")
        self._update(state="downloading", count=offset)
        remaining = model.size - offset
        if remaining and shutil.disk_usage(self.directory).free < remaining + self._reserve_bytes:
            raise OSError(f"Not enough disk space: need {remaining + self._reserve_bytes:,} more bytes "
                          "including the download safety reserve.")
        if remaining:
            headers = {"User-Agent": "TinyStrata-model-download/1", "Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(model.url, headers=headers)
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                self._check_cancelled()
                status = response.status
                if status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if not match:
                        raise ValueError("The download server returned an invalid Content-Range.")
                    first, last, total = map(int, match.groups())
                    if first != offset or total != model.size or not first <= last < total:
                        raise ValueError("The download server returned a different model range or size.")
                    response_bytes = last - first + 1
                elif status == 200:
                    # Some servers ignore Range: restart only after validating headers.
                    offset = 0
                    response_bytes = model.size
                else:
                    raise ValueError(f"Unexpected model download HTTP status: {status}")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != response_bytes:
                    raise ValueError("The download server returned an unexpected Content-Length.")
                encoding = response.headers.get("Content-Encoding", "identity").lower()
                if encoding not in ("", "identity"):
                    raise ValueError("Compressed transfer cannot be used for a resumable model download.")
                self._update(count=offset)
                received = 0
                with self.partial.open("ab" if offset else "wb") as output:
                    while True:
                        self._check_cancelled()
                        block = response.read(min(self._chunk_size, response_bytes - received + 1))
                        if not block:
                            break
                        if received + len(block) > response_bytes:
                            raise ValueError("The download server sent more bytes than its declared range.")
                        output.write(block)
                        received += len(block)
                        self._update(count=offset + received)
                    output.flush()
                if received != response_bytes or offset + received != model.size:
                    raise OSError("Model download ended early; retry to resume the retained partial file.")
        self._check_cancelled()
        if not self._matches(self.partial):
            self._restart_partial = True
            raise ValueError("Model SHA-256 verification failed. The partial file was retained; "
                             "retry to download it again from the beginning.")
        self._check_cancelled()
        # Do not replace a file created by another process while the transfer ran.
        if self.target.exists():
            raise FileExistsError("A target model appeared during download; it was not overwritten.")
        # On the supported Windows host rename fails if the destination exists,
        # including a file created after the check above; replace would overwrite it.
        self.partial.rename(self.target)
        return self.target

    def _run(self) -> None:
        try:
            if not re.fullmatch(r"[0-9a-f]{64}", self.manifest.sha256):
                raise ValueError("The pinned model checksum is missing or invalid.")
            path = self._verified_path
            if path is None or not self._matches(path):
                path = self._download()
            self._verified_path = path
            self._check_cancelled()
            if self._on_ready is not None:
                # Serialize cancellation with the transition into activation.
                # Once loading begins, the public cancel endpoint is a no-op.
                with self._lock:
                    self._check_cancelled()
                    self._state = "loading"
                self._on_ready(path)
            # Activation can finish while shutdown requests cancellation. A verified
            # model remains on disk and can be activated again on the next launch.
            self._check_cancelled()
            with self._lock:
                self._check_cancelled()
                self._ready_path = path
                self._state = "ready"
                self._bytes = self.manifest.size
        except _Cancelled:
            self._update(state="cancelled")
        except Exception as exc:
            with self._lock:
                self._state = "error"
                self._error = str(exc) or type(exc).__name__


class FixedModelBundleDownloader(FixedModelDownloader):
    """Install the pinned language model and projector as one cancellable job.

    Each file uses the same resumable, size-checked and SHA-256-checked transfer.
    Activation happens only after both files have passed verification. The API
    keeps its existing aggregate progress fields and adds per-file status.
    """

    def __init__(self, directory: Path, on_ready: Callable[[Path, Path], None] | None = None,
                 existing_paths: Iterable[Path] = (), projector_paths: Iterable[Path] = (), *,
                 _manifests: tuple[_Manifest, _Manifest] | None = None, **transfer_options):
        model, projector = _manifests or (FIXED_MODEL, FIXED_MMPROJ)
        super().__init__(directory, _manifest=model, **transfer_options)
        self._bundle_ready = on_ready
        self._assets = (
            FixedModelDownloader(directory, existing_paths=existing_paths, _manifest=model, **transfer_options),
            FixedModelDownloader(directory, existing_paths=projector_paths, _manifest=projector, **transfer_options),
        )
        self._active_asset: FixedModelDownloader | None = None

    def _snapshot(self) -> dict:
        result = super()._snapshot()
        assets = [asset.status() for asset in self._assets]
        total = sum(asset["total_bytes"] for asset in assets)
        count = sum(asset["bytes_downloaded"] for asset in assets)
        state = self._state
        active = self._active_asset
        if state in ("downloading", "verifying") and active is not None:
            state = active.status()["state"]
            if state not in ("downloading", "verifying"):
                state = "verifying"
        result.update(state=state, total_bytes=total, bytes_downloaded=count,
                      progress=min(1.0, count / total) if total else 0.0,
                      can_cancel=self._state in ("downloading", "verifying") and not self._closed,
                      asset=active.manifest.filename if active is not None else None,
                      assets=assets)
        return result

    def start(self) -> dict:
        with self._lock:
            if self._closed or self._state == "ready" or self._state in self._ACTIVE:
                return self._snapshot()
            for asset in self._assets:
                asset._cancel.clear()
                with asset._lock:
                    asset._error = None
            self._cancel.clear()
            self._error = None
            self._ready_path = None
            self._active_asset = None
            self._state = "downloading"
            self._thread = threading.Thread(target=self._run, name="model-download", daemon=True)
            self._thread.start()
            return self._snapshot()

    def cancel(self) -> dict:
        with self._lock:
            if self._state in ("downloading", "verifying"):
                self._cancel.set()
                for asset in self._assets:
                    asset._cancel.set()
            return self._snapshot()

    def close(self, timeout: float = 5.0) -> None:
        for asset in self._assets:
            asset._cancel.set()
        super().close(timeout)

    def _run(self) -> None:
        try:
            paths = []
            for asset in self._assets:
                self._check_cancelled()
                with self._lock:
                    self._active_asset = asset
                asset._run()
                self._check_cancelled()
                status = asset.status()
                if status["state"] != "ready":
                    raise ValueError(f'{asset.manifest.filename}: {status["error"] or status["state"]}')
                paths.append(Path(status["path"]))
            with self._lock:
                self._check_cancelled()
                self._state = "loading"
            if self._bundle_ready is not None:
                self._bundle_ready(*paths)
            with self._lock:
                self._check_cancelled()
                self._ready_path = paths[0]
                self._state = "ready"
        except _Cancelled:
            self._update(state="cancelled")
        except Exception as exc:
            with self._lock:
                self._state = "error"
                self._error = str(exc) or type(exc).__name__
