"""Backend selection shared by the launcher and model profiles."""
from __future__ import annotations

import json
import os
import mmap
from functools import lru_cache
from pathlib import Path
import subprocess

from serve import config


def hardware_backend() -> str | None:
    """Prefer a usable NVIDIA driver; otherwise recognize AMD's PCI vendor ID."""
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                                timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode == 0 and "GPU " in result.stdout:
            return "cuda"
    except (OSError, subprocess.TimeoutExpired):
        pass
    if os.name == "nt":
        try:
            result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                "Get-CimInstance Win32_VideoController | ForEach-Object { $_.PNPDeviceID }"],
                capture_output=True, text=True, timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if result.returncode == 0 and "VEN_1002" in result.stdout.upper():
                return "hip"
        except (OSError, subprocess.TimeoutExpired):
            pass
    return None


def hip_ordinal(directories=(), executable=None) -> int | None:
    """Prefer a discrete HIP device, then the largest reported memory capacity."""
    # Query in a short-lived process: DLL search/runtime initialization must not
    # affect the Python API process or its later native model children.
    directories = tuple(directories)
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(map(str, directories)) + os.pathsep + env.get("PATH", "")
    # Existing visibility masks are honored instead of persisting a remapped ordinal.
    if env.get("HIP_VISIBLE_DEVICES") or env.get("ROCR_VISIBLE_DEVICES"):
        return None
    if executable is not None:
        from serve.hip_runtime import runtime_environment
        env = runtime_environment({"backend": "hip", "exe": executable, "lib_dirs": list(directories)}, env)
    from serve.windows_amd_telemetry import hip_cards
    cards = hip_cards(env)
    valid = [card for card in cards if type(card.get("mem_total")) is int and card["mem_total"] > 0]
    discrete = [card for card in valid if card.get("integrated") is False]
    # Unknown integrated status cannot safely be ranked by shared RAM capacity.
    if not discrete and any(card.get("integrated") is None for card in valid):
        return None
    pool = discrete or [card for card in valid if card.get("integrated") is True]
    return max(pool, key=lambda card: (card["mem_total"], -card["index"]))["index"] if pool else None


def executable_backend(executable: str | Path, directories=()) -> str | None:
    """Read cheap native build metadata; never load a model or initialize a GPU."""
    try:
        path = Path(executable).resolve()
        info = path.stat()
    except OSError:
        return None
    return _executable_backend(str(path), info.st_mtime_ns, info.st_size,
                               tuple(map(str, directories)), os.environ.get("PATH", ""))


@lru_cache(maxsize=128)
def _executable_backend(executable: str, modified_ns: int, size: int,
                        directories: tuple[str, ...], search_path: str) -> str | None:
    # Model catalogs are polled frequently. The file identity invalidates this
    # cache when a build is replaced; listing models never repeatedly starts it.
    # Runtime import names identify legacy builds too, without starting them.
    # mmap avoids copying large image-encoder executables during this check.
    try:
        if size < 64:
            imports = Path(executable).read_bytes().lower()
            return "hip" if b"amdhip64" in imports else "cuda" if b"cudart64_" in imports else None
        with open(executable, "rb") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as pe:
            if pe.find(b"amdhip64") >= 0 or pe.find(b"AMDhip64") >= 0:
                return "hip"
            if pe.find(b"cudart64_") >= 0 or pe.find(b"CUDART64_") >= 0:
                return "cuda"
            offset = int.from_bytes(pe[60:64], "little")
            if pe[:2] != b"MZ" or pe[offset:offset + 4] != b"PE\0\0":
                return None
    except (OSError, ValueError):
        return None
    env = dict(os.environ)
    env["PATH"] = search_path
    if directories:
        env["PATH"] = os.pathsep.join(map(str, directories)) + os.pathsep + env.get("PATH", "")
    try:
        result = subprocess.run([str(executable), "--capabilities"], env=env,
                                capture_output=True, text=True, timeout=10,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        report = json.loads(result.stdout) if result.returncode == 0 else {}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        report = {}
    if report.get("backend") in ("cuda", "hip"):
        return report["backend"]
    return None


def configure_backend(cfg: dict, requested: str | None = None, *, explicit_kv: str | None = None) -> dict:
    """Keep automatic KV selection in the engine and preserve saved HIP choices."""
    from serve.model_download import reject_retired_model
    reject_retired_model(cfg.get("arch"), cfg.get("model_name"))
    saved = cfg.get("backend")
    if saved not in (None, "cuda", "hip"):
        raise ValueError("backend must be cuda or hip")
    backend = requested if requested not in (None, "auto") else saved or "cuda"
    if backend not in ("cuda", "hip"):
        raise ValueError("backend must be auto, cuda or hip")
    result = {**cfg, "args": list(cfg.get("args") or []), "backend": backend}
    if backend != (saved or "cuda"):
        # CUDA and HIP ordinals describe different device lists.
        result.pop("gpu", None)
        result.pop("hip_ordinal", None)
    if backend == "hip":
        formats = ("auto", "int8", "fp8", "fp8-mma")
        if explicit_kv is not None and explicit_kv not in formats:
            raise ValueError("The AMD HIP route supports --kv-type auto, int8 or native fp8")
        # Let the model engine choose FP8 or INT8 from its actual capabilities.
        # Keep deliberate HIP choices, including auto, across later starts.
        # A profile from another backend starts with hardware-aware auto unless
        # the caller explicitly selects a format for this backend now.
        previous_kv = config.value_of(result, "kv_type")
        if explicit_kv is not None:
            chosen_kv = explicit_kv
        elif previous_kv in formats and (
                saved == "hip" or (previous_kv == "int8" and cfg.get("arch") == "clover_moe" and
                                    "clover_decode_policy_version" in cfg)):
            chosen_kv = previous_kv
        else:
            chosen_kv = "auto"
        result, _ = config.apply(result, {"kv_type": chosen_kv})
        # The engine chooses growable KV by default. Keep saved allocation modes:
        # an older --kv-whole may be a user's explicit choice, and old settings
        # do not record whether the launcher inserted it automatically.
    return result
