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


def hip_ordinal(directories=()) -> int | None:
    """Prefer a discrete HIP device, then the largest reported memory capacity."""
    # Query in a short-lived process: DLL search/runtime initialization must not
    # affect the Python API process or its later native model children.
    # hipDeviceAttributeIntegrated == 16 in AMD's hip_runtime_api.h. Do not
    # compare only memory sizes: an APU can report more shared RAM than a dGPU.
    script = ("import ctypes,os,pathlib; dirs=[d for d in os.environ.get('PATH','').split(os.pathsep) if d and os.path.isdir(d)]; "
              "handles=[os.add_dll_directory(d) for d in dirs] if hasattr(os,'add_dll_directory') else []; "
              "names=[p for d in dirs "
              "if d for p in pathlib.Path(d).glob('amdhip64*.dll')]; "
              "h=ctypes.CDLL(str(names[0]) if names else 'amdhip64.dll'); "
              "n=ctypes.c_int(); assert h.hipGetDeviceCount(ctypes.byref(n))==0; "
              "totals=[]\nfor i in range(n.value):\n "
              "t=ctypes.c_size_t(); r=h.hipDeviceTotalMem(ctypes.byref(t),i); "
              "integrated=ctypes.c_int(1); h.hipDeviceGetAttribute(ctypes.byref(integrated),16,i); "
              "totals.append((not integrated.value,t.value,-i)) if r==0 and t.value else None\n"
              "print(-max(totals)[2] if totals else '')")
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(map(str, directories)) + os.pathsep + env.get("PATH", "")
    # Existing visibility masks are honored instead of persisting a remapped ordinal.
    if env.get("HIP_VISIBLE_DEVICES") or env.get("ROCR_VISIBLE_DEVICES"):
        return None
    try:
        import sys
        result = subprocess.run([sys.executable, "-c", script], env=env,
                                capture_output=True, text=True, timeout=15,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return int(result.stdout.strip()) if result.returncode == 0 and result.stdout.strip().isdigit() else None
    except (OSError, subprocess.TimeoutExpired):
        return None


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
    """HIP uses INT8 KV; switching backend retains unrelated user settings."""
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
        if explicit_kv not in (None, "auto", "int8"):
            raise ValueError("The AMD HIP route uses INT8 KV; choose --kv-type int8 or auto")
        result, _ = config.apply(result, {"kv_type": "int8"})
    return result
