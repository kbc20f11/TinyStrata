"""Opt-in Intel native build metadata; no probe runs at import time.

The explicit probe starts only ``exe --capabilities``. That native branch
returns before device enumeration, queue construction or model loading. The
result describes compiled paths, not hardware compatibility or quality approval.
Launch, model preparation, and installer import validate this metadata before
publishing settings or replacing a resident model.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess


QWEN_MODEL = "qwen3.6-35b-a3b"
B390_PROFILE = "b390-qwen36"
TIEL_MODEL = "tiel-coder-35b-a3b"
TIEL_PROFILE = "b390-tiel-coder"
_PROFILE_MODELS = {B390_PROFILE: QWEN_MODEL, TIEL_PROFILE: TIEL_MODEL}
_PATHS = ("scalar", "dp4a", "int8_xmx")
_SHA256 = re.compile(r"^[a-fA-F0-9]{64}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class IntelCapabilityError(ValueError):
    """A selected native build has missing, changed or incompatible metadata."""


@dataclass(frozen=True)
class IntelCapabilities:
    executable: str
    sha256: str
    identity: tuple[int, ...]
    report: dict

    @property
    def legacy(self) -> bool:
        return "capabilities_schema" not in self.report


def _identifier(value, description):
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise IntelCapabilityError(f"Invalid Intel {description}")


def _types(value, description):
    if (not isinstance(value, list) or
            any(type(item) is not int or not 0 <= item <= 65535 for item in value) or
            value != sorted(set(value))):
        raise IntelCapabilityError(f"Intel {description} must be sorted unique GGUF type integers")
    return set(value)


def _check_report(report):
    if not isinstance(report, dict):
        raise IntelCapabilityError("Intel capabilities must be a JSON object")
    if report.get("backend") != "sycl" or report.get("experimental") is not True:
        raise IntelCapabilityError("The executable does not identify the experimental sycl backend")
    if report.get("model_families") != []:
        raise IntelCapabilityError("Intel model_families must retain the explicit formal-adoption boundary")
    if report.get("images") is not False or report.get("serve") is not True or report.get("mtp") is not True:
        raise IntelCapabilityError("Intel capabilities must identify the experimental text/MTP server")
    kv = report.get("kv_types")
    if (not isinstance(kv, list) or not kv or any(item not in ("fp32", "int8") for item in kv) or
            len(set(kv)) != len(kv)):
        raise IntelCapabilityError("Invalid Intel KV capability list")
    depth = report.get("experimental_mtp_max_depth")
    if type(depth) is not int or not 1 <= depth <= 7:
        raise IntelCapabilityError("Invalid Intel experimental MTP depth")
    if "capabilities_schema" not in report:
        # Partial new metadata must never silently downgrade into legacy policy.
        if any(key in report for key in ("experimental_models", "experimental_profiles", "build", "weight_types")):
            raise IntelCapabilityError("Intel build metadata is missing capabilities_schema")
        return
    if type(report["capabilities_schema"]) is not int or report["capabilities_schema"] != 1:
        raise IntelCapabilityError("Unsupported Intel capabilities_schema")
    models = report.get("experimental_models")
    if not isinstance(models, list) or not models:
        raise IntelCapabilityError("Intel experimental_models must be a nonempty list")
    for model in models:
        _identifier(model, "experimental model ID")
    if len(set(models)) != len(models):
        raise IntelCapabilityError("Duplicate Intel experimental model ID")
    profiles = report.get("experimental_profiles")
    if not isinstance(profiles, dict):
        raise IntelCapabilityError("Intel experimental_profiles must be an object")
    for name, version in profiles.items():
        _identifier(name, "experimental profile ID")
        if type(version) is not int or version < 1:
            raise IntelCapabilityError("Invalid Intel experimental profile contract version")
    build = report.get("build")
    if (not isinstance(build, dict) or type(build.get("mkl")) is not bool or
            type(build.get("int8_xmx")) is not bool or
            build.get("aot_target") not in ("", "intel_gpu_ptl_h")):
        raise IntelCapabilityError("Invalid Intel compiled build metadata")
    for name, model in _PROFILE_MODELS.items():
        if name in profiles and (profiles[name] != 1 or not build["mkl"] or not build["int8_xmx"] or model not in models):
            raise IntelCapabilityError(f"The {name} contract requires its model and compiled MKL and INT8 XMX")
    paths = report.get("weight_types")
    if not isinstance(paths, dict) or set(paths) != set(_PATHS):
        raise IntelCapabilityError("Intel weight_types must describe scalar, dp4a and int8_xmx paths")
    scalar, dp4a, xmx = (_types(paths[name], name + " weight_types") for name in _PATHS)
    if not scalar or not dp4a <= scalar or not xmx <= dp4a or (not build["int8_xmx"] and xmx):
        raise IntelCapabilityError("Intel compiled quantization paths are inconsistent")


def validate_capabilities(report: dict, *, model: str = QWEN_MODEL, profile: str = "generic",
                          required_types: Mapping | None = None, allow_legacy: bool = False) -> None:
    """Pure compatibility check; never start a process or inspect a device.

    New imports should keep ``allow_legacy=False``. Explicit legacy compatibility
    retains only the existing Qwen generic/B390 route: it cannot prove any newly
    required quantization path or enable another model. Callers remain responsible
    for actual GGUF, selected-device, numerical and quality validation.
    """
    if type(allow_legacy) is not bool:
        raise IntelCapabilityError("allow_legacy must be a boolean")
    _identifier(model, "requested model ID")
    _identifier(profile, "requested profile ID")
    required = {} if required_types is None else required_types
    if not isinstance(required, Mapping):
        raise IntelCapabilityError("required_types must map an execution path to GGUF type integers")
    for path, types in required.items():
        if path not in _PATHS or not isinstance(types, (list, tuple, set, frozenset)):
            raise IntelCapabilityError("Invalid required Intel quantization path")
        if any(type(item) is not int or not 0 <= item <= 65535 for item in types):
            raise IntelCapabilityError("Invalid required Intel GGUF type")
    _check_report(report)
    if "capabilities_schema" not in report:
        if not allow_legacy or model != QWEN_MODEL or profile not in ("generic", B390_PROFILE):
            raise IntelCapabilityError("Legacy Intel metadata is allowed only for an explicitly retained Qwen route")
        if any(required.values()):
            raise IntelCapabilityError("Legacy Intel metadata cannot prove required quantization paths")
        return
    if model not in report["experimental_models"]:
        raise IntelCapabilityError(f"The Intel build does not declare experimental model {model}")
    if profile != "generic" and report["experimental_profiles"].get(profile) != 1:
        raise IntelCapabilityError(f"The Intel build does not declare experimental profile {profile} version 1")
    if profile in _PROFILE_MODELS and model != _PROFILE_MODELS[profile]:
        raise IntelCapabilityError(f"The {profile} profile is specific to {_PROFILE_MODELS[profile]}")
    if model == TIEL_MODEL:
        if profile != TIEL_PROFILE:
            raise IntelCapabilityError("Tiel requires its model-specific b390-tiel-coder profile")
        for path in _PATHS:
            if 18 not in report["weight_types"][path]:
                raise IntelCapabilityError(f"The Intel {path} path lacks GGUF type18 required by Tiel")
    for path, types in required.items():
        missing = set(types) - set(report["weight_types"][path])
        if missing:
            raise IntelCapabilityError(f"The Intel {path} path lacks GGUF types {sorted(missing)}")


def _identity(path):
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise IntelCapabilityError("The Intel executable is not a regular file")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _fingerprint(path):
    before = _identity(path)
    with path.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if _identity(path) != before:
        raise IntelCapabilityError("The Intel executable changed while being hashed")
    return before, sha256


def capability_environment(cfg: dict, inherited: Mapping | None = None) -> dict[str, str]:
    """Use only configured app-local directories and Windows on the child PATH.

    Explicit selector/filter controls retain their values; SDK discovery/PATH
    entries are not needed by this CPU-only command. The parent is unchanged.
    """
    from serve.intel_backend import runtime_environment
    original = os.environ if inherited is None else inherited
    if not isinstance(original, Mapping):
        raise IntelCapabilityError("The inherited Intel environment must be a mapping")
    normalized, names = {}, {}
    for key, value in original.items():
        name = str(key)
        canonical = names.setdefault(name.upper(), name)
        normalized[canonical] = str(value)
    environment = runtime_environment({**cfg, "lib_dirs": []}, normalized)
    base = Path(cfg.get("cwd") or Path.cwd())
    executable = Path(cfg["exe"])
    if not executable.is_absolute():
        executable = base / executable
    directories = cfg.get("lib_dirs", [])
    if not isinstance(directories, list) or not all(isinstance(item, str) and item for item in directories):
        raise IntelCapabilityError("Intel lib_dirs must be a list of app-local directory strings")
    paths = [str(executable.resolve().parent)]
    for directory in directories:
        path = Path(directory)
        paths.append(str((path if path.is_absolute() else base / path).resolve()))
    system_root = next((value for key, value in environment.items() if key.upper() == "SYSTEMROOT"), None)
    if not system_root:
        raise IntelCapabilityError("Intel capability probes require the inherited Windows SystemRoot")
    paths.extend((str(Path(system_root) / "System32"), str(Path(system_root))))
    unique, seen = [], set()
    for path in paths:
        if path.casefold() not in seen:
            seen.add(path.casefold())
            unique.append(path)
    path_key = next((key for key in environment if key.upper() == "PATH"), "PATH")
    environment[path_key] = ";".join(unique)
    return environment


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise IntelCapabilityError("Duplicate key in Intel capability JSON")
        result[key] = value
    return result


def _nonfinite(_):
    raise ValueError("Nonfinite JSON constant")


@lru_cache(maxsize=32)
def _probe_cached(executable, identity, sha256, environment, timeout):
    path = Path(executable)
    if _fingerprint(path) != (identity, sha256):
        raise IntelCapabilityError("The Intel executable changed before its metadata probe")
    try:
        result = subprocess.run([executable, "--capabilities"], env=dict(environment), cwd=str(path.parent),
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
                                errors="strict", timeout=timeout,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, UnicodeError, subprocess.TimeoutExpired) as error:
        raise IntelCapabilityError(f"Intel capability probe failed: {type(error).__name__}") from error
    if _fingerprint(path) != (identity, sha256):
        raise IntelCapabilityError("The Intel executable changed during its metadata probe")
    if result.returncode != 0:
        raise IntelCapabilityError(f"Intel capability probe exited with code {result.returncode}")
    if len(result.stdout) > 65536:
        raise IntelCapabilityError("Intel capability report is too large")
    try:
        report = json.loads(result.stdout, object_pairs_hook=_object, parse_constant=_nonfinite)
    except (ValueError, TypeError) as error:
        raise IntelCapabilityError("Invalid Intel capability JSON") from error
    _check_report(report)
    return report


def probe_capabilities(cfg: dict, *, inherited: Mapping | None = None, timeout: float = 10,
                       expected_sha256: str | None = None) -> IntelCapabilities:
    """Read cached, identity-and-SHA-bound metadata from an explicitly selected exe.

    Hashing on each call prevents a same-size/timestamp replacement from reusing
    stale metadata. Only the subprocess result is cached, keyed by the full
    executable identity, SHA and child environment. Returned reports are copies.
    A successful probe alone does not opt a legacy report into compatibility.
    """
    if cfg.get("backend") != "sycl" or not isinstance(cfg.get("exe"), str) or not cfg["exe"]:
        raise IntelCapabilityError("Intel metadata requires an explicit sycl executable")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise IntelCapabilityError("Intel metadata timeout must be between 0 and 60 seconds")
    if expected_sha256 is not None and (not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256)):
        raise IntelCapabilityError("Invalid expected Intel executable SHA256")
    executable = Path(cfg["exe"])
    if not executable.is_absolute():
        executable = Path(cfg.get("cwd") or Path.cwd()) / executable
    executable = executable.resolve()
    try:
        identity, sha256 = _fingerprint(executable)
        if expected_sha256 is not None and sha256 != expected_sha256.lower():
            raise IntelCapabilityError("Intel executable SHA256 does not match the expected runtime")
        environment = capability_environment(cfg, inherited)
        report = _probe_cached(str(executable), identity, sha256, tuple(sorted(environment.items())), float(timeout))
    except OSError as error:
        raise IntelCapabilityError("Cannot read the selected Intel executable") from error
    return IntelCapabilities(str(executable), sha256, identity, deepcopy(report))
