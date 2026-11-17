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
from serve.intel_models import (MODEL_PROFILES, MODEL_CONTRACTS, TARGET_ONLY_MODELS,
                                QWEN_ENGINE_MODELS, VISION_MODELS, VISION_MODEL_CONTRACTS,
                                VISION_PROTOCOL, VISION_CONTRACT_VERSION, EXPERIMENTAL_KV_MODELS, FP8_KV_FORMAT)


QWEN_MODEL = "qwen3.6-35b-a3b"
B390_PROFILE = "b390-qwen36"
TIEL_MODEL = "tiel-coder-35b-a3b"
TIEL_PROFILE = "b390-tiel-coder"
_PROFILE_MODELS = {profile: model for model, profile in MODEL_PROFILES.items()}
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
    if type(report.get("images")) is not bool or report.get("serve") is not True or type(report.get("mtp")) is not bool:
        raise IntelCapabilityError("Intel capabilities must identify the experimental image/MTP server")
    if report["images"] and (report.get("image_protocol") != VISION_PROTOCOL or
                            type(report.get("vision_contract_version")) is not int or
                            report["vision_contract_version"] != VISION_CONTRACT_VERSION):
        raise IntelCapabilityError("Intel images require the explicit GENI/SVE1 vision contract")
    kv = report.get("kv_types")
    if (not isinstance(kv, list) or not kv or any(item not in ("fp32", "int8", "fp8") for item in kv) or
            len(set(kv)) != len(kv)):
        raise IntelCapabilityError("Invalid Intel KV capability list")
    if "fp8" in kv and ("capabilities_schema" not in report or
            report.get("experimental_kv_models") != EXPERIMENTAL_KV_MODELS or report.get("fp8_kv_format") != FP8_KV_FORMAT):
        raise IntelCapabilityError("FP8 requires the explicit experimental Qwen3.6 E4M3FN codec contract")
    if "fp8" not in kv and any(key in report for key in ("experimental_kv_models", "fp8_kv_format")):
        raise IntelCapabilityError("Partial experimental FP8 metadata is unsupported")
    depth = report.get("experimental_mtp_max_depth")
    if type(depth) is not int or not 0 <= depth <= 7 or (depth == 0) != (report["mtp"] is False):
        raise IntelCapabilityError("Invalid Intel experimental MTP depth")
    if "context_auto" in report or "max_context" in report:
        if (type(report.get("context_auto")) is not bool or type(report.get("max_context")) is not int or
                not 16 <= report["max_context"] <= 262144):
            raise IntelCapabilityError("Invalid Intel native context capability contract")
    if "capabilities_schema" not in report:
        if report["images"]:
            raise IntelCapabilityError("Legacy Intel metadata cannot advertise image input")
        # Partial new metadata must never silently downgrade into legacy policy.
        if any(key in report for key in ("experimental_models", "experimental_profiles", "build", "weight_types", "experimental_model_contracts", "context_auto", "max_context", "image_protocol", "vision_contract_version")):
            raise IntelCapabilityError("Intel build metadata is missing capabilities_schema")
        if report["mtp"] is not True:
            raise IntelCapabilityError("Legacy Intel metadata requires the retained text/MTP server")
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
    if "fp8" in kv and QWEN_MODEL not in models:
        raise IntelCapabilityError("FP8 requires the experimental Qwen3.6 model identity")
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
        if name in profiles and (profiles[name] != 1 or not build["mkl"] or
                model in QWEN_ENGINE_MODELS and not build["int8_xmx"] or model not in models):
            features = "MKL and INT8 XMX" if model in QWEN_ENGINE_MODELS else "MKL"
            raise IntelCapabilityError(f"The {name} contract requires its model and compiled {features}")
    contracts = report.get("experimental_model_contracts", {})
    if not isinstance(contracts, dict) or any(model not in models for model in contracts):
        raise IntelCapabilityError("Intel experimental_model_contracts must describe declared model IDs")
    for model in TARGET_ONLY_MODELS.intersection(models):
        declared = contracts.get(model)
        expected = (VISION_MODEL_CONTRACTS[model] if model in VISION_MODEL_CONTRACTS and
                    isinstance(declared, dict) and declared.get("version") == 2 else MODEL_CONTRACTS[model])
        if not isinstance(declared, dict) or set(declared) != set(expected):
            raise IntelCapabilityError(f"The Intel build lacks the explicit target-only contract for {model}")
        if any(type(declared[key]) is not type(value) or declared[key] != value for key, value in expected.items()):
            raise IntelCapabilityError(f"Unsupported Intel model-specific capability contract for {model}")
        if declared["images"] and not report["images"]:
            raise IntelCapabilityError(f"The Intel build lacks image input declared for {model}")
        if not set(declared["kv_types"]) <= set(kv):
            raise IntelCapabilityError(f"The Intel build lacks the declared KV baseline for {model}")
    paths = report.get("weight_types")
    if not isinstance(paths, dict) or set(paths) != set(_PATHS):
        raise IntelCapabilityError("Intel weight_types must describe scalar, dp4a and int8_xmx paths")
    scalar, dp4a, xmx = (_types(paths[name], name + " weight_types") for name in _PATHS)
    if not scalar or not dp4a <= scalar or not xmx <= dp4a or (not build["int8_xmx"] and xmx):
        raise IntelCapabilityError("Intel compiled quantization paths are inconsistent")


def validate_capabilities(report: dict, *, model: str = QWEN_MODEL, profile: str = "generic",
                          required_types: Mapping | None = None, allow_legacy: bool = False,
                          requested_context: int | str | None = None, require_images: bool = False,
                          requested_kv: str | None = None) -> None:
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
    if requested_kv is not None:
        if requested_kv not in ("int8", "fp32", "fp8"):
            raise IntelCapabilityError("Invalid requested Intel KV format")
        if (requested_kv not in report["kv_types"] or requested_kv == "fp8" and model != QWEN_MODEL
                or model in TARGET_ONLY_MODELS and requested_kv != "int8"):
            raise IntelCapabilityError("The selected Intel runtime lacks the requested model-specific KV format")
    if type(require_images) is not bool:
        raise IntelCapabilityError("require_images must be a boolean")
    if require_images and not supports_images(report, model):
        raise IntelCapabilityError("This Intel model runtime lacks image input; update its runtime or select vision_mode=no")
    if requested_context is not None:
        from serve.intel_models import MODEL_CONTEXT_LIMITS
        model_limit = MODEL_CONTEXT_LIMITS.get(model)
        if model_limit is None:
            raise IntelCapabilityError("Unknown Intel model context contract")
        native_limit = report.get("max_context", 32768)  # Retained old native explicit-only ceiling.
        # Clover shares an executable with Qwen/Tiel. Its trained context is
        # smaller than the binary's top-level ceiling; the exact model contract
        # is checked by _check_report before it may restrict this capability.
        declared_limit = report.get("experimental_model_contracts", {}).get(model, {}).get("context_max")
        if declared_limit is not None:
            native_limit = min(native_limit, declared_limit)
        if native_limit > model_limit:
            raise IntelCapabilityError("Intel native context capability exceeds this model's declared ceiling")
        if requested_context == "auto":
            if report.get("context_auto") is not True:
                raise IntelCapabilityError("This Intel runtime lacks memory-aware context auto; update its model runtime instead of fixing a smaller context")
        elif type(requested_context) is not int or not 16 <= requested_context <= native_limit:
            raise IntelCapabilityError(f"The selected Intel runtime supports explicit context up to {native_limit}; the request was not reduced")
    if "capabilities_schema" not in report:
        if not allow_legacy or model != QWEN_MODEL or profile not in ("generic", B390_PROFILE):
            raise IntelCapabilityError("Legacy Intel metadata is allowed only for an explicitly retained Qwen route")
        if any(required.values()):
            raise IntelCapabilityError("Legacy Intel metadata cannot prove required quantization paths")
        return
    if model not in report["experimental_models"]:
        raise IntelCapabilityError(f"The Intel build does not declare experimental model {model}")
    if model in QWEN_ENGINE_MODELS and (report["mtp"] is not True or report["experimental_mtp_max_depth"] < 1):
        raise IntelCapabilityError("The retained Qwen/Tiel native contract requires MTP support")
    if model not in (*QWEN_ENGINE_MODELS, *TARGET_ONLY_MODELS):
        raise IntelCapabilityError("Unknown Intel model capability contract")
    if profile != "generic" and report["experimental_profiles"].get(profile) != 1:
        raise IntelCapabilityError(f"The Intel build does not declare experimental profile {profile} version 1")
    if profile in _PROFILE_MODELS and model != _PROFILE_MODELS[profile]:
        raise IntelCapabilityError(f"The {profile} profile is specific to {_PROFILE_MODELS[profile]}")
    if model == TIEL_MODEL:
        if profile not in ("generic", TIEL_PROFILE):
            raise IntelCapabilityError("Tiel requires generic or its model-specific b390-tiel-coder profile")
        for path in (("scalar",) if profile == "generic" else _PATHS):
            if 18 not in report["weight_types"][path]:
                raise IntelCapabilityError(f"The Intel {path} path lacks GGUF type18 required by Tiel")
    for path, types in required.items():
        missing = set(types) - set(report["weight_types"][path])
        if missing:
            raise IntelCapabilityError(f"The Intel {path} path lacks GGUF types {sorted(missing)}")


def supports_images(report: dict, model: str) -> bool:
    """Compiled model capability; encoder readiness is checked separately."""
    _check_report(report)
    if model not in VISION_MODELS or report["images"] is not True or model not in report.get("experimental_models", ()):
        return False
    if model in TARGET_ONLY_MODELS:
        return report.get("experimental_model_contracts", {}).get(model, {}).get("images") is True
    return True


def configured_images(cfg: dict) -> bool:
    """Cheap published configuration proof; launch revalidates native bytes."""
    from serve.intel_backend import canonical_model
    try:
        proof = cfg.get("intel_vision_contract")
        if not isinstance(proof, dict) or cfg.get("backend") != "sycl":
            return False
        base = cfg.get("cwd") or os.getcwd()
        executable = cfg.get("exe")
        if not isinstance(executable, str) or not executable:
            return False
        path = executable if os.path.isabs(executable) else os.path.join(base, executable)
        return (type(proof.get("version")) is int and proof["version"] == VISION_CONTRACT_VERSION and
                proof.get("protocol") == VISION_PROTOCOL and
                proof.get("model") == canonical_model(cfg) and
                _SHA256.fullmatch(str(proof.get("sha256", ""))) is not None and
                os.path.normcase(os.path.abspath(path)) == os.path.normcase(str(proof.get("executable", ""))))
    except (OSError, ValueError, KeyError, TypeError):
        return False


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
