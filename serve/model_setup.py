"""Shared launcher and API setup for supported, pinned GGUF models."""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

from serve import config
from serve.model_download import reject_retired_model

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_FILES = ("vocab.json", "merges.txt", "token_type.json", "tokenizer.json", "chat_template.jinja")
NATIVE_ENGINE_NAMES = {"lfm2moe": "tinystrata-lfm2", "gemma4": "tinystrata-gemma4",
                       "inkling2_loop": "tinystrata-inkling2"}
_LEGACY_INKLING_IQ3 = ("Inkling-2-Mini-UD-IQ3_XXS.gguf", 11_663_309_504,
                       "bd7ef0865adf119b5183e45e741c629bafa649769b33b5ebf689bb4bcc4690d6")


def model_architecture(path: str | Path) -> str:
    """Inspect only a local GGUF header; model weights are not read."""
    from serve.gguf_reader import GGUFFile
    from serve.tinyconfig import MODEL_NAMES
    arch = GGUFFile(Path(path)).metadata.get("general.architecture")
    reject_retired_model(arch)
    if arch == "inkling2":
        arch = "inkling2_loop"
    if arch not in MODEL_NAMES:
        raise ValueError(f"Unsupported GGUF model architecture: {arch}")
    return arch


def shared_library_dirs(cfg: dict) -> dict:
    """Reuse a matching installed GPU runtime for another model config."""
    reject_retired_model(cfg.get("arch"), cfg.get("model_name"))
    if cfg.get("backend") == "sycl":
        # The staged executable owns its runtime; Intel must never borrow CUDA
        # or HIP directories from another model's saved configuration.
        directories = list(cfg.get("lib_dirs") or ())
        if cfg.get("exe"):
            executable = Path(cfg["exe"]).expanduser()
            if not executable.is_absolute():
                executable = Path(cfg.get("cwd") or ROOT) / executable
            directories.insert(0, str(executable.resolve().parent))
        return {**cfg, "lib_dirs": list(dict.fromkeys(directories))}
    if os.name == "nt" and cfg.get("backend") == "hip" and cfg.get("exe"):
        from serve.hip_runtime import resolve_hip_runtime
        executable = Path(cfg["exe"]).expanduser()
        if not executable.is_absolute():
            executable = Path(cfg.get("cwd") or ROOT) / executable
        runtime = resolve_hip_runtime(executable, cfg.get("lib_dirs") or ())
        if runtime is not None and runtime.managed:
            # The executable's installation generation owns its SDK selection.
            # An independently launched model can carry older saved lib_dirs.
            return {**cfg, "lib_dirs": list(dict.fromkeys((*runtime.directories,
                                                           *(cfg.get("lib_dirs") or ()))))}
    if (cfg.get("arch") not in ("lfm2moe", "gemma4", "clover_moe", "inkling2_loop") and
            cfg.get("model_name") != "tiel-coder-35b-a3b") or "lib_dirs" in cfg:
        return cfg
    work = ROOT.with_name(ROOT.name + "-work")
    directories = []
    qwen = work / "runtime/.tinystrata/config.json"
    if qwen.is_file():
        try:
            saved = config.load(qwen)
        except (OSError, ValueError):
            saved = {}
        if saved.get("backend", "cuda") == cfg.get("backend", "cuda"):
            directories.extend(saved.get("lib_dirs") or [])
    if cfg.get("backend") != "hip":
        for toolkit in sorted(work.glob("cuda-*"), reverse=True):
            directories.extend((str(toolkit / "bin"), str(toolkit / "bin/x64")))
    directories.append(str(work / "runtime"))
    found = list(dict.fromkeys(str(Path(item).resolve()) for item in directories if Path(item).is_dir()))
    return {**cfg, "lib_dirs": found} if found else cfg


def model_path(cfg: dict) -> Path | None:
    """Return the configured GGUF path, resolving engine-relative paths."""
    args = cfg.get("args", [])
    if not isinstance(args, list) or "--model" not in args[:-1]:
        return None
    path = Path(args[args.index("--model") + 1]).expanduser()
    return (path if path.is_absolute() else Path(cfg.get("cwd") or ".") / path).resolve()


def validate_intel_executable(cfg: dict, *, allow_legacy: bool = True):
    """Check executable build compatibility without loading a GPU or model.

    A missing executable stays in the existing bootstrap workflow; activation
    must still pass check_files. The returned proof binds metadata to actual
    executable bytes. GGUF identity and geometry remain separate checks.
    """
    if cfg.get("backend") != "sycl" or not cfg.get("exe"):
        return None
    executable = Path(cfg["exe"]).expanduser()
    if not executable.is_absolute():
        executable = Path(cfg.get("cwd") or ROOT) / executable
    if not executable.is_file():
        return None
    from serve.intel_backend import canonical_model, context_tokens, engine_args, MODEL_PROFILES, TIEL_MODEL, QWEN_ENGINE_MODELS
    from serve.intel_capabilities import probe_capabilities, validate_capabilities
    proof = probe_capabilities(cfg)
    model = canonical_model(cfg)
    required = None
    if not proof.legacy and model in QWEN_ENGINE_MODELS:
        required = {"scalar": [0, 8, 10, 11, 14, 21, 22, 23, 30]}
        if cfg.get("intel_profile", "generic") in MODEL_PROFILES.values():
            required.update(dp4a=[8, 10, 11, 14, 21, 22, 23], int8_xmx=[14, 21, 22])
        if model == TIEL_MODEL:
            for types in required.values():
                types.append(18)
    elif not proof.legacy:
        selected_path = model_path(cfg)
        if selected_path is not None and selected_path.is_file():
            from serve.gguf_reader import GGUFFile
            required = {"scalar": sorted({tensor.type_id for tensor in GGUFFile(selected_path).tensors})}
    requested_context = context_tokens(cfg)
    validate_capabilities(proof.report, model=model, profile=cfg.get("intel_profile", "generic"),
                          required_types=required, allow_legacy=allow_legacy,
                          requested_context="auto" if requested_context is None else requested_context,
                          require_images=cfg.get("vision_mode", "no") in ("gpu", "cpu"))
    launch = engine_args(cfg, allow_uninstalled=True)
    if "--prefill-mkl" in launch and not proof.report.get("build", {}).get("mkl") and not proof.legacy:
        raise ValueError("The selected Intel runtime does not compile the requested MKL path")
    kv = launch[launch.index("--kv-type") + 1]
    if kv not in proof.report["kv_types"]:
        raise ValueError("The selected Intel runtime does not support the requested KV format")
    return proof


def resolve_executable(cfg: dict) -> Path | None:
    """Resolve an existing engine, also finding a build made after API startup."""
    reject_retired_model(cfg.get("arch"), cfg.get("model_name"))
    from serve.backend import executable_backend
    def usable(path: Path) -> bool:
        if not path.is_file():
            return False
        native = executable_backend(path, cfg.get("lib_dirs") or []) if cfg.get("backend") else None
        return native is None or native == cfg["backend"]
    if cfg.get("exe"):
        path = Path(cfg["exe"]).expanduser()
        path = path if path.is_absolute() else Path(cfg.get("cwd") or ".") / path
        return path.resolve() if usable(path) else None
    suffix = ".exe" if os.name == "nt" else ""
    work = ROOT.with_name(ROOT.name + "-work")
    if cfg.get("backend") == "sycl":
        from serve.intel_backend import supports_model, canonical_model, QWEN_ENGINE_MODELS
        if not supports_model(cfg):
            return None
        if canonical_model(cfg) not in QWEN_ENGINE_MODELS:
            return None  # New families must select their own installed/runtime executable explicitly.
        for folder in (work / "build-intel", work / "build-intel/Release", ROOT / "bin"):
            candidate = folder / ("tinystrata-intel" + suffix)
            if usable(candidate):
                return candidate.resolve()
        return None
    name = NATIVE_ENGINE_NAMES.get(cfg.get("arch"), "tinystrata")
    backend_build = work / ("build-hip" if cfg.get("backend") == "hip" else "build")
    for folder in (backend_build, backend_build / "Release", work / "build-cuda-132" if cfg.get("backend") != "hip" else backend_build,
                   ROOT / "bin", ROOT / "build", ROOT / "build/Release"):
        candidate = folder / (name + suffix)
        if usable(candidate):
            return candidate.resolve()
    return None


def resolve_vision_executable(cfg: dict) -> Path | None:
    """The image encoder is built and installed alongside the engine."""
    selected = cfg.get("vision_exe") or (cfg.get("vision") or {}).get("exe")
    if selected:
        path = Path(selected).expanduser()
        path = path if path.is_absolute() else Path(cfg.get("cwd") or ".") / path
        if path.is_file():
            from serve.backend import executable_backend
            native = executable_backend(path, cfg.get("lib_dirs") or []) if cfg.get("backend") else None
            if native is None or native == cfg["backend"]:
                return path.resolve()
        return None
    engine = resolve_executable(cfg)
    if engine is not None:
        path = engine.with_name("tinystrata-vision" + (".exe" if os.name == "nt" else ""))
        if path.is_file():
            from serve.backend import executable_backend
            native = executable_backend(path, cfg.get("lib_dirs") or []) if cfg.get("backend") else None
            if native is None or native == cfg["backend"]:
                return path.resolve()
    return None


@lru_cache(maxsize=8)
def _artifact_digest(path: str, size: int, modified_ns: int, changed_ns: int) -> str:
    """Hash a legacy managed artifact once per file identity, without loading weights."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def migrate_managed_clover_model(cfg: dict, config_path: str | Path) -> dict:
    if cfg.get("arch") != "clover_moe":
        return cfg
    from serve.model_download import FIXED_CLOVER_MODEL, _LEGACY_CLOVER_MODEL

    legacy = _LEGACY_CLOVER_MODEL
    return _migrate_managed_artifact(cfg, config_path, (legacy.filename, legacy.size, legacy.sha256),
                                     FIXED_CLOVER_MODEL.filename)


def migrate_managed_model(cfg: dict, config_path: str | Path) -> dict:
    if cfg.get("arch") == "inkling2_loop":
        from serve.model_download import FIXED_INKLING_MODEL
        return _migrate_managed_artifact(cfg, config_path, _LEGACY_INKLING_IQ3, FIXED_INKLING_MODEL.filename)
    return migrate_managed_clover_model(cfg, config_path)


def _migrate_managed_artifact(cfg: dict, config_path: str | Path, legacy: tuple[str, int, str], filename: str) -> dict:
    """Advance only the former, verified managed default; retain every local selection.

    Older configs have no provenance. Infer it only from the exact managed path,
    size and checksum of the old pinned artifact. A missing old managed target
    can also advance; its incomplete download is left untouched. Explicit
    ``run.py --model`` selections carry model_source=local, even at that path.
    """
    if cfg.get("model_source") == "local":
        return cfg

    current = model_path(cfg)
    directory = Path(config_path).expanduser().resolve().parent / "model-files"
    old_name, old_size, old_sha = legacy
    if current is None or current != (directory / old_name).resolve():
        return cfg
    if current.exists():
        if not current.is_file():
            return cfg
        try:
            stat = current.stat()
            if stat.st_size != old_size or _artifact_digest(
                    str(current), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != old_sha:
                return cfg
            # Do not adopt a file modified during verification.
            after = current.stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                return cfg
        except OSError:
            return cfg
    new = {**cfg, "args": list(cfg["args"]), "model_source": "managed"}
    target = directory / filename
    new["args"][new["args"].index("--model") + 1] = str(target)
    new.pop("tokenizer", None)
    if not target.is_file():
        new["model_bootstrap"] = True
    return new


def configure_download(cfg: dict, config_path: str | Path) -> dict:
    """Choose the managed destination locally, never from an HTTP request."""
    reject_retired_model(cfg.get("arch"), cfg.get("model_name"))
    from serve.model_download import manifest_for_arch

    args = cfg.get("args", [])
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise ValueError("args must be a list of engine argument strings")
    cfg = migrate_managed_model(cfg, config_path)
    args = cfg["args"] if "args" in cfg else []
    new = {**cfg, "args": list(args)}
    directory = Path(config_path).expanduser().resolve().parent / "model-files"
    new["download_dir"] = str(directory)
    if "--model" not in new["args"]:
        artifact = manifest_for_arch(cfg.get("arch", "qwen35moe"), cfg.get("model_name"))
        new["args"].extend(("--model", str(directory / artifact.filename)))
        if cfg.get("arch") in ("clover_moe", "inkling2_loop"):
            new["model_source"] = "managed"
    return new


def _extract_tokenizer(model: Path, cache: Path) -> None:
    # The same extractor is used for local and API-downloaded model weights.
    from serve.gguf_reader import GGUFFile
    from serve.tokenizer import extract
    from serve.tinyconfig import MODEL_NAMES

    md = GGUFFile(model).metadata
    reject_retired_model(md.get("general.architecture"))
    if md.get("general.architecture") not in (*MODEL_NAMES, "inkling2"):
        raise ValueError(f"Unsupported GGUF model architecture: {md.get('general.architecture')}")
    if not md.get("tokenizer.chat_template"):
        raise ValueError("The GGUF does not contain its chat template")
    print("Extracting the model tokenizer and chat template ...", flush=True)
    extract(model, cache)


def save_settings(config_path: str | Path, cfg: dict) -> None:
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or config.load(path) != cfg:
        config.save(path, cfg)


def configure_vision(cfg: dict, model: Path, *, projector: Path | None = None,
                     verified: bool = False) -> dict:
    """Apply the selected image mode with the pinned, verified projector.

    A GPU encoder starts and warms before the language engine, so its weights
    and workspace are included in the free-VRAM budget. Off needs no assets.
    """
    from serve.model_download import projector_for_arch
    from serve.tinyconfig import VISION_POLICY_VERSION, check_vision_options, disable_vision

    mode = check_vision_options(cfg)
    new = {**cfg, "args": [arg for arg in cfg["args"] if arg != "--vision"], "vision_mode": mode}
    previous = cfg.get("vision") or {}
    if not isinstance(previous, dict):
        raise ValueError("vision must be an object")
    if previous.get("exe") and not new.get("vision_exe"):
        new["vision_exe"] = previous["exe"]
    if projector is not None:
        new["vision_mmproj"] = str(projector)
    elif previous.get("mmproj") and not new.get("vision_mmproj"):
        new["vision_mmproj"] = previous["mmproj"]
    if mode == "no":
        new = disable_vision(new)
    if cfg.get("backend") == "sycl":
        from serve.intel_capabilities import supports_images
        from serve.intel_models import VISION_PROTOCOL, VISION_CONTRACT_VERSION
        from serve.intel_backend import canonical_model
        proof = validate_intel_executable(new)
        if proof is not None and supports_images(proof.report, canonical_model(new)):
            new["intel_vision_contract"] = {"version": VISION_CONTRACT_VERSION,
                "protocol": VISION_PROTOCOL, "model": canonical_model(new),
                "executable": proof.executable, "sha256": proof.sha256}
        else:
            new.pop("intel_vision_contract", None)
    if mode == "no":
        return new
    if cfg.get("backend") == "sycl" and not new.get("intel_vision_contract"):
        new.pop("vision", None)
        new["vision_download_required"] = True
        return new
    artifact = projector_for_arch(cfg.get("arch", "qwen35moe"), cfg.get("model_name"))
    if artifact is None:
        raise ValueError("This model has no supported image projector")
    selected_projector = new.get("vision_mmproj")
    projector = Path(selected_projector or Path(cfg["download_dir"]) / artifact.filename)
    native_vision = cfg.get("arch") == "inkling2_loop"
    if (cfg.get("backend") == "sycl" or native_vision) and not selected_projector and not projector.is_file():
        # Local Intel models may live outside the managed download directory.
        # Reuse their matching projector after the same pinned SHA check.
        beside_model = Path(model).with_name(artifact.filename)
        if beside_model.is_file():
            projector = beside_model
            verified = False
    encoder = None if native_vision else resolve_vision_executable(new)
    valid = projector.is_file() and projector.stat().st_size == artifact.size
    if valid and not verified:
        digest = hashlib.sha256()
        with projector.open("rb") as source:
            for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(block)
        valid = digest.hexdigest() == artifact.sha256
    if native_vision:
        if valid:
            new["inkling_vision"] = new["vision_mmproj"] = str(projector.resolve())
            new.pop("vision_download_required", None)
        else:
            new.pop("inkling_vision", None)
            new["vision_download_required"] = True
        return new
    if valid and encoder is not None:
        if cfg.get("backend") == "sycl":
            new["vision_mmproj"] = str(projector.resolve())
        new["vision"] = {**previous, "exe": str(encoder), "mmproj": str(projector.resolve()),
                         "model": str(model), "gpu": mode == "gpu"}
        if cfg.get("backend") == "sycl":
            new["vision"].update(backend="sycl", elastic=False)
        if cfg.get("arch") == "gemma4":
            new["vision"].setdefault("max_tokens", 1120)
        if "vision_tokens" in cfg:
            new["vision"]["max_tokens"] = cfg["vision_tokens"]
        if cfg.get("arch") == "clover_moe":
            new["vision"].setdefault("max_tokens", 4096)
            new["vision"].setdefault("min_tokens", min(1024, new["vision"]["max_tokens"]))
            if "vision_tokens" in cfg:
                # An explicit image cap wins over a previously saved minimum.
                new["vision"]["min_tokens"] = min(new["vision"]["min_tokens"], cfg["vision_tokens"])
        new["vision_policy_version"] = VISION_POLICY_VERSION
        new["args"].append("--vision")
        new.pop("vision_download_required", None)
    else:
        new.pop("vision", None)
        new["vision_download_required"] = True
    return new


def configure_assistant(cfg: dict, *, assistant: Path | None = None, verified: bool = False) -> dict:
    """Attach only the matching pinned QAT assistant to Gemma's native MTP path."""
    if cfg.get("backend") == "sycl":
        if assistant is not None or cfg.get("draft_model") or "--draft-model" in cfg.get("args", []):
            raise ValueError("Intel target-only text does not support an assistant model")
        return cfg
    if cfg.get("arch") != "gemma4":
        return cfg
    from serve.model_download import FIXED_GEMMA4_ASSISTANT as artifact

    new = {**cfg, "args": list(cfg["args"])}
    args = new["args"]
    configured = args[args.index("--draft-model") + 1] if "--draft-model" in args[:-1] else cfg.get("draft_model")
    path = Path(assistant or configured or Path(cfg["download_dir"]) / artifact.filename).expanduser()
    if not path.is_absolute():
        path = Path(cfg.get("cwd") or ROOT) / path
    path = path.resolve()
    new["draft_model"] = str(path)
    if "--draft-model" in args[:-1]:
        args[args.index("--draft-model") + 1] = str(path)
    else:
        args.extend(("--draft-model", str(path)))
    enabled = config.value_of(new, "mtp") not in (None, 0)
    valid = path.is_file() and path.stat().st_size == artifact.size
    if enabled and valid and not verified:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(block)
        valid = digest.hexdigest() == artifact.sha256
    if enabled and not valid:
        new["assistant_download_required"] = True
    else:
        new.pop("assistant_download_required", None)
    return new


def prepare_installed_model(cfg: dict, config_path: str | Path, model_path: str | Path, *,
                            projector: Path | None = None, projector_verified: bool = False,
                            assistant: Path | None = None, assistant_verified: bool = False) -> dict:
    """Extract a verified/downloaded or local GGUF tokenizer, then save settings.

    A missing executable does not prevent downloading or preparing the model.
    The API can remain available until the engine has been built/configured.
    """
    path = Path(config_path).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"TinyStrata model not found: {model}")
    actual_arch = model_architecture(model)
    if actual_arch != cfg.get("arch", "qwen35moe"):
        raise ValueError("The GGUF architecture differs from this configuration; use --model-family with a separate config")
    if cfg.get("backend") == "sycl":
        from serve.gguf_reader import GGUFFile
        from serve.intel_backend import canonical_model, QWEN_MODEL, TIEL_MODEL, TARGET_ONLY_MODELS
        from serve.model_download import TIEL_MODEL_FILENAME, manifest_for_arch
        metadata = GGUFFile(model).metadata
        selected = canonical_model(cfg)
        name = str(metadata.get("general.name", ""))
        if (selected == QWEN_MODEL and "qwen3.6" not in name.lower() or
                selected == TIEL_MODEL and (name != "Ornith-1.5-35B" or model.name != TIEL_MODEL_FILENAME)):
            raise ValueError("The Intel GGUF identity does not match the selected pinned Qwen3.6 or Tiel artifact")
        if selected in TARGET_ONLY_MODELS:
            artifact = manifest_for_arch(actual_arch, selected)
            if model.name != artifact.filename or model.stat().st_size != artifact.size:
                raise ValueError("The Intel GGUF does not match the selected pinned family artifact")
        validate_intel_executable(cfg)
    st = model.stat()
    identity = f"{model}\n{st.st_size}\n{st.st_mtime_ns}"
    cache = path.parent / "models" / hashlib.sha256(identity.encode()).hexdigest()[:16]
    tokenizer = cache / "tokenizer"
    required = TOKENIZER_FILES + (("merges.json",) if actual_arch == "gemma4" else ())
    if not all((tokenizer / name).is_file() for name in required):
        _extract_tokenizer(model, cache)
    if not all((tokenizer / name).is_file() for name in required):
        raise ValueError("The GGUF tokenizer extraction did not produce all required files")
    new = shared_library_dirs(configure_download(cfg, path))
    args = new["args"]
    args[args.index("--model") + 1] = str(model)
    new["tokenizer"] = str(tokenizer)
    from serve.intel_models import MODEL_LOGS
    log_name = (MODEL_LOGS[selected] if cfg.get("backend") == "sycl" else "engine-tiel-coder.log" if cfg.get("model_name") == "tiel-coder-35b-a3b" else
                {"lfm2moe": "engine-lfm2.log", "gemma4": "engine-gemma4.log",
                 "clover_moe": "engine-clover.log", "inkling2_loop": "engine-inkling2.log"}.get(actual_arch, "engine.log"))
    new["log"] = str(path.parent / log_name)
    exe = resolve_executable(new)
    if exe is not None:
        new["exe"] = str(exe)
        # A downloaded model is not ready until the native engine reaches READY.
        # Keep the control API available across restarts after a load failure.
        if cfg.get("model_bootstrap") is not True:
            new.pop("model_bootstrap", None)
    else:
        new["model_bootstrap"] = True
    new = configure_vision(new, model, projector=projector, verified=projector_verified)
    new = configure_assistant(new, assistant=assistant, verified=assistant_verified)
    save_settings(path, new)
    return new
