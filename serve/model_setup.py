"""Shared launcher and API setup for supported, pinned GGUF models."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from serve import config

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_FILES = ("vocab.json", "merges.txt", "token_type.json", "tokenizer.json", "chat_template.jinja")


def model_architecture(path: str | Path) -> str:
    """Inspect only a local GGUF header; model weights are not read."""
    from serve.gguf_reader import GGUFFile
    from serve.tinyconfig import MODEL_NAMES
    arch = GGUFFile(Path(path)).metadata.get("general.architecture")
    if arch not in MODEL_NAMES:
        raise ValueError(f"Unsupported GGUF model architecture: {arch}")
    return arch


def shared_library_dirs(cfg: dict) -> dict:
    """Reuse an installed portable CUDA runtime for a separate LFM config."""
    if cfg.get("arch") not in ("lfm2moe", "gemma4") or "lib_dirs" in cfg:
        return cfg
    work = ROOT.with_name(ROOT.name + "-work")
    directories = []
    qwen = work / "runtime/.tinystrata/config.json"
    if qwen.is_file():
        try:
            saved = config.load(qwen)
        except (OSError, ValueError):
            saved = {}
        directories.extend(saved.get("lib_dirs") or [])
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


def resolve_executable(cfg: dict) -> Path | None:
    """Resolve an existing engine, also finding a build made after API startup."""
    if cfg.get("exe"):
        path = Path(cfg["exe"]).expanduser()
        path = path if path.is_absolute() else Path(cfg.get("cwd") or ".") / path
        return path.resolve() if path.is_file() else None
    suffix = ".exe" if os.name == "nt" else ""
    work = ROOT.with_name(ROOT.name + "-work")
    name = {"lfm2moe": "tinystrata-lfm2", "gemma4": "tinystrata-gemma4"}.get(cfg.get("arch"), "tinystrata")
    for folder in (work / "build", work / "build/Release", work / "build-cuda-132",
                   ROOT / "bin", ROOT / "build", ROOT / "build/Release"):
        candidate = folder / (name + suffix)
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_vision_executable(cfg: dict) -> Path | None:
    """The image encoder is built and installed alongside the engine."""
    selected = cfg.get("vision_exe") or (cfg.get("vision") or {}).get("exe")
    if selected:
        path = Path(selected).expanduser()
        path = path if path.is_absolute() else Path(cfg.get("cwd") or ".") / path
        return path.resolve() if path.is_file() else None
    engine = resolve_executable(cfg)
    if engine is not None:
        path = engine.with_name("tinystrata-vision" + (".exe" if os.name == "nt" else ""))
        if path.is_file():
            return path.resolve()
    return None


def configure_download(cfg: dict, config_path: str | Path) -> dict:
    """Choose the managed destination locally, never from an HTTP request."""
    from serve.model_download import manifest_for_arch

    args = cfg.get("args", [])
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise ValueError("args must be a list of engine argument strings")
    new = {**cfg, "args": list(args)}
    directory = Path(config_path).expanduser().resolve().parent / "model-files"
    new["download_dir"] = str(directory)
    if "--model" not in new["args"]:
        new["args"].extend(("--model", str(directory / manifest_for_arch(cfg.get("arch", "qwen35moe")).filename)))
    return new


def _extract_tokenizer(model: Path, cache: Path) -> None:
    # The same extractor is used for local and API-downloaded model weights.
    from serve.gguf_reader import GGUFFile
    from serve.tokenizer import extract

    md = GGUFFile(model).metadata
    if md.get("general.architecture") not in ("qwen35moe", "lfm2moe", "gemma4"):
        raise ValueError("This launcher supports qwen35moe, lfm2moe and gemma4 GGUF models")
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
    from serve.model_download import FIXED_GEMMA4_MMPROJ, FIXED_MMPROJ
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
        return disable_vision(new)
    artifact = FIXED_GEMMA4_MMPROJ if cfg.get("arch") == "gemma4" else FIXED_MMPROJ
    projector = Path(new.get("vision_mmproj") or Path(cfg["download_dir"]) / artifact.filename)
    encoder = resolve_vision_executable(new)
    valid = projector.is_file() and projector.stat().st_size == artifact.size
    if valid and not verified:
        digest = hashlib.sha256()
        with projector.open("rb") as source:
            for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(block)
        valid = digest.hexdigest() == artifact.sha256
    if valid and encoder is not None:
        new["vision"] = {**previous, "exe": str(encoder), "mmproj": str(projector.resolve()),
                         "model": str(model), "gpu": mode == "gpu"}
        if cfg.get("arch") == "gemma4":
            new["vision"].setdefault("max_tokens", 1120)
        if "vision_tokens" in cfg:
            new["vision"]["max_tokens"] = cfg["vision_tokens"]
        new["vision_policy_version"] = VISION_POLICY_VERSION
        new["args"].append("--vision")
        new.pop("vision_download_required", None)
    else:
        new.pop("vision", None)
        new["vision_download_required"] = True
    return new


def configure_assistant(cfg: dict, *, assistant: Path | None = None, verified: bool = False) -> dict:
    """Attach only the matching pinned QAT assistant to Gemma's native MTP path."""
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
    new["log"] = str(path.parent / {"lfm2moe": "engine-lfm2.log", "gemma4": "engine-gemma4.log"}.get(actual_arch, "engine.log"))
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
