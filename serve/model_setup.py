"""Shared launcher and API setup for the fixed Qwen3.6 GGUF model."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from serve import config

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_FILES = ("vocab.json", "merges.txt", "token_type.json", "tokenizer.json", "chat_template.jinja")


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
    for folder in (work / "build", work / "build/Release", ROOT / "bin", ROOT / "build", ROOT / "build/Release"):
        candidate = folder / ("tinystrata" + suffix)
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_vision_executable(cfg: dict) -> Path | None:
    """The GPU image encoder is built and installed alongside the engine."""
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
    from serve.model_download import MODEL_FILENAME

    args = cfg.get("args", [])
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise ValueError("args must be a list of engine argument strings")
    new = {**cfg, "args": list(args)}
    directory = Path(config_path).expanduser().resolve().parent / "model-files"
    new["download_dir"] = str(directory)
    if "--model" not in new["args"]:
        new["args"].extend(("--model", str(directory / MODEL_FILENAME)))
    return new


def _extract_tokenizer(model: Path, cache: Path) -> None:
    # The same extractor is used for local and API-downloaded model weights.
    from serve.gguf_reader import GGUFFile
    from serve.tokenizer import extract

    md = GGUFFile(model).metadata
    if md.get("general.architecture") != "qwen35moe":
        raise ValueError("This launcher supports Qwen3.6 qwen35moe GGUF models only")
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
    """Enable resident GPU image encoding with the pinned, verified projector.

    The server starts and warms the encoder before the language engine, so its
    weights and image workspace are already included in the free-VRAM budget.
    """
    from serve.model_download import MMPROJ_FILENAME, MMPROJ_SHA256, MMPROJ_SIZE
    from serve.tinyconfig import VISION_POLICY_VERSION

    new = {**cfg, "args": list(cfg["args"])}
    previous = cfg.get("vision") or {}
    projector = projector or Path(previous.get("mmproj") or Path(cfg["download_dir"]) / MMPROJ_FILENAME)
    encoder = resolve_vision_executable(cfg)
    valid = projector.is_file() and projector.stat().st_size == MMPROJ_SIZE
    if valid and not verified:
        digest = hashlib.sha256()
        with projector.open("rb") as source:
            for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(block)
        valid = digest.hexdigest() == MMPROJ_SHA256
    new["args"] = [arg for arg in new["args"] if arg != "--vision"]
    if valid and encoder is not None:
        new["vision"] = {"exe": str(encoder), "mmproj": str(projector.resolve()),
                         "model": str(model), "gpu": True}
        new["vision_policy_version"] = VISION_POLICY_VERSION
        new["args"].append("--vision")
        new.pop("vision_download_required", None)
    else:
        new.pop("vision", None)
        new["vision_download_required"] = True
    return new


def prepare_installed_model(cfg: dict, config_path: str | Path, model_path: str | Path, *,
                            projector: Path | None = None, projector_verified: bool = False) -> dict:
    """Extract a verified/downloaded or local GGUF tokenizer, then save settings.

    A missing executable does not prevent downloading or preparing the model.
    The API can remain available until the engine has been built/configured.
    """
    path = Path(config_path).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"TinyStrata model not found: {model}")
    st = model.stat()
    identity = f"{model}\n{st.st_size}\n{st.st_mtime_ns}"
    cache = path.parent / "models" / hashlib.sha256(identity.encode()).hexdigest()[:16]
    tokenizer = cache / "tokenizer"
    if not all((tokenizer / name).is_file() for name in TOKENIZER_FILES):
        _extract_tokenizer(model, cache)
    if not all((tokenizer / name).is_file() for name in TOKENIZER_FILES):
        raise ValueError("The GGUF tokenizer extraction did not produce all required files")
    new = configure_download(cfg, path)
    args = new["args"]
    args[args.index("--model") + 1] = str(model)
    new["tokenizer"] = str(tokenizer)
    new["log"] = str(path.parent / "engine.log")
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
    save_settings(path, new)
    return new
