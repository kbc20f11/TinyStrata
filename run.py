"""Launch TinyStrata's API server; local state stays in the sibling work directory."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT.with_name(ROOT.name + "-work") / "runtime"


def set_arg(cfg: dict, flag: str, value) -> None:
    args = cfg.setdefault("args", [])
    if flag in args:
        args[args.index(flag) + 1] = str(value)
    else:
        args.extend((flag, str(value)))


def expert_budget(value: str):
    if value.lower() == "auto":
        return "auto"
    try:
        return float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("expected auto or an expert cache limit in GiB") from e


def context_limit(value: str):
    if value.lower() == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("expected auto or a whole-number context limit") from e


def prepare(a) -> tuple[dict, Path]:
    from serve import config
    from serve.tinyconfig import MEMORY_POLICY_VERSION, automatic_experts, check_files, migrate_settings, validate
    from serve.model_setup import configure_download, model_path, prepare_installed_model, resolve_executable, save_settings

    path = a.config.expanduser().resolve()
    cfg = config.load(path) if path.exists() else {
        "arch": "qwen35moe", "model_name": "qwen3.6-35b-a3b", "cwd": str(ROOT),
        "args": ["--max-context", "auto", "--kv-type", "auto", "--mtp", "3",
                 "--chunk", "512"],
        "memory_policy_version": MEMORY_POLICY_VERSION,
        "host": "127.0.0.1",
    }
    cfg, migrated = migrate_settings(cfg)
    cfg.setdefault("port", 18080)
    if a.model:
        set_arg(cfg, "--model", a.model.expanduser().resolve())
        # An explicit local selection keeps the existing eager-launch behavior.
        cfg.pop("model_bootstrap", None)
    if a.exe:
        cfg["exe"] = str(a.exe.expanduser().resolve())
    elif not cfg.get("exe"):
        candidate = resolve_executable(cfg)
        if candidate is not None:
            cfg["exe"] = str(candidate)
    if getattr(a, "vision_exe", None):
        cfg["vision_exe"] = str(a.vision_exe.expanduser().resolve())
    for key in ("max_context", "kv_type", "mtp", "expert_vram", "vram_reserve_mib", "chunk"):
        value = getattr(a, key)
        if value is not None:
            if key == "expert_vram" and value == "auto":
                cfg = automatic_experts(cfg)
            else:
                cfg, _ = config.apply(cfg, {key: value})
    if a.lib_dir:
        cfg["lib_dirs"] = [str(p.expanduser().resolve()) for p in a.lib_dir]
    if a.port is not None:
        cfg["port"] = a.port
    if a.host is not None:
        cfg["host"] = a.host
    if a.api_key is not None:
        cfg["api_key"] = a.api_key
    if a.gpu is not None:
        cfg["gpu"] = a.gpu
    cfg = configure_download(cfg, path)
    validate(cfg, allow_uninstalled=True)
    check_files(cfg, allow_uninstalled=True)
    model = model_path(cfg)
    if a.model and not model.is_file():
        raise ValueError(f"TinyStrata model not found: {model}")
    cfg["log"] = str(path.parent / "engine.log")
    if model.is_file():
        cfg = prepare_installed_model(cfg, path, model)
    else:
        cfg["model_bootstrap"] = True
        cfg.pop("tokenizer", None)
        save_settings(path, cfg)
    if migrated:
        print("Migrated legacy settings to the API server configuration.", flush=True)
    return cfg, path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, help="existing Qwen3.6 GGUF (optional; download the fixed model through the API)")
    ap.add_argument("--exe", type=Path, help="built tinystrata executable")
    ap.add_argument("--vision-exe", type=Path, help="built tinystrata-vision image encoder")
    ap.add_argument("--config", type=Path, default=RUNTIME / ".tinystrata/config.json")
    ap.add_argument("--max-context", type=context_limit, metavar="auto|TOKENS",
                    help="choose context capacity from available GPU memory (default: auto), or set a token limit")
    ap.add_argument("--kv-type", choices=("auto", "fp32", "fp8", "fp8-mma", "int8"),
                    help="choose native FP8 Attention when supported, otherwise INT8 (default: auto)")
    ap.add_argument("--mtp", type=int)
    ap.add_argument("--expert-vram", type=expert_budget, metavar="auto|GiB",
                    help="fit experts to available VRAM (default: auto), or set an advanced GiB limit; 0 = CPU experts")
    ap.add_argument("--vram-reserve-mib", type=int,
                    help="free GPU memory to leave unused when fitting experts automatically (engine default: 768 MiB)")
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--gpu", type=int, help="GPU index")
    ap.add_argument("--lib-dir", type=Path, action="append", help="CUDA runtime DLL directory (repeatable)")
    ap.add_argument("--port", type=int, help="API port (default: saved setting or 18080)")
    ap.add_argument("--host", help="API bind address (default: 127.0.0.1)")
    ap.add_argument("--api-key", help="API key required by clients (default: unset)")
    ap.add_argument("--configure-only", action="store_true")
    a = ap.parse_args()
    missing = [m for m in ("jinja2", "regex", "psutil") if importlib.util.find_spec(m) is None]
    if missing:
        ap.error("Missing API dependencies: " + ", ".join(missing) +
                 f'. Install with: "{sys.executable}" -m pip install -r "{ROOT / "requirements.txt"}"')
    try:
        cfg, path = prepare(a)
    except (ValueError, OSError, KeyError) as e:
        ap.error(str(e))
    print(f"TinyStrata config: {path}", flush=True)
    if a.configure_only:
        return 0
    command = [sys.executable, "-B", "-m", "serve.server", "--engine", "tinystrata", "--config", str(path),
               "--port", str(cfg.get("port", 18080))]
    try:
        return subprocess.call(command, cwd=ROOT)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
