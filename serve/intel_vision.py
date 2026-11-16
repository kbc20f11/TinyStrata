"""Prepare an Intel encoder on the language engine's actual Windows adapter.

Only startup calls this helper. Device inspection loads neither model weights
nor KV; the encoder then warms and stays resident before native KV admission.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
import subprocess


def _report(executable, arguments, cfg, env):
    result = subprocess.run([str(executable), *arguments], cwd=cfg.get("cwd") or None,
                            env=env, stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, encoding="utf-8", errors="strict", timeout=45,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode or len(result.stdout) > 65536:
        raise ValueError("The Intel image adapter/capability query failed")
    try:
        value = json.loads(result.stdout)
    except (ValueError, TypeError) as error:
        raise ValueError("The Intel image adapter query returned invalid metadata") from error
    if not isinstance(value, dict):
        raise ValueError("The Intel image adapter metadata must be an object")
    return value


def prepare_encoder(cfg, vision, env):
    """Return transient launch preferences, without saving a machine's LUID."""
    from serve.intel_backend import engine_args

    selected = deepcopy(vision)
    base = Path(cfg.get("cwd") or os.getcwd())
    for key in ("exe", "mmproj", "model"):
        value = selected.get(key)
        if isinstance(value, str) and not Path(value).is_absolute():
            selected[key] = str((base / value).resolve())
    report = _report(selected["exe"], ["--capabilities"], cfg, env)
    if not selected.get("gpu"):
        if report.get("backend") != "cpu" and report.get("cpu") is not True:
            raise ValueError("The selected Intel encoder is GPU-only; select vision_mode=gpu or an explicit CPU encoder")
        selected["backend"] = "cpu"
        return selected
    if (report.get("backend") != "sycl" or report.get("sycl") is not True or
            report.get("device_luid") is not True or report.get("warmup_before_ready") is not True):
        raise ValueError("Intel GPU images require the matching SYCL encoder with adapter identity and warmup support")
    args = engine_args(cfg)
    index = args[args.index("--device") + 1]
    executable = Path(cfg["exe"])
    if not executable.is_absolute():
        executable = base / executable
    device = _report(executable, ["--device-info", "--device", str(index)], cfg, env)
    luid = device.get("adapter_luid")
    if (device.get("schema") != 1 or device.get("backend") != "sycl" or
            device.get("vendor_id") != 0x8086 or device.get("device_index") != int(index) or
            not isinstance(luid, str) or re.fullmatch(r"[0-9a-fA-F]{8}:[0-9a-fA-F]{8}", luid) is None or
            type(device.get("adapter_node_mask")) is not int or device["adapter_node_mask"] <= 0):
        raise ValueError("The selected Intel GPU has no valid Windows adapter identity for the image encoder")
    selected.update(backend="sycl", device_luid=luid, elastic=False,
                    input_protocol=report.get("input_protocol", "enc"), language_device=device)
    return selected
