"""Read-only startup gate for a managed Windows installer transaction.

The installer owns the marker and any repair/removal. Launchers only inspect it;
neither a stale marker nor an unreadable owner is permission to start an engine.
"""
from __future__ import annotations

import hmac
import json
import math
import os
from pathlib import Path

GUARD_NAME = ".tinystrata-installing.json"
TRANSACTION_ENV = "TINYSTRATA_INSTALLER_TRANSACTION"
_MAX_BYTES = 65536


class UpdateInProgressError(ValueError):
    """A launch would overlap an active or incomplete installation update."""


def _installation_work(config_path=None, work_root=None):
    if work_root is not None:
        return Path(work_root).expanduser().resolve()
    if config_path is None:
        return None
    path = Path(config_path).expanduser().absolute()
    # Private benchmark/custom configs deliberately have no inferred installation.
    if (not path.name.lower().startswith("config") or path.suffix.lower() != ".json" or
            path.parent.name.lower() != ".tinystrata" or path.parent.parent.name.lower() != "runtime"):
        return None
    return path.parent.parent.parent.resolve()


def _owner_state(pid, started_unix):
    """Return alive/dead/unknown without loading any GPU runtime."""
    try:
        import psutil
    except ImportError:
        return "unknown"
    try:
        process = psutil.Process(pid)
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return "dead"
        observed = process.create_time()
        # Both Windows Process.StartTime and psutil use the creation FILETIME.
        # A small conversion tolerance does not confuse a reused PID's new owner.
        return "alive" if abs(observed - started_unix) <= .01 else "dead"
    except psutil.NoSuchProcess:
        return "dead"
    except (psutil.Error, OSError, ValueError):
        return "unknown"


def assert_installation_ready(config_path=None, work_root=None, allow_installer=False):
    """Reject active/incomplete updates, except their authorized configure-only child.

    Callers may set allow_installer only for a configure-only invocation. The
    matching environment token is insufficient unless its marker owner is alive
    with the recorded process creation time. No marker is ever removed here.
    """
    work = _installation_work(config_path, work_root)
    if work is None:
        return
    marker = work / GUARD_NAME
    try:
        with marker.open("rb") as stream:
            raw = stream.read(_MAX_BYTES + 1)
    except FileNotFoundError:
        return
    except OSError as error:
        raise UpdateInProgressError(
            "TinyStrata's update status cannot be read. Run the installer again to repair this installation; "
            "the update marker was left unchanged.") from error
    try:
        if len(raw) > _MAX_BYTES:
            raise ValueError("oversized marker")
        value = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(value, dict) or type(value.get("schema")) is not int or value["schema"] != 1:
            raise ValueError("unsupported marker")
        pid, started, token = value.get("pid"), value.get("started_unix"), value.get("token")
        if type(pid) is not int or not 0 < pid < (1 << 32):
            raise ValueError("invalid owner")
        if type(started) not in (int, float) or not math.isfinite(started) or started <= 0:
            raise ValueError("invalid creation time")
        if not isinstance(token, str) or not 1 <= len(token) <= 256 or any(ord(c) <= 32 for c in token):
            raise ValueError("invalid token")
    except (ValueError, UnicodeError, OverflowError) as error:
        raise UpdateInProgressError(
            "TinyStrata's update marker is incomplete or invalid. Wait for an active installer, or run the installer "
            "again to repair an interrupted update; the marker was left unchanged.") from error

    owner = _owner_state(pid, started)
    if owner == "alive":
        inherited = os.environ.get(TRANSACTION_ENV, "")
        if allow_installer and hmac.compare_digest(token.encode("utf-8"), inherited.encode("utf-8")):
            return
        raise UpdateInProgressError("TinyStrata is being updated. Wait for the installer to finish before starting it.")
    if owner == "dead":
        raise UpdateInProgressError(
            "A TinyStrata update was interrupted. Run the installer again to repair this installation before starting it; "
            "the update marker was left unchanged.")
    raise UpdateInProgressError(
        "TinyStrata cannot verify the owner of its update marker. Wait for an active installer, or run the installer "
        "again to repair this installation; the marker was left unchanged.")
