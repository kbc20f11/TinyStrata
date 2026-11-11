"""UTF-8 diagnostics for the independently installable optional UI.

This small stdlib-only helper intentionally also exists in serve/stdio.py.
UI-only installation copies ui/** onto older API installations, so its startup
must not require a newly added core API module. Importing does not alter streams.
"""
from __future__ import annotations

import sys


def configure_utf8_stdio(streams=None) -> None:
    """Keep valid Unicode intact and escape lone surrogates in diagnostics."""
    for stream in (sys.stdout, sys.stderr) if streams is None else streams:
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (OSError, ValueError):
            pass  # Closed/disconnected diagnostic streams are not path errors.
