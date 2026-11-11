"""UTF-8 diagnostics for TinyStrata's executable Python entrypoints.

Windows redirected Python streams otherwise inherit an ANSI code page, while
the launcher and log readers use UTF-8. Unicode model/config paths must remain
printable even when they contain characters outside that code page. Importing
this module does not alter streams; only an executable main calls the helper.
The independently installable UI keeps its stdlib-only copy in ui/stdio.py so
it also works when added to an older core installation.
"""
from __future__ import annotations

import sys


def configure_utf8_stdio(streams=None) -> None:
    """Keep valid Unicode intact; represent lone surrogates in diagnostics.

    stdin/native request pipes have their own explicit encodings. Embedded
    callers may replace stdout/stderr with None, StringIO or a closed stream;
    those must not prevent model/configuration work from starting.
    """
    for stream in (sys.stdout, sys.stderr) if streams is None else streams:
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (OSError, ValueError):
            # A closed or disconnected diagnostic stream is not a path error.
            pass
