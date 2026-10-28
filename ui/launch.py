"""Start TinyStrata's optional UI, reusing the current local API and model."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import webbrowser

APP_ID = "tinystrata-ui"
PROFILES = {"qwen36": ("qwen3.6-35b-a3b", "config.json"),
            "gemma4": ("gemma-4-26b-a4b-it", "config-gemma4.json"),
            "lfm2": ("lfm2.5-24b-a2b", "config-lfm2.json")}
MODELS = {model: family for family, (model, _) in PROFILES.items()}


def port_number(value):
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("expected a port from 1 to 65535")
    return port


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def port_occupied(port):
    # Exclusive bind also finds a Windows listener which is still starting.
    with socket.socket() as probe:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind(("127.0.0.1", port))
            return False
        except OSError as error:
            if error.errno in (13, 48, 98, 10013, 10048) or getattr(error, "winerror", None) in (10013, 10048):
                return True
            raise RuntimeError(f"Cannot inspect local port {port}: {error}") from error


def request_json(port, path, *, method="GET", timeout=2, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        body = b"{}" if method == "POST" else None
        request_headers = dict(headers or {})
        if body is not None:
            request_headers["Content-Type"] = "application/json"
        connection.request(method, path, body, request_headers)
        response = connection.getresponse()
        data = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(f"Local port {port} returned HTTP {response.status}: {data}")
        return data
    finally:
        connection.close()


def api_identity(health):
    return (isinstance(health, dict) and health.get("status") == "ok" and
            health.get("service") in ("strata", "tinystrata") and health.get("model") in MODELS)


def process_started(pid):
    """Return Windows' exact creation timestamp in .NET's UTC round-trip form."""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
    try:
        if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                      ctypes.byref(system), ctypes.byref(user)):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)
    ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime
    seconds, fraction = divmod(ticks, 10_000_000)
    utc = datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)
    return utc.strftime("%Y-%m-%dT%H:%M:%S") + f".{fraction:07d}Z"


class Launcher:
    def __init__(self, args, source, work):
        self.source, self.work = source, work
        self.runtime = work / "runtime"
        self.state_dir = self.runtime / "ui"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        settings_path = self.state_dir / "settings.json"
        try:
            settings = read_json(settings_path) if settings_path.is_file() else {}
        except (OSError, ValueError):
            settings = {}
        if not isinstance(settings, dict):
            settings = {}
        self.api_port = port_number(args.api_port if args.api_port is not None else settings.get("api_port", 18080))
        self.port = port_number(args.port if args.port is not None else settings.get("port", 3000))
        self.family = args.model_family or settings.get("model_family", "qwen36")
        if self.family not in PROFILES:
            raise ValueError("Unknown saved model family; specify --model-family qwen36, gemma4 or lfm2")
        if self.api_port == self.port:
            raise ValueError("The UI and API must use different ports")
        self.python = self.runtime / ".venv" / "Scripts" / "python.exe"
        self.url = f"http://127.0.0.1:{self.port}"

    def log(self, message):
        with (self.state_dir / "launch.log").open("a", encoding="utf-8") as output:
            output.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")

    def write_state(self, name, values):
        target = self.state_dir / name
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)

    def save_settings(self):
        self.write_state("settings.json", {"source_dir": str(self.source), "work_dir": str(self.work),
                                         "api_port": self.api_port, "port": self.port,
                                         "model_family": self.family})
        # A small pointer in the normal sibling work directory lets run-ui.bat
        # find a custom installation's private Python without changing Git files.
        location = self.source.with_name(self.source.name + "-work") / "runtime" / "ui" / "location.json"
        location.parent.mkdir(parents=True, exist_ok=True)
        temporary = location.with_name(f"{location.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps({"source_dir": str(self.source), "work_dir": str(self.work)},
                                        indent=2), encoding="utf-8")
        temporary.replace(location)

    def spawn(self, script, arguments, label):
        if not self.python.is_file():
            raise RuntimeError(f"Installed Python was not found: {self.python}. Run the TinyStrata installer first")
        if script is not None and not script.is_file():
            raise RuntimeError(f"Required file was not found: {script}. Update TinyStrata with the UI installer")
        # Console Python retains redirected stdout/stderr. CREATE_NO_WINDOW
        # hides both API and UI helpers without pythonw's missing-stream errors.
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        command = [str(self.python), "-B", *([str(script)] if script is not None else []), *arguments]
        with (self.state_dir / f"{label}.stdout.log").open("ab") as output, \
             (self.state_dir / f"{label}.stderr.log").open("ab") as errors:
            process = subprocess.Popen(command, cwd=self.source, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=errors, creationflags=flags)
        self.log(f"Started {label}: PID {process.pid}")
        self.write_state(f"{label}-process.json", {"pid": process.pid, "command": command})
        return process

    def prepare_api(self, config):
        # Keep run.py's configuration migration/bootstrap, but run the actual
        # server directly so its PID is discoverable by a later installer update.
        arguments = ["--config", str(config), "--model-family", self.family,
                     "--port", str(self.api_port), "--host", "127.0.0.1", "--configure-only"]
        process = self.spawn(self.source / "run.py", arguments, "api-prepare")
        if process.wait() != 0:
            raise RuntimeError(f"API configuration failed; see {self.state_dir / 'api-prepare.stderr.log'}")

    def save_api_process(self, process, config):
        name = {"qwen36": "server-state.json", "gemma4": "server-state-gemma4.json",
                "lfm2": "server-state-lfm2.json"}[self.family]
        target = self.runtime / name
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        values = {"pid": process.pid, "started": process_started(process.pid),
                  "config": str(config), "source": str(self.source), "port": self.api_port}
        temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
        temporary.replace(target)

    def ensure_api(self, no_start=False):
        process = None
        deadline = time.monotonic() + 300
        expected_model, config_name = PROFILES[self.family]
        if port_occupied(self.api_port):
            try:
                health = request_json(self.api_port, "/health")
            except Exception as error:
                raise RuntimeError(f"API port {self.api_port} is occupied or unresponsive; it was left unchanged") from error
            if not api_identity(health):
                raise RuntimeError(f"API port {self.api_port} belongs to another application; it was left unchanged")
            expected_model = health["model"]
            self.family = MODELS[expected_model]
            self.log(f"Reusing the API and selected model {expected_model}; no process was restarted")
            # An unloaded API is intentional for lazy mode or an installation
            # without downloaded weights. The UI leaves loading to the API's
            # usual authenticated requests; it never reads keys or POSTs /load.
            return
        else:
            if no_start:
                raise RuntimeError(f"No TinyStrata API is running on port {self.api_port}; --no-api-start leaves it unchanged")
            config = self.runtime / ".tinystrata" / config_name
            if not config.is_file():
                raise RuntimeError(f"Installed model configuration was not found: {config}. Run the installer first")
            self.prepare_api(config)
            process = self.spawn(None, ["-m", "serve.server", "--engine", "tinystrata", "--config", str(config),
                                 "--port", str(self.api_port), "--host", "127.0.0.1"], "api")
            self.save_api_process(process, config)
        while time.monotonic() < deadline:
            try:
                health = request_json(self.api_port, "/health")
                if not api_identity(health):
                    raise RuntimeError(f"API port {self.api_port} now belongs to another application; it was left unchanged")
                if health["model"] != expected_model:
                    expected_model = health["model"]
                    self.log(f"Following the API's model selection: {expected_model}")
                # Eager loading completes before HTTP startup; lazy/bootstrap
                # servers are ready to use while health.loaded remains false.
                self.family = MODELS[expected_model]
                return
            except (OSError, ValueError, http.client.HTTPException):
                pass
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"TinyStrata exited before becoming ready; see {self.state_dir / 'api.stderr.log'}")
            time.sleep(.4)
        raise RuntimeError(f"TinyStrata did not become ready within 300 seconds; see {self.state_dir / 'api.stderr.log'}")

    def existing_ui(self):
        if not port_occupied(self.port):
            return None
        try:
            health = request_json(self.port, "/local-ui-health")
        except Exception as error:
            raise RuntimeError(f"UI port {self.port} belongs to another application; it was left unchanged") from error
        if not isinstance(health, dict) or health.get("app") != APP_ID:
            raise RuntimeError(f"UI port {self.port} belongs to another application; it was left unchanged")
        if health.get("api_port") != self.api_port:
            raise RuntimeError(f"The UI on port {self.port} uses a different API port; choose another --port")
        return health

    def ensure_ui(self):
        health = self.existing_ui()
        if health is not None:
            self.log(f"Reusing the UI server: PID {health.get('pid')}")
            return
        process = self.spawn(self.source / "ui" / "server.py", ["--port", str(self.port),
                             "--api-port", str(self.api_port)], "ui")
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                health = self.existing_ui()
                if health is not None:
                    return
            except (OSError, ValueError, http.client.HTTPException, RuntimeError):
                # A newly spawned listener may have bound before serving HTTP.
                if process.poll() is not None:
                    break
            if process.poll() is not None:
                break
            time.sleep(.2)
        raise RuntimeError(f"The UI server did not become ready; see {self.state_dir / 'ui.stderr.log'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--work-dir", type=Path, help="state directory (default: source directory's sibling ending in -work)")
    parser.add_argument("--api-port", type=port_number, help="API port (default: saved selection or 18080)")
    parser.add_argument("--port", type=port_number, help="UI port (default: saved selection or 3000)")
    parser.add_argument("--model-family", choices=tuple(PROFILES), help="model for a new API (default: saved selection or qwen36)")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    parser.add_argument("--no-api-start", action="store_true", help="require an existing healthy API; never start it or force model loading")
    parser.add_argument("--configure-only", action="store_true", help="save selections without starting services or opening the browser")
    args = parser.parse_args()
    launcher = None
    try:
        source = args.source_dir.expanduser().resolve()
        default_work = source.with_name(source.name + "-work")
        selected_work = args.work_dir
        if selected_work is None:
            location = default_work / "runtime" / "ui" / "location.json"
            try:
                pointer = read_json(location)
                if Path(pointer["source_dir"]).resolve() == source:
                    selected_work = Path(pointer["work_dir"])
            except (OSError, ValueError, KeyError, TypeError):
                pass
        work = (selected_work or default_work).expanduser().resolve()
        if work.is_relative_to(source):
            raise ValueError("--work-dir must be outside the source directory so local state stays out of Git")
        if not (source / "ui" / "server.py").is_file() or not (source / "ui" / "web" / "index.html").is_file():
            raise ValueError(f"The optional UI was not found in {source}; update using the UI installer")
        launcher = Launcher(args, source, work)
        if args.configure_only:
            launcher.save_settings()
            print(f"TinyStrata UI settings saved: {launcher.state_dir / 'settings.json'}")
            return 0
        launcher.existing_ui()  # Refuse a foreign UI port before starting a model.
        launcher.ensure_api(args.no_api_start)
        launcher.ensure_ui()
        launcher.save_settings()
        message = f"TinyStrata ready: {launcher.url} (API http://127.0.0.1:{launcher.api_port}/v1)"
        launcher.log(message)
        print(message, flush=True)
        if not args.no_browser:
            webbrowser.open(launcher.url)
        return 0
    except Exception as error:
        message = f"TinyStrata UI: {error}"
        if launcher is not None:
            launcher.log(message)
            message += f"\nLog: {launcher.state_dir / 'launch.log'}"
        print(message, file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
