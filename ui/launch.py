"""Start TinyStrata's optional UI, reusing the current local API and model."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
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
            "lfm2": ("lfm2.5-24b-a2b", "config-lfm2.json"),
            "clover": ("clover-35b-a3b-preview", "config-clover.json"),
            "tiel": ("tiel-coder-35b-a3b", "config-tiel-coder.json")}
MODELS = {model: family for family, (model, _) in PROFILES.items()}
FAMILY_ALIASES = {"tiel-coder": "tiel"}


def retired_model(name):
    # The optional UI also supports an older installed core. Keep this check
    # local rather than requiring a newly added serve module at UI startup.
    return isinstance(name, str) and name.strip().lower() in ("gpt-oss", "gptoss", "gpt-oss-20b")


def reject_retired_model(name):
    if retired_model(name):
        raise ValueError("GPT-OSS is no longer supported by TinyStrata. Choose another installed model. "
                         "Existing model files and chats have been kept.")


def profile_path(runtime, family, backend=None):
    """A retired primary config stays intact; Qwen uses its separate profile."""
    parent = Path(runtime) / ".tinystrata"
    if backend == "sycl":
        if family != "qwen36":
            raise ValueError("Experimental Intel supports Qwen3.6 text only")
        return parent / "config-sycl.json"
    path = parent / PROFILES[family][1]
    if family == "qwen36" and path.is_file():
        try:
            cfg = read_json(path)
        except (OSError, ValueError):
            cfg = {}
        if isinstance(cfg, dict) and cfg.get("arch") == "gpt-oss":
            return parent / "config-qwen36.json"
    return path


def available_default_family(runtime):
    """Prefer an already downloaded supported model; never download during migration."""
    arches = {"qwen36": "qwen35moe", "gemma4": "gemma4", "lfm2": "lfm2moe",
              "clover": "clover_moe", "tiel": "qwen35moe"}
    for family, (model, _) in PROFILES.items():
        path = profile_path(runtime, family)
        try:
            cfg = read_json(path)
            if not isinstance(cfg, dict) or cfg.get("arch", "qwen35moe") != arches[family]:
                continue
            reject_retired_model(cfg.get("arch"))
            reject_retired_model(cfg.get("model_name"))
            if cfg.get("model_name", model) != model:
                continue
            args = cfg.get("args", [])
            if not isinstance(args, list) or "--model" not in args[:-1]:
                continue
            selected = Path(args[args.index("--model") + 1]).expanduser()
            if not selected.is_absolute():
                selected = Path(cfg.get("cwd") or ".") / selected
            if selected.is_file():
                return family
        except (OSError, ValueError, TypeError):
            continue
    return "qwen36"


def backend_revision(source):
    """Identify Python code loaded by a UI, independently of changing web assets."""
    digest = hashlib.sha256()
    for name in ("server.py", "launch.py", "stdio.py", "chat_history.py", "web_search.py", "tool_settings.py", "sandbox_runtime.py"):
        digest.update(name.encode("ascii") + b"\0")
        digest.update((Path(source) / "ui" / name).read_bytes())
    return digest.hexdigest()


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


def windows_ui_metadata(pid, port):
    script = r"""$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$request = [Console]::In.ReadToEnd() | ConvertFrom-Json
$item = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + [int]$request.pid)
$owners = @(Get-NetTCPConnection -State Listen -LocalPort ([int]$request.port) -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique)
@{ executable = $item.ExecutablePath; command = $item.CommandLine; parent = $item.ParentProcessId; owners = $owners } | ConvertTo-Json -Compress
"""
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                            input=json.dumps({"pid": pid, "port": port}), text=True, encoding="utf-8",
                            capture_output=True, timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return json.loads(result.stdout) if result.returncode == 0 else {}


def windows_command_arguments(command):
    """Parse the actual Win32 command line, including redirector re-quoting."""
    import ctypes
    from ctypes import wintypes

    shell = ctypes.WinDLL("shell32", use_last_error=True)
    shell.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    shell.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    count = ctypes.c_int()
    values = shell.CommandLineToArgvW(command, ctypes.byref(count))
    if not values:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return [values[index] for index in range(count.value)]
    finally:
        kernel.LocalFree(values)


def matching_ui_process(metadata, pid, command, launcher_pid=None):
    if not isinstance(metadata, dict) or not isinstance(metadata.get("command"), str) or not command:
        return False
    try:
        # Windows' venv python.exe can be a redirector: the actual listener is
        # its direct child running the base interpreter with the same command.
        executable = metadata.get("executable", "")
        executables = [command[0], getattr(sys, "_base_executable", sys.executable)]
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
            kernel.GetModuleFileNameW.restype = wintypes.DWORD
            buffer = ctypes.create_unicode_buffer(32768)
            length = kernel.GetModuleFileNameW(None, buffer, len(buffer))
            if 0 < length < len(buffer):
                # App Execution Aliases can be distinct redirector files; the
                # loaded interpreter image is the reliable file identity.
                executables.append(buffer.value)
        executable_matches = any(Path(candidate).is_file() and os.path.samefile(executable, candidate)
                                 for candidate in executables)
        if os.name == "nt":
            actual = windows_command_arguments(metadata.get("command", ""))
            # Standard CPython venvs can replace argv[0] with the base Python
            # image and add quotes. All script/port/data arguments must still
            # match exactly; only the independently verified interpreter varies.
            command_matches = (len(actual) == len(command) and actual[1:] == command[1:] and
                               any(Path(candidate).is_file() and os.path.samefile(actual[0], candidate)
                                   for candidate in executables))
        else:
            command_matches = metadata.get("command") == subprocess.list2cmdline(command)
        return (executable_matches and command_matches and
                metadata.get("owners") == [pid] and
                (launcher_pid is None or pid == launcher_pid or metadata.get("parent") == launcher_pid))
    except (OSError, TypeError, ValueError):
        return False


def stop_owned_ui(pid, command, started, port, launcher_pid=None):
    """Stop an exact, independently verified Windows UI process, never a PID alone."""
    if os.name != "nt":
        raise RuntimeError("Close the previous UI server before updating it on this platform")
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.TerminateProcess.restype = wintypes.BOOL
    # Holding the handle ensures that a PID reused during inspection cannot be
    # terminated in place of the inspected UI. No process tree is terminated.
    handle = kernel.OpenProcess(0x1000 | 0x0001 | 0x00100000, False, pid)
    if not handle:
        raise RuntimeError("The previous UI process could not be verified; it was left unchanged")
    try:
        if kernel.WaitForSingleObject(handle, 0) != 258:
            return
        observed_started = process_started(pid)
        saved_pid = launcher_pid or pid
        if started and process_started(saved_pid) != started:
            raise RuntimeError("The saved UI process identity has changed; it was left unchanged")
        try:
            metadata = windows_ui_metadata(pid, port)
            valid = matching_ui_process(metadata, pid, command, launcher_pid)
        except (OSError, TypeError, ValueError):
            valid = False
        if not valid or kernel.WaitForSingleObject(handle, 0) != 258 or process_started(pid) != observed_started:
            raise RuntimeError("The previous UI listener or command did not match this installation; it was left unchanged")
        if not kernel.TerminateProcess(handle, 0) or kernel.WaitForSingleObject(handle, 5000) != 0:
            raise RuntimeError("The verified previous UI server did not exit")
    finally:
        kernel.CloseHandle(handle)


class Launcher:
    def __init__(self, args, source, work):
        self.source, self.work = source, work
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        from serve.install_guard import assert_installation_ready
        assert_installation_ready(work_root=self.work, allow_installer=bool(getattr(args, "configure_only", False)))
        self.runtime = work / "runtime"
        self.backend = getattr(args, "backend", None) or "auto"
        self.intel_profile = getattr(args, "intel_profile", None) or "b390-qwen36"
        if getattr(args, "intel_profile", None) and self.backend != "sycl":
            raise ValueError("--intel-profile requires explicit --backend sycl")
        self.state_name = "ui-sycl" if self.backend == "sycl" else "ui"
        self.state_dir = self.runtime / self.state_name
        self.state_dir.mkdir(parents=True, exist_ok=True)
        settings_path = self.state_dir / "settings.json"
        try:
            settings = read_json(settings_path) if settings_path.is_file() else {}
        except (OSError, ValueError):
            settings = {}
        if not isinstance(settings, dict):
            settings = {}
        self.api_port = port_number(args.api_port if args.api_port is not None else settings.get("api_port", 18085 if self.backend == "sycl" else 18080))
        self.port = port_number(args.port if args.port is not None else settings.get("port", 3005 if self.backend == "sycl" else 3000))
        selected_family = args.model_family or settings.get("model_family", "qwen36")
        if args.model_family:
            reject_retired_model(args.model_family)
        elif retired_model(selected_family) and self.backend != "sycl":
            selected_family = available_default_family(self.runtime)
        self.family = FAMILY_ALIASES.get(selected_family, selected_family)
        if self.backend == "sycl" and (self.family != "qwen36" or self.intel_profile != "b390-qwen36"):
            raise ValueError("Experimental Intel supports Qwen3.6 text with profile b390-qwen36 only")
        if self.family not in PROFILES:
            raise ValueError("Unknown saved model family; specify --model-family qwen36, gemma4, lfm2, clover or tiel")
        if self.api_port == self.port:
            raise ValueError("The UI and API must use different ports")
        self.python = self.runtime / ".venv" / "Scripts" / "python.exe"
        self.url = f"http://127.0.0.1:{self.port}"
        self._spawn_log_offsets = {}

    def log(self, message):
        with (self.state_dir / "launch.log").open("a", encoding="utf-8") as output:
            output.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")

    def write_state(self, name, values):
        target = self.state_dir / name
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)

    def save_settings(self):
        settings = {"source_dir": str(self.source), "work_dir": str(self.work),
                    "api_port": self.api_port, "port": self.port, "model_family": self.family}
        if self.backend == "sycl":
            settings.update(backend="sycl", intel_profile=self.intel_profile)
        self.write_state("settings.json", settings)
        # A small pointer in the normal sibling work directory lets run-ui.bat
        # find a custom installation's private Python without changing Git files.
        location = self.source.with_name(self.source.name + "-work") / "runtime" / self.state_name / "location.json"
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
            self._spawn_log_offsets[label] = errors.tell()
            errors.write((f"\n[{datetime.now(timezone.utc).isoformat()}] Starting {label}; "
                          f"source={self.source}\n").encode("utf-8"))
            errors.flush()
            process = subprocess.Popen(command, cwd=self.source, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=errors, creationflags=flags,
                                       env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
        try:
            self.log(f"Started {label}: PID {process.pid}")
            state = {"pid": process.pid, "command": command}
            if label == "ui" and os.name == "nt":
                state["started"] = process_started(process.pid)
            self.write_state(f"{label}-process.json", state)
        except BaseException:
            # Publication failed: leave the previous atomic state record intact
            # and stop only the exact child handle created by this invocation.
            # Never resolve a PID from a stale record or terminate a process tree.
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            raise
        return process

    def validate_ui_backend(self):
        """Check replacement imports before stopping a working UI; no server is created."""
        if not self.python.is_file():
            raise RuntimeError(f"Installed Python was not found: {self.python}; the previous UI was left running")
        code = ("import importlib,sys; sys.path.insert(0,sys.argv[1]); "
                "importlib.import_module('ui.server')")
        try:
            checked = subprocess.run([str(self.python), "-I", "-B", "-c", code, str(self.source)],
                                     cwd=self.source, capture_output=True, text=True,
                                     encoding="utf-8", errors="replace", timeout=30,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError("The replacement UI import check failed; the previous UI was left running") from error
        if checked.returncode:
            detail = (checked.stderr or checked.stdout or "import failed")[-1200:].strip()
            raise RuntimeError(f"The replacement UI could not import; the previous UI was left running: {detail}")

    def prepare_api(self, config):
        # Keep run.py's configuration migration/bootstrap, but run the actual
        # server directly so its PID is discoverable by a later installer update.
        arguments = ["--config", str(config), "--model-family", self.family,
                     "--port", str(self.api_port), "--host", "127.0.0.1", "--configure-only"]
        if self.backend == "sycl":
            arguments += ["--backend", "sycl", "--intel-profile", self.intel_profile]
        elif self.backend != "auto":
            arguments += ["--backend", self.backend]
        process = self.spawn(self.source / "run.py", arguments, "api-prepare")
        if process.wait() != 0:
            raise RuntimeError(f"API configuration failed; see {self.state_dir / 'api-prepare.stderr.log'}")

    def save_api_process(self, process, config):
        name = {"qwen36": "server-state.json", "gemma4": "server-state-gemma4.json",
                "lfm2": "server-state-lfm2.json", "clover": "server-state-clover.json",
                "tiel": "server-state-tiel-coder.json"}[self.family]
        if self.backend == "sycl":
            name = "server-state-sycl.json"
        target = self.runtime / name
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        values = {"pid": process.pid, "started": process_started(process.pid),
                  "config": str(config), "source": str(self.source), "port": self.api_port}
        temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
        temporary.replace(target)

    def ensure_api(self, no_start=False):
        process = None
        deadline = time.monotonic() + 300
        expected_model, _ = PROFILES[self.family]
        if port_occupied(self.api_port):
            try:
                health = request_json(self.api_port, "/health")
            except Exception as error:
                raise RuntimeError(f"API port {self.api_port} is occupied or unresponsive; it was left unchanged") from error
            if not self.selected_api_identity(health):
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
            config = profile_path(self.runtime, self.family, self.backend)
            if not config.is_file():
                raise RuntimeError(f"Installed model configuration was not found: {config}. Run the installer first")
            self.prepare_api(config)
            process = self.spawn(None, ["-m", "serve.server", "--engine", "tinystrata", "--config", str(config),
                                 "--port", str(self.api_port), "--host", "127.0.0.1"], "api")
            self.save_api_process(process, config)
        while time.monotonic() < deadline:
            try:
                health = request_json(self.api_port, "/health")
                if not self.selected_api_identity(health):
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

    def selected_api_identity(self, health):
        if not api_identity(health):
            return False
        if self.backend == "sycl":
            return (health.get("backend") == "sycl" and health.get("experimental") is True and
                    health.get("intel_profile") == self.intel_profile and
                    health.get("model") == PROFILES["qwen36"][0])
        return True

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
            if self.current_ui(health):
                self.log(f"Reusing the UI server: PID {health.get('pid')}")
                return
            # An incomplete update must not stop the working previous UI.
            backend_revision(self.source)
            self.validate_ui_backend()
            self.restart_previous_ui(health)
        process = self.spawn(self.source / "ui" / "server.py", ["--port", str(self.port),
                             "--api-port", str(self.api_port), "--data-dir", str(self.state_dir)], "ui")
        self.ui_process = process
        deadline = time.monotonic() + 15
        last_error = None
        while time.monotonic() < deadline:
            try:
                health = self.existing_ui()
            except (OSError, ValueError, http.client.HTTPException, RuntimeError) as error:
                # A listener may bind before its first HTTP response. Only this
                # network-readiness probe is retryable; ownership and state
                # publication errors below must retain their actual cause.
                last_error = error
                health = None
            if health is not None:
                if not self.current_ui(health):
                    mismatches = ", ".join(self.ui_mismatches(health))
                    raise RuntimeError(f"The started UI serves a different installation or backend ({mismatches}); "
                                       f"see {self.state_dir / 'ui.stderr.log'}")
                pid = health.get("pid")
                command = [str(self.python), "-B", str(self.source / "ui" / "server.py"),
                           "--port", str(self.port), "--api-port", str(self.api_port),
                           "--data-dir", str(self.state_dir)]
                if type(pid) is not int or pid <= 0:
                    raise RuntimeError("The started UI returned an invalid process identity")
                state = {"pid": pid, "launcher_pid": process.pid, "command": command}
                if os.name == "nt":
                    if not matching_ui_process(windows_ui_metadata(pid, self.port), pid, command, process.pid):
                        raise RuntimeError("The UI listener did not match the started process; it was left unchanged")
                    state["started"] = process_started(pid)
                elif pid != process.pid:
                    raise RuntimeError("The UI listener did not match the started process; it was left unchanged")
                try:
                    self.write_state("ui-process.json", state)
                except OSError as error:
                    raise RuntimeError(f"The UI server is ready, but its process record could not be saved: {error}") from error
                return
            if process.poll() is not None:
                break
            time.sleep(.2)
        detail = self.startup_failure_detail(process, "ui", last_error)
        raise RuntimeError(f"The UI server did not become ready{detail}; see {self.state_dir / 'ui.stderr.log'}")

    def startup_failure_detail(self, process, label, last_error=None):
        parts = []
        code = process.poll()
        if code is not None:
            parts.append(f"exit code {code}")
        if last_error is not None:
            parts.append(str(last_error))
        offset = self._spawn_log_offsets.get(label)
        if offset is not None:
            try:
                with (self.state_dir / f"{label}.stderr.log").open("rb") as errors:
                    errors.seek(offset)
                    # Keep old attempts separate and bound diagnostics even if
                    # a broken child writes continuously before exiting.
                    tail = errors.read(65536).decode("utf-8", errors="replace")[-2400:].strip()
                if tail:
                    parts.append(tail)
            except OSError:
                pass  # The original startup failure remains authoritative.
        return ": " + " | ".join(parts) if parts else ""

    def ui_mismatches(self, health):
        if not isinstance(health, dict):
            return ["invalid health response"]
        problems = []
        try:
            if health.get("backend_revision") != backend_revision(self.source):
                problems.append("backend revision")
        except OSError:
            problems.append("backend source files unavailable")
        for key, expected in (("source_dir", self.source), ("data_dir", self.state_dir),
                              ("python_executable", self.python)):
            try:
                if Path(health[key]).resolve() != expected.resolve():
                    problems.append(key)
            except (OSError, KeyError, TypeError, ValueError):
                problems.append(key)
        return problems

    def current_ui(self, health):
        return not self.ui_mismatches(health)

    def restart_previous_ui(self, health):
        try:
            state = read_json(self.state_dir / "ui-process.json")
            pid = health["pid"]
            old_command = [str(self.python), "-B", str(self.source / "ui" / "server.py"),
                           "--port", str(self.port), "--api-port", str(self.api_port)]
            new_command = old_command + ["--data-dir", str(self.state_dir)]
            # Legacy state has no creation timestamp, so OS executable, exact
            # command and listener ownership are all required before upgrading.
            if (type(pid) is not int or pid <= 0 or not isinstance(state, dict) or
                    type(state.get("pid")) is not int or state["pid"] <= 0 or
                    state.get("command") not in (old_command, new_command) or
                    ("started" in state and not isinstance(state["started"], str))):
                raise ValueError("UI process record does not match")
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise RuntimeError("The UI server needs updating, but its ownership could not be verified. Close that UI server and run TinyStrata again; the API and model were left unchanged") from error
        stop_owned_ui(pid, state["command"], state.get("started"), self.port,
                      launcher_pid=state["pid"] if state["pid"] != pid else None)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if not port_occupied(self.port):
                self.log(f"Stopped the verified previous UI server for an update: PID {pid}")
                return
            time.sleep(.1)
        raise RuntimeError("The previous UI port is still occupied; no replacement server was started")


def main():
    # Also works when this file is launched directly from a Unicode install.
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from ui.stdio import configure_utf8_stdio

    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--work-dir", type=Path, help="state directory (default: source directory's sibling ending in -work)")
    parser.add_argument("--api-port", type=port_number, help="API port (default: saved selection or 18080)")
    parser.add_argument("--port", type=port_number, help="UI port (default: saved selection or 3000)")
    parser.add_argument("--model-family", choices=(*PROFILES, *FAMILY_ALIASES), help="model for a new API (default: saved selection or qwen36)")
    parser.add_argument("--backend", choices=("auto", "cuda", "hip", "sycl"),
                        help="explicit sycl uses experimental Qwen3.6 text, config-sycl.json and separate ui-sycl history/settings (API 18085/UI 3005); default selection is unchanged")
    parser.add_argument("--intel-profile", choices=("b390-qwen36",), help="requested Intel tuning profile; requires --backend sycl and does not certify detected hardware")
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
            location = default_work / "runtime" / ("ui-sycl" if args.backend == "sycl" else "ui") / "location.json"
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
        # Preserve the verified API's current family and ports for a desktop
        # retry even when only the UI helper fails to start.
        launcher.save_settings()
        launcher.ensure_ui()
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
