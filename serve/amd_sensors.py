"""Read-only AMD ADLX sensors in an isolated Windows helper process.

ADLX's own capability queries select metrics; adapter identity is the Windows
LUID, never an enumeration index or a GPU-name table. The API process does not
load an AMD driver DLL. A stuck/crashed driver call expires its readings and
ends the helper without blocking the telemetry endpoint.

ABI reference: AMD's ADLX SDK, ISystem2.h and IPerformanceMonitoring.h:
https://github.com/GPUOpen-LibrariesAndSDKs/ADLX/tree/main/SDK/Include
Only Get/IsSupported methods are used; no tuning or device power controls.
"""
from __future__ import annotations

import ctypes as c
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time


_METRICS = ("util", "temp", "power", "pcie")
_LUID = re.compile(r"[0-9a-f]{8}:[0-9a-f]{8}\Z")
_FRESH_SECONDS = 5.0
_HANG_SECONDS = 15.0
_MAX_LINE = 16384


def _finite_number(value, minimum, maximum):
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and minimum <= value <= maximum)


class AmdSensors:
    """Nonblocking snapshots of one HIP adapter's optional Windows sensors.

    ``read()`` returns util (%), temp (C), power (W), metric_status and sources.
    Missing/unsupported values are omitted, not zero-filled. ``power_scope`` is
    ``board`` when total board power is available, otherwise ``chip``. PCIe
    transfer/link telemetry is left to Windows: ADLX GPU1's static bus-info
    methods do not document an instantaneous transfer rate.

    Construction does not wait for driver initialization. One hidden child is
    started at most once; driver failure does not trigger a restart loop.
    """
    def __init__(self, luid):
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._proc = None
        self._thread = None
        self._latest = {}
        self._sample_at = 0.0
        self._message_at = time.monotonic()
        self._failure = None
        self._closed = False
        self._luid = luid.lower() if isinstance(luid, str) else ""
        if os.name != "nt":
            self._failure = "windows_only"
        elif not _LUID.fullmatch(self._luid) or self._luid == "00000000:00000000":
            self._failure = "adapter_identity_unavailable"
        else:
            self._thread = threading.Thread(target=self._run, name="AMD sensor helper", daemon=True)
            self._thread.start()

    def _run(self):
        proc = None
        try:
            proc = subprocess.Popen(
                [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--worker", self._luid],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            # Use the server's existing kill-on-close job; no vendor DLL in the
            # API process, even when the API exits without running close().
            try:
                from .winjob import contain
                contain(proc)
            except (ImportError, OSError, AttributeError):
                pass
            with self._lock:
                self._proc = proc
            if self._stop.is_set():
                return
            listener = threading.Thread(target=self._listen, args=(proc,),
                                        name="AMD sensor pipe", daemon=True)
            listener.start()
            while not self._stop.wait(0.25):
                if proc.poll() is not None:
                    listener.join(timeout=0.25)
                    with self._lock:
                        if not self._failure:
                            self._failure = "sensor_helper_exited"
                    break
                with self._lock:
                    silent_for = time.monotonic() - self._message_at
                if silent_for > _HANG_SECONDS:
                    with self._lock:
                        self._failure = "sensor_helper_timeout"
                    break
        except (OSError, ValueError):
            with self._lock:
                self._failure = "sensor_helper_unavailable"
        finally:
            if proc is not None:
                try:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait(timeout=0.5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                # The pipe reader closes stdout after EOF. Closing a blocked
                # BufferedReader here could wait on its internal read lock.

    def _listen(self, proc):
        try:
            with proc.stdout:
                while not self._stop.is_set():
                    line = proc.stdout.readline(_MAX_LINE + 1)
                    if not line:
                        break
                    if len(line) > _MAX_LINE or not line.endswith(b"\n"):
                        with self._lock:
                            self._failure = "sensor_helper_invalid_output"
                        self._stop.set()
                        break
                    try:
                        message = json.loads(line)
                    except (ValueError, UnicodeError):
                        continue
                    if not isinstance(message, dict) or message.get("type") not in ("ready", "sample", "error"):
                        continue
                    snapshot = self._validate(message)
                    with self._lock:
                        self._message_at = time.monotonic()
                        self._latest = snapshot
                        if message["type"] == "sample":
                            self._sample_at = self._message_at
                        if message["type"] == "error":
                            self._failure = str(message.get("reason", "sensor_unavailable"))[:100]
        except (OSError, ValueError):
            pass

    @staticmethod
    def _validate(message):
        snapshot = {}
        limits = {"util": (0, 100), "temp": (-273.15, 1000), "power": (0, 1000000)}
        for key, bounds in limits.items():
            value = message.get(key)
            if _finite_number(value, *bounds):
                snapshot[key] = value
        statuses = message.get("metric_status", {})
        sources = message.get("metric_sources", {})
        snapshot["metric_status"] = {
            key: statuses.get(key) if isinstance(statuses, dict) and statuses.get(key) in
            ("available", "unsupported", "unavailable", "warming_up") else "unavailable"
            for key in _METRICS
        }
        snapshot["metric_sources"] = {
            key: str(sources[key])[:100] for key in _METRICS
            if isinstance(sources, dict) and isinstance(sources.get(key), str)
        }
        # A success status without a valid value must not create a zero gauge.
        for key in limits:
            if snapshot["metric_status"][key] == "available" and key not in snapshot:
                snapshot["metric_status"][key] = "unavailable"
        if message.get("power_scope") in ("board", "chip"):
            snapshot["power_scope"] = message["power_scope"]
        if isinstance(message.get("sensor_info"), dict):
            snapshot["sensor_info"] = {
                key: message["sensor_info"][key]
                for key in ("luid", "adlx_pci_bus_type", "adlx_pci_lanes")
                if isinstance(message["sensor_info"].get(key), (str, int))
            }
        return snapshot

    def read(self):
        """Return a latest-value snapshot immediately; never call a driver here."""
        with self._lock:
            out = dict(self._latest)
            age = time.monotonic() - self._sample_at
            failure = self._failure
            closed = self._closed
        statuses = dict(out.get("metric_status", {}))
        for key in _METRICS:
            statuses.setdefault(key, "unavailable" if failure or closed else "warming_up")
        if age > _FRESH_SECONDS or failure or closed:
            for key in ("util", "temp", "power"):
                out.pop(key, None)
                if statuses[key] == "available":
                    statuses[key] = "unavailable"
        out["metric_status"] = statuses
        if "metric_sources" in out:
            out["metric_sources"] = dict(out["metric_sources"])
        if "sensor_info" in out:
            out["sensor_info"] = dict(out["sensor_info"])
        if failure:
            out["sensor_error"] = failure
        return out

    def close(self):
        """Stop our own helper, with a bounded wait even after a driver hang."""
        self._stop.set()
        with self._lock:
            self._closed = True
            proc = self._proc
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.kill()
            except OSError:
                pass
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=0.75)


# The following bindings are only instantiated inside --worker. ADLX uses
# stdcall interface methods, cdecl exports, and 8-bit adlx_bool (not BOOL).
def _method(obj, slot, args=(), result=c.c_int32):
    table = c.cast(obj, c.POINTER(c.POINTER(c.c_void_p))).contents
    return c.WINFUNCTYPE(result, c.c_void_p, *args)(table[slot])


def _out(obj, slot, value_type):
    value = value_type()
    status = _method(obj, slot, (c.POINTER(value_type),))(obj, c.byref(value))
    return status, value


def _query(obj, name):
    value = c.c_void_p()
    status = _method(obj, 2, (c.c_wchar_p, c.POINTER(c.c_void_p)))(obj, name, c.byref(value))
    return status, value


def _release(obj):
    if obj:
        _method(obj, 1)(obj)


class _AdapterLuid(c.Structure):
    _fields_ = [("low", c.c_uint32), ("high", c.c_int32)]


class _Adlx:
    def __init__(self, luid):
        self.dll = None
        self.system = c.c_void_p()
        self.gpu = c.c_void_p()
        self.perf = c.c_void_p()
        self.flags = {}
        self.info = {"luid": luid}
        try:
            # Restrict loading to the installed, signed Windows driver location.
            self.dll = c.CDLL("amdadlx64.dll", winmode=0x800)
            self.dll.ADLXInitialize.argtypes = (c.c_uint64, c.POINTER(c.c_void_p))
            self.dll.ADLXInitialize.restype = c.c_int32
            self.dll.ADLXTerminate.argtypes = ()
            self.dll.ADLXTerminate.restype = c.c_int32
            # Ask for the baseline ABI, then QueryInterface newer extensions.
            # Passing the newest SDK version would reject otherwise usable
            # older drivers. IADLXSystem is a singleton and is never Released.
            if self.dll.ADLXInitialize(1 << 48, c.byref(self.system)) != 0 or not self.system:
                raise RuntimeError("adlx_initialization_failed")
            self._select(luid)
            status, self.perf = _out(self.system, 9, c.c_void_p)
            if status != 0 or not self.perf:
                raise RuntimeError("adlx_metrics_unavailable")
            support = c.c_void_p()
            try:
                status = _method(self.perf, 21, (c.c_void_p, c.POINTER(c.c_void_p)))(
                    self.perf, self.gpu, c.byref(support))
                if status != 0 or not support:
                    raise RuntimeError("adlx_capabilities_unavailable")
                for key, slot in (("util", 3), ("temp", 6), ("chip_power", 8), ("board_power", 9)):
                    status, value = _out(support, slot, c.c_uint8)
                    self.flags[key] = bool(value.value) if status == 0 else None
            finally:
                _release(support)
        except BaseException:
            self.close()
            raise

    def _select(self, luid):
        gpu_list = c.c_void_p()
        matches = []
        try:
            status, gpu_list = _out(self.system, 1, c.c_void_p)
            if status != 0 or not gpu_list:
                raise RuntimeError("adlx_adapters_unavailable")
            count = _method(gpu_list, 3, result=c.c_uint32)(gpu_list)
            if not 0 <= count <= 64:
                raise RuntimeError("adlx_adapters_invalid")
            for index in range(count):
                gpu, gpu2 = c.c_void_p(), c.c_void_p()
                try:
                    status = _method(gpu_list, 11, (c.c_uint32, c.POINTER(c.c_void_p)))(
                        gpu_list, index, c.byref(gpu))
                    if status != 0 or not gpu:
                        continue
                    status, gpu2 = _query(gpu, "IADLXGPU2")
                    if status != 0 or not gpu2:
                        continue  # Older interfaces cannot prove adapter identity.
                    status, value = _out(gpu2, 34, _AdapterLuid)
                    actual = f"{value.high & 0xffffffff:08x}:{value.low:08x}"
                    if status != 0 or actual != luid:
                        continue
                    matches.append(gpu2)
                    gpu2 = c.c_void_p()  # Transfer the acquired reference.
                finally:
                    _release(gpu2)
                    _release(gpu)
            if len(matches) != 1:
                raise RuntimeError("adlx_adapter_identity_unavailable")
            self.gpu = matches.pop()
            # Raw adapter information only, not advertised as current PCIe
            # bandwidth/transfer metrics. Windows link properties are separate.
            status, value = _out(self.gpu, 19, c.c_int32)
            if status == 0 and value.value >= 0:
                self.info["adlx_pci_bus_type"] = value.value
            status, value = _out(self.gpu, 20, c.c_uint32)
            if status == 0 and value.value > 0:
                self.info["adlx_pci_lanes"] = value.value
        finally:
            for gpu in matches:
                _release(gpu)
            _release(gpu_list)

    def _status(self):
        statuses = {}
        for key in ("util", "temp"):
            statuses[key] = "unsupported" if self.flags[key] is False else "unavailable"
        power_flags = (self.flags["chip_power"], self.flags["board_power"])
        statuses["power"] = "unsupported" if all(v is False for v in power_flags) else "unavailable"
        statuses["pcie"] = "unsupported"
        return statuses

    def sample(self):
        out = {"type": "sample", "metric_status": self._status(),
               "metric_sources": {}, "sensor_info": dict(self.info)}
        sample = c.c_void_p()
        try:
            status = _method(self.perf, 18, (c.c_void_p, c.POINTER(c.c_void_p)))(
                self.perf, self.gpu, c.byref(sample))
            if status != 0 or not sample:
                return out
            metrics = (("util", "util", 4, 0, 100), ("temp", "temp", 7, -273.15, 1000),
                       ("power", "board_power", 10, 0, 1000000),
                       ("power", "chip_power", 9, 0, 1000000))
            for key, flag, slot, minimum, maximum in metrics:
                if not self.flags[flag] or key in out:
                    continue
                status, value = _out(sample, slot, c.c_double)
                if status == 0 and _finite_number(value.value, minimum, maximum):
                    out[key] = value.value
                    out["metric_status"][key] = "available"
                    out["metric_sources"][key] = "AMD ADLX"
                    if key == "power":
                        out["power_scope"] = "board" if flag == "board_power" else "chip"
            return out
        finally:
            _release(sample)

    def ready(self):
        out = {"type": "ready", "metric_status": self._status(), "sensor_info": dict(self.info)}
        for key, status in out["metric_status"].items():
            if status == "unavailable":
                out["metric_status"][key] = "warming_up"
        return out

    def close(self):
        _release(self.perf)
        _release(self.gpu)
        self.perf, self.gpu = c.c_void_p(), c.c_void_p()
        if self.system:
            self.system = c.c_void_p()
            self.dll.ADLXTerminate()


def _emit(message):
    print(json.dumps(message, allow_nan=False, separators=(",", ":")), flush=True)


def _worker(luid):
    sensors = None
    try:
        if os.name != "nt" or not _LUID.fullmatch(luid):
            raise RuntimeError("adapter_identity_unavailable")
        sensors = _Adlx(luid)
        _emit(sensors.ready())
        while True:
            started = time.monotonic()
            _emit(sensors.sample())
            time.sleep(max(0.0, 1.0 - (time.monotonic() - started)))
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    except (OSError, AttributeError, ValueError, RuntimeError) as error:
        reason = str(error) if isinstance(error, RuntimeError) else "adlx_unavailable"
        _emit({"type": "error", "reason": reason,
               "metric_status": {key: "unavailable" for key in _METRICS}})
    finally:
        if sensors is not None:
            sensors.close()


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--worker":
    _worker(sys.argv[2].lower())
