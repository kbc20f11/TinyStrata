"""Windows AMD identity and dedicated VRAM, without initializing HIP in the API.

HIP ordinals are not registry or DXGI enumeration ordinals. A short-lived HIP
metadata probe supplies the selected adapter's LUID; Windows supplies its name,
dedicated capacity and adapter-wide memory usage. No GPU work is submitted.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys


_HIP_PROBE = r'''
import ctypes as c, json, os, pathlib, re, struct
selected = os.environ.get("TINYSTRATA_HIP_RUNTIME_DLL")
if selected:
    dirs = json.loads(os.environ.get("TINYSTRATA_HIP_RUNTIME_DIRS", "null"))
    dirs = [str(pathlib.Path(selected).parent)] if dirs is None else dirs
    if not isinstance(dirs, list) or any(not isinstance(d, str) or not os.path.isabs(d) or not os.path.isdir(d) for d in dirs):
        raise RuntimeError("Selected HIP runtime library directories are invalid")
else:
    dirs = [d for d in os.environ.get("PATH", "").split(os.pathsep) if d and os.path.isdir(d)]
handles = [os.add_dll_directory(d) for d in dirs] if hasattr(os, "add_dll_directory") else []
names = [p for d in dirs for p in pathlib.Path(d).glob("amdhip64*.dll")]
if selected and (not os.path.isabs(selected) or not os.path.isfile(selected)):
    raise RuntimeError("Selected HIP runtime DLL is missing or not absolute")
kernel = c.WinDLL("kernel32.dll", winmode=0x800)
kernel.GetModuleFileNameW.argtypes = [c.c_void_p, c.c_wchar_p, c.c_uint]
kernel.GetModuleFileNameW.restype = c.c_uint
def loaded_path(library, expected=None):
    loaded = c.create_unicode_buffer(32768)
    length = kernel.GetModuleFileNameW(library._handle, loaded, len(loaded))
    if not 0 < length < len(loaded):
        raise RuntimeError("Cannot identify a loaded HIP runtime library")
    actual = os.path.realpath(loaded.value)
    if expected and os.path.normcase(os.path.realpath(expected)) != os.path.normcase(actual):
        raise RuntimeError("HIP probe loaded a different library from the native runtime")
    return actual
comgr_paths = json.loads(os.environ.get("TINYSTRATA_HIP_COMGR_DLLS", "[]")) if selected else []
if not isinstance(comgr_paths, list) or any(not isinstance(p, str) or not os.path.isabs(p) or not os.path.isfile(p)
        or not re.fullmatch(r"amd_comgr[0-9_]*\.dll", pathlib.Path(p).name, re.I) for p in comgr_paths):
    raise RuntimeError("Selected HIP COMGR libraries are invalid")
comgr_handles, actual_comgr = [], []
for path in comgr_paths:
    library = c.CDLL(path, winmode=0x1100)
    comgr_handles.append(library)
    actual_comgr.append(loaded_path(library, path))
h = c.CDLL(selected, winmode=0x1100) if selected else c.CDLL(str(names[0]) if names else "amdhip64.dll")
actual_dll = loaded_path(h, selected)
h.hipGetDeviceCount.argtypes = [c.POINTER(c.c_int)]
h.hipDeviceGetName.argtypes = [c.c_char_p, c.c_int, c.c_int]
h.hipDeviceTotalMem.argtypes = [c.POINTER(c.c_size_t), c.c_int]
h.hipDeviceGetAttribute.argtypes = [c.POINTER(c.c_int), c.c_int, c.c_int]
get_luid = getattr(h, "hipDeviceGetLuid", None)
if get_luid:
    get_luid.argtypes = [c.c_char_p, c.POINTER(c.c_uint), c.c_int]
n = c.c_int()
assert h.hipGetDeviceCount(c.byref(n)) == 0 and 0 <= n.value <= 32
cards = []
for i in range(n.value):
    name = c.create_string_buffer(256)
    if h.hipDeviceGetName(name, len(name), i) != 0:
        continue
    total = c.c_size_t()
    memory_ok = h.hipDeviceTotalMem(c.byref(total), i) == 0
    integrated = c.c_int()
    attribute_ok = h.hipDeviceGetAttribute(c.byref(integrated), 16, i) == 0
    luid = c.create_string_buffer(8)
    mask = c.c_uint()
    luid_ok = get_luid and get_luid(luid, c.byref(mask), i) == 0
    lo, hi = struct.unpack("<II", luid.raw)
    cards.append({"index": i, "name": name.value.decode("utf-8", "replace"),
                  "mem_total": total.value if memory_ok and total.value else None,
                  "integrated": bool(integrated.value) if attribute_ok else None,
                  "luid": "%08x:%08x" % (hi, lo) if luid_ok and (lo or hi) else None,
                  "node_mask": mask.value if luid_ok else None,
                  "runtime_dll": actual_dll, "comgr_dlls": actual_comgr})
print(json.dumps(cards))
'''


def hip_cards(env=None) -> list[dict]:
    """Visible HIP devices in their runtime order, honoring the exact engine env.

    The child owns all HIP DLL search/runtime side effects. In particular, an
    APU's shared memory is not used to infer which device is the discrete GPU.
    """
    if os.name != "nt":
        return []
    try:
        result = subprocess.run([sys.executable, "-I", "-B", "-c", _HIP_PROBE],
                                env=dict(os.environ) if env is None else dict(env),
                                capture_output=True, text=True, timeout=15,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        cards = json.loads(result.stdout) if result.returncode == 0 else []
        if not isinstance(cards, list) or len(cards) > 32:
            return []
        return [c for c in cards if isinstance(c, dict) and isinstance(c.get("index"), int)
                and isinstance(c.get("name"), str) and c["name"].strip()]
    except (OSError, ValueError, TypeError, subprocess.TimeoutExpired):
        return []


from serve.windows_gpu_memory import (AdapterMemoryUsage as _DedicatedUsage,
                                      dxgi_adapters as _dxgi_adapters)


class WindowsAmdCard:
    """The selected HIP card's Windows readings, with the telemetry reader API."""
    def __init__(self, info):
        self.info = dict(info)
        self._name = self.info.get("name")
        # An APU's HIP total includes shared system RAM, not dedicated VRAM.
        self._total = self.info.get("mem_total") if self.info.get("integrated") is False else None
        if os.name != "nt":
            self._name = self._total = None
        else:
            luid = self.info.get("luid")
            matches = [a for a in _dxgi_adapters() if isinstance(luid, str) and a["luid"] == luid.lower()]
            if len(matches) == 1:
                self._name, self._total = matches[0]["name"], matches[0]["mem_total"]
        self._usage = _DedicatedUsage(self.info.get("luid"), self.info.get("node_mask"))
        self._counters = self._sensors = None
        if os.name == "nt" and isinstance(self.info.get("luid"), str):
            try:
                from serve.windows_gpu_counters import WindowsGpuCounters
                self._counters = WindowsGpuCounters(self.info["luid"], self.info.get("node_mask"))
            except (ImportError, OSError, ValueError, AttributeError):
                pass
            if self.info.get("node_mask") in (None, 0, 1):
                try:
                    from serve.amd_sensors import AmdSensors
                    self._sensors = AmdSensors(self.info["luid"])
                except (ImportError, OSError, ValueError, AttributeError):
                    pass

    def ok(self):
        return bool(self._name)

    def name(self):
        return self._name

    @property
    def total(self):
        """Dedicated capacity for static metadata, even without usage counters."""
        return self._total if isinstance(self._total, int) and not isinstance(self._total, bool) and self._total > 0 else None

    def read(self):
        out = {}
        if self.total is not None:
            out["mem_total"] = self.total
        used = self._usage.read()
        if used is not None:
            out["mem_used"] = used
        readings = []
        sources = {}
        for reader, source in ((self._counters, "Windows"), (self._sensors, "ADLX")):
            try:
                reading = reader.read() if reader else {}
            except Exception:  # optional telemetry cannot break inference or other counters
                reading = {}
            if not isinstance(reading, dict):
                reading = {}
            states = reading.get("metric_status")
            readings.append(states if isinstance(states, dict) else {})
            for key in ("util", "temp", "power", "power_limit", "pcie_gen", "pcie_gen_max", "pcie_width", "pcie_width_max",
                        "pcie_rx_mb", "pcie_tx_mb"):
                value = reading.get(key)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                    continue
                # OS PCIe values describe the actual negotiated link; vendor
                # capability metadata must not replace them. Sensor values use
                # ADLX first, retaining Windows readings when ADLX lacks one.
                if key.startswith("pcie_") and key in out:
                    continue
                out[key] = value
                sources[key] = source
                if key == "power" and reading.get("power_scope") == "chip":
                    out["power_scope"] = "chip"
        groups = {"util": ("util",), "temp": ("temp",), "power": ("power",),
                  "pcie": ("pcie_gen", "pcie_gen_max", "pcie_width", "pcie_rx_mb", "pcie_tx_mb"),
                  "mem_used": ("mem_used",)}
        status = {}
        for group, keys in groups.items():
            if any(out.get(k) is not None for k in keys):
                status[group] = "available"
            elif group == "mem_used":
                status[group] = "unavailable"
            else:
                # A provider that does not implement this metric (e.g. Windows
                # has no watts reading) must not hide ADLX's unsupported flag.
                states = [r[group] for r in readings if group in r]
                status[group] = ("warming_up" if "warming_up" in states else
                                 "unsupported" if states and all(s == "unsupported" for s in states) else "unavailable")
        out["metric_status"] = status
        out["metric_sources"] = sources
        return out

    def close(self):
        self._usage.close()
        for reader in (self._counters, self._sensors):
            if reader:
                try:
                    reader.close()
                except Exception:
                    pass
