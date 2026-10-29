"""Windows AMD identity and dedicated VRAM, without initializing HIP in the API.

HIP ordinals are not registry or DXGI enumeration ordinals. A short-lived HIP
metadata probe supplies the selected adapter's LUID; Windows supplies its name,
dedicated capacity and adapter-wide memory usage. No GPU work is submitted.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import subprocess
import sys
import uuid


_HIP_PROBE = r'''
import ctypes as c, json, os, pathlib, struct
dirs = [d for d in os.environ.get("PATH", "").split(os.pathsep) if d and os.path.isdir(d)]
handles = [os.add_dll_directory(d) for d in dirs] if hasattr(os, "add_dll_directory") else []
names = [p for d in dirs for p in pathlib.Path(d).glob("amdhip64*.dll")]
h = c.CDLL(str(names[0]) if names else "amdhip64.dll")
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
                  "node_mask": mask.value if luid_ok else None})
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


class _Guid(ctypes.Structure):
    _fields_ = [("data1", ctypes.c_uint32), ("data2", ctypes.c_uint16), ("data3", ctypes.c_uint16),
                ("data4", ctypes.c_ubyte * 8)]


class _Luid(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_int32)]


class _AdapterDesc(ctypes.Structure):
    _fields_ = [("description", ctypes.c_wchar * 128), ("vendor", ctypes.c_uint32),
                ("device", ctypes.c_uint32), ("subsystem", ctypes.c_uint32), ("revision", ctypes.c_uint32),
                ("dedicated", ctypes.c_size_t), ("system", ctypes.c_size_t), ("shared", ctypes.c_size_t),
                ("luid", _Luid), ("flags", ctypes.c_uint32)]


def _method(obj, index, args, result=ctypes.c_int32):
    table = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *args)(table[index])


def _dxgi_adapters() -> list[dict]:
    """Read 64-bit dedicated capacity and LUID from Windows' DXGI adapter list."""
    if os.name != "nt":
        return []
    factory = ctypes.c_void_p()
    try:
        dll = ctypes.WinDLL("dxgi.dll", winmode=0x800)  # LOAD_LIBRARY_SEARCH_SYSTEM32
        create = dll.CreateDXGIFactory1
        create.argtypes = [ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p)]
        create.restype = ctypes.c_int32
        iid = _Guid.from_buffer_copy(uuid.UUID("770aae78-f26f-4dba-a829-253c83d1b387").bytes_le)
        if create(ctypes.byref(iid), ctypes.byref(factory)) < 0 or not factory:
            return []
        cards = []
        for i in range(32):
            adapter = ctypes.c_void_p()
            # IDXGIFactory1::EnumAdapters1, after the IUnknown/IDXGIObject/Factory methods.
            if _method(factory, 12, [ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)])(
                    factory, i, ctypes.byref(adapter)) < 0 or not adapter:
                break
            try:
                desc = _AdapterDesc()
                if _method(adapter, 10, [ctypes.POINTER(_AdapterDesc)])(adapter, ctypes.byref(desc)) >= 0:
                    cards.append({"name": desc.description, "mem_total": desc.dedicated or None,
                                  "luid": f"{desc.luid.high & 0xffffffff:08x}:{desc.luid.low:08x}"})
            finally:
                _method(adapter, 2, [], ctypes.c_uint32)(adapter)
        return cards
    except (OSError, AttributeError, ValueError, TypeError):
        return []
    finally:
        if factory:
            _method(factory, 2, [], ctypes.c_uint32)(factory)


class _PdhValue(ctypes.Structure):
    _fields_ = [("status", ctypes.c_uint32), ("value", ctypes.c_int64)]


class _PdhItem(ctypes.Structure):
    _fields_ = [("name", ctypes.c_wchar_p), ("data", _PdhValue)]


class _DedicatedUsage:
    """Adapter-wide instant gauge, matched by LUID rather than display order.

    This is the same counter used by hip_compat/dedicated_vram.hpp when DXCore
    state queries are unavailable. English counter names also work on Japanese
    Windows; no subprocess or vendor DLL is needed for each sample.
    """
    def __init__(self, luid, node_mask):
        self.dll = None
        self.query = ctypes.c_void_p()
        self.counter = ctypes.c_void_p()
        self.prefix = None
        # Multiple physical nodes need an explicit mapping, not an assumed node 0.
        if os.name != "nt" or not isinstance(luid, str) or node_mask not in (None, 0, 1):
            return
        parts = luid.split(":")
        if len(parts) != 2 or any(len(p) != 8 or any(c not in "0123456789abcdef" for c in p.lower()) for p in parts):
            return
        self.prefix = f"luid_0x{parts[0].lower()}_0x{parts[1].lower()}_phys_"
        try:
            self.dll = ctypes.WinDLL("pdh.dll", winmode=0x800)
            d = self.dll
            d.PdhOpenQueryW.argtypes = [ctypes.c_wchar_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
            d.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_size_t,
                                              ctypes.POINTER(ctypes.c_void_p)]
            d.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
            d.PdhGetFormattedCounterArrayW.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
            d.PdhCloseQuery.argtypes = [ctypes.c_void_p]
            for name in ("PdhOpenQueryW", "PdhAddEnglishCounterW", "PdhCollectQueryData",
                         "PdhGetFormattedCounterArrayW", "PdhCloseQuery"):
                getattr(d, name).restype = ctypes.c_int32
            if d.PdhOpenQueryW(None, 0, ctypes.byref(self.query)) != 0:
                self.close()
                return
            if d.PdhAddEnglishCounterW(self.query, r"\GPU Adapter Memory(*)\Dedicated Usage", 0,
                                      ctypes.byref(self.counter)) != 0:
                self.close()
        except (OSError, AttributeError, ValueError, TypeError):
            self.close()

    def read(self):
        if not self.dll or not self.query or not self.counter:
            return None
        try:
            d = self.dll
            if d.PdhCollectQueryData(self.query) != 0:
                return None
            size, count = ctypes.c_uint32(), ctypes.c_uint32()
            status = d.PdhGetFormattedCounterArrayW(self.counter, 0x400, ctypes.byref(size),
                                                   ctypes.byref(count), None)
            if status & 0xffffffff != 0x800007d2 or not 0 < size.value <= 1024 * 1024:
                return None  # PDH_MORE_DATA, bounded metadata buffer
            buffer = ctypes.create_string_buffer(size.value)
            if d.PdhGetFormattedCounterArrayW(self.counter, 0x400, ctypes.byref(size),
                                             ctypes.byref(count), buffer) != 0 or \
                    count.value * ctypes.sizeof(_PdhItem) > len(buffer):
                return None
            items = ctypes.cast(buffer, ctypes.POINTER(_PdhItem))
            matches = []
            for i in range(count.value):
                name = (items[i].name or "").lower()
                if not name.startswith(self.prefix):
                    continue
                if name[len(self.prefix):] != "0" or items[i].data.status not in (0, 1) or items[i].data.value < 0:
                    return None
                matches.append(items[i].data.value)
            return matches[0] if len(matches) == 1 else None
        except (OSError, AttributeError, ValueError, TypeError):
            return None

    def close(self):
        if self.dll and self.query:
            try:
                self.dll.PdhCloseQuery(self.query)
            except (OSError, AttributeError):
                pass
        self.query = ctypes.c_void_p()
        self.counter = ctypes.c_void_p()


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
