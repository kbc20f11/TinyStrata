"""Windows GPU readings matched by adapter LUID, independent of GPU models.

PDH provides WDDM engine utilization. D3DKMT and the PCI bus's device properties
provide temperature and link information when the driver exposes them. No
runtime, kernel driver, administrator access, or vendor DLL is required here.
"""
from __future__ import annotations

import ctypes
import math
import os
import re
import uuid


def _luid_parts(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{8}:[0-9a-fA-F]{8}", value):
        return None
    return tuple(int(x, 16) for x in value.split(":"))


def engine_utilization(items, luid):
    """Sum process contexts on each engine, then use the busiest engine.

    Engines may run simultaneously, so adding 3D, compute and copy utilization
    would exceed 100%. This follows Task Manager's busiest-engine convention.
    Items are (instance name, PDH status, percentage); missing values stay None.
    """
    parts = _luid_parts(luid)
    if parts is None:
        return None
    hi, lo = parts
    pattern = re.compile(rf"(?:^|_)luid_0x{hi:08x}_0x{lo:08x}_phys_(\d+)_eng_(\d+)(?:_|$)", re.I)
    engines = {}
    for name, status, value in items:
        match = pattern.search(name or "")
        if not match:
            continue
        # A logical LUID with several physical nodes needs an explicit mapping.
        if match[1] != "0":
            return None
        if status not in (0, 1) or not isinstance(value, (int, float)) or isinstance(value, bool) \
                or not math.isfinite(value) or value < 0:
            continue
        engine = int(match[2])
        engines[engine] = engines.get(engine, 0.0) + value
    return min(100.0, max(engines.values())) if engines else None


class _CounterValue(ctypes.Structure):
    _fields_ = [("status", ctypes.c_uint32), ("value", ctypes.c_double)]


class _CounterItem(ctypes.Structure):
    _fields_ = [("name", ctypes.c_wchar_p), ("data", _CounterValue)]


class _EngineUsage:
    def __init__(self, luid, allowed=True):
        self.luid, self.dll = luid, None
        self.query, self.counter = ctypes.c_void_p(), ctypes.c_void_p()
        self.status = "unsupported"
        self._collected = False
        if os.name != "nt" or not allowed or _luid_parts(luid) is None:
            return
        try:
            d = self.dll = ctypes.WinDLL("pdh.dll", winmode=0x800)
            signatures = {
                "PdhOpenQueryW": [ctypes.c_wchar_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)],
                "PdhAddEnglishCounterW": [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)],
                "PdhCollectQueryData": [ctypes.c_void_p],
                "PdhGetFormattedCounterArrayW": [ctypes.c_void_p, ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p],
                "PdhCloseQuery": [ctypes.c_void_p],
            }
            for name, args in signatures.items():
                getattr(d, name).argtypes, getattr(d, name).restype = args, ctypes.c_int32
            if d.PdhOpenQueryW(None, 0, ctypes.byref(self.query)) != 0 or \
                    d.PdhAddEnglishCounterW(self.query, r"\GPU Engine(*)\Utilization Percentage", 0,
                                           ctypes.byref(self.counter)) != 0:
                self.close()
                return
            self.status = "warming_up"
            self._collected = d.PdhCollectQueryData(self.query) == 0
        except (OSError, AttributeError, ValueError, TypeError):
            self.close()

    def read(self):
        if not self.dll or not self.query or not self.counter:
            return None
        try:
            d = self.dll
            if d.PdhCollectQueryData(self.query) != 0:
                self.status = "unavailable"
                return None
            if not self._collected:
                self._collected, self.status = True, "warming_up"
                return None
            size, count = ctypes.c_uint32(), ctypes.c_uint32()
            # DOUBLE | NOCAP100: aggregate contexts before clamping the engine.
            fmt = 0x200 | 0x8000
            status = d.PdhGetFormattedCounterArrayW(self.counter, fmt, ctypes.byref(size), ctypes.byref(count), None)
            if status & 0xffffffff != 0x800007d2 or not 0 < size.value <= 4 * 1024 * 1024:
                self.status = "unavailable"
                return None
            buffer = ctypes.create_string_buffer(size.value)
            if d.PdhGetFormattedCounterArrayW(self.counter, fmt, ctypes.byref(size), ctypes.byref(count), buffer) != 0 \
                    or count.value * ctypes.sizeof(_CounterItem) > len(buffer):
                self.status = "unavailable"
                return None
            items = ctypes.cast(buffer, ctypes.POINTER(_CounterItem))
            result = engine_utilization(((items[i].name, items[i].data.status, items[i].data.value)
                                         for i in range(count.value)), self.luid)
            self.status = "available" if result is not None else "warming_up"
            return result
        except (OSError, AttributeError, ValueError, TypeError):
            self.status = "unavailable"
            return None

    def close(self):
        if self.dll and self.query:
            try:
                self.dll.PdhCloseQuery(self.query)
            except (OSError, AttributeError):
                pass
        self.query, self.counter = ctypes.c_void_p(), ctypes.c_void_p()
        self.status = "unsupported"


class _Luid(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_int32)]


class _OpenAdapter(ctypes.Structure):
    _fields_ = [("luid", _Luid), ("handle", ctypes.c_uint32)]


class _CloseAdapter(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32)]


class _QueryAdapter(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("kind", ctypes.c_uint32),
                ("data", ctypes.c_void_p), ("size", ctypes.c_uint32)]


class _Performance(ctypes.Structure):
    # D3DKMT_ADAPTER_PERFDATA (Windows SDK d3dkmthk.h). In particular, Power
    # is 0.1% and Temperature is 0.1 C; they are not interchangeable sensors.
    _fields_ = [("node", ctypes.c_uint32), ("memory_frequency", ctypes.c_uint64),
                ("max_memory_frequency", ctypes.c_uint64), ("oc_memory_frequency", ctypes.c_uint64),
                ("memory_bandwidth", ctypes.c_uint64), ("pcie_bandwidth", ctypes.c_uint64),
                ("fan", ctypes.c_uint32), ("power_percent", ctypes.c_uint32),
                ("temperature", ctypes.c_uint32), ("power_override", ctypes.c_ubyte)]


class _PnpKey(ctypes.Structure):
    _fields_ = [("node", ctypes.c_uint32), ("kind", ctypes.c_uint32),
                ("dest", ctypes.c_void_p), ("length", ctypes.POINTER(ctypes.c_uint32))]


class _Guid(ctypes.Structure):
    _fields_ = [("a", ctypes.c_uint32), ("b", ctypes.c_uint16), ("c", ctypes.c_uint16),
                ("d", ctypes.c_ubyte * 8)]


class _PropertyKey(ctypes.Structure):
    _fields_ = [("guid", _Guid), ("pid", ctypes.c_uint32)]


def _device_instance(path):
    if not isinstance(path, str):
        return None
    at = path.lower().find("\\enum\\")
    if at < 0:
        return None
    value = path[at + len("\\enum\\"):]
    suffix = "\\device parameters"
    if value.lower().endswith(suffix):
        value = value[:-len(suffix)]
    return value if value.lower().startswith("pci\\") and value.count("\\") == 2 else None


class _AdapterProperties:
    def __init__(self, luid, allowed=True):
        self.dll, self.cfg = None, None
        self.handle, self.devnode = 0, None
        self.temperature_status = self.pcie_status = "unsupported"
        self._ever_temperature = False
        parts = _luid_parts(luid)
        if os.name != "nt" or not allowed or parts is None:
            return
        try:
            d = self.dll = ctypes.WinDLL("gdi32.dll", winmode=0x800)
            for name, record in (("D3DKMTOpenAdapterFromLuid", _OpenAdapter),
                                 ("D3DKMTQueryAdapterInfo", _QueryAdapter), ("D3DKMTCloseAdapter", _CloseAdapter)):
                getattr(d, name).argtypes = [ctypes.POINTER(record)]
                getattr(d, name).restype = ctypes.c_int32
            hi, lo = parts
            opened = _OpenAdapter(_Luid(lo, ctypes.c_int32(hi).value), 0)
            if d.D3DKMTOpenAdapterFromLuid(ctypes.byref(opened)) < 0:
                return
            self.handle = opened.handle
            # Do not silently query physical node 0 on a linked adapter.
            count = ctypes.c_uint32()
            if self._query(30, count) < 0 or count.value != 1:
                self.close()
                return
            self.temperature_status = "unavailable"
            buf, length = ctypes.create_unicode_buffer(2048), ctypes.c_uint32(2048)
            record = _PnpKey(0, 1, ctypes.addressof(buf), ctypes.pointer(length))
            if self._query(41, record) < 0 or not 0 < length.value <= len(buf):
                return
            instance = _device_instance(buf.value)
            if instance is None:
                return
            cfg = self.cfg = ctypes.WinDLL("cfgmgr32.dll", winmode=0x800)
            cfg.CM_Locate_DevNodeW.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_wchar_p, ctypes.c_uint32]
            cfg.CM_Locate_DevNodeW.restype = ctypes.c_uint32
            cfg.CM_Get_DevNode_PropertyW.argtypes = [ctypes.c_uint32, ctypes.POINTER(_PropertyKey),
                ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
            cfg.CM_Get_DevNode_PropertyW.restype = ctypes.c_uint32
            node = ctypes.c_uint32()
            if cfg.CM_Locate_DevNodeW(ctypes.byref(node), instance, 0) == 0:
                self.devnode = node.value
                self.pcie_status = "unavailable"
        except (OSError, AttributeError, ValueError, TypeError):
            # PCI property discovery is optional; keep a usable D3DKMT handle
            # for temperature even if PNP/configuration APIs are unavailable.
            if self.temperature_status == "unsupported":
                self.close()
            self.pcie_status = "unsupported"

    def _query(self, kind, record):
        if not self.dll or not self.handle:
            return -1
        query = _QueryAdapter(self.handle, kind, ctypes.addressof(record), ctypes.sizeof(record))
        return self.dll.D3DKMTQueryAdapterInfo(ctypes.byref(query))

    def _property(self, pid):
        if self.cfg is None or self.devnode is None:
            return None
        # SDK pciprop.h: DEVPKEY_PciDevice_CurrentLinkSpeed/Width (9/10),
        # MaxLinkSpeed/Width (11/12). Never replace current with maximum.
        key = _PropertyKey(_Guid.from_buffer_copy(uuid.UUID("3ab22e31-8264-4b4e-9af5-a8d2d8e33e62").bytes_le), pid)
        kind, value, size = ctypes.c_uint32(), ctypes.c_uint32(), ctypes.c_uint32(4)
        if self.cfg.CM_Get_DevNode_PropertyW(self.devnode, ctypes.byref(key), ctypes.byref(kind),
                                           ctypes.byref(value), ctypes.byref(size), 0) != 0 \
                or kind.value != 7 or size.value != 4:  # DEVPROP_TYPE_UINT32
            return None
        return value.value

    def read(self):
        out = {}
        if self.handle:
            try:
                record = _Performance()
                status = self._query(62, record)
                if status >= 0 and 0 < record.temperature < 2000:
                    out["temp"] = record.temperature / 10.0
                    self.temperature_status, self._ever_temperature = "available", True
                elif (status & 0xffffffff) in (0xc0000002, 0xc000000d, 0xc00000bb) or \
                        (status >= 0 and not self._ever_temperature):
                    self.temperature_status = "unsupported"
                else:
                    self.temperature_status = "unavailable"
                # Power is tenths of a percentage, not watts. PCIEBandwidth's
                # interval/direction is unspecified; neither is presented as W/RX.
            except (OSError, AttributeError, ValueError, TypeError):
                self.temperature_status = "unavailable"
        if self.devnode is not None:
            try:
                for pid, key in ((9, "pcie_gen"), (10, "pcie_width"),
                                 (11, "pcie_gen_max"), (12, "pcie_width_max")):
                    value = self._property(pid)
                    valid = value in (1, 2, 3, 4, 5, 6, 7) if "gen" in key else value in (1, 2, 4, 8, 12, 16, 32)
                    if valid:
                        out[key] = value
                self.pcie_status = "available" if "pcie_gen" in out or "pcie_width" in out else "unsupported"
            except (OSError, AttributeError, ValueError, TypeError):
                self.pcie_status = "unavailable"
        return out

    def close(self):
        if self.dll and self.handle:
            try:
                self.dll.D3DKMTCloseAdapter(ctypes.byref(_CloseAdapter(self.handle)))
            except (OSError, AttributeError):
                pass
        self.handle, self.devnode = 0, None
        self.temperature_status = self.pcie_status = "unsupported"


class WindowsGpuCounters:
    """Read-only common OS fallback, with explicit per-field availability."""
    def __init__(self, luid, node_mask=None):
        allowed = node_mask in (None, 0, 1)
        self.usage = _EngineUsage(luid, allowed)
        self.adapter = _AdapterProperties(luid, allowed)

    def read(self):
        out = self.adapter.read()
        utilization = self.usage.read()
        if utilization is not None:
            out["util"] = utilization
        out["metric_status"] = {"util": self.usage.status, "temp": self.adapter.temperature_status,
                                "pcie": self.adapter.pcie_status}
        out["metric_sources"] = {"util": "Windows PDH", "temp": "Windows D3DKMT", "pcie": "Windows PCI properties"}
        return out

    def close(self):
        self.usage.close()
        self.adapter.close()
