"""Read-only DXGI identity and LUID-scoped Windows adapter memory counters.

Shared Usage is system RAM accounting, never an additional VRAM capacity.
"""
from __future__ import annotations

import ctypes
import os
import uuid


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


def dxgi_adapters() -> list[dict]:
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
                                  "vendor_id": desc.vendor, "device_id": desc.device, "flags": desc.flags,
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


class AdapterMemoryUsage:
    """Adapter-wide instant gauge, matched by LUID rather than display order.

    This is the same counter used by hip_compat/dedicated_vram.hpp when DXCore
    state queries are unavailable. English counter names also work on Japanese
    Windows; no subprocess or vendor DLL is needed for each sample.
    """
    def __init__(self, luid, node_mask, counter="Dedicated Usage"):
        if counter not in ("Dedicated Usage", "Shared Usage"):
            raise ValueError("Unknown GPU memory counter")
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
            if d.PdhAddEnglishCounterW(self.query, rf"\GPU Adapter Memory(*)\{counter}", 0,
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
