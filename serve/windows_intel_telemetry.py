"""The actual Intel Level Zero adapter's Windows UMA readings.

Startup INFO comes from the inference process. No GPU runtime is loaded here;
exact LUID identity gates every OS counter. Adapter shared usage overlaps RAM
and includes other processes. It is neither model usage nor dedicated VRAM.
"""
from __future__ import annotations

import math
import re

from serve.windows_gpu_memory import AdapterMemoryUsage, dxgi_adapters
from serve.windows_gpu_counters import WindowsGpuCounters


def selected_adapter(info, adapters):
    """Fail closed for missing identity, duplicate LUIDs and unsupported nodes."""
    if not isinstance(info, dict) or info.get("backend") != "sycl" or info.get("device_vendor") != 0x8086 \
            or info.get("device_integrated") != 1 or info.get("memory_model") != "uma":
        return None
    luid = info.get("adapter_luid")
    if not isinstance(luid, str) or not re.fullmatch(r"[0-9a-fA-F]{8}:[0-9a-fA-F]{8}", luid) \
            or info.get("adapter_node_mask") != 1:
        return None
    matches = [a for a in adapters if isinstance(a, dict) and
               isinstance(a.get("luid"), str) and a["luid"].lower() == luid.lower()]
    if len(matches) != 1:
        return None
    adapter = matches[0]
    if adapter.get("vendor_id") != 0x8086 or adapter.get("device_id") != info.get("device_id") \
            or adapter.get("flags", 0) & 2 or not isinstance(adapter.get("name"), str) \
            or not adapter["name"].strip():
        return None
    return dict(adapter, luid=luid.lower(), node_mask=1)


class WindowsIntelCard:
    total = None  # UMA is not a separate dedicated-VRAM capacity.

    def __init__(self, info):
        self.identity = selected_adapter(info, dxgi_adapters())
        self._usage = self._counters = None
        if self.identity:
            luid, node = self.identity["luid"], self.identity["node_mask"]
            self._usage = AdapterMemoryUsage(luid, node, "Shared Usage")
            self._counters = WindowsGpuCounters(luid, node)

    def ok(self):
        return self.identity is not None

    def name(self):
        return self.identity["name"] if self.identity else None

    def read(self):
        out = {"mem_used": None, "mem_total": None, "shared_mem_used": None,
               "memory_usage_scope": "adapter"}
        try:
            used = self._usage.read() if self._usage else None
            if isinstance(used, int) and not isinstance(used, bool) and used >= 0:
                out["shared_mem_used"] = used
        except Exception:  # optional telemetry must never stop inference
            pass
        try:
            reading = self._counters.read() if self._counters else {}
        except Exception:
            reading = {}
        if not isinstance(reading, dict):
            reading = {}
        # UMA's memory path is not a host-to-discrete-GPU PCIe link. Do not
        # relabel PCI configuration metadata as the model's transfer path.
        for key in ("util", "temp"):
            value = reading.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                out[key] = value
        states = reading.get("metric_status") or {}
        out["metric_status"] = {
            "util": states.get("util", "unavailable"), "temp": states.get("temp", "unavailable"),
            "power": "unsupported", "pcie": "unsupported", "mem_used": "unsupported",
            "shared_mem_used": "available" if out["shared_mem_used"] is not None else "unavailable"}
        out["metric_sources"] = {"shared_mem_used": "Windows PDH (adapter; overlaps system RAM)",
                                 "util": "Windows PDH", "temp": "Windows D3DKMT"}
        return out

    def close(self):
        for reader in (self._usage, self._counters):
            if reader:
                reader.close()
