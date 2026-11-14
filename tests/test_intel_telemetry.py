"""CPU-only identity/counter fixtures; no GPU runtime, inference or browser."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from serve import telemetry as T
from serve import windows_gpu_memory as M
from serve import windows_intel_telemetry as I


ROOT = Path(__file__).resolve().parents[1]
NODE = os.environ.get("TINYSTRATA_TEST_NODE") or shutil.which("node")
LUID = "00000001:000000ab"
INFO = {"backend": "sycl", "device_vendor": 0x8086, "device_id": 1234,
        "device_integrated": 1, "memory_model": "uma", "adapter_luid": LUID,
        "adapter_node_mask": 1}
ADAPTER = {"name": "actual selected Intel GPU", "vendor_id": 0x8086,
           "device_id": 1234, "luid": LUID, "flags": 0, "mem_total": 256 << 20}


class IntelIdentityTests(unittest.TestCase):
    def test_exact_identity_not_adapter_order_or_marketing_name(self):
        other = dict(ADAPTER, luid="00000002:000000ab", name="Intel B390 first")
        selected = I.selected_adapter(dict(INFO, adapter_luid=LUID.upper()), [other, ADAPTER])
        self.assertEqual(selected["name"], ADAPTER["name"])
        self.assertEqual(selected["luid"], LUID)

    def test_missing_ambiguous_wrong_and_linked_identity_fail_closed(self):
        for changes in ({"backend": "cuda"}, {"device_integrated": 0}, {"device_integrated": None},
                        {"memory_model": None}, {"device_vendor": 0x10de}, {"adapter_luid": None},
                        {"adapter_luid": "unknown"}, {"adapter_node_mask": 0}, {"adapter_node_mask": 2}):
            with self.subTest(changes=changes):
                self.assertIsNone(I.selected_adapter(dict(INFO, **changes), [ADAPTER]))
        for cards in ([], [ADAPTER, ADAPTER], [dict(ADAPTER, device_id=999)],
                      [dict(ADAPTER, vendor_id=0x10de)], [dict(ADAPTER, flags=2)]):
            with self.subTest(cards=cards):
                self.assertIsNone(I.selected_adapter(INFO, cards))


class PdhFixture:
    """A real ctypes PDH array with fake system calls, including foreign LUIDs."""
    def __init__(self, entries):
        self.entries = entries
        self.paths = []
        self.dll = SimpleNamespace(
            PdhOpenQueryW=Mock(side_effect=self.open), PdhAddEnglishCounterW=Mock(side_effect=self.add),
            PdhCollectQueryData=Mock(return_value=0),
            PdhGetFormattedCounterArrayW=Mock(side_effect=self.array), PdhCloseQuery=Mock(return_value=0))

    def open(self, source, user, query):
        ctypes.cast(query, ctypes.POINTER(ctypes.c_void_p))[0] = 1
        return 0

    def add(self, query, path, user, counter):
        self.paths.append(path)
        ctypes.cast(counter, ctypes.POINTER(ctypes.c_void_p))[0] = 2
        return 0

    def array(self, counter, fmt, size, count, buffer):
        values = (M._PdhItem * len(self.entries))()
        for row, (name, status, value) in zip(values, self.entries):
            row.name, row.data.status, row.data.value = name, status, value
        ctypes.cast(size, ctypes.POINTER(ctypes.c_uint32))[0] = ctypes.sizeof(values)
        ctypes.cast(count, ctypes.POINTER(ctypes.c_uint32))[0] = len(values)
        if buffer is None:
            return 0x800007d2
        ctypes.memmove(buffer, values, ctypes.sizeof(values))
        self.values = values  # retain string storage until the consumer reads it
        return 0


class MemoryCounterTests(unittest.TestCase):
    def read_fixture(self, entries, counter="Shared Usage"):
        fixture = PdhFixture(entries)
        with patch.object(M, "os", SimpleNamespace(name="nt")), \
                patch.object(M.ctypes, "WinDLL", return_value=fixture.dll, create=True):
            reader = M.AdapterMemoryUsage(LUID, 1) if counter is None else M.AdapterMemoryUsage(LUID, 1, counter)
            value = reader.read()
            reader.close()
        return value, fixture

    def test_shared_usage_uses_only_selected_adapter_and_keeps_zero(self):
        selected = "luid_0x00000001_0x000000ab_phys_0"
        foreign = "luid_0x00000002_0x000000ab_phys_0"
        for expected in (0, 12 << 30):
            value, fixture = self.read_fixture([(foreign, 0, 99 << 30), (selected, 0, expected)])
            self.assertEqual(value, expected)
            self.assertEqual(fixture.paths, [r"\GPU Adapter Memory(*)\Shared Usage"])
            fixture.dll.PdhCloseQuery.assert_called_once()

    def test_duplicate_invalid_or_additional_node_is_unavailable(self):
        selected = "luid_0x00000001_0x000000ab_phys_0"
        other_node = "luid_0x00000001_0x000000ab_phys_1"
        for entries in ([(selected, 0, 10), (selected, 0, 20)], [(selected, 5, 10)],
                        [(selected, 0, -1)], [(selected, 0, 10), (other_node, 0, 20)]):
            with self.subTest(entries=entries):
                self.assertIsNone(self.read_fixture(entries)[0])

    def test_existing_amd_counter_default_remains_dedicated_usage(self):
        value, fixture = self.read_fixture([("luid_0x00000001_0x000000ab_phys_0", 0, 15)], None)
        self.assertEqual(value, 15)
        self.assertEqual(fixture.paths, [r"\GPU Adapter Memory(*)\Dedicated Usage"])
        from serve.windows_amd_telemetry import _DedicatedUsage
        self.assertIs(_DedicatedUsage, M.AdapterMemoryUsage)


class IntelSamplerTests(unittest.TestCase):
    def make_card(self, usage=12 << 30):
        counters = Mock()
        counters.read.return_value = {"util": 50, "temp": None, "mem_total": 64 << 30,
                                      "power": 99, "pcie_gen": 5,
                                      "metric_status": {"util": "available", "temp": "unsupported"}}
        with patch.object(I, "dxgi_adapters", return_value=[ADAPTER]), \
                patch.object(I, "AdapterMemoryUsage") as memory, \
                patch.object(I, "WindowsGpuCounters", return_value=counters):
            memory.return_value.read.return_value = usage
            card = I.WindowsIntelCard(INFO)
            memory.assert_called_once_with(LUID, 1, "Shared Usage")
        return card

    def test_shared_usage_is_separate_from_null_legacy_vram(self):
        card = self.make_card()
        reading = card.read()
        self.assertTrue(card.ok())
        self.assertIsNone(card.total)
        self.assertIsNone(reading["mem_total"])
        self.assertIsNone(reading["mem_used"])
        self.assertEqual(reading["shared_mem_used"], 12 << 30)
        self.assertEqual(reading["memory_usage_scope"], "adapter")
        self.assertNotIn("power", reading)
        self.assertNotIn("pcie_gen", reading)
        self.assertEqual(reading["metric_status"]["temp"], "unsupported")

    def test_counter_failure_is_null_not_zero(self):
        for value in (None, -1, float("nan"), True):
            with self.subTest(value=value):
                reading = self.make_card(value).read()
                self.assertIsNone(reading["shared_mem_used"])
                self.assertEqual(reading["metric_status"]["shared_mem_used"], "unavailable")
        self.assertEqual(self.make_card(0).read()["shared_mem_used"], 0)

    def test_one_ram_pool_and_no_nvml_fallback(self):
        card = self.make_card()
        with patch.object(I, "WindowsIntelCard", return_value=card), \
                patch.object(T, "gpu_reader", side_effect=AssertionError("wrong vendor fallback")), \
                patch.object(T.threading, "Thread"):
            sampler = T.Telemetry(intel_info=INFO)
        sampler.ps = SimpleNamespace(cpu_percent=lambda **kw: 10,
                                     virtual_memory=lambda: SimpleNamespace(total=32 << 30, available=10 << 30),
                                     disk_io_counters=lambda: None)
        reading = sampler.sample()
        self.assertEqual(reading["ram_total"], 32 << 30)
        self.assertEqual(reading["ram_used"], 22 << 30)
        self.assertEqual(reading["gpu_shared_mem_used"], 12 << 30)
        self.assertIsNone(reading["gpu_mem_total"])
        self.assertIsNone(sampler.static["gpu_vram_total_mib"])
        self.assertEqual(sampler.static["gpu_adapter_luid"], LUID)
        self.assertEqual(reading["gpu_metric_status"]["shared_mem_used"], "available")

    def test_missing_or_broken_intel_reader_never_uses_nvml(self):
        with patch.object(T, "gpu_reader", side_effect=AssertionError("wrong vendor fallback")), \
                patch.object(T.threading, "Thread"), patch.object(I, "dxgi_adapters", return_value=[]):
            sampler = T.Telemetry(intel_info={})
            self.assertIsNone(sampler.static["gpu_name"])
            self.assertIsNone(sampler.static["gpu_vram_total_mib"])
            missing = sampler.sample()
            self.assertIsNone(missing["gpu_mem_total"])
            self.assertIsNone(missing["gpu_shared_mem_used"])
            with patch.object(I, "WindowsIntelCard", side_effect=OSError("no counters")):
                failed = T.Telemetry(intel_info=INFO)
                self.assertIsNone(failed.static["gpu_name"])

    def test_existing_non_intel_reader_selection_and_capacity_remain(self):
        card = Mock()
        card.ok.return_value = True
        card.name.return_value = "existing discrete card"
        card.total = 24 << 30
        with patch.object(T, "gpu_reader", return_value=card) as reader, patch.object(T.threading, "Thread"):
            sampler = T.Telemetry(gpu_index=2)
        reader.assert_called_once_with(2, False)
        self.assertEqual(sampler.static["gpu_vram_total_mib"], 24 << 10)
        self.assertNotIn("gpu_memory_model", sampler.static)

    def test_service_passes_actual_native_info_only_for_intel(self):
        from serve.server import Service
        for backend in ("sycl", "cuda", "hip"):
            service = object.__new__(Service)
            service.backend = backend
            service.engine = SimpleNamespace(info=INFO)
            with patch.object(T, "Telemetry") as sampler:
                service.start_telemetry()
                actual = sampler.call_args.kwargs["intel_info"]
                self.assertEqual(actual, INFO if backend == "sycl" else None)
                if actual is not None:
                    self.assertIsNot(actual, INFO)

    def test_public_status_preserves_null_vram_and_reports_shared_scope(self):
        from serve.server import Service
        service = object.__new__(Service)
        service.status_lock = threading.Lock()
        service.status, service.totals = {}, {"requests": 0}
        service.last_timings = service.last_request_at = None
        service.engine = SimpleNamespace(info={}, batch=1)
        service.model, service.started_at = "qwen3.6-35b-a3b", time.time()
        service.vision = service.images = None
        service.reported_ctx, service.vision_available, service.loaded = lambda: 4096, lambda: False, lambda: True
        service.telemetry = SimpleNamespace(snapshot=lambda: {
            "static": {"gpu_name": ADAPTER["name"], "gpu_memory_model": "uma"},
            "now": {"gpu_mem_total": None, "gpu_mem_used": None, "gpu_shared_mem_used": 12 << 30,
                    "ram_total": 32 << 30, "ram_used": 22 << 30}})
        machine = service.v1_status()["machine"]
        self.assertIsNone(machine["gpu"]["total_mib"])
        self.assertIsNone(machine["gpu"]["used_mib"])
        self.assertEqual(machine["gpu"]["shared_used_mib"], 12 << 10)
        self.assertTrue(machine["gpu"]["shared_overlaps_ram"])
        self.assertEqual(machine["gpu"]["memory_usage_scope"], "adapter")
        self.assertEqual(machine["ram"], {"used_gib": 22, "total_gib": 32})


class IntelMemoryUiTests(unittest.TestCase):
    @unittest.skipUnless(NODE, "Node.js is required for frontend CPU checks")
    def test_uma_ignores_vram_capacity_and_does_not_invent_expert_counts(self):
        source = (ROOT / "ui/web/app.js").read_text(encoding="utf-8")
        helpers = source[source.index("function umaMemory("):source.index("let lastMetrics =")]
        about = source[source.index("function renderAbout("):source.index('document.addEventListener("click"', source.index("function renderAbout("))]
        monitor = source[source.index("function renderMonitor("):source.index("function facts(")]
        placement = source[source.index("function expertPlacement("):source.index("// A committed prefix position")]
        script = r'''
const assert = require("node:assert/strict");
const tr = (key, text) => text, fmt = String, esc = String, gb = (v) => String(v / 2**30);
const nodes = new Map();
const element = (id) => ({id, dataset: {}, style: {}, title: "", parentElement: {},
  previousElementSibling: {removeAttribute() {}}, removeAttribute() {}, setAttribute() {}, classList: {toggle() {}}});
const $ = (id) => { if (!nodes.has(id)) nodes.set(id, element(id)); return nodes.get(id); };
$("req-body").closest = (selector) => {
  assert.equal(selector, "table");
  return {querySelector: (selector) => {
    assert.equal(selector, '[data-i18n="monitor.hitRate"]');
    return $("request-hit-header");
  }};
};
const location = {origin: "http://localhost"}, document = {querySelectorAll: () => []};
const StrataI18n = {lang: "en"}, statusIcon = () => "";
const cloverGenerationText = () => null, projectionText = () => null;
const output = {}, facts = (el, rows) => { output[el.id] = rows.filter((r) => r[1] != null && r[1] !== ""); };
const metrics = {}, setGpuMetric = (key, value, unit, sub) => {metrics[key] = {value, unit, sub};};
const setMetric = () => {}, spark = () => {}, imageLive = () => false, modelOp = null, busy = null;
const modelSwitching = false, placementSampleValid = true, contextUsage = () => null, reqShowAll = false;
const gpuMetricState = () => ({label: "Unavailable", hint: ""}), setStableHtml = (el, html) => {el.innerHTML = html;}, renderTotals = () => "";
'''+helpers+about+monitor+placement+r'''
const hw = {gpu_mem_total: 64*2**30, gpu_mem_used: 9*2**30, gpu_shared_mem_used: 12*2**30, ram_total: 32*2**30, ram_used: 22*2**30};
const eng = {memory_model: "uma", kv: "int8", expert_slots: 999, expert_cache_mib: 999999};
const st = {gpu_name: "Selected Intel GPU", gpu_memory_model: "uma"};
const history = {gpu_mem_used: [9], gpu_shared_mem_used: [12]};
const memory = gpuMemoryReading(eng, hw, st, history);
assert.equal(memory.total, null);
assert.equal(memory.used, 12*2**30);
assert.deepEqual(memory.history, [12]);
renderAbout(eng, hw, st);
assert.ok(!JSON.stringify(output).includes("VRAM"));
assert.ok(!JSON.stringify(output).includes("999"));
assert.ok(JSON.stringify(output).includes("shared system RAM"));
assert.equal(output["facts-hw"].find((r) => r[0] === "GPU")[1], "Selected Intel GPU");
const discrete = gpuMemoryReading({}, hw, {}, history);
assert.equal(discrete.total, 64*2**30);
assert.equal(discrete.used, 9*2**30);
assert.equal(gpuMemoryReading(eng, {}, st, {}).used, undefined);
renderMonitor({state: "idle"}, hw, st, eng, history, null, [], null, 0);
assert.equal(metrics.vram.value, "12");
assert.equal(metrics.vram.unit, "GB");
assert.ok(metrics.vram.sub.includes("included in RAM"));
assert.equal($("ml-vram").textContent, "GPU shared RAM");
assert.equal($("slots-bar").parentElement.hidden, true);
assert.equal($("ram-text").textContent, "22 / 32 GB");
assert.equal($("request-hit-header").hidden, true);
assert.ok($("req-body").innerHTML.includes('colspan="8"'));
const request = {time: 1, finish: "stop", prompt_tokens: 84, reused: 0, output_tokens: 59,
  prompt_ms: 1000, decode_tok_s: 30, hit_rate: .5, duration_s: 3};
renderMonitor({state: "idle"}, hw, st, eng, history, null, [request], null, 1);
assert.equal(($("req-body").innerHTML.match(/<td\b/g) || []).length, 8);
assert.ok(!$("req-body").innerHTML.includes("50.0%"));
renderPlacement({engine: eng, hardware_static: st, live: {state: "idle"}});
assert.equal($("placement-track").hidden, true);
assert.ok(!$("chat-placement").title.includes("999"));
assert.ok(!$("chat-placement").title.includes("VRAM"));
renderMonitor({state: "idle"}, hw, {}, {}, history, null, [], null, 0);
assert.equal(metrics.vram.value, "9");
assert.equal(metrics.vram.unit, "/ 64 GB");
assert.equal($("ml-vram").textContent, "VRAM");
assert.equal($("slots-bar").parentElement.hidden, false);
assert.equal($("request-hit-header").hidden, false);
assert.ok($("req-body").innerHTML.includes('colspan="9"'));
renderMonitor({state: "idle"}, hw, {}, {}, history, null, [request], null, 1);
assert.equal(($("req-body").innerHTML.match(/<td\b/g) || []).length, 9);
assert.ok($("req-body").innerHTML.includes("50.0%"));
'''
        result = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_uma_locale_keys_are_available_in_english_and_japanese(self):
        for language in ("en", "ja"):
            catalog = json.loads((ROOT / f"ui/web/locales/{language}.json").read_text(encoding="utf-8"))
            for key in ("metrics.sharedRam", "monitor.sharedRamNote", "monitor.sharedRamHint", "placement.umaHint",
                        "about.umaKv", "about.memoryTopology", "about.umaMemory"):
                self.assertTrue(catalog[key])


if __name__ == "__main__":
    unittest.main()
