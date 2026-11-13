"use strict";
// Actual UI preference, request, metadata and slider code. No network or model.
const fs = require("node:fs"), path = require("node:path"), vm = require("node:vm"), assert = require("node:assert/strict");
const source = fs.readFileSync(path.join(__dirname, "../ui/web/app.js"), "utf8");
let checks = 0;
const check = (value, label) => { assert(value, label); checks++; };
const plain = (value) => JSON.parse(JSON.stringify(value));
const equal = (a, b, label) => { assert.deepEqual(plain(a), b, label); checks++; };
function block(start, end) { const a = source.indexOf(start), b = source.indexOf(end, a); assert(a >= 0 && b > a); return source.slice(a, b); }
function fixture() {
  const nodes = new Map();
  const $ = (id) => { if (!nodes.has(id)) nodes.set(id, {value: "", dataset: {}, children: [], disabled: false}); return nodes.get(id); };
  const c = {health: {status: "ok", model: "qwen3.6-35b-a3b"}, $,
    DEFAULTS: {temperature: 1, top_p: .95, top_k: 20},
    tr: (_, text) => text, segValue: () => "wait", headers: () => ({}),
    thinkingPayload: () => ({enable_thinking: true}), projectionLoaded: () => false};
  vm.createContext(c);
  vm.runInContext(block("function runtimeModelId()", "function apiMessageSerializer("), c);
  vm.runInContext(block("const OLD_DEFAULT_TEMPERATURE", "let settings ="), c);
  vm.runInContext(block("function syncSamplingInputs(", "function renderLocalToolStatus("), c);
  vm.runInContext(block("function outputs()", 'for (const id of ["s-math"'), c);
  vm.runInContext(block("function sharedDefaults(", "async function saveShared("), c);
  const payload = block("  if (!samplingUntouched(settings))", "  if (settings.seed)");
  vm.runInContext(`function requestSampling(settings) { const body = {}; ${payload} return body; }`, c);
  c.settings = c.migrateSampling({...c.DEFAULTS});
  const receive = (temperature, model = c.health.model, top_p = .95, top_k = 20) => c.receiveSamplingDefaults(model,
    {model_alias: model, default_generation_settings: {params: {temperature, top_p, top_k}}});
  return {c, $, receive, shown: () => [$("s-temp").value, $("s-topp").value, $("s-topk").value].map(Number)};
}
async function main() {
  let f = fixture(), c = f.c;
  check(c.samplingUntouched(c.settings), "fresh settings use model mode");
  equal(c.requestSampling(c.settings), {}, "untouched generation omits sampling overrides");
  equal(c.sharedDefaults(c.settings), {enable_thinking: true}, "model mode also omits overrides when sharing defaults");
  c.syncSamplingInputs();
  check(f.$("o-temp").textContent === "—" && f.$("s-temp").disabled, "unknown backend defaults are not displayed as invented1.0");
  check(f.receive(.7), "effective saved server0.7 metadata is accepted"); c.syncSamplingInputs();
  equal(f.shown(), [.7, .95, 20], "drawer shows actual0.7 despite legacy browser1.0 sentinel");
  check(!f.$("s-temp").disabled && f.$("o-temp").textContent === "0.70", "known temperature is editable and accurately labeled");
  f.$("s-temp").value = "1"; f.$("s-temp").oninput();
  const custom = c.storedSampling({temperature: 1, top_p: .95, top_k: 20}, f.$("s-temp").dataset.samplingMode);
  equal(c.requestSampling(custom), {temperature: 1, top_p: .95, top_k: 20}, "manually selecting1.0 sends the literal former sentinel tuple");
  equal(c.sharedDefaults(custom), {enable_thinking: true, temperature: 1, top_p: .95, top_k: 20}, "explicit1.0 is also honored for shared API defaults");
  check(!c.samplingUntouched(c.migrateSampling(plain(custom))), "custom1.0 survives storage/reload without becoming automatic");
  const explicit07 = c.migrateSampling({...c.DEFAULTS, temperature: .7});
  check(explicit07.samplingMode === "custom", "existing non-sentinel0.7 browser customization is retained");
  check(c.migrateSampling({...c.DEFAULTS}).samplingMode === "model", "legacy numeric sentinel migrates once to model mode");
  check(c.migrateSampling({...c.DEFAULTS, temperature: .6, samplingMode: "custom"}).samplingMode === "custom", "explicit legacy-valued0.6 remains custom after migration");
  equal(c.requestSampling({...custom, temperature: 0}), {temperature: 0}, "explicit zero retains greedy request behavior");

  f.$("drawer").dataset.open = "true";
  f.receive(.8); c.syncSamplingInputs();
  equal(f.shown(), [1, .95, 20], "late metadata cannot erase unapplied custom slider values");
  c.syncSamplingInputs(c.DEFAULTS, true);
  equal(f.shown(), [.8, .95, 20], "Reset stages actual current defaults rather than a hardcoded tuple");
  const reset = c.storedSampling({temperature: .8, top_p: .95, top_k: 20}, f.$("s-temp").dataset.samplingMode);
  equal(c.requestSampling(reset), {}, "applying Reset resumes following backend defaults");
  equal(c.requestSampling({...reset, temperature: 0}), {}, "automatic greedy defaults never force later models into greedy sampling");
  c.health = {status: "ok", model: "lfm2.5-24b-a2b"}; f.receive(.1, c.health.model, 1, 50); c.syncSamplingInputs();
  equal(f.shown(), [.1, 1, 50], "automatic values change with the authoritative model");
  c.health = {status: "ok", model: "clover-35b-a3b-preview"}; f.receive(0, c.health.model, 1, 64); c.syncSamplingInputs();
  check(f.$("o-temp").textContent === "0 · greedy" && !f.$("s-temp").disabled && f.$("s-topp").disabled, "actual greedy defaults display0 while retaining the temperature control");
  check(!c.receiveSamplingDefaults(c.health.model, {model_alias: "wrong", default_generation_settings: {params: {temperature: 1, top_p: .95, top_k: 20}}}), "metadata for another model is rejected");
  f.receive(.733, c.health.model, .01, 100); c.syncSamplingInputs();
  equal(f.shown(), [.733, .01, 100], "valid custom defaults are shown without rounding or cutoff clamping");
  check(f.$("s-temp").step === "any" && f.$("s-topp").step === "any" && Number(f.$("s-topp").min) <= .01 && Number(f.$("s-topk").max) >= 100,
    "native range bounds and step preserve custom metadata precision");
  check(f.$("o-temp").textContent === "0.733" && f.$("o-topp").textContent === "0.01", "numeric labels preserve effective precision");

  // Real fetch helper, with deferred CPU-only responses and a model switch.
  f = fixture(); c = f.c; const pending = [], calls = [];
  c.fetch = (url, options) => { calls.push({url, options}); return new Promise((resolve) => pending.push(resolve)); };
  const first = c.loadSamplingDefaults();
  c.health = {status: "ok", model: "gemma-4-26b-a4b-it"}; const second = c.loadSamplingDefaults();
  pending[1]({ok: true, json: async () => ({model_alias: c.health.model, default_generation_settings: {params: {temperature: .9, top_p: .8, top_k: 64}}})});
  await second;
  pending[0]({ok: true, json: async () => ({model_alias: "qwen3.6-35b-a3b", default_generation_settings: {params: {temperature: .7, top_p: .95, top_k: 20}}})});
  await first;
  equal(f.shown(), [.9, .8, 64], "late previous-model response cannot overwrite current model defaults");
  check(calls.every((call) => call.options.cache === "no-store") && calls[1].url.includes("gemma-4-26b-a4b-it"), "requests explicitly identify the model and bypass stale caches");
  c.fetch = async () => ({ok: true, json: async () => ({model_alias: "qwen3.6-35b-a3b", default_generation_settings: {params: {temperature: 1, top_p: .95, top_k: 20}}})});
  await c.loadSamplingDefaults();
  check(f.$("o-temp").textContent === "—", "wrong-model metadata invalidates automatic display instead of presenting stale values");

  // An in-flight default fetch can finish after Reset but before Apply. Reset
  // is a drawer draft: the old saved custom preference must not restore itself.
  f = fixture(); c = f.c; let resolve;
  c.settings = c.migrateSampling({...c.DEFAULTS, temperature: .4});
  f.$("drawer").dataset.open = "true"; c.syncSamplingInputs(c.settings, true);
  c.fetch = () => new Promise((done) => { resolve = done; });
  const inflight = c.loadSamplingDefaults();
  c.syncSamplingInputs(c.DEFAULTS, true);
  resolve({ok: true, json: async () => ({model_alias: c.health.model, default_generation_settings: {params: {temperature: .73, top_p: .93, top_k: 33}}})});
  await inflight;
  equal(f.shown(), [.73, .93, 33], "late defaults refresh honors staged Reset instead of restoring saved custom values");
  check(f.$("s-temp").dataset.samplingMode === "model" && c.settings.samplingMode === "custom", "Reset remains staged independently of unchanged saved preferences");
  f = fixture(); c = f.c; let fetches = 0;
  c.fetch = async () => { fetches++; return {ok: true, json: async () => ({model_alias: c.health.model,
    default_generation_settings: {params: {temperature: 1, top_p: .95, top_k: 20}}})}; };
  for (const health of [{model: "strata"}, {status: "ok", model: "strata"}, {status: "switching", model: "qwen3.6-35b-a3b"}]) {
    c.health = health; await c.loadSamplingDefaults();
  }
  check(fetches === 0 && f.$("o-temp").textContent === "—", "bootstrap and unvalidated health never request unsupported props");
  c.health = {status: "ok", model: "qwen3.6-35b-a3b"}; await c.loadSamplingDefaults();
  check(fetches === 1 && f.$("o-temp").textContent === "1.00", "validated model health subsequently loads and displays actual defaults");
  console.log(`PASS ${checks} sampling default/display/payload checks (no network/model)`);
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
