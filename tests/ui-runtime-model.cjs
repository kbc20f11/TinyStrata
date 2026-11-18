"use strict";
// Exercise the provided UI serializers with synthetic history; no network or model.
const assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path"), vm = require("node:vm");
const C = require("../ui/web/context-compaction.js");
const read = (name) => fs.readFileSync(path.join(__dirname, "..", "ui", "web", name), "utf8");
const app = read("app.js"), resources = read("resource-actions.js");
let checks = 0;
const check = (value, label) => { assert(value, label); checks++; };
const copy = (value) => JSON.parse(JSON.stringify(value));
function block(source, start, end) {
  const a = source.indexOf(start), b = source.indexOf(end, a);
  assert(a >= 0 && b > a, `missing source block: ${start}`);
  return source.slice(a, b);
}
function fixture() {
  const ctx = {
    health: {status: "ok", model: "qwen3.6-35b-a3b"},
    modelCatalog: [{id: "clover-35b-a3b-preview", active: true}],
    messages: [{role: "user", text: "あなたはどのようなモデルですか"}], busy: null,
    settings: {autoCompact: true}, chat: {}, StrataContextCompaction: C,
    clockNow: Date.parse("2026-11-18T14:59:59.900Z"), clockZone: "Asia/Tokyo", clockReads: 0,
    canUseModelImages: () => true, refreshTerminalStats() {}, saveChat() {},
    terminalMessageElement: () => null, tr: (_, fallback) => fallback,
    toast() {}, safeToolMedia: () => false, DOMException, AbortController,
  };
  vm.createContext(ctx);
  vm.runInContext(read("grounding.js"), ctx);
  const realDateContext = ctx.StrataGrounding.getCurrentDateContext;
  ctx.StrataGrounding.getCurrentDateContext = () => {
    ctx.clockReads++;
    return realDateContext({now: () => ctx.clockNow, defaultTimeZone: () => ctx.clockZone});
  };
  vm.runInContext(read("local-tools.js"), ctx);
  vm.runInContext(block(app, "function fileBlock(", "function renderAttachments("), ctx);
  vm.runInContext(block(app, "function assistantMessages(", "function busyHint("), ctx);
  vm.runInContext(block(app, "const UI_SYSTEM_PROMPT", "async function contextJson("), ctx);
  vm.runInContext(block(app, "async function countContext(", "// An answer that used MCP tools"), ctx);
  return ctx;
}
const identityLine = (messages) => messages[0].content.split("\n").find((line) => line.startsWith("The local TinyStrata runtime"));
const hasModel = (messages, model) => identityLine(messages) === `The local TinyStrata runtime reports the current model ID as ${JSON.stringify(model)}.`;
const dateLine = (messages) => messages[0].content.split("\n").find((line) => line.startsWith("Current date from the TinyStrata computer clock:"));

async function main() {
  // Unloaded Intel auto has no allocated capacity until the real /load returns.
  // A send may load once and re-count; reads and other backends never invent it.
  for (const first of [
    {backend: "sycl", context_pending: true, max_context: 0, model: "qwen3.6-35b-a3b"},
    {backend: "sycl", context_pending: false, max_context: 262144, model: "qwen3.6-35b-a3b"},
    {backend: "cuda", context_pending: true, max_context: 0, model: "qwen3.6-35b-a3b"},
    {backend: "sycl", context_pending: true, max_context: 0, model: "other"},
  ]) {
    const state = fixture(), calls = [], request = {model: "qwen3.6-35b-a3b", messages: []};
    state.contextJson = async (url, body) => {
      calls.push({url, body});
      return calls.length === 1 ? first : {model: request.model, max_context: 262144, prompt_tokens: 42};
    };
    const result = await state.countContext(request, new AbortController().signal);
    const load = first.backend === "sycl" && first.context_pending && first.model === request.model;
    check(calls.map((c) => c.url).join() === (load ? "v1/chat/token-count,load,v1/chat/token-count" : "v1/chat/token-count"), "only a pending matching Intel request loads and re-counts");
    check(load ? result.max_context === 262144 : result === first, "capacity comes from the actual count reply");
  }
  {
    const state = fixture(), calls = [], request = {model: "qwen3.6-35b-a3b"};
    state.contextJson = async (url) => {
      calls.push(url);
      if (url === "load") throw new DOMException("cancelled", "AbortError");
      return {backend: "sycl", context_pending: true, max_context: 0, model: request.model};
    };
    await assert.rejects(state.countContext(request, new AbortController().signal), {name: "AbortError"}); checks++;
    check(calls.join() === "v1/chat/token-count,load", "load cancellation does not loop or reuse guessed capacity");
  }
  let f = fixture(), original = copy(f.messages), wire = f.apiMessages();
  check(hasModel(wire, f.health.model), "current /health model supplies runtime identity despite stale catalog");
  check(wire[0].content.startsWith("明示的な言語指定") && wire[0].content.endsWith(f.StrataLocalTools.guidance), "language and tool guidance remain intact");
  check(wire[1].content === original[0].text && JSON.stringify(f.messages) === JSON.stringify(original), "user text and saved messages are not changed");
  const pinned = f.apiMessageSerializer();
  f.health = {status: "ok", model: "gemma-4-26b-a4b-it"};
  check(hasModel(pinned(), "qwen3.6-35b-a3b") && hasModel(f.apiMessages(), f.health.model), "an in-flight request retains its model while the next request follows new health");
  check(!identityLine(f.apiMessageSerializer("clover-35b-a3b-preview")()), "a model absent from authoritative health cannot be asserted");
  for (const health of [{model: "strata"}, {status: "ok", model: "strata"}, {status: "switching", model: "qwen3.6-35b-a3b"},
    {status: "ok", model: ""}, {status: "ok", model: null}]) {
    f.health = health;
    check(!identityLine(f.apiMessages()), "bootstrap, switching or empty health does not invent model identity");
  }
  const custom = '私設モデル / revision "2"\nmetadata';
  f.health = {status: "ok", model: custom};
  check(hasModel(f.apiMessages(), custom) && !identityLine(f.apiMessages()).includes("\n"), "custom model ID is retained as one JSON-quoted data value");

  f = fixture();
  const originalTimes = [{role: "user", text: "2024年の資料の確認", time: 1704067200123},
    {role: "assistant", text: "2024年当時の記録です", time: 1704067200999},
    {role: "user", text: "現在はどうですか", time: f.clockNow}];
  const historical = copy(originalTimes), dated = f.apiMessageSerializer();
  const firstDate = dateLine(dated(originalTimes));
  check(firstDate === "Current date from the TinyStrata computer clock: 2026-11-18 (Asia/Tokyo). Use this as the present, not your training cutoff.", "present date is explicit runtime system context");
  check(f.clockReads === 1 && dateLine(dated(originalTimes)) === firstDate && f.clockReads === 1, "serialization and repeated tool rounds do not re-read a pinned clock");
  f.clockNow += 200;
  check(dateLine(dated(originalTimes)) === firstDate, "crossing midnight cannot rewrite an in-flight request's system prefix");
  check(dateLine(f.apiMessageSerializer()(originalTimes)).includes("2026-11-19 (Asia/Tokyo)"), "the next request receives the new local day");
  check(dateLine(f.apiMessages(originalTimes)).includes("2026-11-19 (Asia/Tokyo)"), "direct serialization also uses the current day");
  f.clockNow += 3600000;
  const sameDay = f.apiMessageSerializer()(originalTimes);
  f.clockNow += 12345;
  check(sameDay[0].content === f.apiMessageSerializer()(originalTimes)[0].content, "seconds and milliseconds do not invalidate the initial system prefix");
  f.clockZone = "UTC";
  check(dateLine(f.apiMessageSerializer()(originalTimes)).includes("2026-11-18 (UTC)") && dateLine(dated(originalTimes)) === firstDate, "timezone changes apply to the next request, never a retained serializer");
  check(JSON.stringify(originalTimes) === JSON.stringify(historical), "runtime dates never rewrite historical text or saved message timestamps");
  check(!/T\d\d:|\.\d{3}Z/.test(dateLine(sameDay)), "initial date context contains no time or millisecond timestamp");
  f.StrataGrounding.getCurrentDateContext = () => {throw new Error("clock unavailable");};
  check(!dateLine(f.apiMessages(originalTimes)), "an unreadable clock cannot invent a date or block serialization");
  f.StrataGrounding = undefined;
  check(!dateLine(f.apiMessageSerializer()(originalTimes)), "older isolated fixtures without the clock module still serialize safely");

  f = fixture();
  const rows = [{role: "user", text: "資料の確認", files: [{name: "conditions.txt", text: "担当: 田中\n期限: 11月20日\n条件: 承認後のみ"}],
    images: [{url: "data:image/png;base64,AAAA"}]},
    {role: "assistant", model: "gpt-oss-20b", text: "確認します。", tools: [{id: "call1", name: "run_python", arguments: {code: "print(2+2)"}, result: "4", state: "done", round: 1, at: 0}]},
    {role: "user", text: "続けて"}];
  original = copy(rows);
  wire = f.apiMessages(rows, "以前のモデルはCloverだった");
  check(hasModel(wire, f.health.model) && wire[1].role === "assistant" && wire[1].content.includes("以前のモデル"), "historical model names stay historical rather than becoming system identity");
  check(wire[2].content[0].text.includes("担当: 田中") && wire[2].content[1].type === "image_url", "document and image input are preserved");
  check(wire.map((row) => row.role).join() === "system,assistant,user,assistant,tool,assistant,user", "summary, assistant tool call and tool result retain their original roles and order");
  check(JSON.stringify(rows) === JSON.stringify(original), "serialization does not modify saved attachments or historical model labels");

  // Use the actual compaction implementation and provided UI wrapper. Change
  // health between asynchronous steps to detect accidental late rebinding.
  f = fixture();
  const history = Array.from({length: 6}, (_, i) => [{role: "user", text: `資料${i}: 数量${i + 1}、担当者${i}、承認後のみ`},
    {role: "assistant", text: `資料${i}を確認`}]).flat();
  const m = {role: "assistant", text: "", reasoning: ""};
  f.messages = history.concat(m);
  f.chat.messages = f.messages; // The compaction checkpoint belongs to this transcript.
  const body = {model: f.health.model, max_tokens: 1024, tools: f.StrataLocalTools.definitions({webSearch: true}), temperature: .7};
  const serialize = f.apiMessageSerializer(body.model), counts = [], summaries = [];
  f.contextJson = async (url, request) => {
    counts.push(copy(request));
    f.health = {status: "ok", model: "clover-35b-a3b-preview"};
    f.clockNow = Date.parse("2026-11-18T15:00:00.100Z");
    return {model: request.model, prompt_tokens: counts.length === 1 ? 14000 : 4000, max_context: 16384};
  };
  f.summarizeContext = async (messages, limit, request) => { summaries.push({messages: copy(messages), model: request.model}); return "担当・期限・数量・承認条件を保持した過去の会話"; };
  let prepared = await f.prepareContext(body, history, m, new AbortController().signal, serialize);
  check(counts.length === 2 && summaries.length === 1, "representative long history is counted, summarized and recounted");
  check([...counts.map((request) => request.messages), summaries[0].messages, prepared.messages].every((messages) => hasModel(messages, body.model)), "count, summary input and final prompt share the original model snapshot");
  check([...counts.map((request) => request.messages), summaries[0].messages, prepared.messages].every((messages) => dateLine(messages).includes("2026-11-18 (Asia/Tokyo)")) && f.clockReads === 1, "count, summary and recount retain the same daily context across midnight");
  check(counts.every((request) => request.model === body.model) && summaries[0].model === body.model, "wire model and runtime identity stay consistent through asynchronous compaction");
  check(prepared.tools === body.tools && prepared.temperature === .7 && f.chat.compaction.model === body.model, "compaction preserves sampling, offered tools and checkpoint ownership");
  f.settings.autoCompact = false;
  prepared = await f.prepareContext(body, history, m, new AbortController().signal, serialize);
  check(hasModel(prepared.messages, body.model), "disabled automatic compaction retains the request snapshot in a later tool round");
  f.settings.autoCompact = true;
  f.contextJson = async () => { const error = new Error("old API"); error.status = 404; throw error; };
  prepared = await f.prepareContext(body, history, m, new AbortController().signal, serialize);
  check(hasModel(prepared.messages, body.model), "older-API retry fallback uses the same snapshot");

  // The resource dialog has its own compaction entry point; it must pin the
  // request too, even though it is cancelled if external switching wins.
  f = fixture(); f.messages = history; f.chat = {}; f.resourceOperation = null;
  Object.assign(f, {effortForModel: () => "on", mcpInfo: {tools: 0}, modelOperationBusy: () => false,
    setBusy() {}, renderChat() {}, contextJson: async (url, request) => {
      manualCounts.push(copy(request)); f.health = {status: "ok", model: "gemma-4-26b-a4b-it"};
      f.clockNow = Date.parse("2026-11-18T15:00:00.100Z");
      return {model: request.model, prompt_tokens: manualCounts.length === 1 ? 14000 : 4000, max_context: 16384};
    }, summarizeContext: async (messages) => { manualSummary = copy(messages); return "過去の担当と条件"; }});
  const manualCounts = []; let manualSummary;
  vm.runInContext(block(resources, "function resourceChatRequest(", "function resourceSnapshot("), f);
  vm.runInContext(block(resources, "async function compactResourceKV(", "async function applyResourcePlacement("), f);
  await assert.rejects(f.compactResourceKV({signal: new AbortController().signal, model: "qwen3.6-35b-a3b", onProgress() {}}), (error) => error.name === "AbortError"); checks++;
  check(manualCounts.length === 2 && manualCounts.every((request) => hasModel(request.messages, request.model)) && hasModel(manualSummary, "qwen3.6-35b-a3b"), "manual KV compaction keeps identity aligned with its request before cancelling an external model change");
  check([...manualCounts.map((request) => request.messages), manualSummary].every((messages) => dateLine(messages).includes("2026-11-18 (Asia/Tokyo)")), "manual KV compaction also pins the date before an asynchronous midnight change");
  console.log(`PASS ${checks} runtime model metadata checks (no network/model)`);
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
