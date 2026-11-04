"use strict";
const assert = require("node:assert/strict");
const C = require("../ui/web/context-compaction.js");
let checks = 0;
const check = (value, label) => { assert(value, label); checks++; };
const clone = (value) => JSON.parse(JSON.stringify(value));
const history = () => Array.from({length: 6}, (_, index) => [
  {role: "user", text: `Question ${index}`, images: index === 0 ? [{name: "photo.png", url: "data:image/png;base64,AAAA"}] : []},
  {role: "assistant", text: `Answer ${index}`, reasoning: "Reasoning", tools: index === 1 ? [{id: "call1", name: "run_python", arguments: {code: "print(42)"}, result: "42", state: "done", round: 1, at: 0}] : []}
]).flat();
const serialize = (messages, summary) => [...(summary ? [{role: "user", content: `Earlier conversation summary:\n${summary}`}, {role: "assistant", content: "I will use this as reference."}] : []),
  ...messages.map((message) => ({role: message.role, content: message.text}))];
async function rejection(promise, code, label) { await assert.rejects(promise, (error) => error.code === code, label); checks++; }
async function main() {
  let messages = history(), original = clone(messages);
  const b = C.budget(32768);
  const lfm = C.summarySettings("lfm2.5-24b-a2b", 2048);
  check(lfm.reasoning_effort === "low" && lfm.reasoning_budget_tokens === 256 && lfm.strata_mcp === false && lfm.tool_choice === "none", "LFM retains required thinking with a bounded budget and no tool execution");
  const gemma = C.summarySettings("gemma-4-26b-a4b-it", 2048);
  check(gemma.reasoning_effort === "none" && gemma.chat_template_kwargs.enable_thinking === false && gemma.max_tokens === 2048, "Gemma summarization explicitly disables thinking");
  for (const model of ["qwen3.6-35b-a3b", "clover-35b-a3b-preview"]) {
    const options = C.summarySettings(model, 2048);
    check(options.reasoning_effort === "none" && options.reasoning_budget_tokens === 0 && options.stream === false && options.temperature === 0, "Qwen and Clover summaries are bounded deterministic text requests");
  }
  check(b.outputReserve === 8192 && b.summaryMax === 2048 && b.trigger === 27852 && b.target === 19660, "defaults reserve space for the answer and trigger before the hard limit");
  check(C.budget(4096).outputReserve === 1024 && C.budget(4096).summaryMax === 512, "small model contexts use proportional reserves");
  check(C.budget(32768, 4096).outputReserve === 4096, "explicit completion budget is respected");
  const planned = C.plan(messages);
  check(planned && planned.covered % 2 === 0 && planned.covered <= 8 && planned.recent.filter((m) => m.role === "user").length >= 2, "only whole old turns are selected, preserving the latest two user turns");
  assert.deepEqual(messages, original); checks++;
  check(C.plan(messages.slice(-4)) === null, "a short history is not needlessly summarized");
  check(C.canCompact(messages) && !C.canCompact(messages.slice(-4)), "manual compaction availability requires an older complete user turn");
  const cp = await C.create(messages, 6, "The user discussed a photo and Python.", {model: "qwen", beforeTokens: 10000, afterTokens: 2000});
  check(await C.valid(cp, messages), "a checkpoint matches its original semantic prefix");
  const presentation = clone(messages); presentation[1].stats = {elapsedMs: 1}; presentation[3].tools[0].open = true;
  check(await C.valid(cp, presentation), "presentation and timing changes do not invalidate the summary");
  const argumentOrder = clone(messages); argumentOrder[3].tools[0].arguments = {code: "print(42)", options: {width: 80, height: 24}};
  const orderedCp = await C.create(argumentOrder, 6, "Facts");
  const reordered = clone(argumentOrder); reordered[3].tools[0].arguments = {options: {height: 24, width: 80}, code: "print(42)"};
  check(await C.valid(orderedCp, reordered), "object key ordering is not part of source identity");
  for (const [mutate, label] of [
    [(v) => {v[0].text += " edited";}, "text edits"],
    [(v) => {v[0].images[0].url = "data:image/png;base64,BBBB";}, "image changes"],
    [(v) => {v[1].reasoning = "changed";}, "reasoning changes"],
    [(v) => {v[3].tools[0].result = "43";}, "tool result changes"],
    [(v) => {v[3].tools[0].arguments.code = "print(43)";}, "tool arguments"],
    [(v) => {v.splice(2, 2);}, "rewound messages"]
  ]) { const changed = clone(messages); mutate(changed); check(!await C.valid(cp, changed), `${label} invalidate a stale summary`); }
  const appended = [...messages, {role: "user", text: "Next question"}];
  check(await C.valid(cp, appended), "appending another user question keeps the checkpoint valid");
  const changedTail = clone(messages); changedTail[8].text = "Edited recent question";
  check(await C.valid(cp, changedTail), "editing recent, uncompressed messages keeps the summary valid");
  const projection = await C.project(messages, cp);
  check(projection.messages[0] === messages[6] && projection.messages.length === 6 && projection.summary === cp.summary, "projection preserves original message and attachment references");
  const bad = {...cp, fingerprint: "0".repeat(64)};
  check((await C.project(messages, bad)).messages.length === messages.length, "a stale checkpoint falls back to the full transcript");
  check(!await C.valid({...cp, covered: 5}, messages), "a checkpoint cannot bisect a user-assistant turn");
  const attached = clone(messages); attached[0].files = [{name: "facts.txt", text: "Original facts"}];
  const attachedCp = await C.create(attached, 6, "The file contains facts.");
  attached[0].files[0].text = "Different facts";
  check(!await C.valid(attachedCp, attached), "file content is part of summary source identity");
  const repeated = C.plan(messages, {checkpoint: cp});
  check(repeated && repeated.from === cp.covered && repeated.covered > cp.covered && repeated.prefix.every((m) => !messages.slice(0, cp.covered).includes(m)), "subsequent compaction summarizes only new old turns and the previous summary");

  const stages = [], counts = [], req = {model: "clover", max_tokens: 1024, tools: [{type: "function", function: {name: "web_search"}}]};
  const run = (extra = {}) => C.prepare({messages, request: req, serialize, count: async (request) => {counts.push(request); return {prompt_tokens: counts.length % 2 ? 14000 : 4000, max_context: 16384};},
    summarize: async () => "Important facts, constraints, decisions, and unresolved work.", onProgress: (stage) => stages.push(stage), ...extra});
  let result = await run();
  check(result.changed && result.beforeTokens === 14000 && result.afterTokens === 4000 && result.targetMet, "a high-context request is summarized and recounted before acceptance");
  check(await C.valid(result.checkpoint, messages) && result.checkpoint.model === "clover", "the returned checkpoint is bound to the original content");
  check((await C.project(messages, {...result.checkpoint, model: "another model"})).checkpoint != null, "summary text is portable between models when the new prompt is recounted");
  check(counts[0].tools === req.tools && result.request.tools === req.tools && result.request.max_tokens === 1024, "request settings and enabled tools survive projection");
  check(stages.join() === "checking,summarizing,recounting", "checking and summary work can be shown separately from normal generation");
  assert.deepEqual(messages, original); checks++;
  let summaries = 0;
  result = await run({count: async () => ({prompt_tokens: 1000, max_context: 16384}), summarize: async () => {summaries++; return "unused";}});
  check(!result.changed && summaries === 0 && result.reason === "below_threshold", "ordinary prompts do not run a hidden summary completion");
  let manualCounts = 0, manualSummaries = 0;
  result = await run({force: true, count: async () => ({prompt_tokens: ++manualCounts === 1 ? 1000 : 400, max_context: 16384}),
    summarize: async () => {manualSummaries++; return "A short manual summary of the older turns.";}});
  check(result.changed && manualCounts === 2 && manualSummaries === 1 && result.beforeTokens === 1000 && result.afterTokens === 400,
    "manual compaction summarizes below the automatic threshold and validates the reduced prompt");
  assert.deepEqual(messages, original); checks++;
  const fullyCompacted = await C.create(messages, 8, "All older turns have been summarized.");
  check(!C.canCompact(messages, {checkpoint: fullyCompacted}), "a repeated manual operation cannot compact the last two user turns");
  manualSummaries = 0;
  result = await run({force: true, checkpoint: fullyCompacted, count: async () => ({prompt_tokens: 800, max_context: 16384}),
    summarize: async () => {manualSummaries++; return "unused";}});
  check(!result.changed && result.reason === "no_older_turns" && result.checkpoint === fullyCompacted && manualSummaries === 0,
    "repeated manual compaction is a friendly no-op when there are no older turns");
  const manualCancelled = new AbortController();
  manualSummaries = 0;
  await assert.rejects(run({force: true, signal: manualCancelled.signal,
    count: async () => {manualCancelled.abort(); return {prompt_tokens: 1000, max_context: 16384};},
    summarize: async () => {manualSummaries++; return "unused";}}), (error) => error.name === "AbortError"); checks++;
  check(manualSummaries === 0, "cancelling manual preparation cannot run or commit a summary");
  result = await run({messages: messages.slice(-4), count: async () => ({prompt_tokens: 15000, max_context: 16384}), summarize: async () => {summaries++; return "unused";}});
  check(!result.changed && summaries === 0 && result.reason === "no_older_turns", "a giant recent question is never silently discarded");
  await rejection(run({messages: messages.slice(-4), count: async () => ({prompt_tokens: 16000, max_context: 16384})}), "recent_context_too_large", "an overflowing recent-only prompt is refused without deletion");
  result = await run({messages: messages.slice(-2), request: {model: "clover"}, count: async () => ({prompt_tokens: 16375, max_context: 16384})});
  check(!result.changed && result.reason === "no_older_turns", "without an explicit output limit, a fitting recent turn can use the remaining context");
  await rejection(run({messages: messages.slice(-2), request: {model: "clover"}, count: async () => ({prompt_tokens: 16376, max_context: 16384})}), "recent_context_too_large", "a recent-only prompt reserves one output token and the engine's eight-token margin");
  await rejection(run({summarize: async () => "   ", count: async () => ({prompt_tokens: 14000, max_context: 16384})}), "invalid_summary", "empty summary rejected");
  let n = 0;
  await rejection(run({count: async () => ({prompt_tokens: ++n === 1 ? 14000 : 14001, max_context: 16384})}), "summary_not_smaller", "larger summary rejected");
  n = 0;
  await rejection(run({count: async () => ({prompt_tokens: ++n === 1 ? 20000 : 16000, max_context: 16384})}), "recent_context_too_large", "overflow rejected");
  n = 0;
  await rejection(run({count: async () => ({prompt_tokens: ++n === 1 ? 14000 : 4000, max_context: n === 1 ? 16384 : 32768})}), "context_changed", "model changes rejected");
  await rejection(run({count: async () => ({prompt_tokens: 14000, max_context: 16384, model: "qwen"})}), "model_changed", "a different model with the same context is rejected");
  n = 0;
  await rejection(run({count: async () => ({prompt_tokens: ++n === 1 ? 14000 : 4000, max_context: 16384, model: n === 1 ? "clover" : "qwen"})}), "model_changed", "a switch during summary cannot apply another model's token count");
  await rejection(run({count: async () => ({prompt_tokens: NaN, max_context: 16384})}), "count_unavailable", "invalid token count rejected");
  n = 0;
  const controller = new AbortController();
  await assert.rejects(run({signal: controller.signal, count: async () => ({prompt_tokens: ++n === 1 ? 14000 : 4000, max_context: 16384}), summarize: async () => {controller.abort(); return "Cancelled summary";}}), (error) => error.name === "AbortError"); checks++;
  check(n === 1, "cancellation during summarizing prevents a candidate count or commit");
  n = 0;
  const mutable = clone(messages);
  await rejection(run({messages: mutable, count: async () => ({prompt_tokens: ++n === 1 ? 14000 : 4000, max_context: 16384}), summarize: async () => {mutable[0].text = "Changed while pending"; return "Stale";}}), "conversation_changed", "source races rejected");
  n = 0;
  const recentRace = clone(messages);
  await rejection(run({messages: recentRace, count: async () => ({prompt_tokens: ++n === 1 ? 14000 : 4000, max_context: 16384}), summarize: async () => {recentRace[10].text = "Edited recent question while pending"; return "Stale";}}), "conversation_changed", "recent source races rejected");
  n = 0;
  await rejection(run({count: async () => ({prompt_tokens: ++n === 1 ? 17000 : 15353, max_context: 16384})}), "recent_context_too_large", "native engine's eight-token margin is kept");
  assert.deepEqual(messages, original); checks++;
  console.log(`PASS ${checks} context compaction checks`);
}
main().catch((error) => {console.error(error); process.exitCode = 1;});
