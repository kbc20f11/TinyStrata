// Conversation compaction changes the prompt projection, never the saved transcript.
// No network, storage, DOM, tokenizer or inference dependencies live in this module.
"use strict";
(function (root) {
  const VERSION = 1;
  const positive = (n) => Number.isFinite(n) && n > 0;
  const aborted = (signal) => { if (signal?.aborted) throw new DOMException("Cancelled", "AbortError"); };
  function failure(message, code) { const error = new Error(message); error.code = code; return error; }
  function budget(context, requestedMax = 0) {
    if (!positive(context)) throw failure("The model context limit is unavailable", "context_unavailable");
    context = Math.floor(context);
    return {context, outputReserve: positive(requestedMax) ? Math.ceil(requestedMax) : Math.max(1, Math.min(8192, Math.floor(context / 4))),
      summaryMax: Math.max(1, Math.min(2048, Math.floor(context / 8))), trigger: Math.floor(context * .85), target: Math.floor(context * .60)};
  }
  function summarySettings(model, maxTokens) {
    const lfm = model === "lfm2.5-24b-a2b";
    return {stream: false, temperature: 0, max_tokens: maxTokens, reasoning_effort: lfm ? "low" : "none",
      reasoning_budget_tokens: lfm ? Math.max(1, Math.min(256, Math.floor(maxTokens / 4))) : 0,
      tool_choice: "none", strata_mcp: false,
      ...(model === "gemma-4-26b-a4b-it" ? {chat_template_kwargs: {enable_thinking: false}} : {})};
  }

  // Only data that can affect the model's transcript belongs to its identity.
  // A tool's expanded/collapsed state and timings can change while a summary is saved.
  const take = (item, keys) => Object.fromEntries(keys.filter((key) => item?.[key] !== undefined).map((key) => [key, item[key]]));
  const image = (item) => take(item, ["name", "url", "data_url"]);
  function semantic(message) {
    const item = take(message, ["role", "text", "reasoning"]);
    item.images = (message.images || []).map(image);
    item.files = (message.files || []).map((file) => take(file, ["name", "text"]));
    item.tools = (message.tools || []).map((tool) => ({...take(tool, ["id", "name", "arguments", "result", "state", "round", "vision", "at"]),
      images: (tool.images || []).map(image)}));
    return item;
  }
  function canonical(value) {
    if (Array.isArray(value)) return "[" + value.map(canonical).join(",") + "]";
    if (value && typeof value === "object") return "{" + Object.keys(value).sort().map((key) => JSON.stringify(key) + ":" + canonical(value[key])).join(",") + "}";
    return JSON.stringify(value);
  }
  async function fingerprint(messages, end = messages.length) {
    if (!Number.isInteger(end) || end < 0 || end > messages.length) throw failure("Invalid compaction boundary", "invalid_checkpoint");
    if (!root.crypto?.subtle) throw failure("Secure conversation fingerprinting is unavailable", "fingerprint_unavailable");
    const bytes = new TextEncoder().encode(canonical(messages.slice(0, end).map(semantic)));
    const digest = await root.crypto.subtle.digest("SHA-256", bytes);
    return Array.from(new Uint8Array(digest), (n) => n.toString(16).padStart(2, "0")).join("");
  }
  function boundary(messages, covered) {
    return Number.isInteger(covered) && covered > 0 && covered <= messages.length &&
      (covered === messages.length || messages[covered]?.role === "user");
  }
  async function valid(checkpoint, messages) {
    if (!checkpoint || checkpoint.version !== VERSION || !boundary(messages, checkpoint.covered) ||
        typeof checkpoint.summary !== "string" || !checkpoint.summary.trim() || checkpoint.summary.length > 65536 ||
        typeof checkpoint.fingerprint !== "string" || !/^[a-f0-9]{64}$/.test(checkpoint.fingerprint)) return false;
    return checkpoint.fingerprint === await fingerprint(messages, checkpoint.covered);
  }
  async function project(messages, checkpoint) {
    const kept = await valid(checkpoint, messages) ? checkpoint : null;
    return {checkpoint: kept, summary: kept?.summary || "", messages: messages.slice(kept?.covered || 0)};
  }

  // Approximate weights choose a whole-turn boundary only. The actual tokenizer
  // must count the projected prompt before any checkpoint is accepted.
  function weight(message) {
    const text = (value) => typeof value === "string" ? value.length : value == null ? 0 : canonical(value).length;
    return Math.max(1, text(message.text) + text(message.reasoning) +
      (message.images || []).length * 4096 + (message.files || []).reduce((sum, file) => sum + text(file.text), 0) +
      (message.tools || []).reduce((sum, tool) => sum + text(tool.arguments) + text(tool.result) + (tool.images || []).length * 4096, 0));
  }
  function plan(messages, {checkpoint = null, keepUserTurns = 2, retainFraction = .4} = {}) {
    const from = checkpoint?.covered || 0;
    if (!Number.isInteger(from) || from < 0 || from > messages.length) return null;
    keepUserTurns = Number.isInteger(keepUserTurns) && keepUserTurns > 0 ? keepUserTurns : 2;
    retainFraction = positive(retainFraction) && retainFraction < 1 ? retainFraction : .4;
    const users = messages.map((message, index) => message.role === "user" ? index : -1).filter((index) => index >= 0);
    if (users.length <= keepUserTurns) return null;
    const last = users[users.length - keepUserTurns];
    const candidates = users.filter((index) => index > from && index <= last);
    if (!candidates.length) return null;
    const weights = messages.map(weight), suffix = new Array(messages.length + 1).fill(0);
    for (let index = messages.length - 1; index >= 0; index--) suffix[index] = suffix[index + 1] + weights[index];
    const active = suffix[from] + (checkpoint?.summary?.length || 0);
    const covered = candidates.find((index) => suffix[index] <= active * retainFraction) ?? candidates[candidates.length - 1];
    return {covered, from, prefix: messages.slice(from, covered), recent: messages.slice(covered), previousSummary: checkpoint?.summary || ""};
  }
  async function create(messages, covered, summary, details = {}) {
    if (!boundary(messages, covered) || typeof summary !== "string" || !summary.trim() || summary.length > 65536)
      throw failure("The conversation summary is invalid", "invalid_summary");
    return {version: VERSION, covered, fingerprint: await fingerprint(messages, covered), summary: summary.trim(),
      model: typeof details.model === "string" ? details.model : "", createdAt: details.createdAt ?? Date.now(),
      beforeTokens: Number.isFinite(details.beforeTokens) ? details.beforeTokens : null,
      afterTokens: Number.isFinite(details.afterTokens) ? details.afterTokens : null};
  }

  // Callback contract: serialize(UI messages, previous summary) -> API messages;
  // count(request) -> {prompt_tokens,max_context}; summarize(API messages,maxTokens,options) -> text.
  // The caller owns committing the returned checkpoint after checking chat ownership.
  async function prepare({messages, checkpoint = null, request, serialize, count, summarize, signal, onProgress = () => {}, keepUserTurns = 2}) {
    aborted(signal);
    const projection = await project(messages, checkpoint);
    aborted(signal);
    const original = {...request, messages: serialize(projection.messages, projection.summary)};
    onProgress("checking");
    const initial = await count(original);
    aborted(signal);
    if (!Number.isFinite(initial?.prompt_tokens) || initial.prompt_tokens < 0) throw failure("The prompt token count is unavailable", "count_unavailable");
    if (initial.model != null && initial.model !== request.model)
      throw failure("The selected model changed before the conversation was counted", "model_changed");
    const limits = budget(initial.max_context, request.max_tokens);
    const unchanged = (reason) => ({request: original, checkpoint: projection.checkpoint, changed: false,
      beforeTokens: initial.prompt_tokens, afterTokens: initial.prompt_tokens, reason, targetMet: initial.prompt_tokens + limits.outputReserve <= limits.target});
    if (initial.prompt_tokens + limits.outputReserve < limits.trigger) return unchanged("below_threshold");
    const selected = plan(messages, {checkpoint: projection.checkpoint, keepUserTurns});
    if (!selected) {
      const actualMinimum = positive(request.max_tokens) ? Math.ceil(request.max_tokens) : 1;
      if (initial.prompt_tokens + actualMinimum + 8 > limits.context)
        throw failure("The recent messages exceed the model context; shorten the message or start a new chat", "recent_context_too_large");
      return unchanged("no_older_turns");
    }
    const sourceFingerprint = await fingerprint(messages, selected.covered);
    const historyFingerprint = await fingerprint(messages);
    aborted(signal);
    onProgress("summarizing");
    const summary = await summarize(serialize(selected.prefix, selected.previousSummary), limits.summaryMax, {signal, ...limits});
    aborted(signal);
    if (typeof summary !== "string" || !summary.trim() || summary.length > 65536)
      throw failure("The model did not produce a usable conversation summary", "invalid_summary");
    const candidate = {...request, messages: serialize(selected.recent, summary.trim())};
    onProgress("recounting");
    const final = await count(candidate);
    aborted(signal);
    if (final?.model != null && final.model !== request.model)
      throw failure("The selected model changed while the conversation was being summarized", "model_changed");
    if (final?.max_context !== initial.max_context) throw failure("The model changed while the conversation was being summarized", "context_changed");
    if (!Number.isFinite(final?.prompt_tokens) || final.prompt_tokens < 0 || final.prompt_tokens >= initial.prompt_tokens)
      throw failure("The conversation summary did not reduce the prompt", "summary_not_smaller");
    if (final.prompt_tokens + limits.outputReserve + 8 > limits.context)
      throw failure("The recent messages still exceed the model context; shorten the message or start a new chat", "recent_context_too_large");
    const next = await create(messages, selected.covered, summary, {model: request.model, beforeTokens: initial.prompt_tokens, afterTokens: final.prompt_tokens});
    aborted(signal);
    if (next.fingerprint !== sourceFingerprint || historyFingerprint !== await fingerprint(messages))
      throw failure("The conversation changed while it was being summarized", "conversation_changed");
    aborted(signal);
    return {request: candidate, checkpoint: next, changed: true, beforeTokens: initial.prompt_tokens,
      afterTokens: final.prompt_tokens, reason: "compacted", targetMet: final.prompt_tokens + limits.outputReserve <= limits.target};
  }
  const api = {VERSION, budget, summarySettings, weight, fingerprint, valid, project, plan, create, prepare};
  root.StrataContextCompaction = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
