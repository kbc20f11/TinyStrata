// Dialog presentation is separate from requests and conversation ownership.
"use strict";
let resourceOperation = null, resourceDialogs = null;
const resourcePlacementSettings = new Map();

function resourceChatRequest(rows) {
  const body = {model: health.model, messages: apiMessages(rows), stream: false, reasoning_effort: effortForModel()};
  if (health.model === "gemma-4-26b-a4b-it") body.chat_template_kwargs = {enable_thinking: body.reasoning_effort !== "none"};
  if (settings.max) body.max_tokens = +settings.max;
  const tools = StrataLocalTools.definitions(settings);
  if (tools.length) body.tools = tools;
  if (settings.mcp !== false && mcpInfo.tools > 0) body.strata_mcp = true;
  return body;
}

function resourceSnapshot() {
  const unloaded = lastMetrics?.live?.state === "unloaded";
  const engine = placementSampleValid && !unloaded && !modelSwitching && !modelOp ? lastMetrics?.engine : null;
  const kv = contextUsage(engine), placement = expertPlacement(engine);
  const configured = resourcePlacementSettings.get(health.model);
  return {model: health.model, busy: modelOperationBusy(),
    kv: {used: kv?.used, total: kv?.total, canCompact: StrataContextCompaction.canCompact(messages, {checkpoint: chat?.compaction})},
    placement: {gpuPct: placement?.gpuPct, cpuPct: placement?.cpuPct, gpu: placement?.gpu, cpu: placement?.cpu, total: placement?.total ?? configured?.expert_total,
      requestedGpuPercent: configured?.gpu_percent ?? null, adjustable: configured?.supported === true,
      requestedGpuExperts: configured?.gpu_experts ?? null, countStep: configured?.count_step,
      legacyBudget: configured?.legacy_budget === true, legacyBudgetGiB: configured?.legacy_budget ? configured.expert_vram_gib : null,
      legacyExpertSlots: configured?.legacy_expert_slots,
      reason: configured?.supported === false ? configured.error || tr("resources.placementDisabled", "Expert placement cannot be adjusted right now.") : ""}};
}

async function readResourceKV({signal, model}) {
  const owner = chatId, source = messages;
  const projection = await StrataContextCompaction.project(source, chat?.compaction);
  const body = {...resourceChatRequest(projection.messages), messages: apiMessages(projection.messages, projection.summary)};
  const count = await contextJson("v1/chat/token-count", body, signal);
  if (model !== health.model || count.model !== model || owner !== chatId || source !== messages)
    throw new Error(tr("resources.modelChanged", "The model changed. Reopen this dialog to continue."));
  return {used: count.prompt_tokens, total: count.max_context,
    canCompact: StrataContextCompaction.canCompact(source, {checkpoint: projection.checkpoint})};
}

async function readResourcePlacement({signal, model}) {
  let value;
  try { value = await modelControl("v1/model/expert-placement", {signal}); }
  catch (error) {
    if (error.status === 404) throw new Error(tr("resources.updateRequired", "Update the TinyStrata API to adjust expert placement."));
    throw error;
  }
  if (model !== health.model || value.placement?.model !== model)
    throw new Error(tr("resources.modelChanged", "The model changed. Reopen this dialog to continue."));
  resourcePlacementSettings.set(model, value.placement);
  return {requestedGpuPercent: value.placement.gpu_percent, requestedGpuExperts: value.placement.gpu_experts,
    total: value.placement.expert_total, countStep: value.placement.count_step, adjustable: value.placement.supported === true,
    legacyBudget: value.placement.legacy_budget === true, legacyBudgetGiB: value.placement.legacy_budget ? value.placement.expert_vram_gib : null,
    legacyExpertSlots: value.placement.legacy_expert_slots,
    reason: value.placement.supported === false ? value.placement.error || tr("resources.placementDisabled", "Expert placement cannot be adjusted right now.") : ""};
}

async function compactResourceKV({signal, model, onProgress}) {
  if (modelOperationBusy() || model !== health.model) throw new Error(tr("resources.busy", "Wait for the current operation to finish."));
  const owner = chat, source = messages;
  resourceOperation = "compact";
  setBusy(false);
  try {
    const body = resourceChatRequest(source);
    const prepared = await StrataContextCompaction.prepare({messages: source, checkpoint: owner?.compaction, request: body, force: true,
      signal, serialize: apiMessages, count: (request) => contextJson("v1/chat/token-count", request, signal),
      summarize: (history, maxTokens) => summarizeContext(history, maxTokens, body, signal),
      onProgress: () => onProgress(tr("resources.compacting", "Summarizing earlier conversation…"))});
    if (!prepared.changed) throw new Error(tr("resources.noChange", "This conversation is already compact enough. Recent turns are kept in full."));
    if (signal.aborted || owner !== chat || source !== messages || model !== health.model) throw new DOMException("Cancelled", "AbortError");
    onProgress(tr("resources.warming", "Loading compacted context…"));
    // Native GEN requires at least one output token. Install the compacted
    // prefix now, without adding its warm-up token to the visible transcript.
    // Tool execution is disabled for this bounded request.
    const warm = {...prepared.request, stream: false, max_tokens: 1, temperature: 0, strata_mcp: false};
    await contextJson("v1/chat/completions", warm, signal);
    if (signal.aborted || owner !== chat || source !== messages || model !== health.model) throw new DOMException("Cancelled", "AbortError");
    owner.compaction = prepared.checkpoint;
    const last = source.findLast((row) => row.role === "assistant");
    if (last) last.compaction = {summary: prepared.checkpoint.summary, beforeTokens: prepared.beforeTokens, afterTokens: prepared.afterTokens};
    saveChat(); renderChat();
    // No cancellable await follows the atomic checkpoint commit. The regular
    // metrics poll updates the footer from the newly warmed native prefix.
    return {beforeTokens: prepared.beforeTokens, afterTokens: prepared.afterTokens, summary: prepared.checkpoint.summary};
  } finally { resourceOperation = null; setBusy(!!busy); }
}

async function applyResourcePlacement({signal, model, gpuPercent, gpuExperts, onProgress}) {
  if (modelOperationBusy() || model !== health.model) throw new Error(tr("resources.busy", "Wait for the current operation to finish."));
  resourceOperation = "placement"; placementSampleValid = false; setBusy(false);
  try {
    const byCount = gpuExperts !== undefined;
    const payload = byCount ? {model, gpu_experts: gpuExperts} : {model, gpu_percent: gpuPercent};
    let result = await modelControl("v1/model/expert-placement", {method: "POST", body: JSON.stringify(payload)});
    const deadline = performance.now() + 300000;
    while (result.operation?.switching) {
      if (performance.now() > deadline) throw new Error(tr("resources.failed", "The operation could not be completed. Please try again."));
      onProgress(tr("resources.applying", "Reloading model…"));
      await new Promise((resolve) => setTimeout(resolve, 750));
      result = await modelControl("v1/model/expert-placement");
    }
    if (result.operation?.error || result.placement?.model !== model ||
        (byCount ? result.placement?.gpu_experts !== gpuExperts : result.placement?.gpu_percent !== gpuPercent))
      throw new Error(result.operation?.error || tr("resources.failed", "The operation could not be completed. Please try again."));
    resourcePlacementSettings.set(model, result.placement);
    await loadHealth();
    const response = await fetch("metrics", {headers: headers()});
    if (response.ok) {lastMetrics = await response.json(); placementSampleValid = true; render(lastMetrics);}
    return result.placement;
  } finally { resourceOperation = null; setBusy(!!busy); }
}

function initResourceDialogs() {
  resourceDialogs = StrataResourceDialogs.create({getSnapshot: resourceSnapshot, readKV: readResourceKV, readPlacement: readResourcePlacement,
    onCompact: compactResourceKV, onPlacement: applyResourcePlacement, tr, format: fmt});
  $("chat-context").onclick = () => resourceDialogs.openKV($("chat-context"));
  $("chat-placement").onclick = () => resourceDialogs.openPlacement($("chat-placement"));
}
