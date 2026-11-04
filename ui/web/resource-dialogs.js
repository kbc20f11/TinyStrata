// Chat resource dialogs own presentation only. The app supplies current-chat
// token counting, compaction and model reloading; no API paths live here.
"use strict";
(function (root) {
  let instance = 0;
  const COPY = {
    kvTitle: "Context usage", kvDescription: "The context available to this chat, including messages and instructions. This is not VRAM usage.",
    used: "Used", available: "Available", unavailable: "Not available", tokens: "{n} tokens", usage: "{used} / {total} tokens ({pct}%)",
    compact: "Compact now", compactNote: "Summarize older exchanges while keeping recent turns. The full original history stays saved; summaries can omit details.",
    compactDisabled: "There are no older exchanges to summarize yet.", compacting: "Summarizing earlier conversation…", compacted: "Compacted: {before} → {after} tokens", summary: "View summary",
    placementTitle: "Expert placement", placementDescription: "Choose how much of this model’s expert weights to keep on the GPU. This is placement, not hardware utilization.",
    actual: "Current placement", gpu: "GPU", cpu: "CPU", automatic: "Automatic", manual: "Custom", target: "GPU target", requested: "Requested: {pct}% GPU", autoRequested: "Automatic placement",
    legacyBudget: "Existing memory limit: {n} GiB. Choose Automatic or Custom to replace it.",
    legacyPlacement: "Existing manual placement. Choose Automatic or Custom to replace it.",
    countDescription: "Choose how many experts across this model’s layers stay in GPU memory. The rest stay on the CPU side. Percentages show placement, not hardware utilization.",
    residentExperts: "Experts on GPU", expertCount: "{count} experts ({pct}%)", expertTarget: "{count} / {total} ({pct}%)",
    requestedExperts: "Requested: {count} / {total} experts on GPU ({pct}%)",
    countNote: "Applying reloads the current model. Available VRAM and the model’s placement granularity can limit the actual number of GPU-resident experts.",
    countStep: "{n} experts per step",
    placementNote: "Applying reloads the current model. Available VRAM and the model’s placement granularity can limit the actual number of GPU-resident experts.",
    placementDisabled: "Expert placement cannot be adjusted right now.", apply: "Apply and reload", applying: "Reloading model…", applied: "Expert placement updated.",
    loading: "Loading…", close: "Close", stop: "Stop", stopping: "Stopping…", stopped: "Summarization stopped.", busy: "Wait for the current operation to finish.",
    failed: "The operation could not be completed. Please try again.", modelChanged: "The model changed. Reopen this dialog to continue.",
  };
  const finite = (n) => typeof n === "number" && Number.isFinite(n);
  const percent = (n) => finite(n) ? Math.max(0, Math.min(100, n)) : null;
  const targetPercent = (n) => n == null ? null : finite(n) ? Math.round(Math.max(0, Math.min(100, n)) / 10) * 10 : null;
  function countGeometry(value) {
    const total = value?.total, step = value?.countStep;
    return Number.isSafeInteger(total) && total > 0 && Number.isSafeInteger(step) && step > 0 && step <= total && total % step === 0 ? {total, step} : null;
  }
  const targetCount = (n, geometry) => n == null || !finite(n) || !geometry ? null :
    Math.min(geometry.total, Math.max(0, Math.round(n / geometry.step) * geometry.step));
  function usage(value) {
    if (!finite(value?.used) || !finite(value?.total) || value.used < 0 || value.total <= 0) return null;
    return {used: value.used, total: value.total, available: Math.max(0, value.total - value.used), pct: Math.max(0, Math.min(100, value.used / value.total * 100))};
  }
  function create(options) {
    if (!options || typeof options.getSnapshot !== "function") throw new TypeError("Resource dialogs need getSnapshot");
    const doc = options.document || root.document;
    if (!doc?.body) throw new Error("Resource dialogs need a document body");
    const id = `resource-dialog-${++instance}`;
    const dialog = doc.createElement("dialog");
    dialog.className = "resource-dialog";
    dialog.setAttribute("aria-labelledby", `${id}-title`);
    dialog.setAttribute("aria-describedby", `${id}-description`);
    dialog.innerHTML = `<header class="resource-dialog__header"><div><p class="resource-dialog__model"></p><h2 id="${id}-title"></h2></div><button type="button" class="st-btn st-btn--icon resource-dialog__dismiss" data-dismiss><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18"/></svg></button></header>
      <div class="resource-dialog__body"><p class="resource-dialog__description" id="${id}-description"></p>
        <section data-kv><div class="resource-dialog__usage"><div class="resource-dialog__donut" role="img"><svg viewBox="0 0 160 160" aria-hidden="true"><circle class="resource-dialog__ring-track" cx="80" cy="80" r="62" pathLength="100"/><circle class="resource-dialog__ring-fill" cx="80" cy="80" r="62" pathLength="100"/></svg><div class="resource-dialog__donut-value"><strong data-percent>–</strong><span data-count>–</span></div></div><dl class="resource-dialog__legend"><div><dt data-used-label></dt><dd data-used>–</dd></div><div><dt data-available-label></dt><dd data-remaining>–</dd></div></dl></div><p class="resource-dialog__note" data-compact-note></p><details class="resource-dialog__summary" hidden><summary></summary><div></div></details></section>
        <section data-placement hidden><h3 data-actual-label></h3><div class="resource-dialog__devices"><div><span data-gpu-label></span><strong data-gpu-actual>–</strong><small class="resource-dialog__device-share" data-gpu-share hidden></small></div><div><span data-cpu-label></span><strong data-cpu-actual>–</strong><small class="resource-dialog__device-share" data-cpu-share hidden></small></div></div><div class="resource-dialog__placement-track" aria-hidden="true"><span></span></div><p class="resource-dialog__requested"></p><fieldset class="resource-dialog__mode"><legend class="resource-dialog__sr-only"></legend><label><input type="radio" name="${id}-mode" value="auto"><span data-auto-label></span></label><label><input type="radio" name="${id}-mode" value="manual"><span data-manual-label></span></label></fieldset><div class="resource-dialog__range"><div><label id="${id}-target-label" for="${id}-range" data-target-label></label><input class="st-input resource-dialog__number" type="number" min="0" inputmode="numeric" aria-labelledby="${id}-target-label" aria-describedby="${id}-step" hidden><output for="${id}-range">–</output></div><input id="${id}-range" type="range" min="0" max="100" step="10" value="50"><div class="resource-dialog__range-ticks" aria-hidden="true"><span>0%</span><span>50%</span><span>100%</span></div></div><p class="resource-dialog__step" id="${id}-step" data-count-step hidden></p><p class="resource-dialog__note" data-placement-note></p></section>
        <p class="resource-dialog__status" role="status" aria-live="polite"></p><p class="resource-dialog__reason" hidden></p>
      </div><footer class="resource-dialog__actions"><button type="button" class="st-btn st-btn--secondary" data-close></button><button type="button" class="st-btn st-btn--primary" data-action></button></footer>`;
    doc.body.appendChild(dialog);
    const find = (selector) => dialog.querySelector(selector);
    const range = find('input[type="range"]'), numberInput = find('input[type="number"]'), modeInputs = [...dialog.querySelectorAll('input[type="radio"]')];
    let kind = "kv", opener = null, epoch = 0, snapshot = {}, cached = null, model = "", pending = false, reading = false, stopping = false;
    let control = null, readControl = null, draft = null, dirty = false, status = null, failureText = "", lastSummary = "", progressText = "", countMode = null, numberEditing = false, dirtyBeforeNumber = false;
    const text = (key, vars = {}) => {
      const fallback = COPY[key] || key;
      const value = options.tr ? options.tr(`resources.${key}`, fallback, vars) : fallback;
      return String(value).replace(/\{(\w+)\}/g, (all, name) => vars[name] == null ? all : String(vars[name]));
    };
    const number = (n, digits = 0) => options.format ? options.format(n, digits) : new Intl.NumberFormat(undefined, {maximumFractionDigits: digits}).format(n);
    const tokens = (n) => text("tokens", {n: number(n)});
    const get = () => {
      const value = options.getSnapshot();
      return value && typeof value === "object" ? value : {};
    };
    function statusText() {
      if (failureText) return failureText;
      if (stopping) return text("stopping");
      if (pending) return progressText || text(kind === "kv" ? "compacting" : "applying");
      if (reading) return text("loading");
      return status ? text(status.key, status.values) : "";
    }
    function render() {
      snapshot = get();
      const changed = dialog.open && String(snapshot.model || "") !== model;
      const data = {...(kind === "kv" ? snapshot.kv : snapshot.placement), ...(cached || {})};
      const busy = !!snapshot.busy || changed;
      find(".resource-dialog__model").textContent = model;
      find(`#${id}-title`).textContent = text(kind === "kv" ? "kvTitle" : "placementTitle");
      countMode = kind === "placement" ? countGeometry(data) : null;
      find(`#${id}-description`).textContent = text(kind === "kv" ? "kvDescription" : countMode ? "countDescription" : "placementDescription");
      find("[data-kv]").hidden = kind !== "kv";
      find("[data-placement]").hidden = kind !== "placement";
      find("[data-dismiss]").setAttribute("aria-label", text("close"));
      find("[data-dismiss]").disabled = pending && kind === "placement";
      find("[data-close]").textContent = text(pending && kind === "kv" ? "stop" : "close");
      find("[data-close]").disabled = stopping || pending && kind === "placement";
      find("[data-action]").textContent = text(pending ? kind === "kv" ? "compacting" : "applying" : kind === "kv" ? "compact" : "apply");
      let reason = changed ? text("modelChanged") : busy ? text("busy") : data.reason || "";
      if (kind === "kv") {
        const sample = usage(data), donut = find(".resource-dialog__donut"), ring = find(".resource-dialog__ring-fill");
        donut.dataset.available = String(!!sample);
        donut.setAttribute("aria-label", sample ? text("usage", {used: number(sample.used), total: number(sample.total), pct: number(sample.pct, 1)}) : text("unavailable"));
        ring.setAttribute("stroke-dasharray", `${sample?.pct || 0} 100`);
        ring.style.visibility = sample && sample.pct > 0 ? "" : "hidden";
        find("[data-percent]").textContent = sample ? number(sample.pct, 1) + "%" : "–";
        find("[data-count]").textContent = sample ? `${number(sample.used)} / ${number(sample.total)}` : text("unavailable");
        find("[data-used-label]").textContent = text("used"); find("[data-available-label]").textContent = text("available");
        find("[data-used]").textContent = sample ? tokens(sample.used) : "–";
        find("[data-remaining]").textContent = sample ? tokens(sample.available) : "–";
        find("[data-compact-note]").textContent = text("compactNote");
        find("[data-action]").disabled = pending || reading || busy || data.canCompact !== true || typeof options.onCompact !== "function";
        if (!reason && !reading && data.canCompact !== true) reason = text("compactDisabled");
        const summary = find(".resource-dialog__summary"); summary.hidden = !lastSummary;
        summary.querySelector("summary").textContent = text("summary"); summary.querySelector("div").textContent = lastSummary;
      } else {
        const gpu = percent(data.gpuPct), cpu = percent(data.cpuPct);
        find("[data-actual-label]").textContent = text("actual"); find("[data-gpu-label]").textContent = text("gpu"); find("[data-cpu-label]").textContent = text("cpu");
        for (const [name, pct] of [["gpu", gpu], ["cpu", cpu]]) {
          const hasCount = countMode && pct != null && Number.isSafeInteger(data[name]);
          find(`[data-${name}-actual]`).textContent = hasCount ? number(data[name]) : pct == null ? "–" : number(pct, 1) + "%";
          find(`[data-${name}-actual]`).setAttribute("aria-label", hasCount ? text("expertCount", {count: number(data[name]), pct: number(pct, 1)}) : pct == null ? text("unavailable") : number(pct, 1) + "%");
          find(`[data-${name}-share]`).textContent = hasCount ? number(pct, 1) + "%" : "";
          find(`[data-${name}-share]`).hidden = !hasCount;
        }
        find(".resource-dialog__placement-track span").style.width = `${gpu || 0}%`;
        find(".resource-dialog__placement-track").dataset.available = String(gpu != null);
        const requested = countMode ? targetCount(data.requestedGpuExperts, countMode) : targetPercent(data.requestedGpuPercent);
        const legacyGiB = finite(data.legacyBudgetGiB) && data.legacyBudgetGiB >= 0;
        const legacy = requested == null && (data.legacyBudget === true || legacyGiB || finite(data.legacyExpertSlots));
        find(".resource-dialog__requested").textContent = legacy ? legacyGiB ? text("legacyBudget", {n: number(data.legacyBudgetGiB, 2)}) : text("legacyPlacement") : requested == null ? text("autoRequested") : countMode ? text("requestedExperts", {count: number(requested), total: number(countMode.total), pct: number(requested / countMode.total * 100, 1)}) : text("requested", {pct: number(requested)});
        if (!dirty && !pending) draft = requested;
        modeInputs[0].checked = draft == null && !(legacy && !dirty); modeInputs[1].checked = draft != null;
        find(".resource-dialog__mode legend").textContent = text("placementTitle");
        find("[data-auto-label]").textContent = text("automatic"); find("[data-manual-label]").textContent = text("manual");
        find("[data-target-label]").textContent = text(countMode ? "residentExperts" : "target");
        range.max = String(countMode?.total || 100); range.step = String(countMode?.step || 10);
        range.dataset.unit = countMode ? "experts" : "percent";
        if (draft != null) range.value = String(draft);
        else if (!dirty && !pending) range.value = String(countMode ? targetCount(data.gpu, countMode) ?? targetCount(countMode.total / 2, countMode) : targetPercent(gpu) ?? 50);
        const disabled = pending || reading || busy || data.adjustable !== true;
        modeInputs.forEach((input) => input.disabled = disabled); range.disabled = disabled || draft == null;
        numberInput.hidden = !countMode; numberInput.disabled = range.disabled;
        numberInput.max = String(countMode?.total || 100); numberInput.step = String(countMode?.step || 10);
        if (!numberEditing) numberInput.value = String(draft ?? range.value);
        find("[data-count-step]").hidden = !countMode;
        find("[data-count-step]").textContent = countMode ? text("countStep", {n: number(countMode.step)}) : "";
        const targetText = draft == null ? text("automatic") : countMode ? text("expertTarget", {count: number(draft), total: number(countMode.total), pct: number(draft / countMode.total * 100, 1)}) : number(draft) + "%";
        range.setAttribute("aria-valuetext", targetText);
        find(".resource-dialog__range output").textContent = targetText;
        const ticks = [...dialog.querySelectorAll(".resource-dialog__range-ticks span")];
        if (ticks.length === 3) ticks.forEach((tick, index) => tick.textContent = countMode ? number(targetCount(countMode.total * index / 2, countMode)) : `${index * 50}%`);
        find("[data-placement-note]").textContent = text(countMode ? "countNote" : "placementNote");
        find("[data-action]").disabled = disabled || typeof options.onPlacement !== "function" || !dirty;
        if (!reason && !reading && data.adjustable !== true) reason = text("placementDisabled");
      }
      const feedback = find(".resource-dialog__status"); feedback.textContent = statusText(); feedback.dataset.state = failureText ? "error" : pending || reading ? "working" : status?.key === "stopped" ? "neutral" : status ? "success" : "neutral";
      find(".resource-dialog__reason").textContent = reason; find(".resource-dialog__reason").hidden = !reason;
      dialog.setAttribute("aria-busy", String(pending || reading));
    }
    async function read() {
      const reader = kind === "kv" ? options.readKV : options.readPlacement;
      if (typeof reader !== "function") {render(); return;}
      const current = ++epoch, selected = kind, selectedModel = model;
      readControl?.abort(); readControl = new AbortController(); reading = true; render();
      try {
        const value = await reader({signal: readControl.signal, model: selectedModel});
        if (current !== epoch || !dialog.open || selected !== kind) return;
        if (String(get().model || "") !== selectedModel) throw new Error(text("modelChanged"));
        cached = value && typeof value === "object" ? value : null;
      } catch (error) { if (current === epoch && error.name !== "AbortError") failureText = error.message || text("failed"); }
      finally { if (current === epoch) {reading = false; render();} }
    }
    function close() {
      if (!dialog.open || pending && kind === "placement") return false;
      ++epoch; readControl?.abort(); if (pending) control?.abort();
      dialog.close(); return true;
    }
    function open(next, trigger) {
      if (pending && kind === "placement") return false;
      if (dialog.open) close();
      kind = next; opener = trigger || doc.activeElement; snapshot = get(); model = String(snapshot.model || "");
      cached = null; pending = false; reading = false; stopping = false; dirty = false; status = null; failureText = ""; lastSummary = ""; progressText = ""; numberEditing = false;
      draft = targetPercent(snapshot.placement?.requestedGpuPercent);
      render(); dialog.showModal(); render(); find("[data-dismiss]").focus(); read(); return true;
    }
    async function act() {
      if (countMode && !commitNumber()) return;
      if (find("[data-action]").disabled || pending) return;
      const selected = kind, selectedModel = model, current = ++epoch;
      control = new AbortController(); const signal = control.signal;
      pending = true; stopping = false; failureText = ""; status = null; progressText = ""; render();
      try {
        const callback = selected === "kv" ? options.onCompact : options.onPlacement;
        const result = await callback({signal, model: selectedModel, ...(selected === "placement" ? countMode ? {gpuExperts: draft} : {gpuPercent: draft} : {}),
          onProgress(value) {if (current === epoch && dialog.open && !signal.aborted && typeof value === "string") {progressText = value; render();}}});
        if (current !== epoch || !dialog.open) return;
        if (signal.aborted) throw new DOMException("Cancelled", "AbortError");
        if (String(get().model || "") !== selectedModel) throw new Error(text("modelChanged"));
        if (selected === "kv") {
          if (typeof result?.summary === "string") lastSummary = result.summary;
          status = finite(result?.beforeTokens) && finite(result?.afterTokens) ? {key: "compacted", values: {before: number(result.beforeTokens), after: number(result.afterTokens)}} : {key: "compacted", values: {before: "–", after: "–"}};
        } else { dirty = false; status = {key: "applied"}; }
        cached = null;
      } catch (error) {
        if (current !== epoch || !dialog.open) return;
        if (signal.aborted || error.name === "AbortError") status = {key: "stopped"};
        else failureText = error.message || text("failed");
      } finally {
        if (current === epoch) {pending = false; stopping = false; render(); if (!failureText && !signal.aborted) read();}
      }
    }
    function stopOrClose() {
      if (pending && kind === "kv") {stopping = true; control?.abort(); render();} else close();
    }
    find("[data-dismiss]").addEventListener("click", close);
    find("[data-close]").addEventListener("click", stopOrClose);
    find("[data-action]").addEventListener("click", act);
    dialog.addEventListener("cancel", (event) => {event.preventDefault(); close();});
    // Block the key's default close signal during a reload. Repeated native
    // close signals are not guaranteed to keep dispatching cancelable events.
    dialog.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && pending && kind === "placement") {event.preventDefault(); event.stopPropagation();}
    }, true);
    dialog.addEventListener("close", () => {if (!dialog.open && opener?.isConnected && typeof opener.focus === "function") opener.focus();});
    modeInputs.forEach((input) => input.addEventListener("change", () => {
      if (!input.checked) return;
      numberEditing = false;
      draft = input.value === "auto" ? null : countMode ? targetCount(+range.value, countMode) : targetPercent(+(range.value || 50)); dirty = true; failureText = ""; status = null; render();
    }));
    range.addEventListener("input", () => {numberEditing = false; draft = countMode ? targetCount(+range.value, countMode) : targetPercent(+range.value); dirty = true; failureText = ""; status = null; render();});
    function commitNumber() {
      if (!numberEditing || !countMode) return true;
      const raw = numberInput.value.trim(), value = Number(raw);
      numberEditing = false;
      if (!raw || !finite(value)) {dirty = dirtyBeforeNumber; render(); return false;}
      draft = targetCount(value, countMode); dirty = true; failureText = ""; status = null; render(); return true;
    }
    numberInput.addEventListener("input", () => {if (!numberEditing) dirtyBeforeNumber = dirty; numberEditing = true; dirty = true; failureText = ""; status = null; render();});
    numberInput.addEventListener("change", commitNumber); numberInput.addEventListener("blur", commitNumber);
    numberInput.addEventListener("keydown", (event) => {if (event.key === "Enter") {event.preventDefault(); commitNumber();}});
    return {openKV: (trigger) => open("kv", trigger), openPlacement: (trigger) => open("placement", trigger),
      refresh: () => {if (dialog.open) render();}, close,
      destroy: () => {if (pending && kind === "placement") return false; close(); ++epoch; readControl?.abort(); control?.abort(); dialog.remove(); return true;},
      get operation() { return pending ? kind : null; }, element: dialog};
  }
  const api = {COPY, targetPercent, countGeometry, targetCount, usage, create};
  root.StrataResourceDialogs = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(globalThis);
