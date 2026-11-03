// Manual Python execution for completed fences. Results are untrusted display
// data; code is executed only by the caller's isolated execution callback.
"use strict";
(function (root) {
  const views = new Set(), attached = new WeakMap(), tickets = new WeakMap();
  const MAX_TEXT = 65536, MAX_FILES = 16, MAX_BYTES = 4 * 1024 * 1024, MAX_SAVED = 8, MAX_SAVED_BYTES = 8 * 1024 * 1024;
  let active = null;
  const tr = (key, english, vars) => typeof root.tr === "function" ? root.tr(key, english, vars) :
    english.replace(/\{(\w+)\}/g, (_, name) => String(vars?.[name] ?? ""));
  const element = (tag, cls, text) => {
    const node = document.createElement(tag); if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text; return node;
  };
  function normalize(value) {
    if (!value || typeof value !== "object") throw new Error("Invalid Python execution result");
    let remaining = MAX_TEXT, bytes = 0;
    const text = (v) => { const s = typeof v === "string" ? v.slice(0, remaining) : ""; remaining -= s.length; return s; };
    const files = [], images = [];
    const accept = (items, image) => {
      for (const item of Array.isArray(items) ? items.slice(0, MAX_FILES) : []) {
        if (!item || typeof item.data_url !== "string") continue;
        const match = item.data_url.match(/^data:(application\/octet-stream|image\/(?:png|jpeg|webp));base64,([A-Za-z0-9+/]*={0,2})$/);
        if (!match || (image && !match[1].startsWith("image/")) || (!image && match[1] !== "application/octet-stream")) continue;
        const size = Math.floor(match[2].length * 3 / 4) - (match[2].endsWith("==") ? 2 : match[2].endsWith("=") ? 1 : 0);
        if (size < 0 || size > MAX_BYTES || bytes + size > 2 * MAX_BYTES) continue;
        bytes += size;
        const name = String(item.name || "output.bin").replace(/[^\p{L}\p{N}_.-]/gu, "_").replace(/^\.+/, "_").slice(0, 120);
        (image ? images : files).push({ name, mime: match[1], size, data_url: item.data_url });
      }
    };
    accept(value.files, false); accept(value.images, true);
    return { stdout: text(value.stdout), stderr: text(value.stderr), result: text(value.result), files, images,
      exitstatus: Number.isInteger(value.exitstatus) && value.exitstatus >= 0 && value.exitstatus <= 255 ? value.exitstatus : 1,
      timeMs: Number.isFinite(value.timeMs) ? Math.max(0, Math.min(value.timeMs, 120000)) : 0,
      errorCode: typeof value.errorCode === "string" ? value.errorCode.slice(0, 60) : null,
      truncated: value.truncated === true };
  }
  function error(message, code) {
    return normalize({ stdout: "", stderr: String(message), result: "", exitstatus: code === "aborted" ? 130 : 1, errorCode: code });
  }
  function stored(view) {
    const saved = view.message.executions;
    if (!saved || typeof saved !== "object" || !Object.hasOwn(saved, view.hash)) return null;
    try { return normalize(saved[view.hash]); } catch (_) { return null; }
  }
  function save(view, value) {
    const old = view.message.executions;
    const entries = old && typeof old === "object" && !Array.isArray(old) ?
      Object.entries(old).filter(([key]) => /^[a-f0-9]{64}$/.test(key) && key !== view.hash).slice(-(MAX_SAVED - 1)) : [];
    const entry = { ...value, at: Date.now() };
    // Keep persistence bounded even for multiple generated image/file results.
    // Drop exports first if a single result alone exceeds the JSON allowance.
    while (new TextEncoder().encode(JSON.stringify(entry)).length > MAX_SAVED_BYTES) {
      entry.truncated = true;
      if (entry.files.length) entry.files.pop(); else if (entry.images.length) entry.images.pop(); else break;
    }
    entries.push([view.hash, entry]);
    const all = Object.fromEntries(entries);
    while (entries.length > 1 && new TextEncoder().encode(JSON.stringify(all)).length > MAX_SAVED_BYTES) {
      const removed = entries.shift(); delete all[removed[0]];
    }
    view.message.executions = all;
    try { Promise.resolve(view.options.onSave?.(view.message)).catch(() => {}); } catch (_) {}
  }
  function running(view) { return active?.message === view.message && active.hash === view.hash; }
  function render(view) {
    const own = running(view), available = typeof view.options.execute === "function";
    const streaming = view.options.streaming !== false || view.code.dataset.syntaxComplete !== "true";
    view.button.hidden = streaming;
    view.button.disabled = !own && (!available || !!active || !!view.options.disabled?.());
    view.button.textContent = own ? tr("codeRun.stop", "Stop") : tr("codeRun.run", "Run Python");
    if (own) view.button.removeAttribute("data-python-run"); else view.button.setAttribute("data-python-run", "");
    view.button.setAttribute("aria-label", view.button.textContent);
    view.button.title = available ? view.button.textContent : tr("codeRun.unavailable", "Python execution is unavailable.");
    const raw = view.message.executions?.[view.hash], lang = root.StrataI18n?.lang || "en";
    if (view.lastRaw === raw && view.lastRunning === own && view.lastLang === lang) return;
    view.lastRaw = raw; view.lastRunning = own; view.lastLang = lang;
    view.output.replaceChildren(); view.output.setAttribute("aria-label", tr("codeRun.output", "Python output"));
    if (own) {
      view.output.hidden = false;
      view.output.append(element("div", "st-code__execution-status", tr("codeRun.running", "Running Python…"))); return;
    }
    const value = stored(view);
    view.output.hidden = !value;
    if (!value) return;
    let status;
    if (value.errorCode === "aborted") status = tr("codeRun.cancelled", "Execution cancelled.");
    else if (["timeout", "startup_timeout"].includes(value.errorCode)) status = tr("codeRun.timeout", "Execution timed out.");
    else if (value.exitstatus) status = tr("codeRun.failed", "Execution failed ({status})", { status: value.exitstatus });
    else status = tr("codeRun.done", "Finished in {seconds} s", { seconds: (value.timeMs / 1000).toFixed(2) });
    const label = element("div", "st-code__execution-status", status); label.setAttribute("role", "status"); view.output.append(label);
    for (const [key, title] of [["stdout", tr("codeRun.stdout", "Output")], ["stderr", tr("codeRun.stderr", "Errors")], ["result", tr("codeRun.result", "Result")]]) {
      if (!value[key]) continue;
      const section = element("section", "st-code__execution-" + key);
      section.append(element("div", "st-code__execution-label", title), element("pre", "", value[key])); view.output.append(section);
    }
    for (const file of value.images) {
      const image = element("img", "st-code__execution-image"); image.src = file.data_url;
      image.alt = file.name || tr("codeRun.image", "Generated image"); image.loading = "lazy"; view.output.append(image);
    }
    if (value.files.length) {
      const list = element("div", "st-code__execution-files"); list.append(element("span", "st-code__execution-label", tr("codeRun.files", "Files")));
      for (const file of value.files) {
        const link = element("a", "st-code__execution-file", file.name); link.href = file.data_url; link.download = file.name;
        link.rel = "noopener noreferrer"; list.append(link);
      }
      view.output.append(list);
    }
    if (!value.stdout && !value.stderr && !value.result && !value.images.length && !value.files.length && !value.exitstatus)
      view.output.append(element("div", "st-code__execution-empty", tr("codeRun.empty", "No output.")));
    if (value.truncated) view.output.append(element("div", "st-code__execution-warning", tr("codeRun.truncated", "Output was shortened to the execution limit.")));
  }
  function refresh() {
    for (const view of views) { if (!view.box.isConnected) views.delete(view); else render(view); }
  }
  async function launch(view) {
    if (running(view)) { active.controller.abort(); return; }
    if (active || view.options.streaming !== false || view.code.dataset.syntaxComplete !== "true" || view.options.disabled?.() || typeof view.options.execute !== "function") return;
    const execution = { message: view.message, hash: view.hash, controller: new AbortController() }; active = execution; refresh();
    let value;
    try { value = normalize(await view.options.execute(view.source, execution.controller.signal)); }
    catch (e) { value = error(e?.message || String(e), execution.controller.signal.aborted ? "aborted" : "execution"); }
    try { save(view, value); }
    finally { if (active === execution) active = null; refresh(); }
  }
  async function hydrate(container, message, options = {}) {
    if (!container?.querySelectorAll || !message || typeof message !== "object") return;
    refresh();
    await Promise.all([...container.querySelectorAll(".st-code > pre > code[data-syntax-lang]")].map(async (code) => {
      if (!["python", "python3", "py", "py3"].includes(code.dataset.syntaxLang.trim().toLowerCase())) return;
      const box = code.parentElement.parentElement, source = code.textContent;
      const existing = attached.get(box);
      if (existing && existing.source === source && existing.message === message) { existing.options = options; render(existing); return; }
      if (existing) { existing.button.remove(); existing.output.remove(); views.delete(existing); attached.delete(box); }
      if (!crypto.subtle) return;
      const ticket = {}; tickets.set(box, ticket);
      const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode("python\0" + source));
      if (!box.isConnected || code.textContent !== source || tickets.get(box) !== ticket) return;
      // A rerender can trigger concurrent hydration while hashing the same DOM.
      const raced = attached.get(box);
      if (raced && raced.source === source && raced.message === message) { raced.options = options; render(raced); return; }
      const hash = [...new Uint8Array(digest)].map((n) => n.toString(16).padStart(2, "0")).join("");
      const head = box.querySelector(".st-code__btns") || box.querySelector(".st-code__head"); if (!head) return;
      const button = element("button", "st-btn st-code__run"); button.type = "button";
      const output = element("div", "st-code__execution"); output.hidden = true; output.setAttribute("role", "region");
      const view = { box, code, source, hash, message, options, button, output };
      button.addEventListener("click", () => { void launch(view); }); head.prepend(button); box.append(output);
      attached.set(box, view); views.add(view); render(view);
    }));
  }
  root.StrataCodeRunner = Object.freeze({ hydrate, cancelAll() { active?.controller.abort(); }, get running() { return active !== null; } });
})(globalThis);
