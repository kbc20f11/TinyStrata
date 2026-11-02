// Local, explicit-language highlighting. Closed fences can be colored while an
// answer streams; open fences wait until the answer ends. No code runs here.
"use strict";
(function (root) {
  const WORKER_URL = new URL("syntax-worker.js?v=20261103.1", document.currentScript.src);
  const MAX_SOURCE = 100000, MAX_CACHE = 2 * 1024 * 1024, MAX_PENDING = 32;
  const languages = new Set("bash c cpp csharp css diff go graphql ini java javascript json kotlin less lua makefile markdown objectivec perl php php-template python python-repl r ruby rust scss shell sql swift typescript vbnet wasm xml yaml powershell cmake dockerfile latex".split(" "));
  const aliases = {
    sh: "bash", zsh: "bash", ksh: "bash", curl: "bash", console: "shell", "shell-session": "shell",
    "c++": "cpp", cc: "cpp", cxx: "cpp", hpp: "cpp", hxx: "cpp", cuda: "cpp", h: "c",
    cs: "csharp", "c#": "csharp", js: "javascript", mjs: "javascript", cjs: "javascript", jsx: "javascript",
    ts: "typescript", tsx: "typescript", py: "python", python3: "python", py3: "python", pycon: "python-repl",
    rb: "ruby", rs: "rust", objc: "objectivec", "objective-c": "objectivec", md: "markdown",
    yml: "yaml", toml: "ini", html: "xml", xhtml: "xml", svg: "xml", xsl: "xml", rss: "xml",
    ps: "powershell", ps1: "powershell", pwsh: "powershell", "cmake.in": "cmake",
    docker: "dockerfile", tex: "latex", mk: "makefile"
  };
  const cache = new Map(), pending = new Map();
  let worker = null, active = null, timer = 0, sequence = 0, cacheSize = 0, pendingSize = 0, refillQueued = false;
  const escape = (s) => s.replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  function language(name) {
    const key = String(name || "").trim().toLowerCase().replace(/^language-/, "");
    const result = Object.hasOwn(aliases, key) ? aliases[key] : key;
    return languages.has(result) ? result : null;
  }
  function cached(key) {
    const value = cache.get(key);
    if (value !== undefined) { cache.delete(key); cache.set(key, value); }
    return value;
  }
  function html(lang, code) {
    const name = language(lang);
    return (name && cached(name + "\0" + code)) ?? escape(code);
  }
  function remember(job, value) {
    const size = job.key.length + value.length;
    if (size > MAX_CACHE) return;
    while (cacheSize + size > MAX_CACHE) {
      const first = cache.keys().next().value, old = cache.get(first);
      cacheSize -= first.length + old.length; cache.delete(first);
    }
    cache.set(job.key, value); cacheSize += size;
  }
  function finish(job, value) {
    clearTimeout(timer); active = null;
    pending.delete(job.key); pendingSize -= job.code.length;
    remember(job, value);
    // Streaming replaces DOM nodes. Address the current copies, not stale nodes
    // captured before the worker started; exact text is checked before insertion.
    for (const el of document.querySelectorAll(`code[data-syntax-job="${job.id}"]`)) {
      if (el.textContent === job.code && language(el.dataset.syntaxLang) === job.lang) {
        el.innerHTML = value; el.dataset.syntaxState = "ready";
      }
      delete el.dataset.syntaxJob;
    }
    pump();
    if (!refillQueued && document.querySelector('code[data-syntax-state="waiting"]')) {
      refillQueued = true;
      queueMicrotask(() => { refillQueued = false; hydrate(document, false, true); });
    }
  }
  function failed() {
    worker?.terminate(); worker = null;
    if (active) finish(active, escape(active.code));
  }
  function pump() {
    if (active || !pending.size) return;
    active = pending.values().next().value;
    try {
      if (!worker) {
        worker = new Worker(WORKER_URL);
        const own = worker;
        own.onmessage = ({data}) => {
          if (worker !== own || !active || data?.id !== active.id) return;
          const value = typeof data.html === "string" && data.html.length <= MAX_CACHE ? data.html : escape(active.code);
          finish(active, value);
        };
        own.onerror = (event) => { event.preventDefault(); if (worker === own) failed(); };
      }
      // Regex grammars must never hold the UI thread or wait indefinitely.
      timer = setTimeout(failed, 2000);
      worker.postMessage({id: active.id, lang: active.lang, code: active.code});
    } catch (_) { failed(); }
  }
  function hydrate(container, streaming = false, waitingOnly = false) {
    const selector = ".st-code > pre > code[data-syntax-lang]" + (waitingOnly ? '[data-syntax-state="waiting"]' : "");
    for (const el of container.querySelectorAll(selector)) {
      if (el.dataset.syntaxState === "ready" || (streaming && el.dataset.syntaxComplete !== "true")) continue;
      const lang = language(el.dataset.syntaxLang), code = el.textContent;
      if (!lang || !code.trim() || code.length > MAX_SOURCE) continue;
      const key = lang + "\0" + code, value = cached(key);
      if (value !== undefined) { el.innerHTML = value; el.dataset.syntaxState = "ready"; continue; }
      let job = pending.get(key);
      if (!job) {
        if (pending.size >= MAX_PENDING || pendingSize + code.length > MAX_CACHE) { el.dataset.syntaxState = "waiting"; continue; }
        job = {id: ++sequence, key, lang, code};
        pending.set(key, job); pendingSize += code.length;
      }
      delete el.dataset.syntaxState;
      el.dataset.syntaxJob = String(job.id);
    }
    pump();
  }
  root.StrataSyntax = {html, hydrate};
})(globalThis);
