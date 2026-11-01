// Render completed Mermaid blocks locally. The parser runs in an opaque sandbox;
// the chat receives only an SVG image, never executable markup or click handlers.
"use strict";
(function (root) {
  const MAX_SOURCE = 32768, MAX_SVG = 2 * 1024 * 1024, MAX_CACHE = 32;
  const cache = new Map();
  let frame = null, boot = null, sequence = 0, pending = null, queue = Promise.resolve();
  let bootTimer = 0, readyListener = null, rejectBoot = null;
  const dark = () => document.documentElement.dataset.theme === "dark";
  const keyOf = (source, theme) => (theme ? "dark:" : "light:") + source;
  function stop() {
    clearTimeout(bootTimer);
    if (readyListener) root.removeEventListener("message", readyListener);
    readyListener = null;
    const reject = rejectBoot; rejectBoot = null;
    reject?.(new Error("Diagram unavailable"));
    frame?.remove(); frame = null; boot = null;
    if (pending) { pending.reject(new Error("Diagram unavailable")); pending = null; }
  }
  function runtime() {
    if (boot) return boot;
    boot = new Promise((resolve, reject) => {
      frame = document.createElement("iframe");
      frame.setAttribute("sandbox", "allow-scripts");
      frame.setAttribute("referrerpolicy", "no-referrer");
      frame.setAttribute("aria-hidden", "true"); frame.tabIndex = -1;
      frame.style.cssText = "position:fixed;left:-10000px;top:0;width:1100px;height:900px;visibility:hidden;border:0;pointer-events:none";
      frame.src = "web/diagram-renderer.html?v=20261101.5";
      const own = frame;
      rejectBoot = reject; bootTimer = setTimeout(stop, 10000);
      const cleanup = () => { clearTimeout(bootTimer); root.removeEventListener("message", ready); readyListener = null; rejectBoot = null; };
      const ready = (event) => {
        if (event.source !== own.contentWindow || event.data?.type !== "tinystrata-diagram-ready") return;
        cleanup(); resolve(own);
      };
      readyListener = ready;
      root.addEventListener("message", ready);
      document.body.appendChild(frame);
    });
    return boot;
  }
  root.addEventListener("message", (event) => {
    if (!frame || event.source !== frame.contentWindow || event.data?.type !== "tinystrata-diagram-result" || !pending || event.data.id !== pending.id) return;
    const {resolve} = pending; pending = null;
    const svg = event.data.svg;
    resolve(event.data.ok && typeof svg === "string" && svg.length <= MAX_SVG
      ? {ok: true, url: "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg)} : {ok: false});
  });
  function result(source, theme) {
    const key = keyOf(source, theme);
    if (cache.has(key)) { const hit = cache.get(key); cache.delete(key); cache.set(key, hit); return hit; }
    if (!source.trim() || source.length > MAX_SOURCE) return {value: {ok: false}, promise: Promise.resolve({ok: false})};
    const entry = {value: null, promise: null};
    entry.promise = queue.then(async () => {
      let timer;
      try {
        const target = await runtime(), id = ++sequence;
        const work = new Promise((resolve, reject) => { pending = {id, resolve, reject}; });
        timer = setTimeout(stop, 12000);
        target.contentWindow.postMessage({type: "tinystrata-diagram-render", id, source, dark: theme}, "*");
        return await work;
      } catch (_) { return {ok: false}; }
      finally { clearTimeout(timer); }
    }).then((value) => { entry.value = value; return value; });
    queue = entry.promise.then(() => undefined, () => undefined);
    cache.set(key, entry);
    if (cache.size > MAX_CACHE) cache.delete(cache.keys().next().value);
    return entry;
  }
  function place(box, value) {
    box.dataset.diagramState = value.ok ? "ready" : "error";
    const button = box.querySelector("[data-diagram-toggle]");
    if (!value.ok) {
      box.dataset.diagramView = "code";
      box.querySelector(".st-diagram__status").hidden = false;
      return;
    }
    const host = box.querySelector(".st-diagram__image");
    box.querySelector(".st-diagram__status").hidden = true;
    const image = document.createElement("img");
    image.src = value.url;
    image.alt = root.tr("diagram.alt", "Mermaid diagram");
    host.replaceChildren(image);
    if (box.dataset.diagramView !== "code") box.dataset.diagramView = "image";
    button.hidden = false;
    button.setAttribute("aria-pressed", String(box.dataset.diagramView === "image"));
  }
  function hydrate(container) {
    const theme = dark(), generation = container._diagramGeneration = (container._diagramGeneration || 0) + 1;
    for (const box of container.querySelectorAll('[data-diagram="mermaid"]')) {
      const source = box.querySelector("pre code").textContent;
      const entry = result(source, theme);
      if (entry.value) { place(box, entry.value); continue; }
      entry.promise.then((value) => {
        if (generation !== container._diagramGeneration || !container.contains(box) || dark() !== theme) return;
        place(box, value);
      });
    }
  }
  function refresh() {
    for (const bubble of document.querySelectorAll('.st-bubble:not([data-streaming="true"])')) hydrate(bubble);
  }
  root.StrataDiagrams = {hydrate, refresh};
  root.addEventListener("pagehide", stop);
})(window);
