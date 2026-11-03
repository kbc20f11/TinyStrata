// Independent TinyStrata broker. This document has an opaque sandbox origin.
// User code runs only in its child worker; no host/API/DOM capabilities cross.
"use strict";
(function () {
  const MAX_TEXT = 65536, MAX_FILES = 16, MAX_BYTES = 4 * 1024 * 1024;
  let worker = null, port = null, started = false;
  function text(value) { return typeof value === "string" ? value.slice(0, MAX_TEXT) : ""; }
  function result(value) {
    if (!value || typeof value !== "object") throw new Error("Invalid worker result");
    let remaining = MAX_TEXT, bytes = 0;
    const take = (s) => { const out = text(s).slice(0, remaining); remaining -= out.length; return out; };
    const files = [];
    for (const f of Array.isArray(value.files) ? value.files.slice(0, MAX_FILES) : []) {
      if (!f || !(f.data instanceof ArrayBuffer) || f.data.byteLength > MAX_BYTES - bytes) continue;
      const name = text(f.name).replace(/[^\p{L}\p{N}_.-]/gu, "_").slice(0, 120);
      bytes += f.data.byteLength;
      files.push({ name: name && !name.startsWith(".") ? name : "output.bin", mime: "application/octet-stream", data: f.data });
    }
    // Raster images are recognized from bytes, never user-supplied HTML/SVG.
    const images = [];
    for (const f of files) {
      const b = new Uint8Array(f.data), ascii = (i, s) => [...s].every((c, n) => b[i + n] === c.charCodeAt(0));
      let mime = null;
      if (b.length >= 8 && b[0] === 137 && ascii(1, "PNG\r\n\x1a\n")) mime = "image/png";
      else if (b.length >= 3 && b[0] === 255 && b[1] === 216 && b[2] === 255) mime = "image/jpeg";
      else if (b.length >= 12 && ascii(0, "RIFF") && ascii(8, "WEBP")) mime = "image/webp";
      if (mime) images.push({ ...f, mime });
    }
    return { stdout: take(value.stdout), stderr: take(value.stderr), result: take(value.result), files, images,
      exitstatus: Number.isInteger(value.exitstatus) && value.exitstatus >= 0 && value.exitstatus <= 255 ? value.exitstatus : 1,
      timeMs: Number.isFinite(value.timeMs) ? Math.max(0, Math.min(value.timeMs, 120000)) : 0,
      truncated: value.truncated === true, errorCode: value.errorCode === "python" ? "python" : null };
  }
  window.addEventListener("message", function initialize(event) {
    if (event.source !== parent || port || event.data?.type !== "tinystrata-sandbox-init" || event.ports.length !== 1) return;
    port = event.ports[0];
    const data = event.data;
    try {
      const blob = new Blob([data.workerSource], { type: "text/javascript" });
      const url = URL.createObjectURL(blob);
      worker = new Worker(url);
      worker.onmessage = (message) => {
        URL.revokeObjectURL(url);
        if (message.data?.type === "ready" && !started) { started = true; port.postMessage({ type: "ready" }); return; }
        if (message.data?.type !== "done") return;
        try {
          const value = result(message.data.value);
          port.postMessage({ type: "done", value }, [...new Set(value.files.map((f) => f.data))]);
        } catch (_) { port.postMessage({ type: "failure", errorCode: "protocol" }); }
        worker.terminate(); worker = null;
      };
      worker.onerror = () => { URL.revokeObjectURL(url); port.postMessage({ type: "failure", errorCode: "worker" }); worker?.terminate(); worker = null; };
      const transfer = data.assets.map((a) => a.data);
      worker.postMessage({ assets: data.assets, code: data.code }, transfer);
    } catch (_) { port.postMessage({ type: "failure", errorCode: "worker" }); }
  });
})();
