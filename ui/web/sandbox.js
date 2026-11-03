// Independent TinyStrata browser sandbox host. Code is never evaluated here.
"use strict";
(function (root) {
  const script = new URL(document.currentScript.src), VERSION = "314.0.7";
  const RUNTIME = new URL("/sandbox-runtime/" + VERSION + "/", location.origin);
  const MAX_CODE = 100000, MAX_TEXT = 65536, MAX_FILES = 16, MAX_BYTES = 4 * 1024 * 1024;
  const ASSETS = Object.freeze([
    ["pyodide.js", 18912, "3141b814715a72e59b51b1b18b9ceae5bf19f7c852417e431bb0a34feadf825c"],
    ["pyodide.asm.mjs", 1250344, "f7cdc8ece80678ceb712f8e65ebe6d3a83203a180c399865f49612a051693635"],
    ["pyodide.asm.wasm", 9598218, "cc36e3cab04fdfc9a63ff13eb52eae2b911bf46c025cc7b281f394bd3de1d5e6"],
    ["python_stdlib.zip", 2545637, "fa1957e5777068fc4f7437f96d860ae2fbe9c19732ba06c84e004ec16dd7dd7a"],
    ["pyodide-lock.json", 119077, "5dc2fc119108bc148c7457dc86e7675b5c87e1cafd420b9c34c1eaef7b36c010"]
  ]);
  let prepared = null, active = null;
  function failure(code, message, status = 1) {
    return { stdout: "", stderr: message, result: "", images: [], files: [], exitstatus: status, timeMs: 0, time: 0, truncated: false, errorCode: code };
  }
  async function fetchBytes(url, expected, signal) {
    const response = await fetch(url, { credentials: "omit", cache: "no-store", redirect: "error", signal });
    if (!response.ok) throw new Error("Sandbox runtime is unavailable");
    const data = await response.arrayBuffer();
    if (data.byteLength !== expected[1]) throw new Error("Sandbox runtime size mismatch");
    const digest = await crypto.subtle.digest("SHA-256", data);
    const hash = [...new Uint8Array(digest)].map((x) => x.toString(16).padStart(2, "0")).join("");
    if (hash !== expected[2]) throw new Error("Sandbox runtime integrity mismatch");
    return { name: expected[0], data };
  }
  async function prepare() {
    if (!prepared) prepared = (async () => {
      const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 30000);
      const source = async (name) => {
        const r = await fetch(new URL(name, script), { credentials: "omit", redirect: "error", signal: controller.signal });
        if (!r.ok) throw new Error("Sandbox bootstrap is unavailable");
        return r.text();
      };
      try {
        const [frame, broker, worker, assets] = await Promise.all([
          source("sandbox-frame.html"), source("sandbox-frame.js"), source("sandbox-worker.js"),
          Promise.all(ASSETS.map((a) => fetchBytes(new URL(a[0], RUNTIME), a, controller.signal)))
        ]);
        return { frame, broker, worker, assets };
      } catch (e) {
        const timedOut = controller.signal.aborted; controller.abort();
        if (timedOut) throw new Error("Python runtime preparation timed out"); throw e;
      }
      finally { clearTimeout(timer); }
    })().catch((e) => { prepared = null; throw e; });
    return prepared;
  }
  function sanitize(value) {
    if (!value || typeof value !== "object") return failure("protocol", "Invalid sandbox response");
    let remaining = MAX_TEXT, bytes = 0;
    const text = (s) => { const out = typeof s === "string" ? s.slice(0, remaining) : ""; remaining -= out.length; return out; };
    const files = [];
    for (const f of Array.isArray(value.files) ? value.files.slice(0, MAX_FILES) : []) {
      if (!f || !(f.data instanceof ArrayBuffer) || f.data.byteLength > MAX_BYTES - bytes) continue;
      bytes += f.data.byteLength;
      files.push({ name: String(f.name || "output.bin").replace(/[^\p{L}\p{N}_.-]/gu, "_").replace(/^\.+/, "_").slice(0, 120), mime: "application/octet-stream", data: f.data });
    }
    const images = (Array.isArray(value.images) ? value.images : []).filter((f) => f &&
      ["image/png", "image/jpeg", "image/webp"].includes(f.mime) && files.some((file) => file.data === f.data && file.name === f.name));
    const timeMs = Number.isFinite(value.timeMs) ? Math.max(0, Math.min(120000, value.timeMs)) : 0;
    const base64 = (buffer) => {
      const bytes = new Uint8Array(buffer); let out = "";
      for (let i = 0; i < bytes.length; i += 16384) out += String.fromCharCode(...bytes.subarray(i, i + 16384));
      return btoa(out);
    };
    const publicFiles = files.map((f) => ({ name: f.name, mime: f.mime, size: f.data.byteLength,
      data_url: "data:application/octet-stream;base64," + base64(f.data) }));
    const publicImages = images.map((f) => ({ name: f.name, mime: f.mime, size: f.data.byteLength,
      data_url: "data:" + f.mime + ";base64," + base64(f.data) }));
    return { stdout: text(value.stdout), stderr: text(value.stderr), result: text(value.result), images: publicImages, files: publicFiles,
      exitstatus: Number.isInteger(value.exitstatus) && value.exitstatus >= 0 && value.exitstatus <= 255 ? value.exitstatus : 1,
      timeMs, time: timeMs / 1000, truncated: value.truncated === true, errorCode: value.errorCode === "python" ? "python" : null };
  }
  async function execute(options = {}) {
    if (!["python", "py", "python3"].includes(String(options.language || "python").toLowerCase())) return failure("unsupported", "Only Python is available");
    if (typeof options.code !== "string" || options.code.length > MAX_CODE) return failure("code_limit", "Code exceeds the execution limit");
    if (options.signal?.aborted) return failure("aborted", "Execution cancelled", 130);
    if (active) return failure("busy", "Another code block is running");
    let cancel = null;
    const marker = {}; active = marker;
    try {
      // Keep cancellation responsive during trusted asset loading too.
      const preload = prepare();
      const aborted = new Promise((_, reject) => { cancel = () => reject(new DOMException("Execution cancelled", "AbortError")); options.signal?.addEventListener("abort", cancel, { once: true }); });
      const data = await Promise.race([preload, aborted]);
      options.signal?.removeEventListener("abort", cancel); cancel = null;
      if (options.signal?.aborted) return failure("aborted", "Execution cancelled", 130);
      return await new Promise((resolve) => {
        const iframe = document.createElement("iframe"), channel = new MessageChannel();
        let timer = 0, done = false, started = false;
        const timeout = Math.min(30000, Math.max(1000, Number(options.timeoutMs) || 10000));
        iframe.hidden = true; iframe.title = "Isolated Python execution";
        iframe.setAttribute("sandbox", "allow-scripts");
        iframe.setAttribute("referrerpolicy", "no-referrer");
        iframe.setAttribute("allow", "camera 'none'; microphone 'none'; geolocation 'none'; clipboard-read 'none'; clipboard-write 'none'");
        function finish(value) {
          if (done) return; done = true; clearTimeout(timer);
          options.signal?.removeEventListener("abort", cancel); channel.port1.close(); iframe.remove(); resolve(value);
        }
        cancel = () => finish(failure("aborted", "Execution cancelled", 130));
        options.signal?.addEventListener("abort", cancel, { once: true });
        timer = setTimeout(() => finish(failure("startup_timeout", "Python initialization timed out", 124)), 30000);
        channel.port1.onmessage = (event) => {
          if (event.data?.type === "ready" && !started) {
            started = true; clearTimeout(timer);
            timer = setTimeout(() => finish(failure("timeout", "Execution timed out", 124)), timeout);
          } else if (event.data?.type === "done") finish(sanitize(event.data.value));
          else if (event.data?.type === "failure") finish(failure("worker", "Python worker could not start"));
        };
        const nonce = [...crypto.getRandomValues(new Uint8Array(18))].map((n) => n.toString(16).padStart(2, "0")).join("");
        iframe.srcdoc = data.frame.replaceAll("__TS_NONCE__", nonce).replace("__TS_FRAME_SCRIPT__", () => data.broker.replace(/<\/script/gi, "<\\/script"));
        iframe.addEventListener("load", () => {
          if (done) return;
          const assets = data.assets.map((a) => ({ name: a.name, data: a.data.slice(0) }));
          iframe.contentWindow.postMessage({ type: "tinystrata-sandbox-init", workerSource: data.worker, assets, code: options.code }, "*", [channel.port2, ...assets.map((a) => a.data)]);
        }, { once: true });
        document.body.appendChild(iframe);
      });
    } catch (error) { return error?.name === "AbortError" ? failure("aborted", "Execution cancelled", 130) : failure("runtime_unavailable", error?.message || "Sandbox runtime is unavailable"); }
    finally { options.signal?.removeEventListener("abort", cancel); if (active === marker) active = null; }
  }
  root.StrataSandbox = Object.freeze({ execute, languages: Object.freeze(["python"]), runtimeVersion: VERSION,
    limits: Object.freeze({ code: MAX_CODE, output: MAX_TEXT, files: MAX_FILES, fileBytes: MAX_BYTES, timeoutMs: 10000 }),
    get busy() { return active !== null; } });
})(globalThis);
