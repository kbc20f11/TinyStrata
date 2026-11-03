// Fresh, disposable Python/WASM worker. Never mount host, IDBFS or OPFS storage.
// Its creator's opaque origin and connect-src 'none' are the security boundary.
"use strict";
(function () {
  const send = self.postMessage.bind(self), MAX_TEXT = 65536, MAX_FILES = 16, MAX_BYTES = 4 * 1024 * 1024;
  const BASE = "https://tinystrata-runtime.invalid/";
  let accepted = false;
  self.onmessage = async (event) => {
    if (accepted) return;
    accepted = true; self.onmessage = null;
    let pyodide = null, stdout = "", stderr = "", result = "", truncated = false, used = 0, started = 0;
    const urls = [];
    const append = (kind, value) => {
      const source = String(value) + "\n", part = source.slice(0, Math.max(0, MAX_TEXT - used));
      used += part.length; if (part.length < source.length) truncated = true;
      if (kind === "out") stdout += part; else stderr += part;
    };
    try {
      const assets = new Map(event.data.assets.map((a) => [a.name, a.data]));
      // The bootstrap fetch shim serves only verified, preloaded bytes. Unknown
      // URLs never delegate to native fetch; CSP also forbids native network.
      self.fetch = async (input) => {
        const url = typeof input === "string" ? input : input?.url || input?.href;
        if (!url?.startsWith(BASE)) throw new TypeError("Network access is disabled");
        const name = url.slice(BASE.length);
        if (!assets.has(name)) throw new TypeError("Package is not available offline");
        return new Response(assets.get(name), { headers: { "Content-Type": name.endsWith(".wasm") ? "application/wasm" : "application/octet-stream" } });
      };
      const blobURL = (name) => {
        const data = assets.get(name); if (!(data instanceof ArrayBuffer)) throw new Error("Missing verified runtime asset");
        const url = URL.createObjectURL(new Blob([data], { type: "text/javascript" })); urls.push(url); return url;
      };
      importScripts(blobURL("pyodide.js"));
      const module = await import(blobURL("pyodide.asm.mjs"));
      const lock = JSON.parse(new TextDecoder().decode(assets.get("pyodide-lock.json")));
      pyodide = await self.loadPyodide({
        indexURL: BASE, packageBaseUrl: BASE, stdLibURL: BASE + "python_stdlib.zip", lockFileContents: lock,
        createPyodideModule: module.default, jsglobals: Object.create(null), packages: [], enableRunUntilComplete: false,
        stdin: () => null, stdout: (s) => append("out", s), stderr: (s) => append("err", s)
      });
      pyodide.FS.mkdirTree("/work"); pyodide.FS.chdir("/work");
      // No package installer and no durable filesystem is exposed. These are
      // virtual files owned by this worker and disappear when it is terminated.
      stdout = ""; stderr = ""; used = 0; truncated = false;
      send({ type: "ready" }); started = performance.now();
      let exitstatus = 0;
      try {
        const value = await pyodide.runPythonAsync(event.data.code);
        if (value !== undefined && value !== null) {
          pyodide.globals.set("_tinystrata_last_result", value);
          const formatted = pyodide.runPython("import reprlib\nreprlib.repr(_tinystrata_last_result)");
          result = String(formatted).slice(0, Math.max(0, MAX_TEXT - used));
          if (String(formatted).length > result.length) truncated = true;
          value?.destroy?.(); pyodide.globals.delete("_tinystrata_last_result");
        }
      } catch (error) { exitstatus = 1; append("err", error?.message || String(error)); }
      const files = []; let bytes = 0;
      function collect(path, prefix, depth) {
        if (depth > 3) { truncated = true; return; }
        for (const name of pyodide.FS.readdir(path)) {
          if (name === "." || name === "..") continue;
          if (files.length >= MAX_FILES) { truncated = true; break; }
          const full = path + "/" + name, stat = pyodide.FS.lstat(full);
          if (pyodide.FS.isDir(stat.mode)) collect(full, prefix + name + "_", depth + 1);
          else if (pyodide.FS.isFile(stat.mode)) {
            if (stat.size > MAX_BYTES - bytes) { truncated = true; continue; }
            const data = pyodide.FS.readFile(full); bytes += data.byteLength;
            files.push({ name: (prefix + name).slice(0, 120), data: data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) });
          }
        }
      }
      collect("/work", "", 0);
      send({ type: "done", value: { stdout, stderr, result, files, exitstatus,
        timeMs: performance.now() - started, truncated, errorCode: exitstatus ? "python" : null } }, files.map((f) => f.data));
    } catch (error) {
      append("err", error?.message || String(error));
      send({ type: "done", value: { stdout, stderr, result: "", files: [], exitstatus: 1, timeMs: started ? performance.now() - started : 0, truncated, errorCode: "python" } });
    } finally { for (const url of urls) URL.revokeObjectURL(url); }
  };
})();
