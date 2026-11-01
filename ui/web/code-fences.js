// Shared display-only boundaries for Markdown and math. Model/history text is
// never rewritten; a misplaced opening fence is recovered only with a language.
"use strict";
(function (root) {
  const languages = new Set(("bash sh shell zsh fish powershell pwsh ps1 bat batch cmd python python3 py javascript js jsx typescript ts tsx " +
    "html svg xml css scss less json jsonc yaml yml toml ini sql ruby rb rust rs go golang java kotlin swift c cpp c++ cs c# csharp " +
    "php perl lua dart r julia haskell clojure elixir erlang scala vue svelte dockerfile makefile cmake diff patch markdown md text plaintext latex tex mermaid").split(" "));
  const escaped = (s, at) => { let n = 0; while (at > 0 && s[--at] === "\\") n++; return n % 2 === 1; };
  function mathEnd(text, at) {
    if (escaped(text, at)) return -1;
    let open, close;
    if (text.startsWith("\\(", at)) { open = "\\("; close = "\\)"; }
    else if (text.startsWith("\\[", at)) { open = "\\["; close = "\\]"; }
    else if (text.startsWith("$$", at)) open = close = "$$";
    else if (text[at] === "$" && !/\s/.test(text[at + 1] || " ") && text[at - 1] !== "$") open = close = "$";
    if (!open) return -1;
    let end = text.indexOf(close, at + open.length);
    while (end >= 0 && (escaped(text, end) || (close === "$" && (/\s/.test(text[end - 1]) || /[\d$]/.test(text[end + 1] || ""))))) end = text.indexOf(close, end + close.length);
    const source = text.slice(at + open.length, end < 0 ? text.length : end);
    if (open === "$" && (/\n|\$/.test(source) || /^\d[\d,.]*(?:\s*[A-Za-z]|\s*$)/.test(source))) return -1;
    return end < 0 ? -1 : end + close.length;
  }
  function tickEnd(text, at, count) {
    const runs = /`+/g; runs.lastIndex = at + count;
    let m; while ((m = runs.exec(text))) if (m[0].length === count) return m.index;
    return -1;
  }
  function quoted(text, from, to) {
    let quote = "";
    for (let i = from; i < to; i++) {
      if (text[i] === "\\") { i++; continue; }
      if (quote) { if (text[i] === quote) quote = ""; }
      else if (text[i] === "'" || text[i] === '"' || text[i] === "`") quote = text[i];
    }
    return !!quote;
  }
  function spans(text) {
    const found = [];
    for (let i = 0; i < text.length;) {
      if (text[i] !== "`" || escaped(text, i)) { i++; continue; }
      let n = 1; while (text[i + n] === "`") n++;
      const end = tickEnd(text, i, n), nl = text.indexOf("\n", i + n);
      if (end >= 0 && (nl < 0 || end < nl)) {
        found.push({start: i, end: end + n, code: text.slice(i + n, end)}); i = end + n;
      } else i += n;
    }
    return found;
  }
  function scan(text) {
    const found = [];
    for (let i = 0; i < text.length;) {
      const math = mathEnd(text, i);
      if (math > i) { i = math; continue; }
      if (text.startsWith("](", i)) {
        const end = text.indexOf(")", i + 2);
        if (end >= 0) { i = end + 1; continue; }
      }
      if (text.startsWith("https://", i) || text.startsWith("http://", i)) {
        while (i < text.length && !/\s/.test(text[i])) i++;
        continue;
      }
      if (text[i] !== "`" && text[i] !== "~") { i++; continue; }
      let n = 1; while (text[i + n] === text[i]) n++;
      if (escaped(text, i)) { i += n; continue; }
      const line = text.lastIndexOf("\n", i - 1) + 1;
      const prefix = text.slice(line, i), standalone = /^ {0,3}$/.test(prefix);
      const nl = text.indexOf("\n", i + n), lineEnd = nl < 0 ? text.length : nl;
      const info = text.slice(i + n, lineEnd).replace(/\r$/, "");
      const tag = info.match(/^([A-Za-z][A-Za-z0-9_.+#-]{0,31})(?=[ \t]|$)/);
      // Same-line paired ticks are inline code, not a broken fenced block.
      const paired = text[i] === "`" ? tickEnd(text, i, n) : -1;
      // A one-line snippet needs a known language and a real body. A bare
      // ```python``` or arbitrary multi-tick span keeps its inline meaning.
      const tail = text[i] === "`" ? info.match(/(`{3,})[ \t]*$/) : null;
      const inlineClose = tail && tail[1].length >= n ? i + n + tail.index : -1;
      if (n >= 3 && tag && languages.has(tag[0].toLowerCase()) && inlineClose >= 0) {
        const code = text.slice(i + n + tag[0].length, inlineClose).replace(/^[ \t]/, "");
        if (code.trim()) {
          const end = inlineClose + tail[1].length;
          found.push({start: i, end, lang: tag[0], code, complete: true});
          i = end; continue;
        }
      }
      const quotedPair = tag && paired >= 0 && paired < lineEnd && quoted(text, i + n + tag[0].length, paired);
      const misplaced = !standalone && prefix.trim() && text[i] === "`" && tag && (!(paired >= 0 && paired < lineEnd) || quotedPair);
      if (n >= 3 && (standalone || misplaced) && !(text[i] === "`" && info.includes("`") && !tag)) {
        const bodyAt = nl < 0 ? text.length : nl + 1;
        const pattern = new RegExp("^ {0,3}" + text[i] + "{" + n + ",}[ \\t]*(?:\\r?\\n|$)", "gm");
        pattern.lastIndex = bodyAt;
        const close = pattern.exec(text);
        let code = text.slice(bodyAt, close ? close.index : text.length);
        // Only the separator after the language is removed. Further indentation
        // and the model's comment/code contents remain verbatim.
        const first = tag ? info.slice(tag[0].length).replace(/^[ \t]/, "") : "";
        if (first.trim()) code = first + (nl >= 0 ? "\n" + code : "");
        if (close) code = code.replace(/\r?\n$/, "");
        found.push({start: i, end: close ? close.index + close[0].length : text.length,
          lang: tag ? tag[0] : info.trim(), code, complete: !!close});
        i = found[found.length - 1].end;
        continue;
      }
      if (text[i] === "`" && paired >= 0) { i = paired + n; continue; }
      i += n;
    }
    return found;
  }
  const api = {scan, spans};
  root.StrataCodeFences = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
