// Completed LaTeX fences: a small document preview, using the existing local
// KaTeX renderer for equations. This is not a TeX compiler: no files, packages,
// external resources or user macros are executed. Unsupported syntax stays code.
"use strict";
(function (root) {
  const LIMIT = 65536, MAX_MATH = 128, MAX_DEPTH = 24;
  const escape = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  const MATH_ENVS = new Set(("math displaymath equation equation* align align* alignat alignat* aligned alignedat gather gather* gathered multline multline* split " +
    "matrix pmatrix bmatrix Bmatrix vmatrix Vmatrix smallmatrix cases array").split(" "));
  const PACKAGES = new Set("amsmath amssymb amsfonts amsthm mathtools geometry inputenc fontenc babel polyglossia fontspec xeCJK ctex unicode-math".split(" "));
  const CLASSES = new Set("article report book minimal standalone jsarticle jarticle bxjsarticle ltjsarticle ctexart".split(" "));
  class Unsupported extends Error {}
  function comments(text) {
    let out = "";
    for (let i = 0; i < text.length; i++) {
      if (text[i] === "\\" && i + 1 < text.length) { out += text[i] + text[++i]; continue; }
      if (text[i] === "%") { while (i < text.length && text[i] !== "\n") i++; if (i < text.length) out += "\n"; }
      else out += text[i];
    }
    return out;
  }
  class Parser {
    constructor(text, labels) {
      this.text = text; this.at = 0; this.math = []; this.labels = labels;
      this.meta = {}; this.document = false; this.closed = false; this.section = [0, 0, 0];
      this.theorems = new Map(); this.counters = new Map();
      for (const name of "theorem lemma proposition corollary definition remark example".split(" "))
        this.theorems.set(name, {title: name[0].toUpperCase() + name.slice(1), counter: name, numbered: true});
    }
    fail(reason = "unsupported") { throw new Unsupported(reason); }
    depth(n) { if (n > MAX_DEPTH) this.fail("limit"); }
    space() { while (/\s/.test(this.text[this.at] || "") && this.at < this.text.length) this.at++; }
    command() {
      if (this.text[this.at] !== "\\") this.fail("syntax");
      this.at++;
      const found = /^[A-Za-z]+\*?/.exec(this.text.slice(this.at));
      if (found) { this.at += found[0].length; return found[0]; }
      if (this.at >= this.text.length) this.fail("syntax");
      return this.text[this.at++];
    }
    group(open = "{", close = "}") {
      this.space();
      if (this.text[this.at++] !== open) this.fail("syntax");
      const start = this.at; let depth = 1;
      while (this.at < this.text.length) {
        const ch = this.text[this.at++];
        if (ch === "\\") { if (this.at < this.text.length) this.at++; continue; }
        if (ch === open) { depth++; this.depth(depth); }
        if (ch === close && --depth === 0) return this.text.slice(start, this.at - 1);
      }
      this.fail("syntax");
    }
    optional() { this.space(); return this.text[this.at] === "[" ? this.group("[", "]") : null; }
    name() { const value = this.group(); if (!/^[A-Za-z][A-Za-z0-9*-]*$/.test(value)) this.fail("syntax"); return value; }
    slot(source, display, raw = source) {
      if (!source.trim() || this.math.length >= MAX_MATH || new TextEncoder().encode(source).length > 16384) this.fail("limit");
      const i = this.math.length; this.math.push({source, display});
      return `<span class="chat-math${display ? " chat-math--display" : ""}" data-latex-math="${i}">${escape(raw)}</span>`;
    }
    delimited() {
      const at = this.at, tail = this.text.slice(at);
      let open, close, display;
      if (tail.startsWith("\\[")) { open = "\\["; close = "\\]"; display = true; }
      else if (tail.startsWith("\\(")) { open = "\\("; close = "\\)"; display = false; }
      else if (tail.startsWith("$$")) { open = close = "$$"; display = true; }
      else { open = close = "$"; display = false; }
      this.at += open.length; const start = this.at;
      while (this.at < this.text.length) {
        if (this.text.startsWith(close, this.at)) {
          const source = this.text.slice(start, this.at);
          this.at += close.length;
          return {html: this.slot(source, display, this.text.slice(at, this.at)), display};
        }
        if (this.text[this.at] === "\\" && !this.text.startsWith(close, this.at)) this.at += 2;
        else this.at++;
      }
      this.fail("syntax");
    }
    environment(name, start) {
      const stack = [name], scan = /\\(begin|end)\s*\{([A-Za-z][A-Za-z0-9*-]*)\}/g;
      scan.lastIndex = this.at;
      let found;
      while ((found = scan.exec(this.text))) {
        // Escaped backslashes are text, not environment delimiters.
        let backslashes = 0; for (let i = found.index - 1; i >= 0 && this.text[i] === "\\"; i--) backslashes++;
        if (backslashes % 2) continue;
        if (found[1] === "begin") { stack.push(found[2]); this.depth(stack.length); }
        else if (stack.pop() !== found[2]) this.fail("syntax");
        if (!stack.length) {
          this.at = scan.lastIndex;
          let source = this.text.slice(start, this.at);
          if (name === "math" || name === "displaymath") source = this.text.slice(this.text.indexOf("}", start) + 1, found.index);
          return this.slot(source, name !== "math");
        }
      }
      this.fail("syntax");
    }
    inline(text, depth) {
      this.depth(depth);
      const original = this.text, originalAt = this.at;
      this.text = text; this.at = 0;
      const out = [];
      try {
        while (this.at < this.text.length) out.push(this.inlineOne(depth));
        return out.join("");
      } finally { this.text = original; this.at = originalAt; }
    }
    inlineOne(depth) {
      this.depth(depth);
      const ch = this.text[this.at];
      if (ch === "$" || this.text.startsWith("\\(", this.at) || this.text.startsWith("\\[", this.at)) return this.delimited().html;
      if (ch === "{") return this.inline(this.group(), depth + 1);
      if (ch === "}" || ch === "&" || ch === "_" || ch === "^" || ch === "#") this.fail("syntax");
      if (ch === "~") { this.at++; return "&nbsp;"; }
      if (ch !== "\\") {
        let end = this.at + 1;
        while (end < this.text.length && !/[\\${}&_^#~]/.test(this.text[end]) && !/^\n[ \t]*\n/.test(this.text.slice(end))) end++;
        const raw = this.text.slice(this.at, end); this.at = end;
        return escape(raw.replace(/\s+/g, " "));
      }
      const command = this.command();
      const tags = {textbf: "strong", emph: "em", textit: "em", texttt: "code", textrm: "span", textsf: "span", textnormal: "span", underline: "u"};
      if (Object.hasOwn(tags, command)) return `<${tags[command]}>${this.inline(this.group(), depth + 1)}</${tags[command]}>`;
      if ("%$&#_{}".includes(command) && command.length === 1) return escape(command);
      if (command === "\\" || command === "newline" || command === "linebreak") return "<br>";
      if ([" ", ",", ";", ":", "!", "quad", "qquad", "enspace", "thinspace"].includes(command)) return " ";
      if (command === "label") { this.group(); return ""; }
      if (command === "LaTeX") return "LaTeX";
      if (command === "TeX") return "TeX";
      if (command === "and") return " · ";
      if (command === "qed") return '<span class="st-latex__qed">□</span>';
      if (command === "ldots" || command === "dots") return "…";
      if (command === "textbackslash") { this.space(); if (this.text.startsWith("{}", this.at)) this.at += 2; return "\\"; }
      this.fail("unsupported");
    }
    endName() {
      const found = /^\\end\s*\{([A-Za-z][A-Za-z0-9*-]*)\}/.exec(this.text.slice(this.at));
      return found ? {name: found[1], length: found[0].length} : null;
    }
    list(name, depth) {
      this.depth(depth); const items = [];
      for (;;) {
        this.space(); const end = this.endName();
        if (end) { if (end.name !== name) this.fail("syntax"); this.at += end.length; break; }
        if (!/^\\item\b/.test(this.text.slice(this.at))) this.fail("syntax");
        this.command(); const label = this.optional();
        const body = this.body(name, depth + 1, true);
        items.push(`<li>${label === null ? "" : `<strong>${this.inline(label, depth + 1)}</strong> `}${body}</li>`);
      }
      const tag = name === "enumerate" ? "ol" : "ul";
      return `<${tag}>${items.join("")}</${tag}>`;
    }
    metadata(command, depth) {
      if (["title", "author", "date"].includes(command)) { this.meta[command] = this.group(); return ""; }
      if (command === "documentclass") {
        this.optional(); const name = this.name(); if (!CLASSES.has(name)) this.fail("unsupported"); return "";
      }
      if (command === "usepackage") {
        this.optional(); const names = this.group().split(",").map((x) => x.trim());
        if (names.some((name) => !PACKAGES.has(name))) this.fail("unsupported");
        this.optional(); return "";
      }
      if (command === "geometry") { this.group(); return ""; }
      if (command === "theoremstyle") { this.group(); return ""; }
      if (command === "newtheorem" || command === "newtheorem*") {
        const name = this.name(), shared = this.optional(), title = this.group(), within = this.optional();
        if (MATH_ENVS.has(name) || ["document", "itemize", "enumerate", "proof"].includes(name)) this.fail("unsupported");
        if (within !== null || (shared !== null && !this.theorems.has(shared))) this.fail("unsupported");
        this.theorems.set(name, {title, counter: shared === null ? name : this.theorems.get(shared).counter, numbered: command !== "newtheorem*"}); return "";
      }
      if (command === "maketitle") {
        return ["title", "author", "date"].map((key) => {
          if (!this.meta[key]) return "";
          const tag = key === "title" ? "h2" : "div";
          return `<${tag} class="st-latex__${key}">${this.inline(this.meta[key], depth + 1)}</${tag}>`;
        }).join("");
      }
      return null;
    }
    body(endEnv = null, depth = 0, stopAtItem = false) {
      this.depth(depth);
      const blocks = [], words = [];
      const flush = () => { const text = words.join("").trim(); if (text) blocks.push(`<p>${text}</p>`); words.length = 0; };
      while (this.at < this.text.length) {
        const end = this.endName();
        if (end) {
          if (end.name !== endEnv) this.fail("syntax");
          if (!stopAtItem) this.at += end.length;
          flush(); return blocks.join("");
        }
        if (stopAtItem && /^\\item\b/.test(this.text.slice(this.at))) { flush(); return blocks.join(""); }
        const gap = /^\n[ \t]*\n(?:\s*\n)*/.exec(this.text.slice(this.at));
        if (gap) { this.at += gap[0].length; flush(); continue; }
        if (this.text.startsWith("$$", this.at) || this.text.startsWith("\\[", this.at)) { flush(); blocks.push(this.delimited().html); continue; }
        if (this.text[this.at] === "\\" && /^\\[A-Za-z]/.test(this.text.slice(this.at))) {
          const start = this.at, command = this.command();
          const meta = this.metadata(command, depth);
          if (meta !== null) { flush(); blocks.push(meta); continue; }
          const headings = {section: 0, subsection: 1, subsubsection: 2}, heading = command.replace(/\*$/, "");
          const level = Object.hasOwn(headings, heading) ? headings[heading] : undefined;
          if (level !== undefined) {
            flush(); this.optional(); const title = this.inline(this.group(), depth + 1);
            let number = "";
            if (!command.endsWith("*")) { this.section[level]++; this.section.fill(0, level + 1); number = this.section.slice(0, level + 1).join(".") + " "; }
            blocks.push(`<h${level + 3} class="st-latex__heading">${number}${title}</h${level + 3}>`); continue;
          }
          if (command === "par") { flush(); continue; }
          if (command === "begin") {
            flush(); const name = this.name();
            if (MATH_ENVS.has(name)) { blocks.push(this.environment(name, start)); continue; }
            if (name === "document") {
              if (this.document || this.closed || endEnv) this.fail("syntax");
              this.document = true; blocks.push(this.body("document", depth + 1)); this.closed = true;
              this.space(); if (this.at !== this.text.length) this.fail("syntax");
              continue;
            }
            if (name === "itemize" || name === "enumerate") { blocks.push(this.list(name, depth + 1)); continue; }
            if (name === "proof" || this.theorems.has(name)) {
              const optional = this.optional(), theorem = this.theorems.get(name);
              let title;
              if (name === "proof") title = optional === null ? escape(this.labels.proof) : this.inline(optional, depth + 1);
              else {
                let number = "";
                if (theorem.numbered) { const count = (this.counters.get(theorem.counter) || 0) + 1; this.counters.set(theorem.counter, count); number = ` ${count}`; }
                title = this.inline(theorem.title, depth + 1) + number + (optional === null ? "" : ` (${this.inline(optional, depth + 1)})`);
              }
              const type = name === "proof" ? "proof" : "theorem";
              blocks.push(`<section class="st-latex__${type}"><strong class="st-latex__${type}-title">${title}</strong>${this.body(name, depth + 1)}${type === "proof" ? '<span class="st-latex__qed">□</span>' : ""}</section>`); continue;
            }
            this.fail("unsupported");
          }
          this.at = start;
        }
        words.push(this.inlineOne(depth + 1));
      }
      if (endEnv) this.fail("syntax");
      flush(); return blocks.join("");
    }
  }
  function parse(source, labels = {}) {
    try {
      if (typeof source !== "string" || !source.trim()) return {ok: false, reason: "empty"};
      if (new TextEncoder().encode(source).length > LIMIT) return {ok: false, reason: "limit"};
      const text = comments(source.replace(/\r\n?/g, "\n")).trim();
      const parser = new Parser(text, {proof: "Proof", ...labels});
      // A bare formula in a latex/tex fence has no prose/document delimiters.
      const firstCommand = /\\([A-Za-z]+)/.exec(text);
      const mathCommand = firstCommand && /^(?:frac|dfrac|tfrac|sqrt|sum|prod|int|lim|sin|cos|tan|alpha|beta|gamma|delta|theta|pi|rightarrow|left|math[a-z]+)$/.test(firstCommand[1]);
      const bare = !/[\n][ \t]*\n|\$|\\[([\]]|\\(?:documentclass|usepackage|title|author|date|maketitle|section|subsection|subsubsection|begin|newtheorem|textbf|emph|textit|texttt)\b/.test(text) &&
        (mathCommand || (!firstCommand && /[_^=+*/<>-]/.test(text)));
      const html = bare ? parser.slot(text, true) : parser.body();
      if (!html.trim()) return {ok: false, reason: "empty"};
      return {ok: true, html, math: parser.math};
    } catch (e) { return {ok: false, reason: e instanceof Unsupported ? e.message : "invalid"}; }
  }
  function failed(box, reason) {
    box.dataset.latexState = "error"; box.dataset.latexView = "code";
    const status = box.querySelector(".st-latex__status"), toggle = box.querySelector("[data-latex-toggle]");
    if (status) { status.hidden = false; status.textContent = root.tr ? root.tr("latex.error", "Could not preview this LaTeX. The source is shown below.") : "Could not preview this LaTeX. The source is shown below."; }
    if (toggle) { toggle.hidden = true; toggle.setAttribute("aria-pressed", "false"); }
    box.dataset.latexError = reason;
  }
  function hydrate(container, enabled = () => true) {
    const generation = container._latexGeneration = (container._latexGeneration || 0) + 1;
    for (const box of container.querySelectorAll('[data-latex="true"]')) {
      if (!enabled() || box.closest('.st-bubble')?.dataset.streaming === "true") continue;
      const code = box.querySelector("pre code"), host = box.querySelector(".st-latex__preview");
      if (!code || !host) continue;
      const parsed = parse(code.textContent, {proof: root.tr ? root.tr("latex.proof", "Proof") : "Proof"});
      if (!parsed.ok) { failed(box, parsed.reason); continue; }
      const current = () => enabled() && generation === container._latexGeneration && container.contains(box) && box.closest('.st-bubble')?.dataset.streaming !== "true";
      const work = parsed.math.map((formula) => root.StrataMath ? root.StrataMath.result(formula.source, formula.display) : Promise.resolve({ok: false}));
      Promise.all(work).then((results) => {
        if (!current()) return;
        if (results.some((r) => !r.ok)) { failed(box, "math"); return; }
        const content = document.createElement("div");
        // The parser emits only fixed tags and escaped text, never raw TeX HTML.
        content.innerHTML = parsed.html;
        for (const slot of content.querySelectorAll("[data-latex-math]")) slot.replaceChildren(results[Number(slot.dataset.latexMath)].element.cloneNode(true));
        host.replaceChildren(...content.childNodes);
        box.dataset.latexState = "ready";
        if (box.dataset.latexView !== "code") box.dataset.latexView = "document";
        const status = box.querySelector(".st-latex__status"), toggle = box.querySelector("[data-latex-toggle]");
        if (status) status.hidden = true;
        if (toggle) { toggle.hidden = false; toggle.setAttribute("aria-pressed", String(box.dataset.latexView === "document")); }
      }).catch(() => { if (current()) failed(box, "unavailable"); });
    }
  }
  const api = {parse, hydrate};
  root.StrataLatex = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
