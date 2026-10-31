// serve/web/i18n.js - the web page's translations.
//
// The English text sits in the code, next to the key:  tr("chat.rewind", "Rewind to here")  and
// tr("chat.hidden", {one: "{n} message hidden", other: "{n} messages hidden"}, {n: 3}).  Static page text is marked
// data-i18n="key" (also data-i18n-title, data-i18n-label, data-i18n-placeholder) and holds its English text in the HTML.
// So the page reads fine with no catalog at all.  Every other language is a catalog, serve/web/locales/<lang>.json
// (a flat {"key": "text"} object; a plural is {"one": "...", "other": "..."}, the CLDR names of the language).
// serve/test_web_i18n.py keeps the catalogs complete: same keys as the English text in the code, same {placeholders}.
//
// The language is the user's choice (About > Settings), kept in this browser.  Nothing is chosen for them from the
// browser's language: a page that is translated in part would mix two languages.
"use strict";
(function (root) {
  // [code, the language's own name].  A new language is a locales/<code>.json file plus one line here.
  // Match TinyStrata's README languages. Old saved language codes still work.
  const LANGUAGES = [["en", "English"], ["zh-CN", "简体中文"], ["ja", "日本語"], ["de", "Deutsch"],
                     ["fr", "Français"], ["es", "Español"], ["pt-BR", "Português (Brasil)"]];
  let lang = "en", catalog = null, rules = null;
  let languageEpoch = 0;
  function normalizeLanguage(code) {
    const value = String(code || "en").replace(/_/g, "-").toLowerCase();
    if (value === "zh" || value.startsWith("zh-")) return "zh-CN";
    if (value === "pt" || value.startsWith("pt-")) return "pt-BR";
    const base = value.split("-")[0];
    return LANGUAGES.some(([key]) => key === base) ? base : "en";
  }

  const fill = (s, vars) => String(s).replace(/\{(\w+)\}/g, (m, k) => (vars && k in vars ? String(vars[k]) : m));
  function pick(entry, vars) {
    if (entry && typeof entry === "object") {
      const n = vars && vars.n != null ? Number(vars.n) : 1;
      let cat = "other";
      try { cat = (rules || new Intl.PluralRules(lang === "en" ? "en" : lang)).select(n); } catch (e) { /* other */ }
      return entry[cat] != null ? entry[cat] : entry.other != null ? entry.other : entry.one;
    }
    return entry;
  }
  // tr(key, english, vars): the catalog's text for the chosen language, else the English text in the code
  function tr(key, english, vars) {
    const own = catalog && Object.prototype.hasOwnProperty.call(catalog, key) ? catalog[key] : undefined;
    const text = pick(own != null ? own : english, vars);
    return fill(text == null ? key : text, vars);
  }

  // marked page text: the English text is read once from the HTML and kept, so switching back restores it
  const ATTRS = [["data-i18n", null], ["data-i18n-title", "title"], ["data-i18n-label", "aria-label"],
                 ["data-i18n-placeholder", "placeholder"]];
  function apply(scope) {
    const doc = scope || root.document;
    if (!doc || !doc.querySelectorAll) return;
    for (const [mark, attr] of ATTRS) {
      for (const el of doc.querySelectorAll(`[${mark}]`)) {
        const key = el.getAttribute(mark);
        const slot = "en_" + mark.replace(/[^a-z]/g, "");
        if (el.dataset[slot] === undefined) el.dataset[slot] = attr ? (el.getAttribute(attr) || "") : el.textContent;
        const text = tr(key, el.dataset[slot]);
        if (attr) el.setAttribute(attr, text); else el.textContent = text;
      }
    }
    if (root.document && root.document.documentElement) root.document.documentElement.lang = lang;
  }

  // load a catalog; English, or a catalog that cannot be read, leaves the English text in the code
  async function setLanguage(code) {
    const epoch = ++languageEpoch;
    code = normalizeLanguage(code);
    let next = null;
    if (code !== "en") {
      try {
        const r = await root.fetch(`web/locales/${code}.json?v=20261031.5`, {cache: "no-store"});
        const j = r.ok ? await r.json() : null;
        if (j && typeof j === "object" && !Array.isArray(j)) next = j;
      } catch (e) { /* the English text stays */ }
      if (!next) code = "en";
    }
    if (epoch !== languageEpoch) return lang;
    lang = code; catalog = next; rules = null;
    try { rules = new Intl.PluralRules(lang); } catch (e) { /* other */ }
    apply();
    return lang;
  }

  const api = {LANGUAGES, tr, apply, setLanguage, normalizeLanguage, get lang() { return lang; }, fill, pick};
  root.StrataI18n = api;
  root.tr = tr;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
