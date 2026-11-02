// All dependencies are pinned local assets. Never import a path from a fence.
"use strict";
importScripts("vendor/highlight/highlight.min.js");
const extras = new Set(["powershell", "cmake", "dockerfile", "latex"]);
hljs.safeMode();
self.onmessage = ({data}) => {
  const {id, lang, code} = data || {};
  if (!Number.isSafeInteger(id) || typeof lang !== "string" || typeof code !== "string" || code.length > 100000) return;
  try {
    if (extras.has(lang) && !hljs.getLanguage(lang)) importScripts(`vendor/highlight/languages/${lang}.min.js`);
    const html = hljs.getLanguage(lang) ? hljs.highlight(code, {language: lang, ignoreIllegals: true}).value : null;
    self.postMessage({id, html});
  } catch (_) { self.postMessage({id, html: null}); }
};
