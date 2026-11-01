// This document has an opaque sandbox origin and cannot access the chat, its
// storage or the network. Do not bind Mermaid's interactive functions.
"use strict";
(function () {
  let serial = 0;
  const engine = window.mermaid;
  const limits = {startOnLoad: false, securityLevel: "strict", htmlLabels: false, layout: "dagre",
    maxTextSize: 32768, maxEdges: 300, suppressErrorRendering: true, fontFamily: "system-ui, sans-serif"};
  engine.initialize(limits);
  const secure = [...new Set([...Object.keys(engine.mermaidAPI.getConfig()), ...Object.keys(limits), "secure", "altFontFamily", "dompurifyConfig"])];
  window.addEventListener("message", async (event) => {
    const message = event.data;
    if (event.source !== parent || message?.type !== "tinystrata-diagram-render" || !Number.isSafeInteger(message.id)) return;
    const reply = (data) => parent.postMessage({type: "tinystrata-diagram-result", id: message.id, ...data}, "*");
    if (typeof message.source !== "string" || message.source.length > limits.maxTextSize) { reply({ok: false}); return; }
    const dark = message.dark === true;
    try {
      engine.initialize({...limits, secure, theme: "base", themeVariables: {
        darkMode: dark, background: dark ? "#030907" : "#f3f7f0",
        primaryColor: dark ? "#123b28" : "#e2efdc", primaryTextColor: dark ? "#d9f2e2" : "#234b32",
        primaryBorderColor: dark ? "#62c595" : "#397a52", lineColor: dark ? "#8ec4a5" : "#497e58",
        secondaryColor: dark ? "#18382c" : "#e8f2e1", tertiaryColor: dark ? "#09251b" : "#f0f6ec",
        textColor: dark ? "#d9f2e2" : "#234b32", fontFamily: "system-ui, sans-serif"
      }});
      const host = document.getElementById("diagram");
      host.replaceChildren();
      const {svg} = await engine.render("tinystrata-diagram-" + (++serial), message.source, host);
      const doc = new DOMParser().parseFromString(svg, "image/svg+xml"), image = doc.documentElement;
      if (doc.querySelector("parsererror") || image.localName !== "svg") throw new Error("Invalid SVG");
      const view = image.getAttribute("viewBox")?.trim().split(/[\s,]+/).map(Number);
      if (view?.length === 4 && view.every(Number.isFinite) && view[2] > 0 && view[3] > 0) {
        image.setAttribute("width", String(view[2])); image.setAttribute("height", String(view[3]));
      }
      reply({ok: true, svg: new XMLSerializer().serializeToString(image)});
      host.replaceChildren();
    } catch (_) { document.getElementById("diagram").replaceChildren(); reply({ok: false}); }
  });
  parent.postMessage({type: "tinystrata-diagram-ready"}, "*");
})();
