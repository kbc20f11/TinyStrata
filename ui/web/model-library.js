// Display metadata only. Installation, download sizes and capabilities come
// from the model-management API, never from this presentation catalogue.
"use strict";
(function (root) {
  const definitions = [
    ["qwen3.6-35b-a3b", "Qwen3.6 35B-A3B", "Qwen", "qwen",
      "General-purpose chat, writing and coding", "会話・文章作成・プログラミングに"],
    ["gemma-4-26b-a4b-it", "Gemma 4 26B-A4B", "Google", "gemma",
      "Conversation, explanations and writing", "会話・説明・文章作成に"],
    ["lfm2.5-24b-a2b", "LFM2.5 24B-A2B", "Liquid AI", "lfm",
      "Everyday text chat and writing", "日常のテキスト会話や文章作成に"],
    ["clover-35b-a3b-preview", "Clover 35B-A3B Preview", "DiffuSpace", "clover",
      "Preview model for chat and text tasks", "会話や文章作成向けのプレビューモデル"],
    ["tiel-coder-35b-a3b", "Tiel Coder 35B-A3B", "peculiar-ragdoll", "tiel",
      "Code generation, debugging and edits", "コード作成・不具合調査・修正に"],
  ];
  const iconSources = {qwen: "web/model-logos/qwen.png", gemma: "web/model-logos/gemma.svg",
    lfm: "web/model-logos/liquid.png", clover: "web/model-logos/diffuspace.jpg", tiel: "web/model-icons.svg"};
  const MODELS = Object.freeze(definitions.map(([id, name, provider, mark, en, ja]) => Object.freeze({
    id, name, provider,
    descriptionKey: `models.description.${mark}`,
    descriptionFallback: en,
    summary: Object.freeze({en, ja}),
    icon: mark,
    iconSrc: iconSources[mark],
    iconHref: iconSources[mark],
  })));
  const byId = new Map(MODELS.map((model) => [model.id, model]));
  const get = (id) => byId.get(id) || null;
  function describe(id, language = "en") {
    const model = get(id);
    if (!model) return null;
    return Object.freeze({...model, descriptionFallback: model.summary[String(language).split("-")[0]] || model.summary.en});
  }
  const api = Object.freeze({MODELS, get, describe, list: () => MODELS});
  root.StrataModelLibrary = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(globalThis);
