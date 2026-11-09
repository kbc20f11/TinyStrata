// Display metadata only. Installation, download sizes and capabilities come
// from the model-management API, never from this presentation catalogue.
"use strict";
(function (root) {
  const definitions = [
    ["qwen3.6-35b-a3b", "Qwen3.6 35B-A3B", "Qwen", "qwen",
      "Most versatile, TinyStrata's default model", "一番汎用性が高い、TinyStrataのデフォルトモデル"],
    ["gemma-4-26b-a4b-it", "Gemma 4 26B-A4B", "Google", "gemma",
      "Warm conversation and natural language", "温かみのある会話、自然な語の出力"],
    ["lfm2.5-24b-a2b", "LFM2.5 24B-A2B", "Liquid AI", "lfm",
      "Fastest and most responsive, no image support", "最も軽快で高速に動作する、画像は非対応"],
    ["clover-35b-a3b-preview", "Clover 35B-A3B Preview", "DiffuSpace", "clover",
      "Balanced conversation and reasoning, slowest to run", "会話能力と知能のバランスがよい、最も低速で動作"],
    ["tiel-coder-35b-a3b", "Tiel Coder 35B-A3B", "peculiar-ragdoll", "tiel",
      "Specialized in programming questions", "プログラミングの質問に特化"],
    ["gpt-oss-20b", "gpt-oss 20B", "OpenAI", "gptoss",
      "Thinks step by step before answering, strong at reasoning", "答える前にじっくり考える、推論に強い"],
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
