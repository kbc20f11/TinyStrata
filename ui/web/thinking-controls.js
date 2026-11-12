// Model-native thinking choices. Effort is prompt steering, not a token budget.
(function (root) {
  "use strict";
  const definitions = {
    toggle: {mode: "toggle", default: "on", options: ["none", "on"]},
    always: {mode: "always", default: "on", options: ["on"]},
    effort: {mode: "effort", default: "medium", options: ["none", "low", "medium", "xhigh"]}
  };
  const modes = {"qwen3.6-35b-a3b": "toggle", "clover-35b-a3b-preview": "toggle",
    "gemma-4-26b-a4b-it": "toggle", "lfm2.5-24b-a2b": "always", "tiel-coder-35b-a3b": "effort"};
  function contract(model, metadata) {
    const candidate = definitions[metadata?.mode];
    if (candidate && metadata.default === candidate.default && Array.isArray(metadata.options)
        && metadata.options.length === candidate.options.length
        && metadata.options.every((value, index) => value === candidate.options[index])) return candidate;
    return definitions[modes[model] || "toggle"];
  }
  function migrate(value) {
    const result = {...value}, saved = value?.thinkingByModel;
    result.thinkingByModel = {};
    // Preserve meaningful explicit legacy choices. The old global OFF works
    // for every switchable model; low/medium have a native meaning for Tiel.
    if (value?.thinking === "none") {
      for (const model of Object.keys(modes)) if (contract(model).options.includes("none")) result.thinkingByModel[model] = "none";
    } else if (["low", "medium"].includes(value?.thinking)) {
      result.thinkingByModel["tiel-coder-35b-a3b"] = value.thinking;
    }
    if (saved && typeof saved === "object" && !Array.isArray(saved)) {
      for (const [model, value] of Object.entries(saved)) {
        if (Object.hasOwn(modes, model) && contract(model).options.includes(value)) result.thinkingByModel[model] = value;
      }
    }
    // The retired global factory value "high" was not a per-model preference.
    // In particular it must not change Tiel's native medium default to xhigh.
    delete result.thinking;
    return result;
  }
  function choice(model, settings, metadata) {
    const spec = contract(model, metadata), value = settings?.thinkingByModel?.[model];
    return spec.options.includes(value) ? value : spec.default;
  }
  function select(settings, model, value, metadata) {
    const result = migrate(settings), spec = contract(model, metadata);
    if (!spec.options.includes(value)) return result;
    result.thinkingByModel[model] = value;
    return result;
  }
  function payload(model, settings, metadata, shared = false) {
    const spec = contract(model, metadata), value = choice(model, settings, metadata);
    if (spec.mode === "effort") return {reasoning_effort: value};
    const enable_thinking = value === "on";
    return shared ? {enable_thinking} : {chat_template_kwargs: {enable_thinking}};
  }
  root.StrataThinkingControls = Object.freeze({contract, migrate, choice, select, payload});
})(globalThis);
