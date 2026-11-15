"""Pure identities/contracts for experimental Intel text engines.

Listing an identity does not install a runtime or establish hardware/quality
support. New families require their own explicit executable and native contract.
"""
QWEN_MODEL = "qwen3.6-35b-a3b"
TIEL_MODEL = "tiel-coder-35b-a3b"
GEMMA_MODEL = "gemma-4-26b-a4b-it"
LFM_MODEL = "lfm2.5-24b-a2b"
CLOVER_MODEL = "clover-35b-a3b-preview"
MODEL_PROFILES = {QWEN_MODEL: "b390-qwen36", TIEL_MODEL: "b390-tiel-coder",
                  GEMMA_MODEL: "b390-gemma4", LFM_MODEL: "b390-lfm25", CLOVER_MODEL: "b390-clover"}
MODEL_ARCHES = {QWEN_MODEL: "qwen35moe", TIEL_MODEL: "qwen35moe", GEMMA_MODEL: "gemma4",
                LFM_MODEL: "lfm2moe", CLOVER_MODEL: "clover_moe"}
MODEL_CONFIGS = {QWEN_MODEL: "config-sycl.json", TIEL_MODEL: "config-tiel-coder-sycl.json",
                 GEMMA_MODEL: "config-gemma4-sycl.json", LFM_MODEL: "config-lfm2-sycl.json",
                 CLOVER_MODEL: "config-clover-sycl.json"}
MODEL_LOGS = {QWEN_MODEL: "engine-intel.log", TIEL_MODEL: "engine-intel-tiel.log",
              GEMMA_MODEL: "engine-intel-gemma4.log", LFM_MODEL: "engine-intel-lfm2.log",
              CLOVER_MODEL: "engine-intel-clover.log"}
TARGET_ONLY_MODELS = frozenset((GEMMA_MODEL, LFM_MODEL, CLOVER_MODEL))
QWEN_ENGINE_MODELS = frozenset((QWEN_MODEL, TIEL_MODEL))
MODEL_CONTRACTS = {
    GEMMA_MODEL: {"version": 1, "architecture": "gemma4", "kv_types": ["int8"], "mtp": False,
                  "experimental_mtp_max_depth": 0, "images": False, "thinking": "on_off", "mode": "target_only"},
    LFM_MODEL: {"version": 1, "architecture": "lfm2moe", "kv_types": ["int8"], "mtp": False,
                "experimental_mtp_max_depth": 0, "images": False, "thinking": "mandatory", "mode": "target_only"},
    CLOVER_MODEL: {"version": 1, "architecture": "clover_moe", "kv_types": ["int8"], "mtp": False,
                   "experimental_mtp_max_depth": 0, "images": False, "thinking": "on_off", "mode": "causal_d0"},
}


def runtime_features(model: str) -> dict:
    """Frontend contract of this route, independent of CUDA family features.

    These are available feature choices, not claims about hardware validation,
    current MTP depth, or measured performance. Native compatibility is checked
    separately before a model is activated.
    """
    if model not in MODEL_PROFILES:
        raise ValueError("Unknown experimental Intel model identity")
    return {"images": False, "assistant": False, "lookup": False,
            "mtp": model in QWEN_ENGINE_MODELS, "target_only": model in TARGET_ONLY_MODELS,
            "clover_draft": False, "clover_modes": ["causal"] if model == CLOVER_MODEL else [],
            "kv_types": ["int8"] if model in TARGET_ONLY_MODELS else ["int8", "fp32"]}
