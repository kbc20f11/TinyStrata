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
MODEL_CONTEXT_LIMITS = {QWEN_MODEL: 262144, TIEL_MODEL: 262144, GEMMA_MODEL: 262144,
                        LFM_MODEL: 131072, CLOVER_MODEL: 131072}
TARGET_ONLY_MODELS = frozenset((GEMMA_MODEL, LFM_MODEL, CLOVER_MODEL))
QWEN_ENGINE_MODELS = frozenset((QWEN_MODEL, TIEL_MODEL))
VISION_MODELS = frozenset((QWEN_MODEL, TIEL_MODEL, GEMMA_MODEL, CLOVER_MODEL))
VISION_PROTOCOL = "geni_sve1"
VISION_CONTRACT_VERSION = 1
EXPERIMENTAL_KV_MODELS = {"fp8": [QWEN_MODEL]}
FP8_KV_FORMAT = "e4m3fn_block32_fp16_scale"
MODEL_CONTRACTS = {
    GEMMA_MODEL: {"version": 1, "architecture": "gemma4", "kv_types": ["int8"], "mtp": False,
                  "experimental_mtp_max_depth": 0, "images": False, "thinking": "on_off", "mode": "target_only", "context_max": 262144},
    LFM_MODEL: {"version": 1, "architecture": "lfm2moe", "kv_types": ["int8"], "mtp": False,
                "experimental_mtp_max_depth": 0, "images": False, "thinking": "mandatory", "mode": "target_only", "context_max": 131072},
    CLOVER_MODEL: {"version": 1, "architecture": "clover_moe", "kv_types": ["int8"], "mtp": False,
                   "experimental_mtp_max_depth": 0, "images": False, "thinking": "on_off", "mode": "causal_d0", "context_max": 131072},
}
VISION_MODEL_CONTRACTS = {
    model: {**MODEL_CONTRACTS[model], "version": 2, "images": True}
    for model in (GEMMA_MODEL, CLOVER_MODEL)
}


def runtime_features(model: str, *, images: bool = False) -> dict:
    """Frontend contract of this route, independent of CUDA family features.

    These are available feature choices, not claims about hardware validation,
    current MTP depth, or measured performance. Native compatibility is checked
    separately before a model is activated.
    """
    if model not in MODEL_PROFILES:
        raise ValueError("Unknown experimental Intel model identity")
    return {"images": bool(images and model in VISION_MODELS), "assistant": False, "lookup": False,
            "mtp": model in QWEN_ENGINE_MODELS, "target_only": model in TARGET_ONLY_MODELS,
            "clover_draft": False, "clover_modes": ["causal"] if model == CLOVER_MODEL else [],
            "kv_types": ["int8"] if model in TARGET_ONLY_MODELS else ["int8", "fp32"]}
