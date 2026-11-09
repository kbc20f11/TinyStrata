"""Model thinking capabilities and API aliases, independent of token budgets."""
from __future__ import annotations

from copy import deepcopy


_MODELS = {
    "qwen3.6-35b-a3b": "toggle", "clover-35b-a3b-preview": "toggle",
    "gemma-4-26b-a4b-it": "toggle", "lfm2.5-24b-a2b": "always",
    "tiel-coder-35b-a3b": "effort",
    # gpt-oss reasons at low, medium or high effort and cannot switch it off
    "gpt-oss-20b": "effort_always",
}
_ALIASES = {
    "qwen36": "qwen3.6-35b-a3b", "qwen3.6": "qwen3.6-35b-a3b", "qwen35moe": "qwen3.6-35b-a3b",
    "clover": "clover-35b-a3b-preview", "clover_moe": "clover-35b-a3b-preview",
    "gemma4": "gemma-4-26b-a4b-it", "gemma-4": "gemma-4-26b-a4b-it",
    "gemma4-26b-a4b": "gemma-4-26b-a4b-it", "gemma4-26b-a4b-it": "gemma-4-26b-a4b-it",
    "lfm2": "lfm2.5-24b-a2b", "lfm2.5": "lfm2.5-24b-a2b", "lfm2moe": "lfm2.5-24b-a2b",
    "tiel": "tiel-coder-35b-a3b", "tiel-coder": "tiel-coder-35b-a3b",
    "gpt-oss": "gpt-oss-20b", "gptoss": "gpt-oss-20b",
}


def _is_effort(mode: str) -> bool:
    return mode in ("effort", "effort_always")


def capabilities(model: str, arch: str | None = None) -> dict | None:
    """Return a fresh catalog value; unknown/custom models keep their legacy controls."""
    name = str(model).strip().lower()
    mode = _MODELS.get(_ALIASES.get(name, name))
    if mode is None:
        return None
    return {"mode": mode, "default": "medium" if _is_effort(mode) else "on",
            "options": ["none", "low", "medium", "xhigh"] if mode == "effort" else
                       ["low", "medium", "xhigh"] if mode == "effort_always" else
                       ["on"] if mode == "always" else ["none", "on"]}


def normalize_effort(value, policy: dict) -> str:
    """Use translator-compatible aliases; binary 'on' never reaches effort_kwargs."""
    key = "on" if value is True else "none" if value is False else str(value).strip().lower()
    effort = _is_effort(policy["mode"])
    if key in ("none", "off", "disabled", "false") or (key == "minimal" and not effort):
        # a model that cannot stop reasoning takes the least of it
        result = "low" if policy["mode"] == "effort_always" else "none"
    elif key in ("on", "enabled", "true", "default", "auto"):
        result = "medium" if effort else "high"
    elif key in ("minimal", "low"):
        result = "low" if effort else "high"
    elif key == "medium":
        result = "medium" if effort else "high"
    elif key in ("high", "xhigh", "max", "maximum"):
        result = "xhigh" if effort else "high"
    else:
        raise ValueError("reasoning_effort: " + ", ".join(policy["options"]))
    return result


def explicit_control(req: dict) -> bool:
    if isinstance(req.get("enable_thinking"), bool):
        return True
    ctk = req.get("chat_template_kwargs")
    if isinstance(ctk, dict) and (isinstance(ctk.get("enable_thinking"), bool) or
                                 ctk.get("reasoning_effort") not in (None, "")):
        return True
    return (req.get("reasoning_effort") not in (None, "") or
            isinstance(req.get("reasoning"), dict) and req["reasoning"].get("effort") not in (None, "") or
            isinstance(req.get("thinking"), dict) or
            isinstance(req.get("output_config"), dict) and req["output_config"].get("effort") not in (None, ""))


def normalize_request(req: dict, policy: dict | None) -> dict:
    """Normalize before generic translators (where minimal otherwise means OFF)."""
    req = deepcopy(req)
    if policy is None:
        return req
    for owner, key in ((req, "reasoning_effort"),
                       (req.get("reasoning"), "effort"),
                       (req.get("output_config"), "effort"),
                       (req.get("chat_template_kwargs"), "reasoning_effort")):
        if isinstance(owner, dict) and owner.get(key) not in (None, ""):
            owner[key] = normalize_effort(owner[key], policy)
    return req


def inject_default(req: dict, api: str, policy: dict, shared: dict,
                   config_effort, default_enable: bool) -> dict:
    if explicit_control(req):
        return req
    if isinstance(shared.get("enable_thinking"), bool):
        enabled, effort = shared["enable_thinking"], None
    elif shared.get("reasoning_effort") not in (None, ""):
        enabled, effort = None, normalize_effort(shared["reasoning_effort"], policy)
    elif config_effort not in (None, ""):
        enabled, effort = None, normalize_effort(config_effort, policy)
    else:
        enabled, effort = default_enable, None
    if policy["mode"] == "always":
        # An old saved OFF preference cannot make the mandatory-thinking model unusable.
        enabled, effort = True, None
    elif policy["mode"] == "effort_always":
        # reasoning cannot be turned off: an old OFF preference gives the default effort
        enabled, effort = (None, effort or policy["default"])
    elif policy["mode"] == "effort" and enabled is not False and effort is None:
        enabled, effort = None, policy["default"]
    if effort is not None:
        if api == "responses":
            req["reasoning"] = {**(req.get("reasoning") or {}), "effort": effort}
        elif api == "anthropic":
            req["output_config"] = {**(req.get("output_config") or {}), "effort": effort}
        else:
            req["reasoning_effort"] = effort
    else:
        req["enable_thinking"] = bool(enabled)
        if api == "anthropic":
            req["thinking"] = {"type": "enabled" if enabled else "disabled"}
    return req


def template_kwargs(kwargs: dict, policy: dict | None, default_enable: bool) -> dict:
    """Canonical native template controls after protocol/explicit-bool precedence."""
    if policy is None:
        return kwargs
    kwargs = dict(kwargs)
    effort = kwargs.get("reasoning_effort")
    if effort not in (None, ""):
        effort = normalize_effort(effort, policy)
    enabled = kwargs.get("enable_thinking")
    if not isinstance(enabled, bool):
        enabled = effort != "none" if effort is not None else default_enable
    if policy["mode"] == "always" and not enabled:
        raise ValueError("LFM2.5-24B-A2B always uses thinking; disabling it is not supported")
    if policy["mode"] == "effort_always":
        enabled = True
    kwargs["enable_thinking"] = enabled
    kwargs.pop("reasoning_effort", None)
    if (policy["mode"] == "effort" and enabled) or policy["mode"] == "effort_always":
        kwargs["reasoning_effort"] = effort if effort not in (None, "none") else policy["default"]
    return kwargs
