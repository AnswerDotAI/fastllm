"High-level completion wrappers with automatic client routing."

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from fastcore.meta import delegates

from .clients import AnthropicClient, GeminiClient, OpenAIClient
from .errors import APIError
from .streaming import StreamSummary
from .types import Completion, Msg, Part, RequestOptions, ToolCall


_ENDPOINTS = ("auto", "responses", "chat")
_PROVIDER_PREFIXES = ("anthropic", "google", "openai", "gemini")

# Built-in OpenAI-compatible provider defaults for model-only usage.
_OPENAI_COMPAT_PROVIDERS = {
    "moonshot": {
        "base_url": "https://api.moonshot.ai/v1",
        "api_key_envs": ("MOONSHOT_API_KEY", "KIMI_API_KEY"),
        "host_tokens": ("moonshot.ai",),
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_envs": ("DEEPSEEK_API_KEY",),
        "host_tokens": ("deepseek.com",),
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_envs": ("GROQ_API_KEY",),
        "host_tokens": ("groq.com",),
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "api_key_envs": ("MISTRAL_API_KEY",),
        "host_tokens": ("mistral.ai",),
    },
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "api_key_envs": ("XAI_API_KEY",),
        "host_tokens": ("x.ai",),
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_envs": ("OPENROUTER_API_KEY",),
        "host_tokens": ("openrouter.ai",),
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_envs": ("TOGETHER_API_KEY",),
        "host_tokens": ("together.xyz",),
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_envs": ("FIREWORKS_API_KEY",),
        "host_tokens": ("fireworks.ai",),
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_envs": ("CEREBRAS_API_KEY",),
        "host_tokens": ("cerebras.ai",),
    },
    "perplexity": {
        "base_url": "https://api.perplexity.ai",
        "api_key_envs": ("PERPLEXITY_API_KEY",),
        "host_tokens": ("perplexity.ai",),
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_envs": ("QWEN_API_KEY", "DASHSCOPE_API_KEY"),
        "host_tokens": ("dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"),
    },
}
_MODEL_VENDOR_HINTS = {
    "kimi": "moonshot",
    "moonshot": "moonshot",
    "deepseek": "deepseek",
    "qwen": "qwen",
    "grok": "xai",
    "mistral": "mistral",
}


def _host(base_url: str = "") -> str: return (urlparse(base_url or "").hostname or "").lower()


def _split_prefix_model(model: str = "") -> tuple[str, str]:
    "Split `vendor/model` style names into `(vendor, model)`."
    raw = (model or "").strip()
    if "/" not in raw: return "", raw
    pref, rest = raw.split("/", 1)
    return pref.strip().lower(), rest.strip()


def _model_name(model: str = "") -> str:
    pref, rest = _split_prefix_model(model)
    if pref in _PROVIDER_PREFIXES and rest: return rest.lower()
    return (model or "").strip().lower()


def _env_token(s: str = "") -> str:
    "Convert arbitrary provider labels into ENV_VAR tokens."
    return re.sub(r"[^A-Za-z0-9]+", "_", (s or "").strip().upper()).strip("_")


def _first_env(*names: str) -> str:
    "Return first non-empty environment variable value."
    for nm in names:
        if not nm: continue
        v = os.getenv(nm)
        if v: return v
    return ""


def _openai_compat_vendor(model: str, *, base_url: str = "") -> str:
    "Infer OpenAI-compatible vendor from host, `vendor/model`, then model prefix hints."
    host = _host(base_url)
    if host:
        for vendor, meta in _OPENAI_COMPAT_PROVIDERS.items():
            if any(tok in host for tok in meta.get("host_tokens", ())): return vendor

    pref, _ = _split_prefix_model(model)
    if pref and pref not in _PROVIDER_PREFIXES: return pref

    m = _model_name(model)
    for hint, vendor in _MODEL_VENDOR_HINTS.items():
        if m.startswith(hint): return vendor
    return ""


def _normalize_model(model: str, *, family: str, vendor: str = "") -> str:
    "Normalize model names by stripping recognized `provider/model` prefixes."
    pref, rest = _split_prefix_model(model)
    if pref in _PROVIDER_PREFIXES and rest: return rest
    if family == "openai_compat" and vendor and pref == vendor and rest: return rest
    return (model or "").strip()


def _openai_compat_base_url(model: str, *, base_url: str = "") -> str:
    "Resolve default base URL for OpenAI-compatible providers."
    if base_url: return base_url
    vendor = _openai_compat_vendor(model, base_url=base_url)
    tok = _env_token(vendor)
    if tok:
        env_url = _first_env(
            f"{tok}_BASE_URL",
            f"{tok}_API_BASE",
            f"FASTLLM_{tok}_BASE_URL",
            f"FASTLLM_{tok}_API_BASE",
        )
        if env_url: return env_url
    meta = _OPENAI_COMPAT_PROVIDERS.get(vendor, {})
    if isinstance(meta.get("base_url"), str) and meta["base_url"]: return meta["base_url"]
    return _first_env("OPENAI_COMPAT_BASE_URL", "FASTLLM_OPENAI_COMPAT_BASE_URL")


def infer_provider(model: str, *, provider: str = "", base_url: str = "") -> str:
    "Infer provider family from base URL, then model prefix."
    _ = provider  # Backward-compatible arg; routing no longer depends on explicit provider.

    host = _host(base_url)
    if "anthropic.com" in host: return "anthropic"
    if "generativelanguage.googleapis.com" in host: return "gemini"
    if host == "api.openai.com": return "openai"

    m = _model_name(model)
    if m.startswith("claude") or "claude-" in m: return "anthropic"
    if m.startswith("gemini") or m.startswith("models/gemini") or "gemini-" in m: return "gemini"
    if m.startswith("gpt"): return "openai"
    return "openai_compat"


def _resolve_api_key(family: str, *, model: str = "", api_key: str = "", base_url: str = "") -> str:
    if api_key: return api_key
    if family == "anthropic": return os.getenv("ANTHROPIC_API_KEY") or ""
    if family == "gemini": return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    if family == "openai_compat":
        vendor = _openai_compat_vendor(model, base_url=base_url)
        tok = _env_token(vendor)
        names = []
        meta = _OPENAI_COMPAT_PROVIDERS.get(vendor, {})
        names.extend(meta.get("api_key_envs", ()))
        if tok:
            names.extend((f"{tok}_API_KEY", f"FASTLLM_{tok}_API_KEY"))
        names.extend(("OPENAI_COMPAT_API_KEY", "FASTLLM_OPENAI_COMPAT_API_KEY", "OPENAI_API_KEY"))
        return _first_env(*names)
    return os.getenv("OPENAI_API_KEY") or ""


def mk_auto_client(model: str, *, api_key: str = "", base_url: str = "", provider: str = "", timeout: float = 60.0):
    "Create a provider client from model/base_url inference."
    _ = provider  # Backward-compatible arg; provider hooks removed.
    raw_model = model
    family = infer_provider(raw_model, base_url=base_url)
    if family == "openai_compat": base_url = _openai_compat_base_url(raw_model, base_url=base_url)
    vendor = _openai_compat_vendor(raw_model, base_url=base_url) if family == "openai_compat" else ""
    key = _resolve_api_key(family, model=raw_model, api_key=api_key, base_url=base_url)
    resolved_provider = "openai" if family == "openai" else ("openai_compat" if family == "openai_compat" else family)
    if family == "anthropic":
        return AnthropicClient(api_key=key, base_url=base_url, provider=resolved_provider, timeout=timeout)
    if family == "gemini":
        return GeminiClient(api_key=key, base_url=base_url, provider=resolved_provider, timeout=timeout)
    return OpenAIClient(api_key=key, base_url=base_url, provider=resolved_provider, timeout=timeout)


def _coerce_part(p: Any) -> Part:
    if isinstance(p, Part): return p
    if p is None: return Part(type="text", text="")
    if isinstance(p, str): return Part(type="text", text=p)
    if isinstance(p, dict):
        d = dict(p)
        typ = str(d.pop("type", "text"))
        txt = d.pop("text", None)
        if txt is None and typ == "text" and isinstance(d.get("content"), str): txt = d.pop("content")
        return Part(type=typ, text=txt, data=(d or None))
    return Part(type="text", text=str(p))


def _tool_call_dict(tc: Any) -> dict[str, Any]:
    "Normalize tool-call-like objects to a canonical `{id,name,arguments}` dict."
    if isinstance(tc, ToolCall):
        return {"id": str(tc.id or ""), "name": str(tc.name or ""), "arguments": dict(tc.arguments or {})}
    if isinstance(tc, dict):
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        args = tc.get("arguments", fn.get("arguments", {}))
        if not isinstance(args, dict):
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                    args = parsed if isinstance(parsed, dict) else {"_value": parsed}
                except Exception:
                    args = {"_raw": args}
            else:
                args = {}
        return {
            "id": str(tc.get("id", tc.get("call_id", tc.get("tool_call_id", ""))) or ""),
            "name": str(tc.get("name", fn.get("name", "")) or ""),
            "arguments": args,
        }
    return {
        "id": str(getattr(tc, "id", "") or ""),
        "name": str(getattr(tc, "name", "") or ""),
        "arguments": dict(getattr(tc, "arguments", {}) or {}),
    }


def _openai_chat_meta_from_raw_events(raw_events: list[dict[str, Any]]) -> dict[str, Any]:
    "Extract OpenAI-compatible assistant metadata (e.g. reasoning_content) from raw stream events."
    chunks = []
    for ev in raw_events or []:
        if not isinstance(ev, dict): continue
        choices = ev.get("choices")
        if not (isinstance(choices, list) and choices and isinstance(choices[0], dict)): continue
        c0 = choices[0]
        d = c0.get("delta") if isinstance(c0.get("delta"), dict) else {}
        rc = d.get("reasoning_content")
        if isinstance(rc, str): chunks.append(rc)
    txt = "".join(chunks)
    return {"reasoning_content": txt} if txt else {}


def _openai_chat_meta_from_completion_raw(raw: Any) -> dict[str, Any]:
    "Extract OpenAI-compatible assistant metadata from completion raw payloads."
    if not isinstance(raw, dict): return {}
    choices = raw.get("choices")
    if not (isinstance(choices, list) and choices and isinstance(choices[0], dict)): return {}
    msg = choices[0].get("message") if isinstance(choices[0].get("message"), dict) else {}
    rc = msg.get("reasoning_content")
    if isinstance(rc, str) and rc: return {"reasoning_content": rc}
    return {}


def _completion_to_msg(c: Completion) -> Msg:
    "Convert a Completion to canonical assistant Msg, preserving tool-calls and provider metadata."
    data = dict(c.message.data or {})
    if c.tool_calls: data["tool_calls"] = [_tool_call_dict(tc) for tc in c.tool_calls]
    if meta := _openai_chat_meta_from_completion_raw(c.raw):
        oac = dict(data.get("openai_chat") or {})
        oac.update(meta)
        data["openai_chat"] = oac
    return Msg(role=c.message.role or "assistant", content=list(c.message.content), data=(data or None))


def _stream_summary_to_msg(s: StreamSummary) -> Msg:
    "Convert StreamSummary to canonical assistant Msg for toolloop replay."
    parts = [Part(type="text", text=s.text)] if s.text else []
    data = {}
    if s.tool_calls: data["tool_calls"] = [_tool_call_dict(tc) for tc in s.tool_calls]
    if meta := _openai_chat_meta_from_raw_events(s.raw_events):
        data["openai_chat"] = meta
    return Msg(role="assistant", content=parts, data=(data or None))


def _coerce_msg(m: Any) -> Msg:
    if isinstance(m, Msg): return m
    if isinstance(m, Completion): return _completion_to_msg(m)
    if isinstance(m, StreamSummary): return _stream_summary_to_msg(m)
    if isinstance(m, str): return Msg(role="user", content=[Part(type="text", text=m)])
    if isinstance(m, dict):
        d = dict(m)
        role = str(d.pop("role", "user"))
        content = d.pop("content", "")
        if isinstance(content, list): parts = [_coerce_part(o) for o in content]
        elif content in (None, ""): parts = []
        else: parts = [_coerce_part(content)]
        if not parts and role != "assistant": parts = [Part(type="text", text="")]
        return Msg(role=role, content=parts, data=(d or None))
    raise TypeError(f"Unsupported message type: {type(m).__name__}")


def _coerce_messages(messages: Any) -> list[Msg]:
    if isinstance(messages, (str, Msg, dict)): return [_coerce_msg(messages)]
    if isinstance(messages, list) or isinstance(messages, tuple): return [_coerce_msg(m) for m in messages]
    return [_coerce_msg(messages)]


def _missing_responses_endpoint(e: Exception) -> bool:
    if isinstance(e, AttributeError): return True
    if isinstance(e, APIError):
        code = e.status_code
        txt = (e.message or "").lower()
        if code in (404, 405, 410, 501): return True
        if code not in (400, 422): return False
        if "responses" in txt and any(k in txt for k in ("not found", "unknown", "unsupported", "invalid")):
            return True
        return False
    if not isinstance(e, httpx.HTTPStatusError): return False
    code = e.response.status_code
    if code in (404, 405, 410, 501): return True
    if code not in (400, 422): return False
    try: txt = (e.response.text or "").lower()
    except Exception: txt = ""
    if code in (400, 422) and "responses" in txt and any(k in txt for k in ("not found", "unknown", "unsupported", "invalid")):
        return True
    return False


def _resolve_endpoint(family: str, endpoint: str) -> str:
    if endpoint not in _ENDPOINTS: raise ValueError(f"Invalid endpoint='{endpoint}', expected one of {_ENDPOINTS}")
    if family in ("anthropic", "gemini"): return endpoint
    if endpoint == "auto": return "responses" if family == "openai" else "chat"
    return endpoint


def _merge_aliases(kwargs: dict) -> dict:
    kw = dict(kwargs)
    if "caching" in kw and "cache" not in kw: kw["cache"] = kw.pop("caching")
    if "api_base" in kw and "base_url" not in kw: kw["base_url"] = kw.pop("api_base")
    kw.pop("provider", None)
    kw.pop("custom_llm_provider", None)
    return kw


async def _openai_complete(c: OpenAIClient, msgs: list[Msg], model: str, opts: Optional[RequestOptions], ep: str,
    fallback: bool, kw: dict):
    if ep == "chat": return await c.achat_complete(msgs, model=model, options=opts, **kw)
    try: return await c.acomplete(msgs, model=model, options=opts, **kw)
    except Exception as e:
        if not fallback or not _missing_responses_endpoint(e): raise
        return await c.achat_complete(msgs, model=model, options=opts, **kw)


async def _openai_stream(c: OpenAIClient, msgs: list[Msg], model: str, opts: Optional[RequestOptions], ep: str,
    fallback: bool, kw: dict):
    if ep == "chat":
        async for d in c.achat_stream(msgs, model=model, options=opts, **kw): yield d
        return
    try:
        async for d in c.astream(msgs, model=model, options=opts, **kw): yield d
        return
    except Exception as e:
        if not fallback or not _missing_responses_endpoint(e): raise
    async for d in c.achat_stream(msgs, model=model, options=opts, **kw): yield d


@delegates(RequestOptions, keep=True)
async def acompletion(model: str, messages: Any, *, stream: bool = False, api_key: str = "", base_url: str = "",
    api_base: str = "", provider: str = "", custom_llm_provider: str = "", endpoint: str = "auto",
    timeout: float = 60.0, options: Optional[RequestOptions] = None, **kwargs):
    "LiteLLM-style async completion wrapper; returns an async iterator when `stream=True`."
    if api_base and not base_url: base_url = api_base
    _ = provider, custom_llm_provider  # Backward-compatible args; routing is inferred from model/base_url.
    kw = _merge_aliases(kwargs)
    msgs = _coerce_messages(messages)
    family = infer_provider(model, base_url=base_url)
    vendor = _openai_compat_vendor(model, base_url=base_url) if family == "openai_compat" else ""
    req_model = _normalize_model(model, family=family, vendor=vendor)
    ep = _resolve_endpoint(family, endpoint)
    c = mk_auto_client(model, api_key=api_key, base_url=base_url, timeout=timeout)
    fallback = family == "openai" and endpoint == "auto"

    if not stream:
        try:
            if family == "anthropic": return await c.acomplete(msgs, model=req_model, options=options, **kw)
            if family == "gemini": return await c.acomplete(msgs, model=req_model, options=options, **kw)
            return await _openai_complete(c, msgs, req_model, options, ep, fallback, kw)
        finally: await c.aclose()

    async def _gen():
        try:
            if family == "anthropic":
                async for d in c.astream(msgs, model=req_model, options=options, **kw): yield d
                return
            if family == "gemini":
                async for d in c.astream(msgs, model=req_model, options=options, **kw): yield d
                return
            async for d in _openai_stream(c, msgs, req_model, options, ep, fallback, kw): yield d
        finally: await c.aclose()

    return _gen()


__all__ = "infer_provider mk_auto_client acompletion".split()
