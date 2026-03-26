"High-level completion wrappers with automatic client routing."

from __future__ import annotations

import os
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from fastcore.meta import delegates

from .clients import AnthropicClient, GeminiClient, OpenAIClient
from .config import ClientConfig
from .types import Msg, Part, RequestOptions


_ENDPOINTS = ("auto", "responses", "chat")


def _norm_provider(provider: str = "") -> str:
    p = (provider or "").strip().lower().replace("-", "_")
    aliases = dict(claude="anthropic", google="gemini", google_ai="gemini", openai_chat="openai")
    return aliases.get(p, p)


def _host(base_url: str = "") -> str: return (urlparse(base_url or "").hostname or "").lower()


def _model_name(model: str = "") -> str:
    m = (model or "").strip().lower()
    for pref in ("anthropic/", "google/", "openai/"):
        if m.startswith(pref): return m.split("/", 1)[1]
    return m


def infer_provider(model: str, *, provider: str = "", base_url: str = "") -> str:
    "Infer provider family from explicit provider, base URL, then model prefix."
    p = _norm_provider(provider)
    if p in ("anthropic", "gemini", "openai"): return p
    if p in ("chatgpt", "azure"): return "openai"
    if p: return "openai_compat"

    host = _host(base_url)
    if "anthropic.com" in host: return "anthropic"
    if "generativelanguage.googleapis.com" in host: return "gemini"
    if host == "api.openai.com": return "openai"

    m = _model_name(model)
    if m.startswith("claude") or "claude-" in m: return "anthropic"
    if m.startswith("gemini") or m.startswith("models/gemini") or "gemini-" in m: return "gemini"
    if m.startswith("gpt"): return "openai"
    return "openai_compat"


def _resolve_api_key(family: str, *, provider: str = "", api_key: str = "") -> str:
    if api_key: return api_key
    p = _norm_provider(provider)
    if p:
        envk = f"{p.upper()}_API_KEY"
        if os.getenv(envk): return os.getenv(envk) or ""
    if family == "anthropic": return os.getenv("ANTHROPIC_API_KEY") or ""
    if family == "gemini": return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    return os.getenv("OPENAI_API_KEY") or ""


def mk_auto_client(model: str, *, api_key: str = "", base_url: str = "", provider: str = "", timeout: float = 60.0):
    "Create a provider client from model/provider/base_url inference."
    p = _norm_provider(provider)
    family = infer_provider(model, provider=p, base_url=base_url)
    key = _resolve_api_key(family, provider=p, api_key=api_key)
    cfg = ClientConfig(model=model, api_key=key, base_url=base_url, provider=p, timeout=timeout)
    if family == "anthropic": return AnthropicClient(cfg)
    if family == "gemini": return GeminiClient(cfg)
    from . import providers as _providers
    _ = _providers
    c = OpenAIClient(cfg)
    if p and hasattr(c, f"use_{p}"): getattr(c, f"use_{p}")()
    if not c.config.provider: c.config.provider = "openai" if family == "openai" else (p or "openai_compat")
    return c


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


def _coerce_msg(m: Any) -> Msg:
    if isinstance(m, Msg): return m
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
    if not isinstance(e, httpx.HTTPStatusError): return False
    code, txt = e.response.status_code, (e.response.text or "").lower()
    if code in (404, 405, 410, 501): return True
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
    if "custom_llm_provider" in kw and "provider" not in kw: kw["provider"] = kw.pop("custom_llm_provider")
    return kw


async def _openai_complete(c: OpenAIClient, msgs: list[Msg], opts: Optional[RequestOptions], ep: str, fallback: bool, kw: dict):
    if ep == "chat": return await c.achat_complete(msgs, options=opts, **kw)
    try: return await c.acomplete(msgs, options=opts, **kw)
    except Exception as e:
        if not fallback or not _missing_responses_endpoint(e): raise
        return await c.achat_complete(msgs, options=opts, **kw)


async def _openai_stream(c: OpenAIClient, msgs: list[Msg], opts: Optional[RequestOptions], ep: str, fallback: bool, kw: dict):
    if ep == "chat":
        async for d in c.achat_stream(msgs, options=opts, **kw): yield d
        return
    try:
        async for d in c.astream(msgs, options=opts, **kw): yield d
        return
    except Exception as e:
        if not fallback or not _missing_responses_endpoint(e): raise
    async for d in c.achat_stream(msgs, options=opts, **kw): yield d


@delegates(RequestOptions, keep=True)
async def acompletion(model: str, messages: Any, *, stream: bool = False, api_key: str = "", base_url: str = "",
    api_base: str = "", provider: str = "", custom_llm_provider: str = "", endpoint: str = "auto",
    timeout: float = 60.0, options: Optional[RequestOptions] = None, **kwargs):
    "LiteLLM-style async completion wrapper; returns an async iterator when `stream=True`."
    if api_base and not base_url: base_url = api_base
    if custom_llm_provider and not provider: provider = custom_llm_provider
    kw = _merge_aliases(kwargs)
    msgs = _coerce_messages(messages)
    family = infer_provider(model, provider=provider, base_url=base_url)
    ep = _resolve_endpoint(family, endpoint)
    c = mk_auto_client(model, api_key=api_key, base_url=base_url, provider=provider, timeout=timeout)
    fallback = family == "openai" and endpoint == "auto"

    if not stream:
        try:
            if family == "anthropic": return await c.acomplete(msgs, options=options, **kw)
            if family == "gemini": return await c.acomplete(msgs, options=options, **kw)
            return await _openai_complete(c, msgs, options, ep, fallback, kw)
        finally: await c.aclose()

    async def _gen():
        try:
            if family == "anthropic":
                async for d in c.astream(msgs, options=options, **kw): yield d
                return
            if family == "gemini":
                async for d in c.astream(msgs, options=options, **kw): yield d
                return
            async for d in _openai_stream(c, msgs, options, ep, fallback, kw): yield d
        finally: await c.aclose()

    return _gen()


@delegates(acompletion, keep=True, but=["stream"])
async def astream(model: str, messages: Any, **kwargs):
    "Convenience async stream wrapper."
    it = await acompletion(model, messages, stream=True, **kwargs)
    async for d in it: yield d


__all__ = "infer_provider mk_auto_client acompletion astream".split()
