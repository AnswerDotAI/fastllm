"Client factory helpers."

from __future__ import annotations

from typing import Optional

from .clients import AnthropicClient, GeminiClient, OpenAIClient
from .config import ClientConfig


def mk_client(family: str, *, model: str, api_key: Optional[str] = None, base_url: str = "", provider: str = ""):
    "Create a fastllm_v2 provider client by family."
    cfg = ClientConfig(model=model, api_key=api_key, base_url=base_url, provider=provider)
    if family in ("openai", "openai_responses", "openai_chat", "openai_compat"):
        from . import providers as _providers
        _ = _providers
        c = OpenAIClient(cfg)
        if provider:
            m = f"use_{provider}"
            if hasattr(c, m): getattr(c, m)()
        return c
    if family == "anthropic": return AnthropicClient(cfg)
    if family == "gemini": return GeminiClient(cfg)
    raise ValueError(f"Unknown family: {family}")
