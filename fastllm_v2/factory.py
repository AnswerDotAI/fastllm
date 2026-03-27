"Client factory helpers."

from __future__ import annotations

from typing import Optional

from .clients import AnthropicClient, GeminiClient, OpenAIClient
from .config import ClientConfig


def mk_client(family: str, *, model: str, api_key: Optional[str] = None, base_url: str = "", provider: str = ""):
    "Create a fastllm_v2 provider client by family."
    _ = provider  # Backward-compatible arg; provider hooks removed.
    if family in ("openai", "openai_responses", "openai_chat", "openai_compat"):
        prov = "openai_chat" if family == "openai_chat" else ("openai_compat" if family == "openai_compat" else "openai")
        return OpenAIClient(ClientConfig(model=model, api_key=api_key, base_url=base_url, provider=prov))
    if family == "anthropic": return AnthropicClient(ClientConfig(model=model, api_key=api_key, base_url=base_url, provider="anthropic"))
    if family == "gemini": return GeminiClient(ClientConfig(model=model, api_key=api_key, base_url=base_url, provider="gemini"))
    raise ValueError(f"Unknown family: {family}")
