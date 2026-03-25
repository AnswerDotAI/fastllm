"Provider overlays for OpenAI-compatible endpoints."

from __future__ import annotations

from fastcore.basics import patch

from ..builtin_specs import openai_ops
from ..oapi import OpenAPIClient
from ..types import Caps
from ..clients import OpenAIClient


def _rebuild(self: OpenAIClient, base_url: str):
    "Rebuild underlying OpenAPI client with a new base URL."
    hdrs = {"Authorization": f"Bearer {self.config.api_key or ''}", "content-type": "application/json", **self.config.default_headers}
    self.api = OpenAPIClient(base_url=base_url, headers=hdrs, timeout=self.config.timeout, ops=openai_ops())


@patch
def use_openrouter(self: OpenAIClient):
    "Apply OpenRouter defaults."
    self.config.base_url = self.config.base_url or "https://openrouter.ai/api/v1"
    self.config.provider = "openrouter"
    _rebuild(self, self.config.base_url)
    self._caps = Caps(**{**self._caps.__dict__, "search": True, "reasoning": True})
    return self


@patch
def use_moonshot(self: OpenAIClient):
    "Apply Moonshot/Kimi defaults."
    self.config.base_url = self.config.base_url or "https://api.moonshot.cn/v1"
    self.config.provider = "moonshot"
    _rebuild(self, self.config.base_url)
    self._caps = Caps(**{**self._caps.__dict__, "search": False, "reasoning": True})
    return self


@patch
def use_vllm(self: OpenAIClient):
    "Apply local vLLM defaults."
    self.config.base_url = self.config.base_url or "http://localhost:8000/v1"
    self.config.provider = "vllm"
    _rebuild(self, self.config.base_url)
    self._caps = Caps(**{**self._caps.__dict__, "search": False, "reasoning": False})
    return self
