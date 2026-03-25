"Provider clients built on the OpenAPI operation layer."

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any, AsyncIterator, Optional

from .builtin_specs import anthropic_ops, gemini_ops, openai_ops
from .config import ClientConfig
from .errors import UnsupportedCapabilityError
from .normalize import normalize_anthropic_event, normalize_anthropic_message, normalize_gemini_event
from .normalize import normalize_gemini_generate, normalize_openai_chat_completion, normalize_openai_chat_delta
from .normalize import normalize_openai_response, normalize_openai_response_event
from .oapi import OpenAPIClient
from .types import Caps, Completion, Delta, Msg, RequestOptions, ToolSpec


_REQ_OPT_FIELDS = {f.name for f in fields(RequestOptions)}
_REQ_OPT_ALIASES = {"headers": "extra_headers", "query": "extra_query", "body": "extra_body"}


def _tool_obj(t: ToolSpec) -> dict:
    "Build provider-agnostic function tool schema."
    return {"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters}


def _provider_part(p, nm: str) -> Optional[dict]:
    "Return provider-native part payload when available."
    if not p.data: return None
    if nm in p.data and isinstance(p.data[nm], dict): return dict(p.data[nm])
    key = f"_{nm}"
    if key in p.data and isinstance(p.data[key], dict): return dict(p.data[key])
    return None


def _merge_opts(options: Optional[RequestOptions], kw: dict[str, Any]) -> RequestOptions:
    "Merge keyword overrides into RequestOptions; unknown kwargs go to native body."
    opts = options or RequestOptions()
    if not kw: return opts

    updates = {}
    native = dict(opts.native or {})
    extra_body = dict(opts.extra_body or {})
    extra_query = dict(opts.extra_query or {})
    extra_headers = dict(opts.extra_headers or {})

    for k,v in kw.items():
        key = _REQ_OPT_ALIASES.get(k, k)
        if key == "native":
            if isinstance(v, dict): native.update(v)
            else: native[key] = v
            continue
        if key == "extra_body":
            if isinstance(v, dict): extra_body.update(v)
            else: extra_body[key] = v
            continue
        if key == "extra_query":
            if isinstance(v, dict): extra_query.update(v)
            else: extra_query[key] = v
            continue
        if key == "extra_headers":
            if isinstance(v, dict): extra_headers.update(v)
            else: extra_headers[key] = str(v)
            continue
        if key in _REQ_OPT_FIELDS: updates[key] = v
        else:
            native[key] = v

    if native: updates["native"] = native
    if extra_body: updates["extra_body"] = extra_body
    if extra_query: updates["extra_query"] = extra_query
    if extra_headers: updates["extra_headers"] = extra_headers
    return replace(opts, **updates) if updates else opts


def _openai_cache(payload: dict, cache: Any):
    "Apply generic cache option to OpenAI-compatible payloads."
    if cache is None: return
    if isinstance(cache, bool):
        payload["store"] = cache
        return
    if isinstance(cache, dict):
        if any(k in cache for k in ("store", "cache", "prompt_cache_key", "service_tier")): payload.update(cache)
        else: payload["cache"] = cache
        return
    payload["store"] = bool(cache)


def _openai_text_format(v: Any) -> Any:
    "Map generic response_format payloads to Responses API text.format."
    if not isinstance(v, dict): return v
    if v.get("type") == "json_schema" and isinstance(v.get("json_schema"), dict):
        js = dict(v["json_schema"])
        js.setdefault("type", "json_schema")
        return js
    return v


def _anthropic_tool_choice(v: Any) -> Optional[dict]:
    "Map generic tool_choice values to Anthropic tool_choice shape."
    if v is None: return None
    if isinstance(v, str):
        mode = v.strip().lower()
        if mode in ("auto",): return {"type": "auto"}
        if mode in ("required", "any", "force"): return {"type": "any"}
        if mode in ("none", "off", "disabled"): return None
        return {"type": "auto"}
    if isinstance(v, dict):
        if "type" in v: return dict(v)
        if "name" in v: return {"type": "tool", **v}
        return dict(v)
    return {"type": "auto"}


def _anthropic_thinking(v: Any) -> Optional[dict]:
    "Map generic reasoning effort to Anthropic thinking config."
    if v is None: return None
    if isinstance(v, dict): return dict(v)
    if not isinstance(v, str): return {"type": "enabled", "budget_tokens": 2048}
    b = dict(minimal=1024, low=2048, medium=4096, high=8192, max=16384, very_high=16384).get(v.strip().lower())
    return {"type": "enabled", "budget_tokens": b or 4096}


def _anthropic_cache_control(cache: Any) -> Optional[dict]:
    "Map generic cache option to Anthropic content-block cache_control."
    if cache is None or cache is False: return None
    if cache is True: return {"type": "ephemeral"}
    if isinstance(cache, str): return {"type": cache}
    if isinstance(cache, dict): return dict(cache)
    return {"type": "ephemeral"}


def _gemini_tool_obj(t: ToolSpec) -> dict:
    "Map ToolSpec to Gemini function declaration."
    return {"name": t.name, "description": t.description, "parameters": t.parameters}


def _gemini_tool_choice(v: Any) -> Optional[dict]:
    "Map generic tool_choice values to Gemini toolConfig."
    if v is None: return None
    if isinstance(v, dict):
        if "functionCallingConfig" in v: return v
        if "mode" in v: return {"functionCallingConfig": v}
        return {"functionCallingConfig": v}
    if not isinstance(v, str): return {"functionCallingConfig": {"mode": "AUTO"}}
    mode = v.strip().lower()
    if mode in ("auto",): return {"functionCallingConfig": {"mode": "AUTO"}}
    if mode in ("none", "off", "disabled"): return {"functionCallingConfig": {"mode": "NONE"}}
    if mode in ("required", "any", "force"): return {"functionCallingConfig": {"mode": "ANY"}}
    return {"functionCallingConfig": {"mode": "AUTO"}}


def _gemini_thinking_config(effort: Optional[str]) -> Optional[dict]:
    "Map generic reasoning effort to Gemini thinking budget."
    if effort is None: return None
    if isinstance(effort, str):
        m = {
            "minimal": 256,
            "low": 512,
            "medium": 1024,
            "high": 2048,
            "max": 4096,
            "very_high": 4096,
        }
        b = m.get(effort.strip().lower())
        if b is not None: return {"thinkingBudget": b}
    if isinstance(effort, dict): return effort
    return {"thinkingBudget": 1024}


def _gemini_cache(body: dict, cache: Any):
    "Apply generic cache option to Gemini payloads."
    if cache is None: return
    if isinstance(cache, str):
        body["cachedContent"] = cache
        return
    if isinstance(cache, dict):
        if "cachedContent" in cache:
            body["cachedContent"] = cache["cachedContent"]
            rest = {k:v for k,v in cache.items() if k != "cachedContent"}
            body.update(rest)
        else:
            body.update(cache)


def _gemini_model_ref(model: str) -> str:
    "Normalize Gemini model route values to `models/...` when needed."
    if model.startswith(("models/", "tunedModels/")): return model
    return f"models/{model}"


class BaseLLMClient:
    "Shared provider-client behavior."
    def __init__(self, config: ClientConfig, *, caps: Caps, api: OpenAPIClient):
        self.config,self._caps,self.api = config,caps,api

    @property
    def caps(self) -> Caps: return self._caps

    def _require_caps(self, names):
        missing = [nm for nm in names if not getattr(self._caps, nm)]
        if missing: raise UnsupportedCapabilityError(f"Unsupported capabilities requested: {', '.join(missing)}")

    async def aclose(self): await self.api.aclose()

    async def acomplete(self, messages: list[Msg], *, options: Optional[RequestOptions] = None, **kwargs) -> Completion:
        raise NotImplementedError

    async def astream(self, messages: list[Msg], *, options: Optional[RequestOptions] = None, **kwargs) -> AsyncIterator[Delta]:
        raise NotImplementedError


class OpenAIClient(BaseLLMClient):
    "OpenAI client with Responses API default and Chat Completions compatibility."
    def __init__(self, config: ClientConfig, *, caps: Optional[Caps] = None, api: Optional[OpenAPIClient] = None):
        hdrs = {"Authorization": f"Bearer {config.api_key or ''}", "content-type": "application/json", **config.default_headers}
        base_url = config.base_url or "https://api.openai.com/v1"
        api = api or OpenAPIClient(base_url=base_url, headers=hdrs, timeout=config.timeout, ops=openai_ops())
        caps = caps or Caps(tools=True, tool_choice=True, streaming=True, search=True, reasoning=True,
            prompt_caching=True, images=True)
        super().__init__(config, caps=caps, api=api)

    def _responses_part(self, role: str, p):
        "Serialize a normalized part for OpenAI Responses input."
        raw = _provider_part(p, "openai")
        if raw is not None: return raw

        if p.type == "text":
            typ = "output_text" if role == "assistant" else "input_text"
            return {"type": typ, "text": p.text or ""}

        if p.type in ("image_url", "input_image", "image"):
            obj = {"type": "input_image", **(p.data or {})}
            if p.text and "image_url" not in obj and "url" not in obj: obj["image_url"] = p.text
            if "url" in obj and "image_url" not in obj: obj["image_url"] = obj.pop("url")
            return obj

        if p.type in ("input_audio", "audio"): return {"type": "input_audio", **(p.data or {})}

        if p.type in ("input_file", "file", "pdf", "document"):
            obj = {"type": "input_file", **(p.data or {})}
            if p.text and all(k not in obj for k in ("file_id", "file_data", "file_url")): obj["file_url"] = p.text
            return obj

        obj = {"type": p.type}
        if p.text is not None: obj["text"] = p.text
        if p.data: obj.update(p.data)
        return obj

    def _responses_messages(self, messages: list[Msg]):
        "Serialize normalized messages to Responses API input format."
        out = []
        for m in messages:
            parts = [self._responses_part(m.role, p) for p in m.content]
            out.append({"role": m.role, "content": parts})
        return out

    def _chat_part(self, p):
        "Serialize a normalized part for OpenAI chat.completions content."
        raw = _provider_part(p, "openai_chat")
        if raw is not None: return raw

        if p.type == "text": return {"type": "text", "text": p.text or ""}

        if p.type in ("image_url", "input_image", "image"):
            d = dict(p.data or {})
            img = d.pop("image_url", None)
            if isinstance(img, dict): iurl = dict(img)
            else:
                url = img or d.pop("url", None) or p.text
                iurl = {"url": url} if url else {}
            if "detail" in d and "detail" not in iurl: iurl["detail"] = d.pop("detail")
            return {"type": "image_url", "image_url": iurl, **d}

        if p.type in ("input_audio", "audio"):
            return {"type": "input_audio", **(p.data or {})}

        if p.type in ("input_file", "file", "pdf", "document"):
            return {"type": "file", **(p.data or {})}

        obj = {"type": p.type}
        if p.text is not None: obj["text"] = p.text
        if p.data: obj.update(p.data)
        return obj

    def _chat_messages(self, messages: list[Msg]):
        "Serialize normalized messages to chat.completions messages."
        res = []
        for m in messages:
            if len(m.content) == 1 and m.content[0].type == "text":
                res.append({"role": m.role, "content": m.content[0].text or ""})
                continue
            cts = [self._chat_part(p) for p in m.content]
            res.append({"role": m.role, "content": cts})
        return res

    def _responses_payload(self, messages: list[Msg], opts: RequestOptions, *, stream: bool):
        "Build Responses API request payload."
        if opts.tools: self._require_caps(["tools"])
        if opts.tool_choice is not None: self._require_caps(["tool_choice"])
        if opts.search: self._require_caps(["search"])
        if opts.reasoning_effort is not None: self._require_caps(["reasoning"])

        payload = {"model": self.config.model, "input": self._responses_messages(messages), "stream": stream}
        if opts.max_tokens is not None: payload["max_output_tokens"] = opts.max_tokens
        if opts.temperature is not None: payload["temperature"] = opts.temperature
        if opts.response_format is not None:
            txt = dict(payload.get("text") or {})
            txt["format"] = _openai_text_format(opts.response_format)
            payload["text"] = txt
        if opts.tools: payload["tools"] = [_tool_obj(t) for t in opts.tools]
        if opts.tool_choice is not None: payload["tool_choice"] = opts.tool_choice
        if opts.reasoning_effort is not None: payload["reasoning"] = {"effort": opts.reasoning_effort}
        if opts.search:
            tools = list(payload.get("tools") or [])
            search_obj = opts.search if isinstance(opts.search, dict) else {}
            tools.append({"type": "web_search_preview", **search_obj})
            payload["tools"] = tools
        _openai_cache(payload, opts.cache)
        if opts.native: payload.update(opts.native)
        if opts.extra_body: payload.update(opts.extra_body)
        return payload

    def _op(self, group: str, *names: str):
        "Return first existing operation from a group."
        g = getattr(self.api, group)
        for nm in names:
            if hasattr(g, nm): return getattr(g, nm)
        raise AttributeError(f"No operation found in group '{group}' for any of: {', '.join(names)}")

    def _chat_payload(self, messages: list[Msg], opts: RequestOptions, *, stream: bool):
        "Build chat.completions payload."
        if opts.tools: self._require_caps(["tools"])
        if opts.tool_choice is not None: self._require_caps(["tool_choice"])
        if opts.search: self._require_caps(["search"])
        if opts.reasoning_effort is not None: self._require_caps(["reasoning"])

        payload = {"model": self.config.model, "messages": self._chat_messages(messages), "stream": stream}
        if opts.max_tokens is not None: payload["max_tokens"] = opts.max_tokens
        if opts.temperature is not None: payload["temperature"] = opts.temperature
        if opts.response_format is not None: payload["response_format"] = opts.response_format
        if opts.tools: payload["tools"] = [{"type": "function", "function": _tool_obj(t) | {"name": t.name}} for t in opts.tools]
        if opts.tool_choice is not None: payload["tool_choice"] = opts.tool_choice
        if opts.reasoning_effort is not None: payload["reasoning_effort"] = opts.reasoning_effort
        if opts.search:
            payload["web_search_options"] = opts.search if isinstance(opts.search, dict) else {}
        _openai_cache(payload, opts.cache)
        if opts.native: payload.update(opts.native)
        if opts.extra_body: payload.update(opts.extra_body)
        return payload

    async def acomplete(self, messages: list[Msg], *, options: Optional[RequestOptions] = None, **kwargs) -> Completion:
        "Responses API non-stream completion."
        opts = _merge_opts(options, kwargs)
        payload = self._responses_payload(messages, opts, stream=False)
        raw = await self._op("responses", "create", "create_response")(_headers=opts.extra_headers,
            _query=opts.extra_query, **payload)
        return normalize_openai_response(raw, model=self.config.model, provider=self.config.provider or "openai")

    async def astream(self, messages: list[Msg], *, options: Optional[RequestOptions] = None, **kwargs) -> AsyncIterator[Delta]:
        "Responses API streaming completion."
        opts = _merge_opts(options, kwargs)
        payload = self._responses_payload(messages, opts, stream=True)
        op = self._op("responses", "create", "create_response")
        async for ev in op(_stream=True, _headers=opts.extra_headers, _query=opts.extra_query, **payload):
            d = normalize_openai_response_event(ev)
            if d is not None: yield d

    async def achat_complete(self, messages: list[Msg], *, options: Optional[RequestOptions] = None, **kwargs) -> Completion:
        "Chat Completions non-stream completion."
        opts = _merge_opts(options, kwargs)
        payload = self._chat_payload(messages, opts, stream=False)
        op = self._op("chat", "create_completions", "create_chat_completion")
        raw = await op(_headers=opts.extra_headers, _query=opts.extra_query, **payload)
        return normalize_openai_chat_completion(raw, model=self.config.model, provider=self.config.provider or "openai_chat")

    async def achat_stream(self, messages: list[Msg], *, options: Optional[RequestOptions] = None, **kwargs) -> AsyncIterator[Delta]:
        "Chat Completions stream."
        opts = _merge_opts(options, kwargs)
        payload = self._chat_payload(messages, opts, stream=True)
        op = self._op("chat", "create_completions", "create_chat_completion")
        async for ev in op(_stream=True, _headers=opts.extra_headers, _query=opts.extra_query, **payload):
            yield normalize_openai_chat_delta(ev)

    async def aresponse_get(self, response_id: str):
        "Get a response object by id."
        return await self._op("responses", "retrieve", "get_response")(response_id=response_id)

    async def aresponse_delete(self, response_id: str):
        "Delete a response object by id."
        return await self._op("responses", "delete", "delete_response")(response_id=response_id)

    async def aresponse_cancel(self, response_id: str):
        "Cancel an in-progress response by id."
        op = getattr(self.api.responses, "cancel", None)
        if op is not None: return await op(response_id=response_id)
        return await self.api.call("/responses/{response_id}/cancel", "POST", route={"response_id": response_id})

    async def aresponse_input_items(self, response_id: str, *, after: Optional[str] = None, limit: Optional[int] = None):
        "List response input items."
        return await self._op("responses", "list_input_items")(response_id=response_id, after=after, limit=limit)

    async def acompact(self, response_ids: list[str], *, model: Optional[str] = None):
        "Compact response history via /responses/compact."
        op = getattr(self.api.responses, "compact", None)
        if op is not None: return await op(response_ids=response_ids, model=model or self.config.model)
        return await self.api.call("/responses/compact", "POST", body={"response_ids": response_ids, "model": model or self.config.model})

    async def ainput_tokens(self, inp: Any, *, model: Optional[str] = None):
        "Estimate input token count via /responses/input_tokens."
        op = getattr(self.api.responses, "input_tokens", None)
        if op is not None: return await op(model=model or self.config.model, input=inp)
        return await self.api.call("/responses/input_tokens", "POST", body={"model": model or self.config.model, "input": inp})


class AnthropicClient(BaseLLMClient):
    "Anthropic native Messages client."
    def __init__(self, config: ClientConfig, *, caps: Optional[Caps] = None, api: Optional[OpenAPIClient] = None):
        hdrs = {
            "x-api-key": config.api_key or "",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            **config.default_headers}
        base_url = config.base_url or "https://api.anthropic.com"
        api = api or OpenAPIClient(base_url=base_url, headers=hdrs, timeout=config.timeout, ops=anthropic_ops())
        caps = caps or Caps(tools=True, tool_choice=True, streaming=True, search=True, reasoning=True,
            prefill=True, citations=True, prompt_caching=True, images=True, pdfs=True)
        super().__init__(config, caps=caps, api=api)

    def _anthropic_part(self, p):
        "Serialize a normalized part for Anthropic content blocks."
        raw = _provider_part(p, "anthropic")
        if raw is not None: return raw

        if p.type == "text": return {"type": "text", "text": p.text or ""}

        if p.type in ("image", "input_image", "image_url"):
            d = dict(p.data or {})
            if "source" in d and isinstance(d["source"], dict): return {"type": "image", **d}
            url = d.pop("url", None) or p.text
            if url: return {"type": "image", "source": {"type": "url", "url": url}, **d}
            return {"type": "image", **d}

        if p.type in ("pdf", "document", "input_file", "file"):
            d = dict(p.data or {})
            if "source" in d and isinstance(d["source"], dict): return {"type": "document", **d}
            url = d.pop("url", None) or p.text
            if url: return {"type": "document", "source": {"type": "url", "url": url}, **d}
            return {"type": "document", **d}

        obj = {"type": p.type}
        if p.text is not None: obj["text"] = p.text
        if p.data: obj.update(p.data)
        return obj

    def _serialize_messages(self, messages: list[Msg], *, cache: Any = None):
        "Serialize normalized messages to Anthropic content blocks."
        res = []
        cache_ctl = _anthropic_cache_control(cache)
        for m in messages:
            blocks = []
            for p in m.content:
                b = self._anthropic_part(p)
                if cache_ctl and m.role in ("user", "system") and "cache_control" not in b:
                    if b.get("type") in ("text", "image", "document"): b["cache_control"] = cache_ctl
                blocks.append(b)
            res.append({"role": m.role, "content": blocks})
        return res

    def _payload(self, messages: list[Msg], opts: RequestOptions, *, stream: bool):
        "Build Anthropic messages payload."
        if opts.tools: self._require_caps(["tools"])
        if opts.tool_choice is not None: self._require_caps(["tool_choice"])
        if opts.search: self._require_caps(["search"])
        if opts.reasoning_effort is not None: self._require_caps(["reasoning"])

        payload = {
            "model": self.config.model,
            "messages": self._serialize_messages(messages, cache=opts.cache),
            "max_tokens": opts.max_tokens or 1024,
            "stream": stream}
        if opts.temperature is not None: payload["temperature"] = opts.temperature
        if opts.tools: payload["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in opts.tools]
        tc = _anthropic_tool_choice(opts.tool_choice)
        if tc is not None: payload["tool_choice"] = tc
        if opts.search: payload["web_search"] = opts.search if isinstance(opts.search, dict) else {}
        think = _anthropic_thinking(opts.reasoning_effort)
        if think is not None:
            payload["thinking"] = think
            bt = think.get("budget_tokens")
            if isinstance(bt, int) and payload["max_tokens"] <= bt:
                payload["max_tokens"] = bt + 256
        if isinstance(opts.cache, dict) and "context_management" in opts.cache:
            payload["context_management"] = opts.cache["context_management"]
        if opts.native: payload.update(opts.native)
        if opts.extra_body: payload.update(opts.extra_body)
        return payload

    async def acomplete(self, messages: list[Msg], *, options: Optional[RequestOptions] = None, **kwargs) -> Completion:
        "Non-stream Anthropic completion."
        opts = _merge_opts(options, kwargs)
        raw = await self.api.messages.create(_headers=opts.extra_headers, _query=opts.extra_query,
            **self._payload(messages, opts, stream=False))
        return normalize_anthropic_message(raw, model=self.config.model, provider=self.config.provider or "anthropic")

    async def astream(self, messages: list[Msg], *, options: Optional[RequestOptions] = None, **kwargs) -> AsyncIterator[Delta]:
        "Stream Anthropic completion."
        opts = _merge_opts(options, kwargs)
        async for ev in self.api.messages.create(_stream=True, _headers=opts.extra_headers, _query=opts.extra_query,
            **self._payload(messages, opts, stream=True)):
            d = normalize_anthropic_event(ev)
            if d is None: continue
            if d.finish_reason == "message_stop": return
            yield d

    async def alist_models(self):
        "List Anthropic models through the OpenAPI layer."
        return await self.api.models.list()


class GeminiClient(BaseLLMClient):
    "Gemini native generateContent/streamGenerateContent client."
    def __init__(self, config: ClientConfig, *, caps: Optional[Caps] = None, api: Optional[OpenAPIClient] = None):
        base_url = config.base_url or "https://generativelanguage.googleapis.com/v1beta"
        hdrs = {"x-goog-api-key": config.api_key or "", **config.default_headers}
        api = api or OpenAPIClient(base_url=base_url, headers=hdrs, timeout=config.timeout, ops=gemini_ops())
        caps = caps or Caps(tools=True, tool_choice=True, streaming=True, reasoning=True, search=True,
            prompt_caching=True, images=True, url_context=True)
        super().__init__(config, caps=caps, api=api)

    def _params(self, opts: RequestOptions, *, stream: bool):
        "Build Gemini query params."
        p = {}
        if stream: p["alt"] = "sse"
        if opts.extra_query: p.update(opts.extra_query)
        return p

    def _gemini_part(self, p):
        "Serialize a normalized part for Gemini contents format."
        raw = _provider_part(p, "gemini")
        if raw is not None: return raw

        if p.type == "text": return {"text": p.text or ""}

        if p.type in ("image", "input_image", "image_url"):
            d = dict(p.data or {})
            if "inlineData" in d or "inline_data" in d:
                v = d.pop("inlineData", d.pop("inline_data", None))
                return {"inlineData": v, **d}
            if "fileData" in d or "file_data" in d:
                v = d.pop("fileData", d.pop("file_data", None))
                return {"fileData": v, **d}
            uri = d.pop("fileUri", None) or d.pop("uri", None) or d.pop("url", None) or p.text
            if uri:
                mt = d.pop("mimeType", None) or d.pop("mime_type", None) or "image/*"
                return {"fileData": {"mimeType": mt, "fileUri": uri}, **d}
            return d

        if p.type in ("input_audio", "audio"):
            d = dict(p.data or {})
            if "inlineData" in d or "inline_data" in d:
                v = d.pop("inlineData", d.pop("inline_data", None))
                return {"inlineData": v, **d}
            return d

        if p.type in ("pdf", "document", "input_file", "file"):
            d = dict(p.data or {})
            if "fileData" in d or "file_data" in d:
                v = d.pop("fileData", d.pop("file_data", None))
                return {"fileData": v, **d}
            uri = d.pop("fileUri", None) or d.pop("uri", None) or d.pop("url", None) or p.text
            if uri:
                mt = d.pop("mimeType", None) or d.pop("mime_type", None) or "application/pdf"
                return {"fileData": {"mimeType": mt, "fileUri": uri}, **d}
            return d

        obj = dict(p.data or {})
        if p.text is not None: obj.setdefault("text", p.text)
        return obj

    def _messages(self, messages: list[Msg]):
        "Serialize normalized messages for Gemini contents format."
        out = []
        for m in messages:
            parts = [self._gemini_part(p) for p in m.content]
            role = "model" if m.role == "assistant" else "user"
            out.append({"role": role, "parts": parts})
        return out

    def _payload(self, messages: list[Msg], opts: RequestOptions):
        "Build Gemini request payload."
        if opts.tools: self._require_caps(["tools"])
        if opts.tool_choice is not None: self._require_caps(["tool_choice"])
        if opts.search: self._require_caps(["search"])

        body = {"contents": self._messages(messages)}
        gen = {}
        if opts.max_tokens is not None: gen["maxOutputTokens"] = opts.max_tokens
        if opts.temperature is not None: gen["temperature"] = opts.temperature
        think = _gemini_thinking_config(opts.reasoning_effort)
        if think is not None: gen["thinkingConfig"] = think
        if gen: body["generationConfig"] = gen

        if opts.tools:
            body["tools"] = [{"functionDeclarations": [_gemini_tool_obj(t) for t in opts.tools]}]
        if opts.search:
            search_obj = opts.search if isinstance(opts.search, dict) else {}
            tools = list(body.get("tools") or [])
            tools.append({"googleSearch": search_obj})
            body["tools"] = tools

        if opts.tool_choice is not None:
            tcfg = _gemini_tool_choice(opts.tool_choice)
            if tcfg: body["toolConfig"] = tcfg

        _gemini_cache(body, opts.cache)
        if opts.native: body.update(opts.native)
        if opts.extra_body: body.update(opts.extra_body)
        return body

    async def acomplete(self, messages: list[Msg], *, options: Optional[RequestOptions] = None, **kwargs) -> Completion:
        "Non-stream Gemini completion."
        opts = _merge_opts(options, kwargs)
        raw = await self.api.models.generate_content(model=_gemini_model_ref(self.config.model), _query=self._params(opts, stream=False),
            _headers=opts.extra_headers, **self._payload(messages, opts))
        return normalize_gemini_generate(raw, model=self.config.model, provider=self.config.provider or "gemini")

    async def astream(self, messages: list[Msg], *, options: Optional[RequestOptions] = None, **kwargs) -> AsyncIterator[Delta]:
        "Stream Gemini completion."
        opts = _merge_opts(options, kwargs)
        emitted = ""
        async for ev in self.api.models.stream_generate_content(model=_gemini_model_ref(self.config.model), _stream=True,
            _query=self._params(opts, stream=True), _headers=opts.extra_headers, **self._payload(messages, opts)):
            d = normalize_gemini_event(ev, emitted)
            if d.text: emitted += d.text
            if d.text or d.finish_reason or d.usage or d.tool_calls: yield d
            if d.finish_reason: return
