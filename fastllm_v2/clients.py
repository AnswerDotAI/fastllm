"Provider clients built on the OpenAPI operation layer."

from __future__ import annotations

import json
from dataclasses import fields, replace
from typing import Any, AsyncIterator, Optional
import warnings
import mimetypes
import httpx

from fastcore.meta import delegates

from .builtin_specs import anthropic_ops, gemini_ops, openai_ops
from .config import ClientConfig
from .errors import APIError, UnsupportedCapabilityError, api_error_from_http
from .normalize import normalize_anthropic_event, normalize_anthropic_message, normalize_gemini_event
from .normalize import normalize_gemini_generate, normalize_openai_chat_completion, normalize_openai_chat_delta
from .normalize import normalize_openai_response, normalize_openai_response_event
from .oapi import OpenAPIClient
from .types import Caps, Completion, Delta, Msg, RequestOptions, ToolCall


_REQ_OPT_FIELDS = {f.name for f in fields(RequestOptions)}
_REQ_OPT_ALIASES = {"headers": "extra_headers", "query": "extra_query", "body": "extra_body"}


def _tool_fn(t: Any) -> Optional[dict]:
    "Extract provider-agnostic function schema from a tool-like object."
    if isinstance(t, dict):
        if isinstance(t.get("function"), dict):
            fn = t["function"]
            name = str(fn.get("name") or "")
            params = fn.get("parameters")
            return dict(name=name, description=str(fn.get("description") or ""),
                parameters=params if isinstance(params, dict) else {})
        if t.get("type") == "function" and "name" in t:
            params = t.get("parameters")
            return dict(name=str(t.get("name") or ""), description=str(t.get("description") or ""),
                parameters=params if isinstance(params, dict) else {})
        if "name" in t and ("parameters" in t or "input_schema" in t):
            params = t.get("parameters", t.get("input_schema"))
            return dict(name=str(t.get("name") or ""), description=str(t.get("description") or ""),
                parameters=params if isinstance(params, dict) else {})
    return None


def _to_json_obj(v: Any) -> dict[str, Any]:
    "Parse tool arguments into dict form, preserving raw when decoding fails."
    if isinstance(v, dict): return dict(v)
    if isinstance(v, str):
        try:
            p = json.loads(v)
            return p if isinstance(p, dict) else {"_value": p}
        except Exception:
            return {"_raw": v}
    return {}


def _canonical_tool_calls(v: Any) -> list[dict[str, Any]]:
    "Normalize tool-call-like payloads to canonical dicts."
    if not isinstance(v, list): return []
    out = []
    for i,tc in enumerate(v):
        if isinstance(tc, ToolCall):
            out.append({"id": str(tc.id or f"call_{i}"), "name": str(tc.name or ""), "arguments": dict(tc.arguments or {})})
            continue
        if isinstance(tc, dict):
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            args = tc.get("arguments", fn.get("arguments", {}))
            out.append({
                "id": str(tc.get("id", tc.get("call_id", tc.get("tool_call_id", f"call_{i}"))) or f"call_{i}"),
                "name": str(tc.get("name", fn.get("name", "")) or ""),
                "arguments": _to_json_obj(args),
            })
            continue
        out.append({
            "id": str(getattr(tc, "id", "") or f"call_{i}"),
            "name": str(getattr(tc, "name", "") or ""),
            "arguments": _to_json_obj(getattr(tc, "arguments", {})),
        })
    return out


def _tool_output_text(msg: Msg, data: dict[str, Any]) -> str:
    "Extract canonical tool output as text."
    if "output" in data:
        out = data.pop("output")
        if isinstance(out, str): return out
        try: return json.dumps(out, ensure_ascii=False)
        except Exception: return str(out)
    if len(msg.content) == 1 and msg.content[0].type == "text":
        return msg.content[0].text or ""
    txt = "".join((p.text or "") for p in msg.content if isinstance(p.text, str))
    if txt: return txt
    if not msg.content: return ""
    try:
        return json.dumps(
            [{"type": p.type, "text": p.text, "data": p.data} for p in msg.content],
            ensure_ascii=False,
        )
    except Exception:
        return ""


def _tool_output_obj(msg: Msg, data: dict[str, Any]) -> Any:
    "Extract canonical tool output as object for Gemini functionResponse."
    if "response" in data: return data.pop("response")
    txt = _tool_output_text(msg, data)
    if not txt: return {}
    try:
        parsed = json.loads(txt)
        return parsed if isinstance(parsed, (dict, list)) else {"content": parsed}
    except Exception:
        return {"content": txt}


def _json_dumps(v: Any) -> str:
    "Compact JSON serializer for tool-argument strings."
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def _strip_openai_file_meta(d: dict[str, Any]) -> dict[str, Any]:
    "Drop OpenAI convenience file fields not accepted by Anthropic/Gemini content parts."
    out = dict(d or {})
    for k in ("filename", "mimeType", "mime_type"):
        out.pop(k, None)
    return out


def _openai_responses_tools(tools: list[Any]) -> list[dict]:
    "Normalize tools to OpenAI Responses API tool shape."
    out = []
    for t in tools:
        if isinstance(t, dict):
            if t.get("type") == "function" and "name" in t: out.append(dict(t)); continue
            if t.get("type") != "function" and "function" not in t and "name" not in t:
                out.append(dict(t))
                continue
        fn = _tool_fn(t)
        if fn is None:
            if isinstance(t, dict): out.append(dict(t)); continue
            raise TypeError(f"Unsupported tool type: {type(t).__name__}")
        out.append(dict(type="function", name=fn["name"], description=fn.get("description", ""),
            parameters=fn.get("parameters") or {}))
    return out


def _openai_chat_tools(tools: list[Any]) -> list[dict]:
    "Normalize tools to OpenAI Chat Completions tool shape."
    out = []
    for t in tools:
        if isinstance(t, dict) and t.get("type") == "function" and isinstance(t.get("function"), dict):
            out.append(dict(t))
            continue
        fn = _tool_fn(t)
        if fn is None:
            if isinstance(t, dict): out.append(dict(t)); continue
            raise TypeError(f"Unsupported tool type: {type(t).__name__}")
        out.append(dict(type="function", function=dict(name=fn["name"], description=fn.get("description", ""),
            parameters=fn.get("parameters") or {})))
    return out


def _anthropic_tools(tools: list[Any]) -> list[dict]:
    "Normalize tools to Anthropic tool shape."
    out = []
    for t in tools:
        if isinstance(t, dict) and "name" in t and "input_schema" in t:
            out.append(dict(t))
            continue
        fn = _tool_fn(t)
        if fn is None:
            if isinstance(t, dict): out.append(dict(t)); continue
            raise TypeError(f"Unsupported tool type: {type(t).__name__}")
        out.append(dict(name=fn["name"], description=fn.get("description", ""),
            input_schema=fn.get("parameters") or {}))
    return out


def _provider_part(p, nm: str) -> Optional[dict]:
    "Return provider-native part payload when available."
    if not p.data: return None
    if nm in p.data and isinstance(p.data[nm], dict): return dict(p.data[nm])
    key = f"_{nm}"
    if key in p.data and isinstance(p.data[key], dict): return dict(p.data[key])
    return None


def _data_url_to_base64(url: Any) -> Optional[tuple[str, str]]:
    "Parse `data:*;base64,...` URLs into `(mime_type, base64_data)`."
    if not isinstance(url, str) or not url.startswith("data:"): return None
    if "," not in url: return None
    header, body = url.split(",", 1)
    if ";base64" not in header or not body: return None
    mime = header[5:].split(";", 1)[0].strip() or "application/octet-stream"
    return mime, body


def _openai_like_image_ref(d: dict[str, Any], text: Optional[str] = None) -> tuple[Optional[str], dict[str, Any]]:
    "Extract OpenAI-style image reference (`image_url`) into a URL or data URL."
    out = dict(d)
    img = out.pop("image_url", None)
    nested = out.pop("input_image", None)
    if img is None and isinstance(nested, dict): img = nested.get("image_url", nested.get("url"))

    ref = None
    if isinstance(img, str): ref = img
    elif isinstance(img, dict):
        ref = img.get("url") or img.get("image_url") or img.get("uri") or img.get("fileUri")
        for k,v in img.items():
            if k in ("url", "image_url", "uri", "fileUri", "detail"): continue
            out.setdefault(k, v)

    if ref is None:
        ref = out.pop("file_url", None) or out.pop("fileUri", None) or out.pop("uri", None) or out.pop("url", None) or text

    # OpenAI-only image hint; don't forward to providers that don't understand it.
    out.pop("detail", None)
    return ref, out


def _openai_like_file_ref(d: dict[str, Any], text: Optional[str] = None) -> tuple[Optional[str], dict[str, Any]]:
    "Extract OpenAI-style file reference (`file_url`) into a URL or data URL."
    out = dict(d)
    nested = out.pop("input_file", None)
    ref = out.pop("file_url", None) or out.pop("fileUri", None) or out.pop("uri", None) or out.pop("url", None) or text

    if isinstance(nested, dict):
        if ref is None:
            ref = nested.get("file_url") or nested.get("fileUri") or nested.get("uri") or nested.get("url")
            if ref is None and isinstance(nested.get("file_data"), str): ref = nested.get("file_data")
        for k in ("mimeType", "mime_type", "filename"):
            if k in nested and k not in out: out[k] = nested[k]
    return ref, out


def _openai_like_video_ref(d: dict[str, Any], text: Optional[str] = None) -> tuple[Optional[str], dict[str, Any]]:
    "Extract OpenAI-style video reference (`video_url`) into a URL or data URL."
    out = dict(d)
    vid = out.pop("video_url", None)
    nested = out.pop("input_video", None)
    if vid is None and isinstance(nested, dict): vid = nested.get("video_url", nested.get("url"))

    ref = None
    if isinstance(vid, str): ref = vid
    elif isinstance(vid, dict):
        ref = vid.get("url") or vid.get("video_url") or vid.get("uri") or vid.get("fileUri")
        for k,v in vid.items():
            if k in ("url", "video_url", "uri", "fileUri"): continue
            out.setdefault(k, v)

    if ref is None:
        ref = out.pop("file_url", None) or out.pop("fileUri", None) or out.pop("uri", None) or out.pop("url", None) or text
    return ref, out


def _audio_format_to_mime(fmt: Any) -> str:
    "Map audio format extension/name to mime type."
    if not isinstance(fmt, str): return "audio/*"
    f = fmt.strip().lower().lstrip(".")
    m = {
        "mp3": "audio/mpeg",
        "mpeg": "audio/mpeg",
        "wav": "audio/wav",
        "webm": "audio/webm",
        "ogg": "audio/ogg",
        "oga": "audio/ogg",
        "m4a": "audio/mp4",
        "mp4": "audio/mp4",
        "aac": "audio/aac",
        "flac": "audio/flac",
    }
    if "/" in f: return f
    return m.get(f, f"audio/{f}")


def _default_filename_for_mime(mime_type: str) -> str:
    "Infer a stable default filename for OpenAI file parts when filename is omitted."
    mt = str(mime_type or "").split(";", 1)[0].strip().lower() or "application/octet-stream"
    ext = mimetypes.guess_extension(mt) or ".bin"
    return f"upload{ext}"


def _ensure_openai_file_data_url(d: dict[str, Any]) -> dict[str, Any]:
    "Ensure OpenAI `file_data` is sent as a data URL (accept raw base64 input for convenience)."
    out = dict(d)
    fd = out.get("file_data")
    if not isinstance(fd, str) or not fd: return out
    parsed = _data_url_to_base64(fd)
    if parsed is not None:
        mt, _ = parsed
        fn = out.get("filename")
        if not isinstance(fn, str) or not fn: out["filename"] = _default_filename_for_mime(mt)
        return out
    mt = out.get("mimeType") or out.get("mime_type")
    if not isinstance(mt, str) or not mt:
        fn = out.get("filename")
        mt = mimetypes.guess_type(str(fn))[0] if isinstance(fn, str) and fn else None
    if not isinstance(mt, str) or not mt: mt = "application/octet-stream"
    out["file_data"] = f"data:{mt};base64,{fd}"
    fn = out.get("filename")
    if not isinstance(fn, str) or not fn: out["filename"] = _default_filename_for_mime(mt)
    return out


def _mime_from_meta(d: dict[str, Any], default: str) -> str:
    "Resolve mime type from explicit fields or filename fallback."
    mt = d.get("mimeType") or d.get("mime_type")
    if isinstance(mt, str) and mt: return mt
    fn = d.get("filename") or d.get("name")
    if isinstance(fn, str) and fn:
        g = mimetypes.guess_type(fn)[0]
        if isinstance(g, str) and g: return g
    return default


def _canonical_media_kind(ptype: str) -> Optional[str]:
    "Map part.type aliases to canonical media kinds."
    t = str(ptype or "").strip().lower()
    if t == "text": return None
    if t in ("image", "image_url", "input_image"): return "input_image"
    if t in ("audio", "input_audio"): return "input_audio"
    if t in ("video", "video_url", "input_video"): return "input_video"
    if t in ("file", "input_file", "pdf", "document"): return "input_file"
    return None


def _message_media_kinds(messages: list[Msg]) -> set[str]:
    "Collect canonical media kinds present in message parts."
    kinds = set()
    for m in messages:
        for p in m.content:
            k = _canonical_media_kind(getattr(p, "type", ""))
            if k is not None: kinds.add(k)
    return kinds


def _validate_media_support(provider: str, model: str, messages: list[Msg]):
    "Validate canonical media kinds against provider capabilities."
    kinds = _message_media_kinds(messages)
    if not kinds: return
    p = (provider or "").strip().lower()
    m = (model or "").strip()

    if p == "anthropic":
        bad = sorted(kinds.intersection({"input_audio", "input_video"}))
        if bad:
            raise UnsupportedCapabilityError(
                f"Model '{m}' on provider '{p}' does not support canonical media type(s): {', '.join(bad)}. "
                "Use provider-native blocks via Part.data['anthropic'] when needed.",
            )
        return

    if p == "openai_compat":
        # Compatibility endpoints vary widely; flag non-text modalities proactively.
        bad = sorted(kinds.intersection({"input_image", "input_audio", "input_video", "input_file"}))
        if bad:
            warnings.warn(
                f"OpenAI-compatible model '{m}' may not fully support canonical media type(s): {', '.join(bad)}. "
                "If this fails, pass provider-native payloads via `native` or Part.data provider keys.",
                UserWarning,
                stacklevel=3,
            )


def _merge_opts(options: Optional[RequestOptions], kw: dict[str, Any]) -> RequestOptions:
    "Merge keyword overrides into RequestOptions; unknown kwargs go to native body."
    opts = options or RequestOptions()
    if not kw: return opts
    if "search" in kw:
        raise TypeError("`search` has been removed; pass provider-native web-search tools via `tools=[...]` "
            "or provider fields via `native`/`extra_body`.")

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


def _gemini_tools(tools: list[Any]) -> list[dict]:
    "Normalize tools to Gemini tools payload shape."
    out, fn_decls = [], []
    for t in tools:
        if isinstance(t, dict):
            if "functionDeclarations" in t and isinstance(t["functionDeclarations"], list):
                out.append(dict(t))
                continue
            if any(k in t for k in ("googleSearch", "googleSearchRetrieval", "codeExecution")) and "function" not in t and "name" not in t:
                out.append(dict(t))
                continue
        fn = _tool_fn(t)
        if fn is None:
            if isinstance(t, dict): out.append(dict(t)); continue
            raise TypeError(f"Unsupported tool type: {type(t).__name__}")
        fn_decls.append(dict(name=fn["name"], description=fn.get("description", ""),
            parameters=fn.get("parameters") or {}))
    if fn_decls: out.insert(0, dict(functionDeclarations=fn_decls))
    return out


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


def _raise_api_error(exc: Exception, *, provider: str, model: str, endpoint: str):
    "Raise structured APIError for HTTP/provider failures with context."
    if isinstance(exc, APIError):
        raise exc.with_context(provider=provider, model=model, endpoint=endpoint) from exc
    if isinstance(exc, httpx.HTTPStatusError):
        raise api_error_from_http(exc, provider=provider, model=model, endpoint=endpoint) from exc
    raise exc


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

        if p.type in ("input_file", "file", "pdf", "document", "input_video", "video", "video_url"):
            d = dict(p.data or {})
            if p.type in ("input_video", "video", "video_url"): ref, d = _openai_like_video_ref(d, p.text)
            else: ref, d = _openai_like_file_ref(d, p.text)
            d = _ensure_openai_file_data_url(d)
            obj = {"type": "input_file", **d}
            if isinstance(ref, str) and all(k not in obj for k in ("file_id", "file_data", "file_url")): obj["file_url"] = ref
            if "url" in obj and "file_url" not in obj: obj["file_url"] = obj.pop("url")
            obj.pop("mimeType", None)
            obj.pop("mime_type", None)
            obj.pop("videoMetadata", None)
            obj.pop("video_metadata", None)
            return obj

        obj = {"type": p.type}
        if p.text is not None: obj["text"] = p.text
        if p.data: obj.update(p.data)
        return obj

    def _responses_messages(self, messages: list[Msg]):
        "Serialize normalized messages to Responses API input format."
        out = []
        for m in messages:
            data = dict(m.data or {})
            raw = data.pop("openai", None)
            if isinstance(raw, dict) and ("role" in raw or "type" in raw):
                out.append(raw)
                continue
            raw_extra = dict(raw) if isinstance(raw, dict) else {}

            if m.role == "tool":
                tcid = str(data.pop("tool_call_id", data.pop("call_id", data.pop("id", ""))) or "")
                item = {"type": "function_call_output", "call_id": tcid, "output": _tool_output_text(m, data)}
                if nm := data.pop("name", None):
                    item["name"] = str(nm)
                for src in (raw_extra, data):
                    for k,v in src.items():
                        if k in ("openai_chat", "anthropic", "gemini"): continue
                        item[k] = v
                out.append(item)
                continue

            tcs = _canonical_tool_calls(data.pop("tool_calls", None)) if m.role == "assistant" else []
            parts = [self._responses_part(m.role, p) for p in m.content]
            if parts or not tcs:
                obj = {"role": m.role, "content": parts}
                for src in (raw_extra, data):
                    for k,v in src.items():
                        if k in ("openai_chat", "anthropic", "gemini"): continue
                        obj[k] = v
                out.append(obj)
            for tc in tcs:
                out.append({
                    "type": "function_call",
                    "call_id": tc["id"],
                    "name": tc["name"],
                    "arguments": _json_dumps(tc.get("arguments") or {}),
                })
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

        if p.type in ("input_file", "file", "pdf", "document", "input_video", "video", "video_url"):
            d = dict(p.data or {})
            if p.type in ("input_video", "video", "video_url"): ref, d = _openai_like_video_ref(d, p.text)
            else: ref, d = _openai_like_file_ref(d, p.text)
            d = _ensure_openai_file_data_url(d)
            obj = {"type": "file", **d}
            if isinstance(ref, str) and all(k not in obj for k in ("file_id", "file_data", "file_url")): obj["file_url"] = ref
            if "url" in obj and "file_url" not in obj: obj["file_url"] = obj.pop("url")
            obj.pop("mimeType", None)
            obj.pop("mime_type", None)
            obj.pop("videoMetadata", None)
            obj.pop("video_metadata", None)
            return obj

        obj = {"type": p.type}
        if p.text is not None: obj["text"] = p.text
        if p.data: obj.update(p.data)
        return obj

    def _chat_messages(self, messages: list[Msg]):
        "Serialize normalized messages to chat.completions messages."
        res = []
        for m in messages:
            data = dict(m.data or {})
            raw = data.pop("openai_chat", None)
            if isinstance(raw, dict) and "role" in raw:
                res.append(raw)
                continue
            raw_extra = dict(raw) if isinstance(raw, dict) else {}

            if m.role == "tool":
                obj = {"role": "tool", "content": _tool_output_text(m, data)}
                tcid = data.pop("tool_call_id", data.pop("call_id", data.pop("id", None)))
                if tcid is not None: obj["tool_call_id"] = str(tcid)
                data.pop("name", None)
                for src in (raw_extra, data):
                    for k,v in src.items():
                        if k in ("openai", "anthropic", "gemini"): continue
                        obj[k] = v
                res.append(obj)
                continue

            tcs = _canonical_tool_calls(data.pop("tool_calls", None)) if m.role == "assistant" else []

            if len(m.content) == 1 and m.content[0].type == "text":
                obj = {"role": m.role, "content": m.content[0].text or ""}
            else:
                cts = [self._chat_part(p) for p in m.content]
                obj = {"role": m.role, "content": cts} if cts else {"role": m.role}

            if tcs:
                obj["tool_calls"] = [{
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": _json_dumps(tc.get("arguments") or {})},
                } for tc in tcs]

            for src in (raw_extra, data):
                for k,v in src.items():
                    if k in ("openai", "anthropic", "gemini"): continue
                    obj[k] = v
            res.append(obj)
        return res

    def _responses_payload(self, messages: list[Msg], opts: RequestOptions, *, stream: bool):
        "Build Responses API request payload."
        _validate_media_support(self.config.provider or "openai", self.config.model, messages)
        if opts.tools: self._require_caps(["tools"])
        if opts.tool_choice is not None: self._require_caps(["tool_choice"])
        if opts.reasoning_effort is not None: self._require_caps(["reasoning"])

        payload = {"model": self.config.model, "input": self._responses_messages(messages), "stream": stream}
        if opts.max_tokens is not None: payload["max_output_tokens"] = opts.max_tokens
        if opts.temperature is not None: payload["temperature"] = opts.temperature
        if opts.response_format is not None:
            txt = dict(payload.get("text") or {})
            txt["format"] = _openai_text_format(opts.response_format)
            payload["text"] = txt
        if opts.tools: payload["tools"] = _openai_responses_tools(opts.tools)
        if opts.tool_choice is not None: payload["tool_choice"] = opts.tool_choice
        if opts.reasoning_effort is not None: payload["reasoning"] = {"effort": opts.reasoning_effort}
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
        _validate_media_support(self.config.provider or "openai", self.config.model, messages)
        if opts.tools: self._require_caps(["tools"])
        if opts.tool_choice is not None: self._require_caps(["tool_choice"])
        if opts.reasoning_effort is not None: self._require_caps(["reasoning"])

        payload = {"model": self.config.model, "messages": self._chat_messages(messages), "stream": stream}
        if opts.max_tokens is not None: payload["max_tokens"] = opts.max_tokens
        if opts.temperature is not None: payload["temperature"] = opts.temperature
        if opts.response_format is not None: payload["response_format"] = opts.response_format
        if opts.tools: payload["tools"] = _openai_chat_tools(opts.tools)
        if opts.tool_choice is not None: payload["tool_choice"] = opts.tool_choice
        if opts.reasoning_effort is not None: payload["reasoning_effort"] = opts.reasoning_effort
        _openai_cache(payload, opts.cache)
        if opts.native: payload.update(opts.native)
        if opts.extra_body: payload.update(opts.extra_body)
        return payload

    @delegates(RequestOptions, keep=True)
    async def acomplete(self, messages: list[Msg], *, options: Optional[RequestOptions] = None, **kwargs) -> Completion:
        "Responses API non-stream completion."
        opts = _merge_opts(options, kwargs)
        payload = self._responses_payload(messages, opts, stream=False)
        try:
            raw = await self._op("responses", "create", "create_response")(_headers=opts.extra_headers,
                _query=opts.extra_query, **payload)
        except Exception as e:
            _raise_api_error(e, provider=self.config.provider or "openai", model=self.config.model, endpoint="responses.create")
        return normalize_openai_response(raw, model=self.config.model, provider=self.config.provider or "openai")

    @delegates(RequestOptions, keep=True)
    async def astream(self, messages: list[Msg], *, options: Optional[RequestOptions] = None, **kwargs) -> AsyncIterator[Delta]:
        "Responses API streaming completion."
        opts = _merge_opts(options, kwargs)
        payload = self._responses_payload(messages, opts, stream=True)
        op = self._op("responses", "create", "create_response")
        try:
            async for ev in op(_stream=True, _headers=opts.extra_headers, _query=opts.extra_query, **payload):
                d = normalize_openai_response_event(ev)
                if d is not None: yield d
        except Exception as e:
            _raise_api_error(e, provider=self.config.provider or "openai", model=self.config.model, endpoint="responses.stream")

    @delegates(RequestOptions, keep=True)
    async def achat_complete(self, messages: list[Msg], *, options: Optional[RequestOptions] = None, **kwargs) -> Completion:
        "Chat Completions non-stream completion."
        opts = _merge_opts(options, kwargs)
        payload = self._chat_payload(messages, opts, stream=False)
        op = self._op("chat", "create_completions", "create_chat_completion")
        try:
            raw = await op(_headers=opts.extra_headers, _query=opts.extra_query, **payload)
        except Exception as e:
            _raise_api_error(e, provider=self.config.provider or "openai_chat", model=self.config.model, endpoint="chat.completions")
        return normalize_openai_chat_completion(raw, model=self.config.model, provider=self.config.provider or "openai_chat")

    @delegates(RequestOptions, keep=True)
    async def achat_stream(self, messages: list[Msg], *, options: Optional[RequestOptions] = None, **kwargs) -> AsyncIterator[Delta]:
        "Chat Completions stream."
        opts = _merge_opts(options, kwargs)
        payload = self._chat_payload(messages, opts, stream=True)
        op = self._op("chat", "create_completions", "create_chat_completion")
        try:
            async for ev in op(_stream=True, _headers=opts.extra_headers, _query=opts.extra_query, **payload):
                yield normalize_openai_chat_delta(ev)
        except Exception as e:
            _raise_api_error(e, provider=self.config.provider or "openai_chat", model=self.config.model, endpoint="chat.stream")

    async def aresponse_get(self, response_id: str):
        "Get a response object by id."
        try:
            return await self._op("responses", "retrieve", "get_response")(response_id=response_id)
        except Exception as e:
            _raise_api_error(e, provider=self.config.provider or "openai", model=self.config.model, endpoint="responses.get")

    async def aresponse_delete(self, response_id: str):
        "Delete a response object by id."
        try:
            return await self._op("responses", "delete", "delete_response")(response_id=response_id)
        except Exception as e:
            _raise_api_error(e, provider=self.config.provider or "openai", model=self.config.model, endpoint="responses.delete")

    async def aresponse_cancel(self, response_id: str):
        "Cancel an in-progress response by id."
        try:
            op = getattr(self.api.responses, "cancel", None)
            if op is not None: return await op(response_id=response_id)
            return await self.api.call("/responses/{response_id}/cancel", "POST", route={"response_id": response_id})
        except Exception as e:
            _raise_api_error(e, provider=self.config.provider or "openai", model=self.config.model, endpoint="responses.cancel")

    async def aresponse_input_items(self, response_id: str, *, after: Optional[str] = None, limit: Optional[int] = None):
        "List response input items."
        try:
            return await self._op("responses", "list_input_items")(response_id=response_id, after=after, limit=limit)
        except Exception as e:
            _raise_api_error(e, provider=self.config.provider or "openai", model=self.config.model, endpoint="responses.input_items")

    async def acompact(self, response_ids: list[str], *, model: Optional[str] = None):
        "Compact response history via /responses/compact."
        try:
            op = getattr(self.api.responses, "compact", None)
            if op is not None: return await op(response_ids=response_ids, model=model or self.config.model)
            return await self.api.call("/responses/compact", "POST", body={"response_ids": response_ids, "model": model or self.config.model})
        except Exception as e:
            _raise_api_error(e, provider=self.config.provider or "openai", model=self.config.model, endpoint="responses.compact")

    async def ainput_tokens(self, inp: Any, *, model: Optional[str] = None):
        "Estimate input token count via /responses/input_tokens."
        try:
            op = getattr(self.api.responses, "input_tokens", None)
            if op is not None: return await op(model=model or self.config.model, input=inp)
            return await self.api.call("/responses/input_tokens", "POST", body={"model": model or self.config.model, "input": inp})
        except Exception as e:
            _raise_api_error(e, provider=self.config.provider or "openai", model=self.config.model, endpoint="responses.input_tokens")


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
            ref, d = _openai_like_image_ref(d, p.text)
            if isinstance(ref, str):
                b64 = _data_url_to_base64(ref)
                if b64 is not None:
                    mt, data = b64
                    return {"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}, **d}
                return {"type": "image", "source": {"type": "url", "url": ref}, **d}
            return {"type": "image", **d}

        if p.type in ("pdf", "document", "input_file", "file"):
            d = dict(p.data or {})
            if "source" in d and isinstance(d["source"], dict): return {"type": "document", **d}
            fdata = d.pop("file_data", None)
            d = _strip_openai_file_meta(d)
            if isinstance(fdata, str) and fdata:
                b64 = _data_url_to_base64(fdata)
                if b64 is not None:
                    mt, data = b64
                else:
                    mt, data = _mime_from_meta(d, "application/pdf"), fdata
                return {"type": "document", "source": {"type": "base64", "media_type": mt, "data": data}, **d}
            ref, d = _openai_like_file_ref(d, p.text)
            if isinstance(ref, str):
                b64 = _data_url_to_base64(ref)
                if b64 is not None:
                    mt, data = b64
                    return {"type": "document", "source": {"type": "base64", "media_type": mt, "data": data}, **d}
                return {"type": "document", "source": {"type": "url", "url": ref}, **d}
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
            data = dict(m.data or {})
            raw = data.pop("anthropic", None)
            if isinstance(raw, dict) and "role" in raw:
                res.append(raw)
                continue
            raw_extra = dict(raw) if isinstance(raw, dict) else {}

            if m.role == "tool":
                tcid = str(data.pop("tool_call_id", data.pop("call_id", data.pop("id", ""))) or "")
                data.pop("name", None)
                if len(m.content) == 1 and m.content[0].type == "text":
                    content = m.content[0].text or ""
                elif m.content:
                    content = [self._anthropic_part(p) for p in m.content]
                else:
                    content = _tool_output_text(m, data)
                tr = {"type": "tool_result", "tool_use_id": tcid, "content": content}
                if "is_error" in data: tr["is_error"] = bool(data.pop("is_error"))
                obj = {"role": "user", "content": [tr]}
                for src in (raw_extra, data):
                    for k,v in src.items():
                        if k in ("openai", "openai_chat", "gemini"): continue
                        obj[k] = v
                res.append(obj)
                continue

            tcs = _canonical_tool_calls(data.pop("tool_calls", None)) if m.role == "assistant" else []
            blocks = []
            for p in m.content:
                b = self._anthropic_part(p)
                if cache_ctl and m.role in ("user", "system") and "cache_control" not in b:
                    if b.get("type") in ("text", "image", "document"): b["cache_control"] = cache_ctl
                blocks.append(b)
            for tc in tcs:
                blocks.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc.get("arguments") or {},
                })
            obj = {"role": m.role, "content": blocks}
            for src in (raw_extra, data):
                for k,v in src.items():
                    if k in ("openai", "openai_chat", "gemini"): continue
                    obj[k] = v
            res.append(obj)
        return res

    def _payload(self, messages: list[Msg], opts: RequestOptions, *, stream: bool):
        "Build Anthropic messages payload."
        _validate_media_support(self.config.provider or "anthropic", self.config.model, messages)
        if opts.tools: self._require_caps(["tools"])
        if opts.tool_choice is not None: self._require_caps(["tool_choice"])
        if opts.reasoning_effort is not None: self._require_caps(["reasoning"])

        payload = {
            "model": self.config.model,
            "messages": self._serialize_messages(messages, cache=opts.cache),
            "max_tokens": opts.max_tokens or 1024,
            "stream": stream}
        if opts.temperature is not None: payload["temperature"] = opts.temperature
        if opts.tools: payload["tools"] = _anthropic_tools(opts.tools)
        tc = _anthropic_tool_choice(opts.tool_choice)
        if tc is not None: payload["tool_choice"] = tc
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

    @delegates(RequestOptions, keep=True)
    async def acomplete(self, messages: list[Msg], *, options: Optional[RequestOptions] = None, **kwargs) -> Completion:
        "Non-stream Anthropic completion."
        opts = _merge_opts(options, kwargs)
        try:
            raw = await self.api.messages.create(_headers=opts.extra_headers, _query=opts.extra_query,
                **self._payload(messages, opts, stream=False))
        except Exception as e:
            _raise_api_error(e, provider=self.config.provider or "anthropic", model=self.config.model, endpoint="messages.create")
        return normalize_anthropic_message(raw, model=self.config.model, provider=self.config.provider or "anthropic")

    @delegates(RequestOptions, keep=True)
    async def astream(self, messages: list[Msg], *, options: Optional[RequestOptions] = None, **kwargs) -> AsyncIterator[Delta]:
        "Stream Anthropic completion."
        opts = _merge_opts(options, kwargs)
        try:
            async for ev in self.api.messages.create(_stream=True, _headers=opts.extra_headers, _query=opts.extra_query,
                **self._payload(messages, opts, stream=True)):
                d = normalize_anthropic_event(ev)
                if d is None: continue
                if d.finish_reason == "message_stop":
                    yield d
                    return
                yield d
        except Exception as e:
            _raise_api_error(e, provider=self.config.provider or "anthropic", model=self.config.model, endpoint="messages.stream")

    async def alist_models(self):
        "List Anthropic models through the OpenAPI layer."
        try:
            return await self.api.models.list()
        except Exception as e:
            _raise_api_error(e, provider=self.config.provider or "anthropic", model=self.config.model, endpoint="models.list")


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
            ref, d = _openai_like_image_ref(d, p.text)
            if isinstance(ref, str):
                b64 = _data_url_to_base64(ref)
                if b64 is not None:
                    mt, data = b64
                    return {"inlineData": {"mimeType": mt, "data": data}, **d}
                mt = d.pop("mimeType", None) or d.pop("mime_type", None) or "image/*"
                return {"fileData": {"mimeType": mt, "fileUri": ref}, **d}
            return d

        if p.type in ("input_audio", "audio"):
            d = dict(p.data or {})
            if "inlineData" in d or "inline_data" in d:
                v = d.pop("inlineData", d.pop("inline_data", None))
                return {"inlineData": v, **d}
            nested = d.pop("input_audio", None)
            if isinstance(nested, dict):
                nd = dict(nested)
                data = nd.pop("data", None)
                fmt = nd.pop("mimeType", None) or nd.pop("mime_type", None) or nd.pop("format", None)
                for k,v in nd.items():
                    if k in ("audio_url", "url", "uri", "fileUri", "file_url"): continue
                    d.setdefault(k, v)
                if isinstance(data, str) and data:
                    return {"inlineData": {"mimeType": _audio_format_to_mime(fmt), "data": data}, **d}
                ref = nd.get("audio_url") or nd.get("url") or nd.get("uri") or nd.get("fileUri") or nd.get("file_url")
                if isinstance(ref, str):
                    mt = d.pop("mimeType", None) or d.pop("mime_type", None) or _audio_format_to_mime(fmt)
                    return {"fileData": {"mimeType": mt, "fileUri": ref}, **d}
            aud = d.pop("audio_url", None)
            if isinstance(aud, dict):
                ref = aud.get("url") or aud.get("audio_url") or aud.get("uri") or aud.get("fileUri") or aud.get("file_url")
                for k,v in aud.items():
                    if k in ("url", "audio_url", "uri", "fileUri", "file_url"): continue
                    d.setdefault(k, v)
            elif isinstance(aud, str): ref = aud
            else: ref = d.pop("fileUri", None) or d.pop("uri", None) or d.pop("file_url", None) or d.pop("url", None) or p.text
            if isinstance(ref, str):
                mt = d.pop("mimeType", None) or d.pop("mime_type", None) or "audio/*"
                return {"fileData": {"mimeType": mt, "fileUri": ref}, **d}
            return d

        if p.type in ("input_video", "video", "video_url"):
            d = dict(p.data or {})
            if "inlineData" in d or "inline_data" in d:
                v = d.pop("inlineData", d.pop("inline_data", None))
                return {"inlineData": v, **d}
            if "fileData" in d or "file_data" in d:
                v = d.pop("fileData", d.pop("file_data", None))
                return {"fileData": v, **d}
            ref, d = _openai_like_video_ref(d, p.text)
            if isinstance(ref, str):
                b64 = _data_url_to_base64(ref)
                if b64 is not None:
                    mt, data = b64
                    if mt == "application/octet-stream": mt = "video/mp4"
                    return {"inlineData": {"mimeType": mt, "data": data}, **d}
                mt = d.pop("mimeType", None) or d.pop("mime_type", None) or "video/mp4"
                vm = d.pop("videoMetadata", d.pop("video_metadata", None))
                obj = {"fileData": {"mimeType": mt, "fileUri": ref}, **d}
                if vm is not None: obj["videoMetadata"] = vm
                return obj
            vm = d.pop("videoMetadata", d.pop("video_metadata", None))
            if vm is not None: return {"videoMetadata": vm, **d}
            return d

        if p.type in ("pdf", "document", "input_file", "file"):
            d = dict(p.data or {})
            if "fileData" in d or "file_data" in d:
                v = d.pop("fileData", d.pop("file_data", None))
                d = _strip_openai_file_meta(d)
                if isinstance(v, dict): return {"fileData": v, **d}
                if isinstance(v, str) and v:
                    b64 = _data_url_to_base64(v)
                    if b64 is not None:
                        mt, data = b64
                        return {"inlineData": {"mimeType": mt, "data": data}, **d}
                    if v.startswith(("http://", "https://", "gs://")):
                        mt = _mime_from_meta(d, "application/pdf")
                        return {"fileData": {"mimeType": mt, "fileUri": v}, **d}
                    mt = _mime_from_meta(d, "application/pdf")
                    return {"inlineData": {"mimeType": mt, "data": v}, **d}
                return {"fileData": v, **d}
            ref, d = _openai_like_file_ref(d, p.text)
            d = _strip_openai_file_meta(d)
            if isinstance(ref, str):
                b64 = _data_url_to_base64(ref)
                if b64 is not None:
                    mt, data = b64
                    return {"inlineData": {"mimeType": mt, "data": data}, **d}
                mt = d.pop("mimeType", None) or d.pop("mime_type", None) or "application/pdf"
                return {"fileData": {"mimeType": mt, "fileUri": ref}, **d}
            return d

        obj = dict(p.data or {})
        if p.text is not None: obj.setdefault("text", p.text)
        return obj

    def _messages(self, messages: list[Msg]):
        "Serialize normalized messages for Gemini contents format."
        out = []
        for m in messages:
            data = dict(m.data or {})
            raw = data.pop("gemini", None)
            if isinstance(raw, dict) and "role" in raw:
                out.append(raw)
                continue
            raw_extra = dict(raw) if isinstance(raw, dict) else {}

            if m.role == "tool":
                tcid = str(data.pop("tool_call_id", data.pop("call_id", data.pop("id", ""))) or "")
                nm = str(data.pop("name", "") or "")
                resp = _tool_output_obj(m, data)
                fr = {"name": (nm or "tool"), "response": resp}
                if tcid: fr["id"] = tcid
                obj = {"role": "user", "parts": [{"functionResponse": fr}]}
                for src in (raw_extra, data):
                    for k,v in src.items():
                        if k in ("openai", "openai_chat", "anthropic"): continue
                        obj[k] = v
                out.append(obj)
                continue

            tcs = _canonical_tool_calls(data.pop("tool_calls", None)) if m.role == "assistant" else []
            parts = [self._gemini_part(p) for p in m.content]
            for tc in tcs:
                fc = {"name": tc["name"], "args": tc.get("arguments") or {}}
                if tc["id"]: fc["id"] = tc["id"]
                parts.append({"functionCall": fc})
            role = "model" if m.role == "assistant" else "user"
            obj = {"role": role, "parts": parts}
            for src in (raw_extra, data):
                for k,v in src.items():
                    if k in ("openai", "openai_chat", "anthropic"): continue
                    obj[k] = v
            out.append(obj)
        return out

    def _payload(self, messages: list[Msg], opts: RequestOptions):
        "Build Gemini request payload."
        _validate_media_support(self.config.provider or "gemini", self.config.model, messages)
        if opts.tools: self._require_caps(["tools"])
        if opts.tool_choice is not None: self._require_caps(["tool_choice"])

        body = {"contents": self._messages(messages)}
        gen = {}
        if opts.max_tokens is not None: gen["maxOutputTokens"] = opts.max_tokens
        if opts.temperature is not None: gen["temperature"] = opts.temperature
        think = _gemini_thinking_config(opts.reasoning_effort)
        if think is not None: gen["thinkingConfig"] = think
        if gen: body["generationConfig"] = gen

        if opts.tools: body["tools"] = _gemini_tools(opts.tools)
        if opts.tool_choice is not None:
            tcfg = _gemini_tool_choice(opts.tool_choice)
            if tcfg: body["toolConfig"] = tcfg

        _gemini_cache(body, opts.cache)
        if opts.native: body.update(opts.native)
        if opts.extra_body: body.update(opts.extra_body)
        return body

    @delegates(RequestOptions, keep=True)
    async def acomplete(self, messages: list[Msg], *, options: Optional[RequestOptions] = None, **kwargs) -> Completion:
        "Non-stream Gemini completion."
        opts = _merge_opts(options, kwargs)
        try:
            raw = await self.api.models.generate_content(model=_gemini_model_ref(self.config.model), _query=self._params(opts, stream=False),
                _headers=opts.extra_headers, **self._payload(messages, opts))
        except Exception as e:
            _raise_api_error(e, provider=self.config.provider or "gemini", model=self.config.model, endpoint="models.generate_content")
        return normalize_gemini_generate(raw, model=self.config.model, provider=self.config.provider or "gemini")

    @delegates(RequestOptions, keep=True)
    async def astream(self, messages: list[Msg], *, options: Optional[RequestOptions] = None, **kwargs) -> AsyncIterator[Delta]:
        "Stream Gemini completion."
        opts = _merge_opts(options, kwargs)
        emitted = ""
        try:
            async for ev in self.api.models.stream_generate_content(model=_gemini_model_ref(self.config.model), _stream=True,
                _query=self._params(opts, stream=True), _headers=opts.extra_headers, **self._payload(messages, opts)):
                d = normalize_gemini_event(ev, emitted)
                if d.text: emitted += d.text
                yield d
                if d.finish_reason: return
        except Exception as e:
            _raise_api_error(e, provider=self.config.provider or "gemini", model=self.config.model, endpoint="models.stream_generate_content")
