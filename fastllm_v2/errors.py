"fastllm_v2 errors."

from __future__ import annotations

from typing import Any, Optional
import json

import httpx


class FastLLMError(Exception):
    "Base fastllm_v2 error."


class UnsupportedCapabilityError(FastLLMError):
    "Raised when a requested feature is unsupported."


class ProtocolError(FastLLMError):
    "Raised when provider payloads do not match expected protocol shape."


class SpecError(FastLLMError):
    "Raised when an OpenAPI spec cannot be parsed as expected."


def _to_text(v: Any) -> str:
    "Best-effort stringify helper."
    if v is None: return ""
    if isinstance(v, str): return v
    try: return json.dumps(v, ensure_ascii=False)
    except Exception: return str(v)


def _retryable(status_code: Optional[int], error_type: Any, code: Any, message: str) -> bool:
    "Classify transient/retryable API failures."
    if isinstance(status_code, int) and (status_code >= 500 or status_code in (408, 409, 425, 429)): return True
    t = str(error_type or "").lower()
    c = str(code or "").lower()
    m = (message or "").lower()
    hints = ("server_error", "internal", "overload", "rate_limit", "timeout", "unavailable", "temporar")
    if any(h in t for h in hints): return True
    if any(h in c for h in hints): return True
    if any(h in m for h in ("try again", "server error", "temporar", "timeout", "overloaded", "unavailable")): return True
    return False


def _req_id(headers: Optional[httpx.Headers]) -> str:
    "Extract provider request id header when available."
    if headers is None: return ""
    for k in ("x-request-id", "request-id", "anthropic-request-id", "x-goog-request-id"):
        v = headers.get(k)
        if v: return str(v)
    return ""


def _parse_http_error_response(resp: httpx.Response) -> tuple[str, str, Any, Any]:
    "Parse common HTTP API error shapes to (message, error_type, code, raw)."
    raw: Any
    try:
        raw = resp.json()
    except Exception:
        raw = resp.text

    msg, et, code, status = "", "", None, None
    if isinstance(raw, dict):
        err = raw.get("error")
        if isinstance(err, dict):
            msg = str(err.get("message") or raw.get("message") or raw.get("detail") or "")
            status = err.get("status", raw.get("status"))
            et = str(err.get("type") or status or raw.get("type") or "")
            code = err.get("code", raw.get("code"))
        elif err is not None:
            msg = _to_text(err)
            status = raw.get("status")
            et = str(raw.get("type") or status or "")
            code = raw.get("code")
        else:
            msg = str(raw.get("message") or raw.get("detail") or raw.get("error_description") or "")
            status = raw.get("status")
            et = str(raw.get("type") or status or "")
            code = raw.get("code")
        if not msg: msg = _to_text(raw)
    else:
        msg = _to_text(raw)
    # Prefer semantic provider codes (e.g. Gemini `status`) over numeric HTTP-like codes.
    if (code is None or isinstance(code, int)) and isinstance(status, str) and status:
        code = status
    if code in (None, "") and isinstance(et, str) and et:
        code = et
    return msg, et, code, raw


def _parse_sse_error_event(event: Any) -> tuple[str, str, Any, Any]:
    "Parse common SSE `error` event shapes to (message, error_type, code, raw)."
    raw = event
    e = event.get("error") if isinstance(event, dict) and isinstance(event.get("error"), dict) else (
        event if isinstance(event, dict) else {"message": _to_text(event)})
    msg = str(e.get("message") or e.get("detail") or _to_text(e))
    et = str(e.get("type") or e.get("status") or "")
    code = e.get("code")
    if code in (None, "") and et: code = et
    return msg, et, code, raw


class APIError(FastLLMError):
    "Structured provider/API error with context."
    def __init__(self, message: str, *, provider: str = "", model: str = "", endpoint: str = "",
        status_code: Optional[int] = None, error_type: str = "", code: Any = None, request_id: str = "",
        retryable: Optional[bool] = None, raw: Any = None):
        self.message = message or "API request failed"
        self.provider = provider or ""
        self.model = model or ""
        self.endpoint = endpoint or ""
        self.status_code = status_code
        self.error_type = error_type or ""
        self.code = code
        self.request_id = request_id or ""
        self.retryable = _retryable(status_code, self.error_type, self.code, self.message) if retryable is None else bool(retryable)
        self.raw = raw
        super().__init__(self.__str__())

    def with_context(self, *, provider: str = "", model: str = "", endpoint: str = "") -> "APIError":
        "Return a copy with missing context fields filled."
        return APIError(
            self.message,
            provider=self.provider or provider,
            model=self.model or model,
            endpoint=self.endpoint or endpoint,
            status_code=self.status_code,
            error_type=self.error_type,
            code=self.code,
            request_id=self.request_id,
            retryable=self.retryable,
            raw=self.raw,
        )

    def __str__(self):
        bits = []
        if self.provider: bits.append(self.provider)
        if self.endpoint: bits.append(self.endpoint)
        if self.model: bits.append(f"model={self.model}")
        if self.status_code is not None: bits.append(f"status={self.status_code}")
        if self.error_type: bits.append(f"type={self.error_type}")
        if self.code not in (None, ""): bits.append(f"code={self.code}")
        if self.request_id: bits.append(f"request_id={self.request_id}")
        if self.retryable: bits.append("retryable=True")
        pref = " ".join(bits)
        return f"{pref}: {self.message}" if pref else self.message


def api_error_from_http(exc: httpx.HTTPStatusError, *, provider: str = "", model: str = "", endpoint: str = "") -> APIError:
    "Build APIError from httpx HTTPStatusError."
    resp = exc.response
    msg, et, code, raw = _parse_http_error_response(resp)
    ep = endpoint
    if not ep and resp.request is not None:
        ep = f"{resp.request.method.upper()} {resp.request.url.path}"
    return APIError(
        msg,
        provider=provider,
        model=model,
        endpoint=ep,
        status_code=resp.status_code,
        error_type=et,
        code=code,
        request_id=_req_id(resp.headers),
        raw=raw,
    )


def api_error_from_event(event: Any, *, provider: str = "", model: str = "", endpoint: str = "") -> APIError:
    "Build APIError from provider SSE/event-level error payload."
    msg, et, code, raw = _parse_sse_error_event(event)
    return APIError(msg, provider=provider, model=model, endpoint=endpoint, error_type=et, code=code, raw=raw)
