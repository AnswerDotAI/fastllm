"Shared async HTTP transport."

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from .errors import ProtocolError
from .sse import SSEvent, aiter_sse


class AsyncTransport:
    "Thin async transport wrapper over httpx."
    def __init__(self, *, timeout: float = 60.0, client: Optional[httpx.AsyncClient] = None,
        base_headers: Optional[Dict[str, str]] = None):
        self._own_client = client is None
        self.client = client or httpx.AsyncClient(timeout=timeout)
        self.base_headers = base_headers or {}

    async def aclose(self):
        if self._own_client: await self.client.aclose()

    def _headers(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        return {**self.base_headers, **(headers or {})}

    @staticmethod
    def _decode(resp: httpx.Response) -> Any:
        "Decode response body using content type."
        ctype = (resp.headers.get("content-type") or "").lower()
        if "application/json" in ctype or ctype.endswith("+json"):
            return resp.json()
        if ctype.startswith("text/") or "application/x-ndjson" in ctype:
            return resp.text
        return resp.content

    async def request(self, method: str, url: str, *, headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None, json_data: Optional[Any] = None, data: Optional[Any] = None,
        files: Optional[Any] = None, raw: bool = False) -> Any:
        "Execute a request and decode JSON/text/binary response."
        resp = await self.client.request(method, url, headers=self._headers(headers), params=params,
            json=json_data, data=data, files=files)
        resp.raise_for_status()
        return resp if raw else self._decode(resp)

    async def request_json(self, method: str, url: str, *, headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None, json_data: Optional[Any] = None) -> Dict[str, Any]:
        data = await self.request(method, url, headers=headers, params=params, json_data=json_data)
        if not isinstance(data, dict): raise ProtocolError(f"Expected dict JSON response, got {type(data).__name__}")
        return data

    async def stream_sse(self, method: str, url: str, *, headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None, json_data: Optional[Any] = None, data: Optional[Any] = None,
        files: Optional[Any] = None) -> AsyncIterator[SSEvent]:
        async with self.client.stream(method, url, headers=self._headers(headers), params=params, json=json_data,
            data=data, files=files) as resp:
            resp.raise_for_status()
            async for event in aiter_sse(resp):
                if not event.data: continue
                yield event

    async def stream_sse_json(self, method: str, url: str, *, headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None, json_data: Optional[Any] = None, data: Optional[Any] = None,
        files: Optional[Any] = None,
        done_token: str = "[DONE]") -> AsyncIterator[Dict[str, Any]]:
        async for event in self.stream_sse(method, url, headers=headers, params=params, json_data=json_data,
            data=data, files=files):
            if event.data == done_token: return
            try: raw = json.loads(event.data)
            except json.JSONDecodeError as e: raise ProtocolError(f"Invalid SSE JSON: {e}") from e
            if isinstance(raw, dict): yield raw
            else: raise ProtocolError(f"Expected SSE JSON object, got {type(raw).__name__}")
