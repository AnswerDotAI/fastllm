"Dynamic OpenAPI operation client (async)."

from __future__ import annotations

from collections import defaultdict
from inspect import Parameter, Signature, signature
from typing import Any, AsyncIterator, Dict, Iterable, Optional
import re
from urllib.parse import quote

from .errors import UnsupportedCapabilityError
from .spec import OpSpec, spec_to_ops
from .transport import AsyncTransport


def _mk_param(name: str, required: bool = False):
    "Create a function signature parameter."
    if required: return Parameter(name, kind=Parameter.POSITIONAL_OR_KEYWORD)
    return Parameter(name, kind=Parameter.POSITIONAL_OR_KEYWORD, default=None)


def _mk_sig(route_params, query_params, body_params):
    "Create a compact operation signature."
    seen, params = set(), []
    for nm in route_params:
        if nm in seen: continue
        params.append(_mk_param(nm, required=True))
        seen.add(nm)
    for nm in list(query_params) + list(body_params):
        if nm in seen: continue
        params.append(_mk_param(nm, required=False))
        seen.add(nm)
    return Signature(params)


class _Op:
    __slots__ = "group name path verb summary route_params query_params body_params streamable client __doc__".split()

    def __init__(self, spec: OpSpec, client: "OpenAPIClient"):
        self.group,self.name,self.path,self.verb = spec.group,spec.name,spec.path,spec.verb
        self.summary,self.route_params,self.query_params = spec.summary,spec.route_params,spec.query_params
        self.body_params,self.streamable,self.client = spec.body_params,spec.streamable,client
        self.__doc__ = self.summary

    def _bind(self, args, kwargs):
        "Bind positional args to unnamed route/query/body params."
        kwargs = dict(kwargs)
        flds = [o for o in self.route_params + self.query_params + self.body_params if o not in kwargs]
        for a,b in zip(args, flds): kwargs[b] = a
        return kwargs

    def _split(self, kwargs):
        "Split kwargs into route/query/body + control kwargs."
        kwargs = dict(kwargs)
        stream = bool(kwargs.pop("_stream", False))
        raw = bool(kwargs.pop("_raw", False))
        headers = kwargs.pop("_headers", None) or {}
        extra_query = kwargs.pop("_query", None) or {}
        has_explicit_body = "_body" in kwargs
        extra_body = kwargs.pop("_body", None) or {}
        data = kwargs.pop("_data", None)
        files = kwargs.pop("_files", None)

        route = {k: kwargs.pop(k) for k in self.route_params if k in kwargs and kwargs[k] is not None}
        query = {k: kwargs.pop(k) for k in self.query_params if k in kwargs and kwargs[k] is not None}
        body = {k: kwargs.pop(k) for k in self.body_params if k in kwargs and kwargs[k] is not None}
        if self.verb in ("GET", "DELETE", "HEAD", "OPTIONS") and not self.body_params and data is None and files is None:
            query.update(kwargs)
        else:
            body.update(kwargs)
        query.update(extra_query)
        body.update(extra_body)
        if self.verb in ("GET", "DELETE", "HEAD", "OPTIONS") and not body and data is None and files is None and not has_explicit_body:
            body = None
        return stream, raw, headers, route, query, body, data, files

    async def _arequest(self, *, raw, headers, route, query, body, data, files):
        "Run standard JSON request for this operation."
        return await self.client.call(self.path, self.verb, headers=headers, route=route, query=query, body=body,
            data=data, files=files, raw=raw)

    async def _astream(self, *, headers, route, query, body, data, files):
        "Run SSE stream request for this operation."
        async for ev in self.client.stream(self.path, self.verb, headers=headers, route=route, query=query, body=body,
            data=data, files=files):
            yield ev

    def __call__(self, *args, **kwargs):
        "Return a coroutine (JSON) or async generator (SSE)."
        kwargs = self._bind(args, kwargs)
        stream, raw, headers, route, query, body, data, files = self._split(kwargs)
        if stream:
            if not self.streamable: raise UnsupportedCapabilityError(f"Operation does not support stream: {self.group}.{self.name}")
            return self._astream(headers=headers, route=route, query=query, body=body, data=data, files=files)
        return self._arequest(raw=raw, headers=headers, route=route, query=query, body=body, data=data, files=files)

    def __str__(self): return f"{self.group}.{self.name}{signature(self)}"
    @property
    def __signature__(self): return _mk_sig(self.route_params, self.query_params, self.body_params)
    __call__.__signature__ = __signature__


class _OpGroup:
    "Simple namespace for grouped operations."
    def __init__(self, name: str, ops: Iterable[_Op]):
        self.name,self.ops = name,list(ops)
        for op in self.ops: setattr(self, op.name, op)

    def __str__(self): return "\n".join(str(o) for o in self.ops)


class OpenAPIClient:
    "Async client built from OpenAPI operation metadata."
    def __init__(self, base_url: str, ops: list[OpSpec], *, headers: Optional[Dict[str, str]] = None,
        timeout: float = 60.0, transport: Optional[AsyncTransport] = None):
        self.base_url = base_url.rstrip("/")
        self.transport = transport or AsyncTransport(timeout=timeout, base_headers=headers)
        self.ops = [_Op(o, self) for o in ops]
        self.func_dict = {f"{o.path}:{o.verb.upper()}": o for o in self.ops}
        by_group = defaultdict(list)
        for op in self.ops: by_group[op.group].append(op)
        self.groups = {k: _OpGroup(k, v) for k,v in by_group.items()}

    async def aclose(self):
        "Close underlying transport resources."
        await self.transport.aclose()

    def _url(self, path: str) -> str:
        "Build absolute URL from path."
        if path.startswith("http://") or path.startswith("https://"): return path
        return f"{self.base_url}{path}"

    def _path(self, path: str, route: Optional[Dict[str, Any]] = None) -> str:
        "Apply route params with URL encoding."
        if not route: return path
        for k,v in route.items():
            s = str(v)
            safe = "/" if "/" in s else ""
            path = path.replace("{" + k + "}", quote(s, safe=safe))
            path = path.replace("{+" + k + "}", quote(str(v), safe="/"))
        path = re.sub(r"\{\+([^}]+)\}", lambda m: "{" + m.group(1) + "}", path)
        return path

    async def call(self, path: str, verb: str, *, headers: Optional[Dict[str, str]] = None,
        route: Optional[Dict[str, Any]] = None, query: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None, data: Optional[Any] = None, files: Optional[Any] = None,
        raw: bool = False) -> Any:
        "Execute an HTTP request and decode response by content type."
        p = self._path(path, route)
        return await self.transport.request(verb, self._url(p), headers=headers, params=query, json_data=body,
            data=data, files=files, raw=raw)

    async def stream(self, path: str, verb: str, *, headers: Optional[Dict[str, str]] = None,
        route: Optional[Dict[str, Any]] = None, query: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None, data: Optional[Any] = None, files: Optional[Any] = None) -> AsyncIterator[Dict[str, Any]]:
        "Execute an SSE request yielding parsed JSON events."
        p = self._path(path, route)
        async for ev in self.transport.stream_sse_json(verb, self._url(p), headers=headers, params=query, json_data=body,
            data=data, files=files):
            yield ev

    def __dir__(self): return super().__dir__() + list(self.groups)
    def __getattr__(self, k):
        if "groups" in vars(self) and k in self.groups: return self.groups[k]
        raise AttributeError(k)

    def __getitem__(self, k):
        "Lookup operation by (path, verb) tuple or path (GET)."
        a,b = k if isinstance(k, tuple) else (k, "GET")
        return self.func_dict[f"{a}:{b.upper()}"]


def client_from_spec(base_url: str, spec: dict, *, headers: dict = None, timeout: float = 60.0,
    transport: Optional[AsyncTransport] = None):
    "Build an OpenAPIClient directly from an OpenAPI-like spec dict."
    return OpenAPIClient(base_url=base_url, headers=headers, timeout=timeout, transport=transport, ops=spec_to_ops(spec))
