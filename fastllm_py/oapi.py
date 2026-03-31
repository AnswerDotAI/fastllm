"Dynamic OpenAPI operation client (async)."

from __future__ import annotations

from collections import defaultdict
from inspect import Parameter, Signature, signature
from typing import Any, AsyncIterator, Dict, Iterable, Optional
import re
from urllib.parse import quote, urljoin, urlparse

import httpx

from .errors import APIError, UnsupportedCapabilityError, api_error_from_http
from .spec import OpSpec, spec_to_ops
from .transport import AsyncTransport


def _type_name(tp: Any) -> str:
    if tp is Any:
        return "Any"
    if hasattr(tp, "__name__"):
        return str(tp.__name__)
    return str(tp).replace("typing.", "")


def _mk_param(name: str, required: bool = False, annotation: Any = Any):
    "Create a function signature parameter."
    ann = annotation if annotation is not None else Any
    if required:
        return Parameter(name, kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=ann)
    return Parameter(name, kind=Parameter.POSITIONAL_OR_KEYWORD, default=None, annotation=ann)


def _mk_kw_param(name: str, required: bool = False, annotation: Any = Any):
    "Create a keyword-only signature parameter."
    ann = annotation if annotation is not None else Any
    if required:
        return Parameter(name, kind=Parameter.KEYWORD_ONLY, annotation=ann)
    return Parameter(name, kind=Parameter.KEYWORD_ONLY, default=None, annotation=ann)


def _mk_sig(route_params, query_params, body_params, required_params=None, param_types=None):
    "Create a compact operation signature."
    req = set(required_params or [])
    ptypes = dict(param_types or {})
    seen, params = set(), []
    for nm in route_params:
        if nm in seen: continue
        params.append(_mk_param(nm, required=True, annotation=ptypes.get(nm, Any)))
        seen.add(nm)
    for nm in list(query_params) + list(body_params):
        if nm in seen: continue
        # Non-route params are keyword-only so required fields can safely follow optional ones.
        params.append(_mk_kw_param(nm, required=(nm in req), annotation=ptypes.get(nm, Any)))
        seen.add(nm)
    return Signature(params)


def _safe_sig_text(obj: Any) -> str:
    "Render signature text without raising."
    try:
        return str(signature(obj))
    except Exception:
        return "(...)"


def _param_names(route_params, query_params, body_params) -> list[str]:
    "Ordered unique parameter names."
    out = []
    for nm in list(route_params) + list(query_params) + list(body_params):
        if nm not in out:
            out.append(nm)
    return out


def _params_text(route_params, query_params, body_params) -> str:
    "Ghapi-like compact params text."
    ns = _param_names(route_params, query_params, body_params)
    return f"({', '.join(ns)})"


def _op_summary(op: "_Op") -> str:
    s = str(op.summary or f"{op.verb} {op.path}")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        s = f"{op.verb} {op.path}"
    if op.docs_url:
        p = urlparse(op.docs_url)
        if p.scheme and p.netloc:
            base = f"{p.scheme}://{p.netloc}"

            def _repl(m):
                href = m.group(1).strip()
                if href.startswith(("http://", "https://", "mailto:", "#")):
                    return f"]({href})"
                if href.startswith(("/", "./", "../")):
                    return f"]({urljoin(base, href)})"
                return f"]({href})"

            s = re.sub(r"\]\(([^)]+)\)", _repl, s)
    return s


def _op_line(op: "_Op", *, markdown: bool = False) -> str:
    head = f"{op.group}.{op.name}"
    if markdown and op.docs_url:
        head = f"[{head}]({op.docs_url})"
    s = f"{head}{_params_text(op.route_params, op.query_params, op.body_params)}"
    summ = _op_summary(op)
    return f"{s}: *{summ}*" if markdown else f"{s}: {summ}"


def _mk_doc(spec: OpSpec) -> str:
    "Render operation docstring with summary, docs URL, and parameter hints."
    lines = [spec.summary or f"{spec.verb} {spec.path}"]
    if spec.docs_url:
        lines += ["", f"Docs: {spec.docs_url}"]

    req = set(spec.required_params or [])
    ordered = []
    for nm in list(spec.route_params) + list(spec.query_params) + list(spec.body_params):
        if nm not in ordered:
            ordered.append(nm)
    if req:
        ordered_req = [nm for nm in ordered if nm in req]
        lines += ["", f"Required: {', '.join(ordered_req)}"]
    if ordered:
        lines += ["", "Parameters:"]
        for nm in ordered:
            typ = _type_name((spec.param_types or {}).get(nm, Any))
            req_s = "required" if nm in req else "optional"
            desc = (spec.param_docs or {}).get(nm, "")
            if desc:
                lines.append(f"- {nm} ({typ}, {req_s}): {desc}")
            else:
                lines.append(f"- {nm} ({typ}, {req_s})")
    return "\n".join(lines)


class _Op:
    __slots__ = "group name path verb summary route_params query_params body_params required_params param_types param_docs docs_url streamable client __doc__".split()

    def __init__(self, spec: OpSpec, client: "OpenAPIClient"):
        self.group,self.name,self.path,self.verb = spec.group,spec.name,spec.path,spec.verb
        self.summary,self.route_params,self.query_params = spec.summary,spec.route_params,spec.query_params
        self.body_params,self.required_params = spec.body_params,spec.required_params
        self.param_types,self.param_docs = spec.param_types,spec.param_docs
        self.docs_url = spec.docs_url
        self.streamable,self.client = spec.streamable,client
        self.__doc__ = _mk_doc(spec)

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
        endpoint = kwargs.pop("_endpoint", "") or ""
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
        return stream, raw, endpoint, headers, route, query, body, data, files

    @staticmethod
    def _infer_model(route: Optional[dict], query: Optional[dict], body: Optional[dict]) -> str:
        "Best-effort model extraction for structured API errors."
        for src in (route, body, query):
            if not isinstance(src, dict):
                continue
            mv = src.get("model")
            if isinstance(mv, str) and mv:
                return mv
        return ""

    def _endpoint_name(self, endpoint: str = "") -> str:
        if endpoint:
            return endpoint
        if self.group and self.name:
            return f"{self.group}.{self.name}"
        return f"{self.verb} {self.path}"

    def _raise_with_context(self, exc: Exception, *, endpoint: str, route: Optional[dict], query: Optional[dict], body: Optional[dict]):
        "Raise APIError with operation context for dynamic op calls."
        provider = getattr(self.client, "provider", "") or ""
        model = self._infer_model(route, query, body)
        ep = self._endpoint_name(endpoint)
        if isinstance(exc, APIError):
            raise exc.with_context(provider=provider, model=model, endpoint=ep) from exc
        if isinstance(exc, httpx.HTTPStatusError):
            raise api_error_from_http(exc, provider=provider, model=model, endpoint=ep) from exc
        raise exc

    async def _arequest(self, *, raw, endpoint, headers, route, query, body, data, files):
        "Run standard JSON request for this operation."
        try:
            return await self.client.call(self.path, self.verb, headers=headers, route=route, query=query, body=body,
                data=data, files=files, raw=raw)
        except Exception as e:
            self._raise_with_context(e, endpoint=endpoint, route=route, query=query, body=body)

    async def _astream(self, *, endpoint, headers, route, query, body, data, files):
        "Run SSE stream request for this operation."
        try:
            async for ev in self.client.stream(self.path, self.verb, headers=headers, route=route, query=query, body=body,
                data=data, files=files):
                yield ev
        except Exception as e:
            self._raise_with_context(e, endpoint=endpoint, route=route, query=query, body=body)

    def __call__(self, *args, **kwargs):
        "Return a coroutine (JSON) or async generator (SSE)."
        kwargs = self._bind(args, kwargs)
        stream, raw, endpoint, headers, route, query, body, data, files = self._split(kwargs)
        if stream:
            if not self.streamable: raise UnsupportedCapabilityError(f"Operation does not support stream: {self.group}.{self.name}")
            return self._astream(endpoint=endpoint, headers=headers, route=route, query=query, body=body, data=data, files=files)
        return self._arequest(raw=raw, endpoint=endpoint, headers=headers, route=route, query=query, body=body, data=data, files=files)

    def __str__(self): return _op_line(self)
    def __repr__(self): return _op_line(self)
    def _repr_markdown_(self): return _op_line(self, markdown=True)

    @property
    def __signature__(self): return _mk_sig(self.route_params, self.query_params, self.body_params, self.required_params, self.param_types)
    __call__.__signature__ = __signature__


class _OpGroup:
    "Simple namespace for grouped operations."
    def __init__(self, name: str, ops: Iterable[_Op]):
        self.name,self.ops = name,list(ops)
        for op in self.ops: setattr(self, op.name, op)

    def __str__(self):
        return "\n".join(_op_line(o) for o in self.ops)

    def __repr__(self): return str(self)

    def _repr_markdown_(self):
        return "\n".join(f"- {_op_line(o, markdown=True)}" for o in self.ops)


class OpenAPIClient:
    "Async client built from OpenAPI operation metadata."
    def __init__(self, base_url: str, ops: list[OpSpec], *, headers: Optional[Dict[str, str]] = None,
        timeout: float = 60.0, transport: Optional[AsyncTransport] = None, provider: str = ""):
        self.base_url = base_url.rstrip("/")
        self.provider = provider or ""
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
    transport: Optional[AsyncTransport] = None, provider: str = ""):
    "Build an OpenAPIClient directly from an OpenAPI-like spec dict."
    return OpenAPIClient(base_url=base_url, headers=headers, timeout=timeout, transport=transport, provider=provider, ops=spec_to_ops(spec))
