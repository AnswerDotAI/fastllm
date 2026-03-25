"OpenAPI-like spec parsing."

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import json
import re

import httpx

from .errors import SpecError


@dataclass(frozen=True)
class OpSpec:
    "Operation metadata used by the dynamic client layer."
    group: str
    name: str
    path: str
    verb: str
    summary: str = ""
    route_params: List[str] = field(default_factory=list)
    query_params: List[str] = field(default_factory=list)
    body_params: List[str] = field(default_factory=list)
    streamable: bool = False


_http_verbs = {"get", "post", "put", "patch", "delete", "options", "head"}
_pat_non_alnum = re.compile(r"[^a-zA-Z0-9]+")
_pat_path_param = re.compile(r"\{([^}]+)\}")


def snake(s: str) -> str:
    "Convert an identifier-ish string to snake_case."
    s = _pat_non_alnum.sub("_", s).strip("_")
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower().strip("_")


def _group_name(op_id: str, path: str) -> tuple[str, str]:
    "Infer group + name from operationId with path fallback."
    if "/" in op_id:
        grp, nm = op_id.split("/", 1)
        return snake(grp), snake(nm)
    if "." in op_id:
        grp, nm = op_id.split(".", 1)
        return snake(grp), snake(nm)
    segs = [s for s in path.strip("/").split("/") if s and not s.startswith("{")]
    grp = snake(segs[0]) if segs else "api"
    return grp, snake(op_id or (segs[-1] if segs else "call"))


def _path_params(path: str) -> list[str]:
    "Extract route params from /x/{id} paths."
    return [snake(o.lstrip("+")) for o in _pat_path_param.findall(path)]


def _collect_params(op: Dict[str, Any], path_desc: Dict[str, Any]) -> tuple[list[str], list[str]]:
    "Collect route and query params from operation + path level params."
    route, query = [], []
    params = list(path_desc.get("parameters") or []) + list(op.get("parameters") or [])
    for p in params:
        if not isinstance(p, dict): continue
        nm, where = snake(str(p.get("name", "")).lstrip("+")), p.get("in")
        if not nm: continue
        if where == "path" and nm not in route: route.append(nm)
        if where == "query" and nm not in query: query.append(nm)
    return route, query


def _resolve_ref(ref: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    "Resolve a local #/components/... JSON pointer."
    if not ref.startswith("#/"): raise SpecError(f"Only local refs supported, got: {ref}")
    cur = spec
    for part in ref[2:].split("/"):
        if not isinstance(cur, dict) or part not in cur: raise SpecError(f"Bad ref: {ref}")
        cur = cur[part]
    if not isinstance(cur, dict): raise SpecError(f"Ref does not resolve to object: {ref}")
    return cur


def _schema_props(schema: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    "Resolve request schema and return properties dict when possible."
    if "$ref" in schema: return _schema_props(_resolve_ref(schema["$ref"], spec), spec)
    if "properties" in schema and isinstance(schema["properties"], dict): return schema["properties"]
    for key in ("oneOf", "anyOf", "allOf"):
        for sub in schema.get(key, []) or []:
            if not isinstance(sub, dict): continue
            props = _schema_props(sub, spec)
            if props: return props
    return {}


def _body_params(op: Dict[str, Any], spec: Dict[str, Any]) -> list[str]:
    "Extract request JSON/body params from requestBody schema."
    rb = op.get("requestBody") or {}
    if "$ref" in rb: rb = _resolve_ref(rb["$ref"], spec)
    if not isinstance(rb, dict): return []
    content = rb.get("content") or {}
    for ctype in ("application/json", "application/x-www-form-urlencoded", "multipart/form-data"):
        body = content.get(ctype) or {}
        schema = body.get("schema") if isinstance(body, dict) else None
        if not isinstance(schema, dict): continue
        props = _schema_props(schema, spec)
        if props: return [snake(k) for k in props]
    return []


def _is_streamable(op: Dict[str, Any]) -> bool:
    "Detect stream-capable operations using explicit hints or SSE responses."
    if bool(op.get("x-fastllm-stream")): return True
    responses = op.get("responses") or {}
    for resp in responses.values():
        if not isinstance(resp, dict): continue
        content = resp.get("content") or {}
        if "text/event-stream" in content: return True
    return False


def spec_to_ops(spec: Dict[str, Any]) -> list[OpSpec]:
    "Convert OpenAPI-like dict to OpSpec list."
    if not isinstance(spec, dict): raise SpecError("spec_to_ops expects a dict")
    paths = spec.get("paths")
    if not isinstance(paths, dict): raise SpecError("spec missing paths")
    res = []
    for path, path_desc in paths.items():
        if not isinstance(path_desc, dict): continue
        for verb, op in path_desc.items():
            if verb.lower() not in _http_verbs: continue
            if not isinstance(op, dict): continue
            op_id = str(op.get("operationId") or f"{verb}_{path}")
            group, name = _group_name(op_id, path)
            route, query = _collect_params(op, path_desc)
            route = route or _path_params(path)
            body = _body_params(op, spec)
            res.append(OpSpec(
                group=group,
                name=name,
                path=path,
                verb=verb.upper(),
                summary=str(op.get("summary") or ""),
                route_params=route,
                query_params=query,
                body_params=body,
                streamable=_is_streamable(op)))
    return res


def merge_ops(*xs: List[OpSpec]) -> list[OpSpec]:
    "Flatten multiple OpSpec lists."
    return [o for x in xs for o in x]


def load_spec_json(text: str) -> Dict[str, Any]:
    "Load OpenAPI spec from JSON text."
    try: obj = json.loads(text)
    except json.JSONDecodeError as e: raise SpecError(f"Invalid JSON spec: {e}") from e
    if not isinstance(obj, dict): raise SpecError(f"Expected JSON object, got {type(obj).__name__}")
    return obj


def load_spec_file(path: str) -> Dict[str, Any]:
    "Load OpenAPI spec from a local JSON file."
    with open(path, "r", encoding="utf-8") as f: return load_spec_json(f.read())


def load_spec_url(url: str, *, timeout: float = 30.0, headers: Dict[str, str] = None) -> Dict[str, Any]:
    "Load OpenAPI spec from URL."
    resp = httpx.get(url, timeout=timeout, headers=headers)
    resp.raise_for_status()
    if isinstance(resp.json(), dict): return resp.json()
    return load_spec_json(resp.text)
