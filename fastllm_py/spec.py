"OpenAPI/Discovery spec parsing and provider operation loaders."

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import json
import os
import re

import httpx
import yaml

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
    required_params: List[str] = field(default_factory=list)
    param_types: Dict[str, Any] = field(default_factory=dict)
    param_docs: Dict[str, str] = field(default_factory=dict)
    docs_url: str = ""
    streamable: bool = False


_http_verbs = {"get", "post", "put", "patch", "delete", "options", "head"}
_pat_non_alnum = re.compile(r"[^a-zA-Z0-9]+")
_pat_path_param = re.compile(r"\{([^}]+)\}")
_pat_md_url = re.compile(r"\[[^\]]+\]\((https?://[^)\s]+)\)")
_pat_url = re.compile(r"(https?://[^\s)>\"]+)")
_pat_curl_json_data = re.compile(r"-d\s+'(\{.*\})'", re.DOTALL)
_pat_required_prefix = re.compile(r"^\s*required\b", re.IGNORECASE)
_core_req_names = {"model", "input", "messages", "max_tokens"}


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
    if segs and re.fullmatch(r"v\d+(?:[a-zA-Z0-9._-]*)?", segs[0]):
        segs = segs[1:]
    grp = snake(segs[0]) if segs else "api"
    return grp, snake(op_id or (segs[-1] if segs else "call"))


def _path_params(path: str) -> list[str]:
    "Extract route params from /x/{id} paths."
    return [snake(o.lstrip("+")) for o in _pat_path_param.findall(path)]


def _resolve_ref(ref: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    "Resolve a local #/components/... JSON pointer."
    if not ref.startswith("#/"):
        raise SpecError(f"Only local refs supported, got: {ref}")
    cur: Any = spec
    for part in ref[2:].split("/"):
        if not isinstance(cur, dict) or part not in cur:
            raise SpecError(f"Bad ref: {ref}")
        cur = cur[part]
    if not isinstance(cur, dict):
        raise SpecError(f"Ref does not resolve to object: {ref}")
    return cur


def _resolve_obj(obj: Any, spec: Dict[str, Any]) -> Any:
    "Resolve a `$ref` object and merge local overrides when present."
    if not isinstance(obj, dict) or "$ref" not in obj:
        return obj
    base = _resolve_ref(str(obj["$ref"]), spec)
    if len(obj) == 1:
        return base
    merged = dict(base)
    merged.update({k: v for k, v in obj.items() if k != "$ref"})
    return merged


def _merge_props(*prop_dicts: Dict[str, Any]) -> Dict[str, Any]:
    "Merge property dicts while preserving insertion order."
    out: Dict[str, Any] = {}
    for d in prop_dicts:
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if k not in out:
                out[k] = v
    return out


def _schema_props(schema: Dict[str, Any], spec: Dict[str, Any], _seen: Optional[set[str]] = None) -> Dict[str, Any]:
    "Resolve request schema and return merged properties for composed schemas."
    if not isinstance(schema, dict):
        return {}

    seen = _seen or set()
    schema = _resolve_obj(schema, spec)

    props = dict(schema.get("properties") or {}) if isinstance(schema.get("properties"), dict) else {}

    # Compose `allOf`/`oneOf`/`anyOf` by unioning all child properties.
    for key in ("allOf", "oneOf", "anyOf"):
        for sub in schema.get(key, []) or []:
            if not isinstance(sub, dict):
                continue
            ref = sub.get("$ref")
            if isinstance(ref, str):
                if ref in seen:
                    continue
                child_seen = set(seen)
                child_seen.add(ref)
            else:
                child_seen = set(seen)
            props = _merge_props(props, _schema_props(sub, spec, child_seen))

    # Handle direct ref last if not already merged above.
    if isinstance(schema.get("$ref"), str):
        ref = str(schema["$ref"])
        if ref not in seen:
            child_seen = set(seen)
            child_seen.add(ref)
            props = _merge_props(props, _schema_props(_resolve_ref(ref, spec), spec, child_seen))

    return props


def _clean_desc(v: Any) -> str:
    "Normalize a description string to a compact one-liner."
    if not isinstance(v, str):
        return ""
    return " ".join(v.strip().split())


def _schema_required(schema: Dict[str, Any], spec: Dict[str, Any], _seen: Optional[set[str]] = None) -> set[str]:
    "Resolve required field names from composed schemas."
    if not isinstance(schema, dict):
        return set()
    seen = _seen or set()
    schema = _resolve_obj(schema, spec)
    req = set(schema.get("required") or [])
    for key in ("allOf", "oneOf", "anyOf"):
        for sub in schema.get(key, []) or []:
            if not isinstance(sub, dict):
                continue
            ref = sub.get("$ref")
            if isinstance(ref, str):
                if ref in seen:
                    continue
                child_seen = set(seen)
                child_seen.add(ref)
            else:
                child_seen = set(seen)
            req |= _schema_required(sub, spec, child_seen)
    return req


def _schema_py_type(schema: Dict[str, Any], spec: Dict[str, Any], _seen: Optional[set[str]] = None) -> Any:
    "Best-effort Python type mapping from JSON schema fragments."
    if not isinstance(schema, dict):
        return Any
    seen = _seen or set()
    schema = _resolve_obj(schema, spec)
    t = schema.get("type")
    if t == "string":
        return str
    if t == "integer":
        return int
    if t == "number":
        return float
    if t == "boolean":
        return bool
    if t == "array":
        return list
    if t == "object":
        return dict
    if t == "null":
        return type(None)

    for key in ("oneOf", "anyOf", "allOf"):
        outs: list[Any] = []
        for sub in schema.get(key, []) or []:
            if not isinstance(sub, dict):
                continue
            ref = sub.get("$ref")
            if isinstance(ref, str):
                if ref in seen:
                    continue
                child_seen = set(seen)
                child_seen.add(ref)
            else:
                child_seen = set(seen)
            out = _schema_py_type(sub, spec, child_seen)
            if out is not Any:
                outs.append(out)
        uniq: list[Any] = []
        for o in outs:
            if o not in uniq:
                uniq.append(o)
        if not uniq:
            continue
        non_none = [o for o in uniq if o is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
        if len(non_none) > 1:
            return Any
        return type(None)
    return Any


def _collect_params(op: Dict[str, Any], path_desc: Dict[str, Any], spec: Dict[str, Any]) -> tuple[list[str], list[str], set[str], Dict[str, Any], Dict[str, str]]:
    "Collect route and query params from operation + path level params."
    route: list[str] = []
    query: list[str] = []
    required: set[str] = set()
    param_types: Dict[str, Any] = {}
    param_docs: Dict[str, str] = {}
    params = list(path_desc.get("parameters") or []) + list(op.get("parameters") or [])
    for p in params:
        if not isinstance(p, dict):
            continue
        p = _resolve_obj(p, spec)
        if not isinstance(p, dict):
            continue
        nm = snake(str(p.get("name", "")).lstrip("+"))
        where = p.get("in")
        if not nm:
            continue
        if where == "path" and nm not in route:
            route.append(nm)
            required.add(nm)
        if where == "query" and nm not in query:
            query.append(nm)
            if bool(p.get("required")):
                required.add(nm)
        sch = p.get("schema")
        if isinstance(sch, dict):
            param_types[nm] = _schema_py_type(sch, spec)
        desc = _clean_desc(p.get("description"))
        if desc:
            param_docs[nm] = desc
    return route, query, required, param_types, param_docs


def _body_params(op: Dict[str, Any], spec: Dict[str, Any]) -> tuple[list[str], set[str], Dict[str, Any], Dict[str, str]]:
    "Extract request JSON/body params from requestBody schema."
    rb = _resolve_obj(op.get("requestBody") or {}, spec)
    if not isinstance(rb, dict):
        return [], set(), {}, {}
    content = rb.get("content") or {}
    for ctype in ("application/json", "application/x-www-form-urlencoded", "multipart/form-data"):
        body = content.get(ctype) or {}
        schema = body.get("schema") if isinstance(body, dict) else None
        if not isinstance(schema, dict):
            continue
        props = _schema_props(schema, spec)
        if props:
            names = [snake(k) for k in props]
            name_set = set(names)
            req = {snake(k) for k in _schema_required(schema, spec) if snake(k) in name_set}
            if not req:
                req |= _infer_required_from_examples(op, name_set)
            ptypes = {snake(k): _schema_py_type(v, spec) for k,v in props.items()}
            pdocs = {}
            for k,v in props.items():
                desc = _clean_desc(v.get("description") if isinstance(v, dict) else None)
                if desc:
                    pdocs[snake(k)] = desc
            return names, req, ptypes, pdocs
    return [], set(), {}, {}


def _extract_request_examples(op: Dict[str, Any]) -> list[Dict[str, Any]]:
    "Extract request-body examples from common provider metadata shapes."
    out: list[Dict[str, Any]] = []
    meta = op.get("x-oaiMeta")
    exs = None
    if isinstance(meta, dict):
        exs = meta.get("examples")
        if exs is None:
            exs = meta.get("example")
    if exs is None:
        exs = op.get("x-examples")

    items = exs if isinstance(exs, list) else [exs] if isinstance(exs, dict) else []
    for ex in items:
        if not isinstance(ex, dict):
            continue
        req = ex.get("request")
        if not isinstance(req, dict):
            continue
        body = req.get("body")
        if isinstance(body, dict):
            out.append(body)
            continue
        curl = req.get("curl")
        if isinstance(curl, str):
            m = _pat_curl_json_data.search(curl)
            if m:
                raw = m.group(1)
                try:
                    j = json.loads(raw)
                except Exception:
                    j = None
                if isinstance(j, dict):
                    out.append(j)
    return out


def _infer_required_from_examples(op: Dict[str, Any], allowed: set[str]) -> set[str]:
    "Infer likely-required body params when schema required list is empty."
    bodies = _extract_request_examples(op)
    if not bodies:
        return set()
    key_sets = [set(snake(k) for k in b.keys()) for b in bodies if isinstance(b, dict)]
    key_sets = [ks for ks in key_sets if ks]
    if not key_sets:
        return set()

    common = set.intersection(*key_sets)
    if len(key_sets) > 1:
        return {k for k in common if k in allowed}

    # Single-example fallback: only infer canonical core request fields.
    return {k for k in common if k in allowed and k in _core_req_names}


def _is_bool_schema(schema: Dict[str, Any], spec: Dict[str, Any], _seen: Optional[set[str]] = None) -> bool:
    "Return True when a schema resolves to / includes a boolean shape."
    if not isinstance(schema, dict):
        return False
    seen = _seen or set()
    schema = _resolve_obj(schema, spec)
    t = schema.get("type")
    if t == "boolean":
        return True
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in schema.get(key, []) or []:
            if not isinstance(sub, dict):
                continue
            ref = sub.get("$ref")
            if isinstance(ref, str):
                if ref in seen:
                    continue
                child_seen = set(seen)
                child_seen.add(ref)
            else:
                child_seen = set(seen)
            if _is_bool_schema(sub, spec, child_seen):
                return True
    return False


def _has_stream_param(op: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    "Best-effort check for request body `stream` parameter."
    rb = _resolve_obj(op.get("requestBody") or {}, spec)
    if not isinstance(rb, dict):
        return False
    content = rb.get("content") or {}
    for ctype in ("application/json", "application/x-www-form-urlencoded", "multipart/form-data"):
        body = content.get(ctype) or {}
        schema = body.get("schema") if isinstance(body, dict) else None
        if not isinstance(schema, dict):
            continue
        props = _schema_props(schema, spec)
        if "stream" in props and _is_bool_schema(props["stream"], spec):
            return True
    return False


def _is_streamable(op: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    "Detect stream-capable operations using explicit hints or SSE responses."
    if bool(op.get("x-fastllm-stream")):
        return True
    responses = op.get("responses") or {}
    for resp in responses.values():
        if not isinstance(resp, dict):
            continue
        content = resp.get("content") or {}
        if "text/event-stream" in content:
            return True
    if _has_stream_param(op, spec):
        return True
    return False


def _first_url(text: Any) -> str:
    "Extract first URL from markdown/plain text."
    if not isinstance(text, str) or not text:
        return ""
    m = _pat_md_url.search(text)
    if m:
        return m.group(1).rstrip(".,;")
    m = _pat_url.search(text)
    if m:
        return m.group(1).rstrip(".,;")
    return ""


def _op_docs_url(op: Dict[str, Any], path_desc: Dict[str, Any], spec: Dict[str, Any]) -> str:
    "Best-effort operation docs URL extraction."
    ext = op.get("externalDocs")
    if isinstance(ext, dict):
        url = ext.get("url")
        if isinstance(url, str) and url:
            return url
    xdoc = op.get("x-docs-url") or op.get("documentationLink")
    if isinstance(xdoc, str) and xdoc:
        return xdoc
    if durl := _first_url(op.get("description")):
        return durl

    pext = path_desc.get("externalDocs") if isinstance(path_desc, dict) else None
    if isinstance(pext, dict):
        purl = pext.get("url")
        if isinstance(purl, str) and purl:
            return purl
    if isinstance(path_desc, dict):
        xdoc = path_desc.get("x-docs-url")
        if isinstance(xdoc, str) and xdoc:
            return xdoc

    tags = op.get("tags")
    stags = spec.get("tags")
    if isinstance(tags, list) and isinstance(stags, list):
        wanted = {str(t) for t in tags if isinstance(t, str)}
        for tg in stags:
            if not isinstance(tg, dict):
                continue
            if str(tg.get("name") or "") not in wanted:
                continue
            t_ext = tg.get("externalDocs")
            if isinstance(t_ext, dict):
                t_url = t_ext.get("url")
                if isinstance(t_url, str) and t_url:
                    return t_url
            t_xdoc = tg.get("x-docs-url")
            if isinstance(t_xdoc, str) and t_xdoc:
                return t_xdoc

    meta = op.get("x-oaiMeta")
    if isinstance(meta, dict):
        grp = meta.get("group")
        pth = meta.get("path")
        if isinstance(grp, str) and grp:
            if isinstance(pth, str) and pth:
                return f"https://platform.openai.com/docs/api-reference/{grp}/{pth}"
            return f"https://platform.openai.com/docs/api-reference/{grp}"
    return ""


def spec_to_ops(spec: Dict[str, Any]) -> list[OpSpec]:
    "Convert OpenAPI-like dict to OpSpec list."
    if not isinstance(spec, dict):
        raise SpecError("spec_to_ops expects a dict")
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        raise SpecError("spec missing paths")

    res: list[OpSpec] = []
    for path, path_desc in paths.items():
        if not isinstance(path_desc, dict):
            continue
        for verb, op in path_desc.items():
            if verb.lower() not in _http_verbs:
                continue
            if not isinstance(op, dict):
                continue
            op_id = str(op.get("operationId") or f"{verb}_{path}")
            group, name = _group_name(op_id, path)
            route, query, required, ptypes, pdocs = _collect_params(op, path_desc, spec)
            route = route or _path_params(path)
            required |= set(route)
            body, body_required, body_types, body_docs = _body_params(op, spec)
            required |= body_required
            for k,v in body_types.items():
                ptypes.setdefault(k, v)
            for k,v in body_docs.items():
                pdocs.setdefault(k, v)
            ordered = route + query + body
            res.append(
                OpSpec(
                    group=group,
                    name=name,
                    path=path,
                    verb=verb.upper(),
                    summary=str(op.get("summary") or ""),
                    route_params=route,
                    query_params=query,
                    body_params=body,
                    required_params=[nm for nm in ordered if nm in required],
                    param_types=ptypes,
                    param_docs=pdocs,
                    docs_url=_op_docs_url(op, path_desc, spec),
                    streamable=_is_streamable(op, spec),
                )
            )
    return res


def merge_ops(*xs: Iterable[OpSpec]) -> list[OpSpec]:
    "Flatten multiple OpSpec iterables."
    return [o for x in xs for o in x]


def load_spec_json(text: str) -> Dict[str, Any]:
    "Load OpenAPI spec from JSON text."
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise SpecError(f"Invalid JSON spec: {e}") from e
    if not isinstance(obj, dict):
        raise SpecError(f"Expected JSON object, got {type(obj).__name__}")
    return obj


def load_spec_yaml(text: str) -> Dict[str, Any]:
    "Load OpenAPI spec from YAML text."
    try:
        obj = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise SpecError(f"Invalid YAML spec: {e}") from e
    if not isinstance(obj, dict):
        raise SpecError(f"Expected YAML object, got {type(obj).__name__}")
    return obj


def load_spec_file(path: str) -> Dict[str, Any]:
    "Load OpenAPI/Discovery spec from local JSON or YAML file."
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    suf = p.suffix.lower()
    if suf in (".yaml", ".yml"):
        return load_spec_yaml(txt)
    if suf == ".json":
        return load_spec_json(txt)

    # Extensionless/unknown: try JSON first, then YAML.
    try:
        return load_spec_json(txt)
    except SpecError:
        return load_spec_yaml(txt)


def load_spec_url(url: str, *, timeout: float = 30.0, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    "Load OpenAPI/Discovery spec from URL (JSON or YAML)."
    resp = httpx.get(url, timeout=timeout, headers=headers)
    resp.raise_for_status()
    ctype = (resp.headers.get("content-type") or "").lower()
    if "json" in ctype:
        obj = resp.json()
        if isinstance(obj, dict):
            return obj
    if "yaml" in ctype or "yml" in ctype:
        return load_spec_yaml(resp.text)

    # Best effort for unknown content-type.
    try:
        obj = resp.json()
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        return load_spec_json(resp.text)
    except SpecError:
        return load_spec_yaml(resp.text)


def _specs_dir(specs_dir: Optional[str] = None) -> Path:
    if specs_dir:
        return Path(specs_dir)
    env_dir = os.getenv("FASTLLM_SPECS_DIR")
    if env_dir:
        return Path(env_dir)
    # .../fastllm/fastllm/spec.py -> .../fastllm/specs
    return Path(__file__).resolve().parent.parent / "specs"


def _find_spec_file(candidates: Iterable[str], *, specs_dir: Optional[str] = None) -> Path:
    d = _specs_dir(specs_dir)
    for nm in candidates:
        p = d / nm
        if p.exists():
            return p
    cands = ", ".join(candidates)
    raise SpecError(f"Spec file not found in {d}: tried [{cands}]. Set FASTLLM_SPECS_DIR if needed.")


@lru_cache(maxsize=16)
def _openapi_ops_from_file(path: str) -> tuple[OpSpec, ...]:
    spec = load_spec_file(path)
    return tuple(spec_to_ops(spec))


def openai_ops(*, spec_path: Optional[str] = None, specs_dir: Optional[str] = None) -> list[OpSpec]:
    "OpenAI OpSpec list parsed from local OpenAPI spec file."
    path = Path(spec_path) if spec_path else _find_spec_file(
        ("openai.with-code-samples.yml", "openai.with-code-samples.yaml", "openai.yml", "openai.yaml", "openai.json"),
        specs_dir=specs_dir,
    )
    return list(_openapi_ops_from_file(str(path.resolve())))


def _discovery_schema_props(name_or_schema: Any, schemas: Dict[str, Any], _seen: Optional[set[str]] = None) -> Dict[str, Any]:
    "Return merged schema properties from discovery schemas by name or inline schema."
    seen = _seen or set()

    schema: Dict[str, Any]
    if isinstance(name_or_schema, str):
        if name_or_schema in seen:
            return {}
        seen = set(seen)
        seen.add(name_or_schema)
        schema = schemas.get(name_or_schema) or {}
    elif isinstance(name_or_schema, dict):
        ref = name_or_schema.get("$ref")
        if isinstance(ref, str):
            return _discovery_schema_props(ref, schemas, seen)
        schema = name_or_schema
    else:
        return {}

    props = dict(schema.get("properties") or {}) if isinstance(schema.get("properties"), dict) else {}
    for key in ("allOf", "oneOf", "anyOf"):
        for sub in schema.get(key, []) or []:
            props = _merge_props(props, _discovery_schema_props(sub, schemas, seen))
    return props


def _discovery_required(name_or_schema: Any, schemas: Dict[str, Any], _seen: Optional[set[str]] = None) -> set[str]:
    "Return merged required names from discovery schemas by name or inline schema."
    seen = _seen or set()

    schema: Dict[str, Any]
    if isinstance(name_or_schema, str):
        if name_or_schema in seen:
            return set()
        seen = set(seen)
        seen.add(name_or_schema)
        schema = schemas.get(name_or_schema) or {}
    elif isinstance(name_or_schema, dict):
        ref = name_or_schema.get("$ref")
        if isinstance(ref, str):
            return _discovery_required(ref, schemas, seen)
        schema = name_or_schema
    else:
        return set()

    req = set(schema.get("required") or [])
    # Gemini discovery specs often encode requiredness in field descriptions
    # ("Required. ...") rather than top-level schema.required arrays.
    if not req:
        props = schema.get("properties")
        if isinstance(props, dict):
            for k, v in props.items():
                if not isinstance(v, dict):
                    continue
                desc = v.get("description")
                if isinstance(desc, str) and _pat_required_prefix.match(desc):
                    req.add(k)
    for key in ("allOf", "oneOf", "anyOf"):
        for sub in schema.get(key, []) or []:
            req |= _discovery_required(sub, schemas, seen)
    return req


def _discovery_py_type(schema: Any) -> Any:
    "Best-effort Python type mapping for Google discovery schema fragments."
    if not isinstance(schema, dict):
        return Any
    t = schema.get("type")
    if t == "string":
        return str
    if t == "integer":
        return int
    if t == "number":
        return float
    if t == "boolean":
        return bool
    if t == "array":
        return list
    if t == "object":
        return dict
    if t == "null":
        return type(None)
    for key in ("oneOf", "anyOf", "allOf"):
        outs = []
        for sub in schema.get(key, []) or []:
            out = _discovery_py_type(sub)
            if out is not Any:
                outs.append(out)
        uniq = []
        for o in outs:
            if o not in uniq:
                uniq.append(o)
        if not uniq:
            continue
        non_none = [o for o in uniq if o is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
        if len(non_none) > 1:
            return Any
        return type(None)
    return Any


def _norm_discovery_path(path: str, version: str) -> str:
    p = "/" + str(path or "").lstrip("/")
    if version:
        pre = f"/{version}/"
        if p.startswith(pre):
            p = "/" + p[len(pre):]
    return p


def _walk_discovery_resources(res: Dict[str, Any], stack: list[str], schemas: Dict[str, Any], version: str, out: list[OpSpec]):
    methods = res.get("methods") or {}
    for mname, m in methods.items():
        if not isinstance(m, dict):
            continue
        verb = str(m.get("httpMethod") or "").upper()
        if verb.lower() not in _http_verbs:
            continue

        group = snake("_".join(stack)) if stack else "api"
        name = snake(mname)
        path = _norm_discovery_path(str(m.get("path") or ""), version)
        summary = str(m.get("description") or "")

        route_params: list[str] = []
        query_params: list[str] = []
        required: set[str] = set()
        param_types: Dict[str, Any] = {}
        param_docs: Dict[str, str] = {}
        for pname, pd in (m.get("parameters") or {}).items():
            if not isinstance(pd, dict):
                continue
            nm = snake(str(pname).lstrip("+"))
            loc = str(pd.get("location") or "")
            if loc == "path" and nm not in route_params:
                route_params.append(nm)
                required.add(nm)
            elif loc == "query" and nm not in query_params:
                query_params.append(nm)
                if bool(pd.get("required")):
                    required.add(nm)
            if "type" in pd:
                param_types[nm] = _discovery_py_type({"type": pd.get("type")})
            elif isinstance(pd.get("schema"), dict):
                param_types[nm] = _discovery_py_type(pd["schema"])
            desc = _clean_desc(pd.get("description"))
            if desc:
                param_docs[nm] = desc

        body_params: list[str] = []
        req = m.get("request") or {}
        req_ref = req.get("$ref") if isinstance(req, dict) else None
        if isinstance(req_ref, str):
            props = _discovery_schema_props(req_ref, schemas)
            body_params = [snake(k) for k in props.keys()]
            name_set = set(body_params)
            required |= {snake(k) for k in _discovery_required(req_ref, schemas) if snake(k) in name_set}
            for k,v in props.items():
                sk = snake(k)
                param_types.setdefault(sk, _discovery_py_type(v))
                desc = _clean_desc(v.get("description") if isinstance(v, dict) else None)
                if desc:
                    param_docs.setdefault(sk, desc)

        streamable = name.startswith("stream_")
        ordered = (route_params or _path_params(path)) + query_params + body_params
        out.append(
            OpSpec(
                group=group,
                name=name,
                path=path,
                verb=verb,
                summary=summary,
                route_params=route_params or _path_params(path),
                query_params=query_params,
                body_params=body_params,
                required_params=[nm for nm in ordered if nm in required],
                param_types=param_types,
                param_docs=param_docs,
                docs_url=str(m.get("documentationLink") or ""),
                streamable=streamable,
            )
        )

    for rname, child in (res.get("resources") or {}).items():
        if isinstance(child, dict):
            _walk_discovery_resources(child, stack + [str(rname)], schemas, version, out)


@lru_cache(maxsize=8)
def _gemini_ops_from_file(path: str) -> tuple[OpSpec, ...]:
    spec = load_spec_file(path)
    if not isinstance(spec, dict):
        raise SpecError("Gemini discovery spec must be an object")

    schemas = spec.get("schemas") or {}
    version = str(spec.get("version") or "")
    resources = spec.get("resources") or {}

    out: list[OpSpec] = []
    for rname, child in resources.items():
        if isinstance(child, dict):
            _walk_discovery_resources(child, [str(rname)], schemas, version, out)

    return tuple(out)


def gemini_ops(*, spec_path: Optional[str] = None, specs_dir: Optional[str] = None) -> list[OpSpec]:
    "Gemini OpSpec list parsed from local discovery JSON spec file."
    path = Path(spec_path) if spec_path else _find_spec_file(("gemini.json",), specs_dir=specs_dir)
    return list(_gemini_ops_from_file(str(path.resolve())))


def anthropic_ops(*, spec_path: Optional[str] = None, specs_dir: Optional[str] = None) -> list[OpSpec]:
    "Anthropic OpSpec list parsed from local OpenAPI spec file."
    path = Path(spec_path) if spec_path else _find_spec_file(("anthropic.yml", "anthropic.yaml", "anthropic.json"), specs_dir=specs_dir)
    return list(_openapi_ops_from_file(str(path.resolve())))
