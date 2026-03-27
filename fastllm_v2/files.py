"Canonical provider-agnostic Files API helpers."

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import mimetypes
from urllib.parse import urlparse

import httpx

from .errors import APIError, UnsupportedCapabilityError, api_error_from_http
from .highlevel import infer_provider, mk_auto_client
from .types import Part


@dataclass(frozen=True)
class FileRef:
    "Provider-agnostic file handle + metadata."
    id: str
    provider: str
    name: str = ""
    filename: str = ""
    mime_type: str = ""
    uri: str = ""
    size_bytes: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def _pick_first(d: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _to_int(v: Any) -> Optional[int]:
    try:
        if v in (None, ""):
            return None
        return int(v)
    except Exception:
        return None


def _file_obj(raw: Dict[str, Any]) -> Dict[str, Any]:
    "Unwrap common `{file: {...}}` envelopes."
    if isinstance(raw.get("file"), dict):
        return raw["file"]
    return raw


def _to_file_ref(raw: Dict[str, Any], provider: str) -> FileRef:
    obj = _file_obj(raw)
    fid = str(_pick_first(obj, "id", "file_id", "name") or "")
    nm = str(_pick_first(obj, "name", "display_name", "displayName") or "")
    fn = str(_pick_first(obj, "filename", "file_name", "display_name", "displayName") or "")
    mt = str(_pick_first(obj, "mime_type", "mimeType") or "")
    uri = str(_pick_first(obj, "uri", "file_uri", "fileUri") or "")
    sz = _to_int(_pick_first(obj, "bytes", "size_bytes", "sizeBytes"))
    return FileRef(id=fid, provider=provider, name=nm, filename=fn, mime_type=mt, uri=uri, size_bytes=sz, raw=raw)


def _list_to_file_refs(raw: Dict[str, Any], provider: str) -> List[FileRef]:
    items = raw.get("data")
    if not isinstance(items, list):
        items = raw.get("files")
    if not isinstance(items, list):
        items = []
    return [_to_file_ref(o if isinstance(o, dict) else {"raw": o}, provider) for o in items]


def _root_url(base_url: str) -> str:
    u = urlparse(base_url or "")
    if not (u.scheme and u.netloc):
        return ""
    return f"{u.scheme}://{u.netloc}"


def _guess_mime(filename: str, mime_type: str) -> str:
    if isinstance(mime_type, str) and mime_type:
        return mime_type
    if isinstance(filename, str) and filename:
        g = mimetypes.guess_type(filename)[0]
        if isinstance(g, str) and g:
            return g
    return "application/octet-stream"


def _read_file_bytes(*, file: Any = None, filename: str = "", mime_type: str = "") -> Tuple[bytes, str, str]:
    "Normalize file/path/bytes-like input to `(bytes, filename, mime_type)`."
    data: bytes
    fn = filename or ""
    if file is None:
        raise ValueError("`file` is required for binary upload. Pass bytes, path, or a binary file object.")

    if isinstance(file, (str, Path)):
        p = Path(file)
        data = p.read_bytes()
        if not fn:
            fn = p.name
    elif isinstance(file, (bytes, bytearray)):
        data = bytes(file)
    elif hasattr(file, "read"):
        data = file.read()
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("File-like object `.read()` must return bytes.")
        data = bytes(data)
        if not fn:
            fn = getattr(file, "name", "") or ""
            if isinstance(fn, str) and fn and "/" in fn:
                fn = Path(fn).name
    else:
        raise TypeError("Unsupported `file` type. Use bytes, path-like, or binary file object.")

    if not fn:
        fn = "upload.bin"
    mt = _guess_mime(fn, mime_type)
    return data, fn, mt


def to_input_file_part(f: FileRef | Dict[str, Any] | str, *, provider: str = "") -> Part:
    "Create a canonical `Part(type='input_file', ...)` from a file reference."
    if isinstance(f, FileRef):
        ref = f
    elif isinstance(f, dict):
        p = provider or str(f.get("provider") or "")
        ref = FileRef(
            id=str(f.get("id") or f.get("name") or ""),
            provider=p,
            name=str(f.get("name") or ""),
            filename=str(f.get("filename") or f.get("display_name") or ""),
            mime_type=str(f.get("mime_type") or f.get("mimeType") or ""),
            uri=str(f.get("uri") or f.get("file_uri") or f.get("fileUri") or ""),
            raw=dict(f),
        )
    else:
        ref = FileRef(id=str(f), provider=provider or "")

    prov = (provider or ref.provider or "").strip().lower()
    if prov == "anthropic":
        if not ref.id:
            raise ValueError("Anthropic file part requires `FileRef.id`.")
        return Part(
            type="input_file",
            data={
                "file_id": ref.id,
                "anthropic": {"type": "document", "source": {"type": "file", "file_id": ref.id}},
            },
        )
    if prov == "gemini":
        uri = ref.uri or ref.id
        if not uri:
            raise ValueError("Gemini file part requires `FileRef.uri` or `FileRef.id`.")
        mt = ref.mime_type or "application/octet-stream"
        return Part(
            type="input_file",
            data={
                "file_url": uri,
                "mimeType": mt,
                "gemini": {"fileData": {"fileUri": uri, "mimeType": mt}},
            },
        )
    # OpenAI / openai_chat / openai_compat default.
    d: Dict[str, Any] = {}
    if ref.id:
        d["file_id"] = ref.id
    elif ref.uri:
        d["file_url"] = ref.uri
    if ref.filename:
        d["filename"] = ref.filename
    if ref.mime_type:
        d["mimeType"] = ref.mime_type
    return Part(type="input_file", data=d or None)


async def _openai_create(c, *, file: Any, filename: str, mime_type: str, purpose: str, headers: Optional[Dict[str, str]] = None):
    b, fn, mt = _read_file_bytes(file=file, filename=filename, mime_type=mime_type)
    return await c.api.call("/files", "POST", headers=headers, body=None, data={"purpose": purpose}, files={"file": (fn, b, mt)})


async def _anthropic_create(c, *, file: Any, filename: str, mime_type: str, purpose: str, headers: Optional[Dict[str, str]] = None):
    b, fn, mt = _read_file_bytes(file=file, filename=filename, mime_type=mime_type)
    h = dict(headers or {})
    h.setdefault("anthropic-beta", "files-api-2025-04-14")
    return await c.api.call("/v1/files", "POST", headers=h, body=None, data={"purpose": purpose}, files={"file": (fn, b, mt)})


async def _gemini_upload_resumable(c, *, file: Any, filename: str, mime_type: str, headers: Optional[Dict[str, str]] = None):
    b, fn, mt = _read_file_bytes(file=file, filename=filename, mime_type=mime_type)
    root = _root_url(c.api.base_url)
    if not root:
        raise ValueError("Could not infer Gemini API root URL for resumable upload.")
    start_url = f"{root}/upload/v1beta/files"

    h1 = {
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(len(b)),
        "X-Goog-Upload-Header-Content-Type": mt,
        "Content-Type": "application/json",
        **(headers or {}),
    }
    # Start resumable session.
    r = await c.api.transport.request("POST", start_url, headers=h1, json_data={"file": {"display_name": fn}}, raw=True)
    upload_url = r.headers.get("x-goog-upload-url")
    if not upload_url:
        raise UnsupportedCapabilityError("Gemini upload session did not return `x-goog-upload-url`.")

    h2 = {
        "Content-Length": str(len(b)),
        "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize",
        **(headers or {}),
    }
    return await c.api.transport.request("POST", upload_url, headers=h2, data=b)


async def afile_create(
    model: str,
    *,
    file: Any = None,
    filename: str = "",
    mime_type: str = "",
    purpose: str = "user_data",
    provider: str = "",
    api_key: str = "",
    base_url: str = "",
    timeout: float = 60.0,
    headers: Optional[Dict[str, str]] = None,
) -> FileRef:
    "Create/upload a file and return canonical `FileRef`."
    c = mk_auto_client(model=model, provider=provider, api_key=api_key, base_url=base_url, timeout=timeout)
    fam = infer_provider(model, provider=provider, base_url=base_url)
    try:
        if fam in ("openai", "openai_compat"):
            raw = await _openai_create(c, file=file, filename=filename, mime_type=mime_type, purpose=purpose, headers=headers)
            return _to_file_ref(raw if isinstance(raw, dict) else {"raw": raw}, c.config.provider or fam)
        if fam == "anthropic":
            raw = await _anthropic_create(c, file=file, filename=filename, mime_type=mime_type, purpose=purpose, headers=headers)
            return _to_file_ref(raw if isinstance(raw, dict) else {"raw": raw}, "anthropic")
        if fam == "gemini":
            raw = await _gemini_upload_resumable(c, file=file, filename=filename, mime_type=mime_type, headers=headers)
            return _to_file_ref(raw if isinstance(raw, dict) else {"raw": raw}, "gemini")
        raise UnsupportedCapabilityError(f"Files API not supported for inferred provider: {fam}")
    except APIError:
        raise
    except httpx.HTTPStatusError as e:
        raise api_error_from_http(e, provider=c.config.provider or fam, model=c.config.model, endpoint="files.create")
    finally:
        await c.aclose()


async def afile_get(
    model: str,
    file_id: str,
    *,
    provider: str = "",
    api_key: str = "",
    base_url: str = "",
    timeout: float = 60.0,
    headers: Optional[Dict[str, str]] = None,
) -> FileRef:
    "Get file metadata as canonical `FileRef`."
    c = mk_auto_client(model=model, provider=provider, api_key=api_key, base_url=base_url, timeout=timeout)
    fam = infer_provider(model, provider=provider, base_url=base_url)
    try:
        if fam in ("openai", "openai_compat"):
            raw = await c.api.call("/files/{file_id}", "GET", headers=headers, route={"file_id": file_id})
            return _to_file_ref(raw if isinstance(raw, dict) else {"raw": raw}, c.config.provider or fam)
        if fam == "anthropic":
            h = dict(headers or {})
            h.setdefault("anthropic-beta", "files-api-2025-04-14")
            raw = await c.api.call("/v1/files/{file_id}", "GET", headers=h, route={"file_id": file_id})
            return _to_file_ref(raw if isinstance(raw, dict) else {"raw": raw}, "anthropic")
        if fam == "gemini":
            nm = file_id if str(file_id).startswith("files/") else f"files/{file_id}"
            raw = await c.api.files.get(name=nm, _headers=headers)
            return _to_file_ref(raw if isinstance(raw, dict) else {"raw": raw}, "gemini")
        raise UnsupportedCapabilityError(f"Files API not supported for inferred provider: {fam}")
    except APIError:
        raise
    except httpx.HTTPStatusError as e:
        raise api_error_from_http(e, provider=c.config.provider or fam, model=c.config.model, endpoint="files.get")
    finally:
        await c.aclose()


async def afile_list(
    model: str,
    *,
    provider: str = "",
    api_key: str = "",
    base_url: str = "",
    timeout: float = 60.0,
    headers: Optional[Dict[str, str]] = None,
    **query,
) -> List[FileRef]:
    "List files as canonical `FileRef` objects."
    c = mk_auto_client(model=model, provider=provider, api_key=api_key, base_url=base_url, timeout=timeout)
    fam = infer_provider(model, provider=provider, base_url=base_url)
    try:
        if fam in ("openai", "openai_compat"):
            raw = await c.api.call("/files", "GET", headers=headers, query=query or None)
            return _list_to_file_refs(raw if isinstance(raw, dict) else {"data": []}, c.config.provider or fam)
        if fam == "anthropic":
            h = dict(headers or {})
            h.setdefault("anthropic-beta", "files-api-2025-04-14")
            raw = await c.api.call("/v1/files", "GET", headers=h, query=query or None)
            return _list_to_file_refs(raw if isinstance(raw, dict) else {"data": []}, "anthropic")
        if fam == "gemini":
            raw = await c.api.files.list(_headers=headers, **query)
            return _list_to_file_refs(raw if isinstance(raw, dict) else {"files": []}, "gemini")
        raise UnsupportedCapabilityError(f"Files API not supported for inferred provider: {fam}")
    except APIError:
        raise
    except httpx.HTTPStatusError as e:
        raise api_error_from_http(e, provider=c.config.provider or fam, model=c.config.model, endpoint="files.list")
    finally:
        await c.aclose()


async def afile_delete(
    model: str,
    file_id: str,
    *,
    provider: str = "",
    api_key: str = "",
    base_url: str = "",
    timeout: float = 60.0,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    "Delete a file and return provider raw response."
    c = mk_auto_client(model=model, provider=provider, api_key=api_key, base_url=base_url, timeout=timeout)
    fam = infer_provider(model, provider=provider, base_url=base_url)
    try:
        if fam in ("openai", "openai_compat"):
            raw = await c.api.call("/files/{file_id}", "DELETE", headers=headers, route={"file_id": file_id})
            return raw if isinstance(raw, dict) else {"raw": raw}
        if fam == "anthropic":
            h = dict(headers or {})
            h.setdefault("anthropic-beta", "files-api-2025-04-14")
            raw = await c.api.call("/v1/files/{file_id}", "DELETE", headers=h, route={"file_id": file_id})
            return raw if isinstance(raw, dict) else {"raw": raw}
        if fam == "gemini":
            nm = file_id if str(file_id).startswith("files/") else f"files/{file_id}"
            raw = await c.api.files.delete(name=nm, _headers=headers)
            return raw if isinstance(raw, dict) else {"raw": raw}
        raise UnsupportedCapabilityError(f"Files API not supported for inferred provider: {fam}")
    except APIError:
        raise
    except httpx.HTTPStatusError as e:
        raise api_error_from_http(e, provider=c.config.provider or fam, model=c.config.model, endpoint="files.delete")
    finally:
        await c.aclose()


async def afile_content(
    model: str,
    file_id: str,
    *,
    provider: str = "",
    api_key: str = "",
    base_url: str = "",
    timeout: float = 60.0,
    headers: Optional[Dict[str, str]] = None,
) -> bytes:
    "Download file content bytes when the provider exposes a file-content endpoint."
    c = mk_auto_client(model=model, provider=provider, api_key=api_key, base_url=base_url, timeout=timeout)
    fam = infer_provider(model, provider=provider, base_url=base_url)
    try:
        if fam in ("openai", "openai_compat"):
            resp = await c.api.call("/files/{file_id}/content", "GET", headers=headers, route={"file_id": file_id}, raw=True)
            if isinstance(resp, httpx.Response):
                return resp.content
            if isinstance(resp, (bytes, bytearray)):
                return bytes(resp)
            return str(resp).encode()
        if fam == "anthropic":
            h = dict(headers or {})
            h.setdefault("anthropic-beta", "files-api-2025-04-14")
            resp = await c.api.call("/v1/files/{file_id}/content", "GET", headers=h, route={"file_id": file_id}, raw=True)
            if isinstance(resp, httpx.Response):
                return resp.content
            if isinstance(resp, (bytes, bytearray)):
                return bytes(resp)
            return str(resp).encode()
        if fam == "gemini":
            raise UnsupportedCapabilityError("Gemini file content download is not exposed in the current built-in ops.")
        raise UnsupportedCapabilityError(f"Files API not supported for inferred provider: {fam}")
    except APIError:
        raise
    except httpx.HTTPStatusError as e:
        raise api_error_from_http(e, provider=c.config.provider or fam, model=c.config.model, endpoint="files.content")
    finally:
        await c.aclose()


__all__ = "FileRef to_input_file_part afile_create afile_get afile_list afile_delete afile_content".split()

