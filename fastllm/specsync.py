"Maintainer utility to refresh local provider specs from official sources."

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import argparse
import json

import httpx
import yaml

from .spec import _specs_dir


OPENAI_SPEC_URL = "https://app.stainless.com/api/spec/documented/openai/openapi.documented.yml"
GEMINI_DISCOVERY_URL = "https://generativelanguage.googleapis.com/$discovery/rest?version=v1beta"
ANTHROPIC_STATS_URL = "https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/main/.stats.yml"

DEFAULT_FILENAMES = {
    "openai": "openai.with-code-samples.yml",
    "gemini": "gemini.json",
    "anthropic": "anthropic.yml",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def _fetch(url: str, *, timeout: float = 60.0, headers: Optional[Dict[str, str]] = None) -> httpx.Response:
    resp = httpx.get(url, timeout=timeout, headers=headers, follow_redirects=True)
    resp.raise_for_status()
    return resp


def _find_key(obj: Any, key: str) -> Optional[Any]:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_key(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_key(v, key)
            if found is not None:
                return found
    return None


def extract_anthropic_openapi_url(stats: Dict[str, Any]) -> str:
    "Extract Anthropic OpenAPI artifact URL from SDK `.stats.yml` metadata."
    for k in ("openapi_spec_url", "openapi_url", "openapiSpecUrl"):
        v = _find_key(stats, k)
        if isinstance(v, str) and v.startswith("http"):
            return v
    raise ValueError("Could not find `openapi_spec_url` in Anthropic stats metadata.")


def resolve_anthropic_spec_url(*, timeout: float = 60.0) -> str:
    "Resolve Anthropic OpenAPI spec URL by reading the SDK `.stats.yml` file."
    resp = _fetch(ANTHROPIC_STATS_URL, timeout=timeout)
    stats = yaml.safe_load(resp.text) or {}
    if not isinstance(stats, dict):
        raise ValueError("Anthropic stats payload is not a YAML mapping.")
    return extract_anthropic_openapi_url(stats)


def _provider_source_url(provider: str, *, timeout: float = 60.0) -> str:
    p = provider.strip().lower()
    if p == "openai":
        return OPENAI_SPEC_URL
    if p == "gemini":
        return GEMINI_DISCOVERY_URL
    if p == "anthropic":
        return resolve_anthropic_spec_url(timeout=timeout)
    raise ValueError(f"Unsupported provider: {provider}")


def _write_if_changed(path: Path, data: bytes, *, write: bool) -> tuple[bool, bool]:
    changed = (not path.exists()) or (path.read_bytes() != data)
    if write and changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return changed, True
    return changed, False


def _sync_one(provider: str, *, specs_dir: Path, timeout: float = 60.0, write: bool = True) -> Dict[str, Any]:
    p = provider.strip().lower()
    src = _provider_source_url(p, timeout=timeout)
    resp = _fetch(src, timeout=timeout)
    data = resp.content

    target = specs_dir / DEFAULT_FILENAMES[p]
    changed, wrote = _write_if_changed(target, data, write=write)
    return {
        "provider": p,
        "source_url": src,
        "target": str(target),
        "bytes": len(data),
        "sha256": _sha256_bytes(data),
        "content_type": resp.headers.get("content-type", ""),
        "fetched_at": _utcnow(),
        "changed": changed,
        "wrote": wrote,
    }


def sync_specs(
    providers: Iterable[str] = ("openai", "gemini", "anthropic"),
    *,
    specs_dir: Optional[str] = None,
    timeout: float = 60.0,
    write: bool = True,
    write_manifest: bool = True,
) -> Dict[str, Any]:
    "Sync provider specs into `specs/` and optionally write a manifest."
    sd = _specs_dir(specs_dir)
    items = [_sync_one(p, specs_dir=sd, timeout=timeout, write=write) for p in providers]
    manifest = {"generated_at": _utcnow(), "specs_dir": str(sd), "items": items}
    if write and write_manifest:
        (sd / "spec_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync fastllm provider specs from official URLs.")
    parser.add_argument(
        "--providers",
        default="openai,gemini,anthropic",
        help="Comma-separated providers to sync (openai,gemini,anthropic).",
    )
    parser.add_argument("--specs-dir", default="", help="Override destination specs directory.")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and diff only; do not write files.")
    parser.add_argument("--no-manifest", action="store_true", help="Skip writing spec_manifest.json.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    "CLI entrypoint: `python -m fastllm.specsync`."
    ns = _parse_args(argv)
    providers = [p.strip() for p in ns.providers.split(",") if p.strip()]
    manifest = sync_specs(
        providers,
        specs_dir=(ns.specs_dir or None),
        timeout=ns.timeout,
        write=(not ns.dry_run),
        write_manifest=(not ns.no_manifest),
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
