# TODO

## Spec Sync Roadmap (Official Sources)

Goal: keep `fastllm_v2` operation coverage current with minimal manual maintenance by auto-refreshing specs from official provider sources and regenerating `official_ops` artifacts.

## Scope

- In scope:
  - OpenAI
  - Gemini
  - Anthropic
- Out of scope (for now): provider proxy/gateway features, routing/fallback orchestration, observability integrations.

## Source of Truth

- OpenAI:
  - Primary: `https://app.stainless.com/api/spec/documented/openai/openapi.documented.yml`
  - Reference repo: `https://github.com/openai/openai-openapi`
- Gemini:
  - Primary discovery doc: `https://generativelanguage.googleapis.com/$discovery/rest?version=v1beta`
- Anthropic:
  - No single official OpenAPI artifact currently exposed.
  - Primary source: official API reference pages under `https://docs.anthropic.com/en/api`.
  - Strategy: maintain a curated provider surface with automated doc-diff checks and explicit beta-header support.

## Planned Architecture

- Add a new module: `fastllm_v2/specsync.py`
- Add optional script entrypoint: `python -m fastllm_v2.specsync`

Core steps:

1. Fetch
- Download provider source docs/specs with conditional requests (`ETag` / `If-None-Match`) when available.
- Store raw snapshots under `fastllm_v2/spec_snapshots/`.
- Write metadata manifest (`provider`, `source_url`, `fetched_at`, `sha256`, `etag`, `status`).

2. Normalize
- OpenAI: parse OpenAPI -> `OpSpec` via existing `spec_to_ops` flow.
- Gemini: parse discovery JSON -> normalized operation records -> `OpSpec`.
- Anthropic: reconcile curated endpoints against latest docs map; emit warnings for added/removed endpoints.

3. Generate
- Regenerate `fastllm_v2/official_ops.py` from normalized operation records.
- Keep generation deterministic (stable sort by `group/name/path/verb`).
- Stamp file header with source metadata and generation timestamp.

4. Validate
- Ensure required key operations exist (e.g. OpenAI `/responses` + `/chat/completions`, Gemini `generateContent` + `streamGenerateContent`, Anthropic `/v1/messages`).
- Run unit tests.
- Run a compact operation diff report (`added`, `removed`, `signature changed`).

## CI Automation Plan

- Add a scheduled workflow (weekly):
  - Run spec sync in dry-run mode.
  - If diff exists, regenerate artifacts and open a PR automatically.
- PR body should include:
  - Source hash changes
  - Endpoint-level diff summary
  - Any breaking-change flags

## Breaking Change Policy

- Non-breaking changes:
  - Added operations/params -> patch/minor release.
- Potentially breaking changes:
  - Removed operation, renamed critical fields, changed streamability -> require manual review + release note + version bump.

## Security and Reliability

- Restrict fetches to allowlisted official domains.
- Use request timeouts and retries with bounded backoff.
- Never execute remote code/data; parse as JSON/YAML only.
- Keep a fallback path: if remote fetch fails, continue using last known committed artifacts.

## Milestones

- [ ] M1: Build `specsync` fetch + manifest + snapshot storage
- [ ] M2: OpenAI autogen wired to `official_ops.py`
- [ ] M3: Gemini discovery autogen wired to `official_ops.py`
- [ ] M4: Anthropic doc-diff checker + curated surface validator
- [ ] M5: Weekly CI auto-refresh PR
- [ ] M6: Release workflow integration + changelog auto-note

## Nice-to-Haves

- [ ] `--provider openai|gemini|anthropic|all`
- [ ] `--dry-run` and `--write` modes
- [ ] Human-readable markdown diff artifact under `docs/spec-diffs/`
- [ ] Optional notification hook for breaking-change detection
