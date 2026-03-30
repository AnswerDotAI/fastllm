# TODO

## Spec Sync Roadmap (Official Sources)

Goal: keep `fastllm` operation coverage current with minimal manual maintenance by auto-refreshing specs from official provider sources and updating the `specs/` sources used by runtime parsers.

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
  - SDK metadata includes a Stainless OpenAPI artifact URL (from `.stats.yml` in `anthropic-sdk-python`).
  - Example artifact URL:
    - `https://storage.googleapis.com/stainless-sdk-openapi-specs/anthropic%2Fanthropic-dd2dcd00a757075370a7e4a7f469a1e2d067c2118684c3b70d7906a8f5cf518b.yml`
  - Strategy: resolve `openapi_spec_url` from `.stats.yml`, then parse that OpenAPI spec directly.

## Planned Architecture

- Add a new module: `fastllm/specsync.py`
- Add optional script entrypoint: `python -m fastllm.specsync`

Core steps:

1. Fetch
- Download provider source docs/specs with conditional requests (`ETag` / `If-None-Match`) when available.
- Store raw snapshots under `fastllm/spec_snapshots/`.
- Write metadata manifest (`provider`, `source_url`, `fetched_at`, `sha256`, `etag`, `status`).

2. Normalize
- OpenAI: parse OpenAPI -> `OpSpec` via existing `spec_to_ops` flow.
- Gemini: parse discovery JSON -> normalized operation records -> `OpSpec`.
- Anthropic: parse OpenAPI -> `OpSpec` via existing `spec_to_ops` flow.

3. Refresh Sources
- Update `specs/openai*.yml|json` and `specs/gemini.json`.
- Keep deterministic parser output (stable sort by `group/name/path/verb`).
- Persist source metadata and generation timestamp in a manifest.

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

- [x] M1: Build `specsync` fetch + manifest + snapshot storage
- [x] M2: OpenAI autogen wired to `specs/openai*.yml|json`
- [x] M3: Gemini discovery autogen wired to `specs/gemini.json`
- [x] M4: Anthropic OpenAPI fetch via `.stats.yml` and parser wiring
- [ ] M5: Weekly CI auto-refresh PR
- [ ] M6: Release workflow integration + changelog auto-note

## Nice-to-Haves

- [ ] `--provider openai|gemini|anthropic|all`
- [ ] `--dry-run` and `--write` modes
- [ ] Human-readable markdown diff artifact under `docs/spec-diffs/`
- [ ] Optional notification hook for breaking-change detection
