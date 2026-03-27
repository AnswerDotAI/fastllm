# fastllm_v2

`fastllm_v2` is a minimal LLM client built around a generic OpenAPI operation layer.

Highlights:

- OpenAPI-like operation specs -> dynamic operation groups/methods
- Async HTTP transport + SSE parsing
- Normalized completion/streaming types
- Built-in families: OpenAI (Responses + Chat), Anthropic (Messages), Gemini (GenerateContent)
- Generic `kwargs` passthrough on high-level clients (unknown kwargs become provider-native body fields)
- Multimodal input mapping (text/image/file/audio) where providers support it
- Generic cache options + provider-specific pass-through
- Provider-agnostic `estimate_cost(...)` from normalized usage + your pricing table
- OpenAI Responses `response_format` compatibility mapped to `text.format`
- Gemini auth on both high-level + dynamic operations (`x-goog-api-key` header)
- OpenAI full operation snapshot from official OpenAPI spec
- Gemini full operation snapshot from official discovery docs
- Expanded Anthropic documented routes (messages, batches, files, models, org admin)
- Generic controls for dynamic calls: `_stream`, `_raw`, `_files`, `_data`, `_headers`, `_query`, `_body`
- High-level async `acompletion` API with automatic provider inference
- Streaming collation via `acollect_stream(acompletion(..., stream=True))`
- Canonical toolloop message replay across OpenAI/Anthropic/Gemini/OpenAI-compatible chat

## Install

```bash
pip install fastllm_v2
```

## Quickstart

```python
import asyncio
from fastllm_v2 import Msg, Part, mk_client

async def main():
    c = mk_client('openai', model='gpt-5-mini', api_key='YOUR_KEY')
    try:
        res = await c.acomplete([Msg(role='user', content=[Part(type='text', text='Say hi')])])
        print(res.message.content[0].text)
    finally:
        await c.aclose()

asyncio.run(main())
```

## One-call API (LiteLLM-style)

You can skip explicit client creation and call a single high-level function:

```python
from fastllm_v2 import acompletion

res = await acompletion(
    model='gpt-5-mini',
    messages=[{'role': 'user', 'content': 'Say hi in one line'}],
    api_key='YOUR_KEY',
)
print(res.message.content[0].text)
```

For built-in providers, `api_key` is optional and auto-resolved from env:

- OpenAI -> `OPENAI_API_KEY`
- Anthropic -> `ANTHROPIC_API_KEY`
- Gemini -> `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
- OpenAI-compatible vendors -> vendor envs when known (for example Moonshot/Kimi -> `MOONSHOT_API_KEY`)

Routing rules in `endpoint='auto'` mode:

- `claude...` -> Anthropic Messages API
- `gemini...` -> Gemini GenerateContent API
- `gpt...` -> OpenAI Responses API (auto fallback to Chat Completions if unsupported)
- otherwise -> OpenAI-compatible Chat Completions

`acompletion` infers routing from `model` and `base_url`.
The `provider`/`custom_llm_provider` arguments remain for backward compatibility but are ignored.

### Model-Only Swap (no code changes)

You can keep call code identical and only change `model=...`.
`fastllm_v2` now resolves provider family, default base URL, and API key env vars automatically.

Examples:

```python
# OpenAI (uses OPENAI_API_KEY)
res = await acompletion(model='gpt-5-mini', messages='Say hi')

# Anthropic (uses ANTHROPIC_API_KEY)
res = await acompletion(model='claude-sonnet-4-5', messages='Say hi')

# Gemini (uses GEMINI_API_KEY or GOOGLE_API_KEY)
res = await acompletion(model='gemini-2.5-flash', messages='Say hi')

# Moonshot/Kimi (uses MOONSHOT_API_KEY + default https://api.moonshot.ai/v1)
res = await acompletion(model='kimi-k2.5', messages='Say hi')
```

OpenAI-compatible `vendor/model` names are also supported:

```python
# Uses QWEN_API_KEY + QWEN_BASE_URL (or built-in DashScope default URL)
res = await acompletion(model='qwen/qwen-plus', messages='Say hi')
```

For unknown OpenAI-compatible vendors, you can still do model-only by setting env vars once:

- `<VENDOR>_API_KEY`
- `<VENDOR>_BASE_URL` (or `<VENDOR>_API_BASE`)

Then call with `model='<vendor>/<model-name>'`.

Streaming is also available:

```python
from fastllm_v2 import acompletion, acollect_stream

summary = await acollect_stream(acompletion(
    model='gpt-5-mini',
    messages='Count from 1 to 3.',
    api_key='YOUR_OPENAI_KEY',
    stream=True,
))
print(summary.text)
print(summary.final)      # aggregated Delta
print(len(summary.raw_events))
```

### Accepted message inputs

`acompletion(messages=...)` accepts:

- `str`
- `dict` (OpenAI-style role/content)
- canonical `Msg`
- `Completion` (assistant turn from previous non-stream call)
- `StreamSummary` (assistant turn from `acollect_stream(...)`)
- list/tuple mixing the above

This lets you append model outputs directly during toolloop flows.

## Generic kwargs (no wrapper churn)

You can pass options as `RequestOptions(...)` or directly as kwargs.
Known option kwargs are mapped (`max_tokens`, `temperature`, `tools`, `tool_choice`, `cache`, etc.).
Unknown kwargs are forwarded to provider payload (`native` body).
`search=` convenience is intentionally not supported; pass provider-native web-search tools in `tools=[...]`.

```python
res = await c.acomplete(
    msgs,
    temperature=0.2,
    cache=True,
    metadata={'team': 'research'},
    headers={'x-trace-id': 'abc'},
    query={'seed': 7},
)
```

## Canonical `reasoning_effort`

`reasoning_effort` is the generic thinking/reasoning control across supported high-level families.

You can pass a string level (for example `none`, `minimal`, `low`, `medium`, `high`) or a provider-native dict when needed.

Mapping by family:

- OpenAI Responses -> `reasoning: {"effort": ...}`
- OpenAI Chat / OpenAI-compatible Chat -> `reasoning_effort` (passthrough)
- Anthropic -> `thinking` config (budget-based mapping)
- Gemini -> `generationConfig.thinkingConfig` (budget-based mapping)

Notes:

- Omitting `reasoning_effort` means provider default behavior (not always "off").
- For exact provider controls, pass native payload fields through `native={...}`.

### Disabling thinking (provider/model caveats)

`reasoning_effort='none'` is not universally supported as a hard "off" switch.

- OpenAI Responses:
  `reasoning_effort='none'` disables reasoning on models that support it (for example `gpt-5.1`).
  Older GPT-5 family models may default to reasoning and may not support `none`.
- Gemini:
  Use `reasoning_effort={'thinkingBudget': 0}` on models that allow disabling (for example 2.5 Flash family).
  Some models (for example 2.5 Pro) do not support thinking-off.
- Anthropic:
  In canonical mode, omit `reasoning_effort` to avoid explicitly enabling thinking.
  For model/version-specific thinking controls (for example adaptive/enabled modes), use `native={"thinking": ...}`.
- OpenAI-compatible chat vendors:
  `reasoning_effort` may be ignored. Use provider-native fields via `native={...}` when needed
  (for example Kimi: `native={"thinking":{"type":"disabled"}}`).

```python
# Same call shape, model-only swap
res = await acompletion(
    model='gpt-5-mini',
    messages=[{'role': 'user', 'content': 'Say hi'}],
    reasoning_effort='low',
)

# Exact Gemini control via native dict pass-through
res = await acompletion(
    model='gemini-2.5-flash',
    messages=[{'role': 'user', 'content': 'Say hi'}],
    reasoning_effort={'thinkingBudget': 0},
)

# Kimi/Moonshot explicit disable via native passthrough
res = await acompletion(
    model='kimi-k2.5',
    messages=[{'role': 'user', 'content': 'Say hi'}],
    native={'thinking': {'type': 'disabled'}},
)
```

## Multimodal + tools (example)

Tools are passed as raw schema dicts (for example, `lisette.lite_mk_func(...)` output).
Use OpenAI-style part payloads as the canonical input format; provider clients normalize as needed.

```python
from fastllm_v2 import Msg, Part

msgs = [Msg(role='user', content=[
    Part(type='text', text='Summarize this image'),
    Part(type='input_image', data={'image_url': 'data:image/png;base64,<base64>'}),
])]

tool = {'type': 'function', 'function': {
    'name': 'lookup',
    'parameters': {'type': 'object', 'properties': {'q': {'type': 'string'}}},
}}
res = await gemini_client.acomplete(
    msgs,
    tools=[tool, {'googleSearch': {}}],
    tool_choice='required',
)
```

For Anthropic and Gemini, the same canonical `input_image + image_url` payload is converted to native
`source`/`inlineData` shapes automatically.

Current canonical multimodal handling:

- OpenAI (Responses/Chat): `input_image`, `input_audio`, `input_file`, `input_video` (video is normalized to file input).
- Gemini: `input_image`, `input_audio`, `input_file`, `input_video` (mapped to `inlineData`/`fileData`).
- Anthropic: `input_image` and `input_file`/`document` are normalized; no dedicated video canonical mapping yet.

## Canonical toolloop contract

Toolloop history can stay provider-agnostic with canonical message objects.

Assistant tool call turn:

```python
Msg(
    role="assistant",
    content=[Part(type="text", text="I'll call tools...")],  # optional text
    data={"tool_calls": [{"id": "...", "name": "...", "arguments": {...}}]},
)
```

Tool result turn:

```python
Msg(
    role="tool",
    content=[Part(type="text", text="tool output text/json")],
    data={"tool_call_id": "...", "name": "..."},
)
```

`fastllm_v2` maps these automatically to:

- OpenAI chat/openai-compatible chat: `assistant.tool_calls` + `role="tool"` messages
- OpenAI Responses: `function_call` + `function_call_output` input items
- Anthropic: `tool_use` + `tool_result`
- Gemini: `functionCall` + `functionResponse`

### Toolloop example (2+ turns)

```python
from fastllm_v2 import Msg, Part, acompletion, acollect_stream

msgs = [Msg(role="user", content=[Part(type="text", text=pr)])]

for _ in range(6):
    res = await acollect_stream(acompletion(
        model=MODEL,
        messages=msgs,
        tools=[simple_add_tool],
        stream=True,
    ))
    msgs.append(res)  # StreamSummary is accepted directly
    if not res.tool_calls:
        break
    for tc in res.tool_calls:
        out = simple_add(**tc.arguments)
        msgs.append(Msg(
            role="tool",
            content=[Part(type="text", text=str(out))],
            data={"tool_call_id": tc.id, "name": tc.name},
        ))
```

### Early compatibility validation

`fastllm_v2` validates canonical media kinds before request build:

- Anthropic + `input_audio` / `input_video` -> raises `UnsupportedCapabilityError`
- `openai_compat` + non-text media -> warning (compatibility varies by vendor)

This provides hard failures for known-invalid inputs while still giving early guidance for uncertain compatibility.

## Caching

Use generic `cache=...` plus pass-through for provider-specific controls:

- OpenAI-compatible: `cache=True` maps to `store=True` by default.
- Anthropic: `cache=True` adds `cache_control` to supported user blocks.
- Gemini: `cache={'cachedContent': 'cachedContents/...'} ` maps to cached context usage.

For provider-specific behavior, pass exact payload fields directly as kwargs or `native`.

Note: `response_format=...` on `OpenAIClient.acomplete(...)` is translated to the Responses API `text.format` shape.

If implicit caching is supported by a provider, `cache=True` is not needed.

## Canonical files API

`fastllm_v2` now includes a provider-agnostic files layer for `openai`, `openai_compat` (openai-chat style),
`anthropic`, and `gemini`:

- `afile_create(...)`
- `afile_get(...)`
- `afile_list(...)`
- `afile_delete(...)`
- `afile_content(...)`
- `to_input_file_part(...)`

```python
from fastllm_v2 import afile_create, to_input_file_part, Msg, Part, acompletion, acollect_stream

# 1) upload once
fref = await afile_create(
    model="gpt-5-mini",      # swap model only: claude..., gemini..., kimi... etc.
    file=open("doc.pdf", "rb"),
    filename="doc.pdf",
    mime_type="application/pdf",
)

# 2) turn uploaded file into a canonical Part
file_part = to_input_file_part(fref)

# 3) reuse in chat/completions
msg = Msg(role="user", content=[
    Part(type="text", text="Summarize this document."),
    file_part,
])
res = await acollect_stream(acompletion(model="gpt-5-mini", messages=[msg], stream=True))
print(res.text)
```

Notes:

- `to_input_file_part(...)` injects provider-native fields automatically (Anthropic/Gemini/OpenAI).
- `gemini` file upload uses resumable upload flow.
- `afile_content(...)` is currently available for OpenAI/OpenAI-compatible and Anthropic;
  Gemini raw content download is not exposed in current built-in ops.

## Error handling

High-level and provider clients now raise structured `APIError` for HTTP and stream-level provider errors.

```python
from fastllm_v2 import APIError

try:
    res = await acollect_stream(acompletion(model=ANTHROPIC_MODEL, messages=[msg], stream=True))
except APIError as e:
    print(e.message)
    print(e.provider, e.model, e.endpoint)
    print(e.status_code, e.error_type, e.code, e.request_id)
    print("retryable?", e.retryable)
    print("raw:", e.raw)   # original provider payload when available
```

This makes transient server failures (for example provider `server_error`) easy to detect and retry.
Provider-specific official error codes/statuses are surfaced in `e.code` (for example OpenAI `context_length_exceeded`,
Anthropic `invalid_request_error`, Gemini `RESOURCE_EXHAUSTED`).

## Cost estimation

`fastllm_v2` normalizes token usage but does not hardcode provider prices.
Pass your own pricing table to keep prices current.

```python
from fastllm_v2 import estimate_cost, ModelPrice

cost = estimate_cost(
    completion,
    prices={
        'gpt-5-mini*': ModelPrice(prompt_per_million=0.25, completion_per_million=2.0)
    },
)
print(cost.total_cost, cost.currency)
```

## OpenAPI layer and endpoint coverage

Built-in specs include:

- OpenAI official OpenAPI snapshot (broad endpoint coverage)
- Gemini official discovery snapshot (broad endpoint coverage)
- Anthropic expanded documented API surface

You can also load provider specs directly and call any operation dynamically:

```python
from fastllm_v2 import client_from_spec, load_spec_file

spec = load_spec_file('openapi.json')
api = client_from_spec('https://api.example.com', spec, headers={'Authorization': 'Bearer X'})

# non-stream
res = await api.responses.create(model='gpt-5-mini', input='hello')

# stream
async for ev in api.responses.create(model='gpt-5-mini', input='hello', stream=True, _stream=True):
    print(ev)
```

`_stream=True` enables SSE transport for stream-capable operations; regular request fields like `stream=True` stay in payload/query.

Dynamic operations also support:

- `_raw=True` to return raw `httpx.Response`
- `_files=...` for multipart uploads
- `_data=...` for form/data requests
- `_headers=...`, `_query=...`, `_body=...` for request overrides

## OpenAI Responses API wrappers

`OpenAIClient` includes convenience wrappers for:

- `POST /responses` (`acomplete`, `astream`)
- `GET /responses/{response_id}` (`aresponse_get`)
- `DELETE /responses/{response_id}` (`aresponse_delete`)
- `POST /responses/{response_id}/cancel` (`aresponse_cancel`)
- `GET /responses/{response_id}/input_items` (`aresponse_input_items`)
- `POST /responses/compact` (`acompact`)
- `POST /responses/input_tokens` (`ainput_tokens`)

And chat compatibility:

- `achat_complete(...)`
- `achat_stream(...)`

## Run tests

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```
