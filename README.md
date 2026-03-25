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

## Generic kwargs (no wrapper churn)

You can pass options as `RequestOptions(...)` or directly as kwargs.
Known option kwargs are mapped (`max_tokens`, `temperature`, `tools`, `tool_choice`, `search`, `cache`, etc.).
Unknown kwargs are forwarded to provider payload (`native` body).

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

## Multimodal + tools (example)

```python
from fastllm_v2 import Msg, Part, ToolSpec

msgs = [Msg(role='user', content=[
    Part(type='text', text='Summarize this image'),
    Part(type='image', data={'inlineData': {'mimeType': 'image/png', 'data': '<base64>'}}),
])]

tool = ToolSpec(name='lookup', parameters={'type': 'object', 'properties': {'q': {'type': 'string'}}})
res = await gemini_client.acomplete(msgs, tools=[tool], tool_choice='required', search=True)
```

## Caching

Use generic `cache=...` plus pass-through for provider-specific controls:

- OpenAI-compatible: `cache=True` maps to `store=True` by default.
- Anthropic: `cache=True` adds `cache_control` to supported user blocks.
- Gemini: `cache={'cachedContent': 'cachedContents/...'} ` maps to cached context usage.

For provider-specific behavior, pass exact payload fields directly as kwargs or `native`.

Note: `response_format=...` on `OpenAIClient.acomplete(...)` is translated to the Responses API `text.format` shape.

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
