import json
import unittest

import httpx

from fastllm_v2 import AnthropicClient, ClientConfig, GeminiClient, Msg, Part, RequestOptions
from fastllm_v2.builtin_specs import anthropic_ops, gemini_ops
from fastllm_v2.oapi import OpenAPIClient
from fastllm_v2.transport import AsyncTransport


def _user(s): return [Msg(role="user", content=[Part(type="text", text=s)])]


class TestAnthropicGemini(unittest.IsolatedAsyncioTestCase):
    async def test_anthropic_complete_stream_and_models(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "claude-sonnet-4-5"}]})

            payload = json.loads(request.content.decode())
            if payload.get("stream"):
                body = (
                    'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}\n\n'
                    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":1,"output_tokens":2}}\n\n'
                    'data: {"type":"message_stop"}\n\n'
                )
                return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json={
                "model": payload["model"],
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 4},
            })

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(base_url="https://api.anthropic.com", headers={"x-api-key": "k"},
            ops=anthropic_ops(), transport=AsyncTransport(client=hc))
        c = AnthropicClient(ClientConfig(model="claude-sonnet-4-5", api_key="k", base_url="https://api.anthropic.com"), api=api)
        try:
            res = await c.acomplete(_user("hello"), options=RequestOptions(max_tokens=32))
            self.assertEqual(res.message.content[0].text, "ok")
            self.assertEqual(res.usage.total_tokens, 7)

            out = []
            async for d in c.astream(_user("hello")):
                out.append(d.text)
            self.assertEqual("".join(out), "hi")

            models = await c.alist_models()
            self.assertEqual(models["data"][0]["id"], "claude-sonnet-4-5")
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_gemini_complete_and_stream(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith(":streamGenerateContent"):
                body = (
                    'data: {"candidates":[{"content":{"parts":[{"text":"hel"}]}}]}\n\n'
                    'data: {"candidates":[{"content":{"parts":[{"text":"hello"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3}}\n\n'
                )
                return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

            return httpx.Response(200, json={
                "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3, "totalTokenCount": 5},
            })

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(base_url="https://generativelanguage.googleapis.com/v1beta", ops=gemini_ops(),
            transport=AsyncTransport(client=hc))
        c = GeminiClient(ClientConfig(model="gemini-2.5-pro", api_key="g", base_url="https://generativelanguage.googleapis.com/v1beta"), api=api)
        try:
            res = await c.acomplete(_user("hello"))
            self.assertEqual(res.message.content[0].text, "ok")
            self.assertEqual(res.usage.total_tokens, 5)

            out = []
            async for d in c.astream(_user("hello")):
                out.append(d.text)
            self.assertEqual("".join(out), "hello")
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_anthropic_cache_multimodal_and_tool_use(self):
        seen = {"payload": None}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content.decode())
            return httpx.Response(200, json={
                "model": "claude-sonnet-4-5",
                "content": [{
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "lookup",
                    "input": {"q": "x"},
                }],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            })

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(base_url="https://api.anthropic.com", headers={"x-api-key": "k"},
            ops=anthropic_ops(), transport=AsyncTransport(client=hc))
        c = AnthropicClient(ClientConfig(model="claude-sonnet-4-5", api_key="k", base_url="https://api.anthropic.com"), api=api)
        msgs = [Msg(role="user", content=[
            Part(type="text", text="describe image"),
            Part(type="image", data={"source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}),
            Part(type="pdf", data={"source": {"type": "base64", "media_type": "application/pdf", "data": "BBBB"}}),
        ])]
        try:
            res = await c.acomplete(msgs, cache=True, tool_choice="auto", reasoning_effort="low",
                tools=[{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}])
            self.assertEqual(res.tool_calls[0].name, "lookup")
            blocks = seen["payload"]["messages"][0]["content"]
            self.assertEqual(blocks[0]["cache_control"]["type"], "ephemeral")
            self.assertEqual(blocks[1]["type"], "image")
            self.assertEqual(blocks[2]["type"], "document")
            self.assertEqual(seen["payload"]["tool_choice"]["type"], "auto")
            self.assertEqual(seen["payload"]["thinking"]["type"], "enabled")
            self.assertGreater(seen["payload"]["max_tokens"], seen["payload"]["thinking"]["budget_tokens"])
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_gemini_tools_search_cache_and_tool_call_norm(self):
        seen = {"payloads": [], "paths": []}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["paths"].append(request.url.path)
            payload = json.loads(request.content.decode())
            seen["payloads"].append(payload)
            if request.url.path.endswith(":streamGenerateContent"):
                body = (
                    'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"lookup","args":{"q":"x"}}}]}}]}\n\n'
                    'data: {"candidates":[{"content":{"parts":[{"text":"done"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3}}\n\n'
                )
                return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

            return httpx.Response(200, json={
                "candidates": [{
                    "content": {"parts": [{"functionCall": {"name": "lookup", "args": {"q": "x"}}}, {"text": "ok"}]},
                    "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3, "totalTokenCount": 5},
            })

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(base_url="https://generativelanguage.googleapis.com/v1beta", ops=gemini_ops(),
            transport=AsyncTransport(client=hc))
        c = GeminiClient(ClientConfig(model="gemini-2.5-pro", api_key="g", base_url="https://generativelanguage.googleapis.com/v1beta"), api=api)
        msgs = [Msg(role="user", content=[
            Part(type="text", text="hello"),
            Part(type="image_url", data={"url": "https://img"}),
        ])]
        try:
            res = await c.acomplete(msgs,
                tools=[
                    {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}},
                    {"googleSearch": {"dynamic": True}},
                ],
                tool_choice="required",
                cache={"cachedContent": "cachedContents/abc"})
            self.assertEqual(res.tool_calls[0].name, "lookup")
            self.assertEqual(res.usage.total_tokens, 5)

            ds = []
            async for d in c.astream(msgs,
                tools=[
                    {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}},
                    {"googleSearch": {}},
                ],
                cache={"cachedContent": "cachedContents/abc"}):
                ds.append(d)
            self.assertEqual(ds[0].tool_calls[0].name, "lookup")

            p = seen["payloads"][0]
            self.assertEqual(p["cachedContent"], "cachedContents/abc")
            self.assertEqual(p["toolConfig"]["functionCallingConfig"]["mode"], "ANY")
            self.assertTrue(any("googleSearch" in t for t in p["tools"]))
            self.assertEqual(p["contents"][0]["parts"][1]["fileData"]["fileUri"], "https://img")
            self.assertTrue(any("/models/" in o for o in seen["paths"]))
            self.assertTrue(all("%2F" not in o for o in seen["paths"]))
        finally:
            await c.aclose()
            await hc.aclose()
