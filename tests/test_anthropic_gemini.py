import json
import unittest

import httpx

from fastllm_v2 import APIError, AnthropicClient, ClientConfig, GeminiClient, Msg, Part, RequestOptions, UnsupportedCapabilityError
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

    async def test_openai_style_image_url_canonicalized_for_anthropic_and_gemini(self):
        b64 = "QUFBQQ=="

        seen_a = {"payload": None}

        def anthropic_handler(request: httpx.Request) -> httpx.Response:
            seen_a["payload"] = json.loads(request.content.decode())
            return httpx.Response(200, json={
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            })

        hc_a = httpx.AsyncClient(transport=httpx.MockTransport(anthropic_handler))
        api_a = OpenAPIClient(base_url="https://api.anthropic.com", headers={"x-api-key": "k"},
            ops=anthropic_ops(), transport=AsyncTransport(client=hc_a))
        c_a = AnthropicClient(ClientConfig(model="claude-sonnet-4-5", api_key="k", base_url="https://api.anthropic.com"), api=api_a)
        msgs = [Msg(role="user", content=[
            Part(type="text", text="What's in the image?"),
            Part(type="input_image", data={"image_url": f"data:image/png;base64,{b64}"}),
        ])]
        try:
            await c_a.acomplete(msgs)
            ib = seen_a["payload"]["messages"][0]["content"][1]
            self.assertEqual(ib["type"], "image")
            self.assertEqual(ib["source"]["type"], "base64")
            self.assertEqual(ib["source"]["media_type"], "image/png")
            self.assertEqual(ib["source"]["data"], b64)
        finally:
            await c_a.aclose()
            await hc_a.aclose()

        seen_g = {"payload": None}

        def gemini_handler(request: httpx.Request) -> httpx.Response:
            seen_g["payload"] = json.loads(request.content.decode())
            return httpx.Response(200, json={
                "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
            })

        hc_g = httpx.AsyncClient(transport=httpx.MockTransport(gemini_handler))
        api_g = OpenAPIClient(base_url="https://generativelanguage.googleapis.com/v1beta", ops=gemini_ops(),
            transport=AsyncTransport(client=hc_g))
        c_g = GeminiClient(ClientConfig(model="gemini-2.5-pro", api_key="g", base_url="https://generativelanguage.googleapis.com/v1beta"), api=api_g)
        try:
            await c_g.acomplete(msgs)
            ib = seen_g["payload"]["contents"][0]["parts"][1]
            self.assertEqual(ib["inlineData"]["mimeType"], "image/png")
            self.assertEqual(ib["inlineData"]["data"], b64)
        finally:
            await c_g.aclose()
            await hc_g.aclose()

    async def test_gemini_openai_style_audio_video_canonicalized(self):
        seen = {"payloads": []}

        def gemini_handler(request: httpx.Request) -> httpx.Response:
            seen["payloads"].append(json.loads(request.content.decode()))
            return httpx.Response(200, json={
                "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
            })

        hc = httpx.AsyncClient(transport=httpx.MockTransport(gemini_handler))
        api = OpenAPIClient(base_url="https://generativelanguage.googleapis.com/v1beta", ops=gemini_ops(),
            transport=AsyncTransport(client=hc))
        c = GeminiClient(ClientConfig(model="gemini-2.5-pro", api_key="g", base_url="https://generativelanguage.googleapis.com/v1beta"), api=api)
        msg_video = [Msg(role="user", content=[
            Part(type="text", text="Summarize video"),
            Part(type="input_video", data={"video_url": "https://example.com/demo.mp4"}),
        ])]
        msg_audio = [Msg(role="user", content=[
            Part(type="text", text="Transcribe audio"),
            Part(type="input_audio", data={"input_audio": {"data": "QUFB", "format": "wav"}}),
        ])]
        try:
            await c.acomplete(msg_video)
            await c.acomplete(msg_audio)
            v = seen["payloads"][0]["contents"][0]["parts"][1]
            a = seen["payloads"][1]["contents"][0]["parts"][1]

            self.assertEqual(v["fileData"]["fileUri"], "https://example.com/demo.mp4")
            self.assertEqual(v["fileData"]["mimeType"], "video/mp4")
            self.assertEqual(a["inlineData"]["mimeType"], "audio/wav")
            self.assertEqual(a["inlineData"]["data"], "QUFB")
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_anthropic_raises_for_unsupported_audio_video(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            })

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(base_url="https://api.anthropic.com", headers={"x-api-key": "k"},
            ops=anthropic_ops(), transport=AsyncTransport(client=hc))
        c = AnthropicClient(ClientConfig(model="claude-sonnet-4-5", api_key="k", base_url="https://api.anthropic.com"), api=api)
        msgs = [Msg(role="user", content=[
            Part(type="text", text="Analyze media"),
            Part(type="input_audio", data={"input_audio": {"data": "QUFB", "format": "wav"}}),
            Part(type="input_video", data={"video_url": "https://example.com/v.mp4"}),
        ])]
        try:
            with self.assertRaises(UnsupportedCapabilityError):
                await c.acomplete(msgs)
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_canonical_input_file_file_data_maps_for_anthropic_and_gemini(self):
        b64 = "QUJDRA=="

        seen_a = {"payload": None}

        def anthropic_handler(request: httpx.Request) -> httpx.Response:
            seen_a["payload"] = json.loads(request.content.decode())
            return httpx.Response(200, json={
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            })

        hc_a = httpx.AsyncClient(transport=httpx.MockTransport(anthropic_handler))
        api_a = OpenAPIClient(base_url="https://api.anthropic.com", headers={"x-api-key": "k"},
            ops=anthropic_ops(), transport=AsyncTransport(client=hc_a))
        c_a = AnthropicClient(ClientConfig(model="claude-sonnet-4-5", api_key="k", base_url="https://api.anthropic.com"), api=api_a)
        msgs = [Msg(role="user", content=[
            Part(type="text", text="Summarize this doc"),
            Part(type="input_file", data={"filename": "doc.pdf", "file_data": b64}),
        ])]
        try:
            await c_a.acomplete(msgs)
            doc = seen_a["payload"]["messages"][0]["content"][1]
            self.assertEqual(doc["type"], "document")
            self.assertEqual(doc["source"]["type"], "base64")
            self.assertEqual(doc["source"]["media_type"], "application/pdf")
            self.assertEqual(doc["source"]["data"], b64)
        finally:
            await c_a.aclose()
            await hc_a.aclose()

        seen_g = {"payload": None}

        def gemini_handler(request: httpx.Request) -> httpx.Response:
            seen_g["payload"] = json.loads(request.content.decode())
            return httpx.Response(200, json={
                "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
            })

        hc_g = httpx.AsyncClient(transport=httpx.MockTransport(gemini_handler))
        api_g = OpenAPIClient(base_url="https://generativelanguage.googleapis.com/v1beta", ops=gemini_ops(),
            transport=AsyncTransport(client=hc_g))
        c_g = GeminiClient(ClientConfig(model="gemini-2.5-pro", api_key="g", base_url="https://generativelanguage.googleapis.com/v1beta"), api=api_g)
        try:
            await c_g.acomplete(msgs)
            doc = seen_g["payload"]["contents"][0]["parts"][1]
            self.assertEqual(doc["inlineData"]["mimeType"], "application/pdf")
            self.assertEqual(doc["inlineData"]["data"], b64)
        finally:
            await c_g.aclose()
            await hc_g.aclose()

    async def test_anthropic_supports_canonical_assistant_tool_calls_and_tool_results(self):
        seen = {"payload": None}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content.decode())
            return httpx.Response(200, json={
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            })

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(base_url="https://api.anthropic.com", headers={"x-api-key": "k"},
            ops=anthropic_ops(), transport=AsyncTransport(client=hc))
        c = AnthropicClient(ClientConfig(model="claude-sonnet-4-5", api_key="k", base_url="https://api.anthropic.com"), api=api)
        msgs = [
            Msg(role="user", content=[Part(type="text", text="calc")]),
            Msg(role="assistant", content=[Part(type="text", text="I'll call a tool")], data={
                "tool_calls": [{"id": "toolu_1", "name": "simple_add", "arguments": {"a": 1, "b": 2}}],
            }),
            Msg(role="tool", content=[Part(type="text", text="3")], data={"tool_call_id": "toolu_1", "name": "simple_add"}),
        ]
        try:
            await c.acomplete(msgs)
            ms = seen["payload"]["messages"]
            self.assertEqual(ms[1]["role"], "assistant")
            self.assertEqual(ms[1]["content"][1]["type"], "tool_use")
            self.assertEqual(ms[1]["content"][1]["id"], "toolu_1")
            self.assertEqual(ms[1]["content"][1]["name"], "simple_add")
            self.assertEqual(ms[1]["content"][1]["input"], {"a": 1, "b": 2})
            self.assertEqual(ms[2]["role"], "user")
            self.assertEqual(ms[2]["content"][0]["type"], "tool_result")
            self.assertEqual(ms[2]["content"][0]["tool_use_id"], "toolu_1")
            self.assertEqual(ms[2]["content"][0]["content"], "3")
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_gemini_supports_canonical_assistant_tool_calls_and_tool_results(self):
        seen = {"payload": None}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content.decode())
            return httpx.Response(200, json={
                "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
            })

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(base_url="https://generativelanguage.googleapis.com/v1beta", ops=gemini_ops(),
            transport=AsyncTransport(client=hc))
        c = GeminiClient(ClientConfig(model="gemini-2.5-pro", api_key="g", base_url="https://generativelanguage.googleapis.com/v1beta"), api=api)
        msgs = [
            Msg(role="user", content=[Part(type="text", text="calc")]),
            Msg(role="assistant", content=[Part(type="text", text="calling tool")], data={
                "tool_calls": [{"id": "call_1", "name": "simple_add", "arguments": {"a": 1, "b": 2}}],
            }),
            Msg(role="tool", content=[Part(type="text", text='{"sum":3}')], data={"tool_call_id": "call_1", "name": "simple_add"}),
        ]
        try:
            await c.acomplete(msgs)
            cts = seen["payload"]["contents"]
            self.assertEqual(cts[1]["role"], "model")
            self.assertEqual(cts[1]["parts"][1]["functionCall"]["id"], "call_1")
            self.assertEqual(cts[1]["parts"][1]["functionCall"]["name"], "simple_add")
            self.assertEqual(cts[1]["parts"][1]["functionCall"]["args"], {"a": 1, "b": 2})
            self.assertEqual(cts[2]["role"], "user")
            fr = cts[2]["parts"][0]["functionResponse"]
            self.assertEqual(fr["name"], "simple_add")
            self.assertEqual(fr["id"], "call_1")
            self.assertEqual(fr["response"], {"sum": 3})
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_anthropic_http_error_is_structured_apierror(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "bad document block"},
            }, headers={"anthropic-request-id": "req_test_123"})

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(base_url="https://api.anthropic.com", headers={"x-api-key": "k"},
            ops=anthropic_ops(), transport=AsyncTransport(client=hc))
        c = AnthropicClient(ClientConfig(model="claude-sonnet-4-5", api_key="k", base_url="https://api.anthropic.com"), api=api)
        try:
            with self.assertRaises(APIError) as ctx:
                await c.acomplete(_user("hello"))
            err = ctx.exception
            self.assertEqual(err.provider, "anthropic")
            self.assertEqual(err.model, "claude-sonnet-4-5")
            self.assertEqual(err.endpoint, "messages.create")
            self.assertEqual(err.status_code, 400)
            self.assertEqual(err.error_type, "invalid_request_error")
            self.assertEqual(err.code, "invalid_request_error")
            self.assertEqual(err.request_id, "req_test_123")
            self.assertFalse(err.retryable)
            self.assertIn("bad document block", err.message)
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_gemini_http_error_surfaces_status_code_string(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={
                "error": {
                    "code": 429,
                    "message": "Quota exceeded",
                    "status": "RESOURCE_EXHAUSTED",
                },
            })

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(base_url="https://generativelanguage.googleapis.com/v1beta", ops=gemini_ops(),
            transport=AsyncTransport(client=hc))
        c = GeminiClient(ClientConfig(model="gemini-2.5-pro", api_key="g", base_url="https://generativelanguage.googleapis.com/v1beta"), api=api)
        try:
            with self.assertRaises(APIError) as ctx:
                await c.acomplete(_user("hello"))
            err = ctx.exception
            self.assertEqual(err.provider, "gemini")
            self.assertEqual(err.status_code, 429)
            self.assertEqual(err.error_type, "RESOURCE_EXHAUSTED")
            self.assertEqual(err.code, "RESOURCE_EXHAUSTED")
            self.assertTrue(err.retryable)
            self.assertIn("Quota exceeded", err.message)
        finally:
            await c.aclose()
            await hc.aclose()
