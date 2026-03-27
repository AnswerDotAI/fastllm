import json
import unittest
import warnings

import httpx

from fastllm_v2 import APIError, ClientConfig, Msg, OpenAIClient, Part, RequestOptions
from fastllm_v2.builtin_specs import openai_ops
from fastllm_v2.oapi import OpenAPIClient
from fastllm_v2.transport import AsyncTransport


def _user(s): return [Msg(role="user", content=[Part(type="text", text=s)])]


class TestOpenAIResponses(unittest.IsolatedAsyncioTestCase):
    async def test_completion_stream_and_latest_response_ops(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/responses" and request.method == "POST":
                payload = json.loads(request.content.decode())
                if payload.get("stream"):
                    body = (
                        'data: {"type":"response.output_text.delta","delta":"hel"}\n\n'
                        'data: {"type":"response.output_text.delta","delta":"lo"}\n\n'
                        'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}\n\n'
                        'data: [DONE]\n\n'
                    )
                    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
                return httpx.Response(200, json={
                    "id": "resp_1",
                    "model": payload.get("model"),
                    "status": "completed",
                    "output": [{
                        "type": "message",
                        "content": [{"type": "output_text", "text": "hi"}]}],
                    "usage": {"input_tokens": 4, "output_tokens": 5, "total_tokens": 9},
                })

            if request.url.path == "/v1/responses/resp_1" and request.method == "GET":
                return httpx.Response(200, json={"id": "resp_1", "status": "completed"})
            if request.url.path == "/v1/responses/resp_1" and request.method == "DELETE":
                return httpx.Response(200, json={"id": "resp_1", "deleted": True})
            if request.url.path == "/v1/responses/resp_1/cancel":
                return httpx.Response(200, json={"id": "resp_1", "status": "cancelled"})
            if request.url.path == "/v1/responses/resp_1/input_items":
                return httpx.Response(200, json={"data": [{"type": "message"}]})
            if request.url.path == "/v1/responses/compact":
                return httpx.Response(200, json={"id": "cmp_1", "status": "completed"})
            if request.url.path == "/v1/responses/input_tokens":
                return httpx.Response(200, json={"total_tokens": 12})

            return httpx.Response(404, json={"error": "not-found"})

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": "Bearer sk-test"},
            ops=openai_ops(),
            transport=AsyncTransport(client=hc),
        )
        c = OpenAIClient(ClientConfig(model="gpt-test", api_key="sk-test", base_url="https://api.openai.com/v1"), api=api)
        try:
            res = await c.acomplete(_user("hello"), options=RequestOptions(max_tokens=32, reasoning_effort="medium"))
            self.assertEqual(res.message.content[0].text, "hi")
            self.assertEqual(res.usage.total_tokens, 9)

            txt = []
            done = None
            async for d in c.astream(_user("hello")):
                if d.text: txt.append(d.text)
                if d.finish_reason: done = d
            self.assertEqual("".join(txt), "hello")
            self.assertEqual(done.usage.total_tokens, 5)

            self.assertEqual((await c.aresponse_get("resp_1"))["id"], "resp_1")
            self.assertEqual((await c.aresponse_delete("resp_1"))["deleted"], True)
            self.assertEqual((await c.aresponse_cancel("resp_1"))["status"], "cancelled")
            self.assertEqual((await c.aresponse_input_items("resp_1"))["data"][0]["type"], "message")
            self.assertEqual((await c.acompact(["resp_1"]))["id"], "cmp_1")
            self.assertEqual((await c.ainput_tokens([{"role": "user", "content": "x"}]))["total_tokens"], 12)
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_chat_completions_supported(self):
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode())
            if payload.get("stream"):
                body = (
                    'data: {"choices":[{"delta":{"content":"a"},"finish_reason":null}]}\n\n'
                    'data: {"choices":[{"delta":{"content":"b"},"finish_reason":"stop"}]}\n\n'
                    'data: [DONE]\n\n'
                )
                return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json={
                "model": payload["model"],
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": "Bearer sk-test"},
            ops=openai_ops(),
            transport=AsyncTransport(client=hc),
        )
        c = OpenAIClient(ClientConfig(model="gpt-test", api_key="sk-test", base_url="https://api.openai.com/v1"), api=api)
        try:
            res = await c.achat_complete(_user("hello"))
            self.assertEqual(res.message.content[0].text, "ok")

            out = []
            async for d in c.achat_stream(_user("hello")):
                out.append(d.text)
            self.assertEqual("".join(out), "ab")
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_responses_stream_error_is_structured_apierror(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = (
                'data: {"type":"error","error":{"type":"server_error","code":"overloaded","message":"temporary issue"}}\n\n'
                'data: [DONE]\n\n'
            )
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": "Bearer sk-test"},
            ops=openai_ops(),
            transport=AsyncTransport(client=hc),
        )
        c = OpenAIClient(ClientConfig(model="gpt-test", api_key="sk-test", base_url="https://api.openai.com/v1"), api=api)
        try:
            with self.assertRaises(APIError) as ctx:
                async for _ in c.astream(_user("hello")):
                    pass
            err = ctx.exception
            self.assertEqual(err.provider, "openai")
            self.assertEqual(err.model, "gpt-test")
            self.assertEqual(err.endpoint, "responses.stream")
            self.assertEqual(err.error_type, "server_error")
            self.assertEqual(err.code, "overloaded")
            self.assertTrue(err.retryable)
            self.assertIn("temporary issue", err.message)
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_openai_http_error_surfaces_provider_code(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={
                "error": {
                    "message": "Input is too long",
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded",
                    "param": "input",
                },
            }, headers={"x-request-id": "req_openai_123"})

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": "Bearer sk-test"},
            ops=openai_ops(),
            transport=AsyncTransport(client=hc),
        )
        c = OpenAIClient(ClientConfig(model="gpt-test", api_key="sk-test", base_url="https://api.openai.com/v1"), api=api)
        try:
            with self.assertRaises(APIError) as ctx:
                await c.acomplete(_user("hello"))
            err = ctx.exception
            self.assertEqual(err.status_code, 400)
            self.assertEqual(err.error_type, "invalid_request_error")
            self.assertEqual(err.code, "context_length_exceeded")
            self.assertEqual(err.request_id, "req_openai_123")
            self.assertIn("Input is too long", err.message)
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_chat_stream_usage_from_choice_usage_field(self):
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode())
            if payload.get("stream"):
                body = (
                    'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
                    'data: {"choices":[{"delta":{},"finish_reason":"stop","usage":{"prompt_tokens":14,"completion_tokens":7,"total_tokens":21,"cached_tokens":8}}]}\n\n'
                    'data: [DONE]\n\n'
                )
                return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": "Bearer sk-test"},
            ops=openai_ops(),
            transport=AsyncTransport(client=hc),
        )
        c = OpenAIClient(ClientConfig(model="gpt-test", api_key="sk-test", base_url="https://api.openai.com/v1"), api=api)
        try:
            done = None
            async for d in c.achat_stream(_user("hello"), stream_options={"include_usage": True}):
                if d.finish_reason: done = d
            self.assertIsNotNone(done)
            self.assertIsNotNone(done.usage)
            self.assertEqual(done.usage.prompt_tokens, 14)
            self.assertEqual(done.usage.completion_tokens, 7)
            self.assertEqual(done.usage.total_tokens, 21)
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_kwargs_passthrough_and_cache(self):
        seen = {"payload": None, "headers": None}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["headers"] = dict(request.headers)
            seen["payload"] = json.loads(request.content.decode())
            return httpx.Response(200, json={
                "id": "resp_2",
                "model": "gpt-test",
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            })

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": "Bearer sk-test"},
            ops=openai_ops(),
            transport=AsyncTransport(client=hc),
        )
        c = OpenAIClient(ClientConfig(model="gpt-test", api_key="sk-test", base_url="https://api.openai.com/v1"), api=api)
        try:
            await c.acomplete(_user("hello"),
                temperature=0.2,
                cache=True,
                custom_toggle=True,
                headers={"x-extra": "1"},
                query={"seed": 7})
            self.assertEqual(seen["payload"]["temperature"], 0.2)
            self.assertEqual(seen["payload"]["store"], True)
            self.assertEqual(seen["payload"]["custom_toggle"], True)
            self.assertEqual(seen["headers"]["x-extra"], "1")
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_search_kwarg_removed_use_tools_instead(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": "Bearer sk-test"},
            ops=openai_ops(),
            transport=AsyncTransport(client=hc),
        )
        c = OpenAIClient(ClientConfig(model="gpt-test", api_key="sk-test", base_url="https://api.openai.com/v1"), api=api)
        try:
            with self.assertRaises(TypeError):
                await c.acomplete(_user("hello"), search=True)
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_tools_lisette_style_schema_is_accepted(self):
        seen = {"payload": None}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content.decode())
            return httpx.Response(200, json={
                "id": "resp_tools",
                "model": "gpt-test",
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            })

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": "Bearer sk-test"},
            ops=openai_ops(),
            transport=AsyncTransport(client=hc),
        )
        c = OpenAIClient(ClientConfig(model="gpt-test", api_key="sk-test", base_url="https://api.openai.com/v1"), api=api)
        try:
            tool = {"type": "function", "function": {
                "name": "simple_add",
                "description": "Add two integers",
                "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
            }}
            await c.acomplete(_user("hello"), tools=[tool], tool_choice="required")
            self.assertEqual(seen["payload"]["tools"][0]["type"], "function")
            self.assertEqual(seen["payload"]["tools"][0]["name"], "simple_add")
            self.assertIn("parameters", seen["payload"]["tools"][0])
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_responses_maps_response_format_to_text_format(self):
        seen = {"payload": None}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content.decode())
            return httpx.Response(200, json={
                "id": "resp_3",
                "model": "gpt-test",
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "{\"city\":\"Istanbul\"}"}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            })

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": "Bearer sk-test"},
            ops=openai_ops(),
            transport=AsyncTransport(client=hc),
        )
        c = OpenAIClient(ClientConfig(model="gpt-test", api_key="sk-test", base_url="https://api.openai.com/v1"), api=api)
        try:
            await c.acomplete(_user("hello"), response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "city_country",
                    "schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            })
            self.assertIn("text", seen["payload"])
            self.assertEqual(seen["payload"]["text"]["format"]["type"], "json_schema")
            self.assertEqual(seen["payload"]["text"]["format"]["name"], "city_country")
            self.assertIn("schema", seen["payload"]["text"]["format"])
            self.assertNotIn("response_format", seen["payload"])
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_openai_video_alias_maps_to_input_file(self):
        seen = {"payload": None}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content.decode())
            return httpx.Response(200, json={
                "id": "resp_video",
                "model": "gpt-test",
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            })

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": "Bearer sk-test"},
            ops=openai_ops(),
            transport=AsyncTransport(client=hc),
        )
        c = OpenAIClient(ClientConfig(model="gpt-test", api_key="sk-test", base_url="https://api.openai.com/v1"), api=api)
        msgs = [Msg(role="user", content=[
            Part(type="text", text="Summarize this video"),
            Part(type="input_video", data={"video_url": "https://example.com/demo.mp4", "mimeType": "video/mp4"}),
        ])]
        try:
            await c.acomplete(msgs)
            p = seen["payload"]["input"][0]["content"][1]
            self.assertEqual(p["type"], "input_file")
            self.assertEqual(p["file_url"], "https://example.com/demo.mp4")
            self.assertNotIn("video_url", p)
            self.assertNotIn("mimeType", p)
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_openai_file_data_raw_base64_is_wrapped_as_data_url(self):
        seen = {"payload": None}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content.decode())
            return httpx.Response(200, json={
                "id": "resp_file",
                "model": "gpt-test",
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            })

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": "Bearer sk-test"},
            ops=openai_ops(),
            transport=AsyncTransport(client=hc),
        )
        c = OpenAIClient(ClientConfig(model="gpt-test", api_key="sk-test", base_url="https://api.openai.com/v1"), api=api)
        msg = Msg("user", content=[
            Part(type="text", text="Summarize this PDF"),
            Part(type="input_file", data={
                "filename": "doc.pdf",
                "file_data": "QUJDRA==",
            }),
        ])
        try:
            await c.acomplete([msg])
            fp = seen["payload"]["input"][0]["content"][1]
            self.assertEqual(fp["type"], "input_file")
            self.assertEqual(fp["filename"], "doc.pdf")
            self.assertEqual(fp["file_data"], "data:application/pdf;base64,QUJDRA==")
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_openai_file_data_data_url_is_preserved(self):
        seen = {"payload": None}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content.decode())
            return httpx.Response(200, json={
                "id": "resp_file_2",
                "model": "gpt-test",
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            })

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": "Bearer sk-test"},
            ops=openai_ops(),
            transport=AsyncTransport(client=hc),
        )
        c = OpenAIClient(ClientConfig(model="gpt-test", api_key="sk-test", base_url="https://api.openai.com/v1"), api=api)
        msg = Msg("user", content=[
            Part(type="text", text="Summarize this PDF"),
            Part(type="input_file", data={
                "filename": "doc.pdf",
                "file_data": "data:application/pdf;base64,QUJDRA==",
            }),
        ])
        try:
            await c.acomplete([msg])
            fp = seen["payload"]["input"][0]["content"][1]
            self.assertEqual(fp["file_data"], "data:application/pdf;base64,QUJDRA==")
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_openai_file_data_missing_filename_is_autofilled_for_responses(self):
        seen = {"payload": None}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content.decode())
            return httpx.Response(200, json={
                "id": "resp_file_3",
                "model": "gpt-test",
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            })

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": "Bearer sk-test"},
            ops=openai_ops(),
            transport=AsyncTransport(client=hc),
        )
        c = OpenAIClient(ClientConfig(model="gpt-test", api_key="sk-test", base_url="https://api.openai.com/v1"), api=api)
        msg = Msg("user", content=[
            Part(type="text", text="Summarize this PDF"),
            Part(type="input_file", data={"file_data": "data:application/pdf;base64,QUJDRA=="}),
        ])
        try:
            await c.acomplete([msg])
            fp = seen["payload"]["input"][0]["content"][1]
            self.assertEqual(fp["type"], "input_file")
            self.assertEqual(fp["filename"], "upload.pdf")
            self.assertEqual(fp["file_data"], "data:application/pdf;base64,QUJDRA==")
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_openai_file_data_missing_filename_is_autofilled_for_chat(self):
        seen = {"payload": None}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content.decode())
            return httpx.Response(200, json={
                "model": "gpt-test",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": "Bearer sk-test"},
            ops=openai_ops(),
            transport=AsyncTransport(client=hc),
        )
        c = OpenAIClient(ClientConfig(model="gpt-test", api_key="sk-test", base_url="https://api.openai.com/v1"), api=api)
        msg = Msg("user", content=[
            Part(type="text", text="Summarize this PDF"),
            Part(type="input_file", data={"file_data": "data:application/pdf;base64,QUJDRA=="}),
        ])
        try:
            await c.achat_complete([msg])
            fp = seen["payload"]["messages"][0]["content"][1]
            self.assertEqual(fp["type"], "file")
            self.assertEqual(fp["filename"], "upload.pdf")
            self.assertEqual(fp["file_data"], "data:application/pdf;base64,QUJDRA==")
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_openai_compat_warns_for_non_text_media(self):
        seen = {"payload": None}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content.decode())
            return httpx.Response(200, json={
                "id": "resp_compat",
                "model": "kimi-k2.5",
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            })

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(
            base_url="https://api.moonshot.ai/v1",
            headers={"Authorization": "Bearer sk-test"},
            ops=openai_ops(),
            transport=AsyncTransport(client=hc),
        )
        cfg = ClientConfig(model="kimi-k2.5", api_key="sk-test", base_url="https://api.moonshot.ai/v1", provider="openai_compat")
        c = OpenAIClient(cfg, api=api)
        msg = Msg("user", content=[
            Part(type="text", text="Describe this image"),
            Part(type="input_image", data={"image_url": "https://example.com/cat.png"}),
        ])
        try:
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                await c.acomplete([msg])
            txts = [str(w.message) for w in rec]
            self.assertTrue(any("OpenAI-compatible model" in t and "input_image" in t for t in txts))
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_chat_payload_supports_assistant_tool_calls_and_tool_messages(self):
        seen = {"payload": None}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content.decode())
            return httpx.Response(200, json={
                "model": "kimi-k2.5",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(
            base_url="https://api.moonshot.ai/v1",
            headers={"Authorization": "Bearer sk-test"},
            ops=openai_ops(),
            transport=AsyncTransport(client=hc),
        )
        cfg = ClientConfig(model="kimi-k2.5", api_key="sk-test", base_url="https://api.moonshot.ai/v1", provider="openai_compat")
        c = OpenAIClient(cfg, api=api)
        msgs = [
            Msg(role="user", content=[Part(type="text", text="calc")]),
            Msg(role="assistant", content=[], data={
                "tool_calls": [{"id": "call_1", "name": "simple_add", "arguments": {"a": 1, "b": 2}}],
                "openai_chat": {"reasoning_content": "hidden-thought"},
            }),
            Msg(role="tool", content=[Part(type="text", text="3")], data={"tool_call_id": "call_1", "name": "simple_add"}),
        ]
        try:
            await c.achat_complete(msgs)
            mm = seen["payload"]["messages"]
            self.assertEqual(mm[1]["role"], "assistant")
            self.assertEqual(mm[1]["tool_calls"][0]["id"], "call_1")
            self.assertEqual(mm[1]["tool_calls"][0]["function"]["name"], "simple_add")
            self.assertEqual(mm[1]["reasoning_content"], "hidden-thought")
            self.assertEqual(mm[2]["role"], "tool")
            self.assertEqual(mm[2]["tool_call_id"], "call_1")
            self.assertEqual(mm[2]["content"], "3")
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_responses_payload_supports_assistant_tool_calls_and_tool_messages(self):
        seen = {"payload": None}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content.decode())
            return httpx.Response(200, json={
                "id": "resp_tools_replay",
                "model": "gpt-test",
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            })

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": "Bearer sk-test"},
            ops=openai_ops(),
            transport=AsyncTransport(client=hc),
        )
        c = OpenAIClient(ClientConfig(model="gpt-test", api_key="sk-test", base_url="https://api.openai.com/v1"), api=api)
        msgs = [
            Msg(role="user", content=[Part(type="text", text="calc")]),
            Msg(role="assistant", content=[], data={"tool_calls": [{"id": "call_2", "name": "simple_add", "arguments": {"a": 2, "b": 3}}]}),
            Msg(role="tool", content=[Part(type="text", text="5")], data={"tool_call_id": "call_2", "name": "simple_add"}),
        ]
        try:
            await c.acomplete(msgs)
            inp = seen["payload"]["input"]
            fc = [it for it in inp if isinstance(it, dict) and it.get("type") == "function_call"]
            fo = [it for it in inp if isinstance(it, dict) and it.get("type") == "function_call_output"]
            self.assertEqual(len(fc), 1)
            self.assertEqual(fc[0]["call_id"], "call_2")
            self.assertEqual(fc[0]["name"], "simple_add")
            self.assertEqual(json.loads(fc[0]["arguments"]), {"a": 2, "b": 3})
            self.assertEqual(len(fo), 1)
            self.assertEqual(fo[0]["call_id"], "call_2")
            self.assertEqual(fo[0]["output"], "5")
        finally:
            await c.aclose()
            await hc.aclose()
