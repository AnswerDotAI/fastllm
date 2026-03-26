import json, unittest

import httpx

from fastllm_v2 import AnthropicClient, ClientConfig, Delta, GeminiClient, Msg, OpenAIClient, Part, ToolCall, acollect_stream
from fastllm_v2.builtin_specs import anthropic_ops, gemini_ops, openai_ops
from fastllm_v2.oapi import OpenAPIClient
from fastllm_v2.transport import AsyncTransport


def _user(s): return [Msg(role="user", content=[Part(type="text", text=s)])]


async def _agen(xs):
    for x in xs:
        yield x


class TestStreamingLossless(unittest.IsolatedAsyncioTestCase):
    async def test_openai_stream_preserves_unknown_events_and_collects(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/responses":
                payload = json.loads(request.content.decode())
                if payload.get("stream"):
                    body = (
                        'data: {"type":"response.created","response":{"id":"resp_1"}}\n\n'
                        'data: {"type":"response.output_text.delta","delta":"1"}\n\n'
                        'data: {"type":"response.output_text.delta","delta":", 2"}\n\n'
                        'data: {"type":"response.output_text.delta","delta":", 3"}\n\n'
                        'data: {"type":"response.output_item.added","item":{"type":"reasoning","summary":[{"type":"summary_text","text":"meta"}]}}\n\n'
                        'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}\n\n'
                        'data: [DONE]\n\n')
                    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
            return httpx.Response(404, json={"error": "not-found"})

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(base_url="https://api.openai.com/v1", headers={"Authorization": "Bearer sk-test"},
            ops=openai_ops(), transport=AsyncTransport(client=hc))
        c = OpenAIClient(ClientConfig(model="gpt-test", api_key="sk-test", base_url="https://api.openai.com/v1"), api=api)
        try:
            summary = await acollect_stream(c.astream(_user("Count from 1 to 3.")))
            self.assertEqual(summary.text, "1, 2, 3")
            self.assertEqual(summary.finish_reason, "completed")
            self.assertEqual(summary.usage.total_tokens, 5)
            self.assertEqual(summary.final.raw["last_event"]["type"], "response.completed")
            self.assertEqual(len(summary.final.raw["events"]), len(summary.raw_events))
            self.assertEqual(summary.raw_events[0]["type"], "response.created")
            self.assertTrue(any(ev.get("type") == "response.output_item.added" for ev in summary.raw_events))
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_anthropic_stream_preserves_message_start_and_message_stop(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/messages":
                payload = json.loads(request.content.decode())
                if payload.get("stream"):
                    body = (
                        'data: {"type":"message_start","message":{"id":"msg_1"}}\n\n'
                        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"ok"}}\n\n'
                        'data: {"type":"message_stop"}\n\n')
                    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
            return httpx.Response(404, json={"error": "not-found"})

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(base_url="https://api.anthropic.com", headers={"x-api-key": "k"},
            ops=anthropic_ops(), transport=AsyncTransport(client=hc))
        c = AnthropicClient(ClientConfig(model="claude-sonnet-4-5", api_key="k", base_url="https://api.anthropic.com"), api=api)
        try:
            summary = await acollect_stream(c.astream(_user("Say ok")))
            self.assertEqual(summary.text, "ok")
            self.assertEqual(summary.raw_events[0]["type"], "message_start")
            self.assertEqual(summary.finish_reason, "message_stop")
            self.assertEqual(summary.deltas[-1].finish_reason, "message_stop")
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_gemini_stream_preserves_metadata_only_events(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith(":streamGenerateContent"):
                body = (
                    'data: {"promptFeedback":{"blockReason":"OTHER"}}\n\n'
                    'data: {"candidates":[{"content":{"parts":[{"text":"hello"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3}}\n\n')
                return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
            return httpx.Response(404, json={"error": "not-found"})

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(base_url="https://generativelanguage.googleapis.com/v1beta", ops=gemini_ops(),
            transport=AsyncTransport(client=hc))
        c = GeminiClient(ClientConfig(model="gemini-2.5-pro", api_key="g", base_url="https://generativelanguage.googleapis.com/v1beta"), api=api)
        try:
            summary = await acollect_stream(c.astream(_user("Say hello")))
            self.assertEqual(summary.raw_events[0]["promptFeedback"]["blockReason"], "OTHER")
            self.assertEqual(summary.text, "hello")
            self.assertEqual(summary.finish_reason, "STOP")
            self.assertEqual(summary.usage.total_tokens, 3)
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_acollect_stream_normalizes_anthropic_chunked_tool_calls(self):
        ds = [
            Delta(tool_calls=[ToolCall(id="toolu_01", name="simple_add", arguments={})],
                raw={"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_01", "name": "simple_add", "input": {}}}),
            Delta(tool_calls=[ToolCall(id="1", name="", arguments={"_delta": '{"a": 547'})],
                raw={"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"a": 547'}}),
            Delta(tool_calls=[ToolCall(id="1", name="", arguments={"_delta": '8954793'})],
                raw={"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '8954793'}}),
            Delta(tool_calls=[ToolCall(id="1", name="", arguments={"_delta": ', "b": 5479'})],
                raw={"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ', "b": 5479'}}),
            Delta(tool_calls=[ToolCall(id="1", name="", arguments={"_delta": '82745}'})],
                raw={"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '82745}'}}),
            Delta(tool_calls=[ToolCall(id="toolu_02", name="simple_add", arguments={})],
                raw={"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_02", "name": "simple_add", "input": {}}}),
            Delta(tool_calls=[ToolCall(id="2", name="", arguments={"_delta": '{"a":5479749754, "b":9875438979}'})],
                raw={"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"a":5479749754, "b":9875438979}'}}),
            Delta(finish_reason="message_stop", raw={"type": "message_stop"}),
        ]
        summary = await acollect_stream(_agen(ds))
        self.assertEqual(len(summary.final.tool_calls), 2)
        self.assertEqual(summary.final.tool_calls[0].id, "toolu_01")
        self.assertEqual(summary.final.tool_calls[0].name, "simple_add")
        self.assertEqual(summary.final.tool_calls[0].arguments, {"a": 5478954793, "b": 547982745})
        self.assertEqual(summary.final.tool_calls[1].id, "toolu_02")
        self.assertEqual(summary.final.tool_calls[1].name, "simple_add")
        self.assertEqual(summary.final.tool_calls[1].arguments, {"a": 5479749754, "b": 9875438979})

    async def test_acollect_stream_normalizes_openai_chunked_tool_calls(self):
        ds = [
            Delta(raw={"type": "response.output_item.added", "item": {"type": "function_call", "call_id": "fc_1", "name": "simple_add"}}),
            Delta(tool_calls=[ToolCall(id="fc_1", name="", arguments={"_delta": '{"a":5478954793'})],
                raw={"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": '{"a":5478954793'}),
            Delta(tool_calls=[ToolCall(id="fc_1", name="", arguments={"_delta": ',"b":547982745}'})],
                raw={"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": ',"b":547982745}'}),
            Delta(raw={"type": "response.output_item.added", "item": {"type": "function_call", "call_id": "fc_2", "name": "simple_add"}}),
            Delta(tool_calls=[ToolCall(id="fc_2", name="", arguments={"_delta": '{"a":5479749754,"b":9875438979}'})],
                raw={"type": "response.function_call_arguments.delta", "item_id": "fc_2", "delta": '{"a":5479749754,"b":9875438979}'}),
            Delta(finish_reason="completed", raw={"type": "response.completed"}),
        ]
        summary = await acollect_stream(_agen(ds))
        self.assertEqual(len(summary.final.tool_calls), 2)
        self.assertEqual(summary.final.tool_calls[0].id, "fc_1")
        self.assertEqual(summary.final.tool_calls[0].name, "simple_add")
        self.assertEqual(summary.final.tool_calls[0].arguments, {"a": 5478954793, "b": 547982745})
        self.assertEqual(summary.final.tool_calls[1].id, "fc_2")
        self.assertEqual(summary.final.tool_calls[1].name, "simple_add")
        self.assertEqual(summary.final.tool_calls[1].arguments, {"a": 5479749754, "b": 9875438979})

    async def test_openai_responses_call_id_and_item_id_are_collated(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/responses":
                payload = json.loads(request.content.decode())
                if payload.get("stream"):
                    body = (
                        'data: {"type":"response.output_item.added","item":{"type":"function_call","id":"fc_abc","call_id":"call_abc","name":"simple_add"}}\n\n'
                        'data: {"type":"response.function_call_arguments.delta","item_id":"fc_abc","delta":"{\\"a\\":5478954793"}\n\n'
                        'data: {"type":"response.function_call_arguments.delta","item_id":"fc_abc","delta":",\\"b\\":547982745}"}\n\n'
                        'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
                        'data: [DONE]\n\n')
                    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
            return httpx.Response(404, json={"error": "not-found"})

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(base_url="https://api.openai.com/v1", headers={"Authorization": "Bearer sk-test"},
            ops=openai_ops(), transport=AsyncTransport(client=hc))
        c = OpenAIClient(ClientConfig(model="gpt-test", api_key="sk-test", base_url="https://api.openai.com/v1"), api=api)
        try:
            summary = await acollect_stream(c.astream(_user("Use tool")))
            self.assertEqual(len(summary.tool_calls), 1)
            self.assertEqual(summary.tool_calls[0].id, "call_abc")
            self.assertEqual(summary.tool_calls[0].name, "simple_add")
            self.assertEqual(summary.tool_calls[0].arguments, {"a": 5478954793, "b": 547982745})
        finally:
            await c.aclose()
            await hc.aclose()

    async def test_chat_stream_chunked_tool_calls_collate_by_index(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/chat/completions":
                payload = json.loads(request.content.decode())
                if payload.get("stream"):
                    body = (
                        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"simple_add:0","type":"function","function":{"name":"simple_add","arguments":""}}]},"finish_reason":null}]}\n\n'
                        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"type":"function","function":{"arguments":"{\\"a\\":5478954793"}}]},"finish_reason":null}]}\n\n'
                        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"type":"function","function":{"arguments":",\\"b\\":547982745}"}}]},"finish_reason":null}]}\n\n'
                        'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"simple_add:1","type":"function","function":{"name":"simple_add","arguments":""}}]},"finish_reason":null}]}\n\n'
                        'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"type":"function","function":{"arguments":"{\\"a\\":5479749754"}}]},"finish_reason":null}]}\n\n'
                        'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"type":"function","function":{"arguments":",\\"b\\":9875438979}"}}]},"finish_reason":null}]}\n\n'
                        'data: {"choices":[{"delta":{},"finish_reason":"stop","usage":{"prompt_tokens":10,"completion_tokens":10,"total_tokens":20}}]}\n\n'
                        'data: [DONE]\n\n')
                    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
            return httpx.Response(404, json={"error": "not-found"})

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(base_url="https://api.openai.com/v1", headers={"Authorization": "Bearer sk-test"},
            ops=openai_ops(), transport=AsyncTransport(client=hc))
        c = OpenAIClient(ClientConfig(model="gpt-test", api_key="sk-test", base_url="https://api.openai.com/v1"), api=api)
        try:
            summary = await acollect_stream(c.achat_stream(_user("Use tool"), stream_options={"include_usage": True}))
            self.assertEqual(summary.finish_reason, "stop")
            self.assertEqual(summary.usage.total_tokens, 20)
            self.assertEqual(len(summary.tool_calls), 2)
            self.assertEqual(summary.tool_calls[0].id, "simple_add:0")
            self.assertEqual(summary.tool_calls[0].name, "simple_add")
            self.assertEqual(summary.tool_calls[0].arguments, {"a": 5478954793, "b": 547982745})
            self.assertEqual(summary.tool_calls[1].id, "simple_add:1")
            self.assertEqual(summary.tool_calls[1].name, "simple_add")
            self.assertEqual(summary.tool_calls[1].arguments, {"a": 5479749754, "b": 9875438979})
        finally:
            await c.aclose()
            await hc.aclose()
