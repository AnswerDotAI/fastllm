import unittest
import os
from unittest.mock import patch

import httpx

from fastllm_v2 import APIError, Completion, Delta, Msg, Part, StreamSummary, ToolCall, acompletion, acollect_stream, infer_provider, mk_auto_client


def _resp_404():
    req = httpx.Request("POST", "https://example.test/v1/responses")
    rsp = httpx.Response(404, request=req, text="Not Found")
    return httpx.HTTPStatusError("404", request=req, response=rsp)


def _comp(text, provider="openai"):
    msg = Msg(role="assistant", content=[Part(type="text", text=text)])
    return Completion(model="test-model", provider=provider, message=msg)


class _FakeOpenAI:
    def __init__(self, *, fail_responses=False, fail_responses_stream=False):
        self.calls,self.closed,self.last_messages = [],False,None
        self.fail_responses,self.fail_responses_stream = fail_responses,fail_responses_stream
        self.config = type("Cfg", (), dict(timeout=60.0, provider="openai"))()

    async def acomplete(self, messages, *, options=None, **kwargs):
        self.last_messages = messages
        self.calls.append("responses")
        if self.fail_responses: raise _resp_404()
        return _comp("responses-ok")

    async def achat_complete(self, messages, *, options=None, **kwargs):
        self.last_messages = messages
        self.calls.append("chat")
        return _comp("chat-ok", provider="openai_compat")

    async def astream(self, messages, *, options=None, **kwargs):
        self.last_messages = messages
        self.calls.append("responses_stream")
        if self.fail_responses_stream: raise _resp_404()
        yield Delta(text="r")
        yield Delta(finish_reason="stop")

    async def achat_stream(self, messages, *, options=None, **kwargs):
        self.last_messages = messages
        self.calls.append("chat_stream")
        yield Delta(text="c")
        yield Delta(finish_reason="stop")

    async def aclose(self): self.closed = True


class _FakeAnthropic:
    def __init__(self):
        self.calls,self.closed = [],False
        self.config = type("Cfg", (), dict(timeout=60.0, provider="anthropic"))()

    async def acomplete(self, messages, *, options=None, **kwargs):
        self.calls.append("anthropic_complete")
        return _comp("anthropic-ok", provider="anthropic")

    async def astream(self, messages, *, options=None, **kwargs):
        self.calls.append("anthropic_stream")
        yield Delta(text="a")
        yield Delta(finish_reason="stop")

    async def aclose(self): self.closed = True


class TestInferProvider(unittest.TestCase):
    def test_infer_provider_from_model_prefix(self):
        self.assertEqual(infer_provider("claude-sonnet-4-5"), "anthropic")
        self.assertEqual(infer_provider("gemini-2.5-flash"), "gemini")
        self.assertEqual(infer_provider("gpt-5-mini"), "openai")
        self.assertEqual(infer_provider("kimi-k2.5"), "openai_compat")

    def test_infer_provider_from_base_url(self):
        self.assertEqual(infer_provider("x", base_url="https://api.anthropic.com"), "anthropic")
        self.assertEqual(infer_provider("x", base_url="https://generativelanguage.googleapis.com/v1beta"), "gemini")
        self.assertEqual(infer_provider("x", base_url="https://api.openai.com/v1"), "openai")
        self.assertEqual(infer_provider("x", base_url="https://api.moonshot.ai/v1"), "openai_compat")

    def test_explicit_provider_is_ignored(self):
        self.assertEqual(infer_provider("claude-sonnet-4-5", provider="gemini"), "anthropic")
        self.assertEqual(infer_provider("gpt-5-mini", provider="moonshot"), "openai")


class TestAutoClientDefaults(unittest.TestCase):
    def test_model_only_kimi_resolves_moonshot_defaults(self):
        with patch.dict(os.environ, {"MOONSHOT_API_KEY": "msk-test"}, clear=True):
            c = mk_auto_client("kimi-k2.5")
            self.addCleanup(lambda: __import__("asyncio").run(c.aclose()))
            self.assertEqual(c.config.provider, "openai_compat")
            self.assertEqual(c.config.base_url, "https://api.moonshot.ai/v1")
            self.assertEqual(c.config.api_key, "msk-test")
            self.assertEqual(c.config.model, "kimi-k2.5")

    def test_vendor_prefix_model_uses_vendor_env_and_strips_prefix(self):
        env = {
            "QWEN_API_KEY": "qwen-test",
            "QWEN_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        }
        with patch.dict(os.environ, env, clear=True):
            c = mk_auto_client("qwen/qwen-plus")
            self.addCleanup(lambda: __import__("asyncio").run(c.aclose()))
            self.assertEqual(c.config.provider, "openai_compat")
            self.assertEqual(c.config.base_url, env["QWEN_BASE_URL"])
            self.assertEqual(c.config.api_key, env["QWEN_API_KEY"])
            self.assertEqual(c.config.model, "qwen-plus")

    def test_provider_prefix_is_stripped_for_native_families(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            c = mk_auto_client("openai/gpt-5-mini")
            self.addCleanup(lambda: __import__("asyncio").run(c.aclose()))
            self.assertEqual(c.config.provider, "openai")
            self.assertEqual(c.config.model, "gpt-5-mini")

    def test_unknown_vendor_prefix_uses_generic_env_convention(self):
        env = {
            "ACME_API_KEY": "acme-key",
            "ACME_BASE_URL": "https://llm.acme.ai/v1",
        }
        with patch.dict(os.environ, env, clear=True):
            c = mk_auto_client("acme/custom-model")
            self.addCleanup(lambda: __import__("asyncio").run(c.aclose()))
            self.assertEqual(c.config.provider, "openai_compat")
            self.assertEqual(c.config.model, "custom-model")
            self.assertEqual(c.config.base_url, env["ACME_BASE_URL"])
            self.assertEqual(c.config.api_key, env["ACME_API_KEY"])


class TestHighLevelAsync(unittest.IsolatedAsyncioTestCase):
    async def test_openai_auto_falls_back_to_chat_when_responses_missing(self):
        fake = _FakeOpenAI(fail_responses=True)
        with patch("fastllm_v2.highlevel.mk_auto_client", return_value=fake):
            res = await acompletion("gpt-5-mini", [dict(role="user", content="hi")], api_key="k", base_url="https://api.openai.com/v1")
        self.assertEqual(res.message.content[0].text, "chat-ok")
        self.assertEqual(fake.calls, ["responses", "chat"])
        self.assertTrue(fake.closed)

    async def test_openai_compat_auto_uses_chat_only(self):
        fake = _FakeOpenAI(fail_responses=True)
        with patch("fastllm_v2.highlevel.mk_auto_client", return_value=fake):
            res = await acompletion("kimi-k2.5", [dict(role="user", content="hi")], api_key="k", base_url="https://api.moonshot.ai/v1")
        self.assertEqual(res.message.content[0].text, "chat-ok")
        self.assertEqual(fake.calls, ["chat"])
        self.assertTrue(fake.closed)

    async def test_openai_auto_falls_back_on_apierror_404(self):
        class _FakeOpenAIApiErr(_FakeOpenAI):
            async def acomplete(self, messages, *, options=None, **kwargs):
                self.calls.append("responses")
                raise APIError("responses endpoint not found", provider="openai", endpoint="responses.create", status_code=404)

        fake = _FakeOpenAIApiErr()
        with patch("fastllm_v2.highlevel.mk_auto_client", return_value=fake):
            res = await acompletion("gpt-5-mini", [dict(role="user", content="hi")], api_key="k", base_url="https://api.openai.com/v1")
        self.assertEqual(res.message.content[0].text, "chat-ok")
        self.assertEqual(fake.calls, ["responses", "chat"])
        self.assertTrue(fake.closed)

    async def test_stream_fallback_from_responses_to_chat(self):
        fake = _FakeOpenAI(fail_responses_stream=True)
        with patch("fastllm_v2.highlevel.mk_auto_client", return_value=fake):
            it = await acompletion("gpt-5-mini", [dict(role="user", content="stream")], api_key="k", base_url="https://api.openai.com/v1", stream=True)
            out = []
            async for d in it:
                if d.text: out.append(d.text)
        self.assertEqual("".join(out), "c")
        self.assertEqual(fake.calls, ["responses_stream", "chat_stream"])
        self.assertTrue(fake.closed)

    async def test_stream_fallback_with_unread_error_response(self):
        class _FakeOpenAIUnread(_FakeOpenAI):
            async def astream(self, messages, *, options=None, **kwargs):
                self.calls.append("responses_stream")
                req = httpx.Request("POST", "https://example.test/v1/responses")
                rsp = httpx.Response(404, request=req, stream=httpx.ByteStream(b"Not Found"))
                raise httpx.HTTPStatusError("404", request=req, response=rsp)
                yield Delta()  # pragma: no cover

        fake = _FakeOpenAIUnread()
        with patch("fastllm_v2.highlevel.mk_auto_client", return_value=fake):
            it = await acompletion("gpt-5-mini", [dict(role="user", content="stream")], api_key="k", base_url="https://api.openai.com/v1", stream=True)
            out = []
            async for d in it:
                if d.text: out.append(d.text)
        self.assertEqual("".join(out), "c")
        self.assertEqual(fake.calls, ["responses_stream", "chat_stream"])
        self.assertTrue(fake.closed)

    async def test_acollect_stream_accepts_acompletion_coroutine(self):
        fake = _FakeOpenAI()
        with patch("fastllm_v2.highlevel.mk_auto_client", return_value=fake):
            summary = await acollect_stream(acompletion(
                "gpt-5-mini",
                [dict(role="user", content="stream")],
                api_key="k",
                base_url="https://api.openai.com/v1",
                stream=True,
            ))
        self.assertEqual(summary.text, "r")
        self.assertEqual(summary.finish_reason, "stop")
        self.assertEqual(summary.final.text, "r")
        self.assertEqual(fake.calls, ["responses_stream"])
        self.assertTrue(fake.closed)

    async def test_anthropic_path_from_model(self):
        seen,fake = dict(family=None),_FakeAnthropic()

        def _builder(model, **kwargs):
            seen["family"] = infer_provider(model, provider=kwargs.get("provider", ""), base_url=kwargs.get("base_url", ""))
            return fake

        with patch("fastllm_v2.highlevel.mk_auto_client", side_effect=_builder):
            res = await acompletion("claude-sonnet-4-5", [dict(role="user", content="hi")], api_key="k")
        self.assertEqual(seen["family"], "anthropic")
        self.assertEqual(res.message.content[0].text, "anthropic-ok")
        self.assertEqual(fake.calls, ["anthropic_complete"])
        self.assertTrue(fake.closed)

    async def test_streamsummary_and_tool_msg_are_coerced_for_chat_toolloop(self):
        fake = _FakeOpenAI()
        ss = StreamSummary(
            text="",
            tool_calls=[ToolCall(id="call_1", name="simple_add", arguments={"a": 1, "b": 2})],
            raw_events=[
                {"choices": [{"delta": {"reasoning_content": "think-1 "}}]},
                {"choices": [{"delta": {"reasoning_content": "think-2"}}]},
            ],
        )
        msgs = [
            Msg(role="user", content=[Part(type="text", text="calc")]),
            ss,
            Msg(role="tool", content=[Part(type="text", text="3")], data={"tool_call_id": "call_1", "name": "simple_add"}),
        ]
        with patch("fastllm_v2.highlevel.mk_auto_client", return_value=fake):
            _ = await acompletion("kimi-k2.5", msgs, api_key="k", base_url="https://api.moonshot.ai/v1")

        self.assertEqual(fake.calls, ["chat"])
        self.assertEqual(fake.last_messages[1].role, "assistant")
        self.assertEqual(fake.last_messages[1].data["tool_calls"][0]["id"], "call_1")
        self.assertEqual(fake.last_messages[1].data["openai_chat"]["reasoning_content"], "think-1 think-2")
        self.assertEqual(fake.last_messages[2].role, "tool")
        self.assertEqual(fake.last_messages[2].data["tool_call_id"], "call_1")

    async def test_completion_is_coerced_with_tool_calls_and_reasoning_meta(self):
        fake = _FakeOpenAI()
        comp = Completion(
            model="kimi-k2.5",
            provider="openai_compat",
            message=Msg(role="assistant", content=[Part(type="text", text="calling tool")]),
            tool_calls=[ToolCall(id="call_2", name="simple_add", arguments={"a": 3, "b": 4})],
            raw={"choices": [{"message": {"reasoning_content": "short-reasoning"}}]},
        )
        msgs = [
            Msg(role="user", content=[Part(type="text", text="calc")]),
            comp,
            Msg(role="tool", content=[Part(type="text", text="7")], data={"tool_call_id": "call_2", "name": "simple_add"}),
        ]
        with patch("fastllm_v2.highlevel.mk_auto_client", return_value=fake):
            _ = await acompletion("kimi-k2.5", msgs, api_key="k", base_url="https://api.moonshot.ai/v1")

        self.assertEqual(fake.calls, ["chat"])
        self.assertEqual(fake.last_messages[1].role, "assistant")
        self.assertEqual(fake.last_messages[1].data["tool_calls"][0]["name"], "simple_add")
        self.assertEqual(fake.last_messages[1].data["openai_chat"]["reasoning_content"], "short-reasoning")
