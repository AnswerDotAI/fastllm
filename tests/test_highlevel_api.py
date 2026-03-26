import unittest
from unittest.mock import patch

import httpx

from fastllm_v2 import Completion, Delta, Msg, Part, acompletion, acollect_stream, infer_provider


def _resp_404():
    req = httpx.Request("POST", "https://example.test/v1/responses")
    rsp = httpx.Response(404, request=req, text="Not Found")
    return httpx.HTTPStatusError("404", request=req, response=rsp)


def _comp(text, provider="openai"):
    msg = Msg(role="assistant", content=[Part(type="text", text=text)])
    return Completion(model="test-model", provider=provider, message=msg)


class _FakeOpenAI:
    def __init__(self, *, fail_responses=False, fail_responses_stream=False):
        self.calls,self.closed = [],False
        self.fail_responses,self.fail_responses_stream = fail_responses,fail_responses_stream
        self.config = type("Cfg", (), dict(timeout=60.0, provider="openai"))()

    async def acomplete(self, messages, *, options=None, **kwargs):
        self.calls.append("responses")
        if self.fail_responses: raise _resp_404()
        return _comp("responses-ok")

    async def achat_complete(self, messages, *, options=None, **kwargs):
        self.calls.append("chat")
        return _comp("chat-ok", provider="openai_compat")

    async def astream(self, messages, *, options=None, **kwargs):
        self.calls.append("responses_stream")
        if self.fail_responses_stream: raise _resp_404()
        yield Delta(text="r")
        yield Delta(finish_reason="stop")

    async def achat_stream(self, messages, *, options=None, **kwargs):
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

    def test_explicit_provider_wins(self):
        self.assertEqual(infer_provider("claude-sonnet-4-5", provider="gemini"), "gemini")
        self.assertEqual(infer_provider("gpt-5-mini", provider="moonshot"), "openai_compat")


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

