import inspect
import unittest

from fastllm_v2 import AnthropicClient, GeminiClient, OpenAIClient, acompletion, astream


class TestDelegatedSignatures(unittest.TestCase):
    def test_acompletion_signature_has_request_option_kwargs(self):
        sig = inspect.signature(acompletion)
        for nm in ("max_tokens", "temperature", "cache", "tools", "tool_choice"):
            self.assertIn(nm, sig.parameters)
        self.assertIn("stream", sig.parameters)
        self.assertIn("options", sig.parameters)
        self.assertIn("kwargs", sig.parameters)

    def test_astream_signature_delegates_from_acompletion(self):
        sig = inspect.signature(astream)
        for nm in ("api_key", "base_url", "provider", "max_tokens", "tools"):
            self.assertIn(nm, sig.parameters)
        self.assertNotIn("stream", sig.parameters)
        self.assertIn("kwargs", sig.parameters)

    def test_client_method_signatures_show_option_fields(self):
        fns = [
            OpenAIClient.acomplete,
            OpenAIClient.astream,
            OpenAIClient.achat_complete,
            OpenAIClient.achat_stream,
            AnthropicClient.acomplete,
            AnthropicClient.astream,
            GeminiClient.acomplete,
            GeminiClient.astream,
        ]
        for fn in fns:
            sig = inspect.signature(fn)
            self.assertIn("max_tokens", sig.parameters)
            self.assertIn("temperature", sig.parameters)
            self.assertIn("options", sig.parameters)
            self.assertIn("kwargs", sig.parameters)
