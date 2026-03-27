import unittest

from fastllm_v2 import AnthropicClient, GeminiClient, OpenAIClient, mk_client


class TestFactoryAndPatches(unittest.TestCase):
    def test_factory(self):
        self.assertIsInstance(mk_client("openai", model="gpt-test", api_key="k"), OpenAIClient)
        self.assertIsInstance(mk_client("openai_compat", model="kimi-k2.5", api_key="k"), OpenAIClient)
        self.assertIsInstance(mk_client("anthropic", model="claude-test", api_key="k"), AnthropicClient)
        self.assertIsInstance(mk_client("gemini", model="gemini-test", api_key="k"), GeminiClient)

    def test_unknown_family(self):
        with self.assertRaises(ValueError):
            mk_client("bogus", model="x")
