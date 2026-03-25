import unittest

from fastllm_v2 import AnthropicClient, GeminiClient, OpenAIClient, mk_client


class TestFactoryAndPatches(unittest.TestCase):
    def test_factory(self):
        self.assertIsInstance(mk_client("openai", model="gpt-test", api_key="k"), OpenAIClient)
        self.assertIsInstance(mk_client("anthropic", model="claude-test", api_key="k"), AnthropicClient)
        self.assertIsInstance(mk_client("gemini", model="gemini-test", api_key="k"), GeminiClient)

    def test_openai_provider_patch(self):
        c = mk_client("openai", model="moonshot-v1-8k", api_key="k", provider="moonshot")
        self.assertEqual(c.config.provider, "moonshot")
        self.assertEqual(c.config.base_url, "https://api.moonshot.cn/v1")

    def test_unknown_family(self):
        with self.assertRaises(ValueError):
            mk_client("bogus", model="x")
