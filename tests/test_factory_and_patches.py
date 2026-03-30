import unittest

from fastllm import AnthropicClient, GeminiClient, OpenAIClient, mk_auto_client


class TestFactoryAndPatches(unittest.TestCase):
    def test_auto_factory(self):
        self.assertIsInstance(mk_auto_client("gpt-test", api_key="k"), OpenAIClient)
        self.assertIsInstance(mk_auto_client("kimi-k2.5", api_key="k"), OpenAIClient)
        self.assertIsInstance(mk_auto_client("claude-test", api_key="k"), AnthropicClient)
        self.assertIsInstance(mk_auto_client("gemini-test", api_key="k"), GeminiClient)

    def test_unknown_family_defaults_to_openai_compat(self):
        self.assertIsInstance(mk_auto_client("bogus-model", api_key="k"), OpenAIClient)
