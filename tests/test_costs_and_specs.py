import unittest

from fastllm_v2 import Completion, CostBreakdown, ModelPrice, Msg, Part, Usage, estimate_cost
from fastllm_v2.builtin_specs import anthropic_ops, gemini_ops, openai_ops


def _keys(xs): return {(o.group, o.name) for o in xs}


class TestCostsAndSpecs(unittest.TestCase):
    def test_estimate_cost_with_cached_tokens(self):
        comp = Completion(
            model="gpt-foo-1",
            message=Msg(role="assistant", content=[Part(type="text", text="ok")]),
            usage=Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500,
                raw={"cached_input_tokens": 400}),
        )
        out = estimate_cost(comp, prices={
            "gpt-foo-*": ModelPrice(prompt_per_million=2.0, completion_per_million=8.0, cached_prompt_per_million=0.5)
        })
        self.assertIsInstance(out, CostBreakdown)
        self.assertEqual(out.cached_prompt_tokens, 400)
        self.assertAlmostEqual(out.prompt_cost, (600 / 1_000_000) * 2.0)
        self.assertAlmostEqual(out.cached_prompt_cost, (400 / 1_000_000) * 0.5)
        self.assertAlmostEqual(out.completion_cost, (500 / 1_000_000) * 8.0)

    def test_estimate_cost_non_strict_missing_price(self):
        out = estimate_cost({"input_tokens": 10, "output_tokens": 5, "model": "unknown"}, strict=False)
        self.assertEqual(out.total_cost, 0.0)

    def test_builtin_specs_cover_more_endpoints(self):
        ok = _keys(openai_ops())
        self.assertIn(("responses", "create"), ok)
        self.assertIn(("chat", "create_completions"), ok)
        self.assertIn(("embeddings", "create"), ok)
        self.assertIn(("images", "generate"), ok)
        self.assertIn(("audio", "speech"), ok)

        ak = _keys(anthropic_ops())
        self.assertIn(("messages", "create"), ak)
        self.assertIn(("messages", "count_tokens"), ak)
        self.assertIn(("files", "list"), ak)

        gk = _keys(gemini_ops())
        self.assertIn(("models", "generate_content"), gk)
        self.assertIn(("models", "count_tokens"), gk)
        self.assertIn(("cached_contents", "create"), gk)
