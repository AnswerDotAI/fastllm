import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from fastllm.specsync import ANTHROPIC_STATS_URL, GEMINI_DISCOVERY_URL, OPENAI_SPEC_URL, extract_anthropic_openapi_url
from fastllm.specsync import sync_specs


def _resp(url: str, *, text: str = "", content_type: str = "text/plain") -> httpx.Response:
    req = httpx.Request("GET", url)
    return httpx.Response(200, request=req, text=text, headers={"content-type": content_type})


class TestSpecSync(unittest.TestCase):
    def test_extract_anthropic_openapi_url_from_stats(self):
        stats = {"sdk": {"artifacts": {"openapi_spec_url": "https://example.test/anthropic.yml"}}}
        self.assertEqual(extract_anthropic_openapi_url(stats), "https://example.test/anthropic.yml")

    def test_sync_specs_writes_files_and_manifest(self):
        anth_url = "https://example.test/anthropic.yml"
        responses = {
            OPENAI_SPEC_URL: _resp(OPENAI_SPEC_URL, text="openapi: 3.1.0\npaths: {}\n", content_type="text/yaml"),
            GEMINI_DISCOVERY_URL: _resp(
                GEMINI_DISCOVERY_URL,
                text='{"version":"v1beta","resources":{},"schemas":{}}',
                content_type="application/json",
            ),
            ANTHROPIC_STATS_URL: _resp(
                ANTHROPIC_STATS_URL,
                text=f"openapi_spec_url: {anth_url}\n",
                content_type="text/yaml",
            ),
            anth_url: _resp(anth_url, text="openapi: 3.1.0\npaths: {}\n", content_type="text/yaml"),
        }

        def fake_get(url, **kwargs):
            return responses[url]

        with tempfile.TemporaryDirectory() as td, patch("fastllm.specsync.httpx.get", side_effect=fake_get):
            out = sync_specs(specs_dir=td, write=True)
            d = Path(td)
            self.assertTrue((d / "openai.with-code-samples.yml").exists())
            self.assertTrue((d / "gemini.json").exists())
            self.assertTrue((d / "anthropic.yml").exists())
            self.assertTrue((d / "spec_manifest.json").exists())

            items = {o["provider"]: o for o in out["items"]}
            self.assertEqual(items["anthropic"]["source_url"], anth_url)
            self.assertTrue(items["openai"]["wrote"])
            self.assertTrue(items["gemini"]["wrote"])
            self.assertTrue(items["anthropic"]["wrote"])

    def test_sync_specs_dry_run_does_not_write(self):
        responses = {
            OPENAI_SPEC_URL: _resp(OPENAI_SPEC_URL, text="openapi: 3.1.0\npaths: {}\n", content_type="text/yaml"),
        }

        def fake_get(url, **kwargs):
            return responses[url]

        with tempfile.TemporaryDirectory() as td, patch("fastllm.specsync.httpx.get", side_effect=fake_get):
            out = sync_specs(["openai"], specs_dir=td, write=False)
            d = Path(td)
            self.assertFalse((d / "openai.with-code-samples.yml").exists())
            self.assertFalse((d / "spec_manifest.json").exists())
            self.assertFalse(out["items"][0]["wrote"])
