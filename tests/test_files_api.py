import unittest
from unittest.mock import patch

import httpx

from fastllm import (
    APIError,
    FileRef,
    Part,
    UnsupportedCapabilityError,
    afile_content,
    afile_create,
    afile_delete,
    afile_get,
    afile_list,
    to_input_file_part,
)


class _Client:
    def __init__(self, provider: str, api, model: str = "test-model"):
        self.provider = provider
        self.model = model
        self.api = api
        self.closed = False

    async def aclose(self):
        self.closed = True


class _OpenAIApi:
    def __init__(self):
        self.calls = []

    async def call(self, path, verb, **kwargs):
        self.calls.append((path, verb, kwargs))
        if path == "/files" and verb == "POST":
            fn, _, mt = kwargs["files"]["file"]
            return {"id": "file_oa_1", "filename": fn, "mime_type": mt, "bytes": 5}
        if path == "/files/{file_id}" and verb == "GET":
            fid = kwargs["route"]["file_id"]
            return {"id": fid, "filename": "doc.txt", "bytes": 12}
        if path == "/files" and verb == "GET":
            return {"data": [{"id": "file_1"}, {"id": "file_2"}]}
        if path == "/files/{file_id}" and verb == "DELETE":
            return {"id": kwargs["route"]["file_id"], "deleted": True}
        if path == "/files/{file_id}/content" and verb == "GET":
            req = httpx.Request("GET", "https://api.openai.com/v1/files/file_oa_1/content")
            return httpx.Response(200, request=req, content=b"hello")
        raise AssertionError(f"Unexpected call: {verb} {path}")


class _AnthropicApi:
    def __init__(self):
        self.calls = []

    async def call(self, path, verb, **kwargs):
        self.calls.append((path, verb, kwargs))
        hdrs = kwargs.get("headers") or {}
        if path.startswith("/v1/files"):
            assert hdrs.get("anthropic-beta") == "files-api-2025-04-14"

        if path == "/v1/files" and verb == "POST":
            fn, _, mt = kwargs["files"]["file"]
            return {"id": "file_ant_1", "filename": fn, "mime_type": mt, "bytes": 7}
        if path == "/v1/files/{file_id}" and verb == "GET":
            fid = kwargs["route"]["file_id"]
            return {"id": fid, "filename": "paper.pdf", "bytes": 77}
        if path == "/v1/files" and verb == "GET":
            return {"data": [{"id": "file_ant_1"}]}
        if path == "/v1/files/{file_id}" and verb == "DELETE":
            return {"id": kwargs["route"]["file_id"], "deleted": True}
        if path == "/v1/files/{file_id}/content" and verb == "GET":
            req = httpx.Request("GET", "https://api.anthropic.com/v1/files/file_ant_1/content")
            return httpx.Response(200, request=req, content=b"%PDF")
        raise AssertionError(f"Unexpected call: {verb} {path}")


class _GeminiTransport:
    def __init__(self):
        self.calls = []

    async def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if url.endswith("/upload/v1beta/files"):
            req = httpx.Request(method, url)
            return httpx.Response(200, request=req, headers={"x-goog-upload-url": "https://upload.example/session/1"})
        if url == "https://upload.example/session/1":
            return {"file": {"name": "files/gm_1", "uri": "gs://bucket/gm_1", "mimeType": "application/pdf", "sizeBytes": "11"}}
        raise AssertionError(f"Unexpected request URL: {url}")


class _GeminiFilesOps:
    def __init__(self):
        self.calls = []

    async def get(self, **kwargs):
        self.calls.append(("get", kwargs))
        return {"name": kwargs["name"], "uri": "gs://bucket/gm_1", "mimeType": "application/pdf"}

    async def list(self, **kwargs):
        self.calls.append(("list", kwargs))
        return {"files": [{"name": "files/gm_1"}, {"name": "files/gm_2"}]}

    async def delete(self, **kwargs):
        self.calls.append(("delete", kwargs))
        return {"name": kwargs["name"], "deleted": True}


class _GeminiApi:
    def __init__(self):
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.transport = _GeminiTransport()
        self.files = _GeminiFilesOps()


class TestFilesPartMapping(unittest.TestCase):
    def test_to_input_file_part_openai_defaults(self):
        p = to_input_file_part(FileRef(id="file_1", provider="openai", filename="doc.pdf", mime_type="application/pdf"))
        self.assertIsInstance(p, Part)
        self.assertEqual(p.type, "input_file")
        self.assertEqual(p.data["file_id"], "file_1")
        self.assertEqual(p.data["filename"], "doc.pdf")
        self.assertEqual(p.data["mimeType"], "application/pdf")

    def test_to_input_file_part_anthropic(self):
        p = to_input_file_part(FileRef(id="file_ant_1", provider="anthropic"))
        self.assertEqual(p.data["file_id"], "file_ant_1")
        src = p.data["anthropic"]["source"]
        self.assertEqual(src["type"], "file")
        self.assertEqual(src["file_id"], "file_ant_1")

    def test_to_input_file_part_gemini(self):
        p = to_input_file_part(FileRef(id="files/gm_1", provider="gemini", uri="gs://bucket/gm_1", mime_type="application/pdf"))
        self.assertEqual(p.data["file_url"], "gs://bucket/gm_1")
        self.assertEqual(p.data["gemini"]["fileData"]["fileUri"], "gs://bucket/gm_1")


class TestFilesApi(unittest.IsolatedAsyncioTestCase):
    async def test_openai_and_openai_compat_files_crud(self):
        made = []

        def _mk(*args, **kwargs):
            c = _Client("openai", _OpenAIApi(), model="gpt-5-mini")
            made.append(c)
            return c

        with patch("fastllm.files.mk_auto_client", side_effect=_mk):
            f = await afile_create("gpt-5-mini", file=b"hello", filename="doc.txt", purpose="assistants")
            g = await afile_get("gpt-5-mini", "file_oa_1")
            ls = await afile_list("gpt-5-mini")
            d = await afile_delete("gpt-5-mini", "file_oa_1")
            content = await afile_content("gpt-5-mini", "file_oa_1")
            # openai_compat uses the same endpoint family for files
            k = await afile_get("kimi-k2.5", "file_oa_1")

        self.assertEqual(f.id, "file_oa_1")
        self.assertEqual(g.id, "file_oa_1")
        self.assertEqual(k.id, "file_oa_1")
        self.assertEqual([x.id for x in ls], ["file_1", "file_2"])
        self.assertEqual(d["deleted"], True)
        self.assertEqual(content, b"hello")
        self.assertTrue(all(c.closed for c in made))
        create_call = made[0].api.calls[0]
        self.assertEqual(create_call[0], "/files")
        self.assertEqual(create_call[1], "POST")
        self.assertEqual(create_call[2]["data"]["purpose"], "assistants")

    async def test_anthropic_files_crud_and_content(self):
        made = []

        def _mk(*args, **kwargs):
            c = _Client("anthropic", _AnthropicApi(), model="claude-sonnet-4-5")
            made.append(c)
            return c

        with patch("fastllm.files.mk_auto_client", side_effect=_mk):
            f = await afile_create("claude-sonnet-4-5", file=b"%PDF", filename="paper.pdf", mime_type="application/pdf")
            g = await afile_get("claude-sonnet-4-5", "file_ant_1")
            ls = await afile_list("claude-sonnet-4-5", limit=5)
            d = await afile_delete("claude-sonnet-4-5", "file_ant_1")
            content = await afile_content("claude-sonnet-4-5", "file_ant_1")

        self.assertEqual(f.id, "file_ant_1")
        self.assertEqual(g.id, "file_ant_1")
        self.assertEqual([x.id for x in ls], ["file_ant_1"])
        self.assertEqual(d["deleted"], True)
        self.assertEqual(content, b"%PDF")
        self.assertTrue(all(c.closed for c in made))
        # list call preserved query passthrough
        list_call = made[2].api.calls[0]
        self.assertEqual(list_call[2]["query"]["limit"], 5)

    async def test_gemini_resumable_upload_and_files_ops(self):
        made = []

        def _mk(*args, **kwargs):
            c = _Client("gemini", _GeminiApi(), model="gemini-2.5-flash")
            made.append(c)
            return c

        with patch("fastllm.files.mk_auto_client", side_effect=_mk):
            f = await afile_create("gemini-2.5-flash", file=b"01234567890", filename="doc.pdf", mime_type="application/pdf")
            g = await afile_get("gemini-2.5-flash", "gm_1")
            ls = await afile_list("gemini-2.5-flash", pageSize=2)
            d = await afile_delete("gemini-2.5-flash", "gm_1")
            with self.assertRaises(UnsupportedCapabilityError):
                await afile_content("gemini-2.5-flash", "gm_1")

        self.assertEqual(f.id, "files/gm_1")
        self.assertEqual(g.id, "files/gm_1")
        self.assertEqual([x.id for x in ls], ["files/gm_1", "files/gm_2"])
        self.assertEqual(d["deleted"], True)
        self.assertTrue(all(c.closed for c in made))

        up_calls = made[0].api.transport.calls
        self.assertEqual(up_calls[0][0], "POST")
        self.assertTrue(up_calls[0][1].endswith("/upload/v1beta/files"))
        self.assertEqual(up_calls[1][1], "https://upload.example/session/1")
        files_calls = made[1].api.files.calls + made[2].api.files.calls + made[3].api.files.calls
        self.assertEqual(files_calls[0][0], "get")
        self.assertEqual(files_calls[1][0], "list")
        self.assertEqual(files_calls[2][0], "delete")

    async def test_http_errors_are_wrapped_as_apierror(self):
        class _ErrApi:
            async def call(self, path, verb, **kwargs):
                req = httpx.Request("POST", "https://api.openai.com/v1/files")
                rsp = httpx.Response(
                    400,
                    request=req,
                    json={"error": {"message": "bad file", "type": "invalid_request_error", "code": "bad_file"}},
                )
                raise httpx.HTTPStatusError("bad", request=req, response=rsp)

        fake = _Client("openai", _ErrApi(), model="gpt-5-mini")
        with patch("fastllm.files.mk_auto_client", return_value=fake):
            with self.assertRaises(APIError) as ctx:
                await afile_create("gpt-5-mini", file=b"x", filename="x.txt")
        err = ctx.exception
        self.assertEqual(err.provider, "openai")
        self.assertEqual(err.endpoint, "files.create")
        self.assertEqual(err.code, "bad_file")
        self.assertTrue(fake.closed)
