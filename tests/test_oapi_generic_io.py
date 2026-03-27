import unittest

import httpx

from fastllm_v2 import OpenAPIClient, spec_to_ops
from fastllm_v2.transport import AsyncTransport


SPEC = {
    "openapi": "3.1.0",
    "paths": {
        "/bin": {
            "get": {"operationId": "misc/get_bin"}
        },
        "/upload": {
            "post": {"operationId": "misc/upload"}
        },
        "/items": {
            "get": {"operationId": "items/list"}
        },
        "/{name}:do": {
            "post": {"operationId": "routeops/do", "parameters": [{"name": "name", "in": "path", "required": True}]}
        },
    },
}


class TestOAPIGenericIO(unittest.IsolatedAsyncioTestCase):
    async def test_binary_and_raw_and_multipart(self):
        seen = {"upload_ct": "", "query": "", "route_path": "", "items_body": b""}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/bin":
                if request.headers.get("x-raw") == "1":
                    return httpx.Response(200, content=b"raw", headers={"content-type": "application/octet-stream"})
                return httpx.Response(200, content=b"abc", headers={"content-type": "application/octet-stream"})

            if request.url.path == "/upload":
                seen["upload_ct"] = request.headers.get("content-type", "")
                return httpx.Response(200, json={"ok": True})

            if request.url.path == "/items":
                seen["query"] = str(request.url.query)
                seen["items_body"] = request.content
                return httpx.Response(200, json={"ok": True})

            if request.url.path.endswith(":do"):
                seen["route_path"] = request.url.path
                return httpx.Response(200, json={"ok": True})

            return httpx.Response(404, json={"error": "not-found"})

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(base_url="https://example.test", ops=spec_to_ops(SPEC), transport=AsyncTransport(client=hc))
        try:
            b = await api.misc.get_bin()
            self.assertEqual(b, b"abc")

            raw = await api.misc.get_bin(_raw=True, _headers={"x-raw": "1"})
            self.assertIsInstance(raw, httpx.Response)
            self.assertEqual(raw.content, b"raw")

            up = await api.misc.upload(_files={"file": ("x.txt", b"hello", "text/plain")})
            self.assertEqual(up["ok"], True)
            self.assertIn("multipart/form-data", seen["upload_ct"])

            await api.items.list(foo="bar", n=2)
            self.assertIn("foo=bar", seen["query"])
            self.assertIn("n=2", seen["query"])
            self.assertEqual(seen["items_body"], b"")

            await api.routeops.do(name="models/gemini-2.5-flash")
            self.assertIn("/models/gemini-2.5-flash:do", seen["route_path"])
        finally:
            await api.aclose()
            await hc.aclose()

    async def test_stream_http_error_body_is_readable(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text='{"error":"bad stream request"}', headers={"content-type": "application/json"})

        tr = AsyncTransport(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        try:
            with self.assertRaises(httpx.HTTPStatusError) as ctx:
                async for _ in tr.stream_sse_json("POST", "https://example.test/sse", json_data={"x": 1}):
                    pass
            self.assertIn("bad stream request", ctx.exception.response.text)
        finally:
            await tr.aclose()

    async def test_multipart_strips_json_content_type_header(self):
        seen = {"content_type": "", "auth": ""}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["content_type"] = request.headers.get("content-type", "")
            seen["auth"] = request.headers.get("authorization", "")
            return httpx.Response(200, json={"ok": True})

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        tr = AsyncTransport(client=hc, base_headers={"authorization": "Bearer k", "content-type": "application/json"})
        api = OpenAPIClient(base_url="https://example.test", ops=spec_to_ops(SPEC), transport=tr)
        try:
            out = await api.misc.upload(_files={"file": ("x.txt", b"hello", "text/plain")}, _data={"purpose": "user_data"})
            self.assertEqual(out["ok"], True)
            self.assertIn("multipart/form-data", seen["content_type"])
            self.assertIn("boundary=", seen["content_type"])
            self.assertEqual(seen["auth"], "Bearer k")
        finally:
            await api.aclose()
            await hc.aclose()
