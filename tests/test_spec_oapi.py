import json
import unittest

import httpx

from fastllm_v2 import OpenAPIClient, client_from_spec, load_spec_json, spec_to_ops
from fastllm_v2.transport import AsyncTransport


MIN_SPEC = {
    "openapi": "3.1.0",
    "paths": {
        "/things/{thing_id}": {
            "post": {
                "operationId": "things/update",
                "parameters": [{"name": "q", "in": "query"}],
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {"type": "object", "properties": {"x": {"type": "integer"}}}}}}}},
        "/events": {
            "post": {
                "operationId": "events/create",
                "x-fastllm-stream": True,
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {"type": "object", "properties": {"stream": {"type": "boolean"}}}}}}}}}}


class TestSpecAndOAPI(unittest.IsolatedAsyncioTestCase):
    async def test_spec_to_ops(self):
        ops = spec_to_ops(MIN_SPEC)
        names = {(o.group, o.name, o.path, o.verb) for o in ops}
        self.assertIn(("things", "update", "/things/{thing_id}", "POST"), names)
        self.assertIn(("events", "create", "/events", "POST"), names)
        self.assertEqual(len(spec_to_ops(load_spec_json(json.dumps(MIN_SPEC)))), len(ops))

    async def test_operation_binding_and_stream(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/events":
                body = 'data: {"type":"tick","delta":"a"}\n\ndata: {"type":"tick","delta":"b"}\n\ndata: [DONE]\n\n'
                return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
            payload = json.loads(request.content.decode()) if request.content else {}
            return httpx.Response(200, json={
                "path": request.url.path,
                "method": request.method,
                "query": dict(request.url.params),
                "json": payload,
            })

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        transport = AsyncTransport(client=client)
        api = OpenAPIClient(base_url="https://example.test", ops=spec_to_ops(MIN_SPEC), transport=transport)
        try:
            res = await api.things.update("abc", q="yes", x=5, extra="v")
            self.assertEqual(res["path"], "/things/abc")
            self.assertEqual(res["query"]["q"], "yes")
            self.assertEqual(res["json"]["x"], 5)
            self.assertEqual(res["json"]["extra"], "v")

            seen = []
            async for ev in api.events.create(_stream=True, stream=True):
                seen.append(ev["delta"])
            self.assertEqual(seen, ["a", "b"])
        finally:
            await api.aclose()
            await client.aclose()

    async def test_client_from_spec(self):
        c = client_from_spec("https://example.test", MIN_SPEC)
        self.assertTrue(hasattr(c, "things"))
        await c.aclose()
