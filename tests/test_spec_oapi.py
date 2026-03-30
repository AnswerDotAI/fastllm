import json
import inspect
import tempfile
import unittest
from pathlib import Path

import httpx

from fastllm import APIError, OpenAPIClient, client_from_spec, gemini_ops, load_spec_json, spec_to_ops
from fastllm.transport import AsyncTransport


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

    async def test_signature_required_types_and_docs(self):
        spec = {
            "openapi": "3.1.0",
            "paths": {
                "/items/{item_id}": {
                    "post": {
                        "operationId": "items/create",
                        "summary": "Create item",
                        "externalDocs": {"url": "https://docs.example.com/items/create"},
                        "parameters": [
                            {"name": "item_id", "in": "path", "required": True, "schema": {"type": "string"}, "description": "Item identifier."},
                            {"name": "q", "in": "query", "required": True, "schema": {"type": "string"}, "description": "Query flag."},
                        ],
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["count"],
                                        "properties": {
                                            "count": {"type": "integer", "description": "How many items."},
                                            "note": {"type": "string", "description": "Optional note."},
                                        },
                                    }
                                }
                            }
                        },
                    }
                }
            },
        }
        api = OpenAPIClient(base_url="https://example.test", ops=spec_to_ops(spec))
        try:
            op = api.items.create
            sig = inspect.signature(op)
            self.assertEqual(sig.parameters["item_id"].annotation, str)
            self.assertEqual(sig.parameters["q"].annotation, str)
            self.assertEqual(sig.parameters["count"].annotation, int)
            self.assertEqual(sig.parameters["note"].annotation, str)
            self.assertIs(sig.parameters["item_id"].default, inspect._empty)
            self.assertIs(sig.parameters["q"].default, inspect._empty)
            self.assertIs(sig.parameters["count"].default, inspect._empty)
            self.assertEqual(sig.parameters["note"].default, None)
            self.assertIn("Docs: https://docs.example.com/items/create", op.__doc__)
            self.assertIn("Required: item_id, q, count", op.__doc__)
            self.assertIn("count (int, required)", op.__doc__)
        finally:
            await api.aclose()

    async def test_signature_handles_optional_query_then_required_body(self):
        spec = {
            "openapi": "3.1.0",
            "paths": {
                "/messages": {
                    "post": {
                        "operationId": "messages/create",
                        "summary": "Create message",
                        "parameters": [
                            {"name": "after", "in": "query", "required": False, "schema": {"type": "string"}},
                        ],
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["model", "max_tokens"],
                                        "properties": {
                                            "model": {"type": "string"},
                                            "max_tokens": {"type": "integer"},
                                            "temperature": {"type": "number"},
                                        },
                                    }
                                }
                            }
                        },
                    }
                }
            },
        }
        api = OpenAPIClient(base_url="https://example.test", ops=spec_to_ops(spec))
        try:
            op = api.messages.create
            sig = inspect.signature(op)
            # Required params are keyword-only and stay inspect-safe.
            self.assertIs(sig.parameters["model"].default, inspect._empty)
            self.assertIs(sig.parameters["max_tokens"].default, inspect._empty)
            self.assertEqual(sig.parameters["after"].default, None)
            self.assertEqual(sig.parameters["model"].kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertEqual(sig.parameters["after"].kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertIn("messages.create", repr(op))
        finally:
            await api.aclose()

    async def test_group_repr_lists_ops(self):
        api = OpenAPIClient(base_url="https://example.test", ops=spec_to_ops(MIN_SPEC))
        try:
            txt = repr(api.things)
            self.assertIn("things.update(", txt)
            self.assertIn("POST /things/{thing_id}", txt)
            md = api.things._repr_markdown_()
            self.assertIn("- things.update(", md)
        finally:
            await api.aclose()

    async def test_docs_url_fallback_from_description_markdown_link(self):
        spec = {
            "openapi": "3.1.0",
            "paths": {
                "/v1/messages": {
                    "post": {
                        "operationId": "messages/messages_post",
                        "summary": "Create a message",
                        "description": "Read the [user guide](https://docs.example.com/messages/guide).",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["model"],
                                        "properties": {"model": {"type": "string"}},
                                    }
                                }
                            }
                        },
                    }
                }
            },
        }
        api = OpenAPIClient(base_url="https://example.test", ops=spec_to_ops(spec))
        try:
            op = api.messages.messages_post
            self.assertEqual(op.docs_url, "https://docs.example.com/messages/guide")
            self.assertIn("(model)", repr(op))
            self.assertIn("Create a message", repr(op))
            self.assertIn("https://docs.example.com/messages/guide", op._repr_markdown_())
        finally:
            await api.aclose()

    async def test_markdown_repr_absolutizes_relative_links_and_flattens_newlines(self):
        spec = {
            "openapi": "3.1.0",
            "paths": {
                "/responses": {
                    "post": {
                        "operationId": "responses/create_response",
                        "summary": "Create with [text](/docs/guides/text)\nfor output.",
                        "externalDocs": {"url": "https://platform.openai.com/docs/api-reference/responses/create"},
                    }
                }
            },
        }
        api = OpenAPIClient(base_url="https://example.test", ops=spec_to_ops(spec))
        try:
            op = api.responses.create_response
            md = op._repr_markdown_()
            self.assertIn("https://platform.openai.com/docs/guides/text", md)
            self.assertNotIn("\n", md)
            self.assertIn(": *Create with [text]", md)
        finally:
            await api.aclose()

    async def test_group_markdown_repr_uses_bulleted_links(self):
        spec = {
            "openapi": "3.1.0",
            "paths": {
                "/items": {
                    "get": {
                        "operationId": "items/list",
                        "summary": "List items",
                        "externalDocs": {"url": "https://docs.example.com/items/list"},
                    },
                    "post": {
                        "operationId": "items/create",
                        "summary": "Create item",
                        "externalDocs": {"url": "https://docs.example.com/items/create"},
                    },
                }
            },
        }
        api = OpenAPIClient(base_url="https://example.test", ops=spec_to_ops(spec))
        try:
            md = api.items._repr_markdown_()
            self.assertIn("- [items.list](https://docs.example.com/items/list)()", md)
            self.assertIn("- [items.create](https://docs.example.com/items/create)()", md)
            self.assertIn(": *List items*", md)
            self.assertIn(": *Create item*", md)
        finally:
            await api.aclose()

    async def test_openai_meta_group_only_sets_docs_url_and_fixes_relative_links(self):
        spec = {
            "openapi": "3.1.0",
            "paths": {
                "/responses/{response_id}/cancel": {
                    "post": {
                        "operationId": "cancelResponse",
                        "summary": "Cancel it. [Learn more](/docs/guides/background).",
                        "x-oaiMeta": {"group": "responses"},
                    }
                }
            },
        }
        api = OpenAPIClient(base_url="https://example.test", ops=spec_to_ops(spec))
        try:
            op = api.responses.cancel_response
            self.assertEqual(op.docs_url, "https://platform.openai.com/docs/api-reference/responses")
            md = op._repr_markdown_()
            self.assertIn("[responses.cancel_response](https://platform.openai.com/docs/api-reference/responses)", md)
            self.assertIn("[Learn more](https://platform.openai.com/docs/guides/background)", md)
        finally:
            await api.aclose()

    async def test_infer_required_from_single_example_core_fields(self):
        spec = {
            "openapi": "3.1.0",
            "paths": {
                "/responses/input_tokens": {
                    "post": {
                        "operationId": "responses/get_input_token_counts",
                        "x-oaiMeta": {
                            "examples": [{
                                "request": {
                                    "curl": "curl ... -d '{\"model\":\"gpt-5\",\"input\":\"Tell me a joke.\"}'"
                                }
                            }]
                        },
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "model": {"type": "string"},
                                            "input": {"type": "string"},
                                            "tools": {"type": "array", "items": {"type": "string"}},
                                        },
                                        "required": [],
                                    }
                                }
                            }
                        },
                    }
                }
            },
        }
        api = OpenAPIClient(base_url="https://example.test", ops=spec_to_ops(spec))
        try:
            sig = inspect.signature(api.responses.get_input_token_counts)
            self.assertIs(sig.parameters["model"].default, inspect._empty)
            self.assertIs(sig.parameters["input"].default, inspect._empty)
            self.assertEqual(sig.parameters["tools"].default, None)
        finally:
            await api.aclose()

    async def test_infer_required_from_multi_examples_intersection(self):
        spec = {
            "openapi": "3.1.0",
            "paths": {
                "/responses/create": {
                    "post": {
                        "operationId": "responses/create",
                        "x-oaiMeta": {
                            "examples": [
                                {"request": {"curl": "curl ... -d '{\"model\":\"gpt-5\",\"input\":\"hello\",\"temperature\":0.2}'"}},
                                {"request": {"curl": "curl ... -d '{\"model\":\"gpt-5\",\"input\":[{\"role\":\"user\",\"content\":\"hi\"}],\"tools\":[]}'"}},
                            ]
                        },
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "model": {"type": "string"},
                                            "input": {"type": "string"},
                                            "temperature": {"type": "number"},
                                            "tools": {"type": "array", "items": {"type": "string"}},
                                        },
                                        "required": [],
                                    }
                                }
                            }
                        },
                    }
                }
            },
        }
        api = OpenAPIClient(base_url="https://example.test", ops=spec_to_ops(spec))
        try:
            sig = inspect.signature(api.responses.create)
            self.assertIs(sig.parameters["model"].default, inspect._empty)
            self.assertIs(sig.parameters["input"].default, inspect._empty)
            self.assertEqual(sig.parameters["temperature"].default, None)
            self.assertEqual(sig.parameters["tools"].default, None)
        finally:
            await api.aclose()

    async def test_discovery_infers_required_from_required_description_prefix(self):
        ds = {
            "version": "v1beta",
            "schemas": {
                "GenerateContentRequest": {
                    "type": "object",
                    "properties": {
                        "contents": {"type": "array", "description": "Required. Conversation input."},
                        "tools": {"type": "array", "description": "Optional. Tools list."},
                    },
                }
            },
            "resources": {
                "models": {
                    "methods": {
                        "generateContent": {
                            "id": "generativelanguage.models.generateContent",
                            "path": "v1beta/{+model}:generateContent",
                            "httpMethod": "POST",
                            "parameters": {
                                "model": {"type": "string", "location": "path", "required": True},
                            },
                            "request": {"$ref": "GenerateContentRequest"},
                            "description": "Generate content",
                        }
                    }
                }
            },
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "gemini.json"
            p.write_text(json.dumps(ds), encoding="utf-8")
            api = OpenAPIClient(base_url="https://example.test", ops=gemini_ops(spec_path=str(p)))
            try:
                sig = inspect.signature(api.models.generate_content)
                self.assertIs(sig.parameters["model"].default, inspect._empty)
                self.assertIs(sig.parameters["contents"].default, inspect._empty)
                self.assertEqual(sig.parameters["tools"].default, None)
            finally:
                await api.aclose()

    async def test_dynamic_op_http_errors_are_wrapped_with_context(self):
        spec = {
            "openapi": "3.1.0",
            "paths": {
                "/{model}:generateContent": {
                    "post": {
                        "operationId": "models/generateContent",
                        "parameters": [{"name": "model", "in": "path", "required": True, "schema": {"type": "string"}}],
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object", "properties": {"contents": {"type": "array"}}}
                                }
                            }
                        },
                    }
                }
            },
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"message": "bad payload", "status": "INVALID_ARGUMENT"}})

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            ops=spec_to_ops(spec),
            transport=AsyncTransport(client=hc),
            provider="gemini",
        )
        try:
            with self.assertRaises(APIError) as ctx:
                await api.models.generate_content(model="models/gemini-2.5-flash", contents=[])
            err = ctx.exception
            self.assertEqual(err.provider, "gemini")
            self.assertEqual(err.endpoint, "models.generate_content")
            self.assertEqual(err.model, "models/gemini-2.5-flash")
            self.assertEqual(err.status_code, 400)
            self.assertEqual(err.code, "INVALID_ARGUMENT")
            self.assertIn("bad payload", err.message)
        finally:
            await api.aclose()
            await hc.aclose()

    async def test_dynamic_stream_http_errors_are_wrapped_with_context(self):
        spec = {
            "openapi": "3.1.0",
            "paths": {
                "/events": {
                    "post": {
                        "operationId": "events/create",
                        "x-fastllm-stream": True,
                    }
                }
            },
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": {"message": "rate limited", "type": "rate_limit_error"}})

        hc = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        api = OpenAPIClient(
            base_url="https://api.openai.com/v1",
            ops=spec_to_ops(spec),
            transport=AsyncTransport(client=hc),
            provider="openai",
        )
        try:
            with self.assertRaises(APIError) as ctx:
                async for _ in api.events.create(_stream=True):
                    pass
            err = ctx.exception
            self.assertEqual(err.provider, "openai")
            self.assertEqual(err.endpoint, "events.create")
            self.assertEqual(err.status_code, 429)
            self.assertEqual(err.error_type, "rate_limit_error")
            self.assertTrue(err.retryable)
            self.assertIn("rate limited", err.message)
        finally:
            await api.aclose()
            await hc.aclose()
