"Built-in operation specs and official snapshots."

from __future__ import annotations

import json

from .official_ops import GEMINI_OFFICIAL_OPS, OPENAI_OFFICIAL_OPS
from .spec import OpSpec, spec_to_ops


# Compact fallbacks when full official snapshots are not desired.
OPENAI_MIN_SPEC = json.loads(r'''
{
  "openapi": "3.1.0",
  "paths": {
    "/responses": {
      "post": {
        "operationId": "responses/create",
        "x-fastllm-stream": true,
        "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {
          "model": {"type": "string"}, "input": {}, "tools": {"type": "array"}, "stream": {"type": "boolean"}
        }}}}}
      }
    },
    "/responses/{response_id}": {
      "get": {"operationId": "responses/retrieve", "parameters": [{"name": "response_id", "in": "path", "required": true}]},
      "delete": {"operationId": "responses/delete", "parameters": [{"name": "response_id", "in": "path", "required": true}]}
    },
    "/responses/{response_id}/cancel": {
      "post": {"operationId": "responses/cancel", "parameters": [{"name": "response_id", "in": "path", "required": true}]}
    },
    "/responses/{response_id}/input_items": {
      "get": {"operationId": "responses/list_input_items", "parameters": [
        {"name": "response_id", "in": "path", "required": true}, {"name": "after", "in": "query"}, {"name": "limit", "in": "query"}
      ]}
    },
    "/responses/compact": {"post": {"operationId": "responses/compact"}},
    "/responses/input_tokens": {"post": {"operationId": "responses/input_tokens"}},
    "/chat/completions": {
      "post": {
        "operationId": "chat/create_completions",
        "x-fastllm-stream": true,
        "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {
          "model": {"type": "string"}, "messages": {"type": "array"}, "tools": {"type": "array"}, "stream": {"type": "boolean"}
        }}}}}
      }
    }
  }
}
''')


ANTHROPIC_MIN_SPEC = json.loads(r'''
{
  "openapi": "3.1.0",
  "paths": {
    "/v1/messages": {
      "post": {
        "operationId": "messages/create",
        "x-fastllm-stream": true,
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "model": {"type": "string"},
                  "messages": {"type": "array"},
                  "system": {},
                  "max_tokens": {"type": "integer"},
                  "temperature": {"type": "number"},
                  "tools": {"type": "array"},
                  "tool_choice": {},
                  "thinking": {"type": "object"},
                  "web_search": {"type": "object"},
                  "context_management": {"type": "object"},
                  "metadata": {"type": "object"},
                  "stream": {"type": "boolean"}
                }
              }
            }
          }
        }
      }
    },
    "/v1/messages/count_tokens": {
      "post": {
        "operationId": "messages/count_tokens",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "model": {"type": "string"},
                  "messages": {"type": "array"},
                  "system": {}
                }
              }
            }
          }
        }
      }
    },
    "/v1/messages/batches": {
      "get": {"operationId": "messages_batches/list"},
      "post": {
        "operationId": "messages_batches/create",
        "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {
          "requests": {"type": "array"}
        }}}}}
      }
    },
    "/v1/messages/batches/{message_batch_id}": {
      "get": {"operationId": "messages_batches/retrieve", "parameters": [{"name": "message_batch_id", "in": "path", "required": true}]}
    },
    "/v1/messages/batches/{message_batch_id}/cancel": {
      "post": {"operationId": "messages_batches/cancel", "parameters": [{"name": "message_batch_id", "in": "path", "required": true}]}
    },
    "/v1/messages/batches/{message_batch_id}/results": {
      "get": {
        "operationId": "messages_batches/results",
        "x-fastllm-stream": true,
        "parameters": [{"name": "message_batch_id", "in": "path", "required": true}],
        "responses": {"200": {"content": {"text/event-stream": {}}}}
      }
    },
    "/v1/models": {"get": {"operationId": "models/list"}},
    "/v1/models/{model_id}": {
      "get": {"operationId": "models/retrieve", "parameters": [{"name": "model_id", "in": "path", "required": true}]}
    },
    "/v1/files": {
      "get": {"operationId": "files/list"},
      "post": {"operationId": "files/create"}
    },
    "/v1/files/{file_id}": {
      "get": {"operationId": "files/retrieve", "parameters": [{"name": "file_id", "in": "path", "required": true}]},
      "delete": {"operationId": "files/delete", "parameters": [{"name": "file_id", "in": "path", "required": true}]}
    },
    "/v1/files/{file_id}/content": {
      "get": {"operationId": "files/content", "parameters": [{"name": "file_id", "in": "path", "required": true}]}
    },
    "/v1/organizations/invites": {
      "get": {"operationId": "organizations_invites/list"},
      "post": {"operationId": "organizations_invites/create"}
    },
    "/v1/organizations/invites/{invite_id}": {
      "get": {"operationId": "organizations_invites/retrieve", "parameters": [{"name": "invite_id", "in": "path", "required": true}]},
      "delete": {"operationId": "organizations_invites/delete", "parameters": [{"name": "invite_id", "in": "path", "required": true}]}
    },
    "/v1/organizations/users": {
      "get": {"operationId": "organizations_users/list"}
    },
    "/v1/organizations/users/{user_id}": {
      "delete": {"operationId": "organizations_users/delete", "parameters": [{"name": "user_id", "in": "path", "required": true}]}
    },
    "/v1/organizations/workspaces": {
      "get": {"operationId": "organizations_workspaces/list"},
      "post": {"operationId": "organizations_workspaces/create"}
    },
    "/v1/organizations/workspaces/{workspace_id}": {
      "get": {"operationId": "organizations_workspaces/retrieve", "parameters": [{"name": "workspace_id", "in": "path", "required": true}]},
      "post": {"operationId": "organizations_workspaces/update", "parameters": [{"name": "workspace_id", "in": "path", "required": true}]},
      "delete": {"operationId": "organizations_workspaces/delete", "parameters": [{"name": "workspace_id", "in": "path", "required": true}]}
    },
    "/v1/organizations/workspaces/{workspace_id}/members": {
      "get": {"operationId": "organizations_workspaces_members/list", "parameters": [{"name": "workspace_id", "in": "path", "required": true}]}
    },
    "/v1/organizations/workspaces/{workspace_id}/members/{user_id}": {
      "post": {"operationId": "organizations_workspaces_members/create", "parameters": [
        {"name": "workspace_id", "in": "path", "required": true}, {"name": "user_id", "in": "path", "required": true}
      ]},
      "delete": {"operationId": "organizations_workspaces_members/delete", "parameters": [
        {"name": "workspace_id", "in": "path", "required": true}, {"name": "user_id", "in": "path", "required": true}
      ]}
    },
    "/v1/organizations/api_keys": {
      "get": {"operationId": "organizations_api_keys/list"},
      "post": {"operationId": "organizations_api_keys/create"}
    },
    "/v1/organizations/api_keys/{api_key_id}": {
      "get": {"operationId": "organizations_api_keys/retrieve", "parameters": [{"name": "api_key_id", "in": "path", "required": true}]},
      "delete": {"operationId": "organizations_api_keys/delete", "parameters": [{"name": "api_key_id", "in": "path", "required": true}]}
    }
  }
}
''')


GEMINI_MIN_SPEC = json.loads(r'''
{
  "openapi": "3.1.0",
  "paths": {
    "/models/{model}:generateContent": {
      "post": {
        "operationId": "models/generate_content",
        "parameters": [{"name": "model", "in": "path", "required": true}],
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "contents": {"type": "array"},
                  "generationConfig": {"type": "object"},
                  "tools": {"type": "array"},
                  "cachedContent": {"type": "string"}
                }
              }
            }
          }
        }
      }
    },
    "/models/{model}:streamGenerateContent": {
      "post": {
        "operationId": "models/stream_generate_content",
        "x-fastllm-stream": true,
        "parameters": [
          {"name": "model", "in": "path", "required": true},
          {"name": "alt", "in": "query"}
        ],
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "contents": {"type": "array"},
                  "generationConfig": {"type": "object"},
                  "tools": {"type": "array"},
                  "cachedContent": {"type": "string"}
                }
              }
            }
          }
        }
      }
    },
    "/models/{model}:countTokens": {
      "post": {
        "operationId": "models/count_tokens",
        "parameters": [{"name": "model", "in": "path", "required": true}]
      }
    },
    "/models/{model}:embedContent": {
      "post": {
        "operationId": "models/embed_content",
        "parameters": [{"name": "model", "in": "path", "required": true}]
      }
    },
    "/models/{model}:batchEmbedContents": {
      "post": {
        "operationId": "models/batch_embed_contents",
        "parameters": [{"name": "model", "in": "path", "required": true}]
      }
    },
    "/cachedContents": {
      "get": {"operationId": "cached_contents/list"},
      "post": {"operationId": "cached_contents/create"}
    },
    "/cachedContents/{cachedContent}": {
      "get": {"operationId": "cached_contents/retrieve", "parameters": [{"name": "cachedContent", "in": "path", "required": true}]},
      "patch": {"operationId": "cached_contents/update", "parameters": [{"name": "cachedContent", "in": "path", "required": true}]},
      "delete": {"operationId": "cached_contents/delete", "parameters": [{"name": "cachedContent", "in": "path", "required": true}]}
    }
  }
}
''')


def openai_ops(*, full: bool = True):
    "OpenAI OpSpec list from official snapshot (default) or compact fallback."
    if not full: return spec_to_ops(OPENAI_MIN_SPEC)
    xs = list(OPENAI_OFFICIAL_OPS)
    have = {(o.group, o.name, o.path, o.verb) for o in xs}
    aliases = [
        OpSpec(group="responses", name="create", path="/responses", verb="POST", streamable=True),
        OpSpec(group="responses", name="retrieve", path="/responses/{response_id}", verb="GET", route_params=["response_id"]),
        OpSpec(group="chat", name="create_completions", path="/chat/completions", verb="POST", streamable=True),
        OpSpec(group="embeddings", name="create", path="/embeddings", verb="POST"),
        OpSpec(group="images", name="generate", path="/images/generations", verb="POST"),
        OpSpec(group="audio", name="speech", path="/audio/speech", verb="POST"),
    ]
    for o in aliases:
        k = (o.group, o.name, o.path, o.verb)
        if k not in have: xs.append(o)
    return xs


def anthropic_ops(*, full: bool = True):
    "Anthropic OpSpec list from expanded documented surface."
    return spec_to_ops(ANTHROPIC_MIN_SPEC)


def gemini_ops(*, full: bool = True):
    "Gemini OpSpec list from discovery snapshot (default) or compact fallback."
    return list(GEMINI_OFFICIAL_OPS) if full else spec_to_ops(GEMINI_MIN_SPEC)
