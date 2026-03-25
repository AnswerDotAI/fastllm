"Core internal types."

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Part:
    "A normalized content part."
    type: str
    text: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class Msg:
    "A normalized message."
    role: str
    content: List[Part]


@dataclass(frozen=True)
class ToolSpec:
    "Normalized tool schema."
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    "Normalized tool call."
    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Usage:
    "Normalized usage."
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Caps:
    "Capability declaration for a client."
    tools: bool = False
    tool_choice: bool = False
    streaming: bool = True
    search: bool = False
    reasoning: bool = False
    prefill: bool = False
    citations: bool = False
    prompt_caching: bool = False
    images: bool = True
    pdfs: bool = False
    url_context: bool = False


@dataclass(frozen=True)
class RequestOptions:
    "Request options shared across providers."
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    cache: Optional[Any] = None
    tools: Optional[List[ToolSpec]] = None
    tool_choice: Optional[Any] = None
    reasoning_effort: Optional[str] = None
    response_format: Optional[Dict[str, Any]] = None
    search: Optional[Any] = None
    native: Optional[Dict[str, Any]] = None
    extra_body: Optional[Dict[str, Any]] = None
    extra_headers: Optional[Dict[str, str]] = None
    extra_query: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class Completion:
    "Normalized completion response."
    model: str
    message: Msg
    finish_reason: Optional[str] = None
    usage: Optional[Usage] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    provider: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Delta:
    "Normalized streaming delta event."
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: Optional[Usage] = None
    raw: Dict[str, Any] = field(default_factory=dict)
