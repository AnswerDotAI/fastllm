"Streaming helpers for lossless event collation."

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .types import Delta, ToolCall, Usage


RawEvent = dict[str, Any]


@dataclass(frozen=True)
class StreamSummary:
    "Collected stream output with full raw-event preservation."
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: Optional[Usage] = None
    deltas: list[Delta] = field(default_factory=list)
    raw_events: list[RawEvent] = field(default_factory=list)
    final: Delta = field(default_factory=Delta)


@dataclass
class _ToolBuf:
    "In-progress normalized tool call assembly state."
    id: str
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    chunks: list[str] = field(default_factory=list)


def _parse_json_args(s: str) -> dict[str, Any]:
    "Parse tool argument JSON, preserving raw text when parsing fails."
    if not s: return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {"_value": obj}
    except json.JSONDecodeError: return {"_raw": s}


def _chunk_from_tool_call(tc: ToolCall) -> str | None:
    "Return incremental argument chunk from a ToolCall if present."
    if len(tc.arguments or {}) != 1: return None
    if "_delta" not in tc.arguments: return None
    v = tc.arguments.get("_delta")
    return v if isinstance(v, str) else str(v)


def _is_empty_tool_args(args: dict[str, Any]) -> bool:
    "Return true when a tool arg payload has no meaningful args yet."
    if not args: return True
    return set(args) == {"_delta"} and not (args.get("_delta") or "")


def _ensure_tool(tools: dict[str, _ToolBuf], order: list[str], key: str, *, tool_id: str = "",
    name: str = "", arguments: dict[str, Any] | None = None) -> _ToolBuf:
    "Get/create tool buffer and merge id/name/arguments."
    if key not in tools:
        order.append(key)
        tools[key] = _ToolBuf(id=(tool_id or key))
    tb = tools[key]
    if tool_id and not tb.id: tb.id = tool_id
    if name and not tb.name: tb.name = name
    if arguments and not _is_empty_tool_args(arguments): tb.arguments = dict(arguments)
    return tb


def _index_from_raw(raw: RawEvent) -> str:
    "Extract normalized content-block index id from provider event raw."
    idx = raw.get("index")
    if idx is None: return ""
    try: return str(int(idx))
    except (TypeError, ValueError): return str(idx)


def _chat_delta_tool_calls(raw: RawEvent) -> list[dict[str, Any]]:
    "Extract chat.completions delta tool call entries when present."
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict): return []
    delta = choices[0].get("delta")
    if not isinstance(delta, dict): return []
    tcs = delta.get("tool_calls")
    if not isinstance(tcs, list): return []
    return [tc for tc in tcs if isinstance(tc, dict)]


def _prime_tool_from_raw(raw: RawEvent, tools: dict[str, _ToolBuf], order: list[str], idx_to_key: dict[str, str],
    id_alias_to_key: dict[str, str]) -> None:
    "Prime tool-call metadata from provider raw events when available."
    typ = raw.get("type")
    if typ == "content_block_start":
        cb = raw.get("content_block") if isinstance(raw.get("content_block"), dict) else {}
        if cb.get("type") != "tool_use": return
        idx = _index_from_raw(raw)
        tool_id = str(cb.get("id") or "")
        key = f"id:{tool_id}" if tool_id else (f"idx:{idx}" if idx else f"raw:{len(order)}")
        if idx: idx_to_key[idx] = key
        args = cb.get("input") if isinstance(cb.get("input"), dict) else {}
        _ensure_tool(tools, order, key, tool_id=tool_id, name=str(cb.get("name") or ""), arguments=args)
        return

    if typ == "response.output_item.added":
        item = raw.get("item") if isinstance(raw.get("item"), dict) else {}
        if item.get("type") != "function_call": return
        item_id = str(item.get("id") or "")
        call_id = str(item.get("call_id") or "")
        tool_id = call_id or item_id
        key = f"id:{tool_id}" if tool_id else f"raw:{len(order)}"
        if item_id: id_alias_to_key[item_id] = key
        if call_id: id_alias_to_key[call_id] = key
        args = item.get("arguments")
        if isinstance(args, str): args = _parse_json_args(args)
        if not isinstance(args, dict): args = {}
        _ensure_tool(tools, order, key, tool_id=tool_id, name=str(item.get("name") or ""), arguments=args)
        return

    # OpenAI-compatible chat.completions tool-call deltas.
    for tc in _chat_delta_tool_calls(raw):
        idx = tc.get("index")
        idxs = "" if idx is None else str(idx)
        tid = str(tc.get("id") or "")
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        prev = idx_to_key.get(idxs) if idxs else None
        key = f"id:{tid}" if tid else (prev or (f"idx:{idxs}" if idxs else f"raw:{len(order)}"))
        if idxs:
            if prev is None or (prev.startswith("idx:") and tid): idx_to_key[idxs] = key
        if tid: id_alias_to_key[tid] = key

        av = fn.get("arguments")
        args = {}
        if isinstance(av, dict): args = av
        elif isinstance(av, str):
            parsed = _parse_json_args(av)
            if parsed and "_raw" not in parsed: args = parsed
        _ensure_tool(tools, order, key, tool_id=tid, name=str(fn.get("name") or ""), arguments=args)


def _tool_key(tc: ToolCall, raw: RawEvent, idx_to_key: dict[str, str], id_alias_to_key: dict[str, str],
    order: list[str]) -> str:
    "Resolve a stable aggregation key for a normalized ToolCall."
    if tc.id and tc.id in id_alias_to_key: return id_alias_to_key[tc.id]
    if tc.id.startswith("toolu_") or tc.id.startswith("fc_"): return f"id:{tc.id}"

    if tc.id and tc.id.isdigit():
        if tc.id in idx_to_key: return idx_to_key[tc.id]
        idx = _index_from_raw(raw)
        if idx and idx in idx_to_key: return idx_to_key[idx]
        return f"idx:{tc.id}"

    if tc.id: return f"id:{tc.id}"
    return f"anon:{len(order)}"


def _final_tool_calls(tools: dict[str, _ToolBuf], order: list[str]) -> list[ToolCall]:
    "Build normalized final tool-call list from aggregation state."
    out = []
    for k in order:
        tb = tools[k]
        args = dict(tb.arguments or {})
        if tb.chunks:
            parsed = _parse_json_args("".join(tb.chunks))
            if parsed and "_raw" not in parsed: args = parsed
            elif not args: args = parsed
        out.append(ToolCall(id=(tb.id or k), name=tb.name, arguments=args))
    return out


async def acollect_stream(it: Any) -> StreamSummary:
    """Collect a Delta stream into text + metadata without dropping raw events.

    `it` can be either:
    - an async iterator of Delta events
    - an awaitable that resolves to an async iterator (e.g. `acompletion(..., stream=True)`)

    Returns a StreamSummary with:
    - `deltas`: all emitted Delta objects, in order
    - `raw_events`: all raw provider events (when present), in order
    - `final`: synthesized final Delta containing aggregate text/tool_calls/usage/finish,
      plus a raw payload including both `last_event` and full `events`.
    """
    if not hasattr(it, "__aiter__") and hasattr(it, "__await__"): it = await it
    if not hasattr(it, "__aiter__"): raise TypeError("acollect_stream expects an async iterator or awaitable stream")

    text, finish, usage = "", None, None
    deltas, raws = [], []
    tools, tool_order, idx_to_key, id_alias_to_key = {}, [], {}, {}

    async for d in it:
        deltas.append(d)
        if d.text: text += d.text
        if d.finish_reason is not None: finish = d.finish_reason
        if d.usage is not None: usage = d.usage
        raw = d.raw or {}
        if raw:
            raws.append(raw)
            _prime_tool_from_raw(raw, tools, tool_order, idx_to_key, id_alias_to_key)
        for tc in d.tool_calls or []:
            key = _tool_key(tc, raw, idx_to_key, id_alias_to_key, tool_order)
            if tc.id: id_alias_to_key.setdefault(tc.id, key)
            tb = _ensure_tool(tools, tool_order, key, tool_id=tc.id, name=tc.name, arguments=tc.arguments)
            if (chunk := _chunk_from_tool_call(tc)) is not None: tb.chunks.append(chunk)

    tcs = _final_tool_calls(tools, tool_order)
    raw = {}
    if raws: raw = {"last_event": raws[-1], "events": raws}
    final = Delta(text=text, tool_calls=tcs, finish_reason=finish, usage=usage, raw=raw)
    return StreamSummary(text=text, tool_calls=tcs, finish_reason=finish, usage=usage,
        deltas=deltas, raw_events=raws, final=final)


__all__ = "StreamSummary acollect_stream".split()
