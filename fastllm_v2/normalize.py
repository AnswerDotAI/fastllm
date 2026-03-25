"Provider payload normalization helpers."

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .errors import ProtocolError
from .types import Completion, Delta, Msg, Part, ToolCall, Usage


def _json_dict(s: Any) -> Dict[str, Any]:
    "Parse json string into dict, preserving raw on failure."
    if isinstance(s, dict): return s
    if not isinstance(s, str): return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {"_raw": s}


def _gemini_tool_calls(parts: list[Any]) -> list[ToolCall]:
    "Extract Gemini functionCall parts as normalized tool calls."
    out = []
    for i,p in enumerate(parts):
        if not isinstance(p, dict): continue
        fc = p.get("functionCall") if isinstance(p.get("functionCall"), dict) else p.get("function_call")
        if not isinstance(fc, dict): continue
        args = fc.get("args")
        out.append(ToolCall(
            id=str(fc.get("id", fc.get("call_id", f"call_{i}"))),
            name=str(fc.get("name", "")),
            arguments=args if isinstance(args, dict) else _json_dict(args)))
    return out


def usage_from_openai(raw: Optional[dict]) -> Optional[Usage]:
    "Normalize OpenAI usage shape(s)."
    if not isinstance(raw, dict): return None
    pt = int(raw.get("prompt_tokens", raw.get("input_tokens", 0)) or 0)
    ct = int(raw.get("completion_tokens", raw.get("output_tokens", 0)) or 0)
    tt = int(raw.get("total_tokens", pt + ct) or (pt + ct))
    return Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=tt, raw=raw)


def usage_from_anthropic(raw: Optional[dict]) -> Optional[Usage]:
    "Normalize Anthropic usage shape."
    if not isinstance(raw, dict): return None
    pt = int(raw.get("input_tokens", 0) or 0)
    ct = int(raw.get("output_tokens", 0) or 0)
    return Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct, raw=raw)


def usage_from_gemini(raw: Optional[dict]) -> Optional[Usage]:
    "Normalize Gemini usageMetadata shape."
    if not isinstance(raw, dict): return None
    pt = int(raw.get("promptTokenCount", 0) or 0)
    ct = int(raw.get("candidatesTokenCount", 0) or 0)
    tt = int(raw.get("totalTokenCount", pt + ct) or (pt + ct))
    return Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=tt, raw=raw)


def normalize_openai_response(raw: Dict[str, Any], *, model: str, provider: str = "openai") -> Completion:
    "Normalize OpenAI Responses API object into Completion."
    out = raw.get("output") or []
    parts, tool_calls = [], []
    for item in out:
        if not isinstance(item, dict): continue
        typ = item.get("type")
        if typ == "message":
            for c in item.get("content") or []:
                if not isinstance(c, dict): continue
                ctyp = c.get("type")
                if ctyp in ("output_text", "text"):
                    parts.append(Part(type="text", text=c.get("text", ""), data=c))
                else:
                    parts.append(Part(type=str(ctyp or "part"), data=c))
        elif typ == "function_call":
            tc = ToolCall(id=str(item.get("call_id", item.get("id", ""))), name=str(item.get("name", "")),
                arguments=_json_dict(item.get("arguments")))
            tool_calls.append(tc)
    if not parts:
        txt = raw.get("output_text")
        if isinstance(txt, str) and txt: parts.append(Part(type="text", text=txt))
        else: parts.append(Part(type="text", text=""))
    return Completion(
        model=str(raw.get("model") or model),
        message=Msg(role="assistant", content=parts),
        finish_reason=str(raw.get("status") or "completed"),
        usage=usage_from_openai(raw.get("usage")),
        tool_calls=tool_calls,
        provider=provider,
        raw=raw)


def normalize_openai_response_event(ev: Dict[str, Any]) -> Optional[Delta]:
    "Normalize OpenAI Responses API stream event into Delta."
    typ = ev.get("type")
    if typ == "response.output_text.delta":
        return Delta(text=str(ev.get("delta") or ""), raw=ev)
    if typ == "response.function_call_arguments.delta":
        tc = ToolCall(id=str(ev.get("item_id", "")), name=str(ev.get("name", "")),
            arguments={"_delta": str(ev.get("delta") or "")})
        return Delta(tool_calls=[tc], raw=ev)
    if typ == "response.completed":
        rsp = ev.get("response") if isinstance(ev.get("response"), dict) else {}
        return Delta(finish_reason=str(rsp.get("status") or "completed"), usage=usage_from_openai(rsp.get("usage")), raw=ev)
    if typ == "error":
        msg = ev.get("error") if isinstance(ev.get("error"), dict) else ev
        raise ProtocolError(f"Responses stream error: {msg}")
    return None


def normalize_openai_chat_completion(raw: Dict[str, Any], *, model: str, provider: str = "openai_chat") -> Completion:
    "Normalize chat.completions response object into Completion."
    choices = raw.get("choices") or []
    if not choices: raise ProtocolError("OpenAI chat response missing choices")
    choice = choices[0] if isinstance(choices[0], dict) else {}
    msg = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    parts, tool_calls = [], []
    cts = msg.get("content")
    if isinstance(cts, str): parts.append(Part(type="text", text=cts))
    elif isinstance(cts, list):
        for c in cts:
            if isinstance(c, dict): parts.append(Part(type=str(c.get("type", "text")), text=c.get("text"), data=c))
    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict): continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        tool_calls.append(ToolCall(id=str(tc.get("id", "")), name=str(fn.get("name", "")),
            arguments=_json_dict(fn.get("arguments"))))
    if not parts: parts = [Part(type="text", text="")]
    return Completion(
        model=str(raw.get("model") or model),
        message=Msg(role="assistant", content=parts),
        finish_reason=choice.get("finish_reason"),
        usage=usage_from_openai(raw.get("usage")),
        tool_calls=tool_calls,
        provider=provider,
        raw=raw)


def normalize_openai_chat_delta(ev: Dict[str, Any]) -> Delta:
    "Normalize a chat completion stream event."
    choices = ev.get("choices") or []
    if not choices or not isinstance(choices[0], dict): return Delta(raw=ev)
    c0 = choices[0]
    d = c0.get("delta") if isinstance(c0.get("delta"), dict) else {}
    txt = d.get("content") if isinstance(d.get("content"), str) else ""
    tcs = []
    for tc in d.get("tool_calls") or []:
        if not isinstance(tc, dict): continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        tcs.append(ToolCall(id=str(tc.get("id", "")), name=str(fn.get("name", "")), arguments=_json_dict(fn.get("arguments"))))
    return Delta(text=txt, tool_calls=tcs, finish_reason=c0.get("finish_reason"), usage=usage_from_openai(ev.get("usage")), raw=ev)


def normalize_anthropic_message(raw: Dict[str, Any], *, model: str, provider: str = "anthropic") -> Completion:
    "Normalize Anthropic message response into Completion."
    content = raw.get("content") or []
    parts, tool_calls = [], []
    for b in content:
        if not isinstance(b, dict): continue
        typ = b.get("type")
        if typ == "text": parts.append(Part(type="text", text=b.get("text", ""), data=b))
        elif typ == "tool_use":
            tool_calls.append(ToolCall(id=str(b.get("id", "")), name=str(b.get("name", "")),
                arguments=b.get("input") if isinstance(b.get("input"), dict) else {}))
            parts.append(Part(type="tool_use", data=b))
    if not parts: parts = [Part(type="text", text="")]
    return Completion(
        model=str(raw.get("model") or model),
        message=Msg(role="assistant", content=parts),
        finish_reason=raw.get("stop_reason"),
        usage=usage_from_anthropic(raw.get("usage")),
        tool_calls=tool_calls,
        provider=provider,
        raw=raw)


def normalize_anthropic_event(ev: Dict[str, Any]) -> Optional[Delta]:
    "Normalize Anthropic SSE event into Delta."
    typ = ev.get("type")
    if typ == "content_block_delta":
        d = ev.get("delta") if isinstance(ev.get("delta"), dict) else {}
        if d.get("type") == "text_delta": return Delta(text=str(d.get("text", "")), raw=ev)
        if d.get("type") == "input_json_delta":
            tc = ToolCall(id=str(ev.get("index", "")), name="", arguments={"_delta": str(d.get("partial_json") or "")})
            return Delta(tool_calls=[tc], raw=ev)
        return None
    if typ == "content_block_start":
        b = ev.get("content_block") if isinstance(ev.get("content_block"), dict) else {}
        if b.get("type") != "tool_use": return None
        tc = ToolCall(id=str(b.get("id", "")), name=str(b.get("name", "")),
            arguments=b.get("input") if isinstance(b.get("input"), dict) else {})
        return Delta(tool_calls=[tc], raw=ev)
    if typ == "message_delta":
        d = ev.get("delta") if isinstance(ev.get("delta"), dict) else {}
        return Delta(finish_reason=d.get("stop_reason"), usage=usage_from_anthropic(ev.get("usage")), raw=ev)
    if typ == "message_stop": return Delta(finish_reason="message_stop", raw=ev)
    if typ == "error": raise ProtocolError(f"Anthropic stream error: {ev}")
    return None


def normalize_gemini_generate(raw: Dict[str, Any], *, model: str, provider: str = "gemini") -> Completion:
    "Normalize Gemini generateContent response."
    cands = raw.get("candidates") or []
    text, finish = "", None
    if cands and isinstance(cands[0], dict):
        c0 = cands[0]
        finish = c0.get("finishReason")
        content = c0.get("content") if isinstance(c0.get("content"), dict) else {}
        parts = content.get("parts") if isinstance(content.get("parts"), list) else []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str))
        tool_calls = _gemini_tool_calls(parts)
    else:
        tool_calls = []
    return Completion(
        model=model,
        message=Msg(role="assistant", content=[Part(type="text", text=text)]),
        finish_reason=finish,
        usage=usage_from_gemini(raw.get("usageMetadata")),
        tool_calls=tool_calls,
        provider=provider,
        raw=raw)


def normalize_gemini_event(ev: Dict[str, Any], emitted: str) -> Delta:
    "Normalize Gemini stream event, returning incremental text delta."
    cands = ev.get("candidates") if isinstance(ev.get("candidates"), list) else []
    txt, finish = "", None
    if cands and isinstance(cands[0], dict):
        c0 = cands[0]
        finish = c0.get("finishReason")
        content = c0.get("content") if isinstance(c0.get("content"), dict) else {}
        parts = content.get("parts") if isinstance(content.get("parts"), list) else []
        txt = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str))
        tool_calls = _gemini_tool_calls(parts)
    else:
        tool_calls = []
    delta_txt = txt[len(emitted):] if txt.startswith(emitted) else txt
    return Delta(text=delta_txt, tool_calls=tool_calls, finish_reason=finish,
        usage=usage_from_gemini(ev.get("usageMetadata")), raw=ev)
