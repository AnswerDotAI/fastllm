"SSE parsing helpers."

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Iterator, Optional

import httpx


@dataclass(frozen=True)
class SSEvent:
    "A parsed SSE event."
    event: Optional[str]
    data: str
    id: Optional[str] = None
    retry: Optional[int] = None


def parse_sse_lines(lines: Iterator[str]) -> Iterator[SSEvent]:
    "Parse SSE line stream into events."
    event, data, event_id, retry = None, [], None, None
    for raw in lines:
        line = raw.rstrip("\n")
        if not line:
            if data: yield SSEvent(event=event, data="\n".join(data), id=event_id, retry=retry)
            event, data, event_id, retry = None, [], None, None
            continue
        if line.startswith(":"): continue
        field, _, value = line.partition(":")
        if value.startswith(" "): value = value[1:]
        if field == "event": event = value
        elif field == "data": data.append(value)
        elif field == "id": event_id = value
        elif field == "retry":
            try: retry = int(value)
            except ValueError: retry = None
    if data: yield SSEvent(event=event, data="\n".join(data), id=event_id, retry=retry)


async def aiter_sse(response: httpx.Response) -> AsyncIterator[SSEvent]:
    "Async SSE parser from an httpx streamed response."
    event, data, event_id, retry = None, [], None, None
    async for raw in response.aiter_lines():
        line = raw.rstrip("\n")
        if not line:
            if data: yield SSEvent(event=event, data="\n".join(data), id=event_id, retry=retry)
            event, data, event_id, retry = None, [], None, None
            continue
        if line.startswith(":"): continue
        field, _, value = line.partition(":")
        if value.startswith(" "): value = value[1:]
        if field == "event": event = value
        elif field == "data": data.append(value)
        elif field == "id": event_id = value
        elif field == "retry":
            try: retry = int(value)
            except ValueError: retry = None
    if data: yield SSEvent(event=event, data="\n".join(data), id=event_id, retry=retry)
