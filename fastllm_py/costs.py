"Provider-agnostic token cost estimation."

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .types import Completion, Delta, Usage


@dataclass(frozen=True)
class ModelPrice:
    "Per-million-token pricing for a model."
    prompt_per_million: float
    completion_per_million: float
    cached_prompt_per_million: Optional[float] = None
    currency: str = "USD"


@dataclass(frozen=True)
class CostBreakdown:
    "Computed request cost from usage + model pricing."
    model: str
    currency: str
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int
    prompt_cost: float
    completion_cost: float
    cached_prompt_cost: float
    total_cost: float


def _to_price(v: Any) -> ModelPrice:
    "Coerce a mapping/dataclass into ModelPrice."
    if isinstance(v, ModelPrice): return v
    if isinstance(v, Mapping):
        pm = float(v.get("prompt_per_million", v.get("prompt", 0.0)) or 0.0)
        cm = float(v.get("completion_per_million", v.get("completion", 0.0)) or 0.0)
        cpm = v.get("cached_prompt_per_million", v.get("cached_prompt"))
        cpm = None if cpm is None else float(cpm)
        cur = str(v.get("currency", "USD"))
        return ModelPrice(prompt_per_million=pm, completion_per_million=cm,
            cached_prompt_per_million=cpm, currency=cur)
    raise TypeError(f"Unsupported price spec: {type(v).__name__}")


def _cached_prompt_tokens(raw: Mapping[str, Any]) -> int:
    "Best-effort cached prompt token extraction across provider shapes."
    cands = [
        raw.get("cached_input_tokens"),
        raw.get("input_cached_tokens"),
        raw.get("cached_tokens"),
        raw.get("cache_read_input_tokens"),
    ]
    d = raw.get("prompt_tokens_details")
    if isinstance(d, Mapping): cands.append(d.get("cached_tokens"))
    for v in cands:
        if v is None: continue
        try: return max(0, int(v))
        except (TypeError, ValueError):
            continue
    return 0


def _extract_usage(x: Any) -> tuple[Usage, Optional[str]]:
    "Extract Usage + optional model from Completion/Delta/Usage/dict."
    if isinstance(x, Completion): return x.usage or Usage(), x.model
    if isinstance(x, Delta): return x.usage or Usage(), None
    if isinstance(x, Usage): return x, None
    if isinstance(x, Mapping):
        pt = int(x.get("prompt_tokens", x.get("input_tokens", 0)) or 0)
        ct = int(x.get("completion_tokens", x.get("output_tokens", 0)) or 0)
        tt = int(x.get("total_tokens", pt + ct) or (pt + ct))
        mdl = x.get("model")
        return Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=tt,
            raw=dict(x)), str(mdl) if mdl is not None else None
    raise TypeError(f"Unsupported usage payload: {type(x).__name__}")


def _resolve_price(model: str, prices: Mapping[str, Any]) -> Optional[ModelPrice]:
    "Resolve exact or wildcard (* suffix) model pricing entry."
    if model in prices: return _to_price(prices[model])
    for k,v in prices.items():
        if not isinstance(k, str): continue
        if k.endswith("*") and model.startswith(k[:-1]): return _to_price(v)
    return None


def estimate_cost(x: Any, *, model: str = "", prices: Optional[Mapping[str, Any]] = None,
    strict: bool = True) -> CostBreakdown:
    """Estimate token cost for a Completion/Delta/Usage-like object.

    Args:
      x: Completion, Delta, Usage, or usage dict.
      model: Optional model override.
      prices: Model pricing map keyed by exact model or wildcard prefix (e.g. "gpt-4.1*").
      strict: Raise if pricing is missing; if false, return zero-cost breakdown.
    """
    usage, mdl = _extract_usage(x)
    mdl = model or mdl or ""
    tbl = prices or {}
    price = _resolve_price(mdl, tbl) if mdl else None
    if price is None:
        if strict: raise KeyError(f"No pricing found for model: {mdl or '<missing>'}")
        price = ModelPrice(prompt_per_million=0.0, completion_per_million=0.0)

    cached_pt = _cached_prompt_tokens(usage.raw or {})
    cached_pt = min(cached_pt, max(0, usage.prompt_tokens))
    normal_pt = max(0, usage.prompt_tokens - cached_pt)

    prompt_cost = (normal_pt / 1_000_000.0) * price.prompt_per_million
    completion_cost = (max(0, usage.completion_tokens) / 1_000_000.0) * price.completion_per_million
    cached_rate = price.cached_prompt_per_million if price.cached_prompt_per_million is not None else price.prompt_per_million
    cached_prompt_cost = (cached_pt / 1_000_000.0) * cached_rate

    total = prompt_cost + completion_cost + cached_prompt_cost
    return CostBreakdown(
        model=mdl,
        currency=price.currency,
        prompt_tokens=max(0, usage.prompt_tokens),
        completion_tokens=max(0, usage.completion_tokens),
        cached_prompt_tokens=cached_pt,
        prompt_cost=prompt_cost,
        completion_cost=completion_cost,
        cached_prompt_cost=cached_prompt_cost,
        total_cost=total)
