"Client configuration."

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ClientConfig:
    "Runtime configuration for a fastllm_v2 client."
    model: str
    api_key: Optional[str] = None
    base_url: str = ""
    timeout: float = 60.0
    provider: str = ""
    default_headers: Dict[str, str] = field(default_factory=dict)
