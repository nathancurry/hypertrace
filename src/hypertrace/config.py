"""Environment-backed runtime configuration."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    db_path: Path
    llm_base_url: str
    llm_api_key: str
    router_model: str
    research_model: str
    review_model: str
    brave_api_key: str
    input_cost_per_million: float
    output_cost_per_million: float
    min_yield: float
    json_mode: bool
    local_usage_multiplier: float = 1.25

    @classmethod
    def from_env(cls) -> Config:
        model = os.getenv("RESEARCH_MODEL", "glm-5.3-flash")
        return cls(
            db_path=Path(os.getenv("HYPERTRACE_DB", "hypertrace.db")).expanduser(),
            llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            router_model=os.getenv("ROUTER_MODEL", model),
            research_model=model,
            review_model=os.getenv("REVIEW_MODEL", model),
            brave_api_key=os.getenv("BRAVE_SEARCH_API_KEY", ""),
            input_cost_per_million=float(os.getenv("LLM_INPUT_COST_PER_MILLION", "0")),
            output_cost_per_million=float(os.getenv("LLM_OUTPUT_COST_PER_MILLION", "0")),
            min_yield=float(os.getenv("HYPERTRACE_MIN_YIELD", "0.05")),
            json_mode=os.getenv("LLM_JSON_MODE", "true").lower() == "true",
            local_usage_multiplier=float(os.getenv("HYPERTRACE_LOCAL_USAGE_MULTIPLIER", "1.25")),
        )

    def require_online(self, max_cost: float | None = None) -> None:
        if not math.isfinite(self.local_usage_multiplier) or self.local_usage_multiplier < 1:
            raise ValueError("HYPERTRACE_LOCAL_USAGE_MULTIPLIER must be at least 1")
        if not self.llm_api_key:
            raise ValueError("Set LLM_API_KEY for research runs")
        if not self.brave_api_key:
            raise ValueError("Set BRAVE_SEARCH_API_KEY for research runs")
        if max_cost is not None and (
            self.input_cost_per_million <= 0 or self.output_cost_per_million <= 0
        ):
            raise ValueError("Set both LLM_*_COST_PER_MILLION values to enforce --max-cost")
