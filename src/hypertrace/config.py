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
    review_primary_base_url: str | None = None
    review_primary_api_key: str | None = None
    review_primary_input_cost_per_million: float | None = None
    review_primary_output_cost_per_million: float | None = None
    review_fallback_base_url: str | None = None
    review_fallback_api_key: str | None = None
    review_fallback_model: str | None = None
    review_fallback_input_cost_per_million: float | None = None
    review_fallback_output_cost_per_million: float | None = None
    review_action_threshold: int = 25
    review_query_limit: int = 3
    max_consecutive_target_searches: int = 3

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
            review_primary_base_url=os.getenv("REVIEW_PRIMARY_BASE_URL") or None,
            review_primary_api_key=os.getenv("REVIEW_PRIMARY_API_KEY") or None,
            review_primary_input_cost_per_million=(
                float(value)
                if (value := os.getenv("REVIEW_PRIMARY_INPUT_COST_PER_MILLION"))
                else None
            ),
            review_primary_output_cost_per_million=(
                float(value)
                if (value := os.getenv("REVIEW_PRIMARY_OUTPUT_COST_PER_MILLION"))
                else None
            ),
            review_fallback_base_url=os.getenv("REVIEW_FALLBACK_BASE_URL") or None,
            review_fallback_api_key=os.getenv("REVIEW_FALLBACK_API_KEY") or None,
            review_fallback_model=os.getenv("REVIEW_FALLBACK_MODEL") or None,
            review_fallback_input_cost_per_million=(
                float(value)
                if (value := os.getenv("REVIEW_FALLBACK_INPUT_COST_PER_MILLION"))
                else None
            ),
            review_fallback_output_cost_per_million=(
                float(value)
                if (value := os.getenv("REVIEW_FALLBACK_OUTPUT_COST_PER_MILLION"))
                else None
            ),
            review_action_threshold=int(os.getenv("HYPERTRACE_REVIEW_ACTION_THRESHOLD", "25")),
            review_query_limit=int(os.getenv("HYPERTRACE_REVIEW_QUERY_LIMIT", "3")),
            max_consecutive_target_searches=int(
                os.getenv("HYPERTRACE_MAX_CONSECUTIVE_TARGET_SEARCHES", "3")
            ),
        )

    def require_online(self, max_cost: float | None = None) -> None:
        if self.review_action_threshold < 1:
            raise ValueError("HYPERTRACE_REVIEW_ACTION_THRESHOLD must be positive")
        if self.max_consecutive_target_searches < 1:
            raise ValueError("HYPERTRACE_MAX_CONSECUTIVE_TARGET_SEARCHES must be positive")
        if not 0 <= self.review_query_limit <= 3:
            raise ValueError("HYPERTRACE_REVIEW_QUERY_LIMIT must be between 0 and 3")
        if not math.isfinite(self.local_usage_multiplier) or self.local_usage_multiplier < 1:
            raise ValueError("HYPERTRACE_LOCAL_USAGE_MULTIPLIER must be at least 1")
        if not self.llm_api_key:
            raise ValueError("Set LLM_API_KEY for research runs")
        if not self.brave_api_key:
            raise ValueError("Set BRAVE_SEARCH_API_KEY for research runs")
        if bool(self.review_primary_base_url) != bool(self.review_primary_api_key):
            raise ValueError("Set both REVIEW_PRIMARY_BASE_URL and REVIEW_PRIMARY_API_KEY")
        if bool(self.review_fallback_base_url) != bool(self.review_fallback_api_key):
            raise ValueError("Set both REVIEW_FALLBACK_BASE_URL and REVIEW_FALLBACK_API_KEY")
        if self.review_fallback_base_url and (
            self.review_primary_input_cost_per_million is None
            or self.review_primary_output_cost_per_million is None
        ):
            raise ValueError("Set REVIEW_PRIMARY_*_COST_PER_MILLION for review failover")
        if self.review_fallback_base_url and (
            self.review_fallback_input_cost_per_million is None
            or self.review_fallback_output_cost_per_million is None
            or not math.isfinite(self.review_fallback_input_cost_per_million)
            or not math.isfinite(self.review_fallback_output_cost_per_million)
            or self.review_fallback_input_cost_per_million < 0
            or self.review_fallback_output_cost_per_million < 0
        ):
            raise ValueError("Set REVIEW_FALLBACK_*_COST_PER_MILLION for fallback billing")
        if any(
            rate is not None and (not math.isfinite(rate) or rate < 0)
            for rate in (
                self.review_primary_input_cost_per_million,
                self.review_primary_output_cost_per_million,
            )
        ):
            raise ValueError("Review primary costs must be nonnegative")
        if max_cost is not None and (
            self.input_cost_per_million <= 0 or self.output_cost_per_million <= 0
        ):
            raise ValueError("Set both LLM_*_COST_PER_MILLION values to enforce --max-cost")
