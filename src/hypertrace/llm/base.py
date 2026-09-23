"""Structured LLM protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResult[T: BaseModel]:
    value: T
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class StructuredLLM(Protocol):
    async def complete(
        self, model: str, system: str, user: str, schema: type[T]
    ) -> LLMResult[T]: ...
