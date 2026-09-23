"""OpenAI-compatible chat completions with validated JSON and bounded retries."""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from hypertrace.llm.base import LLMResult, T
from hypertrace.models import AdversarialReview

logger = logging.getLogger(__name__)


class AttemptObserver(Protocol):
    def started(self, model: str) -> int: ...

    def finished(
        self,
        attempt_id: int,
        outcome: str,
        input_tokens: int | None,
        output_tokens: int | None,
        diagnostics: dict,
    ) -> None: ...


class StructuredOutputError(ValueError):
    def __init__(self, message: str, input_tokens: int = 0, output_tokens: int = 0):
        super().__init__(message)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class ProviderOutputError(StructuredOutputError):
    """The provider returned no usable completion text."""


def _safe_label(value: object, secret: str) -> str:
    if (
        isinstance(value, str)
        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,79}", value)
        and (not secret or secret not in value)
    ):
        return value
    return "<redacted>"


def _response_diagnostics(status: int | None, payload: object, secret: str = "") -> dict:
    data = payload if isinstance(payload, dict) else {}
    choices = data.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    choice = first if isinstance(first, dict) else {}
    message = choice.get("message")
    message = message if isinstance(message, dict) else {}
    if "content" not in message:
        content_state = "missing"
    elif message["content"] is None:
        content_state = "null"
    elif message["content"] == "":
        content_state = "empty_string"
    elif isinstance(message["content"], str) and not message["content"].strip():
        content_state = "blank_string"
    elif isinstance(message["content"], str):
        content_state = "non_empty"
    else:
        content_state = "non_string"
    error = data.get("error")
    if isinstance(error, dict):
        provider_error = {
            "keys": sorted(_safe_label(key, secret) for key in error),
            "type": _safe_label(error["type"], secret) if "type" in error else None,
            "code": _safe_label(error["code"], secret) if "code" in error else None,
            "message_present": "message" in error,
        }
    elif error is not None:
        provider_error = {"value_type": type(error).__name__}
    else:
        provider_error = None
    reason = choice.get("finish_reason")
    return {
        "http_status": status,
        "finish_reason": _safe_label(reason, secret) if reason is not None else None,
        "choices_present": "choices" in data,
        "choice_count": len(choices) if isinstance(choices, list) else None,
        "choice_keys": sorted(_safe_label(key, secret) for key in choice),
        "message_keys": sorted(_safe_label(key, secret) for key in message),
        "content_state": content_state,
        "reasoning_content_chars": (
            len(message["reasoning_content"])
            if isinstance(message.get("reasoning_content"), str)
            else None
        ),
        "reasoning_chars": (
            len(message["reasoning"]) if isinstance(message.get("reasoning"), str) else None
        ),
        "provider_error": provider_error,
        "response_keys": sorted(_safe_label(key, secret) for key in data),
    }


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        json_mode: bool = True,
        max_retries: int = 2,
        client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.json_mode = json_mode
        self.max_retries = max_retries
        self.client = client or httpx.AsyncClient(timeout=60)
        self._owns_client = client is None
        self.attempt_observer: AttemptObserver | None = None

    @staticmethod
    def max_tokens_for(schema: type) -> int:
        # Review is the largest response and reasoning-capable providers may spend
        # completion tokens before producing message.content.
        return 8000 if schema is AdversarialReview else 1500

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def complete(self, model: str, system: str, user: str, schema: type[T]) -> LLMResult[T]:
        schema_text = json.dumps(schema.model_json_schema(), separators=(",", ":"))
        messages = [
            {
                "role": "system",
                "content": f"{system}\nReturn only JSON matching this schema: {schema_text}",
            },
            {"role": "user", "content": user},
        ]
        last_error: Exception | None = None
        input_tokens = 0
        output_tokens = 0
        repair_pending = False
        for attempt in range(
            self.max_retries + 2 if schema is AdversarialReview else self.max_retries + 1
        ):
            is_repair = repair_pending
            body: dict = {
                "model": model,
                "messages": messages,
                "temperature": 0,
            }
            token_field = "max_completion_tokens" if schema is AdversarialReview else "max_tokens"
            body[token_field] = self.max_tokens_for(schema)
            if self.json_mode:
                body["response_format"] = {"type": "json_object"}
            if (
                schema is AdversarialReview
                and model == "glm-5.3-flash"
                and urlsplit(self.base_url).hostname == "api.cheaperinference.com"
            ):
                body["reasoning_effort"] = "low"
            observer = self.attempt_observer
            attempt_id = observer.started(model) if observer else None
            attempt_input: int | None = None
            attempt_output: int | None = None
            outcome = "unknown_failure"
            diagnostics: dict = _response_diagnostics(None, None, self.api_key)
            try:
                response = await self.client.post(
                    f"{self.base_url}/chat/completions",
                    json=body,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=240 if schema is AdversarialReview else 60,
                )
                try:
                    payload = response.json()
                except json.JSONDecodeError:
                    payload = None
                diagnostics = _response_diagnostics(response.status_code, payload, self.api_key)
                response.raise_for_status()
                if not isinstance(payload, dict):
                    raise ProviderOutputError("Provider returned a non-object or non-JSON response")
                if payload.get("error") is not None:
                    raise ProviderOutputError("Provider returned an error field")
                usage = payload.get("usage") or {}
                if isinstance(usage, dict) and (
                    usage.get("prompt_tokens") is not None
                    and usage.get("completion_tokens") is not None
                ):
                    attempt_input = int(usage["prompt_tokens"])
                    attempt_output = int(usage["completion_tokens"])
                    input_tokens += attempt_input
                    output_tokens += attempt_output
                choices = payload.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise ProviderOutputError("Provider returned no completion choices")
                choice = choices[0]
                if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
                    raise ProviderOutputError("Provider returned no completion message")
                content = choice["message"].get("content")
                if content is None or (isinstance(content, str) and not content.strip()):
                    raise ProviderOutputError(
                        f"Provider returned {diagnostics['content_state']} completion content"
                    )
                if not isinstance(content, str):
                    raise ProviderOutputError("Provider returned non-string completion content")
                if choice.get("finish_reason") == "length":
                    raise ProviderOutputError("Provider stopped at the completion token limit")
                try:
                    value = schema.model_validate_json(content)
                except ValidationError as exc:
                    error_types = sorted({item["type"] for item in exc.errors()})
                    if schema is AdversarialReview and not is_repair:
                        try:
                            invalid_object = json.loads(content)
                        except json.JSONDecodeError:
                            pass
                        else:
                            errors = exc.errors(
                                include_input=False, include_context=False, include_url=False
                            )
                            messages = [
                                {
                                    "role": "system",
                                    "content": (
                                        "Correct the JSON object to match this schema: "
                                        f"{schema_text}. Return only JSON; spend minimal reasoning."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": (
                                        "Correct this JSON object to match the schema. "
                                        "Return only the corrected JSON object; do not redo the review.\n"
                                        f"Invalid object: {json.dumps(invalid_object, ensure_ascii=False)}\n"
                                        f"Schema validation errors: {json.dumps(errors, ensure_ascii=False)}"
                                    ),
                                },
                            ]
                            repair_pending = True
                    raise StructuredOutputError(
                        f"Completion failed schema validation: {', '.join(error_types)}"
                    ) from exc
                outcome = "validated"
                return LLMResult(
                    value=value,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=payload.get("model", model),
                )
            except StructuredOutputError as exc:
                last_error = exc
                outcome = (
                    "provider_output_failure"
                    if isinstance(exc, ProviderOutputError)
                    else "invalid_response"
                )
                if not repair_pending:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Previous output failed validation: {str(exc)[:300]}. "
                                "Return a corrected JSON object only."
                            ),
                        }
                    )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                outcome = "transport_failure"
            except httpx.HTTPStatusError as exc:
                outcome = f"http_{exc.response.status_code}"
                if exc.response.status_code not in (429, 500, 502, 503, 504):
                    raise
                last_error = exc
            finally:
                logger.info(
                    "completion_attempt %s", json.dumps({"outcome": outcome, **diagnostics})
                )
                if observer is not None and attempt_id is not None:
                    observer.finished(
                        attempt_id, outcome, attempt_input, attempt_output, diagnostics
                    )
            if is_repair:
                break
            if repair_pending or attempt < self.max_retries:
                continue
            break
        raise StructuredOutputError(
            f"Structured response failed after bounded retries: {last_error}",
            input_tokens,
            output_tokens,
        )
