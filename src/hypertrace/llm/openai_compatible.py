"""OpenAI-compatible chat completions with validated JSON and bounded retries."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from hypertrace.llm.base import LLMResult, T

logger = logging.getLogger(__name__)
PREVIEW_END_CHARS = 160


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


@dataclass(frozen=True)
class StructuredCallPolicy:
    max_completion_tokens: int = 8000
    timeout_seconds: int = 240
    reasoning_effort: str | None = None


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


def _sanitized_preview(content: str) -> str:
    if len(content) > PREVIEW_END_CHARS * 2:
        content = f"{content[:PREVIEW_END_CHARS]}\n…\n{content[-PREVIEW_END_CHARS:]}"
    # Preserve punctuation, whitespace, and fence shape without storing response text.
    return "".join("x" if char.isalnum() or char == "_" else char for char in content)


def _repair_messages(schema_text: str, content: str, error: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Repair this response to match the JSON schema: "
                f"{schema_text}. Preserve every substantive value and statement. "
                "Do not add findings, queries, evidence, or claims. If required information "
                "is absent, leave it absent; do not invent it. Return only corrected JSON; "
                "spend minimal reasoning."
            ),
        },
        {
            "role": "user",
            "content": (
                "Correct only the representation; do not redo the review or research.\n"
                f"Invalid response: {content}\nValidation error: {error}"
            ),
        },
    ]


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

    def structured_call_policy(self, model: str) -> StructuredCallPolicy:
        reasoning_effort = (
            "low"
            if model == "glm-5.3-flash"
            and urlsplit(self.base_url).hostname == "api.cheaperinference.com"
            else None
        )
        return StructuredCallPolicy(reasoning_effort=reasoning_effort)

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def complete(self, model: str, system: str, user: str, schema: type[T]) -> LLMResult[T]:
        policy = self.structured_call_policy(model)
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
        for attempt in range(self.max_retries + 2):
            is_repair = repair_pending
            attempt_kind = "repair" if is_repair else "initial" if attempt == 0 else "retry"
            body: dict = {
                "model": model,
                "messages": messages,
                "temperature": 0,
            }
            body["max_completion_tokens"] = policy.max_completion_tokens
            if self.json_mode:
                body["response_format"] = {"type": "json_object"}
            if policy.reasoning_effort:
                body["reasoning_effort"] = policy.reasoning_effort
            observer = self.attempt_observer
            attempt_id = observer.started(model) if observer else None
            attempt_input: int | None = None
            attempt_output: int | None = None
            outcome = "unknown_failure"
            diagnostics: dict = _response_diagnostics(None, None, self.api_key)
            diagnostics["attempt_kind"] = attempt_kind
            try:
                response = await self.client.post(
                    f"{self.base_url}/chat/completions",
                    json=body,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=policy.timeout_seconds,
                )
                try:
                    payload = response.json()
                except json.JSONDecodeError:
                    payload = None
                diagnostics = _response_diagnostics(response.status_code, payload, self.api_key)
                diagnostics["attempt_kind"] = attempt_kind
                diagnostics["max_completion_tokens"] = policy.max_completion_tokens
                diagnostics["reasoning_effort_requested"] = policy.reasoning_effort
                response.raise_for_status()
                if not isinstance(payload, dict):
                    raise ProviderOutputError("Provider returned a non-object or non-JSON response")
                if payload.get("error") is not None:
                    raise ProviderOutputError("Provider returned an error field")
                usage = payload.get("usage") or {}
                if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                    diagnostics["reported_completion_tokens"] = usage["completion_tokens"]
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
                diagnostics["content_chars"] = len(content) if isinstance(content, str) else None
                if isinstance(content, str):
                    diagnostics["content_sha256"] = sha256(content.encode()).hexdigest()
                if content is None or (isinstance(content, str) and not content.strip()):
                    diagnostics["structured_content_state"] = "empty"
                    raise ProviderOutputError(
                        f"Provider returned {diagnostics['content_state']} completion content"
                    )
                if not isinstance(content, str):
                    raise ProviderOutputError("Provider returned non-string completion content")
                if choice.get("finish_reason") == "length":
                    raise ProviderOutputError("Provider stopped at the completion token limit")
                normalized = content.strip()
                fence = re.fullmatch(
                    r"```(?:json)?[ \t]*\r?\n(.*)\r?\n```", normalized, re.DOTALL | re.IGNORECASE
                )
                if fence:
                    normalized = fence.group(1).strip()
                try:
                    invalid_object = json.loads(normalized)
                except json.JSONDecodeError as exc:
                    diagnostics["structured_content_state"] = "malformed_json"
                    diagnostics["json_parse_error"] = {
                        "message": exc.msg,
                        "line": exc.lineno,
                        "column": exc.colno,
                        "position": exc.pos,
                    }
                    diagnostics["response_preview"] = _sanitized_preview(content)
                    if not is_repair:
                        messages = _repair_messages(
                            schema_text, content, f"JSON parse error: {exc}"
                        )
                        repair_pending = True
                    raise StructuredOutputError(
                        f"Completion contained malformed JSON: {exc}"
                    ) from exc
                try:
                    value = schema.model_validate_json(normalized)
                except ValidationError as exc:
                    error_types = sorted({item["type"] for item in exc.errors()})
                    errors = exc.errors(include_input=False, include_url=False)
                    diagnostics["structured_content_state"] = "valid_json_schema_invalid"
                    diagnostics["validation_errors"] = errors
                    diagnostics["response_preview"] = _sanitized_preview(content)
                    if not is_repair:
                        messages = _repair_messages(
                            schema_text,
                            json.dumps(invalid_object, ensure_ascii=False),
                            f"Schema validation errors: {json.dumps(errors, ensure_ascii=False)}",
                        )
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
                    if (
                        isinstance(exc, ProviderOutputError)
                        and diagnostics["content_state"] in {"empty_string", "blank_string", "null"}
                        and (diagnostics["reasoning_content_chars"] or 0) > 0
                    ):
                        instruction = (
                            "The provider used its completion on reasoning and returned no final JSON. "
                            "Respond immediately with concise JSON matching the schema; "
                            "use minimal reasoning."
                        )
                    else:
                        instruction = (
                            f"Previous output failed validation: {str(exc)[:300]}. "
                            "Return a corrected JSON object only."
                        )
                    messages.append(
                        {
                            "role": "user",
                            "content": instruction,
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
