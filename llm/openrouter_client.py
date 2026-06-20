"""
Thin, structured wrapper around OpenRouter's OpenAI-compatible API.

Why a wrapper instead of calling the OpenAI SDK directly everywhere:
- Single place to handle retries, timeouts, and error translation into
  our own LLMCallError (decoupling agents from a third-party SDK's
  exception types).
- Single place to enforce structured (JSON) output when a schema is
  required, e.g. for classification.
- Makes it trivial to swap providers later without touching agent code.
"""

import json
import time
from typing import Optional, Type, TypeVar

from openai import OpenAI, APIError, APITimeoutError
from pydantic import BaseModel, ValidationError

from config.settings import settings
from utils.exceptions import LLMCallError
from utils.logger import log_event, trace_step

T = TypeVar("T", bound=BaseModel)


class OpenRouterClient:
    """Synchronous client for chat completions via OpenRouter."""

    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )

    def chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_retries: int = 2,
    ) -> str:
        """
        Plain text completion. Raises LLMCallError on failure after retries.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                with trace_step("OpenRouterClient", "chat_completion", model=model, attempt=attempt):
                    response = self._client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=temperature,
                    )
                content = response.choices[0].message.content
                if not content:
                    raise LLMCallError(f"Empty response from model {model}")
                return content.strip()

            except (APIError, APITimeoutError) as exc:
                last_error = exc
                log_event(
                    "OpenRouterClient", "chat_completion_retry",
                    attempt=attempt, error=str(exc),
                )
                time.sleep(3 * attempt)  # simple backoff

        raise LLMCallError(
            f"OpenRouter chat call failed after {max_retries} attempts: {last_error}"
        )

    def chat_structured(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        output_schema: Type[T],
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> T:
        """
        Calls the model and parses its response into a Pydantic schema.

        The system prompt is expected to instruct the model to return
        ONLY valid JSON matching the schema. We additionally validate
        and retry on parse/validation failure, since free-tier models
        are not always perfectly compliant with structured output.
        """
        schema_instruction = (
            f"\n\nYou MUST respond with ONLY valid JSON matching this schema, "
            f"no preamble, no markdown fences:\n{output_schema.model_json_schema()}"
        )
        full_system_prompt = system_prompt + schema_instruction

        last_error: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            raw = self.chat(
                model=model,
                system_prompt=full_system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_retries=1,
            )
            cleaned = raw.replace("```json", "").replace("```", "").strip()
            try:
                data = json.loads(cleaned)
                return output_schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                log_event(
                    "OpenRouterClient", "structured_parse_retry",
                    attempt=attempt, raw_response=cleaned[:200], error=str(exc),
                )

        raise LLMCallError(
            f"Failed to parse structured output into {output_schema.__name__} "
            f"after {max_retries} attempts: {last_error}"
        )


# Singleton — import as `from llm.openrouter_client import openrouter_client`
openrouter_client = OpenRouterClient()