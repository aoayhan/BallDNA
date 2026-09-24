"""Small OpenAI Responses API adapter isolated from product logic."""

from __future__ import annotations

from dataclasses import dataclass

from ball_ai.config import settings


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str


class LLMUnavailableError(RuntimeError):
    """Raised when remote generation cannot be used."""


def generate_text(prompt: str, instructions: str) -> LLMResponse:
    """Generate text with the configured model through the Responses API."""

    if not settings.openai_api_key:
        raise LLMUnavailableError("OPENAI_API_KEY is not configured.")
    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        response = client.responses.create(
            model=settings.openai_model,
            instructions=instructions,
            input=prompt,
            text={"verbosity": "low"},
        )
        if not response.output_text:
            raise LLMUnavailableError("The model returned no text.")
        return LLMResponse(text=response.output_text.strip(), model=settings.openai_model)
    except LLMUnavailableError:
        raise
    except Exception as exc:
        raise LLMUnavailableError(f"Remote generation failed: {exc}") from exc

