"""Shared OpenAI-compatible client (OpenAI, OpenRouter, etc.)."""

from __future__ import annotations

from openai import AsyncOpenAI, OpenAI

from src.config import OPENAI_API_KEY, OPENAI_BASE_URL


def _client_kwargs() -> dict | None:
    if not OPENAI_API_KEY:
        return None
    kwargs: dict = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return kwargs


def create_llm_client() -> OpenAI | None:
    """Build sync client when API key is set; optional base_url for OpenRouter."""
    kwargs = _client_kwargs()
    if not kwargs:
        return None
    return OpenAI(**kwargs)


def create_async_llm_client() -> AsyncOpenAI | None:
    """Build async client when API key is set; optional base_url for OpenRouter."""
    kwargs = _client_kwargs()
    if not kwargs:
        return None
    return AsyncOpenAI(**kwargs)
