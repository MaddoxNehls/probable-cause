"""LLM backends. Provider SDKs are imported lazily, so replay needs neither."""

from __future__ import annotations

import os
from pathlib import Path

from probable_cause.llm.base import (
    Backend,
    BackendError,
    Completion,
    CompletionRequest,
    Message,
)
from probable_cause.llm.cache import CachingBackend, CallBudget
from probable_cause.llm.client import LLM, Usage
from probable_cause.llm.replay import RecordingBackend, ReplayBackend

BACKENDS = ("anthropic", "openrouter", "claude-cli", "replay")
DEFAULT_CACHE_DIR = Path("benchmark/llm-cache")
CLAUDE_CLI_MODEL = "claude-sonnet-5-5"
DEFAULT_MAX_CALLS = 130
# Earlier cache entries were recorded under the CLI alias; they are the same model.
MODEL_ALIASES = {"claude-sonnet-5-5": ("sonnet",)}


def create_backend(name: str, *, model: str | None = None, fixtures: Path | None = None) -> Backend:
    if name == "replay":
        if fixtures is None:
            raise ValueError("the replay backend needs a fixtures path")
        return ReplayBackend(fixtures)
    if name == "anthropic":
        from probable_cause.llm.anthropic_backend import DEFAULT_MODEL, AnthropicBackend

        return AnthropicBackend(model or DEFAULT_MODEL)
    if name == "openrouter":
        from probable_cause.llm.openrouter_backend import DEFAULT_MODEL as OPENROUTER_MODEL
        from probable_cause.llm.openrouter_backend import OpenRouterBackend

        return OpenRouterBackend(model or OPENROUTER_MODEL)
    if name == "claude-cli":
        from probable_cause.llm.claude_cli import ClaudeCliBackend

        return ClaudeCliBackend(model)
    raise ValueError(f"unknown backend {name!r}; choose from {', '.join(BACKENDS)}")


def llm_from_env(*, cache_dir: Path | None = DEFAULT_CACHE_DIR) -> LLM:
    """Build an LLM from PC_LLM_BACKEND / PC_LLM_MODEL / PC_LLM_FIXTURES, with the disk cache
    and the live-call budget (PC_LLM_MAX_CALLS, default 130, ledger in the cache dir).

    Defaults to ``claude-cli`` with Sonnet, the only backend that needs no API key.
    """
    name = os.environ.get("PC_LLM_BACKEND", "claude-cli")
    model = os.environ.get("PC_LLM_MODEL") or (CLAUDE_CLI_MODEL if name == "claude-cli" else None)
    fixtures = os.environ.get("PC_LLM_FIXTURES")
    backend = create_backend(name, model=model, fixtures=Path(fixtures) if fixtures else None)
    if cache_dir is not None and name != "replay":
        budget = CallBudget(
            cache_dir / "calls.jsonl", int(os.environ.get("PC_LLM_MAX_CALLS", DEFAULT_MAX_CALLS))
        )
        backend = CachingBackend(
            backend,
            cache_dir,
            offline=os.environ.get("PC_LLM_OFFLINE") == "1",
            budget=budget,
            model_aliases=MODEL_ALIASES.get(backend.model, ()),
        )
    return LLM(backend)


__all__ = [
    "BACKENDS",
    "LLM",
    "Backend",
    "BackendError",
    "CachingBackend",
    "CallBudget",
    "Completion",
    "CompletionRequest",
    "Message",
    "RecordingBackend",
    "ReplayBackend",
    "Usage",
    "create_backend",
    "llm_from_env",
]
