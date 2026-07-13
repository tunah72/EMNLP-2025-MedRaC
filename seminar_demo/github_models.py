from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Sequence, Type

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


GITHUB_MODELS_BASE_URL = "https://models.github.ai/inference"
GITHUB_API_VERSION = "2026-03-10"
DEFAULT_CHAT_MODEL = "openai/gpt-4.1"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
PROJECT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
_FREE_CHAT_REQUESTS_PER_MINUTE = 10
_CHAT_REQUEST_TIMES: deque[float] = deque()
_CHAT_THROTTLE_LOCK = threading.Lock()


def _throttle_free_chat_requests() -> None:
    while True:
        with _CHAT_THROTTLE_LOCK:
            now = time.monotonic()
            while _CHAT_REQUEST_TIMES and now - _CHAT_REQUEST_TIMES[0] >= 60:
                _CHAT_REQUEST_TIMES.popleft()
            if len(_CHAT_REQUEST_TIMES) < _FREE_CHAT_REQUESTS_PER_MINUTE:
                _CHAT_REQUEST_TIMES.append(now)
                return
            wait_seconds = max(0.0, 60 - (now - _CHAT_REQUEST_TIMES[0]))
        time.sleep(wait_seconds)


def load_project_env() -> bool:
    """Load the repository .env without python-dotenv stack inspection."""
    return load_dotenv(dotenv_path=PROJECT_ENV_PATH, override=False)


def require_github_token() -> str:
    load_project_env()
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise EnvironmentError(
            "GITHUB_TOKEN is required; create a fine-grained token with models:read"
        )
    return token


def validate_openai_catalog_model(model: str, *, embedding: bool = False) -> None:
    if not model.startswith("openai/"):
        kind = "embedding" if embedding else "generation/evaluation"
        raise ValueError(f"GitHub Models {kind} model must use the openai/ publisher")
    if embedding and model != DEFAULT_EMBEDDING_MODEL:
        raise ValueError(
            f"The compatibility index requires {DEFAULT_EMBEDDING_MODEL}, found {model}"
        )


def github_models_client(token: str | None = None) -> OpenAI:
    return OpenAI(
        api_key=token or require_github_token(),
        base_url=GITHUB_MODELS_BASE_URL,
        default_headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        },
    )


class GitHubModelsLLM:
    """Repository-compatible LLM wrapper for OpenAI models hosted by GitHub."""

    def __init__(
        self,
        model_name: str = DEFAULT_CHAT_MODEL,
        *,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 2048,
        client: Any | None = None,
    ) -> None:
        validate_openai_catalog_model(model_name)
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("GitHub Models temperature must be between 0 and 1")
        self.model_name = model_name
        self.model_name_full = f"GitHubModels/{model_name}"
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.client = client or github_models_client()
        self.tokens_used = 0
        self.requests_made = 0

    def get_model_name(self) -> str:
        return self.model_name_full

    def generate(
        self,
        prompts: Sequence[tuple[str, str]],
        schema: Type[BaseModel] | None = None,
        show_progress: bool = True,
        **_: Any,
    ) -> list[tuple[Any, int, int]]:
        del show_progress
        results: list[tuple[Any, int, int]] = []
        for system_msg, user_msg in prompts:
            request: dict[str, Any] = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": self.temperature,
                "seed": self.seed,
                "max_tokens": self.max_tokens,
            }
            if schema is not None:
                request["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": schema.model_json_schema(),
                    },
                }
            _throttle_free_chat_requests()
            response = self.client.chat.completions.create(**request)
            message = response.choices[0].message.content or ""
            if schema is not None:
                payload: Any = json.loads(message)
            else:
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    payload = message
            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            self.requests_made += 1
            self.tokens_used += input_tokens + output_tokens
            results.append((payload, input_tokens, output_tokens))
        return results
