"""infcore Python SDK — клиент к внутреннему gateway (offline-контур).

Тонкий клиент OpenAI-совместимого REST gateway. Без внешних сетевых зависимостей
сверх стандартной библиотеки (или внутреннего зеркала PyPI).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator

__all__ = ["Client", "GenerationParams"]


@dataclass
class GenerationParams:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stop: list[str] = field(default_factory=list)


class Client:
    """Клиент infcore gateway (OpenAI-совместимый)."""

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def chat(self, model: str, messages: list[dict], params: GenerationParams | None = None) -> str:
        raise NotImplementedError("infcore SDK: chat (scaffold)")

    def chat_stream(self, model: str, messages: list[dict],
                    params: GenerationParams | None = None) -> Iterator[str]:
        raise NotImplementedError("infcore SDK: chat_stream (scaffold)")
        yield  # pragma: no cover

    def embeddings(self, model: str, inputs: list[str]) -> list[list[float]]:
        raise NotImplementedError("infcore SDK: embeddings (scaffold)")
