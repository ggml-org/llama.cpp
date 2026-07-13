"""infcore Python SDK — клиент к внутреннему gateway (offline-контур).

Тонкий клиент OpenAI-совместимого REST gateway. Только стандартная библиотека
(urllib) — без внешних сетевых зависимостей; работает в offline-контуре как есть.

Пример:
    from infcore import Client, GenerationParams
    c = Client("http://127.0.0.1:8080", api_key="...")
    print(c.chat("qwen3-moe-a3b", [{"role": "user", "content": "привет"}]))
    for tok in c.chat_stream("qwen3-moe-a3b", [...]):
        print(tok, end="", flush=True)
    vecs = c.embeddings("bge-m3-embed", ["текст 1", "текст 2"])
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Iterator

__all__ = ["Client", "GenerationParams", "InfcoreError"]


class InfcoreError(RuntimeError):
    """Ошибка gateway (не-2xx) или транспортная ошибка. status=None для транспортных."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass
class GenerationParams:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stop: list[str] = field(default_factory=list)

    def to_body(self) -> dict:
        b: dict = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.stop:
            b["stop"] = self.stop
        return b


class Client:
    """Клиент infcore gateway (OpenAI-совместимый)."""

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    # --- внутреннее ----------------------------------------------------------
    def _request(self, path: str, body: dict, stream: bool):
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers, method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)  # noqa: S310 (внутренний контур)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            msg = detail
            try:
                msg = json.loads(detail).get("error", {}).get("message", detail)
            except Exception:  # pragma: no cover - тело не JSON
                pass
            raise InfcoreError(f"gateway {e.code}: {msg}", status=e.code) from None
        except urllib.error.URLError as e:  # pragma: no cover - сеть недоступна
            raise InfcoreError(f"gateway недоступен: {e.reason}") from None
        return resp

    def models(self) -> list[dict]:
        """GET /v1/models — список доступных роли моделей."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + "/v1/models", headers=headers, method="GET")
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)  # noqa: S310
        except urllib.error.HTTPError as e:
            raise InfcoreError(f"gateway {e.code}", status=e.code) from None
        return json.loads(resp.read().decode("utf-8")).get("data", [])

    def chat(self, model: str, messages: list[dict], params: GenerationParams | None = None) -> str:
        """Не-стриминговый chat. Возвращает текст первого choice."""
        body = {"model": model, "messages": messages, "stream": False}
        if params:
            body.update(params.to_body())
        resp = self._request("/v1/chat/completions", body, stream=False)
        j = json.loads(resp.read().decode("utf-8"))
        return j["choices"][0]["message"]["content"]

    def chat_stream(self, model: str, messages: list[dict],
                    params: GenerationParams | None = None) -> Iterator[str]:
        """Стриминговый chat (SSE). Отдаёт дельты content по мере поступления."""
        body = {"model": model, "messages": messages, "stream": True}
        if params:
            body.update(params.to_body())
        resp = self._request("/v1/chat/completions", body, stream=True)
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:  # pragma: no cover
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            if delta:
                yield delta

    def embeddings(self, model: str, inputs: list[str]) -> list[list[float]]:
        """POST /v1/embeddings — векторы для списка строк (порядок сохраняется)."""
        body = {"model": model, "input": inputs}
        resp = self._request("/v1/embeddings", body, stream=False)
        j = json.loads(resp.read().decode("utf-8"))
        return [item["embedding"] for item in j["data"]]
