from __future__ import annotations

import http.client
import json
import time
from typing import Any, Iterable, Iterator


class ServerError(RuntimeError):
    pass


def parse_sse_events(chunks: Iterable[bytes]) -> Iterator[dict[str, Any]]:
    buffer = ""
    data_lines: list[str] = []

    def finish_event() -> dict[str, Any] | None:
        nonlocal data_lines
        if not data_lines:
            return None
        payload = "\n".join(data_lines)
        data_lines = []
        if payload == "[DONE]":
            return None
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ServerError("completion stream contained invalid JSON") from error
        if not isinstance(data, dict):
            raise ServerError("completion stream contained a non-object JSON event")
        return data

    for chunk in chunks:
        buffer += chunk.decode("utf-8")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                data = finish_event()
                if data is not None:
                    yield data
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))

    if buffer:
        if buffer.startswith("data:"):
            data_lines.append(buffer[5:].lstrip(" "))
    data = finish_event()
    if data is not None:
        yield data


class ServerClient:
    def __init__(self, host: str, port: int, timeout_seconds: float) -> None:
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds

    def request_json(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_seconds)
        payload = None if body is None else json.dumps(body)
        headers = {} if payload is None else {"Content-Type": "application/json"}
        try:
            connection.request(method, path, body=payload, headers=headers)
            response = connection.getresponse()
            raw = response.read()
        finally:
            connection.close()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ServerError(f"{method} {path} returned invalid JSON") from error
        if response.status < 200 or response.status >= 300:
            raise ServerError(f"{method} {path} failed with HTTP {response.status}: {data}")
        if not isinstance(data, dict):
            raise ServerError(f"{method} {path} returned a non-object JSON response")
        return data

    def health(self) -> bool:
        try:
            return self.request_json("GET", "/health").get("status") == "ok"
        except (OSError, ServerError):
            return False

    def props(self) -> dict[str, Any]:
        return self.request_json("GET", "/props")

    def tokenize(self, content: str, add_special: bool = True) -> list[int]:
        data = self.request_json("POST", "/tokenize", {
            "content": content,
            "add_special": add_special,
            "parse_special": True,
        })
        tokens = data.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
            raise ServerError("/tokenize returned invalid tokens")
        return tokens

    def apply_template(self, messages: list[dict[str, str]]) -> str:
        data = self.request_json("POST", "/apply-template", {
            "messages": messages,
            "add_generation_prompt": True,
        })
        prompt = data.get("prompt")
        if not isinstance(prompt, str):
            raise ServerError("/apply-template returned no prompt")
        return prompt

    def metrics(self) -> dict[str, float]:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_seconds)
        try:
            connection.request("GET", "/metrics")
            response = connection.getresponse()
            raw = response.read().decode("utf-8")
        finally:
            connection.close()
        if response.status != 200:
            raise ServerError(f"GET /metrics failed with HTTP {response.status}: {raw}")
        metrics: dict[str, float] = {}
        for line in raw.splitlines():
            if not line.startswith("llamacpp:") or "{" in line:
                continue
            name, value = line.split(maxsplit=1)
            metrics[name] = float(value)
        return metrics

    def completion_stream(self, body: dict[str, Any]) -> Iterator[tuple[float, dict[str, Any]]]:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_seconds)
        payload = json.dumps(body)
        try:
            connection.request("POST", "/completion", body=payload, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status < 200 or response.status >= 300:
                raw = response.read().decode("utf-8")
                raise ServerError(f"POST /completion failed with HTTP {response.status}: {raw}")
            for data in parse_sse_events(response):
                yield time.monotonic(), data
        finally:
            connection.close()
