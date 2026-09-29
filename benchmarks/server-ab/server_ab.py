#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


class ConfigError(ValueError):
    pass


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(value: Any, env: dict[str, str] | None = None) -> Any:
    values = os.environ if env is None else env
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in values:
                raise ConfigError(f"environment variable {name} is not set")
            return values[name]

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [expand_env(item, values) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item, values) for key, item in value.items()}
    return value


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError("PyYAML is required; install benchmarks/server-ab/requirements.txt") from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError("configuration root must be a mapping")
    data = expand_env(data)
    required = ("variants", "builds", "models", "workloads", "profiles", "run")
    missing = [name for name in required if not isinstance(data.get(name), dict)]
    if missing:
        raise ConfigError(f"missing configuration mappings: {', '.join(missing)}")
    return data


def balanced_order(
        variants: list[str], blocks: int, strategy: str, seed: int) -> list[tuple[int, int, str]]:
    if not variants or blocks < 1:
        raise ConfigError("at least one variant and one block are required")
    if strategy == "abba":
        if len(variants) != 2:
            raise ConfigError("abba ordering requires exactly two variants")
        first = [variants[0], variants[1], variants[1], variants[0]]
        second = [variants[1], variants[0], variants[0], variants[1]]
        return [
            (block, position, variant)
            for block in range(blocks)
            for position, variant in enumerate(first if block % 2 == 0 else second)
        ]
    if strategy == "random":
        rng = random.Random(seed)
        result: list[tuple[int, int, str]] = []
        for block in range(blocks):
            order = variants.copy()
            rng.shuffle(order)
            result.extend((block, position, variant) for position, variant in enumerate(order))
        return result
    raise ConfigError(f"unknown run order: {strategy}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def load_completed_keys(path: Path) -> set[str]:
    return {
        str(row["run_key"])
        for row in read_jsonl(path)
        if row.get("ok") is True and isinstance(row.get("run_key"), str)
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def request_json(
        url: str,
        payload: dict[str, Any] | None = None,
        timeout: float = 30.0,
        method: str | None = None,
        headers: dict[str, str] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(headers or {})
    request = urllib.request.Request(
        url,
        data=data,
        method=method or ("POST" if data is not None else "GET"),
        headers=request_headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"request failed for {url}: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"expected JSON object from {url}")
    return result


def wait_for_health(base_url: str, timeout: float, process: Any | None = None) -> float:
    started = time.monotonic()
    deadline = started + timeout
    last_error = "server did not respond"
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"server exited with status {process.returncode}")
        try:
            result = request_json(f"{base_url}/health", timeout=min(2.0, timeout))
            if result.get("status") == "ok":
                return time.monotonic() - started
        except RuntimeError as exc:
            last_error = str(exc)
        time.sleep(0.1)
    raise TimeoutError(f"server health timeout: {last_error}")


def erase_slot(base_url: str, slot: int, timeout: float) -> None:
    request_json(f"{base_url}/slots/{slot}?action=erase", {}, timeout=timeout)


class SSEAccumulator:
    def __init__(self, started_at: float):
        self.started_at = started_at
        self.token_times: list[float] = []
        self.tokens: list[int] = []
        self.content: list[str] = []
        self.final: dict[str, Any] = {}
        self.error: str | None = None

    def feed(self, event: dict[str, Any], received_at: float) -> None:
        if "error" in event:
            error = event["error"]
            self.error = str(error.get("message", error)) if isinstance(error, dict) else str(error)
        tokens = event.get("tokens")
        if isinstance(tokens, list):
            for token in tokens:
                if isinstance(token, int):
                    self.tokens.append(token)
                    self.token_times.append(received_at)
        content = event.get("content")
        if isinstance(content, str):
            self.content.append(content)
        if event.get("stop") is True:
            self.final = event

    def finish(self, finished_at: float) -> dict[str, Any]:
        if self.error is not None:
            raise RuntimeError(f"stream returned an error after {len(self.tokens)} tokens: {self.error}")
        if not self.final:
            raise RuntimeError(f"stream ended without a final response after {len(self.tokens)} tokens")
        ttft_ms = None
        if self.token_times:
            ttft_ms = (self.token_times[0] - self.started_at) * 1000.0
        intervals = [
            (right - left) * 1000.0
            for left, right in zip(self.token_times, self.token_times[1:])
        ]
        return {
            "content": "".join(self.content),
            "tokens": self.tokens,
            "ttft_ms": ttft_ms,
            "inter_token_ms": intervals,
            "latency_ms": (finished_at - self.started_at) * 1000.0,
            "timings": self.final.get("timings", {}),
            "tokens_cached": self.final.get("tokens_cached"),
            "tokens_evaluated": self.final.get("tokens_evaluated"),
            "stop_type": self.final.get("stop_type"),
        }


def stream_completion(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    started = time.monotonic()
    accumulator = SSEAccumulator(started)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                value = line[5:].strip()
                if not value or value == "[DONE]":
                    continue
                event = json.loads(value)
                if isinstance(event, dict):
                    accumulator.feed(event, time.monotonic())
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"streaming request failed for {url}: {exc}") from exc
    return accumulator.finish(time.monotonic())


def comma_list(value: str | None) -> set[str] | None:
    if value is None:
        return None
    result = {item.strip() for item in value.split(",") if item.strip()}
    return result or None


def selected(name: str, selection: set[str] | None) -> bool:
    return selection is None or name in selection


def iter_sse_payloads(lines: Iterable[str]) -> Iterable[dict[str, Any]]:
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        value = line[5:].strip()
        if not value or value == "[DONE]":
            continue
        event = json.loads(value)
        if isinstance(event, dict):
            yield event
