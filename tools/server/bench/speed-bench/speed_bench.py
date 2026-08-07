#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse

import requests
from datasets import get_dataset_config_names, load_dataset
from tqdm import tqdm


DATASET_REPO = "nvidia/SPEED-Bench"

@dataclass
class Sample:
    id: str
    category: str
    turns: list[str]


@dataclass
class RequestResult:
    id: str
    category: str
    ok: bool
    turns: int
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str | None
    draft_n: int
    draft_n_accepted: int
    prompt_ms: float | None
    predicted_ms: float | None
    prompt_per_second: float | None
    predicted_per_second: float | None
    error: str | None


def normalize_base_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url:
        raise ValueError("--url cannot be empty")
    if "://" not in url:
        url = "http://" + url
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"invalid --url: {url}")
    if not parsed.path.rstrip("/").endswith("/v1"):
        url = url + "/v1"
    return url.rstrip("/")


def parse_extra_inputs(value: str) -> dict[str, Any]:
    extra = json.loads(value)
    if not isinstance(extra, dict):
        raise ValueError("--extra-inputs must be a JSON object")
    return extra


def extract_turns(row: dict[str, Any]) -> list[str]:
    turns = row.get("turns")
    if isinstance(turns, list) and turns:
        clean_turns = [str(turn).strip() for turn in turns if turn and str(turn).strip()]
        if clean_turns:
            return clean_turns
    raise ValueError("missing or empty turns")


def load_samples(args: argparse.Namespace) -> list[Sample]:
    bench_names = get_dataset_config_names(DATASET_REPO)
    if args.bench not in bench_names:
        raise ValueError(
            f"unknown --bench {args.bench!r}; available benches: {', '.join(bench_names)}"
        )

    dataset = load_dataset(DATASET_REPO, name=args.bench, split="test")
    categories = list(dict.fromkeys(str(category) for category in dataset["category"]))
    requested_categories = None
    if args.category != "all":
        requested_list = [category.strip() for category in args.category.split(",") if category.strip()]
        if not requested_list:
            raise ValueError(
                f"--category must be 'all' or a comma-separated list; available categories: {', '.join(categories)}"
            )
        requested_categories = set(requested_list)
        unknown_categories = [category for category in requested_list if category not in categories]
        if unknown_categories:
            unknown = ", ".join(unknown_categories)
            raise ValueError(
                f"unknown --category {unknown!r} for bench {args.bench!r}; "
                f"available categories: all, {', '.join(categories)}"
            )

    samples: list[Sample] = []
    samples_per_category: dict[str, int] = {}
    skipped = 0
    for index, row_raw in enumerate(dataset):
        row = dict(row_raw)
        category_raw = row.get("category")
        if not isinstance(category_raw, str) or not category_raw.strip():
            skipped += 1
            continue
        category = category_raw.strip()
        if requested_categories is not None and category not in requested_categories:
            continue
        if args.limit is not None and samples_per_category.get(category, 0) >= args.limit:
            continue

        try:
            turns = extract_turns(row)
        except ValueError:
            skipped += 1
            continue
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            skipped += 1
            continue
        sample_id = question_id.strip()
        samples.append(Sample(id=sample_id, category=category, turns=turns))
        samples_per_category[category] = samples_per_category.get(category, 0) + 1

    if not samples:
        raise RuntimeError(f"no samples selected from bench={args.bench} category={args.category}")

    if skipped:
        print(f"speed_bench: skipped {skipped} rows without usable turns")
    return samples


def parse_completion_response(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str | None, str]:
    usage = data.get("usage") or {}
    timings = data.get("timings") or {}
    finish_reason = None
    content = ""
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            content = message["content"]
        elif isinstance(choice.get("text"), str):
            content = choice["text"]
    return usage, timings, finish_reason, content


def run_request(
    endpoint: str,
    model: str | None,
    messages: list[dict[str, str]],
    osl: int,
    extra_inputs: dict[str, Any],
    timeout: float,
) -> tuple[dict[str, Any], float]:
    payload: dict[str, Any] = {
        "messages": messages,
        "max_tokens": osl,
        "stream": False,
    }
    if model:
        payload["model"] = model
    payload.update(extra_inputs)
    payload["max_tokens"] = osl

    start = time.perf_counter()
    response = requests.post(endpoint, json=payload, timeout=timeout)
    latency_s = time.perf_counter() - start
    if response.status_code != 200:
        body = response.text[:500].replace("\n", "\\n")
        raise RuntimeError(f"HTTP {response.status_code}: {body}")
    return response.json(), latency_s


def run_one(
    sample: Sample,
    endpoint: str,
    model: str | None,
    osl: int,
    extra_inputs: dict[str, Any],
    timeout: float,
) -> RequestResult:
    selected_turns = sample.turns
    messages: list[dict[str, str]] = []
    total_latency_s = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    draft_n = 0
    draft_n_accepted = 0
    prompt_ms = 0.0
    predicted_ms = 0.0
    prompt_per_second = None
    predicted_per_second = None
    finish_reason: str | None = None
    try:
        for turn in selected_turns:
            messages.append({"role": "user", "content": turn})
            data, latency_s = run_request(endpoint, model, messages, osl, extra_inputs, timeout)
            total_latency_s += latency_s
            usage, timings, finish_reason, assistant_text = parse_completion_response(data)

            turn_prompt_tokens = int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0)
            turn_completion_tokens_count = int(usage.get("completion_tokens") or timings.get("predicted_n") or 0)
            turn_total_tokens_count = int(usage.get("total_tokens") or (turn_prompt_tokens + turn_completion_tokens_count))
            prompt_tokens += turn_prompt_tokens
            completion_tokens += turn_completion_tokens_count
            total_tokens += turn_total_tokens_count
            draft_n += int(timings.get("draft_n") or 0)
            draft_n_accepted += int(timings.get("draft_n_accepted") or 0)
            prompt_ms += float(timings.get("prompt_ms") or 0)
            predicted_ms += float(timings.get("predicted_ms") or 0)
            if len(selected_turns) == 1 and isinstance(timings.get("prompt_per_second"), (int, float)):
                prompt_per_second = float(timings["prompt_per_second"])
            if len(selected_turns) == 1 and isinstance(timings.get("predicted_per_second"), (int, float)):
                predicted_per_second = float(timings["predicted_per_second"])

            messages.append({"role": "assistant", "content": assistant_text})

        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        if len(selected_turns) > 1:
            prompt_per_second = (prompt_tokens / (prompt_ms / 1000)) if prompt_ms > 0 else None
            predicted_per_second = (completion_tokens / (predicted_ms / 1000)) if predicted_ms > 0 else None

        return RequestResult(
            id=sample.id,
            category=sample.category,
            ok=True,
            turns=len(selected_turns),
            latency_s=total_latency_s,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
            draft_n=draft_n,
            draft_n_accepted=draft_n_accepted,
            prompt_ms=prompt_ms if prompt_ms > 0 else None,
            predicted_ms=predicted_ms if predicted_ms > 0 else None,
            prompt_per_second=prompt_per_second,
            predicted_per_second=predicted_per_second,
            error=None,
        )
    except Exception as exc:
        return RequestResult(
            id=sample.id,
            category=sample.category,
            ok=False,
            turns=len(selected_turns),
            latency_s=total_latency_s,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            finish_reason=None,
            draft_n=0,
            draft_n_accepted=0,
            prompt_ms=None,
            predicted_ms=None,
            prompt_per_second=None,
            predicted_per_second=None,
            error=str(exc),
        )


def summarize_group(category: str, results: list[RequestResult]) -> dict[str, Any]:
    ok_results = [result for result in results if result.ok]
    latencies = [result.latency_s for result in ok_results]
    server_prompt_speeds = [
        result.prompt_per_second
        for result in ok_results
        if result.prompt_per_second is not None
    ]
    server_completion_speeds = [
        result.predicted_per_second
        for result in ok_results
        if result.predicted_per_second is not None
    ]
    turns = sum(result.turns for result in ok_results)
    draft_n = sum(result.draft_n for result in ok_results)
    accepted = sum(result.draft_n_accepted for result in ok_results)

    return {
        "category": category,
        "requests": len(ok_results),
        "turns": turns,
        "failed": len(results) - len(ok_results),
        "avg_prompt_t_s": statistics.mean(server_prompt_speeds) if server_prompt_speeds else None,
        "avg_pred_t_s": statistics.mean(server_completion_speeds) if server_completion_speeds else None,
        "avg_latency": statistics.mean(latencies) if latencies else None,
        "draft_n": draft_n,
        "accepted": accepted,
        "accept_rate": (accepted / draft_n) if draft_n > 0 else None,
    }


def fmt_value(value: Any, kind: str = "") -> str:
    if value is None:
        return "n/a"
    if kind == "int":
        return str(int(value))
    if kind == "rate":
        return f"{float(value):.4f}"
    if kind == "seconds":
        return f"{float(value):.3f}s"
    if kind == "speed":
        return f"{float(value):.2f}"
    if kind == "speedup":
        return f"{float(value):.2f}x"
    return str(value)


def print_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        ("category", "category", ""),
        ("samples", "requests", "int"),
        ("avg_prompt_t/s", "avg_prompt_t_s", "speed"),
        ("avg_pred_t/s", "avg_pred_t_s", "speed"),
        ("avg_latency", "avg_latency", "seconds"),
        ("accept_rate", "accept_rate", "rate"),
    ]
    print_rows(rows, columns)


def print_rows(rows: list[dict[str, Any]], columns: list[tuple[str, str, str]]) -> None:
    rendered_rows = []
    for row in rows:
        rendered_rows.append([fmt_value(row.get(key), kind) for _, key, kind in columns])

    widths = [len(header) for header, _, _ in columns]
    for rendered in rendered_rows:
        for i, cell in enumerate(rendered):
            widths[i] = max(widths[i], len(cell))

    header = "  ".join(header.ljust(widths[i]) for i, (header, _, _) in enumerate(columns))
    print(header)
    print("  ".join("-" * width for width in widths))
    for rendered in rendered_rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(rendered)))


def save_output(path: str, args: argparse.Namespace, samples: list[Sample], results: list[RequestResult], summary: list[dict[str, Any]]) -> None:
    payload = {
        "config": {
            "url": args.url,
            "model": args.model,
            "bench": args.bench,
            "category": args.category,
            "osl": args.osl,
            "concurrency": args.concurrency,
            "extra_inputs": args.extra_inputs,
        },
        "selected_samples": len(samples),
        "completed_samples": sum(1 for result in results if result.ok),
        "failed_samples": sum(1 for result in results if not result.ok),
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SPEED-Bench against an OpenAI-compatible llama-server.")
    parser.add_argument("--url", default="localhost:8080", help="Server URL, for example localhost:8080 or http://localhost:8080/v1")
    parser.add_argument("--model", default=None, help="Optional model name to send in OpenAI requests")
    parser.add_argument("--bench", default="qualitative", help="SPEED-Bench config to run, for example qualitative or throughput_1k")
    parser.add_argument("--category", default="all", help="Category to run within the selected bench; use all for no category filter")
    parser.add_argument("--osl", type=int, default=4096, help="Output sequence length, mapped to max_tokens")
    parser.add_argument("--extra-inputs", default='{"temperature":0}', help="Extra request fields as a JSON object")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent client requests; usually match llama-server --np")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit per category for smoke tests")
    parser.add_argument("--timeout", type=float, default=600, help="Per-request timeout in seconds")
    parser.add_argument("--output", default=None, help="Optional path to save raw results JSON")
    args = parser.parse_args(argv)
    try:
        base_url = normalize_base_url(args.url)
        endpoint = base_url + "/chat/completions"
        extra_inputs = parse_extra_inputs(args.extra_inputs)
        args.extra_inputs = extra_inputs
        samples = load_samples(args)
    except Exception as exc:
        print(f"speed_bench: setup failed: {exc}", file=sys.stderr)
        return 2

    print(f"speed_bench: loaded {len(samples)} samples from bench={args.bench} category={args.category}")

    results: list[RequestResult] = []
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(run_one, sample, endpoint, args.model, args.osl, extra_inputs, args.timeout)
            for sample in samples
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="speed_bench", unit="sample"):
            result = future.result()
            results.append(result)

    elapsed = time.perf_counter() - started
    categories = list(dict.fromkeys(sample.category for sample in samples))
    summary = [
        summarize_group(category, [result for result in results if result.category == category])
        for category in categories
    ]
    summary.append(summarize_group("overall", results))
    print()
    print(f"Summary (elapsed={elapsed:.2f}s)")
    print_table(summary)

    if args.output:
        save_output(args.output, args, samples, results, summary)
        print(f"\nspeed_bench: wrote {args.output}")

    failed = sum(1 for result in results if not result.ok)
    if failed:
        print(f"\nspeed_bench: {failed} samples failed", file=sys.stderr)
        first_error = next((result.error for result in results if result.error), None)
        if first_error:
            print(f"first error: {first_error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
