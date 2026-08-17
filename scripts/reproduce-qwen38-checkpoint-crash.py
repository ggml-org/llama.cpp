#!/usr/bin/env python3
import argparse
import json
import pathlib
import time
import urllib.error
import urllib.request


def request_json(url: str, payload: dict, timeout: int):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, body, time.monotonic() - started, None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body, time.monotonic() - started, f"HTTPError: {exc}"
    except Exception as exc:
        return -1, "", time.monotonic() - started, f"{type(exc).__name__}: {exc}"


def completion_text(body: str) -> str:
    try:
        parsed = json.loads(body)
        return str(parsed.get("content", ""))
    except Exception:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18161/completion")
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=240)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--families", type=int, default=2, choices=range(2, 9))
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    labels = [chr(ord("A") + index) for index in range(args.families)]
    phrases = {
        label: f"{label} checkpoint family records number {112233 + index * 110011} and keeps this exact stable prefix. "
        for index, label in enumerate(labels)
    }
    bases = {
        label: f"{label}-FAMILY-START\n" + phrases[label] * (args.repeats + index * 17)
        for index, label in enumerate(labels)
    }

    continuations = {label: "" for label in labels}
    sequence = []
    for cycle in range(args.cycles):
        sequence.extend(labels)

    summary = []
    for index, family in enumerate(sequence, start=1):
        base = bases[family]
        prompt = base + continuations[family] + f"\nCycle {index}: reply with the family letter only."
        payload = {
            "prompt": prompt,
            "cache_prompt": True,
            "temperature": 0,
            "top_k": 1,
            "seed": 12345,
            "n_predict": 1,
            "stream": False,
        }
        (out / f"request-{index:02d}-{family}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        status, body, elapsed, error = request_json(args.url, payload, args.timeout)
        (out / f"response-{index:02d}-{family}.txt").write_text(body, encoding="utf-8")

        content = completion_text(body)
        if status == 200:
            continuations[family] += f"\nCycle {index} answer: {content}"

        record = {
            "index": index,
            "family": family,
            "status": status,
            "elapsed_seconds": elapsed,
            "error": error,
            "content": content,
        }
        summary.append(record)
        print(json.dumps(record), flush=True)

        if status != 200:
            break

    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0 if len(summary) == len(sequence) and all(item["status"] == 200 for item in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
