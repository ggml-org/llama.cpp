#!/usr/bin/env python3
import argparse
import json
import pathlib
import time
import urllib.error
import urllib.request


def post(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, body, time.monotonic() - start, None
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace"), time.monotonic() - start, repr(exc)
    except Exception as exc:
        return -1, "", time.monotonic() - start, repr(exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18163/completion")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-repeats", type=int, default=1900)
    ap.add_argument("--tail-repeats", type=int, default=110)
    ap.add_argument("--n-predict", type=int, default=1)
    ap.add_argument("--n-probs", type=int, default=0)
    ap.add_argument("--return-tokens", action="store_true")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stable = "Stable checkpoint line 112233 keeps recurrent and attention state aligned for deterministic replay. "
    old_tail = "Original branch suffix 445566 extends the rolling recurrent state before rewind. "
    new_tail = "Replacement branch suffix 778899 must replay from the saved stable checkpoint. "
    base = "LONG-STABLE-PREFIX\n" + stable * args.base_repeats
    prompts = [
        base + "\nEnd stable prefix; answer A.",
        base + old_tail * args.tail_repeats + "\nEnd original branch; answer B.",
        base + new_tail * (args.tail_repeats // 2) + "\nEnd replacement branch; answer C.",
        base + old_tail * (args.tail_repeats + 20) + "\nEnd original branch again; answer D.",
        base + new_tail * (args.tail_repeats // 3) + "\nEnd second replacement; answer E.",
    ]

    results = []
    for idx, prompt in enumerate(prompts, 1):
        payload = {
            "prompt": prompt,
            "cache_prompt": True,
            "temperature": 0,
            "top_k": 1,
            "seed": 424242,
            "n_predict": args.n_predict,
            "n_probs": args.n_probs,
            "return_tokens": args.return_tokens,
            "stream": False,
        }
        (out / f"request-{idx:02d}.json").write_text(json.dumps(payload, indent=2) + "\n")
        status, body, elapsed, error = post(args.url, payload, args.timeout)
        (out / f"response-{idx:02d}.txt").write_text(body)
        usage = {}
        content = ""
        try:
            parsed = json.loads(body)
            usage = parsed.get("timings", {})
            content = parsed.get("content", "")
        except Exception:
            pass
        row = {"index": idx, "status": status, "elapsed_seconds": elapsed, "error": error, "content": content, "timings": usage}
        results.append(row)
        print(json.dumps(row), flush=True)
        if status != 200:
            break
    (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0 if len(results) == len(prompts) and all(r["status"] == 200 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
