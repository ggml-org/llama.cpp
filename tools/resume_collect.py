#!/usr/bin/env python3
"""Resume trace collection for prompts 17, 18, 19."""

import json
import os
import sys
import time
import urllib.request


def main():
    prompts_file = "prompts/dataset_20.jsonl"
    meta_file = "traces/trace_10k.meta.json"
    trace_file = "traces/trace_10k.ids.u16"

    with open(prompts_file, "r") as f:
        prompts = [json.loads(line) for line in f if line.strip()]

    with open(meta_file, "r") as f:
        meta = json.load(f)

    # Completed prompt IDs
    done_ids = {p["id"] for p in meta.get("prompts", [])}
    last_end = meta["prompts"][-1]["end_step"]
    target_bytes = last_end * 48 * 10 * 2

    # Truncate to clean boundary of last completed prompt
    print(f"Truncating {trace_file} to step {last_end} ({target_bytes} bytes)...")
    with open(trace_file, "r+b") as f:
        f.truncate(target_bytes)

    remaining = [p for p in prompts if p["id"] not in done_ids]
    print(f"Remaining prompts to run: {[p['id'] for p in remaining]}")

    for idx, p in enumerate(remaining):
        pid = p["id"]
        cat = p["category"]
        dom = p.get("domain", "")
        text = p["prompt"]

        print(f"\n[Prompt #{pid}] ({cat}/{dom})...", flush=True)

        bytes_before = os.path.getsize(trace_file)
        step_before = bytes_before // (48 * 10 * 2)

        req_data = {
            "prompt": text,
            "n_predict": 512,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "stream": False,
        }
        req = urllib.request.Request(
            "http://127.0.0.1:8080/completion",
            data=json.dumps(req_data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as resp:
            res = json.loads(resp.read().decode("utf-8"))
        dt = time.time() - t0

        bytes_after = os.path.getsize(trace_file)
        step_after = bytes_after // (48 * 10 * 2)
        steps_added = step_after - step_before

        tok_s = res.get("timings", {}).get("predicted_per_second", 0)
        n_tok = res.get("timings", {}).get("predicted_n", 0)

        print(
            f"  -> Done in {dt:.1f}s ({tok_s:.2f} tok/s): generated {n_tok} tokens, logged {steps_added} steps (total steps: {step_after})",
            flush=True,
        )

        meta["prompts"].append(
            {
                "id": pid,
                "category": cat,
                "domain": dom,
                "start_step": int(step_before),
                "end_step": int(step_after),
                "n_steps": int(steps_added),
                "predicted_n": int(n_tok),
                "tok_s": float(tok_s),
                "time_s": float(dt),
            }
        )
        meta["total_decode_tokens"] += int(steps_added)

        with open(meta_file, "w") as mf:
            json.dump(meta, mf, indent=2)

    print(f"\nAll {len(meta['prompts'])} prompts completed!")
    print(f"Total steps in trace: {meta['prompts'][-1]['end_step']}")


if __name__ == "__main__":
    main()
