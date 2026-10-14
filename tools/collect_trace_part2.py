#!/usr/bin/env python3
"""Collect additional trace steps from prompts 20..31."""

import json
import os
import sys
import time
import urllib.request


def main():
    prompts_file = "prompts/dataset_80.jsonl"
    meta_file = "traces/trace_part2.meta.json"
    trace_file = "traces/trace_part2.ids.u16"

    with open(prompts_file, "r") as f:
        all_prompts = [json.loads(line) for line in f if line.strip()]

    target_prompts = [p for p in all_prompts if 20 <= p["id"] <= 31]
    print(f"Targeting {len(target_prompts)} prompts (IDs 20..31)...")

    meta = {"prompts": [], "total_decode_tokens": 0}

    for idx, p in enumerate(target_prompts):
        pid = p["id"]
        cat = p["category"]
        dom = p.get("domain", "")
        text = p["prompt"]

        print(f"\n[{idx+1}/{len(target_prompts)}] Prompt #{pid} ({cat}/{dom})...", flush=True)

        bytes_before = os.path.getsize(trace_file) if os.path.exists(trace_file) else 0
        step_before = bytes_before // (48 * 10 * 2)

        req_data = {
            "prompt": text,
            "n_predict": 256,
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
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                res = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"Error querying prompt {pid}: {e}", file=sys.stderr)
            break
        dt = time.time() - t0

        bytes_after = os.path.getsize(trace_file) if os.path.exists(trace_file) else 0
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

    print(f"\nFinished collection! Logged {meta['total_decode_tokens']} steps to {trace_file}")


if __name__ == "__main__":
    main()
