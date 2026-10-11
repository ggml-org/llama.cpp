#!/usr/bin/env python3
"""Compare frozen servers using the V2 corpus, command profile and SSE collector."""

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/server-ab"))
from server_ab import request_json, stream_completion, wait_for_health

SPEC = importlib.util.spec_from_file_location("sweep", ROOT / "benchmarks/run-fixed-span-sweep.py")
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)


def command(args, context, mtp, arena, stock=False):
    result = SWEEP.server_command(args, context, mtp, arena)
    if stock:
        index = result.index("--kv-stream-arena-mib")
        del result[index:index+2]
    if mtp:
        result += ["--spec-draft-type-k", "q8_0", "--spec-draft-type-v", "q4_0"]
    return result


def compare_matching_rows(actual, control):
    for field in ("context", "mtp", "prompt_tokens", "decode_tokens"):
        if field not in actual or field not in control or actual[field] != control[field]:
            raise ValueError(f"nonmatching control profile: {field}")
    if actual.get("reuse_cache_tokens") != control.get("reuse_cache_tokens"):
        raise ValueError("nonmatching control profile: reuse_cache_tokens")
    result = {}
    for field, count in (("token_ids", actual["decode_tokens"]), ("reuse_token_ids",32), ("fresh_token_ids",32)):
        if field != "token_ids" and field not in actual and field not in control:
            continue
        if len(actual.get(field,[])) != count or len(control.get(field,[])) != count:
            raise ValueError(f"incomplete control output: {field}")
        result[field] = actual[field] == control[field]
    return result


def run_point(args, context, mtp, arena, token_cache):
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"context-{context}-mtp-{mtp}-arena-{arena}.log"
    url = f"http://127.0.0.1:{args.port}"
    row = {"variant": args.variant, "context": context, "mtp": mtp, "arena_mib": arena,
           "server": str(args.server), "log": str(path), "status": "failed",
           "started_unix": time.time(), "uvm": False,
           "command": command(args, context, mtp, arena, args.stock)}
    process = None
    measured_text = None
    started = time.monotonic()
    with path.open("w") as log:
        try:
            process = subprocess.Popen(row["command"], stdout=log,
                                       stderr=subprocess.STDOUT, env=SWEEP.server_environment(False))
            wait_for_health(url, 180, process)
            if "tokens" not in token_cache:
                token_cache["tokens"] = request_json(url + "/tokenize",
                    {"content": args.prefill_text.read_text(), "add_special": False}, timeout=180)["tokens"]
            count = context-args.decode_tokens-4
            if count < 256 or len(token_cache["tokens"]) < count:
                raise ValueError("invalid context or insufficient corpus")
            common = {"temperature": 0, "seed": 123, "ignore_eos": True,
                      "cache_prompt": False, "stream": True, "return_tokens": True}
            warm = stream_completion(url + "/completion", dict(common, prompt=token_cache["tokens"][:256],
                                      n_predict=4), timeout=120)
            if len(warm["tokens"]) != 4:
                raise RuntimeError("incomplete warmup")
            result = stream_completion(url + "/completion", dict(common, prompt=token_cache["tokens"][:count],
                                        n_predict=args.decode_tokens), timeout=1800)
            timing = result["timings"]
            if timing.get("prompt_n") != count or timing.get("predicted_n") != args.decode_tokens or len(result["tokens"]) != args.decode_tokens:
                raise RuntimeError("incomplete measured response")
            row.update(status="ok", prompt_tokens=count, decode_tokens=args.decode_tokens,
                       prefill_tps=timing["prompt_per_second"], decode_tps=timing["predicted_per_second"],
                       prompt_ms=timing["prompt_ms"], decode_ms=timing["predicted_ms"],
                       token_ids=result["tokens"], content=result["content"], timings=timing)
            measured_text = path.read_text(errors="replace")
            if not args.stock:
                pool, resident, ring, active, copies, calls = SWEEP.decode_layout(path)
                row.update(decode_kv_pool_mib=pool, resident_pages=resident, ring_slots=ring,
                           active_pages=active, h2d_mib_per_eval=copies, h2d_calls_per_eval=calls)
            if getattr(args, "verify_reuse", False):
                prefix = token_cache["tokens"][:count]+result["tokens"][:8]
                reused = stream_completion(url + "/completion", dict(common, prompt=prefix,
                                          n_predict=32, cache_prompt=True), timeout=1800)
                fresh = stream_completion(url + "/completion", dict(common, prompt=prefix,
                                         n_predict=32), timeout=1800)
                row.update(reuse_token_ids=reused["tokens"], fresh_token_ids=fresh["tokens"],
                           reuse_cache_tokens=reused["timings"].get("cache_n", 0),
                           reuse_timings=reused["timings"], fresh_timings=fresh["timings"],
                           reuse_matches_fresh=reused["tokens"] == fresh["tokens"])
                # Different prefill chunking can differ even in stock; compare each mode against its matching control.
                if len(reused["tokens"]) != 32 or len(fresh["tokens"]) != 32:
                    raise RuntimeError("incomplete serial prefix replay")
        except Exception as error:
            row["status"] = "failed"
            row["error"] = str(error)
        finally:
            if process is not None:
                SWEEP.stop_process(process)
    text = path.read_text(errors="replace")
    if row["status"] != "ok" and SWEEP.is_arena_capacity_failure(text):
        row["status"] = "capacity_failure"
    accepted, generated = SWEEP.parse_draft_acceptance(measured_text if measured_text is not None else text)
    timing = row.get("timings", {})
    accepted = timing.get("draft_n_accepted", accepted)
    generated = timing.get("draft_n", generated)
    row.update(draft_accepted=accepted, draft_generated=generated, wall_seconds=time.monotonic()-started,
               finished_unix=time.time())
    return row


def run_max_point(args, context, mtp, token_cache):
    args.output.mkdir(parents=True, exist_ok=True)
    probe_args = argparse.Namespace(**vars(args))
    probe_args.uvm = False
    # All profiles use the same four-token verification headroom.
    probe_args.mtp_lengths = (0,1,2,3)
    text = args.prefill_text.read_text()
    url = f"http://127.0.0.1:{args.port}"
    fast_max = SWEEP.find_max_arena_mib(args.arena_mib,
        lambda arena: SWEEP.probe_arena(probe_args, context, mtp, arena, args.output, url, text, token_cache))
    rows = {}
    with (args.output/"arena-probes.jsonl").open("a") as output:
        def fits(arena):
            if arena not in rows:
                row = run_point(args, context, mtp, arena, token_cache)
                output.write(json.dumps(row)+"\n")
                output.flush()
                if row["status"] == "failed":
                    minimum = SWEEP.arena_minimum_mib(Path(row["log"]).read_text(errors="replace"))
                    if minimum is not None:
                        raise SWEEP.ArenaTooSmallError(minimum)
                    raise RuntimeError(f"execution failure during arena qualification; see {row['log']}")
                rows[arena] = row
            return rows[arena]["status"] == "ok"
        maximum = SWEEP.find_max_arena_mib(fast_max, fits, upper_limit=fast_max)
    return rows[maximum]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--contexts", default="8192,98304,131072,196608")
    parser.add_argument("--mtp-lengths", type=SWEEP.parse_mtp_lengths, default=(0,1,2,3))
    parser.add_argument("--arena-mib", type=int, default=2240)
    parser.add_argument("--auto-max-arena", action="store_true",
                        help="find and fully validate each context/mode maximum in 1 MiB steps; no reserve")
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--prefill-text", type=Path, default=ROOT / "benchmarks/data/online-articles-262144-words.txt")
    parser.add_argument("--port", type=int, default=12351)
    parser.add_argument("--stock", action="store_true")
    parser.add_argument("--verify-reuse", action="store_true",
                        help="retain cached-prefix and fresh continuations for matching-stock comparisons after measurement")
    args = parser.parse_args()
    if args.stock and args.auto_max_arena:
        parser.error("stock has no adaptive arena to size")
    args.batch_size = args.ubatch_size = 256
    args.model, args.server, args.output = args.model.resolve(), args.server.resolve(), args.output.resolve()
    contexts = [int(value) for value in args.contexts.split(",")]
    if args.decode_tokens < 16 or args.arena_mib < 1 or any(c < args.decode_tokens+260 or c > 262144 for c in contexts):
        parser.error("invalid decode/context/arena limits")
    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "results.jsonl"
    if results_path.exists():
        parser.error("output already contains results; choose a new directory")
    token_cache = {}
    with results_path.open("x") as output:
        for context in contexts:
            for mtp in args.mtp_lengths:
                row = run_max_point(args, context, mtp, token_cache) if args.auto_max_arena else run_point(args, context, mtp, args.arena_mib, token_cache)
                output.write(json.dumps(row)+"\n")
                output.flush()
                print(json.dumps({k:v for k,v in row.items() if k not in ("token_ids","content")}), flush=True)
                if row["status"] == "failed":
                    raise RuntimeError(f"qualification failure; see {row['log']}")


if __name__ == "__main__":
    main()
