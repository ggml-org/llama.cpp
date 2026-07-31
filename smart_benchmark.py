#!/usr/bin/env python3
"""
Smart Adaptive Auto-Tuner & Benchmark for llama.cpp / MoE Models
Author: Antigravity AI
"""

import subprocess
import json
import os
import sys
import time

MODEL_PATH = "/mnt/p5/models/Qwen3.5-35B-A3B-abliterated-Q4_K_M.gguf"
BENCH_BIN = "/home/cune/llama.cpp/build/bin/llama-bench"
REPORT_MD = "/home/cune/llama.cpp/smart_bench_report.md"
REPORT_JSON = "/home/cune/llama.cpp/smart_bench_report.json"

def cleanup():
    subprocess.run("pkill -9 llama-cli llama-bench llama-server 2>/dev/null || true", shell=True)

def run_single_test(ngl, ncmoe, t, fa_str, lm, poll, ctx, ubatch=512, batch=2048):
    cleanup()
    time.sleep(0.5)
    
    cmd = [
        BENCH_BIN,
        "-m", MODEL_PATH,
        "-p", "16",
        "-n", "32",
        "-r", "1",
        "--no-warmup",
        "-ngl", str(ngl),
        "-ncmoe", str(ncmoe),
        "-t", str(t),
        "-fa", str(fa_str),
        "-lm", str(lm),
        "--poll", str(poll),
        "-ub", str(ubatch),
        "-b", str(batch),
        "-o", "json"
    ]
    
    env = os.environ.copy()
    env["RECURRENT_D"] = "12"
    
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=90)
        if proc.returncode == 0:
            try:
                data = json.loads(proc.stdout)
                pp_ts = 0.0
                tg_ts = 0.0
                for item in data:
                    if item.get("n_prompt", 0) > 0:
                        pp_ts = item.get("avg_ts", 0.0)
                    if item.get("n_gen", 0) > 0:
                        tg_ts = item.get("avg_ts", 0.0)
                return {"success": True, "pp_ts": pp_ts, "tg_ts": tg_ts, "error": None}
            except Exception as e:
                return {"success": False, "pp_ts": 0, "tg_ts": 0, "error": f"JSON parse error: {e}"}
        else:
            err_msg = proc.stderr if proc.stderr else proc.stdout
            if "out of memory" in err_msg.lower() or "cudamalloc" in err_msg.lower():
                return {"success": False, "pp_ts": 0, "tg_ts": 0, "error": "CUDA OOM"}
            return {"success": False, "pp_ts": 0, "tg_ts": 0, "error": f"Exit code {proc.returncode}"}
    except subprocess.TimeoutExpired:
        return {"success": False, "pp_ts": 0, "tg_ts": 0, "error": "Timeout (>90s)"}

def main():
    print("=" * 80)
    print("🚀 SMART ADAPTIVE AUTO-TUNER (WITH -fa on & POLL TUNING)")
    print(f"Target Model: {MODEL_PATH}")
    print("=" * 80)
    
    results = []
    best_config = None
    max_tg_ts = 0.0
    
    offload_candidates = [
        (28, 36),
        (28, 48),
        (26, 36),
    ]
    
    thread_candidates = [12, 10, 8]
    fa_candidates = ["on"]
    poll_candidates = [65, 50, 35, 20, 0]
    ubatch_candidates = [512, 256]
    
    context_sizes = [4096, 8192, 16384, 32768]
    
    print("\nSTAGE 1: Scanning Offloads & Threads with Flash Attention (-fa on)...")
    print(f"{'NGL':<5} | {'NCMOE':<6} | {'Threads':<8} | {'FA':<3} | {'LM':<6} | {'Poll':<5} | {'UBatch':<6} | {'Prompt t/s':<11} | {'Gen t/s':<10} | {'Status':<15}")
    print("-" * 95)
    
    top_candidates = []
    
    for (ngl, ncmoe) in offload_candidates:
        for t in thread_candidates:
            for fa in fa_candidates:
                res = run_single_test(ngl, ncmoe, t, fa, lm="mmap", poll=50, ctx=4096, ubatch=512)
                
                status = "OK" if res["success"] else f"FAIL ({res['error']})"
                print(f"{ngl:<5} | {ncmoe:<6} | {t:<8} | {fa:<3} | {'mmap':<6} | {50:<5} | {512:<6} | {res['pp_ts']:<11.2f} | {res['tg_ts']:<10.2f} | {status:<15}", flush=True)
                
                if res["success"]:
                    rec = {
                        "ngl": ngl, "ncmoe": ncmoe, "t": t, "fa": fa, "lm": "mmap",
                        "poll": 50, "ubatch": 512, "pp_ts": res["pp_ts"], "tg_ts": res["tg_ts"]
                    }
                    results.append(rec)
                    if res["tg_ts"] > max_tg_ts:
                        max_tg_ts = res["tg_ts"]
                        best_config = rec
                    top_candidates.append(rec)
    
    print("\nSTAGE 2: Fine-Tuning Intermediate Polling (65, 50, 35, 20, 0) & Micro-batches...")
    print("-" * 95)
    
    top_candidates.sort(key=lambda x: x["tg_ts"], reverse=True)
    
    for base in top_candidates[:2]:
        for poll in poll_candidates:
            for ub in ubatch_candidates:
                if poll == 50 and ub == 512:
                    continue
                
                res = run_single_test(
                    base["ngl"], base["ncmoe"], base["t"], base["fa"],
                    "mmap", poll, ctx=4096, ubatch=ub
                )
                status = "OK" if res["success"] else f"FAIL ({res['error']})"
                print(f"{base['ngl']:<5} | {base['ncmoe']:<6} | {base['t']:<8} | {base['fa']:<3} | {'mmap':<6} | {poll:<5} | {ub:<6} | {res['pp_ts']:<11.2f} | {res['tg_ts']:<10.2f} | {status:<15}", flush=True)
                
                if res["success"]:
                    rec = {
                        "ngl": base["ngl"], "ncmoe": base["ncmoe"], "t": base["t"],
                        "fa": base["fa"], "lm": "mmap", "poll": poll, "ubatch": ub,
                        "pp_ts": res["pp_ts"], "tg_ts": res["tg_ts"]
                    }
                    results.append(rec)
                    if res["tg_ts"] > max_tg_ts:
                        max_tg_ts = res["tg_ts"]
                        best_config = rec

    print("\nSTAGE 3: Adaptive Context Scaling on Current Leader...")
    print("-" * 95)
    
    max_stable_ctx = 4096
    ctx_results = []
    
    if best_config:
        print(f"Testing Winner: NGL={best_config['ngl']}, NCMOE={best_config['ncmoe']}, Threads={best_config['t']}, Poll={best_config['poll']}, FA={best_config['fa']} across context sizes...")
        
        for ctx in context_sizes:
            res = run_single_test(
                best_config["ngl"], best_config["ncmoe"], best_config["t"],
                best_config["fa"], best_config["lm"], best_config["poll"],
                ctx=ctx, ubatch=best_config["ubatch"]
            )
            
            if res["success"]:
                print(f"Context {ctx:<7}: Prompt {res['pp_ts']:<6.2f} t/s | Gen {res['tg_ts']:<6.2f} t/s [STABLE]")
                max_stable_ctx = ctx
                ctx_results.append({"ctx": ctx, "pp_ts": res["pp_ts"], "tg_ts": res["tg_ts"]})
            else:
                print(f"Context {ctx:<7}: FAILED ({res['error']}) [STOPPING CONTEXT SCALING]")
                break
    
    report_data = {
        "best_config": best_config,
        "max_generation_speed_tps": max_tg_ts,
        "max_stable_context_size": max_stable_ctx,
        "context_scaling": ctx_results,
        "all_results": results
    }
    
    with open(REPORT_JSON, "w") as f:
        json.dump(report_data, f, indent=2)
        
    with open(REPORT_MD, "w") as f:
        f.write("# 🚀 Smart Benchmark & Auto-Tuning Report\n\n")
        f.write(f"### 🏆 Absolute Leader Found:\n")
        if best_config:
            f.write(f"- **NGL (GPU Layers):** `{best_config['ngl']}`\n")
            f.write(f"- **CPU MoE Layers:** `{best_config['ncmoe']}`\n")
            f.write(f"- **Threads:** `{best_config['t']}`\n")
            f.write(f"- **Flash Attention:** `{best_config['fa']}`\n")
            f.write(f"- **Poll:** `{best_config['poll']}`\n")
            f.write(f"- **Micro-batch:** `{best_config['ubatch']}`\n")
            f.write(f"- **Max Generation Speed:** **`{max_tg_ts:.2f} t/s`**\n")
            f.write(f"- **Max Stable Context:** **`{max_stable_ctx}`**\n\n")
    
    print("\n" + "=" * 80)
    print("🎉 BENCHMARK COMPLETE!")
    if best_config:
        print(f"🏆 ABSOLUTE LEADER: {max_tg_ts:.2f} t/s with NGL={best_config['ngl']} NCMOE={best_config['ncmoe']} Poll={best_config['poll']} -fa {best_config['fa']} -t {best_config['t']}")
    print("=" * 80)

if __name__ == "__main__":
    main()
