#!/usr/bin/env python3
import argparse
import os
import re
import statistics
import subprocess
import sys
import time

def parse_args():
    default_bin = "./build-vulkan/bin/llama-speculative-prefill" if os.path.exists("./build-vulkan/bin/llama-speculative-prefill") else "./build/bin/llama-speculative-prefill"

    parser = argparse.ArgumentParser(description="Compare Speculative Prefill ON vs OFF (Baseline)")
    parser.add_argument("-m", "--target-model", help="Path to target model GGUF")
    parser.add_argument("-hf", "--hf-repo", help="Hugging Face repo for target model (<user>/<model>[:quant])")
    parser.add_argument("-hff", "--hf-file", help="Hugging Face model file for target model")
    parser.add_argument("-hft", "--hf-token", help="Hugging Face access token")
    parser.add_argument("-mpd", "-md", "--draft-model", "--spec-prefill-model", help="Path to draft model GGUF")
    parser.add_argument("-hfpd", "-hfd", "--hf-repo-draft", "--draft-hf", "--spec-prefill-hf", help="Hugging Face repo for draft model (<user>/<model>[:quant])")
    parser.add_argument("-hffd", "--hf-file-draft", help="Hugging Face model file for draft model")
    parser.add_argument("-ngl", "--n-gpu-layers", type=int, default=99, help="Number of GPU layers for target model (default: 99)")
    parser.add_argument("-nglpd", "-ngld", "--n-gpu-layers-draft", "--spec-prefill-ngl", type=int, default=99, help="Number of GPU layers for draft model (default: 99)")
    parser.add_argument("-p", "--prompt", help="Prompt text")
    parser.add_argument("-f", "--file", help="File containing prompt text")
    parser.add_argument("--n-prompt", type=int, help="Synthesize prompt of approx N tokens")
    parser.add_argument("-n", "--n-predict", type=int, default=32, help="Number of tokens to generate (default: 32)")
    parser.add_argument("--percentages", default="0.2,0.3,0.5", help="Comma-separated keep percentages (default: 0.2,0.3,0.5)")
    parser.add_argument("--chunk-size", type=int, default=32, help="Chunk size (default: 32)")
    parser.add_argument("--lookahead", type=int, default=4, help="Lookahead count (default: 4)")
    parser.add_argument("--reps", type=int, default=1, help="Repetitions per test (default: 1)")
    parser.add_argument("--bin", default=default_bin, help="Path to binary (default: auto-detected)")
    parser.add_argument("--threads", type=int, default=0, help="Number of threads (0 for auto)")
    return parser.parse_args()

def generate_synthetic_prompt(n_tokens):
    base = (
        "In an ancient kingdom surrounded by mist-covered mountains, scholars gathered in the great library "
        "to study the ancient manuscripts of astronomy, mathematics, and philosophy. They observed the stars "
        "every evening through brass telescopes, recording every shift in planetary alignment with meticulous "
        "precision. Among them was an eager apprentice named Nicholas, who discovered an enigmatic cipher "
        "hidden within the margins of an age-old star chart. "
    )
    repeats = max(1, n_tokens // 75)
    return (base * repeats).strip()

def run_single(bin_path, model_args, prompt, percentage, chunk_size, lookahead, n_predict, threads, ngl, ngld):
    cmd = [
        bin_path,
        *model_args,
        "-ngl", str(ngl),
        "-ngld", str(ngld),
        "-p", prompt,
        "-n", str(n_predict),
        "--spec-prefill-percentage", str(percentage),
        "--spec-prefill-chunk-size", str(chunk_size),
        "--spec-prefill-lookahead", str(lookahead),
    ]
    if threads > 0:
        cmd.extend(["-t", str(threads)])

    start_t = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    wall_ms = (time.perf_counter() - start_t) * 1000.0

    output = proc.stdout + "\n" + proc.stderr

    if proc.returncode != 0:
        print("Error running command (exit %d):\n%s" % (proc.returncode, output))
        return None

    res = {
        "wall_ms": wall_ms,
        "n_prompt_orig": 0,
        "n_prompt_kept": 0,
        "kept_pct": 100.0,
        "draft_eval_ms": 0.0,
        "estimate_ms": 0.0,
        "sparse_prefill_ms": 0.0,
        "ttft_ms": 0.0,
        "gen_tokens": 0,
        "gen_speed_ts": 0.0,
        "generation": "",
    }

    m = re.search(r"prompt tokens = (\d+)", output)
    if m:
        res["n_prompt_orig"] = int(m.group(1))

    m = re.search(r"speculative prefill kept (\d+) / (\d+) tokens \(([\d\.]+)%\)", output)
    if m:
        res["n_prompt_kept"] = int(m.group(1))
        res["kept_pct"] = float(m.group(3))
    else:
        res["n_prompt_kept"] = res["n_prompt_orig"]
        res["kept_pct"] = 100.0

    m = re.search(r"draft eval time: ([\d\.]+) ms, importance estimation time: ([\d\.]+) ms", output)
    if m:
        res["draft_eval_ms"] = float(m.group(1))
        res["estimate_ms"] = float(m.group(2))

    m = re.search(r"sparse target prefill time = ([\d\.]+) ms", output)
    if m:
        res["sparse_prefill_ms"] = float(m.group(1))

    m = re.search(r"total Time-To-First-Token \(TTFT\) = ([\d\.]+) ms", output)
    if m:
        res["ttft_ms"] = float(m.group(1))
    elif res["sparse_prefill_ms"] > 0:
        res["ttft_ms"] = res["sparse_prefill_ms"]

    m = re.search(r"generated (\d+) tokens in ([\d\.]+) ms \(([\d\.]+) tokens/s\)", output)
    if m:
        res["gen_tokens"] = int(m.group(1))
        res["gen_speed_ts"] = float(m.group(3))

    gen_m = re.search(r"--- Generation Start ---\n(.*?)\n--- Generation End ---", output, re.DOTALL)
    if gen_m:
        res["generation"] = gen_m.group(1).strip()

    return res

def main():
    args = parse_args()

    if not os.path.exists(args.bin):
        print("Error: binary %s not found. Please build llama.cpp first." % args.bin)
        sys.exit(1)

    # Validate target model arguments
    if not args.target_model and not args.hf_repo:
        print("Error: either -m/--target-model or -hf/--hf-repo must be specified for target model.")
        sys.exit(1)

    model_args = []
    target_desc = ""
    draft_desc = ""

    if args.target_model:
        if not os.path.exists(args.target_model):
            print("Error: target model file %s not found." % args.target_model)
            sys.exit(1)
        model_args.extend(["-m", args.target_model])
        target_desc = args.target_model
    elif args.hf_repo:
        model_args.extend(["-hf", args.hf_repo])
        target_desc = args.hf_repo
        if args.hf_file:
            model_args.extend(["-hff", args.hf_file])
            target_desc += " (" + args.hf_file + ")"

    if args.hf_token:
        model_args.extend(["-hft", args.hf_token])

    # Validate draft model arguments
    if args.draft_model:
        if not os.path.exists(args.draft_model):
            print("Error: draft model file %s not found." % args.draft_model)
            sys.exit(1)
        model_args.extend(["-md", args.draft_model])
        draft_desc = args.draft_model
    elif args.hf_repo_draft:
        model_args.extend(["-hfd", args.hf_repo_draft])
        draft_desc = args.hf_repo_draft
        if args.hf_file_draft:
            model_args.extend(["-hffd", args.hf_file_draft])
            draft_desc += " (" + args.hf_file_draft + ")"
    else:
        # Default draft model to target model if unspecified
        if args.target_model:
            model_args.extend(["-md", args.target_model])
            draft_desc = args.target_model + " (self-draft)"
        elif args.hf_repo:
            model_args.extend(["-hfd", args.hf_repo])
            if args.hf_file:
                model_args.extend(["-hffd", args.hf_file])
            draft_desc = args.hf_repo + " (self-draft)"

    if args.prompt:
        prompt = args.prompt
    elif args.file:
        with open(args.file, "r") as f:
            prompt = f.read().strip()
    elif args.n_prompt:
        prompt = generate_synthetic_prompt(args.n_prompt)
    else:
        prompt = (
            "Once upon a time in a bustling mountain village, there was a master clockmaker named Jonathan. "
            "Every morning, he wound the town clock high upon the clocktower, watching the villagers below begin "
            "their daily routines. One foggy autumn morning, an enigmatic traveler arrived bearing a broken mechanical "
            "device covered in strange celestial engravings."
        )

    percentages = [float(p.strip()) for p in args.percentages.split(",") if p.strip()]

    print("=" * 80)
    print("SPECULATIVE PREFILL COMPARISON: ON vs OFF (Baseline)")
    print("=" * 80)
    print("Target Model : %s" % target_desc)
    print("Draft Model  : %s" % draft_desc)
    print("Backend      : Vulkan (GPU layers: %d tgt, %d draft)" % (args.n_gpu_layers, args.n_gpu_layers_draft))
    print("Binary Path  : %s" % args.bin)
    print("Chunk Size   : %d" % args.chunk_size)
    print("Lookahead    : %d" % args.lookahead)
    print("Repetitions  : %d" % args.reps)
    print("Gen Tokens   : %d" % args.n_predict)
    print("-" * 80)

    # 1. Warmup
    print("Running warmup...")
    run_single(args.bin, model_args, prompt, 1.0, args.chunk_size, args.lookahead, 4, args.threads, args.n_gpu_layers, args.n_gpu_layers_draft)

    # 2. Run Baseline (OFF: percentage = 1.0)
    print("\n[1/2] Benchmarking Baseline (Speculative Prefill: OFF, Keep: 100%)...")
    base_runs = []
    base_gen = ""
    for r in range(args.reps):
        res = run_single(args.bin, model_args, prompt, 1.0, args.chunk_size, args.lookahead, args.n_predict, args.threads, args.n_gpu_layers, args.n_gpu_layers_draft)
        if res:
            base_runs.append(res)
            base_gen = res["generation"]
            print("  Rep %d: TTFT = %.2f ms | Prefill = %.2f ms" % (r+1, res["ttft_ms"], res["sparse_prefill_ms"]))

    if not base_runs:
        print("Error: baseline runs failed.")
        sys.exit(1)

    base_ttft_avg = statistics.mean([r["ttft_ms"] for r in base_runs])
    base_ttft_std = statistics.stdev([r["ttft_ms"] for r in base_runs]) if len(base_runs) > 1 else 0.0
    n_prompt_total = base_runs[0]["n_prompt_orig"]

    # 3. Run Speculative Prefill (ON: percentage = p)
    spec_results = {}
    for p in percentages:
        print("\n[2/2] Benchmarking Speculative Prefill ON (Keep: %.0f%%)..." % (p * 100.0))
        runs = []
        spec_gen = ""
        for r in range(args.reps):
            res = run_single(args.bin, model_args, prompt, p, args.chunk_size, args.lookahead, args.n_predict, args.threads, args.n_gpu_layers, args.n_gpu_layers_draft)
            if res:
                runs.append(res)
                spec_gen = res["generation"]
                print("  Rep %d: TTFT = %.2f ms | Draft = %.2f ms | Target Prefill = %.2f ms | Kept = %d/%d" % (
                    r+1, res["ttft_ms"], res["draft_eval_ms"], res["sparse_prefill_ms"], res["n_prompt_kept"], res["n_prompt_orig"]))

        if runs:
            ttft_avg = statistics.mean([r["ttft_ms"] for r in runs])
            ttft_std = statistics.stdev([r["ttft_ms"] for r in runs]) if len(runs) > 1 else 0.0
            speedup = base_ttft_avg / ttft_avg if ttft_avg > 0 else 0.0
            spec_results[p] = {
                "runs": runs,
                "ttft_avg": ttft_avg,
                "ttft_std": ttft_std,
                "speedup": speedup,
                "n_kept": runs[0]["n_prompt_kept"],
                "draft_eval_avg": statistics.mean([r["draft_eval_ms"] for r in runs]),
                "target_prefill_avg": statistics.mean([r["sparse_prefill_ms"] for r in runs]),
                "gen_speed_avg": statistics.mean([r["gen_speed_ts"] for r in runs]),
                "gen_text": spec_gen,
            }

    # Print Summary Table
    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY (Prompt Length N = %d tokens)" % n_prompt_total)
    print("=" * 90)
    header = "%-24s | %-13s | %-16s | %-9s | %-14s" % ("Configuration", "Tokens Kept", "TTFT (ms)", "Speedup", "Throughput")
    print(header)
    print("-" * 90)

    base_tp = (n_prompt_total / (base_ttft_avg / 1000.0)) if base_ttft_avg > 0 else 0.0
    print("%-24s | %5d/%-5d (100%%) | %6.2f ± %-5.2f | %-9s | %8.1f t/s" % (
        "Baseline (OFF, 100%)", n_prompt_total, n_prompt_total, base_ttft_avg, base_ttft_std, "1.00x", base_tp))

    for p, data in spec_results.items():
        kept_str = "%5d/%d (%4.1f%%)" % (data["n_kept"], n_prompt_total, p*100.0)
        ttft_str = "%6.2f ± %-5.2f" % (data["ttft_avg"], data["ttft_std"])
        speedup_str = "%5.2fx" % data["speedup"]
        eff_tp = (n_prompt_total / (data["ttft_avg"] / 1000.0)) if data["ttft_avg"] > 0 else 0.0
        cfg_str = "SpecPrefill (ON, p=%.2f)" % p
        print("%-24s | %-13s | %-16s | %-9s | %8.1f t/s" % (
            cfg_str, kept_str, ttft_str, speedup_str, eff_tp))

    print("=" * 90)

    # Detailed Latency Breakdown
    print("\nLATENCY BREAKDOWN (Speculative Prefill Phases):")
    print("-" * 90)
    print("%-10s | %-15s | %-16s | %-22s | %-12s" % ("Keep %", "Draft Prefill", "Lookahead/Attn", "Target Sparse Prefill", "Total TTFT"))
    print("-" * 90)
    for p, data in spec_results.items():
        lah_time = max(0.0, data["ttft_avg"] - data["draft_eval_avg"] - data["target_prefill_avg"])
        print("%5.1f%%     | %8.2f ms     | %9.2f ms     | %12.2f ms           | %7.2f ms" % (
            p*100.0, data["draft_eval_avg"], lah_time, data["target_prefill_avg"], data["ttft_avg"]))
    print("-" * 90)

    # Output Sample Comparison
    print("\nSAMPLE CONTINUATION OUTPUTS:")
    print("-" * 90)
    print("[Baseline (OFF)]:\n\"%s\"\n" % base_gen)
    for p, data in spec_results.items():
        print("[SpecPrefill (ON, p=%.2f)]:\n\"%s\"\n" % (p, data["gen_text"]))
    print("-" * 90)

if __name__ == "__main__":
    main()
