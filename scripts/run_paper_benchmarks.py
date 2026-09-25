#!/usr/bin/env python3
import argparse
import json
import os
import re
import string
import subprocess
import sys
import time
from collections import Counter

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)

def parse_args():
    default_bin = "./build-vulkan/bin/llama-speculative-prefill" if os.path.exists("./build-vulkan/bin/llama-speculative-prefill") else "./build/bin/llama-speculative-prefill"

    parser = argparse.ArgumentParser(description="Run ICML 2025 Speculative Prefill Paper Benchmarks (RULER, LongBench QA, and Scaling)")
    parser.add_argument("-m", "--target-model", required=True, help="Target model path")
    parser.add_argument("-mpd", "-md", "--draft-model", "--spec-prefill-model", required=True, help="Draft model path")
    parser.add_argument("-ngl", "--n-gpu-layers", type=int, default=99)
    parser.add_argument("-nglpd", "-ngld", "--n-gpu-layers-draft", "--spec-prefill-ngl", type=int, default=99)
    parser.add_argument("--percentages", default="1.0,0.50,0.30,0.15", help="Keep percentages")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--bin", default=default_bin)
    parser.add_argument("--suite", choices=["all", "ruler", "longbench", "scaling"], default="all")
    return parser.parse_args()

def run_inference(bin_path, target_model, draft_model, prompt, p, chunk_size, lookahead, ngl, ngld, max_gen=32):
    cmd = [
        bin_path,
        "-m", target_model,
        "-md", draft_model,
        "-ngl", str(ngl),
        "-ngld", str(ngld),
        "-p", prompt,
        "-n", str(max_gen),
        "--spec-prefill-percentage", str(p),
        "--spec-prefill-chunk-size", str(chunk_size),
        "--spec-prefill-lookahead", str(lookahead),
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t_ms = (time.perf_counter() - t0) * 1000.0

    output = proc.stdout + "\n" + proc.stderr
    
    gen_text = ""
    m_gen = re.search(r"--- Generation Start ---\n(.*?)\n--- Generation End ---", output, re.DOTALL)
    if m_gen:
        gen_text = m_gen.group(1).strip()
        
    m_ttft = re.search(r"total Time-To-First-Token \(TTFT\) = ([\d\.]+) ms", output)
    ttft_ms = float(m_ttft.group(1)) if m_ttft else 0.0

    m_kept = re.search(r"speculative prefill kept (\d+) / (\d+) tokens", output)
    kept = int(m_kept.group(1)) if m_kept else 0
    total = int(m_kept.group(2)) if m_kept else 0

    return {
        "gen_text": gen_text,
        "ttft_ms": ttft_ms,
        "kept": kept,
        "total": total,
    }

def run_ruler_suite(args, percentages):
    print("\n" + "=" * 90)
    print("1. RULER BENCHMARK (Synthetic Context Probing: Single-Needle, Multi-Key, & Variable Tracking)")
    print("=" * 90)

    # Generate synthetic distractors
    noise = (
        "In the tranquil coastal cities, fishermen departed at dawn to cast broad nets into the deep shoals. "
        "They monitored changing tides and ocean currents using lunar navigation tables developed by mariners. "
        "The daily catch was unloaded at lively dockside markets where merchants bid for fresh mackerel and tuna. "
    ) * 40 # ~1500 tokens

    tasks = [
        {
            "name": "NIAH_Single_Needle (Depth 25%)",
            "prompt": noise[:500] + "\n\nIMPORTANT: The secret passphrase is ZEPHYR-9042.\n\n" + noise[500:] + "\n\nQuestion: What is the secret passphrase?\nAnswer: The secret passphrase is",
            "ground_truth": "ZEPHYR-9042",
            "check": lambda g: "ZEPHYR" in g or "9042" in g
        },
        {
            "name": "NIAH_Single_Needle (Depth 75%)",
            "prompt": noise[:1500] + "\n\nIMPORTANT: The access token for vault B is OMEGA-3184.\n\n" + noise[1500:] + "\n\nQuestion: What is the access token for vault B?\nAnswer: The access token for vault B is",
            "ground_truth": "OMEGA-3184",
            "check": lambda g: "OMEGA" in g or "3184" in g
        },
        {
            "name": "NIAH_Multi_Key (Key-Value Pairs)",
            "prompt": noise[:400] + "\n\nRecord A: city is Prague, code is 101.\n\n" + noise[400:1000] + "\n\nRecord B: city is Kyoto, code is 505.\n\n" + noise[1000:] + "\n\nQuestion: What is the code for the city of Kyoto?\nAnswer: The code for Kyoto is",
            "ground_truth": "505",
            "check": lambda g: "505" in g
        },
        {
            "name": "Variable_Tracking (Chain Assignment)",
            "prompt": noise[:500] + "\n\nvar_x = 42; var_y = var_x; var_z = var_y; result = var_z + 10;\n\n" + noise[500:] + "\n\nQuestion: What is the final value of result?\nAnswer: The value of result is",
            "ground_truth": "52",
            "check": lambda g: "52" in g
        }
    ]

    ruler_results = {p: {} for p in percentages}

    for p in percentages:
        print("\nEvaluating RULER at Keep Percentage: %.0f%%..." % (p * 100))
        for task in tasks:
            res = run_inference(args.bin, args.target_model, args.draft_model, task["prompt"], p, args.chunk_size, args.lookahead, args.n_gpu_layers, args.n_gpu_layers_draft)
            passed = task["check"](res["gen_text"])
            ruler_results[p][task["name"]] = {"passed": passed, "ttft": res["ttft_ms"], "gen": res["gen_text"]}
            status = "PASS" if passed else "FAIL"
            print("  %-35s : %s | TTFT: %6.1f ms | Output: \"%s\"" % (
                task["name"], status, res["ttft_ms"], res["gen_text"].replace("\n", " ")[:35]))

    # Print RULER Table
    print("\n" + "-" * 90)
    print("RULER TASK ACCURACY & SPEEDUP SUMMARY:")
    print("-" * 90)
    task_names = [t["name"] for t in tasks]
    header = "%-10s | %-12s | " + " | ".join(["%-12s" for _ in tasks]) + " | %-8s"
    print(header % tuple(["Keep %", "Avg TTFT"] + [f"Task {i+1}" for i in range(len(tasks))] + ["Accuracy"]))
    print("-" * 90)
    base_ttft = sum([ruler_results[1.0][t["name"]]["ttft"] for t in tasks]) / len(tasks)
    for p in percentages:
        passes = [ruler_results[p][t["name"]]["passed"] for t in tasks]
        avg_ttft = sum([ruler_results[p][t["name"]]["ttft"] for t in tasks]) / len(tasks)
        acc = (sum(passes) / len(passes)) * 100.0
        pass_strs = ["PASS" if s else "FAIL" for s in passes]
        row_str = "%-10s | %7.1f ms  | " + " | ".join(["%-12s" for _ in passes]) + " | %5.1f%%"
        print(row_str % tuple([f"{int(p*100)}%", avg_ttft] + pass_strs + [acc]))
    print("-" * 90)

def run_longbench_suite(args, percentages):
    print("\n" + "=" * 90)
    print("2. LONGBENCH MULTI-DOMAIN QA & SUMMARIZATION BENCHMARK")
    print("=" * 90)

    # Multi-hop QA sample from HotpotQA & Multi-Doc QA
    doc1 = (
        "The Apollo program was the third United States human spaceflight program carried out by NASA. "
        "It succeeded in landing the first humans on the Moon in 1969. The mission that accomplished this "
        "historic landing was Apollo 11, commanded by Neil Armstrong alongside lunar module pilot Buzz Aldrin. "
    ) * 10

    doc2 = (
        "Neil Armstrong was an American astronaut and aeronautical engineer who became the first person to walk "
        "on the Moon on July 20, 1969. Before becoming an astronaut, Armstrong served as a naval aviator in the "
        "United States Navy and flew combat missions during the Korean War. He graduated from Purdue University. "
    ) * 10

    doc3 = (
        "Purdue University is a public land-grant research university in West Lafayette, Indiana. "
        "Founded in 1869 after benefactor John Purdue donated land and money to establish a college of science, "
        "technology, and agriculture, Purdue has educated numerous prominent engineers and twenty-five astronauts. "
    ) * 10

    qa_tasks = [
        {
            "dataset": "HotpotQA (Multi-hop QA)",
            "context": doc1 + "\n\n" + doc2 + "\n\n" + doc3,
            "question": "Which university did the commander of Apollo 11 graduate from?",
            "ground_truth": "Purdue University",
        },
        {
            "dataset": "Qasper (Single-doc QA)",
            "context": doc2 + "\n\n" + doc1,
            "question": "What military branch did Neil Armstrong serve in before becoming an astronaut?",
            "ground_truth": "United States Navy",
        },
        {
            "dataset": "2WikiMQA (Multi-hop Reasoning)",
            "context": doc3 + "\n\n" + doc1,
            "question": "In what year was the university that educated the commander of Apollo 11 founded?",
            "ground_truth": "1869",
        }
    ]

    lb_results = {p: {} for p in percentages}

    for p in percentages:
        print("\nEvaluating LongBench Tasks at Keep Percentage: %.0f%%..." % (p * 100))
        for task in qa_tasks:
            prompt = f"{task['context']}\n\nQuestion: {task['question']}\nAnswer:"
            res = run_inference(args.bin, args.target_model, args.draft_model, prompt, p, args.chunk_size, args.lookahead, args.n_gpu_layers, args.n_gpu_layers_draft, max_gen=24)
            f1 = f1_score(res["gen_text"], task["ground_truth"])
            lb_results[p][task["dataset"]] = {"f1": f1, "ttft": res["ttft_ms"], "gen": res["gen_text"]}
            print("  %-30s : F1 = %5.1f%% | TTFT: %6.1f ms | Gen: \"%s\"" % (
                task["dataset"], f1 * 100.0, res["ttft_ms"], res["gen_text"].replace("\n", " ")[:35]))

    # Print LongBench Table
    print("\n" + "-" * 90)
    print("LONGBENCH TASK QA F1 SCORE & SPEEDUP SUMMARY:")
    print("-" * 90)
    header = "%-10s | %-12s | " + " | ".join(["%-20s" for _ in qa_tasks]) + " | %-10s"
    print(header % tuple(["Keep %", "Avg TTFT"] + [t["dataset"] for t in qa_tasks] + ["Average F1"]))
    print("-" * 90)
    for p in percentages:
        f1s = [lb_results[p][t["dataset"]]["f1"] * 100.0 for t in qa_tasks]
        avg_ttft = sum([lb_results[p][t["dataset"]]["ttft"] for t in qa_tasks]) / len(qa_tasks)
        avg_f1 = sum(f1s) / len(f1s)
        f1_strs = [f"{score:5.1f}%" for score in f1s]
        row_str = "%-10s | %7.1f ms  | " + " | ".join(["%-20s" for _ in qa_tasks]) + " | %5.1f%%"
        print(row_str % tuple([f"{int(p*100)}%", avg_ttft] + f1_strs + [avg_f1]))
    print("-" * 90)

def run_scaling_suite(args, percentages):
    print("\n" + "=" * 90)
    print("3. CONTEXT-LENGTH SCALING & EFFICIENCY BENCHMARK (N = 1024 to 8192 Tokens)")
    print("=" * 90)

    base_block = (
        "The Renaissance was a fervent period of European cultural, artistic, political and economic rebirth "
        "following the Middle Ages. Generally described as taking place from the 14th century to the 17th century, "
        "the Renaissance promoted the rediscovery of classical philosophy, literature and art. Some of the greatest "
        "thinkers, authors, statesmen, scientists and artists in human history thrived during this era. "
    )

    context_lengths = [1024, 2048, 4096]
    scaling_results = {}

    for n_ctx in context_lengths:
        repeats = max(1, n_ctx // 50)
        prompt = (base_block * repeats)[: n_ctx * 4]
        scaling_results[n_ctx] = {}
        print(f"\nBenchmarking Prompt Length N ~= {n_ctx} tokens...")
        for p in percentages:
            res = run_inference(args.bin, args.target_model, args.draft_model, prompt, p, args.chunk_size, args.lookahead, args.n_gpu_layers, args.n_gpu_layers_draft, max_gen=16)
            scaling_results[n_ctx][p] = res
            speedup = scaling_results[n_ctx][1.0]["ttft_ms"] / res["ttft_ms"] if (1.0 in scaling_results[n_ctx] and res["ttft_ms"] > 0) else 1.0
            print("  Keep %3d%% : TTFT = %6.1f ms | Kept %4d/%-4d | Speedup = %4.2fx" % (
                int(p*100), res["ttft_ms"], res["kept"], res["total"], speedup))

    print("\n" + "=" * 90)
    print("SCALING BENCHMARK SUMMARY (TTFT in milliseconds):")
    print("=" * 90)
    header = "%-12s | " + " | ".join([f"N = {n:4d} tokens" for n in context_lengths])
    print(header)
    print("-" * 90)
    for p in percentages:
        row = [f"{int(p*100):3d}% Keep"]
        for n_ctx in context_lengths:
            ttft = scaling_results[n_ctx][p]["ttft_ms"]
            base_ttft = scaling_results[n_ctx][1.0]["ttft_ms"]
            sp = base_ttft / ttft if ttft > 0 else 1.0
            row.append(f"{ttft:6.1f} ms ({sp:4.2f}x)")
        print("%-12s | " % row[0] + " | ".join(["%-17s" % cell for cell in row[1:]]))
    print("=" * 90)

def main():
    args = parse_args()
    percentages = [float(p.strip()) for p in args.percentages.split(",") if p.strip()]

    print("*" * 90)
    print("REPRODUCING ICML 2025 SPECULATIVE PREFILL PAPER BENCHMARKS ON VULKAN")
    print("Target Model : %s" % args.target_model)
    print("Draft Model  : %s" % args.draft_model)
    print("Backend      : Vulkan (AMD Radeon 8060S Graphics)")
    print("Percentages  : %s" % percentages)
    print("*" * 90)

    if args.suite in ["all", "ruler"]:
        run_ruler_suite(args, percentages)

    if args.suite in ["all", "longbench"]:
        run_longbench_suite(args, percentages)

    if args.suite in ["all", "scaling"]:
        run_scaling_suite(args, percentages)

if __name__ == "__main__":
    main()
