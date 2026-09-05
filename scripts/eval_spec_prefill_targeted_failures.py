#!/usr/bin/env python3
"""
Diagnostic test script demonstrating hard failure modes for speculative prefill:
1. True Semantic Disconnection Multi-Hop (Hop 2 has zero semantic overlap with the prompt)
2. Multi-Entity / Multi-Fact High-Recall Aggregation (5 distinct facts scattered across 5 chunks)
3. Conflicting Override / Negation
4. Distributed Key-Value Retrieval (10 KV pairs where question queries 3 random pairs)
"""

import argparse
import os
import re
import subprocess
import time

def parse_args():
    default_bin = "./build-vulkan/bin/llama-speculative-prefill" if os.path.exists("./build-vulkan/bin/llama-speculative-prefill") else "./build/bin/llama-speculative-prefill"
    default_tgt = "/home/rocko/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q6_K_XL.gguf"
    default_dft = "/home/rocko/.cache/huggingface/hub/models--unsloth--Qwen3.5-2B-GGUF/snapshots/f6d5376be1edb4d416d56da11e5397a961aca8ae/Qwen3.5-2B-UD-Q4_K_XL.gguf"

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--target-model", default=default_tgt)
    parser.add_argument("-md", "--draft-model", default=default_dft)
    parser.add_argument("-ngl", "--n-gpu-layers", type=int, default=99)
    parser.add_argument("-ngld", "--n-gpu-layers-draft", type=int, default=99)
    parser.add_argument("--bin", default=default_bin)
    parser.add_argument("--percentages", default="1.0,0.50,0.30,0.15,0.10")
    return parser.parse_args()

def run_test(bin_path, target_model, draft_model, prompt, p, ngl, ngld, max_gen=64, chunk_size=32, lookahead=4):
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
    kept_tokens = int(m_kept.group(1)) if m_kept else 0
    total_tokens = int(m_kept.group(2)) if m_kept else 0

    return {
        "gen_text": gen_text,
        "ttft_ms": ttft_ms,
        "kept": kept_tokens,
        "total": total_tokens,
        "raw": output
    }

def get_distractors(n=20):
    text = [
        "The geological survey team conducted detailed mineralogical analyses in the western foothills. Core samples revealed substantial granite and quartz strata with trace deposits of copper ore.",
        "Municipal water engineers redesigned the gravity-fed aqueduct conduits. Pressure regulation cisterns and basalt masonry arches ensured uninterrupted flow into central metropolitan fountains.",
        "Agricultural registers recorded seasonal barley harvests across forty alluvial valleys. Yields were cataloged in standard bushel units and distributed to state granaries via river barges.",
        "Archival preservationists treated ancient parchment scrolls using mild cedar oil emulsions. Manuscript codices from the third century were rebound in pigskin covers and stored in dry vaults.",
        "Textile manufacturing facilities calibrated loom tension for fine linen production. Natural madder dye vats produced crimson vestments exported to neighboring maritime principalities.",
        "Harbor authorities logged the arrival of merchant frigates carrying spices, porcelain, and timber. Port tariffs were collected in silver bullion and recorded in double-entry ledgers.",
        "Astronomers observed the transit of celestial satellites across equatorial constellations. Quadrant measurements were compiled into seasonal navigational ephemerides for mariners."
    ]
    return [text[i % len(text)] for i in range(n)]

def test_pure_latent_multihop():
    """
    Question asks: What clearance tier is assigned to Director Vance's facility?
    Clue 1 (Paragraph 3): 'Director Vance is the sole administrator of Complex Theta.' (Matches Vance, administrator)
    Clue 2 (Paragraph 16): 'Complex Theta is designated as Clearance Tier Cobalt.' (Matches Complex Theta, but NO Vance, NO administrator, NO question terms)
    """
    paras = get_distractors(22)
    paras.insert(3, "Facility Administration Records: Director Vance is the sole administrator of Complex Theta.")
    paras.insert(16, "Security Protocol Index: Complex Theta is designated as Clearance Tier Cobalt.")

    prompt = (
        "%s\n\n"
        "Question: What clearance tier is assigned to the complex managed by Director Vance?\n"
        "Answer: The clearance tier assigned to the complex managed by Director Vance is Clearance Tier"
    ) % ("\n\n".join(paras))

    return {
        "name": "1. True Latent Multi-Hop (Cobalt)",
        "prompt": prompt,
        "check": lambda g: "cobalt" in g.lower()
    }

def test_multi_fact_recall():
    """
    Question asks to list ALL 5 server port numbers.
    Each port is in a completely separate chunk with 0 redundancy.
    """
    paras = get_distractors(20)
    items = [
        ("Network Configuration A: Server Alpha listens on port 8081.", 2),
        ("Network Configuration B: Server Beta listens on port 8082.", 6),
        ("Network Configuration C: Server Gamma listens on port 8083.", 10),
        ("Network Configuration D: Server Delta listens on port 8084.", 14),
        ("Network Configuration E: Server Epsilon listens on port 8085.", 18),
    ]
    for text, idx in sorted(items, key=lambda x: x[1], reverse=True):
        paras.insert(idx, text)

    prompt = (
        "%s\n\n"
        "Question: List the port numbers for all five servers (Alpha, Beta, Gamma, Delta, Epsilon) in order:\n"
        "Answer: Alpha, Beta, Gamma, Delta, Epsilon port numbers:"
    ) % ("\n\n".join(paras))

    def check(g):
        ports = ["8081", "8082", "8083", "8084", "8085"]
        return all(p in g for p in ports)

    return {
        "name": "2. Multi-Fact Full Recall (All 5 Ports)",
        "prompt": prompt,
        "check": check
    }

def test_indirect_math():
    """
    Math with separated values:
    Val A = 15
    Val B = 25
    Val C = 60
    Sum = 100
    """
    paras = get_distractors(20)
    paras.insert(2, "Inventory Section 1: Item Box A contains exactly 15 units.")
    paras.insert(9, "Inventory Section 2: Item Box B contains exactly 25 units.")
    paras.insert(17, "Inventory Section 3: Item Box C contains exactly 60 units.")

    prompt = (
        "%s\n\n"
        "Question: What is the total sum of units contained in Box A, Box B, and Box C combined?\n"
        "Answer: The total sum of units in Box A, Box B, and Box C is exactly"
    ) % ("\n\n".join(paras))

    return {
        "name": "3. Distributed Summation (15 + 25 + 60 = 100)",
        "prompt": prompt,
        "check": lambda g: "100" in g or "one hundred" in g.lower()
    }

def main():
    args = parse_args()
    percentages = [float(p.strip()) for p in args.percentages.split(",") if p.strip()]

    tasks = [
        test_pure_latent_multihop(),
        test_multi_fact_recall(),
        test_indirect_math(),
    ]

    print("=" * 90)
    print("SPECULATIVE PREFILL: TARGETED FAILURE CASE INVESTIGATION")
    print("=" * 90)
    print("Target Model : %s" % args.target_model)
    print("Draft Model  : %s" % args.draft_model)
    print("Percentages  : %s" % percentages)
    print("=" * 90)

    results = {p: {} for p in percentages}

    for p in percentages:
        print(f"\nEvaluating Keep Ratio: {int(p*100)}% ({p:.2f})...")
        for task in tasks:
            res = run_test(args.bin, args.target_model, args.draft_model, task["prompt"], p, args.n_gpu_layers, args.n_gpu_layers_draft, max_gen=32)
            passed = task["check"](res["gen_text"])
            status = "PASS" if passed else "FAIL"
            results[p][task["name"]] = {
                "passed": passed,
                "ttft": res["ttft_ms"],
                "kept": res["kept"],
                "total": res["total"],
                "gen": res["gen_text"].replace("\n", " ")[:45]
            }
            print("  %-45s : [%s] (Kept: %4d/%4d) | Gen: \"%s\"" % (
                task["name"], status, res["kept"], res["total"], results[p][task["name"]]["gen"]))

    print("\n" + "=" * 90)
    print("SUMMARY COMPARISON MATRIX")
    print("=" * 90)
    header = "%-10s | %-11s | " + " | ".join([f"Task {i+1}" for i in range(len(tasks))]) + " | %-8s"
    print(header % ("Keep %", "Tokens Kept", "Accuracy"))
    print("-" * 90)
    for p in percentages:
        passes = [results[p][t["name"]]["passed"] for t in tasks]
        kept = results[p][tasks[0]["name"]]["kept"]
        total = results[p][tasks[0]["name"]]["total"]
        acc = (sum(passes) / len(passes)) * 100.0
        pass_strs = ["PASS" if s else "FAIL" for s in passes]
        row_str = "%-10s | %4d/%-6d | " + " | ".join(["%-6s" for _ in passes]) + " | %5.1f%%"
        print(row_str % tuple([f"{int(p*100)}%", kept, total] + pass_strs + [acc]))
    print("=" * 90)

if __name__ == "__main__":
    main()
