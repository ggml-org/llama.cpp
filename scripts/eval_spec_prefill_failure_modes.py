#!/usr/bin/env python3
"""
Stress test / Failure Mode benchmark for Speculative Prefill in llama.cpp.
Identifies challenging scenarios where speculative prefill drops below 100% accuracy:
1. Multi-hop indirect reasoning (bridging without keyword overlap)
2. Global aggregation / counting across non-redundant distributed context
3. Temporal revision / conflicting overrides (adversarial distractors)
4. Multi-step distributed variable tracking across distant chunks
5. Extreme compression (low keep ratio p <= 0.05 / 0.10)
"""

import argparse
import os
import re
import subprocess
import sys
import time

def parse_args():
    default_bin = "./build-vulkan/bin/llama-speculative-prefill" if os.path.exists("./build-vulkan/bin/llama-speculative-prefill") else "./build/bin/llama-speculative-prefill"
    default_tgt = "/home/rocko/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q6_K_XL.gguf"
    default_dft = "/home/rocko/.cache/huggingface/hub/models--unsloth--Qwen3.5-2B-GGUF/snapshots/f6d5376be1edb4d416d56da11e5397a961aca8ae/Qwen3.5-2B-UD-Q4_K_XL.gguf"

    parser = argparse.ArgumentParser(description="Speculative Prefill Failure Mode / Stress Benchmark")
    parser.add_argument("-m", "--target-model", default=default_tgt, help="Target model GGUF")
    parser.add_argument("-md", "--draft-model", default=default_dft, help="Draft model GGUF")
    parser.add_argument("-ngl", "--n-gpu-layers", type=int, default=99)
    parser.add_argument("-ngld", "--n-gpu-layers-draft", type=int, default=99)
    parser.add_argument("--bin", default=default_bin)
    parser.add_argument("--percentages", default="1.0,0.50,0.30,0.15,0.08,0.05")
    return parser.parse_args()

def run_test(bin_path, target_model, draft_model, prompt, p, ngl, ngld, max_gen=40, chunk_size=32, lookahead=4):
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

def build_distractor_background(num_paras=25):
    topics = [
        "In northern agricultural districts, crop rotation schedules were synchronized with solar calendars. Farmers monitored soil moisture and planted legumes every third season to restore nitrogen balance. Yield estimates were cataloged by regional grain administrators.",
        "Geological survey teams mapped mineral deposits along the western mountain ranges. They noted extensive veins of quartz, granite, and copper ore embedded within metamorphic formations. Laboratory assays confirmed high purity in several core samples.",
        "Maritime navigational treatises detailed the seasonal reversal of coastal monsoons. Vessel captains adjusted their trade voyages to harness prevailing trade winds and avoid shallow reefs. Harbor logs recorded arrival dates and cargo manifests.",
        "Urban sanitation works expanded rapidly during the middle administrative period. Brick-lined drainage channels and elevated aqueducts delivered fresh spring water across residential sectors, drastically reducing waterborne illnesses in the central wards.",
        "Archival historians indexed thousands of legal decrees enacted over three dynasties. Scribes preserved fragile scrolls in temperature-controlled stone vaults to prevent moisture decay. Cross-referencing genealogical records resolved land boundary disputes.",
        "Textile manufacturing guilds established standardized dye recipes using natural pigments. Indigo, madder root, and crushed ochre were blended in copper vats under precise temperature controls. Master weavers inspected linen quality before export.",
        "Hydrological engineers constructed earthen levees along the delta tributaries. Seasonal flooding patterns were regulated through sluice gates to irrigate terraced rice paddies. Silt deposits replenished topsoil fertility across the lower basin.",
        "Astronomical observatories recorded lunar occultations and planetary conjunctions with calibrated astrolabes. Star charts were published biannually for maritime navigation and calendar harmonization across all inland provinces."
    ]
    paras = []
    for i in range(num_paras):
        paras.append(topics[i % len(topics)])
    return paras

def create_multi_hop_latent_task():
    paras = build_distractor_background(25)
    paras.insert(4, "Project Leadership Log: The chief architect and primary designer of the Helios Project is Dr. Elena Rostova.")
    paras.insert(18, "Personnel Biography Note: In her personal leisure time, Dr. Elena Rostova is an accomplished player of the harpsichord.")
    
    prompt = (
        "%s\n\n"
        "Question: What musical instrument is played by the chief architect of the Helios Project?\n"
        "Answer: The musical instrument played by the chief architect of the Helios Project is the"
    ) % ("\n\n".join(paras))
    
    return {
        "name": "1. Multi-Hop Latent (No Keyword Overlap in Hop 2)",
        "prompt": prompt,
        "expected": "harpsichord",
        "check": lambda g: "harpsichord" in g.lower()
    }

def create_distributed_aggregation_task():
    paras = build_distractor_background(20)
    incidents = [
        ("Facility Alpha logged exactly 4 security incidents this quarter.", 2),
        ("Facility Beta logged exactly 7 security incidents this quarter.", 6),
        ("Facility Gamma logged exactly 3 security incidents this quarter.", 10),
        ("Facility Delta logged exactly 5 security incidents this quarter.", 14),
        ("Facility Epsilon logged exactly 2 security incidents this quarter.", 18),
        ("Facility Zeta logged exactly 6 security incidents this quarter.", 22),
    ] # Total = 4 + 7 + 3 + 5 + 2 + 6 = 27
    
    for text, idx in sorted(incidents, key=lambda x: x[1], reverse=True):
        paras.insert(idx, text)
        
    prompt = (
        "%s\n\n"
        "Question: What is the total sum of security incidents reported across Facility Alpha, Beta, Gamma, Delta, Epsilon, and Zeta?\n"
        "Answer: The total sum of security incidents across all six facilities is"
    ) % ("\n\n".join(paras))

    return {
        "name": "2. Global Aggregation (6 Distributed Items)",
        "prompt": prompt,
        "expected": "27",
        "check": lambda g: "27" in g or "twenty-seven" in g.lower()
    }

def create_temporal_override_task():
    paras = build_distractor_background(24)
    paras.insert(3, "Security Notice A: The current primary server access code is ALPHA-1111.")
    paras.insert(12, "Security Notice B: Update regarding server configuration: server access code changed to BETA-2222.")
    paras.insert(22, "FINAL OVERRIDE NOTICE: Under emergency security protocol, the active server access code is ZETA-9999. All earlier codes are revoked and obsolete.")

    prompt = (
        "%s\n\n"
        "Question: According to the final active override notice, what is the valid server access code?\n"
        "Answer: The active valid server access code is"
    ) % ("\n\n".join(paras))

    return {
        "name": "3. Temporal Revision / Final Override (Adversarial)",
        "prompt": prompt,
        "expected": "ZETA-9999",
        "check": lambda g: "zeta" in g.lower() or "9999" in g
    }

def create_distributed_var_tracking_task():
    paras = build_distractor_background(22)
    paras.insert(2, "Code Section 1: initialize register_x = 100;")
    paras.insert(8, "Code Section 2: register_x = register_x + 50;")
    paras.insert(14, "Code Section 3: register_x = register_x * 2;")
    paras.insert(20, "Code Section 4: register_x = register_x - 30;")

    prompt = (
        "%s\n\n"
        "Question: Following all four sequential code sections from 1 to 4, what is the final numeric value of register_x?\n"
        "Answer: The final value of register_x is"
    ) % ("\n\n".join(paras))

    return {
        "name": "4. Distributed Variable Tracking",
        "prompt": prompt,
        "expected": "270",
        "check": lambda g: "270" in g
    }

def main():
    args = parse_args()
    percentages = [float(p.strip()) for p in args.percentages.split(",") if p.strip()]

    print("=" * 95)
    print("SPECULATIVE PREFILL FAILURE MODE & STRESS TEST BENCHMARK")
    print("=" * 95)
    print("Target Model : %s" % args.target_model)
    print("Draft Model  : %s" % args.draft_model)
    print("Keep Ratios  : %s" % percentages)
    print("=" * 95)

    tasks = [
        create_multi_hop_latent_task(),
        create_distributed_aggregation_task(),
        create_temporal_override_task(),
        create_distributed_var_tracking_task(),
    ]

    results = {p: {} for p in percentages}

    for p in percentages:
        print(f"\n>>> Running evaluation for Keep Ratio: {int(p*100)}% ({p:.2f}) <<<")
        for task in tasks:
            res = run_test(args.bin, args.target_model, args.draft_model, task["prompt"], p, args.n_gpu_layers, args.n_gpu_layers_draft)
            passed = task["check"](res["gen_text"])
            status = "PASS" if passed else "FAIL"
            results[p][task["name"]] = {
                "passed": passed,
                "ttft": res["ttft_ms"],
                "kept": res["kept"],
                "total": res["total"],
                "gen": res["gen_text"].replace("\n", " ")[:50],
            }
            print("  %-45s : [%s] (Kept: %d/%d) | Output: \"%s\"" % (
                task["name"], status, res["kept"], res["total"], results[p][task["name"]]["gen"]))

    print("\n" + "=" * 95)
    print("FAILURE MODE BENCHMARK SUMMARY TABLE")
    print("=" * 95)
    header = "%-8s | %-11s | " + " | ".join([f"Task {i+1}" for i in range(len(tasks))]) + " | %-8s"
    print(header % ("Keep %", "Tokens Kept", "Accuracy"))
    print("-" * 95)

    for p in percentages:
        passes = [results[p][t["name"]]["passed"] for t in tasks]
        kept_tokens = results[p][tasks[0]["name"]]["kept"]
        total_tokens = results[p][tasks[0]["name"]]["total"]
        acc = (sum(passes) / len(passes)) * 100.0
        pass_strs = ["PASS" if s else "FAIL" for s in passes]
        row_str = "%-8s | %4d/%-6d | " + " | ".join(["%-6s" for _ in passes]) + " | %5.1f%%"
        print(row_str % tuple([f"{int(p*100)}%", kept_tokens, total_tokens] + pass_strs + [acc]))
    print("=" * 95)

if __name__ == "__main__":
    main()
