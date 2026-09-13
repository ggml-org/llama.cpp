#!/usr/bin/env python3
"""
Fictitious / Non-Parametric LongBench Multi-Hop QA:
Tests if LongBench multi-hop fails when models cannot cheat using pre-trained parametric memory.
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
    parser.add_argument("--percentages", default="1.0,0.50,0.30,0.15,0.08")
    return parser.parse_args()

def run_test(bin_path, target_model, draft_model, prompt, p, ngl, ngld, max_gen=32, chunk_size=32, lookahead=4):
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
    output = proc.stdout + "\n" + proc.stderr

    gen_text = ""
    m_gen = re.search(r"--- Generation Start ---\n(.*?)\n--- Generation End ---", output, re.DOTALL)
    if m_gen:
        gen_text = m_gen.group(1).strip()

    m_kept = re.search(r"speculative prefill kept (\d+) / (\d+) tokens", output)
    kept_tokens = int(m_kept.group(1)) if m_kept else 0
    total_tokens = int(m_kept.group(2)) if m_kept else 0

    return {
        "gen_text": gen_text,
        "kept": kept_tokens,
        "total": total_tokens,
    }

def main():
    args = parse_args()
    percentages = [float(p.strip()) for p in args.percentages.split(",") if p.strip()]

    distractors = [
        "The deep-space research vessel Hyperion-1 was commissioned for extrasolar navigation under Captain Robert Vance, who completed flight certification at the Orbital Institute founded in 2185.",
        "Planetary survey vessel Solaris-4 mapped asteroid belts in the Gliese system under Commander Maya Lin, who graduated from the Ceres Technical Academy founded in 2240.",
        "Cargo transport vessel Atlas-7 transported hydrogen fuel cells across outer lunar outposts under Navigator Eric Zhao, trained at the Lunar Flight School established in 2199.",
        "Atmospheric probing vessel Zephyr-1 investigated gas giant storm dynamics in the Kepler perimeter under Pilot Laura Gomez, an alumna of the Jovian Navigation College founded in 2275.",
        "Deep reconnaissance cruiser Vanguard-6 conducted quantum radar calibration under Captain Julian Ross, who attended the Sirius Space Academy founded in 2310.",
        "Mining support cruiser Titan-3 escorted mineral transport convoys along the Kuiper barrier under Officer Daniel Park, who studied at the Neptune Defense Institute founded in 2225.",
        "Hydroponics research station Demeter-2 developed closed-loop atmospheric life support systems under Director Sophia Kim, educated at the Bio-Engineering Academy founded in 2260.",
        "Orbital telescope array Copernicus-8 tracked gravitational microlensing events under Astrophysicist Marcus Silva, certified by the Stellar Observation College founded in 2305.",
        "Planetary defense frigate Aegis-5 patrolled inner asteroid orbital lanes under Commander Nathan Drake, trained at the Martian Aerospace Center established in 2170.",
        "Communication relay platform Hermes-9 maintained laser relay arrays across the Sol grid under Technician Elena Cruz, certified by the Quantum Network Academy founded in 2280."
    ]

    doc_target_1 = "The exploratory starship Zephyr-9 was commissioned for deep-space cartography under the command of Captain Alyssa Thorne."
    doc_target_2 = "Captain Alyssa Thorne completed her advanced astrogation degree at the Pioneer Space Academy, which was founded in 2348 on New Horizon."

    context_paras = list(distractors[:5]) + [doc_target_1] + list(distractors[5:]) + [doc_target_2]
    context = "\n\n".join(context_paras)

    prompt = (
        f"{context}\n\n"
        "Question: In what year was the academy where the commander of starship Zephyr-9 completed her degree founded?\n"
        "Answer: The academy where the commander of starship Zephyr-9 completed her degree was founded in"
    )

    print("=" * 85)
    print("NON-PARAMETRIC LONGBENCH MULTI-HOP QA BENCHMARK")
    print("=" * 85)
    print("Target Year : 2348")
    print("Prompt Size : ~1,000 tokens across 12 distinct non-redundant documents")
    print("-" * 85)

    for p in percentages:
        res = run_test(args.bin, args.target_model, args.draft_model, prompt, p, args.n_gpu_layers, args.n_gpu_layers_draft)
        passed = "2348" in res["gen_text"]
        status = "PASS" if passed else "FAIL"
        print("Keep %3d%% | Kept %4d/%4d tokens | Status: [%-4s] | Output: \"%s\"" % (
            int(p*100), res["kept"], res["total"], status, res["gen_text"].replace("\n", " ")[:45]))
    print("=" * 85)

if __name__ == "__main__":
    main()
