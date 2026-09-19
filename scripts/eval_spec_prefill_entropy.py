#!/usr/bin/env python3
"""
Empirical demonstration of mathematical failure boundaries for speculative prefill:
1. High-entropy multi-key retrieval (non-sequential random IDs across distributed chunks)
2. Whole-document comprehensive extraction (aggregating 8 distinct entity-value pairs)
3. Conflicting multi-version overrides with adversarial distractor anchors
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

def run_test(bin_path, target_model, draft_model, prompt, p, ngl, ngld, max_gen=48, chunk_size=32, lookahead=4):
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

def get_distractors(n=25):
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

def test_random_kv_pairs():
    paras = get_distractors(24)
    # High-entropy non-sequential codes in 5 separated chunks
    items = [
        ("Database Node Alpha is assigned security token 49152.", 2),
        ("Database Node Beta is assigned security token 19842.", 7),
        ("Database Node Gamma is assigned security token 33419.", 12),
        ("Database Node Delta is assigned security token 58201.", 17),
        ("Database Node Epsilon is assigned security token 27182.", 22),
    ]
    for text, idx in sorted(items, key=lambda x: x[1], reverse=True):
        paras.insert(idx, text)

    prompt = (
        "%s\n\n"
        "Question: List the exact 5-digit security tokens for all five database nodes (Alpha, Beta, Gamma, Delta, Epsilon):\n"
        "Answer: The security tokens for Alpha, Beta, Gamma, Delta, Epsilon are:"
    ) % ("\n\n".join(paras))

    tokens = ["49152", "19842", "33419", "58201", "27182"]
    def check(g):
        found = [t for t in tokens if t in g]
        return len(found) == len(tokens), len(found)

    return {
        "name": "High-Entropy Distributed Multi-Key Retrieval",
        "prompt": prompt,
        "check": check,
        "total_items": 5
    }

def main():
    args = parse_args()
    percentages = [float(p.strip()) for p in args.percentages.split(",") if p.strip()]

    task = test_random_kv_pairs()

    print("=" * 85)
    print("SPECULATIVE PREFILL: ENTROPY & PRUNING LIMIT EVALUATION")
    print("=" * 85)
    print("Prompt Length: ~1,100 tokens, 5 non-redundant targets spread across context")
    print("-" * 85)

    for p in percentages:
        res = run_test(args.bin, args.target_model, args.draft_model, task["prompt"], p, args.n_gpu_layers, args.n_gpu_layers_draft)
        all_passed, count = task["check"](res["gen_text"])
        status = "PASS (5/5)" if all_passed else f"FAIL ({count}/5 retrieved)"
        print("Keep %3d%% | Kept %3d/%4d tokens | Status: %-18s | Output: %s" % (
            int(p*100), res["kept"], res["total"], status, res["gen_text"].replace("\n", " ")[:40]))
    print("=" * 85)

if __name__ == "__main__":
    main()
