#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
import time

def parse_args():
    default_bin = "./build-vulkan/bin/llama-speculative-prefill" if os.path.exists("./build-vulkan/bin/llama-speculative-prefill") else "./build/bin/llama-speculative-prefill"

    parser = argparse.ArgumentParser(description="Evaluate Speculative Prefill Quality Impact (Needle-in-a-Haystack & QA)")
    parser.add_argument("-m", "--target-model", required=True, help="Path to target model GGUF")
    parser.add_argument("-mpd", "-md", "--draft-model", "--spec-prefill-model", help="Path to prefill draft model GGUF")
    parser.add_argument("-ngl", "--n-gpu-layers", type=int, default=99, help="GPU layers for target model")
    parser.add_argument("-nglpd", "-ngld", "--n-gpu-layers-draft", "--spec-prefill-ngl", type=int, default=99, help="GPU layers for draft model")
    parser.add_argument("--percentages", default="1.0,0.50,0.30,0.20,0.15", help="Comma-separated keep percentages")
    parser.add_argument("--chunk-size", type=int, default=32, help="Chunk size (default: 32)")
    parser.add_argument("--lookahead", type=int, default=4, help="Lookahead count (default: 4)")
    parser.add_argument("--context-size", type=int, default=1500, help="Approximate prompt context size in words")
    parser.add_argument("--bin", default=default_bin, help="Path to binary")
    return parser.parse_args()

def generate_haystack(target_words):
    paragraphs = [
        "In the quiet valleys of the northern province, astronomers built observatories to chart the movement of celestial bodies. "
        "Every evening, scholars logged coordinates of distant nebulae and recorded fluctuations in stellar brightness with brass instruments. "
        "The archive of records grew into hundreds of bound volumes filled with geometrical proofs and astronomical charts.",
        
        "Trade routes crisscrossed the continent, carrying spices, textiles, and precious metals between bustling harbor cities and inland markets. "
        "Caravans traveled along well-guarded mountain passes, stopping at desert oases to trade horses and exchange news of foreign realms. "
        "Merchant guilds maintained detailed ledgers of commerce, recording tariffs and grain prices across maritime hubs.",
        
        "Architects designed aqueducts and arched bridges to bring clean mountain water into the centers of expanding metropolises. "
        "Engineers perfected the composition of volcanic mortar, enabling the construction of domes and bathhouses that endured for centuries. "
        "Civic planners organized city blocks into grids surrounding central plazas, facilitating administration and commerce.",
        
        "The great library of Alexandria preserved parchment rolls of ancient philosophy, drama, geometry, and medicinal botany. "
        "Translators worked continuously to transcribe foreign manuscripts into standardized scripts, adding scholarly annotations. "
        "Philosophers met under columned porticos to debate the nature of knowledge, virtue, and physical cosmology.",
    ]
    
    text = []
    total = 0
    while total < target_words:
        for p in paragraphs:
            text.append(p)
            total += len(p.split())
            if total >= target_words:
                break
    return text

def run_test(bin_path, target_model, draft_model, prompt, p, chunk_size, lookahead, ngl, ngld):
    cmd = [
        bin_path,
        "-m", target_model,
        "-md", draft_model,
        "-ngl", str(ngl),
        "-ngld", str(ngld),
        "-p", prompt,
        "-n", "32",
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

def main():
    args = parse_args()
    draft_model = args.draft_model if args.draft_model else args.target_model
    percentages = [float(p.strip()) for p in args.percentages.split(",") if p.strip()]

    needle_secret = "DELTA-7492-OMEGA"
    needle_sentence = "IMPORTANT NOTE: The secret project verification passcode is DELTA-7492-OMEGA."
    
    depths = [0.20, 0.50, 0.80] # 20% (early), 50% (middle), 80% (late in document)
    
    print("=" * 85)
    print("SPECULATIVE PREFILL QUALITY EVALUATION: NEEDLE-IN-A-HAYSTACK (NIAH)")
    print("=" * 85)
    print("Target Model : %s" % args.target_model)
    print("Draft Model  : %s" % draft_model)
    print("Target Needle: \"%s\"" % needle_secret)
    print("Depths Tested: %s" % [("%d%%" % int(d*100)) for d in depths])
    print("-" * 85)

    haystack_paras = generate_haystack(args.context_size)

    results_matrix = {} # p -> depth -> dict

    for p in percentages:
        results_matrix[p] = {}
        print("\nEvaluating Keep Percentage: %d%%..." % int(p*100))
        
        for depth in depths:
            insert_idx = int(len(haystack_paras) * depth)
            test_paras = list(haystack_paras)
            test_paras.insert(insert_idx, needle_sentence)
            full_context = "\n\n".join(test_paras)
            
            prompt = (
                "%s\n\n"
                "Question: What is the secret project verification passcode mentioned in the text above?\n"
                "Answer: The secret project verification passcode is"
            ) % full_context

            res = run_test(args.bin, args.target_model, draft_model, prompt, p, args.chunk_size, args.lookahead, args.n_gpu_layers, args.n_gpu_layers_draft)
            
            passed = ("DELTA" in res["gen_text"]) or ("7492" in res["gen_text"]) or ("OMEGA" in res["gen_text"])
            results_matrix[p][depth] = {
                "passed": passed,
                "gen": res["gen_text"],
                "ttft": res["ttft_ms"],
                "kept": res["kept"],
                "total": res["total"]
            }
            status_str = "PASS [FOUND]" if passed else "FAIL [LOST]"
            print("  Depth %2d%%: %s (TTFT: %6.1f ms) | Kept: %d/%d | Gen: \"%s\"" % (
                int(depth*100), status_str, res["ttft_ms"], res["kept"], res["total"], res["gen_text"].replace("\n", " ")[:45]))

    print("\n" + "=" * 85)
    print("QUALITY ACCURACY SUMMARY TABLE (Needle Retrieval vs Keep Ratio)")
    print("=" * 85)
    depth_headers = " | ".join([("Depth %2d%%" % int(d*100)) for d in depths])
    print("Keep %%   | Tokens Kept | Avg TTFT   | %s | Accuracy" % depth_headers)
    print("-" * 85)

    for p in percentages:
        scores = [results_matrix[p][d]["passed"] for d in depths]
        avg_ttft = sum([results_matrix[p][d]["ttft"] for d in depths]) / len(depths)
        n_kept = results_matrix[p][depths[0]]["kept"]
        n_total = results_matrix[p][depths[0]]["total"]
        score_pct = (sum(scores) / len(scores)) * 100.0
        
        depth_results = " | ".join(["PASS" if s else "FAIL" for s in scores])
        cfg_name = "%d%%" % int(p*100)
        print("%-8s | %4d/%-4d | %7.1f ms | %-26s | %5.1f%%" % (
            cfg_name, n_kept, n_total, avg_ttft, depth_results, score_pct))

    print("=" * 85)

if __name__ == "__main__":
    main()
