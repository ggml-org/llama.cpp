#!/usr/bin/env python3
"""
Benchmark: TranslateGemma 4B + AlignAtt streaming policy (EN→ZH)
llama.cpp implementation — comparable to CTranslate2/benchmark/eval_benchmark_translategemma.py

Translates FLORES EN→ZH sentences using llama.cpp with alignment heads.
Computes latency metrics (AL, LAAL, AP, CW) and saves translations for quality scoring.

Usage:
    # First convert TranslateGemma to GGUF:
    #   python3 convert_hf_to_gguf.py /tmp/translategemma-text-only --outfile /tmp/translategemma.gguf

    python3 examples/attn-weights/eval_benchmark_translategemma.py /tmp/translategemma.gguf \\
        --heads examples/attn-weights/translation_heads_en_zh.json \\
        -n 100 --output examples/attn-weights/results_llamacpp.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import llama_attn as ll


# Copy heads JSON from CTranslate2 benchmark if not present locally
CT2_HEADS = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "CTranslate2", "benchmark", "translation_heads_en_zh.json"
)

PROMPT_TEMPLATE = (
    "<bos><start_of_turn>user\n"
    "You are a professional English (en) to Chinese (zh) translator. "
    "Your goal is to accurately convey the meaning and nuances of the "
    "original English text while adhering to Chinese grammar, vocabulary, "
    "and cultural sensitivities.\n"
    "Produce only the Chinese translation, without any additional "
    "explanations or commentary. Please translate the following English "
    "text into Chinese:\n\n\n"
    "{source}<end_of_turn>\n"
    "<start_of_turn>model\n"
)


def find_source_token_range(source_text, vocab):
    """Find token range of source text within the prompt."""
    prefix = PROMPT_TEMPLATE.split("{source}")[0]
    suffix = PROMPT_TEMPLATE.split("{source}")[1]
    prefix_ids = ll.tokenize(vocab, prefix, add_bos=False, special=True)
    full_ids = ll.tokenize(vocab, prefix + source_text + suffix, add_bos=False, special=True)
    suffix_ids = ll.tokenize(vocab, suffix, add_bos=False, special=True)
    return len(prefix_ids), len(full_ids) - len(suffix_ids)


def compute_metrics(alignments, num_src, num_tgt):
    """Compute AL, LAAL, AP, Max CW — same as CTranslate2 benchmark."""
    if not alignments or num_tgt == 0 or num_src == 0:
        return {"al": 0.0, "max_cw": 0, "laal": 0.0, "ap": 0.0}

    mono = []
    max_dep = 0
    for dep in alignments:
        max_dep = max(max_dep, dep)
        mono.append(max_dep)

    ratio = num_src / num_tgt
    total_lag = sum(max(0, (mono[t] + 1) - t * ratio) for t in range(num_tgt))
    al = total_lag / num_tgt

    tau = min(num_src / num_tgt, 1.0)
    laal = al * tau

    total_src_read = sum(mono[t] + 1 for t in range(num_tgt))
    ap = total_src_read / (num_src * num_tgt)

    max_cw = 0
    cw = 0
    prev_needed = 0
    for dep in mono:
        needed = dep + 1
        reads = max(0, needed - prev_needed)
        if reads > 0:
            cw += reads
            max_cw = max(max_cw, cw)
        else:
            cw = 0
        prev_needed = needed
        cw = 0

    return {"al": al, "max_cw": max_cw, "laal": laal, "ap": ap}


def aggregate_ts_weighted_vote(src_attn, ts_scores):
    """TS-weighted vote: weighted argmax across heads."""
    head_argmaxes = np.argmax(src_attn, axis=1)
    weighted = {}
    for h, pos in enumerate(head_argmaxes):
        pos = int(pos)
        weighted[pos] = weighted.get(pos, 0) + ts_scores[h]
    return max(weighted, key=weighted.get)


def aggregate_mean(src_attn):
    avg = src_attn.mean(axis=0)
    return int(np.argmax(avg))


def main():
    parser = argparse.ArgumentParser(description="TranslateGemma benchmark with llama.cpp attention")
    parser.add_argument("model", help="Path to GGUF model file")
    parser.add_argument("-n", type=int, default=100, help="Number of FLORES sentences")
    parser.add_argument("--heads", default=None, help="Alignment heads JSON (default: CTranslate2 benchmark)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top alignment heads")
    parser.add_argument("--strategy", default="ts_weighted_vote", choices=["ts_weighted_vote", "mean"])
    parser.add_argument("--output", default="benchmark/results_llamacpp.json")
    parser.add_argument("--n-ctx", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    # Load alignment heads
    heads_path = args.heads or CT2_HEADS
    if not os.path.exists(heads_path):
        # Try local copy
        local = os.path.join(os.path.dirname(__file__), "translation_heads_en_zh.json")
        if os.path.exists(local):
            heads_path = local
        else:
            print(f"ERROR: Cannot find alignment heads JSON at {heads_path}")
            print("Copy from CTranslate2/benchmark/translation_heads_en_zh.json")
            sys.exit(1)

    with open(heads_path) as f:
        data = json.load(f)

    top_heads = data["token_alignment_heads"][:args.top_k]
    head_layers = [h["layer"] for h in top_heads]
    head_indices = [h["head"] for h in top_heads]
    ts_scores = [h["ts"] for h in top_heads]
    num_heads = len(top_heads)

    print(f"Using {num_heads} alignment heads (strategy={args.strategy})")
    for h in top_heads[:5]:
        print(f"  L{h['layer']:2d} H{h['head']} (TS={h['ts']:.3f})")
    if num_heads > 5:
        print(f"  ... and {num_heads - 5} more")

    # Load FLORES EN→ZH
    print("\nLoading FLORES EN→ZH...")
    from datasets import load_dataset
    ds = load_dataset("openlanguagedata/flores_plus", split="dev")
    en_ds = ds.filter(lambda x: x["iso_639_3"] == "eng" and x["iso_15924"] == "Latn")
    zh_ds = ds.filter(lambda x: x["iso_639_3"] == "cmn" and x["iso_15924"] == "Hans")
    en_map = {row["id"]: row["text"] for row in en_ds}
    zh_map = {row["id"]: row["text"] for row in zh_ds}
    common_ids = sorted(set(en_map) & set(zh_map))
    n = min(args.n, len(common_ids))
    print(f"  {n} sentence pairs")

    # Initialize llama.cpp
    print(f"\nLoading model: {args.model}")
    ll.init()
    model = ll.load_model(args.model)
    vocab = ll.get_vocab(model)
    nv = ll.n_vocab(vocab)
    n_layers = ll.n_layer(model)
    eos_id = ll.vocab_eos(vocab)

    print(f"  {n_layers} layers, vocab={nv}")

    # Find stop token IDs
    stop_ids = set()
    for tok_str in ["<end_of_turn>", "<eos>"]:
        toks = ll.tokenize(vocab, tok_str, add_bos=False, special=True)
        if len(toks) == 1:
            stop_ids.add(toks[0])
    stop_ids.add(eos_id)
    print(f"  Stop IDs: {stop_ids}")

    # Evaluate
    results = []
    al_list, cw_list, laal_list, ap_list = [], [], [], []

    for idx in range(n):
        sid = common_ids[idx]
        source = en_map[sid]
        reference = zh_map[sid]

        prompt = PROMPT_TEMPLATE.format(source=source)
        prompt_tokens = ll.tokenize(vocab, prompt, add_bos=False, special=True)
        prompt_len = len(prompt_tokens)

        src_start, src_end = find_source_token_range(source, vocab)
        num_src = src_end - src_start

        # Create fresh context for each sentence (clean KV cache)
        ctx = ll.create_context(model, n_ctx=args.n_ctx, n_batch=args.n_ctx, attn_weights=True)
        ll.set_attn_heads(ctx, head_layers, head_indices)
        n_c = ll.n_ctx(ctx)

        t0 = time.time()

        # Prefill
        ret = ll.decode_batch(ctx, prompt_tokens, output_last_only=True)
        if ret != 0:
            print(f"  [{idx+1}] prefill failed: {ret}, skipping")
            ll.free_context(ctx)
            continue

        # Autoregressive generation with attention extraction
        generated_ids = []
        alignments = []
        pos = prompt_len

        for step in range(args.max_tokens):
            # Get attention for current token
            attn = ll.get_attn_weights(ctx, -1, num_heads, n_c)

            if attn is not None:
                # Extract attention over source token range only
                n_kv = ll._lib.llama_get_attn_n_kv(ctx)
                if src_start < n_kv and src_end <= n_kv:
                    src_attn = attn[:, src_start:src_end]  # (num_heads, num_src)

                    if args.strategy == "ts_weighted_vote":
                        aligned_pos = aggregate_ts_weighted_vote(src_attn, ts_scores)
                    else:
                        aligned_pos = aggregate_mean(src_attn)
                    alignments.append(aligned_pos)
                else:
                    alignments.append(num_src - 1)
            else:
                alignments.append(num_src - 1)

            # Get next token
            next_tok = ll.argmax_logits(ctx, -1, nv)

            if next_tok in stop_ids or next_tok < 0:
                break

            generated_ids.append(next_tok)

            # Decode next token
            ret = ll.decode_single(ctx, next_tok, pos, output=True)
            if ret != 0:
                break
            pos += 1

        gen_time = time.time() - t0
        ll.free_context(ctx)

        num_tgt = len(generated_ids)
        if num_tgt == 0:
            continue

        # Trim alignments to match generated tokens
        alignments = alignments[:num_tgt]

        # Decode translation text
        pieces = [ll.token_to_piece(vocab, tid) for tid in generated_ids]
        translation = "".join(pieces)

        metrics = compute_metrics(alignments, num_src, num_tgt)
        al_list.append(metrics["al"])
        cw_list.append(metrics["max_cw"])
        laal_list.append(metrics["laal"])
        ap_list.append(metrics["ap"])

        results.append({
            "id": int(sid),
            "source": source,
            "reference": reference,
            "translation": translation,
            "num_src_tokens": num_src,
            "num_tgt_tokens": num_tgt,
            "al": round(metrics["al"], 3),
            "max_cw": metrics["max_cw"],
            "laal": round(metrics["laal"], 3),
            "ap": round(metrics["ap"], 3),
            "gen_time_ms": round(gen_time * 1000),
        })

        if (idx + 1) % 10 == 0:
            avg_al = np.mean(al_list)
            print(f"  [{idx+1}/{n}] Avg AL={avg_al:.2f}, last: {translation[:50]}...", flush=True)

    # Summary
    if not results:
        print("No results!")
        ll.free_model(model)
        ll.cleanup()
        sys.exit(1)

    al_arr = np.array(al_list)
    cw_arr = np.array(cw_list)
    laal_arr = np.array(laal_list)
    ap_arr = np.array(ap_list)

    summary = {
        "system": "TranslateGemma-4B + AlignAtt (llama.cpp)",
        "language_pair": "en-zh",
        "num_sentences": len(results),
        "num_alignment_heads": num_heads,
        "strategy": args.strategy,
        "latency": {
            "avg_al": round(float(np.mean(al_arr)), 3),
            "median_al": round(float(np.median(al_arr)), 3),
            "p90_al": round(float(np.percentile(al_arr, 90)), 3),
            "avg_max_cw": round(float(np.mean(cw_arr)), 1),
            "max_max_cw": int(np.max(cw_arr)),
            "avg_laal": round(float(np.mean(laal_arr)), 3),
            "avg_ap": round(float(np.mean(ap_arr)), 3),
        },
        "avg_gen_time_ms": round(float(np.mean([r["gen_time_ms"] for r in results]))),
    }

    output = {"summary": summary, "sentences": results}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"TranslateGemma + AlignAtt (llama.cpp) — EN→ZH ({len(results)} sentences)")
    print(f"{'='*60}")
    print(f"  Avg AL:      {summary['latency']['avg_al']:.3f}")
    print(f"  Median AL:   {summary['latency']['median_al']:.3f}")
    print(f"  P90 AL:      {summary['latency']['p90_al']:.3f}")
    print(f"  Avg Max CW:  {summary['latency']['avg_max_cw']:.1f}")
    print(f"  Max CW:      {summary['latency']['max_max_cw']}")
    print(f"  Avg LAAL:    {summary['latency']['avg_laal']:.3f}")
    print(f"  Avg AP:      {summary['latency']['avg_ap']:.3f}")
    print(f"  Avg gen:     {summary['avg_gen_time_ms']}ms/sentence")
    print(f"\nSaved to {args.output}")

    ll.free_model(model)
    ll.cleanup()


if __name__ == "__main__":
    main()
