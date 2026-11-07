#!/usr/bin/env python3
'''
  Checks if a text carries the SynthID watermark produced by llama.cpp with --synthid-keys.
  The g-values are computed with the SynthIDTextWatermarkLogitsProcessor from HF transformers,
  the score is the mean g-value: about 50% for unwatermarked text, higher for watermarked text.

  Syntax:
    ./scripts/synthid-check.py -f output.txt --keys 654,400,836 --tokenizer <HF model id or path>

  The tokenizer must be the one of the model that generated the text.
  The text should contain only the generated part, without the prompt.
'''

import argparse
import math
import sys

import torch
from transformers import AutoTokenizer
from transformers.generation.logits_process import SynthIDTextWatermarkLogitsProcessor


def score_tokens(token_ids, keys, ngram_len=5, sampling_table_size=65536, sampling_table_seed=0, context_history_size=1024):
    processor = SynthIDTextWatermarkLogitsProcessor(
        ngram_len=ngram_len,
        keys=keys,
        sampling_table_size=sampling_table_size,
        sampling_table_seed=sampling_table_seed,
        context_history_size=context_history_size,
        device=torch.device("cpu"),
    )

    ids = torch.tensor([token_ids], dtype=torch.long)
    g_values = processor.compute_g_values(ids)                       # (1, n_ngrams, depth)
    mask = processor.compute_context_repetition_mask(ids).float()    # (1, n_ngrams)

    depth = g_values.shape[-1]
    n_scored = int(mask.sum().item())
    if n_scored == 0:
        raise ValueError("text is too short to score")

    mean = (g_values.float() * mask[..., None]).sum().item() / (depth * n_scored)

    # unwatermarked text gives i.i.d. Bernoulli(0.5) g-values
    z_score = (mean - 0.5) / (0.5 / math.sqrt(depth * n_scored))

    return mean, z_score, n_scored


def main():
    parser = argparse.ArgumentParser(description="check a text for the SynthID watermark")
    parser.add_argument("-f", "--file", required=True, help="text file to check")
    parser.add_argument("--keys", required=True, help="comma-separated watermarking keys, same as --synthid-keys")
    parser.add_argument("--tokenizer", required=True, help="HF model id or local path of the tokenizer")
    parser.add_argument("--ngram-len", type=int, default=5)
    args = parser.parse_args()

    keys = [int(k) for k in args.keys.split(",") if k.strip()]
    if not keys:
        sys.exit("error: --keys requires at least one key")

    with open(args.file, encoding="utf-8") as f:
        text = f.read()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) < args.ngram_len:
        sys.exit(f"error: text has {len(token_ids)} tokens, need at least {args.ngram_len}")

    mean, z_score, n_scored = score_tokens(token_ids, keys, ngram_len=args.ngram_len)

    print(f"tokens: {len(token_ids)}, scored ngrams: {n_scored}")
    print(f"watermark score: {100.0 * mean:.1f}% (unwatermarked text scores about 50%)")
    print(f"z-score: {z_score:.2f}")


if __name__ == "__main__":
    main()
