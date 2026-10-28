# Spark DSpark company-configuration reference

This directory builds a private runtime for the validated stochastic reference path. It leaves the normal repository build intact.

The Spark model adapter supports the Spark 4B-1026 target and its five-layer, all-SWA draft with the vanilla Markov head. Target feature inputs are layers 3, 11, 19, 26 and 34. Each feature is RMS-normalized before fusion; the draft uses GELU, headwise sigmoid gates and the SWA RoPE parameters.

The reference uses temperature 1, top-p 0.95 on target and draft, seven stochastic proposals plus a correction or bonus, BF16 KV, standard p/q rejection and normalized max(p-q, 0) correction. Set max_tokens to 2048. Confidence truncation is disabled. All draft queries share the fixed history interval beginning at max(0, first_query_position - 512).

The native llama-server draft chain remains greedy. This change does not add stochastic DSpark to the server. Server sampling flags alone do not reproduce this reference path.

Build the ordinary CUDA library first, then run:

```bash
python tools/spark-dspark-company/build_runtime.py
LD_LIBRARY_PATH=tools/spark-dspark-company/runtime:build/bin \
LLAMA_DSPARK_FIXED_HISTORY=1 \
tools/spark-dspark-company/spark_company_prob_bench --self-test
```

Input is JSONL with id, dataset, index and prompt_tokens fields. Preserve the company thinking/high prompt token IDs. The runtime is loaded before the normal library through LD_LIBRARY_PATH. Example:

```bash
LD_LIBRARY_PATH=tools/spark-dspark-company/runtime:build/bin \
LLAMA_DSPARK_FIXED_HISTORY=1 \
tools/spark-dspark-company/spark_company_prob_bench \
  /path/to/target-bf16.gguf /path/to/draft-bf16.gguf \
  /path/to/cases.jsonl /path/to/results.jsonl 2048 2026
```

Acceptance length is 1 + 7 * sum(accepted_tokens) / sum(draft_tokens). The benchmark also checks the fixed-window visible keys, including histories longer than 512 tokens. It uses CPU probability processing and is an acceptance reference, not the separate optimized CUDA throughput entry point.

The previously completed full acceptance experiment tested 80 cases from each of MATH, GSM8K, LBPP and MBPP with this model implementation and private window override. This export removes unrelated QSA changes and the unused RNN variant. Rebuild and rerun a short model check after applying to a newer base; earlier full-run results do not constitute a full regression run on that newer base.

Do not add model weights, datasets, generated binaries, caches or benchmark outputs to the commit.
