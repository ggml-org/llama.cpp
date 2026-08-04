# Wave 1 pre-flight facts (resolved by the orchestrator)

These are verified before wave 1 starts. The agent should read this and NOT
re-derive them.

## Build
- Baseline tree builds clean after a one-line fix to common/common.h:
  common_params now has `float otel_sample_rate = 1.0f;` (was referenced by
  server-context.cpp:1630 but missing from the struct). This fix is INCLUDED
  in baseline_sha.
- Build command: `cmake --build build --target llama-server llama-bench -- -j8`
  (~30s incremental; full build is longer).

## Model
- Path: ~/Library/Application Support/qwen2.5-0.5b-instruct-q4_k_m.gguf
- ~630M params, Q4_K_M, 462 MiB. Runs on Metal (Apple M1, ~1387 pp64 t/s,
  ~88 tg16 t/s). Small and fast - good for POC.

## Evaluator (verified working)
- Correctness gate: `cd build && ctest -R test-server-prompt-cache` plus a
  logit-diff check against baseline on a fixed 4-prompt set.
- Memory metric: peak RSS via /usr/bin/time -l wrapping llama-bench:
    /usr/bin/time -l ./build/bin/llama-bench -m <model> -p 512 -n 64
  Parse "maximum resident set size" from stderr.
- Baseline peak RSS (qwen 0.5B, pp512+n64, f16 KV): ~594 MB. The target
  metric is -peak_RSS (MAXIMIZE). A q8_0 KV reader should cut the KV portion.

## Workload (frozen for this wave)
- model: ~/Library/Application Support/qwen2.5-0.5b-instruct-q4_k_m.gguf
- bench: -p 512 -n 64 (enough tokens that KV is a meaningful fraction of RSS)
- correctness: ctest test-server-prompt-cache + logit-diff vs baseline

## What S1 actually means here
Wire q8_0 K/V through GGML_OP_TESSERA_PAGED_ATTN instead of f16. The kernel
currently asserts k->type == F32 || F16 at ggml-cpu/ops.cpp:9250 and
ggml-metal.metal:11240. The block-table boundary at src/llama-kv-cache.h:101
already exists. Smallest first step: make the CPU path accept q8_0 with
on-the-fly dequant, then mirror on Metal. ANE can be a follow-up if CPU+Metal
land.
