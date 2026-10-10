# Radix / prefix-cache benchmarks

Related upstream context (search before opening a discussion):

- Host-memory prompt cache: https://github.com/ggml-org/llama.cpp/pull/16391
- Dual-session prefix thrash: https://github.com/ggml-org/llama.cpp/issues/20510
- Unified KV prefix alias notes: https://github.com/ggml-org/llama.cpp/discussions/21961
- Busy-slot prefix share: https://github.com/ggml-org/llama.cpp/issues/27616

Paged KV / full PagedAttention is **out of scope** for the first radix PR (capacity follow-up only).

## Harness

```bash
# server A: baseline (no radix)
llama-server -m model.gguf -c 8192 -np 4 --kv-unified --cache-ram 8192 --no-radix-cache

python tools/server/bench/radix-prefix/radix_prefix_bench.py \
  --url http://127.0.0.1:8080 --workload all --label no-radix --out base.json

# server B: radix on
llama-server -m model.gguf -c 8192 -np 4 --kv-unified --cache-ram 8192 --radix-cache

python tools/server/bench/radix-prefix/radix_prefix_bench.py \
  --url http://127.0.0.1:8080 --workload all --label radix --out radix.json
```

Compare `req_per_s`, `prompt_tokens_computed`, `cache_hit_frac`, and dualagent `after_first_pair`.

## Gates (from design)

| Workload | Phase 1 accept |
|----------|----------------|
| gsp | >= 2x req/s or <= 0.55x computed prompt tokens vs baseline |
| dualagent | after_first_pair TTFT proxy <= 0.3x thrash baseline |
| unique | regression < 3% |
| multiturn | must not regress vs cache_prompt today |
