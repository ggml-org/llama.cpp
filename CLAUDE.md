# llama.cpp-cpuopti — Deterministic Runtime Optimization Fork

This is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that adds a deterministic runtime optimization layer for LLM inference. It targets coding agents and structured-output workloads.

## Project Overview

The optimization layer sits between the llama.cpp frontend and the GGML execution backend. All optimizations are applied at runtime during inference — no retraining, no weight modification, no approximation (Tier 1).

**Target workload:** Coding agents with long sessions, large contexts, repetitive tool schemas, and JSON tool calls.

## Architecture

```
src/llama-opt.h/cpp             — Feature flags, configuration, optimization registry
src/llama-context-hash.h/cpp    — Deterministic block hashing for token sequences
src/llama-kv-cache-dedup.h/cpp  — Context block deduplication (KV reuse for repeated blocks)
src/llama-kv-cache-diff.h/cpp   — Structural KV cache diffing (incremental prefill)
src/llama-schema-skip.h/cpp     — Schema-aware token skipping (grammar-driven fast-forward)
```

Integration points in upstream code:
- `src/llama-kv-cache.cpp` — KV slot allocation hooks for dedup and diffing
- `src/llama-context.cpp` — Graph scheduling hooks for the optimization layer
- `src/llama-grammar.cpp` — Grammar constraint queries for schema-aware skipping

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Feature Flags (CMake options)

| Option | Default | Description |
|---|---|---|
| `LLAMA_OPT_DEDUP` | ON | Context block deduplication (Tier 1 — Exact) |
| `LLAMA_OPT_KV_DIFF` | ON | Structural KV cache diffing (Tier 1 — Exact) |
| `LLAMA_OPT_SCHEMA_SKIP` | ON | Schema-aware token skipping (Tier 1 — Exact) |
| `LLAMA_OPT_PRECOMPUTE` | ON | Deterministic precomputation caching (Tier 1 — Exact) |
| `LLAMA_OPT_ATTN_PRUNE` | OFF | Attention sink pruning (Tier 2 — Approximate) |

To disable all optimizations: `cmake .. -DLLAMA_OPT_DEDUP=OFF -DLLAMA_OPT_KV_DIFF=OFF -DLLAMA_OPT_SCHEMA_SKIP=OFF`

### Runtime Flags (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `LLAMA_OPT_BLOCK_SIZE` | 64 | Token block size for context hashing |
| `LLAMA_OPT_DEDUP_ENABLED` | 1 | Enable/disable dedup at runtime |
| `LLAMA_OPT_DIFF_ENABLED` | 1 | Enable/disable KV diffing at runtime |
| `LLAMA_OPT_SCHEMA_SKIP_ENABLED` | 1 | Enable/disable schema-aware skipping at runtime |
| `LLAMA_OPT_STATS` | 0 | Print optimization statistics per turn |

## Optimization Tiers

- **Tier 1 (Exact):** Bitwise identical outputs vs baseline. On by default.
- **Tier 2 (Approximate):** Output may differ in irrelevant ways. Off by default, opt-in only.

## Hard Constraints

- No retraining or fine-tuning
- No model weight modification
- Deterministic execution (Tier 1)
- Safe fallback to baseline at all times
- Worst-case performance identical to upstream llama.cpp

## Code Conventions

- C++17 (matching upstream llama.cpp)
- Prefix optimization types with `llama_opt_`
- All new files in `src/` prefixed with `llama-opt-` or `llama-kv-cache-` or `llama-schema-`
- Every optimization must be independently feature-flagged
- Use `LLAMA_LOG_INFO` / `LLAMA_LOG_WARN` for diagnostics, never `printf`
- Hash functions must be deterministic across platforms (use FNV-1a or xxHash, not `std::hash`)
- All public types go through `include/llama.h`

## Testing

```bash
cd build
ctest --output-on-failure
```

Optimization-specific tests:
```bash
./bin/test-opt-context-hash     # Block hashing correctness
./bin/test-opt-kv-dedup         # KV dedup correctness and cache hit verification
./bin/test-opt-kv-diff          # KV diff correctness
./bin/test-opt-schema-skip      # Schema-aware skipping correctness
```

## Correctness Validation

Tier 1 optimizations must pass:
- Bitwise comparison of logits vs baseline (optimization disabled)
- Token-by-token output equivalence for deterministic prompts
- Identical results for repeated identical prompts

## Key Files Reference

| File | Purpose |
|---|---|
| `src/llama-kv-cache.h` | KV cache interface — slot allocation, sequence ops |
| `src/llama-kv-cells.h` | Cell storage structure for KV entries |
| `src/llama-memory.h` | Abstract memory interface (`llama_memory_i`) |
| `src/llama-context.h` | Inference context — graph scheduling, batch processing |
| `src/llama-graph.h` | GGML computation graph construction |
| `src/llama-grammar.h` | Grammar constraint parsing and application |
| `src/llama-batch.h` | Token batching and micro-batch handling |
| `ggml/src/ggml.c` | Core tensor operations |
| `ggml/src/ggml-cpu/ops.cpp` | CPU compute kernels (matmul, attention, etc.) |

## Development Workflow

1. All optimizations are modular — one optimization per file pair (`.h` + `.cpp`)
2. Each optimization has its own compile-time flag and runtime toggle
3. Changes to upstream files must be minimal and clearly marked with `// CPUOPTI:` comments
4. Profile before committing — measure overhead cost of hashing/caching/diffing
5. Run correctness tests before and after to verify exact output preservation
