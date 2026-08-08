## Preliminary Important Disclosures

### 1. Bench Tool Position Bug (Fixed)

The benchmark tool had a position accounting bug where the prefill decoded all prompt tokens (including the last), then the generation loop re-decoded the final token at the same position. This caused position collisions affecting all models, not just M-RoPE. The fix matches the prefill pattern used in `speculative-simple.cpp`. All models -- including M-RoPE models like Qwen3.5/3.6 -- are compatible with the deterministic draft filter.

### 2. Filter Functionality Is Restored

The root cause of the earlier 100% draft rejection (a trailing-edge boundary bug in the grammar-constrained decoder) has been fixed. The filter now correctly validates drafts in the pipeline.

### 3. Validated Benchmarks Available

Validated, reproducible benchmarks are available in the comprehensive benchmark report. See [benchmark-overview.md](../../../deterministic-draft-model-poc/docs/benchmark-overview.md) for the full report.

---

## Why

LLM code generation suffers from structural invalidity (unmatched delimiters, syntax errors), expensive post-hoc retries, and cascading repair loops. GBNF grammar masking enforces syntax at the logit level but does not cover semantic style rules or provide deterministic error diagnostics.

A deterministic, pluggable gatekeeper hooks into the speculative decoding engine to intercept draft tokens. By validating drafts via external domain logic before they reach the main model, it eliminates wasteful target model forward passes on known-invalid sequences. This allows the pipeline to scale efficiently to high draft counts ($n_{max} \ge 20$) without stalling throughput, enabling high-performance inference on consumer and low-resource hardware.

## What Changes

The deterministic draft filter provides a **pluggable code-structure validation layer** for speculative decoding. Users add three CLI flags to any llama.cpp binary (the loader is always compiled into libllama; `-DDETERMINISTIC_SPEC_ENABLED=ON` additionally builds the standalone SDK artifacts and the benchmark tool):

1. `--deterministic-draft-model <plugin.so>` -- loads a validation plugin that checks each draft token against a language grammar
2. `--deterministic-draft-n-max <N>` -- sets how many draft tokens the filter evaluates per iteration
3. `--det-draft-accept-all` -- (optional) skips target model verification for filter-accepted tokens, providing 10-28x speedup for code-only generation

The filter intercepts MTP-generated draft tokens before they reach the target model. It runs a batch structural validation sweep via the plugin's shared library, truncates at the first structural error, and passes only the valid prefix to the target model (or directly to output in accept-all mode).

**Auto-configuration**: `--deterministic-draft-model` automatically enables MTP mode (`--spec-type draft-mtp`). If the model lacks MTP heads, llama.cpp fails to start with a clear error. When `--det-draft-n-max` is set > 0, it also sets the MTP draft count -- no need to set `--spec-draft-n-max` separately.

**Plugin architecture**: The plugin is an external shared library (.so/.dylib/.dll) loaded at runtime via dlopen. It implements a simple C API (create, validate, commit, reset) and lives in its own repository -- no llama.cpp source or build system needed. The PoC implements an XGrammar-based grammar-constrained decoder for C, Java, Python, and JavaScript.

## Capabilities

### New Capabilities

- `deterministic-draft`: Pluggable validation gate. Intercepts speculative/MTP draft tokens, runs a batch validation sweep via an external shared library, truncates at the first structural error, and reports error diagnostics.

### Modified Capabilities

None -- no existing specification-level behaviors are altered.

### Future Directions

The current filter is deliberately strict and single-domain (see "Known limits" in [phase3-roadmap.md](phase3-roadmap.md)). Phase 3 candidates - a chain-of-responsibility filter pipeline with transform semantics, and a threshold-gated pass-through for intentional domain switches - are sketched in [phase3-roadmap.md](phase3-roadmap.md). Both are additive to the capability-based plugin contract shipped here; nothing in this PR is blocked on or changed by them.

## Implementation Impact

- **Code (main tree, minimal changes)**: `common/speculative.{h,cpp}`, `common/common.h`, `common/arg.cpp` (New spec type and generic filter pipeline integration). `src/llama-deterministic-draft-serviceloader.cpp` (ServiceLoader: generic dlopen/dlsym layer that resolves a provider's SPI methods at runtime; completely agnostic to the underlying validation logic). `include/llama_deterministic_draft.h` (consumer C API declarations). `include/deterministic_draft_plugin.h` + `include/deterministic_draft_capabilities.h` (SPI/SDK headers defining the plugin boundary). `tools/server/server-context.cpp` (Hook into the speculative server pipeline). `tools/deterministic-draft-bench/` (Dedicated benchmark tool).
- **External artifacts**: `external/` directory (gated by `DETERMINISTIC_SPEC_ENABLED`) produces `external/include/deterministic_draft_plugin.h` and `external/lib/libdeterministic_draft_spec.so` (the ServiceLoader built as a standalone library, from the same source compiled into libllama), introducing zero external dependencies to the core llama binary.
- **Reference Implementation**: `deterministic-draft-model-poc/` - A standalone proof-of-concept project with zero main tree dependencies. It implements an XGrammar grammar-constrained decoder plugin, serving as a reference implementation for syntax-aware code validation.
- **Build**: Main tree builds with `cmake -B build -DDETERMINISTIC_SPEC_ENABLED=ON`. Reference plugins build independently. Plugins are linked at runtime via `--deterministic-draft-model <path_to_so>`.

## Benchmarks & Performance

> **2026-07-18 update**: the accept-all numbers in this section below are **Phase 1** and are now known to be misleading - on RTX 4070 the accept-all treatment actually produced INVALID C and was slower than baseline (the speedups shown come from mismatched/short prompts and an unvalidated correctness path). Phase 2 (2026-07-18, RTX 3060, Qwen3.5-9B) traced and fixed the root causes: accept-all is now gcc-valid AND faster than baseline (2.12x at n_max 16), and the filter caught 3/10 runs where the raw model produced invalid C. See the **Phase 2** section of [benchmark-overview.md](../../../deterministic-draft-model-poc/docs/benchmark-overview.md) for current, validated results; the Phase 1 tables below are kept for historical context only.

Validated, reproducible benchmarks are available in **two forms**:

1. **[Benchmark Overview](../../../deterministic-draft-model-poc/docs/benchmark-overview.md)** -- Key results, mode comparison, hardware comparison, Phase 2 (current) results
2. **[Comprehensive Benchmark](../../../deterministic-draft-model-poc/docs/benchmark-comprehensive.md)** -- Full methodology, multi-language results matrix, automation script, Phase 2 (current) results

### Key Results (Accept-All Mode, n_max=100) - Phase 1 (superseded, see note above)

With `--det-draft-n-max 100 --det-draft-accept-all` on a C/QuickSort prompt (20 lines of real code), the deterministic filter achieves substantial speedups on both GPU and low-power CPU hardware:

| Hardware | Prompt | Baseline (MTP) tps | Treatment tps | Speedup |
|----------|--------|-------------------|--------------|---------|
| GPU (CUDA, RTX 4070) | C quicksort (20 lines) | 12.39 | 116.28 | **9.38x** |
| CPU (N100) | `int fib(int n) {` (1 line) | 0.12 | 3.30 | **28.1x** |

> **Note**: The two hardware platforms use different prompts (20-line code prompt vs single-line). See the comprehensive benchmark for full cross-platform comparison with matched prompts.

**Key observations**:

1. **GPU (RTX 4070): 9.38x speedup** -- With accept-all, large drafts (n_max=100) generate massive amounts of invalid code that the filter eliminates before any GPU forward pass. The near-10x improvement is significant even though GPU verification costs are low per token.

2. **CPU (N100): 28.1x speedup** -- On the Intel N100 edge device (Alder Lake-N, 4 E-cores, ~6W TDP), the same configuration raises throughput from unusable (0.12 tps) to interactive (3.30 tps). Baseline accept rate of 0.4% is transformed to 100% treatment accept rate, with 98.6% reduction in drafted tokens.

3. **The accept-all flag** (`--det-draft-accept-all`) skips target model verification entirely, relying on structural validation as the final arbiter. This is suitable for single-language code-only generation (not mixed markdown/chat output).

See `deterministic-draft-model-poc/docs/benchmark-overview.md` for the full report. Raw benchmark outputs are saved in `deterministic-draft-model-poc/docs/raw/`.
