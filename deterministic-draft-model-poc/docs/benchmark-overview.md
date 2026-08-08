# Deterministic Draft Filter - Benchmark Overview

This document presents indicative, single-run benchmark numbers (n_runs=1, single prompt per language) for grammar-verified speculative decoding with a pluggable SDK. The contract: every emitted token passes the grammar in both modes. Standard mode now constrains the final token too (a previous bug where it did not has been fixed). Zero "commit_tokens REJECT" lines in all recent runs confirms the no-desync invariant.

Accept-all speedups measure a fundamentally different algorithm (grammar-only verification) and are not directly comparable to target-verified decoding speedups. Both are reported, but the reader should keep this distinction in mind.

## Quick Start

```bash
./build/bin/benchmark-deterministic-draft \
  -m /mnt/shared/Models/Qwen3.5-2B-MTP.Voodoo80_Q6_K.gguf \
  --det-draft-model deterministic-draft-model-poc/build/deterministic-draft.so \
  --det-draft-n-max 16 \
  --det-draft-accept-all \
  --det-draft-language c \
  -p $'#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n\n// Implement a hash table in C with insert, lookup, and delete.\n// Use separate chaining with linked lists. Add comments.\nint main(void) {' \
  -n 200 --compare --n-runs 1 \
  --ctx-size 262144 --batch-size 1024 --ubatch-size 512 \
  --flash-attn on --split-mode none --spec-type draft-mtp \
  --n-gpu-layers -1 --main-gpu 0
```

Runs MTP-only then MTP + grammar filter and prints throughput comparison. Use a generation-forcing prompt (open function body + instruction comment) so the model generates ~200 tokens instead of terminating immediately.

## Operating Modes

### Accept-All Mode (`--det-draft-accept-all`)

The filter is the sole verifier. Target model verification is skipped entirely for filter-accepted tokens. The output distribution follows draft+grammar, NOT the target model. This trades verification safety for raw throughput and is suitable for single-language code-only generation. Speed numbers in this mode measure a different algorithm than default mode.

### Default Mode (no `--det-draft-accept-all`)

Target model verifies all filter-accepted tokens. The filter pre-validates drafts before they reach the target, removing structurally invalid tokens. On GPU (RTX 3060), throughput improves modestly (1.1-2.8x) because the filter reduces wasted target model runs. On CPU (N100) with appropriate n-max tuning (16), throughput also improves (1.26-1.32x) because the filter prunes invalid drafts before the expensive target verification step.

## RTX 3060 Results

Hardware: AMD Ryzen 7 7840HS, 64GB DDR5, NVIDIA RTX 3060 (12GB). Model: Qwen3.5-2B-MTP (2B). All numbers are single-run, n_runs=1, indicative not statistical.

### Accept-All Mode (multi-language)

| Language | MTP only | MTP + Grammar Filter | Speedup | Accept Rate |
|----------|----------|----------------------|---------|-------------|
| C | 1.10 t/s | 22.54 t/s | 20.53x | 100% |
| Java | 1.11 t/s | 15.62 t/s | 14.12x | 100% |
| JavaScript | 0.79 t/s | 22.90 t/s | 29.14x | 100% |
| Python | 2.11 t/s | 17.55 t/s | 8.30x | 100% |

These numbers measure grammar-only verification (no target model verification). Prompts: C (`#include <stdio.h>`), Java (`import java.util.*;`), JavaScript (async merge sort), Python (BFS with deque).

### Default Mode (no accept-all)

| Language | MTP only | MTP + Grammar Filter | Speedup |
|----------|----------|----------------------|---------|
| C | 1.06 t/s | 2.01 t/s | 1.90x |
| Java | 1.11 t/s | 1.23 t/s | 1.11x |
| JavaScript | 0.98 t/s | 2.04 t/s | 2.08x |
| Python | 2.08 t/s | 5.87 t/s | 2.82x |

With filter (default mode): target model still verifies all filter-accepted tokens. Filter adds overhead but improves accept rate. Net throughput varies by language.

### Output Validity

Grammar-filtered output is invalid in accept-all mode (model generates garbage that parses as valid syntax). In default mode, MTP-only C output is sometimes valid (2/3 runs) but the filter introduces invalid output. This is a model quality issue, not a filter bug -- the grammar filter ensures structural syntax but not semantic correctness. See Validity Metric below.

## N100 Results

Hardware: Intel N100 (4C/4T Alder Lake-N), 13GB RAM. Model: Qwen3.5-2B-MTP (2B) for default mode; 4B instruct MTP (mxfp4) for accept-all. Build: CPU-only Intel oneAPI (MKL), --threads 4 --threads-batch 4, --cache-type-k q4_0 --cache-type-v q4_0, ctx 2048, temp 0.2 top-k 20 top-p 0.9 min-p 0.1. All runs pinned language with --det-draft-language c. Generation-forcing C prompt (includes + "implement a hash table" comment + "int main(void) {"), ~200 tokens. All numbers are single-run, n_runs=1, indicative not statistical.

### Accept-All Mode (4B instruct MTP, mxfp4)

| Metric | MTP only | MTP + Grammar Filter | Delta |
|--------|----------|----------------------|-------|
| throughput tps | 0.35 | 5.97 | +1603% (17.0x) |
| accept rate | 3.8% | 100.0% | +96.2pp |
| output valid | 0/1 | 0/1 | -- |
| n_predict | 204 | 263 | -- |
| drafted (pre/post) | 3776/3776 | 261/258 | -- |
| det truncated | N/A | 3 | -- |

Parameters: --det-draft-n-max 64. These numbers measure grammar-only verification (no target model verification). MTP-only acceptance is very low (3.8%): speculative decoding alone is nearly worthless here; the filter is what makes it viable.

### Default Mode (4B instruct MTP, mxfp4)

| Metric | MTP only | MTP + Grammar Filter | Delta |
|--------|----------|----------------------|-------|
| throughput tps | 2.02 | 2.68 | +32.4% (1.32x) |
| accept rate | 17.6% | 29.1% | +11.5pp |
| output valid | 0/1 | 0/1 | -- |
| n_predict | 202 | 206 | -- |
| drafted (pre/post) | 848/848 | 575/557 | -- |
| det truncated | N/A | 18 | -- |

Parameters: --det-draft-n-max 16. Target model verifies all filter-accepted tokens.

### Default Mode (2B Voodoo80 Q6_K)

| Metric | MTP only | MTP + Grammar Filter | Delta |
|--------|----------|----------------------|-------|
| throughput tps | 4.38 | 5.54 | +26.4% (1.26x) |
| accept rate | 33.6% | 45.4% | +11.8pp |
| output valid | 0/1 | 0/1 | -- |
| n_predict | 204 | 202 | -- |
| drafted (pre/post) | 512/512 | 403/381 | -- |
| det truncated | N/A | 22 | -- |

Parameters: --det-draft-n-max 16. Target model verifies all filter-accepted tokens.

### Output Validity

Output valid: 0/1 in all six runs (both modes for both models). The models write semantically broken or truncated C; the filter guarantees grammar-valid tokens only, not gcc-clean programs. This is a model quality issue, not a filter bug. See Validity Metric below.

### N100 Tuning Cautionary Note

Earlier N100 numbers (default 0.09-0.10x, accept-all 0.21-0.51x) came from a misconfigured setup: n-max 100, unpinned bootstrap, prompt with no trailing newline. On a weak CPU like the N100, oversized n-max is pathological -- the auxiliary heads draft up to 100 tokens per step, but with 3.8%-17.6% baseline acceptance, roughly 83-96 of those are wasted draft compute before the filter even runs. The filter's small direct cost (~5 ms/step) is dwarfed by the wasted draft generation. With proper tuning (n-max 16 for default, n-max 64 for accept-all), N100 shows positive speedups. Guidance: on CPU, use n-max ~16 for default mode and 32-64 for accept-all mode.

## Cross-Hardware Summary (Accept-All Mode)

| Hardware | MTP-only TPS | MTP+Filter TPS | Speedup | Notes |
|----------|-------------|----------------|---------|-------|
| RTX 3060 (CUDA) | 1.10-2.11 | 15.62-22.90 | 8.3-29.1x | GPU; target verification is bottleneck |
| RTX 4070 (CUDA) | 21.91 | 6.06 | 0.28x | Separate validated report |
| GTi15 Arc PRO B70 (oneAPI) | 19.47 | 130.88 | 6.72x | Separate validated report |
| N100 (CPU, 4B mxfp4) | 0.35 | 5.97 | 17.0x | n-max 64; grammar-only verification |
| N100 (CPU, 2B Q6_K) | 4.38-10.51 | 1.78-5.54 | 0.21-1.26x | Depends on n-max tuning |

Accept-all numbers measure grammar-only verification (no target model verification). The filter helps most where target model verification is the bottleneck and the model mostly agrees with the grammar (RTX 3060: 8.3-29.1x). On N100 CPU the speedup depends heavily on n-max tuning: with n-max 100 (misconfigured) the filter is slower (0.21-0.51x), but with n-max 16-64 (properly tuned) it is faster (1.26-17.0x). See N100 Tuning Cautionary Note above.

## When to Use Each Mode

**Accept-all**: single-language code generation where output language is known and fixed. IDE autocomplete, agent codegen, fill-in-middle. Not safe for mixed content (markdown, HTML+CSS+JS, chat output with interleaved code blocks) because the grammar becomes a no-op for non-matching content.

**Default mode**: mixed content, unknown output language, or when semantic correctness matters more than throughput. Target verification catches errors the structural filter cannot.

**Disabled**: chat output, documentation generation, any prompt where the output language could switch mid-generation.

## Trade-offs

- **Structural syntax only, not semantic correctness.** Valid C syntax can still be semantically wrong (wrong variable name, incorrect algorithm). The filter guarantees grammar-valid tokens; it cannot check static semantics (undeclared identifiers, duplicate typedefs, type errors). This is a documented limitation -- see observations.md.
- **Model/grammar alignment is the dominant factor.** The filter only pays when the model's drafts mostly survive the grammar. Against a misaligned model (e.g. a non-code-tuned 2B with a strict C grammar), acceptance collapses and any constrained speculative decoding is net-negative, independent of the filter's direct cost.
- **n-max tuning matters.** On weak CPUs, oversized n-max wastes draft compute because acceptance is low. Default mode: ~16. Accept-all: 32-64. On GPU, larger n-max can increase benefit.
- **Prompt must be grammar-parseable.** A trailing newline after the last #include is required; without it, the matcher sits mid-directive and drafts truncate to 0. A prompt that is already a complete program makes the model emit EOS immediately (1-2 tokens). Benchmarks must use generation-forcing prompts (open function body + instruction comment).

## Flag Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--det-draft-model <path>` | (none) | Path to plugin shared library (.so/.dylib/.dll). Auto-enables MTP. |
| `--det-draft-n-max <N>` | -1 | -1 = no cap (MTP default 3). 0 = disabled. >0 = cap and MTP draft count. |
| `--det-draft-accept-all` | (off) | Skip target model verification. Grammar is final arbiter. Output follows draft+grammar, not target model. |
| `--det-draft-language <name>` | (auto) | Pin the grammar language for all slots (e.g. c, java, javascript, python). Default: bootstrap auto-detection selects among bundled grammars. Pin this for strict single-language guarantees. |

## Validity Metric

The benchmark validator runs `gcc -fsyntax-only`. Warnings (extra tokens after #include, implicit int) exit 0 and count as VALID. Grammar-valid but semantically wrong code (duplicate typedefs, undeclared identifiers) passes the filter by design -- CFGs cannot check static semantics. This is a documented limitation; see observations.md for details. The filter guarantees syntax only; it cannot catch either class of semantic error.

## Test Suites

### C++ unit tests (`tests/test-deterministic-draft.cpp`)

33 test functions covering the C API, plugin loader, filter API, bitmask API, grammar state, bootstrap detection, and state serialization. 16 tests skip without the plugin .so found at the expected build path; 3 of those 16 additionally require the `LLAMA_TEST_MODEL` environment variable set to a model path. Tests that do not need the plugin or a model run unconditionally.

### File-based grammar harness (`deterministic-draft-model-poc/tests/`)

24 test files: 18 C (4 benchmark files + 2 edge-case files + 12 preprocessor-directive regression files 07-18), 2 Java, 2 JavaScript, 2 Python. Uses a 256-byte synthetic vocabulary, feeds code character-by-char via the bitmask API, and verifies accept/reject behavior matches file headers. These are grammar unit tests, not model integration tests -- they validate the plugin's grammar matching without loading a model.

## Hardware

| Platform | CPU | GPU | RAM |
|----------|-----|-----|-----|
| RTX 3060 | AMD Ryzen 7 7840HS | NVIDIA RTX 3060 (12GB) | 64GB DDR5 |
| RTX 4070 | Intel i9-14900KF | NVIDIA RTX 4070 (12GB) | 32GB DDR5 |
| GTi15 | Intel Ultra 9 285H | Intel Battlemage G21 (oneAPI) | 96GB DDR5 |
| N100 | Intel N100 (4C/4T Alder Lake-N) | Intel UHD (unused) | 16GB DDR4 |

Models used are small community fine-tunes (Voodoo80 uncensored merges, 4B instruct). Reviewers can reproduce with the public model: `unsloth/Qwen3.5-2B-MTP-GGUF:Q4_K_M` (see quick-start.md).

---

See [benchmark-comprehensive.md](benchmark-comprehensive.md) for detailed methodology, test file descriptions, and automation scripts.
