# Deterministic Draft Filter: Comprehensive Multi-Language Benchmark

**Status: XGrammar-based (bitmask-constrained decoding).** The plugin utilizes XGrammar's token-level bitmask constraint (`fill_bitmask`/`commit`), rather than post-hoc text validation. This is grammar-verified speculative decoding with a pluggable SDK: every emitted token passes the grammar in both modes. Zero "commit_tokens REJECT" lines in all recent runs confirms the no-desync invariant. For multi-hardware comparison (RTX 4070, GTi15 Arc PRO B70, N100) see [benchmark-overview.md](benchmark-overview.md).

---

## Phase 2 (2026-07-18, RTX 3060, Qwen3.5-9B) - pre-fix historical reference

> **Historical reference**: These results were measured on 2026-07-18, BEFORE the 2026-07-23 c.gbnf preprocessor tightening and the standard-mode final-token constraint (see [observations.md](observations.md)). They describe a previous code state and should not be treated as current numbers. Re-run section 3.7's automation script against your own model/hardware for current results. The correctness tables below (n_max 16, 10/10 valid) are from that earlier code state; the standard-mode final-token constraint was not yet applied, so the filter did not constrain the final sampled token in default mode.

**Environment**: LXC container, NVIDIA RTX 3060 (12GB), CUDA. **Model**: `Qwopus3.5-9B-Coder-MTP-q8_0.gguf` (qwen35, hybrid SSM + MTP, ~9B). **Prompt**: `#include <stdio.h>\nint main(void) { int x = 0; ` (newline-terminated). **Command shape**: `./build/bin/benchmark-deterministic-draft -m $MODEL --det-draft-model deterministic-draft-model-poc/build/deterministic-draft.so --det-draft-n-max <NMAX> --det-draft-accept-all -p $PROMPT --spec-type draft-mtp -c 4096 -fit on --temp 0.2 --top-k 20 --top-p 0.9 --min-p 0.1 --n-predict 128 --compare --n-runs <N>`. Both baseline and treatment output are validated with `gcc -fsyntax-only`. See [quick-start.md](quick-start.md) for how to reproduce.

The `gcc -fsyntax-only` validity gate is gcc-version-dependent: GCC 14+ errors on implicit function declarations while older gcc only warns, so the same grammar-valid output can score invalid on a newer toolchain (and vice versa). See the validity metric note in [benchmark-overview.md](benchmark-overview.md) before comparing validity numbers across machines.

### Correctness (10 runs, n_max 16)

| Base valid (gcc) | Trt valid (gcc) | Caught by filter |
|-----------------|-----------------|-----------------|
| 7/10 | 10/10 | 3 |

The raw model broke C syntax in 3/10 runs (degenerate declaration spam truncated mid-program; or valid C followed by prose contamination), while the filter was valid in all 10. "Caught by filter" = runs where the raw model produced invalid C but the filter produced valid C. Specific failure outputs are model/prompt/seed-dependent.

### Accept-all throughput by n_max (10 runs for n_max 16, 3 runs for 32/100)

| n_max | Base TPS | Trt TPS | Speedup | Base Acc% | Trt Acc% | Trt valid (gcc) |
|-------|----------|---------|---------|-----------|----------|-----------------|
| 16 | 36.35 | 77.12 | 2.12x | 21.8 | 100.0 | 10/10 |
| 32 | 12.24 | 63.23 | 5.17x | 16.7 | 100.0 | 0/3 |
| 100 | 1.55 | 28.60 | 18.45x | 3.5 | 100.0 | 0/3 |

### Observations

1. **The filter guarantees validity the raw model does not** - MTP-only produced invalid C in 3/10 runs; the filter was valid in 10/10. MTP-only output is now gcc-validated in compare mode (previously unchecked), and the `caught by filter` row counts runs where the raw model broke but the filter held.
2. **Grammar verification is now effectively free** - replacing the per-step `FillNextTokenBitmask` (~1.1-1.8s per call over the ~150k vocab) with O(1) `AcceptToken` probes dropped the sample/accept phase from ~2900ms to ~0.5ms per run. The remaining per-step cost is the model forward passes (draft decode, target decode, catch-up decode).
3. **n_max 16 is the sweet spot for this model** - it is both the fastest configuration (77.12 tps, 2.12x) AND the only one that stays gcc-valid (10/10). The target re-anchors the weak MTP head every ~16 tokens, preventing drift.
4. **Larger drafts drift invalid** - at n_max 32/100 the draft head drifts into grammar-valid-but-degenerate output (comment spam) that fails gcc, despite higher raw speedup. The grammar validates syntax, not coherence.
5. **qwen35 does not share the draft/target KV** - the MTP draft context keeps a separate KV/recurrent cache (upstream `ctx_other` is only honored for GEMMA4_ASSISTANT/EAGLE3), so the target must still decode accepted tokens to populate its own cache. The "target only computes the bonus token" ideal requires shared KV, which this arch does not provide.

### Multi-language coverage (n_max 16, accept-all)

One grammar-parseable prompt per bundled language, same environment/command shape. Validators: C `gcc -fsyntax-only`, Python `python3 compile`, JavaScript `node --check`, Java `javac` (Corretto 25).

| Language | Runs | Base TPS | Trt TPS | Speedup | Base valid | Trt valid | Caught by filter |
|----------|------|----------|---------|---------|-----------|-----------|-----------------|
| C | 10 | 36.35 | 77.12 | 2.12x | 7/10 | 10/10 | 3 |
| Python | 5 | 51.82 | 121.75 | 2.35x | 3/5 | 5/5 | 2 |
| JavaScript | 5 | 56.39 | 111.91 | 1.98x | 0/5 | 0/5 | 0 |
| Java | 5 | 46.04 | 136.26 | 2.96x | 0/5 | 0/5 | 0 |

- Throughput speedup is consistent across languages (~2x-3x) - the accept-all win is language-independent.
- C and Python hold validity (Python after a `target`-trailer grammar fix; see observations.md).
- JavaScript and Java fail real-parser validation because the model degenerates into redeclaration/semantic-error spam and does not complete; those are static-semantic early errors (`const` redeclaration, duplicate constructor, undeclared var, type error) beyond CFG syntax, so out of scope for the grammar. Throughput is unaffected by the validity gap.

The rest of this document (test prompts, command templates, automation script) is the runnable methodology for reproducing these results.

---

## 1. Prerequisites

### Build the benchmark tool

```bash
cmake -B build -DDETERMINISTIC_SPEC_ENABLED=ON
cmake --build build --target benchmark-deterministic-draft -j$(nproc)
```

### Build the deterministic draft plugin

The plugin is a standalone project under `deterministic-draft-model-poc/`. It builds independently of the main llama.cpp tree. The plugin uses XGrammar for grammar-constrained decoding with jump-forward support.

```bash
cd deterministic-draft-model-poc
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -O3"
cmake --build build -j$(nproc)
```

Produces: `deterministic-draft-model-poc/build/deterministic-draft.so`

Building also bundles the `grammars/` directory alongside the `.so` (via a CMake custom target), so the plugin can resolve grammars by language name at runtime without any host-side file path configuration. Grammars are plain `.gbnf` data files - editable without rebuilding the plugin. To use a different grammar bundle location, set `DETERMINISTIC_DRAFT_GRAMMAR_DIR` in the environment before running. The language is auto-detected by the plugin's bootstrap detection from the generated content, or pinned explicitly with the bench's `--det-draft-language <name>` flag.

The plugin requires pybind11 for XGrammar's build (fetched automatically via CMake `FetchContent`). If not installed:
```bash
pip install pybind11
```

### Model

Download an MTP-capable model. The examples below use `unsloth/Qwen3.5-2B-MTP-GGUF`:

```bash
huggingface-cli download unsloth/Qwen3.5-2B-MTP-GGUF \
  --include "*Q4_K_M*" \
  --local-dir ~/models/Qwen3.5-2B-MTP
```

Set `MODEL=~/models/Qwen3.5-2B-MTP/unsloth_Qwen3.5-2B-MTP-GGUF_Q4_K_M.gguf`
(or your download path).

### Verify artifacts

```bash
ls external/lib/libdeterministic_draft_spec.so
ls deterministic-draft-model-poc/build/deterministic-draft.so
```

---

## 2. Test Prompts

Test files are in `deterministic-draft-model-poc/tests/` organized by language:

### C (4 benchmark files + 2 edge-case files + 12 preprocessor regression files)

| File | Description |
|------|-------------|
| `tests/c/01_quicksort.c.test` | Complete quicksort with partition |
| `tests/c/02_linked_list.c.test` | Singly linked list with insert/delete/search |
| `tests/c/03_hash_table.c.test` | Hash table with chaining |
| `tests/c/04_simple_valid.c.test` | Simple valid C fragment |

`tests/c/` also contains `05_missing_semicolon.c.test` (intentionally invalid),
`06_valid_if_else.c.test`, and the preprocessor-directive regression files
`07_garbage_include.c.test` through `18_error.c.test` (accept/reject coverage for
the per-directive `preprocessor` rules in c.gbnf). These are used by
`test_file_based.py` rather than the benchmark matrix - do not use them as
benchmark prompts.

### Python (2 files)

| File | Description |
|------|-------------|
| `tests/python/01_graph.py.test` | Graph class with BFS/DFS traversal |
| `tests/python/02_lru_cache.py.test` | LRU cache with ordered dict |

### JavaScript (2 files)

| File | Description |
|------|-------------|
| `tests/javascript/01_async_queue.js.test` | Async task queue with Promises |
| `tests/javascript/02_merge_sort.js.test` | Merge sort implementation |

### Java (2 files)

| File | Description |
|------|-------------|
| `tests/java/01_bst.java.test` | Generic binary search tree |
| `tests/java/02_producer_consumer.java.test` | Producer-consumer with wait/notify |

### Prompt extraction

Each test file has 3 comment header lines that must be stripped:

```
// TEST: <language>
// DESC: <description>
// EXPECT: accept_all
```

After stripping these, the first ~20 lines of actual code are used as the prompt. For example, `01_quicksort.c.test` yields:

```c
#include <stdio.h>

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
```

> **Keep the prompt's real newlines (Phase 2 requirement).** The grammar matcher is primed with the prompt, so it must be valid input for the grammar. Do NOT flatten the prompt to one line - for C, `#include <stdio.h>` must keep its trailing newline, or the c.gbnf matcher stays stuck inside `include_directive` awaiting `\n`; it then rejects every non-newline token, so drafts truncate to zero and generation degrades to single constrained tokens. See the note in quick-start.md and the automation-script warning in section 3.7.

> **Generation-forcing prompts required.** A prompt that is already a complete program (e.g. a full `main()` function) makes the model emit EOS immediately (1-2 tokens), producing meaningless benchmarks. Use an open function body with an instruction comment: `#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n\n// Implement a hash table in C with insert, lookup, and delete.\n// Use separate chaining with linked lists. Add comments.\nint main(void) {`.

---

## 3. Running Benchmarks

### 3.1 Command template

```bash
./build/bin/benchmark-deterministic-draft \
  -m $MODEL \
  --det-draft-model ./deterministic-draft-model-poc/build/deterministic-draft.so \
  --det-draft-n-max <NMAX> \
  --det-draft-language c \
  -p "<prompt>" \
  -n 200 \
  -ngl 99 \
  -fa on \
  --compare \
  --n-runs 3 \
  2>&1
```

The language is auto-detected by the plugin's bootstrap detection from the generated content, or pinned explicitly with `--det-draft-language <name>`. Pin `--det-draft-language` when you need strict single-language guarantees. Bootstrap detection validates against a UNION of surviving candidate languages: a token valid in ANY surviving candidate passes (e.g. `#` lines keep the Python candidate alive since they parse as Python comments). Pin the language to avoid this.

Note: when `--det-draft-n-max > 0`, it auto-derives `--spec-draft-n-max`. You do NOT need to set `--spec-draft-n-max` separately; if both are set, `--det-draft-n-max` wins.

Note (Phase 2): the prompt must be grammar-parseable - pass it with its real newlines (for C, `#include` must end with a newline). On weak draft heads use a small `--det-draft-n-max` (e.g. 16); large drafts drift into grammar-valid-but-degenerate output that fails the real parser. The bench validates BOTH baseline and treatment output against the real parser and reports `output valid` rows plus a `caught by filter` count (runs where the raw model produced invalid code but the filter produced valid code).

### 3.2 Key flags

| Flag | Purpose |
|------|---------|
| `--compare` | Runs baseline (MTP only) then treatment (MTP + det filter). Outputs comparison table and JSON. |
| `--n-runs 3` | 3 runs per mode, averaged. |
| `--det-draft-n-max <N>` | Sets BOTH the filter cap and MTP draft count when > 0 (auto-derives --spec-draft-n-max). -1 = no cap (MTP default of 3). 0 = filter disabled (warning emitted). |
| `--det-draft-accept-all` | Skip target model verification for filter-accepted tokens. Grammar is sole verifier. Output follows draft+grammar, not target model. See disclosures for scope limitations. |
| `--det-draft-language <name>` | Pin the grammar language (e.g. c, java, javascript, python) instead of relying on bootstrap auto-detection. Use for strict single-language guarantees. |
| `-n 128` | Number of tokens to predict before stopping. |
| `-ngl 99` | Offload all layers to GPU (0 for CPU-only). |
| `-fa on` | Enable flash attention. |

### 3.3 Single test example

```bash
PROMPT=$(tail -n +4 tests/c/01_quicksort.c.test | head -20)

./build/bin/benchmark-deterministic-draft \
  -m $MODEL \
  --det-draft-model ./deterministic-draft-model-poc/build/deterministic-draft.so \
  --det-draft-n-max 3 \
  --det-draft-language c \
  -p "$PROMPT" \
  -n 200 \
  -ngl 99 \
  -fa on \
  --compare \
  --n-runs 3 \
  2>&1
```

### 3.4 Sanity check

A healthy run produces zero "inconsistent sequence positions" errors (KV cache desync bug - previously fixed, confirmed across all benchmarks):

```bash
./build/bin/benchmark-deterministic-draft ... 2>&1 | grep -c "inconsistent sequence positions"
```

A non-zero count indicates a regression. Zero is expected.

### 3.5 Capturing results

The `--compare` flag produces two outputs. The values below are example formatting only - they come from the older tree-sitter-based implementation and will differ with the current XGrammar plugin. Run the automation script in section 3.7 against your own model/hardware for current numbers.

**Comparison table (stderr):**
```
=== Comparison Results ===

                    MTP only    MTP + Grammar Filter    Improvement
  throughput tps:       17.66               441.10             +2398%
  accept rate:           2.6%               100.0%            +97.4pp
  n_predict:              204                 202
  drafted (pre):         5600                 200             28x fewer
  drafted (post):        5600                 200             28x fewer
  det truncated:          N/A                   0
```

**JSON output (stdout):**
```json
{
  "baseline": {"tps": 17.66, "accept_rate": 2.6, "n_predict": 204, "n_drafted_pre": 5600, "n_drafted_post": 5600},
  "treatment": {"tps": 441.10, "accept_rate": 100.0, "n_predict": 202, "n_drafted_pre": 200, "n_drafted_post": 200, "det_truncated": 0},
  "speedup": 24.98
}
```

**Capturing both streams:**
```bash
# Capture stderr (log + comparison) to a file, stdout (JSON) to another
./build/bin/benchmark-deterministic-draft ... 2> run_output.txt > run_results.json
```

### 3.6 Full test suite

The full matrix covers 10 test files at 4 n_max values for two modes (accept-all and default). The loops below run the matrix manually; the automation script in section 3.7 does the same and also captures a summary CSV.

#### Running accept-all mode (recommended for throughput)

```bash
for lang in c python java javascript; do
  for test_file in deterministic-draft-model-poc/tests/${lang}/*.test; do
    name=$(basename "$test_file" .test | cut -c4-)
    PROMPT=$(tail -n +4 "$test_file" | head -20)
    for nmax in 10 30 50 100; do
      echo "=== $lang/$name n_max=$nmax accept-all ==="
      ./build/bin/benchmark-deterministic-draft \
        -m "$MODEL" \
        --det-draft-model deterministic-draft-model-poc/build/deterministic-draft.so \
        --det-draft-n-max "$nmax" \
        --det-draft-accept-all \
        --det-draft-language "$lang" \
        -p "$PROMPT" \
        -n 200 --compare --n-runs 3
      echo ""
    done
  done
done
```

#### Running default mode (target verifies)

Same loop, drop `--det-draft-accept-all` from the command.

### 3.7 Automation script

The following script runs all configurations and captures results to a summary CSV. Set `ACCEPT_ALL=1` for accept-all mode (recommended for throughput benchmarks), or leave unset for default mode.

```bash
#!/bin/bash
# run-comprehensive-bench.sh
# Runs the full multi-language benchmark suite.
#
# Usage:
#   Accept-all mode: ACCEPT_ALL=1 MODEL=<model.gguf> bash run-comprehensive-bench.sh
#   Default mode:    MODEL=<model.gguf> bash run-comprehensive-bench.sh

set -euo pipefail

MODEL="${MODEL:?Set MODEL to the Qwen3.5-2B-MTP GGUF path}"
PLUGIN="deterministic-draft-model-poc/build/deterministic-draft.so"
TEST_DIR="deterministic-draft-model-poc/tests"
ACCEPT_ALL="${ACCEPT_ALL:-}"

OUTPUT_DIR="bench-results-$(date +%Y%m%d-%H%M%S)"
SUMMARY_CSV="${OUTPUT_DIR}/summary.csv"
mode_label="${ACCEPT_ALL:+accept-all}${ACCEPT_ALL:-default}"

N_MAX_VALUES=(10 30 50 100)
N_PREDICT=200
N_RUNS=3
NGL=99

declare -a TESTS
TESTS+=("c,01_quicksort,${TEST_DIR}/c/01_quicksort.c.test")
TESTS+=("c,02_linked_list,${TEST_DIR}/c/02_linked_list.c.test")
TESTS+=("c,03_hash_table,${TEST_DIR}/c/03_hash_table.c.test")
TESTS+=("c,04_simple_valid,${TEST_DIR}/c/04_simple_valid.c.test")
TESTS+=("java,01_bst,${TEST_DIR}/java/01_bst.java.test")
TESTS+=("java,02_producer_consumer,${TEST_DIR}/java/02_producer_consumer.java.test")
TESTS+=("javascript,01_async_queue,${TEST_DIR}/javascript/01_async_queue.js.test")
TESTS+=("javascript,02_merge_sort,${TEST_DIR}/javascript/02_merge_sort.js.test")
TESTS+=("python,01_graph,${TEST_DIR}/python/01_graph.py.test")
TESTS+=("python,02_lru_cache,${TEST_DIR}/python/02_lru_cache.py.test")

mkdir -p "$OUTPUT_DIR"

echo "lang,test,n_max,mode,base_tps,trt_tps,speedup,base_acc_pct,trt_acc_pct,base_n_pre,trt_n_pre,det_trunc,base_n_pred,trt_n_pred,errors" \
  > "$SUMMARY_CSV"

export LD_LIBRARY_PATH=./build/bin:${LD_LIBRARY_PATH:-}

for entry in "${TESTS[@]}"; do
  IFS=',' read -r lang test_name file_path <<< "$entry"

  for n_max in "${N_MAX_VALUES[@]}"; do
    echo ""
    echo "======================================================================"
    echo "  ${lang}: ${test_name} (n_max=${n_max}, mode=${mode_label})"
    echo "======================================================================"

    prompt=$(tail -n +4 "$file_path" | head -20)
    # WARNING (Phase 2): do NOT flatten the prompt to one line (e.g. with
    # `tr '\n' ' '`). The grammar matcher is primed with the prompt, so it
    # must keep its real newlines - for C, `#include` must end with a newline or the
    # c.gbnf matcher stays stuck inside `include_directive` awaiting `\n`, rejecting
    # every non-newline token and truncating all drafts to zero. Pass "$prompt"
    # (newlines preserved), not a flattened variant. The historical version of this
    # script flattened the prompt; that breaks grammar-constrained runs.
    prompt_flat="$prompt"

    out_file="${OUTPUT_DIR}/${lang}_${test_name}_nmax${n_max}_${mode_label}.txt"
    json_file="${OUTPUT_DIR}/${lang}_${test_name}_nmax${n_max}_${mode_label}.json"

    if [ -n "${ACCEPT_ALL:-}" ]; then
      set +e
      ./build/bin/benchmark-deterministic-draft \
        -m "$MODEL" \
        --det-draft-model "$PLUGIN" \
        --det-draft-n-max "$n_max" \
        --det-draft-accept-all \
        --det-draft-language "$lang" \
        -p "$prompt_flat" \
        -n "$N_PREDICT" \
        -ngl "$NGL" \
        -fa on \
        --compare \
        --n-runs "$N_RUNS" \
        2> "$out_file" > "$json_file"
      set -e
    else
      set +e
      ./build/bin/benchmark-deterministic-draft \
        -m "$MODEL" \
        --det-draft-model "$PLUGIN" \
        --det-draft-n-max "$n_max" \
        --det-draft-language "$lang" \
        -p "$prompt_flat" \
        -n "$N_PREDICT" \
        -ngl "$NGL" \
        -fa on \
        --compare \
        --n-runs "$N_RUNS" \
        2> "$out_file" > "$json_file"
      set -e
    fi

    errors=$(grep -c "inconsistent sequence positions" "$out_file" || true)

    json_block=$(cat "$json_file")

    base_tps=$(echo "$json_block" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['baseline']['tps'])" 2>/dev/null || echo "N/A")
    trt_tps=$(echo "$json_block" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['treatment']['tps'])" 2>/dev/null || echo "N/A")
    speedup=$(echo "$json_block" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['speedup'])" 2>/dev/null || echo "N/A")
    base_acc=$(echo "$json_block" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['baseline']['accept_rate'])" 2>/dev/null || echo "N/A")
    trt_acc=$(echo "$json_block" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['treatment']['accept_rate'])" 2>/dev/null || echo "N/A")
    base_n_pre=$(echo "$json_block" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['baseline']['n_drafted_pre'])" 2>/dev/null || echo "N/A")
    trt_n_pre=$(echo "$json_block" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['treatment']['n_drafted_pre'])" 2>/dev/null || echo "N/A")
    det_trunc=$(echo "$json_block" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['treatment']['det_truncated'])" 2>/dev/null || echo "N/A")
    base_n_pred=$(echo "$json_block" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['baseline']['n_predict'])" 2>/dev/null || echo "N/A")
    trt_n_pred=$(echo "$json_block" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['treatment']['n_predict'])" 2>/dev/null || echo "N/A")

    echo "${lang},${test_name},${n_max},${mode_label},${base_tps},${trt_tps},${speedup},${base_acc},${trt_acc},${base_n_pre},${trt_n_pre},${det_trunc},${base_n_pred},${trt_n_pred},${errors}" \
      >> "$SUMMARY_CSV"

    echo "  -> MTP only: ${base_tps} tps, MTP + Filter: ${trt_tps} tps, Speedup: ${speedup}x"
    echo "  -> Errors: ${errors}"
    echo "  -> Saved: ${out_file}"
  done
done

echo ""
echo "======================================================================"
echo "  Complete! Results in ${OUTPUT_DIR}/"
echo "  CSV summary: ${SUMMARY_CSV}"
echo "  Mode: ${mode_label}"
echo "======================================================================"
cat "$SUMMARY_CSV"
```

#### Usage

```bash
export MODEL=~/models/Qwen3.5-2B-MTP/unsloth_Qwen3.5-2B-MTP-GGUF_Q4_K_M.gguf

# Accept-all mode (recommended for throughput benchmarks)
ACCEPT_ALL=1 bash run-comprehensive-bench.sh

# Default mode (target verifies)
bash run-comprehensive-bench.sh
```

#### Expected runtime

With 10 test files x 4 n_max values = 40 configurations x 3 runs each (120 benchmark iterations), the full suite takes approximately 1-2 hours per mode on GPU, longer on CPU.

---

## 4. Disclosures and Scope

### When accept-all is SAFE
- Pure code completion in a known language (IDE autocomplete, fill-in-middle)
- Single-language generation from a known prompt (C, Python, Java, JavaScript)
- Agent codegen where output language is fixed

### When accept-all is NOT SAFE (use default mode or disable filter)
- Chat output with interleaved code blocks (markdown, HTML+CSS+JS)
- Multi-language generation (e.g., HTML + CSS + JS in same output)
- Plain prose, README text, documentation generation
- Any prompt where output language could switch mid-generation

### Why

The filter uses a single grammar per buffer. When text doesn't match that grammar (e.g., markdown prose), the grammar's permissiveness means tokens pass through without structural validation. With accept-all, these unvalidated tokens are committed. The filter becomes a no-op for that content - no benefit, but no harm if the output was correct. The risk is accepting tokens the target model would have rejected.

### Recommendation
Use `--det-draft-accept-all` only when the output language is known and fixed. For mixed-content or unknown-language prompts, use default mode (target verification) with `--det-draft-n-max` set to a moderate value (10-30), or disable the filter entirely.

---

## 5. Accept-All Benchmark Results

> The current (XGrammar) accept-all results are the Phase 2 tables at the top of
> this document. The historical tree-sitter tables previously kept here were
> removed: they described an implementation that no longer exists and are
> recoverable from git history if ever needed. Re-run section 3.7's automation
> script for up-to-date results on your model and hardware.

With `--det-draft-accept-all`, target model verification is skipped for all filter-accepted tokens. The deterministic filter's structural validation becomes the final arbiter. The output distribution follows draft+grammar, NOT the target model. This trades verification safety for raw throughput and is suitable for single-language code-only generation (no mixed markdown/chat output). Speed numbers in this mode measure a fundamentally different algorithm than default mode.

### Known limitations

1. **XGrammar validates syntax, not semantics** - syntactically valid but semantically wrong code passes the filter (e.g., `quick{swap` instead of `quicksort(`)
2. **Output quality not guaranteed** - the filter ensures structural validity but not semantic correctness
3. **Grammar coverage varies by language** - some GBNF grammars may be more permissive than others, affecting filter effectiveness
4. **Higher n_max = more risk** - more drafts per iteration means more chances for the model to produce syntactically valid but semantically wrong tokens

### Validity Metric

The benchmark validator runs `gcc -fsyntax-only`. Warnings (extra tokens after #include, implicit int) exit 0 and count as VALID. Grammar-valid but semantically wrong code (duplicate typedefs, undeclared identifiers) passes the filter by design -- CFGs cannot check static semantics. This is a documented limitation; see observations.md for details. The filter guarantees syntax only; it cannot catch either class of semantic error.

### Current architecture (XGrammar)

1. **Bitmask-constrained decoding** - the plugin's `fill_bitmask()` provides a token-level bitmask of grammar-valid tokens *before* sampling, via XGrammar's compiled grammar + tokenizer-aware `AcceptToken`. This constrains generation proactively rather than validating text after the fact.
2. **Bundled, loadable grammars** - grammars are resolved by language name from a `grammars/` directory bundled alongside the plugin `.so` (or `DETERMINISTIC_DRAFT_GRAMMAR_DIR` if set), not hardcoded or requiring a rebuild to add/edit. The language is auto-detected by the plugin's bootstrap detection from the generated content.
3. **KV cache consistency** - draft truncation is handled via the standard speculative-decoding checkpoint/rollback mechanism already used for the non-deterministic draft path. The grammar matcher has its own state, separate from the KV cache: the plugin implements `rollback`/`reset` for it, and the host rolls the matcher back to the accepted prefix whenever the target rejects part of a filtered draft (standard mode), so matcher state and emitted text never desync.

Note: the tree-sitter-era circuit breaker, non-ASCII token rejection, and separate bonus-token-vs-filter validation described in earlier revisions of this document no longer exist - they were specific workarounds for tree-sitter's text-based validation model and were removed when the plugin was rewritten around XGrammar's bitmask API. The bonus token (final token sampled after the drafted batch) is now bitmask-constrained the same way as every other token, via `common_speculative_sample_and_accept`.

---

## 6. Test Suites

### C++ unit tests (`tests/test-deterministic-draft.cpp`)

33 test functions covering the C API, plugin loader, filter API, bitmask API, grammar state, bootstrap detection, and state serialization. 16 tests skip without the plugin .so found at the expected build path; 3 of those 16 additionally require the `LLAMA_TEST_MODEL` environment variable set to a model path. Tests that do not need the plugin or a model run unconditionally.

### File-based grammar harness (`deterministic-draft-model-poc/tests/`)

24 test files: 18 C (4 benchmark files + 2 edge-case files + 12 preprocessor-directive regression files 07-18), 2 Java, 2 JavaScript, 2 Python. Uses a 256-byte synthetic vocabulary, feeds code character-by-char via the bitmask API, and verifies accept/reject behavior matches file headers. These are grammar unit tests, not model integration tests -- they validate the plugin's grammar matching without loading a model.
