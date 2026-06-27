# Deterministic Draft Filter: Quick Start

> **llama.cpp**: build 9733 (commit f449e0553), GNU 15.2.0, Linux x86_64 (documented benchmark build; build from source for current version)

## Prerequisites

### System packages

```bash
sudo apt install build-essential cmake git python3
```

### pybind11 (required by XGrammar's build)

The plugin utilizes [XGrammar](https://github.com/mlc-ai/xgrammar) for grammar-constrained decoding, which is fetched automatically via CMake `FetchContent`. XGrammar requires pybind11 for its build:

```bash
pip install pybind11
```

### Model

The benchmark requires an MTP (Multi-Token Prediction) model. Download it here:

```bash
huggingface-cli download unsloth/Qwen3.5-2B-MTP-GGUF --include "*Q4_K_M*" --local-dir ~/models/Qwen3.5-2B-MTP
```



## Build Steps

### Step 1: Build llama.cpp core with spec enabled

```bash
cmake -B build -DDETERMINISTIC_SPEC_ENABLED=ON
cmake --build build --config Release -j$(nproc)
```

This produces:

- `build/bin/libllama.so` / `build/bin/libllama-common.so` - core libraries with plugin loader (`.a` for static builds)
- `build/bin/benchmark-deterministic-draft` - benchmark tool
- `external/include/deterministic_draft_plugin.h` - SDK header
- `external/lib/libdeterministic_draft_spec.so` - spec loader shared library

### Step 2: Link SDK artifacts into the PoC project

```bash
cd deterministic-draft-model-poc
./link-sdk.sh
```

This simulates linking the header and `.so` from `external/` into `deterministic-draft-model-poc/lib/`.

### Step 3: Build the PoC plugin (standalone)

```bash
cd deterministic-draft-model-poc
rm -rf build
./link-sdk.sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -O3"
cmake --build build -j$(nproc)
```

This produces:

- `deterministic-draft-model-poc/build/deterministic-draft.so` - the plugin

### Step 4: Verify setup

```bash
./build/bin/benchmark-deterministic-draft \
    -hf unsloth/Qwen3.5-2B-MTP-GGUF:Q4_K_M \
    --spec-type draft-mtp -p "test" -n 1
```

Alternatively, manually download from `unsloth/Qwen3.5-2B-MTP-GGUF` on HuggingFace and pass the file via `-m <path>`.

### Step 5: Run

```bash
# MTP only
./build/bin/benchmark-deterministic-draft -m <model.gguf> --spec-type draft-mtp \
    -p "int fib(int n) {" -n 200

# MTP + grammar filter
./build/bin/benchmark-deterministic-draft -m <model.gguf> \
    --deterministic-draft-model deterministic-draft-model-poc/build/deterministic-draft.so \
    -p "int fib(int n) {" -n 200

# Comparison (runs both automatically)
./build/bin/benchmark-deterministic-draft -m <model.gguf> \
    --deterministic-draft-model deterministic-draft-model-poc/build/deterministic-draft.so \
    -p "int fib(int n) {" -n 200 --compare --n-runs 3
```

The language is auto-detected from the generated content (see Language / Grammar Configuration below) - no `--lang` flag is needed.

## Running Benchmarks / Reproducing Results

### Throughput benchmark (baseline vs accept-all)

The `--compare` flag runs baseline (MTP only) then treatment (MTP + filter + accept-all) and prints a comparison table. The canonical Phase 2 command:

```bash
./build/bin/benchmark-deterministic-draft \
  -m <model.gguf> \
  --det-draft-model deterministic-draft-model-poc/build/deterministic-draft.so \
  --det-draft-n-max 16 \
  --det-draft-accept-all \
  -p "#include <stdio.h>
int main(void) { int x = 0; " \
  --spec-type draft-mtp --n-gpu-layers -1 -c 4096 -fit on \
  --temp 0.2 --top-k 20 --top-p 0.9 --min-p 0.1 --n-predict 128 \
  --compare --n-runs 10
```

Key points:

- **The prompt must be grammar-parseable** (newline after `#include` - see the note in Language / Grammar Configuration below). Without it the grammar cannot validate, and results are meaningless.
- **`--det-draft-n-max` sets BOTH the filter cap and the MTP draft count.** Use a small value (16) on weak draft heads - see the draft-size vs coherence finding in [benchmark-overview.md](benchmark-overview.md).
- **`--det-draft-accept-all`** makes the grammar the sole verifier (the fast path). Drop it for default mode (target verifies).
- **`--compare --n-runs N`** runs N iterations per mode and averages them.

### Reading the comparison table

```
                    MTP only    MTP + Grammar Filter    Delta
  throughput tps:      36.35              77.12             +112.2%
  accept rate:          21.8%             100.0%            +78.2%
  output valid:           7/10             10/10
  caught by filter:        3
```

- `output valid`: runs whose output passed `gcc -fsyntax-only`. Both baseline and treatment are checked (baseline is validated with the same language the plugin resolved for treatment).
- `caught by filter`: runs where the raw model produced invalid code but the filter produced valid code. This is the filter's correctness guarantee (also emitted as `"caught_by_filter"` in the JSON).

### Correctness (caught-by-filter) testing

Whether the raw model produces invalid code - and thus whether `caught by filter` fires - depends on the model, prompt, and seed. A different model is NOT guaranteed to reproduce the same failures. To run the same kind of correctness test on your own model:

1. Use a fixed-language, grammar-parseable prompt (e.g. the C snippet above).
2. Run `--compare --n-runs 10` (more runs = better signal) with `--det-draft-accept-all`.
3. Read the `output valid` rows and `caught by filter` count in the table.

To make failures more likely (so the filter's value is visible), use conditions under which the raw model is error-prone: a smaller/weaker model, a prompt that invites degeneration, or higher temperature. On a strong coder model with an easy prompt the raw model may never fail, so `caught by filter` will be 0 - that means the raw model happened to be correct, not that the filter adds no value.

## Language / Grammar Configuration

The plugin resolves grammars by language name at runtime. Grammars are plain `.gbnf` files bundled alongside the `.so` by a CMake build step - no manual copy needed.

**Language is auto-detected** by the plugin's bootstrap detection. When the filter is first used, the plugin loads all bundled grammars and narrows them down as tokens are generated: each candidate grammar accepts or rejects the streamed tokens, and candidates that reject are dropped. Detection converges to a single language once only one candidate remains accepting, or reports the language as unresolved if a token is genuinely invalid for every candidate. There is no `--lang` flag and no environment variable to force a language - the detected language is whatever the generated content actually is. To steer detection toward a specific language, give the model a prompt that is unambiguously in that language (e.g. a C snippet) so detection resolves to it.

The detected language is reported in the benchmark tool's output and logs (e.g. `=== Output Validation (c) ===` and the `output validation FAILED (c)` / `output VALID (c)` lines), so you can see which grammar was selected.

> **Prompt must be grammar-parseable.** The grammar matcher is primed with the prompt, so the prompt itself must be valid input for the grammar. In particular, a C prompt must terminate its `#include` line with a newline: `#include <stdio.h> int main(void) {` (no `\n`) leaves the `c.gbnf` matcher stuck inside its `preprocessor` rule (`[^\n]* "\n"`), so it accepts *any* non-newline token instead of validating C. Use `#include <stdio.h>\nint main(void) {` so the matcher leaves the preprocessor rule and validates the generated body as C. See the Phase 2 section of [benchmark-overview.md](benchmark-overview.md).

| Bundled grammar | Grammar file |
|-----------------|-------------|
| `c` | `grammars/c.gbnf` |
| `java` | `grammars/java.gbnf` |
| `python` | `grammars/python.gbnf` |
| `javascript` | `grammars/javascript.gbnf` |

**Override grammar location** with the `DETERMINISTIC_DRAFT_GRAMMAR_DIR` environment variable:

```bash
export DETERMINISTIC_DRAFT_GRAMMAR_DIR=/path/to/custom/grammars
./build/bin/benchmark-deterministic-draft ...
```

If unset, the plugin defaults to `<plugin_directory>/grammars/` (the directory containing the `.so`).

**Via the C API** (for programmatic use), call `deterministic_draft_set_language()` or `deterministic_draft_set_grammar()` with an EBNF string.

## Build Structure

The deterministic draft system uses abstraction and separation of concerns to keep changes to the main llama.cpp tree minimal. There are three independent layers:

### 1. Changes to main llama.cpp core

The core changes are small and domain-agnostic. The core knows nothing about XGrammar or any specific validation strategy.

```
llama.cpp/ (main tree)
  ├── include/
  │   ├── deterministic_draft_plugin.h       -- Public C API contract (the SPI / "SDK" header)
  │   ├── deterministic_draft_capabilities.h -- Capability flags shared by both headers
  │   └── llama_deterministic_draft.h        -- Consumer C API declarations (extern "C")
  ├── src/llama-deterministic-draft-serviceloader.cpp  -- ServiceLoader (dlopen/dlsym of the SPI)
  ├── external/include/
  │   ├── deterministic_draft_plugin.h       -- Plugin contract header (for plugin authors)
  │   └── llama_deterministic_draft.h        -- Consumer API header (for end users)
  ├── common/speculative.{h,cpp}            -- Pipeline integration (calls plugin via C API)
  ├── common/arg.cpp                        -- 3 CLI flags + auto-imply MTP
  ├── common/common.h                       -- Enum + params struct
  ├── tools/server/server-context.cpp       -- Server uses speculative pipeline (no bolt-on)
  └── tools/deterministic-draft-bench/      -- Benchmark tool
```

There is no XGrammar-specific code. There is no domain-specific logic. There is no plugin implementation details. The core provides the SPI contract header, a ServiceLoader (`dlopen` + `dlsym`) that resolves a provider's contract methods at runtime, and integrates it into the speculative pipeline.

### 2. External / shared distributed artifacts

These are built when `-DDETERMINISTIC_SPEC_ENABLED=ON` is passed to the main cmake. These artifacts are what third-party plugin authors consume.

```
llama.cpp/external/
  ├── CMakeLists.txt                        -- Installs headers + builds spec .so
  ├── include/
  │   ├── deterministic_draft_plugin.h      -- For plugin authors (implements the plugin)
  │   └── llama_deterministic_draft.h       -- For consumers (links against the .so)
  └── lib/
      └── libdeterministic_draft_spec.so    -- ServiceLoader as a standalone .so (no llama dep)
```

`libdeterministic_draft_spec.so` is built by `external/CMakeLists.txt` from the same ServiceLoader source that is compiled into libllama (`src/llama-deterministic-draft-serviceloader.cpp`), so the loader API is available both inside llama.cpp and as a standalone library for testing plugins without the full build.

The `.so` contains the `dlopen`/`dlsym` loader and all C API wrappers, built from the same source that libllama compiles.

**Two headers, two audiences:**
- `deterministic_draft_plugin.h` - for **plugin authors** who implement a .so (defines `deterministic_draft_create`, `deterministic_draft_fill_bitmask`, `deterministic_draft_commit`, `deterministic_draft_filter_draft`, etc. that the plugin must export)
- `llama_deterministic_draft.h` - for **consumers** who link against `libdeterministic_draft_spec.so` to load and call plugins (declares `llama_deterministic_draft_init`, `llama_deterministic_draft_filter_draft`, `llama_deterministic_draft_free`, etc.)

A consumer who also builds their own plugin (the typical case) needs both headers.

### 3. deterministic-draft-model-poc/ (consumer project)

A standalone project simulating an external consumer - e.g., a law firm, government agency, or commercial vendor with a private implementation in a secure corporate repository. This directory has **no dependency on the main llama.cpp build tree** and the main tree has **no dependency on this directory**.

```
deterministic-draft-model-poc/
  ├── CMakeLists.txt                        -- Standalone CMake project
  ├── link-sdk.sh                           -- Symlinks external artifacts into lib/
  ├── lib/
  │   ├── deterministic_draft_plugin.h      -- Symlinked from external/include/ (plugin author header)
  │   ├── llama_deterministic_draft.h       -- Symlinked from external/include/ (consumer header)
  │   └── libdeterministic_draft_spec.so    -- Symlinked from external/lib/
  ├── src/
  │   └── plugin.cpp                        -- XGrammar plugin implementation
  └── build/
      └── deterministic-draft.so            -- The plugin, loaded at runtime
```

A real consumer would receive the SDK artifacts via distribution (package manager, bundle, internal artifact repo) instead of symlinks. Their private implementation (XGrammar, regex, schema validators, legal citation parsers, etc.) would live entirely in their own repository.

## CLI Flags

| Flag | Alias | Description |
|------|-------|-------------|
| `--deterministic-draft-model <path>` | `--det-draft-model` | Path to the plugin (.so/.dylib/.dll). Auto-enables `--spec-type draft-mtp`. Requires an MTP-enabled model (`n_layer_nextn > 0`). |
| `--deterministic-draft-n-max <N>` | `--det-draft-n-max` | Controls the draft token budget. See behaviour table below. |
| `--det-draft-accept-all` | | Bypass target verification entirely. The filter becomes the sole verifier. Only active with `--det-draft-model`. See Disclosures and Scope below. |
| `--det-draft-language <name>` | | Pin the grammar language (e.g. c, java, javascript, python) for all slots instead of relying on bootstrap auto-detection. Use for strict single-language guarantees. |

### --det-draft-n-max behaviour

| Value | Filter cap | --spec-draft-n-max | MTP auxiliary heads draft |
|-------|------------|-------------------|--------------------------|
| -1 (default) | no cap | untouched (llama.cpp default: 3) | 3 tokens per step |
| 0 | filter disabled (warning emitted) | untouched | 3 tokens per step |
| N > 0 | N tokens | auto-set to N | N tokens per step |

When `--det-draft-n-max` is set to a value greater than 0, it controls the
draft budget at both ends simultaneously - the filter validates up to N tokens
and the MTP auxiliary heads draft up to N tokens. You do not need to set
`--spec-draft-n-max` separately; it is auto-derived.

This is why n_max matters for benchmark results. At n_max=100, the auxiliary
heads draft up to 100 tokens per step. With a 0.4% baseline accept rate (N100),
that means roughly 99 wasted draft tokens per step before the filter. With the
filter and `--det-draft-accept-all`, those 100 tokens are validated
structurally and committed directly - no target verification overhead.

Do not set `--spec-draft-n-max` manually when using `--det-draft-n-max > 0`:
`--det-draft-n-max` overrides it unconditionally (see `common_params_handle_models`
in `common/arg.cpp`).

## Disclosures and Scope

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

## Building a Custom Plugin

A custom plugin (e.g., a private legal citation validator) follows the same pattern as the PoC. The consumer needs both SDK headers and the shared library:

1. Copy the SDK artifacts from `external/`:
    - `deterministic_draft_plugin.h` - **plugin author header**: implement these functions in your plugin .so
    - `llama_deterministic_draft.h` - **consumer header**: use these to load and call the plugin via the .so
    - `libdeterministic_draft_spec.so` - the loader library to link against for standalone testing

2. Implement the plugin: write a .so that exports the `deterministic_draft_*` functions declared in `deterministic_draft_plugin.h`

3. Compile as a shared library (.so/.dylib/.dll), linking against `libdeterministic_draft_spec.so` if you want to test the plugin standalone (without llama.cpp)

4. Load via `--deterministic-draft-model /path/to/your-plugin.so` in any llama.cpp binary (the loader is always compiled into libllama; the CMake option only gates the standalone SDK artifacts)

No llama.cpp source or build system is needed. The plugin lives in its own repository, built and distributed independently. A law firm, government agency, or commercial vendor would keep their plugin implementation in a private repo, consuming only the distributed SDK artifacts.