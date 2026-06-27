# Deterministic Draft Spec SDK

SDK for building and loading deterministic draft plugins (.so/.dylib/.dll) into llama.cpp.

## Overview

A deterministic draft filter constrains speculative decoding to a grammar: every token the drafter proposes is checked against a set of allowed tokens, and the draft is cut short at the first token that does not match.

This repo ships one reference plugin, `deterministic_regex_plugin`, that implements the contract with a single constraint: letters and whitespace only. It is intentionally trivial - its only purpose is to demonstrate the plugin interface, not to be a useful grammar. A real plugin would plug in a full grammar engine (regex, JSON schema, BNF, and so on).

Because the constraint admits only letters and whitespace, every run below discards the digits and punctuation that real code needs. That is the filter working as designed, not a defect - the high `#truncated` and `#target rejected` numbers in the results are a direct consequence.

Two operating modes shown below:

- **Default mode** - the target model still verifies every surviving draft token; the filter constrains only the draft head.
- **Intercept-all mode** (`--det-draft-intercept-all`) - the filter is the sole verifier (MTP only); for other draft types it falls back to default mode with the bonus token also filtered.

See [`deterministic-draft-filter.md`](deterministic-draft-filter.md) for the full flag reference and mode design details.

## Layout

- `CMakeLists.txt` — Root SDK build
- `include/` — Plugin contract + consumer API headers (generated, do not edit)
- `lib/` — `libdeterministic_draft_spec.so` + plugin `.so` artifacts
- `plugins/` — Plugin sources (regex example: `deterministic_regex_plugin.h`/`.c` + `README.md`)

## Build

### 1. Build llama.cpp

From the repo root:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
# or specific to build with CUDA for example
cmake -B build -DDETERMINISTIC_SPEC_ENABLED=ON -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda \
-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_BUILD_TYPE=Release \
-DBUILD_SHARED_LIBS=OFF -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_OPENSSL=ON
cmake --build build --config Release
```

### 2. Build the SDK + plugin

From the repo root:

```sh
cmake --build build --target deterministic_draft_spec_plugin
```

Outputs:

- `lib/libdeterministic_draft_spec.so` — SDK loader library
- `plugins/lib/libdeterministic_draft_spec_plugin.so` — Regex filter plugin

## Run the Plugin

The plugin constrains draft tokens to letters and whitespace only. Run `llama-server` from the repo root.

### 1. EAGLE3

```sh
llama-server \
  -m /home/samueldoyle/AI_LOCAL/Models/eagle3/Qwen3-8B-eagle3-Q4_K_M.gguf \
  -md /home/samueldoyle/AI_LOCAL/Models/eagle3/Qwen3-8B-speculator.eagle3-F16.gguf \
  --spec-type draft-eagle3 \
  --det-draft-model external/plugins/lib/libdeterministic_draft_spec_plugin.so \
  --spec-draft-n-max 4 \
  -c 32768 -ngl 99 -fa on --port 8080
```

### 2. DSPARK

```sh
llama-server \
  -m /home/samueldoyle/AI_LOCAL/Models/dspark/Qwen3-4B-DSpark-Model-Q8_0.gguf \
  -md /home/samueldoyle/AI_LOCAL/Models/dspark/Qwen3-4B-DSpark-Q8_0.gguf \
  --spec-type draft-dspark \
  --det-draft-model external/plugins/lib/libdeterministic_draft_spec_plugin.so \
  --spec-draft-n-max 15 \
  -c 8192 -ngl 99 -fa on --jinja --port 8080
```

For DSPARK, `-m` is the Model file (`Qwen3-4B-DSpark-Model-Q8_0.gguf`) and `-md` is the draft (`Qwen3-4B-DSpark-Q8_0.gguf`). The "Model" suffix file is the TARGET.

### 3. MTP (Multi-Token Prediction)

```sh
llama-server \
  -m /home/samueldoyle/AI_LOCAL/Models/MTP/model_q8_0.gguf \
  --spec-type draft-mtp \
  --det-draft-model external/plugins/lib/libdeterministic_draft_spec_plugin.so \
  -c 8192 -ngl 99 -fa on --port 8080
```

## Verify the Filter

The filter allows only tokens made of letters or whitespace, but it constrains only the **draft head**. In default mode (no `--det-draft-intercept-all`), the target model verifies every draft token and emits the bonus/correction token **unconstrained**, so the output keeps digits, whitespace, and punctuation from the target's own choice.

Greedy sampling (`--temp 0 --top-k 1 --top-p 1.0`) makes the run deterministic. The same request is used for all three drafter types:

```sh
curl -s http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"def f():\n    return 1 + 2\n# what is f() ?","max_tokens":48}'
```

The prompt contains digits (`1`, `2`), whitespace, and punctuation (`:`, `(`, `)`, `+`, `?`, `#`). These appear in the output because the bonus token is unconstrained; only the draft-head tokens are filtered. The `statistics draft-deterministic` line reports draft-head activity only.

### draft-mtp

Start:

```sh
./build/bin/llama-server \
  -m /home/samueldoyle/AI_LOCAL/Models/MTP/model_q8_0.gguf \
  --spec-type draft-mtp \
  --det-draft-model external/plugins/lib/libdeterministic_draft_spec_plugin.so \
  --reasoning off --temp 0 --top-k 1 --top-p 1.0 \
  -c 8192 -ngl 99 -fa on --port 8080
```

Result (actual output):

```
# f() is a function call
# f is a function
# f() returns 3
# f() is a function call that returns 3
# f is a function that returns 3
# f() is
```

```
draft acceptance = 0.68571 (24 accepted / 35 generated), mean len = 2.14
statistics draft-deterministic: #drafts = 22, #truncated = 16, #tokens pre = 65, #tokens post = 35, #target rejected = 11
```

### draft-dspark

Start:

```sh
./build/bin/llama-server \
  -m /home/samueldoyle/AI_LOCAL/Models/dspark/Qwen3-4B-DSpark-Model-Q8_0.gguf \
  -md /home/samueldoyle/AI_LOCAL/Models/dspark/Qwen3-4B-DSpark-Q8_0.gguf \
  --spec-type draft-dspark \
  --spec-draft-n-max 15 \
  --det-draft-model external/plugins/lib/libdeterministic_draft_spec_plugin.so \
  --reasoning off --temp 0 --top-k 1 --top-p 1.0 \
  -c 8192 -ngl 99 -fa on --port 8080
```

Result (actual output):

```
 it is a function. So when you call f(), it returns 3. So the function f() is a function that returns 3. So the function f() is a function that returns 1 + 2. So the function
```

```
draft acceptance = 0.39683 (25 accepted / 63 generated), mean len = 2.79
statistics draft-deterministic: #drafts = 22, #truncated = 18, #tokens pre = 142, #tokens post = 63, #target rejected = 38
```

### draft-eagle3

Start:

```sh
./build/bin/llama-server \
  -m /home/samueldoyle/AI_LOCAL/Models/eagle3/Qwen3-8B-eagle3-Q4_K_M.gguf \
  -md /home/samueldoyle/AI_LOCAL/Models/eagle3/Qwen3-8B-speculator.eagle3-F16.gguf \
  --spec-type draft-eagle3 \
  --spec-draft-n-max 4 \
  --det-draft-model external/plugins/lib/libdeterministic_draft_spec_plugin.so \
  --reasoning off --temp 0 --top-k 1 --top-p 1.0 \
  -c 8192 -ngl 99 -fa on --port 8080
```

Result (actual output):

```
 3

def g():
    return f() + 1
# what is g() ? 4

def h():
    return g() + 1
# what is h() ? 5

def i():
    return
```

```
draft acceptance = 0.59375 (19 accepted / 32 generated), mean len = 2.12
statistics draft-deterministic: #drafts = 28, #truncated = 27, #tokens pre = 106, #tokens post = 32, #target rejected = 13
```

DSPARK may log the following, which is informational, not an error:

```
requested draft size (n_max=15, n_min=0) exceeds the trained block size 7 -- clamping to 7
```

### Reading the statistics

Two counters are logged per request. `draft acceptance` is the generic speculative-decoding metric (server `slot print_timing`); `statistics draft-deterministic` is the regex filter's own accounting of the same run.

**`draft acceptance = RATIO (accepted / generated), mean len = L`**

- `generated` = draft tokens that survived the regex and were sent to target verification; `accepted` = how many of those the target model verified as correct and kept. `RATIO = accepted / generated`.
- `mean len` = average accepted length per step, `1 + accepted / verification_steps`; the leading `1` is the bonus/correction token emitted every step regardless.

**`statistics draft-deterministic: #drafts / #truncated / #tokens pre / #tokens post / #target rejected`**

- `#drafts` = draft batches the filter examined.
- `#truncated` = of those, how many the filter cut short because a token contained a non-letter, non-whitespace byte.
- `#tokens pre` = tokens proposed before filtering; `#tokens post` = tokens that survived the regex.
- `#target rejected` = of the surviving tokens, how many the target model then rejected during verification.

The two lines reconcile: `#tokens post == generated == accepted + #target rejected`. High `#truncated` / `#target rejected` here is expected - the trivial letters-and-whitespace grammar rejects the digits and punctuation that real code output needs.

## Intercept-All Mode

The `--det-draft-intercept-all` flag skips target model verification - the plugin is the sole verifier. Only works with `draft-mtp`. For other draft types (eagle3, dspark, dflash), intercept-all falls back to default mode (target still verifies) and the bonus token is also filtered.

Greedy sampling (`--temp 0 --top-k 1 --top-p 1.0`) makes the run deterministic. The same request as "Verify the Filter" is used:

```sh
curl -s http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"def f():\n    return 1 + 2\n# what is f() ?","max_tokens":48}'
```

### MTP with intercept-all (enabled)

```sh
./build/bin/llama-server \
  -m /home/samueldoyle/AI_LOCAL/Models/MTP/model_q8_0.gguf \
  --spec-type draft-mtp \
  --det-draft-model external/plugins/lib/libdeterministic_draft_spec_plugin.so \
  --det-draft-intercept-all \
  --reasoning off --temp 0 --top-k 1 --top-p 1.0 \
  -c 8192 -ngl 99 -fa on --port 8080
```

Log: `--det-draft-intercept-all is enabled for draft-mtp`

Result (actual output):

```
f
print f
print f
print f
print f
print f
print f
print f
print f
print f
print f
print f
print f
print f
print f
print f
```

```
draft acceptance = 1.00000 (32 accepted / 32 generated), mean len = 3.67
statistics draft-deterministic: #drafts = 15, #truncated = 5, #tokens pre = 45, #tokens post = 32, #target rejected = 0
```

### DSPARK with intercept-all (falls back)

```sh
./build/bin/llama-server \
  -m /home/samueldoyle/AI_LOCAL/Models/dspark/Qwen3-4B-DSpark-Model-Q8_0.gguf \
  -md /home/samueldoyle/AI_LOCAL/Models/dspark/Qwen3-4B-DSpark-Q8_0.gguf \
  --spec-type draft-dspark \
  --spec-draft-n-max 15 \
  --det-draft-model external/plugins/lib/libdeterministic_draft_spec_plugin.so \
  --det-draft-intercept-all \
  --reasoning off --temp 0 --top-k 1 --top-p 1.0 \
  -c 8192 -ngl 99 -fa on --port 8080
```

Warning: `--det-draft-intercept-all requires the standard autoregressive MTP flow. Falling back to default mode with the bonus token also filtered (target still verifies, so performance may degrade)`

Result (actual output):

```
it is a function that returns  ivalues

def fadditionaitionsadditionaitionaitionaitionaitionaitionaitionaitionaitionaitionaitionaitionaitionaitionaition
```

```
draft acceptance = 0.78788 (26 accepted / 33 generated), mean len = 3.89
statistics draft-deterministic: #drafts = 21, #truncated = 18, #tokens pre = 146, #tokens post = 33, #target rejected = 7
```

### EAGLE3 with intercept-all (falls back)

```sh
./build/bin/llama-server \
  -m /home/samueldoyle/AI_LOCAL/Models/eagle3/Qwen3-8B-eagle3-Q4_K_M.gguf \
  -md /home/samueldoyle/AI_LOCAL/Models/eagle3/Qwen3-8B-speculator.eagle3-F16.gguf \
  --spec-type draft-eagle3 \
  --spec-draft-n-max 4 \
  --det-draft-model external/plugins/lib/libdeterministic_draft_spec_plugin.so \
  --det-draft-intercept-all \
  --reasoning off --temp 0 --top-k 1 --top-p 1.0 \
  -c 8192 -ngl 99 -fa on --port 8080
```

Warning: `--det-draft-intercept-all requires the standard autoregressive MTP flow. Falling back to default mode with the bonus token also filtered (target still verifies, so performance may degrade)`

Result (actual output):

```
istringstream
def gfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfuncfunc
```

```
draft acceptance = 0.64444 (29 accepted / 45 generated), mean len = 2.93
statistics draft-deterministic: #drafts = 18, #truncated = 8, #tokens pre = 69, #tokens post = 45, #target rejected = 16
```

The intercept-all output keeps letters and whitespace but drops digits and punctuation: MTP constrains every token because it skips verification, while the fallback types constrain only the bonus token. With this trivial constraint the model still degenerates into a repeated-word loop, so intercept-all output is not representative of a real grammar.

### Hard fail: no --spec-type

```sh
llama-server \
  -m /home/samueldoyle/AI_LOCAL/Models/MTP/model_q8_0.gguf \
  --det-draft-model external/plugins/lib/libdeterministic_draft_spec_plugin.so \
  -c 8192 -ngl 99 -fa on --port 8080
```

Expected: server exits with error `--det-draft-model requires a compatible drafter type (--spec-type draft-mtp, draft-eagle3, draft-dspark, or draft-dflash)`

## Unit Tests

Build and run the deterministic draft unit tests:

```sh
cmake --build build --target test-deterministic-draft
./build/bin/test-deterministic-draft
```

The test suite covers:
- Plugin loader lifecycle (init/free with valid and invalid paths)
- C API wrappers (get_capabilities, set_vocab, fill_bitmask, commit, reset)
- Speculative type enum and params struct
- Intercept-all flag validation (requires plugin + compatible drafter type)
- Intercept-all fallback for non-MTP draft types (dspark, eagle3 -> falls back with bonus filtered; mtp -> stays enabled)
- No --spec-type hard fail
- Plugin state across reset
- filter_draft, apply_bitmask, rollback, commit_tokens
- State serialization round-trip
- Bootstrap detection (per-slot grammar resolution)

## Plugin Interface

Implement the contract from `include/deterministic_draft_plugin.h`:

| Function                         | Description                 |
|----------------------------------|-----------------------------|
| `deterministic_draft_create`     | Allocate plugin state       |
| `deterministic_draft_destroy`    | Free state (NULL-safe)      |
| `deterministic_draft_set_vocab`  | Register token strings      |
| `deterministic_draft_fill_bitmask` | Apply regex to stored vocab |
| `deterministic_draft_commit`     | No-op (stateless)           |
| `deterministic_draft_reset`      | Clear per-slot state        |
| `deterministic_draft_get_capabilities` | BITMASK only           |
