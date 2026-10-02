# Deterministic Draft Spec SDK

SDK for building and loading deterministic draft plugins (.so/.dylib/.dll) into llama.cpp.

## Overview

A deterministic draft filter constrains speculative decoding to a grammar: every token the drafter proposes is checked against a set of allowed tokens, and the draft is cut short at the first token that does not match.

This repo ships one reference plugin, `deterministic_regex_plugin`, that implements the contract with a single constraint: letters and whitespace only. It is intentionally trivial - its only purpose is to demonstrate the plugin interface, not to be a useful grammar. A real plugin would plug in a full grammar engine (regex, JSON schema, BNF, and so on).

Because the constraint admits only letters and whitespace, every run below discards the digits and punctuation that real code needs. That is the filter working as designed, not a defect - the high `#truncated` and `#target rejected` numbers in the results are a direct consequence.

Two operating modes shown below:

- **Default mode** - the target model still verifies every surviving draft token; the filter constrains only the draft head.
- **Intercept-all mode** (`--det-draft-intercept-all`) - the filter is the sole verifier; works with any compatible drafter (mtp, eagle3, dspark, dflash).

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
  --spec-draft-n-max 16 \
  --det-draft-model external/plugins/lib/libdeterministic_draft_spec_plugin.so \
  -c 8192 -ngl 99 -fa on --jinja --port 8080
```

### 2. DSPARK

```sh
llama-server \
  -m /home/samueldoyle/AI_LOCAL/Models/dspark/Qwen3-4B-DSpark-Model-Q8_0.gguf \
  -md /home/samueldoyle/AI_LOCAL/Models/dspark/Qwen3-4B-DSpark-Q8_0.gguf \
  --spec-type draft-dspark \
  --spec-draft-n-max 16 \
  --det-draft-model external/plugins/lib/libdeterministic_draft_spec_plugin.so \
  -c 8192 -ngl 99 -fa on --jinja --port 8080
```

For DSPARK, `-m` is the Model file (`Qwen3-4B-DSpark-Model-Q8_0.gguf`) and `-md` is the draft (`Qwen3-4B-DSpark-Q8_0.gguf`). The "Model" suffix file is the TARGET.

### 3. MTP (Multi-Token Prediction)

```sh
llama-server \
  -m /home/samueldoyle/AI_LOCAL/Models/MTP/Qwen3.5-4B.Q4_K_M.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 16 \
  --det-draft-model external/plugins/lib/libdeterministic_draft_spec_plugin.so \
  -c 8192 -ngl 99 -fa on --jinja --port 8080
```

## Throughput

Median of 3 runs, `max_tokens 256`, greedy (`--temp 0 --top-k 1 --top-p 1.0`), RTX 4070, `--spec-draft-n-max 16` for all drafter types. Same prompt for every cell.

| drafter | `--spec-type` | target model | draft model (`-md`) | baseline (no plugin) | default filter | intercept-all |
|---|---|---|---|---|---|---|
| MTP | `draft-mtp` | Qwen3.5-4B.Q4_K_M (target's own MTP head) | - | 49.85 t/s | 43.47 t/s | 260.80 t/s |
| DSpark | `draft-dspark` | Qwen3 4B | Dspark_Qwen3_4B_Block7 | 158.63 t/s | 123.72 t/s | 304.32 t/s |
| Eagle3 | `draft-eagle3` | Qwen3 8B Awq Compatible Instruct | Qwen3 8B Speculator.Eagle3 | 36.80 t/s | 33.10 t/s | 198.54 t/s |

Read this as indicative only: it uses the trivial `^[A-Za-z]+$` reference plugin, not a real grammar, so the outputs degenerate into repeated-word loops. The default filter is slower than the no-plugin baseline on all three drafters - the constraint truncates drafts and adds per-token filter work, and the target rejects many of the survivors. Intercept-all removes the target veto outright, so every filter-surviving token is kept and the per-round token count rises; that is where the speedup comes from.

These are regex-plugin numbers on current HEAD. They are not comparable to earlier benchmark figures, which used a full XGrammar plugin (five languages) on a pre-MTPv2 build with a different baseline. Read the table as an internal default-vs-intercept-all comparison, not as a plugin speedup over that earlier work.

## Verify the Filter

The filter allows only tokens made of letters or whitespace, but it constrains only the **draft head**. In default mode (no `--det-draft-intercept-all`), the target model verifies every draft token and emits the bonus/correction token **unconstrained**, so the output keeps punctuation and any other character from the target's own choice.

Greedy sampling (`--temp 0 --top-k 1 --top-p 1.0`) makes the run deterministic. The same request is used for all three drafter types:

```sh
curl -s http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"# Write a short story\nOnce upon a time","max_tokens":256}'
```

The `statistics draft-deterministic` line reports draft-head activity only. Each run below uses `--spec-draft-n-max 16`.

### draft-mtp

Start:

```sh
./build/bin/llama-server \
  -m /home/samueldoyle/AI_LOCAL/Models/MTP/Qwen3.5-4B.Q4_K_M.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 16 \
  --det-draft-model external/plugins/lib/libdeterministic_draft_spec_plugin.so \
  --reasoning off --temp 0 --top-k 1 --top-p 1.0 \
  -c 8192 -ngl 99 -fa on --jinja --port 8080
```

Result (actual output):

```
, in a small village nestled between rolling hills and whispering forests, there lived a young girl named Elara. Elara was known for her curiosity and her ability to see the world in a way that others couldn't. While others saw only the ordinary, Elara saw the extraordinary.

One morning, as the sun rose over the village, Elara noticed something peculiar. A small, glowing blue flower was blooming in the middle of the village square, where there was supposed to be a stone fountain. The villagers were busy with their morning chores, but Elara couldn't ignore it. She walked towards the flower, her heart pounding with excitement.

As she approached the flower, she felt a strange warmth radiating from it. Suddenly, the flower began to glow brighter, and a soft voice echoed in her mind. "You have the eyes to see the hidden magic," the voice said. "I am the Guardian of the Village, and I need your help."

Elara listened intently, her curiosity piqued. "What do you need?" she asked.

The Guardian explained that the village's well had run dry, and the villagers were suffering from thirst. The Guardian revealed that the well was cursed, and the only way to break the
```

```
draft acceptance = 0.20593 (125 accepted / 607 generated), mean len = 2.16
statistics draft-deterministic: #drafts = 129, #truncated = 125, #tokens pre = 2017, #tokens post = 607, #target rejected = 482
```

### draft-dspark

Start:

```sh
./build/bin/llama-server \
  -m /home/samueldoyle/AI_LOCAL/Models/dspark/Qwen3-4B-DSpark-Model-Q8_0.gguf \
  -md /home/samueldoyle/AI_LOCAL/Models/dspark/Qwen3-4B-DSpark-Q8_0.gguf \
  --spec-type draft-dspark \
  --spec-draft-n-max 16 \
  --det-draft-model external/plugins/lib/libdeterministic_draft_spec_plugin.so \
  --reasoning off --temp 0 --top-k 1 --top-p 1.0 \
  -c 8192 -ngl 99 -fa on --jinja --port 8080
```

Result (actual output):

```
, there was a young girl named Lila who lived in a small village surrounded by a dense forest. The villagers often spoke of the forest as a place of mystery and danger, but Lila was different. She was curious and brave, and she often ventured into the woods to explore. One day, while exploring the forest, she stumbled upon a hidden glade. In the center of the glade was a shimmering pool of water. As she approached, she noticed that the water was not just water—it was alive. It sparkled with a magical light, and it seemed to respond to her presence. Lila knelt by the pool and touched the water. Suddenly, a voice echoed in her mind, "You have found the Heart of the Forest." Lila was confused but intrigued. She asked, "What is the Heart of the Forest?" The voice replied, "It is the soul of the forest, the source of all life and magic. It has been hidden for centuries, and only those with pure intentions can see it." Lila felt a warm glow spread through her body. She realized that she had been chosen for a purpose. The voice continued, "You must find the three keys to the Heart of the Forest. Each key is guarded by
```

```
draft acceptance = 0.38384 (152 accepted / 396 generated), mean len = 2.77
statistics draft-deterministic: #drafts = 102, #truncated = 77, #tokens pre = 711, #tokens post = 396, #target rejected = 244
```

### draft-eagle3

Start:

```sh
./build/bin/llama-server \
  -m /home/samueldoyle/AI_LOCAL/Models/eagle3/Qwen3-8B-eagle3-Q4_K_M.gguf \
  -md /home/samueldoyle/AI_LOCAL/Models/eagle3/Qwen3-8B-speculator.eagle3-F16.gguf \
  --spec-type draft-eagle3 \
  --spec-draft-n-max 16 \
  --det-draft-model external/plugins/lib/libdeterministic_draft_spec_plugin.so \
  --reasoning off --temp 0 --top-k 1 --top-p 1.0 \
  -c 8192 -ngl 99 -fa on --jinja --port 8080
```

Result (actual output):

```
, in a small village nestled between the mountains and the sea, there lived a curious young girl named Lila. She was known for her love of adventure and her ability to find treasures in the most unexpected places. One day, while exploring the forest near her home, she stumbled upon a hidden cave.

As she entered the cave, she noticed a glowing light at the end of the tunnel. Curious, she followed it and discovered a small room filled with ancient artifacts. In the center of the room was a pedestal with a mysterious box. Lila approached the box and, with a deep breath, opened it to find a map leading to a long-lost treasure.

Excited, Lila carefully studied the map and set off on her journey. She faced many challenges along the way, including treacherous terrain and riddles that tested her wit. But with each obstacle, she grew stronger and more determined. Finally, after days of searching, she reached the location marked on the map.

There, she found a chest filled with gold and jewels. But instead of taking the treasure, she decided to return it to the cave, believing that the real treasure was the journey itself. She returned home, wiser and more fulfilled than ever before.

And so, Lila
```

```
draft acceptance = 0.16576 (122 accepted / 736 generated), mean len = 2.09
statistics draft-deterministic: #drafts = 133, #truncated = 114, #tokens pre = 2078, #tokens post = 736, #target rejected = 614
```

DSPARK may log the following, which is informational, not an error:

```
requested draft size (n_max=16, n_min=0) exceeds the trained block size 7 -- clamping to 7
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

The `--det-draft-intercept-all` flag skips target model verification - the plugin is the sole verifier. Works with any compatible drafter (mtp, eagle3, dspark, dflash). The filter truncates invalid drafts before the target batch is built, so those tokens never reach the target forward; the target forward over the surviving tokens still runs (it populates the target KV cache and feeds drafters that consume target layer inputs).

Greedy sampling (`--temp 0 --top-k 1 --top-p 1.0`) makes the run deterministic. The same request as "Verify the Filter" is used:

```sh
curl -s http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"# Write a short story\nOnce upon a time","max_tokens":256}'
```

Each run below uses `--spec-draft-n-max 16`.

### MTP with intercept-all (enabled)

```sh
./build/bin/llama-server \
  -m /home/samueldoyle/AI_LOCAL/Models/MTP/Qwen3.5-4B.Q4_K_M.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 16 \
  --det-draft-model external/plugins/lib/libdeterministic_draft_spec_plugin.so \
  --det-draft-intercept-all \
  --reasoning off --temp 0 --top-k 1 --top-p 1.0 \
  -c 8192 -ngl 99 -fa on --jinja --port 8080
```

Log: `--det-draft-intercept-all is enabled for draft-mtp`

Result (actual output):

```
 in a village called the Valley of Oakwood lived a village called the Valley of Oakwood lived a village called the Valley of Oakwood lived a young man named Zazoo who lived in the Valley of Oakwood lived a young man named Zazoo who lived in the Valley of Oakwood lived a young man named Zazoo who lived in the Valley of Oakwood lived a young man named Zazoo who lived in the Valley of Oakwood lived a young man named Zazoo who lived in the Valley of Oakwood lived a young man named Zazoo who lived in the Valley of Oakwood lived a young man named Zazoo who lived in the Valley of Oakwood lived a young man named Zazoo who lived in the Valley of Oakwood lived a young man named Zazoo who lived in the Valley of Oakwood lived a young man named Zazoo who lived in the Valley of Oakwood lived a young man named Zazoo who lived in the Valley of Oakwood lived a young man named Zazoo who lived in the Valley of Oakwood lived a young man named Zazoo who lived in the Valley of Oakwood lived a young man named Zazoo who lived in the Valley of Oakwood lived a young man named
```

```
draft acceptance = 1.00000 (235 accepted / 235 generated), mean len = 14.06
statistics draft-deterministic: #drafts = 20, #truncated = 7, #tokens pre = 311, #tokens post = 235, #target rejected = 0
```

### DSPARK with intercept-all (enabled)

```sh
./build/bin/llama-server \
  -m /home/samueldoyle/AI_LOCAL/Models/dspark/Qwen3-4B-DSpark-Model-Q8_0.gguf \
  -md /home/samueldoyle/AI_LOCAL/Models/dspark/Qwen3-4B-DSpark-Q8_0.gguf \
  --spec-type draft-dspark \
  --spec-draft-n-max 16 \
  --det-draft-model external/plugins/lib/libdeterministic_draft_spec_plugin.so \
  --det-draft-intercept-all \
  --reasoning off --temp 0 --top-k 1 --top-p 1.0 \
  -c 8192 -ngl 99 -fa on --jinja --port 8080
```

Log: `--det-draft-intercept-all is enabled for draft-dspark`

Result (actual output):

```
 in a land where the misty hills were covered by the stars and the sky was a canvas of endless blue and silver light

There was a curious creature named Luma who lived in the forested in the heart of the forest

Luma was a creature of the night and the stars

Luma had a shimmering the skin that glowed like the stars in the night sky

Luma was a small creature with a body that shimmered like the stars

Luma and eyes that sparkled like the night sky

Luma was a creature that the forest that lived in the forest

Luma and the forest that lived in the forest

Luma was a creature that lived in the forest

Luma was a creature that lived in the forest

Luma was a creature that lived in the forest

Luma was a creature that lived in the forest

Luma was a creature that lived in the forest

Luma was a creature that lived in the forest

Luma was a creature that lived in the forest

Luma was a creature that lived in the forest

Luma was a creature that lived in the forest

Luma was a creature that lived in the forest

Luma was a creature that lived in the forest

Luma was a
```

```
draft acceptance = 1.00000 (217 accepted / 217 generated), mean len = 7.03
statistics draft-deterministic: #drafts = 38, #truncated = 9, #tokens pre = 260, #tokens post = 217, #target rejected = 0
```

### EAGLE3 with intercept-all (enabled)

```sh
./build/bin/llama-server \
  -m /home/samueldoyle/AI_LOCAL/Models/eagle3/Qwen3-8B-eagle3-Q4_K_M.gguf \
  -md /home/samueldoyle/AI_LOCAL/Models/eagle3/Qwen3-8B-speculator.eagle3-F16.gguf \
  --spec-type draft-eagle3 \
  --spec-draft-n-max 16 \
  --det-draft-model external/plugins/lib/libdeterministic_draft_spec_plugin.so \
  --det-draft-intercept-all \
  --reasoning off --temp 0 --top-k 1 --top-p 1.0 \
  -c 8192 -ngl 99 -fa on --jinja --port 8080
```

Log: `--det-draft-intercept-all is enabled for draft-eagle3`

Result (actual output):

```
 in a small land called The Land of Whispers and there lived lived lived a very curious little creature named Whisper the Wisp

Whisper was Wispy and had One very curious little creature named Whisper the Wispy and had a very curious little creature named Whisper the Wispy and had One very curious little creature named Whisper the Wispy and had One very curious little creature named Whisper the Wispy and had One very curious little creature named Whisper the Wispy and had One very curious little creature named Whisper the Wispy and had One very curious little creature named Whisper the Wispy and had One very curious little creature named Whisper the Wispy and had One very curious little creature named Whisper the Wispy and had One very curious little creature named Whisper the Wispy and had One very curious little creature named Whisper the Wispy and had One very curious little creature named Whisper the Wispy and had One very curious little creature named Whisper the Wispy and had One very curious little creature named Whisper the Wispy and had One very curious little creature named Whisper the Wispy and had One very curious little creature named Whisper the Wis
```

```
draft acceptance = 1.00000 (230 accepted / 230 generated), mean len = 11.95
statistics draft-deterministic: #drafts = 25, #truncated = 12, #tokens pre = 387, #tokens post = 230, #target rejected = 0
```

The intercept-all output keeps letters and whitespace but drops punctuation: every emitted token is filter-constrained, because intercept-all keeps all filter-surviving tokens without target veto. With this trivial `^[A-Za-z]+$` constraint the model still degenerates into a repeated-word loop, so intercept-all output is not representative of a real grammar.

### Hard fail: no --spec-type

```sh
llama-server \
  -m /home/samueldoyle/AI_LOCAL/Models/MTP/Qwen3.5-4B.Q4_K_M.gguf \
  --det-draft-model external/plugins/lib/libdeterministic_draft_spec_plugin.so \
  -c 8192 -ngl 99 -fa on --jinja --port 8080
```

Expected: server exits with error `--det-draft-model requires a compatible drafter type (--spec-type draft-mtp, draft-eagle3, draft-dspark, or draft-dflash)`

## Run All Scenarios

`external/scripts/run-draft-scenarios.sh` runs the six scenarios above (3 drafters x default/intercept-all) end to end: it starts `llama-server` with the reference plugin, issues the request, and prints the output plus the `draft acceptance` / `statistics draft-deterministic` lines.

```sh
# models rooted at $DET_MODELS (default: /home/samueldoyle/AI_LOCAL/Models)
DET_MODELS=/path/to/Models external/scripts/run-draft-scenarios.sh

# or run a subset
external/scripts/run-draft-scenarios.sh ia-mtp ia-dspark ia-eagle3
```

Requires a built `build/bin/llama-server` and `external/plugins/lib/libdeterministic_draft_spec_plugin.so` (see Build above).

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
- Intercept-all support across all compatible drafter types (mtp, dspark, eagle3, dflash)
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
