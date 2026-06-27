# Deterministic Draft Spec SDK

SDK for building and loading deterministic draft plugins (.so/.dylib/.dll) into llama.cpp.

## Overview

A deterministic draft filter constrains speculative decoding to a grammar: every token the drafter proposes is checked against a set of allowed tokens, and the draft is cut short at the first token that does not match.

This repo ships one reference plugin, `deterministic_regex_plugin`, that implements the contract with a single constraint: letters and whitespace only. It is intentionally trivial - its only purpose is to demonstrate the plugin interface, not to be a useful grammar. A real plugin would plug in a full grammar engine (regex, JSON schema, BNF, and so on).

The filter cuts a draft at the first token that fails the rule, so how much it cuts depends on the content. The demo rule admits only letters and whitespace, and on the prose used below the cut lands three to five tokens into a sixteen-token draft - mostly on the drafter's own bad guesses (digits, punctuation) that the target rejects anyway. That is what the `#truncated` and `#target rejected` numbers in the results reflect.

Two operating modes shown below, plus how a plugin declines a round its rule does not describe:

- **Default mode** - the target model still verifies every surviving draft token; the filter constrains only the draft head.
- **Intercept-all mode** (`--det-draft-intercept-all`) - the filter is the sole verifier; works with any compatible drafter (mtp, eagle3, dspark, dflash).
- **Declining a round** - a plugin whose rule does not describe the current content returns `DETERMINISTIC_DRAFT_REST_SKIP` instead of truncating, and the round runs as plain speculative decoding. Without it, a filter loaded for one domain mangles output from another: a letters-only rule applied to a numeric answer truncates every draft and forces the bonus token onto a rule-valid token, so the model is pushed into emitting rule content and can only escape through EOG.

The reference plugin decides this for itself from its own match rate over a rolling window (`DET_DRAFT_REGEX_WINDOW`, default 256 draft tokens; `DET_DRAFT_REGEX_MIN_MATCH`, default 0.5). It starts with no opinion and enforces only once a window says the rule applies.

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

Single run per cell, `max_tokens 256`, greedy (`--temp 0 --top-k 1 --top-p 1.0`), RTX 4070, `--spec-draft-n-max 16` for all drafter types. Same prompt for every cell.

| drafter | `--spec-type` | target model | draft model (`-md`) | baseline (no plugin) | default filter | intercept-all |
|---|---|---|---|---|---|---|
| MTP | `draft-mtp` | Qwen3.5-4B.Q4_K_M (target's own MTP head) | - | 43.57 t/s | 43.30 t/s | 44.15 t/s |
| DSpark | `draft-dspark` | Qwen3 4B | Dspark_Qwen3_4B_Block7 | 136.34 t/s | 111.56 t/s | 139.08 t/s |
| Eagle3 | `draft-eagle3` | Qwen3 8B Awq Compatible Instruct | Qwen3 8B Speculator.Eagle3 | 32.92 t/s | 32.98 t/s | 32.78 t/s |

The plugin declines nearly every round on this prompt, so the two plugin columns track the baseline. The reference rule is letters and whitespace only, the rolling match rate stays under the 0.5 default, and a declined round runs as plain speculative decoding. This is the fallback working as designed, and it is the result to expect on any prompt the rule does not describe.

The constrained path is still there. Forcing the rule on (`DET_DRAFT_REGEX_MIN_MATCH=0 DET_DRAFT_REGEX_WINDOW=1`) restores it, and intercept-all then drops the target veto outright:

| drafter | intercept-all (forced) |
|---|---|
| MTP | 199.25 t/s |
| DSpark | 242.60 t/s |
| Eagle3 | 125.81 t/s |

Read both tables as indicative only: they use the trivial `^[A-Za-z]+$` reference plugin, not a real grammar, so the forced outputs degenerate into repeated-word loops (see "Intercept-All Mode"). The default filter is also slower than the no-plugin baseline on DSpark - the constraint truncates drafts and adds per-token filter work, and the target rejects many of the survivors.

These are regex-plugin numbers on current HEAD. They are not comparable to earlier benchmark figures, which used a full XGrammar plugin (five languages) on a pre-MTPv2 build with a different baseline. Read the tables as an internal comparison, not as a plugin speedup over that earlier work.

## Verify the Filter

The filter allows only tokens made of letters or whitespace, but it constrains only the **draft head**. In default mode (no `--det-draft-intercept-all`), the target model verifies every draft token and emits the bonus/correction token **unconstrained**, so the output keeps punctuation and any other character from the target's own choice.

The reference plugin also declines a round when its rolling match rate says the rule does not describe the content. On the prose prompt below it declines nearly every round, so the three runs show the fallback: clean prose, at baseline speed, with a high `#skipped` count.

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
, in a small village nestled between rolling hills and whispering forests, there lived a young girl named Elara. Elara was known for her curiosity and her ability to see beauty in the most ordinary things. While other children played with toys, Elara spent her days collecting fallen leaves, observing the patterns on spider webs, and listening to the songs of the birds.

One crisp autumn morning, Elara found something unusual. Hidden beneath a pile of dry leaves, she discovered a small, glowing orb that pulsed with a soft, golden light. It was unlike anything she had ever seen before. Intrigued, she picked it up, and the orb seemed to respond to her touch, warming her palm.

As the days passed, the orb grew brighter, and Elara noticed that it had the power to make the world around her more vibrant. Flowers bloomed in the middle of the path, and the colors of the leaves intensified. But wi
```

```
draft acceptance = 0.09158 ( 150 accepted / 1638 generated), mean len = 2.44
statistics draft-deterministic: #drafts = 104, #truncated = 0, #skipped = 104, #tokens pre = 1638, #tokens post = 1638, #target rejected = 1488
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
, there was a young girl named Lila who lived in a small village surrounded by a dense forest. The villagers often spoke of the forest as a place of mystery and danger, but Lila was different. She was curious and brave, and she often ventured into the woods to explore. One day, while exploring the forest, she stumbled upon a hidden glade. In the center of the glade was a shimmering pool of water. As she approached, she noticed that the water was not just water—it was alive. It sparkled with a magical light, and it seemed to respond to her presence. Lila knelt by the pool and touched the water. Suddenly, a voice echoed in her mind, "You have found the Heart of the Forest." The voice was calm and wise, and it spoke of the forest's magic and the importance of balance. Lila realized that the pool was a source of magic that connected the forest to the world beyond. She understood that she had
```

```
draft acceptance = 0.35808 ( 164 accepted / 458 generated), mean len = 2.95
statistics draft-deterministic: #drafts = 90, #truncated = 39, #skipped = 36, #tokens pre = 619, #tokens post = 458, #target rejected = 294
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
, in a small village nestled between the hills of the Valley of Whispers, there lived a curious young girl named Lila. The village was known for its ancient library, which held the secrets of the world. Lila had always been fascinated by the stories within the library's walls, and she spent her days exploring the dusty shelves and reading the old tomes.

One day, while searching for a book on the stars, Lila stumbled upon a hidden room in the library. The door was slightly ajar, and as she stepped inside, the air shimmered with a soft glow. In the center of the room stood a pedestal, and on it was a small, glowing orb. As she reached out to touch it, the orb pulsed with light and a voice echoed in her mind: "To awaken the power of the stars, you must find the three lost constellations."

Lila's heart raced with excitement. She had heard tales of the constellations being lost in the depth
```

```
draft acceptance = 0.07910 ( 140 accepted / 1770 generated), mean len = 2.22
statistics draft-deterministic: #drafts = 115, #truncated = 0, #skipped = 115, #tokens pre = 1770, #tokens post = 1770, #target rejected = 1630
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

**`statistics draft-deterministic: #drafts / #truncated / #skipped / #tokens pre / #tokens post / #target rejected`**

- `#drafts` = draft batches the filter examined.
- `#truncated` = of those, how many the filter cut short because a token contained a non-letter, non-whitespace byte.
- `#skipped` = of those, how many the plugin declined. A declined round runs as plain speculative decoding: the draft is kept whole, the target samples and verifies it, and the bonus token is unconstrained. The reference plugin declines a round while its rolling match rate says the rule does not describe the content.
- `#tokens pre` = tokens proposed before filtering; `#tokens post` = tokens that survived the regex. A skipped round contributes the same count to both.
- `#target rejected` = of the surviving tokens, how many the target model then rejected during verification.

The two lines reconcile: `#tokens post == generated == accepted + #target rejected`. A high `#target rejected` is normal with a noisy drafter: the filter keeps a prefix only up to the first token its rule rejects, so the surviving tail is short and the target discards much of it.

## Intercept-All Mode

The `--det-draft-intercept-all` flag skips target model verification - the plugin is the sole verifier. Works with any compatible drafter (mtp, eagle3, dspark, dflash). The filter truncates invalid drafts before the target batch is built, so those tokens never reach the target forward; the target forward over the surviving tokens still runs (it populates the target KV cache and feeds drafters that consume target layer inputs).

On the prose prompt below the reference plugin declines nearly every round by default (see "Throughput"), so a plain intercept-all run behaves like the baseline and proves nothing about the mode. The runs below force the rule on with `DET_DRAFT_REGEX_MIN_MATCH=0 DET_DRAFT_REGEX_WINDOW=1`: the first makes the plugin enforce from the moment it has an opinion, the second closes that opinion after one draft instead of the 256-token default. That is a knob setting for demonstration only, not a recommended configuration.

Greedy sampling (`--temp 0 --top-k 1 --top-p 1.0`) makes the run deterministic. The same request as "Verify the Filter" is used:

```sh
curl -s http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"# Write a short story\nOnce upon a time","max_tokens":256}'
```

Each run below uses `--spec-draft-n-max 16`.

### MTP with intercept-all (enabled)

```sh
DET_DRAFT_REGEX_MIN_MATCH=0 DET_DRAFT_REGEX_WINDOW=1 \
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
DET_DRAFT_REGEX_MIN_MATCH=0 DET_DRAFT_REGEX_WINDOW=1 \
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
DET_DRAFT_REGEX_MIN_MATCH=0 DET_DRAFT_REGEX_WINDOW=1 \
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

The intercept-all output keeps letters and whitespace but drops punctuation: every emitted token is filter-constrained, because intercept-all keeps all filter-surviving tokens without target veto. With this trivial `^[A-Za-z]+$` constraint the model still degenerates into a repeated-word loop, so intercept-all output is not representative of a real grammar. Note that the mode does not merely cut drafts: it constrains the bonus token too, so a constraint the content does not fit will pull the generation toward whatever the rule does admit. That is the failure the plugin's decline-a-round fallback exists to prevent, and why the default run above stays on the baseline.

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

The script inherits the environment, so pass the forcing knobs the same way to make its intercept-all runs demonstrate the mode rather than decline:

```sh
DET_DRAFT_REGEX_MIN_MATCH=0 DET_DRAFT_REGEX_WINDOW=1 \
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
- Declining a round: the match-rate window and the env knobs that tune it
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
| `deterministic_draft_filter_draft` | Keep the valid prefix; `rest_action` says ENFORCE or SKIP |
| `deterministic_draft_commit`     | No-op (stateless)           |
| `deterministic_draft_reset`      | Clear per-slot state        |
| `deterministic_draft_get_capabilities` | BITMASK only           |
