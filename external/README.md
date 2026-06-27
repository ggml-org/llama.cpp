# Deterministic Draft Spec SDK

SDK for building and loading deterministic draft plugins (.so/.dylib/.dll) into llama.cpp.

## Overview

A deterministic draft filter constrains speculative decoding to a grammar: every token the drafter proposes is checked against a set of allowed tokens, and the draft is cut short at the first token that does not match.

This repo ships one reference plugin, `deterministic_regex_plugin`, that implements the contract with a single constraint: letters and whitespace only. It is intentionally trivial - its only purpose is to demonstrate the plugin interface, not to be a useful grammar. A real plugin would plug in a full grammar engine (regex, JSON schema, BNF, and so on).

The filter cuts a draft at the first token that fails the rule, so how much it cuts depends on the content. On the prose used below the rule is enforced in only part of the rounds, and the cut is short when it happens: the DSPark run removed 192 of 676 proposed tokens (676 -> 484) over 47 of 97 rounds, Eagle3 removed 146 of 1690 (1690 -> 1544) over 12 of 108 rounds, and MTP removed nothing (0 of 98 rounds truncated, every round declined). That is what the `#truncated` and `#target rejected` numbers in the results reflect.

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
cmake -B build -DDETERMINISTIC_SPEC_ENABLED=ON -DCMAKE_BUILD_TYPE=Release
# or specific to build with CUDA for example
cmake -B build -DDETERMINISTIC_SPEC_ENABLED=ON -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda \
-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_BUILD_TYPE=Release \
-DBUILD_SHARED_LIBS=OFF -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_OPENSSL=ON
cmake --build build --config Release
```

`-DDETERMINISTIC_SPEC_ENABLED=ON` is not optional: the root `CMakeLists.txt` declares it as an option and only calls `add_subdirectory(external)` when it is on, so the target in step 2 does not exist without it.

### 2. Build the SDK + plugin

From the repo root:

```sh
cmake --build build --target deterministic_draft_spec_plugin
```

Outputs:

- `external/lib/libdeterministic_draft_spec.so` - SDK loader library
- `external/plugins/lib/libdeterministic_draft_spec_plugin.so` - Regex filter plugin (the `--det-draft-model` path used below)

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

Single run per cell, `max_tokens 256`, greedy (`--temp 0 --top-k 1 --top-p 1.0`), RTX 4070, `--spec-draft-n-max 16` for all drafter types. Same prompt for every cell. Numbers are from the gguf files named in the table, on the commit that ships this README; they are not comparable to runs made with other weights.

| drafter | `--spec-type` | target model (`-m`) | draft model (`-md`) | baseline (no plugin) | default filter | intercept-all |
|---|---|---|---|---|---|---|
| MTP | `draft-mtp` | `Qwen3.5-4B.Q4_K_M.gguf` (target's own MTP head) | - | 65.96 t/s | 61.64 t/s | 59.55 t/s |
| DSpark | `draft-dspark` | `Qwen3-4B-DSpark-Model-Q8_0.gguf` | `Qwen3-4B-DSpark-Q8_0.gguf` | 164.95 t/s | 131.78 t/s | 201.92 t/s |
| Eagle3 | `draft-eagle3` | `Qwen3-8B-eagle3-Q4_K_M.gguf` | `Qwen3-8B-speculator.eagle3-F16.gguf` | 46.76 t/s | 47.55 t/s | 42.39 t/s |

The plugin declines most rounds on this prompt: every round for MTP (98 of 98), 92 of 108 for Eagle3, 36 of 97 for DSpark. A declined round runs as plain speculative decoding, so the default-filter column stays near the baseline. The DSpark column is the exception: of its 61 enforced rounds, 47 truncate and remove 192 tokens, so the default run is well below baseline and the unforced intercept-all run is well above it, because intercept-all skips target verification of the surviving draft tokens. That unforced intercept-all DSpark run scores `draft acceptance = 0.52368 (188 accepted / 359 generated), #target rejected = 171` - not 1.0, because the 36 declined rounds still go through target verification.

The constrained path is still there. Forcing the rule on (`DET_DRAFT_REGEX_MIN_MATCH=0 DET_DRAFT_REGEX_WINDOW=1`) restores it, and intercept-all then drops the target veto outright:

| drafter | intercept-all (forced) |
|---|---|
| MTP | 289.54 t/s |
| DSpark | 288.01 t/s |
| Eagle3 | 201.89 t/s |

Read both tables as indicative only: they use the trivial letters-and-whitespace reference plugin, not a real grammar, so the forced outputs degenerate into repeated-word loops (see "Intercept-All Mode"). The default filter costs throughput where it actually enforces: 61.64 against a 65.96 baseline on MTP and 131.78 against 164.95 on DSpark - the constraint truncates drafts and adds per-token filter work, while the target still rejects many of the survivors. On Eagle3 the two are within run-to-run noise (47.55 against 46.76).

These are regex-plugin numbers on current HEAD. They are not comparable to earlier benchmark figures, which used a full XGrammar plugin (five languages) on a pre-MTPv2 build with a different baseline. Read the tables as an internal comparison, not as a plugin speedup over that earlier work.

## Verify the Filter

The filter allows only tokens made of letters or whitespace, but it constrains only the **draft head**. In default mode (no `--det-draft-intercept-all`), the target model verifies every draft token and emits the bonus/correction token **unconstrained**, so the output keeps punctuation and any other character from the target's own choice.

The reference plugin also declines a round when its rolling match rate says the rule does not describe the content. On the prose prompt below it declines most rounds (98 of 98 for MTP, 92 of 108 for Eagle3, 36 of 97 for DSPark), so the three runs are mostly the fallback: clean prose, near baseline speed, with a high `#skipped` count.

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

One crisp autumn morning, Elara found something unusual in the center of the village square. It was a small, glowing orb that pulsed with a soft, golden light. As she approached it, the orb seemed to react to her presence, emitting a warm hum that resonated in her chest. Suddenly, the ground beneath her feet began to shift, and the orb dissolved into a swarm of tiny, luminous butterflies.

Elara watched in awe as the butterflies danced around her, their wings shimmering with colors she had never seen before. They led her on a journey through the village, transforming familiar streets into a vibrant tapestry of light and sound. The houses seemed to glow from within, and the trees cast dancing shadows that told stories of the village's history.

As the butterflies guided her, Elara realized that the orb was not just a magical object, but a portal to a
```

```
draft acceptance = 0.10221 (  157 accepted /  1536 generated), mean len =  2.60
statistics draft-deterministic: #drafts = 98, #truncated = 0, #skipped = 98, #tokens pre = 1536, #tokens post = 1536, #target rejected = 1379
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
draft acceptance = 0.32438 (  157 accepted /   484 generated), mean len =  2.78
statistics draft-deterministic: #drafts = 97, #truncated = 47, #skipped = 36, #tokens pre = 676, #tokens post = 484, #target rejected = 327
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

As she entered the cave, she was amazed by the glowing crystals that lined the walls. In the center of the cave, she found a small, ancient chest. Curious, she opened it to find a map inside. The map was old and faded, but it depicted a path leading to a legendary treasure said to be hidden deep within the mountains.

Determined to find the treasure, Lila set off on her journey. She faced many challenges along the way, including treacherous terrain, wild animals, and even a storm that nearly washed her away. But with each obstacle, she grew stronger and more determined.

After days of travel, she finally reached the mountain peak where the treasure was said to be hidden. As she approached, she noticed a large stone door with intricate carvings. She used her knowledge of ancient symbols to unlock the door, and with a loud creak, it opened.

Inside, she found a room filled with gold and jewels,
```

```
draft acceptance = 0.09456 (  146 accepted /  1544 generated), mean len =  2.40
statistics draft-deterministic: #drafts = 108, #truncated = 12, #skipped = 92, #tokens pre = 1690, #tokens post = 1544, #target rejected = 1398
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

On the prose prompt below the reference plugin declines most rounds by default (98 of 98 for MTP, 92 of 108 for Eagle3, 36 of 97 for DSPark; see "Throughput"), so a plain intercept-all run is mostly the baseline and demonstrates little about the mode. The runs below force the rule on with `DET_DRAFT_REGEX_MIN_MATCH=0 DET_DRAFT_REGEX_WINDOW=1`: the first makes the plugin enforce from the moment it has an opinion, the second closes that opinion after one draft instead of the 256-token default. That is a knob setting for demonstration only, not a recommended configuration.

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
 in a village called the Valley of Oakwood lived a village called the Valley of Oakwood lived a village called the Valley of Oakwood lived a young man named Zazil who lived in the Valley of Oakwood lived a young man named Zazil who lived in the Valley of Oakwood lived a young man named Zazil who lived in the Valley of Oakwood lived a young man named Zazil who lived in the Valley of Oakwood lived a young man named Zazil who lived in the Valley of Oakwood lived a young man named Zazil who lived in the Valley of Oakwood lived a young man named Zazil who lived in the Valley of Oakwood lived a young man named Zazil who lived in the Valley of Oakwood lived a young man named Zazil who lived in the Valley of Oakwood lived a young man named Zazil who lived in the Valley of Oakwood lived a young man named Zazil who lived in the Valley of Oakwood lived a young man named Zazil who lived in the Valley of Oakwood lived a young man named Zazil who lived in the Valley of Oakwood lived a young man named Zazil who lived in the Valley of Oakwood lived a young man named
```

```
draft acceptance = 1.00000 (  235 accepted /   235 generated), mean len = 14.06
statistics draft-deterministic: #drafts = 20, #truncated = 10, #skipped = 0, #tokens pre = 320, #tokens post = 235, #target rejected = 0
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

There was a curious time traveler named Lira who had a mysterious past the only she could uncover was the key to her own heart

Lira had always been a small figure in the world of the world of time travel but she had a different kind of the power that was not just about the past or the future but about the present and the future

She had a small device that could not just a small device but a time to see the time of the present in the time of the future

Lira was a young girl who could see the future in her eyes of the future

But she had a small problem she was not sure the future of the future was not sure she was a future

She had always wanted to find the truth about her past but she was not sure she was not sure she was a future

One day she met a strange man named Kael who was a traveler who had a traveler who had seen the traveler who had a traveler who had seen the traveler who had seen the traveler who had seen the traveler who had seen the traveler who had seen the traveler who had seen the traveler who had seen the traveler who had
```

```
draft acceptance = 1.00000 (  212 accepted /   212 generated), mean len =  6.58
statistics draft-deterministic: #drafts = 43, #truncated = 22, #skipped = 0, #tokens pre = 298, #tokens post = 212, #target rejected = 0
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
 in a quiet village nestled between the the Earth named Write a named Snowy Mountains and the sky named Azure Skies

In a the land was a was a named Snowy Mountains and a land was a name that was a named Snowy Mountains and a land named Snowy Mountains and a sky named Azure Skies a name that was a named Snowy Mountains and a land named Sky was a name that was a named Sky a name that was a named Sky a name that was a named Sky was a name that was a named Sky a name that was a named Sky was a named Sky a name that was a named Sky a name that was a named Sky a name that was a named Sky a name that was a named Sky a name that was a named Sky a name that was a named Sky a name that was a named Sky a name that was a named Sky a name that was a named Sky a name that was a named Sky a name that was a named Sky a name that was a named Sky a name that was a named Sky a name that was a named Sky a name that was a named Sky a name that was a named Sky a name that was a named Sky a name that was a named Sky a name that was a named Sky a name that was a
```

```
draft acceptance = 1.00000 (  232 accepted /   232 generated), mean len = 14.65
statistics draft-deterministic: #drafts = 23, #truncated = 11, #skipped = 0, #tokens pre = 365, #tokens post = 232, #target rejected = 0
```

The intercept-all output keeps letters and whitespace but drops punctuation: every emitted token is filter-constrained, because intercept-all keeps all filter-surviving tokens without target veto. With this trivial letters-and-whitespace rule the model still degenerates into a repeated-word loop, so intercept-all output is not representative of a real grammar. Note that the mode does not merely cut drafts: it constrains the bonus token too, so a constraint the content does not fit will pull the generation toward whatever the rule does admit. That is the failure the plugin's decline-a-round fallback exists to prevent, and why the default run above stays on the baseline.

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

One caveat on this commit: the script's readiness probe greps the log for `llama_server: listening on`, but the server prints `srv  server_on_li: listening on http://...` (`SRV_INF` in `tools/server/server-common.h`, called from `server_on_listening`). The string never matches, so every scenario waits the full `DET_TIMEOUT` (default 600 s) and then reports `timed out waiting for server` without issuing its request. Lower `DET_TIMEOUT` while that line is stale, or check readiness yourself:

```sh
DET_TIMEOUT=20 external/scripts/run-draft-scenarios.sh ia-mtp
```

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

The test suite registers 20 tests:

- Plugin loader lifecycle (init/free with valid and invalid paths)
- C API wrappers when no plugin is loaded (filter_draft, rollback, apply_bitmask, get_capabilities, get_version)
- Speculative type enum and params struct
- filter query helpers and intercept-all lookup, NULL-handle safe
- Hard fails: no `--spec-type`, no plugin, intercept-all without a plugin
- Intercept-all accepted for all compatible drafter types (mtp, dspark, eagle3, dflash)
- MTP type auto-implied when the plugin is loaded
- `token_to_piece` special-token flag
- Cumulative statistics fields
- Rollback of the bitmask state after a failed init
- Regex plugin: whitespace and EOG handling in `filter_draft` (whitespace is accepted, not a truncation point)
- Regex plugin: declining a round through the match-rate window

The suite does not test plugin state serialization or per-slot grammar resolution: they exist
in the interface (`state_get_size` / `state_get_data` / `state_set_data`) but no test here
touches them. Those, plus per-drafter-type coverage, live in the XGrammar POC suite
(`tests/test-deterministic-draft-xgrammar-poc.cpp`, 34 tests, not registered in CMake).

## Plugin Interface

Implement the contract from `include/deterministic_draft_plugin.h`. All 17 entry points, in the order the header declares them:

| Function | Description |
|---|---|
| `deterministic_draft_create` | Allocate plugin state; NULL on failure |
| `deterministic_draft_destroy` | Free state (NULL-safe) |
| `deterministic_draft_get_capabilities` | Bitmask of `DETERMINISTIC_DRAFT_CAPABILITY_*`; 0 means unusable |
| `deterministic_draft_set_vocab` | Register token strings, vocab size and stop tokens (BITMASK/JUMP_FORWARD plugins) |
| `deterministic_draft_fill_bitmask` | Fill the valid-token bitmask for the next step |
| `deterministic_draft_get_jump_forward` | Longest uniquely determined string to skip; NULL if none |
| `deterministic_draft_commit` | Commit one accepted token to the constraint state |
| `deterministic_draft_rollback` | Undo the last `n_tokens` commits; `n_tokens == 0` must return true |
| `deterministic_draft_filter_draft` | Keep the valid prefix, commit-on-accept; `rest_action` says ENFORCE or SKIP |
| `deterministic_draft_apply_bitmask` | Fill a bitmask and set invalid logits to `-1e30f` in place |
| `deterministic_draft_commit_tokens` | Commit several tokens at once (IDs converted through `set_vocab`) |
| `deterministic_draft_is_terminated` | True when the constraint is terminal (host treats it as EOG); optional |
| `deterministic_draft_reset` | Clear per-slot state |
| `deterministic_draft_state_get_size` | Size in bytes of the serialized constraint state |
| `deterministic_draft_state_get_data` | Write the serialized state into a caller buffer |
| `deterministic_draft_state_set_data` | Restore the constraint state from a buffer |
| `deterministic_draft_get_version` | SPI version string the plugin was built against |

