# Prompt-cache recovery for hybrid/recurrent models (llama.cpp)

> Diagnosis, patch, and writeup are entirely Claude Opus's work.

Fixes `server_prompt_cache::load()` so it stops discarding a usable cached
conversation when a client changes something early in the prompt.

Related upstream issue: [ggml-org/llama.cpp#22746](https://github.com/ggml-org/llama.cpp/issues/22746)
Builds on: [PR #25592](https://github.com/ggml-org/llama.cpp/pull/25592) (checkpoint handling for hybrid/recurrent models)

---

## Runtime arguments

Reference invocation used for testing (Qwen3.6-27B, MTP, Vulkan):

```bash
llama-server \
    --model /mnt/AI/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q4_K_XL.gguf \
    --mmproj /mnt/AI/models/mradermacher/Qwen3.6-27B-GGUF/Qwen3.6-27B.mmproj-Q8_0.gguf \
    --jinja \
    --flash-attn on \
    --parallel 1 \
    --no-mmap \
    --warmup \
    --no-ui \
    --log-timestamps \
    --log-prefix \
    --n-gpu-layers 99 \
    --verbosity 4 \
    --image-min-tokens 1024 \
    --cache-ram 28672 \
    --ctx-checkpoints 64 \
    --checkpoint-min-step 2048 \
    --log-file /mnt/AI/llama-debug/logs/qwen-10003.log \
    --log-prompts-dir /mnt/AI/llama-debug/prompts/qwen-10003 \
    --chat-template-file /mnt/AI/models/qwen3.5_chat_template.jinja \
    --spec-type draft-mtp \
    --spec-draft-n-max 2 \
    --spec-draft-ngl 99 \
    --spec-draft-type-k q8_0 \
    --spec-draft-type-v q8_0 \
    --ctx-size 163840 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --reasoning off \
    --no-reasoning-preserve \
    --reasoning-budget 0 \
    --chat-template-kwargs '{"enable_thinking":false}' \
    --temp 0.7 \
    --top-p 0.95 \
    --min-p 0.00 \
    --top-k 20
```

`--ctx-checkpoints`/`--checkpoint-min-step` above the defaults (32 / 8192) and
`--log-prompts-dir` are what make the checkpoint-salvage behavior and the
prompt-diverging test scenarios observable; neither is required for the fix
itself.

`--chat-template-file` points to a community-fixed template, not the stock
one shipped with the model: [spiritbuun/buun-Qwen3.6-chat_template](https://huggingface.co/spiritbuun/buun-Qwen3.6-chat_template).

## Background

Hybrid models (e.g. Qwen3.5/3.6) mix attention layers with a single rolling
recurrent state. The attention cache can be rewound to any position; the
recurrent state cannot — it only exists at its latest position. When a client
changes a token mid-conversation, everything after that point must be
recomputed regardless. This fix does not change that.

What it does address: `llama-server` was also discarding the part **before**
the change, forcing a full reprocess when a partial one would do.

## Problem

In `server_prompt_cache::load()` (`tools/server/server-task.cpp`):

**1. Wrong selection baseline.**

```cpp
if (f_keep_best < f_keep_cur && sim_best < sim_cur) {
```

`f_keep_best` is seeded from the prompt currently in the slot. If the slot
holds a small unrelated prompt (title generation, subagent call, health
check), its `f_keep` is trivially high, so a large cached conversation with a
lower `f_keep` can never win. Observed: `base f_keep = 0.769` for a 362-token
throwaway prompt vs. `f_keep = 0.22` for the actual 130k-token conversation
in the cache — comparison fails, entry never selected.

The comparison is also redundant: `prompt_save()` runs immediately before
`prompt_load()` (`tools/server/server-context.cpp`), so the slot's own
prompt is already in the cache before selection starts.

**2. `f_keep` guard ignores checkpoints.**

```cpp
// don't trash large prompts
if (f_keep_cur < 0.25f) {
    continue;
}
```

Restoring an entry consumes it (`states.erase()`), so the guard against low
`f_keep` is reasonable in general. But a cached entry carries its context
checkpoints with it. If a checkpoint sits at or below the divergence point,
restoring the entry lets the server resume from that checkpoint instead of
from zero — for hybrid/recurrent memory, that's the difference between
reusing the prefix and reprocessing the whole prompt.

Log excerpt:

```
load:  - looking for better prompt, base f_keep = 0.769, sim = 0.007
update:  - cache state: 2 prompts, 8597.250 MiB
update:    - prompt 0x…12b0: 130011 tokens, checkpoints: 11, 8134.713 MiB
update:    - prompt 0x…9f40:    362 tokens, checkpoints:  2,  462.537 MiB
forcing full prompt re-processing due to lack of cache data
```

## Fix

Keep the guard, but let an entry through when a checkpoint can still salvage
real work, and select by coverage of the new prompt rather than against the
slot's own baseline. Applied as commit `71893574e` on this branch
(`fix/22746-prompt-cache-checkpoint-salvage`) in `tools/server/server-task.cpp`.

## Measurements

Qwen3.6-27B (UD-Q4_K_XL, MTP), Vulkan on an RX 7900 XTX, `--ctx-size 163840`,
`--parallel 1`, `--cache-ram 28672`, `--ctx-checkpoints 64`,
`--checkpoint-min-step 2048`.

Scenario: build a 65k-token conversation, push it out of the slot with a
small unrelated request, edit a message ~20% into the history, come back.
Three identical cycles.

| | without patch | with patch |
|---|---|---|
| cycle 1 | 65103 tokens / 127.0 s — full reprocess | **49176 / 105.1 s** — checkpoint restore |
| cycle 2 | 65103 tokens / 127.3 s — full reprocess | **49176 / 105.2 s** |
| cycle 3 | 65103 tokens / 127.3 s — full reprocess | **49176 / 105.3 s** |
| full reprocesses | 3 / 3 | **0** |
| total | 250147 tokens / 496.0 s | 212197 tokens / 442.9 s |

~15900 tokens and ~22 s saved per incident, matching the prefix before the
divergence point.

### Regression and correctness

- Append-only multi-turn: unchanged, ~26 tokens per turn after cold start.
- Subagent interleave (small unrelated request between turns): 0 full
  reprocesses, main conversation survives.
- Correctness: three facts planted early in a conversation, forced through
  evict → edit → restore, all three recalled correctly.

## Limitations

- The ceiling is the divergence depth: the guard only fires below
  `f_keep = 0.25`, so at most ~25% of the prompt is recoverable this way.
- `n_reuse_min = 4096` is a judgement call, not an optimized value.
- The selection change also affects SWA models; the reasoning holds there
  too (`prompt_save()` runs before `prompt_load()`), but only tested with
  Qwen3.6.
- Tested on Vulkan only.

## Scope

This addresses one of several possible causes of forced full reprocessing
under #22746 — specifically, the server converting a partial cache loss into
a total one. It does not address clients that rewrite their own prompt
history (append-only prefixes should not trigger this at all; if they do,
that's a separate server bug).

## Building

```bash
git checkout fix/22746-prompt-cache-checkpoint-salvage
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j$(nproc)
```

## Credits

Checkpoint-handling groundwork: [PR #25592](https://github.com/ggml-org/llama.cpp/pull/25592)
by krim404. Message-boundary checkpoints: [#24176](https://github.com/ggml-org/llama.cpp/pull/24176)
by aldehir (merged). Discussion: [#22746](https://github.com/ggml-org/llama.cpp/issues/22746).
