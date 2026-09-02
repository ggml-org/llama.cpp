# Speculative prefill

Port of upstream PR [ggml-org/llama.cpp#27692](https://github.com/ggml-org/llama.cpp/pull/27692),
which implements *Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free
Token Importance Estimation* ([arXiv:2502.02789](https://arxiv.org/abs/2502.02789)).

A small draft model prefills the prompt and decodes a few lookahead tokens. The softmax
attention of those tokens over the prompt is captured with a ggml eval callback, smoothed,
max-reduced over heads and layers, and pooled into chunks. Only the top scoring chunks are
fed to the target model. This is lossy: tokens that are dropped are gone.

Ported: `common/speculative-prefill.{h,cpp}`, `llama_set_eval_callback`, the `--spec-prefill-*`
options, `examples/speculative-prefill`, and the llama-server hookup. Not ported: the
llama-bench changes and the python eval scripts from that PR.

## Usage

```bash
llama-server -m target.gguf -mpd draft.gguf --spec-prefill-percentage 0.3
```

The estimator must be a separate small model with the same vocab as the target. MTP, DFlash2,
DSpark and Eagle3 heads are target-dependent and the server refuses to reuse them for this.
The draft context is forced to flash-attn off, because `kq_soft_max` does not exist on the
flash-attn path. If no attention is captured the full prompt is kept.

## Results

Ryzen AI MAX+ 395 / Radeon 8060S, Vulkan, 119 GiB unified memory. Estimator is
Qwen3.5-2B-UD-Q4_K_XL in every run. "eff PP" is original prompt tokens divided by TTFT.

### TTFT vs context length

Qwen3.8-27B-UD-Q4_K_XL, no decode speculation, median of 3 runs. A passcode was planted at
mid-depth and retrieved in 3/3 runs in every cell.

| N tokens | p = off | p = 0.30 | p = 0.15 |
| ---: | ---: | ---: | ---: |
|  1,498 |  6,225 ms | 2,647 ms (2.35x) |  1,516 ms (4.11x) |
|  3,131 | 12,992 ms | 5,199 ms (2.50x) |  3,084 ms (4.21x) |
|  6,212 | 26,494 ms | 9,906 ms (2.67x) |  6,113 ms (4.33x) |
| 12,347 | 54,927 ms | 19,557 ms (2.81x) | 11,896 ms (4.62x) |

Baseline eff PP is nearly flat over that range (241 -> 225 tok/s), so prefill here is close to
linear in N rather than quadratic. The speedup still grows with N, but because the fixed draft
cost amortizes, not because there is super-linear cost to remove.

Decode is unaffected: 11.14 to 11.62 tok/s across all 12 cells.

### With decode speculation

Qwen3.8-27B-UD-Q4_K_XL, 1,869-token prompt, single run each.

| decode spec | p | TTFT | eff PP | tg |
| --- | ---: | ---: | ---: | ---: |
| none        | off  | 5,604 ms |   334 tok/s | 11.56 tok/s |
| none        | 0.15 | 1,587 ms | 1,180 tok/s | 11.66 tok/s |
| MTP n=4     | off  | 5,755 ms |   325 tok/s | 19.47 tok/s |
| MTP n=4     | 0.30 | 2,440 ms |   768 tok/s | 29.33 tok/s |
| MTP n=4     | 0.15 | 1,636 ms | 1,145 tok/s | 23.61 tok/s |
| DFlash2 n=7 | off  | 6,576 ms |   284 tok/s | 19.33 tok/s |
| DFlash2 n=7 | 0.30 | 2,724 ms |   688 tok/s | 30.44 tok/s |
| DFlash2 n=7 | 0.15 | 1,772 ms | 1,057 tok/s | 20.90 tok/s |

The two mechanisms are independent. DFlash2 gains the most because its head also has to prefill
the full prompt, and speculative prefill shrinks that too.

### Qwen3.8-Flash-Next

FP4 ple16 (87 GiB) with the Q8_0 MTP head, 1,869-token prompt, median of 3, needle 3/3.

| p | TTFT | eff PP | tg | MTP accept |
| ---: | ---: | ---: | ---: | ---: |
| off  | 4,920 ms |   380 tok/s | 29.38 tok/s | 71% |
| 0.30 | 2,441 ms |   767 tok/s | 28.53 tok/s | 66% |
| 0.15 | 1,645 ms | 1,139 tok/s | 28.87 tok/s | 69% |

Flash-Next gains less than the dense 27B because it is hybrid: most layers are linear-attention,
so baseline prefill is already closer to linear in N. MTP acceptance and decode rate hold across
keep ratios.

The IQ4_XS build reaches 4,718 -> 1,599 ms over the same sweep when run with `--ngram-on-disk`.

## Caveats

- Greedy (temperature 0) output is not reproducible on this box even with no speculation and no
  speculative prefill: 3 identical requests to the plain target gave 2 different completions.
  Only TTFT is worth quoting from these runs. Cross-arm `tg` and draft-acceptance differences
  are confounded, because each arm generates different text under `ignore_eos`.
- `--spec-prefill-ctx`, `--spec-prefill-device` and `--spec-prefill-ngl` set `enabled = true` as
  a side effect, so passing only a context size makes the server exit with "speculative prefill
  enabled but no draft model was provided".
- The example renumbers kept tokens to positions 0..K-1 instead of keeping their original
  positions. The reference implementation keeps the original ones.
- `t_draft_eval_us` covers the draft prompt prefill only, not the lookahead decode steps. About
  150 ms of the reported TTFT is unattributed, because the eval callback forces node-by-node
  execution.
- Quality falls off when the token budget cannot hold all the non-redundant evidence. p = 0.30 is
  a reasonable default; p = 0.15 still retrieved a needle and summed 3 facts spread across the
  context; below p = 0.10 is risky for prompts with distributed evidence.

`scripts/bench-spec-prefill.sh` reproduces the server measurements.
