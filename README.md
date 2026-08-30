# Numerical gap between 1-token and W-token decode can flip greedy argmax

Tool usage: [examples/compare-single-and-block/README.md](examples/compare-single-and-block/README.md).

## Motivation

This work started from an observation made while reproducing [DFlash](https://github.com/z-lab/dflash) on [llama.cpp](https://github.com/ggml-org/llama.cpp/pull/22105) (and MNN): with `temp = 0` (greedy sampling) and by default the exact-match-then-accept algorithm — assuming no KV-cache corruption, no weight differences, and a correct argmax — the answer of a DFlash-produced output with Qwen3-4B as target model and any draft model should be **exactly the same** as the output of Autoregressive-produced output with Qwen3-4B.

This held for 6 out of 7 test prompts. However, with the prompt *"Explain the Pythagorean theorem"*, the two outputs diverged at some position: the AR trajectory continues with *"relates"*, while the DFlash run produced *"describes"*. A guess is: this phenomenon stems primarily from the **Gap between the logits calculated with single run or a block run on one model**. So here comes this tool.

## Experiment result

I conducted a simple experiment, with prompt 'Explain the Pythagorean theorem', and to generate 100 tokens, on CPU:

```bash
cmake -B build
cmake --build build --target llama-compare-single-and-block -j

./build/bin/llama-compare-single-and-block \
  -m "${MODEL_GGUF}" \
  -p "Explain the Pythagorean theorem" \
  -n 100 -ngl 0 -fa off --block-width 5
```

Here are the results:

```
--- run config ---
flash_attn: 0
n_gpu_layers: 0
n_threads: 32 / batch: 32
n_batch: 512 / n_ubatch: 512
block_width: 5
prompt_tokens: 19
n_gen: 100
model: Qwen3 4B bf16    backend: CPU  fa=off  t=32  W=5
------------------

=== pairwise logit-diff statistics (n=490 cells) ===
pair                                 mean          var      mean|d|       var|d|
l_ar - l_single                 -0.007948     0.001222     0.027564     0.000526
l_ar - l_block                  -0.006654     0.001771     0.030979     0.000855
l_single - l_block (dlogit)     +0.001295     0.001389     0.026934     0.000665

summary: tokens=100 checked=490 flips=6 cells across 2 positions
FLIP at pos=5 (window t=1, block offset j=4):
  single argmax= 57817(' theorem') l=43.2867 | block argmax=   576(' The')l=43.2825
  AR token= 57817(' theorem') l_s=43.2867 l_b=43.2610 dlogit=+0.0256
...
FLIP at pos=14 (window t=10, block offset j=4):
  single argmax= 16555(' describes') l=31.3830 | block argmax= 35616(' relates') l=31.3891
  AR token= 35616(' relates') l_s=31.3680 l_b=31.3891 dlogit=-0.0210
...
  note: single-path argmax also deviates from the AR trajectory — prefill-vs-incremental drift, not a block effect

```

The `t = 0` determinism self-check did not fire (no warning) on this 16-thread CPU run.

How to read the three pairs (same AR token, all cells pooled):

- `l_ar - l_single`: prefill-vs-incremental drift. `l_ar` comes from the original AR run (prompt prefill, then 1 token/step). `l_single` comes from a later window that **wipes KV and re-prefills** `prompt+AR[0..t-1]` as one batch, then steps 1 token at a time. Same tokens, but the prefix KV was written by a different graph shape. At `t = 0` the two graphs match, so any leftover gap there is backend non-determinism (see the self-check in Framework below).
- `l_single - l_block` (`dlogit`): the batch-shape gap this tool isolates. Both paths share that re-prefilled prefix; the only difference is W calls of 1 token vs one call of W tokens.
- `l_ar - l_block`: the two mixed together, not a third independent effect.

Hope you can test the difference on your own backend and share your results. How to run: [examples/compare-single-and-block/README.md](examples/compare-single-and-block/README.md).

## Why does this matter to DFlash?

The target **verify** is a block decode; AR is single-token decode. The logit gap is usually tiny (≈1e-2), but a near-tie can **flip the argmax**, and later tokens then drift.

On this prompt ([PR #22105](https://github.com/ggml-org/llama.cpp/pull/22105), [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) target, [Qwen3-4B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16) draft):

- AR: `…in geometry that relates the sides of a right-angled triangle…`
- DFlash: `…in geometry that describes the relationship between the sides…`

Under exact-match, `describes` enters the output only if the **target** argmaxed to `describes` on that block verify. The tool reproduces the flip with the target model alone.

## Reason for the Gap in llama.cpp

Hypothesis (no per-op kernel dump): a 1-token decode and a W-token decode run the same math on different **compute-graph shapes**. Floating-point addition is not associative, so the rounding can differ. On CPU, `ggml_mul_mat` often takes a GEMV-style path for a 1-row input (`ne11 == 1`) and a GEMM path for a W-row input. `-fa on` tiles attention by query length; GPU scheduling and atomics can add further variation. The KV cache then stores whatever the producing kernel wrote — that is the prefill-vs-incremental drift in `l_ar` vs `l_single`.

This is not a llama.cpp bug: bit-invariance across batch shapes is not a property floating-point kernels provide. Exact-match speculative decoding implicitly relies on that invariance — that is the gap this tool quantifies.

## Framework of this code

Method details are in [examples/compare-single-and-block/README.md](examples/compare-single-and-block/README.md). Two things keep the comparison valid:

- **Full KV wipe + identical re-prefill** before every window (`llama_memory_seq_rm(..., 0, -1, -1)` on both contexts). A partial KV rollback would contaminate the prefix with KV values written under previous iterations' shapes. The next single-step input is always the AR token, even after a flip.
- **Determinism self-check**: the `t = 0` single path replays the AR baseline's exact compute graph, so on a deterministic backend `l_s` must equal `l_ar` bit-for-bit. Any drift (e.g. GPU atomics) means the backend itself is noisy, and a warning is printed. This CPU run printed none.

`j = 0` is the current token on a shared prefix; `j > 0` also includes in-window KV written by each path's own shape.

Data flow of the first pos-14 flip from the CPU run (window `t = 10`, offset `j = 4`):

```
        prefix = prompt + AR[0..9]  ("… a fundamental")
        (bit-identical KV on both contexts after the re-prefill)
                                |
          +---------------------+----------------------+
          | single path         | block path           |
          | 5× 1-token decode   | decode ['fundamental'|
          | of the same window  |  'principle',' in',  |
          | (GEMV-shaped)       |  'geometry',' that'] |
          |                     | as one 5-token call  |
          |                     | (GEMM-shaped)        |
          | relates   31.3680   | relates   31.3891    |
          | describes 31.3830   |                      |
          +---------------------+----------------------+
          argmax: describes     argmax: relates    ==>  FLIP
          (AR baseline: relates 31.3868)
```

The same 5-token input, the same prefix KV, the same weights — only the batch shape differs, and that difference (≈0.02 on these logits) is enough to reorder two near-tied candidates.

## Questions

1. Would this gap affect the quality of DFlash-generated output, especially under real production serving? Exact-match-then-accept and classic same-in-prob rejection sampling define "lossless" slightly differently, but the target model's AR-vs-block numerical gap is present in either paradigm.
2. Besides tightening numerical precision, are there other ways to keep speculative-decoding quality aligned with the autoregressive trajectory?
3. Can others reproduce this run (same prompt, Qwen3-4B bf16, `W=5`, `-n 100`) on your backend and share the pairwise stats / flip count? How to run: [examples/compare-single-and-block/README.md](examples/compare-single-and-block/README.md).

Worth measuring later, for a given prompt set: how the gap and flip rate move with model, backend, `--block-width`, `-fa`, and quantization — and whether some prompts (near-tied tokens) are systematically "harder".
