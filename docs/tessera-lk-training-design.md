# Tessera LK drafter-training driver (native C++)

_Last updated: 2026-07-31_

Source of truth for the native LK (acceptance-rate) drafter-training driver
`tools/quantize/tessera/tessera-train-lk.cpp`. This is "Path A" of the
self-improving flywheel: the autoregressive-drafter trainer. The DFlash/D-PACE
block-drafter trainer (`docs/tessera-dflash-training-design.md`) is "Path B"
and reuses the same plumbing.

This document is the design discussion the LK driver needs before code. Where
this document and `docs/tessera-studio-design.md` disagree, this one wins for
the C++ training path.

## 1. Why native, not Python

The earlier flywheel plan assumed a Python/PyTorch/peft training script. That
is dropped. The driver is C++ (with a Swift orchestrator shelling out to it,
the same way `TesseraTrainingOrchestrator` already shells out to
`llama-finetune`). Reasons:

- The LK objective already lives in ggml-opt (`GGML_OPT_LOSS_TYPE_LK`,
  commit 97e3eda34) and the llama layer already trains against dense labels
  (`src/llama-context.cpp` `opt_epoch_iter`). A Python loop would re-implement
  a graph that already exists and is tested.
- No second runtime, no second dependency tree, no model-format round-trip.
  The drafter is a GGUF; it trains as a GGUF and saves as a GGUF.
- It matches the project's "reuse existing infrastructure, no new subsystems"
  rule.

## 2. What already exists (do not rebuild)

| Piece | Where | Status |
|-------|-------|--------|
| LK loss graph (TV distance = 1 - alpha) | `ggml/src/ggml-opt.cpp` `GGML_OPT_LOSS_TYPE_LK` | landed (97e3eda34) |
| Dense-from-top-k reconstruction | `tools/quantize/tessera/tessera-lk-loss.{h,cpp}` `ts_lk_dense_from_topk` | landed (87e7a97aa) |
| `llama_opt_params.loss_type` threading | `include/llama.h`, `src/llama-context.cpp:3391` | landed (c98e53072) |
| Dense-label epoch path (data=tokens, labels=dense) | `src/llama-context.cpp` `opt_epoch` / `opt_epoch_iter` | landed |
| Trace collection | `tools/imatrix/imatrix.cpp` `--telemetry-out --telemetry-topk K` (`llama.tessera.spec.v1`) | landed |

The driver is the last missing piece: it reads traces, builds the dataset the
llama-layer LK path already expects, and drives the epoch loop.

## 3. The datapoint contract (fixed by the llama layer)

`src/llama-context.cpp` `opt_epoch` already decides the dataset shape for LK:

- data tensor:  type `I32`, `ne_datapoint = n_ctx` - one datapoint is a token
  sequence of length `n_ctx` (`llama_token`).
- labels tensor: type `F32`, `ne_label = n_ctx * n_vocab` - one datapoint is
  `n_ctx` dense distributions of width `n_vocab`, laid out position-major as
  `labels[pos*n_vocab + tok]`.
- Per batch it copies each position's `[n_vocab]` column straight into
  `ggml_opt_labels` (`src/llama-context.cpp:3522-3531`).

So the driver does not get to choose the layout; it must produce exactly this.

## 4. The input side: on-policy distillation (the one real design decision)

A `llama.tessera.spec.v1` record is one speculative-decoding step:

- `prime_token` - the token that seeds the step.
- `drafted_tokens[0..n_dft-1]` - the drafter's proposed tokens.
- `verifier_topk_tokens[i]` / `verifier_topk_probs[i]` for `i in 0..n_dft` -
  the verifier's top-k distribution at the i-th per-prefix forward.

Crucially, `verifier_topk[j]` is the verifier distribution **conditioned on
`prime + drafted[0..j-1]`** - speculative decoding scores the drafter's own
proposed tokens, so the verifier was fed the draft prefix. Therefore the only
input prefix consistent with that label is the draft prefix itself.

This forces the training contract (it is not a preference):

```
n_ctx     = block_size + 1            (block_size == n_dft, fixed per run)
tokens[j] = prime           if j == 0
            drafted[j-1]    if j >= 1
label[j]  = densify(verifier_topk[j]) for j in 0..block_size
```

- `verifier_topk` has exactly `n_dft + 1 = block_size + 1 = n_ctx` entries, so
  **every position is a real training position - no padding, no masking, no
  graph change.** This is why fixed-block records are required.
- It is on-policy distillation: the drafter learns to match the verifier given
  its own draft prefix, which is exactly the conditioning it sees at inference.
  Teacher-forcing on the accepted prefix would pair a draft-conditioned label
  with a mismatched input prefix for every rejected position.
- Records whose `drafted` count != `block_size` are skipped (reported). For a
  fixed drafter config the vast majority of records match.

## 5. Memory constraint (honest)

Dense full-vocab labels are large: the labels tensor is
`ndata * (block_size+1) * n_vocab * 4` bytes. For `n_vocab = 152k`,
`block_size = 4`, `ndata = 1024` that is ~3.1 GB. The driver therefore:

- caps the dataset with `--max-examples` (default 512),
- densifies **straight into** the ggml dataset tensor (no second copy),
- prints the estimated label memory before allocating,
- documents the follow-up: a lazy per-batch densification (custom dataset or a
  llama-layer streaming path) to lift the cap. Not in v1.

## 6. Module split

- `tessera-lk-train-data.{h,cpp}` - pure logic, no llama/ggml. Parses a
  `spec_calib.v2` line, decides usability, and densifies one example into
  caller buffers (reuses `ts_lk_dense_from_topk`). Also auto-detects the modal
  block size. Tested standalone in `test_all.sh` (like `tessera-dataset`).
- `tessera-train-lk.cpp` - executable, links `llama-quantize-impl`. Parses args
  (reuses `common_params_parse` with `LLAMA_EXAMPLE_FINETUNE` for the standard
  training flags; pre-scans its own `--traces/--block-size/--max-examples/
  --dry-run`), loads the drafter, builds the dataset, runs the LK epoch loop,
  saves the GGUF. Mimics `examples/training/finetune.cpp`.

No Tessera-specific code enters `src/`; the driver only consumes the LK path
that is already there.

## 7. CLI

```
tessera-train-lk -m drafter.gguf --traces spec_calib.v2.jsonl -o trained.gguf \
    [--block-size B]      # default: auto-detect modal drafted count
    [--max-examples N]    # default 512 (dense-label memory cap)
    [--dry-run]           # build dataset + print stats, do not train/save
    # plus the standard finetune flags: --epochs --lr --optimizer
    # --val-split --n-gpu-layers --ctx-size (forced to B+1)
```

Forced before model load (mirrors finetune): `load_mode = none` (writable
weights), `cache_type_k/v = f32`, `flash_attn = disabled`, `n_ctx = n_batch =
n_ubatch = block_size+1` (so `opt_period = 1`). Flash attention is forced off
because `FLASH_ATTN_EXT` has no backward pass; the training graph must use the
differentiable non-flash attention path (the same path that routes gradients
through the KV-cache `SET_ROWS`). finetune.cpp forces it off for the same
reason.

## 8. Testing

- `test_lk_train_data.cpp` (standalone, `test_all.sh`): fixture `spec_calib.v2`
  lines -> usability predicate, token layout, and densified label columns
  (top-k probs at their slots, residual mass spread uniformly). Reuses the
  `ts_lk_dense_from_topk` edge cases indirectly.
- End-to-end (manual, needs a real drafter GGUF + traces): loss should fall and
  top-1 accuracy (argmax agreement with the verifier) should rise over epochs.
  Not gated in CI because it needs model weights.

Prerequisite that landed (2026-07-31): native training was blocked fork-wide by
a missing `GGML_OP_SET_ROWS` backward pass (the KV-cache write). It is now
implemented in `ggml/src/ggml.c` as a gather of the output gradient into the
written source rows (`ggml_get_rows(grad, indices)`), with `ggml_get_rows`
extended to read I32 or I64 indices directly (CPU templated on the index type;
Metal instantiates an `_i64` kernel for F32 sources, the only case the backward
gather produces). Validated two ways: a finite-difference harness (analytic
gradient matches central differences to ~1.7e-5 and the closed-form gather
exactly, on I64 indices) and an end-to-end `tessera-train-lk` smoke run on a
tiny locally-generated llama GGUF + synthetic `spec_calib.v2` traces (finite
loss across all epochs, trained GGUF saved). This unblocks `llama-finetune` as
well as `tessera-train-lk`.

## 9. Follow-ups (not v1)

- Lazy per-batch densification to lift the `--max-examples` memory cap.
- Gradient accumulation (`opt_period > 1`) - currently forced to 1 by
  `n_batch = n_ctx`.
- Path B: the DFlash/D-PACE block-drafter driver reuses the arg pre-scan, the
  dataset-build pattern, and the epoch loop; its labels are pre-weighted CE
  rows (see `docs/tessera-dflash-training-design.md`), not dense LK columns.
- Swift orchestrator wiring: point `TesseraTrainingOrchestrator` /
  `RunTrainingTool` at `tessera-train-lk` for the LK step of the flywheel.
