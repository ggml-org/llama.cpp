# DFlash block-drafter training driver - design

Status: design + Stage 0 (go/no-go) findings + offline feature-capture tooling
landed. The training DRIVER (section 5, Stages 1-4) is not written yet.

This documents the plan to train the DFlash/DSpark block drafter on-device, in
idle time, using the D-PACE adaptive position-weight objective. It builds on
landed work:

- D-PACE loss math: `tools/quantize/tessera/tessera-dpace.{h,cpp}` (a0c843bb5)
- Block dataset prep: `TS_DATASET_MODE_DFLASH` in `tessera-dataset` (f6ba75251),
  emits `llama.tessera.dflash-block.v1` per spec step.
- LK loss in ggml-opt: `GGML_OPT_LOSS_TYPE_LK` (97e3eda34) - the autoregressive
  drafter objective. D-PACE is the block-parallel objective.

## 1. The finding that reframes the design

The DFlash drafter is NOT a standalone token-in/token-out model. It is
EAGLE-style, feature-conditioned (`src/models/dflash.cpp`):

- Encoder (`graph<true>`, dflash.cpp:138-153): input is the trunk's hidden
  states at `target_layer_ids`, fused through an FC layer. Input width
  `n_embd_inp_enc = target_layer_ids.size() * n_embd` (dflash.cpp:15). It
  projects features and exposes them for K/V injection.
- Decoder (`graph<false>`, dflash.cpp:243-435): dual-mode. An embd batch
  projects trunk features and injects K/V into the drafter cache; a token batch
  runs noise-block diffusion, attending (non-causal, cache-aware) over
  [committed, MASK...] to emit the draft block.
- The decoder borrows `tok_embd` and `output` (lm_head) from the trunk via
  `cparams.ctx_other` (dflash.cpp:333-341, 416-423).

Consequence: the landed block dataset (target_tokens + acceptance_probs +
weights) is necessary but NOT sufficient. The drafter cannot run a forward pass
without the trunk's target-layer hidden states. Training must supply those
features. This is the real project; the loss is the easy part.

## 2. Stage 0 findings (go/no-go gate)

### 0a. Differentiable masked-block training graph - YES

Every op in the decoder token-batch path is a standard differentiable ggml op:
`ggml_get_rows` (embedding), `build_lora_mm` (matmul), `build_norm`,
`ggml_rope_ext`, `build_attn`, `build_ffn`, `ggml_add` (dflash.cpp:350-427).
This is the same op set trunk finetune already backprops through ggml-opt, so
ggml-opt can build gradients for all of it. No new graph ops are required.

The training graph is the existing inference graph plus a weighted-CE label
fill at the block positions:

    encoder(cached_trunk_features) -> inject K/V into cache
    decoder([anchor, MASK x B])    -> logits [n_vocab, n_tok]
    weighted CE on the B drafted positions

Two setup wrinkles:

1. Borrowed trunk tensors. `tok_embd` and `output` come from the trunk via
   ctx_other. They must be resident at train time but stay FROZEN - the
   `llama_set_param` filter (llama-context.cpp:3373-3387) only marks the
   drafter's own tensors (fc, layers, norms, dspark heads) as trainable, so the
   borrowed tensors receive no gradient. Training therefore needs those two
   trunk weight tensors available (load the trunk for them, or bake copies into
   a training-only drafter GGUF), but does NOT need to run the trunk forward.
2. Two-phase forward. At inference the encoder (embd batch) and decoder (token
   batch) run as separate phases. For training they must be combined into one
   step so gradients flow through both. Since features are cached offline
   (section 3), the encoder forward is cheap and runs at train time on the
   cached features.

### 0b. Logit-to-position layout - block-major, with an off-by-one

Layout (dflash.cpp:173-185, 223-226): `n_tok = n_blocks * block_drafts`,
block-major. Column `b*block_drafts + j` is block b, position j. Position 0 is
the anchor (committed last token); positions 1..block_drafts-1 are the drafted
(MASK) positions. `dflash.block_size` (GGUF metadata) bounds block_drafts and
INCLUDES the anchor, so max drafted = block_size - 1.

Reconciliation with the dataset: `llama.tessera.dflash-block.v1` indexes
target_tokens[k] for k in 0..n_dft-1 over DRAFTED positions only (block_size in
the dataset = n_dft). The model block includes the anchor, so the driver maps:

    dataset target_tokens[k] -> model block position j = k + 1
    anchor position j = 0    -> weight 0 (conditioning input, not a target)
    labels[target_tokens[k], b*block_drafts + (k+1)] = dpace_weights[k]

Fixed training block size: pad short blocks; padded positions get weight 0,
which yields zero gradient, so padding is free.

### 0c. Feature dimensionality and capture

Per token the encoder consumes `n_target_layers * n_embd` floats
(dflash.cpp:15). Example: 3 target layers x 4096 = 12288 floats ~= 48 KB/token
(F32). Features are needed for the committed context the drafter conditions on,
not the draft block itself, so storage scales with context length per example.
F16 halves it; Q8_0-style quantization is the likely production choice. This is
the dominant storage cost.

Capture tap already exists in the runtime - no new primitive needed:
`set_embeddings_layer_inp(lid, enable)` / `set_embeddings_nextn`
(llama-context.cpp:1168-1188), extracted during any trunk forward
(llama-context.cpp:1581-1589). Wiring this into the telemetry tool (imatrix
already runs the trunk forward) is the capture path.

## 3. The weighted-CE insertion point (verified, vindicates piece-3)

- ggml's built-in CE (`ggml_cross_entropy_loss`, ggml.c:6195) requires
  same-shape logits and labels and computes the DENSE CE
  `-sum(b * log_softmax(a))`.
- The llama training layer builds that dense label tensor at graph time
  (llama-context.cpp:3517-3527): zero an F32 [n_vocab, n_ubatch] tensor, then
  write `labels[target, pos] = 1.0` from sparse I32 dataset labels.

That `1.0` is the D-PACE insertion point. Write `dpace_weights[pos]` instead and
the CE becomes exactly `sum_j w_j * (-log q(y_j))` - the D-PACE objective,
gradient scaled per position, with NO new ggml-opt loss type. Weights are
gradient-detached constants baked into the dataset.

## 4. Architecture: offline feature capture (Path 1)

| | Path 1: capture features offline | Path 2: run trunk at train time |
|---|---|---|
| Train-time memory | drafter only | trunk + drafter |
| Idle-time friendly | yes | no (35B trunk resident) |
| Storage | features are large | none |
| Precedent | how EAGLE trains | - |

Path 1 is the on-device-correct choice. Extend telemetry collection so that,
while the trunk runs forward, the target-layer hidden states are extracted per
block context and stored alongside the block. The dataset grows a `features`
field (quantized).

### Reuse vs. own

Reuse as-is: ggml-opt (dataset, opt context, AdamW, epoch); the dense CE loss;
`export-lora`; the D-PACE weight math already baked into the dataset.

Owns:
1. Feature-augmented dataset loader: dflash-block JSONL + features sidecar ->
   ggml_opt_dataset. Keep labels sparse (I32 block targets [B]) + a parallel
   F32 weights array [B] per datapoint; build the dense weighted one-hot at
   graph time (do NOT store [n_vocab, ...] dense).
2. Weight-aware label fill: generalize llama-context.cpp:3517-3527 to take an
   optional per-position weight, default 1.0 so finetune is unchanged. This is
   the one llama-layer change; it must be additive/backward-compatible.
3. Block-diffusion training graph driver: combined encoder+decoder forward,
   weighted CE on the B drafted positions, backprop into drafter params only.
4. CLI + guard eval: acceptance-rate before/after on held-out blocks (reuse
   ts_lk_acceptance_rate / ts_dpace metrics), honest exit codes, dry-run default.

### Feature capture: LANDED

The offline capture tooling (risk #1's first half) is implemented as a
dedicated trunk-only pass in `llama-imatrix`, NOT interleaved into the fragile
speculative telemetry loop (per the constraint above):

- CLI: `llama-imatrix -m trunk.gguf -f calib.txt --features-out <prefix>
  --feature-layers <csv>`. `--feature-layers` is the drafter's
  `target_layer_ids` in concatenation order (e.g. `0,15,31`); the pass needs
  only the trunk + calibration text, no drafter load. Mutually exclusive with
  `--model-draft`.
- Mechanism: enables the existing runtime tap
  `llama_set_embeddings_layer_inp(lid, true)` per target layer, runs the plain
  chunked forward (same structure as `compute_imatrix`, KV cleared per chunk),
  and streams `llama_get_embeddings_layer_inp(lid)` token-major to disk.
  Requesting logits for every token keeps `output_reorder` on the exact proven
  `compute_imatrix` path, so layer buffers come back in batch order.
- Format (owned by `tools/quantize/tessera/tessera-features.{h,cpp}`, shared
  with the future training driver): `<prefix>.bin` = raw f32 blob, row-major
  `[n_tokens, n_layers*n_embd]`, layers concatenated in `--feature-layers`
  order; `<prefix>.json` = header. Schema `llama.tessera.features.v1`. The
  encoder's FC input is a flat read of `row_floats = n_layers*n_embd` floats
  per token. F16/Q8_0 are reserved in the header `dtype` field but not yet
  implemented (f32 is exact and removes conversion risk from the critical path).
- Verified live (stories260K tiny llama, n_embd=64): 23,552 tokens x 192
  floats captured; header/blob size exact; values finite and per-token
  distinct. Gold-standard cross-check: a separate single-layer-1 capture
  matches the layer-1 block of the 3-layer blob BIT-EXACTLY (max diff 0.0 over
  1,507,328 floats), proving layer order + token alignment + deterministic
  batch-order readback. Unit tests: `test_features.cpp` (35 assertions).

Context note: like `compute_imatrix`, KV is cleared per chunk, so each chunk's
first tokens are processed without left context - features are
context-windowed, matching how the drafter conditions at train time.

## 5. Staged plan

- Stage 0 (done): 0a differentiable YES; 0b layout nailed (off-by-one noted);
  0c feature dim + capture tap confirmed. Offline feature-capture tooling
  LANDED (see section 4 "Feature capture: LANDED") - the trunk-only imatrix
  pass + the tessera-features file format. Stage 1 now consumes real captured
  features instead of a hand-built block.
- Stage 1 (throwaway spike): hand-build one feature-augmented block, run one
  epoch of plain CE (weight = 1.0) through the generalized label fill, confirm
  loss decreases. Proves the encoder+decoder+CE plumbing.
- Stage 2: add the weight side-channel; A/B D-PACE vs decay weights on the same
  blocks, confirm the gradient reweights (cross-check ts_dpace_ab_compare).
- Stage 3: CLI (--tessera-dflash-train or a dedicated tool), LoRA export, guard
  eval.
- Stage 4: wire into the Swift RunTrainingTool plug-in point.

## 6. Risks, ranked

1. Feature pipeline is the real project. The capture tooling has LANDED
   (imatrix trunk-only pass + tessera-features format, section 4). Remaining:
   the train-time feature loader (features sidecar -> ggml_opt_dataset, own
   item 1 above) and storage/quantization (f16/Q8_0; f32 capture works today
   but is the dominant storage cost at corpus scale).
2. Combined encoder+decoder training graph. The two-phase inference path must
   be fused into one differentiable step; the MASK-token embedding and the
   borrowed tok_embd/output setup are the fiddly parts. Stage 1 is the gate.
3. Variable block size. Telemetry emits variable n_dft; the training graph
   wants fixed block_drafts. Fix block_size, pad with weight 0 (free).
4. Feature storage size / quantization fidelity.
