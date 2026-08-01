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
  --feature-layers <csv> [--features-warmup N]`. `--feature-layers` is the
  drafter's `target_layer_ids` in concatenation order (e.g. `0,15,31`); the
  pass needs only the trunk + calibration text, no drafter load. Mutually
  exclusive with `--model-draft`. `--features-warmup` (default 256, clamped to
  n_ctx-1) is the per-window context primer: windows overlap by exactly N
  tokens (stride = n_ctx - N) so the emitted rows form one contiguous corpus
  sequence and every emitted token still sees >= N genuine left-context tokens
  - see the context note below.
- Mechanism: enables the existing runtime tap
  `llama_set_embeddings_layer_inp(lid, true)` per target layer, runs the plain
  windowed forward (same structure as `compute_imatrix`, KV cleared per
  window), and streams `llama_get_embeddings_layer_inp(lid)` token-major to
  disk. Windows advance by `stride = n_ctx - warmup` (not n_ctx), so each
  window re-decodes the previous window's trailing `warmup` tokens to prime its
  KV; only positions [warmup, n_ctx) are emitted, giving a contiguous output.
  Requesting logits for every token keeps `output_reorder` on the exact proven
  `compute_imatrix` path, so layer buffers come back in batch order.
- Format (owned by `tools/quantize/tessera/tessera-features.{h,cpp}`, shared
  with the future training driver): `<prefix>.bin` = raw f32 blob, row-major
  `[n_tokens, n_layers*n_embd]`, layers concatenated in `--feature-layers`
  order; `<prefix>.json` = header. Schema `llama.tessera.features.v1`. The
  encoder's FC input is a flat read of `row_floats = n_layers*n_embd` floats
  per token. The header records `chunk_tokens` (window size), `warmup`, and
  `stride` (window advance); in overlap mode `stride == chunk_tokens - warmup`
  and the emitted rows are contiguous, so `ts_features_row_to_token(header, r)`
  reduces to `warmup + r`. Legacy captures (stride absent/0, advanced by a full
  window and dropped a warmup prefix per window) keep the old gappy mapping.
  F16/Q8_0 are reserved in the header `dtype` field but not yet implemented
  (f32 is exact and removes conversion risk from the critical path).
- Verified live (stories260K tiny llama, n_embd=64): with n_ctx=512, warmup=64
  (stride=448) over 23,602 corpus tokens -> 52 windows x 448 = 23,296
  contiguous rows, header/blob size exact, 1.14x decode overhead. Gold-standard
  cross-checks, all BIT-EXACT (max diff 0.0): (a) layer order + alignment - a
  single-layer-1 capture equals the layer-1 block of a [0,1,2] 3-layer blob
  over 1,490,944 floats -> concatenation order, token alignment, deterministic
  batch-order readback; (b) determinism - the full 52-window capture is
  byte-identical across two runs; (c) window-0 invariance - a warmup=64
  single-window capture (448 rows, tokens 64..511) equals rows 64..511 of a
  warmup=0 single-window capture (512 rows, tokens 0..511) over 344,064 bytes
  -> the warmup/overlap machinery changes only WHICH tokens are emitted, never
  the decode. Multi-window contiguity then holds by composition: every window
  runs the same self-contained KV-cleared decode (proven by c), tiled
  contiguously (proven by the row->token math + the live 23,296 count). Unit
  tests: `test_features.cpp` (62 assertions, incl. stride round-trip, stride
  validation, and the overlap/legacy row->token mappings).

Context note: like `compute_imatrix`, KV is cleared per window, so a window's
first tokens lack a full left window. We DECODE those `warmup` tokens (they
build context for the rest of the window) but do NOT emit their features.
Windows advance by `stride = n_ctx - warmup`, overlapping by exactly `warmup`
tokens, so the re-decoded primer is the previous window's tail: the emitted
rows form ONE contiguous corpus sequence (row r == corpus token warmup + r)
rather than dropping a warmup prefix per window, and every emitted token sees
>= warmup genuine left-context tokens spanning the window boundary. This is
mild and expected: the bulk of each window sees a full ~n_ctx window - the same
regime the trunk runs in at inference (finite KV cache) and the regime
EAGLE-style feature capture trains in. `compute_imatrix` applies the same idea
for perplexity (`first = n_ctx/2`, imatrix.cpp:1683); we use a smaller fixed
warmup to avoid discarding half the corpus. The overlap costs n_ctx/(n_ctx -
warmup) decode (~1.07x at warmup=256, n_ctx=4096) and recovers the
warmup-per-window tokens the non-overlap layout threw away.

## 5. Staged plan

- Stage 0 (done): 0a differentiable YES; 0b layout nailed (off-by-one noted);
  0c feature dim + capture tap confirmed. Offline feature-capture tooling
  LANDED (see section 4 "Feature capture: LANDED") - the trunk-only imatrix
  pass + the tessera-features file format. Stage 1 now consumes real captured
  features instead of a hand-built block.
- Stage 1 (throwaway spike): hand-build one feature-augmented block, run one
  epoch of plain CE (weight = 1.0) through the generalized label fill, confirm
  loss decreases. Proves the encoder+decoder+CE plumbing. Assert the
  token-fidelity invariant from section 7.1 (one shared tokenization across
  features, block tokens, and vocab; no detokenize -> retokenize).
- Stage 2: add the weight side-channel; A/B D-PACE vs decay weights on the same
  blocks, confirm the gradient reweights (cross-check ts_dpace_ab_compare).
- Stage 3: CLI (--tessera-dflash-train or a dedicated tool), LoRA export, guard
  eval.
- Stage 4: wire into the Swift RunTrainingTool plug-in point. Harvest traces via
  an interception layer on the inference path, not by instrumenting the agent
  harness (section 7.1).

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

## 7. Driver-design principles (Prime Intellect post-training playbook)

Source: Will Brown (Prime Intellect), "Modern Post-Training: A Deep Dive",
AI Engineer 2026-07 (transcript captured 2026-07-31). Prime Intellect is the
leading open-source post-training shop; their Verifiers + Prime RL stack is the
cloud-scale blueprint for the SAME flywheel Tessera builds on-device (harvest
real-world signal, ground updates in verifiable outcomes, amortize training
compute against inference). Take the CONCEPTS, not the code: Prime is
Python/PyTorch/Torch Titan on GPU clusters and is egress-heavy (cloud
sandboxes, hosted platform, API teachers), none of which fits Tessera's native
C++/Swift + ggml-opt + no-egress doctrine. The alignments below are why the
driver should be built the way it is, plus two concrete hazards to design out.

### 7.1 Two concrete cautions

**Token fidelity (the "renderers" hazard).** Tokenization is many-to-one, so a
text -> tokens -> text round-trip (a chat template stripping a stray newline)
silently changes the token stream. Prime found this causes trainer/inference
mismatch and off-policy drift that only surfaces LATE in long runs; they built
"renderers" (programmable chat templates + logical-prefix-hit detection) to
keep dual message/token streams consistent.

- The DFlash/LK drafter drivers are SAFE BY CONSTRUCTION if they stay in token
  space. Features are keyed by corpus token index (ts_features_row_to_token);
  block labels are token IDs (target_tokens[k] / verifier_argmax); densification
  writes straight into the ggml label tensor (the LK driver already does this,
  no detokenize/retokenize). There is no rendered text in the training path.
- INVARIANT TO ASSERT AT DRIVER LOAD: feature rows, block-dataset tokens, and
  the model vocab share ONE tokenization. This holds by construction - the
  drafter borrows tok_embd + output (lm_head) from the trunk via ctx_other, so
  its vocab IS the trunk vocab; spec_calib.v2 verifier_argmax/target_tokens are
  trunk-vocab IDs; features are captured over the trunk-tokenized corpus - but
  the driver should assert tokenizer/vocab identity explicitly and must NEVER
  detokenize -> retokenize anywhere in the training path.
- WHERE IT BITES LATER: the capability/coding-agent RL loop (the actual
  self-improving loop) trains on chat-templated agent trajectories. THAT path
  must capture tokens at generation time (as the LK driver does) and keep dual
  message/token streams, not reconstruct from rendered text. This is Tessera's
  real "renderers" concern; the drafter drivers avoid it by staying token-side.

**Interception server (trace-harvest wiring).** Prime's pattern: the harness
talks to a fake OpenAI/Anthropic-compatible endpoint; a thin layer intercepts
each request/response, injects logprob capture + temperature, and routes to the
trainer - the harness never knows it is being trained. This decouples
running-the-agent from harvesting-signal, so the same harness runs identically
in deploy and train mode.

- For the DFlash OFFLINE driver this is background (it trains on already
  captured features + traces).
- For the self-improving-loop wiring (Stage 4, Swift RunTrainingTool /
  orchestrator) this is the right shape: harvest traces via a capture layer on
  the INFERENCE PATH (intercept request/response; record tokens + logprobs +
  accept/reject), NOT by instrumenting the agent harness. Tessera already does a
  version of this - spec_calib telemetry intercepts the spec-decode loop; imatrix
  feature capture intercepts the trunk forward. Frame the orchestrator's trace
  harvesting as an interception layer so TesseraAgentLoop stays unmodified.

### 7.2 Design principles to build to

- Loss / algorithm split. Keep the LOSS (weighted CE / D-PACE - the thing taking
  the gradient) separable from the ALGORITHM / data-prep (block assembly,
  dpace_weight baking, feature loading). Concretely: the generalized weighted-CE
  label fill (llama-context.cpp:3517-3527, write dpace_weight not 1.0) is the
  pure loss-side change; block layout + weight baking + feature load stay
  driver-side. This is what keeps finetune unchanged (weight defaults 1.0) and
  makes D-PACE-vs-decay a DATA-side swap (Stage 2 A/B). Mirrors the LK driver
  (densify = algorithm, GGML_OPT_LOSS_TYPE_LK = loss).
- Advantage-as-score. Plain CE = per-position weight 1; D-PACE = adaptive
  per-position weight; LK = acceptance-rate objective. If another drafter
  objective appears, factor it as "swap the loss target / the per-position
  weight", not "write a new training loop". D-PACE's adaptive weights ARE the
  drafter analogue of Prime's position/group reward shaping.
- Teacher = offline verifier traces (OPD framing). On-policy distillation =
  student sequences -> teacher prefill -> reference logprobs -> train. For the
  drafter the teacher is the trunk VERIFIER, and spec_calib.v2 (verifier_argmax
  / verifier_topk) is that teacher signal captured OFFLINE. So DFlash/LK
  training is already distillation from the trunk-as-teacher; no live teacher is
  needed. Live cloud teachers (escalation, decision #9) belong to the
  capability/agent layer, formalizable later as OPD with a teacher pool.
- Async / off-policy is fine. Idle-time training is inherently off-policy
  (traces harvested during usage, trained later). Prime runs ~16 steps off-policy
  BY DESIGN and decouples inference-server from trainer. Do not over-engineer
  on-policyness for the drafter loop (the drafter changes slowly; traces are
  recent). Keep harvest (continuous) decoupled from train (idle-triggered) -
  which TesseraTrainingOrchestrator already does (gates on trace count + idle).
- Group / variance rewards (later, capability axis). For the capability/coding
  loop, adopt group-variance conciseness bonuses (bonus the shortest CORRECT
  answer; optimal length cannot be known up front, so use group variance) to
  bound chain-of-thought - directly addresses agent-loop-runaway (R27). Not
  drafter-relevant now.

Stage hooks: Stage 1 asserts the token-fidelity invariant (7.1); Stage 4 wires
harvesting as an interception layer (7.1).
