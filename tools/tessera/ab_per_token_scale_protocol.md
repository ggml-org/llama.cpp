# A/B protocol: per-token dynamic scale (in_kernel vs auxiliary)

Status: design only.  Implementation deferred to Phase F (W4A4).
Owner: Tessera IterQuant L5 orchestrator.
Last revised: 2026-07-30.

## Context

The W4A4 calibration path computes a per-token dynamic scale

    scale_t = max_c |X[t, c]| / 7

and uses it before the W4A4 FMA so that int4 weight matmuls against
per-token-int4 activations remain in the representable range.  The
design question (Q4 from the W4A4 plan) is how the scale reaches the
kernel:

- `in_kernel`     The kernel re-derives `scale_t` from the loaded
                  activations before the FMA.  The imatrix does not
                  carry the scale on the wire; the calibration step is
                  purely per-tensor statistics.
- `auxiliary`     The calibration step emits a per-token F32 tensor
                  alongside the v2 imatrix.  The kernel reads the
                  auxiliary tensor and skips the recomputation.
- `both`          The imatrix carries the F16-compressed per-token
                  scale; the kernel may use it as a fast path or fall
                  back to re-derivation if the auxiliary tensor is
                  missing.

The two paths are not free.  `in_kernel` adds one F32 reduction per
token at matmul time.  `auxiliary` costs ~1 MB of F16 per tensor at
calibration time and a separate per-token tensor at inference time.
On a 7B model with 128 calibration samples of 2048 tokens and 250
tensors, `auxiliary` is ~125 MB on disk and ~125 MB resident at
inference.  `in_kernel` adds roughly 1 F32 max per output token per
quantized matmul; for a 7B at 1024 tokens that is ~32M extra
reductions, or a few percent of kernel wall-clock on AVX2.

We do not know which path is the right production default.  This
document specifies the A/B protocol that will resolve the question.
The protocol is owned by the IterQuant L5 orchestrator and runs once
per W4A4 release candidate; the result is a single declaration
("`in_kernel` is the production default; `auxiliary` remains
opt-in") that the L5 path consumes.

## Configurations

Two configurations.  Both run the same model, the same calibration
corpus, the same L4 E2E probe, and the same L2 runtime probe.  The
only difference is the per-token scale transport.

| Config  | Calibration step  | Kernel              | Imatrix v2 |
| ------- | ----------------- | ------------------- | ---------- |
| `in_kernel`   | per-tensor only     | re-derive scale_t   | `per_token_dynamic_scale.mode = "in_kernel"`, scale values are zero / mirror |
| `auxiliary`   | emit F16 scale tensor | read scale_t from auxiliary | `per_token_dynamic_scale.mode = "auxiliary"`, scale values are authoritative |

The `both` mode is excluded from the A/B; it is the parity-check
configuration and is exercised by a separate small unit test in
`tests/test-imatrix-v2.py`.

## What stays fixed

- Model.  Use the gemma 4 12B Tessera-corrected GGUF (the existing
  smoke model) and the gemma 4 12B BF16 source.  Same on both arms.
- Calibration corpus.  Use `data/wiki.train.raw` truncated to 128
  chunks of 2048 tokens.  Same on both arms.  The corpus hash goes
  into the imatrix v2 provenance block.
- L4 E2E probe.  `tools/tessera/e2e_probe.py` (Layer 4 in
  `docs/runtime-aware-pipeline.md`) with the four standard prompts
  (`paris`, `gsm8k-easy`, `multi-turn`, `code`).  Same on both arms.
- L2 runtime probe.  `tools/tessera/runtime_probe.py` (Layer 2) with
  the same corpus.  Same on both arms.
- L1 sidecar.  Generated once for the model, reused for both arms.
- W4A4 policy.  The L5 result of the L4+L2 loop is the W4A4 policy
  fed to `tools/tessera/per_tensor_calibrate.py --fitness
  kernel-direct`.  Both arms run the same L5 loop with the same seed.
- The per-tensor GA gene (`llm_int8_threshold`) is fixed at the
  default 6.0 for both arms; the GA sweep is a separate experiment.

## What changes

Only the per-token scale transport.  Concretely, in the W4A4
calibration driver:

    in_kernel    -> calibrate --per-token-scale-mode in_kernel
    auxiliary    -> calibrate --per-token-scale-mode auxiliary

The driver must honour the existing CLI flag
`--per-token-scale-mode {in_kernel, auxiliary, both}` that
`tools/imatrix/imatrix.py` already accepts.  The driver then either
re-derives `scale_t` in the kernel or reads the auxiliary tensor
the calibration step wrote to disk.

## Metrics

We collect four metric families per arm.  All metrics are recorded in
a `llama.tessera.ab-protocol.v1` JSON report written under
`reports/ab-per-token-scale/<date>/<arm>.json`.

### 1. Quality (the dominant signal)

- `ppl_delta`  Perplexity delta of the W4A4 model vs the BF16 source,
               measured on the calibration corpus.  Reported as
               `rel_ppl = exp(log(ppl_w4a4) - log(ppl_bf16)) - 1`.
               Both arms should be within 1% of each other; the
               smaller delta wins.
- `top1_match_rate`  Top-1 token match rate on the L4 E2E probe
               prompts (averaged across the four prompts).  This is
               the L4 metric.  Both arms should be equal or very
               close; the arm with the higher rate wins ties.
- `kl_per_token`  Per-token KL divergence between the W4A4 and BF16
               distributions on the L3 probe (per-token coherence).
               Summed across the 50-token probe.

### 2. Performance (the deciding signal if quality is tied)

- `kernel_wall_clock_ms_per_token`  Median kernel wall-clock per
               output token, measured on the L4 prompt set with
               `LLAMA_TILE640_DEBUG_DEQUANT=0` (no L1 capture
               overhead).  We expect `auxiliary` to be ~1-3% faster
               than `in_kernel` because the F32 reduction moves
               offline.
- `peak_memory_mb`  Peak resident memory during a single 1024-token
               forward.  `auxiliary` is expected to add ~125 MB
               resident for a 7B; `in_kernel` is expected to add
               nothing.
- `compile_time_seconds`  Wall-clock of the ggml/Metal compile path
               for the W4A4 kernel template.  We expect this to be
               identical (the kernel is the same object file; only
               the runtime dispatch differs) and use it as a
               sanity check on the experimental setup.

### 3. Integration cost (the tie-breaker if quality and performance are tied)

- `awq_integration_patches`  Number of source lines changed in the
               existing AWQ path to support the transport.  `in_kernel`
               is expected to be 0 (no AWQ path changes; the kernel
               re-derives the scale from data it already has).
               `auxiliary` is expected to be ~50-100 lines (a new
               kernel argument, a new tensor lookup, and a fallback
               path for the legacy in_kernel mode).
- `metal_shader_dispatch_changes`  Number of Metal shader source
               lines changed.  `in_kernel` should be 0; `auxiliary`
               should be ~10 lines.
- `cuda_kernel_dispatch_changes`  Same as above for CUDA.

### 4. Robustness (informational)

- `scale_quantization_error`  RMSE between the in_kernel-derived
               scale and the auxiliary-stored scale across the
               calibration set.  The F16 compression of the
               auxiliary path is expected to add < 1% relative
               error on the scale values.
- `first_token_kl`  KL divergence on the first generated token
               (the "Paris" test).  Both arms should be equal; this
               is a sanity check on the calibration set.

## Acceptance criterion

The L5 orchestrator declares the winner based on the following
ordered rules.  Ties at any level fall through to the next level.

1. **Quality gate.**  If `ppl_delta` differs by more than 0.5% in
   relative terms, the lower-delta arm wins regardless of
   performance.  Same for `top1_match_rate` if the difference
   exceeds 1 percentage point.
2. **Performance gate.**  If quality is tied, the arm with the
   lower `kernel_wall_clock_ms_per_token` wins, subject to a
   `peak_memory_mb` budget of 1.5x the other arm.  `auxiliary`
   is allowed up to +125 MB on a 7B; beyond that, it loses.
3. **Integration gate.**  If quality and performance are tied, the
   arm with the lower `awq_integration_patches +
   metal_shader_dispatch_changes + cuda_kernel_dispatch_changes`
   sum wins.  `in_kernel` is expected to win this gate; it is
   the tie-breaker for the production default.
4. **Robustness sanity check.**  Both arms must show
   `scale_quantization_error < 5%` and `first_token_kl` within
   the noise floor.  If either arm fails, the A/B is rerun with a
   larger calibration set.

The declaration is a single line in the report:

    {"winner": "in_kernel" | "auxiliary", "margins": {...}, "rules_fired": [1, 3]}

The L5 orchestrator reads this line at boot and dispatches
accordingly.  The losing arm remains available as
`--per-token-scale-mode <loser>` for the A/B rerun and for
experimentation.

## What we expect (the prior)

We expect `in_kernel` to win on quality ties and on the integration
gate.  We expect `auxiliary` to be 1-3% faster on kernel wall-clock
but to add ~125 MB resident memory and ~50 lines of AWQ/Metal/CUDA
plumbing.  On a 7B model that memory hit is uncomfortable; on a
1-3B model it is negligible.  The prior is therefore:

- **7B+ models:** `in_kernel` is the production default.
- **1-3B models:** `auxiliary` is the production default if the
  performance gate fires (which we expect it will, on a 1-3B the
  absolute kernel time is small and the reduction saving is a
  larger fraction of the total).

This is a prior, not a decision.  The A/B exists to falsify it.

## What this protocol does not cover

- The per-tensor GA sweep of `llm_int8_threshold`.  That is a
  separate experiment; the protocol fixes the threshold at 6.0 on
  both arms.
- The mode `both`.  `both` is a parity-check configuration; it is
  exercised by a unit test that compares the F16-compressed
  auxiliary scale to the in_kernel-derived scale on a synthetic
  activation set.  See `tests/test-imatrix-v2.py::test_both_mode_parity`.
- The choice of W4A4 qmin/qmax.  Both arms use the existing
  `pe_qat.py::W4A4_QMIN=-8, W4A4_QMAX=7`.
- The E2 (Tile640+CSR) format change that will eventually store the
  F16 outlier *weight* values (SpQR-style CSR blob).  Per the Q5
  update, the imatrix v2 carries both the activation-outlier
  *indices* AND the F16 activation-outlier *values*, so the GA's
  LLM.int8 decomposition can be reconstructed from the imatrix alone
  and the E2 work is a separate (and orthogonal) workstream.

## How to run

When the W4A4 calibration driver lands:

    tools/tessera/imatrix.py compute \
        --input-dir data/calib/wiki.train.raw.tok-2048 \
        --output reports/ab-per-token-scale/in_kernel.imatrix.gguf \
        --per-token-scale-mode in_kernel \
        --llm-int8-threshold 6.0 \
        --per-family \
        --telemetry-model gemma4-12b-tessera-v1 \
        --telemetry-calibration-corpus data/calib/wiki.train.raw \
        --telemetry-calibration-corpus-hash $(sha256sum < data/calib/wiki.train.raw | cut -c1-64) \
        --tessera-main-tip $(git rev-parse HEAD)

    tools/tessera/imatrix.py compute \
        --input-dir data/calib/wiki.train.raw.tok-2048 \
        --output reports/ab-per-token-scale/auxiliary.imatrix.gguf \
        --per-token-scale-mode auxiliary \
        ...  # same flags as in_kernel

Then run the L4+L2+L3 probe on each arm and record the report under
`reports/ab-per-token-scale/<date>/`.  The `ab_validate.py` driver
emits the winner line and updates the L5 dispatch config.

## References

- `docs/runtime-aware-pipeline.md` -- L1-L6 calibration pipeline
  design.
- `tools/tessera/per_tensor_calibrate.py` -- the per-tensor GA that
  consumes the L2/L4 signals.
- `tools/imatrix/imatrix.py` -- the imatrix v1/v2 tool this protocol
  exercises.
- `tools/tessera/llm_int8_decompose.py` -- the LLM.int8 decomposition
  reader; the W4A4 kernel and the GA's fitness eval read the F16
  outlier values out of the imatrix through this tool.
- `tools/tessera/imatrix_v2_demo.py` -- the synthetic demo that
  validates the v2 schema before the W4A4 path lands.
