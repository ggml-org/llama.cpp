# Tessera Architecture

## Goals

1. **Constitutional quantization**: the per-tensor quantizer must be
   calibrated, with audit trail, schema-versioned policies, and
   reproducible runs.
2. **Runtime-aware calibration**: the offline reference and the C++
   runtime must agree to within F16 precision. The Tile640 kernel
   must expose a debug mode that emits its actual dequantized weights
   so calibration can be measured against the runtime, not a
   reference.
3. **First-class drafter integration**: DFlash / DSpark drafters must
   load as first-class architectures with proper model class hierarchy
   and no legacy-naming fallbacks.
4. **Spec-decoding telemetry**: per-step telemetry with verifier and
   drafter top-k distributions, versioned schema, used for drafter
   fine-tuning.

## Invariants

These invariants are part of the public contract of the project. Any
change that breaks an invariant requires a major version bump and a
deprecation cycle.

### Calibration schema invariants

- The `llama.tessera.per-tensor-calibration.v1` schema is append-only.
  New fields may be added, but existing field semantics are
  immutable. Removing a field requires `v2` with an explicit migration
  path.
- The `llama.speculative.calibration-policy.v1` policy schema
  (consumed by `tile640_quantize_v3.py`) is also append-only. The
  `tensor_families` map keys are stable.
- The `tessera.gemma4.sliding_window_override`,
  `tessera.range_selection`, and `tessera.calibration.activations_source`
  fields in the output GGUF are stable.

### Runtime invariants

- The Tile640 matmul kernel's dequant math (f16 precision) is the
  ground truth for calibration. Any change to the dequant must come
  with a calibration delta showing the new threshold search.
- The `dft.` observer protocol is a stopgap (see
  `docs/audit-2026-07-29.md`). It's being replaced with per-context
  observer state; the new protocol must support both the verifier
  and the drafter without string-prefix hacks.
- The `--no-embedded-mtp` flag bypasses the auto-MTP trigger. It's
  required for any model with `mtp.component.present = true` until the
  MTP path is fully wired. Removing the flag requires a
  backwards-compat shim that keeps the behavior.

### Drafter invariants

- DFlash / DSpark drafters load as `LLM_ARCH_DFLASH` with the
  markov head detected by tensor presence. The
  `LLM_ARCH_DSPARK` is folded into `LLM_ARCH_DFLASH` per upstream
  PR #25173; the conversion is one-way.
- The drafter's verifier is always the same model that produced the
  draft. The verifier and drafter share a tokenizer.

## Subsystems

### `tools/tessera/` — quantizer tooling (Python)

Pure-Python. No C++ bindings, no GPU dependencies. The cluster of
tools:

- `awq_evolve.py` — multi-generation GA over `(alpha, clip,
  outlier_fraction, moment_mix, tail_guard, ternary_threshold)`. The
  search has two levels: a vectorized inner population and
  island populations with a MAP-Elites archive. The `ternary_threshold`
  field was added 2026-07-29 to expose the missing calibration knob.
- `per_tensor_calibrate.py` — per-tensor GA. Smaller, faster, runs
  on the calibrated layer bundles. The output is a per-tensor JSON
  policy consumable by `tile640_quantize_v3.py`.
- `shadow_calibrate.py` — provisional reconstruction scoring.
  Generates a `shadow-policy.json` with the worst-overall-policy
  tensors forced to higher outlier fractions.
- `make_awq_layer_bundles.py` — exports per-tensor weight + imatrix
  snapshots as `.npz` bundles for the GA tools to consume.
- `unsloth_policy.py` — bridges Unsloth's sensitive-module guidance
  into the tessera policy format.
- `evidence_store.py` — appends observer / evolution / acceptance
  evidence as Parquet partitions.
- `hf-evidence.py` — pulls compatible aggregate evidence from the
  Hugging Face tessera calibration commons.

### `tools/ane-mtp/` — ANE prefill toolkit (Python + Objective-C)

The ANE prefill is a Core ML `.mlpackage` that runs the verifier's
prefill on the Apple Neural Engine. Status: WIP — the prefill
.mlpackage is built and the IOSurface async hand-off is implemented,
but the integration with the MTP context is not wired. Use
`--no-embedded-mtp` to bypass the auto-trigger.

### `tools/dspark-gguf-patch/` — legacy dspark preprocessor (Python)

Preprocessor for pre-PR-#25173 dspark drafters. Will be removed
when the legacy converter is no longer in production. See
`docs/audit-2026-07-29.md` for details.

### `tools/tile640/` — main quantizer (Python)

The user-facing quantizer entry point. `calibrate_quantize.py` is the
orchestrator; `quantize_v3.py` is the per-tensor writer. Both are
under `LICENSE-TESSERA` (PolyForm Noncommercial License 1.0.0).

### `tools/imatrix/` — llama-imatrix with spec hook (C++)

The standard llama-imatrix with a spec-decoding hook: when
`--model-draft` is set, it runs the verifier + drafter together and
emits per-step accept/reject telemetry. The hook bypasses
`common_speculative_*` due to off-by-one KV bugs upstream; this is
documented in `docs/audit-2026-07-29.md` and is on the hardening
roadmap.

### `common/`, `src/`, `tools/server/` — C++ additions to llama.cpp

The spec hook, dflash/dspark drafter, MTP wiring, and the imatrix
observer epoch. See `docs/audit-2026-07-29.md` for the per-file
verdict and the hardening roadmap.

## Merge strategy

Each hardening change lands on its own feature branch in an
isolated git worktree. Merges to `main` happen after:

1. The branch's tests pass.
2. The merge applies cleanly with no conflicts.
3. The full test suite passes after the merge.

The merge is performed by the orchestrator (the user's
representative in the conversation). The orchestrator does not
push directly to `main` from a feature branch; merges are rebased
on top of the current `main` and pushed atomically.

For the worktree layout see `docs/architecture-worktrees.md`.
