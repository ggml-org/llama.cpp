## Gemma4A debug handoff (2026-04-02)

Recent breakthrough:

- The conformer collapse after layer 0 was caused by duplicate tensor duplication in `clip_model_loader::load_tensors()`.
- Gemma4A requests some tensors twice:
  - once in the generic layer-loading pass
  - again in the Gemma4A-specific pass
- `get_tensor()` used to `ggml_dup_tensor()` every request into `ctx_data`, even for the same tensor name.
- That allowed `model.layers[il].ln_2_w` to end up pointing at a later zero-initialized duplicate while file data was loaded into the first duplicate found by name.

Concrete evidence:

- `a.blk.0.ln2.weight` on disk is nonzero.
- `LOAD_DBG` in `clip.cpp` showed it loading correctly into `ctx_data`.
- Before the dedupe fix:
  - `g4a_l0_out_norm` was nonzero
  - `g4a_l0_out` was all zeros
  - conformer output collapsed
- After deduping `get_tensor()` by tensor name:
  - `g4a_l0_ln2_w` is nonzero in-graph
  - `g4a_l0_out` is nonzero
  - `g4a_l1_out` is nonzero
  - `g4a_conformer_out` is nonzero for nonzero input

Code changes relevant to this breakthrough:

- `tools/mtmd/clip.cpp`
  - `load_tensors()` now keeps a `data_tensors` map and reuses existing duplicated tensors by name.
  - extra `LOAD_DBG` prints were added for:
    - `a.blk.0.ln2.weight`
    - `a.blk.0.conv_norm.weight`
  - extra tensor dumps were added around:
    - `g4a_l0_out_norm`
    - `g4a_l0_ln2_w`
    - `g4a_l0_ln2_w_rep`
    - `g4a_l0_out`
- `tools/mtmd/models/gemma4a.cpp`
  - layer 0 final norm path is temporarily instrumented.

Current status after the dedupe fix:

- `direct_zero.log`
  - `G4A_CONFORMER_OUT` is zero
  - `G4A_VEC` falls back to the constant bias-driven embedding
- `direct_half.log` / `direct_one.log`
  - `G4A_CONFORMER_OUT` is nonzero
  - `G4A_VEC` is no longer the old constant value
  - `half` and `one` are still very close, but they now differ slightly

Interpretation:

- The "identical constant output regardless of input" bug is broken.
- The loader duplicate bug was real and critical.
- Remaining work is now back to semantic/audio fidelity, not total encoder collapse.

Useful logs:

- `direct_zero.log`
- `direct_half.log`
- `direct_one.log`
