# IFAIRY64 LUT Implementation Plan

Status: Draft (2026-04-21)

Scope of this document:

- Add a dedicated LUT acceleration path for `GGML_TYPE_IFAIRY64`
- Keep the current `tile64_v2` weight semantics and the `qwen2_real` attention layout
- Keep the current activation quantization for now
- Do not regress the existing `GGML_TYPE_IFAIRY` / 7B Fairy2i LUT path

This is intentionally a transition plan. It is not the final weight-only design for 32B Fairy2i. The goal is to unlock a faster path for the currently-correct `tile64_v2` model with minimal semantic risk.

## 1. Current State

The current 32B path is:

- weight format: `tile64_v2`
- GGML type: `GGML_TYPE_IFAIRY64`
- attention layout: `qwen2_real`
- runtime matmul: `ggml_vec_dot_ifairy64_q16_K()`
- activation format consumed by vec-dot: `GGML_TYPE_IFAIRY_Q16`

What works today:

- correctness is good enough for usable output
- `IFAIRY64` now has an ARM SIMD vec-dot path

What is missing:

- `IFAIRY64` cannot use the existing LUT route
- old LUT routing only accepts `GGML_TYPE_IFAIRY`
- old LUT transform/preprocess/qgemm are hard-wired to `QK_IFAIRY == 256`

Key consequence:

- `GGML_IFAIRY_LUT=1` currently has no effect for `tile64_v2` 32B models

## 2. Reuse Assessment

The old LUT path cannot be enabled for `IFAIRY64` as-is.

Hard blockers:

- `ggml_ifairy_lut_can_mul_mat()` only accepts `GGML_TYPE_IFAIRY`
- `ggml_ifairy_lut_transform_tensor()` only accepts `GGML_TYPE_IFAIRY`
- `ggml_ifairy_2w_encode()` reads `block_ifairy`
- LUT packed weight layout assumes `QK_IFAIRY == 256`
- qgemm loops assume `groups_per_block = QK_IFAIRY / 2 = 128`

What can still be reused:

- the threadpool config and env plumbing in `ggml-cpu.c`
- the overall dataflow: `transform weights -> preprocess activations -> qgemm`
- 2-weight direct pattern indexing (`pat = c0 | (c1 << 2)`)
- 16-row tile organization
- the split between `lut16` and `lut_c` activation preprocess variants

Conclusion:

- direct reuse: no
- architectural reuse: yes
- engineering size: medium, not tiny, but not a full rewrite either

## 3. Decision

We will implement a parallel `IFAIRY64` LUT path instead of mutating the existing `IFAIRY` LUT path.

Design choice:

- old path remains untouched for 7B / legacy
- new path is keyed on `GGML_TYPE_IFAIRY64`
- activation quantization remains `GGML_TYPE_IFAIRY_Q16` for this phase

Why this is the right interim choice:

- it preserves current 32B quality
- it minimizes regression risk on the mature legacy path
- it gives us a realistic way to approach legacy-class throughput without first redesigning the whole runtime around weight-only activations

## 4. Target Throughput

Current reference points on this machine:

- `tile64_v2` with current SIMD vec-dot only: about `tg32 ~= 2.2 tok/s`
- `legacy` path with old fast route: about `tg32 ~= 3.2 tok/s`

Because `tile64_v2` weights are larger than `legacy`, the realistic short-to-medium target is lower than legacy:

- Phase 1 target: `tg32 ~= 2.5 - 2.6 tok/s`
- Phase 2 target: `tg32 ~= 2.7 - 2.9 tok/s`
- stretch target after deeper fusion: around `3.0 tok/s`

These targets assume:

- Apple M4 class ARM machine
- `-t 4`
- `-fa off`
- current `qwen2_real` attention layout

## 5. High-Level Architecture

The new route will mirror the old LUT flow, but with `64`-wide weight blocks:

1. Loader gives us `GGML_TYPE_IFAIRY64` tensors
2. `transform_tensor` builds an `IFAIRY64`-specific packed weight representation
3. activations are quantized exactly as today into `block_ifairy_q16`
4. `preprocess` builds LUT tables using `IFAIRY64` block/group mapping
5. `qgemm` consumes the packed `IFAIRY64` tiles plus activation LUT tables

The main difference from the old path is the logical block geometry:

- old: one weight block = `256` complex values = `128` 2-weight groups
- new: one weight block = `64` complex values = `32` 2-weight groups

## 6. File-Level Plan

### 6.1 `ggml/src/ggml-ifairy-lut.h`

Add new types and declarations.

Changes:

- add an `ifairy64_lut_wtile_16` packed tile struct
- add declarations for `IFAIRY64`-specific transform/preprocess/qgemm entry points
- keep the existing `ifairy_lut_wtile_16` unchanged

Recommended new struct:

```c
struct ifairy64_lut_wtile_16 {
    uint8_t qs[QK_IFAIRY64_GROUPS_PER_BLOCK / 2][16];
    float   d_real[16];
    float   d_imag[16];
};
```

New constants:

- `QK_IFAIRY64_GROUPS_PER_BLOCK = QK_IFAIRY64 / 2 = 32`

### 6.2 `ggml/src/ggml-quants.c`

Add `IFAIRY64`-specific 2-weight index encoding helpers.

Changes:

- add `ggml_ifairy64_read_code()`
- add `ggml_ifairy64_2w_get_index_info()`
- add `ggml_ifairy64_2w_index_buffer_size()`
- add `ggml_ifairy64_2w_encode()`

Important note:

- the old `ggml_ifairy_2w_encode()` should stay untouched
- the new encoder should read `block_ifairy64`
- because `IFAIRY64` already stores direct 2-bit phase codes, the encode rule remains:
  - `pat = c0 | (c1 << 2)`

### 6.3 `ggml/src/ggml-ifairy-lut-transform.cpp`

Add a second transform route for `GGML_TYPE_IFAIRY64`.

Changes:

- do not widen the old function by piling special cases into it
- split transform into:
  - existing `GGML_TYPE_IFAIRY` branch
  - new `GGML_TYPE_IFAIRY64` branch

Recommended structure:

- keep `ggml_ifairy_lut_transform_tensor()` as the public entry
- internally dispatch to:
  - `ggml_ifairy_lut_transform_tensor_ifairy()`
  - `ggml_ifairy_lut_transform_tensor_ifairy64()`

New `IFAIRY64` transform must:

- validate `tensor->type == GGML_TYPE_IFAIRY64`
- validate `k % QK_IFAIRY64 == 0`
- build index bytes with `ggml_ifairy64_2w_encode()`
- pack weights into `ifairy64_lut_wtile_16`

Important detail:

- current GGUF stores repeated scales row-by-row
- for now, the transform should simply copy those repeated per-block scales into the packed tile
- do not try to deduplicate or compress scales in Phase 1

### 6.4 `ggml/src/ggml-ifairy-lut.cpp`

Broaden routing and workspace sizing.

Changes:

- `ggml_ifairy_lut_can_mul_mat()` should accept:
  - `src0->type == GGML_TYPE_IFAIRY`
  - or `src0->type == GGML_TYPE_IFAIRY64`
- `ggml_ifairy_lut_get_wsize()` must compute the right group count for each type

Required branching:

- old:
  - `block_k = QK_IFAIRY = 256`
  - `groups_per_block = 128`
- new:
  - `block_k = QK_IFAIRY64 = 64`
  - `groups_per_block = 32`

Do not silently coerce one format into the other.

### 6.5 `ggml/src/ggml-cpu/ggml-cpu.c`

Add `IFAIRY64` LUT routing into `ggml_compute_forward_mul_mat()`.

Changes:

- keep the current `GGML_TYPE_IFAIRY` LUT route unchanged
- add a sibling route for `GGML_TYPE_IFAIRY64`

Important:

- activation quantization remains exactly as today:
  - `F32 -> quantize_row_ifairy_q16_tensor()` or `quantize_row_ifairy_q16_lut_c()`
- this is the main reason the old preprocess logic can be reused structurally

Recommended structure:

- split the big LUT branch into small helpers:
  - `ggml_ifairy_lut_mul_mat_ifairy(...)`
  - `ggml_ifairy_lut_mul_mat_ifairy64(...)`

That keeps old and new code paths readable and limits risk of accidental legacy regressions.

### 6.6 `ggml/src/ggml-ifairy-lut-qgemm.cpp`

This is the core implementation work.

Changes:

- add `IFAIRY64` variants of preprocess and qgemm
- keep old functions intact

Recommended new functions:

- `ggml_ifairy64_lut_preprocess_ex_lut16()`
- `ggml_ifairy64_lut_preprocess_ex_lut_c()`
- `ggml_ifairy64_lut_qgemm_lut16()`
- `ggml_ifairy64_lut_qgemm_fused_lut16()`
- optionally `lut_c` wrappers if we want parity with old routing

#### Preprocess mapping for `IFAIRY64`

The tricky part is activation indexing.

Weights:

- one `IFAIRY64` block covers `64` complex values
- that is `32` 2-weight groups

Activations:

- are still stored as `block_ifairy_q16`
- one `block_ifairy_q16` covers `256` complex values

So preprocess must map:

- `ifairy64 block index blk64`
- to `act_block = blk64 / 4`
- and `subblock = blk64 % 4`
- and `base_off = subblock * 64 + local_group * 2`

This mapping is the main semantic difference from old LUT preprocess.

#### qgemm for `IFAIRY64`

The qgemm shape and logic are conceptually the same:

- iterate over blocks
- iterate over groups in a block
- load per-group LUT tables
- decode packed weight pattern IDs
- accumulate into tile outputs
- apply weight scales and activation scales

But these details change:

- packed tile type is `ifairy64_lut_wtile_16`
- `groups_per_block = 32`
- block stride is based on `QK_IFAIRY64`

Implementation strategy:

- Phase 1 correctness:
  - write a scalar/shared implementation specialized for `IFAIRY64`
  - keep the code structure very close to old qgemm
- Phase 2 performance:
  - optimize ARM hot loops after correctness and integration are stable

### 6.7 `tests/test-ifairy.cpp`

Add correctness coverage for the new LUT route.

New tests:

- `IFAIRY64` 2-weight encoder test
- `IFAIRY64` transform packing test
- `IFAIRY64` LUT backend smoke
- `IFAIRY64` LUT backend `F32 vs Q16` compare
- `IFAIRY64` LUT vs non-LUT output compare on small matrices

Must also re-run:

- existing legacy LUT-only tests
- existing `IFAIRY64` vec-dot compare tests

## 7. Implementation Phases

### Phase 1: correctness-only route

Goal:

- get `GGML_TYPE_IFAIRY64` through a working LUT route
- keep legacy untouched
- prioritize readable separation over maximal code sharing

Deliverables:

- routing accepts `IFAIRY64`
- transform works
- preprocess works
- qgemm works
- end-to-end `llama-cli` runs with `GGML_IFAIRY_LUT=1`

Acceptance:

- bitwise or numerically tight agreement with current non-LUT `IFAIRY64` route on focused tests
- no legacy LUT regressions

### Phase 2: ARM performance tuning

Goal:

- optimize the new `IFAIRY64` LUT route on Apple ARM

Likely hotspots:

- preprocess mapping from `block_ifairy_q16` quarter-blocks
- inner qgemm decode loop
- repeated scale loads

Optimization ideas:

- keep 16-row tiles
- precompute pointer increments for the 4 subblocks
- reduce branchiness in `blk64 -> act_block/subblock` mapping
- unroll group loops in multiples of 2 or 4
- cache scales in registers or tighter scratch layout

Acceptance:

- `tg32` materially better than current vec-dot-only path

### Phase 3: cleanup and default policy

Goal:

- decide whether `IFAIRY64` should enter `auto` routing by default

Only do this when:

- correctness is stable
- speed improvement is clearly reproducible

## 8. Validation Matrix

Each phase should validate all of the following:

1. Build:

- `cmake --build build-rel-lut --target test-ifairy llama-cli llama-bench -j ...`

2. Legacy regression:

- `./build-rel-lut/bin/test-ifairy --ifairy-lut-only`

3. New `IFAIRY64` tests:

- new transform/preprocess/qgemm unit cases

4. End-to-end 32B:

- `GGML_IFAIRY_LUT=1 ./build-rel-lut/bin/llama-cli ...`

5. Bench:

- `GGML_IFAIRY_LUT=1 ./build-rel-lut/bin/llama-bench ...`

Bench protocol:

- do not run multiple benches in parallel
- use the same thread count and prompt/gen settings for before/after
- record both prompt and decode throughput

## 9. Commit Plan

Recommended commit split:

1. `ggml: add ifairy64 2w LUT encoding helpers`
2. `ggml: add ifairy64 LUT transform and routing scaffolding`
3. `ggml: add ifairy64 LUT preprocess and qgemm correctness path`
4. `tests: cover ifairy64 LUT transform and backend compare`
5. `ggml: optimize arm ifairy64 LUT hot loops`
6. `docs: update ifairy64 LUT status and benchmark notes`

## 10. Risks and Tradeoffs

### Risk 1: duplicated code

If we copy too much from old LUT files, maintenance burden rises.

Mitigation:

- duplicate only where geometry really differs
- share helpers where possible
- do not force early abstraction that obscures correctness

### Risk 2: quality confusion

This route still uses activation quantization, which is not training-aligned.

Mitigation:

- document clearly that this is an acceleration step, not the final semantic target
- keep the current non-LUT route available for comparison

### Risk 3: limited upside

Because `tile64_v2` weights are larger than legacy, even a perfect LUT route may not reach legacy throughput.

Mitigation:

- target `2.7 - 2.9 tok/s`, not full `3.2+ tok/s`
- make decisions based on measured gains, not on symmetry with legacy

## 11. Recommended Immediate Next Step

Start with Phase 1, but keep the implementation narrow:

- add `IFAIRY64` 2-weight encode helpers
- add `IFAIRY64` transform path
- add route gating in `ggml_ifairy_lut_can_mul_mat()`

Do not start from qgemm first.

Without the dedicated transform and block/group bookkeeping in place, qgemm changes are too easy to get subtly wrong.
