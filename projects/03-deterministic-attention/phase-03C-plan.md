Project 03C — KV-Cache Invariance + Other Backends
==================================================

Scope
-----

- Ensure incremental decode produces bitwise-identical results to an equivalent single-shot evaluation for the same positions (KV-cache invariance) under `GGML_DETERMINISTIC=1`.
- Port deterministic attention policy to Metal, Vulkan, HIP, and OpenCL backends with single-column kernels/workgroups and no multi-block combines.

Acceptance
----------

- Incremental vs single-shot equivalence:
  - Build two graphs over the same sequence window (positions [P0, P1))
    1) Single-shot: run attention once with KV length = P1 - P0.
    2) Incremental: seed KV=0, then append tokens one by one (or in fixed chunks) until reaching P1 - P0.
  - Assert bitwise equality of logits for the last token across runs on supported backends.
  - Shapes: D∈{64,128,256}, KV∈{256,1024}, B∈{1,8}, GQA∈{1,2}; masks/ALiBi included.

- Backend policy parity:
  - Metal / Vulkan / HIP / OpenCL: deterministic attention paths enforce one-column work per workgroup/block and avoid cross-workgroup reductions (no split-K, no planner); batch invariance verified by tests.

Design Notes
------------

- KV stride and views:
  - Normalize KV cache views to `FATTN_KQ_STRIDE` boundaries; prefer contiguous slices for incremental updates.
  - Avoid separate cache vs current-token reductions; reductions must traverse the same (k) order regardless of how many tokens are cached.

- Dispatcher:
  - Mirror CUDA deterministic dispatcher: prefer vector single-column where available; otherwise single-column tile; quantized K/V must use vector-only deterministic instances.

Tasks
-----

- [ ] Add integration test `tests/test-kvcache-invariance.cpp` (CUDA first; backend-agnostic API calls) that compares incremental vs single-shot.
- [ ] Metal: add deterministic single-column path and `launch_fattn` equivalent constraints.
- [ ] Vulkan: same as Metal; disable multi-block combine in det mode.
- [ ] HIP: mirror CUDA path and env toggles; confirm ROCm kernels respect single-workgroup accumulation.
- [ ] OpenCL / SYCL: add deterministic softmax fallback when FlashAttention is absent.
- [ ] Documentation: update `docs/DETERMINISM.md` with backend notes and KV-cache invariance policy.

Debug & Controls
----------------

- `GGML_DETERMINISTIC=1` enables deterministic policy.
- Reuse CUDA-style toggles where relevant; add backend-scoped disables if needed (e.g., `GGML_DET_ATTENTION_DISABLE_TILE_*`).

Risks
-----

- Backend feature gaps may require interim softmax fallback; keep performance expectations clear.
- Kernel shape coverage differs per backend; keep error messages prescriptive.

