Project 03C — KV‑Cache Invariance First (Reprioritized)
=======================================================

Intent
------
Prioritize KV‑cache invariance before porting deterministic attention to other backends. Under `GGML_DETERMINISTIC=1`, logits for a token at absolute position P must be bitwise identical whether computed via:
- Single‑shot prefill to length P, or
- Incremental decode appending tokens up to P (including chunked prefill/streaming).

Why This First
--------------
- Batch invariance in the attention kernel is necessary but not sufficient; layout/scheduling of the KV cache often changes the reduction order pre‑kernel.
- Aligning KV layout and mask semantics eliminates a class of nondeterminism regardless of backend.

Deterministic Policy (KV‑centric)
---------------------------------
- KV length presented to the attention op is always a multiple of `FATTN_KQ_STRIDE` (currently 256).
- Mask shape is `[KV, PAD(N, GGML_KQ_MASK_PAD), 1, 1]` with `GGML_KQ_MASK_PAD = 64` and is at least N.
- Avoid separate reductions for “cached KV” vs “current tokens” in the graph build: K/V passed to attention must represent one consistent contiguous view in a fixed order regardless of where tokens came from.
- Per‑row reductions remain within a single block/workgroup (handled by 03A/03B dispatcher); this document focuses on producing the same input views to attention across flows.

Acceptance Criteria
-------------------
1) Single‑shot vs incremental decode produce bitwise‑identical logits for the last token at position P, across:
   - D ∈ {64, 128, 256}
   - KV ∈ {256, 1024} (multiple of `FATTN_KQ_STRIDE`)
   - Batch sizes B ∈ {1, 8}
   - GQA ∈ {1, 2}
   - Features: mask on/off, ALiBi on/off; sinks off (plus a smoke with sinks on).
2) Batch invariance preserved under chunked prefill (optional gate): chunk sizes {8, 33} via `RUN_KVCACHE_CHUNK_TESTS=1`.
3) Clear aborts with guidance if invariance preconditions are violated (e.g., KV not multiple of 256, mask not padded).

Specific Files to Target
------------------------
- KV cache construction and views
  - `src/llama-kv-cache.cpp`
    - Ensure `kv_size` and per‑stream K/V 2D views satisfy `FATTN_KQ_STRIDE` alignment when determinism is ON.
    - Audit `get_k(...)`/`get_v(...)` view creation paths for contiguous layout and fixed strides across flows.
    - Confirm `v_trans` handling doesn’t change layout ordering between prefill and decode.

- Graph build for attention (where K/V/mask/sinks are wired)
  - `src/llama-graph.cpp` — `build_attn_mha(...)`
    - Enforce/validate: `n_kv % 256 == 0` when `ggml_is_deterministic()`; otherwise pad/abort with guidance.
    - Mask creation: guarantee `[KV, PAD(N, 64), 1, 1]` regardless of flow; unify any divergent paths.
    - Ensure K/V presented to `ggml_flash_attn_ext` are the same “flattened view” across single‑shot and incremental; avoid any special‑casing that would split reductions.

- Kernel launch invariants (already handled but add sanity)
  - `ggml/src/ggml-cuda/fattn-common.cuh` — asserts already check `K->ne[1] % FATTN_KQ_STRIDE == 0` and mask padding; we will align host‑side graph and KV views to satisfy this across flows.

Plan of Record (KV‑first)
-------------------------
M1 — Define invariance contract + instrumentation (0.5d)
- Add a one‑time INFO log when deterministic mode detects non‑compliant KV or mask and pads/aborts (for triage only; default silent success).
- Document the precise invariance contract in `docs/DETERMINISM.md` (KV/mask layout; no split reductions across cache/current tokens).

M2 — Normalize KV layout/views (0.5–1.0d)
- `llama-kv-cache.cpp`:
  - Verify `kv_size` is a multiple of 256 when determinism is ON; otherwise round‑up allocation and mask unused tail (zero‑fill, masked out).
  - Ensure `k_stream`/`v_stream` `ggml_view_2d(...)` maintain identical stride/offset semantics between single‑shot and update flows.
- `llama-graph.cpp` (`build_attn_mha`):
  - Use consistent permutations and type casts to F16 for K/V before attention, identical across flows.
  - Guarantee `n_kv` reported to attention includes freshly written tokens in the same memory region and order as single‑shot.

M3 — Mask and ALiBi semantics unification (0.5d)
- Centralize mask construction ensuring `PAD(N, 64)` and length ≥ N.
- Confirm ALiBi slope application matches the CUDA path semantics (slope times mask indices); guard against out‑of‑range indices in incremental chunks.

M4 — Test: KV‑cache invariance (1.0d)
- New test: `tests/test-kvcache-invariance.cpp` (backend‑agnostic, uses ggml backends enumeration):
  - Build graphs for single‑shot vs incremental; compare logits of final token bitwise.
  - Grid per Acceptance Criteria; gate chunked tests via `RUN_KVCACHE_CHUNK_TESTS=1`.
  - Skip cleanly if a backend is not present.

M5 — Docs + Runbook (0.5d)
- Add `projects/03-deterministic-attention/runbook-03C.md` with exact build/run commands per backend for the new test.
- Update `docs/DETERMINISM.md` KV section.

Out‑of‑Scope (deferred to post‑KV milestones)
---------------------------------------------
- Metal/Vulkan/OpenCL deterministic attention kernel scheduling (tracked in 03C‑detailed plan; will come after M1–M5 land and soak).
- Quantized K/V expansion on non‑CUDA backends.

Validation Matrix
-----------------
- Backends: CUDA (primary), CPU fallback (softmax), plus availability checks for Metal/Vulkan/OpenCL/HIP without hard requirements for this phase.
- Models: small shapes via synthetic tensors (no model load needed) for unit tests.

Risks & Mitigations
-------------------
- KV round‑up increases memory footprint: document and gate only under `GGML_DETERMINISTIC=1`.
- Legacy flows may depend on non‑padded KV sizes: provide clear error with remediation (enable deterministic mode padding or adjust context size).
- Throughput impact is minimal in this phase; we are not changing kernel selection.

Deliverables
------------
- Code: host‑side KV/view normalization; mask unification; guarded logs; no kernel changes.
- Tests: `test-kvcache-invariance.cpp`.
- Docs: determinism KV section + runbook entries.

