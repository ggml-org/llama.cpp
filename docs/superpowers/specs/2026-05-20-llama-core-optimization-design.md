---
name: llama-core-optimization-2026-05-20
description: Optimization and bug fixes for the core llama library (llama-context.cpp, llama-graph.cpp, llama-kv-cache.cpp). Approach A: 7 bug fixes + 3 high-impact performance optimizations.
metadata:
  type: project-spec
---

# Core Llama Library Optimization — Design Spec

**Date:** 2026-05-20
**Scope:** `src/llama-context.cpp`, `src/llama-graph.cpp`, `src/llama-kv-cache.cpp`, `src/llama-kv-cells.h`
**Approach:** A — All bug fixes + top 3 performance wins (10 targeted changes)

## Context

The core llama library (~47K lines) handles inference graph building, KV cache management, and token generation. A previous pass (commit `eedad9cab`) addressed some issues in `llama-context.cpp`. This pass continues with remaining high-priority items across all three hot-path files.

## Bug Fixes (7 Items)

### B1 — `type_v()` Null Dereference on MLA Models

- **File:** `src/llama-kv-cache.cpp:1117`
- **Issue:** `type_v()` unconditionally accesses `layers[0].v->type`, but MLA models set `v = nullptr` (line 210: `const bool has_v = !is_mla`). Crashes when inspecting cache type for DeepSeek and other MLA models.
- **Fix:** Add null guard: `return layers[0].v ? layers[0].v->type : GGML_TYPE_COUNT;`

### B2 — Unsigned Underflow in `can_reuse` Checks

- **File:** `src/llama-graph.cpp:620, 663, 747`
- **Issue:** `mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs` performs unsigned subtraction. When `n_seqs > n_rs`, the result wraps to a huge number, silently passing the reuse check.
- **Fix:** Guard the subtraction: `mctx->get_recr()->get_n_rs() >= params.ubatch.n_seqs && (mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs) == s_copy_extra->ne[0]`

### B3 — Null Pointer Dereference in `llm_graph_input_pos::can_reuse`

- **File:** `src/llama-graph.cpp:130`
- **Issue:** `pos->ne[0]` accessed without null check. Compare to `llm_graph_input_embd::can_reuse` (line 96-103) which guards with `(!params.ubatch.token) || (tokens && ...)`.
- **Fix:** Add null guard: `(!pos) || (pos->ne[0] == params.ubatch.n_tokens * n_pos_per_embd())`

### B4 — Missing `v_idxs` Reuse Checks

- **File:** `src/llama-graph.cpp:475, 538, 541, 613, 731, 739`
- **Issue:** Six commented-out `self_v_idxs` reuse checks allow graph reuse with wrong V cache indices, causing silent data corruption.
- **Fix:** Re-enable: `res &= (!self_v_idxs) || (self_v_idxs->ne[0] == params.ubatch.n_tokens);` — guard against null to handle models that don't use v_idxs.

### B5 — `~llama_io_read_device()` Not `noexcept`

- **File:** `src/llama-context.cpp:2589-2636`
- **Issue:** Destructor can throw during stack unwinding via `ggml_backend_tensor_copy` or `std::map::at`, calling `std::terminate`. The mirror fix for `llama_io_write_host` was done in commit `eedad9cab`, but `llama_io_read_device` was missed.
- **Fix:** Mark destructor `noexcept`. Wrap buffer copy loop in try-catch, log error on failure.

### B6 — `const_cast` on `const void*` in `copy_tensor_async_rows`

- **File:** `src/llama-context.cpp:1548`
- **Issue:** `const void * dst_data` parameter is cast away via `const_cast<uint8_t *>`. The function writes to this buffer, so `const` is semantically wrong — undefined behavior if caller passes truly const memory.
- **Fix:** Change parameter to `void * dst_data`. Update all callers (they pass non-const pointers). Remove `const_cast`.

### B7 — `llama_kv_cells::cp()` Leaves `used` Set Empty

- **File:** `src/llama-kv-cells.h:141-157`
- **Issue:** `cp()` copies `pos`, `ext`, `seq` arrays but not the `used` set. Currently works because rollback path checks `pos[j] != -1` directly. Any future call to `get_used()`, `used_min()`, or `used_max_p1()` on a copied object returns wrong results.
- **Fix:** After copying arrays, rebuild `used` set: `used.insert(j)` for each `j` where `pos[j] != -1`. Apply to both `cp` overloads.

## Performance Optimizations (3 Items)

### P1 — KV Cache Mask Generation: Eliminate Per-Batch Allocations

- **File:** `src/llama-kv-cache.cpp:1447-1537`
- **Issue:** `set_input_kq_mask_impl` allocates fresh `std::unordered_map<llama_seq_id, uint32_t>` and `std::unordered_map<llama_seq_id, std::vector<uint32_t>>` on every call. For 8192+ cells × multiple streams, this generates millions of hash lookups and heap allocations per batch.
- **Fix:** Replace `std::unordered_map<llama_seq_id, ...>` with `std::array<T, LLAMA_MAX_SEQ>` indexed directly by seq_id. Seq_ids are dense (0-255), so a 256-entry array is L1-cache-sized and eliminates all hashing. `array::reset()` is O(256) vs O(n_cells) for map clearing.
- **Impact:** Multi-stream batches (8-16 sequences, 2048+ context) should see reduced per-token latency.

### P3 — `output_reorder()`: Block Swap Instead of Element-by-Element

- **File:** `src/llama-context.cpp:2056-2086`
- **Issue:** `std::swap_ranges` does 3 reads + 3 writes per element. For 128K vocab × 256 seqs = ~128MB per swap.
- **Fix:** Pre-allocate one temp row in `llama_sampling`. Use 3 `memmove` calls: A→tmp, B→A, tmp→B. Reduces 6N ops to 4N with better cache line utilization. Apply to both float rows and count rows.
- **Impact:** Recurrent models and any workload with unsorted outputs.

### P4 — Training Loop: Batch GPU Label Writes

- **File:** `src/llama-context.cpp:3169-3172`
- **Issue:** `opt_epoch_iter()` calls `ggml_backend_tensor_set` once per label (up to 512×), each writing 4 bytes. Per-call GPU driver overhead: ~5-10 μs.
- **Fix:** Build a `std::vector<float>` filled with 0.0f, set sparse positions to 1.0f, issue one `ggml_backend_tensor_set` for the full row.
- **Impact:** Training with sparse labels reduced from 512 GPU writes to 1.

## Error Handling

- Bug fixes (B1-B4) add defensive guards that prevent crashes and silent corruption.
- Bug fix (B5) prevents `std::terminate` during exception propagation.
- Performance changes maintain existing behavior — no new error paths.

## Testing

1. `cmake -B build -DLLAMA_BUILD_TESTS=ON && cmake --build build -j`
2. `cd build && ctest -C Release --output-on-failure`
3. Targeted tests:
   - B1: Load any DeepSeek/MLA model, verify no crash on type inspection
   - B2-B4: Run multi-sequence batch with varying token counts
   - P1: Multi-stream server benchmark (8-16 seqs, 2K+ context)
   - P3: Recurrent model generation with large vocab
   - P4: Training epoch benchmark with sparse labels

## Out of Scope

- P2: `state_write_data` strided I/O (Approach B)
- P5: `seq_get()` bit-scan optimization (Approach B)
- P6: MoE warmup expert mismatch (Approach B)
- Medium/low priority items from analysis (duplication, code quality)
- Changes to `llama-model.cpp`, `llama-sampler.cpp`, `llama-vocab.cpp` (not analyzed for this pass)
