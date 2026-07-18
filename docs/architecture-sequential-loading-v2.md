# Sequential/Layer-by-Layer Model Loading - V2 Architecture

## 1. Problem Summary

After two rounds of debugging and one failed implementation attempt (`fix/zero-drama`), we know:

**What works:**
- Model loads successfully with mmap — weight tensors point to disk offsets, no RAM copies
- `force_weight_offload` correctly assigns GPU backend to weight ops (fix #1)
- Buffer compatibility checks correctly detect CPU→GPU tensor copy needs (fix #2)
- Non-sequential mode is unbroken (baseline test passes)

**What fails (all 3, unchanged by fix/zero-drama):**
1. ACCESS_VIOLATION at warmup (Qwen 4B + n-gpu-layers 15)
2. GPU OOM: 67GB allocation on 24GB card (Qwen 9B, auto layers)
3. NaN logits → sampler assert (Llama 70B, auto layers)

**Root cause:**
`ggml_gallocr_alloc_graph` allocates ALL split input copies upfront as permanent leafs. The allocator reserves VRAM for every weight copy across every split simultaneously. Per-split streaming is impossible with this monolithic allocation model.

## 2. Design Decision: Fix at GGML Scheduler Level

After evaluating four options (see previous architect task), we choose to fix the scheduler's allocation/compute loop rather than implement per-layer management at the llama.cpp level. Rationale:

- The GGML scheduler already performs the hard work of splitting the graph by backend and weight budget boundaries
- The splits are already correct — each split is an independently executable subgraph with its own input copies
- The only bug is in the ALLOCATION: all splits are allocated at once instead of one at a time
- Fixing this at the scheduler level is ~100 lines of change vs ~1000+ for llama.cpp-level rework
- It keeps all graph-building code in llama-graph.cpp untouched

## 3. Core Insight: `vbuffer_reset` Enables Per-Split Allocation

`ggml_gallocr_alloc_graph` calls `vbuffer_reset` on each backend's GPU buffer at the start of allocation. This frees all previously allocated tensors from that buffer. If we call `ggml_gallocr_alloc_graph` for each split independently (each with its own subgraph), the previous split's memory is freed before the next split's is allocated.

The key constraint: each `ggml_gallocr_alloc_graph` call needs a `ggml_cgraph` that accurately represents the dependency structure so the allocator can track `n_children` properly.

## 4. Implementation Plan

### 4.1 File Changes

Only `ggml/src/ggml-backend.cpp` is modified. No changes to `ggml-alloc.c`, `llama-context.cpp`, `llama-graph.cpp`, or any other file.

### 4.2 New Static Function: `build_split_subgraph()`

```cpp
// Build a ggml_cgraph for a single split that ggml_gallocr can process independently.
// The subgraph contains:
//   - Original weight leafs (mmap'd, data != NULL, skipped by gallocr)
//   - This split's input_dep + input_cpy nodes (GPU weight copies)
//   - This split's compute nodes (whose src[] pointers reference input_cpy)
// On return, subgraph->n_nodes and subgraph->n_leafs are set.
// Caller must provide backend_id arrays sized for subgraph capacity.
static void build_split_subgraph(
    struct ggml_backend_sched * sched,
    struct ggml_cgraph * graph,
    int split_id,
    struct ggml_cgraph * subgraph,
    ggml_backend_buffer_type_t * subgraph_node_backend_ids,
    ggml_backend_buffer_type_t * subgraph_leaf_backend_ids)
{
    struct ggml_backend_sched_split * split = &sched->splits[split_id];
    int backend_id = split->backend_id;

    subgraph->n_nodes = 0;
    subgraph->n_leafs = 0;

    // (1) Add original weight leafs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        // weight tensors have data != NULL (mma'd); gallocr skips allocation for these
        subgraph->leafs[subgraph->n_leafs] = leaf;
        size_t id = hash_id(leaf);
        subgraph_leaf_backend_ids[subgraph->n_leafs] = sched->hv_tensor_backend_ids[id];
        subgraph->n_leafs++;
    }

    // (2) Add input_dep + input_cpy for this split only
    for (int j = 0; j < split->n_inputs; j++) {
        struct ggml_tensor * input = split->inputs[j];
        size_t id = hash_id(input);
        struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, sched->cur_copy);

        // Dependency node: ensures the source tensor stays alive
        struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
        ggml_set_src(input_dep, 0, input);
        subgraph_node_backend_ids[subgraph->n_nodes] = sched->hv_tensor_backend_ids[id];
        subgraph->nodes[subgraph->n_nodes++] = input_dep;

        // The GPU copy (src[0] was rewritten by split_graph to point to this copy)
        subgraph_node_backend_ids[subgraph->n_nodes] = backend_id;
        subgraph->nodes[subgraph->n_nodes++] = input_cpy;
    }

    // (3) Add this split's compute nodes from the original graph
    for (int j = split->i_start; j < split->i_end; j++) {
        subgraph_node_backend_ids[subgraph->n_nodes] = sched->node_backend_ids[j];
        subgraph->nodes[subgraph->n_nodes++] = graph->nodes[j];
    }
}
```

**Critical detail**: The compute nodes at step (3) reference the SAME `input_cpy` objects created at step (2), because `split_graph` (line 1528 in the original code) already rewrote `node->src[j]` to point to these copies. The gallocr sees the full dependency chain: `weight_leaf → input_dep → input_cpy → compute_node.src[j]`.

### 4.3 Modified: `ggml_backend_sched_alloc_splits()`

When `force_weight_offload` is true: pre-reserve GPU buffers to the maximum size needed by any single split, then return (actual allocation happens per-split in `compute_splits`).

```cpp
static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    // --- sequential mode: per-split allocation (new path) ---
    if (sched->force_weight_offload) {
        struct ggml_cgraph * graph = &sched->graph;

        // Pre-allocate subgraph arrays (reused across splits)
        // Worst case: all original nodes + 2*max_inputs per split (dep+cpy) + all leafs
        int max_subgraph_nodes = graph->n_nodes + GGML_SCHED_MAX_SPLIT_INPUTS * 2;
        int max_subgraph_leafs = graph->n_leafs + GGML_SCHED_MAX_SPLIT_INPUTS * sched->n_copies;

        std::vector<ggml_tensor *> subgraph_nodes(max_subgraph_nodes);
        std::vector<ggml_tensor *> subgraph_leafs(max_subgraph_leafs);
        std::vector<ggml_backend_buffer_type_t> subgraph_node_backend_ids(max_subgraph_nodes);
        std::vector<ggml_backend_buffer_type_t> subgraph_leaf_backend_ids(max_subgraph_leafs);

        struct ggml_cgraph subgraph;
        subgraph.nodes = subgraph_nodes.data();
        subgraph.leafs = subgraph_leafs.data();
        subgraph.size  = max_subgraph_nodes;

        // Reserve for each split to size GPU buffers for the worst-case split
        for (int i = 0; i < sched->n_splits; i++) {
            build_split_subgraph(sched, graph, i, &subgraph,
                subgraph_node_backend_ids.data(), subgraph_leaf_backend_ids.data());

            if (!ggml_gallocr_reserve_n(sched->galloc, &subgraph,
                    subgraph_node_backend_ids.data(), subgraph_leaf_backend_ids.data())) {
                return false;
            }
        }

        sched->is_alloc = true;
        return true;
    }

    // --- existing monolithic allocation path (unchanged) ---
    // ... original code ...
}
```

### 4.4 Modified: `ggml_backend_sched_compute_splits()`

In the split iteration loop, when `force_weight_offload` is true, call `ggml_gallocr_alloc_graph` before each split's copy+compute. The gallocr's internal `vbuffer_reset` frees the previous split's GPU memory.

**Current code** (lines 1742-1928) — the split loop:

```cpp
for (int cur_split = 0; cur_split < sched->n_splits; cur_split++) {
    // ... copy inputs for this split ...
    // ... compute this split ...
    // ... prefetch next split ...
}
```

**New code** — add per-split allocation before copy+compute:

```cpp
// --- sequential mode: per-split subgraph (allocate once, reuse across iterations) ---
struct ggml_cgraph subgraph = {};
std::vector<ggml_tensor *> subgraph_nodes;
std::vector<ggml_tensor *> subgraph_leafs;
std::vector<ggml_backend_buffer_type_t> subgraph_node_backend_ids;
std::vector<ggml_backend_buffer_type_t> subgraph_leaf_backend_ids;

if (sched->force_weight_offload) {
    int max_subgraph_nodes = graph->n_nodes + GGML_SCHED_MAX_SPLIT_INPUTS * 2;
    int max_subgraph_leafs = graph->n_leafs + GGML_SCHED_MAX_SPLIT_INPUTS * sched->n_copies;
    subgraph_nodes.resize(max_subgraph_nodes);
    subgraph_leafs.resize(max_subgraph_leafs);
    subgraph_node_backend_ids.resize(max_subgraph_nodes);
    subgraph_leaf_backend_ids.resize(max_subgraph_leafs);
    subgraph.nodes = subgraph_nodes.data();
    subgraph.leafs = subgraph_leafs.data();
    subgraph.size  = max_subgraph_nodes;
}

for (int cur_split = 0; cur_split < sched->n_splits; cur_split++) {
    // ... existing declarations ...

    // --- PER-SPLIT ALLOCATION (new, sequential mode only) ---
    if (sched->force_weight_offload) {
        build_split_subgraph(sched, graph, cur_split, &subgraph,
            subgraph_node_backend_ids.data(), subgraph_leaf_backend_ids.data());

        // ggml_gallocr_alloc_graph calls vbuffer_reset internally,
        // which frees the previous split's GPU allocations
        if (!ggml_gallocr_alloc_graph(sched->galloc, &subgraph)) {
            GGML_LOG_ERROR("%s: failed to allocate split %d\n", __func__, cur_split);
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    // --- COPY INPUTS (existing logic) ---
    // The input_cpy->data pointers were just set by gallocr_alloc_graph above.
    // The copy logic uses ggml_backend_tensor_set_async or tensor_copy
    // to populate them from the mmap'd weight tensors.
    // ... existing copy loop (lines 1748-1876) ...

    // --- COMPUTE (existing logic) ---
    // ... existing compute + event + prefetch (lines 1878-1928) ...

    // For sequential mode Phase 1: disable async prefetch
    if (sched->force_weight_offload) {
        // Ensure compute is done before next split's alloc frees the buffers
        ggml_backend_synchronize(split_backend);
    }
}
```

### 4.5 Disable Async Prefetch in Sequential Mode (Phase 1)

In `ggml_backend_sched_prefetch_split_inputs`, add an early return:

```cpp
static void ggml_backend_sched_prefetch_split_inputs(
    ggml_backend_sched_t sched, int split_id) {
    if (sched->force_weight_offload) {
        return; // Phase 1: synchronous mode for correctness
    }
    // ... existing prefetch logic ...
}
```

This is a performance sacrifice for correctness. Once the per-split allocation works reliably, Phase 2 can re-enable prefetch with proper double-buffering using separate staging buffers.

### 4.6 Modify `ggml_backend_sched_graph_compute_async` Signature

The function needs access to the original graph (not `sched->graph`, which is the modified copy). Change at line ~1955:

```cpp
enum ggml_status ggml_backend_sched_graph_compute_async(
    ggml_backend_sched_t sched, struct ggml_cgraph * graph) {

    // ... existing code ...

    // Pass 'graph' (original) instead of '&sched->graph' (modified copy)
    return ggml_backend_sched_compute_splits(sched, graph);
}
```

The `sched->graph` may be the `graph_copy` with modified leafs. The original `graph` parameter has the original weight leafs needed by `build_split_subgraph`.

Wait — actually, after `ggml_backend_sched_split_graph` runs, `sched->graph` IS modified (node src[] pointers rewritten, input copies created). But `build_split_subgraph` wants the ORIGINAL leafs. The issue is whether `graph` (the parameter) still has the original leafs.

Looking at the existing code at line ~1945:
```cpp
ggml_backend_sched_split_graph(sched, graph);
```
This modifies `graph` in place (rewrites src[] pointers). After this call, `graph` is the modified version. The leafs are unchanged though — `split_graph` adds input copies as leafs but doesn't remove the original weight leafs.

Actually wait — looking more carefully, `split_graph` at lines 1647-1660 adds split input copies as ADDITIONAL leafs to `graph_copy`, but the parameter `graph` that was passed in still has its original leafs. The question is whether `sched->graph` (which `compute_splits` currently uses) has the original leafs or the modified copy.

Let me re-examine. In `ggml_backend_sched_alloc_graph` (line ~1945):
```cpp
ggml_backend_sched_split_graph(sched, graph);
```
Then:
```cpp
// build graph copy with leafs
graph_copy = ggml_new_graph_custom(sched->ctx, ...);
// ... add all leafs (including split input copies) to graph_copy ...
sched->graph = *graph_copy; // This copies the graph_copy INTO sched->graph
```

So `sched->graph` is the modified copy. The original `graph` parameter is the unmodified version with original leafs. That's what we want.

**Fix**: Pass `graph` (the original parameter) to `compute_splits`:

```cpp
static enum ggml_status ggml_backend_sched_compute_splits(
    ggml_backend_sched_t sched,
    struct ggml_cgraph * graph)  // NEW second parameter
{
    // Use 'graph' for weight leafs (original, unmodified)
    // Use 'sched->graph' for compute nodes (modified, src[] rewritten)
```

Wait, but `split_graph` modifies the parameter `graph` directly — it rewrites `node->src[]` on the original graph's nodes. So after `split_graph`, `graph->nodes[j]->src[k]` points to input copies. And `sched->graph.nodes[j]` is the same pointer (it was copied from `graph`).

This means: `graph->nodes[j] == sched->graph.nodes[j]` (same tensor objects). So using either array is fine for compute nodes. For leafs, `graph->n_leafs` is the ORIGINAL count (only weight leafs), while `sched->graph.n_leafs` includes all the split input copies added by split_graph.

**Conclusion**: In `build_split_subgraph`, use `graph->leafs` and `graph->n_leafs` for the original weight leafs, and `sched->graph.nodes` (or equivalently `graph->nodes`) for the compute nodes. Actually, since `graph->nodes == sched->graph.nodes` (same array, just different logical length), either works.

Simpler: Just pass `graph` (the original parameter) to `compute_splits`, and use `graph->leafs` for weight leafs and `graph->nodes` for compute nodes in `build_split_subgraph`.

## 5. Execution Flow (Post-Fix)

```
Model Load (mmap):
  Weight tensors → data = mmap'd addresses (disk offsets, no RAM copies)
  All tensors assigned to CPU backend buffers

Context Init (llama-context.cpp:528-535):
  sched = ggml_backend_sched_new(...)
  sched->force_weight_offload = true
  sched->async_weight_prefetch = true   (disabled in Phase 1)
  sched->max_weight_bytes_per_split = sequential_weight_budget

Batch Processing (llama-context.cpp:1417-1444):
  gf = model.build_graph(gparams)         // build full compute graph
  ggml_backend_sched_alloc_graph(sched, gf)  // calls split_graph + alloc_splits
  
  Inside alloc_graph:
    split_graph(gf):
      - Assigns GPU backend to weight ops (force_weight_offload)
      - Creates N splits at max_weight_bytes_per_split boundaries
      - Creates input_cpy tensors for cross-backend inputs
      - Rewrites node->src[] to point to input_cpy
    alloc_splits() [force_weight_offload=true]:
      - For each split i: build_split_subgraph() + gallocr_reserve_n()
      - GPU buffers sized for worst-case split
      - Returns (actual alloc happens in compute_splits)

  set_inputs(ubatch)                      // fill input tensors
  graph_compute(gf) → ggml_backend_sched_graph_compute_async(sched, gf)
  
  Inside compute_splits (per-split loop):
    Split 0:
      build_split_subgraph(split_0)
      gallocr_alloc_graph(subgraph_0)     // allocates split 0's GPU buffers
      Copy weights: mmap→GPU for split 0  // ggml_backend_tensor_set_async
      Compute split 0                      // GPU runs split 0
      synchronize(GPU)                     // Phase 1: ensure complete
    
    Split 1:
      build_split_subgraph(split_1)
      gallocr_alloc_graph(subgraph_1)     // vbuffer_reset frees split 0's GPU memory
                                           // Allocates split 1's GPU buffers
      Copy weights: mmap→GPU for split 1
      Compute split 1
      synchronize(GPU)
    
    ... continue for all N splits ...

Peak GPU VRAM = max(split_0_weights, split_1_weights, ..., split_N-1_weights)
              + KV cache + activations
              ≈ max_weight_bytes_per_split + KV + activations
```

## 6. VRAM Accounting

For a 70B model in Q4 (~40GB weights) with `max_weight_bytes_per_split = 8GB`:
- Splits: ~5-6 GPU splits
- Peak GPU VRAM: ~8GB (weights) + KV cache + activations
- Fits comfortably in 24GB GPU

For Qwen 4B in Q4 (~3GB weights):
- Splits: 1 or 2 GPU splits (entire model fits in one split budget)
- If 1 split: ~3GB weights + KV cache → no OOM

The 67GB OOM from Test 2 should no longer occur because:
- The old code added ALL splits' input copies as leafs → 8 splits × ~8GB = 64GB reserved
- The new code allocates one split at a time → max ~8GB reserved

## 7. Edge Cases and Error Handling

### 7.1 Single split (small model)
If the model fits in one split (all weights < max_weight_bytes_per_split), `n_splits == 1`. The per-split loop runs once. Behavior is identical to monolithic allocation but with one fewer leaf — no functional difference.

### 7.2 Multi-backend (CPU + GPU)
When some layers are on CPU and some on GPU, splits will span both backends. The per-split allocation only affects GPU backends (CPU tensors have `data != NULL` and are skipped by gallocr). CPU-side tensors remain allocated once and reused.

### 7.3 KV cache spanning splits
The KV cache tensors are allocated separately (in `llama_context` init), not through the scheduler's graph allocator. They persist across all splits. No changes needed.

### 7.4 Embedding layer and lm_head
These are typically large tensors that may be in their own splits. The per-split allocation handles them correctly — each split's subgraph includes only that split's weight copies.

### 7.5 gallocr_alloc_graph failure
If a split fails to allocate, return `GGML_STATUS_ALLOC_FAILED`. The caller (llama_context) already handles this: logs error, returns nullptr, stops inference.

### 7.6 Synchronization before vbuffer_reset
In Phase 1 (no async prefetch), we call `ggml_backend_synchronize` after each split's compute. This ensures the GPU is done with split N's buffers before vbuffer_reset (called by the next `gallocr_alloc_graph`) frees them.

In Phase 2 (async prefetch re-enabled), we need event-based synchronization: wait on the compute event before vbuffer_reset instead of a full device sync.

## 8. Testing Strategy

### Phase 1: Synchronous (no prefetch)
1. Build and test Qwen 4B + `--sequential-load --n-gpu-layers 15` — expect: server starts, warmup succeeds
2. Build and test Qwen 9B + `--sequential-load` — expect: server starts, no OOM
3. Build and test Llama 70B + `--sequential-load` — expect: server starts, valid inference (no NaN)
4. Inference quality check: compare logits vs non-sequential baseline for a small model

### Phase 2: Async prefetch (performance)
1. Re-enable async prefetch
2. Verify no data races or NaN logits
3. Measure tokens/sec — expect: within 30% of non-sequential for models that fit entirely in VRAM

### Regression: Non-sequential mode
- Qwen 4B without `--sequential-load` — must produce identical results to before
- All existing tests must pass

## 9. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| `vbuffer_reset` in gallocr_alloc_graph doesn't actually free GPU memory | Low | High | Add debug logging to verify VRAM usage per split. If gallocr doesn't free, implement explicit `ggml_backend_buffer_free` for input copies between splits. |
| Subgraph node/leaf arrays overflow | Low | High | Pre-allocate arrays sized for max_subgraph_nodes/leafs with generous safety margin. Assert if exceeded. |
| gallocr reserve_n across multiple splits causes buffer fragmentation | Low | Medium | Since reserve_n runs for all splits upfront with the same gallocr instance, tallocs stabilize. Buffer sizes grow monotonically to worst-case split. |
| compute_splits currently uses `sched->graph` not the original `graph` | Low | Medium | Verify that `sched->graph.nodes[j] == graph->nodes[j]` for compute nodes. They are the same tensor objects (copied during split_graph). |

## 10. Implementation Sequence

1. Add `build_split_subgraph()` static function — ~50 lines
2. Modify `ggml_backend_sched_alloc_splits()` — add force_weight_offload branch, ~30 lines
3. Modify `ggml_backend_sched_compute_splits()` — add per-split alloc + sync, ~30 lines
4. Modify `ggml_backend_sched_graph_compute_async()` — pass original graph, ~2 lines
5. Disable async prefetch when force_weight_offload — 1 line in prefetch function
6. Build, test with Qwen 4B, Qwen 9B, Llama 70B
7. If successful: re-enable and test async prefetch in Phase 2
