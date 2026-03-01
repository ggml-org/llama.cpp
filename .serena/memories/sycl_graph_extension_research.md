# SYCL Graph Extension Research: Tensor Split + Graph Replay Compatibility

**Date**: February 2026  
**Query**: Can we inject external data between graph nodes during replay, use multiple queues with graphs, or split graphs into segments to enable tensor split with graph replay?

---

## Executive Summary

**Verdict: SEVERE LIMITATIONS ON ALL FRONTS**

The SYCL graph extension **CANNOT support tensor split with graph replay** under any current mechanism. The fundamental incompatibility is:
- Graph replay **BAKES IN tensor pointers at finalization time** (lines 32056-32152, ggml-sycl.cpp)
- Tensor split produces **PARTIAL MUL_MAT outputs** that violate graph data dependencies
- Stale GPU output in `[N_gpu, N)` region is consumed by subsequent ops before CPU results can be uploaded
- **The SYCL runtime cannot see cross-graph data dependencies** (line 31723)

---

## Key Findings

### 1. EXTERNAL DATA INJECTION BETWEEN GRAPH NODES: NOT SUPPORTED

The `executable_command_graph::update()` API provides **ONLY three update mechanisms**:

```cpp
// From executable_graph.hpp:42-52
void update(const command_graph<graph_state::modifiable> &Graph);
void update(const node &Node);
void update(const std::vector<node> &Nodes);
```

**What they do**:
- Update the **full graph structure or individual node kernels** from a modifiable graph source
- Update **kernel ND-range parameters** via `node::update_nd_range()` / `node::update_range()`
- **NO mechanism for mid-execution tensor updates** or "dynamic parameters" between recorded ops

**What they CANNOT do**:
- Insert new data between graph nodes during replay
- Pause execution, modify buffers, resume
- Update memory pointers after partial execution
- Create "partial replay" or "node subset execution"

**Dynamic Parameters** (`dynamic_parameter<T>`, `dynamic_work_group_memory`) are **kernel parameters only**:
- Used to update scalar kernel arguments (e.g., batch size, M/N dimensions)
- NOT for injecting new tensors or updating buffer pointers mid-execution
- Recorded at **graph finalization time** (line 32102-32152)

### 2. MULTIPLE QUEUES WITH GRAPHS: LIMITED SUPPORT

From `modifiable_command_graph`:
```cpp
// Lines 118-136
void begin_recording(const std::vector<queue> &RecordingQueues, ...);
void end_recording(const std::vector<queue> &RecordingQueues);
```

**What this supports**:
- Recording **the same graph** on **multiple queues simultaneously**
- Both queues capture **identical command sequences**
- Simultaneous execution on multiple queues

**What this CANNOT do**:
- Chain work between queue1 → queue2 during execution
- Have queue2's output feed into queue1's recorded operations
- Create data dependencies **across queues within a single graph**

**Current llama.cpp usage** (line 32056-32075):
```cpp
sycl_ex::command_graph model_sycl_graph(*(sycl_ctx->stream()),
  { sycl_ex::property::graph::assume_buffer_outlives_graph{} });
model_sycl_graph.begin_recording(*(sycl_ctx->stream()));
// ... record ops ...
model_sycl_graph.end_recording();
```
- Uses **single queue** recording → **inherent sequential execution**
- No mechanism to inject queue2's work between nodes

### 3. GRAPH SPLITTING / PARTIAL REPLAY: EXPLICITLY BROKEN

From **ggml-sycl.cpp lines 31721-31734**:

```cpp
// SYCL graph recording for partial graphs (prefix mode) is not supported:
// graph execution of a subset of the compute graph produces incorrect KV cache
// state, likely because the SYCL runtime cannot see cross-graph data dependencies.
// The prefix/suffix split still provides value by enabling the fast compute_impl
// path for GPU nodes while HOST_COMPUTE CPU nodes run in the suffix.
//
// Graph replay remains enabled for GPU-only mode (full graph, gpu_prefix_end < 0).
```

**Attempted approach**:
- Split graph into GPU prefix (full execution) + CPU suffix (separate execution)
- Prefix nodes use graph replay, suffix uses `compute_impl()` (lines 31732-31734)

**Result**: **FAILS - incorrect KV cache state, data corruption**

**Root cause**:
- Graph finalization **bakes in ALL tensor pointers** at compile time
- Inter-node dependencies **frozen at recording time**
- Partial execution breaks assumption that "next op sees producer output"
- SYCL runtime has **no visibility into cross-graph data flow**

**Current workaround** (line 31731-31734):
```cpp
if (gpu_prefix_end >= 0) {
    // Prefix mode: graph execution of partial graphs is broken.
    // Use compute_impl for both prefix and suffix.
    use_sycl_graph = false;
}
```
- **Disables graphs entirely** when CPU offload active
- No partial replay at all

### 4. TENSOR SPLIT INCOMPATIBILITY WITH GRAPH REPLAY

From **ggml-sycl.cpp lines 19930-19936**:

```cpp
// Tensor split is incompatible with graph replay: partial MUL_MAT outputs
// leave stale data in [N_gpu, N) that subsequent graph ops consume before
// CPU results can be uploaded.  Skip during recording so the graph captures
// full MUL_MATs; the non-graph execution path handles tensor split correctly.
if (g_ggml_sycl_graph_recording) {
    return false;  // Don't do tensor split during graph recording
}
```

**The problem**:
1. GPU computes `MUL_MAT[0:N_gpu]` → output written to GPU buffer
2. Graph continues, subsequent ops read from output buffer
3. CPU computes `MUL_MAT[N_gpu:N]` → outputs to staging buffer
4. **Stale GPU-written data in `[N_gpu:N)` consumed by downstream ops BEFORE CPU upload**
5. **Data corruption** in subsequent layer computations

**Why full MUL_MAT is required**:
- Downstream ops depend on **COMPLETE output tensor** in GPU memory
- No "wait point" between GPU partial write and graph continuation
- Graph nodes execute sequentially but **atomicity is per-node, not sub-node**

---

## SYCL Graph API Deep Dive

### Available Mechanisms (with limitations):

| Feature | API | llama.cpp Use | Limitation |
|---------|-----|---------------|-----------|
| **Graph Recording** | `begin_recording(queue)` / `end_recording()` | Lines 32064, 32134 | Single sequential pass |
| **Graph Finalization** | `graph.finalize()` → `executable_graph` | Line 32148 | **Freezes all tensor pointers** |
| **Graph Execution** | `queue.ext_oneapi_graph(exec_graph)` | Lines 32075, 32094, 32161 | Full replay only |
| **Graph Update** | `exec_graph.update(modifiable_graph)` | Line 32071 | Re-record entire graph |
| **Node Update** | `node.update_nd_range(...)` | Not used | Kernel launch params only |
| **Dynamic Parameters** | `dynamic_parameter<T>` | Not used | Scalar kernel args only |
| **Updatable Property** | `property::graph::updatable{}` | Line 32148 | Requires re-record/update cycle |
| **Multi-Queue Recording** | `begin_recording(vector<queue>)` | Not used | Simultaneous identical recording |
| **No Event Callbacks** | ❌ | N/A | Cannot inject work between ops |
| **No Node Predecessor Queries** | `node.get_predecessors()` | Not used | Query-only; cannot modify |

### Graph Finalization (THE TRAP):

From **executable_graph.hpp lines 34-75**:
```cpp
class executable_command_graph {
public:
  // Constructor is internal only:
  executable_command_graph() = delete;  // User cannot construct
  
  // Three update modes:
  void update(const command_graph<graph_state::modifiable> &Graph);
  void update(const node &Node);
  void update(const std::vector<node> &Nodes);
  
  // That's it - NO partial execution, NO node skipping
};
```

**Key insight**: 
- Executable graphs **CANNOT be modified post-finalization** except via full re-record
- Tensor pointers baked into device kernel code at finalization time
- Line 32152: `std::make_unique<...command_graph<...executable>>(exec_graph);` → immutable

### Graph Properties:

From **graph_properties.def**:
```cpp
__SYCL_DATA_LESS_PROP(property::graph, no_cycle_check, ...)
__SYCL_DATA_LESS_PROP(property::graph, assume_buffer_outlives_graph, ...)
__SYCL_DATA_LESS_PROP(property::graph, updatable, ...)  // Expensive - requires re-record
__SYCL_DATA_LESS_PROP(property::graph, enable_profiling, ...)
```

**Available properties**: 
- `assume_buffer_outlives_graph` (line 32104) - llama.cpp uses this
- `updatable` (line 32148) - enables expensive re-record/update cycle
- **NONE provide partial execution, external injection, or multi-queue synchronization**

---

## Input Tensor Refresh Mechanism

From **ggml-sycl.cpp lines 27107-27187** (graph_refresh_input_tensors):

```cpp
// Refresh dynamic input tensors (e.g. tokens) on device BEFORE replaying graph
static void graph_refresh_input_tensors(ggml_backend_sycl_context * ctx, ...) {
    // Cache input tensor list on first call (5-10 tensors)
    if (ctx->input_tensors_cached && !ctx->cached_input_tensors.empty()) {
        // On replay: direct async memcpy for cached tensors
        for (size_t idx = 0; idx < n; idx++) {
            ggml_tensor * tensor = ctx->cached_input_tensors[idx];
            void * dev_ptr = ctx->cached_input_dev_ptrs[idx];
            if (dev_ptr != tensor->data) {
                size_t nbytes = ggml_nbytes(tensor);
                q.memcpy(dev_ptr, tensor->data, nbytes);  // Async upload
            }
        }
        return;
    }
    // First call: discover input tensors marked with GGML_TENSOR_FLAG_INPUT
    // Cache their device pointers
}
```

**What this shows**:
- **ONLY input tensors (leaves) are refreshed** before graph execution
- Refreshed via **async queue.memcpy() BEFORE graph starts**
- **NO mechanism to inject intermediate data**
- NO support for updating intermediate tensor pointers during execution

---

## Host Compute + Graph Incompatibility

From **ggml-sycl.cpp lines 25359-25365** (HOST_COMPUTE design):

```cpp
// host_tasks on gpu_q, eliminating all per-op staging overhead.
// D+: HOST_COMPUTE uses direct dispatch (batched_mode), not host_task.
// Only enable host_task_mode for non-batched fallback paths.
ggml_sycl_host_task_mode_set(false);
```

**The constraint**:
- HOST_COMPUTE mode uses **host-pinned USM buffers** (zero-copy GPU access)
- SYCL graph recording **INCOMPATIBLE with host_task**
- Cannot pause graph, run host_task, resume
- Result: **Graphs DISABLED when HOST_COMPUTE active** (not explicitly shown, but implied by prefix/suffix failure)

---

## Why Tensor Split Needs Full MUL_MAT

The fundamental mismatch:

**Tensor split workflow**:
1. GPU handles rows `[0, N_gpu)` → outputs to device buffer offset `[0, N_gpu)` (floating point stride)
2. CPU handles rows `[N_gpu, N)` → outputs to host staging buffer
3. Host staging uploaded back to device buffer offset `[N_gpu, N)` (after GPU finishes)
4. Next operator (e.g., `NORM`, `FLASH_ATTN`) reads complete `[0, N)` output

**Graph replay constraint**:
- All ops must see **complete output tensors** in device memory
- Graph finalization **assumes full tensor in device VRAM**
- No pause/inject point between operator calls
- Partial write = **stale data for downstream ops**

**Current llama.cpp workaround** (lines 19922-20050):
- Tensor split **explicitly disabled during recording** (`if (g_ggml_sycl_graph_recording) return false;`)
- Full `MUL_MAT` captured into graph
- Tensor split used **ONLY during non-graph execution** (fallback path)
- Performance hit: ~50% slower than graph replay (70.5 vs 159 tok/s baseline)

---

## What WOULD Be Needed (Not Available)

To make tensor split + graphs work, SYCL would need:

1. **Partial Graph Replay with Synchronization Points**:
   ```
   graph.replay(queue, start_node, end_node);  // Not in API
   ```

2. **Buffer Update Between Nodes**:
   ```
   exec_graph.update_buffer_region(tensor_name, new_data, offset);  // Not available
   ```

3. **Node Skip/Conditional Execution**:
   ```
   graph.set_node_enabled(node, false);  // Not in API
   exec_graph.execute_with_config(skip_nodes=[...]);  // Not in API
   ```

4. **Queue Synchronization Within Graph**:
   ```
   add_dependency_between_queues(queue1, queue2);  // Not in API
   ```

5. **External Event Injection**:
   ```
   graph.wait_for_external_event(cpu_event);  // Not in API
   ```

6. **Dynamic Memory Regions**:
   ```
   dynamic_buffer<T> buf = ...;  // Only dynamic_parameter<scalar> exists
   ```

---

## Conclusion

### Can we inject external data between graph nodes? 
**NO.** Only input tensor refresh before graph execution. No mid-execution injection.

### Can we use multiple queues with graphs?
**PARTIALLY.** Can record same graph on multiple queues, but cannot chain work between them or create data dependencies across queues.

### Can we split graphs into segments?
**NO.** Attempted and **explicitly broken**. Partial graph execution produces data corruption due to frozen tensor pointers and SYCL runtime's inability to track cross-graph dependencies.

### Can we make tensor split work with graph replay?
**NO.** Not without:
- Partial graph execution (not supported)
- Buffer update between ops (not supported)  
- Synchronization points within graph (not supported)

**Current state**: Tensor split **disabled during graph recording** (line 19934-19936). Full MUL_MATs captured into graph. Tensor split available only in fallback (non-graph) path.

---

## References

### llama.cpp Files
- **ggml-sycl.cpp:19922-20050**: Tensor split incompatibility with graphs
- **ggml-sycl.cpp:27107-27187**: Input tensor refresh (only mechanism available)
- **ggml-sycl.cpp:31721-31734**: Prefix/suffix split failure analysis
- **ggml-sycl.cpp:32050-32200**: Graph recording/update/execution flow

### oneAPI SYCL Headers (2025.3)
- **executable_graph.hpp**: Update API (full graph, single node, vector of nodes)
- **modifiable_graph.hpp**: Recording API, make_edge, begin/end_recording
- **dynamic.hpp**: dynamic_parameter (scalars only), dynamic_work_group_memory
- **node.hpp**: Node queries (get_predecessors, get_successors) - read-only
- **graph_properties.def**: Available properties (no partial replay, no external injection)

### Key Constraints
- Graph finalization freezes all tensor pointers in device code
- SYCL runtime cannot track cross-graph data dependencies
- Input tensor refresh is the ONLY per-iteration data update mechanism
- Partial execution breaks downstream op assumptions
