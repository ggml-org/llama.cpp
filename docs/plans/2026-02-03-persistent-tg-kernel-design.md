# Persistent TG Kernel Design

> **For Claude:** Use superpowers:executing-plans to implement this plan task-by-task with TDD approach.

**Goal:** Improve Token Generation throughput from ~16 t/s to ~40 t/s on Intel Arc B580

**Primary Optimization:** Full Persistent Kernel with hierarchical work distribution, integrated into the UnifiedKernel class.

**Key Constraints:**
- Must use split barriers (group barriers don't work on Intel Arc hardware)
- Integrate into existing unified-kernel.cpp/hpp, not a separate kernel
- UnifiedKernel class is a black box - callers describe WHAT, class handles HOW
- No stubs - every operation must be fully implemented and tested
- TDD approach - tests written before implementation

---

## Architecture Overview

```
ggml_backend_sycl_graph_compute()
    ├── is_persistent_compatible(cgraph)
    │   └── Check: decode phase, supported quant, XMX hardware
    ├── extract_persistent_plan(cgraph)
    │   └── Walk graph, collect weight pointers, build op table
    └── ctx.unified_kernel->execute_persistent()
        └── Single kernel, all layers, split-barrier sync
```

**Design Decisions:**
- **Hierarchical execution** - All work-groups cooperate on one operation at a time, work-steal tiles within each operation, split-barrier sync between operations
- **Register tiling with cooperative stores** - Keep partial results in registers, use SLM for cross-work-group communication
- **Unified cache async prefetch** - Extend cache with prefetch queue for layer-by-layer streaming (supports models larger than VRAM)
- **Graph-level detection** - Intercept at graph_compute, extract operation metadata, dispatch to persistent kernel

---

## UnifiedKernel Class Design

```cpp
namespace ggml_sycl {

class UnifiedKernel {
public:
    // Construction & Configuration
    explicit UnifiedKernel(sycl::queue& queue, UnifiedCache& cache);
    ~UnifiedKernel();
    void configure(const XMXConfig& xmx_config);

    // Single Operation API (existing functionality)
    void matmul(const MatmulDescriptor& desc);
    void rms_norm(const RmsNormDescriptor& desc);
    void rope(const RopeDescriptor& desc);
    void silu(const SiluDescriptor& desc);
    void softmax(const SoftmaxDescriptor& desc);

    // Persistent Execution API (new functionality)
    void begin_persistent(int n_layers, int batch_size);
    void add_rms_norm(int layer, const void* weights, const void* input, void* output);
    void add_matmul(int layer, const void* weights, const void* input, void* output, MatmulType type);
    void add_attention(int layer, AttentionDescriptor& desc);
    void add_silu_mul(int layer, const void* gate, const void* up, void* output);
    void execute_persistent();

    // State & Diagnostics
    bool supports_persistent() const;
    PersistentStats get_last_stats() const;

private:
    sycl::queue& queue_;
    UnifiedCache& cache_;
    XMXConfig xmx_config_;

    struct PersistentPlan;
    std::unique_ptr<PersistentPlan> current_plan_;

    void* intermediate_buffers_[4];
    int* tile_counter_;
    int* op_barrier_counter_;

    void allocate_persistent_buffers(int hidden_dim, int intermediate_dim);
    void build_operation_table();
    void launch_persistent_kernel();

    static void compute_rms_norm(/* params */);
    static void compute_matmul_tile(/* params */);
    static void compute_attention(/* params */);
    static void compute_silu_mul(/* params */);
};

} // namespace ggml_sycl
```

---

## Persistent Kernel Execution Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PERSISTENT TG KERNEL                             │
├─────────────────────────────────────────────────────────────────────┤
│  for layer = 0 to n_layers-1:                                        │
│    for op in [ATTN_NORM, QKV, ROPE, ATTN, OUT_PROJ,                 │
│               FFN_NORM, GATE, UP, SILU_MUL, DOWN]:                   │
│                                                                      │
│      ┌──────────────────────────────────────────────────────────┐   │
│      │  TILE WORK-STEALING PHASE                                 │   │
│      │  while (tiles_remaining):                                 │   │
│      │    tile_idx = atomic_fetch_add(&tile_counter, 1)         │   │
│      │    if tile_idx >= total_tiles: break                     │   │
│      │    compute_tile(op, tile_idx)  // registers + SLM        │   │
│      └──────────────────────────────────────────────────────────┘   │
│                                                                      │
│      ┌──────────────────────────────────────────────────────────┐   │
│      │  SPLIT BARRIER SYNC (all work-groups)                     │   │
│      │  split_barrier_arrive(ScopeDevice, SemanticsGlobalMem)   │   │
│      │  // Optional: prefetch next op's weights here            │   │
│      │  split_barrier_wait(ScopeDevice, SemanticsGlobalMem)     │   │
│      └──────────────────────────────────────────────────────────┘   │
│                                                                      │
│      reset_tile_counter()  // thread 0 only, after barrier          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Memory Layout

| Buffer | Size (Mistral 7B) | Location | Lifetime |
|--------|-------------------|----------|----------|
| Hidden state | 4096 × FP16 = 8KB | Global + SLM | Per-layer |
| Q, K, V | 4096 × FP16 each = 24KB | Global | Per-attention |
| Attention output | 4096 × FP16 = 8KB | Global | Per-attention |
| FFN intermediate | 11008 × FP16 = 22KB | Global | Per-FFN |
| Tile counter | 4 bytes | Global (atomic) | Reset per-op |
| Op metadata table | ~2KB | Global (read-only) | Entire kernel |

---

## Unified Cache Extensions

```cpp
class UnifiedCache {
public:
    // Async Prefetch Queue (NEW)
    void queue_layer_prefetch(int layer_id,
                              const LayerWeightSet& weights,
                              ggml_layout_mode layout,
                              PrefetchPriority priority = PrefetchPriority::NORMAL);
    void release_layer(int layer_id);
    LayerWeightPointers await_layer(int layer_id);
    bool is_layer_ready(int layer_id) const;

    // Persistent Kernel Support (NEW)
    WeightPointerTable* build_weight_table(int n_layers,
                                           const std::vector<LayerWeightSet>& layer_weights,
                                           ggml_layout_mode layout);
    PersistentBuffers allocate_persistent_buffers(int hidden_dim,
                                                   int intermediate_dim,
                                                   int n_buffers = 4);
    void free_persistent_buffers(PersistentBuffers& buffers);

private:
    std::queue<PrefetchRequest> prefetch_queue_;
    std::mutex prefetch_mutex_;
    std::condition_variable prefetch_cv_;
    std::thread prefetch_worker_;
    void prefetch_worker_loop();
};
```

---

## Graph Detection and Operation Extraction

```cpp
// In ggml-sycl.cpp

static bool is_persistent_compatible(ggml_backend_sycl_context& ctx,
                                      ggml_cgraph* cgraph) {
    if (!env_persistent_tg_enabled()) return false;
    if (!ctx.unified_kernel->supports_persistent()) return false;
    if (!is_decode_phase(cgraph)) return false;
    if (!has_supported_architecture(cgraph)) return false;
    return true;
}

static void extract_persistent_plan(ggml_backend_sycl_context& ctx,
                                     ggml_cgraph* cgraph) {
    UnifiedKernel& kernel = *ctx.unified_kernel;
    ModelDimensions dims = analyze_graph_dimensions(cgraph);

    kernel.begin_persistent(dims.n_layers, dims.batch_size);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor* node = cgraph->nodes[i];
        // Map ggml ops to kernel ops...
    }

    kernel.execute_persistent();
}
```

---

## Core Operation Implementations

Each operation is a complete device function, not a stub.

### RMS Normalization
- Warp reduction for sum of squares
- Cross-warp reduction via SLM
- Split barrier synchronization
- Scale computation and normalization

### Matrix-Vector Multiply (DMMV)
- Cooperative input vector caching in SLM
- Per-warp output row processing
- Dequantization for Q4_0, Q4_K, Q6_K
- Warp reduction for dot product

### Attention (Single Token)
- Q @ K^T score computation
- Online softmax (max, exp, normalize)
- Weighted value aggregation
- GQA/MQA support

### SiLU + Multiply (Fused)
- Element-wise SiLU: x * sigmoid(x)
- Multiply gate × up
- Trivially parallel

---

## Testing Strategy (TDD)

```
Test Hierarchy:
├── Unit Tests (per operation)
│   ├── test_rms_norm_correctness
│   ├── test_dmmv_q4_0_correctness
│   ├── test_dmmv_q4_k_correctness
│   ├── test_dmmv_q6_k_correctness
│   ├── test_silu_mul_correctness
│   ├── test_attention_single_token_correctness
│   └── test_rope_correctness
│
├── Integration Tests (UnifiedKernel class)
│   ├── test_unified_kernel_single_matmul
│   ├── test_unified_kernel_persistent_single_layer
│   ├── test_unified_kernel_persistent_multi_layer
│   └── test_unified_kernel_cache_integration
│
├── System Tests (end-to-end)
│   ├── test_persistent_vs_standard_output_match
│   ├── test_persistent_larger_than_vram_model
│   └── test_persistent_correctness_mistral_7b
│
└── Performance Tests
    ├── bench_persistent_vs_standard_latency
    ├── bench_kernel_launch_count
    └── bench_memory_bandwidth_utilization
```

**Test Files:**
- `ggml/src/ggml-sycl/tests/test-unified-kernel-ops.cpp` - Unit tests for each operation
- `ggml/src/ggml-sycl/tests/test-unified-kernel-persistent.cpp` - Persistent execution tests
- `ggml/src/ggml-sycl/tests/test-unified-cache-prefetch.cpp` - Async prefetch tests

---

## Success Criteria

| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| TG throughput (Mistral 7B, Q4_0) | ~16 t/s | 25 t/s | 40 t/s |
| Kernel launches per token | 280-392 | 1 | 1 |
| Output correctness | - | max_error < 1e-4 | max_error < 1e-5 |
| Memory bandwidth utilization | ~40% | 60% | 80% |
| Works with >VRAM models | No | Yes | Yes |

---

## Implementation Order

1. **UnifiedKernel class scaffold** - Class structure, state management
2. **Async prefetch API** - Cache extensions for layer streaming
3. **RMS norm operation** - First complete operation with tests
4. **DMMV operation** - Matrix-vector with dequantization
5. **SiLU + multiply** - Fused activation
6. **Attention operation** - Score, softmax, value aggregation
7. **Persistent execution loop** - Work-stealing + split barriers
8. **Graph detection** - Intercept TG workloads
9. **Integration testing** - End-to-end validation
10. **Performance tuning** - Optimize for target throughput

---

## Files to Modify

| File | Changes |
|------|---------|
| `unified-kernel.hpp` | Add UnifiedKernel class, operation descriptors |
| `unified-kernel.cpp` | Implement class methods, device functions |
| `unified-cache.hpp` | Add async prefetch API |
| `unified-cache.cpp` | Implement prefetch worker thread |
| `ggml-sycl.cpp` | Add graph detection, operation extraction |
| `dispatch.hpp` | Update kernel type enum if needed |
| `tests/test-unified-kernel-ops.cpp` | NEW: Operation unit tests |
| `tests/test-unified-kernel-persistent.cpp` | NEW: Persistent tests |
| `tests/test-unified-cache-prefetch.cpp` | NEW: Cache tests |

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Split barrier deadlock | Medium | High | Careful barrier placement, extensive testing |
| Numerical accuracy loss | Low | Medium | FP32 accumulators, validation against baseline |
| Memory pressure | Medium | Medium | Layer streaming, careful buffer management |
| Performance regression | Low | High | A/B testing, fallback to standard path |
