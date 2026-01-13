# Unified Tensor Cache Design

**Date:** 2026-01-12
**Status:** Design Complete
**Target:** SYCL backend, Intel Arc GPUs, models exceeding VRAM

## Problem Statement

The current tiered memory components (`unified-cache`, `expert_cache`, `dense_layer_scheduler`, `pinned_chunk_pool`) are fragmented. They need consolidation into a single **Unified Tensor Cache** that:

1. Manages all model weights across three memory tiers (VRAM → Pinned Host → mmap)
2. Places tensors optimally based on priority and available memory
3. Dynamically promotes/demotes based on access patterns
4. Auto-enables when model exceeds VRAM

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                Unified Tensor Cache                 │
├─────────────────────────────────────────────────────┤
│  VRAM Pool          │  Pinned Host Pool  │  mmap   │
│  (owns allocations) │  (8GB chunks)      │ (fallback)
│                     │                    │         │
│  - Hot tensors      │  - Warm tensors    │  - Cold │
│  - Priority-based   │  - GPU-accessible  │  - Disk │
└─────────────────────────────────────────────────────┘
```

**Key properties:**
- Owns both VRAM and host memory pools directly (no ggml buffer wrapping)
- Two-phase loading: enumerate tensors from GGUF first, then place optimally
- Hybrid priority: static type-based + dynamic access-pattern adjustment
- Single API for kernels: `get_tensor_with_location()` returns pointer + tier

## Two-Phase Loading

### Phase 1: Tensor Inventory (in llama-model.cpp)

During GGUF parsing, llama.cpp collects tensor metadata without allocating:

```cpp
struct tensor_info {
    std::string name;
    size_t      size;
    int         layer_id;      // -1 if not layer-specific
    int         expert_id;     // -1 if not expert
    tensor_class type;         // EMBEDDING, ATTENTION, FFN, ROUTER, EXPERT, NORM, OTHER
    int         static_priority;  // 0=highest, computed from type
};

struct tensor_inventory {
    std::vector<tensor_info> tensors;
    size_t total_size;
    size_t dense_size;    // embeddings + attention + ffn + router + norms
    size_t expert_size;   // all expert weights
};
```

### Phase 2: Placement Decision (in SYCL backend)

Backend receives inventory and decides placement:

```cpp
// Called once before allocation begins
void ggml_backend_sycl_set_tensor_inventory(
    ggml_backend_t backend,
    const tensor_inventory& inventory
);
```

Backend sorts by priority, then fills tiers:
1. Sort tensors by `static_priority` (ascending = higher priority first)
2. Fill VRAM with highest-priority tensors until budget exhausted
3. Remaining go to pinned host pool
4. If host exhausted, fall back to mmap (unlikely for RAM-fitting models)

## Static Priority System

Tensors are classified and assigned priorities during inventory collection:

| Priority | Type | Pattern Match | Rationale |
|----------|------|---------------|-----------|
| 0 | Embedding | `token_embd` | Used every input token |
| 0 | Output | `lm_head`, `output` | Used every generated token |
| 1 | Attention | `attn_q`, `attn_k`, `attn_v`, `attn_o` | Every layer, every token |
| 2 | Dense FFN | `ffn_up`, `ffn_down`, `ffn_gate` (non-MoE) | Every layer, every token |
| 2 | Router | `ffn_gate_inp` | Every MoE layer |
| 3 | Experts | `*_exps` | Conditional, ~10-20% usage |
| 4 | Norms | `*_norm` | Tiny size, fast from host |

**Classification function:**

```cpp
tensor_class classify_tensor(const char* name) {
    if (strstr(name, "token_embd")) return EMBEDDING;
    if (strstr(name, "lm_head") || strstr(name, "output.weight")) return OUTPUT;
    if (strstr(name, "attn_")) return ATTENTION;
    if (strstr(name, "_exps")) return EXPERT;
    if (strstr(name, "ffn_gate_inp")) return ROUTER;
    if (strstr(name, "ffn_")) return FFN;
    if (strstr(name, "_norm")) return NORM;

#if GGML_SYCL_DEBUG
    static std::unordered_set<std::string> warned_tensors;
    if (warned_tensors.insert(name).second) {
        GGML_LOG_WARN("[SYCL] Unknown tensor type: %s\n", name);
    }
#endif
    return OTHER;
}
```

## Runtime Access & Dynamic Promotion

### Kernel Interface

```cpp
struct tensor_location {
    void*       ptr;
    memory_tier tier;  // VRAM, PINNED_HOST, MMAP
};

tensor_location cache.get_tensor_with_location(tensor_id id);
```

Kernels can adapt behavior based on tier (e.g., different algorithms for host memory).

### Dynamic Promotion (for experts)

Experts initially placed in host can be promoted to VRAM based on access patterns:

1. **Primary: Router-based prefetch**
   - Read expert IDs from `ids` tensor before MUL_MAT_ID
   - Prefetch needed experts to VRAM asynchronously
   - Reuses existing `graph_preload_moe_experts()` pattern

2. **Fallback: Access-pattern promotion**
   - Track access count + recency per tensor
   - When accessing host-resident tensor, schedule async promotion
   - Score = `0.3 * recency + 0.7 * frequency`
   - High-score tensors get promoted, low-score evicted

### Eviction

When VRAM is full and promotion needed:
- Evict lowest-score tensor from VRAM to make room
- Data isn't lost—host pool still has it (or can reload from mmap)
- Evicted tensor reverts to host-access path

## Memory Pool Architecture

### VRAM Pool

```cpp
class vram_pool {
    sycl::queue& queue_;
    size_t budget_;
    size_t used_ = 0;

    struct allocation {
        void* ptr;
        size_t size;
        tensor_id owner;
    };
    std::vector<allocation> allocations_;

public:
    void* allocate(size_t size, tensor_id owner);
    void deallocate(tensor_id owner);
    size_t available() const { return budget_ - used_; }
};
```

### Pinned Host Pool (existing pinned_chunk_pool)

- 8GB chunks via `sycl::malloc_host`
- Bump allocator with 64-byte alignment
- Lazy growth up to budget

### Unified Cache

```cpp
class unified_tensor_cache {
    vram_pool vram_;
    pinned_chunk_pool host_;

    struct tensor_entry {
        tensor_info info;
        void* host_ptr;           // always set (primary location or backup)
        void* vram_ptr;           // set if resident in VRAM
        memory_tier current_tier;
        uint64_t last_access;
        uint32_t access_count;
    };
    std::unordered_map<tensor_id, tensor_entry> entries_;
};
```

## Integration Points

### 1. llama-model.cpp - Tensor Inventory Collection

During `llama_model_load()`, after GGUF parsing:

```cpp
// Collect inventory before allocation
tensor_inventory inventory;
for (auto& tensor : model.tensors) {
    tensor_info info;
    info.name = tensor.name;
    info.size = ggml_nbytes(tensor.t);
    info.layer_id = extract_layer_id(tensor.name);
    info.expert_id = extract_expert_id(tensor.name);
    info.type = classify_tensor(tensor.name.c_str());
    info.static_priority = priority_from_type(info.type);
    inventory.tensors.push_back(info);
}

// Pass to SYCL backend before allocation
if (backend_is_sycl(model.backend)) {
    ggml_backend_sycl_set_tensor_inventory(model.backend, inventory);
}
```

### 2. ggml-sycl.cpp - New API

```cpp
// Public API
void ggml_backend_sycl_set_tensor_inventory(
    ggml_backend_t backend,
    const tensor_inventory& inventory
);

// Internal: called during tensor allocation
ggml_backend_buffer_type_t ggml_sycl_get_buffer_for_tensor(
    const char* name,
    size_t size
);
```

### 3. Kernel Callsites

Replace direct pointer access with cache lookup:

```cpp
// Before:
const void* weights = src0->data;

// After:
auto [weights, tier] = ctx.cache->get_tensor_with_location(src0);
// tier available if kernel wants to adapt
```

## Kernels Requiring Updates

### Core Matrix Multiplication

| File | Function | What it accesses |
|------|----------|------------------|
| `mmvq.cpp` | `ggml_sycl_op_mul_mat_vec_q` | Quantized weights for generation |
| `mmq.cpp` | `ggml_sycl_op_mul_mat_q` | Quantized weights for batched/prompt |
| `dmmv.cpp` | `ggml_sycl_op_dequantize_mul_mat_vec` | Dequantized mat-vec path |
| `gemm.cpp` | `DnnlGemmWrapper::gemm` | F16/F32 weights via oneDNN |

### MoE-Specific

| File | Function | What it accesses |
|------|----------|------------------|
| `ggml-sycl.cpp` | `ggml_sycl_mul_mat_id` | Expert weight dispatch |
| `mmvq.cpp` | `ggml_sycl_mul_mat_id_vec_q` | Expert weights in MMVQ path |
| `ggml-sycl.cpp` | `graph_preload_moe_experts` | Expert prefetch |

### Embedding/Lookup

| File | Function | What it accesses |
|------|----------|------------------|
| `getrows.cpp` | `ggml_sycl_op_get_rows` | Token embeddings lookup |

### XMX Tiled Kernels

| File | Function | What it accesses |
|------|----------|------------------|
| `mmq.cpp` | XMX Q4_0/Q8_0 tiled paths | Quantized weights via joint_matrix |
| `mmvq.cpp` | XMX MMVQ variants | Quantized weights for generation |
| `ggml-sycl.cpp` | `use_xmx_for_type()` dispatch | Routes to XMX kernels |

### Supporting Operations

| File | Function | What it accesses |
|------|----------|------------------|
| `outprod.cpp` | `ggml_sycl_op_out_prod` | Weight updates |
| `cpy.cpp` | `ggml_sycl_cpy` | Weight copies during reorder |
| `convert.cpp` | Various | Type conversions |

### Update Priority Order

1. `ggml_sycl_op_mul_mat` - main dispatch
2. `ggml_sycl_mul_mat_id` - MoE routing
3. `ggml_sycl_op_get_rows` - embeddings (priority 0 tensors!)
4. `mmvq.cpp` / `mmq.cpp` - including XMX paths
5. XMX tiled kernel variants
6. Rest follow

## Configuration

### Auto-Enable Heuristic

```cpp
bool should_enable_tiered(const tensor_inventory& inv, size_t vram_available) {
    if (inv.total_size <= vram_available * 0.9) {
        return false;  // Fits in VRAM, no tiering needed
    }
    GGML_LOG_INFO("[SYCL] Model size (%.1f GB) exceeds VRAM (%.1f GB), enabling tiered memory\n",
                  inv.total_size / (1024.0*1024.0*1024.0),
                  vram_available / (1024.0*1024.0*1024.0));
    return true;
}
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--sycl-tiered-memory` | auto | Enable tiered memory (auto/on/off) |
| `--sycl-vram-budget` | auto | VRAM budget in GB (auto = total - 1GB reserve) |
| `--sycl-host-budget` | 90% | Percentage of system RAM for pinned pool |

### Environment Variables (debugging)

| Variable | Default | Description |
|----------|---------|-------------|
| `GGML_SYCL_TIERED_DEBUG` | 0 | Log placement decisions and cache activity |
| `GGML_SYCL_FORCE_HOST` | 0 | Force all weights to host (for testing) |

## Testing & Verification

### Unit Tests

| Test | Validates |
|------|-----------|
| `test-unified-cache` | Pool allocation, promotion, eviction |
| `test-tensor-classification` | Priority assignment for all tensor patterns |
| `test-inventory-passing` | llama-model → SYCL backend handoff |

### Integration Tests

| Model | Size | Expected behavior |
|-------|------|-------------------|
| Mistral-7B Q4_0 | ~4GB | Fits in VRAM, tiered disabled |
| GPT-OSS-20B Q8_0 | ~20GB | Dense in VRAM, experts tiered |
| GPT-OSS-120B Q4_0 | ~60GB | Heavy tiering, expert cache active |

### Verification Commands

```bash
# Should see tiered memory auto-enable, cache stats
GGML_SYCL_TIERED_DEBUG=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-cli -m gpt-oss-120b-Q4_0.gguf \
  -ngl 99 -p "Hello" -n 50 2>&1 | grep -E "(tiered|cache|VRAM|pinned)"
```

### Success Criteria

- No "device lost" errors
- No "falling back to CPU memory" warnings
- Reasonable generation speed (>5 t/s for 120B)
- Cache hit rate >80% after warmup
