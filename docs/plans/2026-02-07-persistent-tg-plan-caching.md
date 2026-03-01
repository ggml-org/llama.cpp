# Persistent TG Plan Caching Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use team-driven-development to implement this plan with agent teams.

**Goal:** Optimize the persistent TG kernel's per-token plan building from ~565ms to <1ms by caching the plan template and pooling device allocations.

**Architecture:** The persistent TG kernel executes an entire LLM forward pass in a single kernel launch. Currently `extract_persistent_plan()` rebuilds the plan from scratch every token — 50+ getenv calls, full graph walks, device allocations. We cache the plan after first build, then on subsequent tokens only update mutable pointers (activations, KV cache, RoPE). Device allocations for the ops table and sync counters are pooled across tokens.

**Tech Stack:** C++17, Intel oneAPI SYCL, GGML compute graphs

---

## Team Topology

**Recommended implementers:** 2 (based on 2 parallel tracks)
**Reviewers:** 1 code-reviewer

### Parallel Tracks

| Track | Tasks | Description |
|-------|-------|-------------|
| A | 1 | UnifiedKernel caching API + device allocation pools |
| B | 2 | Debug init refactor in extract_persistent_plan |
| — | 3 | Plan caching caller integration (depends on Task 1) |

### Dependency Graph

```
Task 1 (Track A) ──→ Task 3 (convergence)
Task 2 (Track B) ──→ Task 3 (convergence)
```

### File Ownership Map

| File/Directory | Tasks | Conflict Risk |
|----------------|-------|---------------|
| `ggml/src/ggml-sycl/unified-kernel.hpp` | 1 | None (single task) |
| `ggml/src/ggml-sycl/unified-kernel.cpp` | 1 | None (single task) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp:25647-25733` | 2 | None (disjoint region) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp:25644-26137` | 3 | Sequential after Task 2 |

---

## Background Context for Implementers

### Key Structures

**`UnifiedKernel`** class (`unified-kernel.hpp:2534-2618`): Manages persistent TG execution. Has `current_plan_` (unique_ptr to PersistentPlan), persistent buffer allocations, and sync counters.

**`PersistentPlan`** (`unified-kernel.hpp:386-424`): Contains model dimensions, `vector<OperationDescriptor> operations`, debug state, and `vector<void*> temp_device_allocs`.

**`OperationDescriptor`** (`unified-kernel.hpp:302-338`): Per-operation descriptor with `type`, `layer`, `weights`, `input`, `output`, `aux`, `mask`, dimensions (M/N/K), strides, etc.

**`DeviceOperation`** (`unified-kernel.cpp:3965`): Device-side copy of OperationDescriptor, 64-byte aligned.

### What Changes Between TG Tokens

| Field | Changes? | Why |
|-------|----------|-----|
| `weights` | No | Model weights are static on device |
| `input` | Yes | Activation buffers managed by ggml allocator |
| `output` | Yes | Same as input |
| `aux` | Maybe | RoPE sin/cos change; fused weight ptr stays |
| `mask` | Yes | Attention mask grows with sequence |
| `type`, `M`, `N`, `K` | No | Model structure is static |
| KV cache strides | Yes | KV cache grows each token |

### Current Per-Token Flow (slow)

1. `cancel_persistent()` — frees temp allocs, resets plan
2. `begin_persistent()` — allocs new PersistentPlan
3. `extract_persistent_plan()` — 50 getenvs, graph walks, pointer resolution, op construction
4. `launch_persistent_kernel()` — allocs device ops table, copies, 3 barrier resets, launches
5. `execute_persistent()` — frees temp allocs, resets plan

### Build & Test Commands

```bash
source /opt/intel/oneapi/setvars.sh --force
ninja -C /Apps/llama.cpp/build -j $(nproc)

# Correctness
ONEAPI_DEVICE_SELECTOR=level_zero:0 /Apps/llama.cpp/build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0

# TG benchmark (target: ~76 tok/s)
ONEAPI_DEVICE_SELECTOR=level_zero:0 /Apps/llama.cpp/build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 0 -n 128

# PP benchmark (must stay ~1300 tok/s)
ONEAPI_DEVICE_SELECTOR=level_zero:0 /Apps/llama.cpp/build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 0
```

---

### Task 1: Add Plan Caching and Device Pools to UnifiedKernel

**Track:** A
**Depends on:** None
**File scope:**
- Modify: `ggml/src/ggml-sycl/unified-kernel.hpp:2590-2618` (add members + methods)
- Modify: `ggml/src/ggml-sycl/unified-kernel.cpp:5248-5268` (begin_persistent)
- Modify: `ggml/src/ggml-sycl/unified-kernel.cpp:5628-5666` (add_temp_device_alloc, cancel_persistent, execute_persistent)
- Modify: `ggml/src/ggml-sycl/unified-kernel.cpp:5668-5868` (launch_persistent_kernel)

**Description:**

Add plan caching capability to UnifiedKernel so that the operation list from the first successful execution can be reused on subsequent tokens with only pointer updates. Also pool the device ops table allocation and batch the barrier reset memsets to eliminate per-token device memory management.

**Acceptance Criteria:**

- [ ] `has_cached_plan()` returns true after first successful `execute_persistent()`
- [ ] `begin_plan_update()` creates a working `current_plan_` from cached template
- [ ] `finish_plan_update()` marks plan ready for execution
- [ ] `invalidate_plan_cache()` forces full rebuild on next call
- [ ] Device ops table is allocated once and reused
- [ ] Barrier counters are reset with single memset
- [ ] `get_rows_stable_ptr()` returns pooled allocation
- [ ] All existing tests pass
- [ ] Code follows project conventions (snake_case, 4-space indent)

**Implementation Guide:**

1. **Add public methods to UnifiedKernel** (`unified-kernel.hpp:2590`, before `private:`):

```cpp
    // Plan caching: reuse operation sequence between TG tokens
    bool has_cached_plan() const;
    void begin_plan_update();
    void update_op_pointers(int op_idx, const void * input, void * output,
                            const void * aux = nullptr, const void * mask = nullptr);
    void update_op_attention(int op_idx, const void * q, const void * k_cache,
                             const void * v_cache, const void * mask,
                             void * output,
                             int64_t q_nb0, int64_t q_nb1, int64_t q_nb2, int64_t q_nb3,
                             int64_t k_nb0, int64_t k_nb1, int64_t k_nb2, int64_t k_nb3,
                             int64_t v_nb0, int64_t v_nb1, int64_t v_nb2, int64_t v_nb3,
                             int seq_len);
    void update_op_rope(int op_idx, void * q, void * k, void * rope_dst,
                        const float * cos_cache, const float * sin_cache, int position);
    void finish_plan_update();
    void invalidate_plan_cache();
    void * get_rows_stable_ptr(size_t bytes);
```

2. **Add private members** (`unified-kernel.hpp:2603`, after `current_plan_`):

```cpp
    // Plan caching
    std::vector<OperationDescriptor> cached_ops_;
    PersistentPlan                   cached_plan_template_;
    bool                             plan_cache_valid_ = false;

    // Device ops table pool
    DeviceOperation * d_ops_pool_      = nullptr;
    int               d_ops_pool_size_ = 0;

    // Batched sync counter (tile_counter + barrier_counter + barrier_sense)
    int *  sync_block_      = nullptr;

    // GET_ROWS stable copy pool
    void * get_rows_pool_      = nullptr;
    size_t get_rows_pool_size_ = 0;
```

3. **Implement caching methods** in `unified-kernel.cpp` (add after `cancel_persistent` at line 5643):

```cpp
bool UnifiedKernel::has_cached_plan() const {
    return plan_cache_valid_;
}

void UnifiedKernel::begin_plan_update() {
    // Cancel any in-flight plan but DON'T free cached data
    if (current_plan_) {
        for (void * ptr : current_plan_->temp_device_allocs) {
            sycl::free(ptr, queue_);
        }
        current_plan_.reset();
    }

    // Clone from cached template
    current_plan_ = std::make_unique<PersistentPlan>();
    current_plan_->n_layers          = cached_plan_template_.n_layers;
    current_plan_->batch_size        = cached_plan_template_.batch_size;
    current_plan_->hidden_dim        = cached_plan_template_.hidden_dim;
    current_plan_->intermediate_dim  = cached_plan_template_.intermediate_dim;
    current_plan_->n_heads           = cached_plan_template_.n_heads;
    current_plan_->n_kv_heads        = cached_plan_template_.n_kv_heads;
    current_plan_->head_dim          = cached_plan_template_.head_dim;
    current_plan_->quant_type        = cached_plan_template_.quant_type;
    current_plan_->operations        = cached_ops_;  // copy the vector
}

void UnifiedKernel::update_op_pointers(int op_idx, const void * input, void * output,
                                        const void * aux, const void * mask) {
    if (!current_plan_ || op_idx < 0 || op_idx >= (int) current_plan_->operations.size()) {
        return;
    }
    auto & op = current_plan_->operations[op_idx];
    if (input)  op.input  = input;
    if (output) op.output = output;
    if (aux)    op.aux    = const_cast<void *>(aux);
    if (mask)   op.mask   = mask;
}

void UnifiedKernel::update_op_attention(int op_idx, const void * q, const void * k_cache,
                                         const void * v_cache, const void * mask,
                                         void * output,
                                         int64_t q_nb0, int64_t q_nb1, int64_t q_nb2, int64_t q_nb3,
                                         int64_t k_nb0, int64_t k_nb1, int64_t k_nb2, int64_t k_nb3,
                                         int64_t v_nb0, int64_t v_nb1, int64_t v_nb2, int64_t v_nb3,
                                         int seq_len) {
    if (!current_plan_ || op_idx < 0 || op_idx >= (int) current_plan_->operations.size()) {
        return;
    }
    auto & op  = current_plan_->operations[op_idx];
    op.input   = q;
    op.weights = k_cache;
    op.aux     = const_cast<void *>(static_cast<const void *>(v_cache));
    op.mask    = mask;
    op.output  = output;
    op.q_nb0   = q_nb0;  op.q_nb1 = q_nb1;  op.q_nb2 = q_nb2;  op.q_nb3 = q_nb3;
    op.k_nb0   = k_nb0;  op.k_nb1 = k_nb1;  op.k_nb2 = k_nb2;  op.k_nb3 = k_nb3;
    op.v_nb0   = v_nb0;  op.v_nb1 = v_nb1;  op.v_nb2 = v_nb2;  op.v_nb3 = v_nb3;
    op.N       = seq_len;
}

void UnifiedKernel::update_op_rope(int op_idx, void * q, void * k, void * rope_dst,
                                    const float * cos_cache, const float * sin_cache,
                                    int position) {
    if (!current_plan_ || op_idx < 0 || op_idx >= (int) current_plan_->operations.size()) {
        return;
    }
    auto & op = current_plan_->operations[op_idx];
    op.input  = q;
    op.output = k;  // K pointer stored in output for dual-tensor RoPE
    op.aux    = const_cast<float *>(sin_cache);
    op.weights = cos_cache;
    op.M       = position;
    // rope_dst used for single-tensor mode
    if (rope_dst) {
        op.output = rope_dst;
    }
}

void UnifiedKernel::finish_plan_update() {
    // Plan is already populated with updated pointers, nothing else needed
}

void UnifiedKernel::invalidate_plan_cache() {
    plan_cache_valid_ = false;
    cached_ops_.clear();
}

void * UnifiedKernel::get_rows_stable_ptr(size_t bytes) {
    if (bytes <= get_rows_pool_size_ && get_rows_pool_) {
        return get_rows_pool_;
    }
    if (get_rows_pool_) {
        sycl::free(get_rows_pool_, queue_);
    }
    get_rows_pool_ = sycl::malloc_device(bytes, queue_);
    get_rows_pool_size_ = get_rows_pool_ ? bytes : 0;
    return get_rows_pool_;
}
```

4. **Modify `execute_persistent()`** to cache plan after first execution (line 5649):

Replace the current `execute_persistent()` with:
```cpp
void UnifiedKernel::execute_persistent() {
    if (!current_plan_ || !current_plan_->is_valid()) {
        GGML_LOG_ERROR("UnifiedKernel: execute_persistent called with invalid plan\n");
        return;
    }

    // Launch the persistent kernel
    launch_persistent_kernel();

    // Cache plan template after first successful execution
    if (!plan_cache_valid_) {
        cached_plan_template_.n_layers         = current_plan_->n_layers;
        cached_plan_template_.batch_size       = current_plan_->batch_size;
        cached_plan_template_.hidden_dim       = current_plan_->hidden_dim;
        cached_plan_template_.intermediate_dim = current_plan_->intermediate_dim;
        cached_plan_template_.n_heads          = current_plan_->n_heads;
        cached_plan_template_.n_kv_heads       = current_plan_->n_kv_heads;
        cached_plan_template_.head_dim         = current_plan_->head_dim;
        cached_plan_template_.quant_type       = current_plan_->quant_type;
        cached_ops_ = current_plan_->operations;
        plan_cache_valid_ = true;
        GGML_SYCL_DEBUG("[PERSISTENT-TG] Plan cached: %zu operations\n", cached_ops_.size());
    }

    // Free temporary device allocations (e.g. RoPE cos/sin caches)
    for (void * ptr : current_plan_->temp_device_allocs) {
        sycl::free(ptr, queue_);
    }
    current_plan_->temp_device_allocs.clear();

    // Clear the plan after execution (cached copy remains)
    current_plan_.reset();
}
```

5. **Pool device ops table in `launch_persistent_kernel()`** (line 5809):

Replace:
```cpp
    DeviceOperation * d_ops = sycl::malloc_device<DeviceOperation>(n_ops_device, queue_);
    queue_.memcpy(d_ops, host_ops.data(), host_ops.size() * sizeof(DeviceOperation)).wait();
```

With:
```cpp
    // Reuse device ops allocation when capacity is sufficient
    if (n_ops_device > d_ops_pool_size_) {
        if (d_ops_pool_) sycl::free(d_ops_pool_, queue_);
        d_ops_pool_ = sycl::malloc_device<DeviceOperation>(n_ops_device, queue_);
        d_ops_pool_size_ = d_ops_pool_ ? n_ops_device : 0;
    }
    queue_.memcpy(d_ops_pool_, host_ops.data(), host_ops.size() * sizeof(DeviceOperation)).wait();
```

Then update all references from `d_ops` to `d_ops_pool_` in the rest of `launch_persistent_kernel`. Remove any `sycl::free(d_ops, ...)` call at the end of the function.

6. **Batch barrier resets** in `launch_persistent_kernel()` (line 5833-5838):

In `allocate_persistent_buffers()`, add sync_block allocation:
```cpp
    if (!sync_block_) {
        sync_block_ = sycl::malloc_device<int>(3, queue_);
    }
    tile_counter_    = sync_block_;
    barrier_counter_ = sync_block_ + 1;
    barrier_sense_   = sync_block_ + 2;
```

Replace the three separate memset+wait calls:
```cpp
    queue_.memset(tile_counter_, 0, sizeof(int)).wait();
    queue_.memset(barrier_counter_, 0, sizeof(int)).wait();
    queue_.memset(barrier_sense_, 0, sizeof(int)).wait();
```
With:
```cpp
    queue_.memset(sync_block_, 0, 3 * sizeof(int)).wait();
```

7. **Clean up pools in destructor/free_persistent_buffers**:

In `free_persistent_buffers()` add:
```cpp
    if (d_ops_pool_) { sycl::free(d_ops_pool_, queue_); d_ops_pool_ = nullptr; d_ops_pool_size_ = 0; }
    if (sync_block_) { sycl::free(sync_block_, queue_); sync_block_ = nullptr; }
    if (get_rows_pool_) { sycl::free(get_rows_pool_, queue_); get_rows_pool_ = nullptr; get_rows_pool_size_ = 0; }
    tile_counter_ = nullptr;
    barrier_counter_ = nullptr;
    barrier_sense_ = nullptr;
```

**Commit:**
```bash
git add ggml/src/ggml-sycl/unified-kernel.hpp ggml/src/ggml-sycl/unified-kernel.cpp
git commit -m "sycl: add plan caching and device allocation pools to UnifiedKernel"
```

**Notes for implementer:**
- The `DeviceOperation` type is defined at unified-kernel.cpp:3965 (file-local). Forward-declare or include as needed for the pool member.
- `GGML_SYCL_DEBUG` is a macro that only prints when `g_ggml_sycl_debug` is true.
- The `allocate_persistent_buffers` function is called from `launch_persistent_kernel` — look for where `tile_counter_` is first allocated.
- Keep `cancel_persistent()` working — it should free temp allocs of current_plan_ but NOT invalidate the cache.

---

### Task 2: Refactor Debug Initialization in extract_persistent_plan

**Track:** B
**Depends on:** None
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:25644-25733`

**Description:**

Replace the ~50 `std::getenv()` calls at the top of `extract_persistent_plan()` with a single lazy-initialized static struct. These env vars are read-once configuration but are currently evaluated on every token, adding needless overhead. The mutable per-call debug state (captured pointers, flags) stays per-call.

**Acceptance Criteria:**

- [ ] All env vars are read exactly once (via static initialization)
- [ ] Per-call mutable state (g_persistent_*_dbg structs) still reset each call
- [ ] No functional change in debug behavior
- [ ] Build succeeds with no warnings

**Implementation Guide:**

1. **Add static config struct** above `extract_persistent_plan` (before line 25644):

```cpp
// One-time initialization of persistent TG debug configuration.
// Env vars are read-once; mutable per-call state is separate.
struct PersistentTGDebugConfig {
    bool trace_hash;
    bool set_rows_check;
    bool attn_validate;
    bool rms_validate;
    bool matmul_validate;
    bool add_dump;
    bool mul_dump;
    bool profile_build;
    bool log_ops;
    bool stage_host_ptrs;
    int  max_ops_limit;
    int  rms_dump_instance;

    static const PersistentTGDebugConfig & get() {
        static const PersistentTGDebugConfig cfg = init();
        return cfg;
    }

private:
    static PersistentTGDebugConfig init() {
        PersistentTGDebugConfig c = {};
        c.trace_hash     = false;  // Set by init_sycl_tg_trace() separately
        c.set_rows_check = (std::getenv("GGML_SYCL_PERSISTENT_TG_CHECK_SET_ROWS") != nullptr) ||
                           (std::getenv("GGML_SYCL_PERSISTENT_TG_VALIDATE_SET_ROWS") != nullptr);
        c.attn_validate  = (std::getenv("GGML_SYCL_PERSISTENT_TG_VALIDATE_ATTN") != nullptr);
        c.rms_validate   = (std::getenv("GGML_SYCL_PERSISTENT_TG_VALIDATE_RMS_NORM") != nullptr);
        c.matmul_validate = (std::getenv("GGML_SYCL_PERSISTENT_TG_VALIDATE_MATMUL") != nullptr);
        {
            const char * env = std::getenv("GGML_SYCL_TG_DUMP_ADD");
            c.add_dump = (env && std::atoi(env) != 0);
        }
        {
            const char * env = std::getenv("GGML_SYCL_TG_DUMP_MUL");
            c.mul_dump = (env && std::atoi(env) != 0);
        }
        c.profile_build  = (std::getenv("GGML_SYCL_PERSISTENT_TG_PROFILE_BUILD") != nullptr);
        c.log_ops        = (std::getenv("GGML_SYCL_PERSISTENT_TG_LOG_OPS") != nullptr);
        c.stage_host_ptrs = (std::getenv("GGML_SYCL_PERSISTENT_TG_STAGE_HOST_PTRS") != nullptr);
        c.max_ops_limit  = -1;
        if (const char * env = std::getenv("GGML_SYCL_PERSISTENT_TG_MAX_OPS")) {
            const int val = std::atoi(env);
            if (val > 0) c.max_ops_limit = val;
        }
        c.rms_dump_instance = -1;
        if (const char * env = std::getenv("GGML_SYCL_TG_DUMP_RMS_INSTANCE")) {
            if (*env) {
                char * end = nullptr;
                const long val = std::strtol(env, &end, 10);
                if (end && end != env) c.rms_dump_instance = (int) val;
            }
        }
        return c;
    }
};
```

2. **Replace env var block in extract_persistent_plan** (lines 25647-25733):

Replace the entire block from `init_sycl_tg_trace();` through the `rms_dump_instance` parsing with:

```cpp
    init_sycl_tg_trace();
    const auto & dbg_cfg = PersistentTGDebugConfig::get();

    if (g_sycl_tg_trace_hash) {
        g_persistent_trace_ops.clear();
        g_sycl_trace_op_index = 0;
    }
    g_persistent_debug_hash_ptr = nullptr;

    g_persistent_set_rows_dbg.enabled  = dbg_cfg.set_rows_check;
    g_persistent_set_rows_dbg.captured = false;
    g_persistent_set_rows_dbg.debug_ptr = nullptr;
    g_persistent_set_rows_dbg.debug_count = 0;

    g_persistent_attn_dbg.enabled  = dbg_cfg.attn_validate;
    g_persistent_attn_dbg.captured = false;
    g_persistent_attn_dbg.node = nullptr;
    g_persistent_attn_dbg.debug_ptr = nullptr;
    g_persistent_attn_dbg.debug_floats = 0;
    g_persistent_attn_dbg.layer = -1;
    static bool logged_attn_env = false;
    if (!logged_attn_env) {
        logged_attn_env = true;
        GGML_SYCL_DEBUG("[PERSISTENT-TG] ATTN validate env=%s\n",
                        dbg_cfg.attn_validate ? "enabled" : "(null)");
    }

    g_persistent_rms_dbg.enabled  = dbg_cfg.rms_validate;
    g_persistent_rms_dbg.captured = false;
    g_persistent_rms_dbg.node = nullptr;
    g_persistent_rms_dbg.debug_ptr = nullptr;
    g_persistent_rms_dbg.debug_flag = nullptr;
    g_persistent_rms_dbg.hidden_dim = 0;
    g_persistent_rms_dbg.eps = 0.0f;
    g_persistent_rms_dbg.layer = -1;

    g_persistent_matmul_dbg.enabled  = dbg_cfg.matmul_validate;
    g_persistent_matmul_dbg.captured = false;
    g_persistent_matmul_dbg.node = nullptr;
    g_persistent_matmul_dbg.debug_ptr = nullptr;
    g_persistent_matmul_dbg.debug_flag = nullptr;
    g_persistent_matmul_dbg.out_dim = 0;
    g_persistent_matmul_dbg.layer = -1;
    g_persistent_matmul_dbg.mtype = MatmulType::GENERIC;

    g_persistent_add_dbg.enabled  = dbg_cfg.add_dump;
    g_persistent_add_dbg.captured = false;
    g_persistent_add_dbg.node = nullptr;
    g_persistent_add_dbg.out_dim = 0;
    g_persistent_add_dbg.layer = -1;

    g_persistent_mul_dbg.enabled  = dbg_cfg.mul_dump;
    g_persistent_mul_dbg.captured = false;
    g_persistent_mul_dbg.node = nullptr;
    g_persistent_mul_dbg.out_dim = 0;
    g_persistent_mul_dbg.layer = -1;

    std::array<int, GGML_OP_COUNT> graph_op_counts{};
    std::array<int, GGML_OP_COUNT> plan_op_counts{};
    int skipped_view    = 0;
    int skipped_reshape = 0;
    int skipped_permute = 0;
    int executed_get_rows = 0;
    int executed_cpy      = 0;
    int64_t decode_kv_pos_hint = -1;
    int ops_added = 0;
    const bool profile_build = dbg_cfg.profile_build;
    using clock_t = std::chrono::high_resolution_clock;
    const auto build_start = profile_build ? clock_t::now() : clock_t::time_point{};
    std::array<double, GGML_OP_COUNT> build_ms_by_op{};
    const int max_ops_limit = dbg_cfg.max_ops_limit;
    const bool log_ops = dbg_cfg.log_ops;
    const int rms_dump_instance = dbg_cfg.rms_dump_instance;
```

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: one-time debug init in extract_persistent_plan"
```

**Notes for implementer:**
- `init_sycl_tg_trace()` must still be called per-call (it manages trace state)
- The `g_persistent_*_dbg` global structs contain mutable per-call state and MUST be reset each call
- `logged_attn_env` is already static in the original code — keep it that way
- The variables `profile_build`, `max_ops_limit`, `log_ops`, `rms_dump_instance` are used later in the function, so they must remain as local variables (now initialized from `dbg_cfg`)

---

### Task 3: Integrate Plan Caching in extract_persistent_plan

**Track:** — (convergence)
**Depends on:** Task 1, Task 2
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:25644-26137` (extract_persistent_plan)
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:28431-28515` (graph_compute persistent dispatch)

**Description:**

Add a fast update path to `extract_persistent_plan` that uses the cached plan from Task 1 instead of rebuilding from scratch. When `kernel.has_cached_plan()` is true, walk the graph only to extract updated pointers (activations, KV cache views, RoPE caches), then call the update methods from Task 1. Also use the pooled GET_ROWS allocation. This is the highest-impact change — it transforms per-token plan building from O(full_rebuild) to O(pointer_updates).

**Acceptance Criteria:**

- [ ] First TG token: full plan build (existing code path)
- [ ] Subsequent TG tokens: fast update path using cached plan
- [ ] Correctness: `llama-completion` output matches master
- [ ] Performance: TG128 significantly faster than 1.77 tok/s baseline
- [ ] PP transitions: plan cache invalidated when switching from TG to PP

**Implementation Guide:**

1. **Add fast update path at the top of `extract_persistent_plan`** (after the debug init from Task 2, before model dimension extraction):

```cpp
    // Fast path: reuse cached plan, only update mutable pointers
    if (kernel.has_cached_plan()) {
        kernel.begin_plan_update();

        int op_idx = 0;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];

            // Skip view/reshape ops (same as full path)
            if (!node || node->op == GGML_OP_NONE ||
                node->op == GGML_OP_RESHAPE || node->op == GGML_OP_VIEW ||
                node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_PERMUTE) {
                continue;
            }

            switch (node->op) {
                case GGML_OP_GET_ROWS: {
                    // Pre-execute GET_ROWS (same as full path)
                    ggml_sycl_op_get_rows(ctx, node);
                    q->wait_and_throw();
                    // Use pooled stable copy
                    const size_t bytes = ggml_nbytes(node);
                    void * stable_ptr = kernel.get_rows_stable_ptr(bytes);
                    if (stable_ptr) {
                        const void * out_ptr = get_tensor_ptr_view_fast(node);
                        if (out_ptr) {
                            q->memcpy(stable_ptr, out_ptr, bytes).wait();
                        }
                        materialized_ptrs[node] = stable_ptr;
                    }
                    break;
                }
                case GGML_OP_RMS_NORM: {
                    const void * input_ptr = resolve_input_ptr(node->src[0], 0);
                    void * output_ptr = get_tensor_ptr_view_fast(node);
                    const void * weights_ptr = get_tensor_ptr_fast(node->src[1]);
                    kernel.update_op_pointers(op_idx++, input_ptr, output_ptr);
                    materialized_ptrs[node] = output_ptr;
                    break;
                }
                case GGML_OP_MUL_MAT: {
                    const void * input_ptr = resolve_input_ptr(node->src[1], 0);
                    void * output_ptr = get_tensor_ptr_view_fast(node);
                    // weights don't change, just update input/output
                    kernel.update_op_pointers(op_idx++, input_ptr, output_ptr);
                    materialized_ptrs[node] = output_ptr;
                    break;
                }
                case GGML_OP_ADD:
                case GGML_OP_MUL: {
                    const void * src0_ptr = resolve_input_ptr(node->src[0], 0);
                    const void * src1_ptr = resolve_input_ptr(node->src[1], 0);
                    void * output_ptr = get_tensor_ptr_view_fast(node);
                    kernel.update_op_pointers(op_idx++, src0_ptr, output_ptr, src1_ptr);
                    materialized_ptrs[node] = output_ptr;
                    break;
                }
                case GGML_OP_SILU:
                case GGML_OP_MUL: {
                    // SILU_MUL fused — skip, handled with gate matmul
                    break;
                }
                // TODO: Handle ATTENTION, ROPE, SOFTMAX, SET_ROWS updates
                // For now, fall back to full rebuild for these
                default: {
                    // Unsupported op in fast path — invalidate cache and rebuild
                    kernel.invalidate_plan_cache();
                    kernel.cancel_persistent();
                    goto full_build;
                }
            }
        }

        kernel.finish_plan_update();
        return true;
    }

full_build:
    // Original full plan build code follows...
```

**IMPORTANT**: The fast update path above is a **starting skeleton**. The full implementation requires carefully matching the operation index (`op_idx`) to the cached plan's operation order. The operation sequence in the cached plan includes fused operations (MATMUL_GATE_UP_SILU), so the graph walk must account for fusion.

**A safer initial approach**: For the first iteration, only use the cache for pointer updates on simple ops (RMS_NORM, ADD, MUL) and fall back to full rebuild for any complex op (ATTENTION, ROPE, MATMUL). Measure the improvement. Then expand the cache coverage incrementally.

2. **Invalidate cache on PP transitions** in `ggml_backend_sycl_graph_compute` (around line 28431):

After the persistent TG block (around the `normal_dispatch:` label), add cache invalidation when switching away from persistent TG:

```cpp
normal_dispatch:
    // Invalidate persistent plan cache when falling through to normal dispatch
    // (e.g. PP phase after TG, or unsupported graph)
    if (sycl_ctx->unified_kernel && sycl_ctx->unified_kernel->has_cached_plan()) {
        sycl_ctx->unified_kernel->invalidate_plan_cache();
    }
```

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: integrate plan caching in extract_persistent_plan"
```

**Notes for implementer:**
- The `get_tensor_ptr_fast`, `get_tensor_ptr_view_fast`, `resolve_input_ptr` lambdas are defined earlier in `extract_persistent_plan` — they must remain available in the fast path. This means the fast path should be placed AFTER these lambda definitions (around line 25860).
- The `materialized_ptrs` map and `q` (sycl::queue) must also be available.
- Operation fusion (MATMUL_GATE + MATMUL_UP + SILU_MUL → MATMUL_GATE_UP_SILU) happens in `launch_persistent_kernel`, not during plan building. So the cached ops already have pre-fusion types.
- The fast path MUST maintain the same op_idx ordering as the full build. If there's any mismatch, correctness will break. Start conservative and expand.

---

## Verification

1. **Build**: `source /opt/intel/oneapi/setvars.sh --force && ninja -C build -j $(nproc)`
2. **Correctness**: `ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0`
3. **TG benchmark**: `ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 0 -n 128` — target: significantly faster than 1.77 tok/s
4. **PP benchmark**: `ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 0` — must stay ~1300 tok/s
5. **Disable test**: `GGML_SYCL_PERSISTENT_TG=0` should use legacy dispatch correctly
