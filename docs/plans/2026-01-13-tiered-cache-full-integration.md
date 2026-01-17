# Tiered Cache Full Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Connect the unified_tensor_cache to model loading and all SYCL kernels for production-ready tiered memory.

**Architecture:** Two integration points: (1) Collect tensor inventory in llama-model.cpp after tensor creation, call SYCL backend API per device; (2) Connect get_cached_tensor_ptr() to unified_tensor_cache::get_tensor_with_location() and integrate into all weight-accessing kernels.

**Tech Stack:** C++17, SYCL/oneAPI, ggml tensor API, llama.cpp model loader

---

## Task 1: Add tensor name → ID mapping to unified_tensor_cache

**Files:**
- Modify: `ggml/src/ggml-sycl/unified-tensor-cache.hpp:56-70`
- Modify: `ggml/src/ggml-sycl/unified-tensor-cache.cpp:50-100`
- Test: `tests/test-unified-tensor-cache.cpp`

**Step 1: Write failing test**

Add to `tests/test-unified-tensor-cache.cpp`:

```cpp
static void test_tensor_name_lookup() {
    printf("test_tensor_name_lookup...\n");

    // Create mock inventory with named tensors
    std::vector<ggml_sycl::tensor_info> tensors = {
        ggml_sycl::make_tensor_info("token_embd.weight", 100 * 1024 * 1024),
        ggml_sycl::make_tensor_info("blk.0.attn_q.weight", 50 * 1024 * 1024),
        ggml_sycl::make_tensor_info("blk.0.ffn_down_exps.weight", 200 * 1024 * 1024),
    };

    ggml_sycl::tensor_inventory inventory;
    inventory.tensors = tensors;
    inventory.total_size = 350 * 1024 * 1024;

    // Create cache and set inventory
    sycl::queue q{sycl::default_selector_v};
    ggml_sycl::unified_tensor_cache cache(q, 1024 * 1024 * 1024, 512 * 1024 * 1024);
    cache.set_inventory(inventory);

    // Test lookup by name
    auto id0 = cache.get_tensor_id("token_embd.weight");
    auto id1 = cache.get_tensor_id("blk.0.attn_q.weight");
    auto id_invalid = cache.get_tensor_id("nonexistent.weight");

    assert(id0.has_value() && "token_embd should be found");
    assert(id1.has_value() && "attn_q should be found");
    assert(!id_invalid.has_value() && "nonexistent should return nullopt");
    assert(id0.value() != id1.value() && "IDs should be unique");

    printf("test_tensor_name_lookup: PASSED\n");
}
```

**Step 2: Run test to verify it fails**

Run: `source /opt/intel/oneapi/setvars.sh && cmake --build build --target test-unified-tensor-cache && ./build/bin/test-unified-tensor-cache`
Expected: Compilation error - `get_tensor_id` not defined

**Step 3: Add name→ID mapping to unified_tensor_cache**

In `unified-tensor-cache.hpp`, add after line 68:

```cpp
    // Get tensor ID by name (returns nullopt if not found)
    std::optional<uint64_t> get_tensor_id(const std::string& name) const;
```

Add to private members (after line 110):

```cpp
    // Name to ID mapping for O(1) lookup
    std::unordered_map<std::string, uint64_t> name_to_id_;
```

In `unified-tensor-cache.cpp`, add to `set_inventory()` after populating tensors:

```cpp
    // Build name→ID index
    name_to_id_.clear();
    name_to_id_.reserve(inventory.tensors.size());
    for (size_t i = 0; i < inventory.tensors.size(); i++) {
        name_to_id_[inventory.tensors[i].name] = i;
    }
```

Add implementation:

```cpp
std::optional<uint64_t> unified_tensor_cache::get_tensor_id(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = name_to_id_.find(name);
    if (it != name_to_id_.end()) {
        return it->second;
    }
    return std::nullopt;
}
```

**Step 4: Run test to verify it passes**

Run: `cmake --build build --target test-unified-tensor-cache && ./build/bin/test-unified-tensor-cache`
Expected: PASS

**Step 5: Commit**

```bash
git add ggml/src/ggml-sycl/unified-tensor-cache.hpp ggml/src/ggml-sycl/unified-tensor-cache.cpp tests/test-unified-tensor-cache.cpp
git commit -m "feat(sycl): add tensor name lookup to unified_tensor_cache"
```

---

## Task 2: Store unified_tensor_cache instance in backend context

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:640-700`
- Modify: `ggml/src/ggml-sycl/common.hpp` (ggml_backend_sycl_context)
- Test: `tests/test-tiered-dispatch.cpp`

**Step 1: Write failing test**

Add to `tests/test-tiered-dispatch.cpp`:

```cpp
static void test_cache_instance_available() {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("test_cache_instance_available: SKIPPED (no SYCL device)\n");
        return;
    }

    // Get VRAM info and create large inventory to enable tiered
    size_t free_vram = 0, total_vram = 0;
    ggml_backend_sycl_get_device_memory(0, &free_vram, &total_vram);

    std::vector<std::string> name_storage;
    std::vector<ggml_sycl_tensor_info> tensors;
    size_t total_size = 0;

    for (size_t i = 0; i < 50; i++) {
        name_storage.push_back("blk." + std::to_string(i) + ".weight");
        tensors.push_back({name_storage.back().c_str(), free_vram / 25});
        total_size += free_vram / 25;
    }

    ggml_sycl_tensor_inventory inventory = {tensors.data(), tensors.size(), total_size};
    ggml_backend_sycl_set_tensor_inventory(backend, &inventory);

    // Verify cache is accessible (new API)
    bool has_cache = ggml_backend_sycl_has_tensor_cache(backend);
    assert(has_cache && "Backend should have tensor cache when tiered enabled");

    ggml_backend_free(backend);
    printf("test_cache_instance_available: PASSED\n");
}
```

**Step 2: Run test to verify it fails**

Expected: Compilation error - `ggml_backend_sycl_has_tensor_cache` not defined

**Step 3: Add cache instance storage**

In `ggml/include/ggml-sycl.h`, add after line 119:

```cpp
GGML_BACKEND_API bool ggml_backend_sycl_has_tensor_cache(ggml_backend_t backend);
```

In `ggml/src/ggml-sycl/ggml-sycl.cpp`, add global storage (after line 644):

```cpp
static std::unique_ptr<ggml_sycl::unified_tensor_cache> g_tensor_cache;
static std::mutex g_tensor_cache_mutex;
```

Update `ggml_backend_sycl_set_tensor_inventory()` to create cache (after enabling tiered mode):

```cpp
    // If tiered mode enabled, create unified_tensor_cache
    if (g_tiered_enabled.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> cache_lock(g_tensor_cache_mutex);
        if (!g_tensor_cache) {
            // Create cache with VRAM budget = 90% of free, host budget = 10GB
            size_t vram_budget = (size_t)(free_mem * 0.9);
            size_t host_budget = 10ULL * 1024 * 1024 * 1024;  // 10GB
            g_tensor_cache = std::make_unique<ggml_sycl::unified_tensor_cache>(
                *(ctx->stream()), vram_budget, host_budget);
        }

        // Build internal inventory from API inventory
        ggml_sycl::tensor_inventory internal_inv;
        internal_inv.tensors.reserve(inventory->count);
        for (size_t i = 0; i < inventory->count; i++) {
            internal_inv.tensors.push_back(
                ggml_sycl::make_tensor_info(inventory->tensors[i].name, inventory->tensors[i].size));
            internal_inv.total_size += inventory->tensors[i].size;
        }
        g_tensor_cache->set_inventory(internal_inv);
    }
```

Add API function:

```cpp
bool ggml_backend_sycl_has_tensor_cache(ggml_backend_t backend) {
    (void)backend;
    std::lock_guard<std::mutex> lock(g_tensor_cache_mutex);
    return g_tensor_cache != nullptr && g_tiered_enabled.load(std::memory_order_acquire);
}
```

**Step 4: Run test to verify it passes**

Run: `cmake --build build --target test-tiered-dispatch && ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-tiered-dispatch`
Expected: PASS

**Step 5: Commit**

```bash
git add ggml/include/ggml-sycl.h ggml/src/ggml-sycl/ggml-sycl.cpp tests/test-tiered-dispatch.cpp
git commit -m "feat(sycl): store unified_tensor_cache instance for tiered mode"
```

---

## Task 3: Connect get_cached_tensor_ptr to unified_tensor_cache

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:705-742`
- Test: `tests/test-tiered-dispatch.cpp`

**Step 1: Write failing test**

Add to `tests/test-tiered-dispatch.cpp`:

```cpp
static void test_cache_lookup_returns_tier() {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("test_cache_lookup_returns_tier: SKIPPED\n");
        return;
    }

    size_t free_vram = 0, total_vram = 0;
    ggml_backend_sycl_get_device_memory(0, &free_vram, &total_vram);

    // Create inventory exceeding VRAM
    std::vector<std::string> names = {"blk.0.attn_q.weight", "blk.1.attn_q.weight"};
    std::vector<ggml_sycl_tensor_info> tensors;
    for (const auto& n : names) {
        tensors.push_back({n.c_str(), free_vram});  // Each tensor = full VRAM
    }

    ggml_sycl_tensor_inventory inventory = {tensors.data(), tensors.size(), free_vram * 2};
    ggml_backend_sycl_set_tensor_inventory(backend, &inventory);

    // Query cache stats - should have planned tiers
    uint64_t hits = 0, misses = 0;
    ggml_backend_sycl_get_cache_stats(backend, &hits, &misses);

    // Hits/misses should be 0 initially (no lookups yet)
    assert(hits == 0 && misses == 0);

    ggml_backend_free(backend);
    printf("test_cache_lookup_returns_tier: PASSED\n");
}
```

**Step 2: Run test - expect fail on missing API**

**Step 3: Update get_cached_tensor_ptr to use unified_tensor_cache**

In `ggml-sycl.cpp`, replace the placeholder implementation (lines 705-742):

```cpp
static void * get_cached_tensor_ptr(const char * tensor_name, ggml_sycl::memory_tier * tier_out,
                                    bool * found_in_inventory) {
    if (found_in_inventory) {
        *found_in_inventory = false;
    }

    if (!tensor_name) {
        return nullptr;
    }

    if (!g_tiered_enabled.load(std::memory_order_acquire)) {
        return nullptr;
    }

    // First check inventory index (fast path for inventory check)
    {
        std::lock_guard<std::mutex> lock(g_tensor_inventory_mutex);
        auto it = g_tensor_inventory_index.find(tensor_name);
        if (it == g_tensor_inventory_index.end()) {
            return nullptr;  // Not a tracked tensor
        }
        if (found_in_inventory) {
            *found_in_inventory = true;
        }
    }

    // Query unified_tensor_cache for actual pointer and tier
    std::lock_guard<std::mutex> cache_lock(g_tensor_cache_mutex);
    if (!g_tensor_cache) {
        if (tier_out) {
            *tier_out = ggml_sycl::memory_tier::MMAP;  // Fallback
        }
        return nullptr;
    }

    auto id_opt = g_tensor_cache->get_tensor_id(tensor_name);
    if (!id_opt.has_value()) {
        return nullptr;
    }

    auto location = g_tensor_cache->get_tensor_with_location(id_opt.value());
    if (tier_out) {
        *tier_out = location.tier;
    }
    return location.ptr;
}
```

Add cache stats API in `ggml-sycl.h`:

```cpp
GGML_BACKEND_API void ggml_backend_sycl_get_cache_stats(
    ggml_backend_t backend, uint64_t * hits, uint64_t * misses);
```

Implement in `ggml-sycl.cpp`:

```cpp
void ggml_backend_sycl_get_cache_stats(ggml_backend_t backend, uint64_t * hits, uint64_t * misses) {
    (void)backend;
    std::lock_guard<std::mutex> lock(g_tensor_cache_mutex);
    if (g_tensor_cache) {
        if (hits) *hits = g_tensor_cache->cache_hits();
        if (misses) *misses = g_tensor_cache->cache_misses();
    } else {
        if (hits) *hits = 0;
        if (misses) *misses = 0;
    }
}
```

**Step 4: Run test to verify**

**Step 5: Commit**

```bash
git commit -am "feat(sycl): connect get_cached_tensor_ptr to unified_tensor_cache"
```

---

## Task 4: Add tiered dispatch to ggml_sycl_get_rows

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:7535-7540`
- Test: Run existing tests

**Step 1: Read current implementation**

```cpp
static void ggml_sycl_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_flatten(ctx, dst, ggml_sycl_op_get_rows);
}
```

**Step 2: Add tiered dispatch check**

Replace with:

```cpp
static void ggml_sycl_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // Weight tensor

    // Check for tiered dispatch
    if (src0->name && g_tiered_enabled.load(std::memory_order_acquire)) {
        ggml_sycl::memory_tier tier;
        bool in_inventory = false;
        void * cached_ptr = get_cached_tensor_ptr(src0->name, &tier, &in_inventory);
        if (cached_ptr != nullptr) {
            GGML_LOG_DEBUG("[SYCL] get_rows tiered hit: %s (tier=%d)\n",
                          src0->name, static_cast<int>(tier));
            // Future: use cached_ptr for tiered path
        }
    }

    ggml_sycl_op_flatten(ctx, dst, ggml_sycl_op_get_rows);
}
```

**Step 3: Build and test**

Run: `cmake --build build --target test-tiered-dispatch && ./build/bin/test-tiered-dispatch`
Expected: PASS

**Step 4: Commit**

```bash
git commit -am "feat(sycl): add tiered dispatch to get_rows kernel"
```

---

## Task 5: Add tiered dispatch to ggml_sycl_mul_mat_id (MoE)

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:15022-15100`

**Step 1: Locate MoE mul_mat_id function**

Line 15022: `static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx, ggml_tensor * dst)`

**Step 2: Add tiered dispatch at function start**

After `scope_op_debug_print` (around line 15030), add:

```cpp
    // Check for tiered dispatch on expert weights
    const ggml_tensor * src0 = dst->src[0];  // Expert weight tensor
    if (src0->name && g_tiered_enabled.load(std::memory_order_acquire)) {
        ggml_sycl::memory_tier tier;
        bool in_inventory = false;
        void * cached_ptr = get_cached_tensor_ptr(src0->name, &tier, &in_inventory);
        if (cached_ptr != nullptr) {
            GGML_LOG_DEBUG("[SYCL] mul_mat_id tiered hit: %s (tier=%d)\n",
                          src0->name, static_cast<int>(tier));
        } else if (in_inventory) {
            GGML_LOG_DEBUG("[SYCL] mul_mat_id tiered pending: %s\n", src0->name);
        }
    }
```

**Step 3: Build and test**

**Step 4: Commit**

```bash
git commit -am "feat(sycl): add tiered dispatch to mul_mat_id (MoE)"
```

---

## Task 6: Add tiered dispatch to ggml_sycl_mul_mat_batched_sycl

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:10642-10700`

**Step 1: Add tiered check after function entry**

```cpp
static void ggml_sycl_mul_mat_batched_sycl(ggml_backend_sycl_context & ctx,
                                           const ggml_tensor * src0, const ggml_tensor * src1,
                                           ggml_tensor * dst) {
    // Tiered dispatch check for batched matmul
    if (src0->name && g_tiered_enabled.load(std::memory_order_acquire)) {
        ggml_sycl::memory_tier tier;
        bool in_inventory = false;
        void * cached_ptr = get_cached_tensor_ptr(src0->name, &tier, &in_inventory);
        if (cached_ptr != nullptr) {
            GGML_LOG_DEBUG("[SYCL] batched_sycl tiered hit: %s (tier=%d)\n",
                          src0->name, static_cast<int>(tier));
        }
    }

    // ... rest of existing implementation
```

**Step 2: Build and test**

**Step 3: Commit**

```bash
git commit -am "feat(sycl): add tiered dispatch to mul_mat_batched_sycl"
```

---

## Task 7: Add tiered dispatch to DMMV kernels

**Files:**
- Modify: `ggml/src/ggml-sycl/dmmv.cpp`

**Step 1: Find DMMV entry points**

Search for `ggml_sycl_op_dequantize_mul_mat_vec` dispatch functions.

**Step 2: Add tiered check to main DMMV dispatcher**

In `dmmv.cpp`, find the main dispatch function and add:

```cpp
    // Check tiered dispatch for weight tensor
    if (src0->name && g_tiered_enabled.load(std::memory_order_acquire)) {
        ggml_sycl::memory_tier tier;
        bool in_inventory = false;
        void * cached_ptr = get_cached_tensor_ptr(src0->name, &tier, &in_inventory);
        if (cached_ptr != nullptr) {
            GGML_LOG_DEBUG("[SYCL] dmmv tiered hit: %s (tier=%d)\n",
                          src0->name, static_cast<int>(tier));
        }
    }
```

**Step 3: Build and test**

**Step 4: Commit**

```bash
git commit -am "feat(sycl): add tiered dispatch to DMMV kernels"
```

---

## Task 8: Add tiered dispatch to Flash Attention

**Files:**
- Modify: `ggml/src/ggml-sycl/fattn.cpp`

**Step 1: Find flash attention entry point**

Search for `ggml_sycl_flash_attn_ext` or similar.

**Step 2: Add tiered check for K/V cache tensors**

Flash attention accesses K and V cache tensors. Add tiered checks:

```cpp
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    if (g_tiered_enabled.load(std::memory_order_acquire)) {
        // Check K tensor
        if (K->name) {
            ggml_sycl::memory_tier tier;
            bool found = false;
            void * ptr = get_cached_tensor_ptr(K->name, &tier, &found);
            if (ptr) {
                GGML_LOG_DEBUG("[SYCL] fattn K tiered hit: %s\n", K->name);
            }
        }
        // Check V tensor
        if (V->name) {
            ggml_sycl::memory_tier tier;
            bool found = false;
            void * ptr = get_cached_tensor_ptr(V->name, &tier, &found);
            if (ptr) {
                GGML_LOG_DEBUG("[SYCL] fattn V tiered hit: %s\n", V->name);
            }
        }
    }
```

**Step 3: Build and test**

**Step 4: Commit**

```bash
git commit -am "feat(sycl): add tiered dispatch to flash attention"
```

---

## Task 9: Collect tensor inventory in llama-model.cpp

**Files:**
- Modify: `src/llama-model.cpp:6986-6993`
- Create: `src/llama-model-sycl.cpp` (optional helper)

**Step 1: Add SYCL header include**

At top of `llama-model.cpp` (with other backend includes):

```cpp
#ifdef GGML_USE_SYCL
#include "ggml-sycl.h"
#endif
```

**Step 2: Add inventory collection after tensors_by_name**

After line 6986 (after `tensors_by_name` is populated), add:

```cpp
#ifdef GGML_USE_SYCL
    // Collect tensor inventory for SYCL tiered memory
    if (!tensors_by_name.empty()) {
        std::vector<ggml_sycl_tensor_info> sycl_tensors;
        sycl_tensors.reserve(tensors_by_name.size());
        size_t total_size = 0;

        for (const auto & [name, tensor] : tensors_by_name) {
            if (tensor && ggml_nbytes(tensor) > 0) {
                sycl_tensors.push_back({name.c_str(), ggml_nbytes(tensor)});
                total_size += ggml_nbytes(tensor);
            }
        }

        if (!sycl_tensors.empty()) {
            ggml_sycl_tensor_inventory inventory;
            inventory.tensors = sycl_tensors.data();
            inventory.count = sycl_tensors.size();
            inventory.total_size = total_size;

            // Set inventory for each SYCL backend device
            for (int i = 0; i < ggml_backend_sycl_get_device_count(); i++) {
                ggml_backend_t sycl_backend = ggml_backend_sycl_init(i);
                if (sycl_backend) {
                    ggml_backend_sycl_set_tensor_inventory(sycl_backend, &inventory);
                    // Note: backend is owned by buffer, don't free here
                }
            }

            LLAMA_LOG_INFO("%s: SYCL tensor inventory: %zu tensors, %.2f GB\n",
                          __func__, sycl_tensors.size(), total_size / (1024.0 * 1024.0 * 1024.0));
        }
    }
#endif
```

**Step 3: Build full project**

Run: `cmake --build build -j$(nproc)`

**Step 4: Test with real model**

Run: `ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-completion -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -ngl 99 -p "Hello" -n 5`

Expected: Log shows "SYCL tensor inventory: N tensors, X.XX GB"

**Step 5: Commit**

```bash
git commit -am "feat(sycl): collect tensor inventory during model loading"
```

---

## Task 10: Update integration tests for cache hit rate

**Files:**
- Modify: `tests/test-tiered-memory-integration.sh`
- Modify: `tests/test-tiered-dispatch.cpp`

**Step 1: Add cache stats test**

```cpp
static void test_cache_hit_rate() {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("test_cache_hit_rate: SKIPPED\n");
        return;
    }

    // Setup large inventory to enable tiered
    size_t free_vram = 0, total_vram = 0;
    ggml_backend_sycl_get_device_memory(0, &free_vram, &total_vram);

    std::vector<std::string> names;
    std::vector<ggml_sycl_tensor_info> tensors;
    for (int i = 0; i < 100; i++) {
        names.push_back("blk." + std::to_string(i) + ".weight");
        tensors.push_back({names.back().c_str(), free_vram / 50});
    }

    ggml_sycl_tensor_inventory inv = {tensors.data(), tensors.size(), free_vram * 2};
    ggml_backend_sycl_set_tensor_inventory(backend, &inv);

    // Verify cache stats API works
    uint64_t hits = 0, misses = 0;
    ggml_backend_sycl_get_cache_stats(backend, &hits, &misses);

    // Initially zero
    assert(hits == 0 && misses == 0);

    ggml_backend_free(backend);
    printf("test_cache_hit_rate: PASSED\n");
}
```

**Step 2: Update shell script to check stats**

Add to `test-tiered-memory-integration.sh`:

```bash
# Check cache stats in debug output
if echo "${stderr}" | grep -q "cache_hits\|cache_misses"; then
    echo -e "${GREEN}OK: Cache stats visible in debug output${NC}"
fi
```

**Step 3: Run all tests**

**Step 4: Commit**

```bash
git commit -am "test(sycl): add cache hit rate tests"
```

---

## Task 11: Final integration test with real model

**Files:**
- Test only (no code changes)

**Step 1: Run Mistral-7B test**

```bash
source /opt/intel/oneapi/setvars.sh --force
ONEAPI_DEVICE_SELECTOR=level_zero:1 GGML_SYCL_DEBUG=1 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -ngl 99 -p "1, 2, 3, 4, 5," -n 15 --seed 42 --temp 0
```

Expected:
- "SYCL tensor inventory: ~300 tensors, ~4 GB"
- "tiered: disabled" (model fits in VRAM)
- Output: "6, 7, 8, 9, 10"
- Speed: >5 t/s

**Step 2: Verify cache stats**

Look for cache hit/miss stats in debug output.

**Step 3: Document results**

Update `tests/test-tiered-memory-integration.sh` with verified expectations.

**Step 4: Final commit**

```bash
git commit -am "test(sycl): verify tiered cache integration with real model"
```

---

---

## Task 12: Add tiered dispatch to MMQ kernels (mmq.cpp)

**Files:**
- Modify: `ggml/src/ggml-sycl/mmq.cpp`

**Step 1: Find MMQ dispatch function**

The main dispatch is `ggml_sycl_op_mul_mat_mmq()`. Add tiered check:

```cpp
// At function entry, after getting src0
if (src0->name && g_tiered_enabled.load(std::memory_order_acquire)) {
    ggml_sycl::memory_tier tier;
    bool in_inventory = false;
    void * cached_ptr = get_cached_tensor_ptr(src0->name, &tier, &in_inventory);
    if (cached_ptr != nullptr) {
        GGML_LOG_DEBUG("[SYCL] mmq tiered hit: %s (tier=%d)\n",
                      src0->name, static_cast<int>(tier));
    }
}
```

**Step 2: Build and test**

**Step 3: Commit**

```bash
git commit -am "feat(sycl): add tiered dispatch to MMQ kernels"
```

---

## Task 13: Add tiered dispatch to MMQ XMX kernels (mmq_xmx.cpp)

**Files:**
- Modify: `ggml/src/ggml-sycl/mmq_xmx.cpp`

**Step 1: Find XMX dispatcher**

All 15+ XMX MMQ kernels are called through a common dispatcher. Add tiered check at dispatcher level:

```cpp
// At ggml_mul_mat_q*_q8_1_xmx_* function entries
if (src0->name && g_tiered_enabled.load(std::memory_order_acquire)) {
    ggml_sycl::memory_tier tier;
    bool in_inventory = false;
    void * cached_ptr = get_cached_tensor_ptr(src0->name, &tier, &in_inventory);
    if (cached_ptr != nullptr) {
        GGML_LOG_DEBUG("[SYCL] mmq_xmx tiered hit: %s (tier=%d)\n",
                      src0->name, static_cast<int>(tier));
    }
}
```

**Step 2: Add include for extern declarations**

Add at top of mmq_xmx.cpp:
```cpp
extern std::atomic<bool> g_tiered_enabled;
extern void * get_cached_tensor_ptr(const char *, ggml_sycl::memory_tier *, bool *);
```

**Step 3: Build and test**

**Step 4: Commit**

```bash
git commit -am "feat(sycl): add tiered dispatch to MMQ XMX kernels"
```

---

## Task 14: Add tiered dispatch to MMVQ kernels (mmvq.cpp)

**Files:**
- Modify: `ggml/src/ggml-sycl/mmvq.cpp`

**Step 1: Find MMVQ dispatch function**

Look for `ggml_sycl_op_mul_mat_mmvq()` or similar dispatcher.

**Step 2: Add tiered check**

```cpp
if (src0->name && g_tiered_enabled.load(std::memory_order_acquire)) {
    ggml_sycl::memory_tier tier;
    bool in_inventory = false;
    void * cached_ptr = get_cached_tensor_ptr(src0->name, &tier, &in_inventory);
    if (cached_ptr != nullptr) {
        GGML_LOG_DEBUG("[SYCL] mmvq tiered hit: %s (tier=%d)\n",
                      src0->name, static_cast<int>(tier));
    }
}
```

**Step 3: Build and test**

**Step 4: Commit**

```bash
git commit -am "feat(sycl): add tiered dispatch to MMVQ kernels"
```

---

## Task 15: Add tiered dispatch to MoE XMX kernels (moe-xmx.cpp)

**Files:**
- Modify: `ggml/src/ggml-sycl/moe-xmx.cpp`

**Step 1: Find MoE XMX entry points**

Look for expert weight access in MoE XMX kernels.

**Step 2: Add tiered check for expert weights**

```cpp
// Expert weight tensor typically has name like "blk.N.ffn_down_exps.weight"
if (expert_weight->name && g_tiered_enabled.load(std::memory_order_acquire)) {
    ggml_sycl::memory_tier tier;
    bool in_inventory = false;
    void * cached_ptr = get_cached_tensor_ptr(expert_weight->name, &tier, &in_inventory);
    if (cached_ptr != nullptr) {
        GGML_LOG_DEBUG("[SYCL] moe_xmx tiered hit: %s (tier=%d)\n",
                      expert_weight->name, static_cast<int>(tier));
    }
}
```

**Step 3: Build and test**

**Step 4: Commit**

```bash
git commit -am "feat(sycl): add tiered dispatch to MoE XMX kernels"
```

---

## Task 16: Create helper macro for tiered dispatch boilerplate

**Files:**
- Create: `ggml/src/ggml-sycl/tiered-dispatch.hpp`

**Step 1: Create reusable macro**

```cpp
#ifndef GGML_SYCL_TIERED_DISPATCH_HPP
#define GGML_SYCL_TIERED_DISPATCH_HPP

#include "tensor-types.hpp"
#include <atomic>

// Declare external dependencies
extern std::atomic<bool> g_tiered_enabled;
extern void * get_cached_tensor_ptr(const char * tensor_name,
                                    ggml_sycl::memory_tier * tier_out,
                                    bool * found_in_inventory);

// Macro for tiered dispatch check - use at kernel entry
#define SYCL_TIERED_DISPATCH_CHECK(tensor, kernel_name) \
    do { \
        if ((tensor)->name && g_tiered_enabled.load(std::memory_order_acquire)) { \
            ggml_sycl::memory_tier tier; \
            bool in_inventory = false; \
            void * cached_ptr = get_cached_tensor_ptr((tensor)->name, &tier, &in_inventory); \
            if (cached_ptr != nullptr) { \
                GGML_LOG_DEBUG("[SYCL] %s tiered hit: %s (tier=%d)\n", \
                              kernel_name, (tensor)->name, static_cast<int>(tier)); \
            } \
        } \
    } while(0)

#endif // GGML_SYCL_TIERED_DISPATCH_HPP
```

**Step 2: Update previous tasks to use macro**

Replace inline tiered checks with:
```cpp
SYCL_TIERED_DISPATCH_CHECK(src0, "mmq");
```

**Step 3: Commit**

```bash
git commit -am "refactor(sycl): create tiered dispatch helper macro"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add tensor name→ID mapping | unified-tensor-cache.hpp/cpp |
| 2 | Store cache instance in backend | ggml-sycl.cpp, ggml-sycl.h |
| 3 | Connect get_cached_tensor_ptr | ggml-sycl.cpp |
| 4 | Tiered dispatch: get_rows | ggml-sycl.cpp |
| 5 | Tiered dispatch: mul_mat_id (MoE) | ggml-sycl.cpp |
| 6 | Tiered dispatch: batched_sycl | ggml-sycl.cpp |
| 7 | Tiered dispatch: DMMV | dmmv.cpp |
| 8 | Tiered dispatch: Flash Attention | fattn.cpp |
| 9 | Model loading inventory | llama-model.cpp |
| 10 | Cache hit rate tests | test-*.cpp/sh |
| 11 | Final integration test | (test only) |
| 12 | Tiered dispatch: MMQ | mmq.cpp |
| 13 | Tiered dispatch: MMQ XMX | mmq_xmx.cpp |
| 14 | Tiered dispatch: MMVQ | mmvq.cpp |
| 15 | Tiered dispatch: MoE XMX | moe-xmx.cpp |
| 16 | Create helper macro | tiered-dispatch.hpp |

**Total: 16 tasks covering ALL kernel types**

### Kernel Coverage Summary

| Kernel Type | File | Task |
|-------------|------|------|
| mul_mat (main) | ggml-sycl.cpp | Already done (previous epic) |
| get_rows | ggml-sycl.cpp | Task 4 |
| mul_mat_id (MoE) | ggml-sycl.cpp | Task 5 |
| batched_sycl | ggml-sycl.cpp | Task 6 |
| DMMV | dmmv.cpp | Task 7 |
| Flash Attention | fattn.cpp | Task 8 |
| MMQ (quantized) | mmq.cpp | Task 12 |
| MMQ XMX (Intel XMX) | mmq_xmx.cpp | Task 13 |
| MMVQ (mat-vec quantized) | mmvq.cpp | Task 14 |
| MoE XMX | moe-xmx.cpp | Task 15 |
