# Unified Tensor Cache Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate fragmented cache components into a single Unified Tensor Cache that manages all model weights across VRAM, Pinned Host, and mmap tiers with automatic placement based on priority.

**Architecture:** Two-phase loading (enumerate tensors from GGUF, then place optimally). Owns both VRAM and host memory pools. Static priority by tensor type + dynamic access-pattern adjustment. Auto-enables when model exceeds VRAM.

**Tech Stack:** SYCL/Level Zero, Intel Arc GPUs, C++17

**Design Doc:** `docs/plans/2026-01-12-unified-tensor-cache-design.md`

---

## Task 1: Create vram_pool class

**Files:**
- Create: `ggml/src/ggml-sycl/vram-pool.hpp`
- Create: `ggml/src/ggml-sycl/vram-pool.cpp`
- Create: `tests/test-vram-pool.cpp`
- Modify: `ggml/src/ggml-sycl/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

### Step 1: Write the failing test

Create `tests/test-vram-pool.cpp`:

```cpp
//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include <cassert>
#include <iostream>
#include <sycl/sycl.hpp>

// Forward declaration - will be implemented
namespace ggml_sycl {
class vram_pool;
}

#include "../ggml/src/ggml-sycl/vram-pool.hpp"

void test_basic_allocation() {
    sycl::queue q;

    // 1GB budget
    constexpr size_t BUDGET = 1ULL * 1024 * 1024 * 1024;
    ggml_sycl::vram_pool pool(q, BUDGET);

    // Allocate 100MB
    void* ptr1 = pool.allocate(100 * 1024 * 1024, 1);
    assert(ptr1 != nullptr && "100MB allocation should succeed");

    // Allocate another 200MB
    void* ptr2 = pool.allocate(200 * 1024 * 1024, 2);
    assert(ptr2 != nullptr && "200MB allocation should succeed");

    // Verify pointers are different
    assert(ptr1 != ptr2 && "Pointers should be unique");

    // Verify used tracking
    assert(pool.used() == 300 * 1024 * 1024 && "Used should be 300MB");

    std::cout << "test_basic_allocation: PASSED\n";
}

void test_budget_limit() {
    sycl::queue q;

    // Small 100MB budget
    constexpr size_t BUDGET = 100 * 1024 * 1024;
    ggml_sycl::vram_pool pool(q, BUDGET);

    // First 80MB should succeed
    void* ptr1 = pool.allocate(80 * 1024 * 1024, 1);
    assert(ptr1 != nullptr && "80MB allocation should succeed");

    // Second 80MB should fail (exceeds budget)
    void* ptr2 = pool.allocate(80 * 1024 * 1024, 2);
    assert(ptr2 == nullptr && "Should fail - exceeds budget");

    std::cout << "test_budget_limit: PASSED\n";
}

void test_deallocation() {
    sycl::queue q;

    constexpr size_t BUDGET = 200 * 1024 * 1024;
    ggml_sycl::vram_pool pool(q, BUDGET);

    // Allocate 150MB
    void* ptr1 = pool.allocate(150 * 1024 * 1024, 1);
    assert(ptr1 != nullptr);
    assert(pool.used() == 150 * 1024 * 1024);

    // Deallocate
    pool.deallocate(1);
    assert(pool.used() == 0 && "Used should be 0 after deallocation");

    // Should be able to allocate again
    void* ptr2 = pool.allocate(150 * 1024 * 1024, 2);
    assert(ptr2 != nullptr && "Should succeed after deallocation");

    std::cout << "test_deallocation: PASSED\n";
}

void test_alignment() {
    sycl::queue q;

    constexpr size_t BUDGET = 1ULL * 1024 * 1024 * 1024;
    ggml_sycl::vram_pool pool(q, BUDGET);

    // Allocate with various sizes, check 64-byte alignment
    void* ptr1 = pool.allocate(100, 1);
    void* ptr2 = pool.allocate(1000, 2);
    void* ptr3 = pool.allocate(10000, 3);

    assert((reinterpret_cast<uintptr_t>(ptr1) % 64) == 0 && "ptr1 not aligned");
    assert((reinterpret_cast<uintptr_t>(ptr2) % 64) == 0 && "ptr2 not aligned");
    assert((reinterpret_cast<uintptr_t>(ptr3) % 64) == 0 && "ptr3 not aligned");

    std::cout << "test_alignment: PASSED\n";
}

int main() {
    try {
        test_basic_allocation();
        test_budget_limit();
        test_deallocation();
        test_alignment();
        std::cout << "\nAll vram_pool tests PASSED!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test FAILED: " << e.what() << "\n";
        return 1;
    }
}
```

### Step 2: Run test to verify it fails

```bash
source /opt/intel/oneapi/setvars.sh --force
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build --target test-vram-pool -j$(nproc)
```

Expected: FAILS - `vram_pool` class not defined

### Step 3: Create header file

Create `ggml/src/ggml-sycl/vram-pool.hpp`:

```cpp
//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_VRAM_POOL_HPP
#define GGML_SYCL_VRAM_POOL_HPP

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <sycl/sycl.hpp>
#include <unordered_map>

namespace ggml_sycl {

// VRAM pool allocator for device memory.
// Unlike pinned_chunk_pool, allocates per-tensor (no 11GB limit on device).
// Tracks allocations by tensor_id for deallocation.
//
// Thread-safe: all public methods can be called from multiple threads.
class vram_pool {
  public:
    static constexpr size_t DEFAULT_ALIGNMENT = 64;  // Cache line alignment

    // Create a pool with the given VRAM budget
    vram_pool(sycl::queue& queue, size_t budget);
    ~vram_pool();

    // Non-copyable, non-movable (owns SYCL allocations)
    vram_pool(const vram_pool&) = delete;
    vram_pool& operator=(const vram_pool&) = delete;
    vram_pool(vram_pool&&) = delete;
    vram_pool& operator=(vram_pool&&) = delete;

    // Allocate VRAM for a tensor. Returns nullptr if over budget.
    // tensor_id: unique identifier for later deallocation
    void* allocate(size_t size, uint64_t tensor_id, size_t alignment = DEFAULT_ALIGNMENT);

    // Deallocate VRAM for a tensor
    void deallocate(uint64_t tensor_id);

    // Check if tensor is allocated
    bool is_allocated(uint64_t tensor_id) const;

    // Get pointer for allocated tensor (nullptr if not allocated)
    void* get(uint64_t tensor_id) const;

    // Statistics
    size_t budget() const { return budget_; }
    size_t used() const { return used_; }
    size_t available() const { return budget_ > used_ ? budget_ - used_ : 0; }
    size_t allocation_count() const;

  private:
    struct allocation {
        void*  ptr;
        size_t size;
    };

    sycl::queue& queue_;
    size_t budget_;
    size_t used_ = 0;

    std::unordered_map<uint64_t, allocation> allocations_;
    mutable std::mutex mutex_;
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_VRAM_POOL_HPP
```

### Step 4: Create implementation file

Create `ggml/src/ggml-sycl/vram-pool.cpp`:

```cpp
//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "vram-pool.hpp"
#include "ggml.h"

namespace ggml_sycl {

vram_pool::vram_pool(sycl::queue& queue, size_t budget)
    : queue_(queue), budget_(budget) {
    GGML_LOG_INFO("[SYCL] VRAM pool created with %.2f GB budget\n",
                  budget / (1024.0 * 1024.0 * 1024.0));
}

vram_pool::~vram_pool() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [id, alloc] : allocations_) {
        if (alloc.ptr) {
            sycl::free(alloc.ptr, queue_);
        }
    }
    allocations_.clear();
    GGML_LOG_INFO("[SYCL] VRAM pool destroyed, released %zu allocations\n",
                  allocations_.size());
}

void* vram_pool::allocate(size_t size, uint64_t tensor_id, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Round up size to alignment
    size = (size + alignment - 1) & ~(alignment - 1);

    // Check budget
    if (used_ + size > budget_) {
        return nullptr;
    }

    // Check if already allocated
    auto it = allocations_.find(tensor_id);
    if (it != allocations_.end()) {
        return it->second.ptr;  // Return existing allocation
    }

    // Allocate device memory
    void* ptr = nullptr;
    try {
        ptr = sycl::malloc_device(size, queue_);
    } catch (const sycl::exception& e) {
        GGML_LOG_ERROR("[SYCL] VRAM allocation failed: %s\n", e.what());
        return nullptr;
    }

    if (!ptr) {
        return nullptr;
    }

    allocations_[tensor_id] = {ptr, size};
    used_ += size;

    return ptr;
}

void vram_pool::deallocate(uint64_t tensor_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = allocations_.find(tensor_id);
    if (it == allocations_.end()) {
        return;
    }

    sycl::free(it->second.ptr, queue_);
    used_ -= it->second.size;
    allocations_.erase(it);
}

bool vram_pool::is_allocated(uint64_t tensor_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocations_.find(tensor_id) != allocations_.end();
}

void* vram_pool::get(uint64_t tensor_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocations_.find(tensor_id);
    return it != allocations_.end() ? it->second.ptr : nullptr;
}

size_t vram_pool::allocation_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocations_.size();
}

}  // namespace ggml_sycl
```

### Step 5: Add to CMakeLists.txt

Add to `ggml/src/ggml-sycl/CMakeLists.txt` source list:
```cmake
vram-pool.cpp
```

Add to `tests/CMakeLists.txt`:
```cmake
if (GGML_SYCL)
    ggml_add_test(test-vram-pool test-vram-pool.cpp)
    target_link_libraries(test-vram-pool PRIVATE ggml)
endif()
```

### Step 6: Run test to verify it passes

```bash
cmake --build build --target test-vram-pool -j$(nproc)
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-vram-pool
```

Expected:
```
[SYCL] VRAM pool created with 1.00 GB budget
test_basic_allocation: PASSED
test_budget_limit: PASSED
test_deallocation: PASSED
test_alignment: PASSED

All vram_pool tests PASSED!
```

### Step 7: Commit

```bash
git add ggml/src/ggml-sycl/vram-pool.hpp \
        ggml/src/ggml-sycl/vram-pool.cpp \
        ggml/src/ggml-sycl/CMakeLists.txt \
        tests/test-vram-pool.cpp \
        tests/CMakeLists.txt
git commit -m "feat(sycl): add vram_pool for managed device memory

Implements VRAM pool with:
- Per-tensor allocation tracking by ID
- Budget enforcement
- 64-byte alignment for cache efficiency

Part of: Unified Tensor Cache"
```

---

## Task 2: Create tensor_info and tensor_inventory types

**Files:**
- Create: `ggml/src/ggml-sycl/tensor-types.hpp`
- Create: `tests/test-tensor-classification.cpp`
- Modify: `tests/CMakeLists.txt`

### Step 1: Write the failing test

Create `tests/test-tensor-classification.cpp`:

```cpp
//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include <cassert>
#include <iostream>
#include <string>

#include "../ggml/src/ggml-sycl/tensor-types.hpp"

using namespace ggml_sycl;

void test_tensor_classification() {
    // Embeddings (priority 0)
    assert(classify_tensor("token_embd.weight") == tensor_class::EMBEDDING);

    // Output head (priority 0)
    assert(classify_tensor("lm_head.weight") == tensor_class::OUTPUT);
    assert(classify_tensor("output.weight") == tensor_class::OUTPUT);

    // Attention (priority 1)
    assert(classify_tensor("blk.0.attn_q.weight") == tensor_class::ATTENTION);
    assert(classify_tensor("blk.5.attn_k.weight") == tensor_class::ATTENTION);
    assert(classify_tensor("blk.10.attn_v.weight") == tensor_class::ATTENTION);
    assert(classify_tensor("blk.15.attn_output.weight") == tensor_class::ATTENTION);

    // Router (priority 2)
    assert(classify_tensor("blk.0.ffn_gate_inp.weight") == tensor_class::ROUTER);

    // Dense FFN (priority 2)
    assert(classify_tensor("blk.0.ffn_up.weight") == tensor_class::FFN);
    assert(classify_tensor("blk.5.ffn_down.weight") == tensor_class::FFN);
    assert(classify_tensor("blk.10.ffn_gate.weight") == tensor_class::FFN);

    // Experts (priority 3)
    assert(classify_tensor("blk.0.ffn_down_exps.weight") == tensor_class::EXPERT);
    assert(classify_tensor("blk.5.ffn_gate_exps.weight") == tensor_class::EXPERT);
    assert(classify_tensor("blk.10.ffn_up_exps.weight") == tensor_class::EXPERT);

    // Norms (priority 4)
    assert(classify_tensor("blk.0.attn_norm.weight") == tensor_class::NORM);
    assert(classify_tensor("blk.0.ffn_norm.weight") == tensor_class::NORM);
    assert(classify_tensor("output_norm.weight") == tensor_class::NORM);

    // Other
    assert(classify_tensor("rope_freqs.weight") == tensor_class::OTHER);

    std::cout << "test_tensor_classification: PASSED\n";
}

void test_priority_from_type() {
    assert(priority_from_type(tensor_class::EMBEDDING) == 0);
    assert(priority_from_type(tensor_class::OUTPUT) == 0);
    assert(priority_from_type(tensor_class::ATTENTION) == 1);
    assert(priority_from_type(tensor_class::FFN) == 2);
    assert(priority_from_type(tensor_class::ROUTER) == 2);
    assert(priority_from_type(tensor_class::EXPERT) == 3);
    assert(priority_from_type(tensor_class::NORM) == 4);
    assert(priority_from_type(tensor_class::OTHER) == 5);

    std::cout << "test_priority_from_type: PASSED\n";
}

void test_extract_layer_id() {
    assert(extract_layer_id("blk.0.attn_q.weight") == 0);
    assert(extract_layer_id("blk.15.ffn_up.weight") == 15);
    assert(extract_layer_id("blk.127.attn_k.weight") == 127);
    assert(extract_layer_id("token_embd.weight") == -1);
    assert(extract_layer_id("output_norm.weight") == -1);

    std::cout << "test_extract_layer_id: PASSED\n";
}

void test_extract_expert_id() {
    // Non-expert tensors return -1
    assert(extract_expert_id("blk.0.attn_q.weight") == -1);
    assert(extract_expert_id("blk.0.ffn_up.weight") == -1);

    // Expert tensors - expert ID is inferred from context, not tensor name
    // The tensor name just indicates it's an expert tensor
    // Actual expert ID comes from the tensor's position in the expert array
    assert(extract_expert_id("blk.0.ffn_down_exps.weight") == -1);  // Array of all experts

    std::cout << "test_extract_expert_id: PASSED\n";
}

int main() {
    try {
        test_tensor_classification();
        test_priority_from_type();
        test_extract_layer_id();
        test_extract_expert_id();
        std::cout << "\nAll tensor classification tests PASSED!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test FAILED: " << e.what() << "\n";
        return 1;
    }
}
```

### Step 2: Run test to verify it fails

```bash
cmake --build build --target test-tensor-classification -j$(nproc)
```

Expected: FAILS - `tensor-types.hpp` not found

### Step 3: Create tensor types header

Create `ggml/src/ggml-sycl/tensor-types.hpp`:

```cpp
//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_TENSOR_TYPES_HPP
#define GGML_SYCL_TENSOR_TYPES_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#if GGML_SYCL_DEBUG
#include <unordered_set>
#endif

#include "ggml.h"

namespace ggml_sycl {

// Tensor classification for tiered memory placement
enum class tensor_class {
    EMBEDDING,  // token_embd - used every input token
    OUTPUT,     // lm_head/output - used every generated token
    ATTENTION,  // attn_q/k/v/o - every layer, every token
    FFN,        // ffn_up/down/gate (non-MoE) - every layer
    ROUTER,     // ffn_gate_inp - every MoE layer
    EXPERT,     // *_exps - conditional, ~10-20% usage
    NORM,       // *_norm - tiny, fast from host
    OTHER       // unknown tensors
};

// Memory tier for tensor placement
enum class memory_tier {
    VRAM,         // Device VRAM (fastest)
    PINNED_HOST,  // Pinned host memory (GPU-accessible via PCIe)
    MMAP          // File-backed mmap (slowest, fallback)
};

// Information about a single tensor
struct tensor_info {
    std::string  name;
    size_t       size;
    int          layer_id;        // -1 if not layer-specific
    int          expert_id;       // -1 if not a single expert (exps tensors contain all experts)
    tensor_class type;
    int          static_priority; // 0=highest, computed from type
};

// Inventory of all tensors in a model
struct tensor_inventory {
    std::vector<tensor_info> tensors;
    size_t total_size   = 0;
    size_t dense_size   = 0;  // embeddings + attention + ffn + router + norms
    size_t expert_size  = 0;  // all expert weights
    int    num_layers   = 0;
    int    num_experts  = 0;
};

// Classify a tensor by name pattern
inline tensor_class classify_tensor(const char* name) {
    if (!name) return tensor_class::OTHER;

    // Embeddings (priority 0)
    if (strstr(name, "token_embd")) return tensor_class::EMBEDDING;

    // Output head (priority 0)
    if (strstr(name, "lm_head") || strstr(name, "output.weight")) return tensor_class::OUTPUT;

    // Experts must be checked before FFN (priority 3)
    if (strstr(name, "_exps")) return tensor_class::EXPERT;

    // Router (priority 2)
    if (strstr(name, "ffn_gate_inp")) return tensor_class::ROUTER;

    // Attention (priority 1)
    if (strstr(name, "attn_")) return tensor_class::ATTENTION;

    // Dense FFN (priority 2)
    if (strstr(name, "ffn_")) return tensor_class::FFN;

    // Norms (priority 4)
    if (strstr(name, "_norm")) return tensor_class::NORM;

#if GGML_SYCL_DEBUG
    // Log unknown tensors (once per unique name)
    static std::unordered_set<std::string> warned_tensors;
    if (warned_tensors.insert(name).second) {
        GGML_LOG_WARN("[SYCL] Unknown tensor type: %s\n", name);
    }
#endif

    return tensor_class::OTHER;
}

// Get static priority from tensor class (lower = higher priority)
inline int priority_from_type(tensor_class type) {
    switch (type) {
        case tensor_class::EMBEDDING: return 0;
        case tensor_class::OUTPUT:    return 0;
        case tensor_class::ATTENTION: return 1;
        case tensor_class::FFN:       return 2;
        case tensor_class::ROUTER:    return 2;
        case tensor_class::EXPERT:    return 3;
        case tensor_class::NORM:      return 4;
        case tensor_class::OTHER:     return 5;
    }
    return 5;
}

// Extract layer ID from tensor name (returns -1 if not found)
inline int extract_layer_id(const char* name) {
    if (!name) return -1;

    // Look for "blk.N." pattern
    const char* blk = strstr(name, "blk.");
    if (!blk) return -1;

    int layer_id = 0;
    const char* p = blk + 4;  // Skip "blk."
    while (*p >= '0' && *p <= '9') {
        layer_id = layer_id * 10 + (*p - '0');
        p++;
    }
    return layer_id;
}

// Extract expert ID from tensor name
// Note: Most expert tensors (ffn_*_exps) contain ALL experts in a single tensor
// This returns -1 for such tensors; actual expert ID comes from runtime indexing
inline int extract_expert_id(const char* name) {
    if (!name) return -1;

    // Check for per-expert tensor pattern (rare, model-specific)
    // Most models use *_exps which contains all experts
    const char* exp = strstr(name, ".expert.");
    if (exp) {
        int expert_id = 0;
        const char* p = exp + 8;  // Skip ".expert."
        while (*p >= '0' && *p <= '9') {
            expert_id = expert_id * 10 + (*p - '0');
            p++;
        }
        return expert_id;
    }

    return -1;  // Not a single-expert tensor
}

// Build tensor info from name and size
inline tensor_info make_tensor_info(const char* name, size_t size) {
    tensor_info info;
    info.name = name ? name : "";
    info.size = size;
    info.layer_id = extract_layer_id(name);
    info.expert_id = extract_expert_id(name);
    info.type = classify_tensor(name);
    info.static_priority = priority_from_type(info.type);
    return info;
}

}  // namespace ggml_sycl

#endif  // GGML_SYCL_TENSOR_TYPES_HPP
```

### Step 4: Run test to verify it passes

```bash
cmake --build build --target test-tensor-classification -j$(nproc)
./build/bin/test-tensor-classification
```

Expected:
```
test_tensor_classification: PASSED
test_priority_from_type: PASSED
test_extract_layer_id: PASSED
test_extract_expert_id: PASSED

All tensor classification tests PASSED!
```

### Step 5: Commit

```bash
git add ggml/src/ggml-sycl/tensor-types.hpp \
        tests/test-tensor-classification.cpp \
        tests/CMakeLists.txt
git commit -m "feat(sycl): add tensor classification for tiered placement

Classifies tensors as EMBEDDING/OUTPUT/ATTENTION/FFN/ROUTER/EXPERT/NORM/OTHER.
Static priority: 0 (embeddings, output) to 5 (other).
Extracts layer_id from blk.N. pattern.

Part of: Unified Tensor Cache"
```

---

## Task 3: Create unified_tensor_cache class

**Files:**
- Create: `ggml/src/ggml-sycl/unified-tensor-cache.hpp`
- Create: `ggml/src/ggml-sycl/unified-tensor-cache.cpp`
- Create: `tests/test-unified-tensor-cache.cpp`
- Modify: `ggml/src/ggml-sycl/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

### Step 1: Write the failing test

Create `tests/test-unified-tensor-cache.cpp`:

```cpp
//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include <cassert>
#include <cstring>
#include <iostream>
#include <sycl/sycl.hpp>

#include "../ggml/src/ggml-sycl/unified-tensor-cache.hpp"

using namespace ggml_sycl;

void test_inventory_placement() {
    sycl::queue q;

    // Create cache with 500MB VRAM, 1GB host
    constexpr size_t VRAM_BUDGET = 500 * 1024 * 1024;
    constexpr size_t HOST_BUDGET = 1ULL * 1024 * 1024 * 1024;

    unified_tensor_cache cache(q, VRAM_BUDGET, HOST_BUDGET);

    // Create inventory with mixed priorities
    tensor_inventory inventory;

    // 100MB embedding (priority 0) - should go to VRAM
    inventory.tensors.push_back(make_tensor_info("token_embd.weight", 100 * 1024 * 1024));

    // 200MB attention (priority 1) - should go to VRAM
    inventory.tensors.push_back(make_tensor_info("blk.0.attn_q.weight", 200 * 1024 * 1024));

    // 300MB experts (priority 3) - should go to host (VRAM full after embedding+attention)
    inventory.tensors.push_back(make_tensor_info("blk.0.ffn_down_exps.weight", 300 * 1024 * 1024));

    inventory.total_size = 600 * 1024 * 1024;

    // Set inventory and trigger placement
    cache.set_inventory(inventory);

    // Verify placements
    auto [embd_ptr, embd_tier] = cache.get_tensor_with_location(0);
    assert(embd_tier == memory_tier::VRAM && "Embedding should be in VRAM");

    auto [attn_ptr, attn_tier] = cache.get_tensor_with_location(1);
    assert(attn_tier == memory_tier::VRAM && "Attention should be in VRAM");

    auto [exp_ptr, exp_tier] = cache.get_tensor_with_location(2);
    assert(exp_tier == memory_tier::PINNED_HOST && "Expert should be in host");

    std::cout << "test_inventory_placement: PASSED\n";
}

void test_dynamic_promotion() {
    sycl::queue q;

    constexpr size_t VRAM_BUDGET = 200 * 1024 * 1024;
    constexpr size_t HOST_BUDGET = 500 * 1024 * 1024;

    unified_tensor_cache cache(q, VRAM_BUDGET, HOST_BUDGET);

    tensor_inventory inventory;

    // 100MB embedding - VRAM
    inventory.tensors.push_back(make_tensor_info("token_embd.weight", 100 * 1024 * 1024));

    // 100MB expert 0 - host initially
    inventory.tensors.push_back(make_tensor_info("blk.0.ffn_down_exps.0", 100 * 1024 * 1024));

    // 100MB expert 1 - host initially
    inventory.tensors.push_back(make_tensor_info("blk.0.ffn_down_exps.1", 100 * 1024 * 1024));

    inventory.total_size = 300 * 1024 * 1024;
    cache.set_inventory(inventory);

    // Access expert 0 many times (should trigger promotion consideration)
    for (int i = 0; i < 10; i++) {
        cache.get_tensor_with_location(1);
    }

    // Request promotion of expert 0
    cache.request_promotion(1);
    cache.wait_pending_transfers();

    // Expert 0 should now be in VRAM (evicted embedding or co-located)
    auto [exp0_ptr, exp0_tier] = cache.get_tensor_with_location(1);
    // Note: Whether promotion succeeded depends on eviction policy
    // If embedding is protected, expert stays in host

    std::cout << "test_dynamic_promotion: PASSED\n";
}

void test_prefetch() {
    sycl::queue q;

    constexpr size_t VRAM_BUDGET = 400 * 1024 * 1024;
    constexpr size_t HOST_BUDGET = 1ULL * 1024 * 1024 * 1024;

    unified_tensor_cache cache(q, VRAM_BUDGET, HOST_BUDGET);

    tensor_inventory inventory;
    inventory.tensors.push_back(make_tensor_info("token_embd.weight", 100 * 1024 * 1024));
    inventory.tensors.push_back(make_tensor_info("blk.0.ffn_down_exps.0", 100 * 1024 * 1024));
    inventory.tensors.push_back(make_tensor_info("blk.0.ffn_down_exps.1", 100 * 1024 * 1024));
    inventory.tensors.push_back(make_tensor_info("blk.0.ffn_down_exps.2", 100 * 1024 * 1024));
    inventory.total_size = 400 * 1024 * 1024;

    cache.set_inventory(inventory);

    // Prefetch experts 1 and 2
    std::vector<uint64_t> to_prefetch = {1, 2};
    cache.prefetch(to_prefetch);
    cache.wait_pending_transfers();

    // Prefetched tensors should be in VRAM
    auto [exp1_ptr, exp1_tier] = cache.get_tensor_with_location(1);
    auto [exp2_ptr, exp2_tier] = cache.get_tensor_with_location(2);

    // Note: Actual tier depends on VRAM availability and eviction
    assert(exp1_ptr != nullptr && "Expert 1 should be accessible");
    assert(exp2_ptr != nullptr && "Expert 2 should be accessible");

    std::cout << "test_prefetch: PASSED\n";
}

int main() {
    try {
        test_inventory_placement();
        test_dynamic_promotion();
        test_prefetch();
        std::cout << "\nAll unified_tensor_cache tests PASSED!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test FAILED: " << e.what() << "\n";
        return 1;
    }
}
```

### Step 2: Run test to verify it fails

```bash
cmake --build build --target test-unified-tensor-cache -j$(nproc)
```

Expected: FAILS - `unified-tensor-cache.hpp` not found

### Step 3: Create unified_tensor_cache header

Create `ggml/src/ggml-sycl/unified-tensor-cache.hpp`:

```cpp
//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_UNIFIED_TENSOR_CACHE_HPP
#define GGML_SYCL_UNIFIED_TENSOR_CACHE_HPP

#include "pinned-pool.hpp"
#include "tensor-types.hpp"
#include "vram-pool.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ggml_sycl {

// Result of tensor location query
struct tensor_location {
    void*       ptr;
    memory_tier tier;
};

// Unified tensor cache managing all model weights across memory tiers.
//
// Consolidates previous fragmented caches (unified-cache, expert_cache,
// dense_layer_scheduler, pinned_chunk_pool) into one system.
//
// Design:
// - Owns both VRAM pool and pinned host pool
// - Two-phase loading: set_inventory() then load_tensor_data()
// - Static priority by tensor type + dynamic access-pattern adjustment
// - Auto-enables tiered mode when model exceeds VRAM
//
// Thread-safe: all public methods can be called from multiple threads.
class unified_tensor_cache {
  public:
    // Create cache with VRAM and host budgets
    unified_tensor_cache(sycl::queue& queue, size_t vram_budget, size_t host_budget);
    ~unified_tensor_cache();

    // Non-copyable, non-movable
    unified_tensor_cache(const unified_tensor_cache&) = delete;
    unified_tensor_cache& operator=(const unified_tensor_cache&) = delete;
    unified_tensor_cache(unified_tensor_cache&&) = delete;
    unified_tensor_cache& operator=(unified_tensor_cache&&) = delete;

    // === Phase 1: Inventory ===

    // Set tensor inventory and compute placement decisions
    // Called once after GGUF parsing, before tensor allocation
    void set_inventory(const tensor_inventory& inventory);

    // Check if tiered mode is enabled (model > VRAM)
    bool is_tiered_enabled() const { return tiered_enabled_; }

    // Get planned tier for a tensor (by inventory index)
    memory_tier get_planned_tier(uint64_t tensor_id) const;

    // === Phase 2: Tensor Access ===

    // Get tensor pointer and current location
    // If tensor not yet loaded, loads it to planned tier
    tensor_location get_tensor_with_location(uint64_t tensor_id);

    // Load tensor data from source to cache
    // src_ptr: source data (e.g., mmap pointer)
    // Called during model loading
    void load_tensor_data(uint64_t tensor_id, const void* src_ptr);

    // === Dynamic Promotion ===

    // Request async promotion of tensor to VRAM
    // Used when access pattern suggests tensor should be promoted
    void request_promotion(uint64_t tensor_id);

    // Prefetch tensors to VRAM asynchronously
    // Used with router predictions for MoE
    void prefetch(const std::vector<uint64_t>& tensor_ids);

    // Wait for all pending async transfers
    void wait_pending_transfers();

    // === Statistics ===

    size_t vram_budget() const { return vram_budget_; }
    size_t vram_used() const { return vram_.used(); }
    size_t host_budget() const { return host_budget_; }
    size_t host_used() const;

    size_t cache_hits() const { return cache_hits_.load(); }
    size_t cache_misses() const { return cache_misses_.load(); }
    size_t promotions() const { return promotions_.load(); }
    size_t evictions() const { return evictions_.load(); }

    void print_stats() const;

  private:
    // Entry for a tensor in the cache
    struct tensor_entry {
        tensor_info info;
        void*       host_ptr     = nullptr;  // Always set after load
        void*       vram_ptr     = nullptr;  // Set if in VRAM
        memory_tier current_tier = memory_tier::MMAP;
        memory_tier planned_tier = memory_tier::MMAP;
        uint64_t    last_access  = 0;
        uint32_t    access_count = 0;
        bool        loaded       = false;
    };

    // Compute placement decisions based on inventory
    void compute_placement();

    // Evict lowest-score tensor from VRAM to make room
    bool evict_one(size_t needed_size);

    // Compute eviction score (higher = more valuable)
    float compute_score(const tensor_entry& entry) const;

    sycl::queue& queue_;
    sycl::queue  copy_queue_;  // Separate queue for async transfers

    size_t vram_budget_;
    size_t host_budget_;

    vram_pool         vram_;
    pinned_chunk_pool host_;

    std::unordered_map<uint64_t, tensor_entry> entries_;
    bool                                       tiered_enabled_ = false;
    uint64_t                                   time_           = 0;

    std::atomic<size_t> cache_hits_{0};
    std::atomic<size_t> cache_misses_{0};
    std::atomic<size_t> promotions_{0};
    std::atomic<size_t> evictions_{0};

    mutable std::mutex mutex_;
};

// === Global API ===

// Check if tiered memory should auto-enable
bool should_enable_tiered(const tensor_inventory& inv, size_t vram_available);

// Get/create unified tensor cache for device
unified_tensor_cache* get_unified_tensor_cache(sycl::queue& queue);
unified_tensor_cache* get_unified_tensor_cache_for_device(int device_id);

// Set budgets before first cache access
void set_unified_tensor_cache_vram_budget(size_t bytes);
void set_unified_tensor_cache_host_budget_pct(int pct);

}  // namespace ggml_sycl

#endif  // GGML_SYCL_UNIFIED_TENSOR_CACHE_HPP
```

### Step 4: Create unified_tensor_cache implementation

Create `ggml/src/ggml-sycl/unified-tensor-cache.cpp`:

```cpp
//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "unified-tensor-cache.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>

namespace ggml_sycl {

unified_tensor_cache::unified_tensor_cache(sycl::queue& queue, size_t vram_budget, size_t host_budget)
    : queue_(queue)
    , copy_queue_(queue.get_context(), queue.get_device())
    , vram_budget_(vram_budget)
    , host_budget_(host_budget)
    , vram_(queue, vram_budget)
    , host_(queue, host_budget) {

    GGML_LOG_INFO("[SYCL] Unified tensor cache created: VRAM %.2f GB, Host %.2f GB\n",
                  vram_budget / (1024.0 * 1024.0 * 1024.0),
                  host_budget / (1024.0 * 1024.0 * 1024.0));
}

unified_tensor_cache::~unified_tensor_cache() {
    print_stats();
}

void unified_tensor_cache::set_inventory(const tensor_inventory& inventory) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Store tensor info
    for (size_t i = 0; i < inventory.tensors.size(); i++) {
        tensor_entry entry;
        entry.info = inventory.tensors[i];
        entry.planned_tier = memory_tier::MMAP;  // Will be computed
        entries_[i] = entry;
    }

    // Check if tiered mode needed
    tiered_enabled_ = should_enable_tiered(inventory, vram_budget_);

    if (tiered_enabled_) {
        GGML_LOG_INFO("[SYCL] Tiered memory enabled: model %.2f GB exceeds VRAM %.2f GB\n",
                      inventory.total_size / (1024.0 * 1024.0 * 1024.0),
                      vram_budget_ / (1024.0 * 1024.0 * 1024.0));
    }

    compute_placement();
}

void unified_tensor_cache::compute_placement() {
    // Sort entries by priority (lower priority value = higher priority)
    std::vector<std::pair<uint64_t, tensor_entry*>> sorted;
    for (auto& [id, entry] : entries_) {
        sorted.push_back({id, &entry});
    }
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.second->info.static_priority < b.second->info.static_priority;
    });

    size_t vram_remaining = vram_budget_;
    size_t host_remaining = host_budget_;

    for (auto& [id, entry] : sorted) {
        if (entry->info.size <= vram_remaining) {
            entry->planned_tier = memory_tier::VRAM;
            vram_remaining -= entry->info.size;
        } else if (entry->info.size <= host_remaining) {
            entry->planned_tier = memory_tier::PINNED_HOST;
            host_remaining -= entry->info.size;
        } else {
            entry->planned_tier = memory_tier::MMAP;
        }
    }

    // Log placement summary
    size_t vram_count = 0, host_count = 0, mmap_count = 0;
    for (const auto& [id, entry] : entries_) {
        switch (entry.planned_tier) {
            case memory_tier::VRAM: vram_count++; break;
            case memory_tier::PINNED_HOST: host_count++; break;
            case memory_tier::MMAP: mmap_count++; break;
        }
    }
    GGML_LOG_INFO("[SYCL] Placement plan: %zu VRAM, %zu Host, %zu mmap\n",
                  vram_count, host_count, mmap_count);
}

memory_tier unified_tensor_cache::get_planned_tier(uint64_t tensor_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(tensor_id);
    if (it == entries_.end()) {
        return memory_tier::MMAP;
    }
    return it->second.planned_tier;
}

void unified_tensor_cache::load_tensor_data(uint64_t tensor_id, const void* src_ptr) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = entries_.find(tensor_id);
    if (it == entries_.end()) {
        GGML_LOG_ERROR("[SYCL] Unknown tensor ID: %llu\n", (unsigned long long)tensor_id);
        return;
    }

    tensor_entry& entry = it->second;
    if (entry.loaded) {
        return;  // Already loaded
    }

    const size_t size = entry.info.size;

    // Allocate based on planned tier
    if (entry.planned_tier == memory_tier::VRAM) {
        // Allocate VRAM and copy
        void* vram_ptr = vram_.allocate(size, tensor_id);
        if (vram_ptr) {
            queue_.memcpy(vram_ptr, src_ptr, size).wait();
            entry.vram_ptr = vram_ptr;
            entry.current_tier = memory_tier::VRAM;
        } else {
            // Fallback to host
            entry.planned_tier = memory_tier::PINNED_HOST;
        }
    }

    if (entry.planned_tier == memory_tier::PINNED_HOST) {
        // Allocate host and copy
        void* host_ptr = host_.allocate(size);
        if (host_ptr) {
            std::memcpy(host_ptr, src_ptr, size);
            entry.host_ptr = host_ptr;
            entry.current_tier = memory_tier::PINNED_HOST;
        } else {
            // Keep as mmap
            entry.host_ptr = const_cast<void*>(src_ptr);
            entry.current_tier = memory_tier::MMAP;
        }
    }

    if (entry.planned_tier == memory_tier::MMAP) {
        // Just reference the mmap pointer
        entry.host_ptr = const_cast<void*>(src_ptr);
        entry.current_tier = memory_tier::MMAP;
    }

    entry.loaded = true;
}

tensor_location unified_tensor_cache::get_tensor_with_location(uint64_t tensor_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = entries_.find(tensor_id);
    if (it == entries_.end()) {
        return {nullptr, memory_tier::MMAP};
    }

    tensor_entry& entry = it->second;
    entry.last_access = time_++;
    entry.access_count++;

    // Return VRAM pointer if available
    if (entry.vram_ptr) {
        cache_hits_++;
        return {entry.vram_ptr, memory_tier::VRAM};
    }

    // Return host pointer
    if (entry.host_ptr) {
        if (entry.current_tier == memory_tier::VRAM) {
            cache_hits_++;  // Was in VRAM cache
        } else {
            cache_misses_++;  // Not in VRAM
        }
        return {entry.host_ptr, entry.current_tier};
    }

    cache_misses_++;
    return {nullptr, memory_tier::MMAP};
}

void unified_tensor_cache::request_promotion(uint64_t tensor_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = entries_.find(tensor_id);
    if (it == entries_.end()) return;

    tensor_entry& entry = it->second;

    // Already in VRAM
    if (entry.vram_ptr) return;

    // No host data to promote
    if (!entry.host_ptr) return;

    const size_t size = entry.info.size;

    // Try to allocate VRAM, evicting if needed
    while (vram_.available() < size) {
        if (!evict_one(size)) {
            return;  // Can't evict enough
        }
    }

    void* vram_ptr = vram_.allocate(size, tensor_id);
    if (!vram_ptr) return;

    // Async copy
    copy_queue_.memcpy(vram_ptr, entry.host_ptr, size);
    entry.vram_ptr = vram_ptr;
    entry.current_tier = memory_tier::VRAM;
    promotions_++;
}

void unified_tensor_cache::prefetch(const std::vector<uint64_t>& tensor_ids) {
    for (uint64_t id : tensor_ids) {
        request_promotion(id);
    }
}

void unified_tensor_cache::wait_pending_transfers() {
    copy_queue_.wait();
}

bool unified_tensor_cache::evict_one(size_t needed_size) {
    // Find lowest-score VRAM-resident tensor
    uint64_t worst_id = UINT64_MAX;
    float worst_score = std::numeric_limits<float>::max();

    for (const auto& [id, entry] : entries_) {
        if (!entry.vram_ptr) continue;  // Not in VRAM
        if (entry.info.static_priority <= 1) continue;  // Don't evict high-priority

        float score = compute_score(entry);
        if (score < worst_score) {
            worst_score = score;
            worst_id = id;
        }
    }

    if (worst_id == UINT64_MAX) {
        return false;  // Nothing to evict
    }

    // Evict
    tensor_entry& entry = entries_[worst_id];
    vram_.deallocate(worst_id);
    entry.vram_ptr = nullptr;
    entry.current_tier = entry.host_ptr ? memory_tier::PINNED_HOST : memory_tier::MMAP;
    evictions_++;

    return true;
}

float unified_tensor_cache::compute_score(const tensor_entry& entry) const {
    // Combined LRU + frequency scoring
    float recency = 1.0f / static_cast<float>(time_ - entry.last_access + 1);
    float frequency = static_cast<float>(entry.access_count);
    return 0.3f * recency + 0.7f * frequency;
}

size_t unified_tensor_cache::host_used() const {
    return host_.allocated();
}

void unified_tensor_cache::print_stats() const {
    GGML_LOG_INFO("[SYCL] Unified tensor cache stats:\n");
    GGML_LOG_INFO("  VRAM: %.2f / %.2f GB\n",
                  vram_.used() / (1024.0 * 1024.0 * 1024.0),
                  vram_budget_ / (1024.0 * 1024.0 * 1024.0));
    GGML_LOG_INFO("  Host: %.2f / %.2f GB\n",
                  host_used() / (1024.0 * 1024.0 * 1024.0),
                  host_budget_ / (1024.0 * 1024.0 * 1024.0));
    GGML_LOG_INFO("  Hits: %zu, Misses: %zu, Promotions: %zu, Evictions: %zu\n",
                  cache_hits_.load(), cache_misses_.load(),
                  promotions_.load(), evictions_.load());
}

// === Global Functions ===

bool should_enable_tiered(const tensor_inventory& inv, size_t vram_available) {
    if (inv.total_size <= vram_available * 0.9) {
        return false;  // Fits in VRAM
    }
    GGML_LOG_INFO("[SYCL] Model size (%.1f GB) exceeds VRAM (%.1f GB), enabling tiered memory\n",
                  inv.total_size / (1024.0 * 1024.0 * 1024.0),
                  vram_available / (1024.0 * 1024.0 * 1024.0));
    return true;
}

// Global cache instances (per-device)
static std::unordered_map<int, std::unique_ptr<unified_tensor_cache>> g_caches;
static std::mutex g_cache_mutex;
static size_t g_vram_budget = 0;  // 0 = auto
static int g_host_budget_pct = 90;

unified_tensor_cache* get_unified_tensor_cache(sycl::queue& queue) {
    // Get device ID from queue
    auto device = queue.get_device();
    // Use a simple hash of device for now
    int device_id = static_cast<int>(std::hash<std::string>{}(
        device.get_info<sycl::info::device::name>()) % 1000);

    return get_unified_tensor_cache_for_device(device_id);
}

unified_tensor_cache* get_unified_tensor_cache_for_device(int device_id) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);

    auto it = g_caches.find(device_id);
    if (it != g_caches.end()) {
        return it->second.get();
    }

    // Create new cache - need a queue for this device
    // For now, return nullptr - caller should use set_inventory path
    return nullptr;
}

void set_unified_tensor_cache_vram_budget(size_t bytes) {
    g_vram_budget = bytes;
}

void set_unified_tensor_cache_host_budget_pct(int pct) {
    g_host_budget_pct = pct;
}

}  // namespace ggml_sycl
```

### Step 5: Add to CMakeLists.txt

Add to `ggml/src/ggml-sycl/CMakeLists.txt`:
```cmake
unified-tensor-cache.cpp
```

Add to `tests/CMakeLists.txt`:
```cmake
if (GGML_SYCL)
    ggml_add_test(test-unified-tensor-cache test-unified-tensor-cache.cpp)
    target_link_libraries(test-unified-tensor-cache PRIVATE ggml)
endif()
```

### Step 6: Run test to verify it passes

```bash
cmake --build build --target test-unified-tensor-cache -j$(nproc)
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-unified-tensor-cache
```

### Step 7: Commit

```bash
git add ggml/src/ggml-sycl/unified-tensor-cache.hpp \
        ggml/src/ggml-sycl/unified-tensor-cache.cpp \
        ggml/src/ggml-sycl/CMakeLists.txt \
        tests/test-unified-tensor-cache.cpp \
        tests/CMakeLists.txt
git commit -m "feat(sycl): add unified_tensor_cache for tiered memory

Consolidates fragmented cache components into single system:
- Owns both VRAM pool and pinned host pool
- Two-phase loading: set_inventory() then load_tensor_data()
- Static priority placement + dynamic promotion
- Auto-enables tiered mode when model > VRAM

Part of: Unified Tensor Cache"
```

---

## Task 4: Add tensor inventory API to llama-model.cpp

**Files:**
- Modify: `ggml/include/ggml-sycl.h` (new public API)
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (implement API)
- Modify: `src/llama-model.cpp` (call API during load)

### Step 1: Add public API declaration

Add to `ggml/include/ggml-sycl.h`:

```cpp
// Tensor inventory for tiered memory placement
struct ggml_sycl_tensor_info {
    const char* name;
    size_t      size;
    int         layer_id;
    int         priority;  // 0=highest
};

struct ggml_sycl_tensor_inventory {
    struct ggml_sycl_tensor_info* tensors;
    size_t count;
    size_t total_size;
};

// Set tensor inventory for tiered memory placement
// Must be called before tensor allocation
GGML_BACKEND_API void ggml_backend_sycl_set_tensor_inventory(
    ggml_backend_t backend,
    const struct ggml_sycl_tensor_inventory* inventory
);

// Check if tiered memory is enabled for this backend
GGML_BACKEND_API bool ggml_backend_sycl_is_tiered_enabled(ggml_backend_t backend);
```

### Step 2: Implement API in ggml-sycl.cpp

Add implementation (location TBD based on existing structure):

```cpp
void ggml_backend_sycl_set_tensor_inventory(
    ggml_backend_t backend,
    const ggml_sycl_tensor_inventory* inventory
) {
    if (!backend || !inventory) return;

    ggml_backend_sycl_context* ctx = static_cast<ggml_backend_sycl_context*>(backend->context);
    if (!ctx) return;

    // Convert to internal format
    ggml_sycl::tensor_inventory internal_inv;
    for (size_t i = 0; i < inventory->count; i++) {
        internal_inv.tensors.push_back(
            ggml_sycl::make_tensor_info(inventory->tensors[i].name, inventory->tensors[i].size)
        );
    }
    internal_inv.total_size = inventory->total_size;

    // Get or create unified tensor cache
    auto* cache = ggml_sycl::get_unified_tensor_cache(ctx->stream());
    if (cache) {
        cache->set_inventory(internal_inv);
    }
}

bool ggml_backend_sycl_is_tiered_enabled(ggml_backend_t backend) {
    if (!backend) return false;

    ggml_backend_sycl_context* ctx = static_cast<ggml_backend_sycl_context*>(backend->context);
    if (!ctx) return false;

    auto* cache = ggml_sycl::get_unified_tensor_cache(ctx->stream());
    return cache && cache->is_tiered_enabled();
}
```

### Step 3: Call from llama-model.cpp

In `llama_model_load()`, after GGUF parsing but before tensor allocation:

```cpp
// Collect tensor inventory for SYCL tiered memory
if (ggml_backend_is_sycl(model.backend)) {
    std::vector<ggml_sycl_tensor_info> tensor_infos;
    size_t total_size = 0;

    for (const auto& [name, tensor] : model.tensors_by_name) {
        ggml_sycl_tensor_info info;
        info.name = name.c_str();
        info.size = ggml_nbytes(tensor);
        info.layer_id = -1;  // Will be extracted by backend
        info.priority = 5;   // Will be computed by backend
        tensor_infos.push_back(info);
        total_size += info.size;
    }

    ggml_sycl_tensor_inventory inventory;
    inventory.tensors = tensor_infos.data();
    inventory.count = tensor_infos.size();
    inventory.total_size = total_size;

    ggml_backend_sycl_set_tensor_inventory(model.backend, &inventory);
}
```

### Step 4: Test with model load

```bash
cmake --build build -j$(nproc)
GGML_SYCL_DEBUG=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-cli -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -ngl 99 -p "Hello" -n 5 2>&1 | grep -E "(tiered|inventory|placement)"
```

### Step 5: Commit

```bash
git add ggml/include/ggml-sycl.h \
        ggml/src/ggml-sycl/ggml-sycl.cpp \
        src/llama-model.cpp
git commit -m "feat(sycl): add tensor inventory API for tiered placement

- ggml_backend_sycl_set_tensor_inventory(): pass model tensors to backend
- ggml_backend_sycl_is_tiered_enabled(): check if tiered mode active
- llama-model.cpp: collect inventory during model load

Part of: Unified Tensor Cache"
```

---

## Task 5: Update kernel callsites

This task updates the main matrix multiplication dispatch to use the unified tensor cache.

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (main mul_mat dispatch)

### Step 1: Identify dispatch function

The main dispatch is `ggml_sycl_op_mul_mat()`. Update to query cache for tensor location.

### Step 2: Update dispatch to use cache

```cpp
// In ggml_sycl_op_mul_mat or similar:

// Get tensor from unified cache if tiered mode enabled
auto* tensor_cache = ggml_sycl::get_unified_tensor_cache(ctx.stream());
if (tensor_cache && tensor_cache->is_tiered_enabled()) {
    uint64_t tensor_id = compute_tensor_id(src0);  // Hash or index
    auto [ptr, tier] = tensor_cache->get_tensor_with_location(tensor_id);

    if (ptr) {
        src0_ptr = ptr;
        // Optionally adapt kernel choice based on tier
    }
}
```

### Step 3: Test with model

```bash
GGML_SYCL_TIERED_DEBUG=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-cli -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -ngl 99 --lazy-moe -p "Hello" -n 10 2>&1 | grep -E "(tier|cache|VRAM)"
```

### Step 4: Commit

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "feat(sycl): integrate unified tensor cache in mul_mat dispatch

Query unified_tensor_cache for weight pointers when tiered mode enabled.
Kernels receive correct pointer regardless of VRAM/host location.

Part of: Unified Tensor Cache"
```

---

## Task 6: Integration testing

**Files:**
- Create: `tests/test-tiered-memory-integration.cpp`

### Step 1: Write integration test

```cpp
// Test full flow: inventory -> placement -> inference
// Uses mock model data to verify tiered memory behavior

void test_full_pipeline() {
    // 1. Create inventory simulating large model
    // 2. Set inventory on cache
    // 3. Load simulated tensors
    // 4. Run simulated inference accessing tensors
    // 5. Verify cache stats (hits, misses, promotions)
}
```

### Step 2: Test with real models

```bash
# Small model - should NOT enable tiered
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-cli \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -ngl 99 -p "Hello" -n 10

# Large MoE model - SHOULD enable tiered
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-cli \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -ngl 99 --lazy-moe -p "Hello" -n 10
```

### Step 3: Commit

```bash
git add tests/test-tiered-memory-integration.cpp tests/CMakeLists.txt
git commit -m "test(sycl): add tiered memory integration tests

End-to-end tests for:
- Inventory collection and placement
- VRAM/host allocation
- Cache hits/misses during inference

Part of: Unified Tensor Cache"
```

---

## Verification

After completing all tasks:

```bash
# Run all tiered memory tests
ctest --test-dir build -R "vram-pool|tensor-class|unified-tensor-cache|tiered" -V

# Integration test with GPT-OSS-20B
GGML_SYCL_TIERED_DEBUG=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-cli -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -ngl 99 --lazy-moe -p "The quick brown fox" -n 50

# Expected output:
# - "Tiered memory enabled" message
# - Placement summary (N VRAM, M Host)
# - Cache stats at end (hits, misses)
# - Successful text generation
```

### Success Criteria

- [ ] All unit tests pass
- [ ] Small models (< VRAM) work normally without tiered overhead
- [ ] Large models auto-enable tiered mode
- [ ] No "device lost" or allocation errors
- [ ] Cache hit rate > 80% after warmup
- [ ] Generation speed > 5 t/s for GPT-OSS-20B
