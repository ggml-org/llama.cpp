# Tiered Memory Architecture for MoE - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable MoE models (GPT-OSS-120B+) to use all system RAM as GPU-accessible pinned memory by implementing chunked allocation that bypasses the 11GB per-allocation driver limit.

**Architecture:** Three-tier memory hierarchy (VRAM → Pinned Host → mmap). Pinned host tier uses a pool of 8GB malloc_host chunks with bump allocation. Existing unified cache budget system (`--sycl-unified-cache-host-pct`) controls host tier sizing.

**Tech Stack:** SYCL/Level Zero, Intel Arc GPUs, C++17

**BD Epic:** llama.cpp-2pa

---

## Task 1: Implement pinned_chunk_pool (8GB chunks)

**BD Task:** llama.cpp-j1h

**Files:**
- Create: `ggml/src/ggml-sycl/pinned-pool.hpp`
- Create: `ggml/src/ggml-sycl/pinned-pool.cpp`
- Create: `tests/test-pinned-chunk-pool.cpp`
- Modify: `ggml/src/ggml-sycl/CMakeLists.txt` (add new source)
- Modify: `tests/CMakeLists.txt` (add test)

### Step 1: Write the failing test

Create `tests/test-pinned-chunk-pool.cpp`:

```cpp
#include <sycl/sycl.hpp>
#include <cassert>
#include <cstring>
#include <iostream>

// Forward declarations - will be implemented
namespace ggml_sycl {
class pinned_chunk_pool;
}

void test_basic_allocation() {
    sycl::queue q;

    // 16GB budget, 8GB chunks
    constexpr size_t BUDGET = 16ULL * 1024 * 1024 * 1024;
    ggml_sycl::pinned_chunk_pool pool(q, BUDGET);

    // Allocate 1GB
    void* ptr1 = pool.allocate(1ULL * 1024 * 1024 * 1024);
    assert(ptr1 != nullptr && "1GB allocation should succeed");

    // Allocate another 6GB (still fits in first 8GB chunk)
    void* ptr2 = pool.allocate(6ULL * 1024 * 1024 * 1024);
    assert(ptr2 != nullptr && "6GB allocation should succeed");

    // Allocate 2GB (needs second chunk)
    void* ptr3 = pool.allocate(2ULL * 1024 * 1024 * 1024);
    assert(ptr3 != nullptr && "2GB allocation should trigger new chunk");

    // Verify pointers are different
    assert(ptr1 != ptr2 && ptr2 != ptr3 && "Pointers should be unique");

    std::cout << "test_basic_allocation: PASSED\n";
}

void test_budget_limit() {
    sycl::queue q;

    // Small 10GB budget
    constexpr size_t BUDGET = 10ULL * 1024 * 1024 * 1024;
    ggml_sycl::pinned_chunk_pool pool(q, BUDGET);

    // First 8GB chunk should succeed
    void* ptr1 = pool.allocate(7ULL * 1024 * 1024 * 1024);
    assert(ptr1 != nullptr && "7GB allocation should succeed");

    // Second chunk would exceed budget (7GB used + 8GB new chunk > 10GB budget)
    // But we need space for 4GB more...
    void* ptr2 = pool.allocate(4ULL * 1024 * 1024 * 1024);
    // This should fail since we can't allocate another 8GB chunk
    assert(ptr2 == nullptr && "Should fail - exceeds budget");

    std::cout << "test_budget_limit: PASSED\n";
}

void test_gpu_accessible() {
    sycl::queue q;

    constexpr size_t BUDGET = 16ULL * 1024 * 1024 * 1024;
    ggml_sycl::pinned_chunk_pool pool(q, BUDGET);

    constexpr size_t SIZE = 1024 * 1024;  // 1MB
    void* host_ptr = pool.allocate(SIZE);
    assert(host_ptr != nullptr);

    // Write pattern to host memory
    std::memset(host_ptr, 0xAB, SIZE);

    // Allocate device memory
    void* device_ptr = sycl::malloc_device(SIZE, q);
    assert(device_ptr != nullptr);

    // Copy from pinned host to device (should work if truly pinned)
    q.memcpy(device_ptr, host_ptr, SIZE).wait();

    // Copy back to verify
    std::vector<char> verify(SIZE);
    q.memcpy(verify.data(), device_ptr, SIZE).wait();

    // Check pattern
    for (size_t i = 0; i < SIZE; i++) {
        assert(static_cast<unsigned char>(verify[i]) == 0xAB && "Data mismatch");
    }

    sycl::free(device_ptr, q);
    std::cout << "test_gpu_accessible: PASSED\n";
}

void test_alignment() {
    sycl::queue q;

    constexpr size_t BUDGET = 16ULL * 1024 * 1024 * 1024;
    ggml_sycl::pinned_chunk_pool pool(q, BUDGET);

    // Allocate with various sizes, check 64-byte alignment
    void* ptr1 = pool.allocate(100);
    void* ptr2 = pool.allocate(1000);
    void* ptr3 = pool.allocate(10000);

    assert((reinterpret_cast<uintptr_t>(ptr1) % 64) == 0 && "ptr1 not aligned");
    assert((reinterpret_cast<uintptr_t>(ptr2) % 64) == 0 && "ptr2 not aligned");
    assert((reinterpret_cast<uintptr_t>(ptr3) % 64) == 0 && "ptr3 not aligned");

    std::cout << "test_alignment: PASSED\n";
}

int main() {
    try {
        test_basic_allocation();
        test_budget_limit();
        test_gpu_accessible();
        test_alignment();
        std::cout << "\nAll tests PASSED!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test FAILED: " << e.what() << "\n";
        return 1;
    }
}
```

### Step 2: Run test to verify it fails

```bash
# Add to tests/CMakeLists.txt first, then:
source /opt/intel/oneapi/setvars.sh --force
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build --target test-pinned-chunk-pool -j$(nproc)
```

Expected: Compilation FAILS - `pinned_chunk_pool` class not defined

### Step 3: Create header file

Create `ggml/src/ggml-sycl/pinned-pool.hpp`:

```cpp
#ifndef GGML_SYCL_PINNED_POOL_HPP
#define GGML_SYCL_PINNED_POOL_HPP

#include <sycl/sycl.hpp>
#include <cstddef>
#include <mutex>
#include <vector>

namespace ggml_sycl {

// Pool allocator for pinned host memory using multiple chunks
// Bypasses Intel Level Zero's ~11GB per-allocation limit
class pinned_chunk_pool {
public:
    static constexpr size_t CHUNK_SIZE = 8ULL * 1024 * 1024 * 1024;  // 8GB
    static constexpr size_t DEFAULT_ALIGNMENT = 64;

    pinned_chunk_pool(sycl::queue& queue, size_t budget);
    ~pinned_chunk_pool();

    // Non-copyable, non-movable
    pinned_chunk_pool(const pinned_chunk_pool&) = delete;
    pinned_chunk_pool& operator=(const pinned_chunk_pool&) = delete;
    pinned_chunk_pool(pinned_chunk_pool&&) = delete;
    pinned_chunk_pool& operator=(pinned_chunk_pool&&) = delete;

    // Allocate from pool (returns nullptr if over budget)
    void* allocate(size_t size, size_t alignment = DEFAULT_ALIGNMENT);

    // Mark allocation as free (bump allocator - reclaim when chunk empty)
    void deallocate(void* ptr, size_t size);

    // Statistics
    size_t budget() const { return budget_; }
    size_t allocated() const { return total_allocated_; }
    size_t chunk_count() const;

private:
    struct chunk {
        void*  base;   // malloc_host result
        size_t size;   // CHUNK_SIZE
        size_t used;   // bump pointer offset
        size_t freed;  // bytes deallocated (for reclaim tracking)
    };

    bool grow();  // Add new chunk

    sycl::queue& queue_;
    size_t budget_;
    size_t total_allocated_ = 0;
    std::vector<chunk> chunks_;
    mutable std::mutex mutex_;
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_PINNED_POOL_HPP
```

### Step 4: Create implementation file

Create `ggml/src/ggml-sycl/pinned-pool.cpp`:

```cpp
#include "pinned-pool.hpp"
#include "ggml.h"  // For GGML_LOG_*

namespace ggml_sycl {

pinned_chunk_pool::pinned_chunk_pool(sycl::queue& queue, size_t budget)
    : queue_(queue), budget_(budget) {
    GGML_LOG_INFO("[SYCL] Pinned chunk pool created with %.1f GB budget\n",
                  budget / (1024.0 * 1024.0 * 1024.0));
}

pinned_chunk_pool::~pinned_chunk_pool() {
    for (auto& c : chunks_) {
        if (c.base) {
            sycl::free(c.base, queue_);
        }
    }
    chunks_.clear();
    GGML_LOG_INFO("[SYCL] Pinned chunk pool destroyed, released %zu chunks\n",
                  chunks_.size());
}

void* pinned_chunk_pool::allocate(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Round up size to alignment
    size = (size + alignment - 1) & ~(alignment - 1);

    // Try existing chunks
    for (auto& c : chunks_) {
        size_t aligned_offset = (c.used + alignment - 1) & ~(alignment - 1);
        if (aligned_offset + size <= c.size) {
            void* ptr = static_cast<char*>(c.base) + aligned_offset;
            c.used = aligned_offset + size;
            return ptr;
        }
    }

    // Need new chunk - check budget
    if (total_allocated_ + CHUNK_SIZE > budget_) {
        GGML_LOG_WARN("[SYCL] Pinned pool budget exceeded (%.1f GB used, %.1f GB budget)\n",
                      total_allocated_ / (1024.0 * 1024.0 * 1024.0),
                      budget_ / (1024.0 * 1024.0 * 1024.0));
        return nullptr;
    }

    if (!grow()) {
        return nullptr;
    }

    // Allocate from new chunk
    auto& c = chunks_.back();
    void* ptr = c.base;
    c.used = size;
    return ptr;
}

void pinned_chunk_pool::deallocate(void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Find containing chunk
    for (auto& c : chunks_) {
        char* base = static_cast<char*>(c.base);
        char* p = static_cast<char*>(ptr);
        if (p >= base && p < base + c.size) {
            c.freed += size;
            // Note: bump allocator doesn't reclaim individual allocations
            // Chunk is only released when pool is destroyed
            return;
        }
    }
}

size_t pinned_chunk_pool::chunk_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return chunks_.size();
}

bool pinned_chunk_pool::grow() {
    void* ptr = nullptr;
    try {
        ptr = sycl::malloc_host(CHUNK_SIZE, queue_);
    } catch (const sycl::exception& e) {
        GGML_LOG_ERROR("[SYCL] Failed to allocate 8GB pinned chunk: %s\n", e.what());
        return false;
    }

    if (!ptr) {
        GGML_LOG_ERROR("[SYCL] Failed to allocate 8GB pinned chunk (nullptr)\n");
        return false;
    }

    chunks_.push_back({ptr, CHUNK_SIZE, 0, 0});
    total_allocated_ += CHUNK_SIZE;

    GGML_LOG_INFO("[SYCL] Allocated pinned chunk %zu (total: %.1f GB)\n",
                  chunks_.size(),
                  total_allocated_ / (1024.0 * 1024.0 * 1024.0));

    return true;
}

}  // namespace ggml_sycl
```

### Step 5: Add to CMakeLists.txt

Modify `ggml/src/ggml-sycl/CMakeLists.txt` - add `pinned-pool.cpp` to source list:

```cmake
# Find the list of source files and add:
pinned-pool.cpp
```

Modify `tests/CMakeLists.txt` - add test:

```cmake
if (GGML_SYCL)
    # ... existing SYCL tests ...

    ggml_add_test(test-pinned-chunk-pool test-pinned-chunk-pool.cpp)
    target_link_libraries(test-pinned-chunk-pool PRIVATE ggml)
endif()
```

### Step 6: Run test to verify it passes

```bash
source /opt/intel/oneapi/setvars.sh --force
cmake --build build --target test-pinned-chunk-pool -j$(nproc)
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-pinned-chunk-pool
```

Expected:
```
[SYCL] Pinned chunk pool created with 16.0 GB budget
[SYCL] Allocated pinned chunk 1 (total: 8.0 GB)
[SYCL] Allocated pinned chunk 2 (total: 16.0 GB)
test_basic_allocation: PASSED
[SYCL] Pinned chunk pool created with 10.0 GB budget
[SYCL] Allocated pinned chunk 1 (total: 8.0 GB)
[SYCL] Pinned pool budget exceeded (8.0 GB used, 10.0 GB budget)
test_budget_limit: PASSED
test_gpu_accessible: PASSED
test_alignment: PASSED

All tests PASSED!
```

### Step 7: Commit

```bash
git add ggml/src/ggml-sycl/pinned-pool.hpp \
        ggml/src/ggml-sycl/pinned-pool.cpp \
        ggml/src/ggml-sycl/CMakeLists.txt \
        tests/test-pinned-chunk-pool.cpp \
        tests/CMakeLists.txt
git commit -m "feat(sycl): add pinned_chunk_pool for chunked host memory

Implements 8GB chunk-based allocation pool for pinned host memory.
Bypasses Intel Level Zero's ~11GB per-allocation limit.
Uses bump allocator with 64-byte alignment.

Part of: llama.cpp-2pa (Tiered Memory Architecture)
Closes: llama.cpp-j1h"
```

---

## Task 2: Replace host_cache malloc_host with pool

**BD Task:** llama.cpp-6lk

**Files:**
- Modify: `ggml/src/ggml-sycl/unified-cache.hpp`
- Modify: `ggml/src/ggml-sycl/unified-cache.cpp`
- Modify: `tests/test-pinned-chunk-pool.cpp` (add integration test)

### Step 1: Write the failing integration test

Add to `tests/test-pinned-chunk-pool.cpp`:

```cpp
void test_host_cache_uses_pool() {
    // This test verifies host_cache allocates from pinned pool
    // and doesn't fall back to std::malloc

    sycl::queue q;

    // Get host cache (should use pool internally)
    auto* cache = ggml_sycl::get_host_cache(q);
    assert(cache != nullptr);

    // Allocate several tensors
    const size_t TENSOR_SIZE = 256 * 1024 * 1024;  // 256MB each
    std::vector<void*> ptrs;

    for (int i = 0; i < 10; i++) {
        bool needs_fill = false;
        bool pinned = false;
        void* ptr = cache->ensure_cached_alloc(
            reinterpret_cast<void*>(i + 1),  // fake key
            nullptr,  // src_ptr
            TENSOR_SIZE,
            TENSOR_SIZE,
            ggml_sycl::cache_entry_type::MOE_EXPERT,
            i,   // layer
            i,   // expert
            GGML_LAYOUT_AOS,
            true,   // can_hash
            0,      // content_hash
            nullptr,
            &needs_fill,
            &pinned
        );

        assert(ptr != nullptr && "Allocation should succeed");
        assert(pinned && "Should be pinned allocation, not std::malloc fallback");
        ptrs.push_back(ptr);
    }

    std::cout << "test_host_cache_uses_pool: PASSED\n";
}
```

### Step 2: Run test to verify it fails

```bash
cmake --build build --target test-pinned-chunk-pool -j$(nproc)
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-pinned-chunk-pool
```

Expected: Test may pass/fail depending on current implementation - the goal is to ensure pinned=true

### Step 3: Modify host_cache to use pinned_chunk_pool

Modify `ggml/src/ggml-sycl/unified-cache.hpp`:

Add include and forward declaration near top:
```cpp
#include "pinned-pool.hpp"

namespace ggml_sycl {
// ...
```

Modify `host_cache` class to add pool member:
```cpp
class host_cache {
    // ... existing members ...

    // Add: Pinned memory pool (replaces direct malloc_host)
    std::unique_ptr<pinned_chunk_pool> pinned_pool_;

public:
    host_cache(sycl::queue& queue, size_t budget_bytes);
    // ... rest unchanged ...
};
```

### Step 4: Modify host_cache implementation

Modify `ggml/src/ggml-sycl/unified-cache.cpp`:

In `host_cache` constructor:
```cpp
host_cache::host_cache(sycl::queue& queue, size_t budget_bytes)
    : queue_(queue), budget_(budget_bytes) {
    // Create pinned pool with same budget
    pinned_pool_ = std::make_unique<pinned_chunk_pool>(queue_, budget_bytes);
}
```

In `ensure_cached_alloc`, replace the malloc_host call (around line 380-392):

```cpp
    // OLD CODE:
    // bool   pinned_alloc = true;
    // try {
    //     host_ptr = sycl::malloc_host(dst_size, queue_);
    // } catch (...) {
    //     host_ptr = nullptr;
    // }
    // if (!host_ptr) {
    //     pinned_alloc = false;
    //     host_ptr     = std::malloc(dst_size);
    // }

    // NEW CODE: Use pinned pool instead of direct malloc_host
    bool pinned_alloc = true;
    host_ptr = pinned_pool_->allocate(dst_size);
    if (!host_ptr) {
        // Pool exhausted - fall back to unpinned (last resort)
        pinned_alloc = false;
        host_ptr = std::malloc(dst_size);
        GGML_LOG_WARN("[SYCL] Pinned pool exhausted, falling back to unpinned memory\n");
    }
```

In `free_entry`, update to use pool deallocation:
```cpp
void host_cache::free_entry(host_cache_entry& entry) {
    if (entry.owns_ptr && entry.host_ptr) {
        if (entry.pinned_alloc) {
            // Return to pinned pool
            pinned_pool_->deallocate(entry.host_ptr, entry.size);
        } else {
            std::free(entry.host_ptr);
        }
        used_ -= entry.size;
    }
    entry.host_ptr = nullptr;
    entry.owns_ptr = false;
}
```

### Step 5: Run test to verify it passes

```bash
cmake --build build --target test-pinned-chunk-pool -j$(nproc)
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-pinned-chunk-pool
```

Expected: All tests pass, including `test_host_cache_uses_pool`

### Step 6: Integration test with real model

```bash
# Should see "Allocated pinned chunk" messages, NO "using CPU memory" warnings
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-cli \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -ngl 99 --lazy-moe -p "Hello" -n 10 2>&1 | grep -E "(pinned chunk|CPU memory)"
```

Expected:
```
[SYCL] Allocated pinned chunk 1 (total: 8.0 GB)
[SYCL] Allocated pinned chunk 2 (total: 16.0 GB)
...
```

NO lines containing "using CPU memory" or "falling back"

### Step 7: Commit

```bash
git add ggml/src/ggml-sycl/unified-cache.hpp \
        ggml/src/ggml-sycl/unified-cache.cpp \
        tests/test-pinned-chunk-pool.cpp
git commit -m "feat(sycl): integrate pinned_chunk_pool with host_cache

host_cache now allocates from pinned_chunk_pool instead of
direct malloc_host calls. Eliminates fallback to unpinned
CPU memory for normal operations.

Part of: llama.cpp-2pa (Tiered Memory Architecture)
Closes: llama.cpp-6lk"
```

---

## Task 3: Buffer type routing for tensor placement

**BD Task:** llama.cpp-8lr

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp`
- Modify: `src/llama-model.cpp`
- Create: `tests/test-tensor-placement.cpp`

### Step 1: Write failing test

Create `tests/test-tensor-placement.cpp`:

```cpp
#include <iostream>
#include <cassert>
#include <cstring>

// Test that tensor placement logic routes correctly:
// - Dense tensors: VRAM first, overflow to pinned
// - Expert tensors: VRAM until full, then pinned
// - mmap only when both exhausted

extern "C" {
    // Forward declarations from ggml-sycl
    int ggml_sycl_classify_tensor(const char* name);  // 0=dense, 1=expert, 2=other
}

void test_tensor_classification() {
    // Dense layer patterns
    assert(ggml_sycl_classify_tensor("blk.0.attn_q.weight") == 0);
    assert(ggml_sycl_classify_tensor("blk.5.attn_k.weight") == 0);
    assert(ggml_sycl_classify_tensor("blk.10.ffn_down.weight") == 0);

    // Expert patterns (MoE)
    assert(ggml_sycl_classify_tensor("blk.0.ffn_down_exps.weight") == 1);
    assert(ggml_sycl_classify_tensor("blk.5.ffn_gate_exps.weight") == 1);
    assert(ggml_sycl_classify_tensor("blk.10.ffn_up_exps.weight") == 1);

    // Router (treated as dense - used every token)
    assert(ggml_sycl_classify_tensor("blk.0.ffn_gate_inp.weight") == 0);

    // Other tensors
    assert(ggml_sycl_classify_tensor("token_embd.weight") == 2);
    assert(ggml_sycl_classify_tensor("output_norm.weight") == 2);

    std::cout << "test_tensor_classification: PASSED\n";
}

int main() {
    test_tensor_classification();
    std::cout << "\nAll tests PASSED!\n";
    return 0;
}
```

### Step 2: Run test to verify it fails

```bash
cmake --build build --target test-tensor-placement -j$(nproc)
```

Expected: FAILS - `ggml_sycl_classify_tensor` not defined

### Step 3: Implement tensor classification

Add to `ggml/src/ggml-sycl/ggml-sycl.cpp`:

```cpp
// Classify tensor for tiered placement
// Returns: 0=dense, 1=expert, 2=other
int ggml_sycl_classify_tensor(const char* name) {
    if (!name) return 2;

    // Check for expert patterns (MoE layers)
    if (strstr(name, "_exps.") != nullptr ||
        strstr(name, "_exps_") != nullptr) {
        return 1;  // Expert
    }

    // Check for dense layer patterns
    if (strstr(name, "blk.") != nullptr) {
        // Attention and FFN weights in blocks are dense
        if (strstr(name, "attn_") != nullptr ||
            strstr(name, "ffn_") != nullptr) {
            return 0;  // Dense
        }
    }

    return 2;  // Other (embeddings, norms, etc.)
}
```

### Step 4: Run test to verify it passes

```bash
cmake --build build --target test-tensor-placement -j$(nproc)
./build/bin/test-tensor-placement
```

Expected: `test_tensor_classification: PASSED`

### Step 5: Add placement decision function

Add to `ggml/src/ggml-sycl/ggml-sycl.cpp`:

```cpp
// Select buffer type for tensor based on tier availability
ggml_backend_buffer_type_t ggml_sycl_select_buffer_for_tensor(
    const char* tensor_name,
    size_t tensor_size,
    size_t vram_available,
    size_t pinned_available
) {
    // Try VRAM first for any tensor
    if (tensor_size <= vram_available) {
        return ggml_backend_sycl_buffer_type(0);  // VRAM
    }

    // Overflow to pinned host
    if (tensor_size <= pinned_available) {
        return ggml_backend_sycl_host_buffer_type();  // Pinned host
    }

    // Last resort: CPU buffer (will use mmap)
    return ggml_backend_cpu_buffer_type();
}
```

### Step 6: Commit

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp \
        tests/test-tensor-placement.cpp \
        tests/CMakeLists.txt
git commit -m "feat(sycl): add tensor classification and placement routing

Classifies tensors as dense/expert/other for tiered placement.
Routing: VRAM first, pinned host overflow, CPU/mmap fallback.

Part of: llama.cpp-2pa (Tiered Memory Architecture)
Closes: llama.cpp-8lr"
```

---

## Task 4: Expert cache with LRU/frequency

**BD Task:** llama.cpp-o71

**Files:**
- Create: `ggml/src/ggml-sycl/expert-cache.hpp`
- Create: `ggml/src/ggml-sycl/expert-cache.cpp`
- Create: `tests/test-expert-cache.cpp`
- Modify: `ggml/src/ggml-sycl/CMakeLists.txt`

### Step 1: Write failing test

Create `tests/test-expert-cache.cpp`:

```cpp
#include <sycl/sycl.hpp>
#include <cassert>
#include <iostream>
#include <vector>

namespace ggml_sycl {
class expert_cache;
class pinned_chunk_pool;
}

void test_expert_cache_basic() {
    sycl::queue q;

    // Setup: 1GB VRAM budget, 4GB pinned pool
    constexpr size_t VRAM_BUDGET = 1ULL * 1024 * 1024 * 1024;
    constexpr size_t HOST_BUDGET = 4ULL * 1024 * 1024 * 1024;
    constexpr size_t EXPERT_SIZE = 100 * 1024 * 1024;  // 100MB per expert

    ggml_sycl::pinned_chunk_pool pool(q, HOST_BUDGET);
    ggml_sycl::expert_cache cache(q, pool, VRAM_BUDGET);

    // Pre-populate pinned pool with "expert data"
    std::vector<void*> host_ptrs;
    for (int i = 0; i < 20; i++) {
        void* ptr = pool.allocate(EXPERT_SIZE);
        assert(ptr != nullptr);
        host_ptrs.push_back(ptr);
        cache.register_expert(/*layer=*/0, /*expert=*/i, ptr, EXPERT_SIZE);
    }

    // Access experts 0-9 (should cache to VRAM, 10 * 100MB = 1GB)
    for (int i = 0; i < 10; i++) {
        void* vram_ptr = cache.get_expert(0, i, EXPERT_SIZE);
        assert(vram_ptr != nullptr && "Should return VRAM pointer");
    }

    // Access expert 10 - should evict least recently used (expert 0)
    void* ptr10 = cache.get_expert(0, 10, EXPERT_SIZE);
    assert(ptr10 != nullptr);

    // Access expert 0 again - should cause cache miss, evict expert 1
    void* ptr0 = cache.get_expert(0, 0, EXPERT_SIZE);
    assert(ptr0 != nullptr);

    std::cout << "test_expert_cache_basic: PASSED\n";
}

void test_expert_cache_frequency() {
    sycl::queue q;

    constexpr size_t VRAM_BUDGET = 500 * 1024 * 1024;  // 500MB
    constexpr size_t HOST_BUDGET = 2ULL * 1024 * 1024 * 1024;
    constexpr size_t EXPERT_SIZE = 100 * 1024 * 1024;  // 100MB

    ggml_sycl::pinned_chunk_pool pool(q, HOST_BUDGET);
    ggml_sycl::expert_cache cache(q, pool, VRAM_BUDGET);

    // Register 10 experts
    std::vector<void*> host_ptrs;
    for (int i = 0; i < 10; i++) {
        void* ptr = pool.allocate(EXPERT_SIZE);
        host_ptrs.push_back(ptr);
        cache.register_expert(0, i, ptr, EXPERT_SIZE);
    }

    // Access expert 0 many times (high frequency)
    for (int i = 0; i < 100; i++) {
        cache.get_expert(0, 0, EXPERT_SIZE);
    }

    // Access experts 1-4 once each (fills VRAM: 5 * 100MB = 500MB)
    for (int i = 1; i < 5; i++) {
        cache.get_expert(0, i, EXPERT_SIZE);
    }

    // Access expert 5 - should evict lowest score (not expert 0 due to frequency)
    cache.get_expert(0, 5, EXPERT_SIZE);

    // Expert 0 should still be cached (high frequency protects it)
    assert(cache.is_cached_in_vram(0, 0) && "Expert 0 should remain cached");

    std::cout << "test_expert_cache_frequency: PASSED\n";
}

int main() {
    try {
        test_expert_cache_basic();
        test_expert_cache_frequency();
        std::cout << "\nAll tests PASSED!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test FAILED: " << e.what() << "\n";
        return 1;
    }
}
```

### Step 2: Run test to verify it fails

```bash
cmake --build build --target test-expert-cache -j$(nproc)
```

Expected: FAILS - `expert_cache` class not defined

### Step 3: Create expert_cache header

Create `ggml/src/ggml-sycl/expert-cache.hpp`:

```cpp
#ifndef GGML_SYCL_EXPERT_CACHE_HPP
#define GGML_SYCL_EXPERT_CACHE_HPP

#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace ggml_sycl {

class pinned_chunk_pool;

struct expert_key {
    int layer;
    int expert_id;

    bool operator==(const expert_key& o) const {
        return layer == o.layer && expert_id == o.expert_id;
    }
};

struct expert_key_hash {
    size_t operator()(const expert_key& k) const {
        return std::hash<int>()(k.layer) ^ (std::hash<int>()(k.expert_id) << 16);
    }
};

class expert_cache {
public:
    expert_cache(sycl::queue& compute_queue, pinned_chunk_pool& pool, size_t vram_budget);
    ~expert_cache();

    // Register expert's host location
    void register_expert(int layer, int expert_id, void* host_ptr, size_t size);

    // Get expert pointer (returns VRAM pointer, loads from host if needed)
    void* get_expert(int layer, int expert_id, size_t size);

    // Check if expert is currently in VRAM
    bool is_cached_in_vram(int layer, int expert_id) const;

    // Prefetch experts (async, for overlapping with compute)
    void prefetch(const std::vector<expert_key>& experts, size_t expert_size);

    // Statistics
    size_t vram_used() const { return vram_used_; }
    size_t cache_hits() const { return cache_hits_; }
    size_t cache_misses() const { return cache_misses_; }

private:
    struct vram_entry {
        void* vram_ptr;
        size_t size;
        uint64_t last_access;
        uint32_t access_count;
    };

    struct host_entry {
        void* host_ptr;
        size_t size;
    };

    float compute_score(const vram_entry& e) const;
    void* allocate_vram(size_t size);
    void evict_lowest_score();

    sycl::queue& compute_queue_;
    sycl::queue copy_queue_;
    pinned_chunk_pool& pinned_pool_;

    size_t vram_budget_;
    size_t vram_used_ = 0;
    uint64_t time_ = 0;

    size_t cache_hits_ = 0;
    size_t cache_misses_ = 0;

    std::unordered_map<expert_key, vram_entry, expert_key_hash> vram_entries_;
    std::unordered_map<expert_key, host_entry, expert_key_hash> host_entries_;
    mutable std::mutex mutex_;
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_EXPERT_CACHE_HPP
```

### Step 4: Create expert_cache implementation

Create `ggml/src/ggml-sycl/expert-cache.cpp`:

```cpp
#include "expert-cache.hpp"
#include "pinned-pool.hpp"
#include "ggml.h"

namespace ggml_sycl {

expert_cache::expert_cache(sycl::queue& compute_queue, pinned_chunk_pool& pool, size_t vram_budget)
    : compute_queue_(compute_queue)
    , copy_queue_(compute_queue.get_context(), compute_queue.get_device())
    , pinned_pool_(pool)
    , vram_budget_(vram_budget) {
    GGML_LOG_INFO("[SYCL] Expert cache created with %.1f GB VRAM budget\n",
                  vram_budget / (1024.0 * 1024.0 * 1024.0));
}

expert_cache::~expert_cache() {
    for (auto& [key, entry] : vram_entries_) {
        if (entry.vram_ptr) {
            sycl::free(entry.vram_ptr, compute_queue_);
        }
    }
    GGML_LOG_INFO("[SYCL] Expert cache destroyed (hits=%zu, misses=%zu)\n",
                  cache_hits_, cache_misses_);
}

void expert_cache::register_expert(int layer, int expert_id, void* host_ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    expert_key key{layer, expert_id};
    host_entries_[key] = {host_ptr, size};
}

void* expert_cache::get_expert(int layer, int expert_id, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    expert_key key{layer, expert_id};

    // Check VRAM cache
    auto it = vram_entries_.find(key);
    if (it != vram_entries_.end()) {
        it->second.last_access = time_++;
        it->second.access_count++;
        cache_hits_++;
        return it->second.vram_ptr;
    }

    // Cache miss
    cache_misses_++;

    // Get host pointer
    auto host_it = host_entries_.find(key);
    if (host_it == host_entries_.end()) {
        GGML_LOG_ERROR("[SYCL] Expert not registered: layer=%d expert=%d\n", layer, expert_id);
        return nullptr;
    }

    void* host_ptr = host_it->second.host_ptr;

    // Allocate VRAM (may evict)
    void* vram_ptr = allocate_vram(size);
    if (!vram_ptr) {
        // VRAM exhausted even after eviction - return host pointer for direct access
        return host_ptr;
    }

    // Copy from pinned host to VRAM
    compute_queue_.memcpy(vram_ptr, host_ptr, size).wait();

    vram_entries_[key] = {vram_ptr, size, time_++, 1};
    return vram_ptr;
}

bool expert_cache::is_cached_in_vram(int layer, int expert_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return vram_entries_.find({layer, expert_id}) != vram_entries_.end();
}

void expert_cache::prefetch(const std::vector<expert_key>& experts, size_t expert_size) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (const auto& key : experts) {
        if (vram_entries_.find(key) != vram_entries_.end()) {
            continue;  // Already cached
        }

        auto host_it = host_entries_.find(key);
        if (host_it == host_entries_.end()) {
            continue;  // Not registered
        }

        void* vram_ptr = allocate_vram(expert_size);
        if (!vram_ptr) {
            break;  // VRAM full
        }

        // Async copy
        copy_queue_.memcpy(vram_ptr, host_it->second.host_ptr, expert_size);
        vram_entries_[key] = {vram_ptr, expert_size, time_++, 0};
    }
    // Don't wait - let transfers overlap with compute
}

float expert_cache::compute_score(const vram_entry& e) const {
    // Combined LRU + frequency scoring
    float recency = 1.0f / static_cast<float>(time_ - e.last_access + 1);
    float frequency = static_cast<float>(e.access_count);
    return 0.3f * recency + 0.7f * frequency;
}

void* expert_cache::allocate_vram(size_t size) {
    while (vram_used_ + size > vram_budget_) {
        if (vram_entries_.empty()) {
            return nullptr;  // Nothing to evict
        }
        evict_lowest_score();
    }

    void* ptr = sycl::malloc_device(size, compute_queue_);
    if (ptr) {
        vram_used_ += size;
    }
    return ptr;
}

void expert_cache::evict_lowest_score() {
    if (vram_entries_.empty()) return;

    auto worst = vram_entries_.begin();
    float worst_score = compute_score(worst->second);

    for (auto it = vram_entries_.begin(); it != vram_entries_.end(); ++it) {
        float score = compute_score(it->second);
        if (score < worst_score) {
            worst = it;
            worst_score = score;
        }
    }

    sycl::free(worst->second.vram_ptr, compute_queue_);
    vram_used_ -= worst->second.size;
    vram_entries_.erase(worst);
}

}  // namespace ggml_sycl
```

### Step 5: Add to CMakeLists.txt

Add `expert-cache.cpp` to `ggml/src/ggml-sycl/CMakeLists.txt`

Add test to `tests/CMakeLists.txt`

### Step 6: Run test to verify it passes

```bash
cmake --build build --target test-expert-cache -j$(nproc)
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-expert-cache
```

Expected: All tests pass

### Step 7: Commit

```bash
git add ggml/src/ggml-sycl/expert-cache.hpp \
        ggml/src/ggml-sycl/expert-cache.cpp \
        ggml/src/ggml-sycl/CMakeLists.txt \
        tests/test-expert-cache.cpp \
        tests/CMakeLists.txt
git commit -m "feat(sycl): add expert_cache with LRU/frequency eviction

Implements VRAM↔host expert swapping with:
- LRU + frequency scoring for eviction decisions
- Async prefetch for overlapping transfers
- Direct host access fallback when VRAM exhausted

Part of: llama.cpp-2pa (Tiered Memory Architecture)
Closes: llama.cpp-o71"
```

---

## Task 5: Dense layer scheduler (optional)

**BD Task:** llama.cpp-f94

**Files:**
- Create: `ggml/src/ggml-sycl/dense-scheduler.hpp`
- Create: `ggml/src/ggml-sycl/dense-scheduler.cpp`
- Create: `tests/test-dense-scheduler.cpp`

This task is **optional** - only needed for models where dense layers exceed VRAM.
See design doc for double-buffered prefetch implementation.

Skip this task if not needed for current models.

---

## Verification

After completing Tasks 1-4:

```bash
# Full integration test with GPT-OSS-120B
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-cli \
  -m /Storage/GenAI/models/gpt-oss-120b-Q4_0.gguf \
  -ngl 99 --lazy-moe \
  -p "The quick brown fox" -n 50

# Expected:
# - "Allocated pinned chunk N" messages (multiple 8GB chunks)
# - NO "using CPU memory" warnings
# - Successful text generation
```

### Metrics to monitor:
- Chunk allocation count and total size
- Expert cache hit/miss ratio
- VRAM utilization
- Token generation speed (t/s)
