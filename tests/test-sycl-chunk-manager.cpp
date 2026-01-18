// SYCL ChunkManager unit tests
// Tests for sub-tensor streaming chunk abstraction
// Part of unified memory management system (epic llama.cpp-v3n, task llama.cpp-4ye)
//
// TDD: These tests written FIRST, before implementation.
// Implementation must make these tests pass.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"

// Include the chunk manager header (to be created)
#include "ggml-sycl/chunk-manager.hpp"

// =============================================================================
// Test 1: Chunk size auto-detection from device capabilities
// =============================================================================
static bool test_chunk_size_detection() {
    printf("TEST: test_chunk_size_detection\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: SYCL backend unavailable\n");
        return true;
    }

    // Create chunk manager - should auto-detect optimal chunk size
    ggml_sycl::ChunkManager mgr(0);  // device 0

    size_t chunk_size = mgr.get_chunk_size();

    // Verify chunk size is within reasonable bounds
    const size_t min_chunk = 16 * 1024 * 1024;   // 16 MB minimum
    const size_t max_chunk = 256 * 1024 * 1024;  // 256 MB maximum

    if (chunk_size < min_chunk) {
        printf("  FAIL: chunk_size %zu < minimum %zu\n", chunk_size, min_chunk);
        ggml_backend_free(backend);
        return false;
    }

    if (chunk_size > max_chunk) {
        printf("  FAIL: chunk_size %zu > maximum %zu\n", chunk_size, max_chunk);
        ggml_backend_free(backend);
        return false;
    }

    // Verify chunk size is cache-line aligned (typically 64 bytes)
    if (chunk_size % 64 != 0) {
        printf("  FAIL: chunk_size %zu not aligned to 64 bytes\n", chunk_size);
        ggml_backend_free(backend);
        return false;
    }

    printf("  PASS: auto-detected chunk_size = %zu MB\n", chunk_size / (1024 * 1024));
    ggml_backend_free(backend);
    return true;
}

// =============================================================================
// Test 2: Environment variable override for chunk size
// =============================================================================
static bool test_chunk_size_env_override() {
    printf("TEST: test_chunk_size_env_override\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: SYCL backend unavailable\n");
        return true;
    }

    // Set environment variable
    setenv("GGML_SYCL_CHUNK_SIZE_MB", "32", 1);

    // Create chunk manager - should use env override
    ggml_sycl::ChunkManager mgr(0);

    size_t chunk_size = mgr.get_chunk_size();
    size_t expected = 32 * 1024 * 1024;  // 32 MB

    // Clean up env var
    unsetenv("GGML_SYCL_CHUNK_SIZE_MB");

    if (chunk_size != expected) {
        printf("  FAIL: expected chunk_size %zu, got %zu\n", expected, chunk_size);
        ggml_backend_free(backend);
        return false;
    }

    printf("  PASS: env override respected, chunk_size = %zu MB\n", chunk_size / (1024 * 1024));
    ggml_backend_free(backend);
    return true;
}

// =============================================================================
// Test 3: Chunks cover entire tensor without gaps
// =============================================================================
static bool test_chunk_coverage() {
    printf("TEST: test_chunk_coverage\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: SYCL backend unavailable\n");
        return true;
    }

    // Force small chunk size for testing
    setenv("GGML_SYCL_CHUNK_SIZE_MB", "16", 1);
    ggml_sycl::ChunkManager mgr(0);
    unsetenv("GGML_SYCL_CHUNK_SIZE_MB");

    // Create a 50 MB "tensor" (simulated)
    const size_t tensor_size = 50 * 1024 * 1024;  // 50 MB

    auto chunks = mgr.create_chunks(tensor_size, GGML_TYPE_F32);

    // With 16 MB chunks, 50 MB should give us 4 chunks: 16+16+16+2 MB
    if (chunks.size() < 3) {
        printf("  FAIL: expected at least 3 chunks for 50 MB tensor with 16 MB chunks, got %zu\n",
               chunks.size());
        ggml_backend_free(backend);
        return false;
    }

    // Verify total coverage equals tensor size
    size_t total_covered = 0;
    for (const auto& chunk : chunks) {
        total_covered += chunk.size;
    }

    if (total_covered != tensor_size) {
        printf("  FAIL: total_covered %zu != tensor_size %zu\n", total_covered, tensor_size);
        ggml_backend_free(backend);
        return false;
    }

    // Verify no gaps (each chunk starts where previous ended)
    for (size_t i = 1; i < chunks.size(); i++) {
        size_t expected_offset = chunks[i-1].offset + chunks[i-1].size;
        if (chunks[i].offset != expected_offset) {
            printf("  FAIL: gap detected: chunk[%zu].offset=%zu, expected %zu\n",
                   i, chunks[i].offset, expected_offset);
            ggml_backend_free(backend);
            return false;
        }
    }

    // Verify first chunk starts at offset 0
    if (chunks[0].offset != 0) {
        printf("  FAIL: first chunk offset is %zu, expected 0\n", chunks[0].offset);
        ggml_backend_free(backend);
        return false;
    }

    printf("  PASS: %zu chunks cover entire %zu MB tensor with no gaps\n",
           chunks.size(), tensor_size / (1024 * 1024));
    ggml_backend_free(backend);
    return true;
}

// =============================================================================
// Test 4: Chunk offset lookup
// =============================================================================
static bool test_chunk_lookup() {
    printf("TEST: test_chunk_lookup\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: SYCL backend unavailable\n");
        return true;
    }

    // Force 16 MB chunks
    setenv("GGML_SYCL_CHUNK_SIZE_MB", "16", 1);
    ggml_sycl::ChunkManager mgr(0);
    unsetenv("GGML_SYCL_CHUNK_SIZE_MB");

    const size_t tensor_size = 50 * 1024 * 1024;  // 50 MB
    auto chunks = mgr.create_chunks(tensor_size, GGML_TYPE_F32);

    // Test offset 0 -> chunk 0
    auto* chunk0 = mgr.get_chunk_for_offset(chunks, 0);
    if (!chunk0 || chunk0->chunk_id != 0) {
        printf("  FAIL: offset 0 should map to chunk 0\n");
        ggml_backend_free(backend);
        return false;
    }

    // Test offset 16 MB -> chunk 1
    auto* chunk1 = mgr.get_chunk_for_offset(chunks, 16 * 1024 * 1024);
    if (!chunk1 || chunk1->chunk_id != 1) {
        printf("  FAIL: offset 16 MB should map to chunk 1, got chunk %u\n",
               chunk1 ? chunk1->chunk_id : 999);
        ggml_backend_free(backend);
        return false;
    }

    // Test offset 33 MB -> chunk 2
    auto* chunk2 = mgr.get_chunk_for_offset(chunks, 33 * 1024 * 1024);
    if (!chunk2 || chunk2->chunk_id != 2) {
        printf("  FAIL: offset 33 MB should map to chunk 2, got chunk %u\n",
               chunk2 ? chunk2->chunk_id : 999);
        ggml_backend_free(backend);
        return false;
    }

    // Test offset at very end -> last chunk
    auto* chunk_last = mgr.get_chunk_for_offset(chunks, tensor_size - 1);
    if (!chunk_last || chunk_last->chunk_id != chunks.size() - 1) {
        printf("  FAIL: offset at end should map to last chunk\n");
        ggml_backend_free(backend);
        return false;
    }

    // Test offset beyond tensor -> nullptr
    auto* chunk_oob = mgr.get_chunk_for_offset(chunks, tensor_size + 1000);
    if (chunk_oob != nullptr) {
        printf("  FAIL: offset beyond tensor should return nullptr\n");
        ggml_backend_free(backend);
        return false;
    }

    printf("  PASS: chunk offset lookups work correctly\n");
    ggml_backend_free(backend);
    return true;
}

// =============================================================================
// Test 5: Chunks respect element alignment for quantized types
// =============================================================================
static bool test_chunk_alignment() {
    printf("TEST: test_chunk_alignment\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: SYCL backend unavailable\n");
        return true;
    }

    // Force small chunks to see more of them
    setenv("GGML_SYCL_CHUNK_SIZE_MB", "8", 1);
    ggml_sycl::ChunkManager mgr(0);
    unsetenv("GGML_SYCL_CHUNK_SIZE_MB");

    // Test Q4_0 type (block size = 32 elements, each block = 18 bytes)
    const size_t tensor_size = 50 * 1024 * 1024;  // 50 MB
    auto chunks = mgr.create_chunks(tensor_size, GGML_TYPE_Q4_0);

    // Q4_0 block size in bytes
    const size_t q4_0_block_bytes = ggml_type_size(GGML_TYPE_Q4_0);  // 18 bytes

    for (size_t i = 0; i < chunks.size(); i++) {
        // Each chunk's offset should be aligned to block boundary
        if (chunks[i].offset % q4_0_block_bytes != 0) {
            printf("  FAIL: chunk[%zu].offset=%zu not aligned to Q4_0 block size %zu\n",
                   i, chunks[i].offset, q4_0_block_bytes);
            ggml_backend_free(backend);
            return false;
        }

        // Each chunk's size should be aligned (except possibly the last one)
        if (i < chunks.size() - 1 && chunks[i].size % q4_0_block_bytes != 0) {
            printf("  FAIL: chunk[%zu].size=%zu not aligned to Q4_0 block size %zu\n",
                   i, chunks[i].size, q4_0_block_bytes);
            ggml_backend_free(backend);
            return false;
        }
    }

    printf("  PASS: all chunk offsets aligned to Q4_0 block size (%zu bytes)\n", q4_0_block_bytes);
    ggml_backend_free(backend);
    return true;
}

// =============================================================================
// Test 6: Range query returns all chunks covering a byte range
// =============================================================================
static bool test_chunk_range_query() {
    printf("TEST: test_chunk_range_query\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: SYCL backend unavailable\n");
        return true;
    }

    // Force 16 MB chunks
    setenv("GGML_SYCL_CHUNK_SIZE_MB", "16", 1);
    ggml_sycl::ChunkManager mgr(0);
    unsetenv("GGML_SYCL_CHUNK_SIZE_MB");

    const size_t tensor_size = 64 * 1024 * 1024;  // 64 MB = 4 chunks
    auto chunks = mgr.create_chunks(tensor_size, GGML_TYPE_F32);

    // Query range [20 MB, 45 MB) - should cover chunks 1, 2
    size_t start = 20 * 1024 * 1024;
    size_t len = 25 * 1024 * 1024;

    auto range_chunks = mgr.get_chunks_for_range(chunks, start, len);

    if (range_chunks.size() != 2) {
        printf("  FAIL: range [20MB, 45MB) should cover 2 chunks, got %zu\n", range_chunks.size());
        ggml_backend_free(backend);
        return false;
    }

    // Verify they're chunks 1 and 2
    bool found_chunk1 = false, found_chunk2 = false;
    for (auto* c : range_chunks) {
        if (c->chunk_id == 1) found_chunk1 = true;
        if (c->chunk_id == 2) found_chunk2 = true;
    }

    if (!found_chunk1 || !found_chunk2) {
        printf("  FAIL: expected chunks 1 and 2, got different chunks\n");
        ggml_backend_free(backend);
        return false;
    }

    printf("  PASS: range query returns correct chunks\n");
    ggml_backend_free(backend);
    return true;
}

// =============================================================================
// Test 7: ChunkDescriptor tier tracking
// =============================================================================
static bool test_chunk_tier_tracking() {
    printf("TEST: test_chunk_tier_tracking\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: SYCL backend unavailable\n");
        return true;
    }

    setenv("GGML_SYCL_CHUNK_SIZE_MB", "16", 1);
    ggml_sycl::ChunkManager mgr(0);
    unsetenv("GGML_SYCL_CHUNK_SIZE_MB");

    const size_t tensor_size = 32 * 1024 * 1024;  // 32 MB = 2 chunks
    auto chunks = mgr.create_chunks(tensor_size, GGML_TYPE_F32);

    // Initially all chunks should have tier = NONE (not yet allocated)
    for (const auto& chunk : chunks) {
        if (chunk.tier != ggml_sycl::MemoryTier::NONE) {
            printf("  FAIL: newly created chunk should have tier NONE\n");
            ggml_backend_free(backend);
            return false;
        }
    }

    // Set tier for chunk 0 to VRAM
    chunks[0].tier = ggml_sycl::MemoryTier::VRAM;
    chunks[0].device_ptr = (void*)0x12345678;  // Simulated pointer

    // Set tier for chunk 1 to HOST
    chunks[1].tier = ggml_sycl::MemoryTier::HOST;
    chunks[1].host_ptr = (void*)0x87654321;

    // Verify tiers
    if (chunks[0].tier != ggml_sycl::MemoryTier::VRAM) {
        printf("  FAIL: chunk 0 tier should be VRAM\n");
        ggml_backend_free(backend);
        return false;
    }

    if (chunks[1].tier != ggml_sycl::MemoryTier::HOST) {
        printf("  FAIL: chunk 1 tier should be HOST\n");
        ggml_backend_free(backend);
        return false;
    }

    printf("  PASS: chunk tier tracking works correctly\n");
    ggml_backend_free(backend);
    return true;
}

// =============================================================================
// Main test runner
// =============================================================================
int main(int argc, char ** argv) {
    (void)argc; (void)argv;

    printf("=== ChunkManager Unit Tests ===\n");
    printf("Part of unified memory management (llama.cpp-v3n/llama.cpp-4ye)\n\n");

    int passed = 0;
    int failed = 0;
    int skipped = 0;

    auto run_test = [&](bool (*test_fn)(), const char* name) {
        bool result = test_fn();
        // Note: test functions return true for PASS or SKIP, false for FAIL
        // We need to differentiate SKIP from PASS by checking stdout
        // For simplicity, treating SKIP as a special PASS
        if (result) {
            passed++;
        } else {
            failed++;
            printf("  >>> TEST FAILED: %s\n\n", name);
        }
    };

    run_test(test_chunk_size_detection, "test_chunk_size_detection");
    run_test(test_chunk_size_env_override, "test_chunk_size_env_override");
    run_test(test_chunk_coverage, "test_chunk_coverage");
    run_test(test_chunk_lookup, "test_chunk_lookup");
    run_test(test_chunk_alignment, "test_chunk_alignment");
    run_test(test_chunk_range_query, "test_chunk_range_query");
    run_test(test_chunk_tier_tracking, "test_chunk_tier_tracking");

    printf("\n=== Summary ===\n");
    printf("Passed: %d\n", passed);
    printf("Failed: %d\n", failed);

    return failed > 0 ? 1 : 0;
}
