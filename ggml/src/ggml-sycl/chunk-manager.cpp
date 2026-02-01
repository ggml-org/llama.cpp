// Chunk Manager Implementation for sub-tensor streaming
// Part of unified memory management system (epic llama.cpp-v3n, task llama.cpp-4ye)
//
// TDD GREEN phase: Implementation to pass tests in test-sycl-chunk-manager.cpp

#include "chunk-manager.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "ggml-sycl.h"
#include "ggml-sycl/common.hpp"

namespace ggml_sycl {

// Default chunk sizes
static constexpr size_t DEFAULT_CHUNK_SIZE = 64 * 1024 * 1024;  // 64 MB
static constexpr size_t MIN_CHUNK_SIZE = 16 * 1024 * 1024;      // 16 MB
static constexpr size_t MAX_CHUNK_SIZE = 256 * 1024 * 1024;     // 256 MB
static constexpr size_t CACHE_LINE_ALIGNMENT = 64;

// Detect optimal chunk size based on device capabilities
size_t ChunkManager::detect_optimal_chunk_size(int device_id) {
    // Check for environment variable override first
    const char* env_chunk_size = std::getenv("GGML_SYCL_CHUNK_SIZE_MB");
    if (env_chunk_size) {
        size_t mb = static_cast<size_t>(std::atol(env_chunk_size));
        if (mb > 0 && mb <= 512) {
            size_t chunk_size = mb * 1024 * 1024;
            // Ensure alignment
            chunk_size = (chunk_size / CACHE_LINE_ALIGNMENT) * CACHE_LINE_ALIGNMENT;
            return chunk_size;
        }
    }

    // Auto-detect based on device global memory
    try {
        sycl::queue& q = ggml_sycl_get_device(device_id).default_queue();
        size_t global_mem = q.get_device().get_info<sycl::info::device::global_mem_size>();

        // Use ~1/128 of VRAM as chunk size (heuristic)
        // For 16 GB VRAM -> 128 MB chunks
        // For 8 GB VRAM -> 64 MB chunks
        size_t chunk_size = global_mem / 128;

        // Clamp to reasonable bounds
        chunk_size = std::max(chunk_size, MIN_CHUNK_SIZE);
        chunk_size = std::min(chunk_size, MAX_CHUNK_SIZE);

        // Ensure cache-line alignment
        chunk_size = (chunk_size / CACHE_LINE_ALIGNMENT) * CACHE_LINE_ALIGNMENT;

        return chunk_size;
    } catch (...) {
        // Fallback to default if device query fails
        return DEFAULT_CHUNK_SIZE;
    }
}

// Get alignment requirement for a ggml type
size_t ChunkManager::get_type_alignment(ggml_type type) {
    // Return the block size in bytes for quantized types
    // This ensures chunk boundaries don't split quantization blocks
    size_t type_size = ggml_type_size(type);
    size_t block_size = ggml_blck_size(type);

    if (block_size > 1) {
        // Quantized type: align to block boundary
        return type_size;  // type_size is already per-block for quantized types
    } else {
        // Non-quantized type: align to element size (min 64 for cache line)
        return std::max(type_size, CACHE_LINE_ALIGNMENT);
    }
}

// Constructor
ChunkManager::ChunkManager(int device_id)
    : device_id_(device_id)
    , chunk_size_(detect_optimal_chunk_size(device_id))
    , alignment_(CACHE_LINE_ALIGNMENT) {
}

// Destructor
ChunkManager::~ChunkManager() = default;

// Get current chunk size
size_t ChunkManager::get_chunk_size() const {
    return chunk_size_;
}

// Set chunk size (for testing or manual override)
void ChunkManager::set_chunk_size(size_t size) {
    // Ensure alignment
    chunk_size_ = (size / CACHE_LINE_ALIGNMENT) * CACHE_LINE_ALIGNMENT;
    if (chunk_size_ == 0) {
        chunk_size_ = CACHE_LINE_ALIGNMENT;
    }
}

// Create chunks for a tensor
std::vector<ChunkDescriptor> ChunkManager::create_chunks(size_t tensor_size, ggml_type type) {
    std::vector<ChunkDescriptor> chunks;

    if (tensor_size == 0) {
        return chunks;
    }

    // Get type-specific alignment
    size_t type_align = get_type_alignment(type);

    // Adjust chunk size to be aligned to type boundaries
    size_t aligned_chunk_size = chunk_size_;
    if (type_align > 1) {
        aligned_chunk_size = (chunk_size_ / type_align) * type_align;
        if (aligned_chunk_size == 0) {
            aligned_chunk_size = type_align;
        }
    }

    size_t offset = 0;
    uint32_t chunk_id = 0;

    while (offset < tensor_size) {
        ChunkDescriptor chunk;
        chunk.offset = offset;
        chunk.chunk_id = chunk_id;
        chunk.tier = MemoryTier::NONE;
        chunk.device_ptr = nullptr;
        chunk.host_ptr = nullptr;
        chunk.mmap_ptr = nullptr;
        chunk.last_access = 0;
        chunk.pinned = false;

        // Calculate chunk size
        size_t remaining = tensor_size - offset;
        if (remaining <= aligned_chunk_size) {
            // Last chunk - may be smaller
            chunk.size = remaining;
        } else {
            chunk.size = aligned_chunk_size;
        }

        chunks.push_back(chunk);
        offset += chunk.size;
        chunk_id++;
    }

    return chunks;
}

// Get chunk for a given byte offset
ChunkDescriptor* ChunkManager::get_chunk_for_offset(
    std::vector<ChunkDescriptor>& chunks,
    size_t offset) {

    for (auto& chunk : chunks) {
        if (offset >= chunk.offset && offset < chunk.offset + chunk.size) {
            return &chunk;
        }
    }
    return nullptr;
}

// Get chunk for a given byte offset (const version)
const ChunkDescriptor* ChunkManager::get_chunk_for_offset(
    const std::vector<ChunkDescriptor>& chunks,
    size_t offset) const {

    for (const auto& chunk : chunks) {
        if (offset >= chunk.offset && offset < chunk.offset + chunk.size) {
            return &chunk;
        }
    }
    return nullptr;
}

// Get all chunks overlapping a byte range
std::vector<ChunkDescriptor*> ChunkManager::get_chunks_for_range(
    std::vector<ChunkDescriptor>& chunks,
    size_t start,
    size_t length) {

    std::vector<ChunkDescriptor*> result;
    size_t end = start + length;

    for (auto& chunk : chunks) {
        size_t chunk_end = chunk.offset + chunk.size;

        // Check if chunk overlaps with range [start, end)
        // Overlap exists if: chunk_start < range_end AND chunk_end > range_start
        if (chunk.offset < end && chunk_end > start) {
            result.push_back(&chunk);
        }
    }

    return result;
}

} // namespace ggml_sycl
