// Chunk Manager for sub-tensor streaming
// Part of unified memory management system (epic llama.cpp-v3n, task llama.cpp-4ye)
//
// This header defines the interface. Implementation in chunk-manager.cpp.
// TDD: Interface defined by tests in test-sycl-chunk-manager.cpp

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ggml.h"

namespace ggml_sycl {

// Memory tier enumeration
enum class MemoryTier : uint8_t {
    NONE = 0,   // Not yet allocated
    VRAM = 1,   // GPU device memory
    HOST = 2,   // Pinned host memory
    MMAP = 3,   // Memory-mapped from disk
};

// Describes a single chunk of a tensor
struct ChunkDescriptor {
    size_t offset;          // Byte offset within tensor
    size_t size;            // Chunk size in bytes
    uint32_t chunk_id;      // Unique ID within tensor (0, 1, 2, ...)

    // Current memory tier
    MemoryTier tier;

    // Pointers (only one non-null based on tier)
    void* device_ptr;       // GPU pointer (if tier == VRAM)
    void* host_ptr;         // Pinned host pointer (if tier == HOST)
    void* mmap_ptr;         // MMAP pointer (if tier == MMAP)

    // Metadata for cache management
    uint64_t last_access;   // Timestamp for LRU
    bool pinned;            // If true, cannot be evicted

    // Default constructor
    ChunkDescriptor()
        : offset(0)
        , size(0)
        , chunk_id(0)
        , tier(MemoryTier::NONE)
        , device_ptr(nullptr)
        , host_ptr(nullptr)
        , mmap_ptr(nullptr)
        , last_access(0)
        , pinned(false) {}
};

// Manages chunking of tensors for sub-tensor streaming
class ChunkManager {
public:
    // Constructor - auto-detects optimal chunk size for device
    explicit ChunkManager(int device_id);

    // Destructor
    ~ChunkManager();

    // Get current chunk size in bytes
    size_t get_chunk_size() const;

    // Set chunk size (for testing or manual override)
    void set_chunk_size(size_t size);

    // Create chunks for a tensor of given size and type
    // Returns vector of ChunkDescriptors covering the entire tensor
    std::vector<ChunkDescriptor> create_chunks(size_t tensor_size, ggml_type type);

    // Get the chunk containing a given byte offset
    // Returns nullptr if offset is out of range
    ChunkDescriptor* get_chunk_for_offset(std::vector<ChunkDescriptor>& chunks, size_t offset);
    const ChunkDescriptor* get_chunk_for_offset(const std::vector<ChunkDescriptor>& chunks, size_t offset) const;

    // Get all chunks that overlap with byte range [start, start+length)
    std::vector<ChunkDescriptor*> get_chunks_for_range(
        std::vector<ChunkDescriptor>& chunks,
        size_t start,
        size_t length);

private:
    int device_id_;
    size_t chunk_size_;
    size_t alignment_;

    // Detect optimal chunk size based on device capabilities
    static size_t detect_optimal_chunk_size(int device_id);

    // Get alignment requirement for a ggml type
    static size_t get_type_alignment(ggml_type type);
};

} // namespace ggml_sycl
