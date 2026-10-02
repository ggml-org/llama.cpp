#pragma once

#include "mtmd.h"

#include <memory>
#include <vector>

struct mtmd_embedding_storage;

// Host-only read view. It retains the whole encoded batch, not its scheduler or input chunks.
class MTMD_API mtmd_embedding_view {
public:
    const float * data() const noexcept;
    size_t n_tokens() const noexcept;
    size_t n_embd() const noexcept;
    // Slice token rows without copying; rejection leaves output unchanged.
    bool slice(size_t first, size_t count, mtmd_embedding_view & output) const noexcept;

private:
    friend class mtmd_embedding_output;
    std::shared_ptr<const mtmd_embedding_storage> storage;
    size_t offset = 0;
    size_t tokens = 0;
};

struct mtmd_embedding_chunk {
    const mtmd_input_chunk * chunk;
    size_t tokens;
};

// Publish only complete host results. Existing readers retain the prior generation.
class MTMD_API mtmd_embedding_output {
public:
    // Validate ranges before moving values; failure preserves values and the last output.
    bool publish(const std::vector<mtmd_embedding_chunk> & chunks, size_t n_embd, std::vector<float> & values);
    bool acquire(const mtmd_input_chunk * chunk, mtmd_embedding_view & output) const noexcept;
    // Legacy mutable borrow, valid until replacement, clear, or batch destruction.
    float * borrow(const mtmd_input_chunk * chunk) const noexcept;
    void clear() noexcept;

private:
    std::shared_ptr<mtmd_embedding_storage> storage;
};

// Internal retained-output seam; caller still owns chunk metadata used for positions.
MTMD_API bool mtmd_batch_acquire_output_embd(const mtmd_batch * batch, const mtmd_input_chunk * chunk,
    mtmd_embedding_view & output) noexcept;
MTMD_API void mtmd_batch_clear_output_embd(mtmd_batch * batch) noexcept;

// Check media compatibility and the model's batching limit without changing the batch.
MTMD_API int32_t mtmd_batch_validate_chunk(const std::vector<const mtmd_input_chunk *> & entries,
    const mtmd_input_chunk * chunk, bool support_batch, size_t max_tokens) noexcept;
