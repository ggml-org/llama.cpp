#include "mtmd-embeddings.h"

#include <new>
#include <utility>

struct mtmd_embedding_storage {
    struct range {
        const mtmd_input_chunk * chunk;
        size_t offset;
        size_t tokens;
    };
    size_t n_embd;
    std::vector<float> values;
    std::vector<range> ranges;
};

// Empty views do not expose a host address.
const float * mtmd_embedding_view::data() const noexcept {
    return storage ? storage->values.data() + offset : nullptr;
}

size_t mtmd_embedding_view::n_tokens() const noexcept { return storage ? tokens : 0; }
size_t mtmd_embedding_view::n_embd() const noexcept { return storage ? storage->n_embd : 0; }

// Token bounds already imply that the element offset fits the validated storage.
bool mtmd_embedding_view::slice(size_t first, size_t count, mtmd_embedding_view & output) const noexcept {
    if (!storage || !count || first > tokens || count > tokens - first) return false;
    auto next = *this;
    next.offset += first * storage->n_embd;
    next.tokens = count;
    output = std::move(next);
    return true;
}

// Validate every row range before adopting data; readers of an older result remain independent.
bool mtmd_embedding_output::publish(const std::vector<mtmd_embedding_chunk> & chunks, size_t n_embd,
        std::vector<float> & values) {
    if (chunks.empty() || !n_embd) return false;
    size_t elements = 0;
    for (const auto & chunk : chunks) {
        if (!chunk.chunk || !chunk.tokens || chunk.tokens > (SIZE_MAX - elements) / n_embd) return false;
        elements += chunk.tokens * n_embd;
    }
    if (elements != values.size()) return false;
    try {
        auto next = std::make_shared<mtmd_embedding_storage>();
        next->n_embd = n_embd;
        next->ranges.reserve(chunks.size());
        size_t offset = 0;
        for (const auto & chunk : chunks) {
            next->ranges.push_back({chunk.chunk, offset, chunk.tokens});
            offset += chunk.tokens * n_embd;
        }
        next->values.swap(values);
        storage = std::move(next);
        return true;
    } catch (const std::bad_alloc &) { return false; }
}

// Identity lookup never dereferences caller-owned chunk metadata after encoding.
bool mtmd_embedding_output::acquire(const mtmd_input_chunk * chunk, mtmd_embedding_view & output) const noexcept {
    if (!storage || !chunk) return false;
    for (const auto & range : storage->ranges) {
        if (range.chunk != chunk) continue;
        mtmd_embedding_view next;
        next.storage = storage;
        next.offset = range.offset;
        next.tokens = range.tokens;
        output = std::move(next);
        return true;
    }
    return false;
}

// Preserve the existing mutable-pointer ABI; retained readers treat embeddings as read-only.
float * mtmd_embedding_output::borrow(const mtmd_input_chunk * chunk) const noexcept {
    mtmd_embedding_view view;
    return acquire(chunk, view) ? const_cast<float *>(view.data()) : nullptr;
}

void mtmd_embedding_output::clear() noexcept { storage.reset(); }
