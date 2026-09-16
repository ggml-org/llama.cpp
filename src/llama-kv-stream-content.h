#pragma once

#include "llama-kv-stream-host.h"

#include <functional>
#include <vector>

struct llama_kv_stream_content_state;

// Offsets and sizes refer to encoded bytes in one K or V plane, not float elements.
struct llama_kv_stream_write_span {
    uint32_t layer = 0;
    ggml_kv_stream_operand operand = ggml_kv_stream_operand::none;
    size_t offset = 0;
    const void * data = nullptr;
    size_t bytes = 0;
};

// One token row contains all KV heads. Ranges can stop inside a cache page.
struct llama_kv_stream_rows {
    uint32_t layer = 0;
    ggml_kv_stream_operand operand = ggml_kv_stream_operand::none;
    size_t first = 0, count = 0;
};
struct llama_kv_stream_copy_span {
    llama_kv_stream_rows rows;
    uint64_t cache_id = 0, generation = 0, mirror_epoch = 0;
    const void * data = nullptr;
    size_t bytes = 0;
};

// Move-only encoded snapshot. Cancellation or destruction changes no authoritative bytes.
class llama_kv_stream_write {
public:
    llama_kv_stream_write() = default;
    llama_kv_stream_write(llama_kv_stream_write &&) noexcept = default;
    llama_kv_stream_write & operator=(llama_kv_stream_write &&) noexcept = default;
    llama_kv_stream_write(const llama_kv_stream_write &) = delete;
    llama_kv_stream_write & operator=(const llama_kv_stream_write &) = delete;
    bool pending() const noexcept;
    void cancel() noexcept;

private:
    friend class llama_kv_stream_content;
    struct part { uint32_t layer; ggml_kv_stream_operand operand; size_t offset, begin, bytes; };
    std::shared_ptr<llama_kv_stream_content_state> owner;
    std::shared_ptr<llama_kv_stream_host> backing;
    uint64_t generation = 0;
    std::vector<part> parts;
    std::vector<uint8_t> bytes;
    bool direct = false;
};

// Owner-thread-only bookkeeping for one authoritative host cache and one logical device mirror.
// Use one mutation authority per backing; independent trackers over aliased bytes are not coherent.
// Raw writes through host pointers require invalidate(); callers must gate concurrent CPU/GPU access.
class llama_kv_stream_content {
public:
    explicit llama_kv_stream_content(std::shared_ptr<llama_kv_stream_host> host);
    llama_kv_stream_content(const llama_kv_stream_content &) = delete;
    llama_kv_stream_content & operator=(const llama_kv_stream_content &) = delete;

    std::shared_ptr<llama_kv_stream_host> host() const noexcept;
    uint64_t generation() const noexcept;
    uint64_t mirror_epoch() const noexcept;
    // Invalid coordinates return false without changing output.
    bool dirty(const llama_kv_stream_rows & rows, bool & output) const noexcept;

    // Snapshot all sources before mutation; overlapping writes commit in input order.
    // Failure preserves output. Large restores should use bounded batches rather than duplicate the full cache.
    bool prepare(const std::vector<llama_kv_stream_write_span> & spans, llama_kv_stream_write & output) const;
    // Reject cancelled, foreign, or stale snapshots; successful nonempty commit advances content generation once.
    bool commit(llama_kv_stream_write & write);

    // Fill a private ticket synchronously (span.data must be null). Failed generation never publishes host bytes.
    bool prepare_generated(const std::vector<llama_kv_stream_write_span> & spans,
            const std::function<bool(const llama_kv_stream_write_span &, void *)> & fill, llama_kv_stream_write & output) const;
    // Fill retained authoritative storage directly but defer generation and dirty-row visibility until commit.
    // Cancellation can leave bytes beyond the caller's logical token frontier physically changed.
    bool prepare_direct_generated(const std::vector<llama_kv_stream_write_span> & spans,
            const std::function<bool(const llama_kv_stream_write_span &, void *)> & fill, llama_kv_stream_write & output) const;

    // Preserve new backing bytes but invalidate all mirror rows and outstanding writes, even for the same cache ID.
    bool replace(std::shared_ptr<llama_kv_stream_host> host);
    // Report externally completed host changes; no logical token validity is inferred.
    bool invalidate();
    // Call on every mirror rebind/repartition, even if addresses and arena generation happen to match.
    // Does not invalidate a pending host write; content and device-layout lifecycles are separate.
    bool reset_mirror();

    // Copy only dirty runs in the selected ranges. Callback must finish all its accesses before returning/throwing.
    // Any failure retains all dirty marks; partially copied device data must not be used for attention until retry.
    // Tracked mutation/reentrant flush is blocked during callbacks. A copy callback must not destroy this object.
    bool flush(const std::vector<llama_kv_stream_rows> & rows,
            const std::function<bool(const llama_kv_stream_copy_span &)> & copy);

private:
    std::shared_ptr<llama_kv_stream_content_state> state;
    bool prepare_internal(const std::vector<llama_kv_stream_write_span> & spans, llama_kv_stream_write & output,
            const std::function<bool(const llama_kv_stream_write_span &, void *)> * fill) const;
};
