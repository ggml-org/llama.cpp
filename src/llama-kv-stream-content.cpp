#include "llama-kv-stream-content.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

// Publish a generated ticket only after all synchronous fills complete successfully.
bool llama_kv_stream_content::prepare_generated(const std::vector<llama_kv_stream_write_span> & spans,
        const std::function<bool(const llama_kv_stream_write_span &, void *)> & fill, llama_kv_stream_write & output) const {
    return fill && prepare_internal(spans, output, &fill);
}

struct llama_kv_stream_content_state {
    std::shared_ptr<llama_kv_stream_host> host;
    std::vector<uint64_t> dirty;
    size_t words = 0;
    uint64_t generation = 1, mirror_epoch = 1;
    bool busy = false;
};
struct content_operation {
    bool & busy;
    // Restore admission after copy failure or an exception.
    ~content_operation() { busy = false; }
};

// Bound the compact per-plane bitmap independently of backend storage sizes.
static bool bitmap(const llama_kv_stream_host & host, size_t & words, std::vector<uint64_t> & bits) {
    const size_t rows = host.layout().tokens;
    words = rows / 64 + (rows % 64 != 0);
    const size_t layers = host.config().layers;
    if (!words || !layers || words > bits.max_size()/2/layers) return false;
    bits.assign(words*2*layers, UINT64_MAX);
    return true;
}

// Reject invalid selectors before deriving plane offsets.
static bool plane_valid(const llama_kv_stream_content_state & s, uint32_t layer, ggml_kv_stream_operand operand) {
    return layer < s.host->config().layers && (operand == ggml_kv_stream_operand::k || operand == ggml_kv_stream_operand::v);
}
// A tracked row includes all heads in the selected encoded plane.
static size_t token_bytes(const llama_kv_stream_content_state & s, ggml_kv_stream_operand operand) {
    return operand == ggml_kv_stream_operand::k ? s.host->layout().k_token_bytes : s.host->layout().v_token_bytes;
}
// Keep each layer's K/V bitmaps separate, including their padded final words.
static size_t plane_index(const llama_kv_stream_content_state & s, uint32_t layer, ggml_kv_stream_operand operand) {
    return (size_t(layer)*2 + (operand == ggml_kv_stream_operand::v))*s.words;
}
// Resolve a validated plane only while its authoritative owner is retained.
static uint8_t * plane_data(const llama_kv_stream_content_state & s, uint32_t layer, ggml_kv_stream_operand operand) {
    llama_kv_stream_host_layer data;
    GGML_ASSERT(s.host->layer(layer, data));
    return static_cast<uint8_t *>(operand == ggml_kv_stream_operand::k ? data.k : data.v);
}

// Subtraction-based checks avoid overflowing first + count.
static bool rows_valid(const llama_kv_stream_content_state & s, const llama_kv_stream_rows & r) {
    return plane_valid(s, r.layer, r.operand) && r.first <= s.host->layout().tokens &&
        r.count <= s.host->layout().tokens - r.first;
}
// A full-word mask must not evaluate an undefined shift by 64.
static uint64_t low_bits(size_t n) { return n == 64 ? UINT64_MAX : (uint64_t(1) << n) - 1; }

// Update whole bitmap words, handling partial first/last words without shifting by 64.
static void mark(llama_kv_stream_content_state & s, const llama_kv_stream_rows & r, bool dirty) {
    const size_t base = plane_index(s, r.layer, r.operand);
    for (size_t row = r.first, end = r.first + r.count; row < end;) {
        const size_t n = std::min(size_t(64) - row % 64, end - row);
        const uint64_t mask = low_bits(n) << (row % 64);
        auto & word = s.dirty[base + row / 64];
        if (dirty) word |= mask; else word &= ~mask;
        row += n;
    }
}

// Skip clean/full words; scalar bit search is bounded to one word per run edge.
static size_t find_bit(const llama_kv_stream_content_state & s, const llama_kv_stream_rows & r,
        size_t first, bool dirty) {
    const size_t end = r.first + r.count;
    const size_t base = plane_index(s, r.layer, r.operand);
    for (size_t row = first; row < end;) {
        const size_t n = std::min(size_t(64) - row % 64, end - row);
        uint64_t word = s.dirty[base + row / 64];
        if (!dirty) word = ~word;
        word = (word >> (row % 64)) & low_bits(n);
        if (word) {
            size_t bit = 0;
            while (!(word & 1)) { word >>= 1; ++bit; }
            return row + bit;
        }
        row += n;
    }
    return end;
}

// One tracker must be the mutation authority for this backing; import aliases need external coordination.
llama_kv_stream_content::llama_kv_stream_content(std::shared_ptr<llama_kv_stream_host> host) :
    state(std::make_shared<llama_kv_stream_content_state>()) {
    if (!host || !bitmap(*host, state->words, state->dirty)) throw std::invalid_argument("invalid KV content backing");
    state->host = std::move(host);
}

// Retain backing independently of any device binding or arena generation.
std::shared_ptr<llama_kv_stream_host> llama_kv_stream_content::host() const noexcept { return state->host; }
uint64_t llama_kv_stream_content::generation() const noexcept { return state->generation; }
uint64_t llama_kv_stream_content::mirror_epoch() const noexcept { return state->mirror_epoch; }

// Dirty means not acknowledged in this logical mirror; it does not imply logical token validity.
bool llama_kv_stream_content::dirty(const llama_kv_stream_rows & rows, bool & output) const noexcept {
    if (!rows_valid(*state, rows)) return false;
    output = find_bit(*state, rows, rows.first, true) != rows.first + rows.count;
    return true;
}

// Validate the entire batch and snapshot sources before changing the caller's pending ticket.
bool llama_kv_stream_content::prepare(const std::vector<llama_kv_stream_write_span> & spans, llama_kv_stream_write & output) const {
    return prepare_internal(spans, output, nullptr);
}
bool llama_kv_stream_content::prepare_direct_generated(const std::vector<llama_kv_stream_write_span> & spans,
        const std::function<bool(const llama_kv_stream_write_span &, void *)> & fill, llama_kv_stream_write & output) const {
    if (!fill || state->busy || output.pending()) return false;
    state->busy = true;
    content_operation operation{state->busy};
    try {
        llama_kv_stream_write next;
        if (spans.size() > next.parts.max_size()) return false;
        next.parts.reserve(spans.size());
        for (const auto & span : spans) {
            if (!plane_valid(*state,span.layer,span.operand)) return false;
            const size_t bytes=span.operand == ggml_kv_stream_operand::k ? state->host->layout().k_bytes : state->host->layout().v_bytes;
            if (span.offset > bytes || span.bytes > bytes-span.offset || span.data || !span.bytes) return false;
        }
        next.owner=state;
        next.backing=state->host;
        next.generation=state->generation;
        next.direct=true;
        for (const auto & span : spans) {
            void * destination=plane_data(*state,span.layer,span.operand)+span.offset;
            if (!fill(span,destination)) return false;
            next.parts.push_back({span.layer,span.operand,span.offset,0,span.bytes});
        }
        output=std::move(next);
        return true;
    } catch (const std::bad_alloc &) {
        return false;
    }
}


// Both CPU snapshots and generated payloads share validation, ownership, and atomic publication.
bool llama_kv_stream_content::prepare_internal(const std::vector<llama_kv_stream_write_span> & spans, llama_kv_stream_write & output,
        const std::function<bool(const llama_kv_stream_write_span &, void *)> * fill) const {
    if (state->busy) return false;
    state->busy = true;
    content_operation operation{state->busy};
    try {
        llama_kv_stream_write next;
        size_t total = 0;
        for (const auto & span : spans) {
            if (!plane_valid(*state, span.layer, span.operand)) return false;
            const size_t bytes = span.operand == ggml_kv_stream_operand::k ? state->host->layout().k_bytes : state->host->layout().v_bytes;
            if (span.offset > bytes || span.bytes > bytes - span.offset || (span.bytes && !span.data && !fill) || (fill && span.data) ||
                    span.bytes > next.bytes.max_size() - total) return false;
            total += span.bytes;
        }
        if (spans.size() > next.parts.max_size()) return false;
        next.bytes.resize(total);
        next.parts.reserve(spans.size());
        size_t begin = 0;
        for (const auto & span : spans) {
            if (!span.bytes) continue;
            if (fill) {
                if (!(*fill)(span, next.bytes.data() + begin)) return false;
            } else {
                std::memcpy(next.bytes.data() + begin, span.data, span.bytes);
            }
            next.parts.push_back({span.layer, span.operand, span.offset, begin, span.bytes});
            begin += span.bytes;
        }
        next.owner = state;
        next.generation = state->generation;
        output = std::move(next);
        return true;
    } catch (const std::bad_alloc &) {
        return false;
    }
}

// Byte patches can intersect only part of a quant block; the entire touched token row becomes dirty.
bool llama_kv_stream_content::commit(llama_kv_stream_write & write) {
    if (state->busy || write.owner != state || write.generation != state->generation ||
            (write.direct && write.backing != state->host)) return false;
    state->busy = true;
    content_operation operation{state->busy};
    if (!write.parts.empty()) {
        if (state->generation == UINT64_MAX) return false;
        for (const auto & part : write.parts) {
            if (!write.direct) std::memcpy(plane_data(*state,part.layer,part.operand)+part.offset,
                write.bytes.data()+part.begin,part.bytes);
            const size_t stride = token_bytes(*state, part.operand);
            const size_t first = part.offset / stride;
            const size_t last = (part.offset + part.bytes - 1) / stride;
            mark(*state, {part.layer, part.operand, first, last - first + 1}, true);
        }
        ++state->generation;
    }
    write.cancel();
    return true;
}

// Prepare the new bitmap first; failed replacement leaves backing and pending writes untouched.
bool llama_kv_stream_content::replace(std::shared_ptr<llama_kv_stream_host> host) {
    if (state->busy || !host || state->generation == UINT64_MAX || state->mirror_epoch == UINT64_MAX) return false;
    state->busy = true;
    content_operation operation{state->busy};
    try {
        size_t words;
        std::vector<uint64_t> bits;
        if (!bitmap(*host, words, bits)) return false;
        state->host.swap(host); // release old backing only after all new metadata is published
        state->dirty = std::move(bits);
        state->words = words;
        ++state->generation;
        ++state->mirror_epoch;
        return true;
    } catch (const std::bad_alloc &) {
        return false;
    }
}

// External host changes supersede pending snapshots but do not change physical mirror layout.
bool llama_kv_stream_content::invalidate() {
    if (state->busy || state->generation == UINT64_MAX) return false;
    ++state->generation;
    std::fill(state->dirty.begin(), state->dirty.end(), UINT64_MAX);
    return true;
}

// Even an unchanged pointer can represent a newly assigned resident/ring location.
bool llama_kv_stream_content::reset_mirror() {
    if (state->busy || state->mirror_epoch == UINT64_MAX) return false;
    ++state->mirror_epoch;
    std::fill(state->dirty.begin(), state->dirty.end(), UINT64_MAX);
    return true;
}

// This correctness baseline never acknowledges data while a copy can still reference it.
bool llama_kv_stream_content::flush(const std::vector<llama_kv_stream_rows> & ranges,
        const std::function<bool(const llama_kv_stream_copy_span &)> & copy) {
    if (state->busy) return false;
    for (const auto & r : ranges) if (!rows_valid(*state, r)) return false;
    state->busy = true;
    content_operation operation{state->busy};
    for (const auto & r : ranges) {
        const size_t end = r.first + r.count;
        size_t first = find_bit(*state, r, r.first, true);
        while (first < end) {
            if (!copy) return false;
            const size_t last = find_bit(*state, r, first, false);
            const size_t stride = token_bytes(*state, r.operand);
            const llama_kv_stream_copy_span span{{r.layer, r.operand, first, last - first},
                state->host->cache_id(), state->generation, state->mirror_epoch,
                plane_data(*state, r.layer, r.operand) + first*stride, (last - first)*stride};
            if (!copy(span)) return false;
            first = find_bit(*state, r, last, true);
        }
    }
    for (const auto & r : ranges) mark(*state, r, false);
    return true;
}

// A cancelled/moved/consumed ticket cannot change host bytes.
bool llama_kv_stream_write::pending() const noexcept { return owner != nullptr; }
// Drop encoded payload and retained state, not just a validity flag.
void llama_kv_stream_write::cancel() noexcept { *this = llama_kv_stream_write{}; }
