#include "llama-kv-stream-positions.h"

#include <algorithm>
#include <limits>

bool llama_kv_stream_validate_append(const llama_batch & batch, const llama_kv_stream_append_coordinates & c) noexcept {
    if (batch.n_tokens <= 0 || bool(batch.token) == bool(batch.embd) || c.previous_position < -1 ||
            (c.position_channels != 1 && c.position_channels != 4) ||
            c.first_cell > c.capacity || size_t(batch.n_tokens) > c.capacity - c.first_cell ||
            (batch.seq_id && !batch.n_seq_id)) return false;
    const bool image = batch.embd != nullptr;
    if (image && (!batch.pos || c.decode || c.linear_required)) return false;
    // The batch allocator increments its implicit position cursor after the final row too.
    if (!batch.pos && int64_t(c.previous_position) + batch.n_tokens >= std::numeric_limits<llama_pos>::max()) return false;
    int64_t previous = c.previous_position;
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if ((batch.n_seq_id && batch.n_seq_id[i] != 1) ||
                (batch.seq_id && (!batch.seq_id[i] || batch.seq_id[i][0] != 0))) return false;
        const int64_t position = batch.pos ? batch.pos[i] : int64_t(c.previous_position) + i + 1;
        if (position < 0 || (image && c.position_channels > 1 ? position < previous : position <= previous) ||
                (c.linear_required && position != int64_t(c.first_cell) + i) ||
                (c.position_channels == 1 && position != previous + 1)) return false;
        if (image) for (uint32_t channel = 1; channel < c.position_channels; ++channel) {
            if (batch.pos[size_t(channel) * size_t(batch.n_tokens) + size_t(i)] < 0) return false;
        }
        previous = position;
    }
    return true;
}

bool llama_kv_stream_validate_cells(const llama_kv_cells & cells, size_t count, bool linear_required) noexcept {
    if (count > cells.size() || cells.get_used() != count || cells.used_max_p1() != count) return false;
    llama_pos previous = -1;
    for (size_t i = 0; i < count; ++i) {
        if (cells.is_empty(uint32_t(i)) || cells.seq_count(uint32_t(i)) != 1 || !cells.seq_has(uint32_t(i), 0)) return false;
        const auto position = cells.pos_get(uint32_t(i));
        const auto spatial = cells.ext_get(uint32_t(i));
        if (position < 0 || position < previous || spatial.x < 0 || spatial.y < 0 ||
                (linear_required && int64_t(position) != int64_t(i))) return false;
        previous = position;
    }
    return true;
}

bool llama_kv_stream_suffix(const llama_kv_cells & cells, size_t count,
        llama_pos begin, llama_pos end, size_t & retained) noexcept {
    if (begin <= 0 && end < 0) { retained = 0; return true; }
    if (count > cells.size() || cells.get_used() != count || cells.used_max_p1() != count) return false;
    if (!count) { retained = 0; return true; }
    if (cells.seq_count(0) != 1 || !cells.seq_has(0,0) || cells.seq_count(uint32_t(count-1)) != 1 ||
            !cells.seq_has(uint32_t(count-1),0) || cells.pos_get(0) < 0 ||
            cells.pos_get(uint32_t(count-1)) < cells.pos_get(0)) return false;
    begin = std::max(llama_pos(0), begin);
    if (end >= 0 && end <= begin) { retained = count; return true; }
    const auto lower = [&](llama_pos position) {
        size_t first = 0, last = count;
        while (first < last) {
            const size_t middle = first + (last-first)/2;
            if (cells.pos_get(uint32_t(middle)) < position) first = middle+1;
            else last = middle;
        }
        return first;
    };
    const size_t first = lower(begin), last = end < 0 ? count : lower(end);
    if (first != last && last != count) return false;
    retained = first == last ? count : first;
    return true;
}
