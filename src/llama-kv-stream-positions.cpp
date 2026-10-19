#include "llama-kv-stream-positions.h"
#include "llama-batch.h"

#include <algorithm>
#include <limits>
#include "../ggml/src/ggml-backend-execution.h"

bool llama_kv_stream_validate_append(const llama_batch_ext & batch,
        const llama_kv_stream_append_coordinates & c) noexcept {
    if (batch.tokens.empty() || batch.tokens.size() > INT32_MAX || c.previous_position < -1 ||
            (c.position_channels != 1 && c.position_channels != 4) ||
            c.first_cell > c.capacity || batch.tokens.size() > c.capacity-c.first_cell) return false;
    int64_t previous = c.previous_position;
    for (size_t i = 0; i < batch.tokens.size(); ++i) {
        const auto & token = batch.tokens[i];
        const bool image = token.has_embd;
        if ((token.id != LLAMA_TOKEN_NULL) == image || token.seq_ids.size() != 1 || !token.seq_ids.count(0)) return false;
        if (image) {
            if (c.decode || c.linear_required || !batch.n_embd || token.embd_off > batch.embd.size() ||
                    batch.n_embd > batch.embd.size()-token.embd_off) return false;
        } else if (token.id < 0 || token.id >= batch.n_vocab) return false;
        int64_t position = token.pos[0];
        if (position == -1 && !image) {
            if (previous >= std::numeric_limits<llama_pos>::max()-1) return false;
            position = previous+1;
        }
        if (position < 0 || (image && c.position_channels > 1 ? position < previous : position <= previous) ||
                (c.linear_required && position != int64_t(c.first_cell)+int64_t(i)) ||
                (c.position_channels == 1 && position != previous+1)) return false;
        if (image) for (uint32_t channel = 1; channel < c.position_channels; ++channel)
            if (token.pos[channel] < 0) return false;
        previous = position;
    }
    return true;
}

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

bool llama_kv_stream_bounded_contains(const llama_kv_cells & source,size_t first,size_t end,
        const llama_kv_cells & local,bool match_extended) noexcept {
    if (first > end || end > source.size() || end-first > local.get_used()) return false;
    if (first == end) return true;
    if (source.is_empty(uint32_t(first))) return false;
    const auto & positions=local.seq_pos_get(0);
    auto found=positions.lower_bound({source.pos_get(uint32_t(first)),0});
    for (size_t index=first;index<end;++index,++found) {
        const auto row=uint32_t(index);
        if (source.is_empty(row) || source.seq_count(row) != 1 || !source.seq_has(row,0) ||
                found == positions.end() || found->first != source.pos_get(row) || local.seq_count(found->second) != 1) return false;
        if (match_extended) {
            const auto a=source.ext_get(row), b=local.ext_get(found->second);
            if (a.x != b.x || a.y != b.y || a.tok != b.tok) return false;
        }
    }
    return true;
}

bool llama_kv_stream_mask_bounds(const llama_kv_cells & cells, size_t count, size_t padded,
        const llama_ubatch & batch, bool causal, std::vector<int32_t> & bounds) {
    if (!count || padded < count || padded > INT32_MAX || !batch.n_tokens || batch.n_tokens > INT32_MAX/4 ||
            !batch.pos || !batch.seq_id || !batch.n_seq_id || batch.n_seqs_unq != 1 ||
            !batch.seq_id_unq || batch.seq_id_unq[0] != 0 ||
            (batch.n_pos != 1 && batch.n_pos != 4) || !llama_kv_stream_validate_cells(cells,count,false)) return false;
    std::vector<int32_t> next;
    try { next.resize(size_t(batch.n_tokens)*4); }
    catch (const std::bad_alloc &) { return false; }
    for (uint32_t i=0; i<batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] != 1 || !batch.seq_id[i] || batch.seq_id[i][0] != 0 || batch.pos[i] < 0) return false;
        const auto position=batch.pos[i];
        const bool spatial=causal && batch.is_pos_2d();
        const auto x=spatial ? batch.pos[i+size_t(batch.n_tokens)*2] : 0;
        const auto y=spatial ? batch.pos[i+size_t(batch.n_tokens)] : 0;
        if (spatial && (x < 0 || y < 0)) return false;
        size_t first=0, last=count;
        if (causal) {
            while (first < last) {
                const size_t middle=first+(last-first)/2;
                if (cells.pos_get(uint32_t(middle)) <= position) first=middle+1;
                else last=middle;
            }
        } else first=count;
        size_t end=first;
        if (spatial) {
            size_t equal=first;
            while (equal && cells.pos_get(uint32_t(equal-1)) == position) --equal;
            bool dropped=false;
            for (size_t token=equal; token<first; ++token) {
                if (cells.ext_get(uint32_t(token)).is_2d_gt(x,y)) {
                    if (!dropped) end=token;
                    dropped=true;
                } else if (dropped) return false;
            }
        }
        const ggml_backend_execution_mask_interval interval{0,int32_t(end),int32_t(padded),1};
        std::memcpy(next.data()+size_t(i)*4,&interval,sizeof(interval));
    }
    bounds=std::move(next);
    return true;
}
