#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <utility>

struct llama_kv_stream_prefetch_request {
    size_t layer = 0, first = 0, tokens = 0, pages = 0, slot = 0;
    bool stable = false, submitted = false;
};

// Owner-thread FIFO reservations. Deferred tails occupy slots, so speculative work cannot starve demand.
class llama_kv_stream_prefetch_plan {
public:
    // Allocate metadata by layer/slot count, never by context length. Failure preserves the old plan.
    bool start(const std::vector<size_t> & prefixes, size_t active, size_t padded, size_t page,
            size_t slots, size_t ceiling, size_t stable) {
        if (prefixes.empty() || !active || padded < active || !page || padded-active >= page || !slots || !ceiling || stable > active) return false;
        if ((active-1)/page != (padded-1)/page) return false;
        for (auto prefix : prefixes) if (prefix > padded || (prefix != padded && prefix%page)) return false;
        try {
            llama_kv_stream_prefetch_plan next;
            next.prefixes = prefixes; next.ready.resize(prefixes.size(),false);
            next.records.resize(slots); next.occupied.resize(slots,false);
            next.active = active; next.padded = padded; next.page = page; next.ceiling = ceiling; next.stable = stable;
            next.token = prefixes[0]; *this = std::move(next); return true;
        } catch (const std::bad_alloc &) { return false; }
          catch (const std::length_error &) { return false; }
    }
    // Reserve demand order even when its producer is not ready; later stable requests may still be submitted.
    bool reserve(llama_kv_stream_prefetch_request & output) {
        if (records.empty() || used_pages == records.size()) return false;
        while (layer < prefixes.size() && token == padded) {
            if (++layer < prefixes.size()) token = prefixes[layer];
        }
        if (layer == prefixes.size() || occupied[write]) return false;
        size_t available = 0;
        while (available < records.size()-write && !occupied[write+available]) ++available;
        size_t pages = std::min({ceiling,available,(padded-token)/page+((padded-token)%page != 0)});
        const size_t boundary = stable == active ? padded : stable/page*page;
        if (token < boundary) pages = std::min(pages,(boundary-token)/page+((boundary-token)%page != 0));
        if (!pages) return false;
        const size_t tokens = pages > (padded-token)/page ? padded-token : pages*page;
        llama_kv_stream_prefetch_request r{layer,token,tokens,pages,write,token+std::min(tokens,active-token) <= stable,false};
        records[(head+count)%records.size()] = r;
        for (size_t i = 0; i < pages; ++i) occupied[write+i] = true;
        write = (write+pages)%records.size(); ++count; used_pages += pages; token += tokens;
        output = r; return true;
    }
    bool make_ready(size_t i) { if (i >= ready.size()) return false; ready[i] = true; return true; }
    bool can_submit(const llama_kv_stream_prefetch_request & r) const { return r.layer < ready.size() && (r.stable || ready[r.layer]); }
    bool mark_submitted(size_t i) {
        if (i >= count) return false;
        auto & r = records[(head+i)%records.size()];
        if (r.submitted || !can_submit(r)) return false;
        r.submitted = true; return true;
    }
    // Release a consumed prefix immediately; the remaining request stays at the demand head.
    bool consume(size_t pages) {
        if (!count || !pages) return false;
        auto & r = records[head];
        if (!r.submitted || pages > r.pages) return false;
        for (size_t i = 0; i < pages; ++i) occupied[r.slot+i] = false;
        const size_t tokens = pages > r.tokens/page ? r.tokens : pages*page;
        r.first += tokens; r.tokens -= tokens; r.slot += pages; r.pages -= pages; used_pages -= pages;
        if (!r.pages) { head = (head+1)%records.size(); --count; }
        return true;
    }
    const llama_kv_stream_prefetch_request * request(size_t i) const { return i < count ? &records[(head+i)%records.size()] : nullptr; }
    const llama_kv_stream_prefetch_request * front() const { return count ? &records[head] : nullptr; }
    size_t pending() const { return count; }
    size_t used() const { return used_pages; }
private:
    std::vector<size_t> prefixes;
    std::vector<uint8_t> ready, occupied;
    std::vector<llama_kv_stream_prefetch_request> records;
    size_t active = 0, padded = 0, page = 0, ceiling = 0, stable = 0;
    size_t layer = 0, token = 0, write = 0, head = 0, count = 0, used_pages = 0;
};
