#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <utility>

struct llama_kv_stream_prefetch_request {
    size_t layer = 0, first = 0, tokens = 0, pages = 0, slot = 0;
    bool stable = false, submitted = false, reused = false;
};

struct llama_kv_stream_prefetch_wave {
    size_t requests = 0, pages = 0, tokens = 0, spans = 0;
};

// Describes existing encoded bytes only. Completion and slot protection remain the owner's responsibility.
class llama_kv_stream_ring_cache {
public:
    bool reset(size_t slots,size_t page_tokens) {
        if (!slots || !page_tokens) return false;
        try {
            std::vector<entry> next(slots);
            pages=std::move(next); page=page_tokens; generation=epoch=0;
            return true;
        } catch (const std::bad_alloc &) { return false; }
          catch (const std::length_error &) { return false; }
    }
    size_t size() const { return pages.size(); }
    void clear() { for (auto & p : pages) p={}; }
    void synchronize(uint64_t next_generation,uint64_t next_epoch) {
        if (generation != next_generation || epoch != next_epoch) clear();
        advance(next_generation,next_epoch);
    }
    // Only a known append publication may advance identity while preserving unaffected page records.
    void advance(uint64_t next_generation,uint64_t next_epoch) { generation=next_generation; epoch=next_epoch; }
    bool contains(size_t slot,uint32_t layer,size_t first,size_t live) const {
        return slot < pages.size() && live && live <= page && pages[slot].valid &&
            pages[slot].layer == layer && pages[slot].first == first && pages[slot].live == live;
    }
    bool record(size_t slot,uint32_t layer,size_t first,size_t live) {
        if (slot >= pages.size() || !live || live > page || first%page || first > SIZE_MAX-page) return false;
        pages[slot]={layer,first,live,true}; return true;
    }
    bool invalidate(size_t slot,size_t count) {
        if (!count || slot >= pages.size() || count > pages.size()-slot) return false;
        for (size_t i=0;i<count;++i) pages[slot+i]={};
        return true;
    }
    bool invalidate_rows(uint32_t layer,size_t first,size_t count) {
        if (count > SIZE_MAX-first) return false;
        for (auto & p : pages) if (count && p.valid && p.layer == layer && p.first < first+count && first < p.first+page) p={};
        return true;
    }
private:
    struct entry { uint32_t layer=0; size_t first=0,live=0; bool valid=false; };
    std::vector<entry> pages;
    size_t page=0;
    uint64_t generation=0,epoch=0;
};

// Owner-thread FIFO reservations. Deferred tails occupy slots, so speculative work cannot starve demand.
class llama_kv_stream_prefetch_plan {
public:
    // Allocate metadata by layer/slot count, never by context length. Failure preserves the old plan.
    bool start(const std::vector<size_t> & prefixes, size_t active, size_t padded, size_t page,
            size_t slots, size_t ceiling, size_t stable, const std::vector<uint8_t> & blocked = {}) {
        if (prefixes.empty() || !active || padded < active || !page || padded-active >= page || !slots || !ceiling ||
                stable > active || (!blocked.empty() && blocked.size() != slots)) return false;
        if ((active-1)/page != (padded-1)/page) return false;
        for (auto prefix : prefixes) if (prefix > padded || (prefix != padded && prefix%page)) return false;
        try {
            llama_kv_stream_prefetch_plan next;
            next.prefixes = prefixes; next.ready.resize(prefixes.size(),false);
            next.records.resize(slots);
            next.occupied = blocked.empty() ? std::vector<uint8_t>(slots, 0) : blocked;
            next.has_blocked_slots = std::any_of(blocked.begin(), blocked.end(), [](uint8_t value) { return value != 0; });
            next.active = active; next.padded = padded; next.page = page; next.ceiling = ceiling; next.stable = stable;
            next.token = prefixes[0]; *this = std::move(next); return true;
        } catch (const std::bad_alloc &) { return false; }
          catch (const std::length_error &) { return false; }
    }
    // Reserve demand order even when its producer is not ready; later stable requests may still be submitted.
    bool preview(llama_kv_stream_prefetch_request & output) const {
        if (records.empty() || used_pages == records.size()) return false;
        size_t next_layer=layer,next_token=token,next_write=write;
        while (next_layer < prefixes.size() && next_token == padded) {
            if (++next_layer < prefixes.size()) next_token = prefixes[next_layer];
        }
        if (next_layer == prefixes.size()) return false;
        if (!has_blocked_slots && occupied[next_write]) return false;
        size_t skipped = 0;
        while (occupied[next_write] && skipped < records.size()) {
            next_write = (next_write + 1) % records.size();
            ++skipped;
        }
        if (skipped == records.size()) return false;
        size_t available = 0;
        while (available < records.size()-next_write && !occupied[next_write+available]) ++available;
        size_t pages = std::min({ceiling,available,(padded-next_token)/page+((padded-next_token)%page != 0)});
        const size_t boundary = stable == active ? padded : stable/page*page;
        if (next_token < boundary) pages = std::min(pages,(boundary-next_token)/page+((boundary-next_token)%page != 0));
        if (!pages) return false;
        const size_t tokens = pages > (padded-next_token)/page ? padded-next_token : pages*page;
        output={next_layer,next_token,tokens,pages,next_write,next_token+std::min(tokens,active-next_token) <= stable,false};
        return true;
    }
    // Split a reservation only at a retained/missing boundary; its logical and physical order is unchanged.
    bool reserve(llama_kv_stream_prefetch_request & output,size_t limit=SIZE_MAX) {
        llama_kv_stream_prefetch_request r;
        if (!limit || !preview(r)) return false;
        r.pages=std::min(r.pages,limit);
        r.tokens=r.pages > r.tokens/page ? r.tokens : r.pages*page;
        records[(head+count)%records.size()] = r;
        for (size_t i = 0; i < r.pages; ++i) occupied[r.slot+i] = true;
        write = (r.slot+r.pages)%records.size(); ++count; used_pages += r.pages;
        layer=r.layer; token=r.first+r.tokens;
        output = r; return true;
    }
    bool make_ready(size_t i) { if (i >= ready.size()) return false; ready[i] = true; return true; }
    // Historical uploads are query-independent when the padded extent does not change.
    bool retarget(size_t next_active) {
        if (next_active <= stable || next_active > padded || !page ||
                (next_active-1)/page != (padded-1)/page) return false;
        active=next_active;
        return true;
    }
    bool can_submit(const llama_kv_stream_prefetch_request & r) const { return r.layer < ready.size() && (r.stable || ready[r.layer]); }
    bool mark_submitted(size_t i,bool reused=false) {
        if (i >= count) return false;
        auto & r = records[(head+i)%records.size()];
        if (r.submitted || !can_submit(r) || (reused && !r.stable)) return false;
        r.submitted = true; r.reused=reused; return true;
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
    // Keep the first transfer whole above the grouping target so demand always makes progress.
    bool ready_wave(size_t preferred_pages, size_t max_spans, llama_kv_stream_prefetch_wave & output) const {
        if (!preferred_pages || !max_spans || max_spans > 3 || !count) return false;
        llama_kv_stream_prefetch_wave next;
        const auto * first=request(0);
        size_t logical=first->first, physical=first->slot;
        for (size_t i=0;i<count;++i) {
            const auto * r=request(i);
            if (!r->submitted || r->layer != first->layer || r->first != logical ||
                    !r->pages || r->pages > records.size() || r->slot > records.size()-r->pages ||
                    !r->tokens || r->tokens > SIZE_MAX-logical) break;
            if (next.requests && (next.pages >= preferred_pages || r->pages > preferred_pages-next.pages)) break;
            const bool separate=!next.requests || r->slot != physical;
            if (separate && next.spans == max_spans) break;
            next.spans+=separate; ++next.requests; next.pages+=r->pages; next.tokens+=r->tokens;
            logical+=r->tokens; physical=r->slot+r->pages;
        }
        if (!next.requests) return false;
        output=next; return true;
    }
private:
    std::vector<size_t> prefixes;
    std::vector<uint8_t> ready, occupied;
    std::vector<llama_kv_stream_prefetch_request> records;
    size_t active = 0, padded = 0, page = 0, ceiling = 0, stable = 0;
    size_t layer = 0, token = 0, write = 0, head = 0, count = 0, used_pages = 0;
    bool has_blocked_slots = false;
};
