#include "llama-moe-residency.h"

#include "llama-moe-file-source.h"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>

moe_residency_slot::~moe_residency_slot() {
    if (own_buf) {
        ggml_backend_buffer_free(own_buf);
    }
    if (own_ctx) {
        ggml_free(own_ctx);
    }
}

moe_residency_t & moe_residency_t::instance() {
    static moe_residency_t inst;
    return inst;
}

moe_residency_slot * moe_residency_t::get_or_create(ggml_tensor *  orig_weights,
                                                    int64_t        n_slots,
                                                    int            layer,
                                                    ggml_backend_t target_backend,
                                                    uint64_t       file_offset,
                                                    uint64_t       total_nbytes) {
    std::lock_guard<std::mutex> g(mu);

    auto it = slots.find(orig_weights);
    if (it != slots.end()) {
        GGML_ASSERT(it->second->n_slots == n_slots);
        return it->second.get();
    }

    auto s          = std::make_unique<moe_residency_slot>();
    s->orig_weights = orig_weights;
    s->layer        = layer;
    s->n_slots      = n_slots;
    s->n_expert     = orig_weights->ne[2];
    s->stride       = orig_weights->nb[2];
    s->file_offset  = file_offset;
    s->total_nbytes = total_nbytes;

    // Allocate pool tensor on target backend.
    struct ggml_init_params p{};
    p.mem_size = ggml_tensor_overhead() * 2;
    p.no_alloc = true;
    s->own_ctx = ggml_init(p);
    GGML_ASSERT(s->own_ctx);

    s->pool = ggml_new_tensor_3d(s->own_ctx, orig_weights->type, orig_weights->ne[0], orig_weights->ne[1], n_slots);

    char name[256];
    snprintf(name, sizeof(name), "%s_pool", orig_weights->name);
    ggml_set_name(s->pool, name);

    s->own_buf = ggml_backend_alloc_ctx_tensors(s->own_ctx, target_backend);
    // ggml_backend_buffer_set_usage(s->own_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    GGML_ASSERT(s->own_buf);
    GGML_ASSERT(s->pool->nb[2] == s->stride && "pool/orig per-expert stride mismatch");

    // Init slot tables and LRU. All slots start empty (slot_to_expert = -1).
    // Push slots into LRU back-to-front so that the first eviction picks slot 0,
    // making prefill ordering predictable.
    s->expert_to_slot.assign((size_t) s->n_expert, -1);
    s->slot_to_expert.assign((size_t) n_slots, -1);
    s->lru_iter.assign((size_t) n_slots, s->lru.end());
    for (int64_t i = n_slots - 1; i >= 0; --i) {
        s->lru.push_front(i);
        s->lru_iter[(size_t) i] = s->lru.begin();
    }

    // Madvise orig pages to FREE. Only effective on anonymous mappings, which
    // requires --no-mmap + --override-tensor =CPU. Best-effort: log on failure.
    if (file_offset != 0 && orig_weights->data != nullptr && total_nbytes > 0) {
        const size_t page = (size_t) sysconf(_SC_PAGESIZE);
        uintptr_t    base = (uintptr_t) orig_weights->data;
        uintptr_t    lo   = (base + page - 1) & ~(uintptr_t) (page - 1);
        uintptr_t    hi   = (base + total_nbytes) & ~(uintptr_t) (page - 1);
        if (hi > lo) {
            if (madvise((void *) lo, hi - lo, MADV_FREE) != 0) {
                fprintf(stderr,
                        "moe_residency L%d %s madvise FREE failed: %s "
                        "(base=%p lo=%p hi=%p len=%zu)\n",
                        layer, orig_weights->name, strerror(errno), (void *) base, (void *) lo, (void *) hi,
                        (size_t) (hi - lo));
            } else {
                s->orig_released = true;
            }
        }
    }

    fprintf(stderr, "moe_residency: L%d %s n_slots=%lld n_expert=%lld stride=%zu file_off=%llu released=%d\n", layer,
            orig_weights->name, (long long) n_slots, (long long) s->n_expert, s->stride,
            (unsigned long long) file_offset, (int) s->orig_released);

    auto * raw = s.get();
    slots.emplace(orig_weights, std::move(s));
    return raw;
}

int moe_residency_t::pread_into_pool(moe_residency_slot * slot, int32_t expert_id, int64_t dst_slot) {
    GGML_ASSERT(expert_id >= 0 && expert_id < slot->n_expert);
    GGML_ASSERT(dst_slot >= 0 && dst_slot < slot->n_slots);

    // Stage into a per-thread buffer, then push into the pool tensor through
    // the backend API. Mirrors the existing llama-moe-pool path. On Metal with
    // Shared storage we could pread directly into (uint8_t*)pool->data +
    // dst_slot*stride and save one memcpy — defer that until profiled.
    thread_local std::vector<uint8_t> staging;
    if (staging.size() < slot->stride) {
        staging.resize(slot->stride);
    }

    const uint64_t off = slot->file_offset + (uint64_t) expert_id * slot->stride;
    if (!moe_file_source().pread_into(staging.data(), off, slot->stride)) {
        fprintf(stderr, "moe_residency L%d %s pread fail expert=%d slot=%lld off=%llu stride=%zu\n", slot->layer,
                slot->orig_weights->name, (int) expert_id, (long long) dst_slot, (unsigned long long) off,
                slot->stride);
        return -1;
    }
    ggml_backend_tensor_set(slot->pool, staging.data(), (size_t) dst_slot * slot->stride, slot->stride);
    return 0;
}

void moe_residency_t::touch_lru(moe_residency_slot * slot, int64_t slot_id) {
    slot->lru.erase(slot->lru_iter[(size_t) slot_id]);
    slot->lru.push_front(slot_id);
    slot->lru_iter[(size_t) slot_id] = slot->lru.begin();
}

int64_t moe_residency_t::evict_lru(moe_residency_slot * slot) {
    GGML_ASSERT(!slot->lru.empty());
    return slot->lru.back();
}

void moe_residency_t::resolve_ids(moe_residency_slot * slot,
                                  const int32_t *      ids,
                                  int32_t *            slot_ids_out,
                                  int64_t              n_ids) {
    uint64_t local_misses = 0;

    // Slots used in this call are locked from eviction until the forward
    // consuming slot_ids_out completes. Caller guarantees serialization.
    std::vector<uint8_t> locked((size_t) slot->n_slots, 0);

    for (int64_t i = 0; i < n_ids; ++i) {
        const int32_t e = ids[i];
        GGML_ASSERT(e >= 0 && e < slot->n_expert);

        int32_t s = slot->expert_to_slot[(size_t) e];
        if (s >= 0) {
            slot->total_hits++;
        } else {
            // Find LRU slot not locked in this call.
            int64_t victim = -1;
            for (auto it = slot->lru.rbegin(); it != slot->lru.rend(); ++it) {
                if (!locked[(size_t) *it]) {
                    victim = *it;
                    break;
                }
            }
            if (victim < 0) {
                fprintf(stderr,
                        "moe_residency L%d %s n_ids=%lld n_slots=%lld too small: unique experts in ubatch > n_slots\n",
                        slot->layer, slot->orig_weights->name, (long long) n_ids, (long long) slot->n_slots);
                GGML_ABORT("moe_residency: n_slots too small for ubatch");
            }

            const int32_t old_e = slot->slot_to_expert[(size_t) victim];
            if (old_e >= 0) {
                slot->expert_to_slot[(size_t) old_e] = -1;
            }
            if (pread_into_pool(slot, e, victim) != 0) {
                GGML_ABORT("moe_residency: pread failed");
            }
            slot->expert_to_slot[(size_t) e]      = (int32_t) victim;
            slot->slot_to_expert[(size_t) victim] = e;
            s                                     = (int32_t) victim;
            slot->total_misses++;
            local_misses++;
        }

        locked[(size_t) s] = 1;
        touch_lru(slot, s);
        slot_ids_out[i] = s;
    }

    if (local_misses > 0) {
        const uint64_t total_dec = slot->total_hits + slot->total_misses;
        const double   hit_rate  = total_dec > 0 ? (100.0 * (double) slot->total_hits) / (double) total_dec : 0.0;
        fprintf(stderr, "moe_residency L%d %s ids=%lld local_misses=%llu hits=%llu misses=%llu hit_rate=%.2f%%\n",
                slot->layer, slot->orig_weights->name, (long long) n_ids, (unsigned long long) local_misses,
                (unsigned long long) slot->total_hits, (unsigned long long) slot->total_misses, hit_rate);
    }
}

void moe_residency_t::prefill_from_profile(moe_residency_slot * slot, const std::vector<int32_t> & top_experts) {
    // Load up to n_slots top experts into slots 0..N-1. Idempotent.
    const int64_t n      = std::min<int64_t>(slot->n_slots, (int64_t) top_experts.size());
    int64_t       loaded = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int32_t e = top_experts[(size_t) i];
        if (e < 0 || e >= slot->n_expert) {
            continue;
        }
        if (slot->expert_to_slot[(size_t) e] >= 0) {
            continue;
        }
        if (pread_into_pool(slot, e, i) != 0) {
            continue;
        }
        // Evict any existing occupant of slot i.
        const int32_t old_e = slot->slot_to_expert[(size_t) i];
        if (old_e >= 0) {
            slot->expert_to_slot[(size_t) old_e] = -1;
        }
        slot->expert_to_slot[(size_t) e] = (int32_t) i;
        slot->slot_to_expert[(size_t) i] = e;
        touch_lru(slot, i);
        loaded++;
    }
    fprintf(stderr, "moe_residency prefill L%d %s loaded=%lld/%lld\n", slot->layer, slot->orig_weights->name,
            (long long) loaded, (long long) n);
}

void moe_residency_t::clear() {
    std::lock_guard<std::mutex> g(mu);
    slots.clear();
}
