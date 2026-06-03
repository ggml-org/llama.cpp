#if defined(__APPLE__) && defined(GGML_USE_METAL)

#include "llama-moe-offloader.h"

#include "ggml-backend.h"
#include "ggml-metal.h"

#include <sys/mman.h>
#include <unistd.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <thread>

moe_layer::~moe_layer() {
    for (auto & p : pools) {
        ggml_backend_buffer_free(p.own_buf);
        ggml_free(p.own_ctx);
    }
    ggml_backend_buffer_free(msg_buf);
    ggml_free(msg_ctx);
    if (shared_event) {
        ggml_backend_metal_event_free(shared_event);
    }
}

llama_moe_offloader::llama_moe_offloader(ggml_backend_t backend) : backend(backend) {}

llama_moe_offloader::~llama_moe_offloader() {
    clear();
}

void llama_moe_offloader::start() {
    start_sidecar();
    ggml_backend_metal_set_moe_handler(backend, { hook, this });
}

void llama_moe_offloader::stop() {
    stop_sidecar();
    ggml_backend_synchronize(backend);
    ggml_backend_metal_set_moe_handler(backend, { nullptr, nullptr });
    backend = nullptr;
}

ggml_tensor * llama_moe_offloader::bind_pool(int           layer_idx,
                                             ggml_tensor * orig,
                                             int64_t       n_slots,
                                             uint64_t      file_offset,
                                             int           fd) {
    GGML_ASSERT(file_offset != 0);

    if (!orig) {
        return nullptr;
    }

    std::lock_guard<std::mutex> g(build_mtx);

    if ((int) layers.size() <= layer_idx) {
        layers.resize((size_t) layer_idx + 1);
    }

    auto & layer_ptr = layers[(size_t) layer_idx];

    if (!layer_ptr) {
        auto layer       = std::make_unique<moe_layer>();
        layer->layer_idx = layer_idx;
        layer->n_slots   = n_slots;
        layer->n_expert  = orig->ne[2];

        // lru
        layer->expert_to_slot.assign((size_t) layer->n_expert, -1);
        layer->slot_to_expert.assign((size_t) n_slots, -1);
        layer->lru_clock.assign((size_t) n_slots, 0);
        layer->in_use.assign((size_t) n_slots, 0);

        // shmem message setup
        {
            ggml_init_params p{};
            p.mem_size        = ggml_tensor_overhead() + 64;
            p.no_alloc        = true;
            layer->msg_ctx    = ggml_init(p);
            layer->msg_tensor = ggml_new_tensor_1d(layer->msg_ctx, GGML_TYPE_I8, MOE_MSG_NBYTES);
            char nm[64];
            snprintf(nm, sizeof(nm), "moe_msg_L%d", layer_idx);
            ggml_set_name(layer->msg_tensor, nm);
            layer->msg_buf = ggml_backend_alloc_ctx_tensors(layer->msg_ctx, backend);
            GGML_ASSERT(layer->msg_buf);
            layer->mapped = (uint8_t *) layer->msg_tensor->data;
            memset(layer->mapped, 0, MOE_MSG_NBYTES);
        }

        layer->shared_event = ggml_backend_metal_event_new(backend);

        sidecar_layers.push_back(layer.get());
        layer_ptr = std::move(layer);
    }

    moe_layer & layer = *layer_ptr;

    auto it = layer.name_to_pool_idx.find(orig->name);
    if (it != layer.name_to_pool_idx.end()) {
        return layer.pools[it->second].tensor;
    }

    moe_pool pool;
    pool.file_offset = file_offset;
    pool.fd          = fd;
    pool.stride      = orig->nb[2];
    {
        ggml_init_params p{};
        p.mem_size   = ggml_tensor_overhead() * 2;
        p.no_alloc   = true;
        pool.own_ctx = ggml_init(p);
    }
    pool.tensor = ggml_new_tensor_3d(pool.own_ctx, orig->type, orig->ne[0], orig->ne[1], n_slots);
    GGML_ASSERT(pool.tensor->nb[2] == pool.stride && "pool/orig per-expert stride mismatch");
    {
        char nm[256];
        snprintf(nm, sizeof(nm), "%s_pool", orig->name);
        ggml_set_name(pool.tensor, nm);
    }
    pool.own_buf = ggml_backend_alloc_ctx_tensors(pool.own_ctx, backend);
    GGML_ASSERT(pool.own_buf);

    size_t pool_idx                    = layer.pools.size();
    pool_to_layer[pool.tensor]         = &layer;
    layer.name_to_pool_idx[orig->name] = pool_idx;

    layer.pools.push_back(pool);

    return layer.pools.back().tensor;
}

bool llama_moe_offloader::hook(void *                     user_data,
                               const ggml_tensor *        src0,
                               const ggml_tensor *        src2,
                               ggml_metal_moe_intercept * out) {
    auto * me = (llama_moe_offloader *) user_data;

    auto it = me->pool_to_layer.find(src0);
    if (it == me->pool_to_layer.end()) {
        return false;
    }

    moe_layer & layer = *it->second;

    out->msg_tensor   = layer.msg_tensor;
    out->event        = layer.shared_event;

    int n = (int) (src2->ne[0] * src2->ne[1]);
    GGML_ASSERT(n <= MOE_MAX_IDS);
    out->n = n;

    const int expected_uses = (int) layer.pools.size();

    if (layer.last_src2 == src2 && layer.last_src2_uses < expected_uses) {
        out->reuse       = true;
        out->seq         = layer.last_src2_seq;
        layer.last_src2_uses++;
    } else {
        out->reuse                              = false;
        uint32_t seq                            = layer.next_seq.fetch_add(1, std::memory_order_relaxed) + 1;
        layer.last_src2                         = src2;
        layer.last_src2_seq                     = seq;
        layer.last_src2_uses                    = 1;
        out->seq                                = seq;
        *(int32_t *) (layer.mapped + MOE_OFF_N) = n;
    }

    return true;
}

int64_t llama_moe_offloader::lru_evict(moe_layer & layer) {
    int64_t  victim = -1;
    uint32_t oldest = UINT32_MAX;
    for (int64_t i = 0; i < layer.n_slots; ++i) {
        if (!layer.in_use[i] && layer.lru_clock[i] < oldest) {
            oldest = layer.lru_clock[i];
            victim = i;
        }
    }
    return victim;
}

bool llama_moe_offloader::pread_pool(moe_layer & layer, size_t pool_idx, int32_t expert_id, int64_t dst_slot) {
    moe_pool & p   = layer.pools[pool_idx];
    uint64_t   off = p.file_offset + (uint64_t) expert_id * p.stride;
    void *     dst = (uint8_t *) p.tensor->data + (size_t) dst_slot * p.stride;

    uint8_t * d   = (uint8_t *) dst;
    size_t    got = 0;
    while (got < p.stride) {
        ssize_t r = pread(p.fd, d + got, p.stride - got, (off_t) (off + got));
        if (r < 0 && errno == EINTR) {
            continue;
        }
        if (r <= 0) {
            return false;
        }
        got += (size_t) r;
    }
    return true;
}

struct moe_pread_task {
    moe_layer * layer;
    size_t      pool_idx;
    int32_t     expert_id;
    int64_t     dst_slot;
};

void llama_moe_offloader::resolve(moe_layer & layer, const int32_t * ids, int32_t * out, int n) {
    std::vector<moe_pread_task> tasks;
    tasks.reserve((size_t) n * layer.pools.size());

    std::fill(layer.in_use.begin(), layer.in_use.end(), 0);

    for (int i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        GGML_ASSERT(e >= 0 && e < layer.n_expert);

        int32_t slot = layer.expert_to_slot[(size_t) e];

        if (slot >= 0) {
            layer.lru_clock[slot] = ++layer.lru_time;
            layer.total_hits++;
        } else {
            const int64_t victim = lru_evict(layer);
            if (victim < 0) {
                fprintf(stderr, "moe L%d: n_slots=%lld too small - ubatch has more unique experts than slots\n",
                        layer.layer_idx, (long long) layer.n_slots);
                GGML_ABORT("moe_offloader: n_slots too small for ubatch");
            }
            slot = (int32_t) victim;

            const int32_t old_e = layer.slot_to_expert[(size_t) victim];
            if (old_e >= 0) {
                layer.expert_to_slot[(size_t) old_e] = -1;
            }

            layer.expert_to_slot[(size_t) e]      = slot;
            layer.slot_to_expert[(size_t) victim] = e;
            layer.lru_clock[victim]               = ++layer.lru_time;
            layer.total_misses++;

            for (size_t pi = 0; pi < layer.pools.size(); ++pi) {
                tasks.push_back({ &layer, pi, e, victim });
            }
        }

        layer.in_use[(size_t) slot] = 1;
        out[i]                      = slot;
    }

    if (tasks.empty()) {
        return;
    }

    if (tasks.size() == 1) {
        if (!pread_pool(*tasks[0].layer, tasks[0].pool_idx, tasks[0].expert_id, tasks[0].dst_slot)) {
            GGML_ABORT("moe_offloader: pread failed");
        }
    } else {
        __block bool failed = false;
        dispatch_apply(tasks.size(), DISPATCH_APPLY_AUTO, ^(size_t i) {
            const moe_pread_task & t = tasks[i];
            if (!pread_pool(*t.layer, t.pool_idx, t.expert_id, t.dst_slot)) {
                failed = true;
            }
        });
        if (failed) {
            GGML_ABORT("moe_offloader: parallel pread failed");
        }
    }

    std::atomic_thread_fence(std::memory_order_release);
}

void llama_moe_offloader::sidecar_loop() {
    std::vector<int32_t> remap_buf(MOE_MAX_IDS);

    std::vector<moe_layer *> layers_snapshot;

    while (sidecar_run.load(std::memory_order_relaxed)) {
        bool any = false;

        {
            // will have zero contention after graph finalization
            // and 48 pointers copy is cheap (no realloc)
            std::lock_guard<std::mutex> g(build_mtx);
            layers_snapshot = sidecar_layers;
        }

        for (auto * lp : layers_snapshot) {
            moe_layer & layer = *lp;

            auto *   req_a = (std::atomic<uint32_t> *) (layer.mapped + MOE_OFF_REQ);
            uint32_t req   = req_a->load(std::memory_order_acquire);
            uint32_t done  = layer.last_processed.load(std::memory_order_relaxed);

            if (req <= done) {
                continue;
            }

            int n_ids = *(int32_t *) (layer.mapped + MOE_OFF_N);
            GGML_ASSERT(n_ids > 0 && n_ids <= MOE_MAX_IDS);
            const int32_t * ids = (const int32_t *) (layer.mapped + MOE_OFF_SELECTED);

            resolve(layer, ids, remap_buf.data(), n_ids);

            memcpy(layer.mapped + MOE_OFF_REMAPPED, remap_buf.data(), (size_t) n_ids * sizeof(int32_t));

            layer.last_processed.store(req, std::memory_order_relaxed);
            ggml_backend_metal_event_signal(layer.shared_event, (uint64_t) req);

            any = true;
        }

        if (!any) {
            usleep(10);
        }
    }
}

void llama_moe_offloader::start_sidecar() {
    if (sidecar_run.exchange(true)) {
        return;
    }

    sidecar_thread = std::thread([this] { sidecar_loop(); });
}

void llama_moe_offloader::stop_sidecar() {
    if (!sidecar_run.exchange(false)) {
        return;
    }
    if (sidecar_thread.joinable()) {
        sidecar_thread.join();
    }
}

void llama_moe_offloader::clear() {
    std::lock_guard<std::mutex> g(build_mtx);
    layers.clear();
    sidecar_layers.clear();
    pool_to_layer.clear();
}

#endif
