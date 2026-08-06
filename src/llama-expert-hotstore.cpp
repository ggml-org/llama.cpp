#include "llama-expert-hotstore.h"
#include "llama-expert-heatmap.h"
#include "llama-expert-tier.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <regex>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

// drop the physical pages backing [ptr, ptr+len) that are fully inside the
// range (inward-rounded to page boundaries) so neighbors are never clobbered.
// on mmap'd tensors the pages re-fault from the file; on anonymous memory they
// are discarded (the tier never reads them again while GPU-resident).
static void release_pages(void * ptr, size_t len) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    const size_t page = si.dwPageSize;
#else
    const long page = sysconf(_SC_PAGESIZE);
#endif
    const uintptr_t base   = (uintptr_t) ptr;
    const uintptr_t start  = (base + (uintptr_t) page - 1) & ~((uintptr_t) page - 1);
    const uintptr_t end    = (base + len) & ~((uintptr_t) page - 1);
    if (start < end) {
#ifdef _WIN32
        VirtualFree((LPVOID) start, end - start, MEM_RESET);
#else
        madvise((void *) start, end - start, MADV_DONTNEED);
#endif
    }
}

// uniform strided hash of a memory range: sample N chunks spread across the
// whole slice and combine with FNV-1a (pure integer, bit-exact on every
// device, so a correct copy always hashes identically).
static uint64_t hash_slice(const uint8_t * data, size_t len) {
    constexpr size_t N_CHUNKS = 64;
    constexpr size_t CHUNK    = 256;
    uint64_t h = 1469598103934665603ULL; // FNV-1a offset basis
    if (len == 0) {
        return h;
    }
    const size_t step = len / N_CHUNKS;
    for (size_t i = 0; i < N_CHUNKS; i++) {
        const size_t off = i * step;
        for (size_t j = 0; j < CHUNK && off + j < len; j++) {
            h ^= data[off + j];
            h *= 1099511628211ULL;
        }
    }
    return h;
}

// verify a GPU slot plane against the source slice by hashing both (the GPU
// side via a sparse read-back) and comparing. returns true on match.
static bool verify_gpu_copy(ggml_tensor * dst, size_t slot_off, const uint8_t * src, size_t len) {
    constexpr size_t N_CHUNKS = 64;
    constexpr size_t CHUNK    = 256;
    const uint64_t h_src = hash_slice(src, len);
    uint64_t h_gpu = 1469598103934665603ULL;
    if (len > 0) {
        const size_t step = len / N_CHUNKS;
        for (size_t i = 0; i < N_CHUNKS; i++) {
            const size_t off = i * step;
            const size_t n = std::min(CHUNK, len - off);
            std::vector<uint8_t> buf(n);
            ggml_backend_tensor_get(dst, buf.data(), slot_off + off, n);
            for (size_t j = 0; j < n; j++) {
                h_gpu ^= buf[j];
                h_gpu *= 1099511628211ULL;
            }
        }
    }
    return h_src == h_gpu;
}

// matches the weight tensor of an expert tensor, e.g.:
//   blk.0.ffn_gate_exps.weight
//   blk.3.ffn_down_chexps.weight
// follows the same convention as LLM_FFN_EXPS_REGEX in common.h
static const std::regex g_re_exps_weight("blk\\.(\\d+)\\.ffn_(up|down|gate|gate_up)_(ch|)exps\\.weight");

llama_expert_hotstore::llama_expert_hotstore(
        const llama_model * model, int n_layers, int n_experts, int hot_s, int sync_period,
        float hyst, int dwell) :
    n_layers(n_layers),
    n_experts(n_experts),
    hot_s(hot_s),
    bytes_per_slot(n_layers, 0),
    sync_period(sync_period),
    hyst(hyst),
    dwell(dwell) {
    if (n_layers <= 0) {
        return;
    }
    if (this->hot_s > this->n_experts) {
        LLAMA_LOG_WARN("%s: clamping expert hot store S=%d to n_experts=%d\n", __func__, this->hot_s, this->n_experts);
        this->hot_s = this->n_experts;
    }

    for (const auto & [name, tensor] : llama_internal_get_tensor_map(model)) {
        std::smatch m;
        if (std::regex_search(name, m, g_re_exps_weight)) {
            const int il = std::stoi(m[1].str());
            if (il >= 0 && il < n_layers && tensor->ne[2] > 0) {
                // a slot holds nbytes/n_experts of this tensor
                bytes_per_slot[il] += ggml_nbytes(tensor) / (size_t) tensor->ne[2];
                entries.push_back({il, tensor, {}});
            }
        }
    }

    // entries is fixed from here on; build a per-layer index of stable
    // pointers so copy/resync do not iterate the whole entries vector.
    entries_by_layer.assign(n_layers, {});
    for (auto & e : entries) {
        entries_by_layer[e.layer_idx].push_back(&e);
    }

    if (this->hot_s > 0) {
        slot_to_expert.assign(n_layers, std::vector<int>(this->hot_s, -1));
        dwell_count.assign(n_layers, std::vector<int>(this->hot_s, 0));
    }
}

bool llama_expert_hotstore::allocate(
        const std::vector<ggml_backend_buffer_type_t> & bufts,
        const float * tensor_split, int n_split) {
    if (hot_s <= 0 || entries.empty()) {
        return false;
    }
    if (hot_s > n_experts) {
        throw std::runtime_error(format("%s: hot store S=%d exceeds n_experts=%d",
            __func__, hot_s, n_experts));
    }
    if (n_split <= 0 || (int) bufts.size() < n_split) {
        n_split = (int) bufts.size() > 0 ? (int) bufts.size() : 1;
    }

    n_devices = n_split;
    slot_start.assign(n_devices, 0);
    slot_end.assign(n_devices, 0);

    // per-device slot ranges from the tensor_split fractions (-ts); even split
    // when the fractions are all zero
    {
        float total = 0.0f;
        for (int g = 0; g < n_devices; g++) {
            total += tensor_split ? tensor_split[g] : 0.0f;
        }
        if (total <= 0.0f) {
            total = (float) n_devices;
        }
        int acc = 0;
        for (int g = 0; g < n_devices; g++) {
            slot_start[g] = acc;
            const float frac = tensor_split && tensor_split[g] > 0.0f ? tensor_split[g] : 1.0f;
            slot_end[g] = acc + (int) ((float) hot_s * frac / total);
            acc = slot_end[g];
        }
        slot_end[n_devices - 1] = hot_s; // last device absorbs the remainder
    }

    // per-device no_alloc contexts holding that device's dst + hot_lut tensors
    ctx_dev.resize(n_devices);
    buf_dev.resize(n_devices);
    luts.assign(n_layers, layer_lut{});
    for (int g = 0; g < n_devices; g++) {
        const int local_slots = slot_end[g] - slot_start[g];

        ggml_init_params p = {
            /*.mem_size   =*/ ggml_tensor_overhead() * (entries.size() + 2 * n_layers),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_dev[g] = ggml_context_ptr(ggml_init(p));
        if (!ctx_dev[g]) {
            LLAMA_LOG_ERROR("%s: hot store: failed to create device %d context\n", __func__, g);
            return false;
        }

        // one hot tensor per expert weight tensor: local_slots slot planes
        // plus a zeroed sentinel plane (index local_slots)
        for (auto & e : entries) {
            e.dst.resize(n_devices);
            e.dst[g] = ggml_new_tensor_3d(ctx_dev[g].get(), e.src->type, e.src->ne[0], e.src->ne[1], local_slots + 1);
            ggml_set_name(e.dst[g], (std::string(e.src->name) + ".hot").c_str());
        }
        for (int il = 0; il < n_layers; il++) {
            luts[il].hot_lut.resize(n_devices);
            luts[il].hot_lut[g] = ggml_new_tensor_2d(ctx_dev[g].get(), GGML_TYPE_I32, 1, n_experts);
            luts[il].mask_lut.resize(n_devices);
            luts[il].mask_lut[g] = ggml_new_tensor_2d(ctx_dev[g].get(), GGML_TYPE_F32, 1, local_slots + 1);
        }

        // check the buffer would fit before committing any VRAM
        const size_t need = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_dev[g].get(), bufts[g]);
        if (need == 0) {
            LLAMA_LOG_ERROR("%s: hot store: zero-sized buffer on device %d, disabled\n", __func__, g);
            return false;
        }
        size_t free_mem = 0, total_mem = 0;
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(bufts[g]);
        if (dev) {
            ggml_backend_dev_memory(dev, &free_mem, &total_mem);
        }
        if (dev && free_mem < need) {
            throw std::runtime_error(format("%s: not enough memory to allocate the GPU hot store of %d slots (%zu MiB needed, %zu MiB free on %s)",
                __func__, hot_s, need / (1024 * 1024), free_mem / (1024 * 1024),
                ggml_backend_dev_name(dev)));
        }
        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(ctx_dev[g].get(), bufts[g]);
        if (b == nullptr) {
            throw std::runtime_error(format("%s: unable to allocate hot store buffer of %d slots (%zu MiB)",
                __func__, hot_s, need / (1024 * 1024)));
        }
        buf_dev[g] = ggml_backend_buffer_ptr(b);
        ggml_backend_buffer_set_usage(buf_dev[g].get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_clear(buf_dev[g].get(), 0);

        // sentinel mask: 1.0 for real slots, 0.0 for the sentinel plane, so
        // sentinel-routed hot rows are zeroed after the GPU mul_mat_id.
        std::vector<float> mask_h(local_slots + 1, 1.0f);
        mask_h[local_slots] = 0.0f;
        for (int il = 0; il < n_layers; il++) {
            ggml_backend_tensor_set(luts[il].mask_lut[g], mask_h.data(), 0,
                (local_slots + 1) * sizeof(float));
        }
    }

    // CPU context for the cold_mask tensors
    ggml_init_params params_cpu = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (2 * n_layers) + 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx_cpu = ggml_context_ptr(ggml_init(params_cpu));
    for (int il = 0; il < n_layers; il++) {
        luts[il].cold_mask = ggml_new_tensor_1d(ctx_cpu.get(), GGML_TYPE_I32, n_experts);
    }
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_t b_cpu = ggml_backend_alloc_ctx_tensors_from_buft(ctx_cpu.get(), cpu_buft);
    if (b_cpu) {
        buf_cpu = ggml_backend_buffer_ptr(b_cpu);
        ggml_backend_buffer_set_usage(buf_cpu.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    // register each expert weight tensor with the tier hook so build_lora_mm_id
    // can find its per-device GPU hot tensors and per-device LUTs.
    for (const auto & e : entries) {
        const auto & L = luts[e.layer_idx];
        llama_expert_tier_register(e.src, e.dst, L.hot_lut, L.mask_lut, L.cold_mask);
    }

    return true;
}

llama_expert_hotstore::~llama_expert_hotstore() {
    llama_expert_tier_clear();
}

// device owning a global slot index, or -1 (slot ranges are contiguous)
static int slot_device(const std::vector<int> & slot_start, const std::vector<int> & slot_end, int p) {
    for (int g = 0; g < (int) slot_start.size(); g++) {
        if (p >= slot_start[g] && p < slot_end[g]) {
            return g;
        }
    }
    return -1;
}

bool llama_expert_hotstore::copy_top_s(const llama_expert_heatmap & heatmap) {
    if (is_filled || hot_s <= 0 || entries.empty() || buf_dev.empty()) {
        return false;
    }

    for (int il = 0; il < n_layers; il++) {
        auto & ste = slot_to_expert[il];
        auto & dc  = dwell_count[il];
        // startup batch: the first S experts of each layer go to the GPU
        for (int p = 0; p < hot_s; p++) {
            ste[p] = p;
            dc[p]  = dwell; // initial fill is eligible to be corrected next sync
        }

        for (entry * e : entries_by_layer[il]) {
            const size_t slot = ggml_nbytes(e->src) / (size_t) e->src->ne[2];
            const char * src = e->src->data ? (const char *) ggml_get_data(e->src) : nullptr;
            if (!src) {
                continue;
            }
            for (int p = 0; p < hot_s; p++) {
                const int ex = ste[p];
                if (ex < 0) {
                    continue;
                }
                const int g = slot_device(slot_start, slot_end, p);
                if (g < 0) {
                    continue;
                }
                ggml_backend_tensor_set(e->dst[g], src + (size_t) ex * slot, (size_t) (p - slot_start[g]) * slot, slot);
                // real reclaim: the expert now lives on the GPU, drop its RAM pages
                release_pages((void *) (src + (size_t) ex * slot), slot);
            }
        }
    }

    last_sync_tokens = heatmap.tokens_total;
    is_filled = true;
    update_luts();
    LLAMA_LOG("=== Expert hot store: startup batch moved to GPU ===\n");
    return true;
}

bool llama_expert_hotstore::resync_top_s(const llama_expert_heatmap & heatmap) {
    if (!is_filled || hot_s <= 0 || buf_dev.empty()) {
        return false;
    }

    // release CPU pages of experts whose GPU copy was verified last sync, after
    // a full token has generated from the GPU copy
    for (auto & pr : pending_release) {
        const size_t slot = ggml_nbytes(pr.first->src) / (size_t) pr.first->src->ne[2];
        release_pages((void *) ((const char *) ggml_get_data(pr.first->src) + (size_t) pr.second * slot), slot);
    }
    pending_release.clear();

    // tokens elapsed since the previous sync, used to age dwell counters
    const int64_t elapsed = heatmap.tokens_total - last_sync_tokens;
    int swapped = 0;
    for (int il = 0; il < n_layers; il++) {
        auto & ste = slot_to_expert[il];
        auto & dc  = dwell_count[il];

        // mark the resident experts so the top-S candidates below can be
        // filtered to non-residents
        std::vector<char> resident_set(n_experts, 0);
        for (int p = 0; p < hot_s; p++) {
            if (ste[p] >= 0) {
                resident_set[ste[p]] = 1;
            }
        }

        const std::vector<int> top = heatmap.get_top_s(il, hot_s);

        // find a free slot index (guard: any resident displacement must
        // clear the hysteresis gate, unless the gate is off)
        auto find_slot = [&](int e_cold) -> int {
            // fill empty slots first (no gate on fill)
            for (int p = 0; p < hot_s; p++) {
                if (ste[p] < 0) {
                    return p;
                }
            }
            if (hyst <= 0.0f) {
                // gate off: displace the weakest resident
                int p_worst = -1;
                for (int p = 0; p < hot_s; p++) {
                    if (ste[p] >= 0 && (p_worst < 0 ||
                        heatmap.get_score(il, ste[p]) < heatmap.get_score(il, ste[p_worst]))) {
                        p_worst = p;
                    }
                }
                return p_worst;
            }
            // gate on: coldest resident that has dwelled enough AND is beaten
            // by hyst * this cold expert
            const float s_cold = heatmap.get_score(il, e_cold);
            int p_worst = -1;
            float worst_score = 1e9f;
            for (int p = 0; p < hot_s; p++) {
                if (ste[p] < 0) {
                    continue;
                }
                if (dc[p] < dwell) {
                    continue; // incumbent must keep its slot while under dwell
                }
                if (s_cold >= hyst * heatmap.get_score(il, ste[p])) {
                    const float s_inc = heatmap.get_score(il, ste[p]);
                    if (s_inc < worst_score) {
                        worst_score = s_inc;
                        p_worst     = p;
                    }
                }
            }
            return p_worst;
        };

        int swapped_in_layer = 0;
        for (int e_cold : top) {
            if (swapped_in_layer >= 1) {
                break;
            }
            if (e_cold < 0 || e_cold >= n_experts || resident_set[e_cold]) {
                continue;
            }
            const int p = find_slot(e_cold);
            if (p < 0) {
                break; // no slot free or displaceable under the gate
            }
            const int g = slot_device(slot_start, slot_end, p);
            if (g < 0) {
                continue;
            }
            const int e_out = ste[p];
            bool verified = true;
            for (entry * ent : entries_by_layer[il]) {
                const size_t slot = ggml_nbytes(ent->src) / (size_t) ent->src->ne[2];
                const char * src = ent->src->data ? (const char *) ggml_get_data(ent->src) : nullptr;
                if (!src) {
                    continue;
                }
                const size_t off = (size_t) (p - slot_start[g]) * slot;
                // move the displaced expert's slice back into RAM (repopulate pages)
                if (e_out >= 0) {
                    ggml_backend_tensor_get(ent->dst[g], (void *) (src + (size_t) e_out * slot), off, slot);
                }
                // move the new expert's slice to the GPU
                ggml_backend_tensor_set(ent->dst[g], src + (size_t) e_cold * slot, off, slot);
                // verify the GPU copy landed; release the CPU slice only later
                if (!verify_gpu_copy(ent->dst[g], off, (const uint8_t *) (src + (size_t) e_cold * slot), slot)) {
                    verified = false;
                }
            }
            if (verified) {
                for (entry * ent : entries_by_layer[il]) {
                    const size_t slot = ggml_nbytes(ent->src) / (size_t) ent->src->ne[2];
                    pending_release.emplace_back(ent, e_cold);
                }
            } else if (getenv("LLAMA_EXPERT_DEBUG")) {
                LLAMA_LOG("=== expert hot store: hash mismatch on move-in of expert %d, keeping CPU copy ===\n", e_cold);
            }
            ste[p] = e_cold;
            dc[p]  = -elapsed; // fresh dwell: aging below brings it to 0
            swapped++;
            swapped_in_layer++;
        }

        for (int p = 0; p < hot_s; p++) {
            if (ste[p] >= 0) {
                dc[p] += (int) std::max<int64_t>(elapsed, 0);
            }
        }
    }

    last_sync_tokens = heatmap.tokens_total;
    if (swapped > 0) {
        update_luts();
        if (getenv("LLAMA_EXPERT_DEBUG")) {
            LLAMA_LOG("=== Expert hot store: re-sync swapped %d expert slots ===\n", swapped);
        }
    }
    return swapped > 0;
}

bool llama_expert_hotstore::maybe_resync(const llama_expert_heatmap & heatmap, bool multi_slot) {
    // n_tokens>1 (multi-slot) freezes the hot store: no swapping during the batch
    if (multi_slot || sync_period <= 0 || heatmap.tokens_total <= 0) {
        return false;
    }
    if (heatmap.tokens_total / sync_period > last_sync_tokens / sync_period) {
        return resync_top_s(heatmap);
    }
    return false;
}

int llama_expert_hotstore::slot_of(int layer_idx, int expert_id) const {
    if (layer_idx < 0 || layer_idx >= n_layers || hot_s <= 0) {
        return -1;
    }
    const auto & ste = slot_to_expert[layer_idx];
    for (int p = 0; p < hot_s; p++) {
        if (ste[p] == expert_id) {
            return p;
        }
    }
    return -1;
}

void llama_expert_hotstore::update_luts() {
    if (hot_s <= 0 || luts.empty() || buf_dev.empty()) {
        return;
    }

    std::vector<int32_t> cold_mask_h(n_experts);

    for (int il = 0; il < n_layers; il++) {
        const auto & ste = slot_to_expert[il];

        // per-device LUTs: an expert whose global slot is on device g maps to
        // the LOCAL slot index there; everything else maps to that device's
        // local sentinel slot (zero contribution).
        for (int g = 0; g < n_devices; g++) {
            const int local_slots = slot_end[g] - slot_start[g];
            std::vector<int32_t> hot_lut_h(n_experts, local_slots);
            for (int p = slot_start[g]; p < slot_end[g]; p++) {
                const int e = ste[p];
                if (e >= 0 && e < n_experts) {
                    hot_lut_h[e] = p - slot_start[g];
                }
            }
            ggml_backend_tensor_set(luts[il].hot_lut[g], hot_lut_h.data(), 0,
                n_experts * sizeof(int32_t));
        }

        // defaults: everyone cold
        for (int e = 0; e < n_experts; e++) {
            cold_mask_h[e] = 1;
        }
        // residents override
        for (int p = 0; p < hot_s; p++) {
            const int e = ste[p];
            if (e >= 0 && e < n_experts) {
                cold_mask_h[e] = 0;
            }
        }
        ggml_backend_tensor_set(luts[il].cold_mask, cold_mask_h.data(), 0,
            n_experts * sizeof(int32_t));
    }
}

void llama_expert_hotstore::log_hit_rate(const std::vector<std::pair<int, ggml_tensor *>> & moe_sel) {
    if (moe_sel.empty() || !is_filled) {
        return;
    }
    size_t hits = 0, total = 0;
    for (const auto & kv : moe_sel) {
        const int il = kv.first;
        const ggml_tensor * t = kv.second;
        if (!t || !t->data || t->type != GGML_TYPE_I32) {
            continue;
        }
        const size_t n = ggml_nelements(t);
        std::vector<int32_t> ids(n);
        ggml_backend_tensor_get(t, ids.data(), 0, n * sizeof(int32_t));
        for (size_t i = 0; i < n; i++) {
            const int32_t id = ids[i];
            if (id >= 0 && id < n_experts) {
                total++;
                if (slot_of(il, id) >= 0) {
                    hits++;
                }
            }
        }
    }
    if (total > 0) {
        LLAMA_LOG("=== expert hot hit rate: %zu/%zu = %.1f%% ===\n", hits, total, 100.0f * (float) hits / (float) total);
    }
}

void llama_expert_hotstore::log() const {
    LLAMA_LOG("=== Expert hotstore sizing (S=%d) ===\n", hot_s);
    const bool debug = getenv("LLAMA_EXPERT_DEBUG") != nullptr;
    size_t total = 0;
    for (int il = 0; il < n_layers; il++) {
        total += bytes_per_slot[il];
        if (debug) {
            LLAMA_LOG("  layer %3d: bytes/slot = %zu\n", il, bytes_per_slot[il]);
        }
    }
    LLAMA_LOG("  total bytes/slot across all layers = %zu (%zu MiB)\n",
        total, total / (1024 * 1024));
    if (!buf_dev.empty()) {
        LLAMA_LOG("  GPU hot store allocated: %s, %zu bytes (%zu MiB) for %d+1 slots across %d device(s) (%d expert + 1 sentinel per device)\n",
            ggml_backend_buffer_name(buf_dev[0].get()),
            ggml_backend_buffer_get_size(buf_dev[0].get()),
            ggml_backend_buffer_get_size(buf_dev[0].get()) / (1024 * 1024),
            hot_s, n_devices, hot_s);
    } else if (hot_s > 0) {
        LLAMA_LOG("  hot store DISABLED (%d slots requested)\n", hot_s);
    }
}
