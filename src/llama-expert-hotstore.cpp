#include "llama-expert-hotstore.h"
#include "llama-expert-heatmap.h"
#include "llama-expert-preload.h"
#include "llama-expert-tier.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <regex>

#ifdef _WIN32
#include <windows.h>
#endif

// FNV-1a of the first min(n, 1024) bytes - matches the launch hash sampling
static uint64_t hash_slice_at(const uint8_t * p, size_t n) {
    const size_t m = std::min(n, (size_t) 1024);
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t j = 0; j < m; j++) {
        h ^= p[j];
        h *= 1099511628211ULL;
    }
    return h;
}

static bool verify_gpu_copy(ggml_tensor * dst, size_t slot_off, const ggml_tensor * src_tensor, int expert, const uint8_t * src_data, size_t len) {
    // strict: hash the same first 1024 bytes of the GPU copy and compare to
    // the authoritative data - the launch hash (no-mmap) or the source slice
    // (mmap, where the model tensor is the file-backed ground truth).
    const size_t n = std::min(len, (size_t) 1024);
    std::vector<uint8_t> buf(n);
    ggml_backend_tensor_get(dst, buf.data(), slot_off, n);
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t j = 0; j < n; j++) {
        h ^= buf[j];
        h *= 1099511628211ULL;
    }
    uint64_t expected = llama_expert_preload::expected_hash(src_tensor, expert);
    if (expected == 0 && src_data) {
        expected = 0xcbf29ce484222325ULL;
        for (size_t j = 0; j < n; j++) {
            expected ^= src_data[j];
            expected *= 1099511628211ULL;
        }
    }
    if (expected == 0) {
        return false; // no ground truth - do not trust the copy
    }
    if (getenv("LLAMA_EXPERT_DEBUG")) {
        fprintf(stderr, "hotstore: verify expert %d expected=%016llx gpu=%016llx %s\n",
            expert, (unsigned long long) expected, (unsigned long long) h,
            h == expected ? "MATCH" : "MISMATCH");
    }
    return h == expected;
}

// matches the weight tensor of an expert tensor, e.g.:
//   blk.0.ffn_gate_exps.weight
//   blk.3.ffn_down_chexps.weight
// follows the same convention as LLM_FFN_EXPS_REGEX in common.h
static const std::regex g_re_exps_weight("blk\\.(\\d+)\\.ffn_(up|down|gate|gate_up)_(ch|)exps\\.weight");

llama_expert_hotstore::llama_expert_hotstore(
        const llama_model * model, int n_layers, int n_experts, int hot_s, int sync_period,
        float hyst, int dwell, bool copy_mode) :
    n_layers(n_layers),
    n_experts(n_experts),
    hot_s(hot_s),
    bytes_per_slot(n_layers, 0),
    sync_period(sync_period),
    hyst(hyst),
    dwell(dwell),
    copy_mode(copy_mode) {
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
        gpu_routed.assign(n_layers, std::vector<char>(this->n_experts, 0));
        pending_in.assign(n_layers, {});
        pending_out.assign(n_layers, {});
        // staging: one expert slot per exps tensor, per layer
        cpu_staging_off.assign(n_layers, 0);
        size_t off = 0;
        for (int il = 0; il < n_layers; il++) {
            cpu_staging_off[il] = off;
            for (entry * e : entries_by_layer[il]) {
                off += ggml_nbytes(e->src) / (size_t) e->src->ne[2];
            }
        }
        cpu_staging.resize(off, 0);
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

        // adopt the loader-streamed store buffer if present (the startup batch
        // is already resident and the sentinel planes are zeroed), else
        // allocate a fresh buffer
        ggml_backend_buffer_t pre = llama_expert_preload::take_buffer();
        if (pre) {
            char * base = (char *) ggml_backend_buffer_get_base(pre);
            for (auto & e : entries) {
                const llama_expert_preload::entry * pe = nullptr;
                for (size_t i = 0; i < llama_expert_preload::num_entries(); i++) {
                    const auto * cand = llama_expert_preload::entry_at(i);
                    if (cand->src == e.src) {
                        pe = cand;
                        break;
                    }
                }
                if (!pe) {
                    throw std::runtime_error(format("%s: preload entry missing for %s", __func__, e.src->name));
                }
                ggml_backend_tensor_alloc(pre, e.dst[g], base + pe->gpu_offset);
                if (getenv("LLAMA_EXPERT_DEBUG")) {
                    static int once = 0;
                    if (!once) {
                        once = 1;
                        std::vector<uint8_t> plane(1024);
                        for (int ex = 0; ex < (int) e.dst[g]->ne[2]; ex++) {
                            ggml_backend_tensor_get(e.dst[g], plane.data(), (size_t) ex * pe->plane_bytes, 1024);
                            uint64_t h = 0xcbf29ce484222325ULL;
                            for (int i = 0; i < 1024; i++) {
                                h ^= plane[i];
                                h *= 0x100000001b3ULL;
                            }
                            const uint64_t exp = ex < hot_s ? llama_expert_preload::expected_hash(e.src, ex) : 0;
                            fprintf(stderr, "hotstore: slot %d fnv=%016llx expected=%016llx %s\n",
                                ex, (unsigned long long) h, (unsigned long long) exp,
                                h == exp ? "MATCH" : "MISMATCH");
                        }
                    }
                }
            }
            size_t lut_off = llama_expert_preload::align_up256(llama_expert_preload::entries_size());
            for (int il = 0; il < n_layers; il++) {
                lut_off = llama_expert_preload::align_up256(lut_off);
                ggml_backend_tensor_alloc(pre, luts[il].hot_lut[g], base + lut_off);
                lut_off += llama_expert_preload::align_up256((size_t) n_experts * sizeof(int32_t));
                lut_off = llama_expert_preload::align_up256(lut_off);
                ggml_backend_tensor_alloc(pre, luts[il].mask_lut[g], base + lut_off);
                lut_off += llama_expert_preload::align_up256((size_t) (local_slots + 1) * sizeof(float));
            }
            buf_dev[g] = ggml_backend_buffer_ptr(pre);
            ggml_backend_buffer_set_usage(buf_dev[g].get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            preloaded = true;
        } else {
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
        }

        // sentinel mask: 1.0 for real slots, 0.0 for the sentinel plane, so
        // sentinel-routed hot rows are zeroed after the GPU mul_mat_id.
        std::vector<float> mask_h(local_slots + 1, 1.0f);
        mask_h[local_slots] = 0.0f;
        for (int il = 0; il < n_layers; il++) {
            ggml_backend_tensor_set(luts[il].mask_lut[g], mask_h.data(), 0,
                (local_slots + 1) * sizeof(float));
            if (getenv("LLAMA_EXPERT_DEBUG") && il == 0 && g == 0) {
                static int once = 0;
                if (!once) {
                    once = 1;
                    std::vector<float> rb(local_slots + 1);
                    ggml_backend_tensor_get(luts[il].mask_lut[g], rb.data(), 0,
                        (local_slots + 1) * sizeof(float));
                    fprintf(stderr, "hotstore: mask_lut[0..3]=%.1f %.1f %.1f %.1f [95]=%.1f [96]=%.1f\n",
                        rb[0], rb[1], rb[2], rb[3], rb[95], rb[96]);
                }
            }
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
        luts[il].counts    = ggml_new_tensor_1d(ctx_cpu.get(), GGML_TYPE_I32, n_experts + 1);
    }
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_t b_cpu = ggml_backend_alloc_ctx_tensors_from_buft(ctx_cpu.get(), cpu_buft);
    if (b_cpu) {
        buf_cpu = ggml_backend_buffer_ptr(b_cpu);
        ggml_backend_buffer_set_usage(buf_cpu.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    // the preloaded store carries the startup batch already; publish the
    // matching LUTs now so the tier is correct from the very first token
    // (copy_top_s later re-runs them, idempotently).
    if (preloaded) {
        for (int il = 0; il < n_layers; il++) {
            for (int p = 0; p < hot_s; p++) {
                slot_to_expert[il][p] = p;
            }
        }
        update_luts();
    }


    // register each expert weight tensor with the tier hook so build_lora_mm_id
    // can find its per-device GPU hot tensors and per-device LUTs.
    for (const auto & e : entries) {
        const auto & L = luts[e.layer_idx];
        llama_expert_tier_register(e.src, e.dst, L.hot_lut, L.mask_lut, L.cold_mask, L.counts);
    }

    return true;
}

llama_expert_hotstore::~llama_expert_hotstore() {
    llama_expert_tier_clear();
    llama_expert_preload::clear();
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
            const int pidx = llama_expert_preload::index_of(e->src);
            if (preloaded) {
                // the startup slices were streamed into the store at load
                // (write_entry), so the GPU store is already sound.
                continue;
            }
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
                // page hints are handled by llama_expert_pin (mmap only);
                // DONTNEED here would refault from disk on eviction
            }
        }
        // startup batch is trusted on the GPU (filled from the file, hash-checked);
        // the GPU output counts for it from the first token
        for (int p = 0; p < hot_s; p++) {
            if (ste[p] >= 0) {
                gpu_routed[il][ste[p]] = 1;
            }
        }
    }

    last_sync_tokens = heatmap.tokens_total;
    is_filled = true;
    update_luts();
    fprintf(stderr, "hotstore: startup batch moved to GPU\n");
    return true;
}

// LLAMA_EXPERT_FULL_SYNC: direct swap (1 per layer per token). copy the
// new expert to a free or gate-cleared slot, verify the GPU copy, route
// immediately. no handshake pacing queues - the verify is the safety net.
bool llama_expert_hotstore::resync_full_mirror(const llama_expert_heatmap & heatmap, int budget) {
    if (!is_filled || hot_s <= 0 || buf_dev.empty()) {
        return false;
    }
    const int64_t elapsed = heatmap.tokens_total - last_sync_tokens;
    int swapped = 0;
    for (int il = 0; il < n_layers; il++) {
        auto & ste   = slot_to_expert[il];
        auto & dc    = dwell_count[il];
        auto & rout  = gpu_routed[il];
        std::vector<char> resident_set(n_experts, 0);
        for (int p = 0; p < hot_s; p++) {
            if (ste[p] >= 0) {
                resident_set[ste[p]] = 1;
            }
        }
        const std::vector<int> top = heatmap.get_top_s(il, hot_s);

        auto find_slot = [&](int e_cold) -> int {
            for (int p = 0; p < hot_s; p++) {
                if (ste[p] < 0) {
                    return p;
                }
            }
            const float s_cold = heatmap.get_score(il, e_cold);
            int p_worst = -1;
            float worst_score = 1e9f;
            for (int p = 0; p < hot_s; p++) {
                if (ste[p] < 0) {
                    continue;
                }
                if (dc[p] < dwell) {
                    continue;
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
            if (swapped_in_layer >= budget) {
                break;
            }
            if (resident_set[e_cold]) {
                continue;
            }
            const int p = find_slot(e_cold);
            if (p < 0) {
                break;
            }
            const int g = slot_device(slot_start, slot_end, p);
            if (g < 0) {
                continue;
            }
            const int e_out = ste[p];
            // evicted expert: copy its slices back out (copy-on-read), route it to the CPU
            if (e_out >= 0) {
                for (entry * ent : entries_by_layer[il]) {
                    const int pidx = llama_expert_preload::index_of(ent->src);
                    if (pidx < 0) {
                        continue;
                    }
                    const size_t slot = ggml_nbytes(ent->src) / (size_t) ent->src->ne[2];
                    const size_t off = (size_t) (p - slot_start[g]) * slot;
                    std::vector<uint8_t> out(slot);
                    ggml_backend_tensor_get(ent->dst[g], out.data(), off, slot);
                    llama_expert_preload::set_cpu_slice(pidx, e_out, out.data());
                }
                rout[e_out] = 0;
            }
            // new expert: copy from the host, route immediately (verify dropped)
            bool ok = true;
            for (entry * ent : entries_by_layer[il]) {
                const int pidx = llama_expert_preload::index_of(ent->src);
                const size_t slot = ggml_nbytes(ent->src) / (size_t) ent->src->ne[2];
                const size_t off = (size_t) (p - slot_start[g]) * slot;
                const char * model_src = ent->src->data ? (const char *) ggml_get_data(ent->src) : nullptr;
                const uint8_t * src = pidx >= 0
                    ? llama_expert_preload::cpu_slice(pidx, e_cold)
                    : ((const uint8_t *) model_src) + (size_t) e_cold * slot;
                if (!src) {
                    ok = false;
                    continue;
                }
                ggml_backend_tensor_set(ent->dst[g], src, off, slot);
            }
            if (ok) {
                ste[p] = e_cold;
                rout[e_cold] = 1;
                dc[p]  = -elapsed; // fresh dwell
                swapped++;
                swapped_in_layer++;
            }
        }
    }

    last_sync_tokens = heatmap.tokens_total;
    if (swapped > 0) {
        update_luts();
    }
    return swapped > 0;
}

// byte offset of an exps tensor's single slot within its layer's staging region
static size_t staging_slot_off(const std::vector<llama_expert_hotstore::entry *> & layer, const llama_expert_hotstore::entry * ent) {
    size_t off = 0;
    for (const auto * e : layer) {
        if (e == ent) {
            break;
        }
        off += ggml_nbytes(e->src) / (size_t) e->src->ne[2];
    }
    return off;
}

bool llama_expert_hotstore::resync_top_s(const llama_expert_heatmap & heatmap) {
    if (!is_filled || hot_s <= 0 || buf_dev.empty()) {
        return false;
    }
    if (getenv("LLAMA_EXPERT_DEBUG")) {
        fprintf(stderr, "hotstore: resync tok=%lld\n", (long long) heatmap.tokens_total);
    }

    // tokens elapsed since the previous sync, used to age dwell counters
    const int64_t elapsed = heatmap.tokens_total - last_sync_tokens;
    int changed = 0;
    std::vector<char> dirty(n_layers, 0); // layers whose LUTs need rebuilding
    for (int il = 0; il < n_layers; il++) {
        auto & ste   = slot_to_expert[il];
        auto & dc    = dwell_count[il];
        auto & rout  = gpu_routed[il];
        auto & pin   = pending_in[il];
        auto & pout  = pending_out[il];
        auto & stg   = cpu_staging; // global staging
        const size_t stg_off = cpu_staging_off[il];

        // ---- pending move-outs: verify the staging copy (all per token),
        //      then hand the expert to the CPU and free the slot ------------
        for (auto it = pout.begin(); it != pout.end();) {
            if (!it->verified) {
                if (getenv("LLAMA_EXPERT_NO_VERIFY")) {
                    it->verified = true; // safety dropped: trust the staging copy
                } else {
                bool ok = true;
                for (entry * ent : entries_by_layer[il]) {
                    const int pidx = llama_expert_preload::index_of(ent->src);
                    if (pidx < 0) {
                        continue; // mmap: file-backed, nothing to verify
                    }
                    const size_t slot = ggml_nbytes(ent->src) / (size_t) ent->src->ne[2];
                    const uint64_t exp = llama_expert_preload::expected_hash(ent->src, it->expert);
                    const uint64_t got = hash_slice_at(&stg[stg_off + staging_slot_off(entries_by_layer[il], ent)], slot);
                    if (exp == 0 || exp != got) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    it->verified = true;
                } else if (++it->failures >= 5) {
                    // corruption recovery: the GPU -> staging copy kept failing,
                    // so re-read the expert straight from the gguf file into the
                    // CPU slice (ground truth); only then route it cold
                    bool recovered = true;
                    for (entry * ent : entries_by_layer[il]) {
                        const int pidx = llama_expert_preload::index_of(ent->src);
                        if (pidx < 0) {
                            continue; // mmap: data stays in the file
                        }
                        const size_t slot = ggml_nbytes(ent->src) / (size_t) ent->src->ne[2];
                        std::vector<uint8_t> buf(slot);
                        if (!llama_expert_preload::read_expert(pidx, it->expert, buf.data(), slot)) {
                            recovered = false;
                            continue;
                        }
                        if (llama_expert_preload::expected_hash(ent->src, it->expert) != hash_slice_at(buf.data(), slot)) {
                            recovered = false;
                        }
                        llama_expert_preload::set_cpu_slice(pidx, it->expert, buf.data());
                    }
                    if (recovered) {
                        rout[it->expert] = 0;
                        ste[it->p] = -1;
                        dirty[il] = 1;
                        changed++;
                        it = pout.erase(it);
                        continue;
                    }
                    if (getenv("LLAMA_EXPERT_DEBUG")) {
                        fprintf(stderr, "hotstore: move-out ABORT expert %d after %d fails\n", it->expert, it->failures);
                    }
                    it = pout.erase(it);
                    continue;
                }
                }
            }
            if (it->verified && --it->countdown <= 0) {
                // hand the expert to the CPU and free the slot
                for (entry * ent : entries_by_layer[il]) {
                    const int pidx = llama_expert_preload::index_of(ent->src);
                    if (pidx < 0) {
                        continue; // mmap: data stays in the file
                    }
                    const size_t slot = ggml_nbytes(ent->src) / (size_t) ent->src->ne[2];
                    llama_expert_preload::set_cpu_slice(pidx, it->expert, &stg[stg_off + staging_slot_off(entries_by_layer[il], ent)]);
                }
                rout[it->expert] = 0; // the CPU output counts now
                ste[it->p] = -1;      // the GPU slot is freed
                dirty[il] = 1;
                changed++;
                it = pout.erase(it);
            } else {
                ++it;
            }
        }

        // ---- pending move-ins: verify ALL copies every token; a confirmed
        //      one is routed to the GPU on the following token ---------------
        for (auto it = pin.begin(); it != pin.end();) {
            if (it->verified) {
                ++it;
                continue;
            }
            if (getenv("LLAMA_EXPERT_NO_VERIFY")) {
                it->verified = true; // safety dropped: trust the GPU copy
                ++it;
                continue;
            }
            const int g = slot_device(slot_start, slot_end, it->p);
            bool ok = g >= 0;
            if (g >= 0) {
                for (entry * ent : entries_by_layer[il]) {
                    const int pidx = llama_expert_preload::index_of(ent->src);
                    const size_t slot = ggml_nbytes(ent->src) / (size_t) ent->src->ne[2];
                    const size_t off = (size_t) (it->p - slot_start[g]) * slot;
                    const char * model_src = ent->src->data ? (const char *) ggml_get_data(ent->src) : nullptr;
                    const uint8_t * src = pidx >= 0
                        ? llama_expert_preload::cpu_slice(pidx, it->expert)
                        : ((const uint8_t *) model_src) + (size_t) it->expert * slot;
                    if (!src || !verify_gpu_copy(ent->dst[g], off, ent->src, it->expert, src, slot)) {
                        ok = false;
                    }
                }
            }
            if (ok) {
                it->verified = true;
                it->countdown = 1; // route to the GPU on the next token
                ++it;
            } else if (++it->failures >= 5) {
                // corruption recovery: the CPU slice kept failing to verify, so
                // re-read the expert straight from the gguf file (ground truth)
                // and re-copy it; keep the move-in only if the disk copy verifies
                bool recovered = true;
                const int g2 = it->p >= 0 ? slot_device(slot_start, slot_end, it->p) : -1;
                for (entry * ent : entries_by_layer[il]) {
                    const int pidx = llama_expert_preload::index_of(ent->src);
                    const size_t slot = ggml_nbytes(ent->src) / (size_t) ent->src->ne[2];
                    std::vector<uint8_t> buf(slot);
                    if (pidx < 0 || g2 < 0 || !llama_expert_preload::read_expert(pidx, it->expert, buf.data(), slot)) {
                        recovered = false;
                        continue;
                    }
                    const size_t off = (size_t) (it->p - slot_start[g2]) * slot;
                    ggml_backend_tensor_set(ent->dst[g2], buf.data(), off, slot);
                    if (!verify_gpu_copy(ent->dst[g2], off, ent->src, it->expert, buf.data(), slot)) {
                        recovered = false;
                    }
                }
                if (recovered) {
                    it->verified = true;
                    it->countdown = 1; // route to the GPU on the next token
                    ++it;
                } else {
                    ste[it->p] = -1;
                    it = pin.erase(it);
                }
            } else {
                ++it;
            }
        }
        for (auto it = pin.begin(); it != pin.end();) {
            if (it->verified && --it->countdown <= 0) {
                rout[it->expert] = 1; // the GPU output counts now
                if (!copy_mode) {
                    // move: the expert now lives on the GPU, free its RAM copy
                    for (entry * ent : entries_by_layer[il]) {
                        const int pidx = llama_expert_preload::index_of(ent->src);
                        if (pidx >= 0) {
                            llama_expert_preload::free_cpu_slice(pidx, it->expert);
                        }
                    }
                }
                dirty[il] = 1;
                changed++;
                it = pin.erase(it);
            } else {
                ++it;
            }
        }

        // ---- boundary gate: extract the actual bottom-GPU and top-CPU heats
        //      from the routed sets (the store lags the ideal ordering) and
        //      gate BOTH the eviction and the move-in on them.
        float lowest_gpu = INFINITY, highest_cpu = -INFINITY;
        for (int e = 0; e < n_experts; e++) {
            const float s = heatmap.get_score(il, e);
            if (rout[e]) {
                lowest_gpu = std::min(lowest_gpu, s);
            } else {
                highest_cpu = std::max(highest_cpu, s);
            }
        }
        if (highest_cpu > lowest_gpu) {
            // ---- eviction selection (1/token): a GPU expert leaves only when
            //      a CPU expert beats it by the hysteresis margin and it has
            //      dwelled. gated on the eviction queue capacity.
            int evict_candidate = -1;
            if (!llama_expert_preload::get_no_evict() && !getenv("LLAMA_EXPERT_NO_EVICT") && (int) pout.size() < max_concurrent_moves) {
                // never evict an expert still in the top-S: a swap must move a
                // genuinely cold expert out, not shuffle two GPU-bound experts
                const std::vector<int> top = heatmap.get_top_s(il, hot_s);
                std::vector<char> in_top(n_experts, 0);
                for (int te : top) {
                    in_top[te] = 1;
                }
                std::vector<int> routed;
                for (int e = 0; e < n_experts; e++) {
                    if (rout[e]) {
                        routed.push_back(e);
                    }
                }
                std::sort(routed.begin(), routed.end(), [&](int a, int b) {
                    return heatmap.get_score(il, a) < heatmap.get_score(il, b);
                });
                for (int e : routed) {
                    if (in_top[e]) {
                        break; // monotonic: warmer residents are in the top-S too
                    }
                    int p = -1;
                    for (int pp = 0; pp < hot_s; pp++) {
                        if (ste[pp] == e) {
                            p = pp;
                            break;
                        }
                    }
                    if (p < 0 || dc[p] < dwell) {
                        continue; // not an applicant yet
                    }
                    if (highest_cpu >= hyst * heatmap.get_score(il, e)) {
                        evict_candidate = e;
                    }
                    break; // monotonic: hotter experts fail too
                }
            }
            if (evict_candidate >= 0 && (int) pout.size() < max_concurrent_moves) {
                // queue: at most 2 move-outs per layer at a time
                const int p = [&]() { for (int pp = 0; pp < hot_s; pp++) if (ste[pp] == evict_candidate) return pp; return -1; }();
                const int dg = p >= 0 ? slot_device(slot_start, slot_end, p) : -1;
                if (p >= 0 && dg >= 0) {
                    // copy the evicted expert's slices into the staging buffer
                    for (entry * ent : entries_by_layer[il]) {
                        const int pidx = llama_expert_preload::index_of(ent->src);
                        if (pidx < 0) {
                            continue; // mmap: file-backed, no staging needed
                        }
                        const size_t slot = ggml_nbytes(ent->src) / (size_t) ent->src->ne[2];
                        const size_t off = (size_t) (p - slot_start[dg]) * slot;
                        ggml_backend_tensor_get(ent->dst[dg], &stg[stg_off + staging_slot_off(entries_by_layer[il], ent)], off, slot);
                    }
                    pout.push_back({evict_candidate, p, false, 0, 0});
                    dirty[il] = 1;
                    changed++; // the LUTs do not change yet, but the transition started
                }
            }
        } // boundary gate

        // ---- promote/demote (D2D): the top expert on a worse device that
        //      beats the better device's coldest resident by hysteresis, with
        //      the better slot dwelled, swaps through the staging buffer.
        //      synchronous copies + inline verify (copy_top_s style); both
        //      slots stay routed, only their device changes.
        for (int g = 0; g + 1 < n_devices; g++) {
            float worst_bound = INFINITY;
            int   worst_slot  = -1;
            for (int p = slot_start[g]; p < slot_end[g]; p++) {
                if (ste[p] < 0) {
                    continue;
                }
                const float s = heatmap.get_score(il, ste[p]);
                if (s < worst_bound) {
                    worst_bound = s;
                    worst_slot  = p;
                }
            }
            if (worst_slot < 0 || worst_bound <= 0.0f) {
                continue; // dead resident (no heat yet); eviction will clear it
            }
            if (dc[worst_slot] < dwell + 2) {
                continue; // not aged enough since its last change
            }
            int   best_slot  = -1;
            float best_score = -INFINITY;
            for (int p = slot_start[g+1]; p < slot_end[g+1]; p++) {
                if (ste[p] < 0) {
                    continue;
                }
                const float s = heatmap.get_score(il, ste[p]);
                if (s > best_score) {
                    best_score = s;
                    best_slot  = p;
                }
            }
            if (best_slot < 0 || best_score < hyst * worst_bound) {
                continue;
            }
            const int e_demote  = ste[worst_slot];
            const int e_promote = ste[best_slot];
            for (entry * ent : entries_by_layer[il]) {
                const size_t eslot = ggml_nbytes(ent->src) / (size_t) ent->src->ne[2];
                const size_t off_w = (size_t) (worst_slot - slot_start[g]) * eslot;
                const size_t off_b = (size_t) (best_slot - slot_start[g+1]) * eslot;
                std::vector<uint8_t> s1(eslot), s2(eslot);
                ggml_backend_tensor_get(ent->dst[g],   s1.data(), off_w, eslot); // demote
                ggml_backend_tensor_get(ent->dst[g+1], s2.data(), off_b, eslot); // promote
                ggml_backend_tensor_set(ent->dst[g+1], s1.data(), off_b, eslot);
                ggml_backend_tensor_set(ent->dst[g],   s2.data(), off_w, eslot);
            }
            ste[worst_slot] = e_promote;
            ste[best_slot]  = e_demote;
            // dwell is in resyncs; the counter ages by elapsed tokens per resync
            dc[worst_slot] = -elapsed * (dwell + 2);
            dc[best_slot]  = -elapsed * (dwell + 2);
            dirty[il] = 1;
            changed++;
            if (getenv("LLAMA_EXPERT_DEBUG")) {
                fprintf(stderr, "hotstore: d2d swap promote=%d demote=%d (dev %d -> %d) promote_s=%.2f worst=%.2f hyst=%.2f\n",
                    e_promote, e_demote, g+1, g, best_score, worst_bound, hyst * worst_bound);
            }
            break; // one D2D swap per layer per tick
        }

        // ---- move-in (1/token, gated): fill a free slot with the hottest CPU
        //      expert that should be on the GPU. check the free slot first (the
        //      common blocker), then the move-in queue (rarer), then evaluate.
        int free_slot = -1;
        for (int pp = 0; pp < hot_s; pp++) {
            if (ste[pp] < 0) {
                free_slot = pp;
                break;
            }
        }
        if (free_slot >= 0) {
            if ((int) pin.size() < max_concurrent_moves) { // the queue is rarer than a missing slot
                const std::vector<int> top = heatmap.get_top_s(il, hot_s);
                for (int e_cold : top) {
                    if (e_cold < 0 || e_cold >= n_experts || rout[e_cold]) {
                        continue;
                    }
                    const int dg = slot_device(slot_start, slot_end, free_slot);
                    if (dg < 0) {
                        break;
                    }
                    // copy the new expert in; keep it CPU-counted until verified
                    for (entry * ent : entries_by_layer[il]) {
                        const int pidx = llama_expert_preload::index_of(ent->src);
                        const size_t slot = ggml_nbytes(ent->src) / (size_t) ent->src->ne[2];
                        const size_t off = (size_t) (free_slot - slot_start[dg]) * slot;
                        const char * model_src = ent->src->data ? (const char *) ggml_get_data(ent->src) : nullptr;
                        const uint8_t * src = pidx >= 0
                            ? llama_expert_preload::cpu_slice(pidx, e_cold)
                            : ((const uint8_t *) model_src) + (size_t) e_cold * slot;
                        if (!src) {
                            continue;
                        }
                        ggml_backend_tensor_set(ent->dst[dg], src, off, slot);
                    }
                    ste[free_slot] = e_cold;
                    pin.push_back({e_cold, free_slot, false, 0, 0});
                    dc[free_slot] = -elapsed;
                    dirty[il] = 1;
                    changed++;
                    break; // one move-in per layer per token
                }
            }
        }

        for (int p = 0; p < hot_s; p++) {
            if (ste[p] >= 0) {
                dc[p] += (int) std::max<int64_t>(elapsed, 0);
            }
        }
    }

    last_sync_tokens = heatmap.tokens_total;
    if (changed > 0) {
        update_luts(dirty);
        if (getenv("LLAMA_EXPERT_DEBUG")) {
            fprintf(stderr, "hotstore: re-sync changed %d slots\n", changed);
        }
    }
    return changed > 0;
}

bool llama_expert_hotstore::maybe_resync(const llama_expert_heatmap & heatmap, bool multi_slot) {
    // n_tokens>1 (multi-slot) freezes the hot store: no swapping during the batch
    if (multi_slot || heatmap.tokens_total <= 0) {
        return false;
    }
    if (frozen) {
        return false;
    }
    // continuous adaptive cadence: the sync period grows with the hit rate
    // relative to the target. floor of 10 tokens (never resync more often),
    // stretching to 32 as the hit rate climbs toward 6x the target.
    float ratio = hit_rate_valid ? hit_rate / target_hit_rate() : 0.0f;
    const int eff_period = std::max(10, std::min(32, 10 + (int) (22.0f * (ratio - 1.0f) / 5.0f)));
    if (heatmap.tokens_total / eff_period > last_sync_tokens / eff_period) {
        if (getenv("LLAMA_EXPERT_FULL_SYNC")) {
            // test: mirror the whole store to the top-S on each sync instead of
            // the incremental handshake (pair with --expert-sync-period N)
            return resync_full_mirror(heatmap, std::max(1, hot_s));
        }
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

void llama_expert_hotstore::update_luts(const std::vector<char> & dirty) {
    if (hot_s <= 0 || luts.empty() || buf_dev.empty()) {
        return;
    }
    if (getenv("LLAMA_EXPERT_DEBUG")) {
        static int once = 0;
        if (!once) {
            once = 1;
            const char * base = (const char *) ggml_backend_buffer_get_base(buf_dev[0].get());
            fprintf(stderr, "hotstore: update_luts hot_lut[0] data_off=%td buf_base=%p\n",
                (const char *) luts[0].hot_lut[0]->data - base, (const void *) base);
        }
    }

    std::vector<int32_t> cold_mask_h(n_experts);

    for (int il = 0; il < n_layers; il++) {
        if (!dirty.empty() && !dirty[il]) {
            continue; // this layer's store did not change; LUT already current
        }
        const auto & ste   = slot_to_expert[il];
        const auto & rout  = gpu_routed[il];

        // per-device LUTs: an expert whose slot is on device g AND whose output
        // currently comes from the GPU maps to the LOCAL slot index there;
        // everything else (cold / pending) maps to the sentinel slot.
        for (int g = 0; g < n_devices; g++) {
            const int local_slots = slot_end[g] - slot_start[g];
            std::vector<int32_t> hot_lut_h(n_experts, local_slots);
            for (int p = slot_start[g]; p < slot_end[g]; p++) {
                const int e = ste[p];
                if (e >= 0 && e < n_experts && rout[e]) {
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
        // GPU-counted experts override
        for (int e = 0; e < n_experts; e++) {
            if (rout[e]) {
                cold_mask_h[e] = 0;
            }
        }
        ggml_backend_tensor_set(luts[il].cold_mask, cold_mask_h.data(), 0,
            n_experts * sizeof(int32_t));
        if (getenv("LLAMA_EXPERT_DEBUG") && il == 0) {
            static int once = 0;
            if (!once) {
                once = 1;
                std::vector<int32_t> rb_hot(n_experts);
                ggml_backend_tensor_get(luts[0].hot_lut[0], rb_hot.data(), 0, n_experts * sizeof(int32_t));
                fprintf(stderr, "hotstore: hot_lut[0..7]=%d %d %d %d %d %d %d %d cold_mask[0..3]=%d %d %d %d [96]=%d [97]=%d\n",
                    rb_hot[0], rb_hot[1], rb_hot[2], rb_hot[3], rb_hot[4], rb_hot[5], rb_hot[6], rb_hot[7],
                    ((const int32_t *) luts[0].cold_mask->data)[0],
                    ((const int32_t *) luts[0].cold_mask->data)[1],
                    ((const int32_t *) luts[0].cold_mask->data)[2],
                    ((const int32_t *) luts[0].cold_mask->data)[3],
                    ((const int32_t *) luts[0].cold_mask->data)[96],
                    ((const int32_t *) luts[0].cold_mask->data)[97]);
            }
        }
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
        fprintf(stderr, "hotstore: hit rate %zu/%zu = %.1f%%\n", hits, total, 100.0f * (float) hits / (float) total);
    }
}

void llama_expert_hotstore::reset_counts() {
    for (int il = 0; il < n_layers; il++) {
        ggml_tensor * c = luts[il].counts;
        if (c && c->data) {
            memset(c->data, 0, (size_t) (n_experts + 1) * sizeof(int32_t));
        }
    }
}

void llama_expert_hotstore::read_counts(llama_expert_heatmap & heatmap, int n_tokens) {
    if (luts.size() != (size_t) n_layers) {
        return;
    }
    std::vector<const int32_t *> per_layer((size_t) n_layers, nullptr);
    int64_t total = 0, hot = 0;
    for (int il = 0; il < n_layers; il++) {
        ggml_tensor * c = luts[il].counts;
        const int32_t * cnt = c && c->data ? (const int32_t *) c->data : nullptr;
        per_layer[il] = cnt;
        if (!cnt) {
            continue;
        }
        const auto & rout = gpu_routed[il];
        for (int e = 0; e < n_experts; e++) {
            if (cnt[e] > 0) {
                total += cnt[e];
                if (rout[e]) {
                    hot += cnt[e];
                }
            }
        }
    }
    if (total > 0) {
        const float inst = (float) hot / (float) total;
        hit_rate = hit_rate_valid ? 0.9f * hit_rate + 0.1f * inst : inst;
        hit_rate_valid = true;
    }
    heatmap.update_counts(per_layer, n_tokens);
}

float llama_expert_hotstore::target_hit_rate() const {
    if (n_experts <= 0) {
        return 0.5f;
    }
    const float frac = (float) hot_s / (float) n_experts;
    const float t = 0.8f * std::pow(frac, 0.6f);
    return std::max(0.2f, std::min(0.9f, t));
}

void llama_expert_hotstore::log() const {
    fprintf(stderr, "hotstore: sizing (S=%d)\n", hot_s);
    const bool debug = getenv("LLAMA_EXPERT_DEBUG") != nullptr;
    size_t total = 0;    for (int il = 0; il < n_layers; il++) {
        total += bytes_per_slot[il];
        if (debug) {
            fprintf(stderr, "  layer %3d: bytes/slot = %zu\n", il, bytes_per_slot[il]);
        }
    }
    fprintf(stderr, "  total bytes/slot across all layers = %zu (%zu MiB)\n",
        total, total / (1024 * 1024));
    if (!buf_dev.empty()) {
        fprintf(stderr, "  GPU hot store allocated: %s, %zu bytes (%zu MiB) for %d+1 slots across %d device(s) (%d expert + 1 sentinel per device)\n",
            ggml_backend_buffer_name(buf_dev[0].get()),
            ggml_backend_buffer_get_size(buf_dev[0].get()),
            ggml_backend_buffer_get_size(buf_dev[0].get()) / (1024 * 1024),
            hot_s, n_devices, hot_s);
    } else if (hot_s > 0) {
        fprintf(stderr, "  hot store DISABLED (%d slots requested)\n", hot_s);
    }
}

