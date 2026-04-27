#include "llama-memory-pshard.h"

#include "llama-impl.h"
#include "llama-cparams.h"

#include "ggml-backend.h"

#include <algorithm>

bool llama_memory_pshard::init(
        const std::vector<tensor_spec>         & specs,
        const std::unordered_map<int, int32_t> & layer_backend_ids,
        int32_t                                  cpu_backend_id,
        bool                                     no_alloc) {

    layers.clear();
    streams.clear();
    ctxs.clear();
    bufs.clear();
    bufs_planned_sizes.clear();
    map_layer_ids.clear();
    cpu_bid_     = cpu_backend_id;
    layer_bids_  = layer_backend_ids;

    if (specs.empty()) return true;

    auto * main_gpu_dev  = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    auto * buft_gpu      = ggml_backend_dev_buffer_type(main_gpu_dev);
    auto * buft_cpu_host = ggml_backend_dev_host_buffer_type(main_gpu_dev);

    auto is_sharded = [&](uint32_t il) -> bool {
        auto it = layer_backend_ids.find(il);
        return it == layer_backend_ids.end() || it->second != 0;
    };

    size_t n_pinned  = 0;
    size_t n_sharded = 0;
    for (const auto & sp : specs) {
        if (is_sharded(sp.il)) { n_sharded++; } else { n_pinned++; }
    }

    uint32_t max_n_stream = 1;
    for (const auto & sp : specs) { max_n_stream = std::max(max_n_stream, sp.n_stream); }

    ggml_context * ctx_gpu_pinned  = nullptr;
    ggml_context * ctx_gpu_sharded = nullptr;

    auto make_ctx = [&](size_t n, uint32_t ns) -> ggml_context * {
        size_t overhead = 2u * (1 + ns) * n * ggml_tensor_overhead();
        ggml_init_params params = { overhead, NULL, true };
        ggml_context * c = ggml_init(params);
        if (c) ctxs.emplace_back(c);
        return c;
    };

    if (n_pinned  > 0) { ctx_gpu_pinned  = make_ctx(n_pinned,  max_n_stream); }
    if (n_sharded > 0) { ctx_gpu_sharded = make_ctx(n_sharded, max_n_stream); }
    if ((n_pinned > 0 && !ctx_gpu_pinned) || (n_sharded > 0 && !ctx_gpu_sharded)) {
        LLAMA_LOG_ERROR("%s: failed to create GPU tensor contexts\n", __func__);
        return false;
    }

    for (const auto & sp : specs) {
        const bool sharded = is_sharded(sp.il);
        ggml_context * ctx = sharded ? ctx_gpu_sharded : ctx_gpu_pinned;

        // dim_t2 == 0 means "no second tensor" (MLA caches K-only)
        ggml_tensor * t1 = nullptr;
        ggml_tensor * t2 = nullptr;
        if (sp.is_1d) {
            t1 = ggml_new_tensor_1d(ctx, sp.type_t1, sp.dim_t1);
            if (sp.dim_t2 > 0) {
                t2 = ggml_new_tensor_1d(ctx, sp.type_t2, sp.dim_t2);
            }
        } else {
            t1 = ggml_new_tensor_3d(ctx, sp.type_t1, sp.dim_t1, sp.seq_len, sp.n_stream);
            if (sp.dim_t2 > 0) {
                t2 = ggml_new_tensor_3d(ctx, sp.type_t2, sp.dim_t2, sp.seq_len, sp.n_stream);
            }
        }

        ggml_format_name(t1, "%s_l%d", sp.name_t1, sp.il);
        if (t2) ggml_format_name(t2, "%s_l%d", sp.name_t2, sp.il);

        stream_views sv;
        if (!sp.is_1d && sp.n_stream > 0) {
            for (uint32_t s = 0; s < sp.n_stream; ++s) {
                sv.t1_stream_gpu.push_back(ggml_view_2d(ctx, t1, sp.dim_t1, sp.seq_len, t1->nb[1], s * t1->nb[2]));
                if (t2) {
                    sv.t2_stream_gpu.push_back(ggml_view_2d(ctx, t2, sp.dim_t2, sp.seq_len, t2->nb[1], s * t2->nb[2]));
                }
            }
        }

        map_layer_ids[sp.il] = (int32_t)layers.size();

        layer l;
        l.il     = sp.il;
        l.t1_gpu = t1;
        l.t2_gpu = t2;
        l.t1_cpu = nullptr;
        l.t2_cpu = nullptr;
        layers.push_back(std::move(l));
        streams.push_back(std::move(sv));
    }

    if (ctx_gpu_pinned) {
        const size_t planned_size = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_gpu_pinned, buft_gpu);

        ggml_backend_buffer_t buf;
        if (no_alloc) {
            // dummy 0-size buffer; size accounted via bufs_planned_sizes for memory_breakdown.
            buf = ggml_backend_buft_alloc_buffer(buft_gpu, 0);
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_gpu_pinned); t != nullptr; t = ggml_get_next_tensor(ctx_gpu_pinned, t)) {
                t->buffer = buf;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_gpu_pinned, buft_gpu);
        }
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate GPU buffer for pinned layers\n", __func__);
            return false;
        }
        if (!no_alloc) {
            ggml_backend_buffer_clear(buf, 0);
        }

        LLAMA_LOG_INFO("%s: %10s pinned buffer = %8.2f MiB (%zu layers)%s\n",
            __func__, ggml_backend_buffer_name(buf),
            planned_size / 1024.0 / 1024.0, n_pinned,
            no_alloc ? " (no_alloc)" : "");

        bufs.emplace_back(buf);
        bufs_planned_sizes.push_back(planned_size);
    }

    {
        size_t overhead = 2u * (1 + max_n_stream) * specs.size() * ggml_tensor_overhead();
        ggml_init_params params = { overhead, NULL, true };
        ggml_context * ctx_cpu = ggml_init(params);
        if (!ctx_cpu) {
            LLAMA_LOG_ERROR("%s: failed to create CPU context\n", __func__);
            return false;
        }
        ctxs.emplace_back(ctx_cpu);

        for (size_t i = 0; i < specs.size(); ++i) {
            const auto & sp = specs[i];
            ggml_tensor * t1_cpu = nullptr;
            ggml_tensor * t2_cpu = nullptr;

            if (sp.is_1d) {
                t1_cpu = ggml_new_tensor_1d(ctx_cpu, sp.type_t1, sp.dim_t1);
                if (sp.dim_t2 > 0) {
                    t2_cpu = ggml_new_tensor_1d(ctx_cpu, sp.type_t2, sp.dim_t2);
                }
            } else {
                t1_cpu = ggml_new_tensor_3d(ctx_cpu, sp.type_t1, sp.dim_t1, sp.seq_len, sp.n_stream);
                if (sp.dim_t2 > 0) {
                    t2_cpu = ggml_new_tensor_3d(ctx_cpu, sp.type_t2, sp.dim_t2, sp.seq_len, sp.n_stream);
                }
            }

            ggml_format_name(t1_cpu, "%s_cpu_l%d", sp.name_t1, sp.il);
            if (t2_cpu) ggml_format_name(t2_cpu, "%s_cpu_l%d", sp.name_t2, sp.il);

            layers[i].t1_cpu = t1_cpu;
            layers[i].t2_cpu = t2_cpu;

            if (!sp.is_1d && sp.n_stream > 0) {
                for (uint32_t s = 0; s < sp.n_stream; ++s) {
                    streams[i].t1_stream_cpu.push_back(ggml_view_2d(ctx_cpu, t1_cpu, sp.dim_t1, sp.seq_len, t1_cpu->nb[1], s * t1_cpu->nb[2]));
                    if (t2_cpu) {
                        streams[i].t2_stream_cpu.push_back(ggml_view_2d(ctx_cpu, t2_cpu, sp.dim_t2, sp.seq_len, t2_cpu->nb[1], s * t2_cpu->nb[2]));
                    }
                }
            }
        }

        const size_t planned_size_cpu = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_cpu, buft_cpu_host);

        ggml_backend_buffer_t buf_cpu;
        if (no_alloc) {
            buf_cpu = ggml_backend_buft_alloc_buffer(buft_cpu_host, 0);
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_cpu); t != nullptr; t = ggml_get_next_tensor(ctx_cpu, t)) {
                t->buffer = buf_cpu;
            }
        } else {
            buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(ctx_cpu, buft_cpu_host);
        }
        if (!buf_cpu) {
            LLAMA_LOG_ERROR("%s: failed to allocate CPU-pinned buffer\n", __func__);
            return false;
        }
        if (!no_alloc) {
            ggml_backend_buffer_clear(buf_cpu, 0);
        }
        bufs.emplace_back(buf_cpu);
        bufs_planned_sizes.push_back(planned_size_cpu);
    }

    LLAMA_LOG_INFO("%s: %zu pinned, %zu sharded\n", __func__, n_pinned, n_sharded);
    return true;
}

bool llama_memory_pshard::is_cpu_only(int32_t il) const {
    auto it = layer_bids_.find((int)il);
    return it != layer_bids_.end() && it->second == cpu_bid_;
}

void llama_memory_pshard::activate_gpu(int32_t il) {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end() || !on_activate_gpu) return;
    auto & l = layers[it->second];
    on_activate_gpu(il, l.t1_gpu, l.t2_gpu);
}

void llama_memory_pshard::activate_cpu(int32_t il) {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end() || !on_activate_cpu) return;
    auto & l = layers[it->second];
    on_activate_cpu(il, l.t1_cpu, l.t2_cpu);
}

void llama_memory_pshard::assign_tensors(
        ggml_backend_sched_t sched,
        const std::unordered_map<int, int32_t> & layer_bids,
        const std::vector<ggml_backend_ptr> & backends,
        const pshard_dev_layout & layout) {
    for (auto & l : layers) {
        auto it = layer_bids.find((int)l.il);
        const bool has_bid = it != layer_bids.end() && it->second >= 0 && it->second < (int32_t)backends.size();
        if (!has_bid) {
            continue;
        }
        const int32_t bid = it->second;
        if (bid == layout.compute) {
            activate_gpu(l.il);
        } else if (bid == layout.cpu) {
            activate_cpu(l.il);
        } else {
            activate_gpu(l.il);
            l.t1_gpu->data = NULL; l.t1_gpu->buffer = NULL;
            ggml_backend_sched_set_tensor_backend(sched, l.t1_gpu, backends[bid].get());
            ggml_backend_sched_add_writeback(sched, l.t1_gpu);
            if (l.t2_gpu) {
                l.t2_gpu->data = NULL; l.t2_gpu->buffer = NULL;
                ggml_backend_sched_set_tensor_backend(sched, l.t2_gpu, backends[bid].get());
                ggml_backend_sched_add_writeback(sched, l.t2_gpu);
            }
        }
    }
}
