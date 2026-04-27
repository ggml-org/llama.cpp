#include "llama-memory-pshard.h"

#include "llama-impl.h"
#include "llama-cparams.h"

#include "ggml-backend.h"

#include <algorithm>

bool llama_memory_pshard::init(
        const std::vector<tensor_spec>             & specs,
        const std::unordered_map<int, int32_t>     & layer_backend_ids,
        int32_t                                      cpu_backend_id,
        bool                                         no_alloc,
        ggml_backend_buffer_t                        preload_buf) {

    layers.clear();
    streams.clear();
    ctxs.clear();
    bufs.clear();
    bufs_planned_sizes.clear();
    map_layer_ids.clear();
    external_buf = preload_buf;
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

    if (external_buf) {
        LLAMA_LOG_INFO("%s: %zu layers created, addresses deferred to pack_cache_region\n",
            __func__, layers.size());
    } else {
        if (ctx_gpu_pinned) {
            const size_t planned_size = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_gpu_pinned, buft_gpu);

            ggml_backend_buffer_t buf;
            if (no_alloc) {
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

            for (auto & l : layers) {
                if (!is_sharded(l.il)) l.is_pinned = true;
            }
            bufs.emplace_back(buf);
            bufs_planned_sizes.push_back(planned_size);
        }
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

    LLAMA_LOG_INFO("%s: %zu pinned, %zu sharded, external_buf=%s, mode=%s\n",
        __func__, n_pinned, n_sharded,
        external_buf ? "yes" : "no",
        mode == FULL ? "full" : "cell_granular");

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

void llama_memory_pshard::prepare_for_host_access() {
    uint32_t n_activated = 0;
    for (const auto & l : layers) {
        if (!l.is_pinned) {
            activate_cpu(l.il);
            n_activated++;
        }
    }
    LLAMA_LOG_DEBUG("%s: activated %u/%zu sharded layers to CPU\n", __func__, n_activated, layers.size());
}

void llama_memory_pshard::pin_layer(int32_t il) {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end() || !external_buf) return;
    auto & l = layers[it->second];
    if (l.is_pinned) return;
    GGML_ASSERT(l.t1_gpu_addr && "pin_layer: t1 external address not set");
    if (l.t2_gpu) GGML_ASSERT(l.t2_gpu_addr && "pin_layer: t2 external address not set");
    LLAMA_LOG_DEBUG("pin_layer: il=%d t1_addr=%p t2_addr=%p\n", il, l.t1_gpu_addr, l.t2_gpu_addr);

    l.t1_gpu->data   = l.t1_gpu_addr;
    l.t1_gpu->buffer = external_buf;
    if (l.t2_gpu) {
        l.t2_gpu->data   = l.t2_gpu_addr;
        l.t2_gpu->buffer = external_buf;
    }

    auto & sv = streams[it->second];
    for (auto * v : sv.t1_stream_gpu) { v->data = (char *)l.t1_gpu->data + v->view_offs; }
    if (l.t2_gpu) {
        for (auto * v : sv.t2_stream_gpu) { v->data = (char *)l.t2_gpu->data + v->view_offs; }
    }

    l.is_pinned = true;
    activate_gpu(il);
}

void llama_memory_pshard::unpin_layer(int32_t il) {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) return;
    LLAMA_LOG_DEBUG("unpin_layer: il=%d\n", il);
    auto & l = layers[it->second];
    if (!l.is_pinned) return;
    LLAMA_LOG_DEBUG("%s: layer %d\n", __func__, il);

    l.t1_gpu->data   = NULL;
    l.t1_gpu->buffer = NULL;
    if (l.t2_gpu) {
        l.t2_gpu->data   = NULL;
        l.t2_gpu->buffer = NULL;
    }

    auto & sv = streams[it->second];
    for (auto * v : sv.t1_stream_gpu) { v->data = NULL; }
    if (l.t2_gpu) {
        for (auto * v : sv.t2_stream_gpu) { v->data = NULL; }
    }

    l.is_pinned = false;
    activate_cpu(il);
}

void llama_memory_pshard::set_external_addrs(int32_t il, void * a1, void * a2, size_t sz) {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) return;
    auto & l = layers[it->second];
    l.t1_gpu_addr = a1;
    l.t2_gpu_addr = a2;
    l.alloc_size  = sz;
}

void llama_memory_pshard::refresh_stream_views(int32_t il) {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) return;
    auto & l = layers[it->second];
    auto & sv = streams[it->second];
    if (l.t1_gpu && l.t1_gpu->data) {
        for (auto * v : sv.t1_stream_gpu) v->data = (char *)l.t1_gpu->data + v->view_offs;
    }
    if (l.t2_gpu && l.t2_gpu->data) {
        for (auto * v : sv.t2_stream_gpu) v->data = (char *)l.t2_gpu->data + v->view_offs;
    }
}

size_t llama_memory_pshard::current_pinned_size() const {
    size_t total = 0;
    for (const auto & l : layers) {
        if (l.is_pinned) total += l.alloc_size;
    }
    return total;
}

void llama_memory_pshard::upload_full_one(ggml_tensor * t_gpu, ggml_tensor * t_cpu, ggml_backend_t gpu) {
    if (!t_gpu || !t_cpu || !t_cpu->data) return;
    LLAMA_LOG_DEBUG("upload_full_one: name=%s gpu_data=%p cpu_data=%p bytes=%zu\n",
        t_gpu->name, t_gpu->data, t_cpu->data, ggml_nbytes(t_cpu));
    ggml_backend_tensor_set_async(gpu, t_gpu, t_cpu->data, 0, ggml_nbytes(t_cpu));
}

void llama_memory_pshard::download_full_one(ggml_tensor * t_gpu, ggml_tensor * t_cpu, ggml_backend_t be) {
    if (!t_gpu || !t_cpu || !t_cpu->data) return;
    LLAMA_LOG_DEBUG("download_full_one: name=%s gpu_data=%p cpu_data=%p bytes=%zu\n",
        t_gpu->name, t_gpu->data, t_cpu->data, ggml_nbytes(t_cpu));
    ggml_backend_tensor_get_async(be, t_gpu, t_cpu->data, 0, ggml_nbytes(t_cpu));
}

void llama_memory_pshard::upload_full(int32_t il, ggml_backend_t gpu) {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) return;
    auto & l = layers[it->second];
    GGML_ASSERT(l.t1_cpu && l.t1_cpu->data && "upload_full: CPU t1 not allocated");
    upload_full_one(l.t1_gpu, l.t1_cpu, gpu);
    upload_full_one(l.t2_gpu, l.t2_cpu, gpu);
}

void llama_memory_pshard::download_full(int32_t il, ggml_backend_t be) {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) return;
    auto & l = layers[it->second];
    GGML_ASSERT(l.t1_gpu && l.t1_cpu && l.t1_cpu->data && "download_full: t1 not allocated");
    download_full_one(l.t1_gpu, l.t1_cpu, be);
    download_full_one(l.t2_gpu, l.t2_cpu, be);
}

std::vector<llama_memory_pshard::cell_range>
llama_memory_pshard::batch_ranges(const std::vector<uint32_t> & sorted) {
    std::vector<cell_range> ranges;
    if (sorted.empty()) return ranges;
    uint32_t start = sorted[0], count = 1;
    for (size_t i = 1; i < sorted.size(); i++) {
        if (sorted[i] == start + count) {
            count++;
        } else {
            ranges.push_back({start, count});
            start = sorted[i];
            count = 1;
        }
    }
    ranges.push_back({start, count});
    return ranges;
}

void llama_memory_pshard::upload_cells_one(int32_t il, ggml_tensor * t_gpu, ggml_tensor * t_cpu, ggml_backend_t gpu, bool zero_tail) {
    (void) il;
    if (!t_gpu || !t_cpu || !t_cpu->data || !on_cells_used) return;

    const uint32_t ns     = (uint32_t)t_gpu->ne[2];
    const size_t   t_row  = t_gpu->ne[0] * ggml_element_size(t_gpu);
    const size_t   seq_sz = t_gpu->ne[1];

    LLAMA_LOG_DEBUG("upload_cells_one: il=%d name=%s gpu_data=%p cpu_data=%p ns=%u\n",
        il, t_gpu->name, t_gpu->data, t_cpu->data, ns);

    for (uint32_t s = 0; s < ns; s++) {
        if (write_cells && s < write_cells->size() && (*write_cells)[s].empty()) {
            continue;
        }

        uint32_t n_used = on_cells_used(s);
        size_t base = s * seq_sz * t_row;

        if (n_used > 0) {
            ggml_backend_tensor_set_async(gpu, t_gpu, (char *)t_cpu->data + base, base, n_used * t_row);
        }
        if (zero_tail && n_used < seq_sz) {
            size_t tail = base + n_used * t_row;
            size_t tail_n = seq_sz - n_used;
            ggml_backend_tensor_memset_async(gpu, t_gpu, 0, tail, tail_n * t_row);
        }
    }
}

void llama_memory_pshard::download_cells_one(int32_t il, ggml_tensor * t_gpu, ggml_tensor * t_cpu, ggml_backend_t be) {
    (void) il;
    if (!t_gpu || !t_cpu || !t_cpu->data || !on_cells_used) return;

    const uint32_t ns     = (uint32_t)t_gpu->ne[2];
    const size_t   t_row  = t_gpu->ne[0] * ggml_element_size(t_gpu);
    const size_t   seq_sz = t_gpu->ne[1];

    LLAMA_LOG_DEBUG("download_cells_one: il=%d name=%s gpu_data=%p cpu_data=%p ns=%u\n",
        il, t_gpu->name, t_gpu->data, t_cpu->data, ns);

    for (uint32_t s = 0; s < ns; s++) {
        uint32_t n_used = on_cells_used(s);
        if (n_used == 0) continue;

        size_t base = s * seq_sz * t_row;
        ggml_backend_tensor_get_async(be, t_gpu, (char *)t_cpu->data + base, base, n_used * t_row);
    }
}

void llama_memory_pshard::download_written_one(
        int32_t il, ggml_tensor * t_gpu, ggml_tensor * t_cpu,
        const std::vector<std::vector<uint32_t>> & wc_per_stream, ggml_backend_t be) {
    (void) il;
    if (!t_gpu || !t_cpu || !t_cpu->data) return;

    const size_t   t_row  = t_gpu->ne[0] * ggml_element_size(t_gpu);
    const size_t   seq_sz = t_gpu->ne[1];
    const uint32_t ns     = (uint32_t)t_gpu->ne[2];
    const uint32_t ns_wc  = (uint32_t)std::min((size_t)ns, wc_per_stream.size());

    LLAMA_LOG_DEBUG("download_written_one: il=%d name=%s gpu_data=%p cpu_data=%p ns_wc=%u\n",
        il, t_gpu->name, t_gpu->data, t_cpu->data, ns_wc);

    for (uint32_t s = 0; s < ns_wc; s++) {
        if (wc_per_stream[s].empty()) continue;

        const size_t base = s * seq_sz * t_row;

        std::vector<uint32_t> sorted = wc_per_stream[s];
        std::sort(sorted.begin(), sorted.end());
        sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
        auto ranges = batch_ranges(sorted);

        for (const auto & r : ranges) {
            GGML_ASSERT(r.start + r.count <= seq_sz && "download_written_one: cell range exceeds kv_size");
            size_t off = base + r.start * t_row;
            ggml_backend_tensor_get_async(be, t_gpu, (char *)t_cpu->data + off, off, r.count * t_row);
        }
    }
}

void llama_memory_pshard::upload_cells(int32_t il, ggml_backend_t gpu, bool zero_tail) {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) return;
    auto & l = layers[it->second];
    GGML_ASSERT(l.t1_cpu && l.t1_cpu->data && "upload_cells: CPU t1 not allocated");
    upload_cells_one(il, l.t1_gpu, l.t1_cpu, gpu, zero_tail);
    upload_cells_one(il, l.t2_gpu, l.t2_cpu, gpu, zero_tail);
}

void llama_memory_pshard::download_cells(int32_t il, ggml_backend_t be) {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) return;
    auto & l = layers[it->second];
    GGML_ASSERT(l.t1_cpu && l.t1_cpu->data && "download_cells: CPU t1 not allocated");
    download_cells_one(il, l.t1_gpu, l.t1_cpu, be);
    download_cells_one(il, l.t2_gpu, l.t2_cpu, be);
}

void llama_memory_pshard::download_written(
        int32_t il, const std::vector<std::vector<uint32_t>> & wc_per_stream, ggml_backend_t be) {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) return;
    auto & l = layers[it->second];
    GGML_ASSERT(l.t1_cpu && l.t1_cpu->data && "download_written: CPU t1 not allocated");
    download_written_one(il, l.t1_gpu, l.t1_cpu, wc_per_stream, be);
    download_written_one(il, l.t2_gpu, l.t2_cpu, wc_per_stream, be);
}

void llama_memory_pshard::clear_prefetch() {
    for (auto & l : layers) { l.prefetched_t1 = false; l.prefetched_t2 = false; }
}

bool llama_memory_pshard::prefetch_if_owned(ggml_tensor * t, ggml_backend_t copy_backend) {
    for (auto & l : layers) {
        ggml_tensor * t_gpu = (t == l.t1_gpu) ? l.t1_gpu : (t == l.t2_gpu) ? l.t2_gpu : nullptr;
        if (!t_gpu) continue;
        ggml_tensor * t_cpu = (t == l.t1_gpu) ? l.t1_cpu : l.t2_cpu;
        bool        & flag  = (t == l.t1_gpu) ? l.prefetched_t1 : l.prefetched_t2;

        if (mode == FULL) upload_full_one(t_gpu, t_cpu, copy_backend);
        else              upload_cells_one(l.il, t_gpu, t_cpu, copy_backend, true);
        flag = true;
        return true;
    }
    return false;
}

bool llama_memory_pshard::upload_if_owned(ggml_tensor * t, ggml_backend_t backend) {
    for (auto & l : layers) {
        ggml_tensor * t_gpu = (t == l.t1_gpu) ? l.t1_gpu : (t == l.t2_gpu) ? l.t2_gpu : nullptr;
        if (!t_gpu) continue;
        ggml_tensor * t_cpu = (t == l.t1_gpu) ? l.t1_cpu : l.t2_cpu;
        bool        & flag  = (t == l.t1_gpu) ? l.prefetched_t1 : l.prefetched_t2;

        if (flag) { flag = false; return true; }
        if (mode == FULL) upload_full_one(t_gpu, t_cpu, backend);
        else              upload_cells_one(l.il, t_gpu, t_cpu, backend, true);
        return true;
    }
    return false;
}

bool llama_memory_pshard::download_if_owned(ggml_tensor * t, ggml_backend_t backend) {
    for (auto & l : layers) {
        ggml_tensor * t_gpu = (t == l.t1_gpu) ? l.t1_gpu : (t == l.t2_gpu) ? l.t2_gpu : nullptr;
        if (!t_gpu) continue;
        ggml_tensor * t_cpu = (t == l.t1_gpu) ? l.t1_cpu : l.t2_cpu;

        if (mode == FULL)             download_full_one(t_gpu, t_cpu, backend);
        else if (write_cells)         download_written_one(l.il, t_gpu, t_cpu, *write_cells, backend);
        else                          download_full_one(t_gpu, t_cpu, backend);
        return true;
    }
    return false;
}

void llama_memory_pshard::upload_for_switch(int32_t il, ggml_backend_t be) {
    if (mode == FULL) {
        upload_full(il, be);
    } else {
        upload_cells(il, be, false);
    }
}

void llama_memory_pshard::download_for_switch(int32_t il, ggml_backend_t be) {
    if (mode == FULL) {
        download_full(il, be);
    } else {
        download_cells(il, be);
    }
}

void llama_memory_pshard::assign_tensors(
        ggml_backend_sched_t sched,
        const std::unordered_map<int, int32_t> & layer_bids,
        const std::vector<ggml_backend_ptr> & backends,
        const pshard_dev_layout & layout) {
    for (const auto & l : layers) {
        auto it = layer_bids.find((int)l.il);
        if (l.is_pinned) {
            GGML_ASSERT(l.t1_gpu->data != nullptr && "pinned layer missing GPU address");
            LLAMA_LOG_DEBUG("%s: layer %u -> pinned (GPU)\n", __func__, l.il);
            activate_gpu(l.il);
        } else if (it != layer_bids.end() && it->second >= 0 && it->second < (int32_t)backends.size()) {
            if (it->second == layout.cpu) {
                LLAMA_LOG_DEBUG("%s: layer %u -> CPU (bid=%d)\n", __func__, l.il, it->second);
                activate_cpu(l.il);
            } else {
                LLAMA_LOG_DEBUG("%s: layer %u -> shard (bid=%d)\n", __func__, l.il, it->second);
                activate_gpu(l.il);
                l.t1_gpu->data = NULL; l.t1_gpu->buffer = NULL;
                ggml_backend_sched_set_tensor_backend(sched, l.t1_gpu, backends[it->second].get());
                ggml_backend_sched_add_writeback(sched, l.t1_gpu);
                if (l.t2_gpu) {
                    l.t2_gpu->data = NULL; l.t2_gpu->buffer = NULL;
                    ggml_backend_sched_set_tensor_backend(sched, l.t2_gpu, backends[it->second].get());
                    ggml_backend_sched_add_writeback(sched, l.t2_gpu);
                }
            }
        } else if (!l.is_pinned) {
            LLAMA_LOG_WARN("%s: layer %u has no backend_id in plan -- left unconfigured\n", __func__, l.il);
        }
    }
}
