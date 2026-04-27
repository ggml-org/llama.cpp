#pragma once

#include "llama-pipe-shard.h"

#include <functional>
#include <vector>
#include <unordered_map>

struct llama_model;
struct llama_hparams;
struct ggml_tensor;

struct llama_memory_pshard : llama_memory_pipe_shard_i {
    enum transfer_mode {
        FULL,           // always transfer entire tensors (RS, or KV with v_trans)
        CELL_GRANULAR,  // transfer by cell ranges (KV without v_trans)
    };

    struct stream_views {
        std::vector<ggml_tensor *> t1_stream_gpu;
        std::vector<ggml_tensor *> t2_stream_gpu;
        std::vector<ggml_tensor *> t1_stream_cpu;
        std::vector<ggml_tensor *> t2_stream_cpu;
    };

    // parent cache callbacks set before init
    using activate_fn_t    = std::function<void(int32_t il, ggml_tensor * t1, ggml_tensor * t2)>;
    using cells_used_fn_t  = std::function<uint32_t(uint32_t stream_idx)>;

    transfer_mode   mode = FULL;
    activate_fn_t   on_activate_gpu;
    activate_fn_t   on_activate_cpu;
    cells_used_fn_t on_cells_used;  // null for FULL mode

    struct tensor_spec {
        uint32_t    il;
        ggml_type   type_t1;
        ggml_type   type_t2;
        uint32_t    dim_t1;       // ne[0] for t1
        uint32_t    dim_t2;       // ne[0] for t2
        uint32_t    seq_len;      // ne[1] (kv_size or flat for 1D)
        uint32_t    n_stream;     // ne[2] (1 for RS)
        bool        is_1d;        // true = ggml_new_tensor_1d (RS), false = 3D (KV)
        const char * name_t1 = "cache_k";   // e.g. "cache_k" or "cache_r"
        const char * name_t2 = "cache_v";   // e.g. "cache_v" or "cache_s"
    };

    bool init(
        const std::vector<tensor_spec>             & specs,
        const std::unordered_map<int, int32_t>     & layer_backend_ids,
        int32_t                                      cpu_backend_id,
        bool                                         no_alloc,
        ggml_backend_buffer_t                        preload_buf);

    bool is_cpu_only(int32_t il) const;

    ~llama_memory_pshard() override = default;

    // llama_memory_pipe_shard_i interface
    const std::vector<layer> & get_layers() const override { return layers; }

    void clear_prefetch() override;
    bool prefetch_if_owned(ggml_tensor * t, ggml_backend_t be) override;
    bool upload_if_owned(ggml_tensor * t, ggml_backend_t be) override;
    bool download_if_owned(ggml_tensor * t, ggml_backend_t be) override;

    void upload_for_switch(int32_t il, ggml_backend_t be) override;
    void download_for_switch(int32_t il, ggml_backend_t be) override;

    void activate_gpu(int32_t il) override;
    void activate_cpu(int32_t il) override;
    void prepare_for_host_access() override;
    void pin_layer(int32_t il) override;
    void unpin_layer(int32_t il) override;
    void set_external_addrs(int32_t il, void * a1, void * a2, size_t sz) override;

    void refresh_stream_views(int32_t il) override;

    void assign_tensors(
            ggml_backend_sched_t sched,
            const std::unordered_map<int, int32_t> & layer_bids,
            const std::vector<ggml_backend_ptr> & backends,
            const pshard_dev_layout & layout) override;

    size_t current_pinned_size() const override;

    // stream accessors (KV only, empty for RS)
    const stream_views & get_stream(size_t idx) const { return streams[idx]; }
    const std::vector<ggml_backend_buffer_ptr> & get_bufs() const { return bufs; }

    // planned buffer sizes including dummy no_alloc buffers
    const std::vector<size_t> & get_bufs_planned_sizes() const { return bufs_planned_sizes; }

private:
    std::vector<layer>                     layers;
    std::vector<stream_views>              streams;
    std::unordered_map<int32_t, int32_t>   map_layer_ids;
    std::vector<ggml_context_ptr>          ctxs;
    std::vector<ggml_backend_buffer_ptr>   bufs;
    std::vector<size_t>                    bufs_planned_sizes;
    ggml_backend_buffer_t                  external_buf = nullptr;
    int32_t                                cpu_bid_     = -1;
    std::unordered_map<int, int32_t>       layer_bids_;

    void upload_full(int32_t il, ggml_backend_t gpu);
    void download_full(int32_t il, ggml_backend_t be);
    void upload_cells(int32_t il, ggml_backend_t gpu, bool zero_tail);
    void download_cells(int32_t il, ggml_backend_t be);
    void download_written(int32_t il, const std::vector<std::vector<uint32_t>> & wc_per_stream, ggml_backend_t be);

    void upload_full_one(ggml_tensor * t_gpu, ggml_tensor * t_cpu, ggml_backend_t gpu);
    void download_full_one(ggml_tensor * t_gpu, ggml_tensor * t_cpu, ggml_backend_t be);
    void upload_cells_one(int32_t il, ggml_tensor * t_gpu, ggml_tensor * t_cpu, ggml_backend_t gpu, bool zero_tail);
    void download_cells_one(int32_t il, ggml_tensor * t_gpu, ggml_tensor * t_cpu, ggml_backend_t be);
    void download_written_one(int32_t il, ggml_tensor * t_gpu, ggml_tensor * t_cpu, const std::vector<std::vector<uint32_t>> & wc_per_stream, ggml_backend_t be);

    struct cell_range { uint32_t start; uint32_t count; };
    static std::vector<cell_range> batch_ranges(const std::vector<uint32_t> & sorted);
};
