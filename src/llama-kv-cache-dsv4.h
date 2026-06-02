#pragma once

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

class llama_dsv4_comp_state {
public:
    llama_dsv4_comp_state(
            const llama_model & model,
                    bool        offload,
                bool        unified,
            uint32_t        n_seq_max,
            uint32_t        ratio,
            uint32_t        state_size,
            uint32_t        n_embd_state,
            const char    * name,
        const llama_memory_i::layer_filter_cb & filter);

    void clear(bool data);

    uint32_t get_ratio()    const;
    uint32_t get_state_size() const;
    uint32_t get_n_stream() const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    ggml_tensor * get_kv   (ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_score(ggml_context * ctx, int32_t il) const;

    ggml_tensor * cpy_kv   (ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, int32_t il) const;
    ggml_tensor * cpy_score(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, int32_t il) const;

private:
    struct layer {
        uint32_t il;

        ggml_tensor * kv;
        ggml_tensor * score;
    };

    const uint32_t ratio;
    const uint32_t state_size;
    const uint32_t n_embd_state;
    const uint32_t n_stream;

    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    std::vector<layer> layers;

    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;
};

//
// llama_kv_cache_dsv4
//

// DSV4 uses a normal raw/SWA token cache plus compressed K-only block caches.
// The compressed caches are storage only; DSV4-specific visibility and block
// planning are handled by llama_kv_cache_dsv4_context / llm_graph_input_dsv4.

class llama_kv_cache_dsv4 : public llama_memory_i {
public:
    llama_kv_cache_dsv4(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   swa_full,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_ubatch,
                     uint32_t   n_pad,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse);

    ~llama_kv_cache_dsv4() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache_dsv4 specific API
    //

    llama_kv_cache_iswa * get_raw() const;
    llama_kv_cache      * get_csa() const;
    llama_kv_cache      * get_hca() const;
    llama_kv_cache      * get_lid() const;
    llama_dsv4_comp_state * get_csa_state() const;
    llama_dsv4_comp_state * get_hca_state() const;
    llama_dsv4_comp_state * get_lid_state() const;

private:
    llama_hparams hparams_csa;
    llama_hparams hparams_hca;
    llama_hparams hparams_lid;

    std::unique_ptr<llama_kv_cache_iswa> kv_raw;
    std::unique_ptr<llama_kv_cache>      kv_csa;
    std::unique_ptr<llama_kv_cache>      kv_hca;
    std::unique_ptr<llama_kv_cache>      kv_lid;
    std::unique_ptr<llama_dsv4_comp_state> csa_state;
    std::unique_ptr<llama_dsv4_comp_state> hca_state;
    std::unique_ptr<llama_dsv4_comp_state> lid_state;

    void clear_compressed(bool data);
};

class llama_kv_cache_dsv4_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    struct comp_plan {
        uint32_t ratio = 0;

        // Logical compressed row ids written by the current graph.
        std::vector<int64_t> write_idxs;

        // Position used for compressor RoPE. For a completed block this is the
        // first source-token position of that block.
        std::vector<int32_t> write_pos;

        // Position at which the compressed row becomes visible to attention.
        std::vector<int32_t> write_end;

        // Completed blocks that could not be planned. This should remain empty
        // for the scratch-backed state path.
        std::vector<int32_t> pending_end;

        // Compressor-state row ids updated by the current graph.
        std::vector<int32_t> state_idxs;

        // APE row ids, i.e. pos % ratio, for the compressor-state updates.
        std::vector<int32_t> state_pos;

        // Flattened source row ids used for state-backed commits. Source rows
        // index the graph-local [persistent_state | current_ubatch_scratch]
        // tensor. For overlapped compression the first half is previous rows
        // and the second half is current rows; a final synthetic zero/-inf row
        // may be addressed for the first block's previous half.
        std::vector<int32_t> state_read_idxs;

        // Final compressed-cache row ids written by state-backed commits.
        std::vector<int64_t> state_write_idxs;

        // RoPE positions for state-backed commits.
        std::vector<int32_t> state_write_pos;

        // End positions for state-backed commits.
        std::vector<int32_t> state_write_end;

        // Number of completed compressed rows visible for each query token.
        std::vector<int32_t> n_visible;

        // Maximum compressed rows visible to this ubatch.
        int64_t n_kv = 0;
    };

    llama_kv_cache_dsv4_context(llama_memory_status status);

    llama_kv_cache_dsv4_context(
            llama_kv_cache_dsv4 * kv);

    llama_kv_cache_dsv4_context(
            llama_kv_cache_dsv4 * kv,
            llama_context * lctx,
            bool optimize);

    llama_kv_cache_dsv4_context(
            llama_kv_cache_dsv4 * kv,
            slot_info_vec_t sinfos_raw_base,
            slot_info_vec_t sinfos_raw_swa,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_dsv4_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_dsv4_context specific API
    //

    const llama_kv_cache_iswa_context * get_raw() const;
    const llama_kv_cache_context      * get_csa() const;
    const llama_kv_cache_context      * get_hca() const;
    const llama_kv_cache_context      * get_lid() const;
    const llama_dsv4_comp_state       * get_csa_state() const;
    const llama_dsv4_comp_state       * get_hca_state() const;
    const llama_dsv4_comp_state       * get_lid_state() const;

    const comp_plan & get_csa_plan() const;
    const comp_plan & get_hca_plan() const;
    const comp_plan & get_lid_plan() const;

private:
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    std::vector<comp_plan> plans_csa;
    std::vector<comp_plan> plans_hca;
    std::vector<comp_plan> plans_lid;

    const llama_memory_context_ptr ctx_raw;
    const llama_memory_context_ptr ctx_csa;
    const llama_memory_context_ptr ctx_hca;
    const llama_memory_context_ptr ctx_lid;

    const llama_dsv4_comp_state * csa_state = nullptr;
    const llama_dsv4_comp_state * hca_state = nullptr;
    const llama_dsv4_comp_state * lid_state = nullptr;

    const llama_memory_status status;
};
