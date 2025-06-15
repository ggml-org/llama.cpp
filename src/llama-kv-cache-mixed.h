#pragma once

#include "llama-kv-cache.h"
#include "ggml.h"

#include <cstdint>
#include <memory>
#include <functional>
#include <vector>
#include <chrono>

// Forward declarations
struct llama_model;
struct llama_context;
struct ggml_tensor;

/**
 * Flash Attention Parameters for Custom Operation
 */
struct llama_flash_attn_mixed_params {
    float   scale;            // Scaling factor
    float   max_bias;         // Maximum bias for attention
    float   logit_softcap;    // Logit soft cap
    int32_t layer_id;         // Layer ID for mixed cache access
};


// Mixed KV cache configuration
struct llama_kv_cache_mixed_config {
        // Quantization settings
    bool     enable_quantization = true;    // Enable quantization
    uint32_t quantization_threshold = 4;    // Number of tokens before quantization (reduced for testing)
    uint32_t group_size = 16;               // Number of tokens to quantize at once
    uint32_t max_fp16_window = 1024;        // Maximum number of tokens to keep in FP16 window
    
    // Advanced quantization settings
    bool     adaptive_threshold = false;        // Dynamically adjust threshold based on memory pressure
    float    memory_pressure_threshold = 0.8f;  // Trigger quantization when memory usage > 80%
    uint32_t min_quantization_threshold = 16;   // Minimum threshold for adaptive mode
    uint32_t max_quantization_threshold = 128;  // Maximum threshold for adaptive mode

    uint32_t fp16_window_size = 0;      //> fp16_window_size is the number of tokens in the fp16 window.
    
    // Cache types
    ggml_type hot_type_k  = GGML_TYPE_F16;  // Recent tokens (FP16)
    ggml_type hot_type_v  = GGML_TYPE_F16;
    ggml_type cold_type_k = GGML_TYPE_Q4_0; // Old tokens (quantized)
    ggml_type cold_type_v = GGML_TYPE_Q4_0;
    
    // Performance monitoring
    bool     enable_stats = true;           // Enable quantization statistics
    uint32_t stats_report_interval = 1000; // Report stats every N tokens
};

//> ===================================================================================================
//> Custom Flash Attention Implementation for F32
//> ===================================================================================================
void ggml_compute_forward_flash_attn_ext_f32(
        ggml_tensor * dst,
        int ith,
        int nth,
        void* wdata,
        size_t wsize,
        void * userdatat);

//> =================================================================================================
//> Custom Flash Attention Implementation for Mixed KV Cache
//> =================================================================================================
void ggml_custom_flash_attn_mixed_simple(
    ggml_tensor * dst,
    int ith,
    int nth,
    void* wdata,
    size_t wsize,
    void * userdata
);

/*
 * llama_kv_cache_mixed
 * 
 * Mixed precision KV cache implementation with automatic quantization.
 * 
 * Design Philosophy:
 * ┌─────────────────────────────────────────────────────────────┐
 * │                    Mixed KV Cache                           │
 * │                                                             │
 * │  Hot Data (Recent)     Cold Data (Old)                     │
 * │  ┌─────────────────┐   ┌─────────────────┐                 │
 * │  │   FP16 Buffer   │   │  Quantized      │                 │
 * │  │   [newest N]    │   │  Buffer         │                 │
 * │  │   tokens        │   │  [older tokens] │                 │
 * │  └─────────────────┘   └─────────────────┘                 │
 * │           │                      │                         │
 * │           └──────┬───────────────┘                         │
 * │                  │                                         │
 * │                  ▼                                         │
 * │         ┌─────────────────┐                                │
 * │         │ Merged FP16 View│ ← Always returned to attention │
 * │         │ (dequantized)   │                                │
 * │         └─────────────────┘                                │
 * └─────────────────────────────────────────────────────────────┘
 * 
 * Key Features:
 * - Hot data (recent tokens): stored in FP16 for high precision and fast access
 * - Cold data (old tokens): stored in quantized format (e.g., Q4_0) to save memory
 * - FIFO strategy: when FP16 buffer is full, oldest tokens are quantized
 * - Transparent access: always provides FP16 view externally
 * - Per-layer management: each transformer layer has independent buffers
 * - Configurable quantization: supports different quantization types and thresholds
 * - Performance monitoring: provides quantization statistics and memory usage
 * - Adaptive thresholds: can dynamically adjust based on memory pressure
 */

class llama_kv_cache_mixed : public llama_kv_cache {
public:
    static uint32_t get_padding(const llama_cparams & cparams);

    // this callback is used to filter out layers that should not be included in the cache
    using layer_filter_cb = std::function<bool(int32_t il)>;

    llama_kv_cache_mixed(
            const llama_model &  model,
              layer_filter_cb && filter,
                         bool    v_trans,
                         bool    offload,
                     uint32_t    kv_size,
                     uint32_t    n_seq_max,
                     uint32_t    n_pad,
        const llama_kv_cache_mixed_config & config = {});

    ~llama_kv_cache_mixed();

    //
    // llama_memory_i
    //

    void clear() override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos delta) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    //
    // llama_kv_cache
    //

    void restore() override;
    void commit()  override;

    bool update(llama_context & ctx) override;

    void defrag_sched(float thold) override;

    void set_full() override;

    llama_sbatch sbatch_init(const llama_batch & batch, bool logits_all) override;
    llama_ubatch ubatch_next(llama_sbatch & sbatch, uint32_t n_ubatch, bool embd_pooled) const override;

    // updates the cache head
    bool find_slot(const llama_ubatch & batch) override;

    bool get_can_shift() const override;

    // state write/load
    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1)       override;

    //
    // llama_kv_cache_mixed specific API
    //

    uint32_t get_n() const;
    uint32_t get_size() const;

    // NOTE: Do quantization judgement.
    bool do_quant(int32_t il) const;

    // get views of the current state of the cache (always returns FP16 view)
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_k_quant(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v_quant(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_k_quant_ref(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v_quant_ref(ggml_context * ctx, int32_t il) const;

    // store k_cur and v_cur in the cache based on the current head location
    ggml_tensor * k_quant(ggml_context * ctx, int32_t il) const;
    ggml_tensor * v_quant(ggml_context * ctx, int32_t il) const;
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, int32_t il) const;

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    //
    // Debug methods for testing
    //

    uint32_t get_head() const { return head; }
    uint32_t get_used() const { return used; }
    
    // Get cell information for debugging
    struct cell_info {
        llama_pos pos = -1;
        bool is_empty = true;
        bool valid = false;
    };
    
    cell_info get_cell_info(uint32_t cell_idx) const {
        if (cell_idx >= size) {
            return {-1, true, false};
        }
        const auto & cell = cells[cell_idx];
        return {cell.pos, cell.is_empty(), true};
    }

    // Quantization statistics and management
    struct quantization_stats {
        uint32_t total_tokens_processed = 0;
        uint32_t total_tokens_quantized = 0;
        uint32_t quantization_events = 0;
        float    compression_ratio = 0.0f;
        uint64_t memory_saved_bytes = 0;
        uint32_t current_fp16_tokens = 0;
        
        // Performance metrics
        double   last_quantization_time_ms = 0.0;
        double   total_quantization_time_ms = 0.0;
        double   avg_quantization_time_ms = 0.0;
        
        void reset() {
            total_tokens_processed = 0;
            total_tokens_quantized = 0;
            quantization_events = 0;
            compression_ratio = 0.0f;
            memory_saved_bytes = 0;
            current_fp16_tokens = 0;
            last_quantization_time_ms = 0.0;
            total_quantization_time_ms = 0.0;
            avg_quantization_time_ms = 0.0;
        }
    };
    
    // Get current memory usage and pressure
    struct memory_info {
        size_t total_memory_bytes = 0;
        size_t fp16_memory_bytes = 0;
        size_t quant_memory_bytes = 0;
        float  memory_pressure = 0.0f;  // 0.0 to 1.0
        bool   should_quantize = false;
    };
    
    memory_info get_memory_info() const;
    
    // Get token distribution information for a specific layer
    struct layer_token_info {
        uint32_t n_fp16_tokens = 0;
        uint32_t n_quant_tokens = 0;
        bool     valid = false;
    };
    
    layer_token_info get_layer_token_info(int32_t il) const;

private:
    const llama_model & model;
    const llama_hparams & hparams;
    const llama_kv_cache_mixed_config config;

    // Extended kv_layer structure with both FP16 and quantized tensors
    struct kv_layer_mixed {
        uint32_t il;

        ggml_tensor * k_fp16;
        ggml_tensor * v_fp16;

        ggml_tensor * k_quant;
        ggml_tensor * v_quant;

        // FIFO Quantization state - separate counters for K and V
        mutable int64_t total_tokens = 0;          // total tokens in this layer
        mutable int64_t quant_k_tokens = 0;        // number of quantized K tokens
        mutable int64_t quant_v_tokens = 0;        // number of quantized V tokens
        mutable int64_t fp16_k_tokens = 0;         // number of fp16 K tokens
        mutable int64_t fp16_v_tokens = 0;         // number of fp16 V tokens
        mutable int64_t fp16_start_pos = 0;        // start position of fp16 tokens

        mutable int64_t mixed_k_head = 0;            //> mixed_head is the END of fp16 and START of quant.
        mutable int64_t mixed_v_head = 0;          //> mixed_v_head is the END of fp16 and START of quant.

        uint32_t get_total_cached_tokens() const {
            return total_tokens;
        }
        
        // Helper methods for combined counts
        uint32_t get_total_fp16_tokens() const {
            return fp16_k_tokens; // K and V should be the same, return K count
        }
        
        uint32_t get_total_quant_tokens() const {
            return quant_k_tokens; // K and V should be the same, return K count
        }
    };

    struct kv_cell {
        llama_pos pos   = -1;
        llama_pos delta =  0;

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const kv_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    bool has_shift = false;
    bool do_defrag = false;
    bool v_trans   = true;  // the value tensor is transposed

    uint32_t head = 0; // the location where the batch will be placed in the cache
    uint32_t size = 0; // total number of cells
    uint32_t used = 0; // used cells

    // computed before each graph build
    uint32_t n = 0;

    const uint32_t n_seq_max = 1;

    // required padding
    const uint32_t n_pad = 1;

    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    std::vector<kv_cell>       cells;
    std::vector<kv_layer_mixed> layers;

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    // recovery information
    struct {
        void clear() {
            try {
                // Use swap and clear pattern for safer destruction
                std::unordered_map<uint32_t, kv_cell> empty_map;
                cells.swap(empty_map);
                // empty_map destructor will handle cleanup safely
            } catch (...) {
                // Force clear if swap fails
                cells.clear();
            }
        }

        std::unordered_map<uint32_t, kv_cell> cells;
    } recovery;

    // defrag
    struct {
        std::vector<uint32_t> ids;
    } defrag_info;

    // Helper functions from unified cache
    bool defrag_prepare(int32_t n_max_nodes);
    uint32_t cell_max() const;
    size_t total_size() const;
    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    // Build graph functions
    llm_graph_result_ptr build_graph_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_cgraph * gf) const;

    llm_graph_result_ptr build_graph_defrag(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_cgraph * gf) const;

    llm_graph_result_ptr build_graph_quantize(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_cgraph * gf,
                         int32_t il) const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);
}; 