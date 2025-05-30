#pragma once

#include "llama-kv-cache.h"
#include "ggml.h"

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
    uint32_t quantization_threshold = 32;   // Number of tokens before quantization
    uint32_t group_size = 16;               // Number of tokens to quantize at once
    
    // Advanced quantization settings
    bool     adaptive_threshold = false;        // Dynamically adjust threshold based on memory pressure
    float    memory_pressure_threshold = 0.8f;  // Trigger quantization when memory usage > 80%
    uint32_t min_quantization_threshold = 16;   // Minimum threshold for adaptive mode
    uint32_t max_quantization_threshold = 128;  // Maximum threshold for adaptive mode
    
    // Cache types
    ggml_type hot_type_k  = GGML_TYPE_F16;  // Recent tokens (FP16)
    ggml_type hot_type_v  = GGML_TYPE_F16;
    ggml_type cold_type_k = GGML_TYPE_Q4_0; // Old tokens (quantized)
    ggml_type cold_type_v = GGML_TYPE_Q4_0;
    
    // Performance monitoring
    bool     enable_stats = true;           // Enable quantization statistics
    uint32_t stats_report_interval = 1000; // Report stats every N tokens
};

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

    ~llama_kv_cache_mixed() = default;

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
    
    quantization_stats get_quantization_stats() const { return quant_stats; }
    void reset_quantization_stats() { quant_stats.reset(); }
    
    // Get current memory usage and pressure
    // struct memory_info {
    //     size_t total_memory_bytes = 0;
    //     size_t fp16_memory_bytes = 0;
    //     size_t quant_memory_bytes = 0;
    //     float  memory_pressure = 0.0f;  // 0.0 to 1.0
    //     bool   should_quantize = false;
    // };
    
    // memory_info get_memory_info() const;

private:
    const llama_model & model;
    const llama_hparams & hparams;
    const llama_kv_cache_mixed_config config;

    // Extended kv_layer structure with both FP16 and quantized tensors
    struct kv_layer_mixed {
        // layer index in the model
        uint32_t il;

        // FP16 tensors for recent tokens
        ggml_tensor * k_fp16;
        ggml_tensor * v_fp16;
        
        // Quantized tensors for old tokens
        ggml_tensor * k_quant;
        ggml_tensor * v_quant;
        
        // Dequantized views (for returning FP16 to attention)
        ggml_tensor * k_dequant;  // Temporary tensor for dequantization
        ggml_tensor * v_dequant;  // Temporary tensor for dequantization
        
        // Number of tokens in FP16 buffer
        mutable uint32_t n_fp16_tokens = 0;
        
        // Number of tokens in quantized buffer
        mutable uint32_t n_k_quant_tokens = 0;
        mutable uint32_t n_v_quant_tokens = 0;
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
            cells.clear();
        }

        std::unordered_map<uint32_t, kv_cell> cells;
    } recovery;

    // defrag
    struct {
        std::vector<uint32_t> ids;
    } defrag_info;

    // Quantization management
    struct quantization_manager {
        uint32_t accumulated_tokens = 0;        // Tokens accumulated since last quantization
        uint32_t current_threshold;             // Current dynamic threshold
        bool     quantization_in_progress = false;
        
        // Statistics
        quantization_stats stats;
        
        // Timing
        std::chrono::high_resolution_clock::time_point last_quantization_start;
        
        quantization_manager(uint32_t initial_threshold) : current_threshold(initial_threshold) {}
        
        void reset_accumulation() {
            accumulated_tokens = 0;
        }
        
        bool should_quantize(const llama_kv_cache_mixed_config & config, float memory_pressure) const {
            if (!config.enable_quantization || quantization_in_progress) {
                return false;
            }
            
            // Check basic threshold
            if (accumulated_tokens >= current_threshold) {
                return true;
            }
            
            // Check memory pressure if adaptive mode is enabled
            if (config.adaptive_threshold && memory_pressure > config.memory_pressure_threshold) {
                return accumulated_tokens >= config.min_quantization_threshold;
            }
            
            return false;
        }
        
        void update_threshold(const llama_kv_cache_mixed_config & config, float memory_pressure) {
            if (!config.adaptive_threshold) {
                current_threshold = config.quantization_threshold;
                return;
            }
            
            // Adaptive threshold based on memory pressure
            if (memory_pressure > config.memory_pressure_threshold) {
                // High memory pressure: reduce threshold
                current_threshold = std::max(config.min_quantization_threshold,
                                           current_threshold - config.group_size);
            } else if (memory_pressure < config.memory_pressure_threshold * 0.5f) {
                // Low memory pressure: increase threshold
                current_threshold = std::min(config.max_quantization_threshold,
                                           current_threshold + config.group_size);
            }
        }
    };
    
    mutable quantization_manager quant_mgr;
    mutable quantization_stats quant_stats;

    //
    // Private helper methods
    //

    // Quantize FP16 tokens to quantized format
    void quantize_tokens(int32_t il);
    
    // Quantize oldest tokens using FIFO strategy
    void quantize_oldest_tokens(int32_t il, uint32_t tokens_to_quantize);

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