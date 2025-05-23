#pragma once

#include "llama-kv-cache.h"
#include "ggml.h"

#include <memory>
#include <vector>

// Per-channel quantization type for KV cache
// This quantizes along the token dimension with per-channel scaling factors
#define GGML_TYPE_Q4_0_PC ((ggml_type)100)  // Q4_0 with per-channel quantization
#define QK4_0_PC 256  // Block size for per-channel quantization (256 tokens)

// Per-channel quantization block structure
// Stores quantized data for 256 tokens with per-hidden-dim scaling factors
struct block_q4_0_pc {
    ggml_fp16_t scale;     // per-channel scale factor
    ggml_fp16_t zero;      // per-channel zero point
    uint8_t qs[QK4_0_PC / 2]; // quantized 4-bit values (2 per byte)
};

// Mixed precision KV cache configuration
struct llama_kv_cache_mixed_config {
    uint32_t hot_size = 1024;           // Size of hot (FP16) cache
    uint32_t cold_size = 4096;          // Size of cold (quantized) cache  
    uint32_t group_size = 256;          // Quantization group size (tokens to accumulate before quantizing)
    ggml_type hot_type_k = GGML_TYPE_F16;   // Type for hot cache K
    ggml_type hot_type_v = GGML_TYPE_F16;   // Type for hot cache V
    ggml_type cold_type_k = GGML_TYPE_Q4_0; // Type for cold cache K (quantized)
    ggml_type cold_type_v = GGML_TYPE_Q4_0; // Type for cold cache V (quantized)
    bool enable_quantization = true;    // Enable quantization to cold cache
};

// Per-channel quantization functions
void quantize_row_q4_0_pc(const float * x, block_q4_0_pc * y, int64_t k, int64_t n_channels);
void dequantize_row_q4_0_pc(const block_q4_0_pc * x, float * y, int64_t k, int64_t n_channels);

//
// llama_kv_cache_mixed
//
// Mixed precision KV cache using two unified caches:
// - Hot cache: FP16 storage for recent tokens
// - Cold cache: Quantized storage for older tokens
// Similar to SWA implementation but for mixed precision
//

class llama_kv_cache_mixed : public llama_kv_cache {
public:
    llama_kv_cache_mixed(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
        const llama_kv_cache_mixed_config & config);

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

    bool find_slot(const llama_ubatch & batch) override;

    bool get_can_shift() const override;

    // state write/load
    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1)       override;

    //
    // llama_kv_cache_mixed specific API
    //

    // Get access to individual caches for graph building
    llama_kv_cache_unified * get_kv_hot()  const;
    llama_kv_cache_unified * get_kv_cold() const;

private:
    const llama_kv_cache_mixed_config config;

    // Quantization tracking
    struct quantization_pending {
        void clear() {
            tokens.clear();
        }

        // Track tokens that need to be quantized and moved to cold cache
        std::vector<uint32_t> tokens;  // Token indices that should be moved to cold cache
    };

    bool do_quantize = true;  // Whether to perform quantization and cold storage

    quantization_pending pending;

    // Two unified caches - similar to SWA design
    std::unique_ptr<llama_kv_cache_unified> kv_hot;   // FP16 cache for recent tokens
    std::unique_ptr<llama_kv_cache_unified> kv_cold;  // Quantized cache for older tokens

    // Internal helper functions
    void trigger_quantization();
    bool should_quantize() const;
    void move_tokens_to_cold_cache(const std::vector<uint32_t> & token_indices);
    
    // For debugging - add print statements
    void debug_print_quantization(const char * event) const;
}; 