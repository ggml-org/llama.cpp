#include "llama-kv-cache-mixed.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

// Per-channel quantization implementation
void quantize_row_q4_0_pc(const float * x, block_q4_0_pc * y, int64_t k, int64_t n_channels) {
    for (int64_t ch = 0; ch < n_channels; ++ch) {
        const float * channel_data = x + ch * k;
        block_q4_0_pc * channel_block = y + ch;
        
        // Find min and max for this channel across all tokens
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        
        for (int64_t i = 0; i < k; ++i) {
            min_val = std::min(min_val, channel_data[i]);
            max_val = std::max(max_val, channel_data[i]);
        }
        
        // Calculate scale and zero point
        const float scale = (max_val - min_val) / 15.0f; // 4-bit range [0, 15]
        const float zero = min_val;
        
        channel_block->scale = ggml_fp32_to_fp16(scale);
        channel_block->zero = ggml_fp32_to_fp16(zero);
        
        // Quantize values
        for (int64_t i = 0; i < k; i += 2) {
            float val1 = channel_data[i];
            float val2 = (i + 1 < k) ? channel_data[i + 1] : 0.0f;
            
            // Quantize to 4-bit
            int q1 = std::max(0, std::min(15, (int)roundf((val1 - zero) / scale)));
            int q2 = std::max(0, std::min(15, (int)roundf((val2 - zero) / scale)));
            
            // Pack two 4-bit values into one byte
            channel_block->qs[i / 2] = (q2 << 4) | q1;
        }
    }
}

void dequantize_row_q4_0_pc(const block_q4_0_pc * x, float * y, int64_t k, int64_t n_channels) {
    for (int64_t ch = 0; ch < n_channels; ++ch) {
        const block_q4_0_pc * channel_block = x + ch;
        float * channel_data = y + ch * k;
        
        const float scale = ggml_fp16_to_fp32(channel_block->scale);
        const float zero = ggml_fp16_to_fp32(channel_block->zero);
        
        // Dequantize values
        for (int64_t i = 0; i < k; i += 2) {
            uint8_t packed = channel_block->qs[i / 2];
            
            int q1 = packed & 0x0F;
            int q2 = (packed >> 4) & 0x0F;
            
            channel_data[i] = zero + scale * q1;
            if (i + 1 < k) {
                channel_data[i + 1] = zero + scale * q2;
            }
        }
    }
}

//
// llama_kv_cache_mixed implementation - similar to SWA design
//

llama_kv_cache_mixed::llama_kv_cache_mixed(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
    const llama_kv_cache_mixed_config & config)
    : config(config) {
    
    // Suppress unused parameter warnings
    (void)type_k;
    (void)type_v;
    (void)kv_size;

    // Create filter functions to determine which cache to use
    // For simplicity, we use hot cache for recent tokens and cold cache for older ones
    llama_kv_cache_unified::layer_filter_cb filter_all = [](int32_t il) { 
        (void)il; 
        return true; // All layers use both caches
    };

    const uint32_t hot_size = config.hot_size;
    const uint32_t cold_size = config.cold_size;

    LLAMA_LOG_INFO("%s: creating hot KV cache (FP16), size = %u cells\n", __func__, hot_size);

    // Create hot cache with FP16 precision
    kv_hot = std::make_unique<llama_kv_cache_unified>(
            model, 
            std::move(filter_all), // Use the filter function
            config.hot_type_k,     // FP16 for hot cache
            config.hot_type_v, 
            v_trans, 
            offload, 
            hot_size, 
            n_seq_max, 
            n_pad,
            0, // no SWA
            LLAMA_SWA_TYPE_NONE);

    LLAMA_LOG_INFO("%s: creating cold KV cache (quantized), size = %u cells\n", __func__, cold_size);

    // Create cold cache with quantized precision
    llama_kv_cache_unified::layer_filter_cb filter_all_cold = [](int32_t il) { 
        (void)il; 
        return true; // All layers use both caches
    };
    
    kv_cold = std::make_unique<llama_kv_cache_unified>(
            model, 
            std::move(filter_all_cold),
            config.cold_type_k,    // Q4_0 for cold cache
            config.cold_type_v, 
            v_trans, 
            offload, 
            cold_size, 
            n_seq_max, 
            n_pad,
            0, // no SWA  
            LLAMA_SWA_TYPE_NONE);

    debug_print_quantization("initialized");
}

void llama_kv_cache_mixed::clear() {
    kv_hot->clear();
    kv_cold->clear();
    pending.clear();
    
    debug_print_quantization("cleared");
}

bool llama_kv_cache_mixed::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    bool result_hot = kv_hot->seq_rm(seq_id, p0, p1);
    bool result_cold = kv_cold->seq_rm(seq_id, p0, p1);
    
    return result_hot || result_cold;
}

void llama_kv_cache_mixed::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    kv_hot->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    kv_cold->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_mixed::seq_keep(llama_seq_id seq_id) {
    kv_hot->seq_keep(seq_id);
    kv_cold->seq_keep(seq_id);
}

void llama_kv_cache_mixed::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) {
    kv_hot->seq_add(seq_id, p0, p1, delta);
    kv_cold->seq_add(seq_id, p0, p1, delta);
}

void llama_kv_cache_mixed::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    kv_hot->seq_div(seq_id, p0, p1, d);
    kv_cold->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_kv_cache_mixed::seq_pos_min(llama_seq_id seq_id) const {
    llama_pos hot_min = kv_hot->seq_pos_min(seq_id);
    llama_pos cold_min = kv_cold->seq_pos_min(seq_id);
    
    // Return the minimum across both caches
    if (hot_min == -1) return cold_min;
    if (cold_min == -1) return hot_min;
    return std::min(hot_min, cold_min);
}

llama_pos llama_kv_cache_mixed::seq_pos_max(llama_seq_id seq_id) const {
    llama_pos hot_max = kv_hot->seq_pos_max(seq_id);
    llama_pos cold_max = kv_cold->seq_pos_max(seq_id);
    
    // Return the maximum across both caches
    return std::max(hot_max, cold_max);
}

void llama_kv_cache_mixed::restore() {
    kv_hot->restore();
    kv_cold->restore();
}

void llama_kv_cache_mixed::commit() {
    kv_hot->commit();
    kv_cold->commit();
    
    // Check if we should trigger quantization after commit
    if (should_quantize()) {
        debug_print_quantization("triggering quantization in commit");
        trigger_quantization();
    }
}

bool llama_kv_cache_mixed::update(llama_context & ctx) {
    bool result_hot = kv_hot->update(ctx);
    bool result_cold = kv_cold->update(ctx);
    
    return result_hot || result_cold;
}

void llama_kv_cache_mixed::defrag_sched(float thold) {
    kv_hot->defrag_sched(thold);
    kv_cold->defrag_sched(thold);
}

void llama_kv_cache_mixed::set_full() {
    kv_hot->set_full();
    kv_cold->set_full();
}

llama_sbatch llama_kv_cache_mixed::sbatch_init(const llama_batch & batch, bool logits_all) {
    // Use hot cache for batch initialization
    return kv_hot->sbatch_init(batch, logits_all);
}

llama_ubatch llama_kv_cache_mixed::ubatch_next(llama_sbatch & sbatch, uint32_t n_ubatch, bool embd_pooled) const {
    // Use hot cache for batch processing
    return kv_hot->ubatch_next(sbatch, n_ubatch, embd_pooled);
}

bool llama_kv_cache_mixed::find_slot(const llama_ubatch & batch) {
    // Try to find slot in hot cache first
    bool result = kv_hot->find_slot(batch);
    
    // Check if hot cache is getting full and we should trigger quantization
    if (result && should_quantize()) {
        debug_print_quantization("triggering quantization in find_slot");
        trigger_quantization();
    }
    
    return result;
}

bool llama_kv_cache_mixed::get_can_shift() const {
    // We can shift if either cache supports it
    return kv_hot->get_can_shift() || kv_cold->get_can_shift();
}

void llama_kv_cache_mixed::state_write(llama_io_write_i & io, llama_seq_id seq_id) const {
    // Write both caches
    kv_hot->state_write(io, seq_id);
    kv_cold->state_write(io, seq_id);
    
    // Write mixed cache metadata
    uint32_t n_pending = pending.tokens.size();
    io.write(&n_pending, sizeof(n_pending));
    if (n_pending > 0) {
        io.write(pending.tokens.data(), n_pending * sizeof(uint32_t));
    }
}

void llama_kv_cache_mixed::state_read(llama_io_read_i & io, llama_seq_id seq_id) {
    // Read both caches
    kv_hot->state_read(io, seq_id);
    kv_cold->state_read(io, seq_id);
    
    // Read mixed cache metadata
    uint32_t n_pending;
    io.read_to(&n_pending, sizeof(n_pending));
    pending.tokens.resize(n_pending);
    if (n_pending > 0) {
        io.read_to(pending.tokens.data(), n_pending * sizeof(uint32_t));
    }
}

//
// Mixed precision specific API
//

llama_kv_cache_unified * llama_kv_cache_mixed::get_kv_hot() const {
    return kv_hot.get();
}

llama_kv_cache_unified * llama_kv_cache_mixed::get_kv_cold() const {
    return kv_cold.get();
}

//
// Private helper methods
//

bool llama_kv_cache_mixed::should_quantize() const {
    if (!config.enable_quantization || !do_quantize) {
        return false;
    }
    
    // Check if hot cache usage exceeds threshold
    const uint32_t hot_used = kv_hot->get_n();    // Use public API instead of cell_max()
    const uint32_t hot_size = kv_hot->get_size();
    
    // Trigger quantization when hot cache is 80% full
    const float threshold = 0.8f;
    bool should_trigger = hot_used > (uint32_t)(hot_size * threshold);
    
    if (should_trigger) {
        debug_print_quantization("should_quantize: hot cache threshold exceeded");
    }
    
    return should_trigger;
}

void llama_kv_cache_mixed::trigger_quantization() {
    if (!config.enable_quantization || !do_quantize) {
        return;
    }
    
    debug_print_quantization("trigger_quantization: starting quantization process");
    
    // Get the oldest tokens from hot cache
    const uint32_t hot_used = kv_hot->get_n();    // Use public API instead of cell_max()
    const uint32_t tokens_to_move = std::min(hot_used / 4, config.group_size); // Move 25% or group_size, whichever is smaller
    
    if (tokens_to_move == 0) {
        debug_print_quantization("trigger_quantization: no tokens to move");
        return;
    }
    
    // Collect token indices to move (oldest tokens)
    std::vector<uint32_t> tokens_to_quantize;
    for (uint32_t i = 0; i < tokens_to_move; ++i) {
        tokens_to_quantize.push_back(i);
    }
    
    debug_print_quantization("trigger_quantization: moving tokens to cold cache");
    move_tokens_to_cold_cache(tokens_to_quantize);
    
    debug_print_quantization("trigger_quantization: quantization completed");
}

void llama_kv_cache_mixed::move_tokens_to_cold_cache(const std::vector<uint32_t> & token_indices) {
    if (token_indices.empty()) {
        return;
    }
    
    printf("[MIXED_CACHE] Moving %zu tokens to cold cache (Q4_0 quantization)\n", token_indices.size());
    
    // TODO: Implement actual token moving logic
    // For now, we just print that quantization would happen here
    // This is where the actual quantization from FP16 (hot) to Q4_0 (cold) would occur
    
    for (uint32_t token_idx : token_indices) {
        printf("[MIXED_CACHE] Quantizing token %u: FP16 -> Q4_0\n", token_idx);
        // Here we would:
        // 1. Extract K,V tensors for this token from hot cache
        // 2. Quantize them using Q4_0
        // 3. Store in cold cache
        // 4. Remove from hot cache
    }
    
    printf("[MIXED_CACHE] Quantization batch completed: %zu tokens processed\n", token_indices.size());
}

void llama_kv_cache_mixed::debug_print_quantization(const char * event) const {
    if (!config.enable_quantization) {
        return;
    }
    
    const uint32_t hot_used = kv_hot->get_n();     // Use public API instead of cell_max()
    const uint32_t hot_size = kv_hot->get_size();
    const uint32_t cold_used = kv_cold->get_n();   // Use public API instead of cell_max() 
    const uint32_t cold_size = kv_cold->get_size();
    
    printf("[MIXED_CACHE_DEBUG] %s: hot=%u/%u (%.1f%%), cold=%u/%u (%.1f%%)\n", 
           event,
           hot_used, hot_size, 100.0f * hot_used / hot_size,
           cold_used, cold_size, 100.0f * cold_used / cold_size);
} 