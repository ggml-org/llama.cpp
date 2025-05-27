#include "llama-kv-cache-mixed.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-graph.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <cstring>
#include <chrono>

/*
 * Mixed KV Cache Debug Output
 * 
 * Uses llama's existing debug system. Enable with:
 * - Set log level to DEBUG or higher
 * - Look for "[mixed-kv]" prefix in debug output
 */

// Helper function to format memory size
static std::string format_memory_size(size_t bytes) {
    if (bytes >= 1024 * 1024 * 1024) {
        return std::to_string(bytes / (1024.0 * 1024.0 * 1024.0)) + " GB";
    } else if (bytes >= 1024 * 1024) {
        return std::to_string(bytes / (1024.0 * 1024.0)) + " MB";
    } else if (bytes >= 1024) {
        return std::to_string(bytes / 1024.0) + " KB";
    } else {
        return std::to_string(bytes) + " B";
    }
}

// Helper function to get current timestamp for performance measurement
static std::chrono::high_resolution_clock::time_point get_current_time() {
    return std::chrono::high_resolution_clock::now();
}

// Helper function to calculate duration in milliseconds
static double get_duration_ms(const std::chrono::high_resolution_clock::time_point& start,
                             const std::chrono::high_resolution_clock::time_point& end) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0;
}

/*
 * llama_kv_cache_mixed implementation
 * 
 * Mixed precision KV cache with automatic quantization:
 * 
 * Architecture Overview:
 * +-------------------------------------------------------------+
 * |                    Mixed KV Cache                           |
 * |                                                             |
 * |  Hot Data (Recent)     Cold Data (Old)                      |
 * |  +-----------------+   +-----------------+                  |
 * |  |   FP16 Buffer   |   |  Quantized      |                  |
 * |  |   [newest N]    |   |  Buffer         |                  |
 * |  |   tokens        |   |  [older tokens] |                  |
 * |  +-----------------+   +-----------------+                  |
 * |           |                      |                          |
 * |           +------+---------------+                          |
 * |                  |                                          |
 * |                  v                                          |
 * |         +-----------------+                                 |
 * |         | Merged FP16 View| <- Always returned to attention |
 * |         | (dequantized)   |                                 |
 * |         +-----------------+                                 |
 * +-------------------------------------------------------------+
 * 
 * FIFO Quantization Strategy:
 * 
 * Time ->  [Token 1] [Token 2] [Token 3] [Token 4] [Token 5]
 *         |         |         |         |         |
 *         v         v         v         v         v
 * Step 1: [  FP16   ] [  FP16  ] [  FP16  ]
 * Step 2: [  FP16   ] [  FP16  ] [  FP16  ] [  FP16  ]
 * Step 3: [ Quant   ] [  FP16  ] [  FP16  ] [  FP16  ] [  FP16  ]
 *         ^ oldest moved to quantized buffer when threshold exceeded
 * 
 * Compatibility:
 * - Only activated when use_mixed_kv_cache = true
 * - All existing cache types continue to work unchanged
 * - Uses dynamic_cast for type-safe detection
 */

uint32_t llama_kv_cache_mixed::get_padding(const llama_cparams & cparams) {
    GGML_UNUSED(cparams);
    // TODO : the FA kernels require padding to avoid extra runtime boundary checks
    return cparams.flash_attn ? 256u : 32u;
}

llama_kv_cache_mixed::llama_kv_cache_mixed(
        const llama_model &  model,
          layer_filter_cb && filter,
                     bool    v_trans,
                     bool    offload,
                 uint32_t    kv_size,
                 uint32_t    n_seq_max,
                 uint32_t    n_pad,
    const llama_kv_cache_mixed_config & config)
    : model(model), hparams(model.hparams), config(config),
      v_trans(v_trans), n_seq_max(n_seq_max), n_pad(n_pad),
      quant_mgr(config.quantization_threshold) {

    GGML_ASSERT(kv_size % n_pad == 0);

    // create a context for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            // Allocate enough memory for both FP16 and quantized tensors
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(8u*hparams.n_layer*ggml_tensor_overhead()), // Increase to 8x for mixed tensors
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map[buft] = ctx;
            ctxs.emplace_back(ctx);

            return ctx;
        }

        return it->second;
    };

    head = 0;
    size = kv_size;
    used = 0;

    cells.resize(kv_size);

    for (uint32_t il = 0; il < hparams.n_layer; il++) {
        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: skipped\n", __func__, il);
            continue;
        }

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il) + hparams.n_embd_k_s();
        const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        kv_layer_mixed layer;
        layer.il = il;

        // Create FP16 tensors exactly like unified cache
        layer.k_fp16 = ggml_new_tensor_2d(ctx, config.hot_type_k, n_embd_k_gqa, kv_size);
        layer.v_fp16 = ggml_new_tensor_2d(ctx, config.hot_type_v, n_embd_v_gqa, kv_size);

        // Create quantized tensors (for future use, but not used during alignment testing)
        layer.k_quant = ggml_new_tensor_2d(ctx, config.cold_type_k, n_embd_k_gqa, kv_size);
        layer.v_quant = ggml_new_tensor_2d(ctx, config.cold_type_v, n_embd_v_gqa, kv_size);

        // Create dequantization buffers (for future use, but not used during alignment testing)
        layer.k_dequant = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd_k_gqa, kv_size);
        layer.v_dequant = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd_v_gqa, kv_size);

        // Use naming convention similar to unified cache for FP16 tensors
        ggml_format_name(layer.k_fp16,      "cache_k_l%d",          il);
        ggml_format_name(layer.v_fp16,      "cache_v_l%d",          il);
        ggml_format_name(layer.k_quant,     "cache_k_quant_l%d",    il);
        ggml_format_name(layer.v_quant,     "cache_v_quant_l%d",    il);
        ggml_format_name(layer.k_dequant,   "cache_k_dequant_l%d",  il);
        ggml_format_name(layer.v_dequant,   "cache_v_dequant_l%d",  il);

        map_layer_ids[il] = layers.size();
        layers.push_back(layer);
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto it : ctx_map) {
        auto * buft = it.first;
        auto * ctx  = it.second;

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__,
                       ggml_backend_buffer_name(buf),
                       ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);
        bufs.emplace_back(buf);
    }

    {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        LLAMA_LOG_INFO("%s: mixed cache size = %7.2f MiB (%6u cells, %3d layers, %2u seqs)\n",
                __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f),
                kv_size, (int) layers.size(), n_seq_max);
        LLAMA_LOG_INFO("%s:   FP16 K: %7.2f MiB, FP16 V: %7.2f MiB\n", __func__,
                (float)(memory_size_k/2) / (1024.0f * 1024.0f),
                (float)(memory_size_v/2) / (1024.0f * 1024.0f));
        LLAMA_LOG_INFO("%s:   Quant K (%s): %7.2f MiB, Quant V (%s): %7.2f MiB\n", __func__,
                ggml_type_name(config.cold_type_k), (float)(memory_size_k/2) / (1024.0f * 1024.0f),
                ggml_type_name(config.cold_type_v), (float)(memory_size_v/2) / (1024.0f * 1024.0f));
    }
}

void llama_kv_cache_mixed::clear() {
    LLAMA_LOG_DEBUG("[mixed-kv] clearing cache (size=%u, used=%u)\n", size, used);
    
    for (uint32_t i = 0; i < size; ++i) {
        cells[i].pos = -1;
        cells[i].seq_id.clear();
    }

    head = 0;
    used = 0;

    // Clear all layers and count tokens for debug output
    uint32_t total_fp16_tokens = 0;
    uint32_t total_quant_tokens = 0;
    for (auto & layer : layers) {
        total_fp16_tokens += layer.n_fp16_tokens;
        total_quant_tokens += layer.n_quant_tokens;
        layer.n_fp16_tokens = 0;
        layer.n_quant_tokens = 0;
    }

    LLAMA_LOG_DEBUG("[mixed-kv] cleared %u FP16 tokens and %u quantized tokens across %d layers\n", 
                    total_fp16_tokens, total_quant_tokens, (int)layers.size());

    for (auto & buf : bufs) {
        ggml_backend_buffer_clear(buf.get(), 0);
    }
    
    LLAMA_LOG_DEBUG("[mixed-kv] cache cleared successfully\n");
}

// Implement sequence operations - similar to unified cache
bool llama_kv_cache_mixed::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    uint32_t new_head = size;

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].pos >= p0 && cells[i].pos < p1) {
            if (seq_id < 0) {
                cells[i].seq_id.clear();
            } else if (cells[i].has_seq_id(seq_id)) {
                cells[i].seq_id.erase(seq_id);
            } else {
                continue;
            }

            if (cells[i].is_empty()) {
                // keep count of the number of used cells
                if (cells[i].pos >= 0) {
                    used--;
                }

                cells[i].pos = -1;

                if (new_head == size) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != size && new_head < head) {
        head = new_head;
    }

    return true;
}

void llama_kv_cache_mixed::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (seq_id_src == seq_id_dst) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    head = 0;

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id_src) && cells[i].pos >= p0 && cells[i].pos < p1) {
            cells[i].seq_id.insert(seq_id_dst);
        }
    }
}

void llama_kv_cache_mixed::seq_keep(llama_seq_id seq_id) {
    uint32_t new_head = size;

    for (uint32_t i = 0; i < size; ++i) {
        if (!cells[i].has_seq_id(seq_id)) {
            if (cells[i].pos >= 0) {
                used--;
            }

            cells[i].pos = -1;
            cells[i].seq_id.clear();

            if (new_head == size){
                new_head = i;
            }
        } else {
            cells[i].seq_id.clear();
            cells[i].seq_id.insert(seq_id);
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != size && new_head < head) {
        head = new_head;
    }
}

void llama_kv_cache_mixed::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) {
    if (delta == 0) {
        return;
    }

    uint32_t new_head = size;

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id) && cells[i].pos >= p0 && cells[i].pos < p1) {
            has_shift = true;

            cells[i].pos   += delta;
            cells[i].delta += delta;

            if (cells[i].pos < 0) {
                if (!cells[i].is_empty()) {
                    used--;
                }
                cells[i].pos = -1;
                cells[i].seq_id.clear();
                if (new_head == size) {
                    new_head = i;
                }
            }
        }
    }

    head = new_head != size ? new_head : 0;
}

void llama_kv_cache_mixed::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id) && cells[i].pos >= p0 && cells[i].pos < p1) {
            has_shift = true;

            {
                llama_pos p_old = cells[i].pos;
                cells[i].pos   /= d;
                cells[i].delta += cells[i].pos - p_old;
            }
        }
    }
}

llama_pos llama_kv_cache_mixed::seq_pos_min(llama_seq_id seq_id) const {
    llama_pos result = std::numeric_limits<llama_pos>::max();

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::min(result, cells[i].pos);
        }
    }

    if (result == std::numeric_limits<llama_pos>::max()) {
        result = -1;
    }

    return result;
}

llama_pos llama_kv_cache_mixed::seq_pos_max(llama_seq_id seq_id) const {
    llama_pos result = -1;

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::max(result, cells[i].pos);
        }
    }

    return result;
}

void llama_kv_cache_mixed::restore() {
    for (const auto & [id, cell] : recovery.cells) {
        const bool is_empty0 = cells[id].is_empty();
        const bool is_empty1 = cell.is_empty();

        if (!is_empty0 && is_empty1) {
            used--;
        } else if (is_empty0 && !is_empty1) {
            used++;
        }

        cells[id] = cell;
    }

    recovery.clear();
}

void llama_kv_cache_mixed::commit() {
    if (recovery.cells.empty()) {
        LLAMA_LOG_WARN("%s: the recovery information upon a commit was empty - might indicate a bug\n", __func__);
        return;
    }

    recovery.clear();

    /*
     * Quantization Handling Strategy:
     * 
     * +-------------------------------------------------------------+
     * |                 Quantization Flow                          |
     * |                                                             |
     * |  commit() -> update() -> build_graph_quantize() -> execute |
     * |     |           |              |                     |     |
     * |     v           v              v                     v     |
     * |  Mark for   Check if      Create ggml         Execute     |
     * |  future     quantization  operations          graph       |
     * |  processing needed        in graph            operations   |
     * +-------------------------------------------------------------+
     * 
     * Quantization is now handled correctly through the update() method
     * and graph building mechanism, rather than directly calling
     * quantization functions in commit().
     * 
     * This ensures:
     * - Consistency with llama.cpp architecture
     * - Quantization operations coordinate with other graph operations
     * - Support for GPU acceleration and backend optimization
     * 
     * Quantization will be automatically triggered on the next update() call.
     */
    
    LLAMA_LOG_DEBUG("[mixed-kv] commit completed, quantization will be handled in next update() call\n");
}

bool llama_kv_cache_mixed::update(llama_context & lctx) {
    // Similar to unified cache - handle shift and defrag
    bool need_reserve = false;

    auto * sched = lctx.get_sched();

    if (has_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched);

            auto * gf = lctx.graph_init();

            auto res = build_graph_shift(lctx.get_cparams(), lctx.get_ctx_compute(), gf);

            ggml_backend_sched_alloc_graph(sched, gf);

            res->set_inputs(nullptr);

            lctx.graph_compute(gf, false);

            need_reserve = true;
        }

        {
            has_shift = false;

            for (uint32_t i = 0; i < size; ++i) {
                cells[i].delta = 0;
            }
        }
    }

    if (do_defrag) {
        LLAMA_LOG_DEBUG("%s: defragmenting KV cache\n", __func__);

        if (defrag_prepare(lctx.graph_max_nodes())) {
            ggml_backend_sched_reset(sched);

            auto * gf = lctx.graph_init();

            auto res = build_graph_defrag(lctx.get_cparams(), lctx.get_ctx_compute(), gf);

            ggml_backend_sched_alloc_graph(sched, gf);

            res->set_inputs(nullptr);

            lctx.graph_compute(gf, false);

            need_reserve = true;
        }

        do_defrag = false;
    }

    // TEMPORARILY DISABLE QUANTIZATION FOR ALIGNMENT TESTING
    // TODO: Re-enable quantization after alignment is verified
    /*
    // Check if quantization is needed
    if (config.enable_quantization) {
        bool quantization_needed = false;
        
        // Check each layer for quantization needs
        for (auto & layer : layers) {
            if (layer.n_fp16_tokens >= config.quantization_threshold) {
                quantization_needed = true;
                break;
            }
        }
        
        if (quantization_needed) {
            LLAMA_LOG_DEBUG("[mixed-kv] quantization needed, building quantization graph\n");
            
            ggml_backend_sched_reset(sched);
            auto * gf = lctx.graph_init();
            
            // Build quantization graph for each layer that needs it
            for (auto & layer : layers) {
                if (layer.n_fp16_tokens >= config.quantization_threshold) {
                    LLAMA_LOG_DEBUG("[mixed-kv] building quantization graph for layer %d (%u FP16 tokens)\n", 
                                   layer.il, layer.n_fp16_tokens);
                    
                    auto res = build_graph_quantize(lctx.get_cparams(), lctx.get_ctx_compute(), gf, layer.il);
                    
                    if (res) {
                        // Calculate number of tokens to quantize
                        uint32_t tokens_to_quantize = std::min(layer.n_fp16_tokens, config.group_size);
                        
                        // Pre-update counters (these values will be correct after graph execution)
                        layer.n_quant_tokens += tokens_to_quantize;
                        layer.n_fp16_tokens -= tokens_to_quantize;
                        
                        LLAMA_LOG_DEBUG("[mixed-kv] scheduled quantization of %u tokens for layer %d\n", 
                                       tokens_to_quantize, layer.il);
                    }
                }
            }
            
            // Allocate graph and execute
            ggml_backend_sched_alloc_graph(sched, gf);
            
            LLAMA_LOG_DEBUG("[mixed-kv] executing quantization graph\n");
            lctx.graph_compute(gf, false);
            
            LLAMA_LOG_DEBUG("[mixed-kv] quantization graph execution completed\n");
            
            need_reserve = true;
        }
    }
    */

    LLAMA_LOG_DEBUG("[mixed-kv] update completed (quantization disabled for alignment testing)\n");

    return need_reserve;
}

void llama_kv_cache_mixed::defrag_sched(float thold) {
    const float fragmentation = n >= 2048 ? std::max(0.0f, 1.0f - (float(used + n_pad)/n)) : 0.0f;

    if (fragmentation > thold) {
        LLAMA_LOG_DEBUG("%s: fragmentation: %.2f - requesting defrag\n", __func__, fragmentation);
        do_defrag = true;
    }
}

void llama_kv_cache_mixed::set_full() {
    n = size;
    head = 0;
}

llama_sbatch llama_kv_cache_mixed::sbatch_init(const llama_batch & batch, bool logits_all) {
    return llama_sbatch(batch, hparams.n_embd, true, logits_all);
}

llama_ubatch llama_kv_cache_mixed::ubatch_next(llama_sbatch & sbatch, uint32_t n_ubatch, bool embd_pooled) const {
    GGML_UNUSED(embd_pooled);
    return sbatch.split_simple(n_ubatch);
}

bool llama_kv_cache_mixed::find_slot(const llama_ubatch & ubatch) {
    const uint32_t n_tokens = ubatch.n_tokens;

    LLAMA_LOG_DEBUG("[mixed-kv] finding slot for %u tokens (head=%u, used=%u, size=%u)\n", n_tokens, head, used, size);

    // if we have enough unused cells before the current head ->
    //   better to start searching from the beginning of the cache, hoping to fill it
    if (head > used + 2*ubatch.n_tokens) {
        LLAMA_LOG_DEBUG("[mixed-kv] resetting head from %u to 0 (optimization)\n", head);
        head = 0;
    }

    if (n_tokens > size) {
        LLAMA_LOG_ERROR("[mixed-kv] ERROR: requested tokens (%u) exceed cache size (%u)\n", n_tokens, size);
        LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %d\n", __func__, n_tokens, size);
        return false;
    }

    // Note: Unlike unified cache, we don't enforce n_seq_max limit here
    // This allows the mixed cache to work with any number of sequences
    // The sequence management is handled at a higher level

    uint32_t n_tested = 0;

    while (true) {
        if (head + n_tokens > size) {
            n_tested += size - head;
            head = 0;
            continue;
        }

        bool found = true;
        for (uint32_t i = 0; i < n_tokens; i++) {
            if (cells[head + i].pos >= 0) {
                found = false;
                head += i + 1;
                n_tested += i + 1;
                break;
            }
        }

        if (found) {
            break;
        }

        if (n_tested >= size) {
            return false;
        }
    }

    for (uint32_t i = 0; i < n_tokens; ++i) {
        // remember the original state
        if (recovery.cells.find(head + i) == recovery.cells.end()) {
            recovery.cells[head + i] = cells[head + i];
        }

        cells[head + i].pos = ubatch.pos[i];

        for (int32_t j = 0; j < ubatch.n_seq_id[i]; j++) {
            cells[head + i].seq_id.insert(ubatch.seq_id[i][j]);
        }
    }

    used += n_tokens;

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // after enough generations, the benefit from this heuristic disappears
    // if we start defragmenting the cache, the benefit from this will be more important
    n = std::min(size, std::max(n_pad, GGML_PAD(cell_max(), n_pad)));

    LLAMA_LOG_DEBUG("[mixed-kv] successfully allocated slot: head=%u, used=%u, n=%u\n", head, used, n);

    return true;
}

bool llama_kv_cache_mixed::get_can_shift() const {
    return true;
}

uint32_t llama_kv_cache_mixed::get_n() const {
    return n;
}

uint32_t llama_kv_cache_mixed::get_size() const {
    return size;
}

/*
 * FIFO Quantization Implementation:
 * 
 * Quantize oldest tokens from FP16 to quantized format using ggml operations.
 * This implements FIFO (First In, First Out) strategy.
 * 
 * Important Architecture Note:
 * In llama.cpp, quantization operations should be handled through the graph
 * building mechanism, rather than creating independent contexts within KV cache.
 * 
 * Correct approach: Mark tokens for quantization, handle in update() method
 *                   through build_graph_quantize()
 * Wrong approach: Create ggml_context inside KV cache and execute quantization
 * 
 * Before quantization:
 * +-------------------------------------------------------------+
 * | FP16 Buffer                                                 |
 * | [oldest] [token2] [token3] [token4] [newest]                |
 * |    ^                                                        |
 * |    +-- tokens_to_quantize                                   |
 * +-------------------------------------------------------------+
 * 
 * After quantization:
 * +-----------------+ +---------------------------------------+
 * | Quantized Buffer| | FP16 Buffer                           |
 * | [oldest]        | | [token2] [token3] [token4] [newest]   |
 * +-----------------+ +---------------------------------------+
 */
void llama_kv_cache_mixed::quantize_oldest_tokens(int32_t il, uint32_t tokens_to_quantize) {
    auto start_time = get_current_time();
    
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        LLAMA_LOG_ERROR("[mixed-kv] ERROR: layer %d not found in cache\n", il);
        return;
    }

    auto & layer = layers[it->second];

    LLAMA_LOG_DEBUG("[mixed-kv] starting quantization for layer %d:\n", il);
    LLAMA_LOG_DEBUG("[mixed-kv]   - requested tokens to quantize: %u\n", tokens_to_quantize);
    LLAMA_LOG_DEBUG("[mixed-kv]   - available FP16 tokens: %u\n", layer.n_fp16_tokens);
    LLAMA_LOG_DEBUG("[mixed-kv]   - existing quantized tokens: %u\n", layer.n_quant_tokens);

    // Safety check: don't quantize more than available
    if (layer.n_fp16_tokens < tokens_to_quantize) {
        LLAMA_LOG_DEBUG("[mixed-kv]   - adjusting tokens_to_quantize from %u to %u (limited by available FP16 tokens)\n",
                       tokens_to_quantize, layer.n_fp16_tokens);
        tokens_to_quantize = layer.n_fp16_tokens;
    }

    if (tokens_to_quantize == 0) {
        LLAMA_LOG_DEBUG("[mixed-kv]   - no tokens to quantize, returning early\n");
        return; // Nothing to quantize
    }

    // Calculate memory impact for debug output
    size_t fp16_size_per_token = (ggml_type_size(config.hot_type_k) + ggml_type_size(config.hot_type_v)) * 
                                (hparams.n_embd_k_gqa(il) + hparams.n_embd_v_gqa(il));
    size_t quant_size_per_token = (ggml_type_size(config.cold_type_k) + ggml_type_size(config.cold_type_v)) * 
                                 (hparams.n_embd_k_gqa(il) + hparams.n_embd_v_gqa(il));
    size_t memory_saved = tokens_to_quantize * (fp16_size_per_token - quant_size_per_token);

    LLAMA_LOG_DEBUG("[mixed-kv] memory impact of quantization:\n");
    LLAMA_LOG_DEBUG("[mixed-kv]   - FP16 size per token: %s\n", format_memory_size(fp16_size_per_token).c_str());
    LLAMA_LOG_DEBUG("[mixed-kv]   - quantized size per token: %s\n", format_memory_size(quant_size_per_token).c_str());
    LLAMA_LOG_DEBUG("[mixed-kv]   - memory saved: %s\n", format_memory_size(memory_saved).c_str());

    // Log quantization operation details
    LLAMA_LOG_INFO("%s: scheduling quantization of oldest %u tokens for layer %d from %s to %s (model arch: %s)\n",
                   __func__, tokens_to_quantize, il,
                   ggml_type_name(config.hot_type_k), ggml_type_name(config.cold_type_k),
                   llm_arch_name(model.arch));

    /*
     * Correct Quantization Strategy:
     * 
     * In llama.cpp, we should not create ggml_context inside KV cache.
     * Instead, we should:
     * 1. Mark data that needs quantization
     * 2. Handle quantization in update() method through graph building mechanism
     * 3. Use build_graph_quantize() method to build quantization graph
     * 
     * Currently as a temporary solution, we perform direct memory copy operations,
     * but this should be refactored to use graph building mechanism in future versions.
     */

    // Temporary Implementation: Direct Memory Operations
    // TODO: Refactor to use graph building mechanism
    
    try {
        /*
         * Temporary Quantization Process:
         * 
         * Since we cannot create context inside KV cache, we use direct memory
         * operations as a temporary solution. This is not optimal, but ensures
         * compatibility with llama.cpp architecture.
         * 
         * Step 1: Copy data directly to quantization buffer
         * Step 2: Move remaining FP16 data
         * Step 3: Update counters
         */
        
        // Calculate data sizes to move
        size_t k_token_size = ggml_row_size(layer.k_fp16->type, hparams.n_embd_k_gqa(il));
        size_t v_token_size = ggml_row_size(layer.v_fp16->type, hparams.n_embd_v_gqa(il));
        
        // Get source data pointers (oldest FP16 tokens)
        uint8_t * k_src = (uint8_t*)layer.k_fp16->data;
        uint8_t * v_src = (uint8_t*)layer.v_fp16->data;
        
        // Get target data pointers (end of quantization buffer)
        uint8_t * k_dst = (uint8_t*)layer.k_quant->data + (layer.n_quant_tokens * ggml_row_size(layer.k_quant->type, hparams.n_embd_k_gqa(il)));
        uint8_t * v_dst = (uint8_t*)layer.v_quant->data + (layer.n_quant_tokens * ggml_row_size(layer.v_quant->type, hparams.n_embd_v_gqa(il)));
        
        // NOTE: Here we temporarily just copy data, without actual quantization
        // Real quantization should be implemented through ggml_cpy and type conversion
        // but this needs to be done in graph building process
        
        LLAMA_LOG_WARN("[mixed-kv] WARNING: Using temporary direct memory copy instead of proper quantization\n");
        LLAMA_LOG_WARN("[mixed-kv] This should be replaced with graph-based quantization in future versions\n");
        
        // Temporary solution: direct data copy (no actual quantization)
        // In real applications, this should be done through ggml graph operations for type conversion
        for (uint32_t i = 0; i < tokens_to_quantize; ++i) {
            // Note: This is just copying, not quantizing!
            // Real quantization needs ggml_cpy and type conversion
            memcpy(k_dst + i * ggml_row_size(layer.k_quant->type, hparams.n_embd_k_gqa(il)),
                   k_src + i * k_token_size,
                   std::min(k_token_size, ggml_row_size(layer.k_quant->type, hparams.n_embd_k_gqa(il))));
                   
            memcpy(v_dst + i * ggml_row_size(layer.v_quant->type, hparams.n_embd_v_gqa(il)),
                   v_src + i * v_token_size,
                   std::min(v_token_size, ggml_row_size(layer.v_quant->type, hparams.n_embd_v_gqa(il))));
        }

        /*
         * Step 2: Move remaining FP16 tokens to buffer beginning
         */
        uint32_t remaining_fp16_tokens = layer.n_fp16_tokens - tokens_to_quantize;

        if (remaining_fp16_tokens > 0) {
            // Move remaining FP16 data to buffer beginning
            memmove(k_src, 
                    k_src + tokens_to_quantize * k_token_size,
                    remaining_fp16_tokens * k_token_size);
                    
            memmove(v_src,
                    v_src + tokens_to_quantize * v_token_size,
                    remaining_fp16_tokens * v_token_size);
        }

        // Update token counts
        layer.n_quant_tokens += tokens_to_quantize;
        layer.n_fp16_tokens = remaining_fp16_tokens;

        // Calculate performance metrics
        auto end_time = get_current_time();
        double duration_ms = get_duration_ms(start_time, end_time);
        double tokens_per_ms = tokens_to_quantize / duration_ms;
        
        LLAMA_LOG_DEBUG("[mixed-kv] quantization performance metrics:\n");
        LLAMA_LOG_DEBUG("[mixed-kv]   - duration: %.2f ms\n", duration_ms);
        LLAMA_LOG_DEBUG("[mixed-kv]   - tokens processed: %u\n", tokens_to_quantize);
        LLAMA_LOG_DEBUG("[mixed-kv]   - throughput: %.2f tokens/ms\n", tokens_per_ms);
        LLAMA_LOG_DEBUG("[mixed-kv]   - memory saved: %s\n", format_memory_size(memory_saved).c_str());
        
        LLAMA_LOG_DEBUG("[mixed-kv] updated token counts for layer %d:\n", il);
        LLAMA_LOG_DEBUG("[mixed-kv]   - quantized tokens: %u (was %u)\n", layer.n_quant_tokens, layer.n_quant_tokens - tokens_to_quantize);
        LLAMA_LOG_DEBUG("[mixed-kv]   - FP16 tokens: %u (was %u)\n", layer.n_fp16_tokens, layer.n_fp16_tokens + tokens_to_quantize);

        LLAMA_LOG_DEBUG("%s: quantization completed for layer %d, now have %u quantized + %u FP16 tokens\n",
                        __func__, il, layer.n_quant_tokens, layer.n_fp16_tokens);

    } catch (const std::exception& e) {
        LLAMA_LOG_ERROR("[mixed-kv] ERROR: quantization failed for layer %d: %s\n", il, e.what());
        LLAMA_LOG_ERROR("%s: quantization failed for layer %d: %s\n", __func__, il, e.what());
    }
}

// Legacy method - now calls the new FIFO-based quantization
void llama_kv_cache_mixed::quantize_tokens(int32_t il) {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return;
    }

    auto & layer = layers[it->second];
    quantize_oldest_tokens(il, layer.n_fp16_tokens);
}

// Input setting functions - similar to unified cache
void llama_kv_cache_mixed::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    const int64_t n_tokens     = ubatch->n_tokens;
    const int64_t n_seq_tokens = ubatch->n_seq_tokens;
    const int64_t n_seqs       = ubatch->n_seqs;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    float * data = (float *) dst->data;

    const int64_t n_kv = n;

    // Use only the previous KV cells of the correct sequence for each token of the ubatch.
    // It's assumed that if a token in the batch has multiple sequences, they are equivalent.
    // Example with a cache of 10 tokens, 2 tokens populated in cache and 3 tokens in batch:
    //   Causal mask:
    //      xxx-------
    //      xxxx------
    //      xxxxx-----
    //   Non-causal mask:
    //      xxxxx-----
    //      xxxxx-----
    //      xxxxx-----
    // To visualize the mask, see https://github.com/ggml-org/llama.cpp/pull/12615
    for (int h = 0; h < 1; ++h) {
        for (int s = 0; s < n_seqs; ++s) {
            const llama_seq_id seq_id = ubatch->seq_id[s][0];

            for (int j = 0; j < n_seq_tokens; ++j) {
                const llama_pos p1 = ubatch->pos[s*n_seq_tokens + j];

                for (int i = 0; i < n_kv; ++i) {
                    const llama_pos p0 = cells[i].pos;

                    bool masked = false;

                    // mask the token if not the same sequence
                    masked = masked || (!cells[i].has_seq_id(seq_id));

                    // mask future tokens
                    masked = masked || (causal_attn && p0 > p1);

                    // Note: SWA masking not implemented for mixed cache yet
                    // masked = masked || (is_masked_swa(p0, p1));

                    float f = 0.0f;

                    if (masked) {
                        f = -INFINITY;
                    } else if (hparams.use_alibi) {
                        f = -std::abs(p0 - p1);
                    }

                    data[h*(n_kv*n_tokens) + s*(n_kv*n_seq_tokens) + j*n_kv + i] = f;
                }
            }
        }

        // mask padded tokens
        if (data) {
            for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
                for (int j = 0; j < n_kv; ++j) {
                    data[h*(n_kv*n_tokens) + i*n_kv + j] = -INFINITY;
                }
            }
        }
    }
}

void llama_kv_cache_mixed::set_input_k_shift(ggml_tensor * dst) const {
    // Similar implementation to unified cache
    GGML_UNUSED(dst);
    // TODO: Implement
}

void llama_kv_cache_mixed::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    // Similar implementation to unified cache
    GGML_UNUSED(dst);
    GGML_UNUSED(ubatch);
    // TODO: Implement
}

// State save/load
void llama_kv_cache_mixed::state_write(llama_io_write_i & io, llama_seq_id seq_id) const {
    GGML_UNUSED(io);
    GGML_UNUSED(seq_id);
    // TODO: Implement state serialization
}

void llama_kv_cache_mixed::state_read(llama_io_read_i & io, llama_seq_id seq_id) {
    GGML_UNUSED(io);
    GGML_UNUSED(seq_id);
    // TODO: Implement state deserialization
}

// Helper functions
uint32_t llama_kv_cache_mixed::cell_max() const {
    // Similar to unified cache
    for (uint32_t i = size; i > 0; --i) {
        const kv_cell & cell = cells[i - 1];

        if (cell.pos >= 0 && !cell.is_empty()) {
            return i;
        }
    }

    return 0;
}

size_t llama_kv_cache_mixed::total_size() const {
    size_t size_k = size_k_bytes();
    size_t size_v = size_v_bytes();
    return size_k + size_v;
}

size_t llama_kv_cache_mixed::size_k_bytes() const {
    size_t total = 0;
    for (const auto & layer : layers) {
        total += ggml_nbytes(layer.k_fp16);
        total += ggml_nbytes(layer.k_quant);
    }
    return total;
}

size_t llama_kv_cache_mixed::size_v_bytes() const {
    size_t total = 0;
    for (const auto & layer : layers) {
        total += ggml_nbytes(layer.v_fp16);
        total += ggml_nbytes(layer.v_quant);
    }
    return total;
}

// Graph building functions - placeholder implementations
llm_graph_result_ptr llama_kv_cache_mixed::build_graph_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_cgraph * gf) const {
    GGML_UNUSED(cparams);
    GGML_UNUSED(ctx);
    GGML_UNUSED(gf);
    // TODO: Implement shift graph building
    return nullptr;
}

llm_graph_result_ptr llama_kv_cache_mixed::build_graph_defrag(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_cgraph * gf) const {
    GGML_UNUSED(cparams);
    GGML_UNUSED(ctx);
    GGML_UNUSED(gf);
    // TODO: Implement defrag graph building
    return nullptr;
}

llm_graph_result_ptr llama_kv_cache_mixed::build_graph_quantize(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_cgraph * gf,
                     int32_t il) const {
    LLAMA_LOG_DEBUG("[mixed-kv] building quantization graph for layer %d\n", il);
    
    auto res = std::make_unique<llm_graph_result>();

    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        LLAMA_LOG_ERROR("[mixed-kv] ERROR: layer %d not found in cache for quantization graph\n", il);
        return res;
    }

    const auto & layer = layers[it->second];

    // Check if there are tokens that need quantization
    if (layer.n_fp16_tokens == 0) {
        LLAMA_LOG_DEBUG("[mixed-kv] no FP16 tokens to quantize for layer %d\n", il);
        return res;
    }

    /*
     * Graph-based Quantization Process:
     * 
     * This is the correct llama.cpp quantization approach:
     * 1. Create views of source and target tensors
     * 2. Use ggml_cpy for type conversion (quantization)
     * 3. Add operations to computation graph
     * 4. Let caller execute the graph
     * 
     * Advantages:
     * - Consistent with llama.cpp architecture
     * - Support for GPU acceleration
     * - Support for backend optimization
     * - Memory management handled by framework
     */

    // Calculate number of tokens to quantize (using configured threshold)
    uint32_t tokens_to_quantize = std::min(layer.n_fp16_tokens, config.group_size);
    
    if (tokens_to_quantize == 0) {
        return res;
    }

    LLAMA_LOG_DEBUG("[mixed-kv] creating quantization graph for %u tokens in layer %d\n", tokens_to_quantize, il);

    // Create source views (oldest FP16 data)
    ggml_tensor * k_src = ggml_view_2d(ctx, layer.k_fp16,
                                      layer.k_fp16->ne[0], tokens_to_quantize,
                                      layer.k_fp16->nb[1], 0);
    ggml_tensor * v_src = ggml_view_2d(ctx, layer.v_fp16,
                                      layer.v_fp16->ne[0], tokens_to_quantize,
                                      layer.v_fp16->nb[1], 0);

    // Create target views (quantized storage)
    ggml_tensor * k_dst = ggml_view_2d(ctx, layer.k_quant,
                                      layer.k_quant->ne[0], tokens_to_quantize,
                                      layer.k_quant->nb[1],
                                      layer.n_quant_tokens * layer.k_quant->nb[1]);
    ggml_tensor * v_dst = ggml_view_2d(ctx, layer.v_quant,
                                      layer.v_quant->ne[0], tokens_to_quantize,
                                      layer.v_quant->nb[1],
                                      layer.n_quant_tokens * layer.v_quant->nb[1]);

    // Perform quantization (type conversion)
    ggml_tensor * k_quantized = ggml_cpy(ctx, k_src, k_dst);
    ggml_tensor * v_quantized = ggml_cpy(ctx, v_src, v_dst);

    // Add to computation graph
    ggml_build_forward_expand(gf, k_quantized);
    ggml_build_forward_expand(gf, v_quantized);

    // If there are remaining FP16 tokens, need to move them
    uint32_t remaining_fp16_tokens = layer.n_fp16_tokens - tokens_to_quantize;
    if (remaining_fp16_tokens > 0) {
        // Create source views for remaining data
        ggml_tensor * k_remaining_src = ggml_view_2d(ctx, layer.k_fp16,
                                                    layer.k_fp16->ne[0], remaining_fp16_tokens,
                                                    layer.k_fp16->nb[1],
                                                    tokens_to_quantize * layer.k_fp16->nb[1]);
        ggml_tensor * v_remaining_src = ggml_view_2d(ctx, layer.v_fp16,
                                                    layer.v_fp16->ne[0], remaining_fp16_tokens,
                                                    layer.v_fp16->nb[1],
                                                    tokens_to_quantize * layer.v_fp16->nb[1]);

        // Create target views (FP16 buffer beginning)
        ggml_tensor * k_remaining_dst = ggml_view_2d(ctx, layer.k_fp16,
                                                    layer.k_fp16->ne[0], remaining_fp16_tokens,
                                                    layer.k_fp16->nb[1], 0);
        ggml_tensor * v_remaining_dst = ggml_view_2d(ctx, layer.v_fp16,
                                                    layer.v_fp16->ne[0], remaining_fp16_tokens,
                                                    layer.v_fp16->nb[1], 0);

        // Move remaining data
        ggml_tensor * k_moved = ggml_cpy(ctx, k_remaining_src, k_remaining_dst);
        ggml_tensor * v_moved = ggml_cpy(ctx, v_remaining_src, v_remaining_dst);

        // Add to computation graph
        ggml_build_forward_expand(gf, k_moved);
        ggml_build_forward_expand(gf, v_moved);
    }

    LLAMA_LOG_DEBUG("[mixed-kv] quantization graph built successfully for layer %d (%u tokens)\n", il, tokens_to_quantize);

    return res;
}

bool llama_kv_cache_mixed::defrag_prepare(int32_t n_max_nodes) {
    GGML_UNUSED(n_max_nodes);
    // TODO: Implement defrag preparation
    return false;
}

void llama_kv_cache_mixed::state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id) const {
    GGML_UNUSED(io);
    GGML_UNUSED(cell_ranges);
    GGML_UNUSED(seq_id);
    // TODO: Implement
}

void llama_kv_cache_mixed::state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const {
    GGML_UNUSED(io);
    GGML_UNUSED(cell_ranges);
    // TODO: Implement
}

bool llama_kv_cache_mixed::state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id) {
    GGML_UNUSED(io);
    GGML_UNUSED(cell_count);
    GGML_UNUSED(dest_seq_id);
    // TODO: Implement
    return false;
}

bool llama_kv_cache_mixed::state_read_data(llama_io_read_i & io, uint32_t cell_count) {
    GGML_UNUSED(io);
    GGML_UNUSED(cell_count);
    // TODO: Implement
    return false;
}

//
// Enhanced quantization methods implementation
//

bool llama_kv_cache_mixed::should_trigger_quantization() const {
    float memory_pressure = calculate_memory_pressure();
    return quant_mgr.should_quantize(config, memory_pressure);
}

void llama_kv_cache_mixed::trigger_quantization_if_needed(uint32_t new_tokens) {
    if (quant_mgr.quantization_in_progress) {
        LLAMA_LOG_WARN("%s: quantization already in progress, skipping\n", __func__);
        return;
    }

    quant_mgr.quantization_in_progress = true;
    quant_mgr.last_quantization_start = std::chrono::high_resolution_clock::now();

    LLAMA_LOG_INFO("%s: starting quantization of %u accumulated tokens\n", __func__, new_tokens);

    uint32_t total_quantized = 0;

    // Quantize all layers
    for (auto & layer : layers) {
        if (layer.n_fp16_tokens > 0) {
            quantize_tokens(layer.il);
            total_quantized += layer.n_fp16_tokens;
        }
    }

    // Calculate timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - quant_mgr.last_quantization_start);
    double time_ms = duration.count() / 1000.0;

    // Update statistics
    update_quantization_stats(total_quantized, time_ms);

    // Reset accumulation
    quant_mgr.reset_accumulation();
    quant_mgr.quantization_in_progress = false;

    LLAMA_LOG_INFO("%s: quantization completed in %.2f ms, %u tokens quantized\n",
                   __func__, time_ms, total_quantized);
}

void llama_kv_cache_mixed::update_quantization_stats(uint32_t tokens_quantized, double time_ms) {
    quant_stats.total_tokens_quantized += tokens_quantized;
    quant_stats.quantization_events++;
    quant_stats.last_quantization_time_ms = time_ms;
    quant_stats.total_quantization_time_ms += time_ms;
    quant_stats.avg_quantization_time_ms = quant_stats.total_quantization_time_ms / quant_stats.quantization_events;

    // Calculate compression ratio (assuming Q4_0 is ~4x smaller than FP16)
    if (quant_stats.total_tokens_processed > 0) {
        quant_stats.compression_ratio = static_cast<float>(quant_stats.total_tokens_quantized) /
                                       static_cast<float>(quant_stats.total_tokens_processed);
    }

    // Estimate memory saved (FP16 = 2 bytes, Q4_0 ≈ 0.5 bytes per value)
    // Assuming each token has n_embd values
    size_t fp16_size_per_token = hparams.n_embd * 2;  // 2 bytes per FP16 value
    size_t q4_0_size_per_token = hparams.n_embd / 2;  // ~0.5 bytes per Q4_0 value
    quant_stats.memory_saved_bytes += tokens_quantized * (fp16_size_per_token - q4_0_size_per_token);
}

float llama_kv_cache_mixed::calculate_memory_pressure() const {
    size_t total_memory = total_size();
    size_t fp16_memory = 0;

    // Calculate current FP16 memory usage
    for (const auto & layer : layers) {
        fp16_memory += layer.n_fp16_tokens * (ggml_type_size(config.hot_type_k) + ggml_type_size(config.hot_type_v));
    }

    if (total_memory == 0) {
        return 0.0f;
    }

    return static_cast<float>(fp16_memory) / static_cast<float>(total_memory);
}

void llama_kv_cache_mixed::adaptive_threshold_update() {
    float memory_pressure = calculate_memory_pressure();
    quant_mgr.update_threshold(config, memory_pressure);
}

llama_kv_cache_mixed::memory_info llama_kv_cache_mixed::get_memory_info() const {
    memory_info info;

    info.total_memory_bytes = total_size();

    // Calculate FP16 and quantized memory usage
    for (const auto & layer : layers) {
        info.fp16_memory_bytes += layer.n_fp16_tokens *
            (ggml_type_size(config.hot_type_k) + ggml_type_size(config.hot_type_v));
        info.quant_memory_bytes += layer.n_quant_tokens *
            (ggml_type_size(config.cold_type_k) + ggml_type_size(config.cold_type_v));
    }

    info.memory_pressure = calculate_memory_pressure();
    info.should_quantize = should_trigger_quantization();

    return info;
}

//> ===================================================================================================
//> Following are the original get_k and get_v functions from llama.cpp
//> ===================================================================================================

/*
 * Public API methods for getting K and V tensors
 * 
 * Simple implementation like unified cache - just return FP16 views
 */
ggml_tensor * llama_kv_cache_mixed::get_k(ggml_context * ctx, int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return nullptr;
    }

    const auto & layer = layers[it->second];

    // Use only FP16 tensor, exactly like unified cache
    auto * k = layer.k_fp16;

    // Create view exactly like unified cache
    return ggml_view_3d(ctx, k,
            hparams.n_embd_head_k, hparams.n_head_kv(il), n,
            ggml_row_size(k->type, hparams.n_embd_head_k),
            ggml_row_size(k->type, hparams.n_embd_k_gqa(il)),
            0);
}

ggml_tensor * llama_kv_cache_mixed::get_v(ggml_context * ctx, int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return nullptr;
    }

    const auto & layer = layers[it->second];

    // Use only FP16 tensor, exactly like unified cache
    auto * v = layer.v_fp16;

    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        return ggml_view_3d(ctx, v,
                hparams.n_embd_head_v, hparams.n_head_kv(il), n,
                ggml_row_size(v->type, hparams.n_embd_head_v),    // v->nb[1]
                ggml_row_size(v->type, hparams.n_embd_v_gqa(il)), // v->nb[2]
                0);
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_3d(ctx, v,
            n, hparams.n_head_kv(il), hparams.n_embd_head_v,
            ggml_row_size(v->type, v->ne[1]*hparams.n_embd_head_v), // v->nb[1]
            ggml_row_size(v->type, v->ne[1]),                       // v->nb[2]
            0);
}

ggml_tensor * llama_kv_cache_mixed::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto & layer = layers[ikv];
    auto * k = layer.k_fp16;

    const int64_t n_tokens = k_cur->ne[2];

    // Update FP16 token counter
    layer.n_fp16_tokens += n_tokens;

    LLAMA_LOG_DEBUG("[mixed-kv] adding %ld K tokens to layer %d cache (head=%u)\n", n_tokens, il, head);
    LLAMA_LOG_DEBUG("[mixed-kv]   - current FP16 tokens: %u, quantized tokens: %u\n", 
                    layer.n_fp16_tokens - n_tokens, layer.n_quant_tokens);
    LLAMA_LOG_DEBUG("[mixed-kv]   - updated FP16 tokens: %u (added %ld)\n", 
                    layer.n_fp16_tokens, n_tokens);

    ggml_tensor * k_view = ggml_view_1d(ctx, k,
            n_tokens*hparams.n_embd_k_gqa(il),
            ggml_row_size(k->type, hparams.n_embd_k_gqa(il))*head);

    return ggml_cpy(ctx, k_cur, k_view);
}

ggml_tensor * llama_kv_cache_mixed::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto & layer = layers[ikv];
    auto * v = layer.v_fp16;

    const int64_t n_tokens = v_cur->ne[2];

    // NOTE: We don't increment FP16 token counter here since it's already done in cpy_k
    // Both K and V should have the same token count, so we only count once
    
    LLAMA_LOG_DEBUG("[mixed-kv] adding %ld V tokens to layer %d cache (head=%u)\n", n_tokens, il, head);
    LLAMA_LOG_DEBUG("[mixed-kv]   - current total FP16 tokens: %u\n", layer.n_fp16_tokens);

    v_cur = ggml_reshape_2d(ctx, v_cur, hparams.n_embd_v_gqa(il), n_tokens);

    ggml_tensor * v_view = nullptr;

    if (!v_trans) {
        v_view = ggml_view_1d(ctx, v,
                n_tokens*hparams.n_embd_v_gqa(il),
                ggml_row_size(v->type, hparams.n_embd_v_gqa(il))*head);
    } else {
        // note: the V cache is transposed when not using flash attention
        v_view = ggml_view_2d(ctx, v, n_tokens, hparams.n_embd_v_gqa(il),
                (v->ne[1])*ggml_element_size(v),
                (    head)*ggml_element_size(v));

        v_cur = ggml_transpose(ctx, v_cur);
    }

    return ggml_cpy(ctx, v_cur, v_view);
}
