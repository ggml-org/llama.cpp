#include "llama-kv-cache-mixed.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-graph.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <cstring>
#include <chrono>

// Define missing macros if not available
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef CACHE_LINE_SIZE_F32
#define CACHE_LINE_SIZE_F32 16
#endif

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

uint32_t llama_kv_cache_mixed::get_padding(const llama_cparams & cparams) {
    GGML_UNUSED(cparams);

    return 32u;

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
      v_trans(v_trans), n_seq_max(n_seq_max), n_pad(n_pad) {

    GGML_ASSERT(kv_size % n_pad == 0);

    // create a context for each buffer type (allocator)
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

        // NOTICE: The FP16 tensors are not used during alignment testing, but they are used during quantization.
        layer.k_fp16    = ggml_new_tensor_2d(ctx, config.hot_type_k, n_embd_k_gqa, kv_size);
        layer.v_fp16    = ggml_new_tensor_2d(ctx, config.hot_type_v, n_embd_v_gqa, kv_size);
        // layer.k_fp16    = ggml_new_tensor_2d(ctx, config.hot_type_k, n_embd_k_gqa, config.max_fp16_window + config.quantization_threshold);
        // layer.v_fp16    = ggml_new_tensor_2d(ctx, config.hot_type_v, n_embd_v_gqa, config.max_fp16_window + config.quantization_threshold);

        // Create quantized tensors (for future use, but not used during alignment testing)
        layer.k_quant   = ggml_new_tensor_2d(ctx, config.cold_type_k, n_embd_k_gqa, kv_size);
        layer.v_quant   = ggml_new_tensor_2d(ctx, config.cold_type_v, n_embd_v_gqa, kv_size);

        // Use naming convention similar to unified cache for FP16 tensors
        ggml_format_name(layer.k_fp16,      "cache_k_l%d",          il);
        ggml_format_name(layer.v_fp16,      "cache_v_l%d",          il);
        ggml_format_name(layer.k_quant,     "cache_k_quant_l%d",    il);
        ggml_format_name(layer.v_quant,     "cache_v_quant_l%d",    il);

        map_layer_ids[il] = layers.size();
        layers.push_back(layer);
    }

    //> allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto it : ctx_map) {
        auto * buft = it.first;
        auto * ctx  = it.second;

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_DEBUG("%s: %10s KV buffer size = %8.2f MiB\n", __func__,
                       ggml_backend_buffer_name(buf),
                       ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);
        bufs.emplace_back(buf);
    }

    {
        const size_t memory_size_k = size_k_fp16_bytes();
        const size_t memory_size_v = size_v_fp16_bytes();

        LLAMA_LOG_DEBUG("%s: mixed cache size = %7.2f MiB (%6u cells, %3d layers, %2u seqs)\n",
                __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f),
                kv_size, (int) layers.size(), n_seq_max);
        LLAMA_LOG_DEBUG("%s:   FP16 K: %7.2f MiB, FP16 V: %7.2f MiB\n", __func__,
                (float)(memory_size_k/2) / (1024.0f * 1024.0f),
                (float)(memory_size_v/2) / (1024.0f * 1024.0f));
        LLAMA_LOG_DEBUG("%s:   Quant K (%s): %7.2f MiB, Quant V (%s): %7.2f MiB\n", __func__,
                ggml_type_name(config.cold_type_k), (float)(memory_size_k/2) / (1024.0f * 1024.0f),
                ggml_type_name(config.cold_type_v), (float)(memory_size_v/2) / (1024.0f * 1024.0f));
    }
}

llama_kv_cache_mixed::~llama_kv_cache_mixed() {
    // DEFENSIVE CLEANUP: Ensure safe destruction to prevent heap corruption
    try {
        LLAMA_LOG_DEBUG("[mixed-kv] destructor: starting safe cleanup\n");

        // Clear recovery structures safely first
        try {
            recovery.clear();
            LLAMA_LOG_DEBUG("[mixed-kv] destructor: recovery cleared\n");
        } catch (...) {
            LLAMA_LOG_WARN("[mixed-kv] destructor: exception in recovery cleanup\n");
        }

        // Clear cell structures
        try {
            for (auto& cell : cells) {
                cell.seq_id.clear();
            }
            cells.clear();
            LLAMA_LOG_DEBUG("[mixed-kv] destructor: cells cleared\n");
        } catch (...) {
            LLAMA_LOG_WARN("[mixed-kv] destructor: exception in cell cleanup\n");
        }

        // Clear layers safely
        try {
            layers.clear();
            map_layer_ids.clear();
            LLAMA_LOG_DEBUG("[mixed-kv] destructor: layers cleared\n");
        } catch (...) {
            LLAMA_LOG_WARN("[mixed-kv] destructor: exception in layer cleanup\n");
        }

        // Clear defrag info
        try {
            defrag_info.ids.clear();
            LLAMA_LOG_DEBUG("[mixed-kv] destructor: defrag info cleared\n");
        } catch (...) {
            LLAMA_LOG_WARN("[mixed-kv] destructor: exception in defrag cleanup\n");
        }

        // Reset counters to safe values
        head = 0;
        size = 0;
        used = 0;
        n = 0;

        LLAMA_LOG_DEBUG("[mixed-kv] destructor: cleanup completed successfully\n");
    } catch (const std::exception& e) {
        LLAMA_LOG_ERROR("[mixed-kv] destructor: exception during cleanup: %s\n", e.what());
    } catch (...) {
        LLAMA_LOG_ERROR("[mixed-kv] destructor: unknown exception during cleanup\n");
    }
}

void llama_kv_cache_mixed::clear() {
    LLAMA_LOG_DEBUG("[mixed-kv] clearing cache (size=%u, used=%u)\n", size, used);

    /*
     * Cell clearing operation - Reset all cache slots to initial empty state:
     *
     * Each element in the cells array represents a cache slot. The clear operation will:
     * 1. Set all pos values to -1 (indicating empty)
     * 2. Clear all seq_id sets
     * 3. Reset management counters (head=0, used=0)
     *
     * Before clear():                    After clear():
     * ┌─────┬─────┬─────┬─────┐         ┌─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:2│pos:3│   -->   │pos:-│pos:-│pos:-│pos:-│
     * │seq:1│seq:1│seq:2│seq:2│         │seq:∅│seq:∅│seq:∅│seq:∅│
     * │used │used │used │used │         │empty│empty│empty│empty│
     * └─────┴─────┴─────┴─────┘         └─────┴─────┴─────┴─────┘
     *   ↑                                 ↑
     * used=4                            used=0, head=0
     */
    for (uint32_t i = 0; i < size; ++i) {
        cells[i].pos = -1;        // Mark slot as empty
        cells[i].seq_id.clear();  // Clear sequence ID set
    }

    head = 0;
    used = 0;

    // Clear all layers and count tokens for debug output
    for (auto & layer : layers) {
        layer.quant_k_tokens = 0;
        layer.quant_v_tokens = 0;
        layer.fp16_k_tokens  = 0;
        layer.fp16_v_tokens  = 0;
    }

    for (auto & buf : bufs) {
        ggml_backend_buffer_clear(buf.get(), 0);
    }
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

    /*
     * Cell sequence removal operation - Remove sequence tokens from specified position range:
     *
     * Iterate through all cells, check if each cell's position is within removal range [p0, p1)
     * If within range and contains target sequence, remove that sequence from the cell's seq_id set
     * If cell becomes empty after removal (seq_id set empty), free that slot
     *
     * Example: seq_rm(seq_id=1, p0=1, p1=3) - Remove sequence 1 tokens at positions 1-2
     *
     * Before seq_rm():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:2│pos:3│pos:4│
     * │seq:1│seq:1│seq:1│seq:2│seq:1│  <- Need to remove seq:1 at pos:1-2
     * └─────┴─────┴─────┴─────┴─────┘
     *
     * After seq_rm():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:-│pos:-│pos:3│pos:4│
     * │seq:1│empty│empty│seq:2│seq:1│  <- pos:1,2 cleared and freed
     * └─────┴─────┴─────┴─────┴─────┘
     *         ↑     ↑
     *      new_head candidate positions (for optimizing future allocations)
     */
    for (uint32_t i = 0; i < size; ++i) {
        // Check if cell position is within removal range
        if (cells[i].pos >= p0 && cells[i].pos < p1) {
            if (seq_id < 0) {
                // seq_id < 0 means remove all sequences
                cells[i].seq_id.clear();
            } else if (cells[i].has_seq_id(seq_id)) {
                // Only remove specified sequence ID
                cells[i].seq_id.erase(seq_id);
            } else {
                continue;
            }

            if (cells[i].is_empty()) {
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

    /*
     * Cell sequence copy operation - Copy tokens from source sequence to destination sequence:
     *
     * Iterate through all cells, find cells belonging to source sequence within specified position range
     * Add destination sequence ID to these cells' seq_id set
     * This implements functionality for multiple sequences sharing the same token (e.g. for beam search)
     *
     * Example: seq_cp(seq_src=1, seq_dst=3, p0=1, p1=3) - Copy sequence 1 to sequence 3
     *
     * Before seq_cp():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:2│pos:3│pos:4│
     * │seq:1│seq:1│seq:1│seq:2│seq:1│  <- Copy seq:1's pos:1-2 to seq:3
     * └─────┴─────┴─────┴─────┴─────┘
     *
     * After seq_cp():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:2│pos:3│pos:4│
     * │seq:1│{1,3}│{1,3}│seq:2│seq:1│  <- pos:1,2 now belong to both seq:1 and seq:3
     * └─────┴─────┴─────┴─────┴─────┘
     *         ↑     ↑
     *      Shared tokens (multiple sequences reference same cache slot)
     */
    for (uint32_t i = 0; i < size; ++i) {
        // Check if cell belongs to source sequence and is within specified position range
        if (cells[i].has_seq_id(seq_id_src) && cells[i].pos >= p0 && cells[i].pos < p1) {
            // Add destination sequence ID to this cell (multiple sequences share same token)
            cells[i].seq_id.insert(seq_id_dst);
        }
    }
}

void llama_kv_cache_mixed::seq_keep(llama_seq_id seq_id) {
    uint32_t new_head = size;

    /*
     * Cell sequence keep operation - Keep only specified sequence, clear all others:
     * 
     * Iterate through all cells, completely clear cells not belonging to target sequence,
     * For cells belonging to target sequence, clean multi-sequence state to keep only target sequence
     * This is typically used to switch current active sequence and clean up unwanted branches
     *
     * Example: seq_keep(seq_id=2) - Keep only sequence 2, clear all other sequences
     *
     * Before seq_keep():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:2│pos:3│pos:4│
     * │seq:1│{1,3}│seq:2│{1,2}│seq:1│  <- Keep only seq:2
     * └─────┴─────┴─────┴─────┴─────┘
     *
     * After seq_keep():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:-│pos:-│pos:2│pos:3│pos:-│
     * │empty│empty│seq:2│seq:2│empty│  <- Only cells with seq:2 are kept
     * └─────┴─────┴─────┴─────┴─────┘
     *   ↑     ↑               ↑
     * new_head candidates (for subsequent allocation optimization)
     */
    for (uint32_t i = 0; i < size; ++i) {
        // Check if this cell does not belong to sequence to keep
        if (!cells[i].has_seq_id(seq_id)) {
            // Cell does not belong to target sequence, clear it
            if (cells[i].pos >= 0) {
                used--;  // Decrease used count
            }

            cells[i].pos = -1;           // Mark as free
            cells[i].seq_id.clear();     // Clear sequence IDs

            // Record first free position
            if (new_head == size) {
                //> This only change once. so the new head will be the FIRST free position.
                new_head = i;
            }
        } else {
            // Cell belongs to target sequence, clean its multi-sequence state to keep only target sequence
            cells[i].seq_id.clear();         // Clear all sequence IDs
            cells[i].seq_id.insert(seq_id);  // Insert only target sequence ID
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

    /*
     * Position offset operation for cells sequence - Move positions forward or backward for specified sequence:
     *
     * Iterate through all cells, find cells belonging to target sequence and within specified position range
     * Update their pos and delta values, clear cell if position becomes negative
     * This is used to implement sequence position offsets (like inserting/deleting tokens, position encoding adjustments etc.)
     *
     * Example: seq_add(seq_id=1, p0=2, p1=4, delta=2) - Move positions 2-3 of sequence 1 forward by 2
     *
     * Before seq_add():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:2│pos:3│pos:4│
     * │seq:1│seq:1│seq:1│seq:1│seq:2│  <- Tokens at pos:2-3 of seq:1 need +2 offset
     * └─────┴─────┴─────┴─────┴─────┘
     *               ↑─ range[2,4) ─↑
     *
     * After seq_add():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:4│pos:5│pos:4│
     * │seq:1│seq:1│seq:1│seq:1│seq:2│  <- pos:2→4, pos:3→5, delta accumulated
     * └─────┴─────┴─────┴─────┴─────┘
     *
     * Special case - If delta is negative and makes pos negative, clear that cell:
     * For example with delta=-3, pos:2-3 would become -1,0, cells with negative positions are cleared and freed
     */
    for (uint32_t i = 0; i < size; ++i) {
        // Check if cell belongs to target sequence and is within specified position range
        if (cells[i].has_seq_id(seq_id) && cells[i].pos >= p0 && cells[i].pos < p1) {
            has_shift = true;  // Mark that position shift occurred

            cells[i].pos   += delta;  // Update token position
            cells[i].delta += delta;  // Accumulate offset (used for RoPE etc)

            // If position becomes negative, token is moved out of valid range and needs to be cleared
            if (cells[i].pos < 0) {
                if (!cells[i].is_empty()) {
                    used--;  // Decrease used count
                }
                cells[i].pos = -1;           // Mark as free
                cells[i].seq_id.clear();     // Clear sequence IDs
                if (new_head == size) {
                    new_head = i;            // Record free position
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

    /*
     * Position division operation for cells sequence - Scale down positions proportionally:
     * 
     * Iterate through all cells, find cells belonging to target sequence and within specified position range
     * Divide their positions by divisor d and update accumulated delta offset
     * This is used to implement position scaling (like attention window scaling, position compression etc.)
     *
     * Example: seq_div(seq_id=1, p0=4, p1=8, d=2) - Divide positions 4-7 of sequence 1 by 2
     *
     * Before seq_div():
     * ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:4│pos:5│pos:6│pos:7│pos:8│pos:9│
     * │seq:1│seq:1│seq:1│seq:1│seq:1│seq:1│seq:2│seq:2│
     * └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
     *               ↑─ range[4,8) ─↑   <- These positions need division by 2
     *
     * After seq_div():
     * ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:2│pos:2│pos:3│pos:3│pos:8│pos:9│
     * │seq:1│seq:1│seq:1│seq:1│seq:1│seq:1│seq:2│seq:2│
     * └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
     *               ↑─ 4/2=2  5/2=2  6/2=3  7/2=3 ─↑
     *                  (delta also records position change)
     */
    for (uint32_t i = 0; i < size; ++i) {
        // Check if cell belongs to target sequence and is within specified position range
        if (cells[i].has_seq_id(seq_id) && cells[i].pos >= p0 && cells[i].pos < p1) {
            has_shift = true;  // Mark that position change occurred

            {
                llama_pos p_old = cells[i].pos;    // Save original position
                cells[i].pos   /= d;               // Scale position by division
                cells[i].delta += cells[i].pos - p_old;  // Calculate and accumulate offset
            }
        }
    }
}

llama_pos llama_kv_cache_mixed::seq_pos_min(llama_seq_id seq_id) const {
    llama_pos result = std::numeric_limits<llama_pos>::max();

    /*
     * Find minimum position for specified sequence:
     *
     * Example: Find min position for seq_id=1
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:5│pos:1│pos:3│pos:7│pos:2│
     * │seq:2│seq:1│seq:1│seq:2│seq:1│  <- seq:1 has positions 1,3,2
     * └─────┴─────┴─────┴─────┴─────┘
     *
     * Returns min(1,3,2) = 1
     */
    for (uint32_t i = 0; i < size; ++i) {
        // Check if cell belongs to target sequence
        if (cells[i].has_seq_id(seq_id)) {
            result = std::min(result, cells[i].pos);  // Update minimum position
        }
    }

    if (result == std::numeric_limits<llama_pos>::max()) {
        result = -1;
    }

    return result;
}

llama_pos llama_kv_cache_mixed::seq_pos_max(llama_seq_id seq_id) const {
    llama_pos result = -1;

    /*
     * Find maximum position for specified sequence:
     *
     * Example: Find max position for seq_id=1
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:5│pos:1│pos:3│pos:7│pos:2│
     * │seq:2│seq:1│seq:1│seq:2│seq:1│  <- seq:1 has positions 1,3,2
     * └─────┴─────┴─────┴─────┴─────┘
     *
     * Returns max(1,3,2) = 3
     */
    for (uint32_t i = 0; i < size; ++i) {
        // Check if cell belongs to target sequence
        if (cells[i].has_seq_id(seq_id)) {
            result = std::max(result, cells[i].pos);  // Update maximum position
        }
    }

    return result;
}

void llama_kv_cache_mixed::restore() {
    LLAMA_LOG_DEBUG("[mixed-kv] restoring %zu cells from recovery\n", recovery.cells.size());

    try {
        for (const auto & [id, cell] : recovery.cells) {
            // Validate cell index bounds
            if (id >= size) {
                LLAMA_LOG_ERROR("[mixed-kv] ERROR: recovery cell index %u out of bounds (size=%u)\n", id, size);
                continue;
            }

            /*
             * Restore single cell state and maintain used count correctly:
             *
             * Before restore:     After restore:
             * ┌─────┐             ┌─────┐
             * │pos:2│   <---      │pos:5│  (restore from recovery)
             * │seq:1│             │seq:2│
             * └─────┘             └─────┘
             * used++/used-- adjusted based on cell state changes
             */
            const bool is_empty0 = cells[id].is_empty();  // Whether current cell is empty
            const bool is_empty1 = cell.is_empty();       // Whether restored cell will be empty

            // Adjust used count based on state changes
            if (!is_empty0 && is_empty1) {
                used--;  // RESTORE : occupied -> empty
            } else if (is_empty0 && !is_empty1) {
                used++;  // RESTORE : empty -> occupied
            }

            // Safely restore cell state
            cells[id].pos       = cell.pos;         // Restore position
            cells[id].delta     = cell.delta;       // Restore offset
            cells[id].seq_id    = cell.seq_id;      // Restore sequence ID set

            LLAMA_LOG_DEBUG("[mixed-kv] restored cell %u (pos=%d, seq_ids=%zu)\n",
                           id, cell.pos, cell.seq_id.size());
        }

        // Clear recovery safely using swap pattern
        std::unordered_map<uint32_t, kv_cell> empty_map;
        recovery.cells.swap(empty_map);

        LLAMA_LOG_DEBUG("[mixed-kv] recovery restore completed successfully\n");
    } catch (const std::exception& e) {
        LLAMA_LOG_ERROR("[mixed-kv] ERROR: Exception during recovery restore: %s\n", e.what());
        // Still try to clear recovery
        recovery.cells.clear();
    } catch (...) {
        LLAMA_LOG_ERROR("[mixed-kv] ERROR: Unknown exception during recovery restore\n");
        // Still try to clear recovery
        recovery.cells.clear();
    }
}

void llama_kv_cache_mixed::commit() {
    if (recovery.cells.empty()) {
        LLAMA_LOG_WARN("%s: the recovery information upon a commit was empty - might indicate a bug\n", __func__);
        return;
    }

    //> DEFENSIVE FIX: Clear recovery cells safely to avoid memory corruption crashes
    try {
        // Use swap and clear pattern for safer destruction
        std::unordered_map<uint32_t, kv_cell> empty_map;
        recovery.cells.swap(empty_map);
        // empty_map destructor will handle cleanup safely

        LLAMA_LOG_DEBUG("[mixed-kv] recovery cleared successfully (swapped %zu cells)\n", empty_map.size());
    } catch (const std::exception& e) {
        LLAMA_LOG_ERROR("[mixed-kv] ERROR: Exception during recovery clear: %s\n", e.what());
        // Force clear the recovery structure
        recovery.cells.clear();
    } catch (...) {
        LLAMA_LOG_ERROR("[mixed-kv] ERROR: Unknown exception during recovery clear\n");
        // Force clear the recovery structure
        recovery.cells.clear();
    }

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
            has_shift = false;  // Reset shift flag

            /*
             * Clear all cell deltas:
             *
             * After K-shift operation:
             * ┌─────┬─────┬─────┬─────┐
             * │pos:2│pos:3│pos:4│pos:5│
             * │Δ:+2 │Δ:+2 │Δ:+2 │Δ:+2 │  <- Clear these accumulated deltas
             * └─────┴─────┴─────┴─────┘
             *
             * After delta reset:
             * ┌─────┬─────┬─────┬─────┐
             * │pos:2│pos:3│pos:4│pos:5│
             * │Δ: 0 │Δ: 0 │Δ: 0 │Δ: 0 │  <- Deltas reset to 0
             * └─────┴─────┴─────┴─────┘
             */
            for (uint32_t i = 0; i < size; ++i) {
                cells[i].delta = 0;  // Reset accumulated delta
            }
        }
    }

    if (do_defrag) {
        // NOTICE: Following not used.
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

    do_quant = config.enable_quantization && ( head != 0 && head - cell_max_quantized() >= config.quantization_threshold + config.fp16_window_size );

    if (do_quant) {
        LLAMA_LOG_DEBUG("%s: quantizing KV cache\n", __func__);

        // 标记cells为量化状态
        for (uint32_t i = cell_max_quantized(); i < head - config.fp16_window_size; i++) {
            if (i < size) {
                cells[i].set_quantized(true);
            }
        }

        // 构建量化计算图
        ggml_backend_sched_reset(sched);

        auto * gf = lctx.graph_init();
        auto * ctx = lctx.get_ctx_compute();

        // 对每一层进行量化
        for (size_t i = 0; i < layers.size(); ++i) {
            auto & layer = layers[i];
            
            // 构建 K 量化操作
            auto * k_quant_op = k_quant(ctx, layer.il);
            if (k_quant_op) {
                ggml_build_forward_expand(gf, k_quant_op);
                LLAMA_LOG_DEBUG("[mixed-kv] added K quantization for layer %d\n", layer.il);
            }
            
            // 构建 V 量化操作  
            auto * v_quant_op = v_quant(ctx, layer.il);
            if (v_quant_op) {
                ggml_build_forward_expand(gf, v_quant_op);
                LLAMA_LOG_DEBUG("[mixed-kv] added V quantization for layer %d\n", layer.il);
            }
        }

        ggml_backend_sched_alloc_graph(sched, gf);

        lctx.graph_compute(gf, false);

        need_reserve = true;
        
        do_quant = false;
    }

    LLAMA_LOG_DEBUG("[mixed-kv] update completed (quantization disabled for alignment testing)\n");

    //> IF need reserve, then llama-context will call reserve() to reserve the memory.
    return need_reserve;
}

void llama_kv_cache_mixed::defrag_sched(float thold) {
    // TODO : need adapt to mixed kv cache.
    const float fragmentation = n >= 2048 ? std::max(0.0f, 1.0f - (float(used + n_pad)/n)) : 0.0f;

    if (fragmentation > thold) {
        LLAMA_LOG_DEBUG("%s: fragmentation: %.2f - requesting defrag\n", __func__, fragmentation);
        do_defrag = true;
    }
}

void llama_kv_cache_mixed::set_full() {
    head  = 0;      //> head is the start of the cache (loop buffer)
    n     = size;   //> n is the size of the cache (loop buffer)
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
            //> Some cell may be empty, but the position is not reset to -1.
            if (cells[head + i].pos >= 0) {
                found = false;        // Found occupied slot, current position not usable
                head += i + 1;        // Move head to next possible position
                n_tested += i + 1;    // Update tested slot count
                break;
            }
        }

        if (found) {
            break;
        }

        // NOTICE: Loop termination condition - n_tested >= size means entire cache searched with no free slots
        if (n_tested >= size) {
            return false;   //> Returning false will cause failure
        }
    }

    for (uint32_t i = 0; i < n_tokens; ++i) {
        // Calculate current token's cell index
        const uint32_t cell_idx = head + i;

        // Boundary check: Ensure cell index is within valid range
        if (cell_idx >= size) {
            LLAMA_LOG_ERROR("[mixed-kv] ERROR: cell index %u out of bounds (size=%u)\n", cell_idx, size);
            return false;
        }

        // Check if recovery info already exists for this cell
        // If not, save current state for potential rollback
        if (recovery.cells.find(cell_idx) == recovery.cells.end()) {
            try {
                // Create safe backup of cell state
                kv_cell backup_cell;
                backup_cell.pos = cells[cell_idx].pos;         // Backup position
                backup_cell.delta = cells[cell_idx].delta;     // Backup delta
                backup_cell.seq_id = cells[cell_idx].seq_id;   // Safely copy sequence ID set

                recovery.cells[cell_idx] = std::move(backup_cell);

                LLAMA_LOG_DEBUG("[mixed-kv] stored recovery info for cell %u (pos=%d, seq_ids=%zu)\n",
                               cell_idx, backup_cell.pos, backup_cell.seq_id.size());
            } catch (const std::exception& e) {
                LLAMA_LOG_ERROR("[mixed-kv] ERROR: Failed to store recovery info for cell %u: %s\n", cell_idx, e.what());
                return false;
            }
        }

        // Set new token's position
        cells[cell_idx].pos = ubatch.pos[i];

        // Associate token with corresponding sequences
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; j++) {
            cells[cell_idx].seq_id.insert(ubatch.seq_id[i][j]);
        }
    }

    used += n_tokens;

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // after enough generations, the benefit from this heuristic disappears
    // if we start defragmenting the cache, the benefit from this will be more important

    // NOTE: cell_max() return the last empty cell index.
    n = std::min(size, std::max(n_pad, GGML_PAD(cell_max(), n_pad)));                       //> Virtual head of kv cache.
    n_quantized = std::min(size, std::max(n_pad, GGML_PAD(cell_max_quantized(), n_pad)));   //> Virtual head of quantized kv cache.
    
    // LLAMA_LOG_INFO("\n[mixed-kv] successfully allocated slot: head=%u, used=%u, n=%u, n_quantized=%u, cell_max=%u, cell_max_quantized=%u\n", head, used, n, n_quantized, cell_max(), cell_max_quantized());

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
                // Current query token's position
                const llama_pos p1 = ubatch->pos[s*n_seq_tokens + j];

                // Loop through all tokens in KV cache
                for (int i = 0; i < n_kv; ++i) {
                    // Current key token's position
                    const llama_pos p0 = cells[i].pos;  //> kv_cache idx.

                    bool masked = false;

                    // Rule 1: If key token not in current query token's sequence, mask.
                    masked = masked || (!cells[i].has_seq_id(seq_id));  //> This cell is not in the current query token's sequence.

                    // Rule 2: If causal attention and key token after query token (future), mask.
                    masked = masked || (causal_attn && p0 > p1);            //> p0 in SEQ_LEN > p1 in KV_LEN.

                    float f = 0.0f;

                    if (masked) {
                        // For masked tokens, set attention score to negative infinity
                        f = -INFINITY;
                    } else if (hparams.use_alibi) {
                        // Rule 3: If using ALiBi, compute penalty based on query-key distance
                        f = -std::abs(p0 - p1);
                    }

                    // Write computed mask value to destination tensor
                    data[h*(n_kv*n_tokens) + s*(n_kv*n_seq_tokens) + j*n_kv + i] = f;
                }
            }
        }

        // TODO : Adapt to mixed kv cache.
        // Rule 4: Mask padding tokens in batch
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

uint32_t llama_kv_cache_mixed::cell_max_quantized() const {
    for (uint32_t i = size; i > 0; --i) {
        const kv_cell & cell = cells[i - 1];
        if (cell.pos >= 0 && cell.is_quantized()) {
            return i;
        }
    }

    return 0;
}

//> ===================================================================================================
//> Memory Size Calculation
//> ===================================================================================================

size_t llama_kv_cache_mixed::total_size() const {
    size_t size_k = size_k_fp16_bytes();
    size_t size_v = size_v_fp16_bytes();
    size_t size_k_quant = size_k_quant_bytes();
    size_t size_v_quant = size_v_quant_bytes();

    return size_k + size_v + size_k_quant + size_v_quant;
}

size_t llama_kv_cache_mixed::size_k_fp16_bytes() const {
    size_t total = 0;
    for (const auto & layer : layers) {
        total += ggml_nbytes(layer.k_fp16);
        // total += ggml_nbytes(layer.k_quant);
    }
    return total;
}

size_t llama_kv_cache_mixed::size_v_fp16_bytes() const {
    size_t total = 0;
    for (const auto & layer : layers) {
        total += ggml_nbytes(layer.v_fp16);
        // total += ggml_nbytes(layer.v_quant);
    }
    return total;
}

size_t llama_kv_cache_mixed::size_k_quant_bytes() const {
    size_t total = 0;
    for (const auto & layer : layers) {
        total += ggml_nbytes(layer.k_quant);
    }
    return total;
}

size_t llama_kv_cache_mixed::size_v_quant_bytes() const {
    size_t total = 0;
    for (const auto & layer : layers) {
        total += ggml_nbytes(layer.v_quant);
    }
    return total;
}

//> ===================================================================================================
//> Graph Building Functions
//> ===================================================================================================

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

//> ===================================================================================================
//> Following are the get_k/get_v/get_k_quant/get_v_quant/get_k_quant_ref/get_v_quant_ref functions for mixed kv cache.
//> ===================================================================================================
ggml_tensor * llama_kv_cache_mixed::get_k(ggml_context * ctx, int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return nullptr;
    }

    const auto & layer = layers[it->second];
    auto * k = layer.k_fp16;

    // Create view exactly like unified cache, but limit to actual available tokens
    return ggml_view_3d(ctx, k,
            hparams.n_embd_head_k, hparams.n_head_kv(il), n,
            ggml_row_size(k->type, hparams.n_embd_head_k),
            ggml_row_size(k->type, hparams.n_embd_k_gqa(il)),
            0
        );
}

ggml_tensor * llama_kv_cache_mixed::get_v(ggml_context * ctx, int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return nullptr;
    }

    const auto & layer = layers[it->second];
    auto * v = layer.v_fp16;

    // Create view exactly like unified cache, but limit to actual available tokens
    if (!v_trans) {
        return ggml_view_3d(ctx, v,
                hparams.n_embd_head_v, hparams.n_head_kv(il), n,
                ggml_row_size(v->type, hparams.n_embd_head_v),
                ggml_row_size(v->type, hparams.n_embd_v_gqa(il)),
                0
            );
    }

    // For transposed V tensor
    return ggml_view_3d(ctx, v,
            n, hparams.n_head_kv(il), hparams.n_embd_head_v,
            ggml_row_size(v->type, v->ne[1]*hparams.n_embd_head_v),
            ggml_row_size(v->type, v->ne[1]),
            0
        );
}

ggml_tensor * llama_kv_cache_mixed::get_k_quant(ggml_context * ctx, int32_t il) const {
    auto it = map_layer_ids.find(il);

    if (it == map_layer_ids.end()) {
        return nullptr;
    }

    const auto & layer = layers[it->second];
    auto * k_quant = layer.k_quant;

    // If no quantized tokens, return nullptr
    if (layer.quant_k_tokens == 0) {
        // NOTICE: This can only happen when the graph is pre-built.
        return ggml_view_3d(ctx, k_quant,
                hparams.n_embd_head_k, hparams.n_head_kv(il), n_quantized,
                ggml_row_size(k_quant->type, hparams.n_embd_head_k),
                ggml_row_size(k_quant->type, hparams.n_embd_k_gqa(il)),
                0
            );
    }

    // Create view similar to get_k but for quantized tensor
    return ggml_view_3d(ctx, k_quant,
                hparams.n_embd_head_k, hparams.n_head_kv(il), n_quantized,
                ggml_row_size(k_quant->type, hparams.n_embd_head_k),
                ggml_row_size(k_quant->type, hparams.n_embd_k_gqa(il)),
                0
            );
}

ggml_tensor * llama_kv_cache_mixed::get_v_quant(ggml_context * ctx, int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return nullptr;
    }

    const auto & layer = layers[it->second];
    auto * v_quant = layer.v_quant;

    // If no quantized tokens, return nullptr
    if (layer.quant_v_tokens == 0) {
        // NOTICE: This can only happen when the graph is pre-built
        return ggml_view_3d(ctx, v_quant,
                hparams.n_embd_head_v, hparams.n_head_kv(il), n_quantized,
                ggml_row_size(v_quant->type, hparams.n_embd_head_v),
                ggml_row_size(v_quant->type, hparams.n_embd_v_gqa(il)),
                0
            );
    }

    // Create view similar to get_v but for quantized tensor
    if (!v_trans) {
        return ggml_view_3d(ctx, v_quant,
                hparams.n_embd_head_v, hparams.n_head_kv(il), n_quantized,
                ggml_row_size(v_quant->type, hparams.n_embd_head_v),
                ggml_row_size(v_quant->type, hparams.n_embd_v_gqa(il)),
                0
            );
    }

    // For transposed V tensor
    return ggml_view_3d(ctx, v_quant,
            n_quantized, hparams.n_head_kv(il), hparams.n_embd_head_v,
            ggml_row_size(v_quant->type, v_quant->ne[1]*hparams.n_embd_head_v),
            ggml_row_size(v_quant->type, v_quant->ne[1]),
            0
        );
}

//> ===================================================================================================

ggml_tensor * llama_kv_cache_mixed::get_k_quant_ref(ggml_context * ctx, int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return nullptr;
    }
    const auto & layer = layers[it->second];

    return ggml_view_3d(ctx, layer.k_fp16,
            hparams.n_embd_head_k, hparams.n_head_kv(il), layer.mixed_k_head,
            ggml_row_size(layer.k_fp16->type, hparams.n_embd_head_k),
            ggml_row_size(layer.k_fp16->type, hparams.n_embd_k_gqa(il)),
            0
        );
}

ggml_tensor * llama_kv_cache_mixed::get_v_quant_ref(ggml_context * ctx, int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return nullptr;
    }
    const auto & layer = layers[it->second];

    ggml_tensor * v = layer.v_fp16;

    if (!v_trans) {
        return ggml_view_3d(ctx, v,
                hparams.n_embd_head_v, hparams.n_head_kv(il), layer.mixed_v_head,
                ggml_row_size(v->type, hparams.n_embd_head_v),
                ggml_row_size(v->type, hparams.n_embd_v_gqa(il)),
                0
            );
    }

    return ggml_view_3d(ctx, v,
            layer.mixed_v_head, hparams.n_head_kv(il), hparams.n_embd_head_v,
            ggml_row_size(v->type, v->ne[1]*hparams.n_embd_head_v),
            ggml_row_size(v->type, v->ne[1]),
            0
        );
}

ggml_tensor * llama_kv_cache_mixed::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return nullptr;
    }

    auto & layer = layers[it->second];

    ggml_tensor * k = layer.k_fp16;
    const int64_t n_tokens = k_cur->ne[2];

    layer.fp16_k_tokens += n_tokens;
    layer.total_tokens += n_tokens;         //> Add total tokens in cpy_k function.

    // TODO: You can use k_cur -> data == nullptr check if current is PREBUILD of graph.
    ggml_tensor * k_view = ggml_view_1d(ctx, k,
            n_tokens * hparams.n_embd_k_gqa(il),
            ggml_row_size(k->type, hparams.n_embd_k_gqa(il)) * head
        );

    return ggml_cpy(ctx, k_cur, k_view);
}

ggml_tensor * llama_kv_cache_mixed::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto & layer = layers[ikv];
    auto * v = layer.v_fp16;

    const int64_t n_tokens = v_cur->ne[2];

    layer.fp16_v_tokens += n_tokens;

    // TODO: You can use k_cur -> data == nullptr check if current is PREBUILD of graph.
    v_cur = ggml_reshape_2d(ctx, v_cur, hparams.n_embd_v_gqa(il), n_tokens);

    ggml_tensor * v_view = nullptr;

    // NOTE: `v_trans` = !flash_attn
    if (!v_trans) {
        v_view = ggml_view_1d(ctx, v,
                n_tokens*hparams.n_embd_v_gqa(il),
                ggml_row_size(v->type, hparams.n_embd_v_gqa(il))*head);
    } else {
        // note: the V cache is transposed when not using flash attention
        v_view = ggml_view_2d(ctx, v, n_tokens, hparams.n_embd_v_gqa(il),
                (v->ne[1])*ggml_element_size(v),
                head * ggml_element_size(v));
        v_cur = ggml_transpose(ctx, v_cur);
    }

    return ggml_cpy(ctx, v_cur, v_view);
}

//> ===================================================================================================
//> Following are the k_quant/v_quant functions for mixed kv cache.
//> ===================================================================================================
ggml_tensor * llama_kv_cache_mixed::k_quant(ggml_context * ctx, int32_t il) const {
    // CRITICAL FIX: Use proper layer mapping instead of direct indexing
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        LLAMA_LOG_ERROR("[mixed-kv] ERROR: Layer %d not found in map\n", il);
        return nullptr;
    }

    // Memory Layout Visualization:
    //
    // K FP16 Buffer (Before Quantization):
    // ┌─────────────────────────────────────────────┐
    // │                 FP16 Tokens                 │
    // ├─────────────────────┬───────────────────────┤
    // │  Older Tokens       │  Newer Tokens         │
    // │  (To Quantize)      │  (Keep in FP16)       │
    // ├─────────────────────┼───────────────────────┤
    // │<─────── src_tokens ─┼── remaining tokens ──>│
    // └─────────────────────┴───────────────────────┘
    //                       ↑
    //                  used position
    //
    // Offset Calculation:
    // src_offset_tokens = used - quantization_threshold
    //
    // Example: If used=40, threshold=32
    // Then quantize tokens 8-39 (32 tokens total)
    // And keep tokens 40+ in FP16

    auto & layer = layers[it->second];
    auto * k = layer.k_fp16;

    const size_t src_offset_bytes = ggml_row_size(k->type, hparams.n_embd_k_gqa(il)) * layer.mixed_k_head;
    const size_t dst_offset_bytes = ggml_row_size(layer.k_quant->type, hparams.n_embd_k_gqa(il)) * layer.mixed_k_head;

    const size_t elements_to_quantize = config.quantization_threshold * hparams.n_embd_k_gqa(il);

    //> mixed_k_head = head - config.fp16_window_size;
    layer.mixed_k_head += ((head - layer.mixed_k_head) - config.fp16_window_size);  //> Update the mixed_k_head.

    ggml_tensor * k_need_quantize = ggml_view_1d(ctx, k,
            elements_to_quantize,
            src_offset_bytes
        );

    ggml_tensor * k_quantized = ggml_view_1d(ctx, layer.k_quant,
            elements_to_quantize,
            dst_offset_bytes
        );

    return ggml_cpy(ctx, k_need_quantize, k_quantized);
}

ggml_tensor * llama_kv_cache_mixed::v_quant(ggml_context * ctx, int32_t il) const {
    // CRITICAL FIX: Use proper layer mapping instead of direct indexing
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        LLAMA_LOG_ERROR("[mixed-kv] ERROR: Layer %d not found in map\n", il);
        return nullptr;
    }

    // Memory Layout Visualization:
    //
    // V FP16 Buffer (Before Quantization):
    // ┌─────────────────────────────────────────────┐
    // │                 FP16 Tokens                 │
    // ├─────────────────────┬───────────────────────┤
    // │  Older Tokens       │  Newer Tokens         │
    // │  (To Quantize)      │  (Keep in FP16)       │
    // ├─────────────────────┼───────────────────────┤
    // │<─────── src_tokens ─┼── remaining tokens ──>│
    // └─────────────────────┴───────────────────────┘
    //                       ↑
    //              mixed_head position
    //
    // Offset Calculation:
    // src_offset_tokens = used - quantization_threshold
    //
    // Example: If used=40, threshold=32
    // Then quantize tokens 8-39 (32 tokens total)
    // And keep tokens 40+ in FP16

    auto & layer = layers[it->second];
    auto * v = layer.v_fp16;

    const size_t src_offset_bytes = ggml_row_size(v->type, hparams.n_embd_v_gqa(il)) * layer.mixed_v_head;
    const size_t dst_offset_bytes = ggml_row_size(layer.v_quant->type, hparams.n_embd_v_gqa(il)) * layer.mixed_v_head;

    const size_t elements_to_quantize = config.quantization_threshold * hparams.n_embd_v_gqa(il);

    //> mixed_v_head = head - config.fp16_window_size;
    layer.mixed_v_head += ((head - layer.mixed_v_head) - config.fp16_window_size);  //> Update the mixed_v_head.

    ggml_tensor * v_need_quantize = ggml_view_1d(ctx, v,
            elements_to_quantize,
            src_offset_bytes
        );

    ggml_tensor * v_quantized = ggml_view_1d(ctx, layer.v_quant,
            elements_to_quantize,
            dst_offset_bytes
        );

    return ggml_cpy(ctx, v_need_quantize, v_quantized);
}

//> ===================================================================================================
//> Following are the micro-kernel of flashdecoding kernel.
//> ===================================================================================================

inline static void ggml_vec_mad_f16(const int n, ggml_fp16_t * GGML_RESTRICT y, const ggml_fp16_t * GGML_RESTRICT x, const float v) {
#if defined(GGML_SIMD)
    const int np = (n & ~(GGML_F16_STEP - 1));

    GGML_F16_VEC vx = GGML_F16_VEC_SET1(v);

    GGML_F16_VEC ax[GGML_F16_ARR];
    GGML_F16_VEC ay[GGML_F16_ARR];

    for (int i = 0; i < np; i += GGML_F16_STEP) {
        for (int j = 0; j < GGML_F16_ARR; j++) {
            ax[j] = GGML_F16_VEC_LOAD(x + i + j*GGML_F16_EPR, j);
            ay[j] = GGML_F16_VEC_LOAD(y + i + j*GGML_F16_EPR, j);
            ay[j] = GGML_F16_VEC_FMA(ay[j], ax[j], vx);

            GGML_F16_VEC_STORE(y + i + j*GGML_F16_EPR, ay, j);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(y[i]) + GGML_FP16_TO_FP32(x[i])*v);
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(y[i]) + ggml_fp16_to_fp32(x[i])*v);
    }
#endif
}

inline static void ggml_vec_mad_f32(const int n, float * GGML_RESTRICT y, const float * GGML_RESTRICT x, const float v) {
#if defined(GGML_SIMD)
    const int np = (n & ~(GGML_F32_STEP - 1));

    GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

    GGML_F32_VEC ax[GGML_F32_ARR];
    GGML_F32_VEC ay[GGML_F32_ARR];

    for (int i = 0; i < np; i += GGML_F32_STEP) {
        for (int j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = GGML_F32_VEC_LOAD(x + i + j*GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_FMA(ay[j], ax[j], vx);

            GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] += x[i]*v;
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] += x[i]*v;
    }
#endif
}

//inline static void ggml_vec_scale_f32(const int n, float * y, const float   v) { for (int i = 0; i < n; ++i) y[i] *= v;          }
inline static void ggml_vec_scale_f32(const int n, float * y, const float   v) {
#if defined(GGML_USE_ACCELERATE)
    vDSP_vsmul(y, 1, &v, y, 1, n);
#elif defined(GGML_SIMD)
    const int np = (n & ~(GGML_F32_STEP - 1));

    GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

    GGML_F32_VEC ay[GGML_F32_ARR];

    for (int i = 0; i < np; i += GGML_F32_STEP) {
        for (int j = 0; j < GGML_F32_ARR; j++) {
            ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_MUL(ay[j], vx);

            GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] *= v;
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] *= v;
    }
#endif
}

inline static void ggml_vec_scale_f16(const int n, ggml_fp16_t * y, const float v) {
#if defined(GGML_SIMD)
    const int np = (n & ~(GGML_F16_STEP - 1));

    GGML_F16_VEC vx = GGML_F16_VEC_SET1(v);

    GGML_F16_VEC ay[GGML_F16_ARR];

    for (int i = 0; i < np; i += GGML_F16_STEP) {
        for (int j = 0; j < GGML_F16_ARR; j++) {
            ay[j] = GGML_F16_VEC_LOAD(y + i + j*GGML_F16_EPR, j);
            ay[j] = GGML_F16_VEC_MUL(ay[j], vx);

            GGML_F16_VEC_STORE(y + i + j*GGML_F16_EPR, ay, j);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(y[i])*v);
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(y[i])*v);
    }
#endif
}

//> ===================================================================================================
//> Micro-kernel of flashdecoding kernel.
//> ===================================================================================================

static void flash_decoding_q_f32_kv_f32(
    float* dst,
    float* q_ptr,
    float* k_ptr,
    float* v_ptr,
    const int64_t head_dim,
    const int64_t kv_len
) {
    memset(dst, 0, head_dim * sizeof(float));

    for (int64_t kv_iter = 0; kv_iter < kv_len; ++kv_iter) {    
        float qk_ret = 0.0f;
        for (int64_t hd_iter = 0; hd_iter < head_dim; ++ hd_iter) {
            qk_ret += q_ptr[hd_iter] * k_ptr[kv_iter * head_dim + hd_iter];
        }

        ggml_vec_mad_f32(head_dim, dst, v_ptr, qk_ret);
    }
}

void ggml_compute_forward_flash_attn_ext_f32(
        ggml_tensor * dst,
        int ith,
        int nth,
        void* wdata,
        size_t wsize,
        void * userdatat) {
    
    ggml_tensor * q     = dst->src[0];
    ggml_tensor * k     = dst->src[1];
    ggml_tensor * v     = dst->src[2];
    ggml_tensor * mask  = dst->src[3];
    
    memset(wdata, 0, wsize);

    // LLAMA_LOG_DEBUG("->>>>>>>>>>>>>>> ith: %d, nth: %d.\n", ith, nth);

    GGML_ASSERT(0 <= ith && ith < nth);

    //> QKV must be F32.
    // GGML_ASSERT(q->type == GGML_TYPE_F32);
    // GGML_ASSERT(k->type == GGML_TYPE_F32);
    // GGML_ASSERT(v->type == GGML_TYPE_F32);
    // GGML_ASSERT(mask->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    const int64_t DK = nek0;     //> head_dim
    const int64_t DV = nev0;     //> head_dim
    const int64_t N  = neq1;     //> q_len

    GGML_ASSERT(ne0 == DV);      //> dst -> ne[0] == head_dim
    GGML_ASSERT(ne1 == neq2);    //> dst -> ne[1] == n_heads  
    GGML_ASSERT(ne2 == N);       //> dst -> ne[2] == q_len

    // input tensor rows must be contiguous
    //> QKV cannot do transpose.
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbv0 == ggml_type_size(v->type));

    //> V donot transpose before.
    GGML_ASSERT(neq0 == DK);     //> q -> ne[0] == head_dim
    GGML_ASSERT(nek0 == DK);     //> k -> ne[0] == head_dim
    GGML_ASSERT(nev0 == DV);     //> v -> ne[0] == head_dim

    GGML_ASSERT(neq1 == N);      //> q -> ne[1] == q_len

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    const int64_t rk2 = neq2/nek2;     //> n_q_head / n_kv_head
    const int64_t rk3 = neq3/nek3;     //> n_q_batch / n_kv_batch

    const int64_t rv2 = neq2/nev2;     //> n_q_head / n_v_head
    const int64_t rv3 = neq3/nev3;     //> n_q_batch / n_v_batch

    // parallelize by q rows using ggml_vec_dot_f32

    // total rows in q
    const int nr = neq1*neq2*neq3;     //> number of rows, one row is one head_dim.

    // NOTE: Parallelize by q rows.
    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // Use proper attention scale factor: 1/sqrt(head_dim)
    float scale         = 1.0f / sqrtf((float)DK);
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    // Try to read from op_params if available, otherwise use defaults above
    // Note: op_params is always available but may contain default values
    // memcpy(&scale,         (float *) dst->op_params + 0, sizeof(float));
    // memcpy(&max_bias,      (float *) dst->op_params + 1, sizeof(float));
    // memcpy(&logit_softcap, (float *) dst->op_params + 2, sizeof(float));
    
    // If scale is 0 or 1 (default), use computed scale
    if (scale == 0.0f || scale == 1.0f) {
        scale = 1.0f / sqrtf((float)DK);
    }

    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = neq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    ggml_type    const k_vec_dot_type      = ggml_get_type_traits_cpu(k->type)->vec_dot_type;
    ggml_from_float_t const q_to_vec_dot   = ggml_get_type_traits_cpu(k_vec_dot_type)->from_float;
    ggml_vec_dot_t    const kq_vec_dot     = ggml_get_type_traits_cpu(k->type)->vec_dot;
    ggml_to_float_t   const v_to_float     = ggml_get_type_traits(v->type)->to_float;

    GGML_ASSERT((                            q_to_vec_dot) && "fattn: unsupported K-type");
    GGML_ASSERT((v->type == GGML_TYPE_F32 || v_to_float  ) && "fattn: unsupported V-type");

    // Handle mask data type - can be F32 or F16
    const float * mp_f32 = NULL;
    const ggml_fp16_t * mp_f16 = NULL;

    // loop over n_batch and n_head
    for (int ir = ir0; ir < ir1; ++ir) {
        // q indices
        const int iq3 =  ir / (neq2*neq1);
        const int iq2 = (ir - iq3*neq2*neq1)/neq1;
        const int iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        const uint32_t h = iq2; // head index
        const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

        float S = 0.0f;      // sum
        float M = -INFINITY; // maximum KQ value

        float       * VKQ32 = (float       *) wdata + ith*(1*DK + 2*DV + CACHE_LINE_SIZE_F32); // FP32 VKQ accumulator
        float       * V32   =                 (VKQ32 + 1*DV); // (temporary) FP32 V buffer
        ggml_fp16_t * VKQ16 = (ggml_fp16_t *) (VKQ32 + 1*DV); // (temporary) FP16 VKQ accumulator
        ggml_fp16_t * Q_q   = (ggml_fp16_t *) (VKQ32 + 2*DV); // (temporary) buffer for Q converted to quantized/FP16

        if (v->type == GGML_TYPE_F16) {
            memset(VKQ16, 0, DV*sizeof(ggml_fp16_t));
        } else {
            memset(VKQ32, 0, DV*sizeof(float));
        }

        const ggml_fp16_t * mp = mask ? (ggml_fp16_t *)((char *) mask->data + iq1*mask->nb[1]) : NULL;

        // k indices
        const int ik3 = iq3 / rk3;
        const int ik2 = iq2 / rk2;

        // v indices
        const int iv3 = iq3 / rv3;
        const int iv2 = iq2 / rv2;

        const float * pq = (const float *) ((char *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3));
        q_to_vec_dot(pq, Q_q, DK);

        // online softmax / attention
        // loop over n_kv and n_head_kv
        // ref: https://arxiv.org/pdf/2112.05682.pdf
        for (int64_t ic = 0; ic < nek1; ++ic) {
            const float mv = mp ? slope*ggml_fp16_to_fp32(mp[ic]) : 0.0f;
            if (mv == -INFINITY) {
                continue;
            }

            float s; // KQ value

            //> k_data: [head_dim, kv_len, n_kv_head, n_kv_batch]
            const char * k_data = (const char *) k->data + ( ic*nbk1 + ik2*nbk2 + ik3*nbk3);
            kq_vec_dot(DK, &s, 0, k_data, 0, Q_q, 0, 1);

            s = s*scale; // scale KQ value

            if (logit_softcap != 0.0f) {
                s = logit_softcap*tanhf(s);
            }

            s += mv; // apply mask

            const float Mold = M;

            float ms = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
            float vs = 1.0f; // post-softmax KQ value, expf(s - M)

            const char * v_data = ((const char *) v->data + (ic*nbv1 + iv2*nbv2 + iv3*nbv3));

            if (v->type == GGML_TYPE_F16) {
                if (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f16(DV, VKQ16, ms);
                } else {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                //> VKQ16 = VKQ16 + v_data * expf(s - M)
                ggml_vec_mad_f16(DV, VKQ16, (const ggml_fp16_t *) v_data, vs);
            } else {
                if (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f32(DV, VKQ32, ms);
                } else {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                if (v_to_float) {
                    v_to_float(v_data, V32, DV);
                    ggml_vec_mad_f32(DV, VKQ32, V32, vs);
                } else {
                    // V is F32
                    ggml_vec_mad_f32(DV, VKQ32, (const float *) v_data, vs);
                }
            }

            S = S*ms + vs; // scale and increment sum with partial sum
        }

        if (v->type == GGML_TYPE_F16) {
            for (int64_t d = 0; d < DV; ++d) {
                VKQ32[d] = ggml_fp16_to_fp32(VKQ16[d]);
            }
        }

        // V /= S
        const float S_inv = 1.0f / S;
        ggml_vec_scale_f32(DV, VKQ32, S_inv);

        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        // original
        // memcpy((char *) dst->data + (i1*nb1 + i2*nb2 + i3*nb3), V, nev0*sizeof(float));

        // memset((char *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, 0, nb1);
        // permute(0, 2, 1, 3)
        memcpy((char *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32, nb1);
    }
    
    // 清理宏定义
    #undef ith
    #undef nth
}


/**
 * Flash-Decoding Style Attention Implementation for Mixed KV Cache
 *
 * This implements flash-decoding by splitting the KV sequence across threads,
 * rather than splitting query rows. Each thread processes a chunk of tokens
 * and computes partial attention with log-sum-exp tracking.
 *
 * Key differences from traditional flash attention:
 * - Parallelization across KV sequence dimension instead of query dimension
 * - Each thread computes partial attention for a chunk of KV tokens for ALL queries
 * - Thread 0 performs final log-sum-exp reduction across all chunks
 *
 * Workspace Layout per thread:
 * - chunk_output[N * n_heads * DV]: Attention output for this chunk, for all queries
 * - log_sum_exp[N * n_heads]: Log-sum-exp values for this chunk, for all queries
 * - temp_buffer[DV]: Temporary buffer for intermediate computations
 * - Q_quantized[DK]: Quantized query buffer
 *
 * @param dst Output tensor
 * @param ith Thread index
 * @param nth Total number of threads
 * @param wdata Pointer to workspace
 * @param wsize Size of workspace
 * @param userdata Unused (for compatibility with GGML custom operation interface)
 */
void ggml_custom_flash_attn_mixed_simple(
        ggml_tensor * dst,
        int ith,
        int nth,
        void* wdata,
        size_t wsize,
        void * userdata) {
    GGML_UNUSED(wsize);    // Mark as intentionally unused
    GGML_UNUSED(userdata); // Mark as intentionally unused

    if (!dst) {
        LLAMA_LOG_ERROR("[mixed-kv] ERROR: null dst tensor in custom flash attention\n");
        return;
    }

    llama_flash_attn_mixed_params * flashdecoding_params = (llama_flash_attn_mixed_params *) userdata;

    // LLAMA_LOG_DEBUG("[mixed-kv] Layer id of current call: %d\n", flashdecoding_params->layer_id);

    ggml_tensor * q     = dst->src[0];
    ggml_tensor * k     = dst->src[1];
    ggml_tensor * v     = dst->src[2];
    ggml_tensor * mask  = dst->src[3];
    // ggml_tensor * k_quant = dst->src[4];
    // ggml_tensor * v_quant = dst->src[5];

    if (!q || !k || !v) {
        LLAMA_LOG_ERROR("[mixed-kv] ERROR: null tensors in custom flash attention\n");
        return;
    }

    //> q:    [head_dim, q_len,  n_heads, n_batch]
    //> k:    [head_dim, kv_len, n_heads, n_batch]
    //> v:    [head_dim, kv_len, n_heads, n_batch]
    //> mask: [n_heads,  q_len,  kv_len,  n_batch]

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    const int64_t DK = nek0;      //> head_dim for keys
    const int64_t DV = nev0;      //> head_dim for values
    const int64_t SEQ_LEN  = neq1;      //> q_len
    const int64_t KV_LEN    = nek1; //> kv sequence length
    const int64_t N_KV_HEAD = nek2; //> number of kv heads
    const int64_t N_Q_HEADS   = neq2; //> number of query heads

    GGML_ASSERT(ne0 == DV);       //> dst -> ne[0] == head_dim
    GGML_ASSERT(ne1 == SEQ_LEN);        //> dst -> ne[1] == q_len
    GGML_ASSERT(ne2 == N_Q_HEADS);  //> dst -> ne[2] == N_Q_HEADS

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbv0 == ggml_type_size(v->type));

    GGML_ASSERT(neq0 == DK);     //> q -> ne[0] == head_dim
    GGML_ASSERT(nek0 == DK);     //> k -> ne[0] == head_dim
    GGML_ASSERT(nev0 == DV);     //> v -> ne[0] == head_dim

    GGML_ASSERT(neq1 == SEQ_LEN);      //> q -> ne[1] == q_len

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // Flash-decoding: split KV sequence across threads
    const int64_t kv_chunk_size = (KV_LEN + nth - 1) / nth;             //> split KV sequence into nth chunks
    const int64_t chunk_start = ith * kv_chunk_size;                    //> start of this thread's chunk
    const int64_t chunk_end = MIN(chunk_start + kv_chunk_size, KV_LEN); //> end of this thread's chunk
    const int64_t chunk_len = chunk_end - chunk_start;                  //> length of this thread's chunk

    // Workspace layout per thread:
    //> K_vec = DK, V_vec = DV, result = OUTPUT_SIZE
    const size_t OUTPUT_SIZE    = N_Q_HEADS * SEQ_LEN * DV;
    const size_t LOCAL_MAX_SIZE = N_Q_HEADS * SEQ_LEN;
    float * thread_workspace    = (float *) wdata + ith * (OUTPUT_SIZE + 2 * LOCAL_MAX_SIZE + 1 * DV + 1 * DK + 1 + CACHE_LINE_SIZE_F32);

    const int64_t rk2 = neq2 / nek2;     //> n_q_heads / n_kv_heads
    const int64_t rv2 = neq2 / nev2;     //> n_q_heads / n_kv_heads

    float * chunk_output    = thread_workspace;                                                                 // [N_Q_HEADS * SEQ_LEN * DV]
    float * local_max       = thread_workspace + OUTPUT_SIZE;                                                   // [N_Q_HEADS * SEQ_LEN]
    float * local_exp_sum   = thread_workspace + OUTPUT_SIZE + LOCAL_MAX_SIZE;                                  // [N_Q_HEADS * SEQ_LEN]
    float * temp_buffer     = thread_workspace + OUTPUT_SIZE + 2 * LOCAL_MAX_SIZE;                              // [DV]
    ggml_fp16_t * Q_q       = (ggml_fp16_t *)(thread_workspace + OUTPUT_SIZE + 2 * LOCAL_MAX_SIZE + 1 * DV );   // [DK]
    float * sync_buffer     = thread_workspace + OUTPUT_SIZE + 2 * LOCAL_MAX_SIZE + 1 * DV + 1 * DK;            // [1]

    // Initialize chunk outputs and log_sum_exp for all queries
    memset(chunk_output,   0,           OUTPUT_SIZE * sizeof(float));
    memset(local_exp_sum,  0,           LOCAL_MAX_SIZE * sizeof(float));  // FIX: Initialize exp_sum to 0
    memset(temp_buffer,    0,           DV * sizeof(float));
    memset(Q_q,            0,           DK * sizeof(ggml_fp16_t));
    memset(sync_buffer,    0,           sizeof(float));
    for (int64_t i = 0; i < LOCAL_MAX_SIZE; i++) {
        local_max[i] = -INFINITY;
    }

    // Flash attention parameters (use default values for now)
    const float scale           = 1.0f / sqrtf((float)DK);
    const float max_bias        = 0.0f;
    const float logit_softcap   = 0.0f;

    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(N_Q_HEADS));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // Handle quantization for K/V tensor
    ggml_type const k_vec_dot_type        = ggml_get_type_traits_cpu(k->type) -> vec_dot_type;
    ggml_from_float_t const q_to_vec_dot  = ggml_get_type_traits_cpu(k_vec_dot_type) -> from_float;
    ggml_vec_dot_t const kq_vec_dot       = ggml_get_type_traits_cpu(k->type) -> vec_dot;
    ggml_to_float_t const v_to_float      = ggml_get_type_traits(v->type) -> to_float;

    // Handle mask data type - can be F32 or F16
    const float * mp_f32 = NULL;
    const ggml_fp16_t * mp_f16 = NULL;
    if (mask) {
        if (mask->type == GGML_TYPE_F32) {
            mp_f32 = (const float *)mask->data;
        } else if (mask->type == GGML_TYPE_F16) {
            mp_f16 = (const ggml_fp16_t *)mask->data;
        }
    }

    // Process this chunk of KV tokens for this specific query
    for (int64_t kv_pos = chunk_start; kv_pos < chunk_end; ++ kv_pos) {
        for (int64_t kv_head = 0; kv_head < N_KV_HEAD; ++ kv_head) {
            const char * k_data = (const char *) ((char *) k->data + ( kv_pos * nbk1 + kv_head * nbk2));
            const char * v_data = (const char *) ((char *) v->data + ( kv_pos * nbv1 + kv_head * nbv2));

            GGML_ASSERT(k_data != nullptr);
            GGML_ASSERT(v_data != nullptr);

            const int64_t q_head_start = kv_head * rk2;       //> q_head_start = head / rk2 * rk2
            const int64_t q_head_end   = q_head_start + rk2;  //> q_head_end = q_head_start + rk2

            GGML_ASSERT(q_head_start >= 0);

            for (int64_t q_head = q_head_start; q_head < q_head_end; ++ q_head) {
                for (int64_t q_pos = 0; q_pos < SEQ_LEN; ++ q_pos) {
                    const int64_t output_offset = q_pos * N_Q_HEADS * DV + q_head * DV;
                    const int64_t local_max_idx = q_pos * N_Q_HEADS + q_head;
                    float * output_ptr = chunk_output + output_offset;

                    // NOTE: Q MUST be F32
                    // TODO: cache Q quant.
                    const float * pq = (const float *) ((char *) q->data + q_pos * nbq1 + q_head * nbq2);
                    q_to_vec_dot(pq, Q_q, DK);
                    float s = 0.0f; //> KQ value
                    kq_vec_dot(DK, &s, 0, k_data, 0, Q_q, 0, 1);

                    s = s * scale; // scale KQ value

                    // Compute exponential for softmax
                    float Mold = local_max[local_max_idx];

                    float ms = 1.0f;
                    float vs = 1.0f;

                    if (s > Mold) {
                        local_max[local_max_idx] = s;

                        if (Mold == -INFINITY) {
                            ms = 1.0f;
                        } else {
                            ms = expf(Mold - s);
                        }
                    } else {
                        vs = expf(s - Mold);  // FIX: Use original Mold, not updated local_max
                    }

                    // TODO: support F16 V
                    // GGML_ASSERT(v->type == GGML_TYPE_F32);

                    local_exp_sum[local_max_idx] = local_exp_sum[local_max_idx] * ms + vs;

                    if (ms != 1.0f) {
                        // NOTE: Multiply past sum by ms
                        ggml_vec_scale_f32(DV, (float *)output_ptr, ms);
                    }

                    // ggml_vec_mad_f32(DV, (float *)output_ptr, (const float *)v_data, vs);

                    if (v->type == GGML_TYPE_F32) {
                        // V is already F32, use directly
                        ggml_vec_mad_f32(DV, (float *)output_ptr, (const float *)v_data, vs);
                    } else if (v_to_float) {
                        // V is quantized or F16, convert to F32 first
                        v_to_float(v_data, temp_buffer, DV);
                        ggml_vec_mad_f32(DV, (float *)output_ptr, temp_buffer, vs);
                    } else {
                        // NOTICE: treat as F32 (this shouldn't happen)
                        LLAMA_LOG_WARN("[mixed-kv] WARNING: V is not F32 or F16, treating as F32\n");
                    }
                }
            }
        }
    } //> end of chunk

    //> Barrier-free synchronization: set sync_buffer[0] to 1
    sync_buffer[0] = 1;

    // =======================================================================================
    // BARRIER-FREE SYNCHRONIZATION: All threads must complete before thread 0 can reduce
    // We use a simple busy-wait pattern checking if all chunks have been computed
    // =======================================================================================

    // Thread 0 waits for all other threads and performs reduction
    if (ith == 0 && nth > 1) {
        LLAMA_LOG_DEBUG("[mixed-kv] Starting flash-decoding reduction across %d chunks for %ld queries\n", nth, N_Q_HEADS * SEQ_LEN);

        // Simple busy-wait for all threads to complete their chunk computation
        bool all_threads_ready = false;
        int wait_cycles = 0;
        const int max_wait_cycles = 1000000; // Prevent infinite wait

        // NOTICE: Sync points.
        while (!all_threads_ready && wait_cycles < max_wait_cycles) {
            all_threads_ready = true;
            for (int t = 1; t < nth; ++t) { // Start from 1 since thread 0 is us
                float * t_workspace = (float *) wdata + t * (OUTPUT_SIZE + 2 * LOCAL_MAX_SIZE + 1 * DV + 1 * DK + 1 + CACHE_LINE_SIZE_F32);

                // Check if this thread has completed by checking its sync_buffer
                float * t_sync_buffer = t_workspace + OUTPUT_SIZE + 2 * LOCAL_MAX_SIZE + 1 * DV + 1 * DK;

                // Thread is ready if it set sync_buffer[0] to 1
                if (t_sync_buffer[0] != 1.0f) {
                    all_threads_ready = false;
                    break;
                }
            }
            wait_cycles++;
        }

        if (wait_cycles >= max_wait_cycles) {
            LLAMA_LOG_WARN("[mixed-kv] WARNING: thread synchronization timeout, proceeding with reduction\n");
        }

        // Perform log-sum-exp reduction across all threads
        for (int64_t q_head = 0; q_head < N_Q_HEADS; ++q_head) {
            for (int64_t q_pos = 0; q_pos < SEQ_LEN; ++q_pos) {
                const int64_t output_offset = q_pos * N_Q_HEADS * DV + q_head * DV;
                const int64_t local_max_idx = q_pos * N_Q_HEADS + q_head;

                // Find global maximum across all threads for this query
                float global_max = -INFINITY;
                for (int t = 0; t < nth; ++t) {
                    float * t_workspace = (float *) wdata + t * (OUTPUT_SIZE + 2 * LOCAL_MAX_SIZE + 1 * DV + 1 * DK + 1 + CACHE_LINE_SIZE_F32);
                    float * t_local_max = t_workspace + OUTPUT_SIZE;

                    if (t_local_max[local_max_idx] > global_max) {
                        global_max = t_local_max[local_max_idx];
                    }
                }

                // If all threads had -INFINITY (no valid tokens), skip this query
                if (global_max == -INFINITY) {
                    // Zero out the output for this query
                    float * final_output = (float *) dst->data + output_offset;
                    memset(final_output, 0, DV * sizeof(float));
                    continue;
                }

                // Compute sum of exponentials with global max for numerical stability
                float global_sum = 0.0f;
                for (int t = 0; t < nth; ++t) {
                    float * t_workspace = (float *) wdata + t * (OUTPUT_SIZE + 2 * LOCAL_MAX_SIZE + 1 * DV + 1 * DK + 1 + CACHE_LINE_SIZE_F32);
                    float * t_local_max = t_workspace + OUTPUT_SIZE;
                    float * t_local_exp_sum = t_workspace + OUTPUT_SIZE + LOCAL_MAX_SIZE;

                    if (t_local_max[local_max_idx] != -INFINITY) {
                        // Use the actual exp_sum from the thread, adjusted for global max
                        const float exp_sum_adjustment = expf(t_local_max[local_max_idx] - global_max);
                        global_sum += t_local_exp_sum[local_max_idx] * exp_sum_adjustment;
                    }
                }

                // Normalize factor for final attention weights
                const float norm_factor = 1.0f / global_sum;

                // Combine weighted outputs from all threads
                float * final_output = (float *) dst->data + output_offset;
                memset(final_output, 0, DV * sizeof(float)); // Initialize to zero

                for (int t = 0; t < nth; ++t) {
                    float * t_workspace = (float *) wdata + t * (OUTPUT_SIZE + 2 * LOCAL_MAX_SIZE + 1 * DV + 1 * DK + 1 + CACHE_LINE_SIZE_F32);
                    float * t_chunk_output = t_workspace;
                    float * t_local_max = t_workspace + OUTPUT_SIZE;
                    float * t_local_exp_sum = t_workspace + OUTPUT_SIZE + LOCAL_MAX_SIZE;

                    if (t_local_max[local_max_idx] != -INFINITY) {
                        // FIXED: Correct multi-thread reduction formula
                        // final_output = sum(chunk_output_t * exp(local_max_t - global_max)) / global_sum
                        // Each thread contributes: chunk_output_t * exp(local_max_t - global_max)
                        const float max_adjustment = expf(t_local_max[local_max_idx] - global_max);
                        const float thread_weight = max_adjustment / global_sum;

                        // Add this thread's adjusted contribution
                        const float * thread_output = t_chunk_output + output_offset;
                        ggml_vec_mad_f32(DV, final_output, thread_output, thread_weight);
                    }
                }

                LLAMA_LOG_DEBUG("[mixed-kv] Reduced query (head=%ld, pos=%ld): global_max=%.6f, global_sum=%.6f, norm_factor=%.6f\n",
                               q_head, q_pos, global_max, global_sum, norm_factor);
            }
        }

        LLAMA_LOG_DEBUG("[mixed-kv] Flash-decoding reduction completed for %ld queries across %d threads\n",
                       N_Q_HEADS * SEQ_LEN, nth);

    } else if (nth == 1) {
        // Single-threaded execution: process entire KV sequence and write directly to destination
        LLAMA_LOG_DEBUG("[mixed-kv] Single-threaded flash-decoding execution for %ld queries\n", N_Q_HEADS * SEQ_LEN);

        // For single-threaded execution, normalize the accumulated outputs correctly
        float* thread0_workspace    = (float*)wdata;
        float* local_exp_sum        = thread0_workspace + OUTPUT_SIZE + LOCAL_MAX_SIZE;

        for (int64_t q_head = 0; q_head < N_Q_HEADS; ++q_head) {
            for (int64_t q_pos = 0; q_pos < SEQ_LEN; ++q_pos) {
                const int64_t output_offset = q_pos * N_Q_HEADS * DV + q_head * DV;
                const int64_t local_max_idx = q_pos * N_Q_HEADS + q_head;

                float * final_output = (float *) dst->data + output_offset;
                float * thread_output = thread0_workspace + output_offset;

                // Normalize by the sum of exponentials to get proper softmax weights
                if (local_exp_sum[local_max_idx] > 0.0f) {
                    const float norm_factor = 1.0f / local_exp_sum[local_max_idx];
                    for (int64_t d = 0; d < DV; ++d) {
                        final_output[d] = thread_output[d] * norm_factor;
                    }
                } else {
                    // If sum is 0, set output to 0
                    memset(final_output, 0, DV * sizeof(float));
                }
            }
        }
    }
}
