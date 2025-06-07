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

    // NOTE: `v_trans` = !flash_attn

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

    /*
     * KV Cache Cells Architecture Overview:
     *
     * cells 是 Mixed KV Cache 的核心管理数据结构，用于跟踪每个缓存槽的状态
     * 它是一个固定大小的数组，每个元素代表一个cache slot
     *
     * ┌─────────────────────────────────────────────────────────┐
     * │                    KV Cache Layout                      │
     * │                                                         │
     * │  cells[0]  cells[1]  cells[2]  ...  cells[kv_size-1]   │
     * │  ┌─────┐   ┌─────┐   ┌─────┐         ┌─────┐           │
     * │  │slot │   │slot │   │slot │   ...   │slot │           │
     * │  │  0  │   │  1  │   │  2  │         │ N-1 │           │
     * │  └─────┘   └─────┘   └─────┘         └─────┘           │
     * │     ↑         ↑         ↑               ↑              │
     * │   pos=-1    pos=0     pos=1          pos=N-2           │
     * │  (empty)   (token)   (token)        (token)            │
     * │             seq=1     seq=1          seq=2             │
     * └─────────────────────────────────────────────────────────┘
     *
     * 每个 cell 包含：
     * - pos: token 在序列中的位置 (-1 表示空闲槽位)
     * - seq_id: 该 token 属于哪些序列的集合 (支持多序列共享同一token)
     * - delta: 用于位置偏移计算的累积值 (用于 RoPE、K-shift 等操作)
     *
     * Cache 管理状态：
     * - head: 下一个分配的起始位置指针 (优化查找效率)
     * - used: 当前已使用的slot数量
     * - size: 总的cache容量 (= kv_size)
     */
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
        layer.k_fp16    = ggml_new_tensor_2d(ctx, config.hot_type_k, n_embd_k_gqa, kv_size);
        layer.v_fp16    = ggml_new_tensor_2d(ctx, config.hot_type_v, n_embd_v_gqa, kv_size);

        // Create quantized tensors (for future use, but not used during alignment testing)
        layer.k_quant   = ggml_new_tensor_2d(ctx, config.cold_type_k, n_embd_k_gqa, kv_size);
        layer.v_quant   = ggml_new_tensor_2d(ctx, config.cold_type_v, n_embd_v_gqa, kv_size);

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
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

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

    // Note: ctxs and bufs will be automatically cleaned up by their smart pointer destructors
    // in the correct order (bufs first, then ctxs)
}

void llama_kv_cache_mixed::clear() {
    LLAMA_LOG_DEBUG("[mixed-kv] clearing cache (size=%u, used=%u)\n", size, used);

    /*
     * cells清空操作 - 重置所有缓存槽状态到初始空闲状态：
     *
     * cells 数组中的每个元素都代表一个 cache slot，清空操作将：
     * 1. 将所有 pos 设为 -1 (表示空闲)
     * 2. 清空所有 seq_id 集合
     * 3. 重置管理计数器 (head=0, used=0)
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
        cells[i].pos = -1;        // 标记为空闲槽位
        cells[i].seq_id.clear();  // 清空序列ID集合
    }

    head = 0;
    used = 0;

    // Clear all layers and count tokens for debug output
    uint32_t total_fp16_k_tokens = 0;
    uint32_t total_fp16_v_tokens = 0;
    for (auto & layer : layers) {
        total_fp16_k_tokens += layer.fp16_k_tokens;
        total_fp16_v_tokens += layer.fp16_v_tokens;
        layer.quant_k_tokens = 0;
        layer.quant_v_tokens = 0;
        layer.fp16_k_tokens = 0;
        layer.fp16_v_tokens = 0;
    }

    for (auto & buf : bufs) {
        ggml_backend_buffer_clear(buf.get(), 0);
    }

    LLAMA_LOG_DEBUG("[mixed-kv] cache cleared successfully (cleared %u K tokens, %u V tokens)\n",
                   total_fp16_k_tokens, total_fp16_v_tokens);
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
     * cells序列移除操作 - 从指定位置范围移除序列tokens：
     *
     * 遍历所有cells，检查每个cell的位置是否在移除范围[p0, p1)内
     * 如果在范围内且包含目标序列，则从该cell的seq_id集合中移除该序列
     * 如果移除后cell变为空闲（seq_id集合为空），则释放该slot
     *
     * 例如：seq_rm(seq_id=1, p0=1, p1=3) - 移除序列1在位置1-2的tokens
     *
     * Before seq_rm():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:2│pos:3│pos:4│
     * │seq:1│seq:1│seq:1│seq:2│seq:1│  <- 需要移除位置1-2的seq:1
     * └─────┴─────┴─────┴─────┴─────┘
     *
     * After seq_rm():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:-│pos:-│pos:3│pos:4│
     * │seq:1│empty│empty│seq:2│seq:1│  <- pos:1,2被清空释放
     * └─────┴─────┴─────┴─────┴─────┘
     *         ↑     ↑
     *      new_head 候选位置 (用于优化后续分配)
     */
    for (uint32_t i = 0; i < size; ++i) {
        // 检查该cell的位置是否在移除范围内
        if (cells[i].pos >= p0 && cells[i].pos < p1) {
            if (seq_id < 0) {
                // seq_id < 0 表示移除所有序列
                cells[i].seq_id.clear();
            } else if (cells[i].has_seq_id(seq_id)) {
                // 只移除指定的序列ID
                cells[i].seq_id.erase(seq_id);
            } else {
                // 该cell不包含目标序列，跳过
                continue;
            }

            // 如果cell变为空（没有任何序列使用），则释放该槽位
            if (cells[i].is_empty()) {
                // 更新已使用槽位计数
                if (cells[i].pos >= 0) {
                    used--;
                }

                cells[i].pos = -1;  // 标记为空闲

                // 记录第一个空闲槽位，用于优化后续分配
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
     * cells序列复制操作 - 将源序列的tokens复制给目标序列：
     *
     * 遍历所有cells，找到属于源序列且在指定位置范围内的cells
     * 将目标序列ID添加到这些cells的seq_id集合中
     * 这实现了多序列共享同一token的功能（例如用于beam search）
     *
     * 例如：seq_cp(seq_src=1, seq_dst=3, p0=1, p1=3) - 复制序列1给序列3
     *
     * Before seq_cp():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:2│pos:3│pos:4│
     * │seq:1│seq:1│seq:1│seq:2│seq:1│  <- 复制seq:1的pos:1-2给seq:3
     * └─────┴─────┴─────┴─────┴─────┘
     *
     * After seq_cp():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:2│pos:3│pos:4│
     * │seq:1│{1,3}│{1,3}│seq:2│seq:1│  <- pos:1,2现在同时属于seq:1和seq:3
     * └─────┴─────┴─────┴─────┴─────┘
     *         ↑     ↑
     *      共享tokens (多序列引用同一cache slot)
     */
    for (uint32_t i = 0; i < size; ++i) {
        // 检查该cell是否属于源序列且在指定位置范围内
        if (cells[i].has_seq_id(seq_id_src) && cells[i].pos >= p0 && cells[i].pos < p1) {
            // 将目标序列ID添加到该cell（多序列共享同一token）
            cells[i].seq_id.insert(seq_id_dst);
        }
    }
}

void llama_kv_cache_mixed::seq_keep(llama_seq_id seq_id) {
    uint32_t new_head = size;

    /*
     * cells序列保留操作 - 只保留指定序列，清除其他所有序列：
     *
     * 遍历所有cells，对于不属于目标序列的cells完全清除，
     * 对于属于目标序列的cells，清理多序列状态只保留目标序列
     * 这通常用于切换当前活跃序列，清理不需要的分支
     *
     * 例如：seq_keep(seq_id=2) - 只保留序列2，清除其他所有序列
     *
     * Before seq_keep():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:2│pos:3│pos:4│
     * │seq:1│{1,3}│seq:2│{1,2}│seq:1│  <- 只保留seq:2
     * └─────┴─────┴─────┴─────┴─────┘
     *
     * After seq_keep():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:-│pos:-│pos:2│pos:3│pos:-│
     * │empty│empty│seq:2│seq:2│empty│  <- 只有seq:2的cells被保留
     * └─────┴─────┴─────┴─────┴─────┘
     *   ↑     ↑               ↑
     * new_head候选位置 (用于后续优化分配)
     */
    for (uint32_t i = 0; i < size; ++i) {
        // 检查该cell是否不属于要保留的序列
        if (!cells[i].has_seq_id(seq_id)) {
            // 该cell不属于目标序列，清除它
            if (cells[i].pos >= 0) {
                used--;  // 减少已使用计数
            }

            cells[i].pos = -1;           // 标记为空闲
            cells[i].seq_id.clear();     // 清空序列ID

            // 记录第一个空闲位置
            if (new_head == size){
                new_head = i;
            }
        } else {
            // 该cell属于目标序列，清理它的多序列状态，只保留目标序列
            cells[i].seq_id.clear();         // 清空所有序列ID
            cells[i].seq_id.insert(seq_id);  // 只插入目标序列ID
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
     * cells序列位置偏移操作 - 将指定序列的位置向前或向后移动：
     *
     * 遍历所有cells，找到属于目标序列且在指定位置范围内的cells
     * 更新它们的pos和delta值，如果位置变为负数则清除该cell
     * 这用于实现序列的位置偏移（如插入/删除tokens、位置编码调整等）
     *
     * 例如：seq_add(seq_id=1, p0=2, p1=4, delta=2) - 序列1的位置2-3向前移动2位
     *
     * Before seq_add():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:2│pos:3│pos:4│
     * │seq:1│seq:1│seq:1│seq:1│seq:2│  <- seq:1在pos:2-3的tokens需要+2偏移
     * └─────┴─────┴─────┴─────┴─────┘
     *               ↑─── 范围[2,4) ──↑
     *
     * After seq_add():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:4│pos:5│pos:4│
     * │seq:1│seq:1│seq:1│seq:1│seq:2│  <- pos:2→4, pos:3→5, delta累积
     * └─────┴─────┴─────┴─────┴─────┘
     *
     * 特殊情况 - 如果delta为负且使pos变为负数，则清除该cell：
     * 例如delta=-3时，pos:2-3会变成-1,0，负数位置的cell被清除释放
     */
    for (uint32_t i = 0; i < size; ++i) {
        // 检查该cell是否属于目标序列且在指定位置范围内
        if (cells[i].has_seq_id(seq_id) && cells[i].pos >= p0 && cells[i].pos < p1) {
            has_shift = true;  // 标记发生了位置偏移

            cells[i].pos   += delta;  // 更新token位置
            cells[i].delta += delta;  // 累积偏移量（用于RoPE等）

            // 如果位置变为负数，说明token被移出有效范围，需要清除
            if (cells[i].pos < 0) {
                if (!cells[i].is_empty()) {
                    used--;  // 减少已使用计数
                }
                cells[i].pos = -1;           // 标记为空闲
                cells[i].seq_id.clear();     // 清空序列ID
                if (new_head == size) {
                    new_head = i;            // 记录空闲位置
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
     * cells序列位置除法操作 - 将指定序列的位置按比例缩小：
     *
     * 遍历所有cells，找到属于目标序列且在指定位置范围内的cells
     * 将它们的位置除以除数d，并更新delta累积偏移量
     * 这用于实现位置的比例缩放（如attention window缩放、位置压缩等）
     *
     * 例如：seq_div(seq_id=1, p0=4, p1=8, d=2) - 序列1位置4-7除以2
     *
     * Before seq_div():
     * ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:4│pos:5│pos:6│pos:7│pos:8│pos:9│
     * │seq:1│seq:1│seq:1│seq:1│seq:1│seq:1│seq:2│seq:2│
     * └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
     *               ↑─ 范围[4,8) ─↑   <- 这些位置需要除以2
     *
     * After seq_div():
     * ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:2│pos:2│pos:3│pos:3│pos:8│pos:9│
     * │seq:1│seq:1│seq:1│seq:1│seq:1│seq:1│seq:2│seq:2│
     * └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
     *               ↑─ 4/2=2  5/2=2  6/2=3  7/2=3 ─↑
     *                  (delta同时记录位置变化量)
     */
    for (uint32_t i = 0; i < size; ++i) {
        // 检查该cell是否属于目标序列且在指定位置范围内
        if (cells[i].has_seq_id(seq_id) && cells[i].pos >= p0 && cells[i].pos < p1) {
            has_shift = true;  // 标记发生了位置变化

            {
                llama_pos p_old = cells[i].pos;    // 保存原始位置
                cells[i].pos   /= d;               // 位置除法缩放
                cells[i].delta += cells[i].pos - p_old;  // 计算并累积偏移量
            }
        }
    }
}

llama_pos llama_kv_cache_mixed::seq_pos_min(llama_seq_id seq_id) const {
    llama_pos result = std::numeric_limits<llama_pos>::max();

    /*
     * 查找指定序列的最小位置：
     *
     * 例如：查找seq_id=1的最小位置
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:5│pos:1│pos:3│pos:7│pos:2│
     * │seq:2│seq:1│seq:1│seq:2│seq:1│  <- seq:1的位置有1,3,2
     * └─────┴─────┴─────┴─────┴─────┘
     *
     * 返回 min(1,3,2) = 1
     */
    for (uint32_t i = 0; i < size; ++i) {
        // 检查该cell是否属于目标序列
        if (cells[i].has_seq_id(seq_id)) {
            result = std::min(result, cells[i].pos);  // 更新最小位置
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
     * 查找指定序列的最大位置：
     *
     * 例如：查找seq_id=1的最大位置
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:5│pos:1│pos:3│pos:7│pos:2│
     * │seq:2│seq:1│seq:1│seq:2│seq:1│  <- seq:1的位置有1,3,2
     * └─────┴─────┴─────┴─────┴─────┘
     *
     * 返回 max(1,3,2) = 3
     */
    for (uint32_t i = 0; i < size; ++i) {
        // 检查该cell是否属于目标序列
        if (cells[i].has_seq_id(seq_id)) {
            result = std::max(result, cells[i].pos);  // 更新最大位置
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
             * 恢复单个cell的状态，并正确维护used计数：
             *
             * Before restore:     After restore:
             * ┌─────┐             ┌─────┐
             * │pos:2│   <---      │pos:5│  (从recovery中恢复)
             * │seq:1│             │seq:2│
             * └─────┘             └─────┘
             * used++/used--根据cell状态变化进行调整
             */
            const bool is_empty0 = cells[id].is_empty();  // 当前cell是否为空
            const bool is_empty1 = cell.is_empty();       // 恢复后cell是否为空

            // 根据状态变化调整used计数
            if (!is_empty0 && is_empty1) {
                used--;  // 从占用变为空闲
            } else if (is_empty0 && !is_empty1) {
                used++;  // 从空闲变为占用
            }

            // 安全地恢复cell状态
            cells[id].pos = cell.pos;         // 恢复位置
            cells[id].delta = cell.delta;     // 恢复偏移量
            cells[id].seq_id = cell.seq_id;   // 恢复序列ID集合

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
            has_shift = false;  // 重置偏移标志

            /*
             * 清除所有cells的delta偏移量：
             *
             * After K-shift operation:
             * ┌─────┬─────┬─────┬─────┐
             * │pos:2│pos:3│pos:4│pos:5│
             * │Δ:+2 │Δ:+2 │Δ:+2 │Δ:+2 │  <- 清除这些累积偏移
             * └─────┴─────┴─────┴─────┘
             *
             * After delta reset:
             * ┌─────┬─────┬─────┬─────┐
             * │pos:2│pos:3│pos:4│pos:5│
             * │Δ: 0 │Δ: 0 │Δ: 0 │Δ: 0 │  <- 偏移量被重置
             * └─────┴─────┴─────┴─────┘
             */
            for (uint32_t i = 0; i < size; ++i) {
                cells[i].delta = 0;  // 重置偏移量累积
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

        /*
         * 检查从head开始的连续n_tokens个槽位是否都空闲：
         *
         * 例如：需要分配3个连续槽位
         *
         * Case 1 - 成功找到：
         *     head=2, n_tokens=3
         * ┌─────┬─────┬─────┬─────┬─────┬─────┐
         * │pos:0│pos:1│pos:-│pos:-│pos:-│pos:5│
         * │seq:1│seq:1│empty│empty│empty│seq:2│
         * └─────┴─────┴─────┴─────┴─────┴─────┘
         *               ↑─── 连续3个空闲槽位 ─↑
         *
         * Case 2 - 需要继续查找：
         *     head=2, n_tokens=3
         * ┌─────┬─────┬─────┬─────┬─────┬─────┐
         * │pos:0│pos:1│pos:-│pos:3│pos:-│pos:5│
         * │seq:1│seq:1│empty│seq:1│empty│seq:2│
         * └─────┴─────┴─────┴─────┴─────┴─────┘
         *               ↑     ↑ <- 第2个槽位被占用，从pos:4重新开始
         */
        bool found = true;
        for (uint32_t i = 0; i < n_tokens; i++) {
            // 检查第i个槽位是否被占用
            if (cells[head + i].pos >= 0) {
                found = false;        // 找到占用的槽位，当前位置不可用
                head += i + 1;        // 移动head到下一个可能的位置
                n_tested += i + 1;    // 更新已测试的槽位数
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

    /*
     * 分配连续的n_tokens个槽位并设置它们的状态：
     *
     * 例如：分配3个tokens，从head=5开始
     *
     * Before allocation:
     * ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:2│pos:3│pos:4│pos:-│pos:-│pos:-│
     * │seq:1│seq:1│seq:1│seq:1│seq:1│empty│empty│empty│
     * └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
     *                                   ↑head=5
     *
     * Recovery backup: 先备份原始状态到recovery
     * ┌─recovery.cells[5]─┐ ┌─recovery.cells[6]─┐ ┌─recovery.cells[7]─┐
     * │ pos: -1, seq: {}  │ │ pos: -1, seq: {}  │ │ pos: -1, seq: {}  │
     * └───────────────────┘ └───────────────────┘ └───────────────────┘
     *
     * After allocation:
     * ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:2│pos:3│pos:4│pos:5│pos:6│pos:7│
     * │seq:1│seq:1│seq:1│seq:1│seq:1│seq:2│seq:2│seq:2│  <- 新分配的tokens
     * └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
     *                                   ↑─── 新tokens ─↑
     */
    for (uint32_t i = 0; i < n_tokens; ++i) {
        // 计算当前token对应的cell索引
        const uint32_t cell_idx = head + i;

        // 边界检查：确保cell索引在有效范围内
        if (cell_idx >= size) {
            LLAMA_LOG_ERROR("[mixed-kv] ERROR: cell index %u out of bounds (size=%u)\n", cell_idx, size);
            return false;
        }

        // 检查是否已经为该cell保存了恢复信息
        // 如果没有，需要保存当前状态以便后续可能的回滚操作
        if (recovery.cells.find(cell_idx) == recovery.cells.end()) {
            try {
                // 创建cell状态的安全备份
                kv_cell backup_cell;
                backup_cell.pos = cells[cell_idx].pos;         // 备份位置
                backup_cell.delta = cells[cell_idx].delta;     // 备份偏移量
                backup_cell.seq_id = cells[cell_idx].seq_id;   // 安全复制序列ID集合

                recovery.cells[cell_idx] = std::move(backup_cell);

                LLAMA_LOG_DEBUG("[mixed-kv] stored recovery info for cell %u (pos=%d, seq_ids=%zu)\n",
                               cell_idx, backup_cell.pos, backup_cell.seq_id.size());
            } catch (const std::exception& e) {
                LLAMA_LOG_ERROR("[mixed-kv] ERROR: Failed to store recovery info for cell %u: %s\n", cell_idx, e.what());
                return false;
            }
        }

        // 设置新token的位置
        cells[cell_idx].pos = ubatch.pos[i];

        // 将该token关联到相应的序列
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; j++) {
            cells[cell_idx].seq_id.insert(ubatch.seq_id[i][j]);
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
// void llama_kv_cache_mixed::quantize_oldest_tokens(int32_t il, uint32_t tokens_to_quantize) {
//     GGML_UNUSED(il);
//     GGML_UNUSED(tokens_to_quantize);
//     // TODO: Implement
// }

// // Legacy method - now calls the new FIFO-based quantization
// void llama_kv_cache_mixed::quantize_tokens(int32_t il) {
//     GGML_UNUSED(il);
// }

/*
 * KQ Mask (Attention Mask) 构建函数
 *
 * 目的:
 *   为每个查询（query）token 构建一个 mask，决定它可以与哪些键（key）token 进行交互。
 *   这个 mask 是 attention 机制的核心，用于防止 token "看到" 不该看的信息。
 *
 * Mask 构建规则:
 * 1. 序列隔离 (Sequence Isolation):
 *    一个 token 只能 attend 到属于同一个序列的 key-value pairs。
 *    例如，序列A的token不能 attend 到序列B的token。
 *
 * 2. 因果关系 (Causality):
 *    在自回归生成中，一个 token 只能 attend 到它自己以及它之前的 tokens。
 *    这可以防止模型 "看到未来"，保证生成过程的正确性。
 *
 * 3. ALiBi (Attention with Linear Biases):
 *    如果使用 ALiBi，mask 的值会根据 query 和 key 的相对距离进行惩罚，
 *    距离越远，惩罚越大。
 *
 * 4. 填充处理 (Padding):
 *    对于批处理中因填充而产生的无效 token，其 attention score 会被完全屏蔽。
 *
 * Mask Tensor 示意图 (causal_attn = true):
 *
 *          k_pos=0  k_pos=1  k_pos=2  k_pos=3 (KV Cache)
 *          (seq=1)  (seq=1)  (seq=2)  (seq=1)
 *         +--------+--------+--------+--------+
 * q_pos=1 │   0    │   0    │  -inf  │  -inf  │  <- Query token (pos=1, seq=1)
 * (seq=1) │        │        │ (异构)  │ (未来) │
 *         +--------+--------+--------+--------+
 * q_pos=2 │  -inf  │  -inf  │   0    │  -inf  │  <- Query token (pos=2, seq=2)
 * (seq=2) │ (异构)  │ (异构) │         │ (未来) │
 *         +--------+--------+--------+--------+
 *
 * -  0:      允许 attention
 * - -inf:   禁止 attention (在 softmax 后会变为0)
 * - (异构):  key-value pair 属于不同序列，被 mask
 * - (未来):  key-value pair 在 query token 之后，在因果模型中被 mask
 */
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
                // 当前查询 token 在序列中的位置
                const llama_pos p1 = ubatch->pos[s*n_seq_tokens + j];

                // 遍历所有 KV cache 中的 token
                for (int i = 0; i < n_kv; ++i) {
                    // 当前键 token 在序列中的位置
                    const llama_pos p0 = cells[i].pos;

                    bool masked = false;

                    // 规则 1: 如果 key token 不属于当前 query token 的序列，则屏蔽
                    masked = masked || (!cells[i].has_seq_id(seq_id));

                    // 规则 2: 如果是因果 attention，且 key token 在 query token 之后（未来），则屏蔽
                    masked = masked || (causal_attn && p0 > p1);

                    // 注意：SWA (Sliding Window Attention) 的 masking 在此混合缓存中尚未实现
                    // masked = masked || (is_masked_swa(p0, p1));

                    float f = 0.0f;

                    if (masked) {
                        // 对于被屏蔽的 token，将其 attention score 设置为负无穷
                        f = -INFINITY;
                    } else if (hparams.use_alibi) {
                        // 规则 3: 如果使用 ALiBi，根据 query 和 key 的距离计算惩罚项
                        f = -std::abs(p0 - p1);
                    }

                    // 将计算出的 mask 值写入目标张量
                    data[h*(n_kv*n_tokens) + s*(n_kv*n_seq_tokens) + j*n_kv + i] = f;
                }
            }
        }

        // 规则 4: 屏蔽批处理中的填充 token
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
//> Following are the original get_k and get_v functions from llama.cpp
//> ===================================================================================================

bool llama_kv_cache_mixed::do_quant(int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return false;
    }

    const auto& layer = layers[it->second];

    // Check if we have enough FP16 tokens to trigger quantization
    bool should_quantize = config.enable_quantization &&
                        ( used != 0 && used % config.quantization_threshold == 0 ) &&
                        used >= config.quantization_threshold;

    return should_quantize;
}

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

    // Calculate total tokens available
    const uint32_t total_available_tokens = layer.get_total_cached_tokens();
    const uint32_t tokens_to_use = std::min(total_available_tokens, n);

    LLAMA_LOG_DEBUG("[mixed-kv] get_k layer %d: total_available=%u, n=%u, using=%u\n",
                   il, total_available_tokens, n, tokens_to_use);
    LLAMA_LOG_DEBUG("[mixed-kv]   - quant_k_tokens=%u, fp16_k_tokens=%u\n",
                   layer.quant_k_tokens, used);

    if (tokens_to_use == 0) {
        return nullptr;
    }

    // For now, use only FP16 tensor for simplicity and alignment testing
    // TODO: Implement merged view with quantized data after basic testing
    auto * k = layer.k_fp16;

    // Create view exactly like unified cache, but limit to actual available tokens
    return ggml_view_3d(ctx, k,
            hparams.n_embd_head_k, hparams.n_head_kv(il), tokens_to_use,
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

    // Calculate total tokens available
    const uint32_t total_available_tokens = layer.get_total_cached_tokens();
    const uint32_t tokens_to_use = std::min(total_available_tokens, n);

    LLAMA_LOG_DEBUG("[mixed-kv] get_v layer %d: total_available=%u, n=%u, using=%u\n",
                   il, total_available_tokens, n, tokens_to_use);

    if (tokens_to_use == 0) {
        return nullptr;
    }

    // For now, use only FP16 tensor for simplicity and alignment testing
    // TODO: Implement merged view with quantized data after basic testing
    auto * v = layer.v_fp16;

    // NOTE: v_trans is !flash_attn
    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        return ggml_view_3d(ctx, v,
                hparams.n_embd_head_v, hparams.n_head_kv(il), tokens_to_use,
                ggml_row_size(v->type, hparams.n_embd_head_v),    // v->nb[1]
                ggml_row_size(v->type, hparams.n_embd_v_gqa(il)), // v->nb[2]
                0);
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_3d(ctx, v,
            tokens_to_use, hparams.n_head_kv(il), hparams.n_embd_head_v,
            ggml_row_size(v->type, v->ne[1]*hparams.n_embd_head_v), // v->nb[1]
            ggml_row_size(v->type, v->ne[1]),                       // v->nb[2]
            0);
}

/*
 * Methods for getting quantized K and V tensors
 *
 * Following same pattern as get_k/get_v but for quantized tensors
 */
ggml_tensor * llama_kv_cache_mixed::get_k_quant(ggml_context * ctx, int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return nullptr;
    }

    const auto & layer = layers[it->second];

    // If no quantized tokens, return nullptr
    if (layer.quant_k_tokens == 0) {
        return nullptr;
    }

    auto * k_quant = layer.k_quant;

    if (il == 0) {
        LLAMA_LOG_DEBUG("[mixed-kv] offset: %ld\n", ggml_row_size(k_quant->type, hparams.n_embd_k_gqa(il)) * (layer.quant_k_tokens - config.quantization_threshold));
        LLAMA_LOG_DEBUG("[mixed-kv] hparams.n_embd_head_k: %d\n", hparams.n_embd_head_k);
        LLAMA_LOG_DEBUG("[mixed-kv] hparams.n_head_kv(il): %d\n", hparams.n_head_kv(il));
        LLAMA_LOG_DEBUG("[mixed-kv] config.quantization_threshold: %d\n", config.quantization_threshold);
        LLAMA_LOG_DEBUG("[mixed-kv] layer.quant_k_tokens: %d\n", layer.quant_k_tokens);
    }

    // Create view similar to get_k but for quantized tensor
    return ggml_view_3d(ctx, k_quant,
                hparams.n_embd_head_k, hparams.n_head_kv(il), config.quantization_threshold,
                ggml_row_size(k_quant->type, hparams.n_embd_head_k),
                ggml_row_size(k_quant->type, hparams.n_embd_k_gqa(il)),
                ggml_row_size(k_quant->type, hparams.n_embd_k_gqa(il)) * (layer.quant_k_tokens )
            );
}

ggml_tensor * llama_kv_cache_mixed::get_v_quant(ggml_context * ctx, int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return nullptr;
    }

    const auto & layer = layers[it->second];

    // If no quantized tokens, return nullptr
    if (layer.quant_v_tokens == 0) {
        return nullptr;
    }

    auto * v_quant = layer.v_quant;

    if (il == 0) {
        LLAMA_LOG_DEBUG("[mixed-kv] offset: %ld\n", ggml_row_size(v_quant->type, hparams.n_embd_v_gqa(il)) * (layer.quant_v_tokens - config.quantization_threshold));
        LLAMA_LOG_DEBUG("[mixed-kv] hparams.n_embd_head_v: %d\n", hparams.n_embd_head_v);
        LLAMA_LOG_DEBUG("[mixed-kv] hparams.n_head_kv(il): %d\n", hparams.n_head_kv(il));
        LLAMA_LOG_DEBUG("[mixed-kv] config.quantization_threshold: %d\n", config.quantization_threshold);
        LLAMA_LOG_DEBUG("[mixed-kv] layer.quant_v_tokens: %d\n", layer.quant_v_tokens);
    }

    // NOTE: v_trans is !flash_attn
    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        return ggml_view_3d(ctx, v_quant,
                hparams.n_embd_head_v, hparams.n_head_kv(il), config.quantization_threshold,
                ggml_row_size(v_quant->type, hparams.n_embd_head_v),
                ggml_row_size(v_quant->type, hparams.n_embd_v_gqa(il)),
                ggml_row_size(v_quant->type, hparams.n_embd_v_gqa(il)) * (layer.quant_v_tokens)
            );
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_3d(ctx, v_quant,
            config.quantization_threshold, hparams.n_head_kv(il), hparams.n_embd_head_v,
            ggml_row_size(v_quant->type, v_quant->ne[1]*hparams.n_embd_head_v),
            ggml_row_size(v_quant->type, v_quant->ne[1]),
            ggml_row_size(v_quant->type, hparams.n_embd_v_gqa(il)) * (layer.quant_v_tokens)
        );
}

ggml_tensor * llama_kv_cache_mixed::k_quant(ggml_context * ctx, int32_t il) const {
    // CRITICAL FIX: Use proper layer mapping instead of direct indexing
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        LLAMA_LOG_ERROR("[mixed-kv] ERROR: Layer %d not found in map\n", il);
        return nullptr;
    }

    auto & layer = layers[it->second];
    auto * k = layer.k_fp16;

    // DEFENSIVE FIX: Validate we have enough tokens to quantize
    if (used < config.quantization_threshold) {
        LLAMA_LOG_ERROR("[mixed-kv] ERROR: Not enough tokens to quantize (used=%u, threshold=%u)\n",
                       used, config.quantization_threshold);
        return nullptr;
    }

    LLAMA_LOG_DEBUG("[mixed-kv] quantizing %u K tokens from layer %d (used=%u)\n",
                   config.quantization_threshold, il, used);
    // CRITICAL FIX: Calculate source offset safely
    //
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

    const size_t src_offset_bytes = ggml_row_size(k->type, hparams.n_embd_k_gqa(il)) * (used - config.quantization_threshold);
    const size_t elements_to_quantize = config.quantization_threshold * hparams.n_embd_k_gqa(il);

    // DEFENSIVE FIX: Bounds checking for source tensor
    const size_t k_total_bytes = ggml_nbytes(k);
    const size_t required_bytes = src_offset_bytes + ggml_row_size(k->type, hparams.n_embd_k_gqa(il)) * config.quantization_threshold;
    if (required_bytes > k_total_bytes) {
        LLAMA_LOG_ERROR("[mixed-kv] ERROR: K quantization source out of bounds (need %zu, have %zu)\n",
                       required_bytes, k_total_bytes);
        return nullptr;
    }

    // CRITICAL FIX: Use correct type for destination tensor view
    const size_t dst_offset_bytes = ggml_row_size(layer.k_quant->type, hparams.n_embd_k_gqa(il)) * layer.quant_k_tokens;
    const size_t k_quant_total_bytes = ggml_nbytes(layer.k_quant);
    const size_t dst_required_bytes = dst_offset_bytes + ggml_row_size(layer.k_quant->type, hparams.n_embd_k_gqa(il)) * config.quantization_threshold;

    if (dst_required_bytes > k_quant_total_bytes) {
        LLAMA_LOG_ERROR("[mixed-kv] ERROR: K quantization destination out of bounds (need %zu, have %zu)\n",
                       dst_required_bytes, k_quant_total_bytes);
        return nullptr;
    }

    // Create views with proper bounds checking
    ggml_tensor * k_need_quantize = ggml_view_1d(ctx, k,
            elements_to_quantize,
            src_offset_bytes
        );

    ggml_tensor * k_quantized = ggml_view_1d(ctx, layer.k_quant,
            elements_to_quantize,
            dst_offset_bytes
        );

    // THREAD-SAFE FIX: Update counter before returning (atomic operation would be better)
    const_cast<kv_layer_mixed&>(layer).quant_k_tokens += config.quantization_threshold;

    LLAMA_LOG_DEBUG("[mixed-kv] created K quantization views: src_offset=%zu, dst_offset=%zu, elements=%zu\n",
                   src_offset_bytes, dst_offset_bytes, elements_to_quantize);

    return ggml_cpy(ctx, k_need_quantize, k_quantized);
}

ggml_tensor * llama_kv_cache_mixed::v_quant(ggml_context * ctx, int32_t il) const {
    // CRITICAL FIX: Use proper layer mapping instead of direct indexing
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        LLAMA_LOG_ERROR("[mixed-kv] ERROR: Layer %d not found in map\n", il);
        return nullptr;
    }

    auto & layer = layers[it->second];
    auto * v = layer.v_fp16;

    // DEFENSIVE FIX: Validate we have enough tokens to quantize
    if (used < config.quantization_threshold) {
        LLAMA_LOG_ERROR("[mixed-kv] ERROR: Not enough tokens to quantize (used=%u, threshold=%u)\n",
                       used, config.quantization_threshold);
        return nullptr;
    }

    LLAMA_LOG_DEBUG("[mixed-kv] quantizing %u V tokens from layer %d (used=%u)\n",
                   config.quantization_threshold, il, used);

    // CRITICAL FIX: Calculate source offset safely
    const uint32_t src_offset_tokens = used - config.quantization_threshold;
    const size_t src_offset_bytes = ggml_row_size(v->type, hparams.n_embd_v_gqa(il)) * src_offset_tokens;
    const size_t elements_to_quantize = config.quantization_threshold * hparams.n_embd_v_gqa(il);

    // DEFENSIVE FIX: Bounds checking for source tensor
    const size_t v_total_bytes = ggml_nbytes(v);
    const size_t required_bytes = src_offset_bytes + ggml_row_size(v->type, hparams.n_embd_v_gqa(il)) * config.quantization_threshold;
    if (required_bytes > v_total_bytes) {
        LLAMA_LOG_ERROR("[mixed-kv] ERROR: V quantization source out of bounds (need %zu, have %zu)\n",
                       required_bytes, v_total_bytes);
        return nullptr;
    }

    // CRITICAL FIX: Use correct type for destination tensor view
    const size_t dst_offset_bytes = ggml_row_size(layer.v_quant->type, hparams.n_embd_v_gqa(il)) * layer.quant_v_tokens;
    const size_t v_quant_total_bytes = ggml_nbytes(layer.v_quant);
    const size_t dst_required_bytes = dst_offset_bytes + ggml_row_size(layer.v_quant->type, hparams.n_embd_v_gqa(il)) * config.quantization_threshold;

    if (dst_required_bytes > v_quant_total_bytes) {
        LLAMA_LOG_ERROR("[mixed-kv] ERROR: V quantization destination out of bounds (need %zu, have %zu)\n",
                       dst_required_bytes, v_quant_total_bytes);
        return nullptr;
    }

    // Create views with proper bounds checking
    ggml_tensor * v_need_quantize = ggml_view_1d(ctx, v,
            elements_to_quantize,
            src_offset_bytes);

    ggml_tensor * v_quantized = ggml_view_1d(ctx, layer.v_quant,
            elements_to_quantize,
            dst_offset_bytes);

    // THREAD-SAFE FIX: Update counter before returning (atomic operation would be better)
    const_cast<kv_layer_mixed&>(layer).quant_v_tokens += config.quantization_threshold;

    LLAMA_LOG_DEBUG("[mixed-kv] created V quantization views: src_offset=%zu, dst_offset=%zu, elements=%zu\n",
                   src_offset_bytes, dst_offset_bytes, elements_to_quantize);

    return ggml_cpy(ctx, v_need_quantize, v_quantized);
}

ggml_tensor * llama_kv_cache_mixed::get_k_quant_ref(ggml_context * ctx, int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return nullptr;
    }
    const auto & layer = layers[it->second];

    ggml_tensor * k_ref = ggml_view_3d(ctx, layer.k_fp16,
            hparams.n_embd_head_k, hparams.n_head_kv(il), config.quantization_threshold,
            ggml_row_size(layer.k_fp16->type, hparams.n_embd_head_k),
            ggml_row_size(layer.k_fp16->type, hparams.n_embd_k_gqa(il)),
            ggml_row_size(layer.k_fp16->type, hparams.n_embd_k_gqa(il)) * (layer.quant_k_tokens - config.quantization_threshold)
        );

    return k_ref;
}

ggml_tensor * llama_kv_cache_mixed::get_v_quant_ref(ggml_context * ctx, int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return nullptr;
    }
    const auto & layer = layers[it->second];

    ggml_tensor * v_ref = ggml_view_3d(ctx, layer.v_fp16,
            hparams.n_embd_head_v, hparams.n_head_kv(il), config.quantization_threshold,
            ggml_row_size(layer.v_fp16->type, hparams.n_embd_head_v),
            ggml_row_size(layer.v_fp16->type, hparams.n_embd_v_gqa(il)),
            ggml_row_size(layer.v_fp16->type, hparams.n_embd_v_gqa(il)) * (layer.quant_v_tokens - config.quantization_threshold)
        );

    return v_ref;
}

ggml_tensor * llama_kv_cache_mixed::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto & layer = layers[ikv];
    auto * k = layer.k_fp16;

    // NOTE: k_cur shape is (n_embd_k_gqa(il), n_head, n_tokens, n_batch_size)
    const int64_t n_tokens = k_cur->ne[2];

    if (il == 0) {
        LLAMA_LOG_DEBUG("[mixed-kv] cur shape: %d, %d, %d, %d\n", k_cur->ne[0], k_cur->ne[1], k_cur->ne[2], k_cur->ne[3]);
        LLAMA_LOG_DEBUG("[mixed-kv] cpy_k: adding %ld K tokens to layer %d cache (head=%u)\n", n_tokens, il, head);
        LLAMA_LOG_DEBUG("[mixed-kv]   - before: total=%u, quant_k=%u, quant_v=%u, fp16_k=%u, fp16_v=%u\n",
                    layer.total_tokens, layer.quant_k_tokens, layer.quant_v_tokens, layer.fp16_k_tokens, used);
    }

    // Update token management for FIFO strategy
    if (layer.fp16_k_tokens == 0) {
        // First tokens in this layer
        layer.fp16_start_pos = layer.total_tokens;
    }

    layer.fp16_k_tokens += n_tokens;
    layer.total_tokens += n_tokens;

    if (il == 0) {
        LLAMA_LOG_DEBUG("[mixed-kv]   - after: total=%u, quant_k=%u, quant_v=%u, fp16_k=%u, fp16_v=%u (added %ld K tokens)\n",
                   layer.total_tokens, layer.quant_k_tokens, layer.quant_v_tokens, layer.fp16_k_tokens, used, n_tokens);
    }

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

    LLAMA_LOG_DEBUG("[mixed-kv] cpy_v: adding %ld V tokens to layer %d cache (head=%u)\n", n_tokens, il, head);

    // Update V token counter separately
    layer.fp16_v_tokens += n_tokens;

    LLAMA_LOG_DEBUG("[mixed-kv]   - V tokens updated: fp16_v_tokens=%u (added %ld V tokens)\n",
                   layer.fp16_v_tokens, n_tokens);

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
                (    head)*ggml_element_size(v));
        v_cur = ggml_transpose(ctx, v_cur);
    }

    return ggml_cpy(ctx, v_cur, v_view);
}

// Get current memory usage and pressure information
llama_kv_cache_mixed::memory_info llama_kv_cache_mixed::get_memory_info() const {
    memory_info info;

    // Calculate memory usage for FP16 and quantized tensors
    info.fp16_memory_bytes = size_k_bytes() / 2;  // Half for FP16 (vs full for both FP16+quant)
    info.quant_memory_bytes = size_k_bytes() / 2; // Half for quantized
    info.total_memory_bytes = info.fp16_memory_bytes + info.quant_memory_bytes;

    // Simple memory pressure calculation (can be improved)
    const size_t max_memory = size_k_bytes() + size_v_bytes();
    if (max_memory > 0) {
        info.memory_pressure = (float)info.total_memory_bytes / max_memory;
    }

    // Determine if quantization should be triggered
    info.should_quantize = quant_mgr.should_quantize(config, info.memory_pressure);

    return info;
}

// Get token distribution information for a specific layer
llama_kv_cache_mixed::layer_token_info llama_kv_cache_mixed::get_layer_token_info(int32_t il) const {
    layer_token_info info;

    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return info; // valid = false
    }

    const auto & layer = layers[it->second];
    info.n_fp16_tokens = layer.fp16_k_tokens;
    info.n_quant_tokens = layer.quant_k_tokens; // Use K quant tokens (V should be same)
    info.valid = true;

    return info;
}

//=================================================================================================
// Custom Flash Attention Implementation for Mixed KV Cache with Flash-Decoding
//=================================================================================================

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
    ggml_tensor * k_quant = dst->src[4];
    ggml_tensor * v_quant = dst->src[5];

    if (!q || !k || !v ) {
        LLAMA_LOG_ERROR("[mixed-kv] ERROR: null tensors in custom flash attention\n");
        return;
    }

    //> q:    [head_dim, q_len,  n_heads, n_batch]
    //> k:    [head_dim, kv_len, n_heads, n_batch]
    //> v:    [head_dim, kv_len, n_heads, n_batch]
    //> mask: [n_heads,  q_len,  kv_len,  n_batch]
    //> dst:  [head_dim, n_heads, q_len, n_batch]

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    const int64_t DK = nek0;            //> head_dim for keys
    const int64_t DV = nev0;            //> head_dim for values
    const int64_t SEQ_LEN  = neq1;      //> q_len
    const int64_t KV_LEN    = nek1;     //> kv sequence length
    const int64_t N_KV_HEAD = nek2;     //> number of kv heads
    const int64_t N_Q_HEADS   = neq2;   //> number of query heads
    const int64_t N_BATCH   = ne3;      //> batch size

    GGML_ASSERT(ne0 == DV);             //> dst -> ne[0] == head_dim
    GGML_ASSERT(ne1 == N_Q_HEADS);      //> dst -> ne[1] == n_heads
    GGML_ASSERT(ne2 == SEQ_LEN);        //> dst -> ne[2] == q_len

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

    // Workspace layout per thread (enhanced for multi-type V support):
    //> Similar to standard flash attention workspace layout
    // Note: Output is stored as [DV, N_Q_HEADS, SEQ_LEN] for each batch
    const size_t OUTPUT_SIZE    = DV * N_Q_HEADS * SEQ_LEN;
    const size_t LOCAL_MAX_SIZE = N_Q_HEADS * SEQ_LEN;
    // DEFENSIVE FIX: Calculate workspace size more conservatively
    const size_t workspace_per_thread = OUTPUT_SIZE + 2 * LOCAL_MAX_SIZE + 2 * DV + 1 * DK + 1 + CACHE_LINE_SIZE_F32;

    // CRITICAL FIX: Check workspace size before proceeding
    const size_t total_workspace_needed = workspace_per_thread * nth * sizeof(float);
    if (wsize < total_workspace_needed) {
        LLAMA_LOG_ERROR("[mixed-kv] ERROR: Insufficient workspace size. Need: %zu, Got: %zu, threads: %d\n",
                       total_workspace_needed, wsize, nth);
        return;
    }

    // DEFENSIVE FIX: Add bounds checking for thread workspace
    if (ith >= nth) {
        LLAMA_LOG_ERROR("[mixed-kv] ERROR: Thread index %d out of bounds (max: %d)\n", ith, nth - 1);
        return;
    }

    float * thread_workspace    = (float *) wdata + ith * workspace_per_thread;

    // DEFENSIVE FIX: Validate thread workspace pointer
    if (!thread_workspace || (char*)thread_workspace + workspace_per_thread * sizeof(float) > (char*)wdata + wsize) {
        LLAMA_LOG_ERROR("[mixed-kv] ERROR: Thread workspace %d out of bounds\n", ith);
        return;
    }

    const int64_t rk2 = neq2 / nek2;     //> n_q_heads / n_kv_heads
    const int64_t rv2 = neq2 / nev2;     //> n_q_heads / n_kv_heads

    float * chunk_output    = thread_workspace;                                                                 // [N_Q_HEADS * SEQ_LEN * DV]
    float * local_max       = thread_workspace + OUTPUT_SIZE;                                                   // [N_Q_HEADS * SEQ_LEN]
    float * local_exp_sum   = thread_workspace + OUTPUT_SIZE + LOCAL_MAX_SIZE;                                  // [N_Q_HEADS * SEQ_LEN]
    float * V32_buffer      = thread_workspace + OUTPUT_SIZE + 2 * LOCAL_MAX_SIZE;                              // [DV] - F32 V buffer for conversion
    float * temp_buffer     = thread_workspace + OUTPUT_SIZE + 2 * LOCAL_MAX_SIZE + 1 * DV;                     // [DV] - temp buffer
    ggml_fp16_t * Q_q       = (ggml_fp16_t *)(thread_workspace + OUTPUT_SIZE + 2 * LOCAL_MAX_SIZE + 2 * DV );   // [DK]
    volatile uint32_t * sync_buffer = (volatile uint32_t *)(thread_workspace + OUTPUT_SIZE + 2 * LOCAL_MAX_SIZE + 2 * DV + 1 * DK);  // [1] atomic sync var

    // Initialize chunk outputs and log_sum_exp for all queries
    memset(chunk_output,   0,           OUTPUT_SIZE * sizeof(float));
    memset(local_exp_sum,  0,           LOCAL_MAX_SIZE * sizeof(float));  // FIX: Initialize exp_sum to 0
    memset(V32_buffer,     0,           DV * sizeof(float));
    memset(temp_buffer,    0,           DV * sizeof(float));
    memset(Q_q,            0,           DK * sizeof(ggml_fp16_t));
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

    // Handle quantization for K/V tensor (similar to standard flash attention)
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
            // DEFENSIVE FIX: Add bounds checking for tensor data access
            const size_t k_offset = kv_pos * nbk1 + kv_head * nbk2;
            const size_t v_offset = kv_pos * nbv1 + kv_head * nbv2;

            // Check if offsets are within tensor bounds
            if (k_offset >= ggml_nbytes(k)) {
                LLAMA_LOG_ERROR("[mixed-kv] ERROR: K tensor offset %zu out of bounds (size: %zu)\n",
                               k_offset, ggml_nbytes(k));
                continue;
            }

            if (v_offset >= ggml_nbytes(v)) {
                LLAMA_LOG_ERROR("[mixed-kv] ERROR: V tensor offset %zu out of bounds (size: %zu)\n",
                               v_offset, ggml_nbytes(v));
                continue;
            }

            const char * k_data = (const char *) ((char *) k->data + k_offset);
            const char * v_data = (const char *) ((char *) v->data + v_offset);

            GGML_ASSERT(k_data != nullptr);
            GGML_ASSERT(v_data != nullptr);

            const int64_t q_head_start = kv_head * rk2;       //> q_head_start = head / rk2 * rk2
            const int64_t q_head_end   = q_head_start + rk2;  //> q_head_end = q_head_start + rk2

            GGML_ASSERT(q_head_start >= 0);

            for (int64_t q_head = q_head_start; q_head < q_head_end; ++ q_head) {
                for (int64_t q_pos = 0; q_pos < SEQ_LEN; ++ q_pos) {
                    // CRITICAL FIX: Use consistent output offset calculation for both single and multi-threaded cases
                    // dst layout: [DV, N_Q_HEADS, SEQ_LEN, N_BATCH]
                    // For position (q_head, q_pos), offset = q_head * DV + q_pos * (DV * N_Q_HEADS)
                    const int64_t output_offset = q_head * DV + q_pos * (DV * N_Q_HEADS);
                    const int64_t local_max_idx = q_pos * N_Q_HEADS + q_head;

                    // DEFENSIVE FIX: Add bounds checking for output offset
                    if (output_offset < 0 || output_offset + DV > OUTPUT_SIZE) {
                        LLAMA_LOG_ERROR("[mixed-kv] ERROR: Output offset %ld out of bounds (max: %zu)\n",
                                       output_offset + DV, OUTPUT_SIZE);
                        continue;
                    }

                    if (local_max_idx < 0 || local_max_idx >= LOCAL_MAX_SIZE) {
                        LLAMA_LOG_ERROR("[mixed-kv] ERROR: Local max index %ld out of bounds (max: %zu)\n",
                                       local_max_idx, LOCAL_MAX_SIZE);
                        continue;
                    }

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

                    // Multi-type V support (similar to standard flash attention)
                    local_exp_sum[local_max_idx] = local_exp_sum[local_max_idx] * ms + vs;

                    if (ms != 1.0f) {
                        // NOTE: Multiply past sum by ms
                        ggml_vec_scale_f32(DV, (float *)output_ptr, ms);
                    }

                    // V += v*expf(s - M) - handle different V types
                    if (v->type == GGML_TYPE_F32) {
                        // V is already F32, use directly
                        ggml_vec_mad_f32(DV, (float *)output_ptr, (const float *)v_data, vs);
                    } else if (v_to_float) {
                        // V is quantized or F16, convert to F32 first
                        v_to_float(v_data, V32_buffer, DV);
                        ggml_vec_mad_f32(DV, (float *)output_ptr, V32_buffer, vs);
                    } else {
                        // NOTICE: treat as F32 (this shouldn't happen)
                        LLAMA_LOG_WARN("[mixed-kv] WARNING: V is not F32 or F16, treating as F32\n");
                    }
                }
            }
        }
    } //> end of chunk

    //> Barrier-free synchronization: set sync_buffer[0] to 1 (even if chunk is empty)
    sync_buffer[0] = 1;

    //> =======================================================================================
    //> BARRIER-FREE SYNCHRONIZATION: All threads must complete before thread 0 can reduce
    //> We use a simple busy-wait pattern checking if all chunks have been computed
    //> =======================================================================================

    // Thread 0 waits for all other threads and performs reduction
    if (ith == 0 && nth > 1) {
        // Simple busy-wait for all threads to complete their chunk computation
        bool all_threads_ready = false;
        int wait_cycles = 0;
        const int max_wait_cycles = 1000000; // Prevent infinite wait

        // NOTICE: Sync points.
        while (!all_threads_ready && wait_cycles < max_wait_cycles) {
            all_threads_ready = true;
            for (int t = 1; t < nth; ++t) { // Start from 1 since thread 0 is us
                float * t_workspace = (float *) wdata + t * workspace_per_thread;
                volatile uint32_t * t_sync_buffer = (volatile uint32_t *)(t_workspace + OUTPUT_SIZE + 2 * LOCAL_MAX_SIZE + 2 * DV + 1 * DK);

                // Thread is ready if it set sync_buffer[0] to 1
                if (t_sync_buffer[0] != 1) {
                    all_threads_ready = false;
                    break;
                }
            }
            wait_cycles++;
        }

        if (wait_cycles >= max_wait_cycles) {
            LLAMA_LOG_WARN("[mixed-kv] WARNING: thread synchronization timeout, proceeding with reduction, wait_cycles: %d\n", wait_cycles);
        }

        // Perform log-sum-exp reduction across all threads
        for (int64_t q_head = 0; q_head < N_Q_HEADS; ++q_head) {
            for (int64_t q_pos = 0; q_pos < SEQ_LEN; ++q_pos) {
                // CRITICAL FIX: Use consistent output offset calculation
                // dst layout: [DV, N_Q_HEADS, SEQ_LEN, N_BATCH]
                // For position (q_head, q_pos), offset = q_head * DV + q_pos * (DV * N_Q_HEADS)
                const int64_t output_offset = q_head * DV + q_pos * (DV * N_Q_HEADS);
                const int64_t local_max_idx = q_pos * N_Q_HEADS + q_head;

                // Find global maximum across all threads for this query
                // Only consider threads that actually processed tokens (local_max != -INFINITY)
                float global_max = -INFINITY;
                for (int t = 0; t < nth; ++t) {
                    float * t_workspace = (float *) wdata + t * workspace_per_thread;
                    float * t_local_max = t_workspace + OUTPUT_SIZE;

                    // Only consider threads that processed tokens (not empty chunks)
                    if (t_local_max[local_max_idx] != -INFINITY && t_local_max[local_max_idx] > global_max) {
                        global_max = t_local_max[local_max_idx];
                    }
                }

                // If all threads had -INFINITY (no valid tokens), skip this query
                if (global_max == -INFINITY) {
                    // DEFENSIVE FIX: Bounds check for final output access
                    if (output_offset + DV <= ggml_nelements(dst)) {
                        float * final_output = (float *) dst->data + output_offset;
                        memset(final_output, 0, DV * sizeof(float));
                    } else {
                        LLAMA_LOG_ERROR("[mixed-kv] ERROR: Final output offset %ld out of bounds (dst size: %ld)\n",
                                       output_offset + DV, ggml_nelements(dst));
                    }
                    continue;
                }

                // Compute sum of exponentials with global max for numerical stability
                // Only include threads that actually processed tokens
                float global_sum = 0.0f;
                int active_threads = 0;
                for (int t = 0; t < nth; ++t) {
                    float * t_workspace = (float *) wdata + t * workspace_per_thread;
                    float * t_local_max = t_workspace + OUTPUT_SIZE;
                    float * t_local_exp_sum = t_workspace + OUTPUT_SIZE + LOCAL_MAX_SIZE;

                    // Only include threads that processed tokens (not empty chunks)
                    if (t_local_max[local_max_idx] != -INFINITY && t_local_exp_sum[local_max_idx] > 0.0f) {
                        // FIXED: Numerical stability - clamp exponential difference
                        const float max_diff = t_local_max[local_max_idx] - global_max;
                        const float clamped_diff = fmaxf(-50.0f, fminf(50.0f, max_diff)); // Clamp to prevent overflow
                        const float exp_sum_adjustment = expf(clamped_diff);

                        // Additional safety check
                        if (std::isfinite(exp_sum_adjustment) && exp_sum_adjustment > 0.0f) {
                            global_sum += t_local_exp_sum[local_max_idx] * exp_sum_adjustment;
                            active_threads++;
                        }
                    }
                }

                // Debug: query reduction statistics (can be disabled in production)
                // LLAMA_LOG_DEBUG("[mixed-kv] Query (head=%ld, pos=%ld): active_threads=%d, global_max=%.6f, global_sum=%.6f\n",
                //                q_head, q_pos, active_threads, global_max, global_sum);

                // Normalize factor for final attention weights
                const float norm_factor = 1.0f / global_sum;

                // DEFENSIVE FIX: Bounds check before combining weighted outputs
                if (output_offset + DV > ggml_nelements(dst)) {
                    LLAMA_LOG_ERROR("[mixed-kv] ERROR: Final output offset %ld out of bounds (dst size: %ld)\n",
                                   output_offset + DV, ggml_nelements(dst));
                    continue;
                }

                // Combine weighted outputs from all threads
                float * final_output = (float *) dst->data + output_offset;
                memset(final_output, 0, DV * sizeof(float)); // Initialize to zero

                for (int t = 0; t < nth; ++t) {
                    float * t_workspace = (float *) wdata + t * workspace_per_thread;
                    float * t_chunk_output = t_workspace;
                    float * t_local_max = t_workspace + OUTPUT_SIZE;
                    float * t_local_exp_sum = t_workspace + OUTPUT_SIZE + LOCAL_MAX_SIZE;

                    // Only include contributions from threads that processed tokens
                    if (t_local_max[local_max_idx] != -INFINITY && t_local_exp_sum[local_max_idx] > 0.0f && global_sum > 0.0f) {
                        // FIXED: Numerical stability in thread weight calculation
                        const float max_diff = t_local_max[local_max_idx] - global_max;
                        const float clamped_diff = fmaxf(-50.0f, fminf(50.0f, max_diff)); // Clamp to prevent overflow
                        const float max_adjustment = expf(clamped_diff);

                        // Additional safety check for numerical stability
                        if (std::isfinite(max_adjustment) && max_adjustment > 0.0f && std::isfinite(global_sum) && global_sum > 0.0f) {
                            const float thread_weight = max_adjustment / global_sum;

                            if (std::isfinite(thread_weight) && thread_weight > 0.0f) {
                                // Add this thread's adjusted contribution
                                const float * thread_output = t_chunk_output + output_offset;
                                ggml_vec_mad_f32(DV, final_output, thread_output, thread_weight);
                            }
                        }
                    }
                }
            }
        }
    } else if (nth == 1) {
        // CRITICAL FIX: Single-threaded execution - use consistent output layout
        // For single-threaded execution, normalize the accumulated outputs correctly

        float* thread0_workspace    = (float*)wdata;
        float* local_exp_sum        = thread0_workspace + OUTPUT_SIZE + LOCAL_MAX_SIZE;

        for (int64_t q_head = 0; q_head < N_Q_HEADS; ++q_head) {
            for (int64_t q_pos = 0; q_pos < SEQ_LEN; ++q_pos) {
                // CRITICAL FIX: Use the same output offset calculation as multi-threaded case
                // dst layout: [DV, N_Q_HEADS, SEQ_LEN, N_BATCH]
                // For position (q_head, q_pos), offset = q_head * DV + q_pos * (DV * N_Q_HEADS)
                const int64_t output_offset = q_head * DV + q_pos * (DV * N_Q_HEADS);
                const int64_t local_max_idx = q_pos * N_Q_HEADS + q_head;

                // DEFENSIVE FIX: Bounds check for single-threaded output access
                if (output_offset + DV > ggml_nelements(dst)) {
                    LLAMA_LOG_ERROR("[mixed-kv] ERROR: Single-threaded output offset %ld out of bounds (dst size: %ld)\n",
                                   output_offset + DV, ggml_nelements(dst));
                    continue;
                }

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
