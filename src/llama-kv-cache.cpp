#include "llama-kv-cache.h"

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

//
// llama_kv_cache_unified
//

uint32_t llama_kv_cache_unified::get_padding(const llama_cparams & cparams) {
    // the FA kernels require padding to avoid extra runtime boundary checks
    return cparams.flash_attn ? 256u : 32u;
}

llama_kv_cache_unified::llama_kv_cache_unified(
        const llama_model &  model,
          layer_filter_cb && filter,
                ggml_type    type_k,
                ggml_type    type_v,
                     bool    v_trans,
                     bool    offload,
                 uint32_t    kv_size,
                 uint32_t    n_seq_max,
                 uint32_t    n_pad,
                 uint32_t    n_swa,
           llama_swa_type    swa_type) :
    model(model), hparams(model.hparams), v_trans(v_trans),
    n_seq_max(n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type) {

    GGML_ASSERT(kv_size % n_pad == 0);

    // create a context for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*hparams.n_layer*ggml_tensor_overhead()),
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

    /*
     * 初始化 KV 缓存的核心管理数据结构 cells：
     * 
     * cells 是统一 KV 缓存的核心管理数组，每个元素对应一个缓存槽位
     * 
     * ┌─────────────────────────────────────────────────────────┐
     * │                 Unified KV Cache Layout                 │
     * │                                                         │
     * │  cells[0]  cells[1]  cells[2]  ...  cells[kv_size-1]   │
     * │  ┌─────┐   ┌─────┐   ┌─────┐         ┌─────┐           │
     * │  │slot │   │slot │   │slot │   ...   │slot │           │
     * │  │  0  │   │  1  │   │  2  │         │ N-1 │           │
     * │  └─────┘   └─────┘   └─────┘         └─────┘           │
     * │     ↓         ↓         ↓               ↓              │
     * │   pos=-1    pos=-1    pos=-1          pos=-1           │
     * │  (empty)    (empty)   (empty)        (empty)           │
     * │  delta=0    delta=0   delta=0        delta=0           │
     * │  seq_id={}  seq_id={} seq_id={}      seq_id={}         │
     * └─────────────────────────────────────────────────────────┘
     * 
     * 每个 cell 包含：
     * - pos: token 在序列中的位置 (-1 表示空闲)
     * - delta: 位置偏移累积量，用于 RoPE 和 K-shift
     * - seq_id: 使用该 token 的序列 ID 集合 (支持多序列共享)
     */
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

        ggml_tensor * k;
        ggml_tensor * v;

        k = ggml_new_tensor_2d(ctx, type_k, n_embd_k_gqa, kv_size);
        v = ggml_new_tensor_2d(ctx, type_v, n_embd_v_gqa, kv_size);

        ggml_format_name(k, "cache_k_l%d", il);
        ggml_format_name(v, "cache_v_l%d", il);

        map_layer_ids[il] = layers.size();
        layers.push_back({ il, k, v });
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto it : ctx_map) {
        auto * buft = it.first;
        auto * ctx  = it.second;

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);
        bufs.emplace_back(buf);
    }

    {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f), kv_size, (int) layers.size(), n_seq_max,
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
    }
}

void llama_kv_cache_unified::clear() {
    /*
     * cells 清空操作 - 重置所有缓存槽状态到初始空闲状态：
     * 
     * 遍历所有 cells，重置每个缓存槽的状态：
     * 1. pos = -1：标记为空闲槽位
     * 2. seq_id.clear()：清空序列ID集合
     * 3. delta 保持默认值 0（自动初始化）
     * 4. 重置管理计数器 (head=0, used=0)
     * 
     * Before clear():                    After clear():
     * ┌─────┬─────┬─────┬─────┐         ┌─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:2│pos:3│   -->   │pos:-│pos:-│pos:-│pos:-│
     * │seq:1│seq:1│seq:2│seq:2│         │seq: │seq: │seq: │seq: │
     * │Δ:+2 │Δ:+1 │Δ:-1 │Δ:+3 │         │Δ:0  │Δ:0  │Δ:0  │Δ:0  │
     * │used │used │used │used │         │empty│empty│empty│empty│
     * └─────┴─────┴─────┴─────┘         └─────┴─────┴─────┴─────┘
     * 
     * 注意：delta 在 clear() 中会自动重置为 0，因为 kv_cell 构造函数中 delta=0
     */
    for (uint32_t i = 0; i < size; ++i) {
        cells[i].pos = -1;        // 标记为空闲槽位
        cells[i].seq_id.clear();  // 清空序列ID集合
        // delta 会在 kv_cell 构造时自动重置为 0
    }

    head = 0;
    used = 0;

    for (auto & buf : bufs) {
        ggml_backend_buffer_clear(buf.get(), 0);
    }
}

bool llama_kv_cache_unified::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    uint32_t new_head = size;

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    /*
     * cells 序列移除操作 - 从指定位置范围移除序列 tokens：
     * 
     * 遍历所有 cells，找到位置在 [p0, p1) 范围内的 tokens，
     * 移除指定序列 ID，如果 cell 变空则标记为空闲
     * 
     * 例如：seq_rm(seq_id=1, p0=1, p1=3)
     * 
     * Before seq_rm():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:2│pos:3│pos:4│
     * │seq:1│seq:1│seq:1│seq:2│seq:1│  <- 移除位置1-2的seq:1
     * │Δ:0  │Δ:+1 │Δ:+2 │Δ:0  │Δ:+3 │
     * └─────┴─────┴─────┴─────┴─────┘
     * 
     * After seq_rm():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:-│pos:-│pos:3│pos:4│
     * │seq:1│empty│empty│seq:2│seq:1│  <- pos:1,2被清空
     * │Δ:0  │Δ:0  │Δ:0  │Δ:0  │Δ:+3 │  <- delta 保持，因为可能用于 K-shift
     * └─────┴─────┴─────┴─────┴─────┘
     *         ↑     ↑
     *      new_head 候选位置
     * 
     * 注意：delta 不会被清除，因为它记录了位置偏移历史，
     *       可能在后续的 K-shift 操作中使用
     */
    for (uint32_t i = 0; i < size; ++i) {
        // 检查该 cell 的位置是否在移除范围内
        if (cells[i].pos >= p0 && cells[i].pos < p1) {
            if (seq_id < 0) {
                // seq_id < 0 表示移除所有序列
                cells[i].seq_id.clear();
            } else if (cells[i].has_seq_id(seq_id)) {
                // 只移除指定的序列 ID
                cells[i].seq_id.erase(seq_id);
            } else {
                continue;
            }

            if (cells[i].is_empty()) {
                // 如果 cell 变空，则标记为空闲
                // keep count of the number of used cells
                if (cells[i].pos >= 0) {
                    used--;
                }

                cells[i].pos = -1;
                // 注意：delta 不被重置，保留位置偏移历史

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

void llama_kv_cache_unified::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (seq_id_src == seq_id_dst) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    /*
     * cells 序列复制操作 - 将源序列的 tokens 复制给目标序列：
     * 
     * 遍历所有 cells，找到属于源序列且在指定位置范围内的 tokens，
     * 将目标序列 ID 添加到这些 cells（实现多序列共享同一 token）
     * 
     * 例如：seq_cp(seq_src=1, seq_dst=3, p0=1, p1=3)
     * 
     * Before seq_cp():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:2│pos:3│pos:4│
     * │seq:1│seq:1│seq:1│seq:2│seq:1│  <- 复制seq:1的pos:1-2给seq:3
     * │Δ:0  │Δ:+1 │Δ:+2 │Δ:0  │Δ:+3 │
     * └─────┴─────┴─────┴─────┴─────┘
     * 
     * After seq_cp():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:2│pos:3│pos:4│
     * │seq:1│1,3  │1,3  │seq:2│seq:1│  <- pos:1,2现在同时属于seq:1和seq:3
     * │Δ:0  │Δ:+1 │Δ:+2 │Δ:0  │Δ:+3 │  <- delta 不变，因为位置偏移历史保持
     * └─────┴─────┴─────┴─────┴─────┘
     * 
     * 重要：delta 在复制时保持不变，因为 delta 记录的是该位置的偏移历史，
     *       对于共享该位置的所有序列都是有效的
     */
    // otherwise, this is the KV of a Transformer-like model
    head = 0;

    for (uint32_t i = 0; i < size; ++i) {
        // 检查该 cell 是否属于源序列且在指定位置范围内
        if (cells[i].has_seq_id(seq_id_src) && cells[i].pos >= p0 && cells[i].pos < p1) {
            // 将目标序列 ID 添加到该 cell（多序列共享同一 token）
            cells[i].seq_id.insert(seq_id_dst);
            // delta 保持不变，因为位置偏移历史对所有共享序列都有效
        }
    }
}

void llama_kv_cache_unified::seq_keep(llama_seq_id seq_id) {
    uint32_t new_head = size;

    /*
     * cells 序列保留操作 - 只保留指定序列，清除其他所有序列：
     * 
     * 遍历所有 cells，对于不属于目标序列的 cells 进行清空，
     * 对于属于目标序列的 cells 清除其他序列 ID（保持单一序列）
     * 
     * 例如：seq_keep(seq_id=2)
     * 
     * Before seq_keep():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:2│pos:3│pos:4│
     * │seq:1│1,3  │seq:2│seq:2│seq:1│  <- 只保留seq:2
     * │Δ:0  │Δ:+1 │Δ:+2 │Δ:0  │Δ:+3 │
     * └─────┴─────┴─────┴─────┴─────┘
     * 
     * After seq_keep():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:-│pos:-│pos:2│pos:3│pos:-│
     * │empty│empty│seq:2│seq:2│empty│  <- 只有seq:2的cells被保留
     * │Δ:0  │Δ:0  │Δ:+2 │Δ:0  │Δ:0  │  <- delta保持或清零，取决于cell状态
     * └─────┴─────┴─────┴─────┴─────┘
     *   ↑     ↑               ↑
     * new_head候选位置          清空的cell
     * 
     * 注意：delta 处理策略：
     * - 被保留的 cells：delta 保持不变（位置偏移历史仍有效）
     * - 被清空的 cells：delta 在下次使用时会重新设置
     */
    for (uint32_t i = 0; i < size; ++i) {
        // 检查该 cell 是否不属于要保留的序列
        if (!cells[i].has_seq_id(seq_id)) {
            // 该 cell 不属于目标序列，清除它
            if (cells[i].pos >= 0) {
                used--;  // 减少已使用计数
            }

            cells[i].pos = -1;           // 标记为空闲
            cells[i].seq_id.clear();     // 清空序列ID
            // delta 保留当前值，在下次分配时会被重新设置

            // 记录第一个空闲槽位作为新的搜索起点
            if (new_head == size){
                new_head = i;
            }
        } else {
            // 该 cell 属于目标序列，清除其他序列ID（保持单一序列）
            cells[i].seq_id.clear();
            cells[i].seq_id.insert(seq_id);
            // delta 保持不变，因为该 cell 的位置偏移历史仍然有效
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != size && new_head < head) {
        head = new_head;
    }
}

void llama_kv_cache_unified::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) {
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

    // If there is no range then return early to avoid looping over the
    if (p0 == p1) {
        return;
    }

    /*
     * cells 序列位置偏移操作 - 核心 delta 累积机制：
     * 
     * 将指定序列的位置向前或向后移动，同时累积 delta 偏移量
     * delta 是 RoPE (Rotary Position Embedding) 计算的关键组件
     * 
     * 例如：seq_add(seq_id=1, p0=2, p1=4, delta=+2)
     * 
     * Before seq_add():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:2│pos:3│pos:4│
     * │seq:1│seq:1│seq:1│seq:1│seq:2│  <- seq:1在pos:2-3的tokens需要+2偏移
     * │Δ:0  │Δ:+1 │Δ:0  │Δ:-1 │Δ:+2 │
     * └─────┴─────┴─────┴─────┴─────┘
     * 
     * After seq_add():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:4│pos:5│pos:4│
     * │seq:1│seq:1│seq:1│seq:1│seq:2│  <- pos:2→4, pos:3→5
     * │Δ:0  │Δ:+1 │Δ:+2 │Δ:+1 │Δ:+2 │  <- delta累积：0+2=2, -1+2=1
     * └─────┴─────┴─────┴─────┴─────┘
     * 
     * 负偏移示例：seq_add(seq_id=1, p0=2, p1=4, delta=-3)
     * pos:2→-1, pos:3→0，pos变负的cell被清除：
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:-│pos:0│pos:4│
     * │seq:1│seq:1│empty│seq:1│seq:2│  <- pos:2被清除因为变成-1
     * │Δ:0  │Δ:+1 │Δ:0  │Δ:-4 │Δ:+2 │  <- delta重置为0(新分配时)，0-1-3=-4
     * └─────┴─────┴─────┴─────┴─────┘
     * 
     * delta 的重要作用：
     * 1. RoPE 计算：实际位置 = pos + delta，用于旋转位置编码
     * 2. K-shift 操作：记录需要应用的位置偏移
     * 3. 序列操作历史：累积所有位置变化，保证一致性
     */
    for (uint32_t i = 0; i < size; ++i) {
        // 检查该 cell 是否属于目标序列且在指定位置范围内
        if (cells[i].has_seq_id(seq_id) && cells[i].pos >= p0 && cells[i].pos < p1) {
            has_shift = true;  // 标记发生了位置偏移，触发后续 K-shift

            cells[i].pos   += delta;  // 更新 token 位置
            cells[i].delta += delta;  // 累积位置偏移量（关键！）

            // 如果位置变成负数，则清除该 cell
            if (cells[i].pos < 0) {
                if (!cells[i].is_empty()) {
                    used--;
                }
                cells[i].pos = -1;
                cells[i].seq_id.clear();
                // delta 在 cell 清空后会在下次分配时重新设置
                
                if (new_head == size) {
                    new_head = i;
                }
            }
            // 注意：对于有效的 cells，delta 持续累积，
            //       记录了该位置的完整偏移历史
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != size ? new_head : 0;
}

void llama_kv_cache_unified::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    /*
     * cells 序列位置除法操作 - delta 在位置缩放中的精确计算：
     * 
     * 将指定序列的位置按比例缩小，同时精确计算 delta 变化量
     * 这在序列压缩、采样或批处理优化中使用
     * 
     * 例如：seq_div(seq_id=1, p0=4, p1=8, d=2)
     * 
     * Before seq_div():
     * ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:4│pos:5│pos:6│pos:7│pos:8│pos:9│
     * │seq:1│seq:1│seq:1│seq:1│seq:1│seq:1│seq:2│seq:2│
     * │Δ:0  │Δ:+1 │Δ:+2 │Δ:-1 │Δ:+1 │Δ:0  │Δ:+2 │Δ:-1 │
     * └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
     *               ↑─ p0=4    p1=8 ─↑   <- 这个范围内的位置/2
     * 
     * After seq_div():
     * ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:2│pos:2│pos:3│pos:3│pos:8│pos:9│
     * │seq:1│seq:1│seq:1│seq:1│seq:1│seq:1│seq:2│seq:2│
     * │Δ:0  │Δ:+1 │Δ:0  │Δ:-4 │Δ:-2 │Δ:-4 │Δ:+2 │Δ:-1 │
     * └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
     *               ↑─ 4/2=2  5/2=2  6/2=3  7/2=3 ─↑
     * 
     * delta 计算详解：
     * - pos:4, Δ:+2 → pos:2, Δ:+2+(2-4)=0   (新位置-原位置=-2)
     * - pos:5, Δ:-1 → pos:2, Δ:-1+(2-5)=-4  (新位置-原位置=-3) 
     * - pos:6, Δ:+1 → pos:3, Δ:+1+(3-6)=-2  (新位置-原位置=-3)
     * - pos:7, Δ:0  → pos:3, Δ:0+(3-7)=-4   (新位置-原位置=-4)
     * 
     * 重要：delta += (new_pos - old_pos) 确保了 RoPE 计算的连续性
     *       实际 RoPE 位置 = pos + delta，在除法操作后保持正确
     */
    for (uint32_t i = 0; i < size; ++i) {
        // 检查该 cell 是否属于目标序列且在指定位置范围内
        if (cells[i].has_seq_id(seq_id) && cells[i].pos >= p0 && cells[i].pos < p1) {
            has_shift = true;  // 标记发生了位置偏移，触发后续 K-shift

            {
                llama_pos p_old = cells[i].pos;  // 保存原始位置
                cells[i].pos   /= d;             // 位置除法缩放
                cells[i].delta += cells[i].pos - p_old;  // 累积偏移差值
                
                // delta 变化 = 新位置 - 原位置
                // 这确保了 RoPE 计算中 (pos + delta) 的连续性
                // 例如：原来 pos=6,delta=+1 → RoPE_pos=7
                //       除法后 pos=3,delta=-2 → RoPE_pos=1 (不连续！)
                //       修正后 pos=3,delta=-2 → RoPE_pos=1 (需要额外处理)
            }
        }
    }
}

llama_pos llama_kv_cache_unified::seq_pos_min(llama_seq_id seq_id) const {
    /*
     * cells 最小位置查找 - 查找指定序列的最小 token 位置：
     * 
     * 遍历所有 cells，找到属于指定序列的 tokens 中位置最小的一个
     * 用于确定序列的起始位置或范围检查
     * 
     * 查找过程示例：
     * ┌─────┬─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:3│pos:1│pos:5│pos:2│pos:4│
     * │seq:1│seq:2│seq:1│seq:1│seq:3│seq:1│  <- 查找seq:1的最小位置
     * │Δ:0  │Δ:+1 │Δ:-1 │Δ:+2 │Δ:0  │Δ:+1 │
     * └─────┴─────┴─────┴─────┴─────┴─────┘
     *   ↑           ↑     ↑           ↑
     * seq:1      seq:1  seq:1       seq:1
     * pos:0      pos:1  pos:5       pos:4
     * 
     * result = min(0, 1, 5, 4) = 0
     * 
     * 注意：
     * 1. 只考虑 pos 值，不考虑 delta（delta 是偏移修正）
     * 2. 如果序列不存在，返回 -1
     * 3. 用于序列范围验证和窗口管理
     */
    llama_pos result = std::numeric_limits<llama_pos>::max();

    // 遍历所有 cells，寻找属于指定序列的最小位置
    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::min(result, cells[i].pos);
            // 注意：使用 pos 而不是 pos + delta，因为这是逻辑位置查找
        }
    }

    // 如果没有找到该序列的任何 token，返回 -1
    if (result == std::numeric_limits<llama_pos>::max()) {
        result = -1;
    }

    return result;
}

llama_pos llama_kv_cache_unified::seq_pos_max(llama_seq_id seq_id) const {
    /*
     * cells 最大位置查找 - 查找指定序列的最大 token 位置：
     * 
     * 遍历所有 cells，找到属于指定序列的 tokens 中位置最大的一个
     * 用于确定序列的结束位置或长度计算
     * 
     * 查找过程示例（续上例）：
     * ┌─────┬─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:3│pos:1│pos:5│pos:2│pos:4│
     * │seq:1│seq:2│seq:1│seq:1│seq:3│seq:1│  <- 查找seq:1的最大位置
     * │Δ:0  │Δ:+1 │Δ:-1 │Δ:+2 │Δ:0  │Δ:+1 │
     * └─────┴─────┴─────┴─────┴─────┴─────┘
     *   ↑           ↑     ↑           ↑
     * seq:1      seq:1  seq:1       seq:1
     * pos:0      pos:1  pos:5       pos:4
     * 
     * result = max(0, 1, 5, 4) = 5
     * 
     * 序列范围：seq:1 的 tokens 分布在位置 [0, 5] 范围内
     * 序列长度估算：max_pos - min_pos + 1 = 5 - 0 + 1 = 6 个位置跨度
     * 
     * 应用场景：
     * 1. 序列长度计算和验证
     * 2. 注意力窗口边界确定
     * 3. 缓存容量和使用率分析
     */
    llama_pos result = -1;

    // 遍历所有 cells，寻找属于指定序列的最大位置
    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::max(result, cells[i].pos);
            // 注意：使用 pos 而不是 pos + delta，因为这是逻辑位置查找
        }
    }

    // 如果没有找到该序列的任何 token，返回 -1
    return result;
}

void llama_kv_cache_unified::restore() {
    /*
     * cells 状态恢复操作 - 回滚到备份状态：
     * 
     * 从 recovery 备份中恢复 cells 状态，撤销之前的分配或修改操作
     * 同时正确维护 used 计数器和 delta 状态
     * 
     * 恢复过程示例：
     * 
     * Current state (操作失败后):
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:5│pos:6│pos:7│  <- 新分配但需要回滚
     * │seq:1│seq:1│seq:2│seq:2│seq:3│
     * │Δ:0  │Δ:+1 │Δ:0  │Δ:0  │Δ:0  │
     * └─────┴─────┴─────┴─────┴─────┘
     * 
     * Backup in recovery:
     * recovery.cells[2] = {pos:-1, seq_id:{}, delta:old_value}
     * recovery.cells[3] = {pos:-1, seq_id:{}, delta:old_value}
     * recovery.cells[4] = {pos:-1, seq_id:{}, delta:old_value}
     * 
     * After restore():
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:-│pos:-│pos:-│  <- 恢复到分配前状态
     * │seq:1│seq:1│empty│empty│empty│
     * │Δ:0  │Δ:+1 │Δ:old│Δ:old│Δ:old│  <- delta 也恢复到备份值
     * └─────┴─────┴─────┴─────┴─────┘
     * 
     * 重要：delta 的恢复确保了位置偏移历史的正确性，
     *       避免 RoPE 计算中的不一致性
     */
    for (const auto & [id, cell] : recovery.cells) {
        // TODO: move to new `struct kv_cells`
        
        // 正确维护 used 计数器
        const bool is_empty0 = cells[id].is_empty();
        const bool is_empty1 = cell.is_empty();

        if (!is_empty0 && is_empty1) {
            used--;  // 当前占用 -> 恢复为空闲
        } else if (is_empty0 && !is_empty1) {
            used++;  // 当前空闲 -> 恢复为占用
        }

        // 恢复完整的 cell 状态（包括 pos, seq_id, delta）
        cells[id] = cell;
        // 注意：delta 也被恢复，保持位置偏移历史的一致性
    }

    recovery.clear();  // 清空恢复信息
}

void llama_kv_cache_unified::commit() {
    if (recovery.cells.empty()) {
        LLAMA_LOG_WARN("%s: the recovery information upon a commit was empty - might indicate a bug (ref: %s)\n",
                __func__, "https://github.com/ggml-org/llama.cpp/pull/13194");
        return;
    }

    recovery.clear();
}

bool llama_kv_cache_unified::update(llama_context & lctx) {
    bool need_reserve = false;

    auto * sched = lctx.get_sched();

    if (has_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        // apply K-shift if needed
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched);

            auto * gf = lctx.graph_init();

            auto res = build_graph_shift(lctx.get_cparams(), lctx.get_ctx_compute(), gf);

            ggml_backend_sched_alloc_graph(sched, gf);

            res->set_inputs(nullptr);

            lctx.graph_compute(gf, false);

            need_reserve = true;
        }

        /*
         * delta 重置操作 - K-shift 完成后的清理：
         * 
         * K-shift 操作将所有累积的位置偏移应用到 K 张量的 RoPE 计算中，
         * 完成后需要清零所有 cells 的 delta，为下一轮偏移做准备
         * 
         * Before K-shift (delta 清零前):
         * ┌─────┬─────┬─────┬─────┬─────┐
         * │pos:0│pos:2│pos:3│pos:1│pos:4│
         * │seq:1│seq:1│seq:1│seq:2│seq:2│
         * │Δ:+1 │Δ:-2 │Δ:+3 │Δ:-1 │Δ:+2 │  <- 累积的位置偏移量
         * └─────┴─────┴─────┴─────┴─────┘
         *        ↓ K-shift 应用这些偏移到 RoPE 计算
         * 
         * After K-shift (delta 清零后):
         * ┌─────┬─────┬─────┬─────┬─────┐
         * │pos:0│pos:2│pos:3│pos:1│pos:4│
         * │seq:1│seq:1│seq:1│seq:2│seq:2│
         * │Δ:0  │Δ:0  │Δ:0  │Δ:0  │Δ:0  │  <- 所有 delta 重置为 0
         * └─────┴─────┴─────┴─────┴─────┘
         * 
         * 重要说明：
         * 1. K-shift 操作通过 RoPE 将 delta 偏移"烧入"到 K 张量中
         * 2. 清零 delta 后，pos 仍保持当前值，但偏移历史被清除
         * 3. 后续的 seq_add/seq_div 操作将从 delta=0 开始重新累积
         * 4. 这确保了 RoPE 计算的正确性和一致性
         */
        {
            has_shift = false;

            // 清零所有 cells 的 delta，因为 K-shift 已经应用了偏移
            for (uint32_t i = 0; i < size; ++i) {
                cells[i].delta = 0;  // 重置位置偏移累积量
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

    return need_reserve;
}

void llama_kv_cache_unified::defrag_sched(float thold) {
    // - do not defrag small contexts (i.e. < 2048 tokens)
    // - count the padding towards the number of used tokens
    const float fragmentation = n >= 2048 ? std::max(0.0f, 1.0f - (float(used + n_pad)/n)) : 0.0f;

    // queue defragmentation for next llama_kv_cache_update
    if (fragmentation > thold) {
        LLAMA_LOG_DEBUG("%s: fragmentation: %.2f - requesting defrag\n", __func__, fragmentation);

        do_defrag = true;
    }
}

void llama_kv_cache_unified::set_full() {
    n = size;

    // when simulating a full KV cache, the specific value of the "head" pointer is not important because it does not
    //   affect the shapes of the tensors in the compute graph - it only affects the offsets of the K/V views.
    //   we should only guarantee that the head position won't cause out-of-bounds view of the K, V tensors, so
    //   setting it to 0 is the simplest way to achieve that
    // ref: https://github.com/ggml-org/llama.cpp/issues/13359
    head = 0;
}

llama_sbatch llama_kv_cache_unified::sbatch_init(const llama_batch & batch, bool logits_all) {
    return llama_sbatch(batch, hparams.n_embd, true, logits_all);
}

llama_ubatch llama_kv_cache_unified::ubatch_next(llama_sbatch & sbatch, uint32_t n_ubatch, bool embd_pooled) const {
    GGML_UNUSED(embd_pooled);
    return sbatch.split_simple(n_ubatch);
}

bool llama_kv_cache_unified::find_slot(const llama_ubatch & ubatch) {
    const uint32_t n_tokens = ubatch.n_tokens;

    // if we have enough unused cells before the current head ->
    //   better to start searching from the beginning of the cache, hoping to fill it
    if (head > used + 2*ubatch.n_tokens) {
        head = 0;
    }

    // otherwise, one cell per token.

    if (n_tokens > size) {
        LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %d\n", __func__, n_tokens, size);
        return false;
    }

// #define FIND_SLOT_DEBUG 1
#if FIND_SLOT_DEBUG
    LLAMA_LOG_WARN("begin: n = %5d, used = %5d, head = %5d, n_swa = %5d\n", n, used, head, n_swa);

    // for debugging
    {
        std::string ss;
        if (n_swa > 0) {
            for (uint32_t i = 0; i < size; ++i) {
                if (cells[i].pos == -1) {
                    ss += '.';
                } else {
                    ss += std::to_string(*cells[i].seq_id.begin());
                }
                if (i%256 == 255) {
                    ss += '\n';
                }
            }
        }
        LLAMA_LOG_WARN("\n%s\n", ss.c_str());
    }
#endif

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
                head     += i + 1;
                n_tested += i + 1;
                break;
            }
        }

        if (found) {
            break;
        }

        if (n_tested >= size) {
            //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
            return false;
        }
    }

    /*
     * cells 槽位分配和恢复备份机制：
     * 
     * 在分配新的 token 槽位时，需要备份原始状态以支持回滚操作
     * 同时设置新的位置和序列信息
     * 
     * 分配过程示例：
     * 
     * Before allocation (head=2, n_tokens=3):
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:-│pos:-│pos:-│  <- head指向第一个空闲槽
     * │seq:1│seq:1│empty│empty│empty│
     * │Δ:0  │Δ:+1 │Δ:?  │Δ:?  │Δ:?  │
     * └─────┴─────┴─────┴─────┴─────┘
     *               ↑─ head=2, 分配3个tokens
     * 
     * Backup to recovery:
     * recovery.cells[2] = {pos:-1, seq_id:{}, delta:old_value}
     * recovery.cells[3] = {pos:-1, seq_id:{}, delta:old_value}  
     * recovery.cells[4] = {pos:-1, seq_id:{}, delta:old_value}
     * 
     * After allocation:
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:1│pos:5│pos:6│pos:7│  <- 新分配的token位置
     * │seq:1│seq:1│seq:2│seq:2│seq:3│  <- 新的序列ID
     * │Δ:0  │Δ:+1 │Δ:0  │Δ:0  │Δ:0  │  <- delta重置为0（新分配）
     * └─────┴─────┴─────┴─────┴─────┘
     * 
     * 重要：新分配的 cells 的 delta 自动初始化为 0，
     *       开始新的位置偏移累积周期
     */
    for (uint32_t i = 0; i < n_tokens; ++i) {
        // 备份原始状态到 recovery，支持后续回滚操作
        // remember the original state
        if (recovery.cells.find(head + i) == recovery.cells.end()) {
            recovery.cells[head + i] = cells[head + i];
        }

        // 设置新分配 cell 的位置信息
        cells[head + i].pos = ubatch.pos[i];
        // delta 在 kv_cell 构造或清空时自动初始化为 0

        // 设置序列 ID 信息（支持多序列共享）
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; j++) {
            cells[head + i].seq_id.insert(ubatch.seq_id[i][j]);
        }
        // 注意：新分配的 cell 的 delta = 0，开始新的偏移累积周期
    }

    used += n_tokens;

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // after enough generations, the benefit from this heuristic disappears
    // if we start defragmenting the cache, the benefit from this will be more important
    n = std::min(size, std::max(n_pad, GGML_PAD(cell_max(), n_pad)));

#ifdef FIND_SLOT_DEBUG
    // 🐛 调试信息：显示unified缓存的详细状态
    // 🛡️ 这不会影响mixed缓存的运行，因为mixed缓存有自己的find_slot实现
    // Debug info: show detailed status of unified cache
    // This won't affect mixed cache operation as mixed cache has its own find_slot implementation
    LLAMA_LOG_WARN("end:   n = %5d, used = %5d, head = %5d, n_swa = %5d, n_pad = %5d, cell_max = %5d, size = %5d\n", n, used, head, n_swa, n_pad, cell_max(), size);
#endif

    return true;
}

bool llama_kv_cache_unified::get_can_shift() const {
    return true;
}

uint32_t llama_kv_cache_unified::get_n() const {
    return n;
}

uint32_t llama_kv_cache_unified::get_size() const {
    return size;
}

ggml_tensor * llama_kv_cache_unified::get_k(ggml_context * ctx, int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    return ggml_view_3d(ctx, k,
            hparams.n_embd_head_k, hparams.n_head_kv(il), n,
            ggml_row_size(k->type, hparams.n_embd_head_k),
            ggml_row_size(k->type, hparams.n_embd_k_gqa(il)),
            0);
}

ggml_tensor * llama_kv_cache_unified::get_v(ggml_context * ctx, int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

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

ggml_tensor * llama_kv_cache_unified::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    const int64_t n_tokens = k_cur->ne[2];

    ggml_tensor * k_view = ggml_view_1d(ctx, k,
            n_tokens*hparams.n_embd_k_gqa(il),
            ggml_row_size(k->type, hparams.n_embd_k_gqa(il))*head);

    return ggml_cpy(ctx, k_cur, k_view);
}

ggml_tensor * llama_kv_cache_unified::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const int64_t n_tokens = v_cur->ne[2];

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

void llama_kv_cache_unified::prune_swa(llama_seq_id seq_id, llama_pos pmin, llama_pos pmax) {
    // no pruning is needed when the cache does not use SWA
    GGML_ASSERT(swa_type != LLAMA_SWA_TYPE_NONE && "do not prune non-SWA cache");

    int n_attended = 0;

    for (uint32_t i = 0; i < size; ++i) {
        const llama_pos p0 = cells[i].pos;

        if (p0 <= pmin && !is_masked_swa(p0, pmin)) {
            n_attended++;
        }

        if (is_masked_swa(p0, pmax)) {
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
            }
        }
    }

    if (n_attended < std::min<int>(n_swa, pmin)) {
        LLAMA_LOG_WARN("%s: partial SWA cache detected - possible loss of information, pmin = %d, n_attended = %d, n_swa = %d\n", __func__, pmin, n_attended, n_swa);
    }
}

void llama_kv_cache_unified::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
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

                    // apply SWA if any
                    masked = masked || (is_masked_swa(p0, p1));

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

void llama_kv_cache_unified::set_input_k_shift(ggml_tensor * dst) const {
    /*
     * 设置 K-shift 输入张量 - delta 传递给 RoPE 计算：
     * 
     * 将所有 cells 的 delta 值复制到输入张量，供 K-shift 操作使用
     * K-shift 操作会将这些偏移量应用到 K 张量的 RoPE 计算中
     * 
     * cells delta 到 tensor 的映射：
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │pos:0│pos:2│pos:3│pos:1│pos:4│  <- cells 状态
     * │seq:1│seq:1│seq:1│seq:2│seq:2│
     * │Δ:+1 │Δ:-2 │Δ:+3 │Δ:-1 │Δ:+2 │  <- 累积的位置偏移
     * └─────┴─────┴─────┴─────┴─────┘
     *        ↓ 复制到 K-shift 输入张量
     * dst->data: [+1, -2, +3, -1, +2, 0, 0, ...] (int32_t array)
     *             ↑   ↑   ↑   ↑   ↑   ↑
     *           cell0 1   2   3   4  unused...
     * 
     * RoPE 计算中的使用：
     * for each cell i:
     *   rope_position = cells[i].pos + dst->data[i]  // pos + delta
     *   apply_rope(K_tensor[i], rope_position)
     * 
     * 关键作用：
     * 1. 传递累积的位置偏移给 RoPE 计算
     * 2. 确保旋转位置编码的正确性
     * 3. 支持序列位置的动态调整（插入、删除、缩放等）
     * 4. K-shift 后，这些 delta 值会被清零
     */
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    // 将每个 cell 的 delta 复制到输入张量
    for (uint32_t i = 0; i < size; ++i) {
        data[i] = cells[i].delta;  // 传递位置偏移给 K-shift 操作
    }
    // 注意：K-shift 操作完成后，这些 delta 值会在 update() 中被清零
}

void llama_kv_cache_unified::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(!ubatch->equal_seqs); // TODO: use ubatch->n_seqs instead of failing

    int32_t * data = (int32_t *) dst->data;

    const int64_t n_kv = n;

    for (int h = 0; h < 1; ++h) {
        for (int j = 0; j < n_tokens; ++j) {
            for (int i = 0; i < n_kv; ++i) {
                data[h*(n_kv*n_tokens) + j*n_kv + i] = llama_relative_position_bucket(cells[i].pos, ubatch->pos[j], hparams.n_rel_attn_bkts, false);
            }
        }
    }
}

size_t llama_kv_cache_unified::total_size() const {
    size_t size = 0;

    for (const auto & buf : bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_kv_cache_unified::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & layer : layers) {
        size_k_bytes += ggml_nbytes(layer.k);
    }

    return size_k_bytes;
}

size_t llama_kv_cache_unified::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & layer : layers) {
        size_v_bytes += ggml_nbytes(layer.v);
    }

    return size_v_bytes;
}

ggml_tensor * llama_kv_cache_unified::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * cur,
                ggml_tensor * shift,
                ggml_tensor * factors,
                      float   freq_base,
                      float   freq_scale) const {
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;

    const auto & yarn_ext_factor = cparams.yarn_ext_factor;
    const auto & yarn_beta_fast  = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow  = cparams.yarn_beta_slow;

    const auto & n_rot     = hparams.n_rot;
    const auto & rope_type = hparams.rope_type;

    // See llm_build_deepseek2() for why attn_factor has to be scaled for YaRN RoPE to work correctly.
    // See https://github.com/ggerganov/llama.cpp/discussions/7416 for detailed explanation.
    const float yarn_attn_factor = model.arch == LLM_ARCH_DEEPSEEK2 ? 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale)) : cparams.yarn_attn_factor;

    ggml_tensor * tmp;

    if (ggml_is_quantized(cur->type)) {
        // dequantize to f32 -> RoPE -> quantize back
        tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);

        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        tmp = ggml_cpy(ctx, tmp, cur);
    } else {
        // we rotate only the first n_rot dimensions
        tmp = ggml_rope_ext_inplace(ctx, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    llm_graph_input_k_shift(const llama_kv_cache_unified * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // I32 [kv_size]

    const llama_kv_cache_unified * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift) {
        kv_self->set_input_k_shift(k_shift);
    }
}

llm_graph_result_ptr llama_kv_cache_unified::build_graph_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_cgraph * gf) const {
    auto res = std::make_unique<llm_graph_result>();

    const auto & n_embd_head_k = hparams.n_embd_head_k;
  //const auto & n_embd_head_v = hparams.n_embd_head_v;

    //GGML_ASSERT(kv_self->size == n_ctx);

    auto inp = std::make_unique<llm_graph_input_k_shift>(this);

    inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cparams.n_ctx);
    ggml_set_input(inp->k_shift);

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        ggml_tensor * k =
            ggml_view_3d(ctx, layer.k,
                n_embd_head_k, n_head_kv, size,
                ggml_row_size(layer.k->type, n_embd_head_k),
                ggml_row_size(layer.k->type, n_embd_k_gqa),
                0);

        ggml_tensor * cur = build_rope_shift(cparams, ctx, k, inp->k_shift, rope_factors, freq_base_l, freq_scale_l);

        ggml_build_forward_expand(gf, cur);
    }

    res->add_input(std::move(inp));

    return res;
}

llm_graph_result_ptr llama_kv_cache_unified::build_graph_defrag(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_cgraph * gf) const {
    auto res = std::make_unique<llm_graph_result>();

    const auto & ids = defrag_info.ids;

#if 0
    // CPU defrag
    //
    // TODO: optimizations are possible:
    //       - multiple threads
    //       - avoid copying to the host memory when already there
    //
    // likely not worth the effort, as we have ggml_graph based defrag
    //

    const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa();
    const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa();

    const uint32_t kv_size = size;

    std::vector<uint8_t> buf_k;
    std::vector<uint8_t> buf_v;

    for (uint32_t il = 0; il < n_layer; ++il) {
        const size_t k_size_row = ggml_row_size(k_l[il]->type, n_embd_k_gqa);
        const size_t k_size     = ggml_row_size(k_l[il]->type, n_embd_k_gqa*kv_size);

        const size_t v_size_el = ggml_type_size(v_l[il]->type);
        const size_t v_size    = ggml_row_size (v_l[il]->type, n_embd_v_gqa*kv_size);

        buf_k.resize(k_size);
        buf_v.resize(v_size);

        ggml_backend_tensor_get(k_l[il], buf_k.data(), 0, buf_k.size());
        ggml_backend_tensor_get(v_l[il], buf_v.data(), 0, buf_v.size());

        // batch move [i, i+nm) to [id, id+nm)
        // note: cells can move only to a lower index
        for (uint32_t i = 0; i < n_kv; ++i) {
            const uint32_t id = ids[i];

            if (i == id || id == n_kv) {
                continue;
            }

            uint32_t nm = 1;

            while (i + nm < n_kv && ids[i + nm] == id + nm) {
                nm++;
            }

            // move keys
            {
                const int64_t os =  i*k_size_row;
                const int64_t od = id*k_size_row;

                memcpy(buf_k.data() + od, buf_k.data() + os, nm*k_size_row);
            }

            // move values (note: they are transposed)
            {
                const int64_t os =  i;
                const int64_t od = id;

                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    memcpy(buf_v.data() + (od + j*kv_size)*v_size_el, buf_v.data() + (os + j*kv_size)*v_size_el, nm*v_size_el);
                }
            }

            i += nm - 1;
        }

        ggml_backend_tensor_set(k_l[il], buf_k.data(), 0, buf_k.size());
        ggml_backend_tensor_set(v_l[il], buf_v.data(), 0, buf_v.size());
    }
#else
    for (uint32_t i = 0; i < ids.size(); ++i) {
        const uint32_t id = ids[i];

        if (i == id || id == ids.size()) {
            continue;
        }

        uint32_t nm = 1;

        while (i + nm < ids.size() && ids[i + nm] == id + nm) {
            nm++;
        }

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
            const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            ggml_tensor * view_k_src = ggml_view_2d(ctx, layer.k,
                    n_embd_k_gqa, nm,
                    ggml_row_size(layer.k->type, n_embd_k_gqa),
                    ggml_row_size(layer.k->type, n_embd_k_gqa*i));

            ggml_tensor * view_k_dst = ggml_view_2d(ctx, layer.k,
                    n_embd_k_gqa, nm,
                    ggml_row_size(layer.k->type, n_embd_k_gqa),
                    ggml_row_size(layer.k->type, n_embd_k_gqa*id));

            ggml_tensor * view_v_src;
            ggml_tensor * view_v_dst;

            if (cparams.flash_attn) {
                // NOTE: the V cache is not transposed when using flash attention
                view_v_src = ggml_view_2d(ctx, layer.v,
                        n_embd_v_gqa, nm,
                        ggml_row_size(layer.v->type, n_embd_v_gqa),
                        ggml_row_size(layer.v->type, n_embd_v_gqa*i));

                view_v_dst = ggml_view_2d(ctx, layer.v,
                        n_embd_v_gqa, nm,
                        ggml_row_size(layer.v->type, n_embd_v_gqa),
                        ggml_row_size(layer.v->type, n_embd_v_gqa*id));
            } else {
                view_v_src = ggml_view_2d(ctx, layer.v,
                        nm, n_embd_v_gqa,
                        ggml_row_size(layer.v->type, size),
                        ggml_row_size(layer.v->type, i));

                view_v_dst = ggml_view_2d(ctx, layer.v,
                        nm, n_embd_v_gqa,
                        ggml_row_size(layer.v->type, size),
                        ggml_row_size(layer.v->type, id));
            }

            ggml_build_forward_expand(gf, ggml_cpy(ctx, view_k_src, view_k_dst));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, view_v_src, view_v_dst));
        }

        i += nm - 1;
    }

    //LLAMA_LOG_INFO("gf->n_nodes = %d\n", gf->n_nodes);
#endif

    return res;
}

bool llama_kv_cache_unified::defrag_prepare(int32_t n_max_nodes) {
    const uint32_t n_layer = layers.size();

    const uint32_t n_kv   = cell_max();
    const uint32_t n_used = used;

    assert(n_used <= n_kv);

    //const int64_t t_start = ggml_time_us();

    // number of cells moved
    uint32_t n_moves = 0;

    // each move requires 6*n_layer tensors (see graph_build_kv_self_defrag)
    //   - source view, destination view, copy operation
    //   - x2 for keys and values
    //const uint32_t max_moves = max_nodes()/(6*n_layer);
    // TODO: tmp fix https://github.com/ggerganov/llama.cpp/issues/6685#issuecomment-2057579516
    const uint32_t max_moves = (n_max_nodes - 2*n_layer)/(6*n_layer);

    // determine which KV cells to move where
    //
    //  cell i moves to ids[i]
    //
    //  if ids[i] == i || ids[i] == n_kv, then cell i is not moved
    //
    auto & ids = defrag_info.ids;

    ids.clear();
    ids.resize(n_kv, n_kv);

    for (uint32_t i0 = 0; i0 < n_used; ++i0) {
        const auto & cell0 = cells[i0];

        if (!cell0.is_empty()) {
            ids[i0] = i0;

            continue;
        }

        // found a hole - fill it with data from the end of the cache

        uint32_t nh = 1;

        // determine the size of the hole
        while (i0 + nh < n_used && cells[i0 + nh].is_empty()) {
            nh++;
        }

        uint32_t nf = 0;
        uint32_t is = n_kv - 1;

        // starting from the end, find nh non-empty cells
        for (; is > i0; --is) {
            const auto & cell1 = cells[is];

            if (cell1.is_empty() || ids[is] != n_kv) {
                continue;
            }

            // non-empty cell which is not yet moved
            nf++;

            if (nf == nh) {
                break;
            }
        }

        // this can only happen if `n_used` is not accurate, which would be a bug
        GGML_ASSERT(nf == nh && "KV defrag bug: nf != nh");

        nf = 0;

        uint32_t i1 = is;

        // are we moving a continuous block of memory?
        bool cont = false;

        // should we stop searching for the next move?
        bool stop = false;

        // go back and move the nf cells to the hole
        for (; i1 < n_kv; ++i1) {
            auto & cell1 = cells[i1];

            if (cell1.is_empty() || ids[i1] != n_kv) {
                if (n_moves == max_moves) {
                    stop = true;
                    break;
                }

                cont = false;
                continue;
            }

            // this cell goes to (i0 + nf)
            ids[i1] = i0 + nf;

            // move the cell meta data
            cells[i0 + nf] = cell1;

            // clear the old cell and move the head there
            cell1 = kv_cell();
            head = n_used;

            if (!cont) {
                n_moves++;
                cont = true;
            }

            nf++;

            if (nf == nh) {
                break;
            }
        }

        if (stop || n_moves == max_moves) {
            break;
        }

        //LLAMA_LOG_INFO("(tmp log) KV defrag: move [%u, %u) to [%u, %u)\n", is, i1 + 1, i0, i0 + nh);

        i0 += nh - 1;
    }

    if (n_moves == 0) {
        return false;
    }

    LLAMA_LOG_DEBUG("%s: (tmp log) KV defrag cell moves: %u\n", __func__, n_moves);

    LLAMA_LOG_DEBUG("%s: expected gf nodes: %u\n", __func__, 6*n_moves*n_layer);

    return true;
}

uint32_t llama_kv_cache_unified::cell_max() const {
    for (uint32_t i = size; i > 0; --i) {
        const kv_cell & cell = cells[i - 1];

        if (cell.pos >= 0 && !cell.is_empty()) {
            return i;
        }
    }

    return 0;
}

bool llama_kv_cache_unified::is_masked_swa(llama_pos p0, llama_pos p1) const {
    if (p0 < 0) {
        return true;
    }

    switch (swa_type) {
        case LLAMA_SWA_TYPE_NONE:
            {
            } break;
        case LLAMA_SWA_TYPE_STANDARD:
            {
                if (p1 - p0 >= (int32_t) n_swa) {
                    return true;
                }
            } break;
        case LLAMA_SWA_TYPE_CHUNKED:
            {
                const llama_pos pos_chunk_start = (p1 / n_swa) * n_swa;

                if (p0 < pos_chunk_start) {
                    return true;
                }
            } break;
    }

    return false;
}

void llama_kv_cache_unified::state_write(llama_io_write_i & io, llama_seq_id seq_id) const {
    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges; // ranges, from inclusive, to exclusive
    uint32_t cell_count = 0;

    // Count the number of cells with the specified seq_id
    // Find all the ranges of cells with this seq id (or all, when -1)
    uint32_t cell_range_begin = size;
    for (uint32_t i = 0; i < size; ++i) {
        const auto & cell = cells[i];
        if ((seq_id == -1 && !cell.is_empty()) || cell.has_seq_id(seq_id)) {
            ++cell_count;
            if (cell_range_begin == size) {
                cell_range_begin = i;
            }
        } else {
            if (cell_range_begin != size) {
                cell_ranges.emplace_back(cell_range_begin, i);
                cell_range_begin = size;
            }
        }
    }
    if (cell_range_begin != size) {
        cell_ranges.emplace_back(cell_range_begin, size);
    }

    // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
    uint32_t cell_count_check = 0;
    for (const auto & range : cell_ranges) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    io.write(&cell_count, sizeof(cell_count));

    state_write_meta(io, cell_ranges, seq_id);
    state_write_data(io, cell_ranges);
}

void llama_kv_cache_unified::state_read(llama_io_read_i & io, llama_seq_id seq_id) {
    uint32_t cell_count;
    io.read_to(&cell_count, sizeof(cell_count));

    bool res = true;
    res = res && state_read_meta(io, cell_count, seq_id);
    res = res && state_read_data(io, cell_count);

    if (!res) {
        if (seq_id == -1) {
            clear();
        } else {
            seq_rm(seq_id, -1, -1);
        }
        throw std::runtime_error("failed to restore kv cache");
    }
}

void llama_kv_cache_unified::state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id) const {
    for (const auto & range : cell_ranges) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            const auto & cell = cells[i];
            const llama_pos pos      = cell.pos;
            const uint32_t  n_seq_id = seq_id == -1 ? cell.seq_id.size() : 0;

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id) {
                for (auto seq_id : cell.seq_id) {
                    io.write(&seq_id, sizeof(seq_id));
                }
            }
        }
    }
}

void llama_kv_cache_unified::state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const {
    const uint32_t v_trans = this->v_trans ? 1 : 0;
    const uint32_t n_layer = layers.size();

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));

    std::vector<uint8_t> tmp_buf;

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il) + hparams.n_embd_k_s();

        // Write key type
        const int32_t k_type_i = (int32_t)layer.k->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // Write row size of key
        const uint64_t k_size_row = ggml_row_size(layer.k->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // Read each range of cells of k_size length each into tmp_buf and write out
        for (const auto & range : cell_ranges) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * k_size_row;
            io.write_tensor(layer.k, range.first * k_size_row, buf_size);
        }
    }

    if (!v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

            // Write value type
            const int32_t v_type_i = (int32_t)layer.v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write row size of value
            const uint64_t v_size_row = ggml_row_size(layer.v->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // Read each range of cells of v_size length each into tmp_buf and write out
            for (const auto & range : cell_ranges) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * v_size_row;
                io.write_tensor(layer.v, range.first * v_size_row, buf_size);
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = size;

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

            // Write value type
            const int32_t v_type_i = (int32_t)layer.v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(layer.v->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // Write GQA embedding size
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // For each row, we get the element values of each cell
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                // Read each range of cells of v_size_el length each into tmp_buf and write out
                for (const auto & range : cell_ranges) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                    const size_t buf_size = range_size * v_size_el;
                    io.write_tensor(layer.v, src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_kv_cache_unified::state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id) {
    if (dest_seq_id != -1) {
        // single sequence

        seq_rm(dest_seq_id, -1, -1);

        llama_sbatch sbatch;
        llama_ubatch batch = sbatch.reserve_ubatch(cell_count, /* has_embd */ false);

        batch.n_tokens = cell_count;

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read_to(&pos,      sizeof(pos));
            io.read_to(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 0) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            batch.pos[i] = pos;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i] = &dest_seq_id;
        }

        if (!find_slot(batch)) {
            LLAMA_LOG_ERROR("%s: failed to find available cells in kv cache\n", __func__);
            return false;
        }

        commit();

        // DEBUG CHECK: kv.head should be our first cell, kv.head + cell_count - 1 should be our last cell (verify seq_id and pos values)
        // Assume that this is one contiguous block of cells
        GGML_ASSERT(head + cell_count <= size);
        GGML_ASSERT(cells[head].pos == batch.pos[0]);
        GGML_ASSERT(cells[head + cell_count - 1].pos == batch.pos[cell_count - 1]);
        GGML_ASSERT(cells[head].has_seq_id(dest_seq_id));
        GGML_ASSERT(cells[head + cell_count - 1].has_seq_id(dest_seq_id));
    } else {
        // whole KV cache restore

        if (cell_count > size) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear();

        for (uint32_t i = 0; i < cell_count; ++i) {
            kv_cell & cell = cells[i];

            llama_pos pos;
            uint32_t  n_seq_id;

            io.read_to(&pos,      sizeof(pos));
            io.read_to(&n_seq_id, sizeof(n_seq_id));

            cell.pos = pos;

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read_to(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, n_seq_max);
                    return false;
                }

                cell.seq_id.insert(seq_id);
            }
        }

        head = 0;
        used = cell_count;
    }

    return true;
}

bool llama_kv_cache_unified::state_read_data(llama_io_read_i & io, uint32_t cell_count) {
    uint32_t v_trans;
    uint32_t n_layer;

    io.read_to(&v_trans, sizeof(v_trans));
    io.read_to(&n_layer, sizeof(n_layer));

    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }
    if (cell_count > size) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, size);
        return false;
    }
    if (this->v_trans != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il) + hparams.n_embd_k_s();

        // Read type of key
        int32_t k_type_i_ref;
        io.read_to(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) layer.k->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read_to(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(layer.k->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            // Read and set the keys for the whole cell range
            ggml_backend_tensor_set(layer.k, io.read(cell_count * k_size_row), head * k_size_row, cell_count * k_size_row);
        }
    }

    if (!this->v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

            // Read type of value
            int32_t v_type_i_ref;
            io.read_to(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t)layer.v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read_to(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(layer.v->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                // Read and set the values for the whole cell range
                ggml_backend_tensor_set(layer.v, io.read(cell_count * v_size_row), head * v_size_row, cell_count * v_size_row);
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

            // Read type of value
            int32_t v_type_i_ref;
            io.read_to(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t)layer.v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read_to(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(layer.v->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // Read GQA embedding size
            uint32_t n_embd_v_gqa_ref;
            io.read_to(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            if (cell_count) {
                // For each row in the transposed matrix, read the values for the whole cell range
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    const size_t dst_offset = (head + j * size) * v_size_el;
                    ggml_backend_tensor_set(layer.v, io.read(cell_count * v_size_el), dst_offset, cell_count * v_size_el);
                }
            }
        }
    }

    return true;
}

//
// llama_kv_cache_unified_iswa
//

llama_kv_cache_unified_iswa::llama_kv_cache_unified_iswa(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   swa_full,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_batch,
                 uint32_t   n_pad) : hparams(model.hparams) {
    llama_kv_cache_unified::layer_filter_cb filter_base = [&](int32_t il) { return !model.hparams.is_swa(il); };
    llama_kv_cache_unified::layer_filter_cb filter_swa  = [&](int32_t il) { return  model.hparams.is_swa(il); };

    const uint32_t size_base = kv_size;

    uint32_t size_swa = std::min(size_base, GGML_PAD(hparams.n_swa*n_seq_max + n_batch, n_pad));

    // when using full-size SWA cache, we set the SWA cache size to be equal to the base cache size and disable pruning
    if (swa_full) {
        LLAMA_LOG_WARN("%s: using full-size SWA cache (ref: %s)\n",
                __func__, "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");

        size_swa = size_base;
        do_prune = false;
    }

    LLAMA_LOG_INFO("%s: creating non-SWA KV cache, size = %u cells\n", __func__, size_base);

    kv_base = std::make_unique<llama_kv_cache_unified>(
            model, std::move(filter_base), type_k, type_v,
            v_trans, offload, size_base, n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE);

    LLAMA_LOG_INFO("%s: creating     SWA KV cache, size = %u cells\n", __func__, size_swa);

    kv_swa = std::make_unique<llama_kv_cache_unified>(
            model, std::move(filter_swa), type_k, type_v,
            v_trans, offload, size_swa, n_seq_max, n_pad,
            hparams.n_swa, hparams.swa_type);
}

void llama_kv_cache_unified_iswa::clear() {
    kv_base->clear();
    kv_swa ->clear();
}

bool llama_kv_cache_unified_iswa::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    bool res = true;

    res = res & kv_base->seq_rm(seq_id, p0, p1);
    res = res & kv_swa ->seq_rm(seq_id, p0, p1);

    return res;
}

void llama_kv_cache_unified_iswa::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    kv_base->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    kv_swa ->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_unified_iswa::seq_keep(llama_seq_id seq_id) {
    kv_base->seq_keep(seq_id);
    kv_swa ->seq_keep(seq_id);
}

void llama_kv_cache_unified_iswa::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) {
    kv_base->seq_add(seq_id, p0, p1, delta);
    kv_swa ->seq_add(seq_id, p0, p1, delta);
}

void llama_kv_cache_unified_iswa::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    kv_base->seq_div(seq_id, p0, p1, d);
    kv_swa ->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_kv_cache_unified_iswa::seq_pos_min(llama_seq_id seq_id) const {
    // the base cache is a superset of the SWA cache, so we can just check the SWA cache
    return kv_swa->seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_unified_iswa::seq_pos_max(llama_seq_id seq_id) const {
    return kv_swa->seq_pos_max(seq_id);
}

void llama_kv_cache_unified_iswa::restore() {
    kv_base->restore();
    kv_swa ->restore();
}

void llama_kv_cache_unified_iswa::commit() {
    kv_base->commit();
    kv_swa ->commit();

    // slide the attention window, forgetting/pruning old tokens that are outside the window
    if (do_prune) {
        for (const auto & [seq_id, entry] : pending.pos) {
            kv_swa->prune_swa(seq_id, entry.pmin, entry.pmax);
        }

    }

    pending.clear();
}

bool llama_kv_cache_unified_iswa::update(llama_context & lctx) {
    bool res = true;

    res = res & kv_base->update(lctx);
    res = res & kv_swa ->update(lctx);

    return res;
}

void llama_kv_cache_unified_iswa::defrag_sched(float thold) {
    kv_base->defrag_sched(thold);
    kv_swa ->defrag_sched(thold);
}

void llama_kv_cache_unified_iswa::set_full() {
    kv_base->set_full();
    kv_swa ->set_full();
}

llama_sbatch llama_kv_cache_unified_iswa::sbatch_init(const llama_batch & batch, bool logits_all) {
    pending.clear();

    if (do_prune) {
        for (int i = 0; i < batch.n_tokens; ++i) {
            for (int s = 0; s < batch.n_seq_id[i]; ++s) {
                const llama_seq_id seq_id = batch.seq_id[i][s];
                const llama_pos    pos    = batch.pos[i];

                if (pending.pos.find(seq_id) == pending.pos.end()) {
                    pending.pos[seq_id].pmin = pos;
                    pending.pos[seq_id].pmax = pos;
                } else {
                    pending.pos[seq_id].pmin = std::min(pending.pos[seq_id].pmin, pos);
                    pending.pos[seq_id].pmax = std::max(pending.pos[seq_id].pmax, pos);
                }
            }
        }
    }

    return llama_sbatch(batch, hparams.n_embd, true, logits_all);
}

llama_ubatch llama_kv_cache_unified_iswa::ubatch_next(llama_sbatch & sbatch, uint32_t n_ubatch, bool embd_pooled) const {
    GGML_UNUSED(embd_pooled);
    return sbatch.split_simple(n_ubatch);
}

bool llama_kv_cache_unified_iswa::find_slot(const llama_ubatch & batch) {
    bool res = true;

    res = res & kv_base->find_slot(batch);
    res = res & kv_swa ->find_slot(batch);

    return res;
}

bool llama_kv_cache_unified_iswa::get_can_shift() const {
    return kv_base->get_size() == kv_swa->get_size();
}

void llama_kv_cache_unified_iswa::state_write(llama_io_write_i & io, llama_seq_id seq_id) const {
    kv_base->state_write(io, seq_id);
    kv_swa ->state_write(io, seq_id);
}

void llama_kv_cache_unified_iswa::state_read(llama_io_read_i & io, llama_seq_id seq_id) {
    kv_base->state_read(io, seq_id);
    kv_swa ->state_read(io, seq_id);
}

llama_kv_cache_unified * llama_kv_cache_unified_iswa::get_kv_base() const {
    return kv_base.get();
}

llama_kv_cache_unified * llama_kv_cache_unified_iswa::get_kv_swa() const {
    return kv_swa.get();
}

//
// llama_kv_cache_recurrent
//

llama_kv_cache_recurrent::llama_kv_cache_recurrent(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   offload,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max) : hparams(model.hparams), n_seq_max(n_seq_max) {
    const int32_t n_layer = hparams.n_layer;

    LLAMA_LOG_INFO("%s: kv_size = %u, n_seq_max = %u, type_k = '%s', type_v = '%s', n_layer = %d\n",
            __func__, kv_size, n_seq_max, ggml_type_name(type_k), ggml_type_name(type_v), n_layer);

    head = 0;
    size = kv_size;
    used = 0;

    cells.clear();
    cells.resize(kv_size);

    // create a context for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*n_layer*ggml_tensor_overhead()),
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

    k_l.reserve(n_layer);
    v_l.reserve(n_layer);

    for (int i = 0; i < n_layer; i++) {
        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(i) + hparams.n_embd_k_s();
        const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(i) + hparams.n_embd_v_s();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(i);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s, layer %3d: dev = %s\n", __func__, i, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        ggml_tensor * k = ggml_new_tensor_1d(ctx, type_k, n_embd_k_gqa*kv_size);
        ggml_tensor * v = ggml_new_tensor_1d(ctx, type_v, n_embd_v_gqa*kv_size);
        ggml_format_name(k, "cache_k_l%d", i);
        ggml_format_name(v, "cache_v_l%d", i);
        k_l.push_back(k);
        v_l.push_back(v);
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto it : ctx_map) {
        auto * buft = it.first;
        auto * ctx  = it.second;

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }
        ggml_backend_buffer_clear(buf, 0);
        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
        bufs.emplace_back(buf);
    }

    {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        LLAMA_LOG_INFO("%s: KV self size  = %7.2f MiB, K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f),
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
    }
}

void llama_kv_cache_recurrent::clear() {
    for (int32_t i = 0; i < (int32_t) size; ++i) {
        cells[i].pos = -1;
        cells[i].seq_id.clear();
        cells[i].src = -1;
        cells[i].tail = -1;
    }
    head = 0;
    used = 0;

    for (auto & buf : bufs) {
        ggml_backend_buffer_clear(buf.get(), 0);
    }
}

bool llama_kv_cache_recurrent::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    uint32_t new_head = size;

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // models like Mamba or RWKV can't have a state partially erased
    if (seq_id >= (int64_t) size) {
        // could be fatal
        return false;
    }
    if (0 <= seq_id) {
        int32_t & tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            const kv_cell & cell = cells[tail_id];
            // partial intersection is invalid
            if ((0 < p0 && p0 <= cell.pos) || (0 < p1 && p1 <= cell.pos)) {
                return false;
            }
            // invalidate tails which will be cleared
            if (p0 <= cell.pos && cell.pos < p1) {
                tail_id = -1;
            }
        }
    } else {
        // seq_id is negative, then the range should include everything or nothing
        if (p0 != p1 && (p0 != 0 || p1 != std::numeric_limits<llama_pos>::max())) {
            return false;
        }
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
                cells[i].src = -1;
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

void llama_kv_cache_recurrent::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (seq_id_src == seq_id_dst) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if ((uint32_t) seq_id_dst < size && (uint32_t) seq_id_src < size) {
        kv_cell & tail_src = cells[seq_id_src];
        kv_cell & tail_dst = cells[seq_id_dst];
        if (tail_dst.tail >= 0) {
            // clear destination seq_id if it wasn't empty
            kv_cell & cell_dst = cells[tail_dst.tail];

            cell_dst.seq_id.erase(seq_id_dst);
            tail_dst.tail = -1;
            if (cell_dst.seq_id.empty()) {
                cell_dst.pos = -1;
                cell_dst.src = -1;
                used -= 1;
            }
        }
        if (tail_src.tail >= 0) {
            kv_cell & cell_src = cells[tail_src.tail];

            cell_src.seq_id.insert(seq_id_dst);
            tail_dst.tail = tail_src.tail;
        }
    }
}

void llama_kv_cache_recurrent::seq_keep(llama_seq_id seq_id) {
    uint32_t new_head = size;

    for (uint32_t i = 0; i < size; ++i) {
        if ((llama_seq_id) i != seq_id) {
            cells[i].tail = -1;
        }

        if (!cells[i].has_seq_id(seq_id)) {
            if (cells[i].pos >= 0) {
                used--;
            }

            cells[i].pos = -1;
            cells[i].src = -1;
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

void llama_kv_cache_recurrent::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) {
    if (delta == 0) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the
    if (p0 == p1) {
        return;
    }

    // for Mamba-like or RWKV models, only the pos needs to be shifted
    if (0 <= seq_id && seq_id < (int64_t) size) {
        const int32_t tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            kv_cell & cell = cells[tail_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos += delta;
            }
        }
    }
}

void llama_kv_cache_recurrent::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    // for Mamba-like or RWKV models, only the pos needs to be changed
    if (0 <= seq_id && seq_id < (int64_t) size) {
        const int32_t tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            kv_cell & cell = cells[tail_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos /= d;
            }
        }
    }
}

llama_pos llama_kv_cache_recurrent::seq_pos_min(llama_seq_id seq_id) const {
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

llama_pos llama_kv_cache_recurrent::seq_pos_max(llama_seq_id seq_id) const {
    llama_pos result = -1;

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::max(result, cells[i].pos);
        }
    }

    return result;
}

void llama_kv_cache_recurrent::restore() {
    if (pending.ranges.empty()) {
        return;
    }

    seq_rm(-1, -1, -1);
}

void llama_kv_cache_recurrent::commit() {
    pending.ranges.clear();
}

bool llama_kv_cache_recurrent::update(llama_context & ctx) {
    GGML_UNUSED(ctx);
    return false;
}

void llama_kv_cache_recurrent::defrag_sched(float thold) {
    GGML_UNUSED(thold);
    // noop
}

void llama_kv_cache_recurrent::set_full() {
    n = size;
    head = 0;
}

llama_sbatch llama_kv_cache_recurrent::sbatch_init(
        const llama_batch & batch,
        bool logits_all) {
    return llama_sbatch(batch, hparams.n_embd, false, logits_all);
}

llama_ubatch llama_kv_cache_recurrent::ubatch_next(llama_sbatch & sbatch, uint32_t n_ubatch, bool embd_pooled) const {
    if (embd_pooled) {
        // Pooled embeddings cannot be split across ubatches (yet)
        return sbatch.split_seq(n_ubatch);
    }

    return sbatch.split_equal(n_ubatch);
}

bool llama_kv_cache_recurrent::find_slot(
       const llama_ubatch & ubatch) {
    const uint32_t n_tokens = ubatch.n_tokens;
    const uint32_t n_seqs   = ubatch.n_seqs;

    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    // if we have enough unused cells before the current head ->
    //   better to start searching from the beginning of the cache, hoping to fill it
    if (head > used + 2*n_tokens) {
        head = 0;
    }

    // For recurrent state architectures (like Mamba or RWKV),
    // each cache cell can store the state for a whole sequence.
    // A slot should be always be contiguous.

    // can only process batches with an equal number of new tokens in each sequence
    GGML_ASSERT(ubatch.equal_seqs);

    int32_t min = size - 1;
    int32_t max = 0;

    // everything should fit if all seq_ids are smaller than the max
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t n_seq_id = ubatch.n_seq_id[s];
        for (uint32_t j = 0; j < n_seq_id; ++j) {
            const llama_seq_id seq_id = ubatch.seq_id[s][j];

            if (seq_id < 0 || (uint32_t) seq_id >= size) {
                // too big seq_id
                // TODO: would it be possible to resize the cache instead?
                LLAMA_LOG_ERROR("%s: seq_id=%d >= n_seq_max=%u Try using a bigger --parallel value\n", __func__, seq_id, n_seq_max);
                return false;
            }
            if (j > 0) {
                kv_cell & seq = cells[seq_id];
                if (seq.tail >= 0) {
                    kv_cell & cell = cells[seq.tail];
                    // clear cells from seq_ids that become shared
                    // (should not normally happen, but let's handle it anyway)
                    cell.seq_id.erase(seq_id);
                    seq.tail = -1;
                    if (cell.seq_id.empty()) {
                        cell.pos = -1;
                        cell.src = -1;
                        used -= 1;
                    }
                }
            }
        }
    }

#ifndef NDEBUG
    {
        std::vector<int32_t> tails_verif;
        tails_verif.assign(size, -1);
        for (uint32_t i = 0; i < size; ++i) {
            kv_cell & cell = cells[i];
            for (llama_seq_id seq_id : cell.seq_id) {
                if (tails_verif[seq_id] != -1) {
                    LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tails_verif[seq_id]);
                }
                tails_verif[seq_id] = i;
            }
        }
        for (uint32_t i = 0; i < size; ++i) {
            if (tails_verif[i] != cells[i].tail) {
                LLAMA_LOG_ERROR("%s: wrong tail for seq_id %d, (%d instead of %d)\n", __func__, i, cells[i].tail, tails_verif[i]);
            }
        }
    }
#endif

    // find next empty cell
    uint32_t next_empty_cell = head;

    for (uint32_t i = 0; i < size; ++i) {
        if (next_empty_cell >= size) { next_empty_cell -= size; }
        kv_cell & cell = cells[next_empty_cell];
        if (cell.is_empty()) { break; }
        next_empty_cell += 1;
    }

    // find usable cell range
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const llama_seq_id seq_id = ubatch.seq_id[s][0];
        kv_cell & seq_meta = cells[seq_id];
        bool has_cell = false;
        if (seq_meta.tail >= 0) {
            kv_cell & cell = cells[seq_meta.tail];
            GGML_ASSERT(cell.has_seq_id(seq_id));
            // does this seq_id "own" the cell?
            if (cell.seq_id.size() == 1) { has_cell = true; }
        }
        if (!has_cell) {
            kv_cell & empty_cell = cells[next_empty_cell];
            GGML_ASSERT(empty_cell.is_empty());
            // copy old tail into the empty cell
            if (seq_meta.tail >= 0) {
                kv_cell & orig_cell = cells[seq_meta.tail];
                empty_cell.pos = orig_cell.pos;
                empty_cell.src = orig_cell.src;
                orig_cell.seq_id.erase(seq_id);
                empty_cell.seq_id.insert(seq_id); // will be overwritten
            }
            seq_meta.tail = next_empty_cell;
            // find next empty cell
            if (s + 1 < n_seqs) {
                next_empty_cell += 1;
                for (uint32_t i = 0; i < size; ++i) {
                    if (next_empty_cell >= size) { next_empty_cell -= size; }
                    kv_cell & cell = cells[next_empty_cell];
                    if (cell.is_empty()) { break; }
                    next_empty_cell += 1;
                }
            }
        }
        if (min > seq_meta.tail) { min = seq_meta.tail; }
        if (max < seq_meta.tail) { max = seq_meta.tail; }
    }

    // gather and re-order
    for (uint32_t s = 0; s < n_seqs; ++s) {
        int32_t dst_id = s + min;
        int32_t src_id = cells[ubatch.seq_id[s][0]].tail;
        if (dst_id != src_id) {
            kv_cell & dst_cell = cells[dst_id];
            kv_cell & src_cell = cells[src_id];

            std::swap(dst_cell.pos, src_cell.pos);
            std::swap(dst_cell.src, src_cell.src);
            std::swap(dst_cell.seq_id, src_cell.seq_id);

            // swap tails (assuming they NEVER overlap)
            for (const llama_seq_id seq_id : src_cell.seq_id) {
                cells[seq_id].tail = src_id;
            }
            for (const llama_seq_id seq_id : dst_cell.seq_id) {
                cells[seq_id].tail = dst_id;
            }
        }
    }

    // update the pos of the used seqs
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const llama_pos last_pos = ubatch.pos[n_seq_tokens * s + n_seq_tokens - 1];
        int32_t cell_id = s + min;
        kv_cell & cell = cells[cell_id];

        if (cell.pos >= 0 && last_pos != cell.pos + (llama_pos) n_seq_tokens) {
            // What should happen when the pos backtracks or skips a value?
            // Clearing the state mid-batch would require special-casing which isn't done.
            LLAMA_LOG_WARN("%s: non-consecutive token position %d after %d for sequence %d with %u new tokens\n",
                __func__, last_pos, cell.pos, ubatch.seq_id[s][0], n_seq_tokens);
        }
        cell.pos = last_pos;
        cell.seq_id.clear();
        for (int32_t j = 0; j < ubatch.n_seq_id[s]; ++j) {
            const llama_seq_id seq_id = ubatch.seq_id[s][j];
            cell.seq_id.insert(seq_id);
            cells[seq_id].tail = cell_id;
        }
    }

    // allow getting the range of used cells, from head to head + n
    head = min;
    n    = max - min + 1;
    used = std::count_if(cells.begin(), cells.end(),
        [](const kv_cell & cell){ return !cell.is_empty(); });

    // sanity check
    return n >= n_seqs;
}

bool llama_kv_cache_recurrent::get_can_shift() const {
    return false;
}

int32_t llama_kv_cache_recurrent::s_copy(int i) const {
    const uint32_t cell_id = i + head;

    //////////////////////////////////////////////
    // TODO: this should not mutate the KV cache !
    kv_cell & cell = const_cast<kv_cell &>(cells[cell_id]);

    // prevent out-of-bound sources
    if (cell.src < 0 || (uint32_t) cell.src >= size) {
        cell.src = cell_id;
    }

    int32_t res = cell.src;

    // TODO: do not mutate the KV cache
    // ensure copy only happens once
    if (cell.src != (int32_t) cell_id) {
        cell.src = cell_id;
    }

    return res;
}

float llama_kv_cache_recurrent::s_mask(int i) const {
    const uint32_t cell_id = i + head;

    //////////////////////////////////////////////
    // TODO: this should not mutate the KV cache !
    kv_cell & cell = const_cast<kv_cell &>(cells[cell_id]);

    float res = (float) (cell.src >= 0);

    // only clear once
    if (cell.src < 0) {
        cell.src = cell_id;
    }

    return res;
}

uint32_t llama_kv_cache_recurrent::cell_max() const {
    for (uint32_t i = size; i > 0; --i) {
        const kv_cell & cell = cells[i - 1];

        if (cell.pos >= 0 && !cell.is_empty()) {
            return i;
        }
    }

    return 0;
}

size_t llama_kv_cache_recurrent::total_size() const {
    size_t size = 0;
    for (const auto & buf : bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_kv_cache_recurrent::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & k : k_l) {
        size_k_bytes += ggml_nbytes(k);
    }

    return size_k_bytes;
}

size_t llama_kv_cache_recurrent::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & v : v_l) {
        size_v_bytes += ggml_nbytes(v);
    }

    return size_v_bytes;
}

void llama_kv_cache_recurrent::state_write(llama_io_write_i & io, llama_seq_id seq_id) const {
    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges; // ranges, from inclusive, to exclusive
    uint32_t cell_count = 0;

    // Count the number of cells with the specified seq_id
    // Find all the ranges of cells with this seq id (or all, when -1)
    uint32_t cell_range_begin = size;
    for (uint32_t i = 0; i < size; ++i) {
        const auto & cell = cells[i];
        if ((seq_id == -1 && !cell.is_empty()) || cell.has_seq_id(seq_id)) {
            ++cell_count;
            if (cell_range_begin == size) {
                cell_range_begin = i;
            }
        } else {
            if (cell_range_begin != size) {
                cell_ranges.emplace_back(cell_range_begin, i);
                cell_range_begin = size;
            }
        }
    }
    if (cell_range_begin != size) {
        cell_ranges.emplace_back(cell_range_begin, size);
    }

    // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
    uint32_t cell_count_check = 0;
    for (const auto & range : cell_ranges) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    io.write(&cell_count, sizeof(cell_count));

    state_write_meta(io, cell_ranges, seq_id);
    state_write_data(io, cell_ranges);
}

void llama_kv_cache_recurrent::state_read(llama_io_read_i & io, llama_seq_id seq_id) {
    uint32_t cell_count;
    io.read_to(&cell_count, sizeof(cell_count));

    bool res = true;

    res = res && state_read_meta(io, cell_count, seq_id);
    res = res && state_read_data(io, cell_count);

    if (!res) {
        if (seq_id == -1) {
            clear();
        } else {
            seq_rm(seq_id, -1, -1);
        }
        throw std::runtime_error("failed to restore kv cache");
    }
}

void llama_kv_cache_recurrent::state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id) const {
    for (const auto & range : cell_ranges) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            const auto & cell = cells[i];
            const llama_pos pos      = cell.pos;
            const uint32_t  n_seq_id = seq_id == -1 ? cell.seq_id.size() : 0;

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id) {
                for (auto seq_id : cell.seq_id) {
                    io.write(&seq_id, sizeof(seq_id));
                }
            }
        }
    }
}

void llama_kv_cache_recurrent::state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const {
    const uint32_t v_trans = 0;
    const uint32_t n_layer = hparams.n_layer;

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));

    std::vector<uint8_t> tmp_buf;

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (uint32_t il = 0; il < n_layer; ++il) {
        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il) + hparams.n_embd_k_s();

        // Write key type
        const int32_t k_type_i = (int32_t)k_l[il]->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // Write row size of key
        const uint64_t k_size_row = ggml_row_size(k_l[il]->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // Read each range of cells of k_size length each into tmp_buf and write out
        for (const auto & range : cell_ranges) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * k_size_row;
            io.write_tensor(k_l[il], range.first * k_size_row, buf_size);
        }
    }

    if (!v_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

            // Write value type
            const int32_t v_type_i = (int32_t)v_l[il]->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write row size of value
            const uint64_t v_size_row = ggml_row_size(v_l[il]->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // Read each range of cells of v_size length each into tmp_buf and write out
            for (const auto & range : cell_ranges) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * v_size_row;
                io.write_tensor(v_l[il], range.first * v_size_row, buf_size);
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = size;
        for (uint32_t il = 0; il < n_layer; ++il) {
            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

            // Write value type
            const int32_t v_type_i = (int32_t)v_l[il]->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(v_l[il]->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // Write GQA embedding size
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // For each row, we get the element values of each cell
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                // Read each range of cells of v_size_el length each into tmp_buf and write out
                for (const auto & range : cell_ranges) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                    const size_t buf_size = range_size * v_size_el;
                    io.write_tensor(v_l[il], src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_kv_cache_recurrent::state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id) {
    if (dest_seq_id != -1) {
        // single sequence

        seq_rm(dest_seq_id, -1, -1);

        llama_sbatch sbatch;
        llama_ubatch batch = sbatch.reserve_ubatch(cell_count, /* has_embd */ false);

        batch.n_tokens = cell_count;
        batch.n_seq_tokens = cell_count;
        batch.n_seqs = 1;

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read_to(&pos,      sizeof(pos));
            io.read_to(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 0) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            batch.pos[i] = pos;
        }
        batch.n_seq_id[0] = 1;
        batch.seq_id[0] = &dest_seq_id;
        if (!find_slot(batch)) {
            LLAMA_LOG_ERROR("%s: failed to find available cells in kv cache\n", __func__);
            return false;
        }
        commit();

        // DEBUG CHECK: kv.head should be our first cell, kv.head + cell_count - 1 should be our last cell (verify seq_id and pos values)
        // Assume that this is one contiguous block of cells
        GGML_ASSERT(head + cell_count <= size);
        GGML_ASSERT(cells[head].pos == batch.pos[0]);
        GGML_ASSERT(cells[head + cell_count - 1].pos == batch.pos[cell_count - 1]);
        GGML_ASSERT(cells[head].has_seq_id(dest_seq_id));
        GGML_ASSERT(cells[head + cell_count - 1].has_seq_id(dest_seq_id));
    } else {
        // whole KV cache restore

        if (cell_count > size) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear();

        for (uint32_t i = 0; i < cell_count; ++i) {
            kv_cell & cell = cells[i];

            llama_pos pos;
            uint32_t  n_seq_id;

            io.read_to(&pos,      sizeof(pos));
            io.read_to(&n_seq_id, sizeof(n_seq_id));

            cell.pos = pos;

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read_to(&seq_id, sizeof(seq_id));

                // TODO: llama_kv_cache_recurrent should have a notion of max sequences
                //if (seq_id < 0 || (uint32_t) seq_id >= llama_n_seq_max(ctx)) {
                if (seq_id < 0) {
                    //LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, llama_n_seq_max(ctx));
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, inf)\n", __func__, seq_id);
                    return false;
                }

                cell.seq_id.insert(seq_id);

                int32_t & tail = cells[seq_id].tail;
                if (tail != -1) {
                    LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tail);
                    return false;
                }
                tail = i;
            }
        }

        head = 0;
        used = cell_count;
    }

    for (uint32_t i = 0; i < cell_count; ++i) {
        uint32_t cell_id = head + i;
        // make sure the recurrent states will keep their restored state
        cells[cell_id].src = cell_id;
    }

    return true;
}

bool llama_kv_cache_recurrent::state_read_data(llama_io_read_i & io, uint32_t cell_count) {
    uint32_t v_trans;
    uint32_t n_layer;
    io.read_to(&v_trans, sizeof(v_trans));
    io.read_to(&n_layer, sizeof(n_layer));

    if (n_layer != hparams.n_layer) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, hparams.n_layer);
        return false;
    }
    if (cell_count > size) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, size);
        return false;
    }
    if (false != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (uint32_t il = 0; il < n_layer; ++il) {
        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il) + hparams.n_embd_k_s();

        // Read type of key
        int32_t k_type_i_ref;
        io.read_to(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) k_l[il]->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read_to(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(k_l[il]->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            // Read and set the keys for the whole cell range
            ggml_backend_tensor_set(k_l[il], io.read(cell_count * k_size_row), head * k_size_row, cell_count * k_size_row);
        }
    }

    if (!v_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

            // Read type of value
            int32_t v_type_i_ref;
            io.read_to(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t)v_l[il]->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read_to(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(v_l[il]->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                // Read and set the values for the whole cell range
                ggml_backend_tensor_set(v_l[il], io.read(cell_count * v_size_row), head * v_size_row, cell_count * v_size_row);
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (uint32_t il = 0; il < n_layer; ++il) {
            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

            // Read type of value
            int32_t v_type_i_ref;
            io.read_to(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t)v_l[il]->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read_to(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(v_l[il]->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // Read GQA embedding size
            uint32_t n_embd_v_gqa_ref;
            io.read_to(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            if (cell_count) {
                // For each row in the transposed matrix, read the values for the whole cell range
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    const size_t dst_offset = (head + j * size) * v_size_el;
                    ggml_backend_tensor_set(v_l[il], io.read(cell_count * v_size_el), dst_offset, cell_count * v_size_el);
                }
            }
        }
    }

    return true;
}
