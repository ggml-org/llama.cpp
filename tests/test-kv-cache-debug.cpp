// KV Cache Debug Tool - View cell allocation and usage
//
// This tool provides in-depth analysis of KV cache internals in llama.cpp, including:
// 1. Cache cell allocation and deallocation process
// 2. Dynamic changes in tensor dimensions with token count
// 3. Memory layout for concurrent multi-sequence storage
// 4. Impact of sequence operations on cache state
//
// KV Cache Fundamentals:
// - Each transformer layer has independent K(key) and V(value) caches
// - Cache is managed in "cells", each storing K/V vectors for one token
// - Supports concurrent storage of multiple sequences, each with independent position encoding
// - Fixed cache size triggers reorganization or overwrite when full
//
// ┌─────────────────────────────────────────────────────────────────┐
// │                    KV Cache Architecture                        │
// │                                                                 │
// │  Layer 0:  [K₀] [V₀]     Layer 1:  [K₁] [V₁]                   │
// │            ┌───┐ ┌───┐              ┌───┐ ┌───┐                │
// │  Cell 0 →  │ • │ │ • │    Cell 0 →  │ • │ │ • │                │
// │  Cell 1 →  │ • │ │ • │    Cell 1 →  │ • │ │ • │                │
// │  Cell 2 →  │ • │ │ • │    Cell 2 →  │ • │ │ • │                │
// │  ...       │...│ │...│    ...       │...│ │...│                │
// │  Cell N →  │ • │ │ • │    Cell N →  │ • │ │ • │                │
// │            └───┘ └───┘              └───┘ └───┘                │
// │                                                                 │
// │  Each cell stores one token's K/V vectors for attention         │
// └─────────────────────────────────────────────────────────────────┘

#include "../src/llama-arch.h"
#include "../src/llama-batch.h"
#include "../src/llama-hparams.h"
#include "../src/llama-impl.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-model.h"

#include "../common/common.h"
#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

/*- Helper Functions ----------------------------------------------------------*/

// Create minimal test model
// Constructs a simplified llama_model instance for KV cache testing
//
// ┌─────────────────────────────────────────────────────────────────┐
// │                     Model Construction                          │
// │                                                                 │
// │  Input Parameters:                                              │
// │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
// │  │    arch     │  │   n_layer   │  │   n_head    │             │
// │  │ (LLM_ARCH_  │  │ (# of       │  │ (attention  │             │
// │  │  LLAMA)     │  │  layers)    │  │  heads)     │             │
// │  └─────────────┘  └─────────────┘  └─────────────┘             │
// │         │                │                │                    │
// │         └────────────────┼────────────────┘                    │
// │                          ▼                                     │
// │                 ┌─────────────────┐                            │
// │                 │  llama_model    │                            │
// │                 │   instance      │                            │
// │                 └─────────────────┘                            │
// └─────────────────────────────────────────────────────────────────┘
static std::shared_ptr<llama_model> _make_model(
    llm_arch arch = LLM_ARCH_LLAMA,
    uint32_t n_layer = 2,
    uint32_t n_embd_head_k = 32,
    uint32_t n_embd_head_v = 32,
    uint32_t n_head = 4,
    uint32_t n_head_kv = 1) {

    llama_model_params params;
    params.tensor_buft_overrides = nullptr;
    std::shared_ptr<llama_model> model(new llama_model(params));
    model->hparams = llama_hparams();
    model->arch = arch;

    // Set model parameters that determine KV cache structure
    model->hparams.n_layer = n_layer;
    model->hparams.n_embd_head_k = n_embd_head_k;
    model->hparams.n_embd_head_v = n_embd_head_v;

    // Configure same head settings for all layers
    // In real models, different layers may have different head counts
    if (n_head > 0) {
        auto& n_head_arr = model->hparams.n_head_arr;
        std::fill(n_head_arr.begin(), n_head_arr.end(), n_head);
    }
    if (n_head_kv > 0) {
        auto& n_head_kv_arr = model->hparams.n_head_kv_arr;
        std::fill(n_head_kv_arr.begin(), n_head_kv_arr.end(), n_head_kv);
    }

    return model;
}

/*- Cache Debug Functions -----------------------------------------------------*/

// Print basic KV cache status
// Displays core metrics to understand memory usage
//
// ┌─────────────────────────────────────────────────────────────────┐
// │                    Cache Status Monitor                         │
// │                                                                 │
// │  Cache Metrics:                                                 │
// │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
// │  │ Total Size  │  │ Current N   │  │ Can Shift   │              │
// │  │ (capacity)  │  │ (active)    │  │ (K-shift)   │              │
// │  │     64      │  │     16      │  │    Yes      │              │
// │  └─────────────┘  └─────────────┘  └─────────────┘              │
// │                                                                 │
// │  Cache Layout:                                                  │
// │  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐              │
// │  │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │...│   │63 │              │
// │  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘              │
// │   ▲                       ▲                                     │
// │   │                       │                                     │
// │  head                   active                                  │
// └─────────────────────────────────────────────────────────────────┘
static void print_kv_cache_status(llama_kv_cache_unified * kv_cache, const std::string & title) {
    if (!kv_cache) {
        printf("%s: No KV cache available\n", title.c_str());
        return;
    }
    
    printf("\n╔════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                            %-46s ║\n", title.c_str());
    printf("╚════════════════════════════════════════════════════════════════════════════╝\n");

    // get_size(): Returns total cache capacity (cell count)
    // Fixed at creation time, doesn't change dynamically
    printf("Cache Size: %u cells\n", kv_cache->get_size());
    
    // get_n(): Returns current active cache size
    // Grows with token additions, affects attention computation range
    // Note: Not equal to actual cell count, but attention window size
    printf("Current N (active): %u\n", kv_cache->get_n());
    
    // get_can_shift(): Indicates if cache supports K-shift operation
    // K-shift is an optimization allowing position encoding adjustment
    printf("Can Shift: %s\n", kv_cache->get_can_shift() ? "Yes" : "No");
    
    // Note: total_size(), size_k_bytes(), size_v_bytes() are private
    // These methods provide detailed memory usage but aren't accessible
    printf("Memory Usage: (private methods not accessible)\n");
    
    printf("\n");
}

// Analyze layer tensor structure and memory layout
// Examines detailed state of tensors in KV cache
//
// ┌─────────────────────────────────────────────────────────────────┐
// │                    Tensor Structure Analysis                    │
// │                                                                 │
// │  K Tensor Layout:                                               │
// │  ┌─────────────────────────────────────────────────────────┐    │
// │  │ Dimension 0: n_embd_head_k (32)                         │    │
// │  │ Dimension 1: n_head_kv (1)                              │    │
// │  │ Dimension 2: sequence_length (dynamic: 0→8→16)          │    │
// │  │ Dimension 3: batch_size (1)                             │    │
// │  └─────────────────────────────────────────────────────────┘    │
// │  │ Dimension 0: n_embd_head_k (32)                         │    │
// │  │ Dimension 1: n_head_kv (1)                              │    │
// │  │ Dimension 2: sequence_length (dynamic: 0→8→16)          │    │
// │  │ Dimension 3: batch_size (1)                             │    │
// │  └─────────────────────────────────────────────────────────┘    │
// │                                                                 │
// │  Memory Evolution:                                              │
// │  Initial:  [32, 1, 0, 1] → 0 bytes                              │
// │  Batch 1:  [32, 1, 8, 1] → 512 bytes                            │
// │  Batch 3:  [32, 1, 16, 1] → 1024 bytes                          │
// │                                                                 │
// │  V Tensor: Same structure as K tensor                           │
// └─────────────────────────────────────────────────────────────────┘
static void print_cache_tensors_info(llama_kv_cache_unified * kv_cache, 
                                    const llama_model & model, 
                                    const std::string & title) {
    if (!kv_cache) {
        printf("%s: No KV cache available\n", title.c_str());
        return;
    }
    
    printf("\n=== %s - Tensor Information ===\n", title.c_str());
    
    // 创建临时的ggml context用于获取tensor视图
    // 这不会分配实际内存，只是为了访问tensor的元数据
    ggml_init_params ctx_params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,  // 16MB
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(ctx_params);
    
    if (!ctx) {
        printf("Failed to create ggml context\n");
        return;
    }
    
    // 遍历每一层，检查其KV tensor的状态
    for (int32_t il = 0; il < (int32_t)model.hparams.n_layer; ++il) {
        printf("Layer %d:\n", il);
        
        try {
            // get_k()/get_v()返回指向cache中K/V tensor的视图
            // 这些tensor的维度会随着cache状态动态变化
            ggml_tensor * k_tensor = kv_cache->get_k(ctx, il);
            ggml_tensor * v_tensor = kv_cache->get_v(ctx, il);
            
            if (k_tensor) {
                // K tensor的维度解释：
                // ne[0]: 每个head的K向量维度 (n_embd_head_k)
                // ne[1]: 当前层的KV head数量 (n_head_kv)  
                // ne[2]: 当前活跃的序列长度 (对应get_n()的值)
                // ne[3]: batch维度，通常为1
                printf("  K tensor: [%ld, %ld, %ld, %ld] type=%s, size=%zu bytes\n",
                       k_tensor->ne[0], k_tensor->ne[1], k_tensor->ne[2], k_tensor->ne[3],
                       ggml_type_name(k_tensor->type), ggml_nbytes(k_tensor));
                
                // 检查tensor是否有实际的数据指针
                // NULL指针表示tensor还没有分配内存或已被释放
                if (k_tensor->data) {
                    printf("    Data pointer: %p (has data)\n", k_tensor->data);
                } else {
                    printf("    Data pointer: NULL (no data)\n");
                }
            } else {
                printf("  K tensor: NULL\n");
            }
            
            if (v_tensor) {
                // V tensor的维度结构与K tensor类似
                // 但根据v_trans参数，V tensor可能被转置存储以优化内存访问
                printf("  V tensor: [%ld, %ld, %ld, %ld] type=%s, size=%zu bytes\n",
                       v_tensor->ne[0], v_tensor->ne[1], v_tensor->ne[2], v_tensor->ne[3],
                       ggml_type_name(v_tensor->type), ggml_nbytes(v_tensor));
                
                if (v_tensor->data) {
                    printf("    Data pointer: %p (has data)\n", v_tensor->data);
                } else {
                    printf("    Data pointer: NULL (no data)\n");
                }
            } else {
                printf("  V tensor: NULL\n");
            }
            
        } catch (const std::exception& e) {
            printf("  Error accessing layer %d: %s\n", il, e.what());
        }
    }
    
    ggml_free(ctx);
    printf("\n");
}

// 跟踪和显示序列在cache中的分布情况
// 这个函数帮助理解多序列并发存储的内存布局
//
// ┌─────────────────────────────────────────────────────────────────┐
// │                    Sequence Distribution Map                    │
// │                                                                 │
// │  Cache Cells:                                                   │
// │  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐              │
// │  │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │11 │              │
// │  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘              │
// │                                                                 │
// │  Sequence Mapping:                                              │
// │  Seq 42: ████████████████ [0,3] (4 tokens)                      │
// │  Seq 84: ░░░░░░░░████████ [4,6] (3 tokens)                      │
// │  Seq 126:████████████████ [0,3] (4 tokens, copied from 42)      │
// │                                                                 │
// │  Legend: █ = occupied, ░ = empty                                │
// └─────────────────────────────────────────────────────────────────┘
static void print_sequence_info(llama_kv_cache_unified * kv_cache, 
                               const std::vector<llama_seq_id> & seq_ids,
                               const std::string & title) {
    if (!kv_cache) {
        printf("%s: No KV cache available\n", title.c_str());
        return;
    }
    
    printf("\n=== %s - Sequence Information ===\n", title.c_str());
    
    for (auto seq_id : seq_ids) {
        // seq_pos_min/max()返回指定序列在cache中的位置范围
        // 这些位置对应于transformer中的绝对位置编码
        llama_pos min_pos = kv_cache->seq_pos_min(seq_id);
        llama_pos max_pos = kv_cache->seq_pos_max(seq_id);
        
        printf("Sequence %d: ", seq_id);
        if (min_pos == -1 && max_pos == -1) {
            // 返回-1表示该序列在cache中不存在
            printf("empty\n");
        } else {
            // 显示序列的位置范围和token数量
            // 注意：位置是连续的，但在cache中的存储可能不连续
            printf("range [%d, %d], length %d\n", min_pos, max_pos, max_pos - min_pos + 1);
        }
    }
    printf("\n");
}

/*- Test Functions ------------------------------------------------------------*/

// 主要的KV cache测试函数
// 这个函数通过一系列操作演示cache的工作机制
//
// ┌─────────────────────────────────────────────────────────────────┐
// │                    Test Execution Flow                          │
// │                                                                 │
// │  Step 1: Model Creation                                         │
// │  ┌─────────────┐                                                │
// │  │ Create      │                                                │
// │  │ Test Model  │                                                │
// │  └─────────────┘                                                │
// │         │                                                       │
// │         ▼                                                       │
// │  Step 2: Cache Initialization                                   │
// │  ┌─────────────┐                                                │
// │  │ Initialize  │                                                │
// │  │ KV Cache    │                                                │
// │  └─────────────┘                                                │
// │         │                                                       │
// │         ▼                                                       │
// │  Step 3-7: Token Operations & Analysis                          │
// │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
// │  │ Add Batch 1 │  │ Add Batch 2 │  │ Extend Seq  │              │
// │  │ (Seq 42)    │  │ (Seq 84)    │  │ (Seq 42)    │              │
// │  └─────────────┘  └─────────────┘  └─────────────┘              │
// │         │                │                │                     │
// │         └────────────────┼────────────────┘                     │
// │                          ▼                                      │
// │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
// │  │ Copy Seq    │  │ Remove Seq  │  │ Clear Cache │              │
// │  │ (42→126)    │  │ (84)        │  │ (All)       │              │
// │  └─────────────┘  └─────────────┘  └─────────────┘              │
// └─────────────────────────────────────────────────────────────────┘
static void test_kv_cache_debug() {
    printf("=== Testing KV Cache Debug Tools ===\n");
    
    /*
     * Step 1: Model Creation
     * 
     * ┌─────────────────────────────────────────────────────────────┐
     * │                    Model Architecture                       │
     * │                                                             │
     * │  ┌─────────────┐    ┌─────────────┐                         │
     * │  │   Layer 0   │    │   Layer 1   │                         │
     * │  │             │    │             │                         │
     * │  │ ┌─────────┐ │    │ ┌─────────┐ │                         │
     * │  │ │ 4 Heads │ │    │ │ 4 Heads │ │                         │
     * │  │ │ 32 dim  │ │    │ │ 32 dim  │ │                         │
     * │  │ └─────────┘ │    │ └─────────┘ │                         │
     * │  └─────────────┘    └─────────────┘                         │
     * │                                                             │
     * │  Each layer will have independent K/V cache storage         │
     * └─────────────────────────────────────────────────────────────┘
     */
    auto model = _make_model(LLM_ARCH_LLAMA, 2, 32, 32, 4, 1);
    printf("✓ Test model created (2 layers, 4 heads)\n");
    
    /*
     * Step 2: Cache Initialization
     * 
     * ┌─────────────────────────────────────────────────────────────┐
     * │                    Cache Configuration                      │
     * │                                                             │
     * │  Cache Parameters:                                          │
     * │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
     * │  │   Size: 64  │  │ Type: F16   │  │ Seqs: 4     │          │
     * │  │   cells     │  │ precision   │  │ max         │          │
     * │  └─────────────┘  └─────────────┘  └─────────────┘          │
     * │                                                             │
     * │  Initial Cache Layout:                                      │
     * │  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐          │
     * │  │ ∅ │ ∅ │ ∅ │ ∅ │ ∅ │ ∅ │ ∅ │ ∅ │...│ ∅ │ ∅ │ ∅ │          │
     * │  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘          │
     * │   0   1   2   3   4   5   6   7       60  61  62  63        │
     * │                                                             │
     * │  Legend: ∅ = empty cell                                     │
     * └─────────────────────────────────────────────────────────────┘
     */
    llama_kv_cache_unified::layer_filter_cb filter = [](int32_t il) { 
        (void)il; 
        return true; 
    };
    
    auto kv_cache = std::make_unique<llama_kv_cache_unified>(
        *model, 
        std::move(filter),
        GGML_TYPE_F16,  // K type
        GGML_TYPE_F16,  // V type
        false,          // v_trans
        false,          // offload
        64,             // kv_size
        4,              // n_seq_max
        8,              // n_pad
        0,              // n_swa
        LLAMA_SWA_TYPE_NONE
    );
    
    printf("✓ KV cache created\n");
    
    // 显示初始状态：cache为空，所有tensor维度为0
    print_kv_cache_status(kv_cache.get(), "Initial State");
    print_cache_tensors_info(kv_cache.get(), *model, "Initial State");
    
    /*
     * Step 3: First Token Batch Addition
     * 
     * ┌─────────────────────────────────────────────────────────────┐
     * │                    Batch 1 Processing                      │
     * │                                                             │
     * │  Input Tokens:                                              │
     * │  ┌─────┬─────┬─────┬─────┐                                  │
     * │  │ 101 │ 102 │ 103 │ 104 │                                  │
     * │  │ pos │ pos │ pos │ pos │                                  │
     * │  │  0  │  1  │  2  │  3  │                                  │
     * │  └─────┴─────┴─────┴─────┘                                  │
     * │                                                             │
     * │  Cache After Allocation:                                    │
     * │  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐          │
     * │  │42 │42 │42 │42 │ ∅ │ ∅ │ ∅ │ ∅ │...│ ∅ │ ∅ │ ∅ │          │
     * │  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘          │
     * │   0   1   2   3   4   5   6   7       60  61  62  63        │
     * │                                                             │
     * │  Sequence 42: [0,3] length=4                               │
     * │  Active window: 8 cells (due to padding)                   │
     * └─────────────────────────────────────────────────────────────┘
     */
    printf("\n=== Adding First Batch of Tokens ===\n");
    
    llama_seq_id seq_id_1 = 42;
    llama_batch batch1 = llama_batch_init(4, 0, 1);
    
    // common_batch_add()将token添加到batch中
    // 参数：token_id, position, sequence_ids, need_logits
    // position是该token在序列中的绝对位置
    common_batch_add(batch1, 101, 0, {seq_id_1}, false);
    common_batch_add(batch1, 102, 1, {seq_id_1}, false);
    common_batch_add(batch1, 103, 2, {seq_id_1}, false);
    common_batch_add(batch1, 104, 3, {seq_id_1}, true);  // 最后一个token需要logits
    
    // llama_sbatch将batch转换为内部处理格式
    // 这个过程会分析序列结构和token分布
    llama_sbatch sbatch1(batch1, model->hparams.n_embd, true, false);
    llama_ubatch ubatch1 = sbatch1.split_simple(4);
    
    printf("Batch 1: %u tokens, %u seqs\n", ubatch1.n_tokens, ubatch1.n_seqs);
    
    // find_slot()是cache分配的核心函数
    // 它会在cache中寻找足够的连续空间来存储新的tokens
    if (kv_cache->find_slot(ubatch1)) {
        // commit()确认分配，使更改生效
        // 在此之前，分配是临时的，可以通过restore()撤销
        kv_cache->commit();
        printf("✓ First batch added to cache\n");
        
        print_kv_cache_status(kv_cache.get(), "After First Batch");
        print_cache_tensors_info(kv_cache.get(), *model, "After First Batch");
        print_sequence_info(kv_cache.get(), {seq_id_1}, "After First Batch");
    } else {
        printf("✗ Failed to add first batch to cache\n");
    }
    
    llama_batch_free(batch1);
    
    /*
     * Step 4: Second Sequence Addition
     * 
     * ┌─────────────────────────────────────────────────────────────┐
     * │                    Batch 2 Processing                       │
     * │                                                             │
     * │  Input Tokens (New Sequence):                               │
     * │  ┌─────┬─────┬─────┐                                        │
     * │  │ 201 │ 202 │ 203 │                                        │
     * │  │ pos │ pos │ pos │                                        │
     * │  │  0  │  1  │  2  │                                        │
     * │  └─────┴─────┴─────┘                                        │
     * │                                                             │
     * │  Cache After Allocation:                                    │
     * │  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐          │
     * │  │42 │42 │42 │42 │84 │84 │84 │ ∅ │...│ ∅ │ ∅ │ ∅ │          │
     * │  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘          │
     * │   0   1   2   3   4   5   6   7       60  61  62  63        │
     * │                                                             │
     * │  Sequence 42: [0,3] length=4                                │
     * │  Sequence 84: [0,2] length=3                                │
     * │  Active window: 8 cells (unchanged)                         │
     * └─────────────────────────────────────────────────────────────┘
     */
    printf("\n=== Adding Second Batch of Tokens (Different Sequence) ===\n");
    
    llama_seq_id seq_id_2 = 84;
    llama_batch batch2 = llama_batch_init(3, 0, 1);
    
    // 注意：这个序列的position从0开始，因为它是独立的序列
    // 每个序列都有自己的位置编码空间
    common_batch_add(batch2, 201, 0, {seq_id_2}, false);
    common_batch_add(batch2, 202, 1, {seq_id_2}, false);
    common_batch_add(batch2, 203, 2, {seq_id_2}, true);
    
    llama_sbatch sbatch2(batch2, model->hparams.n_embd, true, false);
    llama_ubatch ubatch2 = sbatch2.split_simple(3);
    
    printf("Batch 2: %u tokens, %u seqs\n", ubatch2.n_tokens, ubatch2.n_seqs);
    
    if (kv_cache->find_slot(ubatch2)) {
        kv_cache->commit();
        printf("✓ Second batch added to cache\n");
        
        print_kv_cache_status(kv_cache.get(), "After Second Batch");
        print_cache_tensors_info(kv_cache.get(), *model, "After Second Batch");
        print_sequence_info(kv_cache.get(), {seq_id_1, seq_id_2}, "After Second Batch");
    } else {
        printf("✗ Failed to add second batch to cache\n");
    }
    
    llama_batch_free(batch2);
    
    /*
     * Step 5: Sequence Extension
     * 
     * ┌─────────────────────────────────────────────────────────────┐
     * │                    Sequence Growth                          │
     * │                                                             │
     * │  Extending Sequence 42:                                     │
     * │  ┌─────┬─────┐                                              │
     * │  │ 105 │ 106 │                                              │
     * │  │ pos │ pos │                                              │
     * │  │  4  │  5  │                                              │
     * │  └─────┴─────┘                                              │
     * │                                                             │
     * │  Cache After Extension:                                     │
     * │  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐          │
     * │  │42 │42 │42 │42 │84 │84 │84 │42 │42 │ ∅ │...│ ∅ │          │
     * │  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘          │
     * │   0   1   2   3   4   5   6   7   8   9       63            │
     * │                                                             │
     * │  Sequence 42: [0,5] length=6 (extended!)                    │
     * │  Sequence 84: [0,2] length=3 (unchanged)                    │
     * │  Active window: 16 cells (expanded to fit longer sequence)  │
     * └─────────────────────────────────────────────────────────────┘
     */
    printf("\n=== Continuing First Sequence ===\n");
    
    llama_batch batch3 = llama_batch_init(2, 0, 1);
    
    // 继续序列42，position从4开始（接续之前的[0,3]）
    common_batch_add(batch3, 105, 4, {seq_id_1}, false);
    common_batch_add(batch3, 106, 5, {seq_id_1}, true);
    
    llama_sbatch sbatch3(batch3, model->hparams.n_embd, true, false);
    llama_ubatch ubatch3 = sbatch3.split_simple(2);
    
    printf("Batch 3: %u tokens, %u seqs\n", ubatch3.n_tokens, ubatch3.n_seqs);
    
    if (kv_cache->find_slot(ubatch3)) {
        kv_cache->commit();
        printf("✓ Third batch added to cache\n");
        
        print_kv_cache_status(kv_cache.get(), "After Third Batch");
        print_sequence_info(kv_cache.get(), {seq_id_1, seq_id_2}, "After Third Batch");
    } else {
        printf("✗ Failed to add third batch to cache\n");
    }
    
    llama_batch_free(batch3);
    
    /*
     * Step 6: Sequence Operations
     * 
     * ┌─────────────────────────────────────────────────────────────┐
     * │                    Sequence Manipulation                    │
     * │                                                             │
     * │  Operation 1: Copy Sequence 42 → 126                        │
     * │  ┌─────────────────┐    copy     ┌─────────────────┐        │
     * │  │ Sequence 42     │────────────▶│ Sequence 126    │        │
     * │  │ [0,1,2,3,4,5]   │             │ [0,1,2,3,4,5]   │        │
     * │  │ (original)      │             │ (duplicate)     │        │
     * │  └─────────────────┘             └─────────────────┘        │
     * │                                                             │
     * │  Operation 2: Remove Sequence 84                            │
     * │  ┌─────────────────┐   remove    ┌─────────────────┐        │
     * │  │ Sequence 84     │────────────▶│     Empty       │        │
     * │  │ [0,1,2]         │             │     Cells       │        │
     * │  │ (deleted)       │             │   Available     │        │
     * │  └─────────────────┘             └─────────────────┘        │
     * └─────────────────────────────────────────────────────────────┘
     */
    printf("\n=== Testing Sequence Operations ===\n");
    
    // seq_cp()复制序列：将源序列的所有K/V数据复制到目标序列
    // 这是一个深拷贝操作，目标序列获得独立的数据副本
    llama_seq_id seq_id_3 = 126;
    printf("Copying sequence %d to %d...\n", seq_id_1, seq_id_3);
    kv_cache->seq_cp(seq_id_1, seq_id_3, -1, -1);  // -1表示复制整个序列
    print_sequence_info(kv_cache.get(), {seq_id_1, seq_id_2, seq_id_3}, "After Sequence Copy");
    
    // seq_rm()删除序列：释放序列占用的cache空间
    // 被删除的cells变为可用状态，可以被新的tokens使用
    printf("Removing sequence %d...\n", seq_id_2);
    kv_cache->seq_rm(seq_id_2, -1, -1);  // -1表示删除整个序列
    print_sequence_info(kv_cache.get(), {seq_id_1, seq_id_2, seq_id_3}, "After Sequence Remove");
    print_kv_cache_status(kv_cache.get(), "After Sequence Remove");
    
    /*
     * Step 7: Cache Cleanup
     * 
     * ┌─────────────────────────────────────────────────────────────┐
     * │                    Cache Reset Operation                    │
     * │                                                             │
     * │  Before Clear:                                              │
     * │  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐          │
     * │  │42 │42 │42 │42 │ ∅ │ ∅ │ ∅ │42 │42 │126│...│ ∅ │          │
     * │  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘          │
     * │                                                             │
     * │  After Clear:                                               │
     * │  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐          │
     * │  │ ∅ │ ∅ │ ∅ │ ∅ │ ∅ │ ∅ │ ∅ │ ∅ │ ∅ │ ∅ │...│ ∅ │          │
     * │  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘          │
     * │                                                             │
     * │  All sequences removed, cache ready for reuse               │
     * └─────────────────────────────────────────────────────────────┘
     */
    printf("\n=== Clearing Cache ===\n");
    kv_cache->clear();
    
    print_kv_cache_status(kv_cache.get(), "After Clear");
    print_sequence_info(kv_cache.get(), {seq_id_1, seq_id_2, seq_id_3}, "After Clear");
    
    printf("✓ KV Cache debug test completed successfully!\n");
}

/*- Main ----------------------------------------------------------------------*/

// 主函数：初始化环境并运行测试
//
// ┌─────────────────────────────────────────────────────────────────┐
// │                    Program Execution Flow                      │
// │                                                                 │
// │  ┌─────────────┐                                                │
// │  │ Initialize  │                                                │
// │  │ Backend     │                                                │
// │  └─────────────┘                                                │
// │         │                                                       │
// │         ▼                                                       │
// │  ┌─────────────┐                                                │
// │  │ Run Cache   │                                                │
// │  │ Debug Tests │                                                │
// │  └─────────────┘                                                │
// │         │                                                       │
// │         ▼                                                       │
// │  ┌─────────────┐                                                │
// │  │ Cleanup &   │                                                │
// │  │ Exit        │                                                │
// │  └─────────────┘                                                │
// └─────────────────────────────────────────────────────────────────┘
int main(int argc, char ** argv) {
    (void)argc;  // Suppress unused parameter warning
    (void)argv;  // Suppress unused parameter warning
    
    printf("=== KV Cache Debug Tool ===\n\n");
    
    // 初始化ggml backend系统
    // 这会加载所有可用的计算后端（CPU, GPU等）
    ggml_backend_load_all();
    printf("ggml backend initialized\n\n");
    
    try {
        test_kv_cache_debug();
        
        printf("\n🎉 All KV cache debug tests completed!\n");
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << "\n";
        return 1;
    }
    
    // 清理backend资源
    llama_backend_free();
    
    return 0;
}
