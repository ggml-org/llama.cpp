#include "llama-kv-cache-paged.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-hparams.h"
#include "llama-model.h"
#include "llama-kv-cache.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

//
// llama_kv_cache_paged_context implementation
//

llama_kv_cache_paged_context::llama_kv_cache_paged_context(llama_memory_status status)
    : status(status), kv_paged(nullptr), ubatch() {
    fprintf(stderr, "llama_kv_cache_paged_context::llama_kv_cache_paged_context(status=%d) called\n", status);
    // ubatch is value-initialized to zero
}

llama_kv_cache_paged_context::llama_kv_cache_paged_context(llama_kv_cache_paged * kv_paged)
    : status(LLAMA_MEMORY_STATUS_SUCCESS), kv_paged(kv_paged), ubatch() {
    fprintf(stderr, "llama_kv_cache_paged_context::llama_kv_cache_paged_context(kv_paged=%p) called\n", (void*)kv_paged);
    // ubatch is value-initialized to zero
}

// Stub implementations for llama_kv_cache_context-like interface
// These are called by graph building code via static_cast
// TODO: Implement proper PagedAttention logic for these methods

ggml_tensor * llama_kv_cache_paged_context::get_k(ggml_context * ctx, int32_t il) const {
    GGML_UNUSED(ctx);
    if (!kv_paged) {
        fprintf(stderr, "ERROR: llama_kv_cache_paged_context::get_k() called with null kv_paged\n");
        return nullptr;
    }
    // Return the full paged K cache tensor for this layer
    // The PagedAttention kernel will handle block indexing
    return kv_paged->get_k_blocks(il);
}

ggml_tensor * llama_kv_cache_paged_context::get_v(ggml_context * ctx, int32_t il) const {
    GGML_UNUSED(ctx);
    if (!kv_paged) {
        fprintf(stderr, "ERROR: llama_kv_cache_paged_context::get_v() called with null kv_paged\n");
        return nullptr;
    }
    // Return the full paged V cache tensor for this layer
    // The PagedAttention kernel will handle block indexing
    return kv_paged->get_v_blocks(il);
}

ggml_tensor * llama_kv_cache_paged_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    if (!kv_paged) {
        fprintf(stderr, "ERROR: llama_kv_cache_paged_context::cpy_k() called with null kv_paged\n");
        return nullptr;
    }

    // Get K cache blocks for this layer
    auto * k_cache = kv_paged->get_k_blocks(il);
    if (!k_cache) {
        return nullptr;
    }

    // Use ggml_paged_cpy to copy K data to paged cache blocks
    // k_cur shape: [head_size, n_heads, n_tokens]
    // k_cache shape: [num_blocks, n_kv_heads, head_size, block_size]
    // k_idxs shape: [n_tokens] - slot index for each token
    return ggml_paged_cpy(ctx, k_cur, k_cache, k_idxs);
}

ggml_tensor * llama_kv_cache_paged_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    if (!kv_paged) {
        fprintf(stderr, "ERROR: llama_kv_cache_paged_context::cpy_v() called with null kv_paged\n");
        return nullptr;
    }

    // Get V cache blocks for this layer
    auto * v_cache = kv_paged->get_v_blocks(il);
    if (!v_cache) {
        return nullptr;
    }

    // Use ggml_paged_cpy to copy V data to paged cache blocks
    // v_cur shape: [head_size, n_heads, n_tokens]
    // v_cache shape: [num_blocks, n_kv_heads, head_size, block_size]
    // v_idxs shape: [n_tokens] - slot index for each token
    return ggml_paged_cpy(ctx, v_cur, v_cache, v_idxs);
}

ggml_tensor * llama_kv_cache_paged_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    // TODO: Proper paged block index calculation
    // For now, create a simple sequential index tensor
    // This won't work correctly for PagedAttention but allows graph building to proceed
    const int64_t n_tokens = ubatch.n_tokens;
    auto * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(result, "k_idxs_paged");
    return result;
}

ggml_tensor * llama_kv_cache_paged_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    // TODO: Proper paged block index calculation
    // For now, create a simple sequential index tensor
    // This won't work correctly for PagedAttention but allows graph building to proceed
    const int64_t n_tokens = ubatch.n_tokens;
    auto * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(result, "v_idxs_paged");
    return result;
}

void llama_kv_cache_paged_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    if (!dst || !ubatch) return;

    // TODO: Proper paged block indexing
    // For now, fill with sequential indices
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    int32_t * data = (int32_t *) dst->data;
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        data[i] = i;
    }
}

void llama_kv_cache_paged_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    if (!dst || !ubatch) return;

    // TODO: Proper paged block indexing
    // For now, fill with sequential indices
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    int32_t * data = (int32_t *) dst->data;
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        data[i] = i;
    }
}

void llama_kv_cache_paged_context::set_input_k_shift(ggml_tensor * dst) const {
    // K shifting not supported with PagedAttention
    GGML_UNUSED(dst);
}

void llama_kv_cache_paged_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    if (!dst || !ubatch) return;

    // TODO: Proper PagedAttention mask handling
    // For now, create a simple causal mask
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_kv = dst->ne[0];  // KV sequence length

    float * data = (float *) dst->data;

    if (causal_attn) {
        // Causal mask: can attend to current and previous tokens
        for (int64_t i = 0; i < n_tokens; ++i) {
            for (int64_t j = 0; j < n_kv; ++j) {
                data[i * n_kv + j] = (j <= i) ? 0.0f : -INFINITY;
            }
        }
    } else {
        // No mask: can attend to all tokens
        for (int64_t i = 0; i < n_tokens * n_kv; ++i) {
            data[i] = 0.0f;
        }
    }
}

void llama_kv_cache_paged_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    // Position bucketing not used with basic PagedAttention
    GGML_UNUSED(dst);
    GGML_UNUSED(ubatch);
}

//
// llama_kv_cache_paged implementation
//

llama_kv_cache_paged::llama_kv_cache_paged(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   block_size,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse)
    : model(model),
      hparams(model.hparams),
      type_k(type_k),
      type_v(type_v),
      n_seq_max(n_seq_max),
      block_size(block_size),
      num_blocks((kv_size + block_size - 1) / block_size) {  // ceil division

    GGML_ASSERT(block_size > 0 && block_size <= 256);
    GGML_ASSERT((block_size & (block_size - 1)) == 0 && "block_size must be power of 2");

    // Check environment variable for debug output
    const char * debug_env = std::getenv("LLAMA_KV_CACHE_DEBUG");
    if (debug_env) {
        debug = std::atoi(debug_env);
    }

    if (debug > 0) {
        fprintf(stderr, "%s: initializing paged KV cache with %u blocks of size %u (total capacity: %u tokens)\n",
                __func__, num_blocks, block_size, num_blocks * block_size);
    }

    // Build layer list (same as standard KV cache)
    const int32_t n_layer = hparams.n_layer;

    for (int32_t il = 0; il < n_layer; ++il) {
        if (filter && !filter(il)) {
            continue;
        }

        // Check if this layer should reuse memory from another layer
        const int32_t il_reuse = reuse ? reuse(il) : -1;

        if (il_reuse >= 0) {
            // Reuse memory from another layer
            auto it = map_layer_ids.find(il_reuse);
            GGML_ASSERT(it != map_layer_ids.end() && "layer to reuse not found");
            map_layer_ids[il] = it->second;
            continue;
        }

        kv_layer layer;
        layer.il = il;

        // Initialize block storage
        layer.blocks.resize(num_blocks);
        for (uint32_t i = 0; i < num_blocks; ++i) {
            layer.blocks[i].id = i;
            layer.blocks[i].is_free = true;
            layer.blocks[i].ref_count = 0;
        }

        // Add to layer list
        const int32_t il_kv = static_cast<int32_t>(layers.size());
        layers.push_back(std::move(layer));
        map_layer_ids[il] = il_kv;
    }

    // Initialize free block list
    for (uint32_t i = 0; i < num_blocks; ++i) {
        free_blocks.push_back(i);
    }

    if (debug > 0) {
        fprintf(stderr, "%s: created %zu layers with %u blocks each\n",
                __func__, layers.size(), num_blocks);
        fprintf(stderr, "%s: map_layer_ids contains %zu entries:\n", __func__, map_layer_ids.size());
        for (const auto & [il, il_kv] : map_layer_ids) {
            fprintf(stderr, "%s:   layer %d -> kv_layer %d\n", __func__, il, il_kv);
        }
    }

    // Allocate tensor memory for blocks
    const int32_t n_embd_k_gqa = hparams.n_embd_k_gqa();
    // const int32_t n_embd_v_gqa = hparams.n_embd_v_gqa();  // unused for now
    const int32_t n_head_kv = hparams.n_head_kv();

    // Create context map for different buffer types
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            // Allocate space for:
            // - 2 base tensors per layer (k_all_blocks, v_all_blocks)
            // - 2 * num_blocks view tensors per layer (k_data, v_data for each block)
            // Total: layers.size() * 2 * (1 + num_blocks)
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*layers.size()*(1 + num_blocks)*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);
            return ctx;
        }
        return it->second.get();
    };

    // Create tensors for each layer
    for (auto & layer : layers) {
        const int32_t il = layer.il;

        // Determine buffer type (CPU or GPU)
        bool offload = model.dev_layer(il) != nullptr;
        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);
        }

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for paged kv cache");
        }

        // Create tensors for all blocks in this layer
        // Shape: [num_blocks, num_kv_heads, head_size, block_size]
        // This matches the expected layout for PagedAttention CUDA kernels
        const int64_t head_size = n_embd_k_gqa / n_head_kv;
        layer.k_all_blocks = ggml_new_tensor_4d(ctx, type_k, num_blocks, n_head_kv, head_size, block_size);
        layer.v_all_blocks = ggml_new_tensor_4d(ctx, type_v, num_blocks, n_head_kv, head_size, block_size);

        ggml_format_name(layer.k_all_blocks, "paged_cache_k_l%d", il);
        ggml_format_name(layer.v_all_blocks, "paged_cache_v_l%d", il);

        // Update individual block pointers to reference parts of the contiguous tensor
        for (uint32_t i = 0; i < num_blocks; ++i) {
            // Create views into the all_blocks tensors
            // Each block is a slice along dimension 0: [num_kv_heads, head_size, block_size]
            // With layout [num_blocks, num_kv_heads, head_size, block_size], we slice the first dim
            const size_t offset = i * layer.k_all_blocks->nb[0];
            layer.blocks[i].k_data = ggml_view_3d(ctx, layer.k_all_blocks,
                n_head_kv, head_size, block_size,
                layer.k_all_blocks->nb[1], layer.k_all_blocks->nb[2], offset);
            layer.blocks[i].v_data = ggml_view_3d(ctx, layer.v_all_blocks,
                n_head_kv, head_size, block_size,
                layer.v_all_blocks->nb[1], layer.v_all_blocks->nb[2], offset);
        }
    }

    // Allocate buffers for all contexts
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for paged kv cache");
        }

        if (debug > 0) {
            fprintf(stderr, "%s: %10s paged KV buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name(buf),
                    ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
        }

        // Clear buffer to avoid NaN values
        ggml_backend_buffer_clear(buf, 0);

        // Store context and buffer pair
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }
}

//
// llama_memory_i interface implementation
//

llama_memory_context_ptr llama_kv_cache_paged::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    GGML_UNUSED(n_ubatch);
    GGML_UNUSED(embd_all);

    const auto & batch = balloc.get_batch();

    if (debug > 0) {
        fprintf(stderr, "%s: processing batch with %d tokens\n",
                __func__, batch.n_tokens);
        fprintf(stderr, "%s: current state: %zu sequences, %zu free blocks\n",
                __func__, block_tables.size(), free_blocks.size());
    }

    // Process each token position to ensure blocks are allocated
    for (int i = 0; i < batch.n_tokens; ++i) {
        // Handle null arrays with defaults:
        // - n_seq_id defaults to 1
        // - seq_id defaults to 0
        // - pos defaults to sequential (i)
        const int n_seqs = batch.n_seq_id ? batch.n_seq_id[i] : 1;

        if (debug > 1) {
            const llama_pos pos_debug = batch.pos ? batch.pos[i] : i;
            fprintf(stderr, "%s: token %d: n_seqs=%d, pos=%d\n",
                    __func__, i, n_seqs, pos_debug);
        }

        for (int j = 0; j < n_seqs; ++j) {
            const llama_seq_id seq_id = batch.seq_id ? batch.seq_id[i][j] : 0;
            const llama_pos pos = batch.pos ? batch.pos[i] : i;

            if (debug > 1) {
                fprintf(stderr, "%s:   seq_id=%d, pos=%d\n",
                        __func__, seq_id, pos);
            }

            // Check if this sequence needs blocks
            auto & blocks = block_tables[seq_id];
            auto & meta = seq_meta[seq_id];

            // Calculate required blocks for this position
            const uint32_t required_blocks = (pos + block_size) / block_size;

            if (debug > 1) {
                fprintf(stderr, "%s:   current blocks: %zu, required: %u\n",
                        __func__, blocks.size(), required_blocks);
            }

            // Allocate more blocks if needed
            while (blocks.size() < required_blocks) {
                uint32_t block_id = allocate_block();
                if (block_id == UINT32_MAX) {
                    fprintf(stderr, "%s: ERROR: failed to allocate block for seq %d at pos %d\n",
                            __func__, seq_id, pos);
                    fprintf(stderr, "%s: ERROR: free_blocks.size()=%zu, block_tables.size()=%zu\n",
                            __func__, free_blocks.size(), block_tables.size());
                    return llama_memory_context_ptr(
                        new llama_kv_cache_paged_context(LLAMA_MEMORY_STATUS_FAILED_PREPARE));
                }
                blocks.push_back(block_id);

                if (debug > 1) {
                    fprintf(stderr, "%s: allocated block %u for seq %d (total blocks: %zu, free remaining: %zu)\n",
                            __func__, block_id, seq_id, blocks.size(), free_blocks.size());
                }
            }

            // Update sequence metadata
            if (meta.pos_min < 0 || pos < meta.pos_min) {
                meta.pos_min = pos;
            }
            if (pos > meta.pos_max) {
                meta.pos_max = pos;
            }
            meta.length = static_cast<uint32_t>(meta.pos_max - meta.pos_min + 1);
        }
    }

    // Populate out_ids based on batch.logits
    // This is required for llama_context to properly track which tokens produce outputs
    auto & out_ids = balloc.get_out_ids();
    out_ids.clear();
    for (int i = 0; i < batch.n_tokens; ++i) {
        // batch.logits should have been populated by balloc.prepare()
        // If logits[i] is non-zero, this token should produce output
        if (batch.logits && batch.logits[i]) {
            out_ids.push_back(i);
        }
    }

    if (debug > 0) {
        fprintf(stderr, "%s: batch initialization complete, %zu outputs\n",
                __func__, out_ids.size());
    }

    return llama_memory_context_ptr(new llama_kv_cache_paged_context(this));
}

llama_memory_context_ptr llama_kv_cache_paged::init_full() {
    if (debug > 0) {
        fprintf(stderr, "%s: creating context for init_full\n", __func__);
    }

    // Return context initialized with this paged cache
    auto ctx = new llama_kv_cache_paged_context(this);

    if (debug > 0) {
        fprintf(stderr, "%s: context created at %p, creating unique_ptr\n", __func__, (void*)ctx);
    }

    llama_memory_context_ptr result(ctx);

    if (debug > 0) {
        fprintf(stderr, "%s: unique_ptr created, returning\n", __func__);
    }

    return result;
}

llama_memory_context_ptr llama_kv_cache_paged::init_update(
        llama_context * lctx,
        bool optimize) {
    GGML_UNUSED(lctx);
    GGML_UNUSED(optimize);
    // TODO: Implement update initialization
    return llama_memory_context_ptr(
        new llama_kv_cache_paged_context(LLAMA_MEMORY_STATUS_NO_UPDATE));
}

bool llama_kv_cache_paged::get_can_shift() const {
    // PagedAttention doesn't support context shifting
    // (blocks are allocated independently)
    return false;
}

void llama_kv_cache_paged::clear(bool data) {
    GGML_UNUSED(data);
    // Free all block tables
    block_tables.clear();
    seq_meta.clear();

    // Reset all blocks to free state
    for (auto & layer : layers) {
        for (auto & block : layer.blocks) {
            block.ref_count = 0;
            block.is_free = true;
        }
    }

    // Rebuild free block list
    free_blocks.clear();
    for (uint32_t i = 0; i < num_blocks; ++i) {
        free_blocks.push_back(i);
    }

    if (debug > 0) {
        fprintf(stderr, "%s: cleared paged KV cache\n", __func__);
    }
}

bool llama_kv_cache_paged::seq_rm(
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1) {
    // Remove tokens in range [p0, p1) from sequence
    if (debug > 0) {
        fprintf(stderr, "%s: called with seq_id=%d, p0=%d, p1=%d\n",
                __func__, seq_id, p0, p1);
    }

    auto it = block_tables.find(seq_id);
    if (it == block_tables.end()) {
        // Sequence doesn't exist - already cleared, return true
        if (debug > 0) {
            fprintf(stderr, "%s: sequence %d doesn't exist, already cleared\n",
                    __func__, seq_id);
        }
        return true;
    }

    // Normalize parameters: p1 < 0 means "to the end"
    if (p0 < 0) {
        p0 = 0;
    }
    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // Get sequence metadata
    auto meta_it = seq_meta.find(seq_id);
    if (meta_it == seq_meta.end()) {
        // No metadata - sequence hasn't been used yet, treat as full removal
        auto & blocks = it->second;
        for (uint32_t block_id : blocks) {
            free_block(block_id);
        }

        block_tables.erase(it);

        if (debug > 0) {
            fprintf(stderr, "%s: removed sequence %d without metadata (%zu blocks freed)\n",
                    __func__, seq_id, blocks.size());
        }

        return true;
    }

    const auto & meta = meta_it->second;

    // Check if we're removing the entire sequence (from start)
    // This includes: removing from position 0, or removing from before/at the minimum position
    // We also treat removal from an uninitialized sequence (pos_min == -1) as full removal when p0 == 0
    bool remove_from_start = (p0 == 0) || (p0 <= meta.pos_min) || (meta.pos_min == -1 && p0 == 0);

    if (remove_from_start) {
        // Removing from the beginning - clear entire sequence
        auto & blocks = it->second;
        for (uint32_t block_id : blocks) {
            free_block(block_id);
        }

        block_tables.erase(it);
        seq_meta.erase(seq_id);

        if (debug > 0) {
            fprintf(stderr, "%s: removed entire sequence %d (%zu blocks freed, p0=%d, pos_min=%d)\n",
                    __func__, seq_id, blocks.size(), p0, meta.pos_min);
        }

        return true;
    }

    // Partial removal from the middle/end is not yet supported
    // This would require tracking which blocks are partially used
    if (debug > 0) {
        fprintf(stderr, "%s: partial sequence removal (p0=%d, p1=%d, pos_min=%d) not yet supported in paged cache\n",
                __func__, p0, p1, meta.pos_min);
    }
    return false;
}

void llama_kv_cache_paged::seq_cp(
        llama_seq_id seq_id_src,
        llama_seq_id seq_id_dst,
        llama_pos p0,
        llama_pos p1) {
    GGML_UNUSED(p1);
    // Copy sequence - in paged attention, this is efficient via block sharing
    auto it_src = block_tables.find(seq_id_src);
    if (it_src == block_tables.end()) {
        return;
    }

    // For simplicity, copy entire sequence (ignore p0, p1 for now)
    GGML_UNUSED(p0);
    auto & src_blocks = it_src->second;

    // Increment reference count on all blocks
    for (uint32_t block_id : src_blocks) {
        for (auto & layer : layers) {
            if (block_id < layer.blocks.size()) {
                layer.blocks[block_id].ref_count++;
            }
        }
    }

    // Share the block table
    block_tables[seq_id_dst] = src_blocks;

    // Copy metadata
    auto it_meta = seq_meta.find(seq_id_src);
    if (it_meta != seq_meta.end()) {
        seq_meta[seq_id_dst] = it_meta->second;
    }

    if (debug > 0) {
        fprintf(stderr, "%s: copied sequence %d to %d (%zu blocks shared)\n",
                __func__, seq_id_src, seq_id_dst, src_blocks.size());
    }
}

void llama_kv_cache_paged::seq_keep(llama_seq_id seq_id) {
    // Remove all sequences except the specified one
    std::vector<llama_seq_id> to_remove;

    for (const auto & entry : block_tables) {
        if (entry.first != seq_id) {
            to_remove.push_back(entry.first);
        }
    }

    for (llama_seq_id sid : to_remove) {
        seq_rm(sid, -1, -1);
    }

    if (debug > 0) {
        fprintf(stderr, "%s: kept only sequence %d\n", __func__, seq_id);
    }
}

void llama_kv_cache_paged::seq_add(
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1,
        llama_pos shift) {
    GGML_UNUSED(p1);
    // Shift positions in sequence
    auto it = seq_meta.find(seq_id);
    if (it == seq_meta.end()) {
        return;
    }

    // Update position metadata
    if (p0 >= 0 && it->second.pos_min >= p0) {
        it->second.pos_min += shift;
    }
    if (p0 >= 0 && it->second.pos_max >= p0) {
        it->second.pos_max += shift;
    }

    if (debug > 0) {
        fprintf(stderr, "%s: shifted sequence %d by %d\n", __func__, seq_id, shift);
    }
}

void llama_kv_cache_paged::seq_div(
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1,
        int d) {
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    // Divide positions (used for attention scaling)
    // For paged attention, this is mostly metadata-only
    auto it = seq_meta.find(seq_id);
    if (it == seq_meta.end()) {
        return;
    }

    if (debug > 0) {
        fprintf(stderr, "%s: divided sequence %d positions by %d\n", __func__, seq_id, d);
    }

    // Position division would affect logical positioning but not block allocation
}

llama_pos llama_kv_cache_paged::seq_pos_min(llama_seq_id seq_id) const {
    auto it = seq_meta.find(seq_id);
    return (it != seq_meta.end()) ? it->second.pos_min : -1;
}

llama_pos llama_kv_cache_paged::seq_pos_max(llama_seq_id seq_id) const {
    auto it = seq_meta.find(seq_id);
    return (it != seq_meta.end()) ? it->second.pos_max : -1;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_paged::memory_breakdown() const {
    // TODO: Implement memory breakdown
    return std::map<ggml_backend_buffer_type_t, size_t>();
}

void llama_kv_cache_paged::state_write(
        llama_io_write_i & io,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) const {
    GGML_UNUSED(io);
    GGML_UNUSED(seq_id);
    GGML_UNUSED(flags);
    // TODO: Implement state serialization
    fprintf(stderr, "%s: state saving not yet implemented for paged cache\n", __func__);
}

void llama_kv_cache_paged::state_read(
        llama_io_read_i & io,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) {
    GGML_UNUSED(io);
    GGML_UNUSED(seq_id);
    GGML_UNUSED(flags);
    // TODO: Implement state deserialization
    fprintf(stderr, "%s: state loading not yet implemented for paged cache\n", __func__);
}

//
// PagedAttention specific functions
//

const std::vector<uint32_t> & llama_kv_cache_paged::get_block_table(llama_seq_id seq_id) const {
    static const std::vector<uint32_t> empty;
    auto it = block_tables.find(seq_id);
    return (it != block_tables.end()) ? it->second : empty;
}

std::vector<int32_t> llama_kv_cache_paged::get_seq_lens() const {
    std::vector<int32_t> lens;
    lens.reserve(seq_meta.size());

    for (const auto & entry : seq_meta) {
        lens.push_back(static_cast<int32_t>(entry.second.length));
    }

    return lens;
}

ggml_tensor * llama_kv_cache_paged::get_k_blocks(int32_t il) const {
    // Map model layer ID to KV cache layer ID
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return nullptr;
    }

    const int32_t il_kv = it->second;
    if (il_kv < 0 || il_kv >= static_cast<int32_t>(layers.size())) {
        return nullptr;
    }

    return layers[il_kv].k_all_blocks;
}

ggml_tensor * llama_kv_cache_paged::get_v_blocks(int32_t il) const {
    // Map model layer ID to KV cache layer ID
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return nullptr;
    }

    const int32_t il_kv = it->second;
    if (il_kv < 0 || il_kv >= static_cast<int32_t>(layers.size())) {
        return nullptr;
    }

    return layers[il_kv].v_all_blocks;
}

ggml_tensor * llama_kv_cache_paged::build_block_tables_tensor(ggml_context * ctx) const {
    // Build block tables tensor for all active sequences
    // Shape: [max_blocks_per_seq, n_seqs]

    // During graph building (before any sequences exist), use default sizes
    size_t max_blocks;
    size_t n_seqs_actual;

    if (block_tables.empty()) {
        // Use defaults for graph building
        n_seqs_actual = n_seq_max;
        // Estimate max blocks based on context size and block size
        // Assume each sequence could use the full context
        max_blocks = (4096 + block_size - 1) / block_size; // default n_ctx = 4096
    } else {
        // Find maximum number of blocks per sequence
        max_blocks = 0;
        for (const auto & [seq_id, blocks] : block_tables) {
            max_blocks = std::max(max_blocks, blocks.size());
        }
        n_seqs_actual = block_tables.size();
    }

    // Create tensor
    ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, max_blocks, n_seqs_actual);
    ggml_set_input(tensor);

    // Fill with block IDs (will be done during set_input)
    // For now, the structure is created
    return tensor;
}

ggml_tensor * llama_kv_cache_paged::build_seq_lens_tensor(ggml_context * ctx) const {
    // Build sequence lengths tensor
    // Shape: [n_seqs]

    // During graph building (before any sequences exist), use default size
    const size_t n_seqs = seq_meta.empty() ? n_seq_max : seq_meta.size();

    // Create tensor
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);
    ggml_set_input(tensor);

    return tensor;
}

//
// Block management (private)
//

uint32_t llama_kv_cache_paged::allocate_block() {
    if (free_blocks.empty()) {
        fprintf(stderr, "%s: ERROR: out of free blocks!\n", __func__);
        return UINT32_MAX;
    }

    uint32_t block_id = free_blocks.back();
    free_blocks.pop_back();

    // Mark block as allocated in all layers
    for (auto & layer : layers) {
        if (block_id < layer.blocks.size()) {
            layer.blocks[block_id].is_free = false;
            layer.blocks[block_id].ref_count = 1;
        }
    }

    if (debug > 1) {
        fprintf(stderr, "%s: allocated block %u (%zu free remaining)\n",
                __func__, block_id, free_blocks.size());
    }

    return block_id;
}

void llama_kv_cache_paged::free_block(uint32_t block_id) {
    if (block_id >= num_blocks) {
        return;
    }

    // Decrement reference count
    for (auto & layer : layers) {
        if (block_id < layer.blocks.size()) {
            auto & block = layer.blocks[block_id];

            if (block.ref_count > 0) {
                block.ref_count--;
            }

            // Free block if reference count reaches zero
            if (block.ref_count == 0 && !block.is_free) {
                block.is_free = true;
                free_blocks.push_back(block_id);

                if (debug > 1) {
                    fprintf(stderr, "%s: freed block %u (%zu free blocks total)\n",
                            __func__, block_id, free_blocks.size());
                }
            }
        }
    }
}

void llama_kv_cache_paged::allocate_blocks_for_sequence(
        llama_seq_id seq_id,
        uint32_t num_tokens) {
    // Calculate number of blocks needed
    uint32_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;

    if (debug > 0) {
        fprintf(stderr, "%s: allocating %u blocks for sequence %d (%u tokens)\n",
                __func__, num_blocks_needed, seq_id, num_tokens);
    }

    // Allocate blocks
    auto & blocks = block_tables[seq_id];
    blocks.reserve(num_blocks_needed);

    for (uint32_t i = 0; i < num_blocks_needed; ++i) {
        uint32_t block_id = allocate_block();
        if (block_id == UINT32_MAX) {
            fprintf(stderr, "%s: ERROR: failed to allocate block %u/%u for sequence %d\n",
                    __func__, i, num_blocks_needed, seq_id);
            return;
        }
        blocks.push_back(block_id);
    }

    // Update sequence metadata
    auto & meta = seq_meta[seq_id];
    meta.length = num_tokens;
    meta.pos_min = 0;
    meta.pos_max = static_cast<llama_pos>(num_tokens - 1);
}

//
// Helper functions (private)
//

size_t llama_kv_cache_paged::total_size() const {
    return size_k_bytes() + size_v_bytes();
}

size_t llama_kv_cache_paged::size_k_bytes() const {
    // TODO: Calculate actual memory size based on tensor layouts
    return 0;
}

size_t llama_kv_cache_paged::size_v_bytes() const {
    // TODO: Calculate actual memory size based on tensor layouts
    return 0;
}

void llama_kv_cache_paged::populate_block_tables_tensor(ggml_tensor * tensor) const {
    if (!tensor || !tensor->data) {
        fprintf(stderr, "%s: ERROR: tensor is null or has no data\n", __func__);
        return;
    }

    if (debug > 0) {
        fprintf(stderr, "%s: populating block tables tensor [%lld, %lld]\n",
                __func__, (long long)tensor->ne[0], (long long)tensor->ne[1]);
    }

    // Tensor layout: [max_blocks_per_seq, n_seqs]
    const int64_t max_blocks = tensor->ne[0];
    const int64_t n_seqs = tensor->ne[1];

    // Initialize all entries to 0 (invalid block ID)
    int32_t * data = reinterpret_cast<int32_t *>(tensor->data);
    memset(data, 0, ggml_nbytes(tensor));

    // Fill in block IDs for each active sequence
    int seq_idx = 0;
    for (const auto & entry : block_tables) {
        if (seq_idx >= n_seqs) {
            fprintf(stderr, "%s: WARNING: more sequences than tensor space (%d >= %lld)\n",
                    __func__, seq_idx, (long long)n_seqs);
            break;
        }

        const auto & blocks = entry.second;
        const int64_t num_blocks = std::min(static_cast<int64_t>(blocks.size()), max_blocks);

        // Copy block IDs for this sequence
        for (int64_t i = 0; i < num_blocks; ++i) {
            data[seq_idx * max_blocks + i] = static_cast<int32_t>(blocks[i]);
        }

        if (debug > 1) {
            fprintf(stderr, "%s:   seq %d: %lld blocks [", __func__, entry.first, (long long)num_blocks);
            for (int64_t i = 0; i < std::min(num_blocks, (int64_t)4); ++i) {
                fprintf(stderr, "%d%s", blocks[i], i < num_blocks - 1 ? ", " : "");
            }
            if (num_blocks > 4) fprintf(stderr, "...");
            fprintf(stderr, "]\n");
        }

        seq_idx++;
    }

    if (debug > 0) {
        fprintf(stderr, "%s: populated %d sequences\n", __func__, seq_idx);
    }
}

void llama_kv_cache_paged::populate_seq_lens_tensor(ggml_tensor * tensor) const {
    if (!tensor || !tensor->data) {
        fprintf(stderr, "%s: ERROR: tensor is null or has no data\n", __func__);
        return;
    }

    if (debug > 0) {
        fprintf(stderr, "%s: populating seq_lens tensor [%lld]\n",
                __func__, (long long)tensor->ne[0]);
    }

    // Tensor layout: [n_seqs]
    const int64_t n_seqs = tensor->ne[0];

    // Initialize all entries to 0
    int32_t * data = reinterpret_cast<int32_t *>(tensor->data);
    memset(data, 0, ggml_nbytes(tensor));

    // Fill in sequence lengths
    int seq_idx = 0;
    for (const auto & entry : block_tables) {
        if (seq_idx >= n_seqs) {
            fprintf(stderr, "%s: WARNING: more sequences than tensor space (%d >= %lld)\n",
                    __func__, seq_idx, (long long)n_seqs);
            break;
        }

        const llama_seq_id seq_id = entry.first;

        // Get length from metadata if available, otherwise calculate from block table
        uint32_t seq_len = 0;
        auto meta_it = seq_meta.find(seq_id);
        if (meta_it != seq_meta.end()) {
            seq_len = meta_it->second.length;
        } else {
            // Fallback: use number of blocks * block_size
            seq_len = static_cast<uint32_t>(entry.second.size() * block_size);
        }

        data[seq_idx] = static_cast<int32_t>(seq_len);

        if (debug > 1) {
            fprintf(stderr, "%s:   seq %d: length = %u\n", __func__, seq_id, seq_len);
        }

        seq_idx++;
    }

    if (debug > 0) {
        fprintf(stderr, "%s: populated %d sequence lengths\n", __func__, seq_idx);
    }
}
