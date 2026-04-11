// paged_kv_upload.h — Block table upload glue, Piece 2b
//
// Bridges paged_kv_cache.h (CPU allocator) to flash_attention_paged.glsl (GPU).
//
// WHAT THIS DOES
// --------------
// Before each decode layer, the GPU needs the block table for this
// (sequence, layer) as a flat uint[] buffer on device. This file provides:
//
//   pkv_upload_block_table()   — copy one layer's block table to a mapped
//                                Vulkan staging buffer, ready for vkCmdCopyBuffer
//   pkv_push_constants()       — fill the PC struct the shader expects
//
// VULKAN MEMORY ASSUMPTIONS
// -------------------------
// The caller owns:
//   - A host-visible staging buffer large enough for max_blocks_per_seq uints
//   - A device-local buffer (BufBT, binding 4) that gets updated each layer
// This is standard double-buffer or ring-buffer upload. We don't manage that
// here — too many ways to do it depending on the Vulkan backend setup.
//
// USAGE (per decode step, per layer):
//
//   // 1. Upload block table for this layer
//   uint32_t n_slots;
//   pkv_upload_block_table(&pkv, seq_id, layer_idx,
//                          staging_ptr, staging_capacity, &n_slots);
//   // vkCmdCopyBuffer(staging -> device BufBT, n_slots * 4 bytes)
//
//   // 2. Fill push constants
//   pkv_pc_t pc = pkv_push_constants(&pkv, Sq, Skv, Hq, q_idx_is_causal);
//   // vkCmdPushConstants(...)
//
//   // 3. Dispatch flash_attention_paged
//   // vkCmdDispatch(Sq, Hq, batch_size)
//
// PIECE DEPENDENCIES
// ------------------
// Piece 1: paged_kv_cache.h     — allocator, block table, LRU
// Piece 2: flash_attention_paged.glsl — GPU shader
// Piece 2b (this): upload glue
// Piece 3 (TODO):  UFM eviction/reload async path

#pragma once
#include "paged_kv_cache.h"
#include <cstring>
#include <cstdio>

// Push constant layout matching flash_attention_paged.glsl PC block exactly.
// Keep in sync with the shader — field order and types must match.
struct pkv_pc_t {
    uint32_t Sq;
    uint32_t Skv;
    uint32_t Hq;
    uint32_t Hkv;
    uint32_t D;
    float    scale;
    uint32_t causal;

    uint32_t kv_stride_per_head;    // HEAD_DIM + 2
    uint32_t kv_heads_stride;       // kv_stride_per_head * 2
    uint32_t token_stride;          // Hkv * kv_heads_stride
    uint32_t bytes_per_phys_block;  // TOKENS_PER_BLOCK * token_stride

    uint32_t max_block_slots;
    uint32_t bs_q;
    uint32_t bs_o;
};

// ── Block table upload ────────────────────────────────────────────────────────
//
// Copies block_table[layer * max_blocks .. +max_blocks] for seq_id into dst.
// dst must be a host-mapped pointer to at least (max_blocks * 4) bytes.
// n_slots_out receives the number of uint32 values written.
//
// Returns false if seq_id not found or dst_capacity too small.
//
// IMPORTANT: physical block ids written here are indices into the KV pool.
// The GPU multiplies by bytes_per_phys_block to get byte offsets.
// This is consistent with pkv_block_t::vram_offset = pool_base + id * bytes_per_block.

inline bool pkv_upload_block_table(const pkv_allocator_t* pkv,
                                    uint32_t seq_id,
                                    uint32_t layer,
                                    void*    dst,
                                    uint32_t dst_capacity_uints,
                                    uint32_t* n_slots_out) {
    auto it = pkv->seqs.find(seq_id);
    if (it == pkv->seqs.end()) {
        fprintf(stderr, "[PKV] pkv_upload_block_table: seq %u not found\n", seq_id);
        return false;
    }
    const pkv_seq_t& seq = it->second;

    if (layer >= seq.n_layers) {
        fprintf(stderr, "[PKV] pkv_upload_block_table: layer %u >= n_layers %u\n",
                layer, seq.n_layers);
        return false;
    }

    uint32_t n_slots = seq.max_blocks;
    if (n_slots > dst_capacity_uints) {
        fprintf(stderr, "[PKV] pkv_upload_block_table: dst too small (%u < %u)\n",
                dst_capacity_uints, n_slots);
        return false;
    }

    const uint32_t* src = seq.block_table.data() + layer * seq.max_blocks;
    memcpy(dst, src, n_slots * sizeof(uint32_t));

    if (n_slots_out) *n_slots_out = n_slots;
    return true;
}

// ── Push constant builder ─────────────────────────────────────────────────────
//
// Computes all derived layout constants from the allocator config.
// Call once per layer dispatch.
//
// Parameters:
//   pkv         - the allocator (has head_dim, n_heads_kv etc.)
//   Sq          - query sequence length (tokens being computed this step, usually 1)
//   Skv         - KV sequence length (total context tokens so far)
//   Hq          - number of query heads
//   causal      - 1 for autoregressive decode, 0 for full attention
//   max_slots   - must match what was uploaded (seq.max_blocks)
//   bs_q, bs_o  - Q/O buffer byte-stride offsets (for batching)
//
// Note on INT8 kv_stride layout:
//   Each K or V vector at a position = HEAD_DIM int8 values + 2 bytes fp16 scale
//   kv_stride_per_head = HEAD_DIM + 2
//   K and V are stored consecutively per head:
//     [K_int8 * HEAD_DIM][K_scale_fp16][V_int8 * HEAD_DIM][V_scale_fp16]
//   kv_heads_stride = 2 * kv_stride_per_head
//   token_stride    = n_heads_kv * kv_heads_stride
//   bytes_per_phys_block = PKV_TOKENS_PER_BLOCK * token_stride

inline pkv_pc_t pkv_push_constants(const pkv_allocator_t* pkv,
                                    uint32_t Sq,
                                    uint32_t Skv,
                                    uint32_t Hq,
                                    uint32_t causal,
                                    uint32_t max_slots,
                                    uint32_t bs_q,
                                    uint32_t bs_o) {
    uint32_t kv_stride_per_head   = pkv->head_dim + 2u;
    uint32_t kv_heads_stride      = kv_stride_per_head * 2u;
    uint32_t token_stride         = pkv->n_heads_kv * kv_heads_stride;
    uint32_t bytes_per_phys_block = PKV_TOKENS_PER_BLOCK * token_stride;

    // Sanity check: does this match what the allocator computed?
    // pkv->bytes_per_block was computed in fp16 (not int8). They WILL differ.
    // int8:  TOKENS * Hkv * (HEAD_DIM + 2) * 2
    // fp16:  TOKENS * Hkv * HEAD_DIM * 2 * 2
    // int8 is smaller. This is expected — the shader uses int8 layout.
    // The pool VRAM allocation should use the larger fp16 size or the
    // caller must ensure the pool was allocated for int8 layout.
    // TODO(Piece 3): reconcile pool allocation with quant format.

    pkv_pc_t pc{};
    pc.Sq                  = Sq;
    pc.Skv                 = Skv;
    pc.Hq                  = Hq;
    pc.Hkv                 = pkv->n_heads_kv;
    pc.D                   = pkv->head_dim;
    pc.scale               = 1.0f / sqrtf((float)pkv->head_dim);
    pc.causal              = causal;
    pc.kv_stride_per_head  = kv_stride_per_head;
    pc.kv_heads_stride     = kv_heads_stride;
    pc.token_stride        = token_stride;
    pc.bytes_per_phys_block = bytes_per_phys_block;
    pc.max_block_slots     = max_slots;
    pc.bs_q                = bs_q;
    pc.bs_o                = bs_o;
    return pc;
}

// ── Block eviction status check ───────────────────────────────────────────────
// Before dispatching the shader, verify all blocks for this layer are in VRAM.
// If any are evicted to RAM, they need to be reloaded first (Piece 3 / UFM).
// Returns the number of blocks currently evicted (should be 0 before dispatch).
inline uint32_t pkv_count_evicted(const pkv_allocator_t* pkv,
                                   uint32_t seq_id,
                                   uint32_t layer) {
    auto it = pkv->seqs.find(seq_id);
    if (it == pkv->seqs.end()) return 0;
    const pkv_seq_t& seq = it->second;

    uint32_t n_slots  = (seq.n_tokens + PKV_TOKENS_PER_BLOCK - 1) / PKV_TOKENS_PER_BLOCK;
    uint32_t evicted  = 0;
    for (uint32_t s = 0; s < n_slots; s++) {
        uint32_t bid = seq.block_table[layer * seq.max_blocks + s];
        if (bid != PKV_INVALID_BLOCK && !pkv->blocks[bid].in_vram)
            evicted++;
    }
    return evicted;
}
