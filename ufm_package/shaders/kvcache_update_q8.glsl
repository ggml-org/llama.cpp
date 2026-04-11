#version 460
#extension GL_EXT_shader_16bit_storage                          : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8         : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16      : require
#extension GL_KHR_shader_subgroup_arithmetic                    : require
#extension GL_KHR_shader_subgroup_basic                         : require

// kvcache_update_q8.glsl — Write one KV head vector in Q8_0 format
//
// Q8_0 FORMAT (matches ggml block_q8_0, QK8_0=32):
//   Each 32-element block: [fp16 scale (2 bytes)][int8 x 32 (32 bytes)] = 34 bytes
//   For HEAD_DIM=128: 4 blocks per vector, 136 bytes total
//   Block b covers elements [b*32 .. b*32+31]
//   Scale = max(abs(elements in block)) / 127.0
//
// This matches exactly what llama.cpp writes when -ctk q8_0 is used,
// so the FA shader can read Q8_0 KV tensors directly.
//
// DISPATCH: one workgroup per (head, token) pair
//   local_size_x = HEAD_DIM (e.g. 128)
//   Each thread handles one element of the head vector.

layout(constant_id = 0) const int HEAD_DIM  = 128;
layout(constant_id = 1) const int BLOCK_SIZE = 32;   // QK8_0
layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly buffer BufSrc { float16_t src[]; };
layout(set = 0, binding = 1) buffer   BufDst { int8_t   dst[]; };

layout(push_constant) uniform PC {
    uint seq_pos;        // token position being written
    uint max_seq;        // KV cache capacity
    uint n_heads;        // number of KV heads
    uint head_dim;       // elements per head (== HEAD_DIM spec const)
    uint head_idx;       // which KV head
    uint layer_idx;      // which layer
    uint cache_stride;   // bytes per token per head = (head_dim/32)*34
} pc;

// Shared scale per Q8_0 block — 4 blocks for HEAD_DIM=128
// General: HEAD_DIM / BLOCK_SIZE blocks
#define MAX_BLOCKS 8   // supports up to HEAD_DIM=256
shared float s_block_scale[MAX_BLOCKS];

void main() {
    uint tid = gl_LocalInvocationID.x;   // 0 .. HEAD_DIM-1
    if (pc.seq_pos >= pc.max_seq) return;

    uint n_blocks = uint(HEAD_DIM) / uint(BLOCK_SIZE);  // e.g. 4 for HEAD_DIM=128

    // Source: fp16 head vector
    uint src_base = pc.head_idx * pc.head_dim;

    // Destination: Q8_0 layout in KV cache
    // Layout: [layer][head][seq_pos][cache_stride bytes]
    // cache_stride = n_blocks * 34 bytes
    uint dst_base = (pc.layer_idx * pc.n_heads + pc.head_idx)
                  * pc.max_seq * pc.cache_stride
                  + pc.seq_pos * pc.cache_stride;

    // ── Step 1: Per-block max reduction ──────────────────────────────────────
    // Each thread belongs to block (tid / BLOCK_SIZE)
    uint block_idx = tid / uint(BLOCK_SIZE);
    float val = float(src[src_base + tid]);
    float local_abs = abs(val);

    // Reduce within each block of 32 using subgroup ops where possible,
    // then write to shared. Use atomic approach for generality:
    // Each thread does subgroupMax within its warp, leader writes.
    // For wave64: threads 0-31 are in same subgroup as 32-63, etc.
    // Simple approach: all threads write to shared via atomicMax equivalent.
    // Since GLSL lacks atomicMax on float, use barrier + sequential write.

    // Write local abs to shared scratch using block_idx
    // (safe because all threads in a block write same index)
    if (tid % uint(BLOCK_SIZE) == 0u) {
        s_block_scale[block_idx] = 0.0;
    }
    barrier();

    // Each thread conditionally updates its block's max
    // Use subgroup reduce within block lanes
    float sg_max = subgroupMax(local_abs);
    // Thread with lowest lane id in this block writes
    uint lane_in_block = gl_SubgroupInvocationID % uint(BLOCK_SIZE);
    if (lane_in_block == 0u) {
        // Safe: only one thread per block writes here due to lane check
        s_block_scale[block_idx] = max(s_block_scale[block_idx], sg_max);
    }
    barrier();

    // Convert max to scale
    if (tid % uint(BLOCK_SIZE) == 0u) {
        s_block_scale[block_idx] = max(s_block_scale[block_idx] / 127.0, 1e-8);
    }
    barrier();

    // ── Step 2: Quantise and write ───────────────────────────────────────────
    float scale = s_block_scale[block_idx];
    int q = clamp(int(round(val / scale)), -127, 127);

    // Q8_0 block layout: [fp16 scale 2 bytes][int8 x 32]
    // Byte offset of this element within dst:
    //   block b starts at b * 34
    //   scale at b*34 + 0..1
    //   data  at b*34 + 2 + (tid % 32)
    uint elem_in_block = tid % uint(BLOCK_SIZE);
    uint block_byte_base = dst_base + block_idx * 34u;

    dst[block_byte_base + 2u + elem_in_block] = int8_t(q);

    // First thread in each block writes the scale
    if (elem_in_block == 0u) {
        uint scale_bits = packHalf2x16(vec2(scale, 0.0));
        dst[block_byte_base + 0u] = int8_t( scale_bits        & 0xFFu);
        dst[block_byte_base + 1u] = int8_t((scale_bits >> 8u) & 0xFFu);
    }
}
