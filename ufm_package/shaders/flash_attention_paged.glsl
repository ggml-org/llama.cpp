#version 460
#extension GL_EXT_shader_16bit_storage                          : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8         : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16      : require
#extension GL_KHR_shader_subgroup_arithmetic                    : require
#extension GL_KHR_shader_subgroup_basic                         : require

// =============================================================================
// flash_attention_paged.glsl — Paged KV Cache attention (Piece 2 GPU shader)
//
// VULKAN 1.2 FIXES (same as flash_attention_kv_quant.glsl):
//   1. Shared array sizes use #define literals, not spec constants.
//   2. Scale load inlined as macros, no unsized array function parameters.
//
// PAGED KV addition vs flash_attention_kv_quant:
//   - Binding 4: block_table[] — physical block index per token slot
//   - KV addresses resolved through block table:
//       phys_block = block_table[token_pos / TOKENS_PER_BLOCK]
//       byte_base  = phys_block * bytes_per_phys_block
//                  + (token_pos % TOKENS_PER_BLOCK) * token_stride
//                  + head_idx * kv_heads_stride
//                  + (K=0 or V=kv_stride_per_head)
//   - PKV_INVALID_BLOCK (0xFFFFFFFF) handled: reads zero
// =============================================================================

#define HEAD_DIM_VAL  128
#define TILE_SIZE_VAL 16

layout(constant_id = 0) const int HEAD_DIM  = HEAD_DIM_VAL;
layout(constant_id = 1) const int TILE_SIZE = TILE_SIZE_VAL;

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly  buffer BufQ  { float16_t Q[];  };
layout(set = 0, binding = 1) readonly  buffer BufKq { int8_t    Kq[]; };
layout(set = 0, binding = 2) readonly  buffer BufVq { int8_t    Vq[]; };
layout(set = 0, binding = 3) writeonly buffer BufO  { float16_t O[];  };
layout(set = 0, binding = 4) readonly  buffer BufBT { uint block_table[]; };

layout(push_constant) uniform PC {
    uint  Sq, Skv, Hq, Hkv, D;
    float scale;
    float logit_softcap;  // 0.0 = disabled; ~50.0 for Gemma 4
    uint  causal;
    uint  kv_stride_per_head;
    uint  kv_heads_stride;
    uint  token_stride;
    uint  bytes_per_phys_block;
    uint  max_block_slots;
    uint  bs_q, bs_o;
} pc;

shared float s_q     [HEAD_DIM_VAL];
shared float s_kv    [TILE_SIZE_VAL * HEAD_DIM_VAL];
shared float s_scores[TILE_SIZE_VAL];
shared float s_o     [HEAD_DIM_VAL];
shared float s_m, s_d, s_rescale;

// Block table address resolution
// Returns 0xFFFFFFFF if block is invalid
uint kv_byte_base(uint kv_pos, uint h_kv, bool is_v) {
    uint slot = kv_pos / uint(TILE_SIZE_VAL);
    if (slot >= pc.max_block_slots) return 0xFFFFFFFFu;
    uint phys = block_table[slot];
    if (phys == 0xFFFFFFFFu) return 0xFFFFFFFFu;
    uint tok  = kv_pos % uint(TILE_SIZE_VAL);
    uint base = phys * pc.bytes_per_phys_block
              + tok  * pc.token_stride
              + h_kv * pc.kv_heads_stride;
    return is_v ? base + pc.kv_stride_per_head : base;
}

#define LOAD_K_SCALE(bb) float(unpackHalf2x16( \
    (uint(uint8_t(Kq[(bb) + uint(HEAD_DIM_VAL) + 1u])) << 8u) | \
     uint(uint8_t(Kq[(bb) + uint(HEAD_DIM_VAL)]))).x)

#define LOAD_V_SCALE(bb) float(unpackHalf2x16( \
    (uint(uint8_t(Vq[(bb) + uint(HEAD_DIM_VAL) + 1u])) << 8u) | \
     uint(uint8_t(Vq[(bb) + uint(HEAD_DIM_VAL)]))).x)

void main() {
    uint tid   = gl_LocalInvocationID.x;
    uint q_idx = gl_WorkGroupID.x;
    uint h_q   = gl_WorkGroupID.y;
    uint b     = gl_WorkGroupID.z;
    uint h_kv  = h_q * pc.Hkv / pc.Hq;

    uint q_base = b * pc.bs_q + h_q * pc.Sq * pc.D + q_idx * pc.D;
    uint o_base = b * pc.bs_o + h_q * pc.Sq * pc.D + q_idx * pc.D;

    s_q[tid] = float(Q[q_base + tid]);
    s_o[tid] = 0.0;
    if (tid == 0u) { s_m = -1e38; s_d = 0.0; }
    subgroupBarrier();

    for (uint kv_start = 0u; kv_start < pc.Skv; kv_start += uint(TILE_SIZE_VAL)) {
        uint tile_len = min(uint(TILE_SIZE_VAL), pc.Skv - kv_start);

        for (uint t = 0u; t < tile_len; t++) {
            uint bb = kv_byte_base(kv_start + t, h_kv, false);
            if (bb == 0xFFFFFFFFu) {
                s_kv[t * uint(HEAD_DIM_VAL) + tid] = 0.0;
            } else {
                s_kv[t * uint(HEAD_DIM_VAL) + tid] = float(Kq[bb + tid]) * LOAD_K_SCALE(bb);
            }
        }
        subgroupBarrier();

        if (tid < tile_len) {
            uint pos    = kv_start + tid;
            bool masked = (pc.causal != 0u) && (pos > q_idx);
            if (masked) {
                s_scores[tid] = -1e38;
            } else {
                float dot = 0.0;
                for (uint d = 0u; d < pc.D; d++)
                    dot += s_q[d] * s_kv[tid * uint(HEAD_DIM_VAL) + d];
                float sc = dot * pc.scale;
                // logit_softcap: scores = cap * tanh(scores / cap)
                // Gemma 4 uses cap=50.0; cap=0.0 means disabled
                if (pc.logit_softcap > 0.0) {
                    sc = pc.logit_softcap * tanh(sc / pc.logit_softcap);
                }
                s_scores[tid] = sc;
            }
        } else {
            if (tid < uint(TILE_SIZE_VAL)) s_scores[tid] = -1e38;
        }
        subgroupBarrier();

        float my_score = (tid < uint(TILE_SIZE_VAL)) ? s_scores[tid] : -1e38;
        float tile_max = subgroupMax(my_score);

        if (tid == 0u) {
            float m_new = max(s_m, tile_max);
            s_rescale   = exp(s_m - m_new);
            s_m         = m_new;
        }
        subgroupBarrier();

        s_o[tid] *= s_rescale;
        subgroupBarrier();

        float my_exp = 0.0;
        if (tid < tile_len) {
            my_exp        = exp(s_scores[tid] - s_m);
            s_scores[tid] = my_exp;
        }
        float tile_sum = subgroupAdd(my_exp);
        if (tid == 0u) s_d = s_d * s_rescale + tile_sum;
        subgroupBarrier();

        for (uint t = 0u; t < tile_len; t++) {
            uint bb = kv_byte_base(kv_start + t, h_kv, true);
            if (bb == 0xFFFFFFFFu) {
                s_kv[t * uint(HEAD_DIM_VAL) + tid] = 0.0;
            } else {
                s_kv[t * uint(HEAD_DIM_VAL) + tid] = float(Vq[bb + tid]) * LOAD_V_SCALE(bb);
            }
        }
        subgroupBarrier();

        for (uint t = 0u; t < tile_len; t++)
            s_o[tid] += s_scores[t] * s_kv[t * uint(HEAD_DIM_VAL) + tid];
        subgroupBarrier();
    }

    O[o_base + tid] = float16_t(s_o[tid] / max(s_d, 1e-8));
}
