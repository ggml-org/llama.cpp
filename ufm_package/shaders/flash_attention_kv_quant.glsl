#version 460
#extension GL_EXT_shader_16bit_storage                          : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8         : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16      : require
#extension GL_KHR_shader_subgroup_arithmetic                    : require
#extension GL_KHR_shader_subgroup_basic                         : require

// flash_attention_kv_quant.glsl — Flash attention with Q8_0 KV cache
//
// WHAT THIS IS
// ------------
// Flash attention reading KV tensors stored in Q8_0 format.
// Q8_0 is llama.cpp's native 8-bit KV quantisation (-ctk q8_0 -ctv q8_0).
// Using Q8_0 directly means no format conversion step — we read the same
// buffer llama.cpp writes.
//
// Q8_0 FORMAT (QK8_0 = 32 elements per block):
//   Block b: [fp16 scale (2 bytes little-endian)][int8 x 32 (32 bytes)]
//   Total: 34 bytes per block
//   For HEAD_DIM=128: 4 blocks per KV vector
//   Element e maps to block e/32, offset within block e%32
//
// MEMORY SAVINGS vs F16:
//   F16:  HEAD_DIM * 2 = 256 bytes per vector
//   Q8_0: (HEAD_DIM/32) * 34 = 136 bytes per vector = 47% reduction

#define HEAD_DIM_VAL  128
#define TILE_SIZE_VAL 16
#define QK8_0         32    // elements per Q8_0 block
#define Q8_BLOCK_BYTES 34   // bytes per Q8_0 block (2 scale + 32 data)

layout(constant_id = 0) const int HEAD_DIM  = HEAD_DIM_VAL;
layout(constant_id = 1) const int TILE_SIZE = TILE_SIZE_VAL;

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

// Q: fp16 query (one token being decoded)
layout(set = 0, binding = 0) readonly  buffer BufQ  { float16_t Q[];  };

// K: Q8_0 quantised KV cache — layout: blocks of [fp16 scale][int8 x 32]
layout(set = 0, binding = 1) readonly  buffer BufKq { int8_t    Kq[]; };

// V: Q8_0 quantised KV cache — same layout
layout(set = 0, binding = 2) readonly  buffer BufVq { int8_t    Vq[]; };

// Output: fp16
layout(set = 0, binding = 3) writeonly buffer BufO  { float16_t O[];  };

layout(push_constant) uniform PC {
    uint  Sq, Skv, Hq, Hkv, D;
    float scale;
    float logit_softcap;  // 0.0 = disabled; ~50.0 for Gemma 4
    uint  causal;
    uint  kv_stride;      // bytes per KV vector = (D/32)*34
    uint  bs_q, bs_k, bs_v, bs_o;
} pc;

shared float s_q     [HEAD_DIM_VAL];
shared float s_kv    [TILE_SIZE_VAL * HEAD_DIM_VAL];
shared float s_scores[TILE_SIZE_VAL];
shared float s_o     [HEAD_DIM_VAL];
shared float s_m, s_d, s_rescale;

// Inlined K and V versions (GLSL doesn't allow unsized array parameters)
#define LOAD_K_ELEM(bb, e) load_q8_elem_k(bb, e)
#define LOAD_V_ELEM(bb, e) load_q8_elem_v(bb, e)

float load_q8_elem_k(uint byte_base, uint elem) {
    uint block      = elem / uint(QK8_0);
    uint in_block   = elem % uint(QK8_0);
    uint block_base = byte_base + block * uint(Q8_BLOCK_BYTES);
    uint lo = uint(uint8_t(Kq[block_base + 0u]));
    uint hi = uint(uint8_t(Kq[block_base + 1u]));
    float sc = float(unpackHalf2x16((hi << 8u) | lo).x);
    return float(Kq[block_base + 2u + in_block]) * sc;
}

float load_q8_elem_v(uint byte_base, uint elem) {
    uint block      = elem / uint(QK8_0);
    uint in_block   = elem % uint(QK8_0);
    uint block_base = byte_base + block * uint(Q8_BLOCK_BYTES);
    uint lo = uint(uint8_t(Vq[block_base + 0u]));
    uint hi = uint(uint8_t(Vq[block_base + 1u]));
    float sc = float(unpackHalf2x16((hi << 8u) | lo).x);
    return float(Vq[block_base + 2u + in_block]) * sc;
}

void main() {
    uint tid   = gl_LocalInvocationID.x;
    uint q_idx = gl_WorkGroupID.x;
    uint h_q   = gl_WorkGroupID.y;
    uint b     = gl_WorkGroupID.z;
    uint h_kv  = h_q * pc.Hkv / pc.Hq;

    uint q_base      = b * pc.bs_q + h_q  * pc.Sq  * pc.D + q_idx * pc.D;
    uint o_base      = b * pc.bs_o + h_q  * pc.Sq  * pc.D + q_idx * pc.D;
    uint k_head_base = b * pc.bs_k + h_kv * pc.Skv * pc.kv_stride;
    uint v_head_base = b * pc.bs_v + h_kv * pc.Skv * pc.kv_stride;

    s_q[tid] = float(Q[q_base + tid]);
    s_o[tid] = 0.0;
    if (tid == 0u) { s_m = -1e38; s_d = 0.0; }
    subgroupBarrier();

    for (uint kv_start = 0u; kv_start < pc.Skv; kv_start += uint(TILE_SIZE_VAL)) {
        uint tile_len = min(uint(TILE_SIZE_VAL), pc.Skv - kv_start);

        // ── Load K tile, dequantise Q8_0 → fp32 ──────────────────────────────
        for (uint t = 0u; t < tile_len; t++) {
            uint bb = k_head_base + (kv_start + t) * pc.kv_stride;
            s_kv[t * uint(HEAD_DIM_VAL) + tid] = load_q8_elem_k(bb, tid);
        }
        subgroupBarrier();

        // ── QK dot products → scores ──────────────────────────────────────────
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
                if (pc.logit_softcap > 0.0)
                    sc = pc.logit_softcap * tanh(sc / pc.logit_softcap);
                s_scores[tid] = sc;
            }
        } else {
            if (tid < uint(TILE_SIZE_VAL)) s_scores[tid] = -1e38;
        }
        subgroupBarrier();

        // ── Online softmax update ─────────────────────────────────────────────
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

        // ── Load V tile, dequantise Q8_0 → fp32, accumulate ──────────────────
        for (uint t = 0u; t < tile_len; t++) {
            uint bb = v_head_base + (kv_start + t) * pc.kv_stride;
            s_kv[t * uint(HEAD_DIM_VAL) + tid] = load_q8_elem_v(bb, tid);
        }
        subgroupBarrier();

        for (uint t = 0u; t < tile_len; t++)
            s_o[tid] += s_scores[t] * s_kv[t * uint(HEAD_DIM_VAL) + tid];
        subgroupBarrier();
    }

    O[o_base + tid] = float16_t(s_o[tid] / max(s_d, 1e-8));
}
