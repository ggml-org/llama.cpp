#version 460
#extension GL_KHR_cooperative_matrix                            : require
#extension GL_KHR_memory_scope_semantics                        : require
#extension GL_KHR_shader_subgroup_basic                         : require
#extension GL_EXT_shader_16bit_storage                          : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16      : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8         : require

// =============================================================================
// linear_coop_q4k_silu.glsl — Fused Q4_K GEMM + SiLU gate (LLaMA FFN)
//
// FIX vs previous version: scale caching (same as linear_coop_q4k.glsl)
//   d, dmin, sc, mv are now hoisted outside the w_col loop.
//   j_sub = w_offset / 32 is constant for all 16 weights in a tile.
//
// Also applies to both _g (gate) and _u (up) weight paths.
// Both paths share the same activation tile s_a.
//
// Duplicate helpers for each binding remain (GLSL has no buffer reference params).
// =============================================================================

layout(local_size_x = 64) in;

layout(set = 0, binding = 0) readonly  buffer BufWgate { uint8_t  wg_bytes[]; };
layout(set = 0, binding = 1) readonly  buffer BufA     { float16_t a[]; };
layout(set = 0, binding = 2) readonly  buffer BufWup   { uint8_t  wu_bytes[]; };
layout(set = 0, binding = 3) writeonly buffer BufO     { float16_t o[]; };

layout(push_constant) uniform PC {
    uint M, N, K;
    uint blocks_per_row;
} pc;

shared float16_t s_wg [16 * 16];
shared float16_t s_wu [16 * 16];
shared float16_t s_a  [16 * 16];
shared float16_t s_out[16 * 16];

// ── Gate helpers ─────────────────────────────────────────────────────────────

void get_scale_min_k4_g(uint sb, uint j, out float sc, out float m) {
    if (j < 4u) {
        sc = float(uint(wg_bytes[sb + j      ]) & 0x3Fu);
        m  = float(uint(wg_bytes[sb + j + 4u ]) & 0x3Fu);
    } else {
        sc = float( (uint(wg_bytes[sb + j + 4u]) & 0x0Fu)
                  | ((uint(wg_bytes[sb + j - 4u]) >> 6u) << 4u) );
        m  = float( (uint(wg_bytes[sb + j + 4u]) >> 4u)
                  | ((uint(wg_bytes[sb + j    ]) >> 6u) << 4u) );
    }
}
uint q4k_nibble_g(uint bb, uint w) {
    uint g = w/64u; uint p = w%32u; uint s = (w%64u)/32u;
    uint bval = uint(wg_bytes[bb + 16u + g*32u + p]);
    return (s==0u) ? (bval & 0xFu) : (bval >> 4u);
}

// ── Up helpers ────────────────────────────────────────────────────────────────

void get_scale_min_k4_u(uint sb, uint j, out float sc, out float m) {
    if (j < 4u) {
        sc = float(uint(wu_bytes[sb + j      ]) & 0x3Fu);
        m  = float(uint(wu_bytes[sb + j + 4u ]) & 0x3Fu);
    } else {
        sc = float( (uint(wu_bytes[sb + j + 4u]) & 0x0Fu)
                  | ((uint(wu_bytes[sb + j - 4u]) >> 6u) << 4u) );
        m  = float( (uint(wu_bytes[sb + j + 4u]) >> 4u)
                  | ((uint(wu_bytes[sb + j    ]) >> 6u) << 4u) );
    }
}
uint q4k_nibble_u(uint bb, uint w) {
    uint g = w/64u; uint p = w%32u; uint s = (w%64u)/32u;
    uint bval = uint(wu_bytes[bb + 16u + g*32u + p]);
    return (s==0u) ? (bval & 0xFu) : (bval >> 4u);
}

float silu(float x) { return x / (1.0 + exp(-x)); }

void main() {
    uint tid      = gl_LocalInvocationID.x;
    uint tile_row = gl_WorkGroupID.y;
    uint tile_col = gl_WorkGroupID.x;
    uint out_row_base = tile_row * 16u;
    uint out_col_base = tile_col * 16u;
    if (out_row_base >= pc.M || out_col_base >= pc.N) return;

    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc_g, acc_u;
    for (uint i = 0u; i < acc_g.length(); i++) { acc_g[i] = float16_t(0.0); acc_u[i] = float16_t(0.0); }

    for (uint blk = 0u; blk < pc.blocks_per_row; blk++) {
        for (uint sub_tile = 0u; sub_tile < 16u; sub_tile++) {
            uint w_offset = sub_tile * 16u;
            uint j_sub    = w_offset / 32u;   // constant for all w_col in this tile

            // Dequant gate weights — scale hoisted
            for (uint i = 0u; i < 4u; i++) {
                uint elem  = tid * 4u + i;
                uint w_row = out_row_base + elem / 16u;
                uint w_col = elem % 16u;
                float16_t dq = float16_t(0.0);
                if (w_row < pc.M) {
                    uint bb   = (w_row * pc.blocks_per_row + blk) * 144u;
                    float d   = float(unpackHalf2x16((uint(wg_bytes[bb+1u])<<8u)|uint(wg_bytes[bb])).x);
                    float dmin= float(unpackHalf2x16((uint(wg_bytes[bb+3u])<<8u)|uint(wg_bytes[bb+2u])).x);
                    float sc, mv;
                    get_scale_min_k4_g(bb + 4u, j_sub, sc, mv);
                    dq = float16_t(d * sc * float(q4k_nibble_g(bb, w_offset + w_col)) - dmin * mv);
                }
                s_wg[elem] = dq;
            }

            // Dequant up weights — scale hoisted
            for (uint i = 0u; i < 4u; i++) {
                uint elem  = tid * 4u + i;
                uint w_row = out_row_base + elem / 16u;
                uint w_col = elem % 16u;
                float16_t dq = float16_t(0.0);
                if (w_row < pc.M) {
                    uint bb   = (w_row * pc.blocks_per_row + blk) * 144u;
                    float d   = float(unpackHalf2x16((uint(wu_bytes[bb+1u])<<8u)|uint(wu_bytes[bb])).x);
                    float dmin= float(unpackHalf2x16((uint(wu_bytes[bb+3u])<<8u)|uint(wu_bytes[bb+2u])).x);
                    float sc, mv;
                    get_scale_min_k4_u(bb + 4u, j_sub, sc, mv);
                    dq = float16_t(d * sc * float(q4k_nibble_u(bb, w_offset + w_col)) - dmin * mv);
                }
                s_wu[elem] = dq;
            }

            // Shared activation tile
            uint k_base = blk * 256u + w_offset;
            for (uint i = 0u; i < 4u; i++) {
                uint elem = tid * 4u + i;
                uint k_abs = k_base + elem / 16u;
                uint n_abs = out_col_base + elem % 16u;
                s_a[elem] = (k_abs < pc.K && n_abs < pc.N) ? a[k_abs * pc.N + n_abs] : float16_t(0.0);
            }
            barrier();

            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matA;
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> matWg, matWu;
            coopMatLoad(matA,  s_a,  0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
            coopMatLoad(matWg, s_wg, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
            coopMatLoad(matWu, s_wu, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
            acc_g = coopMatMulAdd(matWg, matA, acc_g);
            acc_u = coopMatMulAdd(matWu, matA, acc_u);
            barrier();
        }
    }

    for (uint i = 0u; i < acc_g.length(); i++)
        acc_g[i] = float16_t(silu(float(acc_g[i])) * float(acc_u[i]));

    bool boundary = (out_row_base + 16u > pc.M) || (out_col_base + 16u > pc.N);
    if (!boundary) {
        coopMatStore(acc_g, o, out_row_base * pc.N + out_col_base, pc.N,
                     gl_CooperativeMatrixLayoutRowMajor);
    } else {
        coopMatStore(acc_g, s_out, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
        barrier();
        for (uint i = 0u; i < 4u; i++) {
            uint elem = tid * 4u + i;
            uint r = elem / 16u, c = elem % 16u;
            uint gr = out_row_base + r, gc = out_col_base + c;
            if (gr < pc.M && gc < pc.N) o[gr * pc.N + gc] = s_out[elem];
        }
    }
}
