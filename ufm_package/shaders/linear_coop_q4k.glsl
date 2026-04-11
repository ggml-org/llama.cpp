#version 460
#extension GL_KHR_cooperative_matrix                            : require
#extension GL_KHR_memory_scope_semantics                        : require
#extension GL_KHR_shader_subgroup_basic                         : require
#extension GL_EXT_shader_16bit_storage                          : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16      : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8         : require

// =============================================================================
// linear_coop_q4k.glsl — Fused Q4_K dequant + WMMA GEMM
//
// FIX vs previous version: scale caching
// ───────────────────────────────────────
// Previous: q4k_dequant() called per weight = d, dmin, sc, m recomputed 256×
//   per 16×16 tile. Each call reads the 16-byte block header and runs
//   get_scale_min_k4 for every single weight.
//
// Now: per tile-row, hoist d, dmin, sc, m outside the w_col loop.
//   A 16-wide WMMA tile (w_offset..w_offset+15) always falls within ONE Q4_K
//   sub-block of 32 weights (j = w_offset/32, constant for all w_col 0..15
//   since w_offset is a multiple of 16 and w_col < 16).
//   So d, dmin, sc, m are the same for all 16 weights in a row.
//
//   Savings: 256 get_scale_min_k4 calls → 16 per tile (one per output row).
//   get_scale_min_k4 involves several byte reads and bit shifts — not free.
//
// Q4_K BLOCK FORMAT: 144 bytes = 2+2+12+128 (d fp16, dmin fp16, 12 scale bytes, 128 qs bytes)
// Sub-blocks: 8 × 32 weights, indexed j = w_in_block / 32
// Nibble layout: group g = w/64, position p = w%32, sub s = (w%64)/32
// =============================================================================

layout(local_size_x = 64) in;

layout(set = 0, binding = 0) readonly  buffer BufW { uint8_t   w_bytes[]; };
layout(set = 0, binding = 1) readonly  buffer BufA { float16_t a[]; };
layout(set = 0, binding = 2) writeonly buffer BufO { float16_t o[]; };

layout(push_constant) uniform PC {
    uint M, N, K;
    uint blocks_per_row;
} pc;

shared float16_t s_w  [16 * 16];
shared float16_t s_a  [16 * 16];
shared float16_t s_out[16 * 16];

void get_scale_min_k4(uint scales_base, uint j, out float sc, out float m) {
    if (j < 4u) {
        sc = float(uint(w_bytes[scales_base + j      ]) & 0x3Fu);
        m  = float(uint(w_bytes[scales_base + j + 4u ]) & 0x3Fu);
    } else {
        sc = float( (uint(w_bytes[scales_base + j + 4u]) & 0x0Fu)
                  | ((uint(w_bytes[scales_base + j - 4u]) >> 6u) << 4u) );
        m  = float( (uint(w_bytes[scales_base + j + 4u]) >> 4u)
                  | ((uint(w_bytes[scales_base + j    ]) >> 6u) << 4u) );
    }
}

uint q4k_nibble(uint bb, uint w) {
    uint g    = w / 64u;
    uint p    = w % 32u;
    uint s    = (w % 64u) / 32u;
    uint bval = uint(w_bytes[bb + 16u + g * 32u + p]);
    return (s == 0u) ? (bval & 0xFu) : (bval >> 4u);
}

void main() {
    uint tid      = gl_LocalInvocationID.x;
    uint tile_row = gl_WorkGroupID.y;
    uint tile_col = gl_WorkGroupID.x;

    uint out_row_base = tile_row * 16u;
    uint out_col_base = tile_col * 16u;
    if (out_row_base >= pc.M || out_col_base >= pc.N) return;

    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc;
    for (uint i = 0u; i < acc.length(); i++) acc[i] = float16_t(0.0);

    for (uint blk = 0u; blk < pc.blocks_per_row; blk++) {
        for (uint sub_tile = 0u; sub_tile < 16u; sub_tile++) {
            uint w_offset = sub_tile * 16u;
            // Sub-block index j is constant for this entire tile:
            // w_offset is a multiple of 16, w_col < 16, so w_offset+w_col < w_offset+16
            // j = (w_offset + w_col) / 32 = w_offset / 32  (since w_col < 16 can't change j)
            uint j_sub = w_offset / 32u;  // 0..7

            // 64 threads × 4 = 256 elements = 16×16 tile
            for (uint i = 0u; i < 4u; i++) {
                uint elem  = tid * 4u + i;
                uint w_row = out_row_base + elem / 16u;  // which output row
                uint w_col = elem % 16u;                  // which weight within tile

                float16_t dq = float16_t(0.0);
                if (w_row < pc.M) {
                    uint bb = (w_row * pc.blocks_per_row + blk) * 144u;

                    // FIX: hoist d, dmin, sc, m — all constant for this (row, tile)
                    // Each thread computes for its own row, so this is per-element.
                    // Compared to q4k_dequant(), we avoid re-reading the 12 scale bytes
                    // and re-running get_scale_min_k4 for every w_col — j_sub is the same.
                    uint lo_d = uint(w_bytes[bb]);
                    uint hi_d = uint(w_bytes[bb + 1u]);
                    float d    = float(unpackHalf2x16((hi_d << 8u) | lo_d).x);
                    uint lo_m = uint(w_bytes[bb + 2u]);
                    uint hi_m = uint(w_bytes[bb + 3u]);
                    float dmin = float(unpackHalf2x16((hi_m << 8u) | lo_m).x);
                    float sc, mv;
                    get_scale_min_k4(bb + 4u, j_sub, sc, mv);

                    float nibble = float(q4k_nibble(bb, w_offset + w_col));
                    dq = float16_t(d * sc * nibble - dmin * mv);
                }
                s_w[elem] = dq;
            }

            uint k_base = blk * 256u + w_offset;
            for (uint i = 0u; i < 4u; i++) {
                uint elem  = tid * 4u + i;
                uint k_off = elem / 16u;
                uint n_off = elem % 16u;
                uint k_abs = k_base + k_off;
                uint n_abs = out_col_base + n_off;
                s_a[elem] = (k_abs < pc.K && n_abs < pc.N)
                            ? a[k_abs * pc.N + n_abs]
                            : float16_t(0.0);
            }

            barrier();

            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> matW;
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matA;
            coopMatLoad(matW, s_w, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
            coopMatLoad(matA, s_a, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
            acc = coopMatMulAdd(matW, matA, acc);

            barrier();
        }
    }

    bool boundary = (out_row_base + 16u > pc.M) || (out_col_base + 16u > pc.N);
    if (!boundary) {
        coopMatStore(acc, o, out_row_base * pc.N + out_col_base, pc.N,
                     gl_CooperativeMatrixLayoutRowMajor);
    } else {
        coopMatStore(acc, s_out, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
        barrier();
        for (uint i = 0u; i < 4u; i++) {
            uint elem = tid * 4u + i;
            uint r = elem / 16u, c = elem % 16u;
            uint gr = out_row_base + r, gc = out_col_base + c;
            if (gr < pc.M && gc < pc.N) o[gr * pc.N + gc] = s_out[elem];
        }
    }
}
