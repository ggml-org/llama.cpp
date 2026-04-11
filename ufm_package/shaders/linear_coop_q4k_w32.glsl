#version 460
#extension GL_KHR_cooperative_matrix                            : require
#extension GL_KHR_memory_scope_semantics                        : require
#extension GL_KHR_shader_subgroup_basic                         : require
#extension GL_EXT_shader_16bit_storage                          : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16      : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8         : require

// =============================================================================
// linear_coop_q4k_w32.glsl — Q4_K fused dequant + WMMA, WAVE32 variant
//
// WHY WAVE32 (from RDNA4 ISA reference, April 2025):
//   Wave64 executes VALU/VMEM instructions TWICE: once for low 32 work-items,
//   once for high 32. Wave32 issues each instruction only once.
//   Additionally, dynamic VGPR allocation is only available in wave32 mode,
//   which can improve occupancy when register pressure is high.
//
// COMPARED TO linear_coop_q4k.glsl (wave64):
//   - local_size_x = 32 instead of 64
//   - Each thread owns 8 coopmat elements (256/32) instead of 4 (256/64)
//   - Inner loops run: for i in 0..7 instead of 0..3
//   - Coverage per workgroup: still 16×16 output tile
//   - LDS usage: unchanged (same arrays, half the threads writing)
//   - Barrier domain: halved — potentially faster barrier resolution
//
// HOW TO A/B TEST:
//   1. Compile both shaders, create two pipelines in ggml-vulkan.cpp
//   2. Run llama-bench -r 5 with each pipeline
//   3. Keep whichever gives better tokens/sec
//   RDNA4 performance guide says workgroup multiples of 64 are safest for
//   cross-GPU compat, but wave32 with dual-issue is preferred for
//   pure RDNA4 compute. Test both.
//
// SPEC NOTE: VK_KHR_cooperative_matrix maps gl_ScopeSubgroup to the wave.
//   In wave32 mode the subgroup size is 32. coopmat<float16_t, Subgroup, 16, 16>
//   with 32 threads → each thread holds 16*16/32 = 8 elements. length() = 8.
// =============================================================================

layout(local_size_x = 32) in;   // WAVE32

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

void get_scale_min_k4(uint sb, uint j, out float sc, out float m) {
    if (j < 4u) {
        sc = float(uint(w_bytes[sb + j      ]) & 0x3Fu);
        m  = float(uint(w_bytes[sb + j + 4u ]) & 0x3Fu);
    } else {
        sc = float( (uint(w_bytes[sb + j + 4u]) & 0x0Fu)
                  | ((uint(w_bytes[sb + j - 4u]) >> 6u) << 4u) );
        m  = float( (uint(w_bytes[sb + j + 4u]) >> 4u)
                  | ((uint(w_bytes[sb + j    ]) >> 6u) << 4u) );
    }
}

uint q4k_nibble(uint bb, uint w) {
    uint g = w / 64u; uint p = w % 32u; uint s = (w % 64u) / 32u;
    uint bval = uint(w_bytes[bb + 16u + g * 32u + p]);
    return (s == 0u) ? (bval & 0xFu) : (bval >> 4u);
}

void main() {
    uint tid      = gl_LocalInvocationID.x;   // 0..31
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
            uint j_sub    = w_offset / 32u;

            // Wave32: 32 threads × 8 elements = 256 = 16×16 tile
            for (uint i = 0u; i < 8u; i++) {
                uint elem  = tid * 8u + i;
                uint w_row = out_row_base + elem / 16u;
                uint w_col = elem % 16u;
                float16_t dq = float16_t(0.0);
                if (w_row < pc.M) {
                    uint bb    = (w_row * pc.blocks_per_row + blk) * 144u;
                    float d    = float(unpackHalf2x16((uint(w_bytes[bb+1u])<<8u)|uint(w_bytes[bb])).x);
                    float dmin = float(unpackHalf2x16((uint(w_bytes[bb+3u])<<8u)|uint(w_bytes[bb+2u])).x);
                    float sc, mv;
                    get_scale_min_k4(bb + 4u, j_sub, sc, mv);
                    dq = float16_t(d * sc * float(q4k_nibble(bb, w_offset + w_col)) - dmin * mv);
                }
                s_w[elem] = dq;
            }

            uint k_base = blk * 256u + w_offset;
            for (uint i = 0u; i < 8u; i++) {
                uint elem  = tid * 8u + i;
                uint k_abs = k_base + elem / 16u;
                uint n_abs = out_col_base + elem % 16u;
                s_a[elem] = (k_abs < pc.K && n_abs < pc.N)
                            ? a[k_abs * pc.N + n_abs] : float16_t(0.0);
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
        for (uint i = 0u; i < 8u; i++) {
            uint elem = tid * 8u + i;
            uint r = elem / 16u, c = elem % 16u;
            uint gr = out_row_base + r, gc = out_col_base + c;
            if (gr < pc.M && gc < pc.N) o[gr * pc.N + gc] = s_out[elem];
        }
    }
}
