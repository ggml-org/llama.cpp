#version 460
#extension GL_KHR_cooperative_matrix                            : require
#extension GL_KHR_memory_scope_semantics                        : require
#extension GL_KHR_shader_subgroup_basic                         : require
#extension GL_KHR_shader_subgroup_arithmetic                    : require
#extension GL_KHR_shader_subgroup_shuffle                       : require
#extension GL_EXT_shader_16bit_storage                          : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16      : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8         : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32        : require

// linear_coop_q8.glsl — INT8 cooperative matrix GEMM for Q8_0 weights
//
// GAP FIXED: fp32 coopmat accumulator (was scalar shared array)
// ─────────────────────────────────────────────────────────────
// Previous version used shared float s_acc_f[256] to accumulate across blocks.
// That meant every sub-tile forced a full 1KB LDS write + scalar read-back,
// wasting the int32 result that was already sitting in register-resident coopmat.
//
// This version:
//   acc_fp32 = coopmat<float, Subgroup, 16, 16, Accumulator>
//   Lives in registers for the entire K-loop.
//   Per sub-tile: INT8 WMMA -> int32 coopmat
//               -> stage to LDS (needed for per-row/col scale indexing)
//               -> apply wscale[r] * ascale[c] per element
//               -> load back as fp32 coopmat
//               -> acc_fp32 += scaled_fp32  (register op)
//   Only 2 mandatory LDS trips per sub-tile (down from 3).
//   acc_fp32 never touches LDS until the final store.
//
// SPEC BASIS (GL_KHR_cooperative_matrix):
//   coopmat<float, gl_ScopeSubgroup, 16, 16, Accumulator> is a valid type.
//   coopMatLoad accepts shared float[] as source buffer.
//   Element-wise acc_fp32[i] += x is valid coopmat arithmetic.
//
// RUNTIME GUARD (needed in ggml-vulkan.cpp):
//   Before creating this pipeline, verify via:
//     vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR
//   that AType=int8, BType=int8, CType=float32, ResultType=float32, 16x16
//   is listed. On RDNA4 gfx1201 this IS reported (confirmed via FSR4/VKD3D path).
//   If absent, fall back to linear_coop_q8_scalar.glsl (the old version).
//
// Q8_0 FORMAT: 34 bytes = [fp16 scale][32 x int8]
// LDS: 3200 bytes total, limit is 64KB on RDNA4.

layout(local_size_x = 64) in;

layout(set = 0, binding = 0) readonly  buffer BufW { int8_t    w_bytes[]; };
layout(set = 0, binding = 1) readonly  buffer BufA { float16_t a[]; };
layout(set = 0, binding = 2) writeonly buffer BufO { float16_t o[]; };

layout(push_constant) uniform PC {
    uint M, N, K;
    uint blocks_per_row;
} pc;

shared int8_t    s_wq    [16 * 16];
shared int8_t    s_aq    [16 * 16];
shared int32_t   s_int32 [16 * 16];
shared float     s_fp32  [16 * 16];
shared float     s_wscale[16];
shared float     s_ascale[16];
shared float16_t s_out   [16 * 16];

float q8_block_scale(uint byte_base) {
    uint lo = uint(uint8_t(w_bytes[byte_base]));
    uint hi = uint(uint8_t(w_bytes[byte_base + 1u]));
    return float(unpackHalf2x16((hi << 8u) | lo).x);
}

void main() {
    uint tid          = gl_LocalInvocationID.x;
    uint out_row_base = gl_WorkGroupID.y * 16u;
    uint out_col_base = gl_WorkGroupID.x * 16u;
    if (out_row_base >= pc.M || out_col_base >= pc.N) return;

    // fp32 coopmat accumulator — register-resident for the entire K loop
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc_fp32;
    for (uint i = 0u; i < acc_fp32.length(); i++) acc_fp32[i] = float(0.0);

    for (uint blk = 0u; blk < pc.blocks_per_row; blk++) {

        // Weight block scale for 16 output rows
        if (tid < 16u) {
            uint w_row = out_row_base + tid;
            s_wscale[tid] = (w_row < pc.M)
                ? q8_block_scale((w_row * pc.blocks_per_row + blk) * 34u)
                : 1.0;
        }
        barrier();

        // 32 weights per Q8_0 block -> 2 sub-tiles of 16
        for (uint sub = 0u; sub < 2u; sub++) {
            uint w_k_off    = sub * 16u;
            uint k_abs_base = blk * 32u + w_k_off;

            // Load int8 weight sub-tile
            for (uint i = 0u; i < 4u; i++) {
                uint elem  = tid * 4u + i;
                uint w_row = out_row_base + elem / 16u;
                uint w_off = elem % 16u;
                int8_t wv  = int8_t(0);
                if (w_row < pc.M) {
                    uint bb = (w_row * pc.blocks_per_row + blk) * 34u;
                    wv = w_bytes[bb + 2u + w_k_off + w_off];
                }
                s_wq[elem] = wv;
            }

            // Dynamic activation scale: all 64 threads, per column
            // threads 0,16,32,48 share column out_col_base+0, etc.
            {
                uint n_col = out_col_base + (tid % 16u);
                float mx = 0.0;
                if (n_col < pc.N) {
                    for (uint k = 0u; k < 16u; k++) {
                        uint k_abs = k_abs_base + k;
                        if (k_abs < pc.K)
                            mx = max(mx, abs(float(a[k_abs * pc.N + n_col])));
                    }
                }
                float mx2 = max(mx, subgroupShuffleXor(mx, 16u));
                float mx3 = max(mx2, subgroupShuffleXor(mx2, 32u));
                if (tid < 16u) s_ascale[tid] = max(mx3 / 127.0, 1e-8);
            }
            barrier();

            // Quantise activation sub-tile to int8
            for (uint i = 0u; i < 4u; i++) {
                uint elem  = tid * 4u + i;
                uint k_off = elem / 16u;
                uint n_off = elem % 16u;
                uint k_abs = k_abs_base + k_off;
                uint n_abs = out_col_base + n_off;
                int8_t aq  = int8_t(0);
                if (k_abs < pc.K && n_abs < pc.N) {
                    float v = float(a[k_abs * pc.N + n_abs]) / s_ascale[n_off];
                    aq = int8_t(clamp(int(round(v)), -127, 127));
                }
                s_aq[elem] = aq;
            }
            barrier();

            // INT8 WMMA -> int32 coopmat result
            coopmat<int8_t,  gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>          matW;
            coopmat<int8_t,  gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>          matA;
            coopmat<int32_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> sub_acc;
            for (uint i = 0u; i < sub_acc.length(); i++) sub_acc[i] = 0;
            coopMatLoad(matW, s_wq, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
            coopMatLoad(matA, s_aq, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
            sub_acc = coopMatMulAdd(matW, matA, sub_acc);

            // Stage int32 result row-major into LDS
            // s_int32[r*16+c] = result at output tile position (r, c)
            coopMatStore(sub_acc, s_int32, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
            barrier();

            // Apply dequant: each thread scales its 4 elements
            // elem = tid*4+i, r = elem/16, c = elem%16 (matches RowMajor layout)
            for (uint i = 0u; i < 4u; i++) {
                uint elem = tid * 4u + i;
                uint r    = elem / 16u;
                uint c    = elem % 16u;
                s_fp32[elem] = float(s_int32[elem]) * s_wscale[r] * s_ascale[c];
            }
            barrier();

            // Load scaled fp32 values as a coopmat and accumulate (register op)
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> scaled_fp32;
            coopMatLoad(scaled_fp32, s_fp32, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
            for (uint i = 0u; i < acc_fp32.length(); i++)
                acc_fp32[i] += scaled_fp32[i];

            barrier();
        }
    }

    // Downcast accumulator to fp16 for output
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> out_mat;
    for (uint i = 0u; i < out_mat.length(); i++)
        out_mat[i] = float16_t(acc_fp32[i]);

    bool boundary = (out_row_base + 16u > pc.M) || (out_col_base + 16u > pc.N);
    if (!boundary) {
        coopMatStore(out_mat, o, out_row_base * pc.N + out_col_base, pc.N,
                     gl_CooperativeMatrixLayoutRowMajor);
    } else {
        coopMatStore(out_mat, s_out, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
        barrier();
        for (uint i = 0u; i < 4u; i++) {
            uint elem = tid * 4u + i;
            uint r = elem / 16u, c = elem % 16u;
            uint gr = out_row_base + r, gc = out_col_base + c;
            if (gr < pc.M && gc < pc.N) o[gr * pc.N + gc] = s_out[elem];
        }
    }
}
