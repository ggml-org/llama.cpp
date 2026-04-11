#version 460
#extension GL_KHR_cooperative_matrix                            : require
#extension GL_KHR_memory_scope_semantics                        : require
#extension GL_KHR_shader_subgroup_basic                         : require
#extension GL_EXT_shader_16bit_storage                          : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16      : require

// linear_coop_32.glsl — 32x32 tiled fp16 cooperative matrix GEMM
//
// FIX vs previous version:
//   Boundary write path staged through s_stage[16*16] instead of s_A[32*32].
//   Previous version wrote a 16x16 sub-tile into s_A offset 0 then read back
//   from it — fine for q=0, but q=1,2,3 write to the same offset 0 while
//   s_A rows 16-31 still hold stale tile data. Using a dedicated 16x16
//   staging array eliminates any aliasing.

layout(local_size_x = 64) in;

layout(set = 0, binding = 0) readonly  buffer BufA { float16_t A[]; };
layout(set = 0, binding = 1) readonly  buffer BufB { float16_t B[]; };
layout(set = 0, binding = 2) writeonly buffer BufC { float16_t C[]; };

layout(push_constant) uniform PC {
    uint M, N, K;
    float alpha;
} pc;

shared float16_t s_A[32 * 32];      // A tile
shared float16_t s_B[32 * 32];      // B tile
shared float16_t s_stage[16 * 16];  // FIX: dedicated boundary output staging

void load_A_tile(uint row_base, uint k_base, bool bounds) {
    uint tid = gl_LocalInvocationID.x;
    for (uint i = 0u; i < 16u; i++) {
        uint elem = tid * 16u + i;
        uint r = elem / 32u, c = elem % 32u;
        uint gr = row_base + r, gc = k_base + c;
        s_A[elem] = (!bounds || (gr < pc.M && gc < pc.K))
                    ? A[gr * pc.K + gc] : float16_t(0.0);
    }
}

void load_B_tile(uint k_base, uint col_base, bool bounds) {
    uint tid = gl_LocalInvocationID.x;
    for (uint i = 0u; i < 16u; i++) {
        uint elem = tid * 16u + i;
        uint r = elem / 32u, c = elem % 32u;
        uint gr = k_base + r, gc = col_base + c;
        s_B[elem] = (!bounds || (gr < pc.K && gc < pc.N))
                    ? B[gr * pc.N + gc] : float16_t(0.0);
    }
}

void main() {
    uint tid      = gl_LocalInvocationID.x;
    uint row_base = gl_WorkGroupID.y * 32u;
    uint col_base = gl_WorkGroupID.x * 32u;
    if (row_base >= pc.M || col_base >= pc.N) return;

    bool need_bounds = (row_base + 32u > pc.M) || (col_base + 32u > pc.N);

    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc[4];
    for (uint q = 0u; q < 4u; q++)
        for (uint i = 0u; i < acc[q].length(); i++)
            acc[q][i] = float16_t(0.0);

    for (uint k = 0u; k < pc.K; k += 32u) {
        bool bounds = need_bounds || (k + 32u > pc.K);

        load_A_tile(row_base, k, bounds);
        load_B_tile(k, col_base, bounds);
        barrier();

        for (uint ki = 0u; ki < 2u; ki++) {
            uint k_off = ki * 16u;

            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a_top, a_bot;
            coopMatLoad(a_top, s_A, 0u  * 32u + k_off, 32u, gl_CooperativeMatrixLayoutRowMajor);
            coopMatLoad(a_bot, s_A, 16u * 32u + k_off, 32u, gl_CooperativeMatrixLayoutRowMajor);

            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b_left, b_right;
            coopMatLoad(b_left,  s_B, k_off * 32u + 0u,  32u, gl_CooperativeMatrixLayoutRowMajor);
            coopMatLoad(b_right, s_B, k_off * 32u + 16u, 32u, gl_CooperativeMatrixLayoutRowMajor);

            acc[0] = coopMatMulAdd(a_top, b_left,  acc[0]);
            acc[1] = coopMatMulAdd(a_top, b_right, acc[1]);
            acc[2] = coopMatMulAdd(a_bot, b_left,  acc[2]);
            acc[3] = coopMatMulAdd(a_bot, b_right, acc[3]);
        }
        barrier();
    }

    if (pc.alpha != 1.0) {
        for (uint q = 0u; q < 4u; q++)
            for (uint i = 0u; i < acc[q].length(); i++)
                acc[q][i] = float16_t(float(acc[q][i]) * pc.alpha);
    }

    bool boundary = (row_base + 32u > pc.M) || (col_base + 32u > pc.N);
    if (!boundary) {
        coopMatStore(acc[0], C, (row_base +  0u)*pc.N + (col_base +  0u), pc.N, gl_CooperativeMatrixLayoutRowMajor);
        coopMatStore(acc[1], C, (row_base +  0u)*pc.N + (col_base + 16u), pc.N, gl_CooperativeMatrixLayoutRowMajor);
        coopMatStore(acc[2], C, (row_base + 16u)*pc.N + (col_base +  0u), pc.N, gl_CooperativeMatrixLayoutRowMajor);
        coopMatStore(acc[3], C, (row_base + 16u)*pc.N + (col_base + 16u), pc.N, gl_CooperativeMatrixLayoutRowMajor);
    } else {
        // FIX: use s_stage (16x16) not s_A (32x32 reused from tile load)
        const uint sub_row_off[4] = {0u,  0u, 16u, 16u};
        const uint sub_col_off[4] = {0u, 16u,  0u, 16u};
        for (uint q = 0u; q < 4u; q++) {
            coopMatStore(acc[q], s_stage, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
            barrier();
            for (uint i = 0u; i < 4u; i++) {
                uint elem = tid * 4u + i;
                uint r = elem / 16u, c = elem % 16u;
                uint gr = row_base + sub_row_off[q] + r;
                uint gc = col_base + sub_col_off[q] + c;
                if (gr < pc.M && gc < pc.N) C[gr * pc.N + gc] = s_stage[elem];
            }
            barrier();
        }
    }
}
