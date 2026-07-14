#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#else
#define REQD_SUBGROUP_SIZE_64
#endif

// subgroup size for q1 kernels
#ifndef FA_SG
#define FA_SG 64
#endif
#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define REQD_FA_SG __attribute__((intel_reqd_sub_group_size(FA_SG)))
#else
#define REQD_FA_SG
#endif

#ifdef cl_khr_subgroup_shuffle
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable
#define HAS_SUBGROUP_SHUFFLE 1
#elif defined(cl_qcom_subgroup_shuffle)
#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
#define HAS_SUBGROUP_SHUFFLE 1
#endif

#define ACC_TYPE float
#define ACC_TYPE4 float4
#define Q_DATA_TYPE4 float4
#define KV_DATA_TYPE4 half4
#define O_DATA_TYPE4 float4
#define MASK_DATA_TYPE half
#define CONVERT_Q_ACC4(x) (x)
#define CONVERT_KV_ACC4(x) convert_float4(x)
#define CONVERT_O_DATA4(x) (x)

#define DK_VEC (DK/4)
#define DV_VEC (DV/4)

#ifndef FA_PARTIAL_FLOATS
#define FA_PARTIAL_FLOATS (2 + DV)
#endif
#define Q1_WG_SIZE FA_SG

// The kernels are built with -cl-finite-math-only. On some older Adreno GPUs,
// infinite operand can cause undefined behavior and miscompilation for exp.
// Therefore, a large negative value is used instead.
#define FA_M_INIT (-3.0e38f)

// Drop full unroll at DK>=192 — Adreno compiler host-memory budget.
#if DK >= 192
#define FA_UNROLL
#else
#define FA_UNROLL _Pragma("unroll")
#endif

// N_SPLIT>1 splits DK/DV across threads to cut per-thread register use.
#ifndef N_SPLIT
#define N_SPLIT 1
#endif

#define SPLIT_DK_VEC (DK_VEC / N_SPLIT)
#define SPLIT_DV_VEC (DV_VEC / N_SPLIT)

#if N_SPLIT > 1
#define WG_SIZE (BLOCK_M * N_SPLIT)
#else
#define WG_SIZE (BLOCK_M)
#endif

inline float get_alibi_slope(
    const float max_bias, const uint h, const uint n_head_log2, const float m0, const float m1
) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    return pow(base, exph);
}

// Adreno compiler crashes when attempting to compile the entire program for DK=512,
// FA_DECODE_ONLY allows bypass the encoding kernel.
#if !defined(FA_DECODE_ONLY) && !defined(FA_MQ_ONLY)
#ifndef FA_TILE_NAME
#define FA_TILE_NAME flash_attn_f32_f16
#endif
__kernel void FA_TILE_NAME(
    const global void * q_void, ulong q_offset,
#ifdef FA_K_IMG
    __read_only image1d_buffer_t k_img, ulong k_offset_unused,
#else
    const global void * k_void, ulong k_offset,
#endif
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset,
    const global void * k_pad_void,
    const global void * v_pad_void,
    const global void * mask_pad_void,
    const global char * blk,
    const int n_kv_blocks,
    const ulong mask_pad_nb1,
    const ulong mask_pad_nb2,
    const ulong mask_pad_nb3
) {
    const int tid = get_local_id(0);
    const int block_q_idx = get_group_id(0);
    const int head_batch_idx = get_global_id(1);

#if N_SPLIT > 1
    const int q_lane    = tid / N_SPLIT;
    const int split_idx = tid % N_SPLIT;
#else
    const int q_lane    = tid;
    const int split_idx = 0;
#endif

    const int my_query_row = block_q_idx * BLOCK_M + q_lane;
    const int query_valid = my_query_row < n_q;

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;

    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;
    const int mask_head_idx = mask_void != NULL ? head_idx % mask_ne2 : 0;
    const int mask_batch_idx = mask_void != NULL ? batch_idx % mask_ne3 : 0;

    const global char* q_base = (const global char*)q_void + q_offset;
#ifndef FA_K_IMG
    const global char* k_base = (const global char*)k_void + k_offset;
#endif
    const global char* v_base = (const global char*)v_void + v_offset;
    global char* o_base = (global char*)o_void + o_offset;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }
    const global char* mask_pad_base = NULL;
    if (mask_pad_void != NULL) {
        mask_pad_base = (const global char*)mask_pad_void + mask_batch_idx * mask_pad_nb3 + mask_head_idx * mask_pad_nb2;
    }
    const global char* blk_base = NULL;
    if (blk != NULL) {
        const int n_q_blocks = (n_q + BLOCK_M - 1) / BLOCK_M;
        blk_base = blk + (((mask_batch_idx * mask_ne2) + mask_head_idx) * n_q_blocks + block_q_idx) * n_kv_blocks;
    }

    ACC_TYPE4 q_priv[SPLIT_DK_VEC];
    const int dk_off = split_idx * SPLIT_DK_VEC;
    if (query_valid) {
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + my_query_row * q_nb1;
        const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
        FA_UNROLL
        for (int i = 0; i < SPLIT_DK_VEC; ++i) {
            q_priv[i] = CONVERT_Q_ACC4(q_ptr[dk_off + i]);
        }
    } else {
        FA_UNROLL
        for (int i = 0; i < SPLIT_DK_VEC; ++i) {
            q_priv[i] = (ACC_TYPE4)(0.0f);
        }
    }

    ACC_TYPE4 o_acc[SPLIT_DV_VEC];
    FA_UNROLL
    for (int i = 0; i < SPLIT_DV_VEC; ++i) {
        o_acc[i] = (ACC_TYPE4)(0.0f);
    }

    ACC_TYPE m_i = FA_M_INIT;
    ACC_TYPE l_i = 0.0f;

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    __local KV_DATA_TYPE4 l_k[BLOCK_N][DK_VEC];
    __local KV_DATA_TYPE4 l_v[BLOCK_N][DV_VEC];

#if N_SPLIT > 1 && !defined(HAS_SUBGROUP_SHUFFLE)
    __local ACC_TYPE local_partial[BLOCK_N][WG_SIZE];
    __local ACC_TYPE local_p[BLOCK_M][BLOCK_N];
    __local ACC_TYPE local_softmax_scale[BLOCK_M];
    __local ACC_TYPE local_l_inv[BLOCK_M];
#endif

    for (int k_start = 0; k_start < n_kv; k_start += BLOCK_N) {
        char blk_cur = 1;
        if (blk_base != NULL) {
            blk_cur = blk_base[k_start / BLOCK_N];
            if (blk_cur == 0) continue;
        }

        const int use_kv_pad = k_pad_void != NULL && k_start + BLOCK_N > n_kv;
        const int k_tile_start = use_kv_pad ? 0 : k_start;
        const ulong k_tile_nb2 = use_kv_pad ? (ulong) BLOCK_N * k_nb1 : k_nb2;
        const ulong k_tile_nb3 = use_kv_pad ? (ulong) n_head_kv * k_tile_nb2 : k_nb3;
        const ulong v_tile_nb2 = use_kv_pad ? (ulong) BLOCK_N * v_nb1 : v_nb2;
        const ulong v_tile_nb3 = use_kv_pad ? (ulong) n_head_kv * v_tile_nb2 : v_nb3;
#ifdef FA_K_IMG
        // K via texture cache for the bulk (aligned) tiles; the ragged last
        // tile (use_kv_pad) still reads the f32-strided pad buffer from global.
        const global char* k_tile_base = use_kv_pad ? (const global char*) k_pad_void : (const global char*) 0;
        const int k_pitch_px_row   = (int)(k_nb1 >> 3);
        const int k_pitch_px_head  = (int)(k_nb2 >> 3);
        const int k_pitch_px_batch = (int)(k_nb3 >> 3);
#else
        const global char* k_tile_base = use_kv_pad ? (const global char*) k_pad_void : k_base;
#endif
        const global char* v_tile_base = use_kv_pad ? (const global char*) v_pad_void : v_base;

        for (int i = tid; i < BLOCK_N * DK_VEC; i += WG_SIZE) {
            const int row = i / DK_VEC;
            const int col = i % DK_VEC;
            const int k_row_idx = k_tile_start + row;
            if (use_kv_pad || k_row_idx < n_kv) {
#ifdef FA_K_IMG
                if (use_kv_pad) {
                    const ulong k_row_offset = batch_idx * k_tile_nb3 + head_kv_idx * k_tile_nb2 + k_row_idx * k_nb1;
                    l_k[row][col] = ((__global KV_DATA_TYPE4*)(k_tile_base + k_row_offset))[col];
                } else {
                    const int k_row_px = batch_idx * k_pitch_px_batch + head_kv_idx * k_pitch_px_head + k_row_idx * k_pitch_px_row;
                    l_k[row][col] = read_imageh(k_img, k_row_px + col);
                }
#else
                const ulong k_row_offset = batch_idx * k_tile_nb3 + head_kv_idx * k_tile_nb2 + k_row_idx * k_nb1;
                l_k[row][col] = ((__global KV_DATA_TYPE4*)(k_tile_base + k_row_offset))[col];
#endif
            } else {
                l_k[row][col] = (KV_DATA_TYPE4)(0.0h);
            }
        }
        for (int i = tid; i < BLOCK_N * DV_VEC; i += WG_SIZE) {
            const int row = i / DV_VEC;
            const int col = i % DV_VEC;
            const int v_row_idx = k_tile_start + row;
            if (use_kv_pad || v_row_idx < n_kv) {
                const ulong v_row_offset = batch_idx * v_tile_nb3 + head_kv_idx * v_tile_nb2 + v_row_idx * v_nb1;
                l_v[row][col] = ((__global KV_DATA_TYPE4*)(v_tile_base + v_row_offset))[col];
            } else {
                l_v[row][col] = (KV_DATA_TYPE4)(0.0h);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

#if N_SPLIT > 1 && defined(HAS_SUBGROUP_SHUFFLE)
        {
            const int dv_off = split_idx * SPLIT_DV_VEC;
            for (int j = 0; j < BLOCK_N; j += 2) {
                const int k_row0 = k_start + j;
                const int k_row1 = k_start + j + 1;

                ACC_TYPE partial0 = 0.0f;
                ACC_TYPE partial1 = 0.0f;
                FA_UNROLL
                for (int k = 0; k < SPLIT_DK_VEC; k++) {
                    const ACC_TYPE4 qk = q_priv[k];
                    ACC_TYPE4 dot0 = qk * CONVERT_KV_ACC4(l_k[j  ][dk_off + k]);
                    ACC_TYPE4 dot1 = qk * CONVERT_KV_ACC4(l_k[j+1][dk_off + k]);
                    partial0 += dot0.s0 + dot0.s1 + dot0.s2 + dot0.s3;
                    partial1 += dot1.s0 + dot1.s1 + dot1.s2 + dot1.s3;
                }

                FA_UNROLL
                for (int step = 1; step < N_SPLIT; step <<= 1) {
                    partial0 += sub_group_shuffle_xor(partial0, step);
                    partial1 += sub_group_shuffle_xor(partial1, step);
                }

                ACC_TYPE score0 = partial0 * scale;
                ACC_TYPE score1 = partial1 * scale;

                if (!query_valid) { score0 = FA_M_INIT; score1 = FA_M_INIT; }
                if (is_causal) {
                    if (k_row0 > (n_kv - n_q + my_query_row)) score0 = FA_M_INIT;
                    if (k_row1 > (n_kv - n_q + my_query_row)) score1 = FA_M_INIT;
                }
                if (k_row0 >= n_kv) score0 = FA_M_INIT;
                if (k_row1 >= n_kv) score1 = FA_M_INIT;

                if (query_valid && mask_base != NULL && blk_cur != 2) {
                    if (use_kv_pad && mask_pad_base != NULL) {
                        const global MASK_DATA_TYPE* mask_ptr =
                            (const global MASK_DATA_TYPE*)(mask_pad_base + my_query_row * mask_pad_nb1);
                        score0 += slope * (ACC_TYPE)mask_ptr[j];
                        score1 += slope * (ACC_TYPE)mask_ptr[j + 1];
                    } else {
                        const global MASK_DATA_TYPE* mask_ptr =
                            (const global MASK_DATA_TYPE*)(mask_base + my_query_row * mask_nb1);
                        if (k_row0 < n_kv) score0 += slope * (ACC_TYPE)mask_ptr[k_row0];
                        if (k_row1 < n_kv) score1 += slope * (ACC_TYPE)mask_ptr[k_row1];
                    }
                }

                if (logit_softcap > 0.0f) {
                    score0 = logit_softcap * tanh(score0 / logit_softcap);
                    score1 = logit_softcap * tanh(score1 / logit_softcap);
                }

                const ACC_TYPE m_new = max(m_i, max(score0, score1));
                // Whole tile masked (m_new == FA_M_INIT): force the exp() args
                // far negative so the tile contributes 0, not exp(0)=1.
                const ACC_TYPE m_exp = (m_new == FA_M_INIT) ? 0.0f : m_new;
                const ACC_TYPE sp    = native_exp(m_i - m_exp);
                const ACC_TYPE p0    = native_exp(score0 - m_exp);
                const ACC_TYPE p1    = native_exp(score1 - m_exp);

                FA_UNROLL
                for (int i = 0; i < SPLIT_DV_VEC; ++i) {
                    o_acc[i] = o_acc[i] * sp
                             + p0 * CONVERT_KV_ACC4(l_v[j  ][dv_off + i])
                             + p1 * CONVERT_KV_ACC4(l_v[j+1][dv_off + i]);
                }
                l_i = l_i * sp + p0 + p1;
                m_i = m_new;
            }
        }
#elif N_SPLIT > 1
        // N_SPLIT>1 fallback (no shuffle): 3-phase local-memory reduction.
        // Phase 1 — partial dots for all BLOCK_N tokens.
        for (int j = 0; j < BLOCK_N; ++j) {
            ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
            FA_UNROLL
            for (int k = 0; k < SPLIT_DK_VEC; k++) {
                dot_acc = mad(q_priv[k], CONVERT_KV_ACC4(l_k[j][dk_off + k]), dot_acc);
            }
            local_partial[j][tid] =
                dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);  // 1 barrier: partial dots visible

        // Phase 2 — split_idx==0 reduces partial sums and computes block softmax.
        if (split_idx == 0) {
            if (query_valid) {
                ACC_TYPE m_new = m_i;
                for (int j = 0; j < BLOCK_N; ++j) {
                    const int k_row = k_start + j;
                    ACC_TYPE score = 0.0f;
                    FA_UNROLL
                    for (int s = 0; s < N_SPLIT; s++) {
                        score += local_partial[j][q_lane * N_SPLIT + s];
                    }
                    score *= scale;

                    if (is_causal && k_row > (n_kv - n_q + my_query_row)) score = FA_M_INIT;
                    if (k_row >= n_kv) score = FA_M_INIT;

                    if (mask_base != NULL && blk_cur != 2) {
                        if (use_kv_pad && mask_pad_base != NULL) {
                            const global MASK_DATA_TYPE* mask_ptr =
                                (const global MASK_DATA_TYPE*)(mask_pad_base + my_query_row * mask_pad_nb1);
                            score += slope * (ACC_TYPE)mask_ptr[j];
                        } else {
                            const global MASK_DATA_TYPE* mask_ptr =
                                (const global MASK_DATA_TYPE*)(mask_base + my_query_row * mask_nb1);
                            if (k_row < n_kv) score += slope * (ACC_TYPE)mask_ptr[k_row];
                        }
                    }

                    if (logit_softcap > 0.0f) {
                        score = logit_softcap * tanh(score / logit_softcap);
                    }

                    m_new = max(m_new, score);
                    local_p[q_lane][j] = score;
                }

                const ACC_TYPE m_exp = (m_new == FA_M_INIT) ? 0.0f : m_new;
                const ACC_TYPE sp = native_exp(m_i - m_exp);
                ACC_TYPE l_new = l_i * sp;
                for (int j = 0; j < BLOCK_N; ++j) {
                    const ACC_TYPE p = native_exp(local_p[q_lane][j] - m_exp);
                    local_p[q_lane][j] = p;
                    l_new += p;
                }
                local_softmax_scale[q_lane] = sp;
                l_i = l_new;
                m_i = m_new;
            } else {
                local_softmax_scale[q_lane] = 1.0f;
                for (int j = 0; j < BLOCK_N; ++j) local_p[q_lane][j] = 0.0f;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Phase 3 — V accumulate using broadcast probabilities.
        {
            const ACC_TYPE sp_block = local_softmax_scale[q_lane];
            const int dv_off = split_idx * SPLIT_DV_VEC;
            FA_UNROLL
            for (int i = 0; i < SPLIT_DV_VEC; ++i) {
                o_acc[i] *= sp_block;
            }
            for (int j = 0; j < BLOCK_N; ++j) {
                const ACC_TYPE p = local_p[q_lane][j];
                FA_UNROLL
                for (int i = 0; i < SPLIT_DV_VEC; ++i) {
                    o_acc[i] = mad(p, CONVERT_KV_ACC4(l_v[j][dv_off + i]), o_acc[i]);
                }
            }
        }
#else
        // N_SPLIT==1: j+=4 unroll. Requires BLOCK_N % 4 == 0.
        if (query_valid) {
            for (int j = 0; j < BLOCK_N; j += 4) {
                const int k_row0 = k_start + j;
                const int k_row1 = k_start + j + 1;
                const int k_row2 = k_start + j + 2;
                const int k_row3 = k_start + j + 3;

                ACC_TYPE4 dot_acc0 = (ACC_TYPE4)(0.0f);
                ACC_TYPE4 dot_acc1 = (ACC_TYPE4)(0.0f);
                ACC_TYPE4 dot_acc2 = (ACC_TYPE4)(0.0f);
                ACC_TYPE4 dot_acc3 = (ACC_TYPE4)(0.0f);
                FA_UNROLL
                for (int k = 0; k < DK_VEC; k++) {
                    const ACC_TYPE4 qk = q_priv[k];
                    dot_acc0 = mad(qk, CONVERT_KV_ACC4(l_k[j][k]),   dot_acc0);
                    dot_acc1 = mad(qk, CONVERT_KV_ACC4(l_k[j+1][k]), dot_acc1);
                    dot_acc2 = mad(qk, CONVERT_KV_ACC4(l_k[j+2][k]), dot_acc2);
                    dot_acc3 = mad(qk, CONVERT_KV_ACC4(l_k[j+3][k]), dot_acc3);
                }
                ACC_TYPE s0 = (dot_acc0.s0 + dot_acc0.s1 + dot_acc0.s2 + dot_acc0.s3) * scale;
                ACC_TYPE s1 = (dot_acc1.s0 + dot_acc1.s1 + dot_acc1.s2 + dot_acc1.s3) * scale;
                ACC_TYPE s2 = (dot_acc2.s0 + dot_acc2.s1 + dot_acc2.s2 + dot_acc2.s3) * scale;
                ACC_TYPE s3 = (dot_acc3.s0 + dot_acc3.s1 + dot_acc3.s2 + dot_acc3.s3) * scale;

                if (is_causal) {
                    const int causal_limit = n_kv - n_q + my_query_row;
                    if (k_row0 > causal_limit) s0 = FA_M_INIT;
                    if (k_row1 > causal_limit) s1 = FA_M_INIT;
                    if (k_row2 > causal_limit) s2 = FA_M_INIT;
                    if (k_row3 > causal_limit) s3 = FA_M_INIT;
                }
                if (k_row0 >= n_kv) s0 = FA_M_INIT;
                if (k_row1 >= n_kv) s1 = FA_M_INIT;
                if (k_row2 >= n_kv) s2 = FA_M_INIT;
                if (k_row3 >= n_kv) s3 = FA_M_INIT;

                if (mask_base != NULL && blk_cur != 2) {
                    if (use_kv_pad && mask_pad_base != NULL) {
                        const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_pad_base + my_query_row * mask_pad_nb1);
                        s0 += slope * (ACC_TYPE)mask_ptr[j];
                        s1 += slope * (ACC_TYPE)mask_ptr[j + 1];
                        s2 += slope * (ACC_TYPE)mask_ptr[j + 2];
                        s3 += slope * (ACC_TYPE)mask_ptr[j + 3];
                    } else {
                        const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base + my_query_row * mask_nb1);
                        if (k_row0 < n_kv) s0 += slope * (ACC_TYPE)mask_ptr[k_row0];
                        if (k_row1 < n_kv) s1 += slope * (ACC_TYPE)mask_ptr[k_row1];
                        if (k_row2 < n_kv) s2 += slope * (ACC_TYPE)mask_ptr[k_row2];
                        if (k_row3 < n_kv) s3 += slope * (ACC_TYPE)mask_ptr[k_row3];
                    }
                }

                if (logit_softcap > 0.0f) {
                    s0 = logit_softcap * tanh(s0 / logit_softcap);
                    s1 = logit_softcap * tanh(s1 / logit_softcap);
                    s2 = logit_softcap * tanh(s2 / logit_softcap);
                    s3 = logit_softcap * tanh(s3 / logit_softcap);
                }

                const ACC_TYPE m_new      = max(m_i, max(max(s0, s1), max(s2, s3)));
                // Whole tile masked (m_new == FA_M_INIT): force the exp() args
                // far negative so the tile contributes 0, not exp(0)=1.
                const ACC_TYPE m_exp      = (m_new == FA_M_INIT) ? 0.0f : m_new;
                const ACC_TYPE scale_prev = native_exp(m_i - m_exp);
                const ACC_TYPE p0         = native_exp(s0 - m_exp);
                const ACC_TYPE p1         = native_exp(s1 - m_exp);
                const ACC_TYPE p2         = native_exp(s2 - m_exp);
                const ACC_TYPE p3         = native_exp(s3 - m_exp);

                FA_UNROLL
                for (int i = 0; i < DV_VEC; ++i) {
                    o_acc[i] = mad(p3, CONVERT_KV_ACC4(l_v[j+3][i]),
                               mad(p2, CONVERT_KV_ACC4(l_v[j+2][i]),
                               mad(p1, CONVERT_KV_ACC4(l_v[j+1][i]),
                               mad(p0, CONVERT_KV_ACC4(l_v[j][i]),
                               o_acc[i] * scale_prev))));
                }
                l_i = l_i * scale_prev + p0 + p1 + p2 + p3;
                m_i = m_new;
            }
        }
#endif
        // End of tile: every thread must finish reading l_k/l_v before the
        // next iteration's load overwrites them (WAR hazard on local memory).
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write output.
#if N_SPLIT > 1 && defined(HAS_SUBGROUP_SHUFFLE)
    if (query_valid) {
        ACC_TYPE sinks_sp = 1.0f;
        if (sinks_void != NULL) {
            const global ACC_TYPE* sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
            const ACC_TYPE m_sink  = sinks_ptr[head_idx];
            const ACC_TYPE m_final = max(m_i, m_sink);
            sinks_sp = exp(m_i - m_final);
            l_i = l_i * sinks_sp + exp(m_sink - m_final);
            m_i = m_final;
        }
        const ACC_TYPE l_inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        const int dv_off = split_idx * SPLIT_DV_VEC;
        const ulong o_row_offset = batch_idx * o_nb3 + my_query_row * o_nb2 + head_idx * o_nb1;
        global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
        if (l_inv > 0.0f) {
            FA_UNROLL
            for (int i = 0; i < SPLIT_DV_VEC; ++i) {
                o_row[dv_off + i] = CONVERT_O_DATA4(o_acc[i] * sinks_sp * l_inv);
            }
        } else {
            FA_UNROLL
            for (int i = 0; i < SPLIT_DV_VEC; ++i) {
                o_row[dv_off + i] = (O_DATA_TYPE4)(0.0f);
            }
        }
    }
#elif N_SPLIT > 1
    if (split_idx == 0) {
        ACC_TYPE sinks_sp = 1.0f;
        if (query_valid && sinks_void != NULL) {
            const global ACC_TYPE* sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
            const ACC_TYPE m_sink = sinks_ptr[head_idx];
            const ACC_TYPE m_final = max(m_i, m_sink);
            sinks_sp = exp(m_i - m_final);
            l_i = l_i * sinks_sp + exp(m_sink - m_final);
            m_i = m_final;
        }
        local_softmax_scale[q_lane] = sinks_sp;
        local_l_inv[q_lane] = (query_valid && l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (query_valid) {
        const ACC_TYPE sinks_sp = local_softmax_scale[q_lane];
        const ACC_TYPE l_inv    = local_l_inv[q_lane];
        const int dv_off = split_idx * SPLIT_DV_VEC;
        const ulong o_row_offset = batch_idx * o_nb3 + my_query_row * o_nb2 + head_idx * o_nb1;
        global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
        if (l_inv > 0.0f) {
            FA_UNROLL
            for (int i = 0; i < SPLIT_DV_VEC; ++i) {
                o_row[dv_off + i] = CONVERT_O_DATA4(o_acc[i] * sinks_sp * l_inv);
            }
        } else {
            FA_UNROLL
            for (int i = 0; i < SPLIT_DV_VEC; ++i) {
                o_row[dv_off + i] = (O_DATA_TYPE4)(0.0f);
            }
        }
    }
#else
    if (query_valid) {
        if (sinks_void != NULL) {
            const global ACC_TYPE* sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
            const ACC_TYPE m_sink = sinks_ptr[head_idx];
            const ACC_TYPE m_final = max(m_i, m_sink);

            const ACC_TYPE scale_o = exp(m_i - m_final);
            FA_UNROLL
            for (int i = 0; i < DV_VEC; ++i) {
                o_acc[i] *= scale_o;
            }

            l_i = l_i * exp(m_i - m_final) + exp(m_sink - m_final);
        }

        const ulong o_row_offset = batch_idx * o_nb3 + my_query_row * o_nb2 + head_idx * o_nb1;
        global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
        if (l_i > 0.0f) {
            const ACC_TYPE l_inv = 1.0f / l_i;
            FA_UNROLL
            for (int i = 0; i < DV_VEC; ++i) {
                o_row[i] = CONVERT_O_DATA4(o_acc[i] * l_inv);
            }
        } else {
            FA_UNROLL
            for (int i = 0; i < DV_VEC; ++i) {
                o_row[i] = (O_DATA_TYPE4)(0.0f);
            }
        }
    }
#endif
}
#endif  // !FA_DECODE_ONLY

// allow bypassing decode kernels to avoid compiler crash for DK=512 on Adreno GPUs
#ifndef FA_PREFILL_ONLY
#ifndef FA_MQ_ONLY  // q1 excluded from the MQ-only (g8) program
REQD_FA_SG
__kernel void flash_attn_f32_f16_q1(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset
) {
    const int tid = get_local_id(0);
    const int head_batch_idx = get_global_id(1);

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;

    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void + q_offset;
#ifndef FA_K_IMG
    const global char* k_base = (const global char*)k_void + k_offset;
#endif
    const global char* v_base = (const global char*)v_void + v_offset;
    global char* o_base = (global char*)o_void + o_offset;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx = head_idx % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    // Q is uniform across WG threads (n_q=1). Share via local memory to
    // avoid per-thread q_priv[DK_VEC] dynamic-indexed private array that
    // spills to DDR on Adreno.
    __local ACC_TYPE4 q_shared[DK_VEC];
    const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2;
    const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
    for (int i = tid; i < DK_VEC; i += Q1_WG_SIZE) {
        q_shared[i] = CONVERT_Q_ACC4(q_ptr[i]);
    }
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    const global ACC_TYPE* sinks_ptr = NULL;
    if (sinks_void != NULL) {
        sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
    }

    ACC_TYPE m_i = (sinks_ptr != NULL) ? sinks_ptr[head_idx] : FA_M_INIT;
    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const global KV_DATA_TYPE4* k_ptr = (const global KV_DATA_TYPE4*)(k_base + k_row_offset);
        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
        FA_UNROLL
        for (int k = 0; k < DK_VEC; k++) {
            dot_acc = mad(q_shared[k], CONVERT_KV_ACC4(k_ptr[k]), dot_acc);
        }
        ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
        if (mask_base != NULL) {
            const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base);
            score += slope * (ACC_TYPE)mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }
        m_i = max(m_i, score);
    }

    const ACC_TYPE m_final = sub_group_reduce_max(m_i);

    ACC_TYPE4 o_acc[DV_VEC];
    FA_UNROLL
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);
    ACC_TYPE l_i = 0.0f;

    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
        const global KV_DATA_TYPE4* k_ptr = (const global KV_DATA_TYPE4*)(k_base + k_row_offset);
        const global KV_DATA_TYPE4* v_ptr = (const global KV_DATA_TYPE4*)(v_base + v_row_offset);
        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
        FA_UNROLL
        for (int k = 0; k < DK_VEC; k++) {
            dot_acc = mad(q_shared[k], CONVERT_KV_ACC4(k_ptr[k]), dot_acc);
        }
        ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
        if (mask_base != NULL) {
            const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base);
            score += slope * (ACC_TYPE)mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }
        const ACC_TYPE p = exp(score - m_final);
        l_i += p;
        FA_UNROLL
        for (int i = 0; i < DV_VEC; i++) {
            o_acc[i] = mad(p, CONVERT_KV_ACC4(v_ptr[i]), o_acc[i]);
        }
    }

    __local ACC_TYPE4 local_o_comp[Q1_WG_SIZE];
    const ACC_TYPE l_red = sub_group_reduce_add(l_i);

    const ulong o_row_offset = batch_idx * o_nb3 + head_idx * o_nb1;
    global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
    ACC_TYPE l_final = l_red;

    if (sinks_ptr != NULL) {
        l_final += exp(sinks_ptr[head_idx] - m_final);
    }

    if (l_final > 0.0f) {
        const ACC_TYPE l_inv = 1.0f / l_final;
        for (int i = 0; i < DV_VEC; i++) {
            local_o_comp[tid] = o_acc[i];
            sub_group_barrier(CLK_LOCAL_MEM_FENCE);
            FA_UNROLL
            for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
                if (tid < s) local_o_comp[tid] += local_o_comp[tid + s];
                sub_group_barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (tid == 0) {
                o_row[i] = CONVERT_O_DATA4(local_o_comp[0] * l_inv);
            }
        }
    } else if (tid == 0) {
        FA_UNROLL
        for (int i = 0; i < DV_VEC; ++i) o_row[i] = (O_DATA_TYPE4)(0.0f);
    }
}

#endif  // !FA_MQ_ONLY (q1)
// decode variant for large DV (e.g. Gemma-4 DK=DV=512 global layers).
#define VEC_NSG          4
#define VEC_WG_SIZE      (Q1_WG_SIZE * VEC_NSG)
#define Q1V_DV_PER_THREAD ((DV_VEC + Q1_WG_SIZE - 1) / Q1_WG_SIZE)

// allow bypassing the kernel to avoid compiler crash for DK=512 on Adreno GPUs
#if !defined(FA_DECODE_MINIMAL) && !defined(FA_MQ_ONLY)
REQD_SUBGROUP_SIZE_64
__kernel void flash_attn_f32_f16_q1_vec(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset
) {
    const int tid             = get_local_id(0);
    const int sgid            = tid / Q1_WG_SIZE;   // subgroup index (0..VEC_NSG-1)
    const int tid_sg          = tid % Q1_WG_SIZE;   // lane within subgroup
    const int head_batch_idx  = get_global_id(1);

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx  = head_batch_idx % n_head;

    const int gqa_ratio   = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char * q_base = (const global char *) q_void + q_offset;
    const global char * k_base = (const global char *) k_void + k_offset;
    const global char * v_base = (const global char *) v_void + v_offset;
    global       char * o_base = (global       char *) o_void + o_offset;

    const global char * mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx  = head_idx  % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char *) mask_void + mask_offset +
                    mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    // Q is uniform across the WG — stage in __local once. All WG threads load.
    __local ACC_TYPE4 q_shared[DK_VEC];
    {
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2;
        const global Q_DATA_TYPE4 * q_ptr = (const global Q_DATA_TYPE4 *) (q_base + q_row_offset);
        for (int i = tid; i < DK_VEC; i += VEC_WG_SIZE) {
            q_shared[i] = CONVERT_Q_ACC4(q_ptr[i]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    const global ACC_TYPE * sinks_ptr = NULL;
    if (sinks_void != NULL) {
        sinks_ptr = (const global ACC_TYPE *) ((const global char *) sinks_void + sinks_offset);
    }

    // per-thread DV slice within its subgroup
    // DV=512 -> 2x float4 = 32 bytes; DV=256 -> 1x float4 - no spill
    ACC_TYPE4 o_acc[Q1V_DV_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < Q1V_DV_PER_THREAD; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);

    // each subgroup independently runs the FA-2 online softmax over its slice of n_kv.
    // sinks are not folded into per-subgroup m_i — they're added once in
    // the cross-subgroup merge to avoid double-counting.
    ACC_TYPE m_i = FA_M_INIT;
    ACC_TYPE l_i = 0.0f;

    const int kv_per_sg = (n_kv + VEC_NSG - 1) / VEC_NSG;
    const int kv_start  = sgid * kv_per_sg;
    const int kv_end    = min(n_kv, kv_start + kv_per_sg);

    for (int k_idx = kv_start; k_idx < kv_end; ++k_idx) {
        const ulong k_row_off = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const ulong v_row_off = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
        const global KV_DATA_TYPE4 * k_ptr = (const global KV_DATA_TYPE4 *) (k_base + k_row_off);
        const global KV_DATA_TYPE4 * v_ptr = (const global KV_DATA_TYPE4 *) (v_base + v_row_off);

        // Q*K^T: each thread accumulates its DK slice; subgroup-reduce the partial.
        ACC_TYPE4 dot4 = (ACC_TYPE4)(0.0f);
        for (int k = tid_sg; k < DK_VEC; k += Q1_WG_SIZE) {
            dot4 = mad(q_shared[k], CONVERT_KV_ACC4(k_ptr[k]), dot4);
        }
        ACC_TYPE dot_partial = dot4.s0 + dot4.s1 + dot4.s2 + dot4.s3;
        ACC_TYPE score = sub_group_reduce_add(dot_partial) * scale;

        if (mask_base != NULL) {
            const global MASK_DATA_TYPE * mask_ptr = (const global MASK_DATA_TYPE *) mask_base;
            score += slope * (ACC_TYPE) mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }

        // FA-2 online update. All threads in the subgroup see the same score,
        // so m_i and l_i evolve identically across lanes within the subgroup.
        const ACC_TYPE m_new      = max(m_i, score);
        const ACC_TYPE scale_prev = native_exp(m_i - m_new);
        const ACC_TYPE p          = native_exp(score - m_new);

        int idx = 0;
        for (int dv_idx = tid_sg; dv_idx < DV_VEC; dv_idx += Q1_WG_SIZE, ++idx) {
            o_acc[idx] = mad(p, CONVERT_KV_ACC4(v_ptr[dv_idx]), o_acc[idx] * scale_prev);
        }
        l_i = l_i * scale_prev + p;
        m_i = m_new;
    }

    // Cross-subgroup merge via __local. Each subgroup publishes (m_i, l_i)
    // and its o_acc slice; subgroup 0 then folds them into the final norm
    // and writes the row.
    __local ACC_TYPE  sg_m[VEC_NSG];
    __local ACC_TYPE  sg_l[VEC_NSG];
    __local ACC_TYPE4 sg_o[VEC_NSG][DV_VEC];

    if (tid_sg == 0) {
        sg_m[sgid] = m_i;
        sg_l[sgid] = l_i;
    }
    {
        int idx = 0;
        for (int dv_idx = tid_sg; dv_idx < DV_VEC; dv_idx += Q1_WG_SIZE, ++idx) {
            sg_o[sgid][dv_idx] = o_acc[idx];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (sgid == 0) {
        // m_final = max over all subgroups' m_i, plus the sink (if any).
        ACC_TYPE m_final = sg_m[0];
        #pragma unroll
        for (int s = 1; s < VEC_NSG; ++s) {
            m_final = max(m_final, sg_m[s]);
        }
        if (sinks_ptr != NULL) {
            m_final = max(m_final, sinks_ptr[head_idx]);
        }

        ACC_TYPE l_final = 0.0f;
        #pragma unroll
        for (int s = 0; s < VEC_NSG; ++s) {
            l_final += sg_l[s] * native_exp(sg_m[s] - m_final);
        }
        if (sinks_ptr != NULL) {
            l_final += native_exp(sinks_ptr[head_idx] - m_final);
        }
        const ACC_TYPE l_inv = (l_final > 0.0f) ? (1.0f / l_final) : 0.0f;

        const ulong o_row_offset = batch_idx * o_nb3 + head_idx * o_nb1;
        global O_DATA_TYPE4 * o_row = (global O_DATA_TYPE4 *) (o_base + o_row_offset);

        // Each thread in subgroup 0 writes its DV slice, folding all subgroups'
        // contributions with the rescale factor.
        int idx = 0;
        for (int dv_idx = tid_sg; dv_idx < DV_VEC; dv_idx += Q1_WG_SIZE, ++idx) {
            ACC_TYPE4 o_merged = (ACC_TYPE4)(0.0f);
            #pragma unroll
            for (int s = 0; s < VEC_NSG; ++s) {
                const ACC_TYPE alpha = native_exp(sg_m[s] - m_final);
                o_merged = mad((ACC_TYPE4)(alpha), sg_o[s][dv_idx], o_merged);
            }
            o_row[dv_idx] = CONVERT_O_DATA4(o_merged * l_inv);
        }
    }
}

#endif  // !FA_DECODE_MINIMAL

#ifndef FA_DECODE_ONLY

// flash_attn_f32_f16_q1_local_tile
// one WG per (q_idx, q_head)

#define LT_KC 32
#define LT_WG 128

#ifndef FA_MQ_ONLY  // q1_local_tile excluded from the MQ-only (g8) program
REQD_SUBGROUP_SIZE_64
__kernel void flash_attn_f32_f16_q1_local_tile(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void * mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void * sinks_void,
    const ulong sinks_offset
) {
    const int q_idx     = get_global_id(0) / LT_WG;
    const int head_idx  = get_global_id(1);
    const int batch_idx = get_global_id(2);
    const int tid       = get_local_id(0);

    const int gqa_ratio   = n_head_kv > 0 ? (n_head / n_head_kv) : 1;
    const int head_kv_idx = head_idx / gqa_ratio;

    const float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    __local half  k_tile[LT_KC * DK];   // 32*128*2 = 8 KB at DK=128
    __local half  v_tile[LT_KC * DV];   // 8 KB
    __local float red[LT_WG];           // 512 B reduction scratch
    __local float score_shared;         // broadcast score (each K-step)

    // Each thread owns one float of Q at index `tid` (assumes LT_WG == DK).
    const global char * q_row_base = (const global char *) q_void + q_offset +
                                     batch_idx * q_nb3 + head_idx * q_nb2 + (ulong) q_idx * q_nb1;
    float q_val = ((const global float *) q_row_base)[tid];

    const global char * mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx  = head_idx  % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char *) mask_void + mask_offset +
                    mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2 +
                    (ulong) q_idx * mask_nb1;
    }

    float o_val = 0.0f;
    float m_i = FA_M_INIT;
    float l_i = 0.0f;

    for (int kb = 0; kb < n_kv; kb += LT_KC) {
        const int tile_len = min(LT_KC, n_kv - kb);

        // Stage K and V tiles into __local.
        for (int i = tid; i < tile_len * DK; i += LT_WG) {
            const int j = i / DK;
            const int d = i % DK;
            const int kv_idx = kb + j;
            const global char * k_row = (const global char *) k_void + k_offset +
                                        batch_idx * k_nb3 + head_kv_idx * k_nb2 +
                                        (ulong) kv_idx * k_nb1;
            const global char * v_row = (const global char *) v_void + v_offset +
                                        batch_idx * v_nb3 + head_kv_idx * v_nb2 +
                                        (ulong) kv_idx * v_nb1;
            k_tile[j * DK + d] = ((const global half *) k_row)[d];
            v_tile[j * DV + d] = ((const global half *) v_row)[d];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int j = 0; j < tile_len; ++j) {
            const int kv_idx = kb + j;

            // Q·K dot via __local tree-reduce.
            red[tid] = q_val * convert_float(k_tile[j * DK + tid]);
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int stride = LT_WG >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    red[tid] += red[tid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (tid == 0) {
                float s = red[0] * scale;
                if (mask_base != NULL) {
                    const global MASK_DATA_TYPE * mask_ptr = (const global MASK_DATA_TYPE *) mask_base;
                    s += slope * (float) mask_ptr[kv_idx];
                }
                if (logit_softcap > 0.0f) {
                    s = logit_softcap * tanh(s / logit_softcap);
                }
                score_shared = s;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            const float s     = score_shared;
            const float m_new = fmax(m_i, s);
            const float alpha = native_exp(m_i - m_new);
            const float beta  = native_exp(s   - m_new);

            o_val = o_val * alpha + beta * convert_float(v_tile[j * DV + tid]);
            l_i   = l_i   * alpha + beta;
            m_i   = m_new;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Fold attention sinks into the running (m, l, o), if present.
    if (sinks_void != NULL) {
        const global float * sinks_ptr =
            (const global float *) ((const global char *) sinks_void + sinks_offset);
        const float m_sink = sinks_ptr[head_idx];
        const float m_new  = fmax(m_i, m_sink);
        const float alpha  = native_exp(m_i    - m_new);
        const float beta   = native_exp(m_sink - m_new);
        o_val = o_val * alpha;
        l_i   = l_i * alpha + beta;
        m_i   = m_new;
    }

    const float l_inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    global float * o_row = (global float *) ((global char *) o_void + o_offset +
                                              batch_idx * o_nb3 + head_idx * o_nb1 +
                                              (ulong) q_idx * o_nb2);
    o_row[tid] = o_val * l_inv;
}

// flash_attn_f32_f16_q1_local_mq_split

#define LMQ_WG  64
#define LMQ_KC  32
#define LMQ_DPL 2   // DK / LMQ_WG at DK=128

#endif  // !FA_MQ_ONLY (q1_local_tile)
#ifndef MQ_GQA
#define MQ_GQA 4
#endif

#ifndef FA_PARTIAL_FLOATS
#define FA_PARTIAL_FLOATS (2 + DV)
#endif

#ifndef FA_MQ_ONLY  // q1_local_mq_split excluded from the MQ-only (g8) program
REQD_SUBGROUP_SIZE_64
__kernel void flash_attn_f32_f16_q1_local_mq_split(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void * mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    global float * partial_void,
    const int n_splits,
    const int kv_per_split
) {
    const int tid              = get_local_id(0);  // 0..LMQ_WG-1
    const int kvhead_batch_idx = get_global_id(1);
    const int split_q_idx      = get_global_id(2);
    const int split_idx        = split_q_idx % n_splits;
    const int q_idx            = split_q_idx / n_splits;

    const int batch_idx   = kvhead_batch_idx / n_head_kv;
    const int head_kv_idx = kvhead_batch_idx % n_head_kv;

    const int kv_start = split_idx * kv_per_split;
    const int kv_end   = min(kv_start + kv_per_split, n_kv);

    const ulong record_stride = (ulong) FA_PARTIAL_FLOATS;

    if (kv_start >= kv_end) {
        // Empty split — write sentinel for each Q-head so merge treats it as 0.
        if (tid == 0) {
            #pragma unroll
            for (int h = 0; h < MQ_GQA; ++h) {
                const int head_idx = head_kv_idx * MQ_GQA + h;
                const ulong rec_idx = ((((ulong) batch_idx * n_head + head_idx) * n_q + q_idx)
                                       * n_splits + split_idx);
                global float * rec = partial_void + rec_idx * record_stride;
                rec[0] = FA_M_INIT;
                rec[1] = 0.0f;
            }
        }
        return;
    }

    const global char * q_base = (const global char *) q_void + q_offset;
    const global char * k_base = (const global char *) k_void + k_offset;
    const global char * v_base = (const global char *) v_void + v_offset;

    // Stage MQ_GQA Q rows in __local (MQ_GQA × DK floats).
    __local float q_shared[MQ_GQA * DK];
    for (int i = tid; i < MQ_GQA * DK; i += LMQ_WG) {
        const int h        = i / DK;
        const int d        = i % DK;
        const int head_idx = head_kv_idx * MQ_GQA + h;
        const ulong q_row_off = batch_idx * q_nb3 + head_idx * q_nb2 + (ulong) q_idx * q_nb1;
        const global float * q_ptr = (const global float *) (q_base + q_row_off);
        q_shared[h * DK + d] = q_ptr[d];
    }

    // K/V tile staging buffers (16 KB combined at DK=DV=128 KC=32).
    __local half k_tile[LMQ_KC * DK];
    __local half v_tile[LMQ_KC * DV];

    // Per-h state held in private registers.
    float o_acc[MQ_GQA][LMQ_DPL];
    float m_i[MQ_GQA];
    float l_i[MQ_GQA];
    float slope[MQ_GQA];
    #pragma unroll
    for (int h = 0; h < MQ_GQA; ++h) {
        m_i[h] = FA_M_INIT;
        l_i[h] = 0.0f;
        slope[h] = get_alibi_slope(max_bias, head_kv_idx * MQ_GQA + h, n_head_log2, m0, m1);
        #pragma unroll
        for (int p = 0; p < LMQ_DPL; ++p) o_acc[h][p] = 0.0f;
    }

    // Per-h mask pointers.
    const global char * mask_base[MQ_GQA];
    if (mask_void != NULL) {
        const int mask_batch_idx = batch_idx % mask_ne3;
        const global char * mask_base_b = (const global char *) mask_void + mask_offset +
                                          mask_batch_idx * mask_nb3 +
                                          (ulong) q_idx * mask_nb1;
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
            const int head_idx      = head_kv_idx * MQ_GQA + h;
            const int mask_head_idx = head_idx % mask_ne2;
            mask_base[h] = mask_base_b + mask_head_idx * mask_nb2;
        }
    } else {
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) mask_base[h] = NULL;
    }

    barrier(CLK_LOCAL_MEM_FENCE);  // Ensure Q staged before first dot.

    for (int kb = kv_start; kb < kv_end; kb += LMQ_KC) {
        const int tile_len = min((int) LMQ_KC, kv_end - kb);

        // Cooperative load K + V tile.
        for (int i = tid; i < tile_len * DK; i += LMQ_WG) {
            const int j = i / DK;
            const int d = i % DK;
            const int kv_idx = kb + j;
            const global char * k_row = k_base + batch_idx * k_nb3 + head_kv_idx * k_nb2 + (ulong) kv_idx * k_nb1;
            const global char * v_row = v_base + batch_idx * v_nb3 + head_kv_idx * v_nb2 + (ulong) kv_idx * v_nb1;
            k_tile[j * DK + d] = ((const global half *) k_row)[d];
            v_tile[j * DV + d] = ((const global half *) v_row)[d];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Process each cache row in the tile.
        for (int j = 0; j < tile_len; ++j) {
            const int kv_idx = kb + j;

            // Dot product per h: lane owns LMQ_DPL D-elements at (tid*LMQ_DPL..).
            float score[MQ_GQA];
            #pragma unroll
            for (int h = 0; h < MQ_GQA; ++h) {
                float contrib = 0.0f;
                #pragma unroll
                for (int p = 0; p < LMQ_DPL; ++p) {
                    const int d = tid * LMQ_DPL + p;
                    contrib += q_shared[h * DK + d] * (float) k_tile[j * DK + d];
                }
                float s = sub_group_reduce_add(contrib) * scale;
                if (mask_base[h] != NULL) {
                    const global MASK_DATA_TYPE * mask_ptr = (const global MASK_DATA_TYPE *) mask_base[h];
                    s += slope[h] * (float) mask_ptr[kv_idx];
                }
                if (logit_softcap > 0.0f) {
                    s = logit_softcap * tanh(s / logit_softcap);
                }
                score[h] = s;
            }

            // Online softmax update + V accumulation per h.
            float p_h[MQ_GQA];
            float sp_h[MQ_GQA];
            #pragma unroll
            for (int h = 0; h < MQ_GQA; ++h) {
                const float m_new = fmax(m_i[h], score[h]);
                sp_h[h] = native_exp(m_i[h] - m_new);
                p_h[h]  = native_exp(score[h] - m_new);
                l_i[h]  = l_i[h] * sp_h[h] + p_h[h];
                m_i[h]  = m_new;
            }

            #pragma unroll
            for (int p = 0; p < LMQ_DPL; ++p) {
                const int d = tid * LMQ_DPL + p;
                const float v_val = (float) v_tile[j * DV + d];
                #pragma unroll
                for (int h = 0; h < MQ_GQA; ++h) {
                    o_acc[h][p] = o_acc[h][p] * sp_h[h] + p_h[h] * v_val;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);  // Before next tile load overwrites k/v_tile.
    }

    // write partial records: one per (h, split)
    #pragma unroll
    for (int h = 0; h < MQ_GQA; ++h) {
        const int head_idx = head_kv_idx * MQ_GQA + h;
        const ulong rec_idx = ((((ulong) batch_idx * n_head + head_idx) * n_q + q_idx)
                               * n_splits + split_idx);
        global float * rec   = partial_void + rec_idx * record_stride;
        global float * rec_o = rec + 2;

        if (tid == 0) {
            rec[0] = m_i[h];
            rec[1] = l_i[h];
        }
        #pragma unroll
        for (int p = 0; p < LMQ_DPL; ++p) {
            const int d = tid * LMQ_DPL + p;
            rec_o[d] = o_acc[h][p];
        }
    }
}

#endif  // !FA_MQ_ONLY (q1_local_mq_split)
#ifndef MQ_NSG
#define MQ_NSG 4
#endif
#define MQ_WG_SIZE (Q1_WG_SIZE * MQ_NSG)

REQD_SUBGROUP_SIZE_64
__kernel void flash_attn_f32_f16_q1_vec_mq(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset
) {
    const int tid              = get_local_id(0);
    const int sgid             = tid / Q1_WG_SIZE;   // subgroup 0..MQ_NSG-1
    const int tid_sg           = tid % Q1_WG_SIZE;   // lane 0..63
    const int kvhead_batch_idx = get_global_id(1);

    const int batch_idx   = kvhead_batch_idx / n_head_kv;
    const int head_kv_idx = kvhead_batch_idx % n_head_kv;

    const global char * q_base = (const global char *) q_void + q_offset;
    const global char * k_base = (const global char *) k_void + k_offset;
    const global char * v_base = (const global char *) v_void + v_offset;
    global       char * o_base = (global       char *) o_void + o_offset;

    __local ACC_TYPE4 q_shared[MQ_GQA * DK_VEC];
    for (int i = tid; i < MQ_GQA * DK_VEC; i += MQ_WG_SIZE) {
        const int h        = i / DK_VEC;
        const int k        = i % DK_VEC;
        const int head_idx = head_kv_idx * MQ_GQA + h;
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2;
        const global Q_DATA_TYPE4 * q_ptr = (const global Q_DATA_TYPE4 *) (q_base + q_row_offset);
        q_shared[h * DK_VEC + k] = CONVERT_Q_ACC4(q_ptr[k]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // per-h ALiBi slope
    float slope[MQ_GQA];
    #pragma unroll
    for (int h = 0; h < MQ_GQA; ++h) {
        slope[h] = get_alibi_slope(max_bias, head_kv_idx * MQ_GQA + h, n_head_log2, m0, m1);
    }

    // per-h mask row pointer
    const global char * mask_base[MQ_GQA];
    if (mask_void != NULL) {
        const int mask_batch_idx = batch_idx % mask_ne3;
        const global char * mask_base_b = (const global char *) mask_void + mask_offset +
                                          mask_batch_idx * mask_nb3;
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
            const int head_idx      = head_kv_idx * MQ_GQA + h;
            const int mask_head_idx = head_idx % mask_ne2;
            mask_base[h] = mask_base_b + mask_head_idx * mask_nb2;
        }
    } else {
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) mask_base[h] = NULL;
    }

    const global ACC_TYPE * sinks_ptr = NULL;
    if (sinks_void != NULL) {
        sinks_ptr = (const global ACC_TYPE *) ((const global char *) sinks_void + sinks_offset);
    }

    // per-thread per-h DV slice.
    ACC_TYPE4 o_acc[MQ_GQA][Q1V_DV_PER_THREAD];
    ACC_TYPE  m_i[MQ_GQA];
    ACC_TYPE  l_i[MQ_GQA];
    #pragma unroll
    for (int h = 0; h < MQ_GQA; ++h) {
        m_i[h] = FA_M_INIT;
        l_i[h] = 0.0f;
        #pragma unroll
        for (int i = 0; i < Q1V_DV_PER_THREAD; ++i) o_acc[h][i] = (ACC_TYPE4)(0.0f);
    }

    // each subgroup independently sweeps its slice of n_kv.
    const int kv_per_sg = (n_kv + MQ_NSG - 1) / MQ_NSG;
    const int kv_start  = sgid * kv_per_sg;
    const int kv_end    = min(n_kv, kv_start + kv_per_sg);

    for (int k_idx = kv_start; k_idx < kv_end; ++k_idx) {
        const ulong k_row_off = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const ulong v_row_off = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
        const global KV_DATA_TYPE4 * k_ptr = (const global KV_DATA_TYPE4 *) (k_base + k_row_off);
        const global KV_DATA_TYPE4 * v_ptr = (const global KV_DATA_TYPE4 *) (v_base + v_row_off);

        // Q*K^T: load each K stride once, dot against all MQ_GQA Q rows.
        ACC_TYPE4 dot4[MQ_GQA];
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) dot4[h] = (ACC_TYPE4)(0.0f);
        for (int k = tid_sg; k < DK_VEC; k += Q1_WG_SIZE) {
            const ACC_TYPE4 k_vec = CONVERT_KV_ACC4(k_ptr[k]);
            #pragma unroll
            for (int h = 0; h < MQ_GQA; ++h) {
                dot4[h] = mad(q_shared[h * DK_VEC + k], k_vec, dot4[h]);
            }
        }

        ACC_TYPE score[MQ_GQA];
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
            const ACC_TYPE dot_partial = dot4[h].s0 + dot4[h].s1 + dot4[h].s2 + dot4[h].s3;
            ACC_TYPE s = sub_group_reduce_add(dot_partial) * scale;
            if (mask_base[h] != NULL) {
                const global MASK_DATA_TYPE * mask_ptr = (const global MASK_DATA_TYPE *) mask_base[h];
                s += slope[h] * (ACC_TYPE) mask_ptr[k_idx];
            }
            if (logit_softcap > 0.0f) {
                s = logit_softcap * tanh(s / logit_softcap);
            }
            score[h] = s;
        }

        // FA-2 online softmax update — V load amortized across MQ_GQA heads.
        // p, scale_prev are computed per h; the V vector is loaded once
        // per dv stride and reused MQ_GQA times.
        ACC_TYPE p_h[MQ_GQA];
        ACC_TYPE sp_h[MQ_GQA];
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
            const ACC_TYPE m_new = max(m_i[h], score[h]);
            sp_h[h] = native_exp(m_i[h] - m_new);
            p_h[h]  = native_exp(score[h] - m_new);
            l_i[h]  = l_i[h] * sp_h[h] + p_h[h];
            m_i[h]  = m_new;
        }

        int idx = 0;
        for (int dv_idx = tid_sg; dv_idx < DV_VEC; dv_idx += Q1_WG_SIZE, ++idx) {
            const ACC_TYPE4 v_vec = CONVERT_KV_ACC4(v_ptr[dv_idx]);
            #pragma unroll
            for (int h = 0; h < MQ_GQA; ++h) {
                o_acc[h][idx] = mad(p_h[h], v_vec, o_acc[h][idx] * sp_h[h]);
            }
        }
    }

    // cross subgroup merge
    __local ACC_TYPE  sg_m[MQ_GQA][MQ_NSG];
    __local ACC_TYPE  sg_l[MQ_GQA][MQ_NSG];
    __local ACC_TYPE4 sg_o[MQ_NSG][DV_VEC];

    if (tid_sg == 0) {
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
            sg_m[h][sgid] = m_i[h];
            sg_l[h][sgid] = l_i[h];
        }
    }

    #pragma unroll
    for (int h = 0; h < MQ_GQA; ++h) {
        // each subgroup publishes its o_acc slice for head h.
        {
            int idx = 0;
            for (int dv_idx = tid_sg; dv_idx < DV_VEC; dv_idx += Q1_WG_SIZE, ++idx) {
                sg_o[sgid][dv_idx] = o_acc[h][idx];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (sgid == 0) {
            const int head_idx = head_kv_idx * MQ_GQA + h;

            ACC_TYPE m_final = sg_m[h][0];
            #pragma unroll
            for (int s = 1; s < MQ_NSG; ++s) {
                m_final = max(m_final, sg_m[h][s]);
            }
            if (sinks_ptr != NULL) {
                m_final = max(m_final, sinks_ptr[head_idx]);
            }

            ACC_TYPE l_final = 0.0f;
            #pragma unroll
            for (int s = 0; s < MQ_NSG; ++s) {
                l_final += sg_l[h][s] * native_exp(sg_m[h][s] - m_final);
            }
            if (sinks_ptr != NULL) {
                l_final += native_exp(sinks_ptr[head_idx] - m_final);
            }
            const ACC_TYPE l_inv = (l_final > 0.0f) ? (1.0f / l_final) : 0.0f;

            const ulong o_row_offset = batch_idx * o_nb3 + head_idx * o_nb1;
            global O_DATA_TYPE4 * o_row = (global O_DATA_TYPE4 *) (o_base + o_row_offset);
            for (int dv_idx = tid_sg; dv_idx < DV_VEC; dv_idx += Q1_WG_SIZE) {
                ACC_TYPE4 o_merged = (ACC_TYPE4)(0.0f);
                #pragma unroll
                for (int s = 0; s < MQ_NSG; ++s) {
                    const ACC_TYPE alpha = native_exp(sg_m[h][s] - m_final);
                    o_merged = mad((ACC_TYPE4)(alpha), sg_o[s][dv_idx], o_merged);
                }
                o_row[dv_idx] = CONVERT_O_DATA4(o_merged * l_inv);
            }
        }
        // Barrier guards next h's overwrite of sg_o.
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

#ifndef MQ_NSG_SPLIT
#define MQ_NSG_SPLIT 4
#endif
#define MQ_SPLIT_WG_SIZE (Q1_WG_SIZE * MQ_NSG_SPLIT)

#ifndef FA_PARTIAL_FLOATS
#define FA_PARTIAL_FLOATS (2 + DV)
#endif

REQD_SUBGROUP_SIZE_64
__kernel void flash_attn_f32_f16_q1_vec_mq_split(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void * mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    global float * partial_void,
    const int n_splits,
    const int kv_per_split
) {
    const int tid              = get_local_id(0);
    const int sgid             = tid / Q1_WG_SIZE;
    const int tid_sg           = tid % Q1_WG_SIZE;
    const int kvhead_batch_idx = get_global_id(1);
    const int split_q_idx      = get_global_id(2);
    const int split_idx        = split_q_idx % n_splits;
    const int q_idx            = split_q_idx / n_splits;

    const int batch_idx   = kvhead_batch_idx / n_head_kv;
    const int head_kv_idx = kvhead_batch_idx % n_head_kv;

    const int kv_start = split_idx * kv_per_split;
    const int kv_end   = min(kv_start + kv_per_split, n_kv);

    const ulong record_stride = (ulong) FA_PARTIAL_FLOATS;

    if (kv_start >= kv_end) {
        // write sentinel for each of the MQ_GQA Q-heads so the
        // merge pass treats this slot as dropped
        if (tid == 0) {
            #pragma unroll
            for (int h = 0; h < MQ_GQA; ++h) {
                const int head_idx = head_kv_idx * MQ_GQA + h;
                const ulong rec_idx = ((((ulong) batch_idx * n_head + head_idx) * n_q + q_idx)
                                       * n_splits + split_idx);
                global float * rec = partial_void + rec_idx * record_stride;
                rec[0] = FA_M_INIT;
                rec[1] = 0.0f;
            }
        }
        return;
    }

    const global char * q_base = (const global char *) q_void + q_offset;
    const global char * k_base = (const global char *) k_void + k_offset;
    const global char * v_base = (const global char *) v_void + v_offset;

    // stage MQ_GQA Q rows in __local once (uniform across WG)
    __local ACC_TYPE4 q_shared[MQ_GQA * DK_VEC];
    for (int i = tid; i < MQ_GQA * DK_VEC; i += MQ_SPLIT_WG_SIZE) {
        const int h        = i / DK_VEC;
        const int k        = i % DK_VEC;
        const int head_idx = head_kv_idx * MQ_GQA + h;
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + (ulong) q_idx * q_nb1;
        const global Q_DATA_TYPE4 * q_ptr = (const global Q_DATA_TYPE4 *) (q_base + q_row_offset);
        q_shared[h * DK_VEC + k] = CONVERT_Q_ACC4(q_ptr[k]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float slope[MQ_GQA];
    #pragma unroll
    for (int h = 0; h < MQ_GQA; ++h) {
        slope[h] = get_alibi_slope(max_bias, head_kv_idx * MQ_GQA + h, n_head_log2, m0, m1);
    }

    const global char * mask_base[MQ_GQA];
    if (mask_void != NULL) {
        const int mask_batch_idx = batch_idx % mask_ne3;
        const global char * mask_base_b = (const global char *) mask_void + mask_offset +
                                          mask_batch_idx * mask_nb3 +
                                          (ulong) q_idx * mask_nb1;
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
            const int head_idx      = head_kv_idx * MQ_GQA + h;
            const int mask_head_idx = head_idx % mask_ne2;
            mask_base[h] = mask_base_b + mask_head_idx * mask_nb2;
        }
    } else {
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) mask_base[h] = NULL;
    }

    ACC_TYPE4 o_acc[MQ_GQA][Q1V_DV_PER_THREAD];
    ACC_TYPE  m_i[MQ_GQA];
    ACC_TYPE  l_i[MQ_GQA];
    #pragma unroll
    for (int h = 0; h < MQ_GQA; ++h) {
        m_i[h] = FA_M_INIT;
        l_i[h] = 0.0f;
        #pragma unroll
        for (int i = 0; i < Q1V_DV_PER_THREAD; ++i) o_acc[h][i] = (ACC_TYPE4)(0.0f);
    }

    // each subgroup independently sweeps its slice of the split's kv range.
    const int kv_len   = kv_end - kv_start;
    const int kv_per_sg = (kv_len + MQ_NSG_SPLIT - 1) / MQ_NSG_SPLIT;
    const int kv_lo    = kv_start + sgid * kv_per_sg;
    const int kv_hi    = min(kv_end, kv_lo + kv_per_sg);

    for (int k_idx = kv_lo; k_idx < kv_hi; ++k_idx) {
        const ulong k_row_off = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const ulong v_row_off = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
        const global KV_DATA_TYPE4 * k_ptr = (const global KV_DATA_TYPE4 *) (k_base + k_row_off);
        const global KV_DATA_TYPE4 * v_ptr = (const global KV_DATA_TYPE4 *) (v_base + v_row_off);

        ACC_TYPE4 dot4[MQ_GQA];
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) dot4[h] = (ACC_TYPE4)(0.0f);
        for (int k = tid_sg; k < DK_VEC; k += Q1_WG_SIZE) {
            const ACC_TYPE4 k_vec = CONVERT_KV_ACC4(k_ptr[k]);
            #pragma unroll
            for (int h = 0; h < MQ_GQA; ++h) {
                dot4[h] = mad(q_shared[h * DK_VEC + k], k_vec, dot4[h]);
            }
        }

        ACC_TYPE score[MQ_GQA];
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
            const ACC_TYPE dot_partial = dot4[h].s0 + dot4[h].s1 + dot4[h].s2 + dot4[h].s3;
            ACC_TYPE s = sub_group_reduce_add(dot_partial) * scale;
            if (mask_base[h] != NULL) {
                const global MASK_DATA_TYPE * mask_ptr = (const global MASK_DATA_TYPE *) mask_base[h];
                s += slope[h] * (ACC_TYPE) mask_ptr[k_idx];
            }
            if (logit_softcap > 0.0f) {
                s = logit_softcap * tanh(s / logit_softcap);
            }
            score[h] = s;
        }

        ACC_TYPE p_h[MQ_GQA];
        ACC_TYPE sp_h[MQ_GQA];
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
            const ACC_TYPE m_new = max(m_i[h], score[h]);
            sp_h[h] = native_exp(m_i[h] - m_new);
            p_h[h]  = native_exp(score[h] - m_new);
            l_i[h]  = l_i[h] * sp_h[h] + p_h[h];
            m_i[h]  = m_new;
        }

        int idx = 0;
        for (int dv_idx = tid_sg; dv_idx < DV_VEC; dv_idx += Q1_WG_SIZE, ++idx) {
            const ACC_TYPE4 v_vec = CONVERT_KV_ACC4(v_ptr[dv_idx]);
            #pragma unroll
            for (int h = 0; h < MQ_GQA; ++h) {
                o_acc[h][idx] = mad(p_h[h], v_vec, o_acc[h][idx] * sp_h[h]);
            }
        }
    }

    // per-h cross-subgroup merge
    __local ACC_TYPE  sg_m[MQ_GQA][MQ_NSG_SPLIT];
    __local ACC_TYPE  sg_l[MQ_GQA][MQ_NSG_SPLIT];
    __local ACC_TYPE4 sg_o[MQ_NSG_SPLIT][DV_VEC];

    if (tid_sg == 0) {
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
            sg_m[h][sgid] = m_i[h];
            sg_l[h][sgid] = l_i[h];
        }
    }

    #pragma unroll
    for (int h = 0; h < MQ_GQA; ++h) {
        {
            int idx = 0;
            for (int dv_idx = tid_sg; dv_idx < DV_VEC; dv_idx += Q1_WG_SIZE, ++idx) {
                sg_o[sgid][dv_idx] = o_acc[h][idx];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (sgid == 0) {
            const int head_idx = head_kv_idx * MQ_GQA + h;

            // fold per-subgroup (m, l) into split-level (m_c, l_c)
            ACC_TYPE m_c = sg_m[h][0];
            #pragma unroll
            for (int s = 1; s < MQ_NSG_SPLIT; ++s) {
                m_c = max(m_c, sg_m[h][s]);
            }
            ACC_TYPE l_c = 0.0f;
            #pragma unroll
            for (int s = 0; s < MQ_NSG_SPLIT; ++s) {
                l_c += sg_l[h][s] * native_exp(sg_m[h][s] - m_c);
            }

            const ulong rec_idx = ((((ulong) batch_idx * n_head + head_idx) * n_q + q_idx)
                                   * n_splits + split_idx);
            global float  * rec   = partial_void + rec_idx * record_stride;
            global float4 * rec_o = (global float4 *) (rec + 2);

            if (tid_sg == 0) {
                rec[0] = (float) m_c;
                rec[1] = (float) l_c;
            }
            // each thread writes its DV slice of the merged O.
            for (int dv_idx = tid_sg; dv_idx < DV_VEC; dv_idx += Q1_WG_SIZE) {
                ACC_TYPE4 o_merged = (ACC_TYPE4)(0.0f);
                #pragma unroll
                for (int s = 0; s < MQ_NSG_SPLIT; ++s) {
                    const ACC_TYPE alpha = native_exp(sg_m[h][s] - m_c);
                    o_merged = mad((ACC_TYPE4)(alpha), sg_o[s][dv_idx], o_merged);
                }
                rec_o[dv_idx] = o_merged;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// Cluster-parallel variant of _q1_vec_mq_split
//
// Tthe baseline keeps one 256B K row in flight per subgroup (32 lanes cooperate
// on one position, serialized by the reduce+exp chain). This kernel
// takes q1_split's memory-level parallelism at MQ's read-once traffic:
//   - the 64-lane subgroup is split into FA_CL_NCL clusters of FA_CL_C lanes;
//   - each cluster owns its own KV position stream (positions strided by
//     FA_CL_NCL) with private per-cluster online-softmax state, hence FA_CL_NCL
//     independent K rows in flight per subgroup, no cross-cluster serial chain;
//   - within a cluster, lanes split DK for the dot (cluster-reduce via
//     sub_group_shuffle_xor, steps < FA_CL_C stay inside the cluster) and
//     split DV for o_acc (each lane owns dv indices {lic + FA_CL_C*i} — the
//     same slice for every position, so accumulation is lane-local);
//   - merge stage 1 folds the FA_CL_NCL cluster partials with cross-cluster
//     shuffles (distances >= FA_CL_C); stage 2 is the baseline cross-subgroup
//     LDS merge (o published by cluster 0's lanes, layout-identical to the
//     baseline's sg_o).
// The KV sweep runs a UNIFORM trip count (max over clusters) with a clamped
// row address + FA_M_INIT score on the tail — keeps every shuffle convergent
// (p = exp(FA_M_INIT - m) underflows to 0, so clamped-row reads are inert).
// Register cost vs baseline: o_acc grows from DV_VEC/64 to DV_VEC/FA_CL_C
// float4 per lane per head — FA_CL_C=8 / MQ_GQA=4 => 16 float4 (256B).

#ifdef HAS_SUBGROUP_SHUFFLE  // cluster reduce/merge needs shuffles; absent -> kernel dropped, dispatch falls back

#ifndef FA_CL_C
#define FA_CL_C 8
#endif

// The lane striping requires DK/DV to divide evenly across the cluster;
// otherwise (e.g. DK=40 with FA_CL_C=16 -> zero-size arrays) compile the
// kernel out — host soft-create falls back silently.
#if (DK_VEC % FA_CL_C) == 0 && (DV_VEC % FA_CL_C) == 0
#define FA_CL_NCL (Q1_WG_SIZE / FA_CL_C)   // clusters (position streams) per subgroup
#define FA_CL_DK  (DK_VEC / FA_CL_C)       // half4s of K per lane per row
#define FA_CL_DV  (DV_VEC / FA_CL_C)       // float4s of o_acc per lane per head

// explicit "half" sub-group attribute routes this fp16-heavy kernel to a slow
// codegen path on the X1 compiler. X2 keeps the pin: its driver miscompile
// without it.
#ifdef FA_C8_NO_SG_PIN
#define FA_C8_SG_ATTR
#else
// REQD_FA_SG pins the HW subgroup on Intel (intel_reqd_sub_group_size(FA_SG),
// host passes -D FA_SG=32); empty on Adreno. REQD_SUBGROUP_SIZE_64 pins 64 on
// Adreno; empty on Intel.
#define FA_C8_SG_ATTR REQD_FA_SG REQD_SUBGROUP_SIZE_64
#endif

FA_C8_SG_ATTR
__kernel void flash_attn_f32_f16_q1_vec_mq_split_c8(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void * mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    global float * partial_void,
    const int n_splits,
    const int kv_per_split
) {
    const int tid              = get_local_id(0);
    const int sgid             = tid / Q1_WG_SIZE;
    const int tid_sg           = tid % Q1_WG_SIZE;
    const int cl               = tid_sg / FA_CL_C;   // cluster id
    const int lic              = tid_sg % FA_CL_C;   // lane in cluster
    const int kvhead_batch_idx = get_global_id(1);
    const int split_q_idx      = get_global_id(2);
    const int split_idx        = split_q_idx % n_splits;
    const int q_idx            = split_q_idx / n_splits;

    const int batch_idx   = kvhead_batch_idx / n_head_kv;
    const int head_kv_idx = kvhead_batch_idx % n_head_kv;

    const int kv_start = split_idx * kv_per_split;
    const int kv_end   = min(kv_start + kv_per_split, n_kv);

    const ulong record_stride = (ulong) FA_PARTIAL_FLOATS;

    if (kv_start >= kv_end) {
        if (tid == 0) {
            #pragma unroll
            for (int h = 0; h < MQ_GQA; ++h) {
                const int head_idx = head_kv_idx * MQ_GQA + h;
                const ulong rec_idx = ((((ulong) batch_idx * n_head + head_idx) * n_q + q_idx)
                                       * n_splits + split_idx);
                global float * rec = partial_void + rec_idx * record_stride;
                rec[0] = FA_M_INIT;
                rec[1] = 0.0f;
            }
        }
        return;
    }

    const global char * q_base = (const global char *) q_void + q_offset;
    const global char * k_base = (const global char *) k_void + k_offset;
    const global char * v_base = (const global char *) v_void + v_offset;

    // Stage MQ_GQA Q rows in __local once (uniform across WG).
    __local ACC_TYPE4 q_shared[MQ_GQA * DK_VEC];
    for (int i = tid; i < MQ_GQA * DK_VEC; i += MQ_SPLIT_WG_SIZE) {
        const int h        = i / DK_VEC;
        const int k        = i % DK_VEC;
        const int head_idx = head_kv_idx * MQ_GQA + h;
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + (ulong) q_idx * q_nb1;
        const global Q_DATA_TYPE4 * q_ptr = (const global Q_DATA_TYPE4 *) (q_base + q_row_offset);
        q_shared[h * DK_VEC + k] = CONVERT_Q_ACC4(q_ptr[k]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float slope[MQ_GQA];
    #pragma unroll
    for (int h = 0; h < MQ_GQA; ++h) {
        slope[h] = get_alibi_slope(max_bias, head_kv_idx * MQ_GQA + h, n_head_log2, m0, m1);
    }

    const global char * mask_base[MQ_GQA];
    if (mask_void != NULL) {
        const int mask_batch_idx = batch_idx % mask_ne3;
        const global char * mask_base_b = (const global char *) mask_void + mask_offset +
                                          mask_batch_idx * mask_nb3 +
                                          (ulong) q_idx * mask_nb1;
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
            const int head_idx      = head_kv_idx * MQ_GQA + h;
            const int mask_head_idx = head_idx % mask_ne2;
            mask_base[h] = mask_base_b + mask_head_idx * mask_nb2;
        }
    } else {
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) mask_base[h] = NULL;
    }

    // Per-CLUSTER online-softmax state (uniform across the cluster's lanes);
    // o_acc holds this lane's DV slice {lic + FA_CL_C*i}.
    ACC_TYPE4 o_acc[MQ_GQA][FA_CL_DV];
    ACC_TYPE  m_i[MQ_GQA];
    ACC_TYPE  l_i[MQ_GQA];
    #pragma unroll
    for (int h = 0; h < MQ_GQA; ++h) {
        m_i[h] = FA_M_INIT;
        l_i[h] = 0.0f;
        #pragma unroll
        for (int i = 0; i < FA_CL_DV; ++i) o_acc[h][i] = (ACC_TYPE4)(0.0f);
    }

    const int kv_len    = kv_end - kv_start;
    const int kv_per_sg = (kv_len + MQ_NSG_SPLIT - 1) / MQ_NSG_SPLIT;
    const int kv_lo     = kv_start + sgid * kv_per_sg;
    const int kv_hi     = min(kv_end, kv_lo + kv_per_sg);

    // Uniform trip count across the subgroup: every cluster runs n_iter
    // iterations; tail positions clamp the row address and drop the score to
    // FA_M_INIT so shuffles stay convergent and the contribution is exactly 0.
    const int n_iter = (kv_hi - kv_lo + FA_CL_NCL - 1) / FA_CL_NCL;
    const ulong kv_row_base = batch_idx * k_nb3 + head_kv_idx * k_nb2;
    const ulong v_row_base  = batch_idx * v_nb3 + head_kv_idx * v_nb2;

    for (int it = 0; it < n_iter; ++it) {
        const int k_idx = kv_lo + cl + it * FA_CL_NCL;
        const int valid = k_idx < kv_hi;
        const int k_safe = valid ? k_idx : (kv_hi - 1);

        const global KV_DATA_TYPE4 * k_ptr = (const global KV_DATA_TYPE4 *) (k_base + kv_row_base + (ulong) k_safe * k_nb1);
        const global KV_DATA_TYPE4 * v_ptr = (const global KV_DATA_TYPE4 *) (v_base + v_row_base  + (ulong) k_safe * v_nb1);

        // Dot: this lane covers DK elements {lic + FA_CL_C*i} of the cluster's row.
        ACC_TYPE4 dot4[MQ_GQA];
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) dot4[h] = (ACC_TYPE4)(0.0f);
        #pragma unroll
        for (int i = 0; i < FA_CL_DK; ++i) {
            const int kk = lic + FA_CL_C * i;
            const ACC_TYPE4 k_vec = CONVERT_KV_ACC4(k_ptr[kk]);
            #pragma unroll
            for (int h = 0; h < MQ_GQA; ++h) {
                dot4[h] = mad(q_shared[h * DK_VEC + kk], k_vec, dot4[h]);
            }
        }

        // Cluster-reduce (xor steps < FA_CL_C stay inside the cluster) + score.
        ACC_TYPE score[MQ_GQA];
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
            ACC_TYPE s = dot4[h].s0 + dot4[h].s1 + dot4[h].s2 + dot4[h].s3;
            #pragma unroll
            for (int step = 1; step < FA_CL_C; step <<= 1) {
                s += sub_group_shuffle_xor(s, step);
            }
            s *= scale;
            if (mask_base[h] != NULL) {
                const global MASK_DATA_TYPE * mask_ptr = (const global MASK_DATA_TYPE *) mask_base[h];
                s += slope[h] * (ACC_TYPE) mask_ptr[k_safe];
            }
            if (logit_softcap > 0.0f) {
                s = logit_softcap * tanh(s / logit_softcap);
            }
            score[h] = valid ? s : FA_M_INIT;
        }

        // Per-cluster online update — identical math to the baseline, but the
        // serial chain is per cluster (depth n_iter, not kv_per_sg).
        ACC_TYPE p_h[MQ_GQA];
        ACC_TYPE sp_h[MQ_GQA];
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
            const ACC_TYPE m_new = max(m_i[h], score[h]);
            sp_h[h] = native_exp(m_i[h] - m_new);
            p_h[h]  = native_exp(score[h] - m_new);
            l_i[h]  = l_i[h] * sp_h[h] + p_h[h];
            m_i[h]  = m_new;
        }

        // V accumulate on this lane's DV slice (p = 0 on tail -> inert).
        #pragma unroll
        for (int i = 0; i < FA_CL_DV; ++i) {
            const ACC_TYPE4 v_vec = CONVERT_KV_ACC4(v_ptr[lic + FA_CL_C * i]);
            #pragma unroll
            for (int h = 0; h < MQ_GQA; ++h) {
                o_acc[h][i] = mad(p_h[h], v_vec, o_acc[h][i] * sp_h[h]);
            }
        }
    }

    // Merge stage 1: fold the FA_CL_NCL cluster partials inside the subgroup.
    // Lanes with equal lic across clusters hold the SAME dv slice, so a
    // cross-cluster xor-reduce (distances FA_CL_C..Q1_WG_SIZE/2) sums o
    // slice-wise; m/l fold the same way. All shuffles are subgroup-convergent.
    #pragma unroll
    for (int h = 0; h < MQ_GQA; ++h) {
        ACC_TYPE m_c = m_i[h];
        #pragma unroll
        for (int step = FA_CL_C; step < Q1_WG_SIZE; step <<= 1) {
            m_c = max(m_c, sub_group_shuffle_xor(m_c, step));
        }
        const ACC_TYPE alpha = native_exp(m_i[h] - m_c);
        ACC_TYPE l_c = l_i[h] * alpha;
        #pragma unroll
        for (int step = FA_CL_C; step < Q1_WG_SIZE; step <<= 1) {
            l_c += sub_group_shuffle_xor(l_c, step);
        }
        #pragma unroll
        for (int i = 0; i < FA_CL_DV; ++i) {
            ACC_TYPE4 o = o_acc[h][i] * alpha;
            #pragma unroll
            for (int step = FA_CL_C; step < Q1_WG_SIZE; step <<= 1) {
                o.s0 += sub_group_shuffle_xor(o.s0, step);
                o.s1 += sub_group_shuffle_xor(o.s1, step);
                o.s2 += sub_group_shuffle_xor(o.s2, step);
                o.s3 += sub_group_shuffle_xor(o.s3, step);
            }
            o_acc[h][i] = o;
        }
        m_i[h] = m_c;
        l_i[h] = l_c;
    }

    // Merge stage 2: baseline cross-subgroup LDS merge. Cluster 0's lanes hold
    // the subgroup's merged o (dv indices {lic + FA_CL_C*i}) — same sg_o layout
    // and fold loop as q1_vec_mq_split.
    __local ACC_TYPE  sg_m[MQ_GQA][MQ_NSG_SPLIT];
    __local ACC_TYPE  sg_l[MQ_GQA][MQ_NSG_SPLIT];
    __local ACC_TYPE4 sg_o[MQ_NSG_SPLIT][DV_VEC];

    if (tid_sg == 0) {
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
            sg_m[h][sgid] = m_i[h];
            sg_l[h][sgid] = l_i[h];
        }
    }

    #pragma unroll
    for (int h = 0; h < MQ_GQA; ++h) {
        if (cl == 0) {
            #pragma unroll
            for (int i = 0; i < FA_CL_DV; ++i) {
                sg_o[sgid][lic + FA_CL_C * i] = o_acc[h][i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (sgid == 0) {
            const int head_idx = head_kv_idx * MQ_GQA + h;

            ACC_TYPE m_c = sg_m[h][0];
            #pragma unroll
            for (int s = 1; s < MQ_NSG_SPLIT; ++s) {
                m_c = max(m_c, sg_m[h][s]);
            }
            ACC_TYPE l_c = 0.0f;
            #pragma unroll
            for (int s = 0; s < MQ_NSG_SPLIT; ++s) {
                l_c += sg_l[h][s] * native_exp(sg_m[h][s] - m_c);
            }

            const ulong rec_idx = ((((ulong) batch_idx * n_head + head_idx) * n_q + q_idx)
                                   * n_splits + split_idx);
            global float  * rec   = partial_void + rec_idx * record_stride;
            global float4 * rec_o = (global float4 *) (rec + 2);

            if (tid_sg == 0) {
                rec[0] = (float) m_c;
                rec[1] = (float) l_c;
            }
            for (int dv_idx = tid_sg; dv_idx < DV_VEC; dv_idx += Q1_WG_SIZE) {
                ACC_TYPE4 o_merged = (ACC_TYPE4)(0.0f);
                #pragma unroll
                for (int s = 0; s < MQ_NSG_SPLIT; ++s) {
                    const ACC_TYPE alpha = native_exp(sg_m[h][s] - m_c);
                    o_merged = mad((ACC_TYPE4)(alpha), sg_o[s][dv_idx], o_merged);
                }
                rec_o[dv_idx] = o_merged;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

#endif  // DK_VEC/DV_VEC divisible by FA_CL_C
#endif  // HAS_SUBGROUP_SHUFFLE (q1_vec_mq_split_c8)

REQD_SUBGROUP_SIZE_64
__kernel void flash_attn_f32_f16_q1_vec_mq_split_k_img(
    const global void * q_void, ulong q_offset,
    __read_only image1d_buffer_t k_img,
    const global void * v_void, ulong v_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void * mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    global float * partial_void,
    const int n_splits,
    const int kv_per_split
) {
    const int tid              = get_local_id(0);
    const int sgid             = tid / Q1_WG_SIZE;
    const int tid_sg           = tid % Q1_WG_SIZE;
    const int kvhead_batch_idx = get_global_id(1);
    const int split_q_idx      = get_global_id(2);
    const int split_idx        = split_q_idx % n_splits;
    const int q_idx            = split_q_idx / n_splits;

    const int batch_idx   = kvhead_batch_idx / n_head_kv;
    const int head_kv_idx = kvhead_batch_idx % n_head_kv;

    const int kv_start = split_idx * kv_per_split;
    const int kv_end   = min(kv_start + kv_per_split, n_kv);

    const ulong record_stride = (ulong) FA_PARTIAL_FLOATS;

    if (kv_start >= kv_end) {
        if (tid == 0) {
            #pragma unroll
            for (int h = 0; h < MQ_GQA; ++h) {
                const int head_idx = head_kv_idx * MQ_GQA + h;
                const ulong rec_idx = ((((ulong) batch_idx * n_head + head_idx) * n_q + q_idx)
                                       * n_splits + split_idx);
                global float * rec = partial_void + rec_idx * record_stride;
                rec[0] = FA_M_INIT;
                rec[1] = 0.0f;
            }
        }
        return;
    }

    const global char * q_base = (const global char *) q_void + q_offset;
    const global char * v_base = (const global char *) v_void + v_offset;

    __local ACC_TYPE4 q_shared[MQ_GQA * DK_VEC];
    for (int i = tid; i < MQ_GQA * DK_VEC; i += MQ_SPLIT_WG_SIZE) {
        const int h        = i / DK_VEC;
        const int k        = i % DK_VEC;
        const int head_idx = head_kv_idx * MQ_GQA + h;
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + (ulong) q_idx * q_nb1;
        const global Q_DATA_TYPE4 * q_ptr = (const global Q_DATA_TYPE4 *) (q_base + q_row_offset);
        q_shared[h * DK_VEC + k] = CONVERT_Q_ACC4(q_ptr[k]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float slope[MQ_GQA];
    #pragma unroll
    for (int h = 0; h < MQ_GQA; ++h) {
        slope[h] = get_alibi_slope(max_bias, head_kv_idx * MQ_GQA + h, n_head_log2, m0, m1);
    }

    const global char * mask_base[MQ_GQA];
    if (mask_void != NULL) {
        const int mask_batch_idx = batch_idx % mask_ne3;
        const global char * mask_base_b = (const global char *) mask_void + mask_offset +
                                          mask_batch_idx * mask_nb3 +
                                          (ulong) q_idx * mask_nb1;
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
            const int head_idx      = head_kv_idx * MQ_GQA + h;
            const int mask_head_idx = head_idx % mask_ne2;
            mask_base[h] = mask_base_b + mask_head_idx * mask_nb2;
        }
    } else {
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) mask_base[h] = NULL;
    }

    ACC_TYPE4 o_acc[MQ_GQA][Q1V_DV_PER_THREAD];
    ACC_TYPE  m_i[MQ_GQA];
    ACC_TYPE  l_i[MQ_GQA];
    #pragma unroll
    for (int h = 0; h < MQ_GQA; ++h) {
        m_i[h] = FA_M_INIT;
        l_i[h] = 0.0f;
        #pragma unroll
        for (int i = 0; i < Q1V_DV_PER_THREAD; ++i) o_acc[h][i] = (ACC_TYPE4)(0.0f);
    }

    // K pitches in pixel units, pixel = 1 half4 = 8 B -> byte_stride >> 3.
    const int pitch_px_row   = (int)(k_nb1 >> 3);
    const int pitch_px_head  = (int)(k_nb2 >> 3);
    const int pitch_px_batch = (int)(k_nb3 >> 3);

    const int kv_len    = kv_end - kv_start;
    const int kv_per_sg = (kv_len + MQ_NSG_SPLIT - 1) / MQ_NSG_SPLIT;
    const int kv_lo     = kv_start + sgid * kv_per_sg;
    const int kv_hi     = min(kv_end, kv_lo + kv_per_sg);

    for (int k_idx = kv_lo; k_idx < kv_hi; ++k_idx) {
        const int k_row_px = batch_idx * pitch_px_batch +
                             head_kv_idx * pitch_px_head +
                             k_idx * pitch_px_row;

        const ulong v_row_off = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
        const global KV_DATA_TYPE4 * v_ptr = (const global KV_DATA_TYPE4 *) (v_base + v_row_off);

        ACC_TYPE4 dot4[MQ_GQA];
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) dot4[h] = (ACC_TYPE4)(0.0f);
        for (int k = tid_sg; k < DK_VEC; k += Q1_WG_SIZE) {
            const half4     k_h4  = read_imageh(k_img, k_row_px + k);
            const ACC_TYPE4 k_vec = CONVERT_KV_ACC4(k_h4);
            #pragma unroll
            for (int h = 0; h < MQ_GQA; ++h) {
                dot4[h] = mad(q_shared[h * DK_VEC + k], k_vec, dot4[h]);
            }
        }

        ACC_TYPE score[MQ_GQA];
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
            const ACC_TYPE dot_partial = dot4[h].s0 + dot4[h].s1 + dot4[h].s2 + dot4[h].s3;
            ACC_TYPE s = sub_group_reduce_add(dot_partial) * scale;
            if (mask_base[h] != NULL) {
                const global MASK_DATA_TYPE * mask_ptr = (const global MASK_DATA_TYPE *) mask_base[h];
                s += slope[h] * (ACC_TYPE) mask_ptr[k_idx];
            }
            if (logit_softcap > 0.0f) {
                s = logit_softcap * tanh(s / logit_softcap);
            }
            score[h] = s;
        }

        ACC_TYPE p_h[MQ_GQA];
        ACC_TYPE sp_h[MQ_GQA];
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
            const ACC_TYPE m_new = max(m_i[h], score[h]);
            sp_h[h] = native_exp(m_i[h] - m_new);
            p_h[h]  = native_exp(score[h] - m_new);
            l_i[h]  = l_i[h] * sp_h[h] + p_h[h];
            m_i[h]  = m_new;
        }

        int idx = 0;
        for (int dv_idx = tid_sg; dv_idx < DV_VEC; dv_idx += Q1_WG_SIZE, ++idx) {
            const ACC_TYPE4 v_vec = CONVERT_KV_ACC4(v_ptr[dv_idx]);
            #pragma unroll
            for (int h = 0; h < MQ_GQA; ++h) {
                o_acc[h][idx] = mad(p_h[h], v_vec, o_acc[h][idx] * sp_h[h]);
            }
        }
    }

    __local ACC_TYPE  sg_m[MQ_GQA][MQ_NSG_SPLIT];
    __local ACC_TYPE  sg_l[MQ_GQA][MQ_NSG_SPLIT];
    __local ACC_TYPE4 sg_o[MQ_NSG_SPLIT][DV_VEC];

    if (tid_sg == 0) {
        #pragma unroll
        for (int h = 0; h < MQ_GQA; ++h) {
            sg_m[h][sgid] = m_i[h];
            sg_l[h][sgid] = l_i[h];
        }
    }

    #pragma unroll
    for (int h = 0; h < MQ_GQA; ++h) {
        {
            int idx = 0;
            for (int dv_idx = tid_sg; dv_idx < DV_VEC; dv_idx += Q1_WG_SIZE, ++idx) {
                sg_o[sgid][dv_idx] = o_acc[h][idx];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (sgid == 0) {
            const int head_idx = head_kv_idx * MQ_GQA + h;

            ACC_TYPE m_c = sg_m[h][0];
            #pragma unroll
            for (int s = 1; s < MQ_NSG_SPLIT; ++s) {
                m_c = max(m_c, sg_m[h][s]);
            }
            ACC_TYPE l_c = 0.0f;
            #pragma unroll
            for (int s = 0; s < MQ_NSG_SPLIT; ++s) {
                l_c += sg_l[h][s] * native_exp(sg_m[h][s] - m_c);
            }

            const ulong rec_idx = ((((ulong) batch_idx * n_head + head_idx) * n_q + q_idx)
                                   * n_splits + split_idx);
            global float  * rec   = partial_void + rec_idx * record_stride;
            global float4 * rec_o = (global float4 *) (rec + 2);

            if (tid_sg == 0) {
                rec[0] = (float) m_c;
                rec[1] = (float) l_c;
            }
            for (int dv_idx = tid_sg; dv_idx < DV_VEC; dv_idx += Q1_WG_SIZE) {
                ACC_TYPE4 o_merged = (ACC_TYPE4)(0.0f);
                #pragma unroll
                for (int s = 0; s < MQ_NSG_SPLIT; ++s) {
                    const ACC_TYPE alpha = native_exp(sg_m[h][s] - m_c);
                    o_merged = mad((ACC_TYPE4)(alpha), sg_o[s][dv_idx], o_merged);
                }
                rec_o[dv_idx] = o_merged;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
#endif  // !FA_DECODE_ONLY

#ifndef FA_MQ_ONLY  // q1_split + merge excluded from the MQ-only (g8) program
__kernel void flash_attn_f32_f16_q1_split(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void * mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    global float * partial_void,
    const int n_splits,
    const int kv_per_split
) {
    const int tid              = get_local_id(0);
    const int head_batch_idx   = get_global_id(1);
    const int split_q_idx      = get_global_id(2);
    const int split_idx        = split_q_idx % n_splits;
    const int q_idx            = split_q_idx / n_splits;
    const int batch_idx        = head_batch_idx / n_head;
    const int head_idx         = head_batch_idx % n_head;
    const int gqa_ratio        = n_head / n_head_kv;
    const int head_kv_idx      = head_idx / gqa_ratio;

    const int kv_start = split_idx * kv_per_split;
    const int kv_end   = min(kv_start + kv_per_split, n_kv);

    const ulong record_stride = (ulong) FA_PARTIAL_FLOATS;
    const ulong record_idx    = ((((ulong) batch_idx * n_head + head_idx) * n_q + q_idx)
                                 * n_splits + split_idx);
    global float  * rec       = partial_void + record_idx * record_stride;
    global float4 * rec_o     = (global float4 *) (rec + 2);

    if (kv_start >= kv_end) {
        // Empty split: leave sentinel partial for merge.
        if (tid == 0) {
            rec[0] = FA_M_INIT;
            rec[1] = 0.0f;
        }
        return;
    }

    const global char * q_base = (const global char *) q_void + q_offset;
    const global char * k_base = (const global char *) k_void + k_offset;
    const global char * v_base = (const global char *) v_void + v_offset;

    const global char * mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx  = head_idx  % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char *) mask_void + mask_offset +
                    mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2 +
                    (ulong) q_idx * mask_nb1;
    }

    // share Q via local memory (n_q=1 per split -> uniform across WG).
    __local ACC_TYPE4 q_shared[DK_VEC];
    const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + (ulong) q_idx * q_nb1;
    const global Q_DATA_TYPE4 * q_ptr = (const global Q_DATA_TYPE4 *) (q_base + q_row_offset);
    for (int i = tid; i < DK_VEC; i += Q1_WG_SIZE) {
        q_shared[i] = CONVERT_Q_ACC4(q_ptr[i]);
    }
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);

    const float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    // pass 1a — split-local max.
    ACC_TYPE m_i = FA_M_INIT;
    for (int k_idx = kv_start + tid; k_idx < kv_end; k_idx += Q1_WG_SIZE) {
        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const global KV_DATA_TYPE4 * k_ptr = (const global KV_DATA_TYPE4 *) (k_base + k_row_offset);
        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
        #pragma unroll
        for (int k = 0; k < DK_VEC; ++k) {
            dot_acc = mad(q_shared[k], CONVERT_KV_ACC4(k_ptr[k]), dot_acc);
        }
        ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
        if (mask_base != NULL) {
            const global MASK_DATA_TYPE * mask_ptr = (const global MASK_DATA_TYPE *) (mask_base);
            score += slope * (ACC_TYPE) mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }
        m_i = max(m_i, score);
    }

    const ACC_TYPE m_c = sub_group_reduce_max(m_i);

    // pass 1b — softmax-weighted V accumulate.
    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);
    ACC_TYPE l_i = 0.0f;

    for (int k_idx = kv_start + tid; k_idx < kv_end; k_idx += Q1_WG_SIZE) {
        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
        const global KV_DATA_TYPE4 * k_ptr = (const global KV_DATA_TYPE4 *) (k_base + k_row_offset);
        const global KV_DATA_TYPE4 * v_ptr = (const global KV_DATA_TYPE4 *) (v_base + v_row_offset);
        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
        #pragma unroll
        for (int k = 0; k < DK_VEC; ++k) {
            dot_acc = mad(q_shared[k], CONVERT_KV_ACC4(k_ptr[k]), dot_acc);
        }
        ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
        if (mask_base != NULL) {
            const global MASK_DATA_TYPE * mask_ptr = (const global MASK_DATA_TYPE *) (mask_base);
            score += slope * (ACC_TYPE) mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }
        const ACC_TYPE p = exp(score - m_c);
        l_i += p;
        #pragma unroll
        for (int i = 0; i < DV_VEC; ++i) {
            o_acc[i] = mad(p, CONVERT_KV_ACC4(v_ptr[i]), o_acc[i]);
        }
    }

    __local ACC_TYPE4 local_o[Q1_WG_SIZE];
    const ACC_TYPE l_c = sub_group_reduce_add(l_i);

    if (tid == 0) {
        rec[0] = (float) m_c;
        rec[1] = (float) l_c;
    }
    for (int i = 0; i < DV_VEC; ++i) {
        local_o[tid] = o_acc[i];
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) local_o[tid] += local_o[tid + s];
            sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (tid == 0) {
            rec_o[i] = local_o[0];
        }
    }
}

// FD Pass 2: merge per-split partials into final O
// empty splits drop via exp(-INF)=0.
__kernel void flash_attn_f32_merge(
    const global float * partial_void,
    global void * o_void,
    const ulong o_offset,
    const int n_head,
    const int n_splits,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const global void * sinks_void,
    const ulong sinks_offset,
    const int n_q
) {
    const int lane           = get_local_id(0);  // 0..DV_VEC-1
    const int head_batch_idx = get_global_id(1);
    const int q_idx          = get_global_id(2);
    const int batch_idx      = head_batch_idx / n_head;
    const int head_idx       = head_batch_idx % n_head;

    const ulong record_stride = (ulong) FA_PARTIAL_FLOATS;
    const ulong record_idx_0  = (((ulong) batch_idx * n_head + head_idx) * n_q + q_idx) * n_splits;
    const global float * rec0 = partial_void + record_idx_0 * record_stride;

    __local ACC_TYPE m_final_shared;
    __local ACC_TYPE l_final_shared;
    if (lane == 0) {
        ACC_TYPE m = FA_M_INIT;
        for (int c = 0; c < n_splits; ++c) {
            const ACC_TYPE m_c = rec0[c * record_stride + 0];
            m = max(m, m_c);
        }
        ACC_TYPE m_sink = 0.0f;
        bool has_sink = false;
        if (sinks_void != NULL) {
            const global ACC_TYPE * sinks_ptr =
                (const global ACC_TYPE *) ((const global char *) sinks_void + sinks_offset);
            m_sink = sinks_ptr[head_idx];
            has_sink = true;
            m = max(m, m_sink);
        }
        ACC_TYPE l = 0.0f;
        for (int c = 0; c < n_splits; ++c) {
            const ACC_TYPE m_c = rec0[c * record_stride + 0];
            const ACC_TYPE l_c = rec0[c * record_stride + 1];
            if (m_c > FA_M_INIT) {
                l += l_c * exp(m_c - m);
            }
        }
        if (has_sink) {
            l += exp(m_sink - m);
        }
        m_final_shared = m;
        l_final_shared = l;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const ACC_TYPE m_final = m_final_shared;
    const ACC_TYPE l_final = l_final_shared;
    const ACC_TYPE l_inv   = (l_final > 0.0f) ? (1.0f / l_final) : 0.0f;

    ACC_TYPE4 o = (ACC_TYPE4)(0.0f);
    for (int c = 0; c < n_splits; ++c) {
        const global float * rec_c   = rec0 + c * record_stride;
        const ACC_TYPE       m_c     = rec_c[0];
        if (m_c <= FA_M_INIT) continue;
        const global float4 * rec_oc = (const global float4 *) (rec_c + 2);
        const ACC_TYPE scale_c = exp(m_c - m_final);
        o = mad((ACC_TYPE4)(scale_c), rec_oc[lane], o);
    }
    o = o * l_inv;

    const ulong o_row_offset = (ulong) batch_idx * o_nb3 + (ulong) q_idx * o_nb2 + (ulong) head_idx * o_nb1;
    global O_DATA_TYPE4 * o_row = (global O_DATA_TYPE4 *) ((global char *) o_void + o_offset + o_row_offset);
    o_row[lane] = CONVERT_O_DATA4(o);
}
#endif  // !FA_MQ_ONLY (q1_split + merge)
#endif  // !FA_PREFILL_ONLY (decode kernels)
