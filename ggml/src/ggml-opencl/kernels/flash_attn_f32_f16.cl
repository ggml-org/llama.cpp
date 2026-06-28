#pragma OPENCL EXTENSION cl_khr_fp16 : enable

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
#define Q1_WG_SIZE 64

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
__kernel void flash_attn_f32_f16(
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
    const global char* k_base = (const global char*)k_void + k_offset;
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
        const global char* k_tile_base = use_kv_pad ? (const global char*) k_pad_void : k_base;
        const global char* v_tile_base = use_kv_pad ? (const global char*) v_pad_void : v_base;

        for (int i = tid; i < BLOCK_N * DK_VEC; i += WG_SIZE) {
            const int row = i / DK_VEC;
            const int col = i % DK_VEC;
            const int k_row_idx = k_tile_start + row;
            if (use_kv_pad || k_row_idx < n_kv) {
                const ulong k_row_offset = batch_idx * k_tile_nb3 + head_kv_idx * k_tile_nb2 + k_row_idx * k_nb1;
                l_k[row][col] = ((__global KV_DATA_TYPE4*)(k_tile_base + k_row_offset))[col];
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
    const global char* k_base = (const global char*)k_void + k_offset;
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
    barrier(CLK_LOCAL_MEM_FENCE);

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

    __local ACC_TYPE local_m[Q1_WG_SIZE];
    local_m[tid] = m_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    FA_UNROLL
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_m[tid] = max(local_m[tid], local_m[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const ACC_TYPE m_final = local_m[0];

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

    __local ACC_TYPE local_l[Q1_WG_SIZE];
    __local ACC_TYPE4 local_o_comp[Q1_WG_SIZE];
    local_l[tid] = l_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    FA_UNROLL
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_l[tid] += local_l[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const ulong o_row_offset = batch_idx * o_nb3 + head_idx * o_nb1;
    global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
    ACC_TYPE l_final = local_l[0];

    if (sinks_ptr != NULL) {
        l_final += exp(sinks_ptr[head_idx] - m_final);
    }

    if (l_final > 0.0f) {
        const ACC_TYPE l_inv = 1.0f / l_final;
        for (int i = 0; i < DV_VEC; i++) {
            local_o_comp[tid] = o_acc[i];
            barrier(CLK_LOCAL_MEM_FENCE);
            FA_UNROLL
            for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
                if (tid < s) local_o_comp[tid] += local_o_comp[tid + s];
                barrier(CLK_LOCAL_MEM_FENCE);
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

// Flash-decoding split pass. gid(2) = q_idx * n_splits + split_idx.
// Partial record per split: [m, l, O[DV]]. Merge kernel applies sink + norm.
#define FA_PARTIAL_FLOATS (2 + DV)

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

    // Share Q via local memory (n_q=1 per split -> uniform across WG).
    __local ACC_TYPE4 q_shared[DK_VEC];
    const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + (ulong) q_idx * q_nb1;
    const global Q_DATA_TYPE4 * q_ptr = (const global Q_DATA_TYPE4 *) (q_base + q_row_offset);
    for (int i = tid; i < DK_VEC; i += Q1_WG_SIZE) {
        q_shared[i] = CONVERT_Q_ACC4(q_ptr[i]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    // Pass 1a — split-local max.
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

    __local ACC_TYPE local_m[Q1_WG_SIZE];
    local_m[tid] = m_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_m[tid] = max(local_m[tid], local_m[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const ACC_TYPE m_c = local_m[0];

    // Pass 1b — softmax-weighted V accumulate.
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

    __local ACC_TYPE  local_l[Q1_WG_SIZE];
    __local ACC_TYPE4 local_o[Q1_WG_SIZE];
    local_l[tid] = l_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_l[tid] += local_l[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const ACC_TYPE l_c = local_l[0];

    if (tid == 0) {
        rec[0] = (float) m_c;
        rec[1] = (float) l_c;
    }
    for (int i = 0; i < DV_VEC; ++i) {
        local_o[tid] = o_acc[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) local_o[tid] += local_o[tid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (tid == 0) {
            rec_o[i] = local_o[0];
        }
    }
}

// FD Pass 2: merge per-split partials into final O. Empty splits drop via exp(-INF)=0.
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
