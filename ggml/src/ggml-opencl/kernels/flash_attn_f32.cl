#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define ACC_TYPE float
#define ACC_TYPE4 float4
#define DATA_TYPE float
#define DATA_TYPE4 float4
#define MASK_DATA_TYPE half
#define CONVERT_ACC4(x) (x)
#define CONVERT_DATA4(x) (x)

#define DK_VEC (DK/4)
#define DV_VEC (DV/4)
#define WG_SIZE (BLOCK_M)
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
__kernel void flash_attn_f32(
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
    const int block_q_idx = get_group_id(0);
    const int head_batch_idx = get_global_id(1);

    const int my_query_row = block_q_idx * BLOCK_M + tid;

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

    ACC_TYPE4 q_priv[DK_VEC];
    if (my_query_row < n_q) {
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + my_query_row * q_nb1;
        const global DATA_TYPE4* q_ptr = (const global DATA_TYPE4*)(q_base + q_row_offset);
        FA_UNROLL
        for (int i = 0; i < DK_VEC; ++i) {
            q_priv[i] = CONVERT_ACC4(q_ptr[i]);
        }
    }

    ACC_TYPE4 o_acc[DV_VEC];
    FA_UNROLL
    for (int i = 0; i < DV_VEC; ++i) {
        o_acc[i] = (ACC_TYPE4)(0.0f);
    }
    ACC_TYPE m_i = FA_M_INIT;
    ACC_TYPE l_i = 0.0f;

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    __local DATA_TYPE4 l_k[BLOCK_N][DK_VEC];
    __local DATA_TYPE4 l_v[BLOCK_N][DV_VEC];

    for (int k_start = 0; k_start < n_kv; k_start += BLOCK_N) {
        for (int i = tid; i < BLOCK_N * DK_VEC; i += WG_SIZE) {
            const int row = i / DK_VEC;
            const int col = i % DK_VEC;
            const int k_row_idx = k_start + row;
            if (k_row_idx < n_kv) {
                const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_row_idx * k_nb1;
                l_k[row][col] = ((__global DATA_TYPE4*)(k_base + k_row_offset))[col];
            }
        }
        for (int i = tid; i < BLOCK_N * DV_VEC; i += WG_SIZE) {
            const int row = i / DV_VEC;
            const int col = i % DV_VEC;
            const int v_row_idx = k_start + row;
            if (v_row_idx < n_kv) {
                const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + v_row_idx * v_nb1;
                l_v[row][col] = ((__global DATA_TYPE4*)(v_base + v_row_offset))[col];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (my_query_row >= n_q) {
            continue;
        }

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
                dot_acc0 = mad(qk, CONVERT_ACC4(l_k[j][k]),   dot_acc0);
                dot_acc1 = mad(qk, CONVERT_ACC4(l_k[j+1][k]), dot_acc1);
                dot_acc2 = mad(qk, CONVERT_ACC4(l_k[j+2][k]), dot_acc2);
                dot_acc3 = mad(qk, CONVERT_ACC4(l_k[j+3][k]), dot_acc3);
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

            if (mask_base != NULL) {
                const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base + my_query_row * mask_nb1);
                if (k_row0 < n_kv) s0 += slope * (ACC_TYPE)mask_ptr[k_row0];
                if (k_row1 < n_kv) s1 += slope * (ACC_TYPE)mask_ptr[k_row1];
                if (k_row2 < n_kv) s2 += slope * (ACC_TYPE)mask_ptr[k_row2];
                if (k_row3 < n_kv) s3 += slope * (ACC_TYPE)mask_ptr[k_row3];
            }

            if (logit_softcap > 0.0f) {
                s0 = logit_softcap * tanh(s0 / logit_softcap);
                s1 = logit_softcap * tanh(s1 / logit_softcap);
                s2 = logit_softcap * tanh(s2 / logit_softcap);
                s3 = logit_softcap * tanh(s3 / logit_softcap);
            }

            const ACC_TYPE m_new      = max(m_i, max(max(s0, s1), max(s2, s3)));
            const ACC_TYPE scale_prev = native_exp(m_i - m_new);
            const ACC_TYPE p0         = native_exp(s0 - m_new);
            const ACC_TYPE p1         = native_exp(s1 - m_new);
            const ACC_TYPE p2         = native_exp(s2 - m_new);
            const ACC_TYPE p3         = native_exp(s3 - m_new);

            FA_UNROLL
            for (int i = 0; i < DV_VEC; ++i) {
                o_acc[i] = mad(p3, CONVERT_ACC4(l_v[j+3][i]),
                           mad(p2, CONVERT_ACC4(l_v[j+2][i]),
                           mad(p1, CONVERT_ACC4(l_v[j+1][i]),
                           mad(p0, CONVERT_ACC4(l_v[j][i]),
                           o_acc[i] * scale_prev))));
            }
            l_i = l_i * scale_prev + p0 + p1 + p2 + p3;
            m_i = m_new;
        }
    }

    if (my_query_row < n_q) {
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
        global DATA_TYPE4 *o_row = (global DATA_TYPE4 *)(o_base + o_row_offset);
        if (l_i > 0.0f) {
            const ACC_TYPE l_inv = 1.0f / l_i;
            FA_UNROLL
            for (int i = 0; i < DV_VEC; ++i) {
                o_row[i] = CONVERT_DATA4(o_acc[i] * l_inv);
            }
        } else {
            FA_UNROLL
            for (int i = 0; i < DV_VEC; ++i) {
                o_row[i] = (DATA_TYPE4)(0.0f);
            }
        }
    }
}

__kernel void flash_attn_f32_q1(
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

    ACC_TYPE4 q_priv[DK_VEC];
    const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2;
    const global DATA_TYPE4* q_ptr = (const global DATA_TYPE4*)(q_base + q_row_offset);
    FA_UNROLL
    for (int i = 0; i < DK_VEC; ++i) {
        q_priv[i] = CONVERT_ACC4(q_ptr[i]);
    }

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    const global ACC_TYPE* sinks_ptr = NULL;
    if (sinks_void != NULL) {
        sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
    }

    ACC_TYPE m_i = (sinks_ptr != NULL) ? sinks_ptr[head_idx] : FA_M_INIT;
    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const global DATA_TYPE4* k_ptr = (const global DATA_TYPE4*)(k_base + k_row_offset);
        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
        FA_UNROLL
        for (int k = 0; k < DK_VEC; k++) {
            dot_acc = mad(q_priv[k], CONVERT_ACC4(k_ptr[k]), dot_acc);
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
        const global DATA_TYPE4* k_ptr = (const global DATA_TYPE4*)(k_base + k_row_offset);
        const global DATA_TYPE4* v_ptr = (const global DATA_TYPE4*)(v_base + v_row_offset);
        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
        FA_UNROLL
        for (int k = 0; k < DK_VEC; k++) {
            dot_acc = mad(q_priv[k], CONVERT_ACC4(k_ptr[k]), dot_acc);
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
            o_acc[i] = mad(p, CONVERT_ACC4(v_ptr[i]), o_acc[i]);
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
    global DATA_TYPE4 *o_row = (global DATA_TYPE4 *)(o_base + o_row_offset);
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
                o_row[i] = CONVERT_DATA4(local_o_comp[0] * l_inv);
            }
        }
    } else if (tid == 0) {
        FA_UNROLL
        for (int i = 0; i < DV_VEC; ++i) o_row[i] = (DATA_TYPE4)(0.0f);
    }
}
