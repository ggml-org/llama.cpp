#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define ACC_TYPE float
#define ACC_TYPE4 float4
#define DATA_TYPE half
#define DATA_TYPE4 half4
#define CONVERT_ACC4(x) convert_float4(x)
#define CONVERT_DATA4(x) convert_half4(x)

#define DK_VEC (DK/4)
#define DV_VEC (DV/4)
#define WG_SIZE (BLOCK_M)
#define Q1_WG_SIZE 64

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
__kernel void flash_attn_f16(
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
        #pragma unroll
        for (int i = 0; i < DK_VEC; ++i) {
            q_priv[i] = CONVERT_ACC4(q_ptr[i]);
        }
    }

    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) {
        o_acc[i] = (ACC_TYPE4)(0.0f);
    }
    ACC_TYPE m_i = -INFINITY;
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

        for (int j = 0; j < BLOCK_N; j += 2) {
            const int k_row0 = k_start + j;
            const int k_row1 = k_start + j + 1;

            ACC_TYPE4 dot_acc0 = (ACC_TYPE4)(0.0f);
            ACC_TYPE4 dot_acc1 = (ACC_TYPE4)(0.0f);
            #pragma unroll
            for (int k = 0; k < DK_VEC; k++) {
                dot_acc0 = mad(q_priv[k], CONVERT_ACC4(l_k[j][k]), dot_acc0);
                dot_acc1 = mad(q_priv[k], CONVERT_ACC4(l_k[j+1][k]), dot_acc1);
            }
            ACC_TYPE score0 = (dot_acc0.s0 + dot_acc0.s1 + dot_acc0.s2 + dot_acc0.s3) * scale;
            ACC_TYPE score1 = (dot_acc1.s0 + dot_acc1.s1 + dot_acc1.s2 + dot_acc1.s3) * scale;

            if (is_causal) {
                if (k_row0 > (n_kv - n_q + my_query_row)) score0 = -INFINITY;
                if (k_row1 > (n_kv - n_q + my_query_row)) score1 = -INFINITY;
            }

            if (k_row0 >= n_kv) score0 = -INFINITY;
            if (k_row1 >= n_kv) score1 = -INFINITY;

            if (mask_base != NULL) {
                const global DATA_TYPE* mask_ptr = (const global DATA_TYPE*)(mask_base + my_query_row * mask_nb1);
                if (k_row0 < n_kv) score0 += slope * (ACC_TYPE)mask_ptr[k_row0];
                if (k_row1 < n_kv) score1 += slope * (ACC_TYPE)mask_ptr[k_row1];
            }

            if (logit_softcap > 0.0f) {
                score0 = logit_softcap * tanh(score0 / logit_softcap);
                score1 = logit_softcap * tanh(score1 / logit_softcap);
            }

            const ACC_TYPE m_new = max(m_i, max(score0, score1));
            const ACC_TYPE p0 = exp(score0 - m_new);
            const ACC_TYPE p1 = exp(score1 - m_new);
            const ACC_TYPE scale_prev = exp(m_i - m_new);

            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_acc[i] = o_acc[i] * scale_prev + p0 * CONVERT_ACC4(l_v[j][i]) + p1 * CONVERT_ACC4(l_v[j+1][i]);
            }
            l_i = l_i * scale_prev + p0 + p1;
            m_i = m_new;
        }
    }

    if (my_query_row < n_q) {
        if (sinks_void != NULL) {
            const global ACC_TYPE* sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
            const ACC_TYPE m_sink = sinks_ptr[head_idx];
            const ACC_TYPE m_final = max(m_i, m_sink);

            const ACC_TYPE scale_o = exp(m_i - m_final);
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_acc[i] *= scale_o;
            }

            l_i = l_i * exp(m_i - m_final) + exp(m_sink - m_final);
        }

        const ulong o_row_offset = batch_idx * o_nb3 + my_query_row * o_nb2 + head_idx * o_nb1;
        global DATA_TYPE4 *o_row = (global DATA_TYPE4 *)(o_base + o_row_offset);
        if (l_i > 0.0f) {
            const ACC_TYPE l_inv = 1.0f / l_i;
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_row[i] = CONVERT_DATA4(o_acc[i] * l_inv);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_row[i] = (DATA_TYPE4)(0.0f);
            }
        }
    }
}

__kernel void flash_attn_f16_q1(
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
    #pragma unroll
    for (int i = 0; i < DK_VEC; ++i) {
        q_priv[i] = CONVERT_ACC4(q_ptr[i]);
    }

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    const global ACC_TYPE* sinks_ptr = NULL;
    if (sinks_void != NULL) {
        sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
    }

    ACC_TYPE m_i = (sinks_ptr != NULL) ? sinks_ptr[head_idx] : -INFINITY;
    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const global DATA_TYPE4* k_ptr = (const global DATA_TYPE4*)(k_base + k_row_offset);
        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
        #pragma unroll
        for (int k = 0; k < DK_VEC; k++) {
            dot_acc = mad(q_priv[k], CONVERT_ACC4(k_ptr[k]), dot_acc);
        }
        ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
        if (mask_base != NULL) {
            const global DATA_TYPE* mask_ptr = (const global DATA_TYPE*)(mask_base);
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
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_m[tid] = max(local_m[tid], local_m[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const ACC_TYPE m_final = local_m[0];

    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);
    ACC_TYPE l_i = 0.0f;

    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
        const global DATA_TYPE4* k_ptr = (const global DATA_TYPE4*)(k_base + k_row_offset);
        const global DATA_TYPE4* v_ptr = (const global DATA_TYPE4*)(v_base + v_row_offset);
        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
        #pragma unroll
        for (int k = 0; k < DK_VEC; k++) {
            dot_acc = mad(q_priv[k], CONVERT_ACC4(k_ptr[k]), dot_acc);
        }
        ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
        if (mask_base != NULL) {
            const global DATA_TYPE* mask_ptr = (const global DATA_TYPE*)(mask_base);
            score += slope * (ACC_TYPE)mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }
        const ACC_TYPE p = exp(score - m_final);
        l_i += p;
        #pragma unroll
        for (int i = 0; i < DV_VEC; i++) {
            o_acc[i] = mad(p, CONVERT_ACC4(v_ptr[i]), o_acc[i]);
        }
    }

    __local ACC_TYPE local_l[Q1_WG_SIZE];
    __local ACC_TYPE4 local_o_comp[Q1_WG_SIZE];
    local_l[tid] = l_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
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
            #pragma unroll
            for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
                if (tid < s) local_o_comp[tid] += local_o_comp[tid + s];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (tid == 0) {
                o_row[i] = CONVERT_DATA4(local_o_comp[0] * l_inv);
            }
        }
    } else if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < DV_VEC; ++i) o_row[i] = (DATA_TYPE4)(0.0f);
    }
}
