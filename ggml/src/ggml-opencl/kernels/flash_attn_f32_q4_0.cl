#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#define FA_HAVE_INT_DOT 1
#endif

#ifdef cl_khr_subgroup_shuffle
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable
#define HAS_SUBGROUP_SHUFFLE 1
#elif defined(cl_qcom_subgroup_shuffle)
#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
#define HAS_SUBGROUP_SHUFFLE 1
#endif

// Flash attention: Q=f32, K=q4_0, V=q4_0.
// Block = half d + uchar qs[16]; qs[j] low/high nibble -> elem j / j+16.
// Dequant: val[i] = d * (nibble_i - 8). dp4a path runs on raw 0..15 nibbles
// and applies the -8*sum(q) correction once per block (needs Q q_sum).

#define ACC_TYPE float
#define ACC_TYPE4 float4
#define Q_DATA_TYPE4 float4
#define O_DATA_TYPE4 float4
#define MASK_DATA_TYPE half
#define CONVERT_Q_ACC4(x) (x)
#define CONVERT_O_DATA4(x) (x)

#define DK_VEC (DK/4)
#define DV_VEC (DV/4)
#define Q1_WG_SIZE 64

// The kernels are built with -cl-finite-math-only. On some older Adreno GPUs,
// infinite operand can cause undefined behavior and miscompilation for exp.
// Therefore, a large negative value is used instead.
#define FA_M_INIT (-3.0e38f)

#define QK4_0 32
#define Q4_0_BLOCK_SIZE 18

#define DK_Q4_BLOCKS (DK / QK4_0)
#define DV_Q4_BLOCKS (DV / QK4_0)

inline float dot_q4_0_f32(const global char * block_ptr, ACC_TYPE4 * q_slice) {
    float d = vload_half(0, (const global half *)block_ptr);
    const global uchar * qs = (const global uchar *)(block_ptr + 2);

    float sum = 0.0f;
    // Low nibbles -> elems 0..15.
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        float4 nv = (float4)((float)(int)(qs[g*4 + 0] & 0x0F) - 8.0f,
                             (float)(int)(qs[g*4 + 1] & 0x0F) - 8.0f,
                             (float)(int)(qs[g*4 + 2] & 0x0F) - 8.0f,
                             (float)(int)(qs[g*4 + 3] & 0x0F) - 8.0f);
        sum += dot(q_slice[g], nv);
    }
    // High nibbles -> elems 16..31.
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        float4 nv = (float4)((float)(int)(qs[g*4 + 0] >> 4) - 8.0f,
                             (float)(int)(qs[g*4 + 1] >> 4) - 8.0f,
                             (float)(int)(qs[g*4 + 2] >> 4) - 8.0f,
                             (float)(int)(qs[g*4 + 3] >> 4) - 8.0f);
        sum += dot(q_slice[4 + g], nv);
    }
    return sum * d;
}

#ifdef FA_HAVE_INT_DOT
inline uint pack_i8x4(char a, char b, char c, char d) {
    return ((uint)(uchar)a)       |
           ((uint)(uchar)b) <<  8  |
           ((uint)(uchar)c) << 16  |
           ((uint)(uchar)d) << 24;
}

// Returns (qd, q_sum); q_sum feeds the -8*sum(q) bias correction.
typedef struct {
    float qd;
    int   q_sum;
} q4_q_block_info;

inline q4_q_block_info quant_q_block_int8_packed_q4(const ACC_TYPE4 * q_block,
                                                    uint *            out_packed) {
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float4 av = fabs(q_block[i]);
        amax = fmax(amax, fmax(fmax(av.s0, av.s1), fmax(av.s2, av.s3)));
    }
    float qd  = amax / 127.0f;
    float qid = (amax > 0.0f) ? 127.0f / amax : 0.0f;

    int q_sum = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float4 v = q_block[i] * qid;
        char a = (char)((int)round(v.s0));
        char b = (char)((int)round(v.s1));
        char c = (char)((int)round(v.s2));
        char d = (char)((int)round(v.s3));
        out_packed[i] = pack_i8x4(a, b, c, d);
        q_sum += (int)a + (int)b + (int)c + (int)d;
    }
    q4_q_block_info info = { qd, q_sum };
    return info;
}

// k_packed[0..3] = low nibbles (Q elems 0..15), k_packed[4..7] = high (16..31).
inline void pack_q4_0_nibbles(const global uchar * qs, uint * k_packed) {
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        uchar b0 = qs[g*4 + 0];
        uchar b1 = qs[g*4 + 1];
        uchar b2 = qs[g*4 + 2];
        uchar b3 = qs[g*4 + 3];
        k_packed[g] =
              ((uint)(b0 & 0x0F))       |
              ((uint)(b1 & 0x0F)) <<  8 |
              ((uint)(b2 & 0x0F)) << 16 |
              ((uint)(b3 & 0x0F)) << 24;
        k_packed[4 + g] =
              ((uint)(b0 >> 4))         |
              ((uint)(b1 >> 4)) <<  8   |
              ((uint)(b2 >> 4)) << 16   |
              ((uint)(b3 >> 4)) << 24;
    }
}

inline float dot_q4_0_int(const global char * k_block_ptr,
                          const uint *        q_packed,
                          float               q_d,
                          int                 q_sum) {
    float kd = vload_half(0, (const global half *)k_block_ptr);
    const global uchar * k_qs = (const global uchar *)(k_block_ptr + 2);

    uint k_packed[8];
    pack_q4_0_nibbles(k_qs, k_packed);

    int sum = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        sum = dot_acc_sat_4x8packed_ss_int(q_packed[i], k_packed[i], sum);
    }
    // Correct raw-nibble sum: (nibble - 8) bias -> subtract 8 * q_sum.
    return (float)(sum - 8 * q_sum) * q_d * kd;
}
#endif // FA_HAVE_INT_DOT

inline void dequant_q4_0_f32(const global char * block_ptr, ACC_TYPE4 * out) {
    float d = vload_half(0, (const global half *)block_ptr);
    const global uchar * qs = (const global uchar *)(block_ptr + 2);

    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        out[g] = d * (float4)((float)(int)(qs[g*4 + 0] & 0x0F) - 8.0f,
                              (float)(int)(qs[g*4 + 1] & 0x0F) - 8.0f,
                              (float)(int)(qs[g*4 + 2] & 0x0F) - 8.0f,
                              (float)(int)(qs[g*4 + 3] & 0x0F) - 8.0f);
    }
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        out[4 + g] = d * (float4)((float)(int)(qs[g*4 + 0] >> 4) - 8.0f,
                                  (float)(int)(qs[g*4 + 1] >> 4) - 8.0f,
                                  (float)(int)(qs[g*4 + 2] >> 4) - 8.0f,
                                  (float)(int)(qs[g*4 + 3] >> 4) - 8.0f);
    }
}

// max_bias<=0 returns 1.0 so score += 1.0 * mask[k] stays a no-op multiplier.
inline float get_alibi_slope(float max_bias, int head_idx, int n_head_log2, float m0, float m1) {
    if (max_bias <= 0.0f) return 1.0f;
    float base = (head_idx < n_head_log2) ? m0 : m1;
    int   exph = (head_idx < n_head_log2) ? (head_idx + 1) : (2*(head_idx - n_head_log2) + 1);
    return pow(base, (float)exph);
}

// q1 decode: one query row per WG, threads sweep KV positions.
__kernel void flash_attn_f32_q4_0_q1(
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
    const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
    #pragma unroll
    for (int i = 0; i < DK_VEC; ++i) {
        q_priv[i] = CONVERT_Q_ACC4(q_ptr[i]);
    }

#ifdef FA_HAVE_INT_DOT
    // Quantise Q once per thread: 8 uints + qd + q_sum per block.
    uint  q_packed[DK_Q4_BLOCKS * 8];
    float q_d_scale[DK_Q4_BLOCKS];
    int   q_sum_arr[DK_Q4_BLOCKS];
    #pragma unroll
    for (int b = 0; b < DK_Q4_BLOCKS; ++b) {
        q4_q_block_info info = quant_q_block_int8_packed_q4(&q_priv[b * 8], &q_packed[b * 8]);
        q_d_scale[b] = info.qd;
        q_sum_arr[b] = info.q_sum;
    }
#endif

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    const global ACC_TYPE* sinks_ptr = NULL;
    if (sinks_void != NULL) {
        sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
    }

    // One-pass online softmax (FA-2): single sweep over kv positions,
    // updating per-thread (m_i, l_i, o_acc) per K. Eliminates the second
    // K read of the original two-pass implementation.
    ACC_TYPE m_i = (sinks_ptr != NULL) ? sinks_ptr[head_idx] : FA_M_INIT;
    ACC_TYPE l_i = 0.0f;
    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);

    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
        const global char* k_row = k_base + batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const global char* v_row = v_base + batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;

        ACC_TYPE score = 0.0f;
        #pragma unroll
        for (int b = 0; b < DK_Q4_BLOCKS; b++) {
#ifdef FA_HAVE_INT_DOT
            score += dot_q4_0_int(k_row + b * Q4_0_BLOCK_SIZE,
                                   &q_packed[b * 8], q_d_scale[b], q_sum_arr[b]);
#else
            score += dot_q4_0_f32(k_row + b * Q4_0_BLOCK_SIZE, &q_priv[b * 8]);
#endif
        }
        score *= scale;

        if (mask_base != NULL) {
            const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base);
            score += slope * (ACC_TYPE)mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }

        // Online softmax step.
        const ACC_TYPE m_new = max(m_i, score);
        const ACC_TYPE alpha = exp(m_i  - m_new);
        const ACC_TYPE p     = exp(score - m_new);

        l_i = alpha * l_i + p;
        #pragma unroll
        for (int i = 0; i < DV_VEC; ++i) o_acc[i] *= alpha;

        #pragma unroll
        for (int b = 0; b < DV_Q4_BLOCKS; b++) {
            ACC_TYPE4 v_dequant[8];
            dequant_q4_0_f32(v_row + b * Q4_0_BLOCK_SIZE, v_dequant);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                o_acc[b * 8 + i] = mad(p, v_dequant[i], o_acc[b * 8 + i]);
            }
        }

        m_i = m_new;
    }

    // Cross-thread reduce: max(m_i) -> m_final, rescale per-thread l_i and
    // o_acc by alpha = exp(m_i_thread - m_final) before sum-reduce.
    __local ACC_TYPE local_m[Q1_WG_SIZE];
    local_m[tid] = m_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_m[tid] = max(local_m[tid], local_m[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const ACC_TYPE m_final = local_m[0];

    const ACC_TYPE alpha_final = exp(m_i - m_final);
    l_i *= alpha_final;
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] *= alpha_final;

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
            #pragma unroll
            for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
                if (tid < s) local_o_comp[tid] += local_o_comp[tid + s];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (tid == 0) {
                o_row[i] = CONVERT_O_DATA4(local_o_comp[0] * l_inv);
            }
        }
    } else if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < DV_VEC; ++i) o_row[i] = (O_DATA_TYPE4)(0.0f);
    }
}

// Flash-decoding split pass for q4_0 KV. Merge kernel is type-agnostic and
// shared with the f16/q8_0 FA kernels.
#define FA_PARTIAL_FLOATS (2 + DV)

__kernel void flash_attn_f32_q4_0_q1_split(
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
    const int tid            = get_local_id(0);
    const int head_batch_idx = get_global_id(1);
    const int split_q_idx    = get_global_id(2);
    const int split_idx      = split_q_idx % n_splits;
    const int q_idx          = split_q_idx / n_splits;
    const int batch_idx      = head_batch_idx / n_head;
    const int head_idx       = head_batch_idx % n_head;
    const int gqa_ratio      = n_head / n_head_kv;
    const int head_kv_idx    = head_idx / gqa_ratio;

    const int kv_start = split_idx * kv_per_split;
    const int kv_end   = min(kv_start + kv_per_split, n_kv);

    const ulong record_stride = (ulong) FA_PARTIAL_FLOATS;
    const ulong record_idx    = ((((ulong) batch_idx * n_head + head_idx) * n_q + q_idx)
                                 * n_splits + split_idx);
    global float  * rec       = partial_void + record_idx * record_stride;
    global float4 * rec_o     = (global float4 *) (rec + 2);

    if (kv_start >= kv_end) {
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

    ACC_TYPE4 q_priv[DK_VEC];
    const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + (ulong) q_idx * q_nb1;
    const global Q_DATA_TYPE4 * q_ptr = (const global Q_DATA_TYPE4 *) (q_base + q_row_offset);
    #pragma unroll
    for (int i = 0; i < DK_VEC; ++i) {
        q_priv[i] = CONVERT_Q_ACC4(q_ptr[i]);
    }

#ifdef FA_HAVE_INT_DOT
    uint  q_packed[DK_Q4_BLOCKS * 8];
    float q_d_scale[DK_Q4_BLOCKS];
    int   q_sum_arr[DK_Q4_BLOCKS];
    #pragma unroll
    for (int b = 0; b < DK_Q4_BLOCKS; ++b) {
        q4_q_block_info info = quant_q_block_int8_packed_q4(&q_priv[b * 8], &q_packed[b * 8]);
        q_d_scale[b] = info.qd;
        q_sum_arr[b] = info.q_sum;
    }
#endif

    const float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    // One-pass online softmax (FA-2): single sweep over the split's K range.
    ACC_TYPE m_i = FA_M_INIT;
    ACC_TYPE l_i = 0.0f;
    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);

    for (int k_idx = kv_start + tid; k_idx < kv_end; k_idx += Q1_WG_SIZE) {
        const global char * k_row = k_base + batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const global char * v_row = v_base + batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
        ACC_TYPE score = 0.0f;
        #pragma unroll
        for (int b = 0; b < DK_Q4_BLOCKS; ++b) {
#ifdef FA_HAVE_INT_DOT
            score += dot_q4_0_int(k_row + b * Q4_0_BLOCK_SIZE,
                                   &q_packed[b * 8], q_d_scale[b], q_sum_arr[b]);
#else
            score += dot_q4_0_f32(k_row + b * Q4_0_BLOCK_SIZE, &q_priv[b * 8]);
#endif
        }
        score *= scale;
        if (mask_base != NULL) {
            const global MASK_DATA_TYPE * mask_ptr = (const global MASK_DATA_TYPE *) (mask_base);
            score += slope * (ACC_TYPE) mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }

        // Online softmax step.
        const ACC_TYPE m_new = max(m_i, score);
        const ACC_TYPE alpha = exp(m_i  - m_new);
        const ACC_TYPE p     = exp(score - m_new);

        l_i = alpha * l_i + p;
        #pragma unroll
        for (int i = 0; i < DV_VEC; ++i) o_acc[i] *= alpha;

        #pragma unroll
        for (int b = 0; b < DV_Q4_BLOCKS; ++b) {
            ACC_TYPE4 v_dequant[8];
            dequant_q4_0_f32(v_row + b * Q4_0_BLOCK_SIZE, v_dequant);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                o_acc[b * 8 + i] = mad(p, v_dequant[i], o_acc[b * 8 + i]);
            }
        }

        m_i = m_new;
    }

    // Cross-thread reduce: max(m_i) -> m_c, rescale per-thread l_i and o_acc
    // by alpha = exp(m_i_thread - m_c) before sum-reduce.
    __local ACC_TYPE local_m[Q1_WG_SIZE];
    local_m[tid] = m_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_m[tid] = max(local_m[tid], local_m[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const ACC_TYPE m_c = local_m[0];

    const ACC_TYPE alpha_final = exp(m_i - m_c);
    l_i *= alpha_final;
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] *= alpha_final;

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

// Prefill: q4_0 K/V, n_q > 1. BLOCK_M × BLOCK_N tiling.
// K in local as packed nibbles + per-block scale; V dequant -> half in local.
// Requires DK % QK4_0 == 0 and DV % QK4_0 == 0.
#define KV_DATA_TYPE4 half4
#define CONVERT_KV_ACC4(x) convert_float4(x)

#define DK_Q4_BLOCKS_PREFILL (DK / QK4_0)
#define DV_Q4_BLOCKS_PREFILL (DV / QK4_0)

// N_SPLIT>1 splits DK/DV across N_SPLIT threads per query row; needs
// sub_group_shuffle_xor and DK_Q4_BLOCKS_PREFILL % N_SPLIT == 0.
#ifndef N_SPLIT
#define N_SPLIT 1
#endif

#if N_SPLIT > 1
#define SPLIT_DK_VEC        (DK_VEC / N_SPLIT)
#define SPLIT_DV_VEC        (DV_VEC / N_SPLIT)
#define SPLIT_DK_Q4_BLOCKS  (DK_Q4_BLOCKS_PREFILL / N_SPLIT)
#define WG_SIZE             (BLOCK_M * N_SPLIT)
#else
#define SPLIT_DK_VEC        DK_VEC
#define SPLIT_DV_VEC        DV_VEC
#define SPLIT_DK_Q4_BLOCKS  DK_Q4_BLOCKS_PREFILL
#define WG_SIZE             BLOCK_M
#endif

__kernel void flash_attn_f32_q4_0(
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
    // blk: per-(qblock,kvblock) class from flash_attn_blk_f16
    // (0=masked, 1=mixed, 2=unmasked). NULL disables the prepass opt.
    const global void * blk_void
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
    const int head_idx  = head_batch_idx % n_head;

    const int gqa_ratio   = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;
    const int mask_head_idx  = mask_void != NULL ? head_idx  % mask_ne2 : 0;
    const int mask_batch_idx = mask_void != NULL ? batch_idx % mask_ne3 : 0;

    const global char * q_base = (const global char *) q_void + q_offset;
    const global char * k_base = (const global char *) k_void + k_offset;
    const global char * v_base = (const global char *) v_void + v_offset;
    global       char * o_base = (global       char *) o_void + o_offset;

    const global char * mask_base = NULL;
    if (mask_void != NULL) {
        mask_base = (const global char *) mask_void + mask_offset +
                    mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    // BLK_PREPASS_BM may differ from this kernel's BLOCK_M; scale q-block idx.
    #ifndef BLK_PREPASS_BM
    #define BLK_PREPASS_BM BLOCK_M
    #endif
    const global char * blk_base = NULL;
    int n_kv_blocks = 0;
    if (blk_void != NULL) {
        n_kv_blocks = (n_kv + BLOCK_N - 1) / BLOCK_N;
        const int n_q_blocks_prepass = (n_q + BLK_PREPASS_BM - 1) / BLK_PREPASS_BM;
        const int prepass_q_block    = (block_q_idx * BLOCK_M) / BLK_PREPASS_BM;
        blk_base = (const global char *) blk_void +
                   (((mask_batch_idx * mask_ne2) + mask_head_idx) * n_q_blocks_prepass + prepass_q_block) * n_kv_blocks;
    }

    const int dk_off_vec = split_idx * SPLIT_DK_VEC;
    ACC_TYPE4 q_priv[SPLIT_DK_VEC];
    if (query_valid) {
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + my_query_row * q_nb1;
        const global float4 * q_ptr = (const global float4 *) (q_base + q_row_offset);
        #pragma unroll
        for (int i = 0; i < SPLIT_DK_VEC; ++i) {
            q_priv[i] = q_ptr[dk_off_vec + i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < SPLIT_DK_VEC; ++i) q_priv[i] = (ACC_TYPE4)(0.0f);
    }

#ifdef FA_HAVE_INT_DOT
    uint  q_packed_pf[SPLIT_DK_Q4_BLOCKS * 8];
    float q_d_pf[SPLIT_DK_Q4_BLOCKS];
    int   q_sum_pf[SPLIT_DK_Q4_BLOCKS];
    #pragma unroll
    for (int b = 0; b < SPLIT_DK_Q4_BLOCKS; ++b) {
        q4_q_block_info info = quant_q_block_int8_packed_q4(&q_priv[b * 8], &q_packed_pf[b * 8]);
        q_d_pf[b]   = info.qd;
        q_sum_pf[b] = info.q_sum;
    }
#endif

    const int dv_off_vec = split_idx * SPLIT_DV_VEC;
    ACC_TYPE4 o_acc[SPLIT_DV_VEC];
    #pragma unroll
    for (int i = 0; i < SPLIT_DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);

    ACC_TYPE m_i = FA_M_INIT;
    ACC_TYPE l_i = 0.0f;

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

#ifdef FA_HAVE_INT_DOT
    __local uint  l_k_packed[BLOCK_N][DK_Q4_BLOCKS_PREFILL * 8];
    __local float l_k_scale [BLOCK_N][DK_Q4_BLOCKS_PREFILL];
#else
    __local half4 l_k[BLOCK_N][DK_VEC];
#endif

    __local half4 l_v[BLOCK_N][DV_VEC];

    for (int k_start = 0; k_start < n_kv; k_start += BLOCK_N) {
        // Skip fully-masked KV tiles (uniform branch across WG).
        char blk_cur = 1;
        if (blk_base != NULL) {
            blk_cur = blk_base[k_start / BLOCK_N];
            if (blk_cur == 0) continue;
        }

        {
#ifdef FA_HAVE_INT_DOT
            const int k_blocks_per_row = DK_Q4_BLOCKS_PREFILL;
            const int n_blocks_total = BLOCK_N * k_blocks_per_row;
            for (int i = tid; i < n_blocks_total; i += WG_SIZE) {
                const int row = i / k_blocks_per_row;
                const int blk = i % k_blocks_per_row;
                const int k_row_idx = k_start + row;
                if (k_row_idx < n_kv) {
                    const ulong k_row_off = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_row_idx * k_nb1;
                    const global char * blk_ptr = k_base + k_row_off + blk * Q4_0_BLOCK_SIZE;
                    const float df = (float) vload_half(0, (const global half *) blk_ptr);
                    const global uchar * qs = (const global uchar *)(blk_ptr + 2);
                    l_k_scale[row][blk] = df;
                    uint k_packed[8];
                    pack_q4_0_nibbles(qs, k_packed);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        l_k_packed[row][blk * 8 + j] = k_packed[j];
                    }
                } else {
                    l_k_scale[row][blk] = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) l_k_packed[row][blk * 8 + j] = 0u;
                }
            }
#else
            // Fallback: dequant q4_0 -> half in local memory.
            const int k_blocks_per_row = DK_Q4_BLOCKS_PREFILL;
            const int n_blocks_total = BLOCK_N * k_blocks_per_row;
            for (int i = tid; i < n_blocks_total; i += WG_SIZE) {
                const int row = i / k_blocks_per_row;
                const int blk = i % k_blocks_per_row;
                const int k_row_idx = k_start + row;
                if (k_row_idx < n_kv) {
                    const ulong k_row_off = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_row_idx * k_nb1;
                    const global char * blk_ptr = k_base + k_row_off + blk * Q4_0_BLOCK_SIZE;
                    const float df = (float) vload_half(0, (const global half *) blk_ptr);
                    const global uchar * qs = (const global uchar *)(blk_ptr + 2);
                    #pragma unroll
                    for (int g = 0; g < 4; ++g) {
                        float4 vlo = df * (float4)((float)(int)(qs[g*4 + 0] & 0x0F) - 8.0f,
                                                   (float)(int)(qs[g*4 + 1] & 0x0F) - 8.0f,
                                                   (float)(int)(qs[g*4 + 2] & 0x0F) - 8.0f,
                                                   (float)(int)(qs[g*4 + 3] & 0x0F) - 8.0f);
                        float4 vhi = df * (float4)((float)(int)(qs[g*4 + 0] >> 4) - 8.0f,
                                                   (float)(int)(qs[g*4 + 1] >> 4) - 8.0f,
                                                   (float)(int)(qs[g*4 + 2] >> 4) - 8.0f,
                                                   (float)(int)(qs[g*4 + 3] >> 4) - 8.0f);
                        l_k[row][blk * 8 + g    ] = (half4)((half)vlo.s0, (half)vlo.s1, (half)vlo.s2, (half)vlo.s3);
                        l_k[row][blk * 8 + 4 + g] = (half4)((half)vhi.s0, (half)vhi.s1, (half)vhi.s2, (half)vhi.s3);
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) l_k[row][blk * 8 + j] = (half4)(0.0h);
                }
            }
#endif
        }
        // V tile load — dequant V -> half in local memory.
        {
            const int v_blocks_per_row = DV_Q4_BLOCKS_PREFILL;
            const int n_blocks_total = BLOCK_N * v_blocks_per_row;
            for (int i = tid; i < n_blocks_total; i += WG_SIZE) {
                const int row = i / v_blocks_per_row;
                const int blk = i % v_blocks_per_row;
                const int v_row_idx = k_start + row;
                if (v_row_idx < n_kv) {
                    const ulong v_row_off = batch_idx * v_nb3 + head_kv_idx * v_nb2 + v_row_idx * v_nb1;
                    const global char * blk_ptr = v_base + v_row_off + blk * Q4_0_BLOCK_SIZE;
                    const float df = (float) vload_half(0, (const global half *) blk_ptr);
                    const global uchar * qs = (const global uchar *)(blk_ptr + 2);
                    #pragma unroll
                    for (int g = 0; g < 4; ++g) {
                        float4 vlo = df * (float4)((float)(int)(qs[g*4 + 0] & 0x0F) - 8.0f,
                                                   (float)(int)(qs[g*4 + 1] & 0x0F) - 8.0f,
                                                   (float)(int)(qs[g*4 + 2] & 0x0F) - 8.0f,
                                                   (float)(int)(qs[g*4 + 3] & 0x0F) - 8.0f);
                        float4 vhi = df * (float4)((float)(int)(qs[g*4 + 0] >> 4) - 8.0f,
                                                   (float)(int)(qs[g*4 + 1] >> 4) - 8.0f,
                                                   (float)(int)(qs[g*4 + 2] >> 4) - 8.0f,
                                                   (float)(int)(qs[g*4 + 3] >> 4) - 8.0f);
                        l_v[row][blk * 8 + g    ] = (half4)((half)vlo.s0, (half)vlo.s1, (half)vlo.s2, (half)vlo.s3);
                        l_v[row][blk * 8 + 4 + g] = (half4)((half)vhi.s0, (half)vhi.s1, (half)vhi.s2, (half)vhi.s3);
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) l_v[row][blk * 8 + j] = (half4)(0.0h);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // QK dot + online softmax. N_SPLIT>1 reduces per-thread partials via shuffle_xor.
#if N_SPLIT > 1
        {
#else
        if (query_valid) {
#endif
            const int k_blk_base = split_idx * SPLIT_DK_Q4_BLOCKS;
            for (int j = 0; j < BLOCK_N; j += 4) {
                const int k_row0 = k_start + j;
                const int k_row1 = k_start + j + 1;
                const int k_row2 = k_start + j + 2;
                const int k_row3 = k_start + j + 3;

                ACC_TYPE s0, s1, s2, s3;
#ifdef FA_HAVE_INT_DOT
                s0 = 0.0f; s1 = 0.0f; s2 = 0.0f; s3 = 0.0f;
                #pragma unroll
                for (int b_local = 0; b_local < SPLIT_DK_Q4_BLOCKS; ++b_local) {
                    const int b = k_blk_base + b_local;
                    int sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
                    #pragma unroll
                    for (int g = 0; g < 8; ++g) {
                        const uint qp = q_packed_pf[b_local * 8 + g];
                        sum0 = dot_acc_sat_4x8packed_ss_int(qp, l_k_packed[j  ][b * 8 + g], sum0);
                        sum1 = dot_acc_sat_4x8packed_ss_int(qp, l_k_packed[j+1][b * 8 + g], sum1);
                        sum2 = dot_acc_sat_4x8packed_ss_int(qp, l_k_packed[j+2][b * 8 + g], sum2);
                        sum3 = dot_acc_sat_4x8packed_ss_int(qp, l_k_packed[j+3][b * 8 + g], sum3);
                    }
                    const float qd    = q_d_pf[b_local];
                    const int   q_sum = q_sum_pf[b_local];
                    s0 += (float)(sum0 - 8 * q_sum) * qd * l_k_scale[j  ][b];
                    s1 += (float)(sum1 - 8 * q_sum) * qd * l_k_scale[j+1][b];
                    s2 += (float)(sum2 - 8 * q_sum) * qd * l_k_scale[j+2][b];
                    s3 += (float)(sum3 - 8 * q_sum) * qd * l_k_scale[j+3][b];
                }
#else
                ACC_TYPE4 dot_acc0 = (ACC_TYPE4)(0.0f);
                ACC_TYPE4 dot_acc1 = (ACC_TYPE4)(0.0f);
                ACC_TYPE4 dot_acc2 = (ACC_TYPE4)(0.0f);
                ACC_TYPE4 dot_acc3 = (ACC_TYPE4)(0.0f);
                #pragma unroll
                for (int k = 0; k < SPLIT_DK_VEC; ++k) {
                    const ACC_TYPE4 qk = q_priv[k];
                    const int k_abs = dk_off_vec + k;
                    dot_acc0 = mad(qk, CONVERT_KV_ACC4(l_k[j  ][k_abs]), dot_acc0);
                    dot_acc1 = mad(qk, CONVERT_KV_ACC4(l_k[j+1][k_abs]), dot_acc1);
                    dot_acc2 = mad(qk, CONVERT_KV_ACC4(l_k[j+2][k_abs]), dot_acc2);
                    dot_acc3 = mad(qk, CONVERT_KV_ACC4(l_k[j+3][k_abs]), dot_acc3);
                }
                s0 = dot_acc0.s0 + dot_acc0.s1 + dot_acc0.s2 + dot_acc0.s3;
                s1 = dot_acc1.s0 + dot_acc1.s1 + dot_acc1.s2 + dot_acc1.s3;
                s2 = dot_acc2.s0 + dot_acc2.s1 + dot_acc2.s2 + dot_acc2.s3;
                s3 = dot_acc3.s0 + dot_acc3.s1 + dot_acc3.s2 + dot_acc3.s3;
#endif

#if N_SPLIT > 1
                // Power-of-2 N_SPLIT: shuffle_xor butterfly. N_SPLIT=3 (DK=96):
                // explicit 3-lane shuffle.
                #if (N_SPLIT & (N_SPLIT - 1)) == 0
                    #pragma unroll
                    for (int step = 1; step < N_SPLIT; step <<= 1) {
                        s0 += sub_group_shuffle_xor(s0, step);
                        s1 += sub_group_shuffle_xor(s1, step);
                        s2 += sub_group_shuffle_xor(s2, step);
                        s3 += sub_group_shuffle_xor(s3, step);
                    }
                #else
                    const uint tri_base = (get_sub_group_local_id() / N_SPLIT) * N_SPLIT;
                    s0 = sub_group_shuffle(s0, tri_base + 0) + sub_group_shuffle(s0, tri_base + 1) + sub_group_shuffle(s0, tri_base + 2);
                    s1 = sub_group_shuffle(s1, tri_base + 0) + sub_group_shuffle(s1, tri_base + 1) + sub_group_shuffle(s1, tri_base + 2);
                    s2 = sub_group_shuffle(s2, tri_base + 0) + sub_group_shuffle(s2, tri_base + 1) + sub_group_shuffle(s2, tri_base + 2);
                    s3 = sub_group_shuffle(s3, tri_base + 0) + sub_group_shuffle(s3, tri_base + 1) + sub_group_shuffle(s3, tri_base + 2);
                #endif
                if (!query_valid) { s0 = FA_M_INIT; s1 = FA_M_INIT; s2 = FA_M_INIT; s3 = FA_M_INIT; }
#endif
                s0 *= scale; s1 *= scale; s2 *= scale; s3 *= scale;

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

                if (query_valid && mask_base != NULL && blk_cur != 2) {
                    const global MASK_DATA_TYPE * mask_ptr =
                        (const global MASK_DATA_TYPE *) (mask_base + my_query_row * mask_nb1);
                    if (k_row0 < n_kv) s0 += slope * (ACC_TYPE) mask_ptr[k_row0];
                    if (k_row1 < n_kv) s1 += slope * (ACC_TYPE) mask_ptr[k_row1];
                    if (k_row2 < n_kv) s2 += slope * (ACC_TYPE) mask_ptr[k_row2];
                    if (k_row3 < n_kv) s3 += slope * (ACC_TYPE) mask_ptr[k_row3];
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

                #pragma unroll
                for (int i = 0; i < SPLIT_DV_VEC; ++i) {
                    const int i_abs = dv_off_vec + i;
                    o_acc[i] = mad(p3, CONVERT_KV_ACC4(l_v[j+3][i_abs]),
                               mad(p2, CONVERT_KV_ACC4(l_v[j+2][i_abs]),
                               mad(p1, CONVERT_KV_ACC4(l_v[j+1][i_abs]),
                               mad(p0, CONVERT_KV_ACC4(l_v[j  ][i_abs]),
                               o_acc[i] * scale_prev))));
                }
                l_i = l_i * scale_prev + p0 + p1 + p2 + p3;
                m_i = m_new;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write output.
    if (query_valid) {
        if (sinks_void != NULL) {
            const global ACC_TYPE * sinks_ptr =
                (const global ACC_TYPE *) ((const global char *) sinks_void + sinks_offset);
            const ACC_TYPE m_sink  = sinks_ptr[head_idx];
            const ACC_TYPE m_final = max(m_i, m_sink);
            const ACC_TYPE scale_o = exp(m_i - m_final);
            #pragma unroll
            for (int i = 0; i < SPLIT_DV_VEC; ++i) o_acc[i] *= scale_o;
            l_i = l_i * scale_o + exp(m_sink - m_final);
            m_i = m_final;
        }
        const ACC_TYPE l_inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        const ulong o_row_offset = batch_idx * o_nb3 + my_query_row * o_nb2 + head_idx * o_nb1;
        global float4 * o_row = (global float4 *) (o_base + o_row_offset);
        if (l_inv > 0.0f) {
            #pragma unroll
            for (int i = 0; i < SPLIT_DV_VEC; ++i) o_row[dv_off_vec + i] = o_acc[i] * l_inv;
        } else {
            #pragma unroll
            for (int i = 0; i < SPLIT_DV_VEC; ++i) o_row[dv_off_vec + i] = (float4)(0.0f);
        }
    }
}

// FD Pass 2: merge split partials. Identical across q4_0/q8_0/f16; each FA
// source owns a copy since kernels compile per-source-program.
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
    const int lane           = get_local_id(0);
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
