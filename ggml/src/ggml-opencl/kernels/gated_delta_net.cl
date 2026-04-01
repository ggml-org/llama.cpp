#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// gated_delta_net
//------------------------------------------------------------------------------
kernel void kernel_gated_delta_net_f32(
        global uchar * q,
        ulong offset_q,
        global uchar * k,
        ulong offset_k,
        global uchar * v,
        ulong offset_v,
        global uchar * g,
        ulong offset_g,
        global uchar * beta,
        ulong offset_beta,
        global uchar * state_in,
        ulong offset_state_in,
        global uchar * dst,
        ulong offset_dst,
        int H,
        int n_tokens,
        int n_seqs,
        ulong sq1,
        ulong sq2,
        ulong sq3,
        ulong sv1,
        ulong sv2,
        ulong sv3,
        ulong sb1,
        ulong sb2,
        ulong sb3,
        int rq1,
        int rq3,
        int S_v,
        int kda,
        float scale) {
    const int col      = get_global_id(0);
    const int h_idx    = get_global_id(1);
    const int sequence = get_global_id(2);

    if (col >= S_v || h_idx >= H || sequence >= n_seqs) {
        return;
    }

    const int iq1 = h_idx / rq1;
    const int iq3 = sequence / rq3;

    global uchar * q_base = q + offset_q;
    global uchar * k_base = k + offset_k;
    global uchar * v_base = v + offset_v;
    global uchar * g_base = g + offset_g;
    global uchar * b_base = beta + offset_beta;
    global uchar * state_base = state_in + offset_state_in;
    global uchar * dst_base = dst + offset_dst;

    const ulong state_offset_elems = ((ulong)sequence * (ulong)H + (ulong)h_idx) * (ulong)S_v * (ulong)S_v;
    const ulong attn_score_elems   = (ulong)S_v * (ulong)H * (ulong)n_tokens * (ulong)n_seqs;

    global float * state_out = (global float *)(dst_base + attn_score_elems * sizeof(float)) + state_offset_elems;
    global const float * curr_state = (global const float *)state_base + state_offset_elems;
    global float * attn_data = (global float *)dst_base + (((ulong)sequence * (ulong)n_tokens * (ulong)H + (ulong)h_idx) * (ulong)S_v);

    float s_private[128];
    for (int i = 0; i < S_v; ++i) {
        s_private[i] = curr_state[i * S_v + col];
    }

    for (int t = 0; t < n_tokens; ++t) {
        global const float * q_t = (global const float *)(q_base + (ulong)iq3 * sq3 + (ulong)t * sq2 + (ulong)iq1 * sq1);
        global const float * k_t = (global const float *)(k_base + (ulong)iq3 * sq3 + (ulong)t * sq2 + (ulong)iq1 * sq1);
        global const float * v_t = (global const float *)(v_base + (ulong)sequence * sv3 + (ulong)t * sv2 + (ulong)h_idx * sv1);

        const ulong gb_offset = (ulong)sequence * sb3 + (ulong)t * sb2 + (ulong)h_idx * sb1;
        global const float * beta_t = (global const float *)(b_base + gb_offset);
        global const float * g_t    = (global const float *)(g_base + gb_offset * (ulong)(kda ? S_v : 1));

        const float beta_val = *beta_t;

        float kv_col = 0.0f;
        if (!kda) {
            const float g_val = exp(*g_t);
            for (int i = 0; i < S_v; ++i) {
                kv_col += s_private[i] * k_t[i];
            }

            const float delta_col = (v_t[col] - g_val * kv_col) * beta_val;
            float attn_col = 0.0f;
            for (int i = 0; i < S_v; ++i) {
                s_private[i] = g_val * s_private[i] + k_t[i] * delta_col;
                attn_col += s_private[i] * q_t[i];
            }
            attn_data[col] = attn_col * scale;
        } else {
            for (int i = 0; i < S_v; ++i) {
                kv_col += exp(g_t[i]) * s_private[i] * k_t[i];
            }

            const float delta_col = (v_t[col] - kv_col) * beta_val;
            float attn_col = 0.0f;
            for (int i = 0; i < S_v; ++i) {
                s_private[i] = exp(g_t[i]) * s_private[i] + k_t[i] * delta_col;
                attn_col += s_private[i] * q_t[i];
            }
            attn_data[col] = attn_col * scale;
        }

        attn_data += S_v * H;
    }

    for (int i = 0; i < S_v; ++i) {
        state_out[i * S_v + col] = s_private[i];
    }
}
