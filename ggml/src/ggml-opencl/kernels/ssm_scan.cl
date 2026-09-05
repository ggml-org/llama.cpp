// Mamba2 fused SSM scan kernel. One workgroup per (head, dim, seq); WG size =
// 64 threads. Each thread owns c_factor = d_state/64 state elements in
// private registers; the state stays resident across the n_tokens t-loop
//
// References:
//   ggml/src/ggml-cuda/ssm-scan.cu:117 ssm_scan_f32_group
//   ggml/src/ggml-cpu/ops.cpp:9368 ggml_compute_forward_ssm_scan_f32

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_khr_subgroups
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#if defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#else
#define REQD_SUBGROUP_SIZE_64
#endif

inline float softplus_f32(float x) {
    return (x <= 20.0f) ? log(1.0f + exp(x)) : x;
}

// d_state = 128 (most Mamba-2 models, e.g. mamba2-2.7B, Codestral-Mamba).
// WG = 64 threads, each holds 2 state elements (tid and tid+64) per row.
//
// SSM_R = rows of `dim` handled per workgroup (compile-time; default 1 keeps the
// original one-row kernel bit-identical). B and C are indexed by (group, token)
// only -- never by dim -- so with one row per workgroup, every head_dim
// workgroups of a head re-read the identical B/C rows. Handling SSM_R rows per
// workgroup cuts that traffic by SSM_R, and costs SSM_R state registers per
// thread plus SSM_R subgroup reductions per token. dt/dA are per-head and stay
// hoisted out of the row loop. Requires head_dim % SSM_R == 0, enforced at
// dispatch.
#ifndef SSM_R
#define SSM_R 1
#endif
#ifndef SSM_KNAME
#define SSM_KNAME kernel_ssm_scan_f32_mamba2_d128
#endif
REQD_SUBGROUP_SIZE_64
kernel void SSM_KNAME(
    global const char * src0_base, ulong src0_off,
    global const char * src1_base, ulong src1_off,
    global const char * src2_base, ulong src2_off,
    global const char * src3_base, ulong src3_off,
    global const char * src4_base, ulong src4_off,
    global const char * src5_base, ulong src5_off,
    global const char * src6_base, ulong src6_off,
    global       char * dst_base,  ulong dst_off,
    ulong s0_nb2, ulong s0_nb3,
    ulong x_nb2,  ulong x_nb3,
    ulong dt_nb1, ulong dt_nb2,
    ulong A_nb1,
    ulong B_nb2,  ulong B_nb3,
    ulong C_nb2,  ulong C_nb3,
    ulong s_off_bytes,
    int   head_dim, int n_head, int n_group, int n_tokens,
    int   K,        int n_seqs
) {
    const int d_state = 128;

    const int tid     = (int) get_local_id(0);
    const int wg_x    = (int) get_group_id(0);
    const int seq_id  = (int) get_group_id(1);

    // Each workgroup owns SSM_R consecutive dim rows of one head, so the B/C
    // loads below are issued once and reused across all SSM_R rows.
    const int rows_per_head = head_dim / SSM_R;
    const int head_id = wg_x / rows_per_head;
    const int dim_id  = (wg_x - head_id * rows_per_head) * SSM_R;
    const int g       = head_id / (n_head / n_group);

    src0_base += src0_off;
    src1_base += src1_off;
    src2_base += src2_off;
    src3_base += src3_off;
    src4_base += src4_off;
    src5_base += src5_off;
    src6_base += src6_off;
    dst_base  += dst_off;

    const int seq_slot = ((global const int *) src6_base)[seq_id];

    const ulong state_base_off = (ulong)seq_slot * s0_nb3 + (ulong)head_id * s0_nb2
                                + (ulong)dim_id * d_state * sizeof(float);
    global const float * s0_warp = (global const float *)(src0_base + state_base_off);
    const ulong state_out_off = (ulong)seq_id * s0_nb3 + (ulong)head_id * s0_nb2
                              + (ulong)dim_id * d_state * sizeof(float);
    global float * s_warp = (global float *)(dst_base + s_off_bytes + state_out_off);

    global const char * x_seq  = src1_base + (ulong)seq_id * x_nb3;
    global const char * dt_seq = src2_base + (ulong)seq_id * dt_nb2;
    global const char * B_seq  = src4_base + (ulong)seq_id * B_nb3 + (ulong)g * d_state * sizeof(float);
    global const char * C_seq  = src5_base + (ulong)seq_id * C_nb3 + (ulong)g * d_state * sizeof(float);

    const ulong y_dim_total = (ulong)n_head * head_dim;
    global float * y_seq = (global float *)dst_base
                           + (ulong)seq_id * (ulong)n_tokens * y_dim_total;

    const float A_val = ((global const float *)src3_base)[(ulong)head_id * A_nb1 / sizeof(float)];

    // c_factor = 2: each thread owns 2 state elements (tid and tid+64) per row.
    float state0[SSM_R];
    float state1[SSM_R];
    #pragma unroll
    for (int r = 0; r < SSM_R; ++r) {
        state0[r] = s0_warp[(ulong)r * d_state + tid];
        state1[r] = s0_warp[(ulong)r * d_state + tid + 64];
    }

    for (int t = 0; t < n_tokens; ++t) {
        // per-head, shared by all SSM_R rows
        const float dt_h        = ((global const float *)(dt_seq + (ulong)t * dt_nb1))[head_id];
        const float dt_softplus = softplus_f32(dt_h);
        const float dA          = exp(dt_softplus * A_val);

        // per-(group, token): loaded ONCE and reused across all SSM_R rows
        const float B0 = ((global const float *)(B_seq + (ulong)t * B_nb2))[tid];
        const float B1 = ((global const float *)(B_seq + (ulong)t * B_nb2))[tid + 64];
        const float C0 = ((global const float *)(C_seq + (ulong)t * C_nb2))[tid];
        const float C1 = ((global const float *)(C_seq + (ulong)t * C_nb2))[tid + 64];

        global const float * x_row = (global const float *)(x_seq + (ulong)t * x_nb2)
                                     + (ulong)head_id * head_dim + dim_id;

        #pragma unroll
        for (int r = 0; r < SSM_R; ++r) {
            const float x_dt = x_row[r] * dt_softplus;

            state0[r] = state0[r] * dA + B0 * x_dt;
            state1[r] = state1[r] * dA + B1 * x_dt;
            const float partial = state0[r] * C0 + state1[r] * C1;

            const float sum = sub_group_reduce_add(partial);
            if (tid == 0) {
                y_seq[(ulong)t * y_dim_total + (ulong)head_id * head_dim + dim_id + r] = sum;
            }
        }

        // Rollback snapshots. Slot 0 is the final state, written after the loop;
        // slots 1..K-1 hold the state as it stood after each of the last K-1
        // tokens, so a speculative verify can rewind when a drafted token is
        // rejected. `slot` counts back from the last token, matching the CUDA
        // reference. K == 1 (every non-speculative graph) leaves this dead.
        const int slot = n_tokens - 1 - t;
        if (slot > 0 && slot < K) {
            global float * snap = (global float *)(dst_base + s_off_bytes
                + ((ulong)slot * (ulong)n_seqs + (ulong)seq_id) * s0_nb3
                + (ulong)head_id * s0_nb2
                + (ulong)dim_id * d_state * sizeof(float));
            #pragma unroll
            for (int r = 0; r < SSM_R; ++r) {
                snap[(ulong)r * d_state + tid]      = state0[r];
                snap[(ulong)r * d_state + tid + 64] = state1[r];
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < SSM_R; ++r) {
        s_warp[(ulong)r * d_state + tid]      = state0[r];
        s_warp[(ulong)r * d_state + tid + 64] = state1[r];
    }
}

// d_state = 256 (Falcon-H1). WG = 64 threads, each holds 4 state elements.
REQD_SUBGROUP_SIZE_64
kernel void kernel_ssm_scan_f32_mamba2_d256(
    global const char * src0_base, ulong src0_off,
    global const char * src1_base, ulong src1_off,
    global const char * src2_base, ulong src2_off,
    global const char * src3_base, ulong src3_off,
    global const char * src4_base, ulong src4_off,
    global const char * src5_base, ulong src5_off,
    global const char * src6_base, ulong src6_off,
    global       char * dst_base,  ulong dst_off,
    ulong s0_nb2, ulong s0_nb3,
    ulong x_nb2,  ulong x_nb3,
    ulong dt_nb1, ulong dt_nb2,
    ulong A_nb1,
    ulong B_nb2,  ulong B_nb3,
    ulong C_nb2,  ulong C_nb3,
    ulong s_off_bytes,
    int   head_dim, int n_head, int n_group, int n_tokens,
    int   K,        int n_seqs
) {
    const int d_state = 256;

    const int tid     = (int) get_local_id(0);
    const int wg_x    = (int) get_group_id(0);
    const int seq_id  = (int) get_group_id(1);

    const int head_id = wg_x / head_dim;
    const int dim_id  = wg_x - head_id * head_dim;
    const int g       = head_id / (n_head / n_group);

    src0_base += src0_off;
    src1_base += src1_off;
    src2_base += src2_off;
    src3_base += src3_off;
    src4_base += src4_off;
    src5_base += src5_off;
    src6_base += src6_off;
    dst_base  += dst_off;

    const int seq_slot = ((global const int *) src6_base)[seq_id];

    const ulong state_base_off = (ulong)seq_slot * s0_nb3 + (ulong)head_id * s0_nb2
                                + (ulong)dim_id * d_state * sizeof(float);
    global const float * s0_warp = (global const float *)(src0_base + state_base_off);
    const ulong state_out_off = (ulong)seq_id * s0_nb3 + (ulong)head_id * s0_nb2
                              + (ulong)dim_id * d_state * sizeof(float);
    global float * s_warp = (global float *)(dst_base + s_off_bytes + state_out_off);

    global const char * x_seq  = src1_base + (ulong)seq_id * x_nb3;
    global const char * dt_seq = src2_base + (ulong)seq_id * dt_nb2;
    global const char * B_seq  = src4_base + (ulong)seq_id * B_nb3 + (ulong)g * d_state * sizeof(float);
    global const char * C_seq  = src5_base + (ulong)seq_id * C_nb3 + (ulong)g * d_state * sizeof(float);

    const ulong y_dim_total = (ulong)n_head * head_dim;
    global float * y_seq = (global float *)dst_base
                           + (ulong)seq_id * (ulong)n_tokens * y_dim_total;

    const float A_val = ((global const float *)src3_base)[(ulong)head_id * A_nb1 / sizeof(float)];

    // c_factor = 4: each thread owns 4 state elements.
    float state0 = s0_warp[tid];
    float state1 = s0_warp[tid + 64];
    float state2 = s0_warp[tid + 128];
    float state3 = s0_warp[tid + 192];

    for (int t = 0; t < n_tokens; ++t) {
        const float dt_h        = ((global const float *)(dt_seq + (ulong)t * dt_nb1))[head_id];
        const float dt_softplus = softplus_f32(dt_h);
        const float dA          = exp(dt_softplus * A_val);
        const float x_val       = ((global const float *)(x_seq + (ulong)t * x_nb2))[(ulong)head_id * head_dim + dim_id];
        const float x_dt        = x_val * dt_softplus;

        global const float * B_t = (global const float *)(B_seq + (ulong)t * B_nb2);
        global const float * C_t = (global const float *)(C_seq + (ulong)t * C_nb2);

        const float B0 = B_t[tid];
        const float B1 = B_t[tid + 64];
        const float B2 = B_t[tid + 128];
        const float B3 = B_t[tid + 192];
        const float C0 = C_t[tid];
        const float C1 = C_t[tid + 64];
        const float C2 = C_t[tid + 128];
        const float C3 = C_t[tid + 192];

        state0 = state0 * dA + B0 * x_dt;
        state1 = state1 * dA + B1 * x_dt;
        state2 = state2 * dA + B2 * x_dt;
        state3 = state3 * dA + B3 * x_dt;
        const float partial = state0 * C0 + state1 * C1 + state2 * C2 + state3 * C3;

        const float sum = sub_group_reduce_add(partial);
        if (tid == 0) {
            y_seq[(ulong)t * y_dim_total + (ulong)head_id * head_dim + dim_id] = sum;
        }

        // Rollback snapshots -- see the d128 kernel above.
        const int slot = n_tokens - 1 - t;
        if (slot > 0 && slot < K) {
            global float * snap = (global float *)(dst_base + s_off_bytes
                + ((ulong)slot * (ulong)n_seqs + (ulong)seq_id) * s0_nb3
                + (ulong)head_id * s0_nb2
                + (ulong)dim_id * d_state * sizeof(float));
            snap[tid]       = state0;
            snap[tid + 64]  = state1;
            snap[tid + 128] = state2;
            snap[tid + 192] = state3;
        }
    }

    s_warp[tid]       = state0;
    s_warp[tid + 64]  = state1;
    s_warp[tid + 128] = state2;
    s_warp[tid + 192] = state3;
}
