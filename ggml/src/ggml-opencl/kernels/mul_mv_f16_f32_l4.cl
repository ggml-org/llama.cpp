#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#ifdef cl_khr_subgroup_shuffle
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable
#define HAS_SUBGROUP_SHUFFLE 1
#elif defined(cl_qcom_subgroup_shuffle)
#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
#define HAS_SUBGROUP_SHUFFLE 1
#endif

// Assumes row size (ne00) is a multiple of 4
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int nrows = ne11;
    int r0 = get_group_id(0);
    int im = get_group_id(2);

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset_src0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;

    global half4 * x4 = (global half4 *) (src0 + offset_src0);

    for (int r1 = 0; r1 < nrows; ++r1) {
        ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

        global float4 * y4 = (global float4 *) (src1 + offset_src1);

        float sumf = 0;
        for (int i = get_sub_group_local_id(); i < ne00/4; i += get_max_sub_group_size()) {
            sumf += convert_float(x4[i].s0) * y4[i].s0;
            sumf += convert_float(x4[i].s1) * y4[i].s1;
            sumf += convert_float(x4[i].s2) * y4[i].s2;
            sumf += convert_float(x4[i].s3) * y4[i].s3;
        }

        float all_sum = sub_group_reduce_add(sumf);
        if (get_sub_group_local_id() == 0) {
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
    }
}

// Each subgroup produces DR_NDST outputs, assumes ne11 == 1
#define MUL_MAT_F16_F32_L4_DR_NDST 4

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4_dr(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global float*)((global char*)dst  + offsetd);

    const int r0_base = get_group_id(0) * MUL_MAT_F16_F32_L4_DR_NDST;
    const int im      = get_group_id(2);

    const int i12 = im % ne12;
    const int i13 = im / ne12;

    // assume ne11 == 1
    const ulong offset_src1 = i12*nb12 + i13*nb13;
    global float4 * y4 = (global float4 *)(src1 + offset_src1);

    global half4 * x4[MUL_MAT_F16_F32_L4_DR_NDST];
    float          sumf[MUL_MAT_F16_F32_L4_DR_NDST];

    const ulong   k_head_off = (i12/r2)*nb02 + (i13/r3)*nb03;

    #pragma unroll
    for (int n = 0; n < MUL_MAT_F16_F32_L4_DR_NDST; ++n) {
        int       r0   = r0_base + n;
        int       r0c  = r0 < ne01 ? r0 : 0;
        ulong     off  = (ulong)r0c*nb01 + k_head_off;
        x4[n]   = (global half4 *)(src0 + off);
        sumf[n] = 0.0f;
    }

    const int n_chunks = ne00 / 4;
    const int sg_size  = get_max_sub_group_size();
    const int lid      = get_sub_group_local_id();

    for (int i = lid; i < n_chunks; i += sg_size) {
        float4 q = y4[i];
        #pragma unroll
        for (int n = 0; n < MUL_MAT_F16_F32_L4_DR_NDST; ++n) {
            float4 k = convert_float4(x4[n][i]);
            sumf[n] = mad(k.s0, q.s0, sumf[n]);
            sumf[n] = mad(k.s1, q.s1, sumf[n]);
            sumf[n] = mad(k.s2, q.s2, sumf[n]);
            sumf[n] = mad(k.s3, q.s3, sumf[n]);
        }
    }

    #pragma unroll
    for (int n = 0; n < MUL_MAT_F16_F32_L4_DR_NDST; ++n) {
        float reduced = sub_group_reduce_add(sumf[n]);
        int   r0      = r0_base + n;
        if (lid == 0 && r0 < ne01) {
            dst[im*ne1*ne0 + r0] = reduced;
        }
    }
}

// Kernels for decoding, Adreno only for now
#define MUL_MAT_F16_F32_L4_DR_LS_R2_MAX 8

#ifdef ADRENO_GPU
#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
#define sub_group_shuffle_xor(val, mask) qcom_sub_group_shuffle_xor((val), (mask), CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, 0.0f)

REQD_SUBGROUP_SIZE_64
kernel void kernel_mul_mat_f16_f32_l4_dr_ls(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global float*)((global char*)dst  + offsetd);

    const int r0_base = get_group_id(0) * 2;
    const int kv_grp  = get_group_id(2);   // KV head group; im = kv_grp*r2 + q

    const int i12_kv = kv_grp % ne02;
    const int i13_kv = kv_grp / ne02;

    const int lid     = get_sub_group_local_id();
    const int subhalf = lid >> 5;          // 0 or 1 (which K row in the WG)
    const int intra   = lid & 31;          // 0..31 (lane within the half)

    const int r0  = r0_base + subhalf;
    const int r0c = r0 < ne01 ? r0 : 0;    // clamp OOB to row 0; skip write below

    // K row pointer for this lane (one K row per half-wave).
    const ulong k_off = (ulong)r0c*nb01 + (ulong)i12_kv*nb02 + (ulong)i13_kv*nb03;
    global half4 * x4 = (global half4 *)(src0 + k_off);

    global float4 * y4[MUL_MAT_F16_F32_L4_DR_LS_R2_MAX];
    #pragma unroll
    for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
        const int i12_q = i12_kv*r2 + q;
        const ulong q_off = (ulong)i12_q*nb12 + (ulong)i13_kv*nb13;
        y4[q] = (global float4 *)(src1 + q_off);
    }

    float partial[MUL_MAT_F16_F32_L4_DR_LS_R2_MAX];
    #pragma unroll
    for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
        partial[q] = 0.0f;
    }

    const int n_chunks = ne00 / 4;

    for (int i = intra; i < n_chunks; i += 32) {
        float4 k = convert_float4(x4[i]);

        #pragma unroll
        for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
            if (q < r2) {
                float4 v = y4[q][i];
                partial[q] = mad(k.s0, v.s0, partial[q]);
                partial[q] = mad(k.s1, v.s1, partial[q]);
                partial[q] = mad(k.s2, v.s2, partial[q]);
                partial[q] = mad(k.s3, v.s3, partial[q]);
            }
        }
    }

    // half-wave reduction
    #pragma unroll
    for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
        if (q < r2) {
            partial[q] += sub_group_shuffle_xor(partial[q],  1u);
            partial[q] += sub_group_shuffle_xor(partial[q],  2u);
            partial[q] += sub_group_shuffle_xor(partial[q],  4u);
            partial[q] += sub_group_shuffle_xor(partial[q],  8u);
            partial[q] += sub_group_shuffle_xor(partial[q], 16u);
        }
    }

    if (intra == 0 && r0 < ne01) {
        #pragma unroll
        for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
            if (q < r2) {
                const int im = i12_kv*r2 + q + i13_kv*ne12;
                dst[im*ne1*ne0 + r0] = partial[q];
            }
        }
    }
}

REQD_SUBGROUP_SIZE_64
kernel void kernel_mul_mat_f16_f32_l4_dr_lq(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global float*)((global char*)dst  + offsetd);

    const int r0_base = get_group_id(0) * 4;
    const int kv_grp  = get_group_id(2);

    const int i12_kv = kv_grp % ne02;
    const int i13_kv = kv_grp / ne02;

    const int lid   = get_sub_group_local_id();
    const int subq  = lid >> 4;            // 0..3 (which K row)
    const int intra = lid & 15;            // 0..15 (lane within quarter)

    const int r0  = r0_base + subq;
    const int r0c = r0 < ne01 ? r0 : 0;

    const ulong k_off = (ulong)r0c*nb01 + (ulong)i12_kv*nb02 + (ulong)i13_kv*nb03;
    global half4 * x4 = (global half4 *)(src0 + k_off);

    global float4 * y4[MUL_MAT_F16_F32_L4_DR_LS_R2_MAX];
    #pragma unroll
    for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
        const int i12_q = i12_kv*r2 + q;
        const ulong q_off = (ulong)i12_q*nb12 + (ulong)i13_kv*nb13;
        y4[q] = (global float4 *)(src1 + q_off);
    }

    float partial[MUL_MAT_F16_F32_L4_DR_LS_R2_MAX];
    #pragma unroll
    for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
        partial[q] = 0.0f;
    }

    const int n_chunks = ne00 / 4;

    for (int i = intra; i < n_chunks; i += 16) {
        float4 k = convert_float4(x4[i]);

        #pragma unroll
        for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
            if (q < r2) {
                float4 v = y4[q][i];
                partial[q] = mad(k.s0, v.s0, partial[q]);
                partial[q] = mad(k.s1, v.s1, partial[q]);
                partial[q] = mad(k.s2, v.s2, partial[q]);
                partial[q] = mad(k.s3, v.s3, partial[q]);
            }
        }
    }

    // quarter-wave reduction
    #pragma unroll
    for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
        if (q < r2) {
            partial[q] += sub_group_shuffle_xor(partial[q], 1u);
            partial[q] += sub_group_shuffle_xor(partial[q], 2u);
            partial[q] += sub_group_shuffle_xor(partial[q], 4u);
            partial[q] += sub_group_shuffle_xor(partial[q], 8u);
        }
    }

    if (intra == 0 && r0 < ne01) {
        #pragma unroll
        for (int q = 0; q < MUL_MAT_F16_F32_L4_DR_LS_R2_MAX; ++q) {
            if (q < r2) {
                const int im = i12_kv*r2 + q + i13_kv*ne12;
                dst[im*ne1*ne0 + r0] = partial[q];
            }
        }
    }
}
#endif // ADRENO_GPU

// Multi-row variant: each workgroup processes N_ROWS_PER_WG K rows instead of
// 1, amortizing dispatch overhead. The default kernel above launches one WG
// per (r0, im) which means ~262K workgroups for Qwen3.6 attn KQ at d=16k —
// 64 threads each doing one mad and a sub_group_reduce. Per-call wall time is
// ~8× over the K-bandwidth ideal because of wave-dispatch + memory-latency
// overhead. This variant collapses 8 of those WGs into one and caches Q once
// per Q-row in __local across the 8 K-row computations.
//
// Dispatched when ne11 == 1 (decode: single Q row) and ne01 % N_ROWS == 0,
// with global x = ne01/N_ROWS * subgroup_size.
#define N_ROWS_PER_WG 8
#define N_OUTS_PER_WG 8

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4_x8(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char *)((global char *)src0 + offset0);
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int sgs_sz  = get_max_sub_group_size();

    const int r0_base = get_group_id(0) * N_ROWS_PER_WG;
    const int im      = get_group_id(2);

    const int i12 = im % ne12;
    const int i13 = im / ne12;

    // Single Q row only (decode). Cache Q once in __local for reuse across
    // the N_ROWS K-row computations.
    const ulong offset_src1 = (i12) * nb12 + (i13) * nb13;
    global float4 * y4 = (global float4 *)(src1 + offset_src1);

    __local float4 q_loc[64];   // ne00/4 max for sub_group_size 64
    if (sgs_lid < ne00 / 4) {
        q_loc[sgs_lid] = y4[sgs_lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (int dr = 0; dr < N_ROWS_PER_WG; ++dr) {
        const int r0 = r0_base + dr;
        if (r0 >= ne01) return;

        const ulong offset_src0 = r0 * nb01 + (i12 / r2) * nb02 + (i13 / r3) * nb03;
        global half4 * x4 = (global half4 *)(src0 + offset_src0);

        float sumf = 0.0f;
        for (int i = sgs_lid; i < ne00 / 4; i += sgs_sz) {
            const half4   k4 = x4[i];
            const float4  q  = q_loc[i];
            sumf += convert_float(k4.s0) * q.s0
                  + convert_float(k4.s1) * q.s1
                  + convert_float(k4.s2) * q.s2
                  + convert_float(k4.s3) * q.s3;
        }

        const float all_sum = sub_group_reduce_add(sumf);
        if (sgs_lid == 0) {
            dst[im * ne1 * ne0 + r0] = all_sum;  // ne11 == 1, so r1==0
        }
    }
}

// Streaming-Q multi-output variant for the KQV-shaped matmul: src0 has small
// ne01 (e.g. DV=256) but large ne00 (n_kv, up to 16384 at d=16k). The x8
// kernel can't handle this because its per-WG __local Q cache is sized for
// ne00 <= 256. This variant streams Q from global (no cache) but still packs
// N_OUTS_PER_WG = 8 outputs per workgroup. Q is re-read once per output
// inside the inner loop; Adreno L1 absorbs the 8× redundancy since adjacent
// outputs in one WG hit the same Q cache lines per iter.
//
// Dispatched for the same shape pattern as x8 (ne11 == 1, ne01 divisible by 8)
// when ne00 > 256, i.e. when the x8 path can't be used.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4_y8(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char *)((global char *)src0 + offset0);
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int sgs_sz  = get_max_sub_group_size();

    const int r0_base = get_group_id(0) * N_OUTS_PER_WG;
    const int im      = get_group_id(2);

    const int i12 = im % ne12;
    const int i13 = im / ne12;

    // Q (= src1) base pointer; r1 == 0 since ne11 == 1.
    const ulong offset_src1 = (i12) * nb12 + (i13) * nb13;
    global float4 * y4 = (global float4 *)(src1 + offset_src1);

    // Per-output base pointers (per row of src0). Computed once; inner loop
    // strides float4 indices across them.
    global half4 * x4_o[N_OUTS_PER_WG];
    #pragma unroll
    for (int o = 0; o < N_OUTS_PER_WG; ++o) {
        const int r0 = r0_base + o;
        // Pre-cap: if r0 OOB, point to the first row (harmless reads, output
        // suppressed at write-time). Keeps the inner loop unconditional.
        const int r0c = (r0 < ne01) ? r0 : 0;
        const ulong off = r0c * nb01 + (i12 / r2) * nb02 + (i13 / r3) * nb03;
        x4_o[o] = (global half4 *)(src0 + off);
    }

    float sum[N_OUTS_PER_WG] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for (int i = sgs_lid; i < ne00 / 4; i += sgs_sz) {
        const float4 q4 = y4[i];
        #pragma unroll
        for (int o = 0; o < N_OUTS_PER_WG; ++o) {
            const half4 v4 = x4_o[o][i];
            sum[o] += convert_float(v4.s0) * q4.s0
                    + convert_float(v4.s1) * q4.s1
                    + convert_float(v4.s2) * q4.s2
                    + convert_float(v4.s3) * q4.s3;
        }
    }

    #pragma unroll
    for (int o = 0; o < N_OUTS_PER_WG; ++o) {
        const int r0 = r0_base + o;
        const float s = sub_group_reduce_add(sum[o]);
        if (sgs_lid == 0 && r0 < ne01) {
            dst[im * ne1 * ne0 + r0] = s;
        }
    }
}

// Lane-utilization variant of _x8 for the long-context KQ matmul (Adreno X2).
//
// Background: _x8 runs the inner K-loop as
//   for (i = sgs_lid; i < ne00/4; i += sgs_sz)
// At DK=128 (Qwen3/Qwen3.6/etc. attention head dim), ne00/4 = 32 but
// sgs_sz = 64, so the loop runs ONCE with lanes 0..31 active and lanes
// 32..63 idle. With N_ROWS_PER_WG=8 outer iters, half of every wave-cycle
// is wasted and the kernel issues only 32 cache-line requests per K-row
// instead of 64. The depth-sweep on Qwen3-30B-A3B (tg128@d=16k) attributes
// ~9.5% of the 76 GB/s coalesced-read peak — almost exactly the 50% lane
// idle × ~20% small-WG ceiling.
//
// Fix: pair K-rows across the warp. Lanes 0..31 process row (2p+0), lanes
// 32..63 process row (2p+1), in parallel. 4 pair iters cover the same 8
// outputs. Per-pair reduction stays within each 32-lane half via
// sub_group_shuffle_xor with masks <32 (per opencl_ssm_scan_mamba2:
// shuffle_xor with mask>=32 silently miscompiles on Adreno X2; we never
// cross the half-warp boundary).
//
// Same total compute (32×8 = 64×4 = 256 lane-mads), same total K bytes per
// WG, but 2× the per-wave-cycle memory-issue parallelism. Most beneficial
// for DK in [128, 256] where the per-row inner loop is only 1-2 iters.
//
// Dispatch grid identical to _x8: nth0=64, ((ne01/8)*64, 1, ne12*ne13).
// Opt-in via env var GGML_OPENCL_MM_KQ_PAIR=1 on the host side.
#define N_OUTS_PAIR  8
#define N_PAIRS_PAIR (N_OUTS_PAIR / 2)

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4_x8_pair(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char *)((global char *)src0 + offset0);
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int half_id = sgs_lid >> 5;     // 0 = lower half, 1 = upper half
    const int lane_h  = sgs_lid & 31;     // lane 0..31 within half

    const int r0_base = get_group_id(0) * N_OUTS_PAIR;
    const int im      = get_group_id(2);

    const int i12 = im % ne12;
    const int i13 = im / ne12;

    // Single Q row (decode). Cache Q once in __local for reuse across all
    // N_OUTS_PAIR K-rows. Both halves of the warp broadcast-read q_loc.
    const ulong offset_src1 = (i12) * nb12 + (i13) * nb13;
    global float4 * y4 = (global float4 *)(src1 + offset_src1);

    __local float4 q_loc[64];   // ne00/4 max for sub_group_size 64
    if (sgs_lid < ne00 / 4) {
        q_loc[sgs_lid] = y4[sgs_lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int dk_vec = ne00 / 4;

    #pragma unroll
    for (int p = 0; p < N_PAIRS_PAIR; ++p) {
        // Lower half handles even row, upper half handles odd row.
        // Dispatch guarantees ne01 % 8 == 0, so r0 is always in range.
        const int r0 = r0_base + 2 * p + half_id;

        const ulong offset_src0 = r0 * nb01 + (i12 / r2) * nb02 + (i13 / r3) * nb03;
        global half4 * x4 = (global half4 *)(src0 + offset_src0);

        float sumf = 0.0f;
        // Stride 32 within the 32-lane half. For DK=128 this runs once.
        for (int i = lane_h; i < dk_vec; i += 32) {
            const half4  k4 = x4[i];
            const float4 q  = q_loc[i];
            sumf += convert_float(k4.s0) * q.s0
                  + convert_float(k4.s1) * q.s1
                  + convert_float(k4.s2) * q.s2
                  + convert_float(k4.s3) * q.s3;
        }

        // Reduce within 32-lane half. All 64 lanes execute; the shuffle is
        // self-contained within each half because masks are <32.
        sumf += sub_group_shuffle_xor(sumf, 16);
        sumf += sub_group_shuffle_xor(sumf, 8);
        sumf += sub_group_shuffle_xor(sumf, 4);
        sumf += sub_group_shuffle_xor(sumf, 2);
        sumf += sub_group_shuffle_xor(sumf, 1);

        if (lane_h == 0) {
            dst[im * ne1 * ne0 + r0] = sumf;
        }
    }
}

// GQA-coalesced KQ variant for the long-context fa=0 path on Adreno X2.
//
// Background: at DK=128 the regular _x8 kernel reads each K-row r2 times
// (once per Q-head sharing the K-head — Qwen3-30B-A3B has r2 = n_head_q /
// n_head_kv = 8 with n_head_q=32, n_head_kv=4). At d=16k that's 16 MB of
// unique K read as 128 MB. The L2 path saturates near 56 GB/s, but most of
// the BW pays for redundant fetches; effective unique-K BW is 9.5% of peak.
//
// Fix: one WG per K-head (instead of per Q-head). The 64-lane warp partitions
// across GQA_RATIO=8 Q-heads (8 lanes each at DK=128 → 32 dk_vec elements /
// 8 lanes = 4 mads per lane per K-row). Each K-row is read once from global
// and contributes to GQA_RATIO outputs. Q vectors for the gqa_ratio Q-heads
// are pre-staged in __local.
//
// Specialized for DK=128 and GQA_RATIO=8 (Qwen3-30B-A3B / Qwen3-4B /
// Qwen3.6-A3B). Other shapes fall through to the regular _x8 path.
//
// Dispatch (separate from _x8): nth0=64, ((ne01/8)*64, 1, ne02*ne13).
// Opt-in via env var GGML_OPENCL_MM_KQ_GQA=1 on the host.
#define N_K_ROWS_GQA   16
#define GQA_RATIO_GQA  8
#define LANES_PER_QH   8    // 64 / GQA_RATIO_GQA
#define DK_VEC_GQA     32   // DK / 4 for DK=128

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4_x8_gqa4(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char *)((global char *)src0 + offset0);
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int q_id    = sgs_lid >> 3;       // 0..7: which Q-head (8 per WG)
    const int lane_q  = sgs_lid & 7;        // 0..7: lane within Q-head partition

    const int r0_base = get_group_id(0) * N_K_ROWS_GQA;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;           // K-head index (also K2 batch)
    const int i03 = im_kv / ne02;           // n13 batch index

    // GQA Q-heads sharing this K-head: i12 ∈ [i02*r2, i02*r2 + r2).
    // (r2 is the dispatch's guarantee == GQA_RATIO_GQA at this gate.)
    const int q_head_lo = i02 * GQA_RATIO_GQA;

    // Stage all GQA_RATIO Q vectors (one per Q-head sharing the K-head) into
    // __local. Each Q-head has DK_VEC_GQA = 32 float4 elements; 64 lanes load
    // 2 per Q-head (32 lanes × 2) using the first 32 lanes per Q-head.
    __local float4 q_loc[GQA_RATIO_GQA * DK_VEC_GQA];   // 4 × 32 = 128 float4
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_GQA; ++qh) {
        const int qh_idx = q_head_lo + qh;
        global float4 * y4 = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
        // Only lanes [0, DK_VEC_GQA) load; broadcast-safe.
        if (sgs_lid < DK_VEC_GQA) {
            q_loc[qh * DK_VEC_GQA + sgs_lid] = y4[sgs_lid];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // K base offset for this WG. All 8 K-rows × 4 Q-heads share this K-head.
    const ulong offset_src0_base = (i02) * nb02 + (i03 / r3) * nb03;

    #pragma unroll
    for (int dr = 0; dr < N_K_ROWS_GQA; ++dr) {
        const int r0 = r0_base + dr;
        // Dispatch guarantees ne01 % N_K_ROWS_GQA == 0, so r0 < ne01.

        const ulong offset_src0 = r0 * nb01 + offset_src0_base;
        global half4 * x4 = (global half4 *)(src0 + offset_src0);

        // Each lane reads 4 K elements covering DK_VEC_GQA = 32 float4 with
        // LANES_PER_QH = 8 lanes per Q-head partition. The 8 Q-head
        // partitions in the warp all read the same K-row in parallel (8
        // duplicated reads per element, served by L1 once cached).
        float sumf = 0.0f;
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int i = lane_q + t * LANES_PER_QH;   // 8, 16, 24-step
            const half4  k4 = x4[i];
            const float4 q  = q_loc[q_id * DK_VEC_GQA + i];
            sumf += convert_float(k4.s0) * q.s0
                  + convert_float(k4.s1) * q.s1
                  + convert_float(k4.s2) * q.s2
                  + convert_float(k4.s3) * q.s3;
        }

        // Reduce within 8-lane Q-head partition (masks < 8, well within the
        // safe-shuffle range for Adreno X2's 32-lane cluster bound).
        sumf += sub_group_shuffle_xor(sumf, 4);
        sumf += sub_group_shuffle_xor(sumf, 2);
        sumf += sub_group_shuffle_xor(sumf, 1);

        if (lane_q == 0) {
            // im for this Q-head: linearized (i12=q_head_lo+q_id, i13=i03)
            // matching the original _x8 dst[im * ne1 * ne0 + r0] indexing.
            const int im_out = i03 * ne12 + (q_head_lo + q_id);
            dst[im_out * ne1 * ne0 + r0] = sumf;
        }
    }
}

// GQA-coalesced KQV variant for the long-context fa=0 path on Adreno X2.
//
// Companion to _x8_gqa4 — applies the same GQA-replay-elimination trick to
// the second mul_mat per attention block. KQV shape: src0 = V cache
// [ne00=n_kv, ne01=DV=128, ne02=n_head_kv=4]; src1 = softmax(KQ)
// [ne10=n_kv, ne11=1, ne12=n_head_q=32]. r2 = ne12/ne02 = 8.
//
// Current _y8 dispatches one WG per Q-head × 8 DV-row outputs. It re-reads
// each V slab r2=8 times across Q-heads. Total V read ≈ 128 MB / call at
// d=16k for only 16 MB unique V data.
//
// _y8_gqa: one WG per K-head × 8 DV-row outputs (dispatch z = ne02 * ne13).
// Each WG produces N_DV_ROWS_Y8GQA × GQA_RATIO_Y8GQA = 8 × 8 = 64 outputs.
// V is read once per K-head (per DV-row group); softmax read once per
// (q_head, dv_group) — total softmax read unchanged, total V read /= r2 = 8.
//
// Inner loop: each of 64 lanes processes one K-position chunk (4 elements).
// Per iter, each lane reads 8 V values (one per DV row, all shared K-head)
// and 8 softmax values (one per Q-head). Per iter, each lane does 64 mads
// (8 DV × 8 Q-heads). Accumulator footprint per lane: 64 fp32 = 256 B
// (significant; watch for register pressure).
//
// Dispatch grid: nth0=64, ((ne01/8)*64, 1, ne02*ne13). Opt-in via env var
// GGML_OPENCL_MM_KQV_GQA=1.
#define N_DV_ROWS_Y8GQA  8
#define GQA_RATIO_Y8GQA  8

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4_y8_gqa(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char *)((global char *)src0 + offset0);
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int sgs_sz  = get_max_sub_group_size();

    const int r0_base = get_group_id(0) * N_DV_ROWS_Y8GQA;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;           // K-head index
    const int i03 = im_kv / ne02;           // n13 batch index

    // GQA Q-heads sharing this K-head.
    const int q_head_lo = i02 * GQA_RATIO_Y8GQA;

    // Q (= softmax(KQ)) base pointers per Q-head. Streaming — no caching;
    // each Q-head's softmax is read once per WG. Across all WGs (ne01/8
    // DV-groups × ne02 K-heads), total softmax reads = ne01/8 × ne12 × n_kv.
    global float4 * y4_q[GQA_RATIO_Y8GQA];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_Y8GQA; ++qh) {
        const int qh_idx = q_head_lo + qh;
        y4_q[qh] = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
    }

    // 8 V-row base pointers (per DV output). All 8 share K-head i02.
    global half4 * x4_o[N_DV_ROWS_Y8GQA];
    #pragma unroll
    for (int o = 0; o < N_DV_ROWS_Y8GQA; ++o) {
        const int r0 = r0_base + o;
        const int r0c = (r0 < ne01) ? r0 : 0;
        const ulong off = r0c * nb01 + (i02) * nb02 + (i03 / r3) * nb03;
        x4_o[o] = (global half4 *)(src0 + off);
    }

    // 64 accumulators: 8 DV rows × 8 Q-heads.
    float sum[N_DV_ROWS_Y8GQA][GQA_RATIO_Y8GQA] = { {0.0f} };

    for (int i = sgs_lid; i < ne00 / 4; i += sgs_sz) {
        // Load 8 V values (one per DV row), same K-head, K-pos = i.
        half4 v[N_DV_ROWS_Y8GQA];
        #pragma unroll
        for (int o = 0; o < N_DV_ROWS_Y8GQA; ++o) {
            v[o] = x4_o[o][i];
        }
        // Load 8 softmax values (one per Q-head).
        float4 q[GQA_RATIO_Y8GQA];
        #pragma unroll
        for (int qh = 0; qh < GQA_RATIO_Y8GQA; ++qh) {
            q[qh] = y4_q[qh][i];
        }
        // 64 mads.
        #pragma unroll
        for (int o = 0; o < N_DV_ROWS_Y8GQA; ++o) {
            const float4 vf = (float4)(convert_float(v[o].s0),
                                       convert_float(v[o].s1),
                                       convert_float(v[o].s2),
                                       convert_float(v[o].s3));
            #pragma unroll
            for (int qh = 0; qh < GQA_RATIO_Y8GQA; ++qh) {
                sum[o][qh] += vf.s0 * q[qh].s0
                            + vf.s1 * q[qh].s1
                            + vf.s2 * q[qh].s2
                            + vf.s3 * q[qh].s3;
            }
        }
    }

    // Reduce each of 64 accumulators across the 64-lane subgroup.
    #pragma unroll
    for (int o = 0; o < N_DV_ROWS_Y8GQA; ++o) {
        const int r0 = r0_base + o;
        #pragma unroll
        for (int qh = 0; qh < GQA_RATIO_Y8GQA; ++qh) {
            const float s = sub_group_reduce_add(sum[o][qh]);
            if (sgs_lid == 0 && r0 < ne01) {
                const int im_out = i03 * ne12 + (q_head_lo + qh);
                dst[im_out * ne1 * ne0 + r0] = s;
            }
        }
    }
}

// image1d_buffer_t (texture-cache) variant of _x8_gqa4.
//
// Background: profiling _x8_gqa4 N_K_ROWS_GQA=16 on Qwen3-30B-A3B (Adreno X2):
// KQ stayed at 69% of decode time @ d=16k with effective K-read BW of only
// ~7.3 GB/s — ~10% of the coalesced __global peak (76 GB/s). The kernel is
// memory-stall bound on serial K-row fetches; widening the per-WG window
// via N=16 didn't fix it. The prefill mul_mm_f16_f32_kq uses Adreno's
// texture-cache (image1d_buffer_t) path for K, which is a separate cache
// from L2 and historically gives much higher effective BW on Adreno.
//
// Identical math to _x8_gqa4. The only change is K is bound as a
// __read_only image1d_buffer_t over a sub-buffer covering the K cache for
// this call (offset already baked into the sub-buffer). The pixel format
// is {CL_RGBA, CL_FLOAT} → 1 pixel = 16 bytes = 1 half8 = 2 half4. Two
// pixels per K-row per lane (lane_q + t*8 for t∈{0,1}) cover 4 half4 worth
// of K data, matching _x8_gqa4's per-lane coverage. as_half8(float4) is a
// bit-cast: bytes preserved end-to-end (the prefill kq path uses the same
// pattern).
//
// Layout note: at decode KQ, the K view on Adreno is permuted (nb01 > nb02 —
// head-major). The host computes k_bytes generically from
// (ne01-1)*nb01 + (ne02-1)*nb02 + (ne03-1)*nb03 + ne00*sizeof(half) so the
// image covers the full byte span regardless of layout. Pitches in this
// kernel still come from nb01/nb02/nb03 unchanged, so the addressing math
// is identical to _x8_gqa4.
//
// Dispatch (separate from _x8_gqa4): same grid as the regular variant.
// Opt-in via env var GGML_OPENCL_MM_KQ_GQA_IMG=1 on the host.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4_x8_gqa4_img(
        __read_only image1d_buffer_t src0_img,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int q_id    = sgs_lid >> 3;       // 0..7: which Q-head (8 per WG)
    const int lane_q  = sgs_lid & 7;        // 0..7: lane within Q-head partition

    const int r0_base = get_group_id(0) * N_K_ROWS_GQA;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;
    const int i03 = im_kv / ne02;

    const int q_head_lo = i02 * GQA_RATIO_GQA;

    __local float4 q_loc[GQA_RATIO_GQA * DK_VEC_GQA];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_GQA; ++qh) {
        const int qh_idx = q_head_lo + qh;
        global float4 * y4 = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
        if (sgs_lid < DK_VEC_GQA) {
            q_loc[qh * DK_VEC_GQA + sgs_lid] = y4[sgs_lid];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Image pixel = 1 float4 = 16 bytes = 8 fp16 (= 2 half4). Pitches in pixel
    // units. Sub-buffer is bound at offset0 host-side so offset0 is not
    // threaded through here. Each lane reads 2 pixels per K-row covering 4
    // half4 worth of K data (same coverage as _x8_gqa4) via a slightly
    // different access pattern: lane lane_q ∈ [0,8) reads pixels {lane_q,
    // lane_q+8} (covering half4 positions {2*lane_q, 2*lane_q+1, 2*lane_q+16,
    // 2*lane_q+17}). Coalesced burst: 8 lanes in the same Q-head partition
    // read 8 consecutive pixels per iter.
    const int pitch_px_row  = (int)(nb01 >> 4);
    const int pitch_px_head = (int)(nb02 >> 4);
    const int pitch_px_n13  = (int)(nb03 >> 4);

    const int head_px_base = i02 * pitch_px_head + (i03 / r3) * pitch_px_n13;

    #pragma unroll
    for (int dr = 0; dr < N_K_ROWS_GQA; ++dr) {
        const int r0 = r0_base + dr;
        const int row_px_base = r0 * pitch_px_row + head_px_base;

        float sumf = 0.0f;
        #pragma unroll
        for (int t = 0; t < 2; ++t) {
            const int p = lane_q + t * LANES_PER_QH;          // pixel idx in row, 0..15
            const half8 k8 = as_half8(read_imagef(src0_img, row_px_base + p));
            const int   i0 = 2 * p;                            // first half4 idx
            const float4 qa = q_loc[q_id * DK_VEC_GQA + i0    ];
            const float4 qb = q_loc[q_id * DK_VEC_GQA + i0 + 1];
            sumf += convert_float(k8.s0) * qa.s0
                  + convert_float(k8.s1) * qa.s1
                  + convert_float(k8.s2) * qa.s2
                  + convert_float(k8.s3) * qa.s3
                  + convert_float(k8.s4) * qb.s0
                  + convert_float(k8.s5) * qb.s1
                  + convert_float(k8.s6) * qb.s2
                  + convert_float(k8.s7) * qb.s3;
        }

        sumf += sub_group_shuffle_xor(sumf, 4);
        sumf += sub_group_shuffle_xor(sumf, 2);
        sumf += sub_group_shuffle_xor(sumf, 1);

        if (lane_q == 0) {
            const int im_out = i03 * ne12 + (q_head_lo + q_id);
            dst[im_out * ne1 * ne0 + r0] = sumf;
        }
    }
}

// image1d_buffer_t (texture-cache) variant of _y8_gqa for KQV decode.
//
// Companion to _x8_gqa4_img. Same texture-cache idea applied to V: each
// per-WG inner-loop iter previously did 8 V-row loads from __global; now
// they go through the image cache instead.
//
// V layout note: at decode KQV, ggml typically TRANSPOSES V vs the K layout
// so n_kv is the FAST dim and DV is the SLOW dim. So:
//   - ne00 = n_kv (contracting dim of KQV matmul)
//   - ne01 = DV  (= 128 for Qwen3 family)
//   - ne02 = n_head_kv
//   - nb01 is the DV stride (= n_kv_alloc * 2 bytes typically)
// The host computes k_bytes generically the same way as for KQ.
//
// Image format: CL_RGBA/CL_HALF_FLOAT — 8-byte pixels = 1 half4. Same per-
// iter access pattern as _y8_gqa (1 half4 per V-row per lane per iter),
// just via read_imageh instead of __global load. Tried CL_RGBA/CL_FLOAT
// (16-byte pixels, 2 half4 per pixel) first — same total compute but with
// 2x per-iter staging which pushed register pressure over a cliff (-16%
// @ d=16k vs KQV_GQA). The 1:1 half4-per-pixel mapping keeps the kernel
// register-equivalent to _y8_gqa.
//
// Dispatch grid: same as _y8_gqa. Opt-in via env GGML_OPENCL_MM_KQV_GQA_IMG=1.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4_y8_gqa_img(
        __read_only image1d_buffer_t src0_img,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int sgs_sz  = get_max_sub_group_size();

    const int r0_base = get_group_id(0) * N_DV_ROWS_Y8GQA;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;
    const int i03 = im_kv / ne02;

    const int q_head_lo = i02 * GQA_RATIO_Y8GQA;

    // Q (= softmax(KQ)) base pointers per Q-head. Streaming — no caching.
    global float4 * y4_q[GQA_RATIO_Y8GQA];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_Y8GQA; ++qh) {
        const int qh_idx = q_head_lo + qh;
        y4_q[qh] = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
    }

    // Pixel-space pitches for V image. CL_RGBA/CL_HALF_FLOAT: 1 pixel = 8 bytes = 1 half4.
    const int pitch_px_row  = (int)(nb01 >> 3);
    const int pitch_px_head = (int)(nb02 >> 3);
    const int pitch_px_n13  = (int)(nb03 >> 3);

    const int head_px_base = i02 * pitch_px_head + (i03 / r3) * pitch_px_n13;

    // Per-DV-row pixel base.
    int row_px_base[N_DV_ROWS_Y8GQA];
    #pragma unroll
    for (int o = 0; o < N_DV_ROWS_Y8GQA; ++o) {
        const int r0  = r0_base + o;
        const int r0c = (r0 < ne01) ? r0 : 0;
        row_px_base[o] = r0c * pitch_px_row + head_px_base;
    }

    // 64 accumulators: 8 DV rows × 8 Q-heads.
    float sum[N_DV_ROWS_Y8GQA][GQA_RATIO_Y8GQA] = { {0.0f} };

    // Same per-iter access pattern as _y8_gqa: 1 half4 per V-row per lane.
    for (int i = sgs_lid; i < ne00 / 4; i += sgs_sz) {
        // V loads via image: 8 V-row pixels, all at column i in the row.
        half4 v[N_DV_ROWS_Y8GQA];
        #pragma unroll
        for (int o = 0; o < N_DV_ROWS_Y8GQA; ++o) {
            v[o] = read_imageh(src0_img, row_px_base[o] + i);
        }
        // Softmax (global): 8 float4, one per Q-head.
        float4 q[GQA_RATIO_Y8GQA];
        #pragma unroll
        for (int qh = 0; qh < GQA_RATIO_Y8GQA; ++qh) {
            q[qh] = y4_q[qh][i];
        }
        // 64 mads.
        #pragma unroll
        for (int o = 0; o < N_DV_ROWS_Y8GQA; ++o) {
            const float4 vf = (float4)(convert_float(v[o].s0),
                                       convert_float(v[o].s1),
                                       convert_float(v[o].s2),
                                       convert_float(v[o].s3));
            #pragma unroll
            for (int qh = 0; qh < GQA_RATIO_Y8GQA; ++qh) {
                sum[o][qh] += vf.s0 * q[qh].s0
                            + vf.s1 * q[qh].s1
                            + vf.s2 * q[qh].s2
                            + vf.s3 * q[qh].s3;
            }
        }
    }

    // Reduce each accumulator across the 64-lane subgroup.
    #pragma unroll
    for (int o = 0; o < N_DV_ROWS_Y8GQA; ++o) {
        const int r0 = r0_base + o;
        #pragma unroll
        for (int qh = 0; qh < GQA_RATIO_Y8GQA; ++qh) {
            const float s = sub_group_reduce_add(sum[o][qh]);
            if (sgs_lid == 0 && r0 < ne01) {
                const int im_out = i03 * ne12 + (q_head_lo + qh);
                dst[im_out * ne1 * ne0 + r0] = s;
            }
        }
    }
}

// r2=4 specialization of the GQA-coalesced KQ image kernel.
//
// Companion to `_x8_gqa4_img` (r2=8, Qwen3-30B-A3B / Qwen3.6-35B-A3B). This
// variant fans the 64-lane subgroup across 4 Q-heads (16 lanes per partition)
// so each lane covers 32/16 = 2 half4 of one K-row per pass. With
// CL_RGBA/CL_FLOAT image pixels (16 bytes = 1 half8 = 2 half4) this maps to
// exactly 1 pixel read per lane per K-row — no wasted BW.
//
// Targets: Llama-3-8B (n_head=32, n_kv=8 → r2=4), Llama-3.2-1B/3B (same),
// Qwen3-4B/8B (same). Other shapes fall through.
//
// Same dispatch grid as the r2=8 variant: nth0=64, ((ne01/16)*64, 1,
// ne02*ne13). Opt-in via env var GGML_OPENCL_MM_KQ_GQA_R4_IMG=1 on the host.
// Shape gate: ne11==1 && ne00==128 && r2==4 && r3==1 && ne01%16==0.
//
// Reduction uses sub_group_shuffle_xor with masks 8, 4, 2, 1 — all <32, safe
// on Adreno X2's silent-miscompile boundary.
#define N_K_ROWS_GQA_R4   16
#define GQA_RATIO_R4      4
#define LANES_PER_QH_R4   16    // = 64 / GQA_RATIO_R4
#define DK_VEC_R4         32    // DK / 4 for DK=128

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4_x8_gqa_r4_img(
        __read_only image1d_buffer_t src0_img,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int q_id    = sgs_lid >> 4;       // 0..3
    const int lane_q  = sgs_lid & 15;       // 0..15

    const int r0_base = get_group_id(0) * N_K_ROWS_GQA_R4;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;
    const int i03 = im_kv / ne02;

    const int q_head_lo = i02 * GQA_RATIO_R4;

    __local float4 q_loc[GQA_RATIO_R4 * DK_VEC_R4];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_R4; ++qh) {
        const int qh_idx = q_head_lo + qh;
        global float4 * y4 = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
        if (sgs_lid < DK_VEC_R4) {
            q_loc[qh * DK_VEC_R4 + sgs_lid] = y4[sgs_lid];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int pitch_px_row  = (int)(nb01 >> 4);
    const int pitch_px_head = (int)(nb02 >> 4);
    const int pitch_px_n13  = (int)(nb03 >> 4);

    const int head_px_base = i02 * pitch_px_head + (i03 / r3) * pitch_px_n13;

    #pragma unroll
    for (int dr = 0; dr < N_K_ROWS_GQA_R4; ++dr) {
        const int r0 = r0_base + dr;
        const int row_px_base = r0 * pitch_px_row + head_px_base;

        // 1 pixel = 1 half8 = 2 half4. With LANES_PER_QH_R4=16 lanes
        // and DK_VEC_R4=32 half4/row, lane_q reads pixel `lane_q`.
        const int p = lane_q;
        const half8 k8 = as_half8(read_imagef(src0_img, row_px_base + p));
        const int   i0 = 2 * p;
        const float4 qa = q_loc[q_id * DK_VEC_R4 + i0    ];
        const float4 qb = q_loc[q_id * DK_VEC_R4 + i0 + 1];

        float sumf =
              convert_float(k8.s0) * qa.s0
            + convert_float(k8.s1) * qa.s1
            + convert_float(k8.s2) * qa.s2
            + convert_float(k8.s3) * qa.s3
            + convert_float(k8.s4) * qb.s0
            + convert_float(k8.s5) * qb.s1
            + convert_float(k8.s6) * qb.s2
            + convert_float(k8.s7) * qb.s3;

        // Reduce within 16-lane Q-head partition.
        sumf += sub_group_shuffle_xor(sumf, 8);
        sumf += sub_group_shuffle_xor(sumf, 4);
        sumf += sub_group_shuffle_xor(sumf, 2);
        sumf += sub_group_shuffle_xor(sumf, 1);

        if (lane_q == 0) {
            const int im_out = i03 * ne12 + (q_head_lo + q_id);
            dst[im_out * ne1 * ne0 + r0] = sumf;
        }
    }
}

// DK=256, r2=2 specialization for Gemma-3-4B (n_head=8, n_head_kv=4,
// head_dim=256). 64-lane subgroup partitioned across 2 Q-heads (32 lanes
// each). DK_VEC=64 half4 per K-row; each lane reads 1 CL_RGBA/CL_FLOAT
// pixel = 2 half4 — clean 1:1 mapping. Reduction masks {16,8,4,2,1} are
// all <32 (safe on Adreno X2). Q stage: 2×64=128 float4 = 2 KB local.
//
// Opt-in via GGML_OPENCL_MM_KQ_GQA_R2_DK256_IMG=1.
#define N_K_ROWS_GQA_R2_DK256   16
#define GQA_RATIO_R2            2
#define LANES_PER_QH_R2         32    // = 64 / GQA_RATIO_R2
#define DK_VEC_DK256            64    // DK / 4 for DK=256

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4_x8_gqa_r2_dk256_img(
        __read_only image1d_buffer_t src0_img,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src1 = (global char *)((global char *)src1 + offset1);
    dst  = (global float*)((global char *)dst  + offsetd);

    const int sgs_lid = get_sub_group_local_id();
    const int q_id    = sgs_lid >> 5;       // 0..1
    const int lane_q  = sgs_lid & 31;       // 0..31

    const int r0_base = get_group_id(0) * N_K_ROWS_GQA_R2_DK256;
    const int im_kv   = get_group_id(2);

    const int i02 = im_kv % ne02;
    const int i03 = im_kv / ne02;

    const int q_head_lo = i02 * GQA_RATIO_R2;

    // Stage 2 Q-heads × 64 float4 = 128 float4 (2 KB). Each of 64 lanes
    // loads exactly one float4 per Q-head (DK_VEC_DK256 == subgroup size).
    __local float4 q_loc[GQA_RATIO_R2 * DK_VEC_DK256];
    #pragma unroll
    for (int qh = 0; qh < GQA_RATIO_R2; ++qh) {
        const int qh_idx = q_head_lo + qh;
        global float4 * y4 = (global float4 *)(src1 + qh_idx * nb12 + i03 * nb13);
        q_loc[qh * DK_VEC_DK256 + sgs_lid] = y4[sgs_lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int pitch_px_row  = (int)(nb01 >> 4);
    const int pitch_px_head = (int)(nb02 >> 4);
    const int pitch_px_n13  = (int)(nb03 >> 4);

    const int head_px_base = i02 * pitch_px_head + (i03 / r3) * pitch_px_n13;

    #pragma unroll
    for (int dr = 0; dr < N_K_ROWS_GQA_R2_DK256; ++dr) {
        const int r0 = r0_base + dr;
        const int row_px_base = r0 * pitch_px_row + head_px_base;

        // 1 pixel per lane = 2 half4 = 8 fp16. 32 lanes × 2 half4 = 64 half4 covers DK=256.
        const int p = lane_q;
        const half8 k8 = as_half8(read_imagef(src0_img, row_px_base + p));
        const int   i0 = 2 * p;
        const float4 qa = q_loc[q_id * DK_VEC_DK256 + i0    ];
        const float4 qb = q_loc[q_id * DK_VEC_DK256 + i0 + 1];

        float sumf =
              convert_float(k8.s0) * qa.s0
            + convert_float(k8.s1) * qa.s1
            + convert_float(k8.s2) * qa.s2
            + convert_float(k8.s3) * qa.s3
            + convert_float(k8.s4) * qb.s0
            + convert_float(k8.s5) * qb.s1
            + convert_float(k8.s6) * qb.s2
            + convert_float(k8.s7) * qb.s3;

        sumf += sub_group_shuffle_xor(sumf, 16);
        sumf += sub_group_shuffle_xor(sumf, 8);
        sumf += sub_group_shuffle_xor(sumf, 4);
        sumf += sub_group_shuffle_xor(sumf, 2);
        sumf += sub_group_shuffle_xor(sumf, 1);

        if (lane_q == 0) {
            const int im_out = i03 * ne12 + (q_head_lo + q_id);
            dst[im_out * ne1 * ne0 + r0] = sumf;
        }
    }
}
