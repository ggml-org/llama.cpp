#include "common.h"

// ref: ggml.c:ggml_compute_forward_ssm_conv_f32
kernel void kernel_ssm_conv_f32_f32(
        constant ggml_metal_kargs_ssm_conv & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t ir = tgpig.x;
    const int64_t i2 = tgpig.y;
    const int64_t i3 = tgpig.z;

    const int64_t nc  = args.ne10;
  //const int64_t ncs = args.ne00;
  //const int64_t nr  = args.ne01;
  //const int64_t n_t = args.ne1;
  //const int64_t n_s = args.ne2;

    device const float * s = (device const float *) ((device const char *) src0 + ir*args.nb01 + i2*args.nb00 + i3*args.nb02);
    device const float * c = (device const float *) ((device const char *) src1 + ir*args.nb11);
    device       float * x = (device       float *) ((device       char *) dst  + ir*args.nb0  + i2*args.nb1  + i3*args.nb2);

    float sumf = 0.0f;

    for (int64_t i0 = 0; i0 < nc; ++i0) {
        sumf += s[i0] * c[i0];
    }

    x[0] = sumf;
}

kernel void kernel_ssm_conv_f32_f32_4(
        constant ggml_metal_kargs_ssm_conv & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t ir = tgpig.x;
    const int64_t i2 = tgpig.y;
    const int64_t i3 = tgpig.z;

    const int64_t nc  = args.ne10;
  //const int64_t ncs = args.ne00;
  //const int64_t nr  = args.ne01;
  //const int64_t n_t = args.ne1;
  //const int64_t n_s = args.ne2;

    device const float4 * s = (device const float4 *) ((device const char *) src0 + ir*args.nb01 + i2*args.nb00 + i3*args.nb02);
    device const float4 * c = (device const float4 *) ((device const char *) src1 + ir*args.nb11);
    device       float  * x = (device       float  *) ((device       char *) dst  + ir*args.nb0  + i2*args.nb1  + i3*args.nb2);

    float sumf = 0.0f;

    for (int64_t i0 = 0; i0 < nc/4; ++i0) {
        sumf += dot(s[i0], c[i0]);
    }

    x[0] = sumf;
}

constant short FC_ssm_conv_bs   [[function_constant(FC_SSM_CONV + 0)]];

// Batched version: each threadgroup processes multiple tokens for better efficiency
// Thread layout: each thread handles one token, threadgroup covers BATCH_SIZE tokens
kernel void kernel_ssm_conv_f32_f32_batched(
        constant ggml_metal_kargs_ssm_conv & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    // tgpig.x = row index (ir)
    // tgpig.y = batch of tokens (i2_base / BATCH_SIZE)
    // tgpig.z = sequence index (i3)
    // tpitg.x = thread within batch (0..BATCH_SIZE-1)
    const short BATCH_SIZE = FC_ssm_conv_bs;

    const int64_t ir      = tgpig.x;
    const int64_t i2_base = tgpig.y * BATCH_SIZE;
    const int64_t i3      = tgpig.z;
    const int64_t i2_off  = tpitg.x;
    const int64_t i2      = i2_base + i2_off;

    const int64_t nc  = args.ne10;  // conv kernel size (typically 4)
    const int64_t n_t = args.ne1;   // number of tokens

    // Bounds check for partial batches at the end
    if (i2 >= n_t) {
        return;
    }

    // Load conv weights (shared across all tokens for this row)
    device const float * c = (device const float *) ((device const char *) src1 + ir*args.nb11);

    // Load source for this specific token
    device const float * s = (device const float *) ((device const char *) src0 + ir*args.nb01 + i2*args.nb00 + i3*args.nb02);

    // Output location for this token
    device float * x = (device float *) ((device char *) dst + ir*args.nb0 + i2*args.nb1 + i3*args.nb2);

    float sumf = 0.0f;
    for (int64_t i0 = 0; i0 < nc; ++i0) {
        sumf += s[i0] * c[i0];
    }

    x[0] = sumf;
}

kernel void kernel_ssm_conv_f32_f32_batched_4(
        constant ggml_metal_kargs_ssm_conv & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    // tgpig.x = row index (ir)
    // tgpig.y = batch of tokens (i2_base / BATCH_SIZE)
    // tgpig.z = sequence index (i3)
    // tpitg.x = thread within batch (0..BATCH_SIZE-1)
    const short BATCH_SIZE = FC_ssm_conv_bs;

    const int64_t ir      = tgpig.x;
    const int64_t i2_base = tgpig.y * BATCH_SIZE;
    const int64_t i3      = tgpig.z;
    const int64_t i2_off  = tpitg.x;
    const int64_t i2      = i2_base + i2_off;

    const int64_t nc  = args.ne10;  // conv kernel size (typically 4)
    const int64_t n_t = args.ne1;   // number of tokens

    // Bounds check for partial batches at the end
    if (i2 >= n_t) {
        return;
    }

    // Load conv weights (shared across all tokens for this row)
    device const float4 * c = (device const float4 *) ((device const char *) src1 + ir*args.nb11);

    // Load source for this specific token
    device const float4 * s = (device const float4 *) ((device const char *) src0 + ir*args.nb01 + i2*args.nb00 + i3*args.nb02);

    // Output location for this token
    device float * x = (device float *) ((device char *) dst + ir*args.nb0 + i2*args.nb1 + i3*args.nb2);

    float sumf = 0.0f;
    for (int64_t i0 = 0; i0 < nc/4; ++i0) {
        sumf += dot(s[i0], c[i0]);
    }

    x[0] = sumf;
}

// ref: ggml.c:ggml_compute_forward_ssm_scan_f32, Mamba-2 part
// Optimized version: reduces redundant memory loads by having one thread load shared values
kernel void kernel_ssm_scan_f32(
        constant ggml_metal_kargs_ssm_scan & args,
        device const void * src0,
        device const void * src1,
        device const void * src2,
        device const void * src3,
        device const void * src4,
        device const void * src5,
        device const void * src6,
        device      float * dst,
        threadgroup float * shared [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgptg[[simdgroups_per_threadgroup]],
        uint3    tgpg[[threadgroups_per_grid]]) {
    constexpr short NW = N_SIMDWIDTH;

    // Shared memory layout:
    // [0..sgptg*NW-1]: partial sums for reduction (existing)
    // [sgptg*NW..sgptg*NW+sgptg-1]: pre-computed x_dt values for each token in batch
    // [sgptg*NW+sgptg..sgptg*NW+2*sgptg-1]: pre-computed dA values for each token in batch
    threadgroup float * shared_sums = shared;
    threadgroup float * shared_x_dt = shared + sgptg * NW;
    threadgroup float * shared_dA   = shared + sgptg * NW + sgptg;

    shared_sums[tpitg.x] = 0.0f;

    const int32_t i0 = tpitg.x;
    const int32_t i1 = tgpig.x;
    const int32_t ir = tgpig.y; // current head
    const int32_t i3 = tgpig.z; // current seq

    const int32_t nc  = args.d_state;
    const int32_t nr  = args.d_inner;
    const int32_t nh  = args.n_head;
    const int32_t ng  = args.n_group;
    const int32_t n_t = args.n_seq_tokens;

    const int32_t s_off = args.s_off;

    device const int32_t * ids = (device const int32_t *) src6;

    device const float * s0_buff = (device const float *) ((device const char *) src0 + ir*args.nb02 + ids[i3]*args.nb03);
    device       float * s_buff  = (device       float *) ((device       char *) dst  + ir*args.nb02 +      i3*args.nb03 + s_off);

    const int32_t i = i0 + i1*nc;
    const int32_t g = ir / (nh / ng); // repeat_interleave

    float s0 = s0_buff[i];
    float s  = 0.0f;

    device const float * A = (device const float *) ((device const char *) src3 + ir*args.nb31); // {ne30, nh}

    const float A0 = A[i0%args.ne30];

    device const float * x  = (device const float *)((device const char *) src1 + i1*args.nb10  + ir*args.nb11 + i3*args.nb13); // {dim, nh, nt, ns}
    device const float * dt = (device const float *)((device const char *) src2 + ir*args.nb20  + i3*args.nb22);                // {nh, nt, ns}
    device const float * B  = (device const float *)((device const char *) src4 +  g*args.nb41  + i3*args.nb43);                // {d_state, ng, nt, ns}
    device const float * C  = (device const float *)((device const char *) src5 +  g*args.nb51  + i3*args.nb53);                // {d_state, ng, nt, ns}

    device float * y = dst + (i1 + ir*(nr) + i3*(n_t*nh*nr)); // {dim, nh, nt, ns}

    for (int i2 = 0; i2 < n_t; i2 += sgptg) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Pre-compute x_dt and dA for this batch of tokens
        // Only first sgptg threads do the loads and expensive math
        if (i0 < sgptg && i2 + i0 < n_t) {
            // ns12 and ns21 are element strides (nb12/nb10, nb21/nb20)
            device const float * x_t  = x  + i0 * args.ns12;
            device const float * dt_t = dt + i0 * args.ns21;

            const float dt0  = dt_t[0];
            const float dtsp = dt0 <= 20.0f ? log(1.0f + exp(dt0)) : dt0;
            shared_x_dt[i0] = x_t[0] * dtsp;
            shared_dA[i0]   = dtsp;  // Store dtsp, compute exp(dtsp * A0) per-thread since A0 varies
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int t = 0; t < sgptg && i2 + t < n_t; t++) {
            const float x_dt = shared_x_dt[t];
            const float dA   = exp(shared_dA[t] * A0);

            s = (s0 * dA) + (B[i0] * x_dt);

            const float sumf = simd_sum(s * C[i0]);

            if (tiisg == 0) {
                shared_sums[t*NW + sgitg] = sumf;
            }

            // recurse
            s0 = s;

            B  += args.ns42;
            C  += args.ns52;
        }

        // Advance pointers for next batch
        x  += sgptg * args.ns12;
        dt += sgptg * args.ns21;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float sumf = simd_sum(shared_sums[sgitg*NW + tiisg]);

        if (tiisg == 0 && i2 + sgitg < n_t) {
            y[sgitg*nh*nr] = sumf;
        }

        y += sgptg*nh*nr;
    }

    s_buff[i] = s;
}
