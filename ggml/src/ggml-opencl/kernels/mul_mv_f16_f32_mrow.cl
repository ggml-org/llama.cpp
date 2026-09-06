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

// Multi-row f16xf32 GEMV for the DECODE path (single token, ne11*ne12 small).
// The legacy kernel_mul_mat_f16_f32_1row runs ONE 64-lane subgroup per workgroup =
// one output row per WG, which caps memory-level parallelism at roughly half of
// LPDDR5x peak. This variant packs MROW subgroups per workgroup, each
// computing a distinct output row, so a WG keeps 64*MROW loads in flight. The
// activation column y (shared by every output row) is staged into __local ONCE per
// WG and reused across the MROW rows, cutting redundant activation reads. Used for
// the f16 attention projections (Q/K/V/O) and lm_head, which dominate decode.
// Numerically equivalent to _1row (same f16->f32 widening, same float4 partial sums,
// same subgroup-reduce order), so byte-identical to the per-op path.

#define MROW 16

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_mrow(
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
        int r3,
        __local float * ysh
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global float*)((global char*)dst + offsetd);

    int r0   = get_group_id(0) * MROW + get_local_id(1);  // output row
    int r1   = get_group_id(1);                            // token (ne11)
    int im   = get_group_id(2);
    int lid  = get_sub_group_local_id();                   // 0..63
    int nsg  = get_local_size(1);                          // == MROW

    int i12 = im % ne12;
    int i13 = im / ne12;

    ulong offset_src1 = r1*nb11 + (i12)*nb12 + (i13)*nb13;
    global float * y = (global float *) (src1 + offset_src1);

    // Cooperatively stage the activation column (ne00 floats) into __local once per
    // WG and reuse across the MROW rows. Staging is the actual win here: dropping it
    // (each subgroup re-reading y from global) regresses below the 1-row kernel.
    for (int i = get_local_id(1)*get_sub_group_size() + lid; i < ne00; i += nsg*get_sub_group_size()) {
        ysh[i] = y[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (r0 >= ne01) {
        return;
    }

    ulong offset_src0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    global half * x = (global half *) (src0 + offset_src0);

    // The vector path below casts the row pointer to half4, which must be 8-byte aligned.
    // A row address is r0*nb01 + ..., and a permuted or strided src0 leaves nb01/nb02/nb03
    // unconstrained -- ne00 % 4 == 0 bounds the element count per row, not the byte stride
    // between rows. Take the vector path only when this work-item's row is actually
    // aligned; the scalar loop below has no such requirement.
    const bool row_aligned = (((ulong) x) & 7) == 0;

    float sumf = 0.0f;
    if (ne00 < 128 || !row_aligned) {
        for (int i = lid; i < ne00; i += get_sub_group_size()) {
            sumf += (float) x[i] * ysh[i];
        }
        float all_sum = sub_group_reduce_add(sumf);
        if (lid == 0) {
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
    } else {
        global half4 * x4 = (global half4 *) x;
        __local float4 * ysh4 = (__local float4 *) ysh;
        for (int i = lid; i < ne00/4; i += get_sub_group_size()) {
            float4 yv = ysh4[i];
            sumf += (float) x4[i].s0 * yv.s0;
            sumf += (float) x4[i].s1 * yv.s1;
            sumf += (float) x4[i].s2 * yv.s2;
            sumf += (float) x4[i].s3 * yv.s3;
        }
        float all_sum = sub_group_reduce_add(sumf);
        if (lid == 0) {
            for (int i = 4*(ne00/4); i < ne00; ++i) {
                all_sum += (float) x[i] * ysh[i];
            }
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
    }
}

// Register-blocked variant: each 64-lane subgroup accumulates RPT consecutive
// output rows instead of one. The staged activation is reused across all RPT rows,
// and each lane keeps RPT independent weight loads in flight per column step ->
// more memory-level parallelism on the streaming f16 weight read (the BW limiter),
// plus RPT fewer staging barriers per output row. Per-row reduction order is
// identical to _mrow, so byte-identical to the per-op path. Dispatch guarantees
// ne00 >= 128 and ne00 % 4 == 0, so only the half4 path is needed (no tail).
#define MROW_RB_BODY(RPT)                                                                 \
    src0 = (global char*)((global char*)src0 + offset0);                                  \
    src1 = (global char*)((global char*)src1 + offset1);                                  \
    dst  = (global float*)((global char*)dst + offsetd);                                  \
    int r0b = (get_group_id(0) * get_local_size(1) + get_local_id(1)) * (RPT);            \
    int r1  = get_group_id(1);                                                            \
    int im  = get_group_id(2);                                                            \
    int lid = get_sub_group_local_id();                                                   \
    int nsg = get_local_size(1);                                                          \
    int i12 = im % ne12;                                                                  \
    int i13 = im / ne12;                                                                  \
    ulong off_y = r1*nb11 + i12*nb12 + i13*nb13;                                          \
    global float * y = (global float *) (src1 + off_y);                                   \
    for (int i = get_local_id(1)*get_sub_group_size() + lid; i < ne00;                    \
         i += nsg*get_sub_group_size()) {                                                 \
        ysh[i] = y[i];                                                                    \
    }                                                                                     \
    barrier(CLK_LOCAL_MEM_FENCE);                                                         \
    __local float4 * ysh4 = (__local float4 *) ysh;                                       \
    global half4 * xr[RPT];                                                               \
    _Pragma("unroll")                                                                     \
    for (int rr = 0; rr < (RPT); ++rr) {                                                  \
        int row = r0b + rr;                                                               \
        if (row > ne01 - 1) row = ne01 - 1;                                               \
        xr[rr] = (global half4 *) (src0 + (ulong)row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03); \
    }                                                                                     \
    float sumf[RPT];                                                                      \
    _Pragma("unroll")                                                                     \
    for (int rr = 0; rr < (RPT); ++rr) sumf[rr] = 0.0f;                                   \
    for (int i = lid; i < ne00/4; i += get_sub_group_size()) {                            \
        float4 yv = ysh4[i];                                                              \
        _Pragma("unroll")                                                                 \
        for (int rr = 0; rr < (RPT); ++rr) {                                              \
            half4 xv = xr[rr][i];                                                         \
            sumf[rr] += (float) xv.s0 * yv.s0 + (float) xv.s1 * yv.s1                      \
                      + (float) xv.s2 * yv.s2 + (float) xv.s3 * yv.s3;                     \
        }                                                                                 \
    }                                                                                     \
    _Pragma("unroll")                                                                     \
    for (int rr = 0; rr < (RPT); ++rr) {                                                  \
        float s = sub_group_reduce_add(sumf[rr]);                                         \
        int row = r0b + rr;                                                               \
        if (lid == 0 && row < ne01) {                                                     \
            dst[im*ne1*ne0 + r1*ne0 + row] = s;                                           \
        }                                                                                 \
    }

// half8 (128-bit) load variant: Adreno's load/store unit issues 128-bit
// transactions, so half4 (64-bit) loads may leave the load path half-idle. This
// processes 8 weight elements per lane per step via half8. Accumulation groups
// elements in 8s rather than 4s, so it is NOT bit-identical to _1row (float add is
// non-associative) -- experimental BW probe, gate on ne00 % 8 == 0.
#define MROW_H8_BODY(RPT)                                                                 \
    src0 = (global char*)((global char*)src0 + offset0);                                  \
    src1 = (global char*)((global char*)src1 + offset1);                                  \
    dst  = (global float*)((global char*)dst + offsetd);                                  \
    int r0b = (get_group_id(0) * get_local_size(1) + get_local_id(1)) * (RPT);            \
    int r1  = get_group_id(1);                                                            \
    int im  = get_group_id(2);                                                            \
    int lid = get_sub_group_local_id();                                                   \
    int nsg = get_local_size(1);                                                          \
    int i12 = im % ne12;                                                                  \
    int i13 = im / ne12;                                                                  \
    ulong off_y = r1*nb11 + i12*nb12 + i13*nb13;                                          \
    global float * y = (global float *) (src1 + off_y);                                   \
    for (int i = get_local_id(1)*get_sub_group_size() + lid; i < ne00;                    \
         i += nsg*get_sub_group_size()) {                                                 \
        ysh[i] = y[i];                                                                    \
    }                                                                                     \
    barrier(CLK_LOCAL_MEM_FENCE);                                                         \
    __local float4 * ysh4 = (__local float4 *) ysh;                                       \
    global half8 * xr[RPT];                                                               \
    _Pragma("unroll")                                                                     \
    for (int rr = 0; rr < (RPT); ++rr) {                                                  \
        int row = r0b + rr;                                                               \
        if (row > ne01 - 1) row = ne01 - 1;                                               \
        xr[rr] = (global half8 *) (src0 + (ulong)row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03); \
    }                                                                                     \
    float sumf[RPT];                                                                      \
    _Pragma("unroll")                                                                     \
    for (int rr = 0; rr < (RPT); ++rr) sumf[rr] = 0.0f;                                   \
    for (int i = lid; i < ne00/8; i += get_sub_group_size()) {                            \
        float4 y0 = ysh4[2*i];                                                            \
        float4 y1 = ysh4[2*i + 1];                                                        \
        _Pragma("unroll")                                                                 \
        for (int rr = 0; rr < (RPT); ++rr) {                                              \
            half8 xv = xr[rr][i];                                                         \
            sumf[rr] += (float) xv.s0 * y0.s0 + (float) xv.s1 * y0.s1                      \
                      + (float) xv.s2 * y0.s2 + (float) xv.s3 * y0.s3                      \
                      + (float) xv.s4 * y1.s0 + (float) xv.s5 * y1.s1                      \
                      + (float) xv.s6 * y1.s2 + (float) xv.s7 * y1.s3;                     \
        }                                                                                 \
    }                                                                                     \
    _Pragma("unroll")                                                                     \
    for (int rr = 0; rr < (RPT); ++rr) {                                                  \
        float s = sub_group_reduce_add(sumf[rr]);                                         \
        int row = r0b + rr;                                                               \
        if (lid == 0 && row < ne01) {                                                     \
            dst[im*ne1*ne0 + r1*ne0 + row] = s;                                           \
        }                                                                                 \
    }

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_mrow_h8(
        global char * src0, ulong offset0,
        global char * src1, ulong offset1,
        global float * dst, ulong offsetd,
        int ne00, int ne01, int ne02,
        ulong nb00, ulong nb01, ulong nb02, ulong nb03,
        int ne10, int ne11, int ne12,
        ulong nb10, ulong nb11, ulong nb12, ulong nb13,
        int ne0, int ne1, int r2, int r3,
        __local float * ysh
) {
    MROW_H8_BODY(1)
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_mrow_h8r2(
        global char * src0, ulong offset0,
        global char * src1, ulong offset1,
        global float * dst, ulong offsetd,
        int ne00, int ne01, int ne02,
        ulong nb00, ulong nb01, ulong nb02, ulong nb03,
        int ne10, int ne11, int ne12,
        ulong nb10, ulong nb11, ulong nb12, ulong nb13,
        int ne0, int ne1, int r2, int r3,
        __local float * ysh
) {
    MROW_H8_BODY(2)
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_mrow_r2(
        global char * src0, ulong offset0,
        global char * src1, ulong offset1,
        global float * dst, ulong offsetd,
        int ne00, int ne01, int ne02,
        ulong nb00, ulong nb01, ulong nb02, ulong nb03,
        int ne10, int ne11, int ne12,
        ulong nb10, ulong nb11, ulong nb12, ulong nb13,
        int ne0, int ne1, int r2, int r3,
        __local float * ysh
) {
    MROW_RB_BODY(2)
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_mrow_r4(
        global char * src0, ulong offset0,
        global char * src1, ulong offset1,
        global float * dst, ulong offsetd,
        int ne00, int ne01, int ne02,
        ulong nb00, ulong nb01, ulong nb02, ulong nb03,
        int ne10, int ne11, int ne12,
        ulong nb10, ulong nb11, ulong nb12, ulong nb13,
        int ne0, int ne1, int r2, int r3,
        __local float * ysh
) {
    MROW_RB_BODY(4)
}
