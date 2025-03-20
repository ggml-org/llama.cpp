//------------------------------------------------------------------------------
// This file is contains additional mulmat kernels
// (and potentially other kernels).
//------------------------------------------------------------------------------
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supportedby OpenCL implementation on your device."
#endif

#ifdef cl_khr_subgroups
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#elif defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#error "Subgroup not supported on your device."
#endif

#ifdef cl_intel_required_subgroup_size
// Always use subgroup size of 32 on Intel.
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
// Always use subgroups size of 64 on Adreno.
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#else
// TODO: do not know how to choose subgroup size on other GPUs.
#error "Selecting subgroup size is not supported on your device."
#endif

#define QK4_0                   32
#define QR4_0                   2
#define QK4_1                   32
#define QR4_1                   2
#define QK5_0                   32
#define QR5_0                   2
#define QK5_1                   32
#define QR5_1                   2
#define QK8_0                   32
#define QR8_0                   1
#define QK_K                    256
#define K_QUANTS_PER_ITERATION  2

typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;

//------------------------------------------------------------------------------
// block_q4_0
//------------------------------------------------------------------------------
struct block_q4_0
{
    half d;
    uint8_t qs[QK4_0 / 2];
};

//------------------------------------------------------------------------------
// block_q6_K
//------------------------------------------------------------------------------
// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    half d;             // super-block scale
} block_q6_K;

//------------------------------------------------------------------------------
// These are the variant for matmatmul, based on the matvecmul kernel with
// flattened block_q4_0.
//------------------------------------------------------------------------------

// Common dot prod.
inline float mm_block_q_4_0_dot_y_flat(
        global uchar * x,
        global half  * dh,
        float sumy,
        float16 yl,
        int il
) {
    float           d   = *dh;
    global ushort * qs  = ((global ushort *)x + il/2);
    float           acc = 0.f;

    acc += yl.s0 * (qs[0] & 0x000F);
    acc += yl.s1 * (qs[0] & 0x0F00);
    acc += yl.s8 * (qs[0] & 0x00F0);
    acc += yl.s9 * (qs[0] & 0xF000);

    acc += yl.s2 * (qs[1] & 0x000F);
    acc += yl.s3 * (qs[1] & 0x0F00);
    acc += yl.sa * (qs[1] & 0x00F0);
    acc += yl.sb * (qs[1] & 0xF000);

    acc += yl.s4 * (qs[2] & 0x000F);
    acc += yl.s5 * (qs[2] & 0x0F00);
    acc += yl.sc * (qs[2] & 0x00F0);
    acc += yl.sd * (qs[2] & 0xF000);

    acc += yl.s6 * (qs[3] & 0x000F);
    acc += yl.s7 * (qs[3] & 0x0F00);
    acc += yl.se * (qs[3] & 0x00F0);
    acc += yl.sf * (qs[3] & 0xF000);

    return d * (sumy * -8.f + acc);
}

#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define N_DST 8 // each SIMD group works on 8 rows (in weights matrix)
#define N_SIMDGROUP 1 // number of SIMD groups in a thread group
#define N_SIMDWIDTH 16 // assuming SIMD group size is 16
#elif defined (ADRENO_GPU)
#define N_DST 8
#define N_SIMDGROUP 1
#define N_SIMDWIDTH 64
#endif
//
// This variant performs 1d blocking with 8x output.
// Eeach simdgroup outputs 8 values on `n0` dim (row in the output matrix).
//
inline void mul_mat_q_n_f32_1d_8x_flat(
        global uchar * src0_q,
        global half  * src0_d,
        global float * src1,
        global float * dst,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    const int nb = ne00/QK4_0;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    // (r0 * N_SIMDGROUP + get_sub_group_id()) is the linear global id of
    // a SIMD group in the grid. Each SIMD group produces N_DST values in the
    // result, hence uses nb blocks, i.e., the offset becomes first_row*nb.
    // Currently with llama2 7B, im is always 0.
    // TODO: how to handle im/gqa*(nb*ne0)?
    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    int i12 = im%ne12;
    int i13 = im/ne12;

    // The number of scales is the same as the number of blocks.
    ulong offset0_d = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    // Each block contains QK4_0/2 uchars, hence offset for qs is as follows.
    ulong offset0_q = (first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02)) * QK4_0/2;

    global uchar * x = (global uchar *) src0_q + offset0_q;
    global half  * d = (global half  *) src0_d + offset0_d;
    global float * y = (global float *) src1   + r1*ne10 + im*ne00*ne1;

    float16 yl;
    float8 sumf = (float8)(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);

    int ix = get_sub_group_local_id()/2;
    int il = 8*(get_sub_group_local_id()%2);

    global float * yb = y + ix*QK4_0 + il;

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/2) {
        float sumy = 0.f;

        sumy += yb[0];
        sumy += yb[1];
        sumy += yb[2];
        sumy += yb[3];
        sumy += yb[4];
        sumy += yb[5];
        sumy += yb[6];
        sumy += yb[7];

        sumy += yb[16];
        sumy += yb[17];
        sumy += yb[18];
        sumy += yb[19];
        sumy += yb[20];
        sumy += yb[21];
        sumy += yb[22];
        sumy += yb[23];

        yl.s0 = yb[0];
        yl.s1 = yb[1]/256.f;

        yl.s2 = yb[2];
        yl.s3 = yb[3]/256.f;

        yl.s4 = yb[4];
        yl.s5 = yb[5]/256.f;

        yl.s6 = yb[6];
        yl.s7 = yb[7]/256.f;

        yl.s8 = yb[16]/16.f;
        yl.s9 = yb[17]/4096.f;

        yl.sa = yb[18]/16.f;
        yl.sb = yb[19]/4096.f;

        yl.sc = yb[20]/16.f;
        yl.sd = yb[21]/4096.f;

        yl.se = yb[22]/16.f;
        yl.sf = yb[23]/4096.f;

        sumf.s0 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 0*nb*QK4_0/2, d + ib + 0*nb, sumy, yl, il);
        sumf.s1 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 1*nb*QK4_0/2, d + ib + 1*nb, sumy, yl, il);
        sumf.s2 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 2*nb*QK4_0/2, d + ib + 2*nb, sumy, yl, il);
        sumf.s3 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 3*nb*QK4_0/2, d + ib + 3*nb, sumy, yl, il);

        sumf.s4 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 4*nb*QK4_0/2, d + ib + 4*nb, sumy, yl, il);
        sumf.s5 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 5*nb*QK4_0/2, d + ib + 5*nb, sumy, yl, il);
        sumf.s6 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 6*nb*QK4_0/2, d + ib + 6*nb, sumy, yl, il);
        sumf.s7 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 7*nb*QK4_0/2, d + ib + 7*nb, sumy, yl, il);

        yb += QK4_0 * (N_SIMDWIDTH/2);
    }

    float8 tot = (float8)(
        sub_group_reduce_add(sumf.s0), sub_group_reduce_add(sumf.s1),
        sub_group_reduce_add(sumf.s2), sub_group_reduce_add(sumf.s3),
        sub_group_reduce_add(sumf.s4), sub_group_reduce_add(sumf.s5),
        sub_group_reduce_add(sumf.s6), sub_group_reduce_add(sumf.s7)
    );

    if (get_sub_group_local_id() == 0) {
        if (first_row + 0 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 0] = tot.s0;
        }
        if (first_row + 1 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 1] = tot.s1;
        }
        if (first_row + 2 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 2] = tot.s2;
        }
        if (first_row + 3 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 3] = tot.s3;
        }

        if (first_row + 4 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 4] = tot.s4;
        }
        if (first_row + 5 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 5] = tot.s5;
        }
        if (first_row + 6 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 6] = tot.s6;
        }
        if (first_row + 7 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 7] = tot.s7;
        }
    }
}

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_0_f32_1d_8x_flat(
        global uchar * src0_q,
        global half  * src0_d,
        global float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    mul_mat_q_n_f32_1d_8x_flat(src0_q, src0_d, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3);
}

#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define N_DST 16 // each SIMD group works on 8 rows (in weights matrix)
#define N_SIMDGROUP 1 // number of SIMD groups in a thread group
#define N_SIMDWIDTH 16 // assuming SIMD group size is 16
#elif defined (ADRENO_GPU)
#define N_DST 16
#define N_SIMDGROUP 1
#define N_SIMDWIDTH 64
#endif
//
// This variant performs 1d blocking with 16x output.
// Eeach simdgroup outputs 16 values on `n0` dim (row in the output matrix).
//
inline void mul_mat_q_n_f32_1d_16x_flat(
        global uchar * src0_q,
        global half  * src0_d,
        global float * src1,
        global float * dst,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    const int nb = ne00/QK4_0;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    // (r0 * N_SIMDGROUP + get_sub_group_id()) is the linear global id of
    // a SIMD group in the grid. Each SIMD group produces N_DST values in the
    // result, hence uses nb blocks, i.e., the offset becomes first_row*nb.
    // Currently with llama2 7B, im is always 0.
    // TODO: how to handle im/gqa*(nb*ne0)?
    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    int i12 = im%ne12;
    int i13 = im/ne12;

    // The number of scales is the same as the number of blocks.
    ulong offset0_d = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    // Each block contains QK4_0/2 uchars, hence offset for qs is as follows.
    ulong offset0_q = (first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02)) * QK4_0/2;

    global uchar * x = (global uchar *) src0_q + offset0_q;
    global half  * d = (global half  *) src0_d + offset0_d;
    global float * y = (global float *) src1   + r1*ne10 + im*ne00*ne1;

    float16 yl;
    float16 sumf = (float16)(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                             0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);

    int ix = get_sub_group_local_id()/2;
    int il = 8*(get_sub_group_local_id()%2);

    global float * yb = y + ix*QK4_0 + il;

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/2) {
        float sumy = 0.f;

        sumy += yb[0];
        sumy += yb[1];
        sumy += yb[2];
        sumy += yb[3];
        sumy += yb[4];
        sumy += yb[5];
        sumy += yb[6];
        sumy += yb[7];

        sumy += yb[16];
        sumy += yb[17];
        sumy += yb[18];
        sumy += yb[19];
        sumy += yb[20];
        sumy += yb[21];
        sumy += yb[22];
        sumy += yb[23];

        yl.s0 = yb[0];
        yl.s1 = yb[1]/256.f;

        yl.s2 = yb[2];
        yl.s3 = yb[3]/256.f;

        yl.s4 = yb[4];
        yl.s5 = yb[5]/256.f;

        yl.s6 = yb[6];
        yl.s7 = yb[7]/256.f;

        yl.s8 = yb[16]/16.f;
        yl.s9 = yb[17]/4096.f;

        yl.sa = yb[18]/16.f;
        yl.sb = yb[19]/4096.f;

        yl.sc = yb[20]/16.f;
        yl.sd = yb[21]/4096.f;

        yl.se = yb[22]/16.f;
        yl.sf = yb[23]/4096.f;

        sumf.s0 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 +  0*nb*QK4_0/2, d + ib +  0*nb, sumy, yl, il);
        sumf.s1 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 +  1*nb*QK4_0/2, d + ib +  1*nb, sumy, yl, il);
        sumf.s2 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 +  2*nb*QK4_0/2, d + ib +  2*nb, sumy, yl, il);
        sumf.s3 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 +  3*nb*QK4_0/2, d + ib +  3*nb, sumy, yl, il);

        sumf.s4 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 +  4*nb*QK4_0/2, d + ib +  4*nb, sumy, yl, il);
        sumf.s5 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 +  5*nb*QK4_0/2, d + ib +  5*nb, sumy, yl, il);
        sumf.s6 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 +  6*nb*QK4_0/2, d + ib +  6*nb, sumy, yl, il);
        sumf.s7 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 +  7*nb*QK4_0/2, d + ib +  7*nb, sumy, yl, il);

        sumf.s8 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 +  8*nb*QK4_0/2, d + ib +  8*nb, sumy, yl, il);
        sumf.s9 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 +  9*nb*QK4_0/2, d + ib +  9*nb, sumy, yl, il);
        sumf.sa += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 10*nb*QK4_0/2, d + ib + 10*nb, sumy, yl, il);
        sumf.sb += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 11*nb*QK4_0/2, d + ib + 11*nb, sumy, yl, il);

        sumf.sc += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 12*nb*QK4_0/2, d + ib + 12*nb, sumy, yl, il);
        sumf.sd += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 13*nb*QK4_0/2, d + ib + 13*nb, sumy, yl, il);
        sumf.se += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 14*nb*QK4_0/2, d + ib + 14*nb, sumy, yl, il);
        sumf.sf += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 15*nb*QK4_0/2, d + ib + 15*nb, sumy, yl, il);

        yb += QK4_0 * (N_SIMDWIDTH/2);
    }

    float16 tot = (float16)(
        sub_group_reduce_add(sumf.s0), sub_group_reduce_add(sumf.s1),
        sub_group_reduce_add(sumf.s2), sub_group_reduce_add(sumf.s3),
        sub_group_reduce_add(sumf.s4), sub_group_reduce_add(sumf.s5),
        sub_group_reduce_add(sumf.s6), sub_group_reduce_add(sumf.s7),

        sub_group_reduce_add(sumf.s8), sub_group_reduce_add(sumf.s9),
        sub_group_reduce_add(sumf.sa), sub_group_reduce_add(sumf.sb),
        sub_group_reduce_add(sumf.sc), sub_group_reduce_add(sumf.sd),
        sub_group_reduce_add(sumf.se), sub_group_reduce_add(sumf.sf)
    );

    if (get_sub_group_local_id() == 0) {
        if (first_row + 0 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 0] = tot.s0;
        }
        if (first_row + 1 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 1] = tot.s1;
        }
        if (first_row + 2 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 2] = tot.s2;
        }
        if (first_row + 3 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 3] = tot.s3;
        }

        if (first_row + 4 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 4] = tot.s4;
        }
        if (first_row + 5 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 5] = tot.s5;
        }
        if (first_row + 6 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 6] = tot.s6;
        }
        if (first_row + 7 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 7] = tot.s7;
        }

        if (first_row + 8 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 8] = tot.s8;
        }
        if (first_row + 9 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 9] = tot.s9;
        }
        if (first_row + 10 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 10] = tot.sa;
        }
        if (first_row + 11 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 11] = tot.sb;
        }

        if (first_row + 12 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 12] = tot.sc;
        }
        if (first_row + 13 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 13] = tot.sd;
        }
        if (first_row + 14 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 14] = tot.se;
        }
        if (first_row + 15 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 15] = tot.sf;
        }
    }
}

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_0_f32_1d_16x_flat(
        global uchar * src0_q,
        global half  * src0_d,
        global float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    mul_mat_q_n_f32_1d_16x_flat(src0_q, src0_d, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3);
}

//------------------------------------------------------------------------------
// kernel_mul_mat_q4_0_f32_flat_v0
//------------------------------------------------------------------------------
inline float block_q_4_0_dot_y_flat_v2(
    half   x,
    half   d,
    float  sumy,
    float4 yl
) {
    uchar2 q = as_uchar2(x);
    float acc = 0.0f;

    acc += (q.s0 & 0x0F) * yl.s0;
    acc += (q.s1 & 0x0F) * yl.s1;

    acc += (q.s0 & 0xF0) * yl.s2;
    acc += (q.s1 & 0xF0) * yl.s3;

    return d * (sumy * -8.f + acc);;
}

inline float block_q_4_0_dot_y_flat_v4(
    float  x,
    half   d,
    float  sumy,
    float8 yl
) {
    uchar4 q = as_uchar4(x);
    float acc = 0.0f;

    acc += (q.s0 & 0x0F) * yl.s0;
    acc += (q.s1 & 0x0F) * yl.s1;
    acc += (q.s2 & 0x0F) * yl.s2;
    acc += (q.s3 & 0x0F) * yl.s3;

    acc += (q.s0 & 0xF0) * yl.s4;
    acc += (q.s1 & 0xF0) * yl.s5;
    acc += (q.s2 & 0xF0) * yl.s6;
    acc += (q.s3 & 0xF0) * yl.s7;

    return d * (sumy * -8.f + acc);;
}

inline float block_q_4_0_dot_y_flat_v8(
    float2  x,
    half    d,
    float   sumy,
    float16 yl
) {
    uchar8 q = as_uchar8(x);
    float acc = 0.0f;

    acc += (q.s0 & 0x0F) * yl.s0;
    acc += (q.s1 & 0x0F) * yl.s1;
    acc += (q.s2 & 0x0F) * yl.s2;
    acc += (q.s3 & 0x0F) * yl.s3;
    acc += (q.s4 & 0x0F) * yl.s4;
    acc += (q.s5 & 0x0F) * yl.s5;
    acc += (q.s6 & 0x0F) * yl.s6;
    acc += (q.s7 & 0x0F) * yl.s7;

    acc += (q.s0 & 0xF0) * yl.s8;
    acc += (q.s1 & 0xF0) * yl.s9;
    acc += (q.s2 & 0xF0) * yl.sa;
    acc += (q.s3 & 0xF0) * yl.sb;
    acc += (q.s4 & 0xF0) * yl.sc;
    acc += (q.s5 & 0xF0) * yl.sd;
    acc += (q.s6 & 0xF0) * yl.se;
    acc += (q.s7 & 0xF0) * yl.sf;

    return d * (sumy * -8.f + acc);;
}

#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define THREADS_PER_BLK 4   // Number of threads per block, or each thread process 1/THREADS_PER_BLK of a block
#define N_DST           4
#define N_SIMDGROUP     1
#define N_SIMDWIDTH     16
#elif defined (ADRENO_GPU)
#define THREADS_PER_BLK 4
#define N_DST           4
#define N_SIMDGROUP     1
#define N_SIMDWIDTH     64
#endif

#if THREADS_PER_BLK == 2                // Each thread processes 1/2 block
#   define ACT_TY                       float16
#   define Q_BLK_LD_TY                  float2
#   define block_q_4_0_dot_y_flat       block_q_4_0_dot_y_flat_v8
#elif THREADS_PER_BLK == 4              // Each thread processes 1/4 block
#   define ACT_TY                       float8
#   define Q_BLK_LD_TY                  float
#   define block_q_4_0_dot_y_flat       block_q_4_0_dot_y_flat_v4
#elif THREADS_PER_BLK == 8              // Each thread processes 1/8 block
#   define ACT_TY                       float4
#   define Q_BLK_LD_TY                  half
#   define block_q_4_0_dot_y_flat       block_q_4_0_dot_y_flat_v2
#endif

#define BTYES_PER_THREAD_IN_BLK         (QK4_0/2/THREADS_PER_BLK)

#if N_DST == 2
#   define  SUM_TY                      float2
#elif N_DST == 4
#   define  SUM_TY                      float4
#elif N_DST == 8
#   define  SUM_TY                      float8
#elif N_DST == 16
#   define  SUM_TY                      float16
#endif

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_0_f32_flat_v0(
        global uchar * src0_q,
        global half  * src0_d,
        global float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    const int nb = ne00/QK4_0;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    int i12 = im%ne12;
    int i13 = im/ne12;

    // The number of scales is the same as the number of blocks.
    ulong offset0_d = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    // Each block contains QK4_0/2 uchars, hence offset for qs is as follows.
    ulong offset0_q = (first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02)) * QK4_0/2;

    global uchar * x = (global uchar *) src0_q + offset0_q;
    global half  * d = (global half  *) src0_d + offset0_d;
    global float * y = (global float *) src1   + r1*ne10 + im*ne00*ne1;

    int ix = get_sub_group_local_id()/THREADS_PER_BLK;
    int il = get_sub_group_local_id()%THREADS_PER_BLK;

    global float * yb = y + ix*QK4_0 + BTYES_PER_THREAD_IN_BLK*il;

    // Registers for caching activation
    ACT_TY yl = 0.f;

    // Registers for caching quants
    Q_BLK_LD_TY q_blk_0 = 0, q_blk_1 = 0;
#if N_DST == 4 || N_DST == 8 || N_DST == 16
    Q_BLK_LD_TY q_blk_2 = 0, q_blk_3 = 0;
#endif
#if N_DST == 8 || N_DST == 16
    Q_BLK_LD_TY q_blk_4 = 0, q_blk_5 = 0, q_blk_6 = 0, q_blk_7 = 0;
#endif

    // Partial sum
    SUM_TY sumf = 0.f;

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/THREADS_PER_BLK) {
        float sumy = 0.f;

        q_blk_0 = *(global Q_BLK_LD_TY*)(x + ib*QK4_0/2 + BTYES_PER_THREAD_IN_BLK*il + 0*nb*QK4_0/2);
        q_blk_1 = *(global Q_BLK_LD_TY*)(x + ib*QK4_0/2 + BTYES_PER_THREAD_IN_BLK*il + 1*nb*QK4_0/2);
#if N_DST == 4 || N_DST == 8 || N_DST == 16
        q_blk_2 = *(global Q_BLK_LD_TY*)(x + ib*QK4_0/2 + BTYES_PER_THREAD_IN_BLK*il + 2*nb*QK4_0/2);
        q_blk_3 = *(global Q_BLK_LD_TY*)(x + ib*QK4_0/2 + BTYES_PER_THREAD_IN_BLK*il + 3*nb*QK4_0/2);
#endif
#if N_DST == 8 || N_DST == 16
        q_blk_4 = (*(global Q_BLK_LD_TY*)(x + ib*QK4_0/2 + BTYES_PER_THREAD_IN_BLK*il + 4*nb*QK4_0/2));
        q_blk_5 = (*(global Q_BLK_LD_TY*)(x + ib*QK4_0/2 + BTYES_PER_THREAD_IN_BLK*il + 5*nb*QK4_0/2));
        q_blk_6 = (*(global Q_BLK_LD_TY*)(x + ib*QK4_0/2 + BTYES_PER_THREAD_IN_BLK*il + 6*nb*QK4_0/2));
        q_blk_7 = (*(global Q_BLK_LD_TY*)(x + ib*QK4_0/2 + BTYES_PER_THREAD_IN_BLK*il + 7*nb*QK4_0/2));
#endif

        // Load activation
#if THREADS_PER_BLK == 2    // Each thread processes 1/2 block
        yl.s01234567 = *(global float8 *)(yb);
        yl.s89abcdef = *(global float8 *)(yb + 16);

        sumy += yl.s0;
        sumy += yl.s1;
        sumy += yl.s2;
        sumy += yl.s3;
        sumy += yl.s4;
        sumy += yl.s5;
        sumy += yl.s6;
        sumy += yl.s7;
        sumy += yl.s8; yl.s8 /= 16.f;
        sumy += yl.s9; yl.s9 /= 16.f;
        sumy += yl.sa; yl.sa /= 16.f;
        sumy += yl.sb; yl.sb /= 16.f;
        sumy += yl.sc; yl.sc /= 16.f;
        sumy += yl.sd; yl.sd /= 16.f;
        sumy += yl.se; yl.se /= 16.f;
        sumy += yl.sf; yl.sf /= 16.f;
#elif THREADS_PER_BLK == 4  // Each thread processes 1/4 block
        yl.s0123 = *(global float4 *)(yb);
        yl.s4567 = *(global float4 *)(yb + 16);

        sumy += yl.s0;
        sumy += yl.s1;
        sumy += yl.s2;
        sumy += yl.s3;
        sumy += yl.s4; yl.s4 /= 16.f;
        sumy += yl.s5; yl.s5 /= 16.f;
        sumy += yl.s6; yl.s6 /= 16.f;
        sumy += yl.s7; yl.s7 /= 16.f;
#elif THREADS_PER_BLK == 8  // Each thread processes 1/8 block
        yl.s01 = *(global float2 *)(yb);
        yl.s23 = *(global float2 *)(yb + 16);

        sumy += yl.s0;
        sumy += yl.s1;
        sumy += yl.s2; yl.s2 /= 16.f;
        sumy += yl.s3; yl.s3 /= 16.f;
#endif

        sumf.s0 += block_q_4_0_dot_y_flat(q_blk_0, *(d + ib + 0*nb), sumy, yl);
        sumf.s1 += block_q_4_0_dot_y_flat(q_blk_1, *(d + ib + 1*nb), sumy, yl);
#if N_DST == 4 || N_DST == 8 || N_DST == 16
        sumf.s2 += block_q_4_0_dot_y_flat(q_blk_2, *(d + ib + 2*nb), sumy, yl);
        sumf.s3 += block_q_4_0_dot_y_flat(q_blk_3, *(d + ib + 3*nb), sumy, yl);
#endif
#if N_DST == 8 || N_DST == 16
        sumf.s4 += block_q_4_0_dot_y_flat(q_blk_4, *(d + ib + 4*nb), sumy, yl);
        sumf.s5 += block_q_4_0_dot_y_flat(q_blk_5, *(d + ib + 5*nb), sumy, yl);
        sumf.s6 += block_q_4_0_dot_y_flat(q_blk_6, *(d + ib + 6*nb), sumy, yl);
        sumf.s7 += block_q_4_0_dot_y_flat(q_blk_7, *(d + ib + 7*nb), sumy, yl);
#endif

        yb += QK4_0 * (N_SIMDWIDTH/THREADS_PER_BLK);
    }

    SUM_TY tot = (SUM_TY)(
          sub_group_reduce_add(sumf.s0), sub_group_reduce_add(sumf.s1)
#if N_DST == 4 || N_DST == 8 || N_DST == 16
        , sub_group_reduce_add(sumf.s2), sub_group_reduce_add(sumf.s3)
#endif
#if N_DST == 8 || N_DST == 16
        , sub_group_reduce_add(sumf.s4), sub_group_reduce_add(sumf.s5)
        , sub_group_reduce_add(sumf.s6), sub_group_reduce_add(sumf.s7)
#endif
    );

    if (get_sub_group_local_id() == 0) {
        if (first_row + 0 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 0] = tot.s0;
        }
        if (first_row + 1 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 1] = tot.s1;
        }
#if N_DST == 4 || N_DST == 8 || N_DST == 16
        if (first_row + 2 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 2] = tot.s2;
        }
        if (first_row + 3 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 3] = tot.s3;
        }
#endif
#if N_DST == 8 || N_DST == 16
        if (first_row + 4 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 4] = tot.s4;
        }
        if (first_row + 5 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 5] = tot.s5;
        }
        if (first_row + 6 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 6] = tot.s6;
        }
        if (first_row + 7 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 7] = tot.s7;
        }
#endif
    }
}

//------------------------------------------------------------------------------
// Using image1d_buffer_t

#if defined(cl_qcom_subgroup_shuffle)
#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
float qcom_sub_group_reduce_add(float sum) {
    sum += qcom_sub_group_shuffle_down(sum, 32, CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, 0.f);
    sum += qcom_sub_group_shuffle_down(sum, 16, CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, 0.f);
    sum += qcom_sub_group_shuffle_down(sum,  8, CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, 0.f);
    sum += qcom_sub_group_shuffle_down(sum,  4, CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, 0.f);
    sum += qcom_sub_group_shuffle_down(sum,  2, CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, 0.f);
    sum += qcom_sub_group_shuffle_down(sum,  1, CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, 0.f);
    return sum;
}
#define sub_group_reduce_add qcom_sub_group_reduce_add
#else
#define sub_group_reduce_add sub_group_reduce_add
#endif

#undef THREADS_PER_BLK
#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define THREADS_PER_BLK 4   // Number of threads per block, or each thread process 1/THREADS_PER_BLK of a block
#define N_DST           4
#define N_SIMDGROUP     1
#define N_SIMDWIDTH     16
#elif defined (ADRENO_GPU)
#define THREADS_PER_BLK 4
#define N_DST           4
#define N_SIMDGROUP     1
#define N_SIMDWIDTH     64
#endif

#if THREADS_PER_BLK == 2                // Each thread processes 1/2 block
#   define ACT_TY                       float16
#   define Q_BLK_LD_TY                  float2
#   define EXTRACT_BLK_DATA(tmp, part)  *((float2*)&tmp + part)
#   define block_q_4_0_dot_y_flat       block_q_4_0_dot_y_flat_v8
#elif THREADS_PER_BLK == 4              // Each thread processes 1/4 block
#   define ACT_TY                       float8
#   define Q_BLK_LD_TY                  float
#   define EXTRACT_BLK_DATA(tmp, part)  *((float*)&tmp + part)
#   define block_q_4_0_dot_y_flat       block_q_4_0_dot_y_flat_v4
#elif THREADS_PER_BLK == 8              // Each thread processes 1/8 block
#   define ACT_TY                       float4
#   define Q_BLK_LD_TY                  half
#   define EXTRACT_BLK_DATA(tmp, part)  *((half*)&tmp + part)
#   define block_q_4_0_dot_y_flat       block_q_4_0_dot_y_flat_v2
#endif

#define BTYES_PER_THREAD_IN_BLK         (QK4_0/2/THREADS_PER_BLK)

#if N_DST == 2
#   define  SUM_TY                      float2
#elif N_DST == 4
#   define  SUM_TY                      float4
#elif N_DST == 8
#   define  SUM_TY                      float8
#elif N_DST == 16
#   define  SUM_TY                      float16
#endif

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_0_f32_flat_img_v0(
        read_only image1d_buffer_t src0_q,
        read_only image1d_buffer_t src0_d,
        global float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    const int nb = ne00/QK4_0;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    int i12 = im%ne12;
    int i13 = im/ne12;

    // The number of scales is the same as the number of blocks.
    ulong offset0_d = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    // Each block contains QK4_0/2 uchars, hence offset for qs is as follows.
    ulong offset0_q = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    global float * y = (global float *) src1   + r1*ne10 + im*ne00*ne1;

    int ix = get_sub_group_local_id()/THREADS_PER_BLK;
    int il = get_sub_group_local_id()%THREADS_PER_BLK;

    global float * yb = y + ix*QK4_0 + BTYES_PER_THREAD_IN_BLK*il;

    // Registers for caching activation
    ACT_TY yl = 0.f;

    // Registers for caching quants
    Q_BLK_LD_TY q_blk_0 = 0, q_blk_1 = 0;
#if N_DST == 4 || N_DST == 8 || N_DST == 16
    Q_BLK_LD_TY q_blk_2 = 0, q_blk_3 = 0;
#endif
#if N_DST == 8 || N_DST == 16
    Q_BLK_LD_TY q_blk_4 = 0, q_blk_5 = 0, q_blk_6 = 0, q_blk_7 = 0;
#endif

    // Partial sum
    SUM_TY sumf = 0.f;

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/THREADS_PER_BLK) {
        float sumy = 0.f;;

        float4 tmp;
        tmp = read_imagef(src0_q, offset0_q + ib + 0*nb);
        q_blk_0 = EXTRACT_BLK_DATA(tmp, il);
        tmp = read_imagef(src0_q, offset0_q + ib + 1*nb);
        q_blk_1 = EXTRACT_BLK_DATA(tmp, il);
#if N_DST == 4 || N_DST == 8 || N_DST == 16
        tmp = read_imagef(src0_q, offset0_q + ib + 2*nb);
        q_blk_2 = EXTRACT_BLK_DATA(tmp, il);
        tmp = read_imagef(src0_q, offset0_q + ib + 3*nb);
        q_blk_3 = EXTRACT_BLK_DATA(tmp, il);
#endif
#if N_DST == 8 || N_DST == 16
        tmp = read_imagef(src0_q, offset0_q + ib + 4*nb);
        q_blk_4 = EXTRACT_BLK_DATA(tmp, il);
        tmp = read_imagef(src0_q, offset0_q + ib + 5*nb);
        q_blk_5 = EXTRACT_BLK_DATA(tmp, il);
        tmp = read_imagef(src0_q, offset0_q + ib + 6*nb);
        q_blk_6 = EXTRACT_BLK_DATA(tmp, il);
        tmp = read_imagef(src0_q, offset0_q + ib + 7*nb);
        q_blk_7 = EXTRACT_BLK_DATA(tmp, il);
#endif

        // Load activation
#if THREADS_PER_BLK == 2    // Each thread processes 1/2 block
        yl.s01234567 = *(global float8 *)(yb);
        yl.s89abcdef = *(global float8 *)(yb + 16);

        sumy += yl.s0;
        sumy += yl.s1;
        sumy += yl.s2;
        sumy += yl.s3;
        sumy += yl.s4;
        sumy += yl.s5;
        sumy += yl.s6;
        sumy += yl.s7;
        sumy += yl.s8; yl.s8 /= 16.f;
        sumy += yl.s9; yl.s9 /= 16.f;
        sumy += yl.sa; yl.sa /= 16.f;
        sumy += yl.sb; yl.sb /= 16.f;
        sumy += yl.sc; yl.sc /= 16.f;
        sumy += yl.sd; yl.sd /= 16.f;
        sumy += yl.se; yl.se /= 16.f;
        sumy += yl.sf; yl.sf /= 16.f;
#elif THREADS_PER_BLK == 4  // Each thread processes 1/4 block
        yl.s0123 = *(global float4 *)(yb);
        yl.s4567 = *(global float4 *)(yb + 16);

        sumy += yl.s0;
        sumy += yl.s1;
        sumy += yl.s2;
        sumy += yl.s3;
        sumy += yl.s4; yl.s4 /= 16.f;
        sumy += yl.s5; yl.s5 /= 16.f;
        sumy += yl.s6; yl.s6 /= 16.f;
        sumy += yl.s7; yl.s7 /= 16.f;
#elif THREADS_PER_BLK == 8  // Each thread processes 1/8 block
        yl.s01 = *(global float2 *)(yb);
        yl.s23 = *(global float2 *)(yb + 16);

        sumy += yl.s0;
        sumy += yl.s1;
        sumy += yl.s2; yl.s2 /= 16.f;
        sumy += yl.s3; yl.s3 /= 16.f;
#endif

        sumf.s0 += block_q_4_0_dot_y_flat(q_blk_0, read_imageh(src0_d, offset0_d + ib + 0*nb).s0, sumy, yl);
        sumf.s1 += block_q_4_0_dot_y_flat(q_blk_1, read_imageh(src0_d, offset0_d + ib + 1*nb).s0, sumy, yl);
#if N_DST == 4 || N_DST == 8 || N_DST == 16
        sumf.s2 += block_q_4_0_dot_y_flat(q_blk_2, read_imageh(src0_d, offset0_d + ib + 2*nb).s0, sumy, yl);
        sumf.s3 += block_q_4_0_dot_y_flat(q_blk_3, read_imageh(src0_d, offset0_d + ib + 3*nb).s0, sumy, yl);
#endif
#if N_DST == 8 || N_DST == 16
        sumf.s4 += block_q_4_0_dot_y_flat(q_blk_4, read_imageh(src0_d, offset0_d + ib + 4*nb).s0, sumy, yl);
        sumf.s5 += block_q_4_0_dot_y_flat(q_blk_5, read_imageh(src0_d, offset0_d + ib + 5*nb).s0, sumy, yl);
        sumf.s6 += block_q_4_0_dot_y_flat(q_blk_6, read_imageh(src0_d, offset0_d + ib + 6*nb).s0, sumy, yl);
        sumf.s7 += block_q_4_0_dot_y_flat(q_blk_7, read_imageh(src0_d, offset0_d + ib + 7*nb).s0, sumy, yl);
#endif

        yb += QK4_0 * (N_SIMDWIDTH/THREADS_PER_BLK);
    }

    SUM_TY tot = (SUM_TY)(
          sub_group_reduce_add(sumf.s0), sub_group_reduce_add(sumf.s1)
#if N_DST == 4 || N_DST == 8 || N_DST == 16
        , sub_group_reduce_add(sumf.s2), sub_group_reduce_add(sumf.s3)
#endif
#if N_DST == 8 || N_DST == 16
        , sub_group_reduce_add(sumf.s4), sub_group_reduce_add(sumf.s5)
        , sub_group_reduce_add(sumf.s6), sub_group_reduce_add(sumf.s7)
#endif
    );

    if (get_sub_group_local_id() == 0) {
        if (first_row + 0 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 0] = tot.s0;
        }
        if (first_row + 1 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 1] = tot.s1;
        }
#if N_DST == 4 || N_DST == 8 || N_DST == 16
        if (first_row + 2 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 2] = tot.s2;
        }
        if (first_row + 3 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 3] = tot.s3;
        }
#endif
#if N_DST == 8 || N_DST == 16
        if (first_row + 4 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 4] = tot.s4;
        }
        if (first_row + 5 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 5] = tot.s5;
        }
        if (first_row + 6 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 6] = tot.s6;
        }
        if (first_row + 7 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 7] = tot.s7;
        }
#endif
    }
}

//------------------------------------------------------------------------------
// kernel_mul_mv_q6_K_f32
//------------------------------------------------------------------------------

#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define N_DST 1 // number of rows each SIMD group works on
#define N_SIMDGROUP 2 // number of SIMD groups in a thread group
#define N_SIMDWIDTH 16 // SIMD group size
#elif defined (ADRENO_GPU)
#define N_DST 1
#define N_SIMDGROUP 2
#define N_SIMDWIDTH 64
#endif

#define BLOCK_STRIDE (N_SIMDWIDTH/16) // number of blocks each subgroup processes

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q6_K_f32(
        global void * src0,
        ulong offset0,
        global float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    uchar kmask1 = 0x03;
    uchar kmask2 = 0x0C;
    uchar kmask3 = 0x30;
    uchar kmask4 = 0xC0;

    int nb = ne00/QK_K;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int row = N_SIMDGROUP * r0 + get_sub_group_id();

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset_src0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    global block_q6_K * x = (global block_q6_K *) src0 + row*nb + offset_src0;
    global float      * yy = (global float     *) src1 + r1*ne10 + im*ne00*ne1;

    float sumf = 0;

    // For Q6_K quantization, 16 values forms a subblock, 16 subblock forms a
    // block. Values in a subblock shares a scale that is quantized with 8 bits;
    // the entire block shares a single floating point scale.
    // For work distribution, each thread processes a subblock (16 weights), hence
    // 16 threads process a (super) block -- a subgroup thus handles SIMDWIDTH/16
    // (super) blocks -- this is the block stride.
    // The 16 threads that process a (super) block are split into 2 portions, each has
    // 8 threads; each portion works on 8 subblocks.
    // For subgroup of 16 threads, the entire subgroup works on a single (super) block
    // before moving to the next (super) block. Thread0 - thread7 work on the
    // first 8 subblocks; thread8 - thread15 works on the last 8 subblocks.
    // Thread0 - thread3 work on subblocks 0, 2, 4, 6; thread4 - thread7 work on
    // subblocks 1, 3, 5, 7. Each thread does not work on an entire subblock, but
    // works on a total of 16 weight values.
    int tid  = get_sub_group_local_id()/BLOCK_STRIDE; // first block_stride groups have tid=0
    int ix   = get_sub_group_local_id()%BLOCK_STRIDE; // first block is 0..block_stride-1
    int ip   = tid/8;   // first or second half of (super) block (0 or 1)
    int il   = tid%8;   // each half has 8 parts, one per scale
    int n    = 4;       // 4 scales at a time (and 4 sums)
    int l0   = n*il;    // offset into half-block, 0..28
    int is   = 8*ip + l0/16; // 0, 1, 8, 9

    int y_offset = 128*ip + l0;
    int q_offset_l = 64*ip + l0;
    int q_offset_h = 32*ip + l0;

    for (int i = ix; i < nb; i += BLOCK_STRIDE) {

        global uint8_t * q1 = x[i].ql + q_offset_l;
        global uint8_t * q2 = q1 + QK_K/8;
        global uint8_t * qh = x[i].qh + q_offset_h;
        global int8_t  * sc = x[i].scales + is;

        global float * y = yy + i * QK_K + y_offset;

        float dall = x[i].d;

        float4 sums = {0.f, 0.f, 0.f, 0.f};

        sums.s0 += y[0+ 0] * ((float)((q1[0] & 0xF) | ((qh[0] & kmask1) << 4)) - 32.f);
        sums.s1 += y[0+32] * ((float)((q2[0] & 0xF) | ((qh[0] & kmask2) << 2)) - 32.f);
        sums.s2 += y[0+64] * ((float)((q1[0]  >> 4) | ((qh[0] & kmask3) << 0)) - 32.f);
        sums.s3 += y[0+96] * ((float)((q2[0]  >> 4) | ((qh[0] & kmask4) >> 2)) - 32.f);

        sums.s0 += y[1+ 0] * ((float)((q1[1] & 0xF) | ((qh[1] & kmask1) << 4)) - 32.f);
        sums.s1 += y[1+32] * ((float)((q2[1] & 0xF) | ((qh[1] & kmask2) << 2)) - 32.f);
        sums.s2 += y[1+64] * ((float)((q1[1]  >> 4) | ((qh[1] & kmask3) << 0)) - 32.f);
        sums.s3 += y[1+96] * ((float)((q2[1]  >> 4) | ((qh[1] & kmask4) >> 2)) - 32.f);

        sums.s0 += y[2+ 0] * ((float)((q1[2] & 0xF) | ((qh[2] & kmask1) << 4)) - 32.f);
        sums.s1 += y[2+32] * ((float)((q2[2] & 0xF) | ((qh[2] & kmask2) << 2)) - 32.f);
        sums.s2 += y[2+64] * ((float)((q1[2]  >> 4) | ((qh[2] & kmask3) << 0)) - 32.f);
        sums.s3 += y[2+96] * ((float)((q2[2]  >> 4) | ((qh[2] & kmask4) >> 2)) - 32.f);

        sums.s0 += y[3+ 0] * ((float)((q1[3] & 0xF) | ((qh[3] & kmask1) << 4)) - 32.f);
        sums.s1 += y[3+32] * ((float)((q2[3] & 0xF) | ((qh[3] & kmask2) << 2)) - 32.f);
        sums.s2 += y[3+64] * ((float)((q1[3]  >> 4) | ((qh[3] & kmask3) << 0)) - 32.f);
        sums.s3 += y[3+96] * ((float)((q2[3]  >> 4) | ((qh[3] & kmask4) >> 2)) - 32.f);

        sumf += dall * (sums.s0 * sc[0] + sums.s1 * sc[2] + sums.s2 * sc[4] + sums.s3 * sc[6]);
    }

    float tot = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id() == 0) {
        dst[r1*ne0 + im*ne0*ne1 + row] = tot;
    }
}
