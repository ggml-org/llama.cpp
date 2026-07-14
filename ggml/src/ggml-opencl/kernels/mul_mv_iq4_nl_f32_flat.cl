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

#define QK4_NL 32

typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;

constant float kvalues_iq4nl[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
      1.f,   13.f,  25.f,  38.f,  53.f,  69.f,  89.f, 113.f
};

//------------------------------------------------------------------------------
// block_iq4_nl
//------------------------------------------------------------------------------
struct block_iq4_nl
{
    half d;
    uint8_t qs[QK4_NL / 2];
};

// Compute dot product between half a block of iq4_nl quants and activations.
// x points to the quant bytes, dh points to the scale.
// yl has 16 activation values: [0..7] for low nibbles, [8..15] for high nibbles.
// il indicates offset into the quant bytes (0 or 8).
inline float block_iq4_nl_dot_y_flat(
        global uchar * x,
        global half  * dh,
        private float * yl,
        int il
) {
    float d = *dh;
    global uchar * qs = x + il;
    float acc = 0.f;
    for (int i = 0; i < 8; ++i) {
        acc += yl[i]   * kvalues_iq4nl[qs[i] & 0x0F];
        acc += yl[i+8] * kvalues_iq4nl[qs[i] >> 4];
    }
    return d * acc;
}

#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define N_DST 8 // each subgroup works on 8 rows
#define N_SUBGROUP 1 // number of subgroups in a thread group
#define N_SUBGROUP_SIZE 16 // assuming subgroup size is 16
#elif defined (ADRENO_GPU)
#define N_DST 8
#define N_SUBGROUP 1
#define N_SUBGROUP_SIZE 64
#endif

inline void mul_vec_q_n_f32_8x_flat(
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
    const ulong nb = ne00/QK4_NL;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int first_row = (r0 * N_SUBGROUP + get_sub_group_id()) * N_DST;

    int i12 = im%ne12;
    int i13 = im/ne12;

    // The number of scales is the same as the number of blocks.
    ulong offset0_d = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    // Each block contains QK4_NL/2 uchars, hence offset for qs is as follows.
    ulong offset0_q = (first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02)) * QK4_NL/2;

    global uchar * x = (global uchar *) src0_q + offset0_q;
    global half  * d = (global half  *) src0_d + offset0_d;
    global float * y = (global float *) src1   + r1*ne10 + im*ne00*ne1;

    float yl[16];
    float8 sumf = 0.f;

    int ix = get_sub_group_local_id()/2;
    int il = 8*(get_sub_group_local_id()%2);

    global float * yb = y + ix*QK4_NL + il;

    for (int ib = ix; ib < nb; ib += N_SUBGROUP_SIZE/2) {
        for (int i = 0; i < 8; ++i) {
            yl[i]   = yb[i];
            yl[i+8] = yb[i+16];
        }

        sumf.s0 += block_iq4_nl_dot_y_flat(x + ib*QK4_NL/2 + 0*nb*QK4_NL/2, d + ib + 0*nb, yl, il);
        sumf.s1 += block_iq4_nl_dot_y_flat(x + ib*QK4_NL/2 + 1*nb*QK4_NL/2, d + ib + 1*nb, yl, il);
        sumf.s2 += block_iq4_nl_dot_y_flat(x + ib*QK4_NL/2 + 2*nb*QK4_NL/2, d + ib + 2*nb, yl, il);
        sumf.s3 += block_iq4_nl_dot_y_flat(x + ib*QK4_NL/2 + 3*nb*QK4_NL/2, d + ib + 3*nb, yl, il);

        sumf.s4 += block_iq4_nl_dot_y_flat(x + ib*QK4_NL/2 + 4*nb*QK4_NL/2, d + ib + 4*nb, yl, il);
        sumf.s5 += block_iq4_nl_dot_y_flat(x + ib*QK4_NL/2 + 5*nb*QK4_NL/2, d + ib + 5*nb, yl, il);
        sumf.s6 += block_iq4_nl_dot_y_flat(x + ib*QK4_NL/2 + 6*nb*QK4_NL/2, d + ib + 6*nb, yl, il);
        sumf.s7 += block_iq4_nl_dot_y_flat(x + ib*QK4_NL/2 + 7*nb*QK4_NL/2, d + ib + 7*nb, yl, il);

        yb += QK4_NL * (N_SUBGROUP_SIZE/2);
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
kernel void kernel_mul_mv_iq4_nl_f32_flat(
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

    mul_vec_q_n_f32_8x_flat(src0_q, src0_d, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3);
}
