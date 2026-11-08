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

//------------------------------------------------------------------------------
// mul_vec_q_n_f32
//------------------------------------------------------------------------------
// Compute inner product between half a block of iq4_nl and 16 floats (yl).
// il indicates where the quants begin (0 or 8).
inline float block_iq4_nl_dot_y(
        global struct block_iq4_nl * qb_curr,
        private float * yl,
        int il
) {
    float d = qb_curr->d;
    float acc = 0.f;
    global uchar * qs = qb_curr->qs + il;
    for (int i = 0; i < 8; ++i) {
        acc += yl[i]   * kvalues_iq4nl[qs[i] & 0x0F];
        acc += yl[i+8] * kvalues_iq4nl[qs[i] >> 4];
    }
    return d * acc;
}

#ifdef INTEL_GPU
#define N_DST 4 // each subgroup group works on 4 rows
#define N_SUBGROUP 1 // number of subgroups in a thread group
#define N_SUBGROUP_SIZE 16 // assuming subgroup size is 16
#elif defined (ADRENO_GPU)
#define N_DST 4
#define N_SUBGROUP 1
#define N_SUBGROUP_SIZE 64
#endif

inline void mul_vec_q_n_f32(
        global void * src0,
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

    ulong offset0 = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    global struct block_iq4_nl * x = (global struct block_iq4_nl *) src0 + offset0;
    global float               * y = (global float               *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[16];       // src1 vector cache
    float sumf[N_DST]={0.f};

    int ix = get_sub_group_local_id()/2;
    int il = 8*(get_sub_group_local_id()%2);

    global float * yb = y + ix * QK4_NL + il;

    // each thread in a SIMD group deals with half a block.
    for (int ib = ix; ib < nb; ib += N_SUBGROUP_SIZE/2) {
        for (int i = 0; i < 8; ++i) {
            yl[i]   = yb[i];
            yl[i+8] = yb[i+16];
        }

        for (int row = 0; row < N_DST; row++) {
            sumf[row] += block_iq4_nl_dot_y(x+ib+row*nb, yl, il);
        }

        yb += QK4_NL * (N_SUBGROUP_SIZE/2);
    }

    float tot[N_DST] = {
        sub_group_reduce_add(sumf[0]), sub_group_reduce_add(sumf[1]),
        sub_group_reduce_add(sumf[2]), sub_group_reduce_add(sumf[3])};
    for (int row = 0; row < N_DST; ++row) {
        if (get_sub_group_local_id() == 0 && first_row + row < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = tot[row];
        }
    }
}

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_iq4_nl_f32(
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

    mul_vec_q_n_f32(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3);
}
