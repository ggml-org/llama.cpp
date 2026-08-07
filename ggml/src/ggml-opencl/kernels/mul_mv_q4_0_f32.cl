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
// mul_vec_q_n_f32
//------------------------------------------------------------------------------
// function for calculate inner product between half a q4_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_4_0_dot_y(
        global struct block_q4_0 * qb_curr,
        float sumy,
        private float * yl,
        int il
) {
    float d = qb_curr->d;
    float2 acc = 0.f;
    global ushort * qs = ((global ushort *)qb_curr + 1 + il/2);
    for (int i = 0; i < 8; i+=2) {
        acc.s0 += yl[i + 0] * (qs[i / 2] & 0x000F)
                + yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc.s1 += yl[i + 8] * (qs[i / 2] & 0x00F0)
                + yl[i + 9] * (qs[i / 2] & 0xF000);
    }
    return d * (sumy * -8.f + acc.s0 + acc.s1);
}

#ifdef INTEL_GPU
#define N_DST 4 // each SIMD group works on 4 rows
#define N_SIMDGROUP 1 // number of SIMD groups in a thread group
#define N_SIMDWIDTH 16 // assuming SIMD group size is 16
#elif defined (ADRENO_GPU)
#define N_DST 4
#define N_SIMDGROUP 1
#define N_SIMDWIDTH 64
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

    const ulong nb = ne00/QK4_0;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    // (r0 * N_SIMDGROUP + get_sub_group_id()) is essenatially the linear global
    // id of a SIMD group in the grid.
    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset0 = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    global struct block_q4_0 * x = (global struct block_q4_0 *) src0 + offset0;
    global float             * y = (global float             *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[16];       // src1 vector cache
    float sumf[N_DST]={0.f};

    int ix = get_sub_group_local_id()/2;
    int il = 8*(get_sub_group_local_id()%2);

    global float * yb = y + ix * QK4_0 + il;

    // each thread in a SIMD group deals with half a block.
    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/2) {
        float sumy = 0;
        for (int i = 0; i < 8; i += 2) {
            sumy += yb[i] + yb[i+1];
            yl[i+0] = yb[i+ 0];
            yl[i+1] = yb[i+ 1]/256.f;
            sumy += yb[i+16] + yb[i+17];
            yl[i+8] = yb[i+16]/16.f;
            yl[i+9] = yb[i+17]/4096.f;
        }

        for (int row = 0; row < N_DST; row++) {
            sumf[row] += block_q_4_0_dot_y(x+ib+row*nb, sumy, yl, il);
        }

        // One thread in a SIMD group (i.e., subgroup) handles a half block,
        // hence then entire SIMD group handles SIMDWIDTH/2 blocks.
        // y points to the activation matrix (of type float). Therefore for
        // one thread, the # of blocks y should advance is SIMDWIDTH/2 (because
        // SIMDWIDTH/2 blocks are processed by a SIMD group) - in terms of
        // floats, it is QK4_0 * (SIMDWIDTH/2), where QK4_0 is the block size.
        yb += QK4_0 * (N_SIMDWIDTH/2);
    }

    // The above does not work for Adreno - it produces incorrect results for
    // row = 1, 2, 3 and only row = 0 gives the correct result.
    // If N_DST is changed, the below array must be initialized accordingly.
    // This also seems to perform better on Intel.
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
kernel void kernel_mul_mat_q4_0_f32(
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
