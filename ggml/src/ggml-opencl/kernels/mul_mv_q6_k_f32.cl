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

    if (row >= ne01) {
        return;
    }

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
