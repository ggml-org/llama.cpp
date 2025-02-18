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
// block_q4_1
//------------------------------------------------------------------------------
struct block_q4_1
{
    half d;
    half m;
    uint8_t qs[QK4_1 / 2];
};

//------------------------------------------------------------------------------
// block_q5_0
//------------------------------------------------------------------------------
struct block_q5_0
{
    half d;
    uint32_t qh;
    uint8_t qs[QK5_0 / 2];
};

//------------------------------------------------------------------------------
// block_q5_1
//------------------------------------------------------------------------------
struct block_q5_1
{
    half d;
    half m;
    uint32_t qh;
    uint8_t qs[QK5_1 / 2];
};

//------------------------------------------------------------------------------
// block_q8_0
//------------------------------------------------------------------------------
struct block_q8_0
{
    half d;
    int8_t qs[QK8_0];
};

//------------------------------------------------------------------------------
// block_q2_K
//------------------------------------------------------------------------------
struct block_q2_K
{
    uint8_t scales[16];
    uint8_t qs[64];
    half d;
    half dmin;
};

//------------------------------------------------------------------------------
// block_q3_K
//------------------------------------------------------------------------------
struct block_q3_K
{
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    half d;
};

//------------------------------------------------------------------------------
// block_q4_K
//------------------------------------------------------------------------------
struct block_q4_K
{
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

//------------------------------------------------------------------------------
// block_q5_K
//------------------------------------------------------------------------------
struct block_q5_K
{
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

//------------------------------------------------------------------------------
// block_q6_K
//------------------------------------------------------------------------------
struct block_q6_K
{
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};

//------------------------------------------------------------------------------
// dequantize_q4_0_f32, dequantize_q4_0_f16
//------------------------------------------------------------------------------
void dequantize_q4_0_f32(global struct block_q4_0 * xb, short il, float16 * reg) {
    global ushort * qs = ((global ushort *)xb + 1);
    float d1 = il ? (xb->d / 16.h) : xb->d;
    float d2 = d1 / 256.f;
    float md = -8.h * xb->d;
    ushort mask0 = il ? 0x00F0 : 0x000F;
    ushort mask1 = mask0 << 8;

    reg->s0 = d1 * (qs[0] & mask0) + md;
    reg->s1 = d2 * (qs[0] & mask1) + md;

    reg->s2 = d1 * (qs[1] & mask0) + md;
    reg->s3 = d2 * (qs[1] & mask1) + md;

    reg->s4 = d1 * (qs[2] & mask0) + md;
    reg->s5 = d2 * (qs[2] & mask1) + md;

    reg->s6 = d1 * (qs[3] & mask0) + md;
    reg->s7 = d2 * (qs[3] & mask1) + md;

    reg->s8 = d1 * (qs[4] & mask0) + md;
    reg->s9 = d2 * (qs[4] & mask1) + md;

    reg->sa = d1 * (qs[5] & mask0) + md;
    reg->sb = d2 * (qs[5] & mask1) + md;

    reg->sc = d1 * (qs[6] & mask0) + md;
    reg->sd = d2 * (qs[6] & mask1) + md;

    reg->se = d1 * (qs[7] & mask0) + md;
    reg->sf = d2 * (qs[7] & mask1) + md;
}

void dequantize_q4_0_f16(global struct block_q4_0 * xb, short il, half16 * reg) {
    global ushort * qs = ((global ushort *)xb + 1);
    half d1 = il ? (xb->d / 16.h) : xb->d;
    half d2 = d1 / 256.h;
    half md = -8.h * xb->d;
    ushort mask0 = il ? 0x00F0 : 0x000F;
    ushort mask1 = mask0 << 8;

    reg->s0 = d1 * (qs[0] & mask0) + md;
    reg->s1 = d2 * (qs[0] & mask1) + md;

    reg->s2 = d1 * (qs[1] & mask0) + md;
    reg->s3 = d2 * (qs[1] & mask1) + md;

    reg->s4 = d1 * (qs[2] & mask0) + md;
    reg->s5 = d2 * (qs[2] & mask1) + md;

    reg->s6 = d1 * (qs[3] & mask0) + md;
    reg->s7 = d2 * (qs[3] & mask1) + md;

    reg->s8 = d1 * (qs[4] & mask0) + md;
    reg->s9 = d2 * (qs[4] & mask1) + md;

    reg->sa = d1 * (qs[5] & mask0) + md;
    reg->sb = d2 * (qs[5] & mask1) + md;

    reg->sc = d1 * (qs[6] & mask0) + md;
    reg->sd = d2 * (qs[6] & mask1) + md;

    reg->se = d1 * (qs[7] & mask0) + md;
    reg->sf = d2 * (qs[7] & mask1) + md;
}

//------------------------------------------------------------------------------
// add
//------------------------------------------------------------------------------

// general-purpose kernel for addition of two tensors
// pros: works for non-contiguous tensors, supports broadcast across dims 1, 2 and 3
// cons: not very efficient
kernel void kernel_add(
        global char * src0,
        ulong  offset0,
        global char * src1,
        ulong  offset1,
        global char * dst,
        ulong  offsetd,
        int   ne00,
        int   ne01,
        int   ne02,
        int   ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int   ne10,
        int   ne11,
        int   ne12,
        int   ne13,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int   ne0,
        int   ne1,
        int   ne2,
        int   ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst = dst + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int i13 = i03 % ne13;
    int i12 = i02 % ne12;
    int i11 = i01 % ne11;

    global char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    global char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    global char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1;

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const int i10 = i0 % ne10;
        *((global float *)(dst_ptr + i0*nb0)) = *((global float *)(src0_ptr + i0*nb00)) + *((global float *)(src1_ptr + i10*nb10));
    }
}

// assumption: src1 is a row
// broadcast src1 into src0
kernel void kernel_add_row(
        global float4 * src0,
        ulong  offset0,
        global float4 * src1,
        ulong  offset1,
        global float4 * dst,
        ulong  offsetd,
        int ne
) {
    src0 = (global float4*)((global char*)src0 + offset0);
    src1 = (global float4*)((global char*)src1 + offset1);
    dst = (global float4*)((global char*)dst + offsetd);

    // This performs better than using %.
    uint gid = get_global_id(0);
    uint idx1 = gid - (gid/ne)*ne; // get_global_id(0) % ne
    dst[gid] = src0[gid] + src1[idx1];
}

//------------------------------------------------------------------------------
// mul
//------------------------------------------------------------------------------
kernel void kernel_mul(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global char * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        int ne13,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int i13 = i03 % ne13;
    int i12 = i02 % ne12;
    int i11 = i01 % ne11;

    global char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    global char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    global char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1;

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const int i10 = i0 % ne10;
        *((global float *)(dst_ptr + i0*nb0)) = *((global float *)(src0_ptr + i0*nb00)) * *((global float *)(src1_ptr + i10*nb10));
    }
}

// assumption: src1 is a row
// broadcast src1 into src0
kernel void kernel_mul_row(
        global float4 * src0,
        ulong offset0,
        global float4 * src1,
        ulong offset1,
        global float4 * dst,
        ulong offsetd,
        int ne
) {
    src0 = (global float4*)((global char*)src0 + offset0);
    src1 = (global float4*)((global char*)src1 + offset1);
    dst = (global float4*)((global char*)dst + offsetd);

    // This performs better than using %.
    uint gid = get_global_id(0);
    uint idx1 = gid - (gid/ne)*ne; // get_global_id(0) % ne
    dst[gid] = src0[gid] * src1[idx1];
}

//------------------------------------------------------------------------------
// scale
//------------------------------------------------------------------------------
kernel void kernel_scale(
        global float4 * src0,
        ulong offset0,
        global float4 * dst,
        ulong offsetd,
        float scale
) {
    src0 = (global float4*)((global char*)src0 + offset0);
    dst = (global float4*)((global char*)dst + offsetd);
    dst[get_global_id(0)] = src0[get_global_id(0)] * scale;
}

//------------------------------------------------------------------------------
// gelu
//------------------------------------------------------------------------------
#define GELU_COEF_A     0.044715f
#define SQRT_2_OVER_PI  0.79788456080286535587989211986876f

kernel void kernel_gelu(
    global float * src0,
    ulong offset0,
    global float * dst,
    ulong offsetd
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    float x = src0[get_global_id(0)];

    dst[get_global_id(0)] = 0.5f*x*(1.0f + tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

kernel void kernel_gelu_4(
    global float4 * src0,
    ulong offset0,
    global float4 * dst,
    ulong offsetd
) {
    src0 = (global float4*)((global char*)src0 + offset0);
    dst = (global float4*)((global char*)dst + offsetd);

    float4 x = src0[get_global_id(0)];

    dst[get_global_id(0)] = 0.5f*x*(1.0f + tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

//------------------------------------------------------------------------------
// silu
//------------------------------------------------------------------------------
kernel void kernel_silu(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    float x = src0[get_global_id(0)];
    dst[get_global_id(0)] = x / (1.0f + exp(-x));
}

kernel void kernel_silu_4(
        global float4 * src0,
        ulong offset0,
        global float4 * dst,
        ulong offsetd
) {
    src0 = (global float4*)((global char*)src0 + offset0);
    dst = (global float4*)((global char*)dst + offsetd);

    float4 x = src0[get_global_id(0)];
    dst[get_global_id(0)] = x / (1.0f + exp(-x));
}

//------------------------------------------------------------------------------
// relu
//------------------------------------------------------------------------------
kernel void kernel_relu(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    dst[get_global_id(0)] = fmax(0.0f, src0[get_global_id(0)]);
}

//------------------------------------------------------------------------------
// clamp
//------------------------------------------------------------------------------
kernel void kernel_clamp(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        float min,
        float max
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    dst[get_global_id(0)] = src0[get_global_id(0)] < min ?
        min :
        (src0[get_global_id(0)] > max ? max : src0[get_global_id(0)]);
}

//------------------------------------------------------------------------------
// norm
//------------------------------------------------------------------------------
kernel void kernel_norm(
        global void * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        float eps,
        local float * sum
) {
    src0 = (global void*)((global char*)src0 + offset0);
    dst = (global void*)((global char*)dst + offsetd);

    global float * x = (global float *) ((global char *) src0 + get_group_id(0)*nb01);

    // MEAN
    // parallel sum
    sum[get_local_id(0)] = 0.0f;
    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        sum[get_local_id(0)] += x[i00];
    }
    // reduce
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = get_local_size(0)/2; i > 0; i /= 2) {
        if (get_local_id(0) < i) {
            sum[get_local_id(0)] += sum[get_local_id(0) + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean  = sum[0] / ne00;

    // recenter and VARIANCE
    barrier(CLK_LOCAL_MEM_FENCE);
    global float * y = dst + get_group_id(0)*ne00;
    sum[get_local_id(0)] = 0.0f;
    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        y[i00] = x[i00] - mean;
        sum[get_local_id(0)] += y[i00] * y[i00];
    }

    // reduce
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = get_local_size(0)/2; i > 0; i /= 2) {
        if (get_local_id(0) < i) {
            sum[get_local_id(0)] += sum[get_local_id(0) + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float variance = sum[0] / ne00;

    float scale = 1.0f/sqrt(variance + eps);
    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        y[i00] = y[i00] * scale;
    }
}

//------------------------------------------------------------------------------
// rms_norm
//------------------------------------------------------------------------------
// This kernel depends on subgroup size.
kernel void kernel_rms_norm(
        global void * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        float eps,
        local float * sum // Note, the size depends on number of subgroups
) {
    src0 = (global void*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    global float4 * x = (global float4 *) ((global char *) src0 + get_group_id(0)*nb01);
    global float * x_scalar = (global float *) x;
    float4 sumf = 0;
    float all_sum = 0;

    // parallel sum
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        sumf += x[i00] * x[i00];
    }
    all_sum = sumf.s0 + sumf.s1 + sumf.s2 + sumf.s3;
    all_sum = sub_group_reduce_add(all_sum);
    if (get_sub_group_local_id() == 0) {
        sum[get_sub_group_id()] = all_sum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    // broadcast
    for (uint i = get_local_size(0) / get_max_sub_group_size() / 2; i > 0; i /= 2) {
       if (get_local_id(0) < i) {
           sum[get_local_id(0)] += sum[get_local_id(0) + i];
       }
    }
    if (get_local_id(0) == 0) {
        for (int i = 4 * (ne00 / 4); i < ne00; i++) {
            sum[0] += x_scalar[i];
        }
        sum[0] /= ne00;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const float mean  = sum[0];
    const float scale = 1.0f/sqrt(mean + eps);

    global float4 * y = (global float4 *) (dst + get_group_id(0)*ne00);
    global float * y_scalar = (global float *) y;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        y[i00] = x[i00] * scale;
    }
    if (get_local_id(0) == 0) {
        for (int i00 = 4 * (ne00 / 4); i00 < ne00; i00++) {
            y_scalar[i00] = x_scalar[i00] * scale;
        }
    }
}

//------------------------------------------------------------------------------
// diag_mask_inf kernels
//------------------------------------------------------------------------------
kernel void kernel_diag_mask_inf(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int n_past
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int i02 = get_global_id(2);
    int i01 = get_global_id(1);
    int i00 = get_global_id(0);

    if (i00 > n_past + i01) {
        dst[i02*ne01*ne00 + i01*ne00 + i00] = -INFINITY;
    } else {
        dst[i02*ne01*ne00 + i01*ne00 + i00] = src0[i02*ne01*ne00 + i01*ne00 + i00];
    }
}

kernel void kernel_diag_mask_inf_8(
        global float4 * src0,
        ulong offset0,
        global float4 * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int n_past
) {
    src0 = (global float4*)((global char*)src0 + offset0);
    dst = (global float4*)((global char*)dst + offsetd);

    int i = 2*get_global_id(0);

    dst[i+0] = src0[i+0];
    dst[i+1] = src0[i+1];
    int i4 = 4*i;
    int i02 = i4/(ne00*ne01); i4 -= i02*ne00*ne01;
    int i01 = i4/(ne00);      i4 -= i01*ne00;
    int i00 = i4;
    for (int k = 3; k >= 0; --k) {
        if (i00 + 4 + k <= n_past + i01) {
            break;
        }
        (&dst[i+1])[k] = -INFINITY;
        if (i00 + k > n_past + i01) {
            (&dst[i])[k] = -INFINITY;
        }
    }
}

//------------------------------------------------------------------------------
// softmax
//------------------------------------------------------------------------------
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_soft_max(
        global float * src0,
        ulong offset0,
        global float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        float scale,
        float max_bias,
        float m0,
        float m1,
        int n_head_log2
) {
    src0 = (global float*)((global char*)src0 + offset0);
    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    global float * psrc0 = src0 + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;
    global float * pmask = src1 != src0 ? src1 + i01*ne00 : 0;
    global float * pdst  = dst  + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    float slope = 1.0f;

    // ALiBi
    if (max_bias > 0.0f) {
        int h = i02;

        float base = h < n_head_log2 ? m0 : m1;
        int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float lmax = -INFINITY;
    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        lmax = fmax(lmax, psrc0[i00]*scale + (pmask ? slope*pmask[i00] : 0.0f));
    }
    float max = sub_group_reduce_max(lmax);

    // parallel sum
    float lsum = 0.0f;
    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        float exp_psrc0 = exp((psrc0[i00]*scale + (pmask ? slope*pmask[i00] : 0.0f)) - max);
        lsum += exp_psrc0;
        // Remember the result of exp here. exp is expensive, so we really do not
        // wish to compute it twice.
        pdst[i00] = exp_psrc0;
    }

    const float sum = sub_group_reduce_add(lsum);

    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        pdst[i00] /= sum;
    }
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_soft_max_4(
        global float * src0,
        ulong offset0,
        global float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        float scale,
        float max_bias,
        float m0,
        float m1,
        int n_head_log2
) {
    src0 = (global float*)((global char*)src0 + offset0);
    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    global float4 * psrc4 = (global float4 *)(src0 + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);
    global float4 * pmask = src1 != src0 ? (global float4 *)(src1 + i01*ne00) : 0;
    global float4 * pdst4 = (global float4 *)(dst  + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);

    float slope = 1.0f;

    // ALiBi
    if (max_bias > 0.0f) {
        int h = i02;

        float base = h < n_head_log2 ? m0 : m1;
        int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float4 lmax4 = -INFINITY;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        lmax4 = fmax(lmax4, psrc4[i00]*scale + (pmask ? slope*pmask[i00] : 0.0f));
    }
    float lmax = fmax(fmax(lmax4.s0, lmax4.s1), fmax(lmax4.s2, lmax4.s3));

    const float max = sub_group_reduce_max(lmax);

    // parallel sum
    float4 lsum4 = 0.0f;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        const float4 exp_psrc4 = exp((psrc4[i00]*scale + (pmask ? slope*pmask[i00] : 0.0f)) - max);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }
    float lsum = lsum4.s0 + lsum4.s1 + lsum4.s2 + lsum4.s3;

    const float sum = sub_group_reduce_add(lsum);

    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        pdst4[i00] /= sum;
    }
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_soft_max_f16(
        global float * src0,
        ulong offset0,
        global half * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        float scale,
        float max_bias,
        float m0,
        float m1,
        int n_head_log2
) {
    src0 = (global float *)((global char *)src0 + offset0);
    src1 = (global half *)((global char *)src1 + offset1);
    dst = (global float *)((global char *)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    global float * psrc0 = src0 + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;
    global half  * pmask = (global char *)src1 != (global char *)src0 ? src1 + i01*ne00 : 0;
    global float * pdst  = dst  + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    float slope = 1.0f;

    // ALiBi
    if (max_bias > 0.0f) {
        int h = i02;

        float base = h < n_head_log2 ? m0 : m1;
        int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float lmax = -INFINITY;
    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        lmax = fmax(lmax, psrc0[i00]*scale + (pmask ? slope*pmask[i00] : 0.0f));
    }
    float max = sub_group_reduce_max(lmax);

    // parallel sum
    float lsum = 0.0f;
    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        float exp_psrc0 = exp((psrc0[i00]*scale + (pmask ? slope*pmask[i00] : 0.0f)) - max);
        lsum += exp_psrc0;
        // Remember the result of exp here. exp is expensive, so we really do not
        // wish to compute it twice.
        pdst[i00] = exp_psrc0;
    }

    const float sum = sub_group_reduce_add(lsum);

    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        pdst[i00] /= sum;
    }
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_soft_max_4_f16(
        global float * src0,
        ulong offset0,
        global half * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        float scale,
        float max_bias,
        float m0,
        float m1,
        int n_head_log2
) {
    src0 = (global float *)((global char *)src0 + offset0);
    src1 = (global half *)((global char *)src1 + offset1);
    dst = (global float *)((global char *)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    global float4 * psrc4 = (global float4 *)(src0 + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);
    global half4  * pmask = (global char *)src1 != (global char *)src0 ? (global half4 *)(src1 + i01*ne00) : 0;
    global float4 * pdst4 = (global float4 *)(dst  + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);

    float slope = 1.0f;

    // ALiBi
    if (max_bias > 0.0f) {
        int h = i02;

        float base = h < n_head_log2 ? m0 : m1;
        int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float4 lmax4 = -INFINITY;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        lmax4 = fmax(lmax4, psrc4[i00]*scale + slope*(pmask ? convert_float4(pmask[i00]) : 0.0f));
    }
    float lmax = fmax(fmax(lmax4.s0, lmax4.s1), fmax(lmax4.s2, lmax4.s3));

    const float max = sub_group_reduce_max(lmax);

    // parallel sum
    float4 lsum4 = 0.0f;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        const float4 exp_psrc4 = exp((psrc4[i00]*scale + slope*(pmask ? convert_float4(pmask[i00]) : 0.0f)) - max);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }
    float lsum = lsum4.s0 + lsum4.s1 + lsum4.s2 + lsum4.s3;

    const float sum = sub_group_reduce_add(lsum);

    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        pdst4[i00] /= sum;
    }
}

//------------------------------------------------------------------------------
// kernel_rope
//------------------------------------------------------------------------------
float rope_yarn_ramp(float low, float high, int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
float2 rope_yarn(
    float theta_extrap, float freq_scale, float2 corr_dims, int i0, float ext_factor, float mscale
) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.s0, corr_dims.s1, i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    return (float2)(cos(theta) * mscale, sin(theta) * mscale);
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_fac(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
float rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * log(n_ctx_orig / (n_rot * 2 * M_PI_F)) / (2 * log(base));
}

float2 rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow
) {
    // start and end correction dims
    return (float2)(
        max(0.0f,         floor(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base))),
        min(n_dims - 1.0f, ceil(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)))
    );
}

kernel void kernel_rope_norm_f32(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * src2,
        ulong offset2,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    src2 = (global float*)((global char*)src2 + offset2);
    dst = (global float*)((global char*)dst + offsetd);

    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;

    float theta_base = (float) pos[i2];
    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        if (i0 < n_dims) {
            int ic = i0/2;

            float theta = theta_base * pow(freq_base, inv_ndims*i0);

            float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

            global float * src       = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global float * dst_data  = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            float x0 = src[0];
            float x1 = src[1];

            dst_data[0] = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
            dst_data[1] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
        } else {
            global float * src      = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global float * dst_data = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

kernel void kernel_rope_norm_f16(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * src2,
        ulong offset2,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    src2 = (global float*)((global char*)src2 + offset2);
    dst = (global float*)((global char*)dst + offsetd);

    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;

    float theta_base = (float) pos[i2];
    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        if (i0 < n_dims) {
            int ic = i0/2;

            float theta = theta_base * pow(freq_base, inv_ndims*i0);

            float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

            global half * src       = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global half * dst_data  = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            float x0 = src[0];
            float x1 = src[1];

            dst_data[0] = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
            dst_data[1] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
        } else {
            global half * src      = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global half * dst_data = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

kernel void kernel_rope_neox_f32(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * src2,
        ulong offset2,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    src2 = (global float*)((global char*)src2 + offset2);
    dst = (global float*)((global char*)dst + offsetd);

    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;

    float theta_base = (float) pos[i2];
    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        if (i0 < n_dims) {
            int ic = i0/2;

            const float theta = theta_base * pow(freq_base, inv_ndims*i0);

            const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

            global float * src      = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
            global float * dst_data = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

            const float x0 = src[0];
            const float x1 = src[n_dims/2];

            dst_data[0]        = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
            dst_data[n_dims/2] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
        } else {
            global float * const src = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global float * dst_data  = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

kernel void kernel_rope_neox_f16(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * src2,
        ulong offset2,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    src2 = (global float*)((global char*)src2 + offset2);
    dst = (global float*)((global char*)dst + offsetd);

    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;

    float theta_base = (float) pos[i2];
    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        if (i0 < n_dims) {
            int ic = i0/2;

            const float theta = theta_base * pow(freq_base, inv_ndims*i0);

            const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

            global half * src       = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
            global half * dst_data  = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

            const float x0 = src[0];
            const float x1 = src[n_dims/2];

            dst_data[0]        = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
            dst_data[n_dims/2] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
        } else {
            global half * const src = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global half * dst_data  = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

//------------------------------------------------------------------------------
// cpy
//------------------------------------------------------------------------------

kernel void kernel_cpy_f16_f16(
        global half * src0,
        ulong offset0,
        global half * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global half*)((global char*)src0 + offset0);
    dst = (global half*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    int i3 = n / (ne2*ne1*ne0);
    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    global half * dst_data = (global half *) ((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        global const half * src = (global half *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        dst_data[i00] = src[0];
    }
}

kernel void kernel_cpy_f16_f32(
        global half * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {

    src0 = (global half*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    int i3 = n / (ne2*ne1*ne0);
    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    global float * dst_data = (global float *) ((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        global half * src = (global half *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        dst_data[i00] = src[0];
    }
}

kernel void kernel_cpy_f32_f16(
        global float * src0,
        ulong offset0,
        global half * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global half*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    int i3 = n / (ne2*ne1*ne0);
    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    global half * dst_data = (global half *) ((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        global const float * src = (global float *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        dst_data[i00] = src[0];
    }
}

kernel void kernel_cpy_f32_f32(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    int i3 = n / (ne2*ne1*ne0);
    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    global float * dst_data = (global float *) ((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        global const float * src = (global float *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        dst_data[i00] = src[0];
    }
}

//------------------------------------------------------------------------------
// get_rows
//------------------------------------------------------------------------------
kernel void kernel_get_rows_f32(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb1,
        ulong nb2
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);

    int r = ((global int *) ((global char *) src1 + i11*nb11 + i10*nb10))[0];

    int i02 = i11;

    for (int ind = get_local_id(0); ind < ne00; ind += get_local_size(0)) {
        ((global float *) ((global char *) dst + i11*nb2 + i10*nb1))[ind] =
            ((global float *) ((global char *) src0 + r*nb01 + i02*nb02))[ind];
    }
}

kernel void kernel_get_rows_f16(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb1,
        ulong nb2
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);

    int r = ((global int32_t *) ((global char *) src1 + i11*nb11 + i10*nb10))[0];

    int i02 = i11;

    for (int ind = get_local_id(0); ind < ne00; ind += get_local_size(0)) {
        ((global float *) ((global char *) dst + i11*nb2 + i10*nb1))[ind] =
            ((global half *) ((global char *) src0 + r*nb01 + i02*nb02))[ind];
    }
}

kernel void kernel_get_rows_q4_0(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb1,
        ulong nb2
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    const int NL = 2;

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);

    int r = ((global int32_t *) ((global char *) src1 + i11*nb11 + i10*nb10))[0];

    int i02 = i11;

    for (int ind = get_local_id(0); ind < ne00/16; ind += get_local_size(0)) {
        float16 temp;
        dequantize_q4_0_f32(
            ((global struct block_q4_0 *) ((global char *) src0 + r*nb01 + i02*nb02)) + ind/NL, ind%NL, &temp);
        *(((global float16 *) ((global char *) dst + i11*nb2 + i10*nb1)) + ind) = temp;
    }
}

//------------------------------------------------------------------------------
// mul_mat_f32_f32
//------------------------------------------------------------------------------
#define N_F32_F32 4

kernel void kernel_mul_mat_f32_f32(
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

    int r0 = get_group_id(0);
    int rb = get_group_id(1)*N_F32_F32;
    int im = get_group_id(2);

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset_src0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;

    global float * x = (global float *) (src0 + offset_src0);

    if (ne00 < 128) {
        for (int row = 0; row < N_F32_F32; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

            global float * y = (global float *) (src1 + offset_src1);

            float sumf = 0;
            for (int i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size()) {
                sumf += (float) x[i] * (float) y[i];
            }

            float all_sum = sub_group_reduce_add(sumf);
            if (get_sub_group_local_id() == 0) {
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    } else {
        global float4 * x4 = (global float4 *)x;
        for (int row = 0; row < N_F32_F32; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

            global float  * y  = (global float  *) (src1 + offset_src1);
            global float4 * y4 = (global float4 *) y;

            float sumf = 0;
            for (int i = get_sub_group_local_id(); i < ne00/4; i += get_max_sub_group_size()) {
                sumf += (float) x4[i].s0 * y4[i].s0;
                sumf += (float) x4[i].s1 * y4[i].s1;
                sumf += (float) x4[i].s2 * y4[i].s2;
                sumf += (float) x4[i].s3 * y4[i].s3;
            }

            float all_sum = sub_group_reduce_add(sumf);
            if (get_sub_group_local_id() == 0) {
                for (int i = 4*(ne00/4); i < ne00; ++i) {
                    all_sum += (float) x[i] * y[i];
                }
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    }
}

//------------------------------------------------------------------------------
// mul_mat_f16_f16
//------------------------------------------------------------------------------
#define N_F16_F16 4

kernel void kernel_mul_mat_f16_f16(
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
        int r3)
{
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int r0 = get_group_id(0);
    int rb = get_group_id(1)*N_F16_F16;
    int im = get_group_id(2);

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset_src0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;

    global half * x = (global half *) (src0 + offset_src0);

    if (ne00 < 128) {
        for (int row = 0; row < N_F16_F16; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

            global half * y = (global half *) (src1 + offset_src1);

            float sumf = 0;
            for (int i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size()) {
                sumf += (half) x[i] * (half) y[i];
            }

            float all_sum = sub_group_reduce_add(sumf);
            if (get_sub_group_local_id() == 0) {
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    } else {
        global half4 * x4 = (global half4 *)x;
        for (int row = 0; row < N_F16_F16; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

            global half  * y  = (global half  *) (src1 + offset_src1);
            global half4 * y4 = (global half4 *) y;

            float sumf = 0;
            for (int i = get_sub_group_local_id(); i < ne00/4; i += get_max_sub_group_size()) {
                sumf += (half) x4[i].s0 * y4[i].s0;
                sumf += (half) x4[i].s1 * y4[i].s1;
                sumf += (half) x4[i].s2 * y4[i].s2;
                sumf += (half) x4[i].s3 * y4[i].s3;
            }

            float all_sum = sub_group_reduce_add(sumf);
            if (get_sub_group_local_id() == 0) {
                for (int i = 4*(ne00/4); i < ne00; ++i) {
                    all_sum += (half) x[i] * y[i];
                }
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    }
}

//------------------------------------------------------------------------------
// mul_mat_f16_f32_1row
//------------------------------------------------------------------------------
kernel void kernel_mul_mat_f16_f32_1row(
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

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset_src0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    global half  * x = (global half  *) (src0 + offset_src0);
    global float * y = (global float *) (src1 + offset_src1);

    float sumf = 0;
    if (ne00 < 128) {
        for (int i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size()) {
            sumf += (float) x[i] * (float) y[i];
        }
        float all_sum = sub_group_reduce_add(sumf);
        if (get_sub_group_local_id() == 0) {
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
    } else {
        global half4  * x4 = (global half4  *) x;
        global float4 * y4 = (global float4 *) y;
        for (int i = get_sub_group_local_id(); i < ne00/4; i += get_max_sub_group_size()) {
            sumf += (float) x4[i].s0 * y4[i].s0;
            sumf += (float) x4[i].s1 * y4[i].s1;
            sumf += (float) x4[i].s2 * y4[i].s2;
            sumf += (float) x4[i].s3 * y4[i].s3;
        }
        float all_sum = sub_group_reduce_add(sumf);
        if (get_sub_group_local_id() == 0) {
            for (int i = 4*(ne00/4); i < ne00; ++i) {
                all_sum += (float) x[i] * y[i];
            }
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
    }

}

//------------------------------------------------------------------------------
// mul_mat_f16_f32
//------------------------------------------------------------------------------
#define N_F16_F32 4

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32(
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

    int r0 = get_group_id(0);
    int rb = get_group_id(1)*N_F16_F32;
    int im = get_group_id(2);

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset_src0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;

    global half * x = (global half *) (src0 + offset_src0);

    if (ne00 < 128) {
        for (int row = 0; row < N_F16_F32; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

            global float * y = (global float *) (src1 + offset_src1);

            float sumf = 0;
            for (int i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size()) {
                sumf += convert_float(x[i]) * y[i];
            }

            float all_sum = sub_group_reduce_add(sumf);
            if (get_sub_group_local_id() == 0) {
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    } else {
        global half4 * x4 = (global half4 *)x;
        for (int row = 0; row < N_F16_F32; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

            global float  * y  = (global float  *) (src1 + offset_src1);
            global float4 * y4 = (global float4 *) y;

            float sumf = 0;
            for (int i = get_sub_group_local_id(); i < ne00/4; i += get_max_sub_group_size()) {
                sumf += convert_float(x4[i].s0) * y4[i].s0;
                sumf += convert_float(x4[i].s1) * y4[i].s1;
                sumf += convert_float(x4[i].s2) * y4[i].s2;
                sumf += convert_float(x4[i].s3) * y4[i].s3;
            }

            float all_sum = sub_group_reduce_add(sumf);
            if (get_sub_group_local_id() == 0) {
                for (int i = 4*(ne00/4); i < ne00; ++i) {
                    all_sum += (float) x[i] * y[i];
                }
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    }
}

//------------------------------------------------------------------------------
// mul_mat_f16_f32_l4
//------------------------------------------------------------------------------
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

//
// This variant unrolls the loops and uses vector types instead of pointers.
// It improves performance on Adreno but not so much on Intel.
//
inline float block_q_4_0_dot_y_v(
        global struct block_q4_0 * qb_curr,
        float sumy,
        float16 yl,
        int il
) {
    float d = qb_curr->d;
    float acc = 0.f;
    global ushort * qs = ((global ushort *)qb_curr + 1 + il/2);

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
#define N_DST 4 // each SIMD group works on 4 rows
#define N_SIMDGROUP 1 // number of SIMD groups in a thread group
#define N_SIMDWIDTH 16 // assuming SIMD group size is 16
#elif defined (ADRENO_GPU)
#define N_DST 4
#define N_SIMDGROUP 1
#define N_SIMDWIDTH 64
#endif

inline void mul_vec_q_n_f32_v(
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

    float16 yl;       // src1 vector cache
    float4 sumf = (float4)(0.f, 0.f, 0.f, 0.f);

    int ix = get_sub_group_local_id()/2;
    int il = 8*(get_sub_group_local_id()%2);

    global float * yb = y + ix * QK4_0 + il;

    // each thread in a SIMD group deals with half a block.
    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/2) {
        float sumy = 0;

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

        sumf.s0 += block_q_4_0_dot_y_v(x+ib+0*nb, sumy, yl, il);
        sumf.s1 += block_q_4_0_dot_y_v(x+ib+1*nb, sumy, yl, il);
        sumf.s2 += block_q_4_0_dot_y_v(x+ib+2*nb, sumy, yl, il);
        sumf.s3 += block_q_4_0_dot_y_v(x+ib+3*nb, sumy, yl, il);

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
    float4 tot = (float4)(
        sub_group_reduce_add(sumf.s0), sub_group_reduce_add(sumf.s1),
        sub_group_reduce_add(sumf.s2), sub_group_reduce_add(sumf.s3)
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
    }
}

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_0_f32_v(
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

    mul_vec_q_n_f32_v(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3);
}

//------------------------------------------------------------------------------
// kernel_convert_block_q4_0
// Convert the block_q4_0 format to 2 separate arrays (AOS -> SOA).
// This kernel does not deshuffle the bits.
//------------------------------------------------------------------------------
kernel void kernel_convert_block_q4_0(
    global struct block_q4_0 * src0,
    global uchar * dst_q,
    global half  * dst_d
) {
    global struct block_q4_0 * b = (global struct block_q4_0 *) src0 + get_global_id(0);
    global uchar * q = (global uchar *) dst_q + QK4_0/2*get_global_id(0);
    global half  * d = (global half *) dst_d + get_global_id(0);

    *d = b->d;

    for (int i = 0; i < QK4_0/2; ++i) {
        q[i] = b->qs[i];
    }
}

kernel void kernel_restore_block_q4_0(
    global uchar * src_q,
    global half  * src_d,
    global struct block_q4_0 * dst
) {
    global struct block_q4_0 * b = (global struct block_q4_0 *) dst + get_global_id(0);
    global uchar * q = (global uchar *) src_q + QK4_0/2*get_global_id(0);
    global half  * d = (global half *) src_d + get_global_id(0);

    b->d = *d;
    for (int i = 0; i < QK4_0/2; ++i) {
        b->qs[i] = q[i];
    }
}

//------------------------------------------------------------------------------
// mul_vec_q_n_f32_flat
//
// This variation uses flat arrays (struct of arrays, SOA) representation for
// quant tensors.
//------------------------------------------------------------------------------

// This function requires the original shuffled weights.
// As a reminder, the original weights are shuffled so that (q[0], q[16]) are
// packed together in a byte, so are (q[1], q[17]) and so on.
inline float block_q_4_0_dot_y_flat(
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
#define N_DST 4 // each SIMD group works on 4 rows
#define N_SIMDGROUP 1 // number of SIMD groups in a thread group
#define N_SIMDWIDTH 16 // assuming SIMD group size is 32
#elif defined (ADRENO_GPU)
#define N_DST 4
#define N_SIMDGROUP 1
#define N_SIMDWIDTH 64
#endif

inline void mul_vec_q_n_f32_flat(
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
    const ulong nb = ne00/QK4_0;

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
    float4 sumf = (float4)(0.f, 0.f, 0.f, 0.f);

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

        sumf.s0 += block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 0*nb*QK4_0/2, d + ib + 0*nb, sumy, yl, il);
        sumf.s1 += block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 1*nb*QK4_0/2, d + ib + 1*nb, sumy, yl, il);
        sumf.s2 += block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 2*nb*QK4_0/2, d + ib + 2*nb, sumy, yl, il);
        sumf.s3 += block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 3*nb*QK4_0/2, d + ib + 3*nb, sumy, yl, il);

        yb += QK4_0 * (N_SIMDWIDTH/2);
    }

    float4 tot = (float4)(
        sub_group_reduce_add(sumf.s0), sub_group_reduce_add(sumf.s1),
        sub_group_reduce_add(sumf.s2), sub_group_reduce_add(sumf.s3)
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
    }
}

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_0_f32_flat(
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

    mul_vec_q_n_f32_flat(src0_q, src0_d, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3);
}

//
// This variant outputs 8 values.
//
#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define N_DST 8 // each SIMD group works on 8 rows
#define N_SIMDGROUP 1 // number of SIMD groups in a thread group
#define N_SIMDWIDTH 16 // assuming SIMD group size is 32
#elif defined (ADRENO_GPU)
#define N_DST 8
#define N_SIMDGROUP 1
#define N_SIMDWIDTH 64
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
    const ulong nb = ne00/QK4_0;

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
    float8 sumf = 0.f;

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

        sumf.s0 += block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 0*nb*QK4_0/2, d + ib + 0*nb, sumy, yl, il);
        sumf.s1 += block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 1*nb*QK4_0/2, d + ib + 1*nb, sumy, yl, il);
        sumf.s2 += block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 2*nb*QK4_0/2, d + ib + 2*nb, sumy, yl, il);
        sumf.s3 += block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 3*nb*QK4_0/2, d + ib + 3*nb, sumy, yl, il);

        sumf.s4 += block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 4*nb*QK4_0/2, d + ib + 4*nb, sumy, yl, il);
        sumf.s5 += block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 5*nb*QK4_0/2, d + ib + 5*nb, sumy, yl, il);
        sumf.s6 += block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 6*nb*QK4_0/2, d + ib + 6*nb, sumy, yl, il);
        sumf.s7 += block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 7*nb*QK4_0/2, d + ib + 7*nb, sumy, yl, il);

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
kernel void kernel_mul_mat_q4_0_f32_8x_flat(
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
