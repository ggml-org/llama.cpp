static inline float ifairy_bf16_to_f32(ushort h) {
    return as_float(((uint) h) << 16);
}

static inline ushort ifairy_f32_to_bf16(float x) {
    uint bits = as_uint(x);
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        return (ushort) ((bits >> 16) | 64U);
    }
    return (ushort) ((bits + (0x7fffU + ((bits >> 16) & 1U))) >> 16);
}

static inline float ifairy_pair_sum_sq(uint x) {
    const float xr = ifairy_bf16_to_f32((ushort) (x & 0xffffU));
    const float xi = ifairy_bf16_to_f32((ushort) (x >> 16));
    return xr*xr + xi*xi;
}

static inline uint ifairy_rms_norm_pair(uint x, float scale) {
    const float xr = ifairy_bf16_to_f32((ushort) (x & 0xffffU));
    const float xi = ifairy_bf16_to_f32((ushort) (x >> 16));

    const uint r = (uint) ifairy_f32_to_bf16(xr * scale);
    const uint i = (uint) ifairy_f32_to_bf16(xi * scale);
    return r | (i << 16);
}

/**
 * Normalizes each row of iFairy split-layout values using the CPU iFairy
 * RMSNorm bit semantics: each F32 lane is treated as two packed BF16 halves.
 */
kernel void kernel_ifairy_rms_norm(
        global char * src0,
        ulong offset0,
        global char * dst,
        ulong offsetd,
        int   ne00,
        int   ne01,
        int   ne02,
        int   ne03,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        float eps,
        local float * sum
) {
    src0 = src0 + offset0;
    dst  = dst  + offsetd;

    const int i03 = get_group_id(2);
    const int i02 = get_group_id(1);
    const int i01 = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);

    global char * x = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    global char * y = dst  + i03*nb3  + i02*nb2  + i01*nb1;

    float row_sum = 0.0f;
    for (int i00 = lid; i00 < ne00; i00 += lsz) {
        const uint bits = *((global uint *) (x + 4*i00));
        row_sum += ifairy_pair_sum_sq(bits);
    }

    sum[lid] = row_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsz >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            sum[lid] += sum[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float scale = rsqrt(sum[0] / (float) ne00 + eps);
    for (int i00 = lid; i00 < ne00; i00 += lsz) {
        const uint bits = *((global uint *) (x + 4*i00));
        *((global uint *) (y + 4*i00)) = ifairy_rms_norm_pair(bits, scale);
    }
}

/**
 * Fuses iFairy RMSNorm with the following ordinary F32 MUL used for norm
 * weights. The multiply is intentionally not complex multiplication.
 */
kernel void kernel_ifairy_rms_norm_mul(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global char * dst,
        ulong offsetd,
        int   ne00,
        int   ne01,
        int   ne02,
        int   ne03,
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
        ulong nb1,
        ulong nb2,
        ulong nb3,
        float eps,
        local float * sum
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    const int i03 = get_group_id(2);
    const int i02 = get_group_id(1);
    const int i01 = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);

    global char * x = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    global char * f = src1 + (i03 % ne13)*nb13 + (i02 % ne12)*nb12 + (i01 % ne11)*nb11;
    global char * y = dst  + i03*nb3  + i02*nb2  + i01*nb1;

    float row_sum = 0.0f;
    for (int i00 = lid; i00 < ne00; i00 += lsz) {
        const uint bits = *((global uint *) (x + 4*i00));
        row_sum += ifairy_pair_sum_sq(bits);
    }

    sum[lid] = row_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsz >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            sum[lid] += sum[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float scale = rsqrt(sum[0] / (float) ne00 + eps);
    for (int i00 = lid; i00 < ne00; i00 += lsz) {
        const int  i10    = i00 % ne10;
        const uint bits   = *((global uint *) (x + 4*i00));
        const uint packed = ifairy_rms_norm_pair(bits, scale);
        const float w     = *((global float *) (f + i10*nb10));
        *((global float *) (y + 4*i00)) = as_float(packed) * w;
    }
}
