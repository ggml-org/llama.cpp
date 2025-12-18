#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define GELU_COEF_A     0.044715f
#define GELU_QUICK_COEF -1.702f
#define SQRT_2_OVER_PI  0.79788456080286535587989211986876f
#define SQRT_2_INV      0.70710678118654752440084436210484f

//------------------------------------------------------------------------------
// geglu
//------------------------------------------------------------------------------
kernel void kernel_geglu(
    global char * src0,
    ulong  offset0,
    global char * src1,
    ulong  offset1,
    global char * dst,
    ulong  offsetd,
    ulong nb01,
    ulong nb11,
    int ne0,
    ulong nb1,
    int ne00_off,
    int ne10_off
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    global float * src0_row = (global float *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
    global float * src1_row = (global float *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
    global float * dst_row  = (global float *) ((global char *) dst  + get_group_id(0)*nb1);

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float gelu = 0.5f*x0*(1.0f + tanh(SQRT_2_OVER_PI*x0*(1.0f + GELU_COEF_A*x0*x0)));

        dst_row[i0] = gelu*x1;
    }
}

kernel void kernel_geglu_f16(
    global char * src0,
    ulong  offset0,
    global char * src1,
    ulong  offset1,
    global char * dst,
    ulong  offsetd,
    ulong nb01,
    ulong nb11,
    int ne0,
    ulong nb1,
    int ne00_off,
    int ne10_off
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    global half * src0_row = (global half *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
    global half * src1_row = (global half *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
    global half * dst_row  = (global half *) ((global char *) dst  + get_group_id(0)*nb1);

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const half x0 = src0_row[i0];
        const half x1 = src1_row[i0];

        const half gelu = 0.5f*x0*(1.0f + tanh(SQRT_2_OVER_PI*x0*(1.0f + GELU_COEF_A*x0*x0)));

        dst_row[i0] = gelu*x1;
    }
}

//------------------------------------------------------------------------------
// reglu
//------------------------------------------------------------------------------
kernel void kernel_reglu(
    global char * src0,
    ulong  offset0,
    global char * src1,
    ulong  offset1,
    global char * dst,
    ulong  offsetd,
    ulong nb01,
    ulong nb11,
    int ne0,
    ulong nb1,
    int ne00_off,
    int ne10_off
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    global float * src0_row = (global float *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
    global float * src1_row = (global float *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
    global float * dst_row  = (global float *) ((global char *) dst  + get_group_id(0)*nb1);

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        dst_row[i0] = x0*x1*(x0 > 0.0f);
    }
}

kernel void kernel_reglu_f16(
    global char * src0,
    ulong  offset0,
    global char * src1,
    ulong  offset1,
    global char * dst,
    ulong  offsetd,
    ulong nb01,
    ulong nb11,
    int ne0,
    ulong nb1,
    int ne00_off,
    int ne10_off
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    global half * src0_row = (global half *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
    global half * src1_row = (global half *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
    global half * dst_row  = (global half *) ((global char *) dst  + get_group_id(0)*nb1);

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const half x0 = src0_row[i0];
        const half x1 = src1_row[i0];

        dst_row[i0] = x0*x1*(x0 > 0.0f);
    }
}

//------------------------------------------------------------------------------
// swiglu
//------------------------------------------------------------------------------
kernel void kernel_swiglu(
    global char * src0,
    ulong  offset0,
    global char * src1,
    ulong  offset1,
    global char * dst,
    ulong  offsetd,
    ulong nb01,
    ulong nb11,
    int ne0,
    ulong nb1,
    int ne00_off,
    int ne10_off
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    global float * src0_row = (global float *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
    global float * src1_row = (global float *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
    global float * dst_row  = (global float *) ((global char *) dst  + get_group_id(0)*nb1);

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float silu = x0 / (1.0f + exp(-x0));

        dst_row[i0] = silu*x1;
    }
}

kernel void kernel_swiglu_f16(
    global char * src0,
    ulong  offset0,
    global char * src1,
    ulong  offset1,
    global char * dst,
    ulong  offsetd,
    ulong nb01,
    ulong nb11,
    int ne0,
    ulong nb1,
    int ne00_off,
    int ne10_off
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    global half * src0_row = (global half *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
    global half * src1_row = (global half *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
    global half * dst_row  = (global half *) ((global char *) dst  + get_group_id(0)*nb1);

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const half x0 = src0_row[i0];
        const half x1 = src1_row[i0];

        const half silu = x0 / (1.0f + exp(-x0));

        dst_row[i0] = silu*x1;
    }
}

//------------------------------------------------------------------------------
// swiglu_oai
//------------------------------------------------------------------------------
kernel void kernel_swiglu_oai(
    global char * src0,
    ulong         offset0,
    global char * src1,
    ulong         offset1,
    global char * dst,
    ulong         offsetd,
    ulong         nb01,
    ulong         nb11,
    int           ne0,
    ulong         nb1,
    int           ne00_off,
    int           ne10_off,
    float         limit,
    float         alpha
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    global float * src0_row = (global float *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
    global float * src1_row = (global float *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
    global float * dst_row  = (global float *) ((global char *) dst  + get_group_id(0)*nb1);

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        float x0 = src0_row[i0];
        float x1 = src1_row[i0];

        x0 = min(x0, limit);
        x1 = max(min(x1, limit), -limit);

        float out_glu = x0 / (1.0f + exp(-x0 * alpha));
        out_glu = out_glu * (1.0f + x1);

        dst_row[i0] = out_glu;
    }
}

//------------------------------------------------------------------------------
// geglu_erf
//------------------------------------------------------------------------------
kernel void kernel_geglu_erf(
    global char * src0,
    ulong  offset0,
    global char * src1,
    ulong  offset1,
    global char * dst,
    ulong  offsetd,
    ulong nb01,
    ulong nb11,
    int ne0,
    ulong nb1,
    int ne00_off,
    int ne10_off
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    global float * src0_row = (global float *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
    global float * src1_row = (global float *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
    global float * dst_row  = (global float *) ((global char *) dst  + get_group_id(0)*nb1);

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float gelu_erf = 0.5f*x0*(1.0f + erf(x0*SQRT_2_INV));

        dst_row[i0] = gelu_erf*x1;
    }
}

kernel void kernel_geglu_erf_f16(
    global char * src0,
    ulong  offset0,
    global char * src1,
    ulong  offset1,
    global char * dst,
    ulong  offsetd,
    ulong nb01,
    ulong nb11,
    int ne0,
    ulong nb1,
    int ne00_off,
    int ne10_off
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    global half * src0_row = (global half *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
    global half * src1_row = (global half *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
    global half * dst_row  = (global half *) ((global char *) dst  + get_group_id(0)*nb1);

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const half x0 = src0_row[i0];
        const half x1 = src1_row[i0];

        const half gelu_erf = 0.5f*x0*(1.0f + erf(x0*SQRT_2_INV));

        dst_row[i0] = gelu_erf*x1;
    }
}

//------------------------------------------------------------------------------
// geglu_quick
//------------------------------------------------------------------------------
kernel void kernel_geglu_quick(
    global char * src0,
    ulong  offset0,
    global char * src1,
    ulong  offset1,
    global char * dst,
    ulong  offsetd,
    ulong nb01,
    ulong nb11,
    int ne0,
    ulong nb1,
    int ne00_off,
    int ne10_off
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    global float * src0_row = (global float *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
    global float * src1_row = (global float *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
    global float * dst_row  = (global float *) ((global char *) dst  + get_group_id(0)*nb1);

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float gelu_quick = x0*(1.0f/(1.0f + exp(GELU_QUICK_COEF*x0)));

        dst_row[i0] = gelu_quick*x1;
    }
}

kernel void kernel_geglu_quick_f16(
    global char * src0,
    ulong  offset0,
    global char * src1,
    ulong  offset1,
    global char * dst,
    ulong  offsetd,
    ulong nb01,
    ulong nb11,
    int ne0,
    ulong nb1,
    int ne00_off,
    int ne10_off
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    global half * src0_row = (global half *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
    global half * src1_row = (global half *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
    global half * dst_row  = (global half *) ((global char *) dst  + get_group_id(0)*nb1);

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const half x0 = src0_row[i0];
        const half x1 = src1_row[i0];

        const half gelu_quick = x0*(1.0f/(1.0f + exp(GELU_QUICK_COEF*x0)));

        dst_row[i0] = gelu_quick*x1;
    }
}
