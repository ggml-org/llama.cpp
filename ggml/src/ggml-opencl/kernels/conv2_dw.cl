#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void kernel_conv2d_dw_f32_f32(
        global const float * weights,
        long                 weights_offset,
        global const float * input,
        long                 input_offset,
        global       float * output,
        long                 output_offset,
        long C,
        long N,
        long KW,
        long KH,
        long W,
        long H,
        long OW,
        long OH,
        int s0,
        int s1,
        int p0,
        int p1,
        int d0,
        int d1,
        long nb01,
        long nb03,
        long nb11,
        long nb12,
        long nb13,
        long nb1,
        long nb2,
        long nb3
) {
    weights = (global const float *) ((global const char *) weights + weights_offset);
    input = (global const float *) ((global const char *) input + input_offset);
    output = (global float *) ((global char *) output + output_offset);

    const long gid = get_global_id(0);
    const long OHW = OH * OW;
    const long COHW = C * OHW;
    const long total = N * COHW;
    if (gid >= total) {
        return;
    }

    const long n = gid / COHW;
    const long c_rem = gid - n * COHW;
    const long c = c_rem / OHW;
    const long oh = (c_rem - c * OHW) / OW;
    const long ow = c_rem - c * OHW - oh * OW;

    float sum = 0.0f;
    for (long kh = 0; kh < KH; ++kh) {
        const long ih = oh * s1 + kh * d1 - p1;
        if (ih < 0 || ih >= H) {
            continue;
        }
        for (long kw = 0; kw < KW; ++kw) {
            const long iw = ow * s0 + kw * d0 - p0;
            if (iw < 0 || iw >= W) {
                continue;
            }
            const long weight_idx = kw + kh * nb01 + c * nb03;
            const long input_idx = iw + ih * nb11 + c * nb12 + n * nb13;
            sum += weights[weight_idx] * input[input_idx];
        }
    }

    const long output_idx = ow + oh * nb1 + c * nb2 + n * nb3;
    output[output_idx] = sum;
}

kernel void kernel_conv2d_dwf16_f32(
        global const half * weights,
        long                weights_offset,
        global const float * input,
        long                 input_offset,
        global       float * output,
        long                 output_offset,
        long C,
        long N,
        long KW,
        long KH,
        long W,
        long H,
        long OW,
        long OH,
        int s0,
        int s1,
        int p0,
        int p1,
        int d0,
        int d1,
        long nb01,
        long nb03,
        long nb11,
        long nb12,
        long nb13,
        long nb1,
        long nb2,
        long nb3
) {
    weights = (global const half *) ((global const char *) weights + weights_offset);
    input = (global const float *) ((global const char *) input + input_offset);
    output = (global float *) ((global char *) output + output_offset);

    const long gid = get_global_id(0);
    const long OHW = OH * OW;
    const long COHW = C * OHW;
    const long total = N * COHW;
    if (gid >= total) {
        return;
    }

    const long n = gid / COHW;
    const long c_rem = gid - n * COHW;
    const long c = c_rem / OHW;
    const long oh = (c_rem - c * OHW) / OW;
    const long ow = c_rem - c * OHW - oh * OW;

    float sum = 0.0f;
    for (long kh = 0; kh < KH; ++kh) {
        const long ih = oh * s1 + kh * d1 - p1;
        if (ih < 0 || ih >= H) {
            continue;
        }
        for (long kw = 0; kw < KW; ++kw) {
            const long iw = ow * s0 + kw * d0 - p0;
            if (iw < 0 || iw >= W) {
                continue;
            }
            const long weight_idx = kw + kh * nb01 + c * nb03;
            const long input_idx = iw + ih * nb11 + c * nb12 + n * nb13;
            sum += (float) weights[weight_idx] * input[input_idx];
        }
    }

    const long output_idx = ow + oh * nb1 + c * nb2 + n * nb3;
    output[output_idx] = sum;
}
