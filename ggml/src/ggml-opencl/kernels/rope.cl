#pragma OPENCL EXTENSION cl_khr_fp16 : enable

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

kernel void kernel_rope_multi_f32(
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
        float beta_slow,
        int4 sections
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

    const int sect_dims = sections.s0 + sections.s1 + sections.s2 + sections.s3;
    const int sec_w = sections.s1 + sections.s0;

    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        if (i0 < n_dims) {
            int ic = i0/2;

            const int sector = (i0 / 2) % sect_dims;
            float theta_base = 0.0f;

            if (sector < sections.s0) {
                theta_base = pos[i2];
            }
            else if (sector >= sections.s0 && sector < sec_w) {
                theta_base = pos[i2 + ne2 * 1];
            }
            else if (sector >= sec_w && sector < sec_w + sections.s2) {
                theta_base = pos[i2 + ne2 * 2];
            }
            else if (sector >= sec_w + sections.s2) {
                theta_base = pos[i2 + ne2 * 3];
            }

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

kernel void kernel_rope_multi_f16(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * src2,
        ulong offset2,
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
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        int4 sections
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

    const int sect_dims = sections.s0 + sections.s1 + sections.s2 + sections.s3;
    const int sec_w = sections.s1 + sections.s0;

    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        if (i0 < n_dims) {
            int ic = i0/2;

            const int sector = (i0 / 2) % sect_dims;
            float theta_base = 0.0f;

            if (sector < sections.s0) {
                theta_base = pos[i2];
            }
            else if (sector >= sections.s0 && sector < sec_w) {
                theta_base = pos[i2 + ne2 * 1];
            }
            else if (sector >= sec_w && sector < sec_w + sections.s2) {
                theta_base = pos[i2 + ne2 * 2];
            }
            else if (sector >= sec_w + sections.s2) {
                theta_base = pos[i2 + ne2 * 3];
            }

            const float theta = theta_base * pow(freq_base, inv_ndims*i0);

            const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

            global half * src      = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
            global half * dst_data = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

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

kernel void kernel_rope_vision_f32(
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
        float beta_slow,
        int4 sections
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

    const int sect_dims = sections.s0 + sections.s1;
    const int sec_w = sections.s1 + sections.s0;

    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        int ic = i0/2;

        const int sector = (i0/2) % sect_dims;
        float theta_base = 0.0f;

        if (sector < sections.s0) {
            const int p = sector;
            theta_base = pos[i2] * pow(freq_base, inv_ndims*2.0f*p);
        } else if (sector >= sections.s0 && sector < sec_w) {
            const int p = sector - sections.s0;
            theta_base = pos[i2 + ne2] * pow(freq_base, inv_ndims*2.0f*p);
        }

        const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

        float2 cos_sin_theta = rope_yarn(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

        global float * src      = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
        global float * dst_data = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

        const float x0 = src[0];
        const float x1 = src[n_dims];

        dst_data[0]      = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
        dst_data[n_dims] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
    }
}

kernel void kernel_rope_vision_f16(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * src2,
        ulong offset2,
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
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        int4 sections
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

    const int sect_dims = sections.s0 + sections.s1;
    const int sec_w = sections.s1 + sections.s0;

    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        int ic = i0/2;

        const int sector = (i0/2) % sect_dims;
        float theta_base = 0.0f;

        if (sector < sections.s0) {
            const int p = sector;
            theta_base = pos[i2] * pow(freq_base, inv_ndims*2.0f*p);
        } else if (sector >= sections.s0 && sector < sec_w) {
            const int p = sector - sections.s0;
            theta_base = pos[i2 + ne2] * pow(freq_base, inv_ndims*2.0f*p);
        }

        const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

        float2 cos_sin_theta = rope_yarn(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

        global half * src      = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
        global half * dst_data = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

        const float x0 = src[0];
        const float x1 = src[n_dims];

        dst_data[0]      = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
        dst_data[n_dims] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
    }
}
