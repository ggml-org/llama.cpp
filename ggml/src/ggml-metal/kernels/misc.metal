#include "common.h"

kernel void kernel_argmax_f32(
        constant ggml_metal_kargs_argmax & args,
        device   const char * src0,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    device const float * x_row = (device const float *) ((device const char *) src0 + tgpig * args.nb01);

    float   lmax = -INFINITY;
    int32_t larg = -1;

    for (int i00 = tpitg; i00 < args.ne00; i00 += ntg) {
        if (x_row[i00] > lmax) {
            lmax = x_row[i00];
            larg = i00;
        }
    }

    // find the argmax value in the block
    float max_val = simd_max(lmax);
    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

    device int32_t * dst_i32 = (device int32_t *) dst;

    threadgroup   float * shared_maxval = (threadgroup   float *) shmem;
    threadgroup int32_t * shared_argmax = (threadgroup int32_t *) shmem + N_SIMDWIDTH;

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_maxval[tiisg] = -INFINITY;
            shared_argmax[tiisg] = -1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_maxval[sgitg] = max_val;
            shared_argmax[sgitg] = arg_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = shared_maxval[tiisg];
        arg_val = shared_argmax[tiisg];

        float max_val_reduced   = simd_max(max_val);
        int32_t arg_val_reduced = simd_max(select(-1, arg_val, max_val == max_val_reduced));

        dst_i32[tgpig] = arg_val_reduced;

        return;
    }

    dst_i32[tgpig] = arg_val;
}


kernel void kernel_diag_f32(
        constant ggml_metal_kargs_diag & args,
        device   const char * src0,
        device         char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;

    const int32_t i3 = tgpig.z;
    const int32_t i2 = tgpig.y;
    const int32_t i1 = tgpig.x;

    device const float * src0_ptr = (device const float *)(src0 +                i2*args.nb02 + i3*args.nb03);
    device       float * dst_ptr  = (device       float *)(dst  + i1*args.nb01 + i2*args.nb2  + i3*args.nb3);

    for (int i0 = tiitg; i0 < args.ne0; i0 += NW) {
        dst_ptr[i0] = i0 == i1 ? src0_ptr[i0] : 0.0f;
    }
}

constant bool FC_rope_is_imrope [[function_constant(FC_ROPE + 0)]];

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int i0, float ext_factor, float mscale,
    thread float * cos_theta, thread float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    *cos_theta = cos(theta) * mscale;
    *sin_theta = sin(theta) * mscale;
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_fac(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
static float rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * log(n_ctx_orig / (n_rot * 2 * M_PI_F)) / (2 * log(base));
}

static void rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
    // start and end correction dims
    dims[0] = max(0.0f,         floor(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base)));
    dims[1] = min(n_dims - 1.0f, ceil(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)));
}

template<typename T>
kernel void kernel_rope_norm(
        constant ggml_metal_kargs_rope & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 tptg [[threads_per_threadgroup]],
        uint3   tgpig[[threadgroup_position_in_grid]]) {
    const int i3 = tgpig[2];
    const int i2 = tgpig[1];
    const int i1 = tgpig[0];

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    device const int32_t * pos = (device const int32_t *) src1;

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.f/args.n_dims;

    float cos_theta;
    float sin_theta;

    for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {
        if (i0 < args.n_dims) {
            const int ic = i0/2;

            const float theta = theta_base * pow(args.freq_base, inv_ndims*i0);

            const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;

            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);

            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);

            const float x0 = src[0];
            const float x1 = src[1];

            dst_data[0] = x0*cos_theta - x1*sin_theta;
            dst_data[1] = x0*sin_theta + x1*cos_theta;
        } else {
            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

template<typename T>
kernel void kernel_rope_neox(
        constant ggml_metal_kargs_rope & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 tptg [[threads_per_threadgroup]],
        uint3   tgpig[[threadgroup_position_in_grid]]) {
    const int i3 = tgpig[2];
    const int i2 = tgpig[1];
    const int i1 = tgpig[0];

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    device const int32_t * pos = (device const int32_t *) src1;

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.f/args.n_dims;

    float cos_theta;
    float sin_theta;

    for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {
        if (i0 < args.n_dims) {
            const int ic = i0/2;

            const float theta = theta_base * pow(args.freq_base, inv_ndims*i0);

            const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;

            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);

            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + ic*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + ic*args.nb0);

            const float x0 = src[0];
            const float x1 = src[args.n_dims/2];

            dst_data[0]             = x0*cos_theta - x1*sin_theta;
            dst_data[args.n_dims/2] = x0*sin_theta + x1*cos_theta;
        } else {
            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

template<typename T>
kernel void kernel_rope_multi(
        constant ggml_metal_kargs_rope & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 tptg [[threads_per_threadgroup]],
        uint3   tgpig[[threadgroup_position_in_grid]]) {
    const int i3 = tgpig[2];
    const int i2 = tgpig[1];
    const int i1 = tgpig[0];

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    device const int32_t * pos = (device const int32_t *) src1;

    const float inv_ndims = -1.f/args.n_dims;

    float cos_theta;
    float sin_theta;

    for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {
        if (i0 < args.n_dims) {
            const int ic = i0/2;

            // mrope theta calculations
            // note: the rest is the same as kernel_rope_neox
            const int sect_dims = args.sect_0 + args.sect_1 + args.sect_2 + args.sect_3;
            const int sec_w01   = args.sect_0 + args.sect_1;               // end of section 1
            const int sec_w012  = args.sect_0 + args.sect_1 + args.sect_2; // end of section 2
            const int sector    = ic % sect_dims;

            float theta_base;
            if (FC_rope_is_imrope) {
                if (sector % 3 == 1 && sector < 3 * args.sect_1) { // h
                    theta_base = (float) pos[i2 + args.ne02 * 1];
                } else if (sector % 3 == 2 && sector < 3 * args.sect_2) { // w
                    theta_base = (float) pos[i2 + args.ne02 * 2];
                } else if (sector % 3 == 0 && sector < 3 * args.sect_0) { // t
                    theta_base = (float) pos[i2 + args.ne02 * 0];
                } else { // e
                    theta_base = (float) pos[i2 + args.ne02 * 3];
                }
            } else {
                if (sector < args.sect_0) {
                    theta_base = (float) pos[i2];
                } else if (sector < sec_w01) {
                    theta_base = (float) pos[i2 + args.ne02 * 1];
                } else if (sector < sec_w012) {
                    theta_base = (float) pos[i2 + args.ne02 * 2];
                } else {
                    theta_base = (float) pos[i2 + args.ne02 * 3];
                }
            }
            // end of mrope

            const float theta = theta_base * pow(args.freq_base, inv_ndims*i0);

            const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;

            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);

            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + ic*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + ic*args.nb0);

            const float x0 = src[0];
            const float x1 = src[args.n_dims/2];

            dst_data[0]             = x0*cos_theta - x1*sin_theta;
            dst_data[args.n_dims/2] = x0*sin_theta + x1*cos_theta;
        } else {
            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

template<typename T>
kernel void kernel_rope_vision(
        constant ggml_metal_kargs_rope & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 tptg [[threads_per_threadgroup]],
        uint3   tgpig[[threadgroup_position_in_grid]]) {
    const int i3 = tgpig[2];
    const int i2 = tgpig[1];
    const int i1 = tgpig[0];

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    device const int32_t * pos = (device const int32_t *) src1;

    const float inv_ndims = -1.f/args.n_dims;

    float cos_theta;
    float sin_theta;

    for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {
        if (i0 < 2*args.n_dims) { // different from kernel_rope_multi
            const int ic = i0/2;

            // mrope theta calculations (only support 2 dimensions)
            const int sect_dims = args.sect_0 + args.sect_1;
            const int sector    = ic % sect_dims;

            float p;
            float theta_base;
            if (sector < args.sect_1) {
                p = (float) sector;
                theta_base = (float) pos[i2];
            } else {
                p = (float) sector - args.sect_0;
                theta_base = (float) pos[i2 + args.ne02];
            }

            const float theta = theta_base * pow(args.freq_base, 2.0f * inv_ndims * p);
            // end of mrope

            const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;

            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);

            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + ic*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + ic*args.nb0);

            const float x0 = src[0];
            const float x1 = src[args.n_dims]; // different from kernel_rope_multi

            dst_data[0]           = x0*cos_theta - x1*sin_theta;
            dst_data[args.n_dims] = x0*sin_theta + x1*cos_theta; // different from kernel_rope_multi
        } else {
            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

typedef decltype(kernel_rope_norm<float>) kernel_rope_norm_t;
typedef decltype(kernel_rope_neox<float>) kernel_rope_neox_t;
typedef decltype(kernel_rope_multi<float>) kernel_rope_multi_t;
typedef decltype(kernel_rope_vision<float>) kernel_rope_vision_t;

template [[host_name("kernel_rope_norm_f32")]] kernel kernel_rope_norm_t kernel_rope_norm<float>;
template [[host_name("kernel_rope_norm_f16")]] kernel kernel_rope_norm_t kernel_rope_norm<half>;

template [[host_name("kernel_rope_neox_f32")]] kernel kernel_rope_neox_t kernel_rope_neox<float>;
template [[host_name("kernel_rope_neox_f16")]] kernel kernel_rope_neox_t kernel_rope_neox<half>;

template [[host_name("kernel_rope_multi_f32")]] kernel kernel_rope_multi_t kernel_rope_multi<float>;
template [[host_name("kernel_rope_multi_f16")]] kernel kernel_rope_multi_t kernel_rope_multi<half>;

template [[host_name("kernel_rope_vision_f32")]] kernel kernel_rope_vision_t kernel_rope_vision<float>;
template [[host_name("kernel_rope_vision_f16")]] kernel kernel_rope_vision_t kernel_rope_vision<half>;

typedef void (im2col_t)(
        constant ggml_metal_kargs_im2col & args,
        device const float * x,
        device        char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]);

template <typename T>
kernel void kernel_im2col(
        constant ggml_metal_kargs_im2col & args,
        device const float * x,
        device        char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
//    const int64_t IC = tgpg[0];
    const int64_t OH = tgpg[1];
    const int64_t OW = tgpg[2];

    const int64_t KH = ntg[1];
    const int64_t KW = ntg[2];

          int64_t in  = tpitg[0];
    const int64_t ikh = tpitg[1];
    const int64_t ikw = tpitg[2];

    const int64_t iic = tgpig[0];
    const int64_t ioh = tgpig[1];
    const int64_t iow = tgpig[2];

    const int64_t iiw = iow*args.s0 + ikw*args.d0 - args.p0;
    const int64_t iih = ioh*args.s1 + ikh*args.d1 - args.p1;

    int64_t offset_dst = (in*OH*OW + ioh*OW + iow)*args.CHW + (iic*(KH*KW) + ikh*KW + ikw);

    device T * pdst = (device T *) (dst);

    if (iih < 0 || iih >= args.IH || iiw < 0 || iiw >= args.IW) {
        while (in < args.N) {
            pdst[offset_dst] = 0.0f;
            offset_dst += ntg[0]*args.CHW*OH*OW;

            in += ntg[0];
        }
    } else {
        int64_t offset_src = in*args.ofs0 + iic*args.ofs1 + iih*args.IW + iiw;

        while (in < args.N) {
            pdst[offset_dst] = x[offset_src];

            offset_dst += ntg[0]*args.CHW*OH*OW;
            offset_src += ntg[0]*args.ofs0;

            in += ntg[0];
        }
    }
}

template [[host_name("kernel_im2col_f32")]] kernel im2col_t kernel_im2col<float>;
template [[host_name("kernel_im2col_f16")]] kernel im2col_t kernel_im2col<half>;

// TODO: optimize
typedef void (im2col_ext_t)(
        constant ggml_metal_kargs_im2col & args,
        device const float * x,
        device        char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]);

template <typename T>
kernel void kernel_im2col_ext(
        constant ggml_metal_kargs_im2col & args,
        device const float * x,
        device        char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],      // tgpg[0] = D x IC x KH x KW, CHW = IC x KH x KW
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {  // [M, 1, 1]
    const int64_t KHW = (int64_t)args.KHW;

    const int64_t d   = tgpig[0] / args.CHW;
    const int64_t chw = tgpig[0] % args.CHW;
    const int64_t tgpig_0 = chw / KHW;  // 0 ~ (IC - 1)
    const int64_t HW = tgpig[0] % KHW;

    const int64_t tpitg_0 = (d * ntg[0]) + tpitg[0];
    if (tpitg_0 >= args.N) {
        return;
    }

    const int64_t tpitg_1 = HW / args.KW;
    const int64_t tpitg_2 = HW % args.KW;

    const int64_t iiw = tgpig[2] * args.s0 + tpitg_2 * args.d0 - args.p0;
    const int64_t iih = tgpig[1] * args.s1 + tpitg_1 * args.d1 - args.p1;

    const int64_t offset_dst =
        (tpitg_0 * tgpg[1] * tgpg[2] + tgpig[1] * tgpg[2] + tgpig[2]) * args.CHW +
        (tgpig_0 * KHW + tpitg_1 * args.KW + tpitg_2);

    device T * pdst = (device T *) (dst);

    if (iih < 0 || iih >= args.IH || iiw < 0 || iiw >= args.IW) {
        pdst[offset_dst] = 0.0f;
    } else {
        const int64_t offset_src = tpitg_0 * args.ofs0 + tgpig_0 * args.ofs1;
        pdst[offset_dst] = x[offset_src + iih * args.IW + iiw];
    }
}

template [[host_name("kernel_im2col_ext_f32")]] kernel im2col_ext_t kernel_im2col_ext<float>;
template [[host_name("kernel_im2col_ext_f16")]] kernel im2col_ext_t kernel_im2col_ext<half>;

template <typename TK>
kernel void kernel_conv_2d(
        constant ggml_metal_kargs_conv_2d & args,
        device const char * weights,
        device const char * src,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3    tgpg[[threadgroups_per_grid]],
        uint3   tpitg[[thread_position_in_threadgroup]],
        uint3     ntg[[threads_per_threadgroup]]) {

    const uint threads_per_tg = ntg.x * ntg.y * ntg.z;
    const uint tg_index = (tgpig.z * tgpg.y + tgpig.y) * tgpg.x + tgpig.x;
    const uint local_thread = tpitg.z * (ntg.x * ntg.y) + tpitg.y * ntg.x + tpitg.x;
    const uint thread_index = tg_index * threads_per_tg + local_thread;
    const uint64_t total_threads = (uint64_t) threads_per_tg * tgpg.x * tgpg.y * tgpg.z;
    const uint64_t total_outputs = (uint64_t) args.N * args.OC * args.OH * args.OW;

    for (uint64_t index = thread_index; index < total_outputs; index += total_threads) {
        uint64_t tmp = index;

        const int32_t ow = tmp % args.OW; tmp /= args.OW;
        const int32_t oh = tmp % args.OH; tmp /= args.OH;
        const int32_t oc = tmp % args.OC; tmp /= args.OC;
        const int32_t  n = tmp;

        float acc = 0.0f;

        const int32_t base_x = ow*args.s0 - args.p0;
        const int32_t base_y = oh*args.s1 - args.p1;

        int32_t ky_start = 0;
        if (base_y < 0) {
            ky_start = (-base_y + args.d1 - 1)/args.d1;
        }
        int32_t ky_end = args.KH;
        const int32_t y_max = args.IH - 1 - base_y;
        if (y_max < 0) {
            ky_end = ky_start;
        } else if (base_y + (args.KH - 1)*args.d1 >= args.IH) {
            ky_end = min(ky_end, y_max/args.d1 + 1);
        }

        int32_t kx_start = 0;
        if (base_x < 0) {
            kx_start = (-base_x + args.d0 - 1)/args.d0;
        }
        int32_t kx_end = args.KW;
        const int32_t x_max = args.IW - 1 - base_x;
        if (x_max < 0) {
            kx_end = kx_start;
        } else if (base_x + (args.KW - 1)*args.d0 >= args.IW) {
            kx_end = min(kx_end, x_max/args.d0 + 1);
        }

        if (ky_start < ky_end && kx_start < kx_end) {
            const uint64_t src_base_n = (uint64_t) n  * args.nb13;
            const uint64_t w_base_oc  = (uint64_t) oc * args.nb03;

            for (int32_t ic = 0; ic < args.IC; ++ic) {
                const uint64_t src_base_nc = src_base_n + (uint64_t) ic * args.nb12;
                const uint64_t w_base_ocic = w_base_oc  + (uint64_t) ic * args.nb02;

                for (int32_t ky = ky_start; ky < ky_end; ++ky) {
                    const int32_t iy = base_y + ky*args.d1;
                    const uint64_t src_base_row = src_base_nc + (uint64_t) iy * args.nb11;
                    const uint64_t w_base_row   = w_base_ocic + (uint64_t) ky * args.nb01;

                    for (int32_t kx = kx_start; kx < kx_end; ++kx) {
                        const int32_t ix = base_x + kx*args.d0;
                        const uint64_t src_offs = src_base_row + (uint64_t) ix * args.nb10;
                        const uint64_t w_offs   = w_base_row   + (uint64_t) kx * args.nb00;

                        const float x = *(device const float *)(src + src_offs);
                        const float w = (float) (*(device const TK *)(weights + w_offs));

                        acc += x * w;
                    }
                }
            }
        }

        const uint64_t dst_offs =
            (uint64_t) n  * args.nb3 +
            (uint64_t) oc * args.nb2 +
            (uint64_t) oh * args.nb1 +
            (uint64_t) ow * args.nb0;

        *(device float *)(dst + dst_offs) = acc;
    }
}

template [[host_name("kernel_conv_2d_f32_f32")]]
kernel void kernel_conv_2d<float>(
        constant ggml_metal_kargs_conv_2d & args,
        device const char * weights,
        device const char * src,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3    tgpg[[threadgroups_per_grid]],
        uint3   tpitg[[thread_position_in_threadgroup]],
        uint3     ntg[[threads_per_threadgroup]]);

template [[host_name("kernel_conv_2d_f16_f32")]]
kernel void kernel_conv_2d<half>(
        constant ggml_metal_kargs_conv_2d & args,
        device const char * weights,
        device const char * src,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3    tgpg[[threadgroups_per_grid]],
        uint3   tpitg[[thread_position_in_threadgroup]],
        uint3     ntg[[threads_per_threadgroup]]);

typedef void (conv_transpose_1d_t)(
        constant ggml_metal_kargs_conv_transpose_1d & args,
        device const float * src0,
        device const float * src1,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3    tgpg[[threadgroups_per_grid]]);

template <typename T>
kernel void kernel_conv_transpose_1d(
        constant ggml_metal_kargs_conv_transpose_1d & args,
        device const     T * src0,
        device const float * src1,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3   tgpg[[threadgroups_per_grid]]) {

    // For output position j on the time axis, only input positions
    //   i such that i*s0 <= j < i*s0 + K
    // contribute -- i.e. i in [ceil((j - K + 1)/s0), floor(j/s0)]
    // intersected with [0, IL-1]. That's at most ceil(K/s0) values
    // (typically 2 for stride==K/2 transposed convs).
    const int32_t j  = tgpig[0];
    const int32_t s0 = args.s0;
    const int32_t K  = args.K;
    const int32_t IL = args.IL;

    int32_t i_min;
    {
        int32_t a = j - K + 1;
        i_min = a <= 0 ? 0 : (a + s0 - 1) / s0; // ceil(a/s0) for a>0
    }
    int32_t i_max = j / s0;
    if (i_max > IL - 1) i_max = IL - 1;

    float v = 0.0f;
    if (i_min <= i_max) {
        for (int64_t c = 0; c < args.IC; c++) {
            const int32_t kernel_offset = c * tgpg[1] * K + K * tgpig[1];
            const int32_t input_offset  = c * IL;

            for (int32_t i = i_min; i <= i_max; i++) {
                v += float(src0[kernel_offset + j - i * s0]) * src1[input_offset + i];
            }
        }
    }

    device float * dst_ptr = (device float *) (dst + tgpig[0] * args.nb0 + tgpig[1] * args.nb1);

    dst_ptr[0] = v;
}

template [[host_name("kernel_conv_transpose_1d_f32_f32")]]
kernel void kernel_conv_transpose_1d<float>(
    constant ggml_metal_kargs_conv_transpose_1d & args,
    device const float * src0,
    device const float * src1,
    device        char * dst,
    uint3   tgpig[[threadgroup_position_in_grid]],
    uint3    tgpg[[threadgroups_per_grid]]);

template [[host_name("kernel_conv_transpose_1d_f16_f32")]]
kernel void kernel_conv_transpose_1d<half>(
    constant ggml_metal_kargs_conv_transpose_1d & args,
    device const half  * src0,
    device const float * src1,
    device        char * dst,
    uint3   tgpig[[threadgroup_position_in_grid]],
    uint3    tgpg[[threadgroups_per_grid]]);


typedef void (conv_transpose_2d_t)(
        constant ggml_metal_kargs_conv_transpose_2d & args,
        device const float * src0,
        device const float * src1,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3    tgpg[[threadgroups_per_grid]]);

template <typename T>
kernel void kernel_conv_transpose_2d(
        constant ggml_metal_kargs_conv_transpose_2d & args,
        device const T * src0,
        device const float * src1,
        device        char * dst,
        threadgroup float * shared_sum [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3   tpitg[[thread_position_in_threadgroup]],
        uint3     ntg[[threads_per_threadgroup]]) {

    const int64_t out_x = tgpig[0];
    const int64_t out_y = tgpig[1];
    const int64_t out_c = tgpig[2];

    const int64_t kw = tpitg[0];
    const int64_t kh = tpitg[1];

    float v = 0.0f;

    for (int64_t in_c = 0; in_c < args.IC; in_c++) {
        int64_t in_y = out_y - kh;

        if (in_y < 0 || in_y % args.s0) continue;

        in_y /= args.s0;

        if (in_y >= args.IH) continue;

        int64_t in_x = out_x - kw;

        if (in_x < 0 || in_x % args.s0) continue;

        in_x /= args.s0;

        if (in_x >= args.IW) continue;

        const int64_t input_idx = (args.IW * args.IH) * in_c + (args.IW) * in_y + in_x;
        const int64_t kernel_idx = (args.KH * args.KW * args.OC) * in_c + (args.KH * args.KW) * out_c + (args.KW) * kh + kw;

        v += (float)src0[kernel_idx] * src1[input_idx];
    }

    const uint tid = tpitg.y * ntg.x + tpitg.x;
    shared_sum[tid] = v;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        const uint num_threads = ntg.x * ntg.y;
        for (uint i = 0; i < num_threads; i++) {
            total += shared_sum[i];
        }

        device float * dst_ptr = (device float *) (dst + out_x*args.nb0 + out_y * args.nb1 + out_c*args.nb2);
        dst_ptr[0] = total;
    }
}

template [[host_name("kernel_conv_transpose_2d_f32_f32")]]
kernel void kernel_conv_transpose_2d<float>(
    constant ggml_metal_kargs_conv_transpose_2d & args,
    device const float * src0,
    device const float * src1,
    device        char * dst,
    threadgroup float * shared_sum [[threadgroup(0)]],
    uint3   tgpig[[threadgroup_position_in_grid]],
    uint3   tpitg[[thread_position_in_threadgroup]],
    uint3     ntg[[threads_per_threadgroup]]);

template [[host_name("kernel_conv_transpose_2d_f16_f32")]]
kernel void kernel_conv_transpose_2d<half>(
    constant ggml_metal_kargs_conv_transpose_2d & args,
    device const half  * src0,
    device const float * src1,
    device        char * dst,
    threadgroup float * shared_sum [[threadgroup(0)]],
    uint3   tgpig[[threadgroup_position_in_grid]],
    uint3   tpitg[[thread_position_in_threadgroup]],
    uint3     ntg[[threads_per_threadgroup]]);

constant bool FC_upscale_aa [[function_constant(FC_UPSCALE + 0)]];

kernel void kernel_upscale_nearest_f32(
    constant ggml_metal_kargs_upscale & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3/args.sf3;
    const int64_t i02 = i2/args.sf2;
    const int64_t i01 = i1/args.sf1;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const int64_t i00 = i0/args.sf0;

        device const float * src0_ptr = (device const float *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);
        device       float * dst_ptr  = (device       float *) (dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1  +  i0*args.nb0);

        dst_ptr[0] = src0_ptr[0];
    }
}

static inline float bilinear_tri(float x) {
    return MAX(0.0f, 1.0f - fabs(x));
}

kernel void kernel_upscale_bilinear_f32(
    constant ggml_metal_kargs_upscale & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3 / args.sf3;
    const int64_t i02 = i2 / args.sf2;

    const float   f01  = ((float)i1 + args.poffs) / args.sf1 - args.poffs;
    const int64_t i01  = MAX(0, MIN(args.ne01 - 1, (int64_t)floor(f01)));
    const int64_t i01p = MAX(0, MIN(args.ne01 - 1, i01 + 1));
    const float   fd1  = MAX(0.0f, MIN(1.0f, f01 - (float)i01));

    src0 += i03*args.nb03 + i02*args.nb02;

    device float * dst_ptr = (device float *)(dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1);

    if (FC_upscale_aa) {
        const float support0  = MAX(1.0f, 1.0f / args.sf0);
        const float invscale0 = 1.0f / support0;
        const float support1  = MAX(1.0f, 1.0f / args.sf1);
        const float invscale1 = 1.0f / support1;

        for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
            const float f00 = ((float)i0 + args.poffs) / args.sf0 - args.poffs;

            int64_t x_min = MAX((int64_t)0, (int64_t)floor(f00 - support0 + args.poffs));
            int64_t x_max = MIN(args.ne00,  (int64_t)ceil (f00 + support0 + args.poffs));

            int64_t y_min = MAX((int64_t)0, (int64_t)floor(f01 - support1 + args.poffs));
            int64_t y_max = MIN(args.ne01,  (int64_t)ceil (f01 + support1 + args.poffs));

            float sum = 0.0f;
            float wsum = 0.0f;

            for (int64_t sy = y_min; sy < y_max; ++sy) {
                const float wy = MAX(0.0f, 1.0f - fabs((float)sy - f01) * invscale1);
                for (int64_t sx = x_min; sx < x_max; ++sx) {
                    const float wx = MAX(0.0f, 1.0f - fabs((float)sx - f00) * invscale0);
                    const float w  = wx * wy;
                    device const float * src_ptr = (device const float *)(src0 + sy*args.nb01 + sx*args.nb00);
                    sum  += (*src_ptr) * w;
                    wsum += w;
                }
            }

            const float v = (wsum > 0.0f) ? (sum / wsum) : 0.0f;
            dst_ptr[i0] = v;
        }
    } else {
        for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
            const float   f00  = ((float)i0 + args.poffs) / args.sf0 - args.poffs;
            const int64_t i00  = MAX(0, MIN(args.ne00 - 1, (int64_t)floor(f00)));
            const int64_t i00p = MAX(0, MIN(args.ne00 - 1, i00 + 1));
            const float   fd0  = MAX(0.0f, MIN(1.0f, f00 - (float)i00));

            device const float * src00 = (device const float *)(src0 + i01*args.nb01  + i00*args.nb00);
            device const float * src10 = (device const float *)(src0 + i01*args.nb01  + i00p*args.nb00);
            device const float * src01 = (device const float *)(src0 + i01p*args.nb01 + i00*args.nb00);
            device const float * src11 = (device const float *)(src0 + i01p*args.nb01 + i00p*args.nb00);

            const float v =
                (*src00) * (1.0f - fd0) * (1.0f - fd1) +
                (*src10) * fd0          * (1.0f - fd1) +
                (*src01) * (1.0f - fd0) * fd1 +
                (*src11) * fd0          * fd1;

            dst_ptr[i0] = v;
        }
    }
}

template <typename T>
kernel void kernel_conv_3d(
        constant ggml_metal_kargs_conv_3d & args,
        device const  char * src0, // Weights [IC * OC, KD, KH, KW]
        device const  char * src1, // Inputs  [IC * N,  ID, IH, IW]
        device       char  * dst,  // Outputs [OC * N,  OD, OH, OW]
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]]) {

    // 1. Un-flatten the spatial dimension from Grid X
    int64_t spatial_idx = tgpig.x * 32 + tpitg.x;

    if (spatial_idx >= args.OW * args.OH * args.OD) {
        return; // Thread falls outside the spatial volume
    }

    int64_t od = spatial_idx / (args.OW * args.OH);
    int64_t oh = (spatial_idx / args.OW) % args.OH;
    int64_t ow = spatial_idx % args.OW;

    // 2. Map Y to Channels, Z to Batch
    int64_t oc = tgpig.y;
    int64_t batch_idx = tgpig.z;

    // 3. Calculate anchor coordinates in the Input volume
    int64_t i_w_base = ow * args.s0 - args.p0;
    int64_t i_h_base = oh * args.s1 - args.p1;
    int64_t i_d_base = od * args.s2 - args.p2;

    float sum = 0.0f;

    // 4. Gather Loop (Iterate over Input Channels -> Depth -> Height -> Width)
    for (int64_t ic = 0; ic < args.IC; ++ic) {

        // ggml packs batch and channel together in the 4th dimension
        int64_t src_cn_idx = batch_idx * args.IC + ic;
        int64_t w_cn_idx   = oc * args.IC + ic;

        for (int64_t kz = 0; kz < args.KD; ++kz) {
            int64_t id = i_d_base + kz * args.d2;
            if (id < 0 || id >= args.ID) continue; // Boundary check (Padding)

            for (int64_t ky = 0; ky < args.KH; ++ky) {
                int64_t ih = i_h_base + ky * args.d1;
                if (ih < 0 || ih >= args.IH) continue;

                for (int64_t kx = 0; kx < args.KW; ++kx) {
                    int64_t iw = i_w_base + kx * args.d0;
                    if (iw < 0 || iw >= args.IW) continue;

                    // Convert multi-dimensional coordinates to flat byte offsets
                    int64_t w_idx = kx*args.nb00 + ky*args.nb01 + kz*args.nb02 + w_cn_idx*args.nb03;
                    int64_t i_idx = iw*args.nb10 + ih*args.nb11 + id*args.nb12 + src_cn_idx*args.nb13;

                    // Dereference memory and cast weights to f32 if they were f16
                    float w_val = (float)*(device const T*)((device const char*)src0 + w_idx);
                    float i_val = *(device const float*)((device const char*)src1 + i_idx);

                    sum += w_val * i_val;
                }
            }
        }
    }

    // 5. Write the accumulated value out to RAM
    int64_t dst_cn_idx = batch_idx * args.OC + oc;
    int64_t d_idx = ow*args.nb0 + oh*args.nb1 + od*args.nb2 + dst_cn_idx*args.nb3;

    *(device float*)(dst + d_idx) = sum;
}

// Explicit instantiations so the JIT compiler can find them by name
template [[host_name("kernel_conv_3d_f32_f32")]]
kernel void kernel_conv_3d<float>(
    constant ggml_metal_kargs_conv_3d & args,
    device const char * src0,
    device const char * src1,
    device       char  * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]]);

// Explicit instantiation for f16 weights
template [[host_name("kernel_conv_3d_f16_f32")]]
kernel void kernel_conv_3d<half>(
    constant ggml_metal_kargs_conv_3d & args,
    device const char  * src0,
    device const char * src1,
    device       char  * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]]);


static inline float bicubic_weight1(float x) {
    const float a = -0.75f;
    return ((a + 2) * x - (a + 3)) * x * x + 1;
}

static inline float bicubic_weight2(float x) {
    const float a = -0.75f;
    return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
}

kernel void kernel_upscale_bicubic_f32(
    constant ggml_metal_kargs_upscale & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3 / args.sf3;
    const int64_t i02 = i2 / args.sf2;

    const float   f01 = ((float)i1 + args.poffs) / args.sf1 - args.poffs;
    const int64_t i01 = (int64_t)floor(f01);
    const float   fd1 = f01 - (float)i01;

    const float w_y0 = bicubic_weight2(fd1 + 1.0f);
    const float w_y1 = bicubic_weight1(fd1);
    const float w_y2 = bicubic_weight1(1.0f - fd1);
    const float w_y3 = bicubic_weight2(2.0f - fd1);

    const device const char * src_slice = src0 + i03 * args.nb03 + i02 * args.nb02;

    device float * dst_ptr = (device float *)(dst + i3 * args.nb3 + i2 * args.nb2 + i1 * args.nb1);

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const float   f00 = ((float)i0 + args.poffs) / args.sf0 - args.poffs;
        const int64_t i00 = (int64_t)floor(f00);
        const float   fd0 = f00 - (float)i00;

        const float w_x0 = bicubic_weight2(fd0 + 1.0f);
        const float w_x1 = bicubic_weight1(fd0);
        const float w_x2 = bicubic_weight1(1.0f - fd0);
        const float w_x3 = bicubic_weight2(2.0f - fd0);

        float sum = 0.0f;

        for (int dy = -1; dy <= 2; ++dy) {
            const int64_t iy = MAX(0, MIN(args.ne01 - 1, i01 + dy));
            const float wy = (dy == -1) ? w_y0 : (dy == 0) ? w_y1 : (dy == 1) ? w_y2 : w_y3;

            for (int dx = -1; dx <= 2; ++dx) {
                const int64_t ix = MAX(0, MIN(args.ne00 - 1, i00 + dx));
                const float wx = (dx == -1) ? w_x0 : (dx == 0) ? w_x1 : (dx == 1) ? w_x2 : w_x3;

                device const float * src_ptr = (device const float *)(src_slice + iy * args.nb01 + ix * args.nb00);
                sum += (*src_ptr) * wx * wy;
            }
        }

        dst_ptr[i0] = sum;
    }
}

kernel void kernel_roll_f32(
    constant ggml_metal_kargs_roll & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    device const float * src0_ptr = (device const float *) src0;
    device       float * dst_ptr  = (device       float *) dst;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        // apply shifts and wrap around
        int64_t i00 = i0 - args.s0;
        int64_t i01 = i1 - args.s1;
        int64_t i02 = i2 - args.s2;
        int64_t i03 = i3 - args.s3;

        if (i00 < 0) { i00 += args.ne00; } else if (i00 >= args.ne00) { i00 -= args.ne00; }
        if (i01 < 0) { i01 += args.ne01; } else if (i01 >= args.ne01) { i01 -= args.ne01; }
        if (i02 < 0) { i02 += args.ne02; } else if (i02 >= args.ne02) { i02 -= args.ne02; }
        if (i03 < 0) { i03 += args.ne03; } else if (i03 >= args.ne03) { i03 -= args.ne03; }

        int64_t src_idx = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00 + i00;
        int64_t dst_idx = i3 *args.ne2 *args.ne1 *args.ne0  + i2 *args.ne1 *args.ne0  + i1 *args.ne0  + i0;

        dst_ptr[dst_idx] = src0_ptr[src_idx];
    }
}

template <typename T>
kernel void kernel_pad_impl(
    constant ggml_metal_kargs_pad & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {
    const int32_t i3 = tgpig.z;
    const int32_t i2 = tgpig.y;
    const int32_t k0 = tgpig.x/args.ne1;
    const int32_t i1 = tgpig.x - k0*args.ne1;

    const int32_t i03 = i3;
    const int32_t i02 = i2;
    const int32_t i01 = i1;

    device const T * src0_ptr = (device const T *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       T * dst_ptr  = (device       T *) (dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1);

    for (int32_t l0 = 0; l0 < 1024; l0 += ntg.x) {
        const int32_t i0 = k0*1024 + tpitg.x + l0;
        if (i0 >= args.ne0) {
            break;
        }

        if (i0 < args.ne00 && i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
            dst_ptr[i0] = src0_ptr[i0];
        } else {
            dst_ptr[i0] = 0.0f;
        }
    }
}

typedef decltype(kernel_pad_impl<float>) kernel_pad_t;

template [[host_name("kernel_pad_f32")]]   kernel kernel_pad_t kernel_pad_impl<float>;
template [[host_name("kernel_pad_f32_4")]] kernel kernel_pad_t kernel_pad_impl<float4>;

// TODO: this is slow - optimize
kernel void kernel_pad_reflect_1d_f32(
    constant   ggml_metal_kargs_pad_reflect_1d & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3  tgpg[[threadgroups_per_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3;
    const int64_t i02 = i2;
    const int64_t i01 = i1;

    device const float * src0_ptr = (device const float *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       float * dst_ptr  = (device       float *) (dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1);

    if (i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
        for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
            if (i0 < args.p0) {
                dst_ptr[i0] = src0_ptr[args.p0 - i0];
            } else if (i0 < args.ne0 - args.p1) {
                dst_ptr[i0] = src0_ptr[i0 - args.p0];
            } else {
                dst_ptr[i0] = src0_ptr[(args.ne0 - args.p1 - args.p0) - (args.p1 + 1 - (args.ne0 - i0)) - 1];
            }
        }
    }
}

kernel void kernel_arange_f32(
    constant   ggml_metal_kargs_arange & args,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    device float * dst_ptr = (device float *) dst;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        dst_ptr[i0] = args.start + args.step * i0;
    }
}

kernel void kernel_timestep_embedding_f32(
    constant  ggml_metal_kargs_timestep_embedding & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    int i = tgpig.x;
    device float * embed_data = (device float *)(dst + i*args.nb1);

    int half_ = args.dim / 2;
    for (int j = tpitg.x; j < half_; j += ntg.x) {
        float timestep = ((device float *)src0)[i];
        float freq = (float)exp(-log((float)args.max_period) * j / half_);
        float arg = timestep * freq;
        embed_data[j        ] = cos(arg);
        embed_data[j + half_] = sin(arg);
    }

    if (args.dim % 2 != 0 && tpitg.x == 0) {
        embed_data[2 * half_] = 0.f;
    }
}

// bitonic sort implementation following the CUDA kernels as reference
typedef void (argsort_t)(
        constant   ggml_metal_kargs_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

template<ggml_sort_order order>
kernel void kernel_argsort_f32_i32(
        constant   ggml_metal_kargs_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    // bitonic sort
    const int col = tpitg[0];
    const int ib  = tgpig[0] / args.ne01;

    const int i00 = ib*ntg.x;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    device const float * src0_row = (device const float *) (src0 + args.nb01*i01 + args.nb02*i02 + args.nb03*i03);

    // initialize indices
    shmem_i32[col] = i00 + col;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 2; k <= ntg.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (shmem_i32[col] >= args.ne00 ||
                       (shmem_i32[ixj] <  args.ne00 && (order == GGML_SORT_ORDER_ASC ?
                            src0_row[shmem_i32[col]] > src0_row[shmem_i32[ixj]] :
                            src0_row[shmem_i32[col]] < src0_row[shmem_i32[ixj]]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                } else {
                    if (shmem_i32[ixj] >= args.ne00 ||
                       (shmem_i32[col] <  args.ne00 && (order == GGML_SORT_ORDER_ASC ?
                            src0_row[shmem_i32[col]] < src0_row[shmem_i32[ixj]] :
                            src0_row[shmem_i32[col]] > src0_row[shmem_i32[ixj]]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int64_t i0 = ib*args.top_k;

    // copy the result to dst without the padding
    if (i0 + col < args.ne0 && col < args.top_k) {
        dst += i0 + args.ne0*i01 + args.ne0*args.ne1*i02 + args.ne0*args.ne1*args.ne2*i03;

        dst[col] = shmem_i32[col];
    }
}

template [[host_name("kernel_argsort_f32_i32_asc")]]  kernel argsort_t kernel_argsort_f32_i32<GGML_SORT_ORDER_ASC>;
template [[host_name("kernel_argsort_f32_i32_desc")]] kernel argsort_t kernel_argsort_f32_i32<GGML_SORT_ORDER_DESC>;

typedef void (argsort_merge_t)(
        constant   ggml_metal_kargs_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

template<ggml_sort_order order>
kernel void kernel_argsort_merge_f32_i32(
        constant   ggml_metal_kargs_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {

    const int im  = tgpig[0] / args.ne01;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    const int start = im * (2 * args.len);

    const int len0 = MIN(args.len, MAX(0, args.ne0 - (int)(start)));
    const int len1 = MIN(args.len, MAX(0, args.ne0 - (int)(start + args.len)));

    const int total = len0 + len1;

    device const int32_t * tmp0 = tmp + start
        + i01*args.ne0
        + i02*args.ne0*args.ne01
        + i03*args.ne0*args.ne01*args.ne02;

    device const int32_t * tmp1 = tmp0 + args.len;

    dst += start
        + i01*args.top_k
        + i02*args.top_k*args.ne01
        + i03*args.top_k*args.ne01*args.ne02;

    device const float * src0_row = (device const float *)(src0
        + args.nb01*i01
        + args.nb02*i02
        + args.nb03*i03);

    if (total == 0) {
        return;
    }

    const int chunk = (total + ntg.x - 1) / ntg.x;

    const int k0 = tpitg.x * chunk;
    const int k1 = MIN(MIN(k0 + chunk, total), args.top_k);

    if (k0 >= args.top_k) {
        return;
    }

    if (k0 >= total) {
        return;
    }

    int low  = k0 > len1 ? k0 - len1 : 0;
    int high = MIN(k0, len0);

    // binary-search partition (i, j) such that i + j = k
    while (low < high) {
        const int mid = (low + high) >> 1;

        const int32_t idx0 = tmp0[mid];
        const int32_t idx1 = tmp1[k0 - mid - 1];

        const float val0 = src0_row[idx0];
        const float val1 = src0_row[idx1];

        bool take_left;
        if (order == GGML_SORT_ORDER_ASC) {
            take_left = (val0 <= val1);
        } else {
            take_left = (val0 >= val1);
        }

        if (take_left) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    int i = low;
    int j = k0 - i;

    // keep the merge fronts into registers
    int32_t idx0 = 0;
    float   val0 = 0.0f;
    if (i < len0) {
        idx0 = tmp0[i];
        val0 = src0_row[idx0];
    }

    int32_t idx1 = 0;
    float   val1 = 0.0f;
    if (j < len1) {
        idx1 = tmp1[j];
        val1 = src0_row[idx1];
    }

    for (int k = k0; k < k1; ++k) {
        int32_t out_idx;

        if (i >= len0) {
            while (k < k1) {
                dst[k++] = tmp1[j++];
            }
            break;
        } else if (j >= len1) {
            while (k < k1) {
                dst[k++] = tmp0[i++];
            }
            break;
        } else {
            bool take_left;

            if (order == GGML_SORT_ORDER_ASC) {
                take_left = (val0 <= val1);
            } else {
                take_left = (val0 >= val1);
            }

            if (take_left) {
                out_idx = idx0;
                ++i;
                if (i < len0) {
                    idx0 = tmp0[i];
                    val0 = src0_row[idx0];
                }
            } else {
                out_idx = idx1;
                ++j;
                if (j < len1) {
                    idx1 = tmp1[j];
                    val1 = src0_row[idx1];
                }
            }
        }

        dst[k] = out_idx;
    }
}

template [[host_name("kernel_argsort_merge_f32_i32_asc")]]  kernel argsort_merge_t kernel_argsort_merge_f32_i32<GGML_SORT_ORDER_ASC>;
template [[host_name("kernel_argsort_merge_f32_i32_desc")]] kernel argsort_merge_t kernel_argsort_merge_f32_i32<GGML_SORT_ORDER_DESC>;


kernel void kernel_pool_2d_max_f32(
        constant    ggml_metal_kargs_pool_2d & args,
        device  const float * src0,
        device        float * dst,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    const int idx = gid;
    const int I_HW = args.IH * args.IW;
    const int O_HW = args.OH * args.OW;
    const int nc = idx / O_HW;
    const int cur_oh = idx % O_HW / args.OW;
    const int cur_ow = idx % O_HW % args.OW;

    device const float * i_ptr = src0 + nc * I_HW;
    device       float * o_ptr = dst  + nc * O_HW;

    const int start_h = cur_oh * args.s1 - args.p1;
    const int bh = MAX(0,  start_h);
    const int eh = MIN(args.IH, start_h + args.k1);
    const int start_w = cur_ow * args.s0 - args.p0;
    const int bw = MAX(0,  start_w);
    const int ew = MIN(args.IW, start_w + args.k0);

    float res = -INFINITY;

    for (int i = bh; i < eh; i += 1) {
        for (int j = bw; j < ew; j += 1) {
            res = MAX(res, i_ptr[i * args.IW + j]);
        }
    }

    o_ptr[cur_oh * args.OW + cur_ow] = res;
}

kernel void kernel_pool_2d_avg_f32(
        constant    ggml_metal_kargs_pool_2d & args,
        device  const float * src0,
        device        float * dst,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    const int idx = gid;
    const int I_HW = args.IH * args.IW;
    const int O_HW = args.OH * args.OW;
    const int nc = idx / O_HW;
    const int cur_oh = idx % O_HW / args.OW;
    const int cur_ow = idx % O_HW % args.OW;

    device const float * i_ptr = src0 + nc * I_HW;
    device       float * o_ptr = dst  + nc * O_HW;

    const int start_h = cur_oh * args.s1 - args.p1;
    const int bh = MAX(0,  start_h);
    const int eh = MIN(args.IH, start_h + args.k1);
    const int start_w = cur_ow * args.s0 - args.p0;
    const int bw = MAX(0,  start_w);
    const int ew = MIN(args.IW, start_w + args.k0);
    // const float scale = 1. / ((eh - bh) * (ew - bw));
    const float scale = 1. / (args.k0 * args.k1);

    float res = 0;

    for (int i = bh; i < eh; i += 1) {
        for (int j = bw; j < ew; j += 1) {
            float cur = i_ptr[i * args.IW + j];
            res += cur * scale;
        }
    }

    o_ptr[cur_oh * args.OW + cur_ow] = res;
}


kernel void kernel_pool_1d_max_f32(
        constant        ggml_metal_kargs_pool_1d & args,
        device  const   float * src,
        device          float * dst,
        uint            gid [[thread_position_in_grid]]
) {

    if (gid >= args.np) {
        return;
    }

    const int ow  = (int)gid % args.OW;
    const int row = (int)gid / args.OW;

    const int base = ow * args.s0 - args.p0;

    float acc = -INFINITY;

    const int src_off = row * args.IW;
    const int dst_off = row * args.OW;

    for (int ki = 0; ki < args.k0; ++ki) {
        int j = base + ki;
        if (j < 0 || j >= args.IW){
            continue;
        }
        float v = src[src_off + j];
        acc = max(acc, v);
    }

    dst[dst_off + ow] = acc;
}

kernel void kernel_pool_1d_avg_f32(
        constant        ggml_metal_kargs_pool_1d & args,
        device  const   float * src,
        device          float * dst,
        uint            gid [[thread_position_in_grid]]
) {

    if (gid >= args.np) {
        return;
    }

    const int ow  = (int)gid % args.OW;
    const int row = (int)gid / args.OW;

    const int base = ow * args.s0 - args.p0;

    float acc = 0.0f;
    int   cnt = 0;

    const int src_off = row * args.IW;
    const int dst_off = row * args.OW;

    for (int ki = 0; ki < args.k0; ++ki) {
        const int j = base + ki;
        if (j < 0 || j >= args.IW) {
            continue;
        }
        acc += src[src_off + j];
        cnt += 1;
    }

    dst[dst_off + ow] = (cnt > 0) ? (acc / (float)cnt) : 0.0f;
}

kernel void kernel_opt_step_adamw_f32(
        constant    ggml_metal_kargs_opt_step_adamw & args,
        device       float * x,
        device const float * g,
        device       float * g_m,
        device       float * g_v,
        device const float * pars,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    const float alpha  = pars[0];
    const float beta1  = pars[1];
    const float beta2  = pars[2];
    const float eps    = pars[3];
    const float wd     = pars[4];
    const float beta1h = pars[5];
    const float beta2h = pars[6];

    const float gi = g[gid];
    const float gmi = g_m[gid] * beta1 +      gi * (1.0f - beta1);
    const float gvi = g_v[gid] * beta2 + gi * gi * (1.0f - beta2);

    g_m[gid] = gmi;
    g_v[gid] = gvi;

    const float mh =      gmi * beta1h;
    const float vh = sqrt(gvi * beta2h) + eps;

    x[gid] = x[gid] * (1.0f - alpha * wd) - alpha * mh / vh;
}

kernel void kernel_opt_step_sgd_f32(
        constant    ggml_metal_kargs_opt_step_sgd & args,
        device       float * x,
        device const float * g,
        device const float * pars,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    x[gid] = x[gid] * (1.0f - pars[0] * pars[1]) - pars[0] * g[gid];
}

template<typename T>
kernel void kernel_memset(
        constant ggml_metal_kargs_memset & args,
        device T * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = args.val;
}

typedef decltype(kernel_memset<int64_t>) kernel_memset_t;

template [[host_name("kernel_memset_i64")]] kernel kernel_memset_t kernel_memset<int64_t>;

constant short FC_count_equal_nsg [[function_constant(FC_COUNT_EQUAL + 0)]];

template<typename T>
kernel void kernel_count_equal(
        constant ggml_metal_kargs_count_equal & args,
        device   const char * src0,
        device   const char * src1,
        device   atomic_int * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const short NSG = FC_count_equal_nsg;

    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    if (i3 >= args.ne03 || i2 >= args.ne02 || i1 >= args.ne01) {
        return;
    }

    int sum = 0;

    device const char * base0 = src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03;
    device const char * base1 = src1 + i1*args.nb11 + i2*args.nb12 + i3*args.nb13;

    for (int64_t i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {
        const T v0 = *(device const T *)(base0 + i0*args.nb00);
        const T v1 = *(device const T *)(base1 + i0*args.nb10);
        sum += (v0 == v1);
    }

    sum = simd_sum(sum);

    if (tiisg == 0) {
        shmem_i32[sgitg] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        float v = 0.0f;
        if (tpitg.x < NSG) {
            v = shmem_i32[tpitg.x];
        }

        float total = simd_sum(v);
        if (tpitg.x == 0) {
            atomic_fetch_add_explicit(dst, (int32_t) total, memory_order_relaxed);
        }
    }
}

typedef decltype(kernel_count_equal<int32_t>) kernel_count_equal_t;

template [[host_name("kernel_count_equal_i32")]] kernel kernel_count_equal_t kernel_count_equal<int32_t>;
