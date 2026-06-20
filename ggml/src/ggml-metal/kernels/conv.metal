#include "common.h"

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
