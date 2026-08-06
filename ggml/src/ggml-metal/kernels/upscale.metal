#include "common.h"

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
