#include "common.h"

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
