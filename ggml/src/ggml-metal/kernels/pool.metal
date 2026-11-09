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
        threadgroup    char * shmem [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint ntg[[threads_per_threadgroup]]) {
    if (tgpig >= args.np) {
        return;
    }

    const int out = (int) tgpig;
    const int ow  = out % (int) args.OW;
    const int row = out / (int) args.OW;

    const int base     = ow * args.s0 - args.p0;
    const int src_off  = row * args.IW;
    const int dst_off  = row * args.OW;

    float lmax = -INFINITY;

    for (int ki = tpitg; ki < args.k0; ki += ntg) {
        const int j = base + ki;
        if (j >= 0 && j < args.IW) {
            lmax = max(lmax, src[src_off + j]);
        }
    }

    float max_val = simd_max(lmax);

    threadgroup float * shared_max = (threadgroup float *) shmem;

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_max[tiisg] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_max[sgitg] = max_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = shared_max[tiisg];
        max_val = simd_max(max_val);
    }

    if (tpitg == 0) {
        dst[dst_off + ow] = max_val;
    }
}

kernel void kernel_pool_1d_avg_f32(
        constant        ggml_metal_kargs_pool_1d & args,
        device  const   float * src,
        device          float * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint ntg[[threads_per_threadgroup]]) {
    if (tgpig >= args.np) {
        return;
    }

    const int out = (int) tgpig;
    const int ow  = out % (int) args.OW;
    const int row = out / (int) args.OW;

    const int base     = ow * args.s0 - args.p0;
    const int src_off  = row * args.IW;
    const int dst_off  = row * args.OW;

    float lsum = 0.0f;
    int   lcnt = 0;

    for (int ki = tpitg; ki < args.k0; ki += ntg) {
        const int j = base + ki;
        if (j >= 0 && j < args.IW) {
            lsum += src[src_off + j];
            lcnt += 1;
        }
    }

    float sum = simd_sum(lsum);
    float cnt = simd_sum((float) lcnt);

    threadgroup float * shared_sum = (threadgroup float *) shmem;
    threadgroup float * shared_cnt = (threadgroup float *) shmem + N_SIMDWIDTH;

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_sum[tiisg] = 0.0f;
            shared_cnt[tiisg] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_sum[sgitg] = sum;
            shared_cnt[sgitg] = cnt;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = shared_sum[tiisg];
        cnt = shared_cnt[tiisg];

        sum = simd_sum(sum);
        cnt = simd_sum(cnt);
    }

    if (tpitg == 0) {
        dst[dst_off + ow] = (cnt > 0.0f) ? (sum / cnt) : 0.0f;
    }
}
