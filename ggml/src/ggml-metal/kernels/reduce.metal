#include "common.h"

kernel void kernel_op_sum_f32(
        constant ggml_metal_kargs_sum & args,
        device const float * src0,
        device       float * dst,
        threadgroup  float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {

    if (args.np == 0) {
        return;
    }

    // TODO: become function constant
    const uint nsg = (ntg.x + 31) / 32;

    float sumf = 0;

    for (uint64_t i0 = tpitg.x; i0 < args.np; i0 += ntg.x) {
        sumf += src0[i0];
    }

    sumf = simd_sum(sumf);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0;

    if (sgitg == 0) {
        float v = 0;

        if (tpitg.x < nsg) {
            v = shmem_f32[tpitg.x];
        }

        total = simd_sum(v);

        if (tpitg.x == 0) {
            dst[0] = total;
        }
    }
}

constant short FC_sum_rows_op [[function_constant(FC_SUM_ROWS + 0)]];

template <typename T0, typename T>
kernel void kernel_sum_rows_impl(
        constant ggml_metal_kargs_sum_rows & args,
        device const char * src0,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
#define FC_OP  FC_sum_rows_op

    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    threadgroup T0 * shmem_t = (threadgroup T0 *) shmem;

    if (sgitg == 0) {
        shmem_t[tiisg] = 0.0f;
    }

    device const T0 * src_row = (device const T0 *) (src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03);
    device       T  * dst_row = (device       T  *) (dst  + i1*args.nb1  + i2*args.nb2  + i3*args.nb3);

    T0 sumf = T0(0.0f);

    for (int64_t i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {
        sumf += src_row[i0];
    }

    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_t[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_t[tiisg];
    sumf = simd_sum(sumf);

    if (tpitg.x == 0) {
        if (FC_OP == OP_SUM_ROWS_NUM_MEAN) {
            if (is_same<float4, T0>::value) {
                dst_row[0] = sum(sumf) / (4*args.ne00);
            } else {
                dst_row[0] = sum(sumf) / args.ne00;
            }
        } else {
            dst_row[0] = sum(sumf);
        }
    }

#undef FC_OP
}

typedef decltype(kernel_sum_rows_impl<float, float>) kernel_sum_rows_t;

template [[host_name("kernel_sum_rows_f32_f32")]]   kernel kernel_sum_rows_t kernel_sum_rows_impl<float,  float>;
template [[host_name("kernel_sum_rows_f32_f32_4")]] kernel kernel_sum_rows_t kernel_sum_rows_impl<float4, float>;

template<typename T>
kernel void kernel_cumsum_blk(
        constant ggml_metal_kargs_cumsum_blk & args,
        device const char * src0,
        device       char * tmp,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int ib = tgpig[0]/args.ne01;

    const int i00 = ib*ntg.x;
    const int i01 = tgpig[0]%args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    device const float * src0_row = (device const float *) (src0 +
            args.nb01*i01 +
            args.nb02*i02 +
            args.nb03*i03);

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;

    float v = 0.0f;

    if (i00 + tpitg.x < args.ne00) {
        v = src0_row[i00 + tpitg.x];
    }

    float s = simd_prefix_inclusive_sum(v);

    if (tiisg == N_SIMDWIDTH - 1) {
        shmem_f32[sgitg] = s;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        shmem_f32[tiisg] = simd_prefix_exclusive_sum(shmem_f32[tiisg]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    s += shmem_f32[sgitg];

    device float * dst_row = (device float *) dst +
        args.ne00*i01 +
        args.ne00*args.ne01*i02 +
        args.ne00*args.ne01*args.ne02*i03;

    if (i00 + tpitg.x < args.ne00) {
        dst_row[i00 + tpitg.x] = s;
    }

    if (args.outb && tpitg.x == ntg.x - 1) {
        device float * tmp_row = (device float *) tmp +
            args.net0*i01 +
            args.net0*args.net1*i02 +
            args.net0*args.net1*args.net2*i03;

        tmp_row[ib] = s;
    }
}

typedef decltype(kernel_cumsum_blk<float>) kernel_cumsum_blk_t;

template [[host_name("kernel_cumsum_blk_f32")]] kernel kernel_cumsum_blk_t kernel_cumsum_blk<float>;

template<typename T>
kernel void kernel_cumsum_add(
        constant ggml_metal_kargs_cumsum_add & args,
        device const char * tmp,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int ib = tgpig[0]/args.ne01;

    if (ib == 0) {
        return;
    }

    const int i00 = ib*ntg.x;
    const int i01 = tgpig[0]%args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    device const float * tmp_row = (device const float *) (tmp +
            args.nbt1*i01 +
            args.nbt2*i02 +
            args.nbt3*i03);

    device float * dst_row = (device float *) dst +
        args.ne00*i01 +
        args.ne00*args.ne01*i02 +
        args.ne00*args.ne01*args.ne02*i03;

    if (i00 + tpitg.x < args.ne00) {
        dst_row[i00 + tpitg.x] += tmp_row[ib - 1];
    }
}

typedef decltype(kernel_cumsum_add<float>) kernel_cumsum_add_t;

template [[host_name("kernel_cumsum_add_f32")]] kernel kernel_cumsum_add_t kernel_cumsum_add<float>;
