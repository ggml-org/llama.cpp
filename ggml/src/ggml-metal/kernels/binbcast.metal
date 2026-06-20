#include "common.h"

// OP: 0 - add, 1 - sub, 2 - mul, 3 - div
constant short FC_bin_op [[function_constant(FC_BIN + 0)]];
constant short FC_bin_f  [[function_constant(FC_BIN + 1)]];
constant bool  FC_bin_rb [[function_constant(FC_BIN + 2)]];
constant bool  FC_bin_cb [[function_constant(FC_BIN + 3)]];

template <typename T0, typename T1, typename T>
kernel void kernel_bin_fuse_impl(
        constant ggml_metal_kargs_bin & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
#define FC_OP FC_bin_op
#define FC_F  FC_bin_f
#define FC_RB FC_bin_rb
#define FC_CB FC_bin_cb

    if (FC_RB) {
        // row broadcast
        const uint i0 = tgpig.y*args.ne00 + tgpig.x;
        const uint i1 = FC_CB ? tgpig.x%args.ne10 : tgpig.x;

        device const T0 * src0_row = (device const T0 *) (src0);
        device       T  * dst_row  = (device       T  *) (dst);

        if (FC_F == 1) {
            device const T1 * src1_row = (device const T1 *) (src1 + args.o1[0]);

            if (FC_OP == 0) {
                dst_row[i0] = src0_row[i0] + src1_row[i1];
            }

            if (FC_OP == 1) {
                dst_row[i0] = src0_row[i0] - src1_row[i1];
            }

            if (FC_OP == 2) {
                dst_row[i0] = src0_row[i0] * src1_row[i1];
            }

            if (FC_OP == 3) {
                dst_row[i0] = src0_row[i0] / src1_row[i1];
            }
        } else {
            T0 res = src0_row[i0];

            if (FC_OP == 0) {
                FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                    res += ((device const T1 *) (src1 + args.o1[j]))[i1];
                }
            }

            if (FC_OP == 1) {
                FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                    res -= ((device const T1 *) (src1 + args.o1[j]))[i1];
                }
            }

            if (FC_OP == 2) {
                FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                    res *= ((device const T1 *) (src1 + args.o1[j]))[i1];
                }
            }

            if (FC_OP == 3) {
                FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                    res /= ((device const T1 *) (src1 + args.o1[j]))[i1];
                }
            }

            dst_row[i0] = res;
        }
    } else {
        const int i03 = tgpig.z;
        const int i02 = tgpig.y;
        const int i01 = tgpig.x;

        if (i01 >= args.ne01) {
            return;
        }

        const int i13 = i03%args.ne13;
        const int i12 = i02%args.ne12;
        const int i11 = i01%args.ne11;

        device const T0 * src0_ptr = (device const T0 *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + args.offs);
        device       T  * dst_ptr  = (device       T  *) (dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1  + args.offs);

        if (FC_F == 1) {
            device const T1 * src1_ptr = (device const T1 *) (src1 + args.o1[0] + i13*args.nb13 + i12*args.nb12 + i11*args.nb11);

            for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
                const int i10 = FC_CB ? i0%args.ne10 : i0;

                if (FC_OP == 0) {
                    dst_ptr[i0] = src0_ptr[i0] + src1_ptr[i10];
                }

                if (FC_OP == 1) {
                    dst_ptr[i0] = src0_ptr[i0] - src1_ptr[i10];
                }

                if (FC_OP == 2) {
                    dst_ptr[i0] = src0_ptr[i0] * src1_ptr[i10];
                }

                if (FC_OP == 3) {
                    dst_ptr[i0] = src0_ptr[i0] / src1_ptr[i10];
                }
            }
        } else {
            device const T1 * src1_ptr[8];
            FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                src1_ptr[j] = (device const T1 *) (src1 + args.o1[j] + i13*args.nb13 + i12*args.nb12 + i11*args.nb11);
            }

            for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
                const int i10 = FC_CB ? i0%args.ne10 : i0;

                T res = src0_ptr[i0];

                if (FC_OP == 0) {
                    FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                        res += src1_ptr[j][i10];
                    }
                }

                if (FC_OP == 1) {
                    FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                        res -= src1_ptr[j][i10];
                    }
                }

                if (FC_OP == 2) {
                    FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                        res *= src1_ptr[j][i10];
                    }
                }

                if (FC_OP == 3) {
                    FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                        res /= src1_ptr[j][i10];
                    }
                }

                dst_ptr[i0] = res;
            }
        }
    }

#undef FC_OP
#undef FC_F
#undef FC_RB
#undef FC_CB
}

typedef decltype(kernel_bin_fuse_impl<float, float, float>) kernel_bin_fuse_t;

template [[host_name("kernel_bin_fuse_f32_f32_f32")]]   kernel kernel_bin_fuse_t kernel_bin_fuse_impl<float,  float,  float>;
template [[host_name("kernel_bin_fuse_f32_f32_f32_4")]] kernel kernel_bin_fuse_t kernel_bin_fuse_impl<float4, float4, float4>;

kernel void kernel_add_id(
        constant ggml_metal_kargs_add_id & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i1 = tgpig.x;
    const int i2 = tgpig.y;

    const int i11 = *((device const int32_t *) (src2 + i1*sizeof(int32_t) + i2*args.nb21));

    const size_t nb1 = args.ne0 * sizeof(float);
    const size_t nb2 = args.ne1 * nb1;

    device       float * dst_row  = (device       float *)((device char *)dst  +  i1*nb1       + i2*nb2);
    device const float * src0_row = (device const float *)((device char *)src0 +  i1*args.nb01 + i2*args.nb02);
    device const float * src1_row = (device const float *)((device char *)src1 + i11*args.nb11);

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        dst_row[i0] = src0_row[i0] + src1_row[i0];
    }
}

template<typename T>
kernel void kernel_repeat(
        constant ggml_metal_kargs_repeat & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    const int i03 = i3%args.ne03;
    const int i02 = i2%args.ne02;
    const int i01 = i1%args.ne01;

    device const char * src0_ptr = src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01;
    device       char * dst_ptr  = dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const int i00 = i0%args.ne00;
        *((device T *)(dst_ptr + i0*args.nb0)) = *((device T *)(src0_ptr + i00*args.nb00));
    }
}

typedef decltype(kernel_repeat<float>) kernel_repeat_t;

template [[host_name("kernel_repeat_f32")]] kernel kernel_repeat_t kernel_repeat<float>;
template [[host_name("kernel_repeat_f16")]] kernel kernel_repeat_t kernel_repeat<half>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_repeat_bf16")]] kernel kernel_repeat_t kernel_repeat<bfloat>;
#endif
template [[host_name("kernel_repeat_i32")]] kernel kernel_repeat_t kernel_repeat<int>;
template [[host_name("kernel_repeat_i16")]] kernel kernel_repeat_t kernel_repeat<short>;
