#include "common.h"

constant short FC_unary_op [[function_constant(FC_UNARY + 0)]];
constant bool  FC_unary_cnt[[function_constant(FC_UNARY + 1)]];

template <typename T0, typename T, typename TC>
kernel void kernel_unary_impl(
        constant ggml_metal_kargs_unary & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
#define FC_OP  FC_unary_op
#define FC_CNT FC_unary_cnt

    device const T0 * src0_ptr;
    device       T  * dst_ptr;

    int i0;

    if (FC_CNT) {
        i0 = tgpig.x;

        src0_ptr = (device const T0 *) (src0);
        dst_ptr  = (device       T  *) (dst);
    } else {
        const int i03 = tgpig.z;
        const int i02 = tgpig.y;
        const int k0  = tgpig.x/args.ne01;
        const int i01 = tgpig.x - k0*args.ne01;

        i0 = k0*ntg.x + tpitg.x;

        src0_ptr = (device const T0 *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
        dst_ptr  = (device       T  *) (dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1 );
    }

    {
        //threadgroup_barrier(mem_flags::mem_none);

        if (!FC_CNT) {
            if (i0 >= args.ne0) {
                return;
            }
        }

        const TC x = (TC) src0_ptr[i0];

        if (FC_OP == OP_UNARY_NUM_SCALE) {
            dst_ptr[i0] = (T) (args.scale * x + args.bias);
        }

        if (FC_OP == OP_UNARY_NUM_FILL) {
            dst_ptr[i0] = (T) args.val;
        }

        if (FC_OP == OP_UNARY_NUM_CLAMP) {
            dst_ptr[i0] = (T) clamp(x, args.min, args.max);
        }

        if (FC_OP == OP_UNARY_NUM_SQR) {
            dst_ptr[i0] = (T) (x * x);
        }

        if (FC_OP == OP_UNARY_NUM_SQRT) {
            dst_ptr[i0] = (T) sqrt(x);
        }

        if (FC_OP == OP_UNARY_NUM_SIN) {
            dst_ptr[i0] = (T) sin(x);
        }

        if (FC_OP == OP_UNARY_NUM_COS) {
            dst_ptr[i0] = (T) cos(x);
        }

        if (FC_OP == OP_UNARY_NUM_LOG) {
            dst_ptr[i0] = (T) log(x);
        }

        if (FC_OP == OP_UNARY_NUM_LEAKY_RELU) {
            dst_ptr[i0] = (T) (TC(x > 0)*x + TC(x <= 0)*(x * args.slope));
        }

        if (FC_OP == OP_UNARY_NUM_TANH) {
            dst_ptr[i0] = (T) precise::tanh(x);
        }

        if (FC_OP == OP_UNARY_NUM_RELU) {
            dst_ptr[i0] = (T) fmax(0, x);
        }

        if (FC_OP == OP_UNARY_NUM_SIGMOID) {
            dst_ptr[i0] = (T) (1 / (1 + exp(-x)));
        }

        if (FC_OP == OP_UNARY_NUM_GELU) {
            dst_ptr[i0] = (T) (0.5*x*(1 + precise::tanh(SQRT_2_OVER_PI*x*(1 + GELU_COEF_A*x*x))));
        }

        if (FC_OP == OP_UNARY_NUM_GELU_ERF) {
            dst_ptr[i0] = (T) (0.5*x*(1 + erf_approx(SQRT_2_INV*x)));
        }

        if (FC_OP == OP_UNARY_NUM_GELU_QUICK) {
            dst_ptr[i0] = (T) (x * (1/(1 + exp(GELU_QUICK_COEF*x))));
        }

        if (FC_OP == OP_UNARY_NUM_SILU) {
            dst_ptr[i0] = (T) (x / (1 + exp(-x)));
        }

        if (FC_OP == OP_UNARY_NUM_ELU) {
            dst_ptr[i0] = (T) elu_approx(x);
        }

        if (FC_OP == OP_UNARY_NUM_NEG) {
            dst_ptr[i0] = (T) -x;
        }

        if (FC_OP == OP_UNARY_NUM_ABS) {
            dst_ptr[i0] = (T) fabs(x);
        }

        if (FC_OP == OP_UNARY_NUM_SGN) {
            dst_ptr[i0] = T(x > 0) - T(x < 0);
        }

        if (FC_OP == OP_UNARY_NUM_STEP) {
            dst_ptr[i0] = T(x > 0);
        }

        if (FC_OP == OP_UNARY_NUM_HARDSWISH) {
            dst_ptr[i0] = (T) (x * fmax(0, fmin(1, x/6 + 0.5)));
        }

        if (FC_OP == OP_UNARY_NUM_HARDSIGMOID) {
            dst_ptr[i0] = (T) fmax(0, fmin(1, x/6 + 0.5));
        }

        if (FC_OP == OP_UNARY_NUM_EXP) {
            dst_ptr[i0] = (T) exp(x);
        }

        if (FC_OP == OP_UNARY_NUM_SOFTPLUS) {
            dst_ptr[i0] = (T) select(log(1 + exp(x)), x, x > 20);
        }

        if (FC_OP == OP_UNARY_NUM_EXPM1) {
            // TODO: precise implementation
            dst_ptr[i0] = (T) (exp(x) - 1);
        }

        if (FC_OP == OP_UNARY_NUM_FLOOR) {
            dst_ptr[i0] = (T) floor(x);
        }

        if (FC_OP == OP_UNARY_NUM_CEIL) {
            dst_ptr[i0] = (T) ceil(x);
        }

        if (FC_OP == OP_UNARY_NUM_ROUND) {
            dst_ptr[i0] = (T) round(x);
        }

        if (FC_OP == OP_UNARY_NUM_TRUNC) {
            dst_ptr[i0] = (T) trunc(x);
        }

        if (FC_OP == OP_UNARY_NUM_XIELU) {
            const TC xi      = x;
            const TC gate    = TC(xi > TC(0.0f));
            const TC clamped = fmin(xi, TC(args.val));
            const TC y_pos   = TC(args.scale) * xi * xi + TC(args.bias) * xi;
            const TC y_neg   = (exp(clamped) - TC(1.0f) - xi) * TC(args.slope) + TC(args.bias) * xi;
            dst_ptr[i0] = (T) (gate * y_pos + (TC(1.0f) - gate) * y_neg);
        }
    }

#undef FC_OP
#undef FC_CNT
}

typedef decltype(kernel_unary_impl<float, float, float>) kernel_unary_t;

template [[host_name("kernel_unary_f32_f32")]]   kernel kernel_unary_t kernel_unary_impl<float,  float,  float>;
template [[host_name("kernel_unary_f32_f32_4")]] kernel kernel_unary_t kernel_unary_impl<float4, float4, float4>;
template [[host_name("kernel_unary_f16_f16")]]   kernel kernel_unary_t kernel_unary_impl<half,   half,   float>;
template [[host_name("kernel_unary_f16_f16_4")]] kernel kernel_unary_t kernel_unary_impl<half4,  half4,  float4>;

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
template [[host_name("kernel_repeat_i32")]] kernel kernel_repeat_t kernel_repeat<int>;
template [[host_name("kernel_repeat_i16")]] kernel kernel_repeat_t kernel_repeat<short>;

template<typename T>
kernel void kernel_reglu(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const T * src0_row = (device const T *) ((device const char *) src0 + tgpig*args.nb01) + args.i00;
    device const T * src1_row = (device const T *) ((device const char *) src1 + tgpig*args.nb11) + args.i10;
    device       T * dst_row  = (device       T *) ((device       char *) dst  + tgpig*args.nb1);

    for (int i0 = tpitg; i0 < args.ne0; i0 += ntg) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        dst_row[i0] = (T)(x0*x1*(x0 > 0.0f));
    }
}

typedef decltype(kernel_reglu<float>) kernel_reglu_t;

template [[host_name("kernel_reglu_f32")]] kernel kernel_reglu_t kernel_reglu<float>;
template [[host_name("kernel_reglu_f16")]] kernel kernel_reglu_t kernel_reglu<half>;

template<typename T>
kernel void kernel_geglu(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const T * src0_row = (device const T *) ((device const char *) src0 + tgpig*args.nb01) + args.i00;
    device const T * src1_row = (device const T *) ((device const char *) src1 + tgpig*args.nb11) + args.i10;
    device       T * dst_row  = (device       T *) ((device       char *) dst  + tgpig*args.nb1);

    for (int i0 = tpitg; i0 < args.ne0; i0 += ntg) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float gelu = 0.5f*x0*(1.0f + precise::tanh(SQRT_2_OVER_PI*x0*(1.0f + GELU_COEF_A*x0*x0)));

        dst_row[i0] = (T)(gelu*x1);
    }
}

typedef decltype(kernel_geglu<float>) kernel_geglu_t;

template [[host_name("kernel_geglu_f32")]] kernel kernel_geglu_t kernel_geglu<float>;
template [[host_name("kernel_geglu_f16")]] kernel kernel_geglu_t kernel_geglu<half>;

template<typename T>
kernel void kernel_swiglu(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const T * src0_row = (device const T *) ((device const char *) src0 + tgpig*args.nb01) + args.i00;
    device const T * src1_row = (device const T *) ((device const char *) src1 + tgpig*args.nb11) + args.i10;
    device       T * dst_row  = (device       T *) ((device       char *) dst  + tgpig*args.nb1);

    for (int i0 = tpitg; i0 < args.ne0; i0 += ntg) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float silu = x0 / (1.0f + exp(-x0));

        dst_row[i0] = (T)(silu*x1);
    }
}

typedef decltype(kernel_swiglu<float>) kernel_swiglu_t;

template [[host_name("kernel_swiglu_f32")]] kernel kernel_swiglu_t kernel_swiglu<float>;
template [[host_name("kernel_swiglu_f16")]] kernel kernel_swiglu_t kernel_swiglu<half>;

template<typename T>
kernel void kernel_swiglu_oai(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const T * src0_row = (device const T *) ((device const char *) src0 + tgpig*args.nb01) + args.i00;
    device const T * src1_row = (device const T *) ((device const char *) src1 + tgpig*args.nb11) + args.i10;
    device       T * dst_row  = (device       T *) ((device       char *) dst  + tgpig*args.nb1);

    for (int i0 = tpitg; i0 < args.ne0; i0 += ntg) {
        float x0 = src0_row[i0];
        float x1 = src1_row[i0];

        x0 = min(x0, args.limit);
        x1 = max(min(x1, args.limit), -args.limit);

        float out_glu = x0 / (1.0f + exp(-x0 * args.alpha));
        out_glu = out_glu * (1.0f + x1);

        dst_row[i0] = (T)out_glu;
    }
}

typedef decltype(kernel_swiglu_oai<float>) kernel_swiglu_oai_t;

template [[host_name("kernel_swiglu_oai_f32")]] kernel kernel_swiglu_oai_t kernel_swiglu_oai<float>;
template [[host_name("kernel_swiglu_oai_f16")]] kernel kernel_swiglu_oai_t kernel_swiglu_oai<half>;

template<typename T>
kernel void kernel_geglu_erf(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const T * src0_row = (device const T *) ((device const char *) src0 + tgpig*args.nb01) + args.i00;
    device const T * src1_row = (device const T *) ((device const char *) src1 + tgpig*args.nb11) + args.i10;
    device       T * dst_row  = (device       T *) ((device       char *) dst  + tgpig*args.nb1);

    for (int i0 = tpitg; i0 < args.ne0; i0 += ntg) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float gelu_erf = 0.5f*x0*(1.0f+erf_approx<float>(x0*SQRT_2_INV));

        dst_row[i0] = (T)(gelu_erf*x1);
    }
}

typedef decltype(kernel_geglu_erf<float>) kernel_geglu_erf_t;

template [[host_name("kernel_geglu_erf_f32")]] kernel kernel_geglu_erf_t kernel_geglu_erf<float>;
template [[host_name("kernel_geglu_erf_f16")]] kernel kernel_geglu_erf_t kernel_geglu_erf<half>;

template<typename T>
kernel void kernel_geglu_quick(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const T * src0_row = (device const T *) ((device const char *) src0 + tgpig*args.nb01) + args.i00;
    device const T * src1_row = (device const T *) ((device const char *) src1 + tgpig*args.nb11) + args.i10;
    device       T * dst_row  = (device       T *) ((device       char *) dst  + tgpig*args.nb1);

    for (int i0 = tpitg; i0 < args.ne0; i0 += ntg) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float gelu_quick = x0*(1.0f/(1.0f+exp(GELU_QUICK_COEF*x0)));

        dst_row[i0] = (T)(gelu_quick*x1);
    }
}

typedef decltype(kernel_geglu_quick<float>) kernel_geglu_quick_t;

template [[host_name("kernel_geglu_quick_f32")]] kernel kernel_geglu_quick_t kernel_geglu_quick<float>;
template [[host_name("kernel_geglu_quick_f16")]] kernel kernel_geglu_quick_t kernel_geglu_quick<half>;

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


template<uint32_t ttype>
bool _ggml_vec_tri_cmp(const int i, const int r);

template<>
bool _ggml_vec_tri_cmp</* GGML_TRI_TYPE_LOWER */ 3>(const int i, const int r) {
    return i < r;
}

template<>
bool _ggml_vec_tri_cmp</* GGML_TRI_TYPE_LOWER_DIAG */ 2>(const int i, const int r) {
    return i <= r;
}

template<>
bool _ggml_vec_tri_cmp</* GGML_TRI_TYPE_UPPER */ 1>(const int i, const int r) {
    return i > r;
}

template<>
bool _ggml_vec_tri_cmp</* GGML_TRI_TYPE_UPPER_DIAG */ 0>(const int i, const int r) {
    return i >= r;
}

template<typename T, int ttype>
kernel void kernel_tri(
        constant ggml_metal_kargs_tri & args,
        device const char * src0,
        device const char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    if (i3 >= args.ne03 || i2 >= args.ne02 || i1 >= args.ne01) {
        return;
    }

    device const T * src_row = (device const T *) ((device const char *) src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03);
    device       T * dst_row = (device       T *) ((device       char *) dst  + i1*args.nb1  + i2*args.nb2  + i3*args.nb3);

    // Each thread is a single element of the row if ne00 < max threads per
    // threadgroup, so this will loop once for each index that this thread is
    // responsible for
    for (int64_t i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {
        // Use the comparison as a mask for branchless
        dst_row[i0] = static_cast<T>(_ggml_vec_tri_cmp<ttype>(i0, i1)) * src_row[i0];
    }
}

typedef decltype(kernel_tri<float, 0>) kernel_tri_t;

template [[host_name("kernel_tri_f32_0")]] kernel kernel_tri_t kernel_tri<float, 0>;
template [[host_name("kernel_tri_f32_1")]] kernel kernel_tri_t kernel_tri<float, 1>;
template [[host_name("kernel_tri_f32_2")]] kernel kernel_tri_t kernel_tri<float, 2>;
template [[host_name("kernel_tri_f32_3")]] kernel kernel_tri_t kernel_tri<float, 3>;
template [[host_name("kernel_tri_f16_0")]] kernel kernel_tri_t kernel_tri<half, 0>;
template [[host_name("kernel_tri_f16_1")]] kernel kernel_tri_t kernel_tri<half, 1>;
template [[host_name("kernel_tri_f16_2")]] kernel kernel_tri_t kernel_tri<half, 2>;
template [[host_name("kernel_tri_f16_3")]] kernel kernel_tri_t kernel_tri<half, 3>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_tri_bf16_0")]] kernel kernel_tri_t kernel_tri<bfloat, 0>;
template [[host_name("kernel_tri_bf16_1")]] kernel kernel_tri_t kernel_tri<bfloat, 1>;
template [[host_name("kernel_tri_bf16_2")]] kernel kernel_tri_t kernel_tri<bfloat, 2>;
template [[host_name("kernel_tri_bf16_3")]] kernel kernel_tri_t kernel_tri<bfloat, 3>;
#endif

