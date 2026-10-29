#include "common.h"

constant short FC_unary_op [[function_constant(FC_UNARY + 0)]];
constant bool  FC_unary_cnt[[function_constant(FC_UNARY + 1)]];

template <typename T, typename U>
inline constexpr T ggml_metal_unary_cast(U x) {
    return (T) x;
}

#if defined(GGML_METAL_HAS_BF16)
// MSL has no C-style cast from float vectors to bfloat vectors
template <>
inline bfloat4 ggml_metal_unary_cast<bfloat4, float4>(float4 x) {
    return bfloat4(bfloat(x[0]), bfloat(x[1]), bfloat(x[2]), bfloat(x[3]));
}

template <>
inline bfloat4 ggml_metal_unary_cast<bfloat4, float>(float x) {
    return bfloat4(bfloat(x), bfloat(x), bfloat(x), bfloat(x));
}
#endif

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
        i0 = tgpig.x*ntg.x + tpitg.x;

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
        } else {
            if (i0 >= args.ne00*args.ne01*args.ne02*args.ne03) {
                return;
            }
        }

        const TC x = (TC) src0_ptr[i0];

        if (FC_OP == OP_UNARY_NUM_SCALE) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(args.scale * x + args.bias);
        }

        if (FC_OP == OP_UNARY_NUM_FILL) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(args.val);
        }

        if (FC_OP == OP_UNARY_NUM_CLAMP) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(clamp(x, args.min, args.max));
        }

        if (FC_OP == OP_UNARY_NUM_SQR) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(x * x);
        }

        if (FC_OP == OP_UNARY_NUM_SQRT) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(sqrt(x));
        }

        if (FC_OP == OP_UNARY_NUM_SIN) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(sin(x));
        }

        if (FC_OP == OP_UNARY_NUM_COS) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(cos(x));
        }

        if (FC_OP == OP_UNARY_NUM_LOG) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(log(x));
        }

        if (FC_OP == OP_UNARY_NUM_LEAKY_RELU) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(TC(x > 0)*x + TC(x <= 0)*(x * args.slope));
        }

        if (FC_OP == OP_UNARY_NUM_TANH) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(precise::tanh(x));
        }

        if (FC_OP == OP_UNARY_NUM_RELU) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(fmax(0, x));
        }

        if (FC_OP == OP_UNARY_NUM_SIGMOID) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(1 / (1 + exp(-x)));
        }

        if (FC_OP == OP_UNARY_NUM_GELU) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(0.5*x*(1 + precise::tanh(SQRT_2_OVER_PI*x*(1 + GELU_COEF_A*x*x))));
        }

        if (FC_OP == OP_UNARY_NUM_GELU_ERF) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(0.5*x*(1 + erf_approx(SQRT_2_INV*x)));
        }

        if (FC_OP == OP_UNARY_NUM_GELU_QUICK) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(x * (1/(1 + exp(GELU_QUICK_COEF*x))));
        }

        if (FC_OP == OP_UNARY_NUM_SILU) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(x / (1 + exp(-x)));
        }

        if (FC_OP == OP_UNARY_NUM_ELU) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(elu_approx(x));
        }

        if (FC_OP == OP_UNARY_NUM_NEG) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(-x);
        }

        if (FC_OP == OP_UNARY_NUM_ABS) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(fabs(x));
        }

        if (FC_OP == OP_UNARY_NUM_SGN) {
            dst_ptr[i0] = T(x > 0) - T(x < 0);
        }

        if (FC_OP == OP_UNARY_NUM_STEP) {
            dst_ptr[i0] = T(x > 0);
        }

        if (FC_OP == OP_UNARY_NUM_HARDSWISH) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(x * fmax(0, fmin(1, x/6 + 0.5)));
        }

        if (FC_OP == OP_UNARY_NUM_HARDSIGMOID) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(fmax(0, fmin(1, x/6 + 0.5)));
        }

        if (FC_OP == OP_UNARY_NUM_EXP) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(exp(x));
        }

        if (FC_OP == OP_UNARY_NUM_SOFTPLUS) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(select(log(1 + exp(x)), x, x > 20));
        }

        if (FC_OP == OP_UNARY_NUM_EXPM1) {
            // TODO: precise implementation
            dst_ptr[i0] = ggml_metal_unary_cast<T>(exp(x) - 1);
        }

        if (FC_OP == OP_UNARY_NUM_FLOOR) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(floor(x));
        }

        if (FC_OP == OP_UNARY_NUM_CEIL) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(ceil(x));
        }

        if (FC_OP == OP_UNARY_NUM_ROUND) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(round(x));
        }

        if (FC_OP == OP_UNARY_NUM_TRUNC) {
            dst_ptr[i0] = ggml_metal_unary_cast<T>(trunc(x));
        }

        if (FC_OP == OP_UNARY_NUM_XIELU) {
            const TC xi      = x;
            const TC gate    = TC(xi > TC(0.0f));
            const TC clamped = fmin(xi, TC(args.val));
            const TC y_pos   = TC(args.scale) * xi * xi + TC(args.bias) * xi;
            const TC y_neg   = (exp(clamped) - TC(1.0f) - xi) * TC(args.slope) + TC(args.bias) * xi;
            dst_ptr[i0] = ggml_metal_unary_cast<T>(gate * y_pos + (TC(1.0f) - gate) * y_neg);
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
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_unary_bf16_bf16")]]   kernel kernel_unary_t kernel_unary_impl<bfloat,  bfloat,  float>;
template [[host_name("kernel_unary_bf16_bf16_4")]] kernel kernel_unary_t kernel_unary_impl<bfloat4, bfloat4, float4>;
#endif

kernel void kernel_silu_back_f32(
        constant ggml_metal_kargs_silu_back & args,
        device const float * dy,
        device const float * x,
        device       float * dx,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.ne) {
        return;
    }

    const float s = 1.0f / (1.0f + exp(-x[gid]));
    dx[gid] = dy[gid] * s * (1.0f + x[gid] * (1.0f - s));
}

template<typename T>
kernel void kernel_reglu(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint2 tpitg[[thread_position_in_threadgroup]],
        uint2   ntg[[threads_per_threadgroup]],
        uint2  ntgs[[threadgroups_per_grid]]) {
    device const T * src0_row = (device const T *) ((device const char *) src0 + tgpig.y*args.nb01) + args.i00;
    device const T * src1_row = (device const T *) ((device const char *) src1 + tgpig.y*args.nb11) + args.i10;
    device       T * dst_row  = (device       T *) ((device       char *) dst  + tgpig.y*args.nb1);

    for (int i0 = tgpig.x*ntg.x + tpitg.x; i0 < args.ne0; i0 += ntg.x*ntgs.x) {
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
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint2 tpitg[[thread_position_in_threadgroup]],
        uint2   ntg[[threads_per_threadgroup]],
        uint2  ntgs[[threadgroups_per_grid]]) {
    device const T * src0_row = (device const T *) ((device const char *) src0 + tgpig.y*args.nb01) + args.i00;
    device const T * src1_row = (device const T *) ((device const char *) src1 + tgpig.y*args.nb11) + args.i10;
    device       T * dst_row  = (device       T *) ((device       char *) dst  + tgpig.y*args.nb1);

    for (int i0 = tgpig.x*ntg.x + tpitg.x; i0 < args.ne0; i0 += ntg.x*ntgs.x) {
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
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint2 tpitg[[thread_position_in_threadgroup]],
        uint2   ntg[[threads_per_threadgroup]],
        uint2  ntgs[[threadgroups_per_grid]]) {
    device const T * src0_row = (device const T *) ((device const char *) src0 + tgpig.y*args.nb01) + args.i00;
    device const T * src1_row = (device const T *) ((device const char *) src1 + tgpig.y*args.nb11) + args.i10;
    device       T * dst_row  = (device       T *) ((device       char *) dst  + tgpig.y*args.nb1);

    for (int i0 = tgpig.x*ntg.x + tpitg.x; i0 < args.ne0; i0 += ntg.x*ntgs.x) {
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
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint2 tpitg[[thread_position_in_threadgroup]],
        uint2   ntg[[threads_per_threadgroup]],
        uint2  ntgs[[threadgroups_per_grid]]) {
    device const T * src0_row = (device const T *) ((device const char *) src0 + tgpig.y*args.nb01) + args.i00;
    device const T * src1_row = (device const T *) ((device const char *) src1 + tgpig.y*args.nb11) + args.i10;
    device       T * dst_row  = (device       T *) ((device       char *) dst  + tgpig.y*args.nb1);

    for (int i0 = tgpig.x*ntg.x + tpitg.x; i0 < args.ne0; i0 += ntg.x*ntgs.x) {
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
kernel void kernel_swiglu_clamp(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint2 tpitg[[thread_position_in_threadgroup]],
        uint2   ntg[[threads_per_threadgroup]],
        uint2  ntgs[[threadgroups_per_grid]]) {
    device const T * src0_row = (device const T *) ((device const char *) src0 + tgpig.y*args.nb01) + args.i00;
    device const T * src1_row = (device const T *) ((device const char *) src1 + tgpig.y*args.nb11) + args.i10;
    device       T * dst_row  = (device       T *) ((device       char *) dst  + tgpig.y*args.nb1);

    for (int i0 = tgpig.x*ntg.x + tpitg.x; i0 < args.ne0; i0 += ntg.x*ntgs.x) {
        const float gate = min((float) src0_row[i0], args.limit);
        const float up = clamp((float) src1_row[i0], -args.limit, args.limit);

        dst_row[i0] = (T)(gate / (1.0f + exp(-gate)) * up);
    }
}

typedef decltype(kernel_swiglu_clamp<float>) kernel_swiglu_clamp_t;

template [[host_name("kernel_swiglu_clamp_f32")]] kernel kernel_swiglu_clamp_t kernel_swiglu_clamp<float>;
template [[host_name("kernel_swiglu_clamp_f16")]] kernel kernel_swiglu_clamp_t kernel_swiglu_clamp<half>;

template<typename T>
kernel void kernel_geglu_erf(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint2 tpitg[[thread_position_in_threadgroup]],
        uint2   ntg[[threads_per_threadgroup]],
        uint2  ntgs[[threadgroups_per_grid]]) {
    device const T * src0_row = (device const T *) ((device const char *) src0 + tgpig.y*args.nb01) + args.i00;
    device const T * src1_row = (device const T *) ((device const char *) src1 + tgpig.y*args.nb11) + args.i10;
    device       T * dst_row  = (device       T *) ((device       char *) dst  + tgpig.y*args.nb1);

    for (int i0 = tgpig.x*ntg.x + tpitg.x; i0 < args.ne0; i0 += ntg.x*ntgs.x) {
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
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint2 tpitg[[thread_position_in_threadgroup]],
        uint2   ntg[[threads_per_threadgroup]],
        uint2  ntgs[[threadgroups_per_grid]]) {
    device const T * src0_row = (device const T *) ((device const char *) src0 + tgpig.y*args.nb01) + args.i00;
    device const T * src1_row = (device const T *) ((device const char *) src1 + tgpig.y*args.nb11) + args.i10;
    device       T * dst_row  = (device       T *) ((device       char *) dst  + tgpig.y*args.nb1);

    for (int i0 = tgpig.x*ntg.x + tpitg.x; i0 < args.ne0; i0 += ntg.x*ntgs.x) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float gelu_quick = x0*(1.0f/(1.0f+exp(GELU_QUICK_COEF*x0)));

        dst_row[i0] = (T)(gelu_quick*x1);
    }
}

typedef decltype(kernel_geglu_quick<float>) kernel_geglu_quick_t;

template [[host_name("kernel_geglu_quick_f32")]] kernel kernel_geglu_quick_t kernel_geglu_quick<float>;
template [[host_name("kernel_geglu_quick_f16")]] kernel kernel_geglu_quick_t kernel_geglu_quick<half>;
