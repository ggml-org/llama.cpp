//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_DNNL_OPS_HPP
#define GGML_SYCL_DNNL_OPS_HPP

#include "common.hpp"

#if GGML_SYCL_DNNL

#include "dnnl.hpp"
#include "dnnl_sycl.hpp"

// Forward declaration
struct ggml_backend_sycl_context;

//
// DnnlSoftmaxWrapper - Softmax primitive using oneDNN
//
class DnnlSoftmaxWrapper {
public:
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;

    template<typename T>
    static constexpr dt to_dt() {
        if constexpr (std::is_same_v<T, float>) return dt::f32;
        else if constexpr (std::is_same_v<T, sycl::half>) return dt::f16;
        else static_assert(sizeof(T) == 0, "Unsupported type");
    }

    // Softmax along the last dimension (axis = -1)
    // src/dst shape: [batch, features] or [n3, n2, n1, n0] with softmax on n0
    static void softmax(
        ggml_backend_sycl_context & ctx,
        const void * src,
        void * dst,
        int64_t batch,      // Product of all dimensions except the softmax axis
        int64_t features,   // Size of softmax axis (innermost dimension)
        float scale,        // Pre-softmax scale factor
        dt data_type,
        const queue_ptr & q)
    {
        auto stream = ctx.stream_dnnl(q);
        auto eng = ctx.engine_dnnl(q);

        // 2D layout: [batch, features] with softmax on axis 1
        dnnl::memory::dims dims = {batch, features};
        auto src_md = dnnl::memory::desc(dims, data_type, tag::ab);
        auto dst_md = dnnl::memory::desc(dims, data_type, tag::ab);

        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

        // Apply pre-softmax scale if not 1.0
        if (scale != 1.0f) {
            // Scale input before softmax using eltwise linear: y = scale * x
            dnnl::post_ops po;
            // Note: oneDNN softmax doesn't support pre-ops, so we may need
            // to apply scale separately if needed
        }

        auto softmax_pd = dnnl::softmax_forward::primitive_desc(
            eng, dnnl::prop_kind::forward_inference,
            dnnl::algorithm::softmax_accurate,
            src_md, dst_md, 1, attr);  // axis = 1 (features dimension)

        auto src_mem = dnnl::memory(src_md, eng, const_cast<void*>(src));
        auto dst_mem = dnnl::memory(dst_md, eng, dst);

        auto scratchpad_md = softmax_pd.scratchpad_desc();
        auto scratchpad_mem = ctx.get_scratchpad_mem(scratchpad_md, eng, q);

        auto softmax_prim = dnnl::softmax_forward(softmax_pd);

        std::unordered_map<int, dnnl::memory> args;
        args.insert({DNNL_ARG_SRC, src_mem});
        args.insert({DNNL_ARG_DST, dst_mem});
        args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem});

        softmax_prim.execute(stream, args);
    }
};

//
// DnnlEltwiseWrapper - Element-wise operations using oneDNN
//
class DnnlEltwiseWrapper {
public:
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    using alg = dnnl::algorithm;

    template<typename T>
    static constexpr dt to_dt() {
        if constexpr (std::is_same_v<T, float>) return dt::f32;
        else if constexpr (std::is_same_v<T, sycl::half>) return dt::f16;
        else static_assert(sizeof(T) == 0, "Unsupported type");
    }

    // Supported element-wise operations
    enum class op {
        SILU,       // x * sigmoid(x) = swish
        GELU,       // GELU with tanh approximation
        GELU_ERF,   // GELU with erf
        RELU,       // max(0, x)
        TANH,       // tanh(x)
        EXP,        // exp(x)
        SQRT,       // sqrt(x)
        ABS,        // |x|
        SIGMOID,    // 1 / (1 + exp(-x))
        LOG         // ln(x)
    };

    // Map our op enum to oneDNN algorithm
    static alg to_dnnl_algorithm(op operation) {
        switch (operation) {
            case op::SILU:     return alg::eltwise_swish;
            case op::GELU:     return alg::eltwise_gelu_tanh;
            case op::GELU_ERF: return alg::eltwise_gelu_erf;
            case op::RELU:     return alg::eltwise_relu;
            case op::TANH:     return alg::eltwise_tanh;
            case op::EXP:      return alg::eltwise_exp;
            case op::SQRT:     return alg::eltwise_sqrt;
            case op::ABS:      return alg::eltwise_abs;
            case op::SIGMOID:  return alg::eltwise_logistic;
            case op::LOG:      return alg::eltwise_log;
            default:           return alg::eltwise_relu;  // fallback
        }
    }

    // Get alpha parameter for operations that need it
    static float get_alpha(op operation) {
        switch (operation) {
            case op::SILU:  return 1.0f;  // swish with beta=1
            case op::RELU:  return 0.0f;  // standard relu (no leak)
            default:        return 0.0f;
        }
    }

    // Element-wise unary operation
    static void eltwise(
        ggml_backend_sycl_context & ctx,
        op operation,
        const void * src,
        void * dst,
        int64_t nelements,
        dt data_type,
        const queue_ptr & q)
    {
        auto stream = ctx.stream_dnnl(q);
        auto eng = ctx.engine_dnnl(q);

        // 1D layout for element-wise
        dnnl::memory::dims dims = {nelements};
        auto md = dnnl::memory::desc(dims, data_type, tag::a);

        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

        alg algorithm = to_dnnl_algorithm(operation);
        float alpha = get_alpha(operation);
        float beta = 0.0f;

        auto eltwise_pd = dnnl::eltwise_forward::primitive_desc(
            eng, dnnl::prop_kind::forward_inference,
            algorithm, md, md, alpha, beta, attr);

        auto src_mem = dnnl::memory(md, eng, const_cast<void*>(src));
        auto dst_mem = dnnl::memory(md, eng, dst);

        auto scratchpad_md = eltwise_pd.scratchpad_desc();
        auto scratchpad_mem = ctx.get_scratchpad_mem(scratchpad_md, eng, q);

        auto eltwise_prim = dnnl::eltwise_forward(eltwise_pd);

        std::unordered_map<int, dnnl::memory> args;
        args.insert({DNNL_ARG_SRC, src_mem});
        args.insert({DNNL_ARG_DST, dst_mem});
        args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem});

        eltwise_prim.execute(stream, args);
    }

    // In-place element-wise (src == dst)
    static void eltwise_inplace(
        ggml_backend_sycl_context & ctx,
        op operation,
        void * data,
        int64_t nelements,
        dt data_type,
        const queue_ptr & q)
    {
        eltwise(ctx, operation, data, data, nelements, data_type, q);
    }
};

//
// DnnlBinaryWrapper - Binary operations using oneDNN
//
class DnnlBinaryWrapper {
public:
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    using alg = dnnl::algorithm;

    template<typename T>
    static constexpr dt to_dt() {
        if constexpr (std::is_same_v<T, float>) return dt::f32;
        else if constexpr (std::is_same_v<T, sycl::half>) return dt::f16;
        else static_assert(sizeof(T) == 0, "Unsupported type");
    }

    // Supported binary operations
    enum class op {
        ADD,    // a + b
        SUB,    // a - b
        MUL,    // a * b
        DIV,    // a / b
        MIN,    // min(a, b)
        MAX     // max(a, b)
    };

    // Map our op enum to oneDNN algorithm
    static alg to_dnnl_algorithm(op operation) {
        switch (operation) {
            case op::ADD: return alg::binary_add;
            case op::SUB: return alg::binary_sub;
            case op::MUL: return alg::binary_mul;
            case op::DIV: return alg::binary_div;
            case op::MIN: return alg::binary_min;
            case op::MAX: return alg::binary_max;
            default:      return alg::binary_add;  // fallback
        }
    }

    // Binary operation for same-shape tensors (no broadcasting)
    static void binary(
        ggml_backend_sycl_context & ctx,
        op operation,
        const void * src0,
        const void * src1,
        void * dst,
        int64_t nelements,
        dt data_type,
        const queue_ptr & q)
    {
        auto stream = ctx.stream_dnnl(q);
        auto eng = ctx.engine_dnnl(q);

        // 1D layout for element-wise binary
        dnnl::memory::dims dims = {nelements};
        auto md = dnnl::memory::desc(dims, data_type, tag::a);

        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

        alg algorithm = to_dnnl_algorithm(operation);

        auto binary_pd = dnnl::binary::primitive_desc(
            eng, algorithm, md, md, md, attr);

        auto src0_mem = dnnl::memory(md, eng, const_cast<void*>(src0));
        auto src1_mem = dnnl::memory(md, eng, const_cast<void*>(src1));
        auto dst_mem = dnnl::memory(md, eng, dst);

        auto scratchpad_md = binary_pd.scratchpad_desc();
        auto scratchpad_mem = ctx.get_scratchpad_mem(scratchpad_md, eng, q);

        auto binary_prim = dnnl::binary(binary_pd);

        std::unordered_map<int, dnnl::memory> args;
        args.insert({DNNL_ARG_SRC_0, src0_mem});
        args.insert({DNNL_ARG_SRC_1, src1_mem});
        args.insert({DNNL_ARG_DST, dst_mem});
        args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem});

        binary_prim.execute(stream, args);
    }
};

//
// DnnlReductionWrapper - Reduction operations using oneDNN
//
class DnnlReductionWrapper {
public:
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    using alg = dnnl::algorithm;

    template<typename T>
    static constexpr dt to_dt() {
        if constexpr (std::is_same_v<T, float>) return dt::f32;
        else if constexpr (std::is_same_v<T, sycl::half>) return dt::f16;
        else static_assert(sizeof(T) == 0, "Unsupported type");
    }

    // Supported reduction operations
    enum class op {
        SUM,        // sum of elements
        MEAN,       // mean of elements
        MAX,        // max element
        MIN,        // min element
        SUM_SQ,     // sum of squares
        NORM_LP_1,  // L1 norm
        NORM_LP_2   // L2 norm
    };

    // Map our op enum to oneDNN algorithm
    static alg to_dnnl_algorithm(op operation) {
        switch (operation) {
            case op::SUM:       return alg::reduction_sum;
            case op::MEAN:      return alg::reduction_mean;
            case op::MAX:       return alg::reduction_max;
            case op::MIN:       return alg::reduction_min;
            case op::SUM_SQ:    return alg::reduction_mul;  // Need custom
            case op::NORM_LP_1: return alg::reduction_norm_lp_sum;
            case op::NORM_LP_2: return alg::reduction_norm_lp_power_p_sum;
            default:            return alg::reduction_sum;
        }
    }

    // Reduce along the last dimension
    // Input: [batch, features], Output: [batch, 1]
    static void reduce_last_dim(
        ggml_backend_sycl_context & ctx,
        op operation,
        const void * src,
        void * dst,
        int64_t batch,
        int64_t features,
        dt data_type,
        const queue_ptr & q)
    {
        auto stream = ctx.stream_dnnl(q);
        auto eng = ctx.engine_dnnl(q);

        dnnl::memory::dims src_dims = {batch, features};
        dnnl::memory::dims dst_dims = {batch, 1};

        auto src_md = dnnl::memory::desc(src_dims, data_type, tag::ab);
        auto dst_md = dnnl::memory::desc(dst_dims, data_type, tag::ab);

        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

        alg algorithm = to_dnnl_algorithm(operation);
        float p = (operation == op::NORM_LP_2) ? 2.0f : 1.0f;
        float eps = 0.0f;

        auto reduction_pd = dnnl::reduction::primitive_desc(
            eng, algorithm, src_md, dst_md, p, eps, attr);

        auto src_mem = dnnl::memory(src_md, eng, const_cast<void*>(src));
        auto dst_mem = dnnl::memory(dst_md, eng, dst);

        auto scratchpad_md = reduction_pd.scratchpad_desc();
        auto scratchpad_mem = ctx.get_scratchpad_mem(scratchpad_md, eng, q);

        auto reduction_prim = dnnl::reduction(reduction_pd);

        std::unordered_map<int, dnnl::memory> args;
        args.insert({DNNL_ARG_SRC, src_mem});
        args.insert({DNNL_ARG_DST, dst_mem});
        args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem});

        reduction_prim.execute(stream, args);
    }
};

#endif // GGML_SYCL_DNNL

#endif // GGML_SYCL_DNNL_OPS_HPP
