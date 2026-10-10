#include "ggml.h"
#include "mmq.hpp"
#include "quantize.hpp"

#include <algorithm>
#include <future>
#include <vector>

// view of the stacks [s0, s0 + n) of a stacked ztensor, the stacks are the outermost dim of its buffer
// the view descs must outlive the view
static zdnn_ztensor ggml_zdnn_stack_view(zdnn_tensor_desc * pre_tfm_desc,
                                         zdnn_tensor_desc * tfm_desc,
                                   const zdnn_ztensor     & src,
                                         int64_t            n_stacks,
                                         int64_t            s0,
                                         int64_t            n) {

    GGML_ASSERT(src.buffer_size % n_stacks == 0);
    const uint64_t stack_bytes = src.buffer_size / n_stacks;

    *pre_tfm_desc = *src.pre_transformed_desc;
    *tfm_desc     = *src.transformed_desc;

    switch (pre_tfm_desc->layout) {
        case ZDNN_3DS:
            {
                pre_tfm_desc->dim3 = n;
            } break;
        case ZDNN_2DS:
            {
                pre_tfm_desc->dim2 = n;
            } break;
        default:
            GGML_ABORT("%s: unsupported stacked layout", __func__);
    }

    tfm_desc->dim4 = n;
    GGML_ASSERT(zdnn_getsize_ztensor(tfm_desc) == n*stack_bytes);

    zdnn_ztensor view = src;
    view.pre_transformed_desc = pre_tfm_desc;
    view.transformed_desc     = tfm_desc;
    view.buffer               = (char *) src.buffer + s0*stack_bytes;
    view.buffer_size          = n * stack_bytes;
    return view;
}

// zero bias of n_stacks stacks, created once per shape as it never changes
static const zdnn_ztensor & ggml_zdnn_zero_bias(ggml_backend_zdnn_context * ctx, int64_t n_stacks, int64_t ne0) {
    std::unique_ptr<ggml_backend_zdnn_buffer> & bias = ctx->zero_bias[{ n_stacks, ne0 }];

    if (!bias) {
        bias = std::make_unique<ggml_backend_zdnn_buffer>();

        std::vector<float> zeros(n_stacks * ne0, 0.0f);

        zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &bias->pre_tfm_desc, n_stacks, ne0);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&bias->pre_tfm_desc, &bias->tfm_desc));
        ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&bias->pre_tfm_desc, &bias->tfm_desc, &bias->ztensor));
        ZDNN_CHECK(zdnn_transform_ztensor(&bias->ztensor, zeros.data()));
    }

    return bias->ztensor;
}

// the weights are stacked into groups along ne00, and each group of each row of the inputs and weights has its own scale
// zDNN multiplies the groups in stacked calls, then the partial results of each group are scaled and summed here
static void ggml_zdnn_mul_mat_q_grouped(ggml_backend_zdnn_context * ctx,
                                        int32_t                     n_threads,
                                  const ggml_tensor               * inputs,
                                  const ggml_backend_zdnn_buffer  * weights_extra,
                                        ggml_tensor               * output) {
    const int64_t ne10 = inputs->ne[0];
    const int64_t ne11 = inputs->ne[1];
    const int64_t ne0  = output->ne[0];

    const int64_t n_groups = weights_extra->pre_tfm_desc.dim3;
    const int64_t gs       = weights_extra->pre_tfm_desc.dim2;

    // the last group is zero padded when gs does not divide ne10
    GGML_ASSERT(n_groups * gs >= ne10 && (n_groups - 1) * gs < ne10);

    // the rounding errors of the inputs are stacked after the inputs, see ggml_zdnn_quantize_inputs
    int8_t * inputs_q = ggml_zdnn_resize_scratch_max(ctx->scratch_q, 2 * n_groups * gs * ne11);
    std::vector<float> inputs_scales(n_groups * ne11);
    ggml_zdnn_quantize_inputs(n_threads, inputs, gs, weights_extra->channel_scales.data(),
                            inputs_q, inputs_q + n_groups * gs * ne11, inputs_scales.data());

    zdnn_tensor_desc inputs_pre_tfm_desc;
    zdnn_tensor_desc inputs_tfm_desc;
    zdnn_ztensor     inputs_ztensor;

    zdnn_init_pre_transformed_desc(ZDNN_3DS, INT8, &inputs_pre_tfm_desc, 2 * n_groups, ne11, gs);
    ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&inputs_pre_tfm_desc, QUANTIZED_INT8, &inputs_tfm_desc));
    ggml_zdnn_init_scratch_ztensor(ctx, &inputs_pre_tfm_desc, &inputs_tfm_desc, &inputs_ztensor, 0);
    ZDNN_CHECK(zdnn_transform_quantized_ztensor(&inputs_ztensor, false, -127, 127, inputs_q));

    // each group of the partial results is read back as its own 2D ztensor
    zdnn_tensor_desc group_pre_tfm_desc;
    zdnn_tensor_desc group_tfm_desc;

    zdnn_init_pre_transformed_desc(ZDNN_2D, FP32, &group_pre_tfm_desc, ne11, ne0);
    ZDNN_CHECK(zdnn_generate_transformed_desc(&group_pre_tfm_desc, &group_tfm_desc));

    const uint64_t group_bytes = zdnn_getsize_ztensor(&group_tfm_desc);

    // stacked calls with a large output are much slower per group, so run the groups in chunks
    // 8 MiB per chunk was benchmarked to be the best on Llama-3.2-1B Q8_0
    // always re-measure with different models to see if different sizes need a dispatching logic
    constexpr int64_t chunk_bytes = 8*1024*1024;

    const int64_t n_chunk = std::clamp<int64_t>(chunk_bytes / (int64_t)group_bytes, 1, n_groups);

    // all zeros, so it is already the pre-computed bias for every chunk
    const zdnn_ztensor & bias_ztensor = ggml_zdnn_zero_bias(ctx, n_chunk, ne0);

    // two partial buffers, so the next chunk runs on the NNPA while this one is summed
    // each holds the results of a chunk of inputs, followed by the results of their rounding errors
    zdnn_tensor_desc partial_pre_tfm_desc;
    zdnn_tensor_desc partial_tfm_desc;
    zdnn_ztensor     partial_ztensor[2];

    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &partial_pre_tfm_desc, 2 * n_chunk, ne11, ne0);
    ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&partial_pre_tfm_desc, QUANTIZED_DLFLOAT16, &partial_tfm_desc));
    GGML_ASSERT(zdnn_getsize_ztensor(&partial_tfm_desc) == (uint64_t)(2 * n_chunk) * group_bytes);
    for (int32_t i = 0; i < 2; i++) {
        ggml_zdnn_init_scratch_ztensor(ctx, &partial_pre_tfm_desc, &partial_tfm_desc, &partial_ztensor[i], 1 + i);
    }

    const float * weights_scales = weights_extra->scales.data();
    float * partial     = ggml_zdnn_resize_scratch_max(ctx->scratch_f32, 2 * ne11 * ne0);
    float * partial_res = partial + ne11 * ne0;

    const auto sum_chunk = [&](const zdnn_ztensor & chunk, int64_t g0, int64_t n) {
        zdnn_ztensor group_ztensor;
        zdnn_init_ztensor(&group_pre_tfm_desc, &group_tfm_desc, &group_ztensor);

        for (int64_t g = g0; g < g0 + n; g++) {
            group_ztensor.buffer_size    = group_bytes;
            group_ztensor.is_transformed = true;

            group_ztensor.buffer = (char *)chunk.buffer + (g - g0)*group_bytes;
            ZDNN_CHECK(zdnn_transform_origtensor(&group_ztensor, partial));

            group_ztensor.buffer = (char *)chunk.buffer + (n_chunk + g - g0)*group_bytes;
            ZDNN_CHECK(zdnn_transform_origtensor(&group_ztensor, partial_res));

            const float * sw = weights_scales + g*ne0;
            for (int64_t i1 = 0; i1 < ne11; i1++) {
                const float   sa = inputs_scales[g*ne11 + i1];
                const float * p  = partial     + i1*ne0;
                const float * pr = partial_res + i1*ne0;
                float       * y  = (float *)output->data + i1*ne0;

                if (g == 0) {
                    for (int64_t i0 = 0; i0 < ne0; i0++) {
                        y[i0] = sa * sw[i0] * (p[i0] + pr[i0] * (1.0f/254.0f));
                    }
                } else {
                    for (int64_t i0 = 0; i0 < ne0; i0++) {
                        y[i0] += sa * sw[i0] * (p[i0] + pr[i0] * (1.0f/254.0f));
                    }
                }
            }
        }
    };

    zdnn_tensor_desc view_pre_tfm_desc[6];
    zdnn_tensor_desc view_tfm_desc[6];
    std::future<void> sum_task;

    for (int64_t c = 0, g0 = 0; g0 < n_groups; c++, g0 += n_chunk) {
        const int64_t n = std::min(n_chunk, n_groups - g0);

        const zdnn_ztensor inputs_view     = ggml_zdnn_stack_view(&view_pre_tfm_desc[0], &view_tfm_desc[0], inputs_ztensor,         2 * n_groups, g0,            n);
        const zdnn_ztensor inputs_res_view = ggml_zdnn_stack_view(&view_pre_tfm_desc[1], &view_tfm_desc[1], inputs_ztensor,         2 * n_groups, n_groups + g0, n);
        const zdnn_ztensor weights_view    = ggml_zdnn_stack_view(&view_pre_tfm_desc[2], &view_tfm_desc[2], weights_extra->ztensor, n_groups,     g0,            n);
        const zdnn_ztensor bias_view       = ggml_zdnn_stack_view(&view_pre_tfm_desc[3], &view_tfm_desc[3], bias_ztensor,           n_chunk,      0,             n);

        // the partial buffer of chunk c - 2 is summed by now, as sum_task is waited on before starting the next sum
        zdnn_ztensor & chunk = partial_ztensor[c % 2];
        chunk.pre_transformed_desc = &partial_pre_tfm_desc;
        chunk.transformed_desc     = &partial_tfm_desc;
        chunk.is_transformed       = false;

        zdnn_ztensor chunk_view     = ggml_zdnn_stack_view(&view_pre_tfm_desc[4], &view_tfm_desc[4], chunk, 2 * n_chunk, 0,       n);
        zdnn_ztensor chunk_res_view = ggml_zdnn_stack_view(&view_pre_tfm_desc[5], &view_tfm_desc[5], chunk, 2 * n_chunk, n_chunk, n);
        ZDNN_CHECK(zdnn_quantized_matmul_op(&inputs_view, &weights_view, &bias_view,
                                            MATMUL_OP_ADDITION, -127, 127, true, false, true,
                                            nullptr, &chunk_view));
        ZDNN_CHECK(zdnn_quantized_matmul_op(&inputs_res_view, &weights_view, &bias_view,
                                            MATMUL_OP_ADDITION, -127, 127, true, false, true,
                                            nullptr, &chunk_res_view));

        if (sum_task.valid()) {
            sum_task.get();
        }

        if (n_threads > 1) {
            sum_task = std::async(std::launch::async, sum_chunk, chunk, g0, n);
        } else {
            sum_chunk(chunk, g0, n);
        }
    }

    if (sum_task.valid()) {
        sum_task.get();
    }

}

void ggml_zdnn_mul_mat_q(
          ggml_backend_zdnn_context * ctx,
    const               ggml_tensor * src0,
    const               ggml_tensor * src1,
                        ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    const enum ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const ggml_tensor * weights = src0;
    const ggml_tensor * inputs  = src1;
          ggml_tensor * output  = dst;

    GGML_ASSERT(inputs->type == GGML_TYPE_F32);

    ggml_backend_zdnn_buffer * weights_extra = (ggml_backend_zdnn_buffer *)weights->extra;
    ggml_backend_zdnn_buffer * output_extra  = (ggml_backend_zdnn_buffer *)output->extra;

    // starting threads costs more than it saves for small inputs, e.g. during token generation
    const int32_t n_threads = ne10*ne11 >= 65536 ? ctx->n_threads : 1;

    if (weights_extra->pre_tfm_desc.layout == ZDNN_3DS) {
        GGML_ASSERT(weights_extra->pre_tfm_desc.dim1 == weights->ne[1] && "weights_extra->pre_tfm_desc.dim1 must match weights->ne[1]");

        ggml_zdnn_mul_mat_q_grouped(ctx, n_threads, inputs, weights_extra, output);

        // the result is written to output->data directly, so the output ztensor is stale
        output_extra->ztensor.is_transformed = false;
        return;
    }

    ggml_backend_zdnn_buffer * bias_extra = (ggml_backend_zdnn_buffer *)output_extra->extra;

    GGML_ASSERT(weights_extra->pre_tfm_desc.dim2 == weights->ne[0] && "weights_extra->pre_tfm_desc.dim2 must match weights->ne[0]");
    GGML_ASSERT(weights_extra->pre_tfm_desc.dim1 == weights->ne[1] && "weights_extra->pre_tfm_desc.dim1 must match weights->ne[1]");

    // zDNN takes one scale per zTensor, so quantize each row of the inputs with its own scale
    // the rounding errors of the inputs are stacked after the inputs, see ggml_zdnn_quantize_inputs
    int8_t * inputs_q = ggml_zdnn_resize_scratch_max(ctx->scratch_q, 2 * ne10 * ne11);
    std::vector<float> inputs_scales(ne11);
    ggml_zdnn_quantize_inputs(n_threads, inputs, ne10, weights_extra->channel_scales.data(),
                            inputs_q, inputs_q + ne10 * ne11, inputs_scales.data());

    zdnn_tensor_desc inputs_pre_tfm_desc;
    zdnn_tensor_desc inputs_tfm_desc;
    zdnn_ztensor     inputs_ztensor;

    zdnn_init_pre_transformed_desc(ZDNN_3DS, INT8, &inputs_pre_tfm_desc, 2, ne11, ne10);
    ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&inputs_pre_tfm_desc, QUANTIZED_INT8, &inputs_tfm_desc));
    ggml_zdnn_init_scratch_ztensor(ctx, &inputs_pre_tfm_desc, &inputs_tfm_desc, &inputs_ztensor, 0);
    ZDNN_CHECK(zdnn_transform_quantized_ztensor(&inputs_ztensor, false, -127, 127, inputs_q));

    zdnn_tensor_desc partial_pre_tfm_desc;
    zdnn_tensor_desc partial_tfm_desc;
    zdnn_ztensor     partial_ztensor;

    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &partial_pre_tfm_desc, 2, ne11, ne0);
    ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&partial_pre_tfm_desc, QUANTIZED_DLFLOAT16, &partial_tfm_desc));
    ggml_zdnn_init_scratch_ztensor(ctx, &partial_pre_tfm_desc, &partial_tfm_desc, &partial_ztensor, 1);

    // both stacks are multiplied by the same unstacked weights in one broadcast call
    // the bias is all zeros, so it is already the pre-computed bias
    ZDNN_CHECK(zdnn_quantized_matmul_op(&inputs_ztensor, &weights_extra->ztensor, &bias_extra->ztensor,
                                        MATMUL_OP_ADDITION, -127, 127, true, false, true,
                                        nullptr, &partial_ztensor));

    // each stack of the partial results is read back as its own 2D ztensor
    zdnn_tensor_desc stack_pre_tfm_desc;
    zdnn_tensor_desc stack_tfm_desc;

    zdnn_init_pre_transformed_desc(ZDNN_2D, FP32, &stack_pre_tfm_desc, ne11, ne0);
    ZDNN_CHECK(zdnn_generate_transformed_desc(&stack_pre_tfm_desc, &stack_tfm_desc));

    const uint64_t stack_bytes = zdnn_getsize_ztensor(&stack_tfm_desc);
    GGML_ASSERT(partial_ztensor.buffer_size == 2*stack_bytes);

    float * partial[2] = { (float *)output->data, ggml_zdnn_resize_scratch_max(ctx->scratch_f32, ne11 * ne0) };

    ggml_zdnn_parallel_for(n_threads, 2, [&](int64_t s_start, int64_t s_end) {
        zdnn_ztensor stack_ztensor;
        zdnn_init_ztensor(&stack_pre_tfm_desc, &stack_tfm_desc, &stack_ztensor);

        for (int64_t s = s_start; s < s_end; s++) {
            stack_ztensor.buffer         = (char *)partial_ztensor.buffer + s*stack_bytes;
            stack_ztensor.buffer_size    = stack_bytes;
            stack_ztensor.is_transformed = true;
            ZDNN_CHECK(zdnn_transform_origtensor(&stack_ztensor, partial[s]));
        }
    });

    // the result is written to output->data directly, so the output ztensor is stale
    output_extra->ztensor.is_transformed = false;

    const float * weights_scales = weights_extra->scales.data();
    ggml_zdnn_parallel_for(n_threads, ne1, [&](int64_t i1_start, int64_t i1_end) {
        for (int64_t i1 = i1_start; i1 < i1_end; i1++) {
            const float   sa = inputs_scales[i1];
            const float * pr = partial[1] + i1*ne0;
            float       * y  = partial[0] + i1*ne0;
            for (int64_t i0 = 0; i0 < ne0; i0++) {
                y[i0] = sa * weights_scales[i0] * (y[i0] + pr[i0] * (1.0f/254.0f));
            }
        }
    });
}
