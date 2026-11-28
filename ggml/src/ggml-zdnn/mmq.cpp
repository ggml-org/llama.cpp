#include "ggml.h"
#include "mmq.hpp"
#include "quantize.hpp"
#include "utils.hpp"

#include <algorithm>
#include <atomic>
#include <vector>

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

    // weights read straight into a host buffer skip set_tensor, so transform them on first use
    ggml_zdnn_load_weights(weights_extra, weights);

    // the weights are stacked into groups along ne00, and each group of each row of the inputs and weights has its own scale
    const bool    stacked  = weights_extra->pre_tfm_desc.layout == ZDNN_3DS;
    const int64_t n_groups = stacked ? weights_extra->pre_tfm_desc.dim3 : 1;
    const int64_t gs       = stacked ? weights_extra->pre_tfm_desc.dim2 : ne10;

    GGML_ASSERT(weights_extra->pre_tfm_desc.dim1 == weights->ne[1] && "weights_extra->pre_tfm_desc.dim1 must match weights->ne[1]");

    // the last group is zero padded when gs does not divide ne10
    GGML_ASSERT(n_groups * gs >= ne10 && (n_groups - 1) * gs < ne10);

    // starting threads costs more than it saves for small inputs, e.g. during token generation
    const int32_t n_threads = ne10*ne11 >= 65536 ? ctx->n_threads : 1;

    // the read-back and the sum work on the outputs, and below 1 MiB of them one thread is faster than starting more
    const int32_t n_threads_out = ne0*ne11 >= 262144 ? n_threads : 1;

    // zDNN takes one scale per zTensor, so quantize each group of each row of the inputs with its own scale
    // the rounding errors of the inputs are stacked after the inputs, see ggml_zdnn_quantize_inputs
    ggml_zdnn_pool_alloc<int8_t> inputs_q_alloc(ctx->pool, 2 * n_groups * gs * ne11);
    int8_t * inputs_q = inputs_q_alloc.get();
    std::vector<float> inputs_scales(n_groups * ne11);
    ggml_zdnn_quantize_inputs(n_threads, inputs, gs, weights_extra->channel_scales.data(),
                            inputs_q, inputs_q + n_groups * gs * ne11, inputs_scales.data());

    // each NNPA call reads its weights once, so split the groups first, then the rows down to 64 per call
    // benchmarked on Granite-4.2-8B shapes: a call costs about ne10*ne0*(64 + ne11), and below this limit threads cost more than they save
    constexpr int64_t min_chunk_rows = 64;

    const int32_t n_threads_calls = ne10*ne0*(min_chunk_rows + ne11) >= 256*1024*1024 ? ctx->n_threads : 1;
    const int64_t n_chunks        = std::max<int64_t>(1, std::min<int64_t>(n_threads_calls / n_groups, ne11 / min_chunk_rows));
    const int64_t n_calls         = n_groups * n_chunks;
    const int32_t n_workers       = (int32_t)std::min<int64_t>(n_threads_calls, n_calls);
    const int64_t max_rows        = (ne11 + n_chunks - 1) / n_chunks;

    // the bias cache and the pool are not thread safe, so use them before the workers start
    // the bias is all zeros, so it is already the pre-computed bias
    const zdnn_ztensor & bias_ztensor = ggml_zdnn_zero_bias(ctx, 1, ne0);

    zdnn_tensor_desc max_pre_tfm_desc;
    zdnn_tensor_desc max_tfm_desc;

    zdnn_init_pre_transformed_desc(ZDNN_3DS, INT8, &max_pre_tfm_desc, 2, max_rows, gs);
    ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&max_pre_tfm_desc, QUANTIZED_INT8, &max_tfm_desc));
    const uint64_t inputs_bytes = zdnn_getsize_ztensor(&max_tfm_desc);

    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &max_pre_tfm_desc, 2, max_rows, ne0);
    ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&max_pre_tfm_desc, QUANTIZED_DLFLOAT16, &max_tfm_desc));
    const uint64_t partial_bytes = zdnn_getsize_ztensor(&max_tfm_desc);

    ggml_zdnn_pool_alloc<uint8_t> inputs_alloc(ctx->pool, n_workers * inputs_bytes);
    ggml_zdnn_pool_alloc<uint8_t> partials_alloc(ctx->pool, n_calls * partial_bytes);

    const auto chunk_start = [&](int64_t c) {
        return c*ne11 / n_chunks;
    };

    std::atomic<int64_t> next_call = 0;

    ggml_zdnn_parallel_for(n_workers, n_workers, [&](int64_t worker_start, int64_t) {
        uint8_t * inputs_buffer = inputs_alloc.get() + worker_start*inputs_bytes;

        for (int64_t call = next_call.fetch_add(1, std::memory_order_relaxed); call < n_calls; call = next_call.fetch_add(1, std::memory_order_relaxed)) {
            const int64_t g  = call / n_chunks;
            const int64_t r0 = chunk_start(call % n_chunks);
            const int64_t nr = chunk_start(call % n_chunks + 1) - r0;

            zdnn_tensor_desc inputs_pre_tfm_desc;
            zdnn_tensor_desc inputs_tfm_desc;
            zdnn_ztensor     inputs_ztensor;

            zdnn_init_pre_transformed_desc(ZDNN_3DS, INT8, &inputs_pre_tfm_desc, 2, nr, gs);
            ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&inputs_pre_tfm_desc, QUANTIZED_INT8, &inputs_tfm_desc));
            ggml_zdnn_init_scratch_ztensor(&inputs_pre_tfm_desc, &inputs_tfm_desc, &inputs_ztensor, inputs_buffer);

            zdnn_tensor_desc view_pre_tfm_desc;
            zdnn_tensor_desc view_tfm_desc;

            for (int64_t s = 0; s < 2; s++) {
                zdnn_ztensor inputs_view = ggml_zdnn_stack_view(&view_pre_tfm_desc, &view_tfm_desc, inputs_ztensor, 2, s, 1);
                ZDNN_CHECK(zdnn_transform_quantized_ztensor(&inputs_view, false, -127, 127, inputs_q + ((s*n_groups + g)*ne11 + r0)*gs));
            }
            inputs_ztensor.is_transformed = true;

            zdnn_tensor_desc partial_pre_tfm_desc;
            zdnn_tensor_desc partial_tfm_desc;
            zdnn_ztensor     partial_ztensor;

            zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &partial_pre_tfm_desc, 2, nr, ne0);
            ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&partial_pre_tfm_desc, QUANTIZED_DLFLOAT16, &partial_tfm_desc));
            ggml_zdnn_init_scratch_ztensor(&partial_pre_tfm_desc, &partial_tfm_desc, &partial_ztensor, partials_alloc.get() + call*partial_bytes);

            // both stacks are multiplied by the same group of the weights in one broadcast call
            zdnn_tensor_desc weights_pre_tfm_desc;
            zdnn_tensor_desc weights_tfm_desc;
            const zdnn_ztensor weights_ztensor = stacked
                ? ggml_zdnn_stack_view(&weights_pre_tfm_desc, &weights_tfm_desc, weights_extra->ztensor, n_groups, g, 1)
                : weights_extra->ztensor;

            ZDNN_CHECK(zdnn_quantized_matmul_op(&inputs_ztensor, &weights_ztensor, &bias_ztensor,
                                                MATMUL_OP_ADDITION, -127, 127, true, false, true,
                                                nullptr, &partial_ztensor));
        }
    });

    // stack 0 of group 0 is read back into the output, the others wait here to be summed in order
    ggml_zdnn_pool_alloc<float> partial_alloc(ctx->pool, (2 * n_groups - 1) * ne11 * ne0);
    float * y_data = (float *)output->data;

    const auto partial_at = [&](int64_t s, int64_t g) {
        const int64_t i = s*n_groups + g;
        return i == 0 ? y_data : partial_alloc.get() + (i - 1)*ne11*ne0;
    };

    // each stack of the partial results is read back as its own 2D ztensor
    ggml_zdnn_parallel_for(n_threads_out, 2 * n_calls, [&](int64_t t_start, int64_t t_end) {
        zdnn_tensor_desc stack_pre_tfm_desc;
        zdnn_tensor_desc stack_tfm_desc;
        zdnn_ztensor     stack_ztensor;

        for (int64_t t = t_start; t < t_end; t++) {
            const int64_t call = t / 2;
            const int64_t s    = t % 2;
            const int64_t g    = call / n_chunks;
            const int64_t r0   = chunk_start(call % n_chunks);
            const int64_t nr   = chunk_start(call % n_chunks + 1) - r0;

            zdnn_init_pre_transformed_desc(ZDNN_2D, FP32, &stack_pre_tfm_desc, nr, ne0);
            ZDNN_CHECK(zdnn_generate_transformed_desc(&stack_pre_tfm_desc, &stack_tfm_desc));
            zdnn_init_ztensor(&stack_pre_tfm_desc, &stack_tfm_desc, &stack_ztensor);

            const uint64_t stack_bytes = zdnn_getsize_ztensor(&stack_tfm_desc);
            GGML_ASSERT(2*stack_bytes <= partial_bytes);

            stack_ztensor.buffer         = partials_alloc.get() + call*partial_bytes + s*stack_bytes;
            stack_ztensor.buffer_size    = stack_bytes;
            stack_ztensor.is_transformed = true;
            ZDNN_CHECK(zdnn_transform_origtensor(&stack_ztensor, partial_at(s, g) + r0*ne0));
        }
    });

    const float * weights_scales = weights_extra->scales.data();
    ggml_zdnn_parallel_for(n_threads_out, ne1, [&](int64_t i1_start, int64_t i1_end) {
        for (int64_t i1 = i1_start; i1 < i1_end; i1++) {
            float * y = y_data + i1*ne0;

            // the partial results of group 0 are already in y
            for (int64_t g = 0; g < n_groups; g++) {
                ggml_zdnn_dequantize_row(y, partial_at(0, g) + i1*ne0, partial_at(1, g) + i1*ne0,
                                         weights_scales + g*ne0, inputs_scales[g*ne11 + i1], ne0, g > 0);
            }
        }
    });
}
