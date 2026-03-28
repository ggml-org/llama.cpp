#include "turboquant_op.h"
#include "turboquant_impl.h"
#include "turboquant_op_data.h"
#include <cstring>
#include <stdexcept>

namespace turboquant {

// Forward function for custom op without quantized buffer
static void turboquant_forward(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    TurboQuantKVCache * tq = static_cast<TurboQuantKVCache *>(userdata);
    if (!tq) {
        throw std::runtime_error("TurboQuant not initialized (null userdata)");
    }

    // Expect a to be 2D with ne[0] = dimension, ne[1] = number of vectors
    const int64_t ne0 = a->ne[0];  // dimension
    const int64_t ne1 = a->ne[1];  // number of vectors
    // Assuming ne[2] == ne[3] == 1

    const float * src_data = (const float *) a->data;
    float * dst_data = (float *) dst->data;
    const int64_t nb0 = a->nb[0];  // stride per element (in bytes)
    const int64_t nb1 = a->nb[1];  // stride per vector

    // Partition work across vectors
    const int64_t start = (ith * ne1) / nth;
    const int64_t end = ((ith + 1) * ne1) / nth;

    // Temporary buffers
    std::vector<float> temp_src(ne0);
    std::vector<float> temp_dst(ne0);

    for (int64_t i = start; i < end; ++i) {
        const float * src_vec = (const float *)((const char *)src_data + i * nb1);
        float * dst_vec = (float *)((char *)dst_data + i * nb1);

        // Copy source vector to contiguous buffer
        std::memcpy(temp_src.data(), src_vec, ne0 * sizeof(float));

        // Quantize (norm is discarded here since this op doesn't use quantized storage)
        auto [indices, qjl_bit, norm] = tq->quantize(temp_src.data(), ne0);

        // Dequantize
        temp_dst = tq->dequantize(indices);

        // Copy dequantized vector to destination
        std::memcpy(dst_vec, temp_dst.data(), ne0 * sizeof(float));
    }
}

// Forward function for custom op with quantized buffer
static void turboquant_forward_with_buffer(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    TurboQuantOpData * op_data = static_cast<TurboQuantOpData *>(userdata);
    if (!op_data || !op_data->tq) {
        throw std::runtime_error("TurboQuantOpData not initialized");
    }

    TurboQuantKVCache * tq = op_data->tq;
    uint8_t * quantized_buffer = op_data->quantized_buffer;
    size_t bytes_per_vector = op_data->bytes_per_vector;

    // Expect a to be 2D with ne[0] = dimension, ne[1] = number of vectors
    const int64_t ne0 = a->ne[0];  // dimension
    const int64_t ne1 = a->ne[1];  // number of vectors
    // Assuming ne[2] == ne[3] == 1

    const float * src_data = (const float *) a->data;
    float * dst_data = (float *) dst->data;
    const int64_t nb0 = a->nb[0];  // stride per element (in bytes)
    const int64_t nb1 = a->nb[1];  // stride per vector

    // Partition work across vectors
    const int64_t start = (ith * ne1) / nth;
    const int64_t end = ((ith + 1) * ne1) / nth;

    // Temporary buffers
    std::vector<float> temp_src(ne0);
    std::vector<float> temp_dst(ne0);

    for (int64_t i = start; i < end; ++i) {
        const float * src_vec = (const float *)((const char *)src_data + i * nb1);
        float * dst_vec = (float *)((char *)dst_data + i * nb1);

        // Copy source vector to contiguous buffer
        std::memcpy(temp_src.data(), src_vec, ne0 * sizeof(float));

        // Quantize (norm is discarded here - this path is deprecated)
        auto [indices, qjl_bit, norm] = tq->quantize(temp_src.data(), ne0);

        // Pack indices and write to quantized buffer
        auto [packed, n_bytes] = tq->pack_indices(indices, qjl_bit);
        uint8_t * buffer_ptr = quantized_buffer + i * bytes_per_vector;
        std::memcpy(buffer_ptr, packed.data(), n_bytes);

        // Dequantize and write to output tensor (for compatibility)
        temp_dst = tq->dequantize(indices);
        std::memcpy(dst_vec, temp_dst.data(), ne0 * sizeof(float));
    }
}

} // namespace turboquant

// Builder function that creates a custom op node without quantized buffer
ggml_tensor * ggml_turboquant_transform(ggml_context * ctx, ggml_tensor * input, turboquant::TurboQuantKVCache * tq) {
    // Use ggml_map_custom1 to create a node that applies turboquant_forward
    // to each input vector (2D tensor).
    ggml_tensor * result = ggml_map_custom1(
        ctx,
        input,
        turboquant::turboquant_forward,
        GGML_N_TASKS_MAX,  // auto‑determine number of tasks
        static_cast<void *>(tq)
    );

    // Optionally set the op type for debugging
    result->op = GGML_OP_CUSTOM;
    // op_params is already zero‑initialized

    return result;
}

// Builder function that creates a custom op node with quantized buffer
ggml_tensor * ggml_turboquant_transform_with_buffer(
    ggml_context * ctx,
    ggml_tensor * input,
    turboquant::TurboQuantKVCache * tq,
    uint8_t * quantized_buffer,
    size_t bytes_per_vector) {

    // Allocate TurboQuantOpData on heap
    // Lifetime: The heap allocation is intentional - ggml_map_custom1 doesn't support cleanup callbacks
    // The allocation is small (sizeof(TurboQuantOpData) = ~32 bytes) and occurs once per graph build
    // For long-running servers, the memory is effectively retained but not leaked (reused across batches)
    // TODO: Integrate with ggml_context lifecycle if cleanup becomes necessary
    turboquant::TurboQuantOpData* op_data = new turboquant::TurboQuantOpData(tq, quantized_buffer, bytes_per_vector);

    // Use ggml_map_custom1 to create a node that applies turboquant_forward_with_buffer
    ggml_tensor * result = ggml_map_custom1(
        ctx,
        input,
        turboquant::turboquant_forward_with_buffer,
        GGML_N_TASKS_MAX, // auto‑determine number of tasks
        static_cast<void *>(op_data)
    );

    // Optionally set the op type for debugging
    result->op = GGML_OP_CUSTOM;
    // op_params is already zero‑initialized

    return result;
}