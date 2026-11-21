#include "ggml.h"
#include "mmid.hpp"
#include "quantize.hpp"
#include "utils.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <vector>

struct ggml_zdnn_mmid_row {
    int32_t slot;
    int32_t token;
};

// rows of one expert that run in one NNPA call
struct ggml_zdnn_mmid_task {
    int32_t expert;
    int32_t n_rows;
    const ggml_zdnn_mmid_row * rows;
};

// memory of one worker, the ztensors are 4K aligned
struct ggml_zdnn_mmid_scratch {
    int8_t  * inputs_q;
    float   * inputs_scales;
    float   * partial;
    uint8_t * inputs_buffer;
    uint8_t * partial_buffer;
};

static inline void ggml_zdnn_mmid_init_ztensor(zdnn_tensor_desc * pre_tfm_desc,
                                               zdnn_tensor_desc * tfm_desc,
                                               zdnn_ztensor     * ztensor,
                                               uint8_t          * buffer) {
    zdnn_init_quantized_ztensor(pre_tfm_desc, tfm_desc, 1.0f, 0.0f, ztensor);
    ztensor->buffer      = buffer;
    ztensor->buffer_size = zdnn_getsize_ztensor(tfm_desc);
}

// the inputs of each group are followed by their rounding errors as extra rows, so one call reads the weights once
static void ggml_zdnn_mul_mat_id_task(const ggml_tensor              * inputs,
                                      const ggml_backend_zdnn_buffer * weights_extra,
                                      const zdnn_ztensor             & bias_ztensor,
                                      const ggml_zdnn_mmid_task      & task,
                                      const ggml_zdnn_mmid_scratch   & scratch,
                                            ggml_tensor              * dst) {
    const ggml_tensor * weights = dst->src[0];

    const int64_t ne0  = dst->ne[0];
    const int64_t ne10 = inputs->ne[0];
    const int64_t ne11 = inputs->ne[1];
    const int64_t n_r  = task.n_rows;

    const bool    stacked  = weights_extra->pre_tfm_desc.layout == ZDNN_3DS;
    const int64_t n_stacks = stacked ? weights_extra->pre_tfm_desc.dim3 : 1;
    const int64_t n_groups = n_stacks / weights->ne[2];
    const int64_t gs       = stacked ? weights_extra->pre_tfm_desc.dim2 : ne10;

    const float * channel_scales = weights_extra->channel_scales.data() + task.expert*ne10;
    const float * weights_scales = weights_extra->scales.data() + task.expert*n_groups*ne0;

    for (int64_t r = 0; r < n_r; r++) {
        const ggml_zdnn_mmid_row & row = task.rows[r];
        const float * x = (const float *)((const char *)inputs->data + (row.slot % ne11)*inputs->nb[1] + row.token*inputs->nb[2]);

        ggml_zdnn_quantize_row(x, ne10, gs, channel_scales,
                               scratch.inputs_q + r*gs, scratch.inputs_q + (n_r + r)*gs, 2*n_r*gs,
                               scratch.inputs_scales + r, n_r);
    }

    zdnn_tensor_desc inputs_pre_tfm_desc;
    zdnn_tensor_desc inputs_tfm_desc;
    zdnn_ztensor     inputs_ztensor;

    zdnn_init_pre_transformed_desc(ZDNN_3DS, INT8, &inputs_pre_tfm_desc, n_groups, 2 * n_r, gs);
    ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&inputs_pre_tfm_desc, QUANTIZED_INT8, &inputs_tfm_desc));
    ggml_zdnn_mmid_init_ztensor(&inputs_pre_tfm_desc, &inputs_tfm_desc, &inputs_ztensor, scratch.inputs_buffer);
    ZDNN_CHECK(zdnn_transform_quantized_ztensor(&inputs_ztensor, false, -127, 127, scratch.inputs_q));

    zdnn_tensor_desc partial_pre_tfm_desc;
    zdnn_tensor_desc partial_tfm_desc;
    zdnn_ztensor     partial_ztensor;

    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &partial_pre_tfm_desc, n_groups, 2 * n_r, ne0);
    ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&partial_pre_tfm_desc, QUANTIZED_DLFLOAT16, &partial_tfm_desc));
    ggml_zdnn_mmid_init_ztensor(&partial_pre_tfm_desc, &partial_tfm_desc, &partial_ztensor, scratch.partial_buffer);

    zdnn_tensor_desc view_pre_tfm_desc;
    zdnn_tensor_desc view_tfm_desc;
    const zdnn_ztensor weights_ztensor = n_stacks > 1
        ? ggml_zdnn_stack_view(&view_pre_tfm_desc, &view_tfm_desc, weights_extra->ztensor, n_stacks, task.expert*n_groups, n_groups)
        : weights_extra->ztensor;

    // the bias is all zeros, so it is already the pre-computed bias
    ZDNN_CHECK(zdnn_quantized_matmul_op(&inputs_ztensor, &weights_ztensor, &bias_ztensor,
                                        MATMUL_OP_ADDITION, -127, 127, true, false, true,
                                        nullptr, &partial_ztensor));

    // each group of the partial results is read back as its own 2D ztensor
    zdnn_tensor_desc group_pre_tfm_desc;
    zdnn_tensor_desc group_tfm_desc;
    zdnn_ztensor     group_ztensor;

    zdnn_init_pre_transformed_desc(ZDNN_2D, FP32, &group_pre_tfm_desc, 2 * n_r, ne0);
    ZDNN_CHECK(zdnn_generate_transformed_desc(&group_pre_tfm_desc, &group_tfm_desc));
    zdnn_init_ztensor(&group_pre_tfm_desc, &group_tfm_desc, &group_ztensor);

    const uint64_t group_bytes = zdnn_getsize_ztensor(&group_tfm_desc);
    GGML_ASSERT(partial_ztensor.buffer_size == (uint64_t)n_groups * group_bytes);

    for (int64_t g = 0; g < n_groups; g++) {
        group_ztensor.buffer         = (char *)partial_ztensor.buffer + g*group_bytes;
        group_ztensor.buffer_size    = group_bytes;
        group_ztensor.is_transformed = true;
        ZDNN_CHECK(zdnn_transform_origtensor(&group_ztensor, scratch.partial));

        const float * sw = weights_scales + g*ne0;
        for (int64_t r = 0; r < n_r; r++) {
            const ggml_zdnn_mmid_row & row = task.rows[r];

            const float   sa = scratch.inputs_scales[g*n_r + r];
            const float * p  = scratch.partial + r*ne0;
            const float * pr = scratch.partial + (n_r + r)*ne0;
            float       * y  = (float *)((char *)dst->data + row.slot*dst->nb[1] + row.token*dst->nb[2]);

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
}

bool ggml_zdnn_supports_mul_mat_id(const ggml_backend_zdnn_device_context * ctx_dev, const ggml_tensor * op) {
    const ggml_tensor * weights = op->src[0];
    const ggml_tensor * inputs  = op->src[1];

    if (weights->type != GGML_TYPE_Q8_0 || inputs->type != GGML_TYPE_F32 || !ctx_dev->has_parmblkformat_1) {
        return false;
    }

    // the ztensor belongs to the weights, and the rows of the inputs are read with their strides
    if (weights->view_src != nullptr || !ggml_is_contiguous(weights) || inputs->nb[0] != sizeof(float)) {
        return false;
    }

    // benchmarked on Granite-3.0-1B-A400M and Granite-4.0-H-Tiny Q8_0: the CPU is faster for fewer tokens at 6 threads and up
    if (inputs->ne[2] < std::max<int64_t>(1, ctx_dev->n_threads - 4)) {
        return false;
    }

    const int64_t max_dim = ctx_dev->max_size;

    const int64_t gs       = ggml_zdnn_q_group_size(weights);
    const int64_t n_groups = gs > 0 ? (weights->ne[0] + gs - 1) / gs : 1;

    if (weights->ne[0] > max_dim || weights->ne[1] > max_dim || weights->ne[2] * n_groups > max_dim) {
        return false;
    }

    zdnn_tensor_desc pre_tfm_desc;
    zdnn_tensor_desc tfm_desc;
    ggml_zdnn_q_weights_desc(&pre_tfm_desc, &tfm_desc, weights);

    return zdnn_getsize_ztensor(&tfm_desc) <= zdnn_get_nnpa_max_tensor_size();
}

void ggml_zdnn_mul_mat_id(ggml_backend_zdnn_context * ctx, ggml_tensor * dst) {
    const ggml_tensor * weights = dst->src[0];
    const ggml_tensor * inputs  = dst->src[1];
    const ggml_tensor * ids     = dst->src[2];

    const int64_t ne0      = dst->ne[0];
    const int64_t ne10     = inputs->ne[0];
    const int64_t n_expert = weights->ne[2];

    GGML_ASSERT(dst->nb[0] == sizeof(float));
    GGML_ASSERT(inputs->nb[0] == sizeof(float));

    ggml_backend_zdnn_buffer * weights_extra = (ggml_backend_zdnn_buffer *)weights->extra;

    // weights read straight into a host buffer skip set_tensor, so transform them on first use
    if (!weights_extra->ztensor.is_transformed) {
        ggml_zdnn_load_tensor_q(weights_extra, weights);
    }

    // group the rows by expert, a skipped slot has no matrix and its output is zero
    const auto id_at = [&](int64_t slot, int64_t token) {
        return *(const int32_t *)((const char *)ids->data + slot*ids->nb[0] + token*ids->nb[1]);
    };

    std::vector<int32_t> expert_first(n_expert + 1, 0);
    for (int64_t token = 0; token < ids->ne[1]; token++) {
        for (int64_t slot = 0; slot < ids->ne[0]; slot++) {
            const int32_t id = id_at(slot, token);
            GGML_ASSERT(id >= -1 && id < n_expert);

            if (id == -1) {
                memset((char *)dst->data + slot*dst->nb[1] + token*dst->nb[2], 0, ne0*sizeof(float));
                continue;
            }

            expert_first[id + 1]++;
        }
    }

    for (int64_t e = 0; e < n_expert; e++) {
        expert_first[e + 1] += expert_first[e];
    }

    std::vector<ggml_zdnn_mmid_row> rows(expert_first[n_expert]);
    if (rows.empty()) {
        return;
    }

    std::vector<int32_t> expert_next(expert_first.begin(), expert_first.end() - 1);
    for (int64_t token = 0; token < ids->ne[1]; token++) {
        for (int64_t slot = 0; slot < ids->ne[0]; slot++) {
            const int32_t id = id_at(slot, token);
            if (id != -1) {
                rows[expert_next[id]++] = { (int32_t)slot, (int32_t)token };
            }
        }
    }

    // split popular experts to balance the workers, benchmarked: 64 rows is up to 18% faster than 256 at 8 threads
    constexpr int64_t chunk_rows = 64;

    std::vector<ggml_zdnn_mmid_task> tasks;
    int64_t max_rows = 0;
    for (int64_t e = 0; e < n_expert; e++) {
        for (int64_t r = expert_first[e]; r < expert_first[e + 1]; r += chunk_rows) {
            const int64_t n = std::min<int64_t>(chunk_rows, expert_first[e + 1] - r);
            tasks.push_back({ (int32_t)e, (int32_t)n, rows.data() + r });
            max_rows = std::max(max_rows, n);
        }
    }

    const bool    stacked  = weights_extra->pre_tfm_desc.layout == ZDNN_3DS;
    const int64_t n_stacks = stacked ? weights_extra->pre_tfm_desc.dim3 : 1;
    const int64_t n_groups = n_stacks / n_expert;
    const int64_t gs       = stacked ? weights_extra->pre_tfm_desc.dim2 : ne10;

    // the bias cache and the pool are not thread safe, so use them before the workers start
    const zdnn_ztensor & bias_ztensor = ggml_zdnn_zero_bias(ctx, n_groups, ne0);

    zdnn_tensor_desc pre_tfm_desc;
    zdnn_tensor_desc tfm_desc;

    zdnn_init_pre_transformed_desc(ZDNN_3DS, INT8, &pre_tfm_desc, n_groups, 2 * max_rows, gs);
    ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&pre_tfm_desc, QUANTIZED_INT8, &tfm_desc));
    const uint64_t inputs_bytes = zdnn_getsize_ztensor(&tfm_desc);

    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &pre_tfm_desc, n_groups, 2 * max_rows, ne0);
    ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&pre_tfm_desc, QUANTIZED_DLFLOAT16, &tfm_desc));
    const uint64_t partial_bytes = zdnn_getsize_ztensor(&tfm_desc);

    const uint64_t inputs_q_bytes      = GGML_PAD(2 * n_groups * max_rows * gs, 4096);
    const uint64_t inputs_scales_bytes = GGML_PAD(n_groups * max_rows * sizeof(float), 4096);
    const uint64_t partial_f32_bytes   = GGML_PAD(2 * max_rows * ne0 * sizeof(float), 4096);
    const uint64_t worker_bytes        = inputs_q_bytes + inputs_scales_bytes + partial_f32_bytes + inputs_bytes + partial_bytes;

    // starting threads costs more than it saves for very small inputs
    const int32_t n_workers = (int64_t)rows.size()*ne10 >= 4096 ? (int32_t)std::min<int64_t>(ctx->n_threads, tasks.size()) : 1;

    ggml_zdnn_pool_alloc<uint8_t> scratch_alloc(ctx->pool, n_workers * worker_bytes);

    std::atomic<int64_t> next_task = 0;

    ggml_zdnn_parallel_for(n_workers, n_workers, [&](int64_t worker_start, int64_t) {
        uint8_t * mem = scratch_alloc.get() + worker_start*worker_bytes;

        const ggml_zdnn_mmid_scratch scratch = {
            /* .inputs_q       = */ (int8_t *)mem,
            /* .inputs_scales  = */ (float *)(mem + inputs_q_bytes),
            /* .partial        = */ (float *)(mem + inputs_q_bytes + inputs_scales_bytes),
            /* .inputs_buffer  = */ mem + inputs_q_bytes + inputs_scales_bytes + partial_f32_bytes,
            /* .partial_buffer = */ mem + inputs_q_bytes + inputs_scales_bytes + partial_f32_bytes + inputs_bytes,
        };

        for (int64_t t = next_task.fetch_add(1, std::memory_order_relaxed); t < (int64_t)tasks.size(); t = next_task.fetch_add(1, std::memory_order_relaxed)) {
            ggml_zdnn_mul_mat_id_task(inputs, weights_extra, bias_ztensor, tasks[t], scratch, dst);
        }
    });
}
