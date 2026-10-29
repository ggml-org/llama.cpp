#include "sampling.cuh"

#ifdef GGML_CUDA_USE_CUB
#include <cub/cub.cuh>
#include <map>

struct sample_input {
    ggml_sample_params params;
    const float * logits;
    const float * q;
};

struct ggml_cuda_sampling_context {
    int device;
    int n_vocab = 0;
    int n_rows = 0;
    void * storage = nullptr;
    void * workspace = nullptr;
    size_t workspace_size = 0;
    sample_input * input = nullptr;
    ggml_sample_result * output = nullptr;
    std::map<int, std::pair<cudaGraph_t, cudaGraphExec_t>> graphs;

    explicit ggml_cuda_sampling_context(int device) : device(device) {}
    ~ggml_cuda_sampling_context() { clear(); }
    void clear() {
        ggml_cuda_set_device(device);
        for (auto & entry : graphs) {
            CUDA_CHECK(cudaGraphExecDestroy(entry.second.second));
            CUDA_CHECK(cudaGraphDestroy(entry.second.first));
        }
        graphs.clear();
        CUDA_CHECK(cudaFree(storage));
        CUDA_CHECK(cudaFree(workspace));
        CUDA_CHECK(cudaFree(input));
        CUDA_CHECK(cudaFree(output));
        storage = workspace = nullptr;
        input = nullptr;
        output = nullptr;
    }
};

static __global__ void sample_logits(float * keys, uint64_t * sort_keys, int * ids, int n_vocab, const sample_input * input) {
    const auto & params = input->params;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (id >= n_vocab) {
        return;
    }
    float value = input->logits[(int64_t) params.rows[row] * n_vocab + id];
    for (int i = 0; i < params.n_bias; ++i) {
        if (id == params.bias_ids[i]) {
            value += params.bias_values[i];
        }
    }
    keys[row * n_vocab + id] = value;
    const uint32_t bits = __float_as_uint(value);
    const uint32_t ordered = bits ^ ((uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U);
    // High bits keep rows separate when sorting the whole batch in one pass.
    sort_keys[row * n_vocab + id] = ((uint64_t) (params.n_draft - row) << 32) | ordered;
    ids[row * n_vocab + id] = id;
}

static __global__ void sample_weights(const float * logits, const int * ids, float * weights,
        const float * cdf, int n_vocab, const sample_input * input) {
    const auto & params = input->params;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (col >= n_vocab) {
        return;
    }
    const int offset = row * n_vocab;
    const float maximum = logits[offset + ids[offset]];
    const float value = logits[offset + ids[offset + col]];
    bool keep = true;
    if (cdf && params.top_p < 1.0f && col >= params.min_keep && col > 0) {
        keep = cdf[offset + col - 1] < params.top_p * cdf[offset + n_vocab - 1];
    }
    const float scale = cdf || params.temperature_first ? 1.0f / params.temperature : 1.0f;
    weights[offset + col] = keep ? expf((value - maximum) * scale) : 0.0f;
}

static __global__ void sample_residual(const float * p_weights, const float * cdf, const int * ids,
        float * weights, float * p_selected, int n_vocab, const sample_input * input) {
    const auto & params = input->params;
    const auto * q = input->q;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (col >= n_vocab) {
        return;
    }
    const int offset = row * n_vocab;
    const float total = cdf[offset + n_vocab - 1];
    const float p = p_weights[offset + col] / total;
    if (ids[offset + col] == params.draft[row]) {
        p_selected[row] = p;
    }
    weights[offset + col] = fmaxf(0.0f, p - q[((int64_t) params.q_row + row) * n_vocab + ids[offset + col]]);
}

static __device__ int sample_find(const float * cdf, int size, float uniform) {
    const float threshold = uniform * cdf[size - 1];
    int lo = 0;
    int hi = size - 1;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (cdf[mid] <= threshold) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

static __global__ void sample_finish(const float * p_selected, const float * cdf, const float * residual_cdf,
        const int * ids, int n_vocab, const sample_input * input, ggml_sample_result * result) {
    const auto & params = input->params;
    const auto * q = input->q;
    result->n_tokens = 0;
    for (int row = 0; row <= params.n_draft; ++row) {
        if (!(cdf[(row + 1) * n_vocab - 1] > 0.0f) || !isfinite(cdf[(row + 1) * n_vocab - 1])) {
            return;
        }
    }
    int row = 0;
    for (; row < params.n_draft; ++row) {
        const int token = params.draft[row];
        const float q_x = q[((int64_t) params.q_row + row) * n_vocab + token];
        if (!(q_x > 0.0f && params.uniforms[row] < fminf(1.0f, p_selected[row] / q_x))) {
            break;
        }
        result->tokens[row] = token;
    }
    const int offset = row * n_vocab;
    const float * distribution = row < params.n_draft && residual_cdf[offset + n_vocab - 1] > 0.0f
        ? residual_cdf + offset : cdf + offset;
    const int selected = sample_find(distribution, n_vocab, params.uniforms[params.n_draft + row]);
    result->tokens[row] = ids[offset + selected];
    result->n_tokens = row + 1;
}
#endif

bool ggml_cuda_sample(ggml_backend_cuda_context & ctx, const ggml_tensor * logits,
        const ggml_tensor * draft_probs, const ggml_sample_params * params, ggml_sample_result * result) {
#ifndef GGML_CUDA_USE_CUB
    GGML_UNUSED(ctx); GGML_UNUSED(logits); GGML_UNUSED(draft_probs); GGML_UNUSED(params); GGML_UNUSED(result);
    return false;
#else
    if (!logits || !params || !result || logits->type != GGML_TYPE_F32 || !ggml_is_contiguous(logits) ||
            logits->ne[2] != 1 || logits->ne[3] != 1 ||
            logits->ne[0] <= 0 || logits->ne[0] > INT_MAX / (GGML_SAMPLE_MAX_DRAFT + 1) ||
            params->n_draft < 0 || params->n_draft > GGML_SAMPLE_MAX_DRAFT ||
            params->n_bias < 0 || params->n_bias > GGML_SAMPLE_MAX_BIAS ||
            !std::isfinite(params->temperature) || params->temperature <= 0.0f ||
            !std::isfinite(params->top_p) || params->top_p <= 0.0f || params->top_p > 1.0f ||
            params->min_keep < 0 || params->min_keep > logits->ne[0]) {
        return false;
    }
    const int n_vocab = logits->ne[0];
    const int n_rows = params->n_draft + 1;
    for (int row = 0; row < n_rows; ++row) {
        if (params->rows[row] < 0 || params->rows[row] >= logits->ne[1]) {
            return false;
        }
    }
    if (params->n_draft > 0) {
        if (!draft_probs || draft_probs->type != GGML_TYPE_F32 || !ggml_is_contiguous(draft_probs) ||
                draft_probs->ne[2] != 1 || draft_probs->ne[3] != 1 ||
                draft_probs->ne[0] != n_vocab || params->q_row < 0 ||
                (int64_t) params->q_row + params->n_draft > draft_probs->ne[1]) {
            return false;
        }
        for (int row = 0; row < params->n_draft; ++row) {
            if (params->draft[row] < 0 || params->draft[row] >= n_vocab) {
                return false;
            }
        }
    }
    for (int i = 0; i < 2 * params->n_draft + 1; ++i) {
        if (!(params->uniforms[i] >= 0.0f && params->uniforms[i] < 1.0f)) {
            return false;
        }
    }

    ggml_cuda_set_device(ctx.device);
    const auto stream = ctx.stream();
#ifdef USE_CUDA_GRAPH
    static const bool use_graphs = std::getenv("GGML_CUDA_DISABLE_GRAPHS") == nullptr;
#else
    const bool use_graphs = false;
#endif
    if (!ctx.sampling) {
        ctx.sampling = std::make_shared<ggml_cuda_sampling_context>(ctx.device);
    }
    auto & state = *ctx.sampling;
    if (state.n_vocab != n_vocab || state.n_rows < n_rows) {
        state.clear();
        state.n_vocab = n_vocab;
        state.n_rows = n_rows;
        CUDA_CHECK(cudaMalloc(&state.storage, (size_t) n_rows * n_vocab * (6 * sizeof(float) + 2 * sizeof(int) + 2 * sizeof(uint64_t)) + GGML_SAMPLE_MAX_DRAFT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.input, sizeof(sample_input)));
        CUDA_CHECK(cudaMalloc(&state.output, sizeof(ggml_sample_result)));
    }
    const size_t capacity = (size_t) state.n_rows * n_vocab;
    float * keys = (float *) state.storage;
    float * weights = keys + capacity;
    float * cdf = weights + capacity;
    float * cutoff_cdf = cdf + capacity;
    float * residual = cutoff_cdf + capacity;
    float * residual_cdf = residual + capacity;
    int * ids = (int *) (residual_cdf + capacity);
    int * indices = ids + capacity;
    uint64_t * sort_keys = (uint64_t *) (indices + capacity);
    uint64_t * sorted_keys = sort_keys + capacity;
    float * p_selected = (float *) (sorted_keys + capacity);
    size_t scan_size = 0;
    size_t sort_size = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, scan_size, weights, cdf, n_vocab, stream));
    int row_bits = 0;
    while ((1 << row_bits) < n_rows) { ++row_bits; }
    const int count = n_vocab * n_rows;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(nullptr, sort_size, sort_keys, sorted_keys, indices, ids, count, 0, 32 + row_bits, stream));
    if (state.workspace && sort_size > state.workspace_size) {
        for (auto & entry : state.graphs) {
            CUDA_CHECK(cudaGraphExecDestroy(entry.second.second));
            CUDA_CHECK(cudaGraphDestroy(entry.second.first));
        }
        state.graphs.clear();
        CUDA_CHECK(cudaFree(state.workspace));
        state.workspace = nullptr;
    }
    if (!state.workspace) {
        state.workspace_size = std::max(scan_size, sort_size);
        CUDA_CHECK(cudaMalloc(&state.workspace, state.workspace_size));
    }
    sample_input input = {*params, (const float *) logits->data, draft_probs ? (const float *) draft_probs->data : nullptr};
    CUDA_CHECK(cudaMemcpyAsync(state.input, &input, sizeof(input), cudaMemcpyHostToDevice, stream));
    if (!use_graphs || !state.graphs.count(n_rows)) {
        if (use_graphs) {
            CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        }
        const dim3 blocks((n_vocab + 255) / 256, n_rows, 1);
        sample_logits<<<blocks, 256, 0, stream>>>(keys, sort_keys, indices, n_vocab, state.input);
        CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(state.workspace, sort_size, sort_keys, sorted_keys,
                indices, ids, count, 0, 32 + row_bits, stream));
        sample_weights<<<blocks, 256, 0, stream>>>(keys, ids, weights, nullptr, n_vocab, state.input);
        for (int row = 0; row < n_rows; ++row) {
            CUDA_CHECK(cub::DeviceScan::InclusiveSum(state.workspace, scan_size, weights + row * n_vocab,
                    cutoff_cdf + row * n_vocab, n_vocab, stream));
        }
        sample_weights<<<blocks, 256, 0, stream>>>(keys, ids, weights, cutoff_cdf, n_vocab, state.input);
        for (int row = 0; row < n_rows; ++row) {
            CUDA_CHECK(cub::DeviceScan::InclusiveSum(state.workspace, scan_size, weights + row * n_vocab,
                    cdf + row * n_vocab, n_vocab, stream));
        }
        if (params->n_draft > 0) {
            const dim3 draft_blocks(blocks.x, params->n_draft, 1);
            sample_residual<<<draft_blocks, 256, 0, stream>>>(weights, cdf, ids, residual, p_selected, n_vocab, state.input);
            for (int row = 0; row < params->n_draft; ++row) {
                CUDA_CHECK(cub::DeviceScan::InclusiveSum(state.workspace, scan_size, residual + row * n_vocab,
                        residual_cdf + row * n_vocab, n_vocab, stream));
            }
        }
        sample_finish<<<1, 1, 0, stream>>>(p_selected, cdf, residual_cdf, ids, n_vocab, state.input, state.output);
        if (use_graphs) {
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t executable = nullptr;
            CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            state.graphs[n_rows] = {graph, executable};
        }
    }
    if (use_graphs) {
        CUDA_CHECK(cudaGraphLaunch(state.graphs.at(n_rows).second, stream));
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(result, state.output, sizeof(*result), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return result->n_tokens > 0;
#endif
}
