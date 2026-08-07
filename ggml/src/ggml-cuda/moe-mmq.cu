#include "mmid.cuh"
#include "mmq-cutlass.cuh"
#include "mmvq.cuh"
#include "moe-mmq.cuh"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>

struct moe_cutlass_stage_config {
    int  tile_n;
    bool swap_ab;
};

struct moe_cutlass_config {
    bool disabled;
    bool log_config;
    bool pdl;
    bool prefix_schedule;
    bool cta_quant;
    bool cta_activation;
    bool validate_support;
    bool inplace_weights;
    int  activation_rows;
    moe_cutlass_stage_config w13;
    moe_cutlass_stage_config w2;
};

static bool moe_cutlass_env_flag(const char * name, bool default_value = false) {
    const char * value = std::getenv(name);
    return value == nullptr ? default_value : std::atoi(value) != 0;
}

static int moe_cutlass_env_int(const char * name, int default_value, int min_value, int max_value) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    errno             = 0;
    char *     end    = nullptr;
    const long result = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || result < min_value || result > max_value) {
        GGML_ABORT("invalid value for %s: %s", name, value);
    }
    return (int) result;
}

static int moe_cutlass_env_tile_n(const char * name) {
    const int value = moe_cutlass_env_int(name, 0, 0, 128);
    if (value != 0 && value != 32 && value != 64 && value != 128) {
        GGML_ABORT("%s must be 0, 32, 64, or 128", name);
    }
    return value;
}

static int moe_cutlass_env_activation_rows() {
    const int value = moe_cutlass_env_int("GGML_CUDA_MOE_MMQ_CUTLASS_ACTIVATION_ROWS", 4, 1, 8);
    if (value != 1 && value != 4 && value != 8) {
        GGML_ABORT("GGML_CUDA_MOE_MMQ_CUTLASS_ACTIVATION_ROWS must be 1, 4, or 8");
    }
    return value;
}

static const moe_cutlass_config & moe_cutlass_get_config() {
    static const moe_cutlass_config config = {
        moe_cutlass_env_flag("GGML_CUDA_MOE_MMQ_DISABLE"),
        moe_cutlass_env_flag("GGML_CUDA_MOE_MMQ_LOG_CONFIG"),
        moe_cutlass_env_flag("GGML_CUDA_MOE_MMQ_CUTLASS_PDL"),
        !moe_cutlass_env_flag("GGML_CUDA_MOE_MMQ_CUTLASS_PREFIX_DISABLE"),
        !moe_cutlass_env_flag("GGML_CUDA_MOE_MMQ_CUTLASS_CTA_QUANT_DISABLE"),
        !moe_cutlass_env_flag("GGML_CUDA_MOE_MMQ_CUTLASS_CTA_ACTIVATION_DISABLE"),
        moe_cutlass_env_flag("GGML_CUDA_MOE_MMQ_CUTLASS_VALIDATE_SUPPORT"),
        moe_cutlass_env_flag("GGML_CUDA_MOE_MMQ_CUTLASS_INPLACE_WEIGHTS"),
        moe_cutlass_env_activation_rows(),
        {
            moe_cutlass_env_tile_n("GGML_CUDA_MOE_MMQ_CUTLASS_W13_TILE_N"),
            moe_cutlass_env_int("GGML_CUDA_MOE_MMQ_CUTLASS_W13_SWAP_AB", 1, 0, 1) != 0,
        },
        {
            moe_cutlass_env_tile_n("GGML_CUDA_MOE_MMQ_CUTLASS_W2_TILE_N"),
            moe_cutlass_env_int("GGML_CUDA_MOE_MMQ_CUTLASS_W2_SWAP_AB", 1, 0, 1) != 0,
        },
    };
    return config;
}

static ggml_cuda_cutlass_config moe_cutlass_gemm_config(
        const moe_cutlass_stage_config & config,
        int64_t                          n_rows,
        int64_t                          n_cols,
        int                              n_experts,
        bool                             pdl) {
    int tile_n = config.tile_n;
    if (tile_n == 0) {
        if (config.swap_ab) {
            const int64_t rows_per_expert = (n_rows + n_experts - 1) / n_experts;
            tile_n = rows_per_expert <= 32 ? 32 : rows_per_expert <= 64 ? 64 : 128;
        } else {
            tile_n = n_cols % 128 == 0 ? 128 : n_cols % 64 == 0 ? 64 : 32;
        }
    }
    return { tile_n, config.swap_ab, pdl, false };
}

static bool moe_cutlass_shapes(const ggml_cuda_moe_mmq_args & args) {
    constexpr int64_t n_expert      = 128;
    constexpr int64_t n_expert_used = 4;

    const int64_t n_embd   = args.input->ne[0];
    const int64_t n_ff     = args.activation->ne[0];
    const int64_t n_tokens = args.ids->ne[1];

    return n_embd == 2880 && n_ff == 2880 && args.gate_up->type == GGML_TYPE_MXFP4 &&
           args.down->type == GGML_TYPE_MXFP4 && args.input->type == GGML_TYPE_F32 && args.ids->type == GGML_TYPE_I32 &&
           args.gate_up_dst->type == GGML_TYPE_F32 && args.gate_up_bias->type == GGML_TYPE_F32 &&
           args.gate_up_biased->type == GGML_TYPE_F32 && args.activation->type == GGML_TYPE_F32 &&
           args.down_dst->type == GGML_TYPE_F32 && args.down_bias->type == GGML_TYPE_F32 &&
           args.down_biased->type == GGML_TYPE_F32 && args.weights->type == GGML_TYPE_F32 &&
           args.weighted->type == GGML_TYPE_F32 && args.dst->type == GGML_TYPE_F32 && args.gate_up->ne[0] == n_embd &&
           args.gate_up->ne[1] == 2 * n_ff && args.gate_up->ne[2] == n_expert && args.gate_up->ne[3] == 1 &&
           args.input->ne[1] == 1 && args.input->ne[2] == n_tokens && args.input->ne[3] == 1 &&
           args.ids->ne[0] == n_expert_used && args.ids->ne[2] == 1 && args.ids->ne[3] == 1 &&
           args.gate_up_dst->ne[0] == 2 * n_ff && args.gate_up_dst->ne[1] == n_expert_used &&
           args.gate_up_dst->ne[2] == n_tokens && args.gate_up_dst->ne[3] == 1 &&
           args.gate_up_bias->ne[0] == 2 * n_ff && args.gate_up_bias->ne[1] == n_expert &&
           args.gate_up_bias->ne[2] == 1 && args.gate_up_bias->ne[3] == 1 &&
           ggml_are_same_shape(args.gate_up_biased, args.gate_up_dst) && args.activation->ne[1] == n_expert_used &&
           args.activation->ne[2] == n_tokens && args.activation->ne[3] == 1 && args.down->ne[0] == n_ff &&
           args.down->ne[1] == n_embd && args.down->ne[2] == n_expert && args.down->ne[3] == 1 &&
           args.down_dst->ne[0] == n_embd && args.down_dst->ne[1] == n_expert_used &&
           args.down_dst->ne[2] == n_tokens && args.down_dst->ne[3] == 1 && args.down_bias->ne[0] == n_embd &&
           args.down_bias->ne[1] == n_expert && args.down_bias->ne[2] == 1 && args.down_bias->ne[3] == 1 &&
           ggml_are_same_shape(args.down_biased, args.down_dst) && args.weights->ne[0] == 1 &&
           args.weights->ne[1] == n_expert_used && args.weights->ne[2] == n_tokens && args.weights->ne[3] == 1 &&
           ggml_are_same_shape(args.weighted, args.down_dst) && args.dst->ne[0] == n_embd &&
           args.dst->ne[1] == n_tokens && args.dst->ne[2] == 1 && args.dst->ne[3] == 1;
}

static bool moe_cutlass_layouts(const ggml_cuda_moe_mmq_args & args) {
    return ggml_is_contiguous(args.gate_up) && ggml_is_contiguous(args.input) && ggml_is_contiguous_rows(args.ids) &&
           args.ids->nb[1] >= ggml_row_size(args.ids->type, args.ids->ne[0]) && ggml_is_contiguous(args.gate_up_dst) &&
           ggml_is_contiguous(args.gate_up_bias) && ggml_is_contiguous(args.gate_up_biased) &&
           ggml_is_contiguous(args.activation) && ggml_is_contiguous(args.down) && ggml_is_contiguous(args.down_dst) &&
           ggml_is_contiguous(args.down_bias) && ggml_is_contiguous(args.down_biased) &&
           ggml_is_contiguous(args.weights) && ggml_is_contiguous(args.weighted) && ggml_is_contiguous(args.dst);
}

static bool moe_cutlass_buffers(const ggml_cuda_moe_mmq_args & args, ggml_backend_buffer_type_t buffer_type) {
    const std::array<const ggml_tensor *, 14> tensors = {
        args.gate_up,        args.input,      args.ids,      args.gate_up_dst, args.gate_up_bias,
        args.gate_up_biased, args.activation, args.down,     args.down_dst,    args.down_bias,
        args.down_biased,    args.weights,    args.weighted, args.dst,
    };
    return std::all_of(tensors.begin(), tensors.end(), [buffer_type](const ggml_tensor * tensor) {
        return tensor->buffer && ggml_backend_buffer_get_type(tensor->buffer) == buffer_type;
    });
}

static __global__ void moe_cutlass_validate_equal_kernel(
        const uint8_t * lhs, const uint8_t * rhs, size_t size, int * mismatch) {
    for (size_t index = (size_t) blockIdx.x * blockDim.x + threadIdx.x; index < size;
         index += (size_t) blockDim.x * gridDim.x) {
        if (lhs[index] != rhs[index]) {
            atomicExch(mismatch, 1);
            return;
        }
    }
}

static void moe_cutlass_validate_equal(
        ggml_backend_cuda_context & ctx, const void * lhs, const void * rhs, size_t size, const char * name) {
    if (size == 0) {
        return;
    }

    ggml_cuda_pool_alloc<int> mismatch(ctx.pool());
    int * mismatch_data = mismatch.alloc(1);
    CUDA_CHECK(cudaMemsetAsync(mismatch_data, 0, sizeof(int), ctx.stream()));
    constexpr int threads = 256;
    const int blocks = (int) std::min<size_t>((size + threads - 1) / threads, 1024);
    moe_cutlass_validate_equal_kernel<<<blocks, threads, 0, ctx.stream()>>>(
        (const uint8_t *) lhs, (const uint8_t *) rhs, size, mismatch_data);
    CUDA_CHECK(cudaGetLastError());

    int mismatch_host = 0;
    CUDA_CHECK(cudaMemcpyAsync(&mismatch_host, mismatch_data, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    if (mismatch_host != 0) {
        GGML_ABORT("CUTLASS MoE support validation failed for %s", name);
    }
}

static bool moe_cutlass_run(
        ggml_backend_cuda_context &      ctx,
        const ggml_cuda_moe_mmq_args &   args,
        const moe_cutlass_config &        config,
        const ggml_cuda_cutlass_weight & w13_weight,
        const ggml_cuda_cutlass_weight & w2_weight,
        int32_t *                         ids_src1,
        int32_t *                         ids_dst,
        int32_t *                         row_expert,
        const int32_t *                   expert_bounds,
        int                               sm_count,
        int64_t                           ids_stride) {
    const int64_t n_experts     = args.gate_up->ne[2];
    const int64_t n_expert_used = args.ids->ne[0];
    const int64_t n_tokens      = args.ids->ne[1];
    const int64_t n_rows        = n_tokens * n_expert_used;
    const int64_t n_embd        = args.input->ne[0];
    const int64_t n_ff          = args.activation->ne[0];
    const int64_t w13_k         = w13_weight.k;
    const int64_t w2_k          = w2_weight.k;
    cudaStream_t stream         = ctx.stream();

    const ggml_cuda_cutlass_config w13_config =
        moe_cutlass_gemm_config(config.w13, n_rows, 2 * n_ff, n_experts, config.pdl);
    const ggml_cuda_cutlass_config w2_config =
        moe_cutlass_gemm_config(config.w2, n_rows, n_embd, n_experts, config.pdl);

    GGML_ASSERT(w13_k >= n_embd && w13_k % 128 == 0);
    GGML_ASSERT(w2_k >= n_ff && w2_k % 128 == 0);

    ggml_cuda_pool_alloc<uint8_t> w13_input(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> w13_scales(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> w2_input(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> w2_scales(ctx.pool());
    uint8_t * w13_input_data = w13_input.alloc(ggml_cuda_cutlass_activation_size(GGML_TYPE_MXFP4, n_rows, w13_k));
    uint8_t * w13_scale_data =
        w13_scales.alloc(ggml_cuda_moe_cutlass_scale_size(n_rows, n_experts, w13_k, false));
    uint8_t * w2_input_data = w2_input.alloc(ggml_cuda_cutlass_activation_size(GGML_TYPE_MXFP4, n_rows, w2_k));
    uint8_t * w2_scale_data =
        w2_scales.alloc(ggml_cuda_moe_cutlass_scale_size(n_rows, n_experts, w2_k, false));

    const bool cta_quant = config.cta_quant && ggml_cuda_moe_cutlass_quantize_broadcast_cta(
        (const float *) args.input->data, (const int32_t *) args.ids->data, ids_src1, ids_dst, row_expert,
        expert_bounds, w13_input_data, w13_scale_data, n_embd, w13_k, args.input->nb[2] / sizeof(float),
        n_tokens, n_experts, n_expert_used, ids_stride, false, stream);
    if (!cta_quant) {
        ggml_cuda_moe_cutlass_quantize_broadcast(
            (const float *) args.input->data, (const int32_t *) args.ids->data, ids_src1, expert_bounds,
            w13_input_data, w13_scale_data, n_embd, w13_k, args.input->nb[2] / sizeof(float), n_tokens, n_experts,
            n_expert_used, ids_stride, false, stream);
    }
    if (config.validate_support && cta_quant) {
        ggml_cuda_pool_alloc<uint8_t> reference_input(ctx.pool());
        ggml_cuda_pool_alloc<uint8_t> reference_scales(ctx.pool());
        const size_t input_size = ggml_cuda_cutlass_activation_size(GGML_TYPE_MXFP4, n_rows, w13_k);
        const size_t scale_size = ggml_cuda_moe_cutlass_scale_size(n_rows, n_experts, w13_k, false);
        uint8_t * reference_input_data = reference_input.alloc(input_size);
        uint8_t * reference_scale_data = reference_scales.alloc(scale_size);
        ggml_cuda_moe_cutlass_quantize_broadcast(
            (const float *) args.input->data, (const int32_t *) args.ids->data, ids_src1, expert_bounds,
            reference_input_data, reference_scale_data, n_embd, w13_k, args.input->nb[2] / sizeof(float), n_tokens,
            n_experts, n_expert_used, ids_stride, false, stream);
        moe_cutlass_validate_equal(ctx, w13_input_data, reference_input_data, input_size, "input activation");
        moe_cutlass_validate_equal(ctx, w13_scale_data, reference_scale_data, scale_size, "input scales");
    }

    ggml_cuda_cutlass_weight_wait_ready(w13_weight, stream);
    if (!ggml_cuda_cutlass_grouped_gemm(
            ctx, w13_weight, w13_input_data, w13_scale_data, expert_bounds, row_expert, args.gate_up_dst->data,
            n_experts, n_rows, 2 * n_ff, w13_k, sm_count, w13_config, stream, true)) {
        GGML_ABORT("required CUTLASS W13 launch failed");
    }

    const bool cta_activation = config.cta_activation && ggml_cuda_moe_cutlass_w13_epilogue_cta(
        args.gate_up_dst->data, (const float *) args.gate_up_bias->data, (const int32_t *) args.ids->data, ids_dst,
        row_expert, expert_bounds, w2_input_data, w2_scale_data, n_ff, w2_k, n_rows, n_experts, n_expert_used,
        config.activation_rows, ids_stride, false, stream);
    if (!cta_activation) {
        ggml_cuda_moe_cutlass_w13_epilogue(
            args.gate_up_dst->data, (const float *) args.gate_up_bias->data, (const int32_t *) args.ids->data, ids_dst,
            expert_bounds, w2_input_data, w2_scale_data, n_ff, w2_k, n_rows, n_experts, n_expert_used, ids_stride,
            false, stream);
    }
    if (config.validate_support && cta_activation) {
        ggml_cuda_pool_alloc<uint8_t> reference_input(ctx.pool());
        ggml_cuda_pool_alloc<uint8_t> reference_scales(ctx.pool());
        const size_t input_size = ggml_cuda_cutlass_activation_size(GGML_TYPE_MXFP4, n_rows, w2_k);
        const size_t scale_size = ggml_cuda_moe_cutlass_scale_size(n_rows, n_experts, w2_k, false);
        uint8_t * reference_input_data = reference_input.alloc(input_size);
        uint8_t * reference_scale_data = reference_scales.alloc(scale_size);
        ggml_cuda_moe_cutlass_w13_epilogue(
            args.gate_up_dst->data, (const float *) args.gate_up_bias->data, (const int32_t *) args.ids->data, ids_dst,
            expert_bounds, reference_input_data, reference_scale_data, n_ff, w2_k, n_rows, n_experts, n_expert_used,
            ids_stride, false, stream);
        moe_cutlass_validate_equal(ctx, w2_input_data, reference_input_data, input_size, "W13 activation");
        moe_cutlass_validate_equal(ctx, w2_scale_data, reference_scale_data, scale_size, "W13 activation scales");
    }

    ggml_cuda_cutlass_weight_wait_ready(w2_weight, stream);
    if (!ggml_cuda_cutlass_grouped_gemm(
            ctx, w2_weight, w2_input_data, w2_scale_data, expert_bounds, row_expert, args.down_dst->data,
            n_experts, n_rows, n_embd, w2_k, sm_count, w2_config, stream, true)) {
        GGML_ABORT("required CUTLASS W2 launch failed");
    }

    ggml_cuda_moe_cutlass_w2_finalize(
        args.down_dst->data, (const float *) args.down_bias->data, (const float *) args.weights->data,
        (const int32_t *) args.ids->data, ids_src1, (float *) args.dst->data, n_embd, n_tokens, n_expert_used,
        ids_stride, stream);
    return true;
}

bool ggml_cuda_moe_cutlass_prefill_requested() {
    return ggml_cuda_cutlass_compiled() && !moe_cutlass_env_flag("GGML_CUDA_CUTLASS_DISABLE") &&
        !moe_cutlass_get_config().disabled;
}

bool ggml_cuda_moe_mmq(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_mmq_args & args) {
    const moe_cutlass_config & config = moe_cutlass_get_config();
    const auto & device_info = ggml_cuda_info().devices[ctx.device];
    const int64_t n_tokens = args.ids->ne[1];

    ggml_cuda_cutlass_weight cached_w13;
    ggml_cuda_cutlass_weight cached_w2;
    const bool inplace_weights =
        ggml_cuda_cutlass_get_inplace_weight(ctx, args.gate_up, cached_w13) &&
        ggml_cuda_cutlass_get_inplace_weight(ctx, args.down, cached_w2);
    const int mmvq_max_batch = get_mmvq_mmid_max_batch(args.gate_up->type, device_info.cc);

    if (!ggml_cuda_moe_cutlass_prefill_requested() || !blackwell_mma_available(device_info.cc) ||
        n_tokens <= mmvq_max_batch || (n_tokens < 256 && !inplace_weights) || n_tokens >= (1 << 22) ||
        (size_t) n_tokens > device_info.smpbo / sizeof(uint32_t)) {
        return false;
    }
    if (!moe_cutlass_shapes(args) || !moe_cutlass_layouts(args) ||
        !moe_cutlass_buffers(args, ggml_backend_cuda_buffer_type(ctx.device))) {
        return false;
    }
#ifdef USE_CUDA_GRAPH
    if (ctx.any_cuda_graph_enabled()) {
        if (inplace_weights) {
            GGML_ABORT("in-place CUTLASS MoE weights cannot fall back while CUDA Graphs are enabled");
        }
        return false;
    }
#endif

    const int64_t n_experts     = args.gate_up->ne[2];
    const int64_t n_expert_used = args.ids->ne[0];
    const int64_t n_rows        = n_expert_used * n_tokens;
    const int64_t ids_stride    = args.ids->nb[1] / ggml_element_size(args.ids);

    if (config.log_config) {
        static bool logged = false;
        if (!logged) {
            GGML_LOG_INFO(
                "CUTLASS MoE: prefix=%d cta-quant=%d cta-activation=%d pdl=%d inplace-weights=%d activation-rows=%d "
                "w13-tile-n=%d w13-swap=%d w2-tile-n=%d w2-swap=%d\n",
                config.prefix_schedule, config.cta_quant, config.cta_activation, config.pdl,
                config.inplace_weights, config.activation_rows, config.w13.tile_n, config.w13.swap_ab, config.w2.tile_n,
                config.w2.swap_ab);
            logged = true;
        }
    }

    if (ctx.cutlass_weight_stream == nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.cutlass_weight_stream, cudaStreamNonBlocking));
    }

    ggml_cuda_cutlass_weight w13_weight = cached_w13;
    ggml_cuda_cutlass_weight w2_weight  = cached_w2;
    if (!inplace_weights) {
        if (!ggml_cuda_cutlass_repack_weight(
                ctx, args.gate_up, w13_weight, ctx.cutlass_weight_stream, false, !config.inplace_weights)) {
            return false;
        }
        if (!ggml_cuda_cutlass_repack_weight(
                ctx, args.down, w2_weight, ctx.cutlass_weight_stream, false, !config.inplace_weights)) {
            if (config.inplace_weights) {
                GGML_ABORT("failed to prepare W2 after replacing the W13 weight layout");
            }
            return false;
        }
    }

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), n_experts + 1);
    ggml_cuda_pool_alloc<int32_t> row_expert(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> prefix_block_counts(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> prefix_block_offsets(ctx.pool());

    bool prefix_plan = false;
    if (config.prefix_schedule) {
        const int n_blocks = ggml_cuda_mm_ids_prefix_block_count((int) n_tokens, (int) n_expert_used);
        prefix_block_counts.alloc((size_t) n_blocks * n_experts);
        prefix_block_offsets.alloc((size_t) n_blocks * n_experts);
        prefix_plan = ggml_cuda_launch_mm_ids_prefix(
            (const int32_t *) args.ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(), row_expert.get(),
            prefix_block_counts.get(), prefix_block_offsets.get(), (int) n_experts, (int) n_tokens,
            (int) n_expert_used, (int) ids_stride, ctx.stream());
    }
    if (!prefix_plan) {
        ggml_cuda_launch_mm_ids_helper(
            (const int32_t *) args.ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(), (int) n_experts,
            (int) n_tokens, (int) n_expert_used, (int) args.input->ne[1], (int) ids_stride,
            (int) (args.input->nb[2] / args.input->nb[1]), true, ctx.stream());
    }

    return moe_cutlass_run(
        ctx, args, config, w13_weight, w2_weight, ids_src1.get(), ids_dst.get(),
        prefix_plan ? row_expert.get() : nullptr, expert_bounds.get(), device_info.nsm, ids_stride);
}
