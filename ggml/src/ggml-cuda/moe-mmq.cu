#include "mmid.cuh"
#include "mmq.cuh"
#include "moe-mmq-epilogues.cuh"
#include "moe-mmq-mxfp8.cuh"
#include "moe-mmq-tma.cuh"
#include "moe-mmq.cuh"
#ifdef GGML_CUDA_MOE_PROFILE
#    include "moe-profile.cuh"
#endif

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <vector>

enum class moe_mmq_w13_epilogue_mode {
    staged,
    fused,
    fused_quant,
    tma_epilogue,
};

enum class moe_mmq_w2_epilogue_mode {
    staged,
    fused,
    tma_weighted,
    tma_atomic,
};

struct moe_mmq_stage_config {
    bool persistent;
    int  tile_rows;
    int  cta_multiplier;
    bool output_tile_major;
};

struct moe_mmq_config {
    bool                        disabled;
    bool                        shared_plan;
    bool                        padded_shapes;
    bool                        log_config;
    bool                        log_distribution;
    bool                        use_cp_async;
    bool                        use_weight_pipeline;
    bool                        async_repack;
    bool                        tma_warp_specialized;
    bool                        tma_require;
    bool                        tma_tail_elide;
    int                         repack_cache_entries;
    int                         activation_format;
    moe_mmq_w13_epilogue_mode   w13_epilogue;
    moe_mmq_w2_epilogue_mode    w2_epilogue;
    ggml_cuda_moe_weight_layout weight_layout;
    moe_mmq_stage_config        w13;
    moe_mmq_stage_config        w2;
};

static bool moe_mmq_is_tma_layout(ggml_cuda_moe_weight_layout layout) {
    return layout == ggml_cuda_moe_weight_layout::tma || layout == ggml_cuda_moe_weight_layout::tma_inplace;
}

static bool moe_mmq_env_flag(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && std::atoi(value) != 0;
}

static int moe_mmq_env_int(const char * name, int default_value, int min_value, int max_value) {
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

static int moe_mmq_env_tile_rows(const char * name) {
    const int value = moe_mmq_env_int(name, 0, 0, 128);
    if (value != 0 && value != 32 && value != 64 && value != 128) {
        GGML_ABORT("%s must be 0, 32, 64, or 128", name);
    }
    return value;
}

static moe_mmq_w13_epilogue_mode moe_mmq_env_w13_epilogue() {
    const char * value = std::getenv("GGML_CUDA_MOE_MMQ_W13_EPILOGUE");
    if (value == nullptr || std::strcmp(value, "fused") == 0) {
        return moe_mmq_w13_epilogue_mode::fused;
    }
    if (std::strcmp(value, "staged") == 0) {
        return moe_mmq_w13_epilogue_mode::staged;
    }
    if (std::strcmp(value, "fused-quant") == 0) {
        return moe_mmq_w13_epilogue_mode::fused_quant;
    }
    if (std::strcmp(value, "tma-epilogue") == 0) {
        return moe_mmq_w13_epilogue_mode::tma_epilogue;
    }
    GGML_ABORT("GGML_CUDA_MOE_MMQ_W13_EPILOGUE must be staged, fused, fused-quant, or tma-epilogue");
    return moe_mmq_w13_epilogue_mode::fused;
}

static int moe_mmq_env_activation_format() {
    const char * value = std::getenv("GGML_CUDA_MOE_MMQ_ACTIVATION_FORMAT");
    if (value == nullptr || std::strcmp(value, "mxfp4") == 0) {
        return GGML_CUDA_MOE_ACTIVATION_MXFP4;
    }
    if (std::strcmp(value, "mxfp8") == 0) {
        return GGML_CUDA_MOE_ACTIVATION_MXFP8;
    }
    GGML_ABORT("GGML_CUDA_MOE_MMQ_ACTIVATION_FORMAT must be mxfp4 or mxfp8");
    return GGML_CUDA_MOE_ACTIVATION_MXFP4;
}

static moe_mmq_w2_epilogue_mode moe_mmq_env_w2_epilogue() {
    const char * value = std::getenv("GGML_CUDA_MOE_MMQ_W2_EPILOGUE");
    if (value == nullptr || std::strcmp(value, "fused") == 0) {
        return moe_mmq_w2_epilogue_mode::fused;
    }
    if (std::strcmp(value, "staged") == 0) {
        return moe_mmq_w2_epilogue_mode::staged;
    }
    if (std::strcmp(value, "tma-weighted") == 0) {
        return moe_mmq_w2_epilogue_mode::tma_weighted;
    }
    if (std::strcmp(value, "tma-atomic") == 0) {
        return moe_mmq_w2_epilogue_mode::tma_atomic;
    }
    GGML_ABORT("GGML_CUDA_MOE_MMQ_W2_EPILOGUE must be staged, fused, tma-weighted, or tma-atomic");
    return moe_mmq_w2_epilogue_mode::fused;
}

static ggml_cuda_moe_weight_layout moe_mmq_env_weight_layout() {
    const char * value = std::getenv("GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT");
    if (value == nullptr || std::strcmp(value, "canonical") == 0) {
        return ggml_cuda_moe_weight_layout::canonical;
    }
    if (std::strcmp(value, "interleaved") == 0) {
        return ggml_cuda_moe_weight_layout::interleaved;
    }
    if (std::strcmp(value, "split") == 0) {
        return ggml_cuda_moe_weight_layout::split;
    }
    if (std::strcmp(value, "tma") == 0) {
        return ggml_cuda_moe_weight_layout::tma;
    }
    if (std::strcmp(value, "tma-inplace") == 0) {
        return ggml_cuda_moe_weight_layout::tma_inplace;
    }
    GGML_ABORT("GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT must be canonical, interleaved, split, tma, or tma-inplace");
    return ggml_cuda_moe_weight_layout::canonical;
}

static const moe_mmq_config & moe_mmq_get_config() {
    static const moe_mmq_config config = [] {
        const bool                 persistent_disabled = moe_mmq_env_flag("GGML_CUDA_MOE_MMQ_PERSISTENT_DISABLE");
        const moe_mmq_stage_config w13                 = {
            !persistent_disabled && !moe_mmq_env_flag("GGML_CUDA_MOE_MMQ_W13_PERSISTENT_DISABLE"),
            moe_mmq_env_tile_rows("GGML_CUDA_MOE_MMQ_W13_TILE_ROWS"),
            moe_mmq_env_int("GGML_CUDA_MOE_MMQ_W13_CTA_MULTIPLIER", 1, 1, 8),
            moe_mmq_env_flag("GGML_CUDA_MOE_MMQ_W13_OUTPUT_TILE_MAJOR"),
        };
        const moe_mmq_stage_config w2 = {
            !persistent_disabled && !moe_mmq_env_flag("GGML_CUDA_MOE_MMQ_W2_PERSISTENT_DISABLE"),
            moe_mmq_env_tile_rows("GGML_CUDA_MOE_MMQ_W2_TILE_ROWS"),
            moe_mmq_env_int("GGML_CUDA_MOE_MMQ_W2_CTA_MULTIPLIER", 1, 1, 8),
            moe_mmq_env_flag("GGML_CUDA_MOE_MMQ_W2_OUTPUT_TILE_MAJOR"),
        };
        moe_mmq_config result = {
            moe_mmq_env_flag("GGML_CUDA_MOE_MMQ_DISABLE"),
            !moe_mmq_env_flag("GGML_CUDA_MOE_MMQ_SHARED_PLAN_DISABLE"),
            moe_mmq_env_flag("GGML_CUDA_MOE_MMQ_PADDED_TEST"),
            moe_mmq_env_flag("GGML_CUDA_MOE_MMQ_LOG_CONFIG"),
            moe_mmq_env_flag("GGML_CUDA_MOE_MMQ_LOG_DISTRIBUTION"),
            moe_mmq_env_flag("GGML_CUDA_MOE_MMQ_CP_ASYNC"),
            moe_mmq_env_flag("GGML_CUDA_MOE_MMQ_WEIGHT_PIPELINE"),
            moe_mmq_env_flag("GGML_CUDA_MOE_MMQ_REPACK_ASYNC"),
            moe_mmq_env_flag("GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED"),
            moe_mmq_env_flag("GGML_CUDA_MOE_MMQ_TMA_REQUIRE"),
            !moe_mmq_env_flag("GGML_CUDA_MOE_MMQ_TMA_TAIL_DISABLE"),
            moe_mmq_env_int("GGML_CUDA_MOE_MMQ_REPACK_CACHE_ENTRIES", 2, 2, 16),
            moe_mmq_env_activation_format(),
            moe_mmq_env_w13_epilogue(),
            moe_mmq_env_w2_epilogue(),
            moe_mmq_env_weight_layout(),
            w13,
            w2,
        };
        if (!result.shared_plan) {
            result.w13.persistent = false;
            result.w2.persistent  = false;
        }
        if (result.use_weight_pipeline) {
            if (result.weight_layout != ggml_cuda_moe_weight_layout::split) {
                GGML_ABORT("GGML_CUDA_MOE_MMQ_WEIGHT_PIPELINE requires split weights");
            }
            result.use_cp_async = true;
        }
        if (result.tma_warp_specialized && !moe_mmq_is_tma_layout(result.weight_layout)) {
            GGML_ABORT("GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED requires tma weights");
        }
        if (result.tma_require && !moe_mmq_is_tma_layout(result.weight_layout)) {
            GGML_ABORT("GGML_CUDA_MOE_MMQ_TMA_REQUIRE requires tma weights");
        }
        if (result.weight_layout == ggml_cuda_moe_weight_layout::tma_inplace && !result.tma_require) {
            GGML_ABORT("tma-inplace weights require GGML_CUDA_MOE_MMQ_TMA_REQUIRE");
        }
        if (result.w13_epilogue == moe_mmq_w13_epilogue_mode::tma_epilogue &&
            !moe_mmq_is_tma_layout(result.weight_layout)) {
            GGML_ABORT("the TMA W13 epilogue requires TMA weights");
        }
        if ((result.w2_epilogue == moe_mmq_w2_epilogue_mode::tma_weighted ||
             result.w2_epilogue == moe_mmq_w2_epilogue_mode::tma_atomic) &&
            !moe_mmq_is_tma_layout(result.weight_layout)) {
            GGML_ABORT("TMA W2 epilogues require TMA weights");
        }
        if (result.activation_format == GGML_CUDA_MOE_ACTIVATION_MXFP8 &&
            (!moe_mmq_is_tma_layout(result.weight_layout) ||
             result.w13_epilogue != moe_mmq_w13_epilogue_mode::tma_epilogue)) {
            GGML_ABORT("MXFP8 activations require TMA weights and the TMA W13 epilogue");
        }
        return result;
    }();
    return config;
}

static int moe_mmq_tile_rows(const moe_mmq_stage_config & config, int64_t n_tokens) {
    if (config.tile_rows != 0) {
        return config.tile_rows;
    }
    return n_tokens <= 512 ? 32 : n_tokens <= 2048 ? 64 : 128;
}

template <int J>
static void moe_mmq_build_tile_offsets(const int32_t * expert_bounds,
                                       int32_t *       tile_offsets,
                                       int             n_experts,
                                       cudaStream_t    stream) {
    build_moe_mmq_tile_offsets<J><<<1, 1, 0, stream>>>(expert_bounds, tile_offsets, n_experts);
    CUDA_CHECK(cudaGetLastError());
}

static void moe_mmq_build_tile_offsets(int             tile_rows,
                                       const int32_t * expert_bounds,
                                       int32_t *       tile_offsets,
                                       int             n_experts,
                                       cudaStream_t    stream) {
    if (tile_rows == 32) {
        moe_mmq_build_tile_offsets<32>(expert_bounds, tile_offsets, n_experts, stream);
    } else if (tile_rows == 64) {
        moe_mmq_build_tile_offsets<64>(expert_bounds, tile_offsets, n_experts, stream);
    } else {
        GGML_ASSERT(tile_rows == 128);
        moe_mmq_build_tile_offsets<128>(expert_bounds, tile_offsets, n_experts, stream);
    }
}

static bool ggml_cuda_moe_mmq_shapes(const ggml_cuda_moe_mmq_args & args, const moe_mmq_config & config) {
    constexpr int64_t n_expert      = 128;
    constexpr int64_t n_expert_used = 4;

    const int64_t n_embd       = args.input->ne[0];
    const int64_t n_ff         = args.activation->ne[0];
    const int64_t n_tokens     = args.ids->ne[1];
    const bool    native_shape = n_embd == 2880 && n_ff == 2880;
    const bool    padded_shape = config.padded_shapes && n_embd == 2944 && n_ff == 2944;

    return (native_shape || padded_shape) && args.gate_up->type == GGML_TYPE_MXFP4 &&
           args.down->type == GGML_TYPE_MXFP4 && args.input->type == GGML_TYPE_F32 && args.ids->type == GGML_TYPE_I32 &&
           args.gate_up_dst->type == GGML_TYPE_F32 && args.gate_up_bias->type == GGML_TYPE_F32 &&
           args.gate_up_biased->type == GGML_TYPE_F32 && args.activation->type == GGML_TYPE_F32 &&
           args.down_dst->type == GGML_TYPE_F32 && args.down_bias->type == GGML_TYPE_F32 &&
           args.down_biased->type == GGML_TYPE_F32 && args.weights->type == GGML_TYPE_F32 &&
           args.weighted->type == GGML_TYPE_F32 && args.dst->type == GGML_TYPE_F32 && args.gate_up->ne[0] == n_embd &&
           args.gate_up->ne[1] == 2 * n_ff && args.gate_up->ne[2] == n_expert && args.gate_up->ne[3] == 1 &&
           args.input->ne[0] == n_embd && args.input->ne[1] == 1 && args.input->ne[2] == n_tokens &&
           args.input->ne[3] == 1 && args.ids->ne[0] == n_expert_used && args.ids->ne[2] == 1 && args.ids->ne[3] == 1 &&
           args.gate_up_dst->ne[0] == 2 * n_ff && args.gate_up_dst->ne[1] == n_expert_used &&
           args.gate_up_dst->ne[2] == n_tokens && args.gate_up_dst->ne[3] == 1 &&
           args.gate_up_bias->ne[0] == 2 * n_ff && args.gate_up_bias->ne[1] == n_expert &&
           args.gate_up_bias->ne[2] == 1 && args.gate_up_bias->ne[3] == 1 &&
           ggml_are_same_shape(args.gate_up_biased, args.gate_up_dst) && args.activation->ne[0] == n_ff &&
           args.activation->ne[1] == n_expert_used && args.activation->ne[2] == n_tokens &&
           args.activation->ne[3] == 1 && args.down->ne[0] == n_ff && args.down->ne[1] == n_embd &&
           args.down->ne[2] == n_expert && args.down->ne[3] == 1 && args.down_dst->ne[0] == n_embd &&
           args.down_dst->ne[1] == n_expert_used && args.down_dst->ne[2] == n_tokens && args.down_dst->ne[3] == 1 &&
           args.down_bias->ne[0] == n_embd && args.down_bias->ne[1] == n_expert && args.down_bias->ne[2] == 1 &&
           args.down_bias->ne[3] == 1 && ggml_are_same_shape(args.down_biased, args.down_dst) &&
           args.weights->ne[0] == 1 && args.weights->ne[1] == n_expert_used && args.weights->ne[2] == n_tokens &&
           args.weights->ne[3] == 1 && ggml_are_same_shape(args.weighted, args.down_dst) && args.dst->ne[0] == n_embd &&
           args.dst->ne[1] == n_tokens && args.dst->ne[2] == 1 && args.dst->ne[3] == 1;
}

static bool ggml_cuda_moe_mmq_layouts(const ggml_cuda_moe_mmq_args & args) {
    return ggml_is_contiguous(args.gate_up) && ggml_is_contiguous(args.input) && ggml_is_contiguous_rows(args.ids) &&
           args.ids->nb[1] >= ggml_row_size(args.ids->type, args.ids->ne[0]) && ggml_is_contiguous(args.gate_up_dst) &&
           ggml_is_contiguous(args.gate_up_bias) && ggml_is_contiguous(args.gate_up_biased) &&
           ggml_is_contiguous(args.activation) && ggml_is_contiguous(args.down) && ggml_is_contiguous(args.down_dst) &&
           ggml_is_contiguous(args.down_bias) && ggml_is_contiguous(args.down_biased) &&
           ggml_is_contiguous(args.weights) && ggml_is_contiguous(args.weighted) && ggml_is_contiguous(args.dst);
}

static bool ggml_cuda_moe_mmq_buffers(const ggml_cuda_moe_mmq_args & args, ggml_backend_buffer_type_t buffer_type) {
    const std::array<const ggml_tensor *, 14> tensors = {
        args.gate_up,        args.input,      args.ids,      args.gate_up_dst, args.gate_up_bias,
        args.gate_up_biased, args.activation, args.down,     args.down_dst,    args.down_bias,
        args.down_biased,    args.weights,    args.weighted, args.dst,
    };
    return std::all_of(tensors.begin(), tensors.end(), [buffer_type](const ggml_tensor * tensor) {
        return tensor->buffer && ggml_backend_buffer_get_type(tensor->buffer) == buffer_type;
    });
}

static const char * moe_mmq_w13_epilogue_name(moe_mmq_w13_epilogue_mode mode) {
    switch (mode) {
        case moe_mmq_w13_epilogue_mode::staged:
            return "staged";
        case moe_mmq_w13_epilogue_mode::fused:
            return "fused";
        case moe_mmq_w13_epilogue_mode::fused_quant:
            return "fused-quant";
        case moe_mmq_w13_epilogue_mode::tma_epilogue:
            return "tma-epilogue";
    }
    GGML_ABORT("invalid MoE W13 epilogue mode");
    return "invalid";
}

static const char * moe_mmq_w2_epilogue_name(moe_mmq_w2_epilogue_mode mode) {
    switch (mode) {
        case moe_mmq_w2_epilogue_mode::staged:
            return "staged";
        case moe_mmq_w2_epilogue_mode::fused:
            return "fused";
        case moe_mmq_w2_epilogue_mode::tma_weighted:
            return "tma-weighted";
        case moe_mmq_w2_epilogue_mode::tma_atomic:
            return "tma-atomic";
    }
    GGML_ABORT("invalid MoE W2 epilogue mode");
    return "invalid";
}

static const char * moe_mmq_weight_layout_name(ggml_cuda_moe_weight_layout layout) {
    switch (layout) {
        case ggml_cuda_moe_weight_layout::canonical:
            return "canonical";
        case ggml_cuda_moe_weight_layout::interleaved:
            return "interleaved";
        case ggml_cuda_moe_weight_layout::split:
            return "split";
        case ggml_cuda_moe_weight_layout::tma:
            return "tma";
        case ggml_cuda_moe_weight_layout::tma_inplace:
            return "tma-inplace";
    }
    GGML_ABORT("invalid MoE weight layout");
    return "invalid";
}

static double moe_mmq_tile_fill(const std::vector<int32_t> & bounds, int n_experts, int tile_rows, int64_t n_rows) {
    int64_t scheduled_rows = 0;
    for (int i = 0; i < n_experts; ++i) {
        const int64_t n = bounds[i + 1] - bounds[i];
        scheduled_rows += GGML_PAD(n, tile_rows);
    }
    return scheduled_rows == 0 ? 1.0 : (double) n_rows / scheduled_rows;
}

static void moe_mmq_log_distribution(const int32_t * expert_bounds,
                                     int             n_experts,
                                     int64_t         n_rows,
                                     int             w13_tile_rows,
                                     int             w2_tile_rows,
                                     cudaStream_t    stream) {
    std::vector<int32_t> bounds(n_experts + 1);
    CUDA_CHECK(
        cudaMemcpyAsync(bounds.data(), expert_bounds, bounds.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int n_active = 0;
    int n_min    = INT_MAX;
    int n_max    = 0;
    for (int i = 0; i < n_experts; ++i) {
        const int n = bounds[i + 1] - bounds[i];
        if (n != 0) {
            ++n_active;
            n_min = std::min(n_min, n);
            n_max = std::max(n_max, n);
        }
    }

    GGML_LOG_INFO(
        "MoE MMQ distribution: rows=%lld active=%d/%d min=%d max=%d "
        "w13-tile-fill=%.4f w2-tile-fill=%.4f\n",
        (long long) n_rows, n_active, n_experts, n_active == 0 ? 0 : n_min, n_max,
        moe_mmq_tile_fill(bounds, n_experts, w13_tile_rows, n_rows),
        moe_mmq_tile_fill(bounds, n_experts, w2_tile_rows, n_rows));
}

bool ggml_cuda_moe_mmq(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_mmq_args & args) {
    const moe_mmq_config & config      = moe_mmq_get_config();
    const auto &           device_info = ggml_cuda_info().devices[ctx.device];
    const int              cc          = device_info.cc;
    const int64_t          n_tokens    = args.ids->ne[1];

    const bool inplace_layout = config.weight_layout == ggml_cuda_moe_weight_layout::tma_inplace;
    if (config.disabled || !blackwell_mma_available(cc) || (!inplace_layout && n_tokens < 256) ||
        n_tokens >= (1 << 22) || (size_t) n_tokens > device_info.smpbo / sizeof(uint32_t)) {
        return false;
    }

    if (!ggml_cuda_moe_mmq_shapes(args, config) || !ggml_cuda_moe_mmq_layouts(args) ||
        !ggml_cuda_moe_mmq_buffers(args, ggml_backend_cuda_buffer_type(ctx.device)) ||
        (!config.shared_plan && (config.w13_epilogue == moe_mmq_w13_epilogue_mode::fused_quant ||
                                 config.w13_epilogue == moe_mmq_w13_epilogue_mode::tma_epilogue))) {
        if (config.tma_require) {
            GGML_ABORT("required TMA MoE MMQ path is not supported by this graph");
        }
        return false;
    }
#ifdef USE_CUDA_GRAPH
    if (config.weight_layout != ggml_cuda_moe_weight_layout::canonical && ctx.any_cuda_graph_enabled()) {
        if (config.tma_require) {
            GGML_ABORT("repacked MoE weights are not compatible with CUDA graph capture");
        }
        return false;
    }
#endif

    const int64_t n_experts     = args.gate_up->ne[2];
    const int64_t n_expert_used = args.ids->ne[0];
    const int64_t n_rows        = n_expert_used * n_tokens;
    const int64_t ids_stride    = args.ids->nb[1] / ggml_element_size(args.ids);
    const int     w13_tile_rows = moe_mmq_tile_rows(config.w13, n_tokens);
    const int     w2_tile_rows  = moe_mmq_tile_rows(config.w2, n_tokens);

    if (config.log_config) {
        static bool logged = false;
        if (!logged) {
            GGML_LOG_INFO(
                "MoE MMQ: plan=%d w13-epilogue=%s w2-epilogue=%s weights=%s cp-async=%d weight-pipeline=%d "
                "async-repack=%d repack-cache=%d activation-format=%s tma-warp=%d tma-require=%d tma-tail=%d "
                "w13={persistent=%d,tile=%d,cta=%d,output-major=%d} "
                "w2={persistent=%d,tile=%d,cta=%d,output-major=%d} padded=%d\n",
                config.shared_plan, moe_mmq_w13_epilogue_name(config.w13_epilogue),
                moe_mmq_w2_epilogue_name(config.w2_epilogue),
                moe_mmq_weight_layout_name(config.weight_layout), config.use_cp_async, config.use_weight_pipeline,
                config.async_repack, config.repack_cache_entries,
                config.activation_format == GGML_CUDA_MOE_ACTIVATION_MXFP8 ? "mxfp8" : "mxfp4",
                config.tma_warp_specialized, config.tma_require, config.tma_tail_elide, config.w13.persistent,
                w13_tile_rows, config.w13.cta_multiplier, config.w13.output_tile_major, config.w2.persistent,
                w2_tile_rows, config.w2.cta_multiplier, config.w2.output_tile_major, config.padded_shapes);
            logged = true;
        }
    }

    ggml_cuda_moe_weight_view         w13_weight;
    ggml_cuda_moe_weight_view         w2_weight;
    const ggml_cuda_moe_weight_layout w13_layout =
        config.w13.persistent ? config.weight_layout : ggml_cuda_moe_weight_layout::canonical;
    const ggml_cuda_moe_weight_layout w2_layout =
        config.w2.persistent ? config.weight_layout : ggml_cuda_moe_weight_layout::canonical;
    cudaStream_t repack_stream = ctx.stream();
    if (config.async_repack &&
        (w13_layout != ggml_cuda_moe_weight_layout::canonical || w2_layout != ggml_cuda_moe_weight_layout::canonical)) {
        if (ctx.moe_weight_stream == nullptr) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.moe_weight_stream, cudaStreamNonBlocking));
        }
        repack_stream = ctx.moe_weight_stream;
    }
    {
#ifdef GGML_CUDA_MOE_PROFILE
        const ggml_cuda_moe_profile_scope profile_scope("ffn_moe.weight_repack");
#endif
        if (!ggml_cuda_moe_repack_weight(ctx, args.gate_up, w13_layout, w13_weight, repack_stream,
                                         config.repack_cache_entries, !config.async_repack) ||
            !ggml_cuda_moe_repack_weight(ctx, args.down, w2_layout, w2_weight, repack_stream,
                                         config.repack_cache_entries, !config.async_repack)) {
            if (config.tma_require) {
                GGML_ABORT("required TMA MoE weight repack is not available");
            }
            return false;
        }
    }

    if (moe_mmq_is_tma_layout(config.weight_layout) &&
        (!config.shared_plan || !config.w13.persistent || !config.w2.persistent ||
         !ggml_cuda_moe_mmq_tma_supported(w13_weight, w13_tile_rows, config.tma_warp_specialized, device_info.smpbo,
                                          config.w13_epilogue == moe_mmq_w13_epilogue_mode::tma_epilogue ?
                                              GGML_CUDA_MOE_MMQ_EPILOGUE_W13 :
                                              GGML_CUDA_MOE_MMQ_EPILOGUE_NONE) ||
         !ggml_cuda_moe_mmq_tma_supported(w2_weight, w2_tile_rows, config.tma_warp_specialized, device_info.smpbo))) {
        if (config.tma_require) {
            GGML_ABORT("required TMA MoE MMQ launch does not fit this device or shape");
        }
        return false;
    }

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool());
    if (config.shared_plan) {
        ids_src1.alloc(n_rows);
        ids_dst.alloc(n_rows);
        expert_bounds.alloc(n_experts + 1);
#ifdef GGML_CUDA_MOE_PROFILE
        const ggml_cuda_moe_profile_scope profile_scope("ffn_moe.shared_ids_helper");
#endif
        const int si1  = args.ids->nb[1] / ggml_element_size(args.ids);
        const int sis1 = args.input->nb[2] / args.input->nb[1];
        ggml_cuda_launch_mm_ids_helper((const int32_t *) args.ids->data, ids_src1.get(), ids_dst.get(),
                                       expert_bounds.get(), n_experts, n_tokens, n_expert_used, args.input->ne[1], si1,
                                       sis1,
                                       /*write_inverse=*/true, ctx.stream());
        CUDA_CHECK(cudaGetLastError());
        if (config.log_distribution) {
            moe_mmq_log_distribution(expert_bounds.get(), n_experts, n_rows, w13_tile_rows, w2_tile_rows, ctx.stream());
        }
    }

    ggml_cuda_pool_alloc<int32_t> w13_tile_offsets(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> w2_tile_offsets(ctx.pool());
    const int32_t *               w13_offsets = nullptr;
    const int32_t *               w2_offsets  = nullptr;
    if (config.w13.persistent) {
        w13_offsets = w13_tile_offsets.alloc(n_experts + 1);
        moe_mmq_build_tile_offsets(w13_tile_rows, expert_bounds.get(), w13_tile_offsets.get(), n_experts, ctx.stream());
    }
    if (config.w2.persistent) {
        if (config.w13.persistent && w13_tile_rows == w2_tile_rows) {
            w2_offsets = w13_offsets;
        } else {
            w2_offsets = w2_tile_offsets.alloc(n_experts + 1);
            moe_mmq_build_tile_offsets(w2_tile_rows, expert_bounds.get(), w2_tile_offsets.get(), n_experts,
                                       ctx.stream());
        }
    }

    ggml_cuda_pool_alloc<char> activation_q(ctx.pool());
    const int64_t              activation_q_ne0 = GGML_PAD(args.activation->ne[0], MATRIX_ROW_PADDING);
    const int *                activation_q_ptr = nullptr;
    if (config.w13_epilogue == moe_mmq_w13_epilogue_mode::fused_quant ||
        config.w13_epilogue == moe_mmq_w13_epilogue_mode::tma_epilogue) {
        const bool   fallback = args.down->ne[1] % 128 != 0;
        const int    j_max    = ggml_cuda_mmq_get_J_max(GGML_TYPE_MXFP4, fallback, cc, n_expert_used);
        const size_t activation_q_size =
            config.activation_format == GGML_CUDA_MOE_ACTIVATION_MXFP8 ?
                ggml_cuda_moe_mxfp8_size(n_rows, activation_q_ne0) + j_max * sizeof(block_mxfp8_mmq) :
                n_rows * activation_q_ne0 * sizeof(block_fp4_mmq) / QK_FP4_MMQ + j_max * sizeof(block_fp4_mmq);
        activation_q_ptr = (const int *) activation_q.alloc(activation_q_size);
    }

    ggml_cuda_pool_alloc<char> input_mxfp8(ctx.pool());
    const int *                input_mxfp8_ptr = nullptr;
    if (config.activation_format == GGML_CUDA_MOE_ACTIVATION_MXFP8) {
        const int64_t input_ne0 = GGML_PAD(args.input->ne[0], MATRIX_ROW_PADDING);
        const bool    fallback  = args.gate_up->ne[1] % 128 != 0;
        const int     j_max     = ggml_cuda_mmq_get_J_max(GGML_TYPE_MXFP4, fallback, cc, n_expert_used);
        input_mxfp8_ptr         = (const int *) input_mxfp8.alloc(ggml_cuda_moe_mxfp8_size(n_rows, input_ne0) +
                                                                  j_max * sizeof(block_mxfp8_mmq));
#ifdef GGML_CUDA_MOE_PROFILE
        const ggml_cuda_moe_profile_scope profile_scope("ffn_moe.activation_quant_mxfp8");
#endif
        ggml_cuda_moe_quantize_scatter_mxfp8((const float *) args.input->data, ids_src1.get(), input_mxfp8.get(),
                                             args.input->ne[0], args.input->nb[2] / sizeof(float), input_ne0, n_tokens,
                                             n_rows, n_expert_used, ctx.stream());
    }

    ggml_cuda_moe_mmq_state w13_state = {
        ids_src1.ptr,
        ids_dst.ptr,
        expert_bounds.ptr,
        n_experts,
        n_tokens,
        n_expert_used,
        true,
        w13_offsets,
        w13_tile_rows,
        device_info.nsm * config.w13.cta_multiplier,
        config.w13.output_tile_major,
        config.use_cp_async,
        config.use_weight_pipeline,
        config.tma_warp_specialized,
        nullptr,
        w13_weight,
    };
    if (config.w13_epilogue == moe_mmq_w13_epilogue_mode::tma_epilogue) {
        w13_state.epilogue         = GGML_CUDA_MOE_MMQ_EPILOGUE_W13;
        w13_state.bias             = (const float *) args.gate_up_bias->data;
        w13_state.activation_q     = activation_q.get();
        w13_state.activation_q_ne0 = activation_q_ne0;
        w13_state.epilogue_width   = args.activation->ne[0];
    }
    w13_state.src1_q            = input_mxfp8_ptr;
    w13_state.activation_format = config.activation_format;
    w13_state.tma_tail_elide    = config.tma_tail_elide;
    w13_state.logical_k         = args.gate_up->ne[0];
    ggml_cuda_moe_weight_wait_ready(w13_weight, ctx.stream());
    ggml_cuda_mul_mat_q(ctx, args.gate_up, args.input, args.ids, args.gate_up_dst,
                        config.shared_plan ? &w13_state : nullptr);

    if (config.w13_epilogue == moe_mmq_w13_epilogue_mode::staged) {
#ifdef GGML_CUDA_MOE_PROFILE
        const ggml_cuda_moe_profile_scope profile_scope("ffn_moe.staged_swiglu_oai");
#endif
        ggml_cuda_moe_mmq_w13_epilogue_staged(args, ids_stride, ctx.stream());
    } else if (config.w13_epilogue == moe_mmq_w13_epilogue_mode::fused) {
#ifdef GGML_CUDA_MOE_PROFILE
        const ggml_cuda_moe_profile_scope profile_scope("ffn_moe.fused_swiglu_oai");
#endif
        ggml_cuda_moe_mmq_w13_epilogue_fused(args, ids_stride, ctx.stream());
    } else if (config.w13_epilogue == moe_mmq_w13_epilogue_mode::fused_quant) {
#ifdef GGML_CUDA_MOE_PROFILE
        const ggml_cuda_moe_profile_scope profile_scope("ffn_moe.fused_swiglu_oai_quant");
#endif
        ggml_cuda_moe_mmq_w13_epilogue_quantize(
            args, ids_dst.get(), activation_q.get(), activation_q_ne0, ids_stride, ctx.stream());
    }

    ggml_cuda_moe_mmq_state w2_state = {
        ids_src1.ptr,
        ids_dst.ptr,
        expert_bounds.ptr,
        n_experts,
        n_tokens,
        n_expert_used,
        true,
        w2_offsets,
        w2_tile_rows,
        device_info.nsm * config.w2.cta_multiplier,
        config.w2.output_tile_major,
        config.use_cp_async,
        config.use_weight_pipeline,
        config.tma_warp_specialized,
        activation_q_ptr,
        w2_weight,
    };
    if (config.w2_epilogue == moe_mmq_w2_epilogue_mode::tma_weighted ||
        config.w2_epilogue == moe_mmq_w2_epilogue_mode::tma_atomic) {
        w2_state.epilogue      = config.w2_epilogue == moe_mmq_w2_epilogue_mode::tma_weighted ?
                                     GGML_CUDA_MOE_MMQ_EPILOGUE_W2_WEIGHTED :
                                     GGML_CUDA_MOE_MMQ_EPILOGUE_W2_ATOMIC;
        w2_state.bias          = (const float *) args.down_bias->data;
        w2_state.route_weights = (const float *) args.weights->data;
        w2_state.epilogue_dst = config.w2_epilogue == moe_mmq_w2_epilogue_mode::tma_weighted ?
                                    (float *) args.weighted->data :
                                    (float *) args.dst->data;
        w2_state.epilogue_width = args.dst->ne[0];
    }
    w2_state.activation_format = config.activation_format;
    w2_state.tma_tail_elide    = config.tma_tail_elide;
    w2_state.logical_k         = args.down->ne[0];
    if (config.w2_epilogue == moe_mmq_w2_epilogue_mode::tma_atomic) {
        CUDA_CHECK(cudaMemsetAsync(args.dst->data, 0, ggml_nbytes(args.dst), ctx.stream()));
    }
    ggml_cuda_moe_weight_wait_ready(w2_weight, ctx.stream());
    ggml_cuda_mul_mat_q(ctx, args.down, args.activation, args.ids, args.down_dst,
                        config.shared_plan ? &w2_state : nullptr);

    if (config.w2_epilogue == moe_mmq_w2_epilogue_mode::staged) {
#ifdef GGML_CUDA_MOE_PROFILE
        const ggml_cuda_moe_profile_scope profile_scope("ffn_moe.w2_epilogue_staged");
#endif
        ggml_cuda_moe_mmq_w2_epilogue_staged(args, ids_stride, ctx.stream());
    } else if (config.w2_epilogue == moe_mmq_w2_epilogue_mode::fused) {
#ifdef GGML_CUDA_MOE_PROFILE
        const ggml_cuda_moe_profile_scope profile_scope("ffn_moe.w2_epilogue_fused");
#endif
        ggml_cuda_moe_mmq_w2_epilogue_fused(args, ids_stride, ctx.stream());
    } else if (config.w2_epilogue == moe_mmq_w2_epilogue_mode::tma_weighted) {
#ifdef GGML_CUDA_MOE_PROFILE
        const ggml_cuda_moe_profile_scope profile_scope("ffn_moe.weighted_reduce");
#endif
        ggml_cuda_moe_mmq_reduce_weighted(args, ctx.stream());
    }

    return true;
}
