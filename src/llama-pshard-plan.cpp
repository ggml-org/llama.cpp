#include "llama-pshard-plan.h"

#include "llama-benchmark.h"
#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-model.h"
#include "llama-model-loader.h"

#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdlib>
#include <memory>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include <atomic>
#include <mutex>
#include <thread>

static std::mutex g_probe_mutex;

static std::vector<llama_device_memory_data> llama_get_device_memory_data_safe(
        const char * path_model, const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<llama_device> & devs, uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train, uint32_t & hp_n_expert, uint32_t & hp_n_embd_r,
        ggml_log_level log_level,
        llama_probe_hook_t probe_hook = nullptr,
        void * probe_hook_data = nullptr) {
    std::lock_guard<std::mutex> lock(g_probe_mutex);
    return llama_get_device_memory_data(path_model, mparams, cparams, devs,
        hp_ngl, hp_n_ctx_train, hp_n_expert, hp_n_embd_r, log_level, probe_hook, probe_hook_data);
}

const char * llama_get_overflow_pattern(size_t il, llama_layer_fraction lf) {
    constexpr size_t n_strings = 1000;
    GGML_ASSERT(il < n_strings);
    switch (lf) {
        case LLAMA_LAYER_FRACTION_ATTN: {
            static std::array<std::string, n_strings> p;
            if (p[il].empty()) { p[il] = "blk\\." + std::to_string(il) + "\\.ffn_(up|gate|down).*"; }
            return p[il].c_str();
        }
        case LLAMA_LAYER_FRACTION_UP: {
            static std::array<std::string, n_strings> p;
            if (p[il].empty()) { p[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|down).*"; }
            return p[il].c_str();
        }
        case LLAMA_LAYER_FRACTION_GATE: {
            static std::array<std::string, n_strings> p;
            if (p[il].empty()) { p[il] = "blk\\." + std::to_string(il) + "\\.ffn_down.*"; }
            return p[il].c_str();
        }
        case LLAMA_LAYER_FRACTION_MOE: {
            static std::array<std::string, n_strings> p;
            if (p[il].empty()) { p[il] = "blk\\." + std::to_string(il) + "\\.ffn_(up|down|gate)_(ch|)exps"; }
            return p[il].c_str();
        }
        default:
            return nullptr;
    }
}

static void llama_pshard_generate_overrides(
        uint32_t n_pinned,
        uint32_t n_layers,
        ggml_backend_buffer_type_t gpu_buft,
        ggml_backend_buffer_type_t host_buft,
        struct llama_model_tensor_buft_override * tensor_buft_overrides,
        llama_layer_fraction overflow_type,
        llama_pshard_strategy strategy,
        const pshard_dev_layout & layout,
        bool pin_from_back = false,
        bool output_on_gpu = false,
        uint32_t n_attn_pinned = 0) {
    thread_local std::array<std::string, 1000> patterns_layer;
    thread_local std::array<std::string, 1000> patterns_layer_attn;
    thread_local std::array<std::string, 1000> patterns_layer_ffn;
    thread_local std::string pat_output = "^output";

    const uint32_t il_pin_start = pin_from_back ? (n_layers - n_pinned) : 0;
    GGML_ASSERT(n_layers <= 1000 && "pshard: n_layers exceeds thread_local pattern array capacity");
    const uint32_t il_pin_end   = pin_from_back ? n_layers : n_pinned;
    const uint32_t il_boundary_raw = pin_from_back ? (il_pin_start > 0 ? il_pin_start - 1 : UINT32_MAX) : il_pin_end;
    const uint32_t il_boundary = (overflow_type != LLAMA_LAYER_FRACTION_NONE && il_boundary_raw < n_layers) ? il_boundary_raw : UINT32_MAX;
    const bool output_on_cpu = !output_on_gpu;

    static constexpr size_t OVERRIDE_CAP = 4096;
    size_t itbo = 0;

    auto emit = [&](const char * pat, ggml_backend_buffer_type_t buft, int32_t bid) {
        GGML_ASSERT(itbo + 1 < OVERRIDE_CAP && "override array overflow");
        tensor_buft_overrides[itbo] = { pat, buft, bid };
        itbo++;
    };

    {
        thread_local std::string pat_tok_embd = "^token_embd";
        const int32_t out_bid = output_on_cpu ? layout.cpu : layout.compute;
        emit(pat_output.c_str(), output_on_cpu ? host_buft : gpu_buft, out_bid);
        emit(pat_tok_embd.c_str(), host_buft, layout.cpu);
    }

    // one pattern cache per thread
    for (uint32_t il = 0; il < n_layers; il++) {
        if (patterns_layer[il].empty())      { patterns_layer[il]      = "blk\\." + std::to_string(il) + "\\..*"; }
        if (patterns_layer_attn[il].empty()) { patterns_layer_attn[il] = "blk\\." + std::to_string(il) + "\\.attn_(q|k|v|output|q_norm|k_norm).*"; }
        if (patterns_layer_ffn[il].empty())  { patterns_layer_ffn[il]  = "blk\\." + std::to_string(il) + "\\.ffn_((up|gate|down)\\.|(up|down|gate|gate_up)_(ch|)exps).*"; }

        if (il == il_boundary) {
            const char * overflow_pat = llama_get_overflow_pattern(il, overflow_type);
            if (overflow_pat) {
                emit(overflow_pat, host_buft, (strategy == LLAMA_PSHARD_STATIC_FITPARAMS_DENSEPRIO_MOEONLY) ? layout.cpu : layout.shard(il));
            }
            emit(patterns_layer[il].c_str(), gpu_buft, layout.compute);
        } else if (il >= il_pin_start && il < il_pin_end) {
            emit(patterns_layer[il].c_str(), gpu_buft, layout.compute);
        } else {
            const bool    use_alternating_shards = strategy == LLAMA_PSHARD_GPUONLY_LAYERPIN_LAYERSTREAM;
            const int32_t shard_bid = use_alternating_shards ? layout.shard(il) : layout.shard_a;

            switch (strategy) {
                case LLAMA_PSHARD_STATIC_FITPARAMS_DENSEPRIO_MOEONLY:
                    emit(patterns_layer[il].c_str(), host_buft, layout.cpu);
                    break;

                case LLAMA_PSHARD_GPUONLY_LAYERPIN_LAYERSTREAM:
                    emit(patterns_layer[il].c_str(), host_buft, shard_bid);
                    break;

                case LLAMA_PSHARD_GPUONLY_ATTNPIN_FFNSTREAM:
                    emit(patterns_layer_ffn[il].c_str(), host_buft, shard_bid);
                    emit(patterns_layer[il].c_str(), gpu_buft, layout.compute);
                    break;

                case LLAMA_PSHARD_DYNAMIC_FFNCPU_ATTNSTREAM:
                    emit(patterns_layer_ffn[il].c_str(), host_buft, layout.cpu);
                    emit(patterns_layer[il].c_str(), host_buft, shard_bid);
                    break;

                case LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS:
                    if (n_attn_pinned > 0 && il < n_attn_pinned) {
                        emit(patterns_layer_ffn[il].c_str(), host_buft, layout.cpu);
                        emit(patterns_layer[il].c_str(), gpu_buft, layout.compute);
                    } else {
                        emit(patterns_layer[il].c_str(), host_buft, layout.cpu);
                    }
                    break;

                default: break;
            }
        }
    }
    tensor_buft_overrides[itbo] = { nullptr, nullptr, -1 };
    LLAMA_LOG_DEBUG("%s: %zu overrides emitted\n", __func__, itbo);
}

// overflow names by enum value
static const char * const PSHARD_FRAC_NAMES[] = { "NONE", "ATTN", "UP", "GATE", "MOE" };

struct llama_pshard_search_ctx {
    const char                               * path_model;
    const struct llama_model_params           * mparams;
    const struct llama_context_params         * cparams;
    struct llama_model_tensor_buft_override   * overrides;
    uint32_t                                   n_layers;
    size_t                                     vram_free;
    ggml_backend_buffer_type_t                 gpu_buft;
    ggml_backend_buffer_type_t                 host_buft;
    pshard_dev_layout                          layout;
    bool                                       is_moe;
    bool                                       has_rs = false;

    // optional TPS predictor
    const llama_benchmark_predictor          * predictor = nullptr;
    uint32_t                                   kv_size   = 0;
    uint32_t                                   cache_ubatch = 0;
};

struct llama_pshard_tps_hook_data {
    const llama_benchmark_predictor * predictor;
    int      cpu_backend_id;
    uint32_t kv_size;
    int32_t  batch_size;
    uint32_t n_outputs;
    bool     has_rs;
    float  * out_tps;
};

static void pshard_tps_probe_hook(llama_context * ctx, void * user_data) {
    auto * d = (llama_pshard_tps_hook_data *) user_data;
    if (!d || !d->predictor || !ctx) return;

    {
        llama_memory_context_ptr mctx;
        auto * mem = ctx->get_memory();
        if (mem) {
            mctx = mem->init_full();
        }
        uint32_t n_tokens = (uint32_t)d->batch_size;
        uint32_t n_seqs   = ctx->n_seq_max();
        ctx->graph_reserve(n_tokens, n_seqs, n_seqs, mctx.get(), true);
    }

    d->predictor->clear_cache();
    double tps = d->predictor->predict_tps(ctx->get_sched(), d->cpu_backend_id, d->kv_size, d->batch_size, d->n_outputs, d->has_rs);
    if (d->out_tps) {
        *d->out_tps = (float)tps;
    }
}

static std::vector<llama_device_memory_data> llama_pshard_probe_memory(
        const llama_pshard_search_ctx & ctx,
        const llama_model_params      & mparams,
        const llama_context_params    & cparams,
        ggml_log_level                  log_level,
        llama_probe_hook_t              probe_hook = nullptr,
        void                          * probe_hook_data = nullptr) {
    std::vector<llama_device> devs;
    uint32_t hp_ngl = 0, hp_n_ctx_train = 0, hp_n_expert = 0, hp_n_embd_r = 0;

    auto data_tier = llama_get_device_memory_data_safe(
        ctx.path_model, &mparams, &cparams, devs,
        hp_ngl, hp_n_ctx_train, hp_n_expert, hp_n_embd_r,
        log_level, probe_hook, probe_hook_data);

    if (ctx.cache_ubatch != 0 && ctx.cache_ubatch != cparams.n_ubatch) {
        llama_context_params cparams_cache = cparams;
        cparams_cache.n_batch  = std::max(cparams_cache.n_batch, ctx.cache_ubatch);
        cparams_cache.n_ubatch = ctx.cache_ubatch;

        std::vector<llama_device> devs_cache;
        uint32_t c_ngl = 0, c_n_ctx_train = 0, c_n_expert = 0, c_n_embd_r = 0;
        auto data_cache = llama_get_device_memory_data_safe(
            ctx.path_model, &mparams, &cparams_cache, devs_cache,
            c_ngl, c_n_ctx_train, c_n_expert, c_n_embd_r,
            log_level);

        for (size_t i = 0; i < data_tier.size() && i < data_cache.size(); i++) {
            data_tier[i].mb.context = data_cache[i].mb.context;
        }
    }

    return data_tier;
}

struct llama_pshard_tier_prune {
    uint32_t hi_pinned[LLAMA_PSHARD_COUNT];
    uint32_t hi_attn;
    bool     skip[LLAMA_PSHARD_COUNT];

    void init(uint32_t n_layers) {
        for (int s = 0; s < LLAMA_PSHARD_COUNT; s++) {
            hi_pinned[s] = n_layers;
            skip[s] = false;
        }
        hi_attn = n_layers;
    }

    void update(int s, const llama_pshard_plan & plan) {
        if (!plan.is_viable) {
            skip[s] = true;
        } else if (s == LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS) {
            if (plan.n_attn_pinned == 0) {
                skip[s] = true;
            } else {
                hi_attn = plan.n_attn_pinned;
                hi_pinned[s] = plan.n_pinned;
            }
        } else {
            if (plan.n_pinned == 0) {
                skip[s] = true;
            } else {
                hi_pinned[s] = plan.n_pinned;
            }
        }
    }
};

static llama_pshard_plan llama_pshard_search_strategy(
        const llama_pshard_search_ctx & ctx,
        llama_pshard_strategy strategy,
        uint32_t hi_hint = UINT32_MAX,
        uint32_t lo_hint = 0) {

    const auto * mparams    = ctx.mparams;
    const auto * cparams    = ctx.cparams;
    auto * tensor_buft_overrides = ctx.overrides;
    const auto   n_layers   = ctx.n_layers;
    const auto   vram_free  = ctx.vram_free;
    const auto   gpu_buft   = ctx.gpu_buft;
    const auto   host_buft  = ctx.host_buft;
    const auto & layout     = ctx.layout;
    const auto   is_moe     = ctx.is_moe;

    llama_pshard_plan plan;
    plan.strategy   = strategy;
    plan.batch_size = cparams->n_batch;

    const bool use_pshard = (strategy != LLAMA_PSHARD_STATIC_FITPARAMS_DENSEPRIO_MOEONLY);

    const uint32_t hi_default = use_pshard ? (n_layers - 1) : n_layers;
    uint32_t lo = lo_hint, hi = (hi_hint < hi_default) ? hi_hint : hi_default;
    uint32_t best_n_pinned = lo_hint;
    int64_t mem_lo = 0, mem_hi = (int64_t)vram_free * 2;

    // try the upper bound first
    {
        llama_pshard_generate_overrides(hi, n_layers, gpu_buft, host_buft,
            tensor_buft_overrides, LLAMA_LAYER_FRACTION_NONE, strategy, layout, false, false);
        llama_model_params mp = *mparams;
        mp.pshard = use_pshard;
        mp.n_gpu_layers = use_pshard ? (n_layers + 1) : (hi + 1);
        mp.tensor_buft_overrides = tensor_buft_overrides;
        try {
            const auto d = llama_pshard_probe_memory(ctx, mp, *cparams, GGML_LOG_LEVEL_ERROR);
            const int64_t gpu_used = d[0].mb.total();
            LLAMA_LOG_INFO("%s: [%s] n_pinned=%u -> %.1f MiB (model=%.1f cache=%.1f compute=%.1f) budget %.1f %s (hi-first)\n",
                __func__, llama_pshard_strategy_name(strategy), hi,
                gpu_used / (1024.0 * 1024.0),
                d[0].mb.model / (1024.0 * 1024.0), d[0].mb.context / (1024.0 * 1024.0), d[0].mb.compute / (1024.0 * 1024.0),
                vram_free / (1024.0 * 1024.0),
                gpu_used <= (int64_t)vram_free ? "FITS" : "OVER");
            if (gpu_used <= (int64_t)vram_free) {
                best_n_pinned = hi;
                lo = hi + 1;
            } else {
                mem_hi = gpu_used;
            }
        } catch (...) {
            LLAMA_LOG_WARN("%s: [%s] hi-first probe failed (n_pinned=%u)\n", __func__, llama_pshard_strategy_name(strategy), hi);
        }
    }

    while (lo <= hi) {
        uint32_t mid;
        if (mem_hi > mem_lo && mem_hi > (int64_t)vram_free) {
            int64_t target = (int64_t)vram_free;
            mid = lo + (uint32_t)((double)(target - mem_lo) * (hi - lo) / (mem_hi - mem_lo));
            if (mid <= lo) mid = lo + 1;
            if (mid > hi)  mid = hi;
        } else {
            mid = (lo + hi) / 2;
        }

        llama_pshard_generate_overrides(mid, n_layers, gpu_buft, host_buft,
            tensor_buft_overrides, LLAMA_LAYER_FRACTION_NONE, strategy, layout, false, false);

        llama_model_params mp = *mparams;
        mp.pshard = use_pshard;
        mp.n_gpu_layers = use_pshard ? (n_layers + 1) : (mid + 1);
        mp.tensor_buft_overrides = tensor_buft_overrides;

        try {
            const auto d = llama_pshard_probe_memory(ctx, mp, *cparams, GGML_LOG_LEVEL_ERROR);
            const int64_t gpu_used = d[0].mb.total();

            LLAMA_LOG_INFO("%s: [%s] n_pinned=%u -> %.1f MiB (model=%.1f cache=%.1f compute=%.1f) budget %.1f %s\n",
                __func__, llama_pshard_strategy_name(strategy), mid,
                gpu_used / (1024.0 * 1024.0),
                d[0].mb.model   / (1024.0 * 1024.0),
                d[0].mb.context / (1024.0 * 1024.0),
                d[0].mb.compute / (1024.0 * 1024.0),
                vram_free / (1024.0 * 1024.0),
                gpu_used <= (int64_t)vram_free ? "FITS" : "OVER");

            if (gpu_used <= (int64_t)vram_free) {
                best_n_pinned = mid;
                lo = mid + 1;
                mem_lo = gpu_used;
            } else {
                if (mid == 0) break;
                hi = mid - 1;
                mem_hi = gpu_used;
            }
        } catch (...) {
            LLAMA_LOG_WARN("%s: [%s] probe failed (n_pinned=%u)\n", __func__, llama_pshard_strategy_name(strategy), mid);
            if (mid == 0) break;
            hi = mid - 1;
        }
    }

    llama_layer_fraction best_overflow = LLAMA_LAYER_FRACTION_NONE;
    const uint32_t            fallback_n_pinned = best_n_pinned;                    // known fit
    const llama_layer_fraction fallback_overflow = LLAMA_LAYER_FRACTION_NONE;
    if (best_n_pinned < n_layers && !(use_pshard && best_n_pinned >= n_layers - 1)) {
        const uint32_t frac_n_pinned = best_n_pinned + 1; // pin one more layer, partially
        auto try_frac = [&](llama_layer_fraction frac) -> bool {
            llama_pshard_generate_overrides(frac_n_pinned, n_layers, gpu_buft, host_buft,
                tensor_buft_overrides, frac, strategy, layout, false, false);
            llama_model_params mp = *mparams;
            mp.pshard = use_pshard;
            mp.n_gpu_layers = use_pshard ? (n_layers + 1) : (frac_n_pinned + 1);
            mp.tensor_buft_overrides = tensor_buft_overrides;
            try {
                const auto d = llama_pshard_probe_memory(ctx, mp, *cparams, GGML_LOG_LEVEL_ERROR);
                return d[0].mb.total() <= (int64_t)vram_free;
            } catch (...) {
                LLAMA_LOG_WARN("%s: [%s] overflow probe failed (frac=%d)\n", __func__, llama_pshard_strategy_name(strategy), (int)frac);
                return false;
            }
        };
        // try one partial boundary layer
        if (try_frac(LLAMA_LAYER_FRACTION_ATTN)) {
            best_n_pinned = frac_n_pinned;
            best_overflow = LLAMA_LAYER_FRACTION_ATTN;
            if (try_frac(LLAMA_LAYER_FRACTION_UP))   { best_overflow = LLAMA_LAYER_FRACTION_UP;
            if (try_frac(LLAMA_LAYER_FRACTION_GATE)) { best_overflow = LLAMA_LAYER_FRACTION_GATE;
            if (try_frac(LLAMA_LAYER_FRACTION_MOE))  { best_overflow = LLAMA_LAYER_FRACTION_MOE; }}}
        }
    }

    plan.n_pinned      = best_n_pinned;
    plan.overflow      = best_overflow;
    plan.output_on_gpu = false;

    llama_pshard_generate_overrides(best_n_pinned, n_layers, gpu_buft, host_buft,
        tensor_buft_overrides, best_overflow, strategy, layout, false, plan.output_on_gpu);
    {
        llama_model_params mp = *mparams;
        mp.pshard = use_pshard;
        mp.n_gpu_layers = use_pshard ? (n_layers + 1) : (best_n_pinned + 1);
        mp.tensor_buft_overrides = tensor_buft_overrides;
        llama_pshard_tps_hook_data tps_data = { ctx.predictor, layout.cpu, ctx.kv_size, (int32_t)cparams->n_batch, cparams->n_seq_max, ctx.has_rs, &plan.tps };
        auto * hook     = ctx.predictor ? pshard_tps_probe_hook : nullptr;
        auto * hookdata = ctx.predictor ? (void *)&tps_data     : nullptr;

        try {
            const auto d = llama_pshard_probe_memory(ctx, mp, *cparams, GGML_LOG_LEVEL_ERROR, hook, hookdata);
            plan.total_vram_req   = d[0].mb.total();
            plan.scratch_measured = d[0].mb.compute;
            plan.cache_measured   = d[0].mb.context;
            plan.is_viable = ((int64_t)plan.total_vram_req <= (int64_t)vram_free);
        } catch (...) {
            LLAMA_LOG_WARN("%s: [%s] final measurement probe failed (n_pinned=%u)\n", __func__, llama_pshard_strategy_name(strategy), best_n_pinned);
            plan.is_viable = false;
        }
    }

    // drop the partial boundary layer if the final probe exceeds budget
    if (!plan.is_viable && best_overflow != LLAMA_LAYER_FRACTION_NONE) {
        best_n_pinned = fallback_n_pinned;
        best_overflow = fallback_overflow;
        plan.n_pinned = best_n_pinned;
        plan.overflow = best_overflow;
        llama_pshard_generate_overrides(best_n_pinned, n_layers, gpu_buft, host_buft,
            tensor_buft_overrides, best_overflow, strategy, layout, false, plan.output_on_gpu);
        llama_model_params mp = *mparams;
        mp.pshard = use_pshard;
        mp.n_gpu_layers = use_pshard ? (n_layers + 1) : (best_n_pinned + 1);
        mp.tensor_buft_overrides = tensor_buft_overrides;
        llama_pshard_tps_hook_data tps_data = { ctx.predictor, layout.cpu, ctx.kv_size, (int32_t)cparams->n_batch, cparams->n_seq_max, ctx.has_rs, &plan.tps };
        auto * hook     = ctx.predictor ? pshard_tps_probe_hook : nullptr;
        auto * hookdata = ctx.predictor ? (void *)&tps_data     : nullptr;
        try {
            const auto d = llama_pshard_probe_memory(ctx, mp, *cparams, GGML_LOG_LEVEL_ERROR, hook, hookdata);
            plan.total_vram_req   = d[0].mb.total();
            plan.scratch_measured = d[0].mb.compute;
            plan.cache_measured   = d[0].mb.context;
            plan.is_viable = ((int64_t)plan.total_vram_req <= (int64_t)vram_free);
        } catch (...) {
            plan.is_viable = false;
        }
    }

    for (const auto * ov = tensor_buft_overrides; ov->pattern; ++ov) {
        plan.overrides.push_back({ov->pattern, ov->buft, ov->backend_id});
    }

    return plan;
}

static llama_pshard_plan llama_pshard_search_attn_pin(
        const llama_pshard_search_ctx & ctx,
        uint32_t hi_attn_hint = UINT32_MAX,
        uint32_t hi_full_hint = UINT32_MAX,
        uint32_t lo_full_hint = 0) {

    const auto * mparams    = ctx.mparams;
    const auto * cparams    = ctx.cparams;
    auto * tensor_buft_overrides = ctx.overrides;
    const auto   n_layers   = ctx.n_layers;
    const auto   vram_free  = ctx.vram_free;
    const auto   gpu_buft   = ctx.gpu_buft;
    const auto   host_buft  = ctx.host_buft;
    const auto & layout     = ctx.layout;
    const auto   is_moe     = ctx.is_moe;

    llama_pshard_plan plan;
    plan.strategy = LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS;

    auto measure_vram = [&](uint32_t n_full, uint32_t n_attn, bool out_gpu) -> llama_memory_breakdown_data {
        llama_pshard_generate_overrides(n_full, n_layers, gpu_buft, host_buft,
            tensor_buft_overrides, LLAMA_LAYER_FRACTION_NONE, LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS,
            layout, false, out_gpu, n_attn);

        llama_model_params mp = *mparams;
        mp.pshard = true;
        mp.n_gpu_layers = n_layers + 1;
        mp.tensor_buft_overrides = tensor_buft_overrides;

        const auto d = llama_pshard_probe_memory(ctx, mp, *cparams, GGML_LOG_LEVEL_ERROR);
        return d[0].mb;
    };

    // phase 1: maximize attention layers on GPU
    uint32_t n_attn = 0;
    {
        uint32_t lo = 0, hi = (hi_attn_hint < n_layers) ? hi_attn_hint : n_layers;
        int64_t mem_lo = 0, mem_hi = (int64_t)vram_free * 2;

        try {
            auto mb = measure_vram(0, hi, false);
            int64_t gpu_used = mb.total();
            LLAMA_LOG_INFO("%s: [STATIC_ATTNPRIO_ALLMODELS p1] n_attn=%u -> %.1f MiB (model=%.1f cache=%.1f compute=%.1f) budget %.1f %s (hi-first)\n",
                __func__, hi, gpu_used / (1024.0 * 1024.0),
                mb.model / (1024.0 * 1024.0), mb.context / (1024.0 * 1024.0), mb.compute / (1024.0 * 1024.0),
                vram_free / (1024.0 * 1024.0),
                gpu_used <= (int64_t)vram_free ? "FITS" : "OVER");
            if (gpu_used <= (int64_t)vram_free) {
                n_attn = hi;
                lo = hi + 1;
            } else {
                mem_hi = gpu_used;
            }
        } catch (...) {
            LLAMA_LOG_WARN("%s: [STATIC_ATTNPRIO_ALLMODELS p1] hi-first probe failed (n_attn=%u)\n", __func__, hi);
        }

        while (lo <= hi) {
            uint32_t mid;
            if (mem_hi > mem_lo && mem_hi > (int64_t)vram_free) {
                mid = lo + (uint32_t)((double)((int64_t)vram_free - mem_lo) * (hi - lo) / (mem_hi - mem_lo));
                if (mid <= lo) mid = lo + 1;
                if (mid > hi)  mid = hi;
            } else {
                mid = (lo + hi) / 2;
            }

            try {
                auto mb = measure_vram(0, mid, false);
                int64_t gpu_used = mb.total();
                LLAMA_LOG_INFO("%s: [STATIC_ATTNPRIO_ALLMODELS p1] n_attn=%u -> %.1f MiB (model=%.1f cache=%.1f compute=%.1f) budget %.1f %s\n",
                    __func__, mid, gpu_used / (1024.0 * 1024.0),
                    mb.model / (1024.0 * 1024.0), mb.context / (1024.0 * 1024.0), mb.compute / (1024.0 * 1024.0),
                    vram_free / (1024.0 * 1024.0),
                    gpu_used <= (int64_t)vram_free ? "FITS" : "OVER");

                if (gpu_used <= (int64_t)vram_free) {
                    n_attn = mid;
                    lo = mid + 1;
                    mem_lo = gpu_used;
                } else {
                    if (mid == 0) break;
                    hi = mid - 1;
                    mem_hi = gpu_used;
                }
            } catch (...) {
                LLAMA_LOG_WARN("%s: [STATIC_ATTNPRIO_ALLMODELS p1] probe failed (n_attn=%u)\n", __func__, mid);
                if (mid == 0) break;
                hi = mid - 1;
            }
        }
    }

    // moe tries output on gpu before ffn pinning
    // dense tries output on gpu after ffn pinning
    bool output_on_gpu = false;

    if (is_moe && n_attn >= n_layers) {
        try {
            auto mb = measure_vram(0, n_attn, true);
            int64_t gpu_used = mb.total();
            LLAMA_LOG_INFO("%s: [STATIC_ATTNPRIO_ALLMODELS p1b] output_on_gpu probe -> %.1f MiB (model=%.1f cache=%.1f compute=%.1f) budget %.1f %s\n",
                __func__, gpu_used / (1024.0 * 1024.0),
                mb.model / (1024.0 * 1024.0), mb.context / (1024.0 * 1024.0), mb.compute / (1024.0 * 1024.0),
                vram_free / (1024.0 * 1024.0),
                gpu_used <= (int64_t)vram_free ? "FITS" : "OVER");
            if (gpu_used <= (int64_t)vram_free) {
                output_on_gpu = true;
            }
        } catch (...) {
            LLAMA_LOG_WARN("%s: [STATIC_ATTNPRIO_ALLMODELS p1b] output_on_gpu probe failed\n", __func__);
        }
    }

    // phase 2: maximize fully pinned layers
    uint32_t n_full = 0;
    {
        uint32_t hi_full_max = (hi_full_hint < n_attn) ? hi_full_hint : n_attn;
        uint32_t lo = lo_full_hint, hi = hi_full_max;
        int64_t mem_lo = 0, mem_hi = (int64_t)vram_free * 2;

        try {
            auto mb = measure_vram(hi, n_attn, output_on_gpu);
            int64_t gpu_used = mb.total();
            LLAMA_LOG_INFO("%s: [STATIC_ATTNPRIO_ALLMODELS p2] n_full=%u n_attn=%u -> %.1f MiB (model=%.1f cache=%.1f compute=%.1f) budget %.1f %s (hi-first)\n",
                __func__, hi, n_attn, gpu_used / (1024.0 * 1024.0),
                mb.model / (1024.0 * 1024.0), mb.context / (1024.0 * 1024.0), mb.compute / (1024.0 * 1024.0),
                vram_free / (1024.0 * 1024.0),
                gpu_used <= (int64_t)vram_free ? "FITS" : "OVER");
            if (gpu_used <= (int64_t)vram_free) {
                n_full = hi;
                lo = hi + 1;
            } else {
                mem_hi = gpu_used;
            }
        } catch (...) {
            LLAMA_LOG_WARN("%s: [STATIC_ATTNPRIO_ALLMODELS p2] hi-first probe failed (n_full=%u, n_attn=%u)\n", __func__, hi, n_attn);
        }

        while (lo <= hi) {
            uint32_t mid;
            if (mem_hi > mem_lo && mem_hi > (int64_t)vram_free) {
                mid = lo + (uint32_t)((double)((int64_t)vram_free - mem_lo) * (hi - lo) / (mem_hi - mem_lo));
                if (mid <= lo) mid = lo + 1;
                if (mid > hi)  mid = hi;
            } else {
                mid = (lo + hi) / 2;
            }

            try {
                auto mb = measure_vram(mid, n_attn, output_on_gpu);
                int64_t gpu_used = mb.total();
                LLAMA_LOG_INFO("%s: [STATIC_ATTNPRIO_ALLMODELS p2] n_full=%u n_attn=%u -> %.1f MiB (model=%.1f cache=%.1f compute=%.1f) budget %.1f %s\n",
                    __func__, mid, n_attn, gpu_used / (1024.0 * 1024.0),
                    mb.model / (1024.0 * 1024.0), mb.context / (1024.0 * 1024.0), mb.compute / (1024.0 * 1024.0),
                    vram_free / (1024.0 * 1024.0),
                    gpu_used <= (int64_t)vram_free ? "FITS" : "OVER");

                if (gpu_used <= (int64_t)vram_free) {
                    n_full = mid;
                    lo = mid + 1;
                    mem_lo = gpu_used;
                } else {
                    if (mid == 0) break;
                    hi = mid - 1;
                    mem_hi = gpu_used;
                }
            } catch (...) {
                LLAMA_LOG_WARN("%s: [STATIC_ATTNPRIO_ALLMODELS p2] probe failed (n_full=%u, n_attn=%u)\n", __func__, mid, n_attn);
                if (mid == 0) break;
                hi = mid - 1;
            }
        }
    }

    // dense tries output on gpu only after all layers fit
    if (!is_moe && !output_on_gpu && n_full >= n_layers) {
        try {
            auto mb = measure_vram(n_full, n_attn, true);
            int64_t gpu_used = mb.total();
            LLAMA_LOG_INFO("%s: [STATIC_ATTNPRIO_ALLMODELS p3] output_on_gpu probe (n_full=%u) -> %.1f MiB (model=%.1f cache=%.1f compute=%.1f) budget %.1f %s\n",
                __func__, n_full, gpu_used / (1024.0 * 1024.0),
                mb.model / (1024.0 * 1024.0), mb.context / (1024.0 * 1024.0), mb.compute / (1024.0 * 1024.0),
                vram_free / (1024.0 * 1024.0),
                gpu_used <= (int64_t)vram_free ? "FITS" : "OVER");
            if (gpu_used <= (int64_t)vram_free) {
                output_on_gpu = true;
            }
        } catch (...) {
            LLAMA_LOG_WARN("%s: [STATIC_ATTNPRIO_ALLMODELS p3] output_on_gpu probe failed\n", __func__);
        }
    }

    plan.n_pinned      = n_full;
    plan.n_attn_pinned = n_attn;
    plan.output_on_gpu = output_on_gpu;

    llama_pshard_generate_overrides(n_full, n_layers, gpu_buft, host_buft,
        tensor_buft_overrides, LLAMA_LAYER_FRACTION_NONE, LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS,
        layout, false, plan.output_on_gpu, n_attn);
    {
        llama_model_params mp = *mparams;
        mp.pshard = true;
        mp.n_gpu_layers = n_layers + 1;
        mp.tensor_buft_overrides = tensor_buft_overrides;
        llama_pshard_tps_hook_data tps_data = { ctx.predictor, layout.cpu, ctx.kv_size, (int32_t)cparams->n_batch, cparams->n_seq_max, ctx.has_rs, &plan.tps };
        auto * hook     = ctx.predictor ? pshard_tps_probe_hook : nullptr;
        auto * hookdata = ctx.predictor ? (void *)&tps_data     : nullptr;

        try {
            const auto d = llama_pshard_probe_memory(ctx, mp, *cparams, GGML_LOG_LEVEL_ERROR, hook, hookdata);
            plan.total_vram_req   = d[0].mb.total();
            plan.scratch_measured = d[0].mb.compute;
            plan.cache_measured   = d[0].mb.context;
            plan.is_viable = ((int64_t)plan.total_vram_req <= (int64_t)vram_free);
        } catch (...) {
            LLAMA_LOG_WARN("%s: [STATIC_ATTNPRIO_ALLMODELS] final measurement probe failed (n_full=%u, n_attn=%u)\n", __func__, n_full, n_attn);
            plan.is_viable = false;
        }
    }

    for (const auto * ov = tensor_buft_overrides; ov->pattern; ++ov) {
        plan.overrides.push_back({ov->pattern, ov->buft, ov->backend_id});
    }

    return plan;
}

llama_pshard_plan_registry * llama_pshard_registry_create(uint32_t n_tier_max, uint32_t n_seq_max) {
    auto * registry = new llama_pshard_plan_registry();
    registry->init(n_tier_max, n_seq_max);
    return registry;
}

void llama_pshard_registry_free(llama_pshard_plan_registry * registry) {
    delete registry;
}

// plan cache serialization

uint64_t pshard_registry_fingerprint(
        const struct llama_model_params * mparams,
        const struct llama_context_params * cparams,
        int64_t model_file_size) {

    (void) mparams;

    uint64_t h = 0xcbf29ce484222325ULL;
    auto mix = [&](uint64_t v) { h ^= v; h *= 0x100000001b3ULL; };

    mix(cparams->n_ctx);
    mix(cparams->n_seq_max);
    mix(cparams->n_threads);
    mix((uint64_t)cparams->flash_attn_type);
    mix((uint64_t)cparams->type_k);
    mix((uint64_t)cparams->type_v);
    mix((uint64_t)model_file_size);
    mix((uint64_t)pshard_strategy_from_env());

    return h;
}

static std::string pshard_plan_to_ot(const llama_pshard_plan & plan, ggml_backend_buffer_type_t host_buft) {
    std::string ot;
    const char * buft_name = ggml_backend_buft_name(host_buft);
    for (size_t i = 0; i < plan.overrides.size(); i++) {
        if (i > 0) ot += ',';
        ot += plan.overrides[i].pattern;
        ot += '=';
        ot += buft_name;
        ot += ':';
        ot += std::to_string(plan.overrides[i].backend_id);
    }
    return ot;
}

static const char * pshard_overflow_name(int overflow) {
    static const char * const names[] = { "NONE", "ATTN", "UP", "GATE", "MOE" };
    return (overflow >= 0 && overflow < 5) ? names[overflow] : "NONE";
}

static int pshard_overflow_from_name(const char * name) {
    if (!name) return 0;
    if (strcmp(name, "ATTN") == 0) return 1;
    if (strcmp(name, "UP")   == 0) return 2;
    if (strcmp(name, "GATE") == 0) return 3;
    if (strcmp(name, "MOE")  == 0) return 4;
    return 0;
}

static const size_t PSHARD_MIB = 1024ULL * 1024ULL;
static const int PSHARD_CACHE_MAX_SECTIONS = 10;
static const int PSHARD_CACHE_MAX_VARIANTS = 8;

static uint32_t pshard_bytes_to_mib_ceil(size_t bytes) {
    return (uint32_t)((bytes + PSHARD_MIB - 1) / PSHARD_MIB);
}

static size_t pshard_mib_to_bytes(uint32_t mib) {
    return (size_t)mib * PSHARD_MIB;
}

static size_t pshard_mib_to_bytes(double mib) {
    return (size_t)(mib * (double)PSHARD_MIB + 0.5);
}

static bool pshard_parse_variant_header(const std::string & line, uint32_t & budget_mib, uint32_t & cache_ubatch) {
    cache_ubatch = 0;
    if (sscanf(line.c_str(), "[variant budget=%u cache_ubatch=%u]", &budget_mib, &cache_ubatch) == 2) {
        return true;
    }
    return sscanf(line.c_str(), "[variant budget=%u]", &budget_mib) == 1;
}

static bool pshard_plan_is_better(const llama_pshard_plan & candidate, const llama_pshard_plan & current);

bool pshard_registry_save(
        const llama_pshard_plan_registry * registry, uint64_t fingerprint,
        const char * cache_path, ggml_backend_buffer_type_t host_buft,
        const llama_context_params * cparams) {
    if (!registry || !cache_path) return false;

    struct cache_section {
        std::string header;
        std::vector<std::string> lines;
    };
    std::vector<cache_section> sections;

    FILE * existing = fopen(cache_path, "r");
    if (existing) {
        char line[8192];
        cache_section * cur = nullptr;
        while (fgets(line, sizeof(line), existing)) {
            std::string s = line;
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
            if (s.compare(0, 13, "[fingerprint=") == 0) {
                sections.push_back({s, {}});
                cur = &sections.back();
            } else if (cur) {
                cur->lines.push_back(s);
            }
        }
        fclose(existing);
    }

    std::vector<cache_section> preserved_sections;
    std::vector<std::vector<std::string>> preserved_variants;

    size_t inferred_budget = registry->pshard_disabled ? registry->baseline_vram_req : 0;
    if (!registry->pshard_disabled) {
        for (const auto & plan : registry->best_plans) {
            if (plan.is_viable) {
                inferred_budget = std::max(inferred_budget, plan.total_vram_req);
            }
        }
    }
    const uint32_t budget_mib = registry->budget_mib
        ? registry->budget_mib
        : pshard_bytes_to_mib_ceil(inferred_budget);
    const uint32_t cache_ubatch = registry->cache_ubatch
        ? registry->cache_ubatch
        : (registry->tier_sizes.empty() ? 0 : registry->tier_sizes.back());

    for (const auto & sec : sections) {
        uint64_t fp = 0;
        sscanf(sec.header.c_str(), "[fingerprint=0x%" SCNx64, &fp);
        if (fp != fingerprint) {
            preserved_sections.push_back(sec);
            continue;
        }

        std::vector<std::string> cur_variant;
        uint32_t cur_budget = 0;
        uint32_t cur_cache_ubatch = 0;
        bool in_variant = false;
        auto flush_variant = [&]() {
            if (in_variant && (cur_budget != budget_mib || cur_cache_ubatch != cache_ubatch)) {
                preserved_variants.push_back(cur_variant);
            }
            cur_variant.clear();
            cur_budget = 0;
            cur_cache_ubatch = 0;
            in_variant = false;
        };

        for (const auto & ln : sec.lines) {
            uint32_t parsed_budget = 0;
            uint32_t parsed_cache_ubatch = 0;
            if (pshard_parse_variant_header(ln, parsed_budget, parsed_cache_ubatch)) {
                flush_variant();
                in_variant = true;
                cur_budget = parsed_budget;
                cur_cache_ubatch = parsed_cache_ubatch;
                cur_variant.push_back(ln);
            } else if (in_variant) {
                cur_variant.push_back(ln);
            }
        }
        flush_variant();
    }

    while ((int)preserved_sections.size() >= PSHARD_CACHE_MAX_SECTIONS) {
        preserved_sections.erase(preserved_sections.begin());
    }
    while ((int)preserved_variants.size() >= PSHARD_CACHE_MAX_VARIANTS) {
        preserved_variants.erase(preserved_variants.begin());
    }

    FILE * f = fopen(cache_path, "w");
    if (!f) {
        LLAMA_LOG_WARN("%s: could not write plan cache: %s\n", __func__, cache_path);
        return false;
    }

    fprintf(f, "# Generated file. Edit at your own risk.\n");

    for (const auto & sec : preserved_sections) {
        fprintf(f, "\n%s\n", sec.header.c_str());
        for (const auto & ln : sec.lines) {
            fprintf(f, "%s\n", ln.c_str());
        }
    }

    fprintf(f, "\n[fingerprint=0x%016" PRIx64 "]\n", fingerprint);
    if (cparams) {
        const char * fa_str = "unknown";
        switch (cparams->flash_attn_type) {
            case LLAMA_FLASH_ATTN_TYPE_DISABLED: fa_str = "off";  break;
            case LLAMA_FLASH_ATTN_TYPE_ENABLED:  fa_str = "on";   break;
            case LLAMA_FLASH_ATTN_TYPE_AUTO:     fa_str = "auto"; break;
        }
        const int forced_strategy = pshard_strategy_from_env();
        fprintf(f, "# n_ctx=%u n_seq_max=%u n_threads=%d fa=%s type_k=%d type_v=%d strategy=%s\n",
            cparams->n_ctx, cparams->n_seq_max, cparams->n_threads,
            fa_str, (int)cparams->type_k, (int)cparams->type_v,
            forced_strategy >= 0 ? llama_pshard_strategy_name((llama_pshard_strategy)forced_strategy) : "auto");
    }

    for (const auto & variant : preserved_variants) {
        fprintf(f, "\n");
        for (const auto & ln : variant) {
            fprintf(f, "%s\n", ln.c_str());
        }
    }

    fprintf(f, "\n[variant budget=%u cache_ubatch=%u]\n", budget_mib, cache_ubatch);
    if (registry->pshard_disabled) {
        fprintf(f, "pshard_disabled=1 baseline_vram=%.1f\n", registry->baseline_vram_req / (1024.0 * 1024.0));
    } else {
        for (size_t t = 0; t < registry->tier_sizes.size(); t++) {
            const auto & plan = registry->best_plans[t];
            if (!plan.is_viable) {
                fprintf(f, "[tier %zu bs=%u] not_viable\n", t, registry->tier_sizes[t]);
                continue;
            }
            fprintf(f, "[tier %zu bs=%u]\n", t, registry->tier_sizes[t]);
            fprintf(f, "strategy=%s n_pinned=%u n_attn_pinned=%u overflow=%s tps=%.2f vram=%.1f output_on_gpu=%d pin_from_back=%d\n",
                llama_pshard_strategy_name(plan.strategy),
                plan.n_pinned, plan.n_attn_pinned,
                pshard_overflow_name(plan.overflow),
                plan.tps, plan.total_vram_req / (1024.0 * 1024.0),
                (int)plan.output_on_gpu, (int)plan.pin_from_back);
            fprintf(f, "ot=%s\n", pshard_plan_to_ot(plan, host_buft).c_str());
        }
    }

    fclose(f);
    LLAMA_LOG_INFO("%s: saved budget=%u MiB cache_ubatch=%u variant with %zu tier plans to %s\n",
        __func__, budget_mib, cache_ubatch, registry->tier_sizes.size(), cache_path);
    return true;
}

bool pshard_registry_load(
        llama_pshard_plan_registry * registry, uint64_t fingerprint,
        const char * cache_path, ggml_backend_buffer_type_t host_buft,
        size_t current_budget, bool require_exact_budget) {
    if (!registry || !cache_path) return false;

    FILE * f = fopen(cache_path, "r");
    if (!f) return false;

    char line[8192];
    bool in_section = false;

    char fp_header[64];
    snprintf(fp_header, sizeof(fp_header), "[fingerprint=0x%016" PRIx64 "]", fingerprint);

    struct tier_data {
        uint32_t bs = 0;
        bool viable = false;
        llama_pshard_strategy strategy = LLAMA_PSHARD_STATIC_FITPARAMS_DENSEPRIO_MOEONLY;
        uint32_t n_pinned = 0;
        uint32_t n_attn_pinned = 0;
        int overflow = 0;
        float tps = 0.0f;
        double vram_mib = 0.0;
        int output_on_gpu = 0;
        int pin_from_back = 0;
        std::string ot_line;
    };
    struct variant_data {
        uint32_t budget_mib = 0;
        uint32_t cache_ubatch = 0;
        bool pshard_disabled = false;
        double baseline_vram_mib = 0.0;
        std::vector<tier_data> tiers;
    };
    std::vector<variant_data> variants;
    variant_data * cur_variant = nullptr;

    while (fgets(line, sizeof(line), f)) {
        std::string s = line;
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();

        if (s.compare(0, 13, "[fingerprint=") == 0) {
            if (in_section) break; // hit next section, stop
            in_section = (s == fp_header);
            continue;
        }
        if (!in_section) continue;

        uint32_t variant_budget = 0;
        uint32_t variant_cache_ubatch = 0;
        if (pshard_parse_variant_header(s, variant_budget, variant_cache_ubatch)) {
            variants.push_back({});
            cur_variant = &variants.back();
            cur_variant->budget_mib = variant_budget;
            cur_variant->cache_ubatch = variant_cache_ubatch;
            continue;
        }
        if (!cur_variant) continue;

        if (s.rfind("pshard_disabled=1", 0) == 0) {
            double baseline_mib = 0.0;
            if (sscanf(s.c_str(), "pshard_disabled=1 baseline_vram=%lf", &baseline_mib) == 1) {
                cur_variant->pshard_disabled = true;
                cur_variant->baseline_vram_mib = baseline_mib;
            } else {
                LLAMA_LOG_WARN("%s: malformed pshard_disabled line: %s\n", __func__, s.c_str());
            }
            continue;
        }

        if (s.compare(0, 5, "[tier") == 0) {
            tier_data td = {};
            size_t tier_idx = 0;
            if (s.find("not_viable") != std::string::npos) {
                if (sscanf(s.c_str(), "[tier %zu bs=%u]", &tier_idx, &td.bs) < 2) {
                    LLAMA_LOG_WARN("%s: malformed tier header (not_viable): %s\n", __func__, s.c_str());
                    continue;
                }
                td.viable = false;
            } else {
                if (sscanf(s.c_str(), "[tier %zu bs=%u]", &tier_idx, &td.bs) < 2) {
                    LLAMA_LOG_WARN("%s: malformed tier header: %s\n", __func__, s.c_str());
                    continue;
                }
                td.viable = true;
            }
            cur_variant->tiers.push_back(td);
        } else if (s.compare(0, 9, "strategy=") == 0 && !cur_variant->tiers.empty()) {
            auto & td = cur_variant->tiers.back();
            char strat_name[64] = {}, overflow_name[16] = {};
            if (sscanf(s.c_str(), "strategy=%63s n_pinned=%u n_attn_pinned=%u overflow=%15s tps=%f vram=%lf",
                   strat_name, &td.n_pinned, &td.n_attn_pinned, overflow_name, &td.tps, &td.vram_mib) < 4) {
                LLAMA_LOG_WARN("%s: malformed strategy line: %s\n", __func__, s.c_str());
                td.viable = false;
                continue;
            }

            const char * ogg = strstr(s.c_str(), "output_on_gpu=");
            const char * pfb = strstr(s.c_str(), "pin_from_back=");
            if (!ogg || !pfb) {
                LLAMA_LOG_WARN("%s: missing output_on_gpu/pin_from_back, invalidating cache: %s\n", __func__, s.c_str());
                td.viable = false;
                continue;
            }
            td.output_on_gpu = atoi(ogg + 14);
            td.pin_from_back = atoi(pfb + 14);

            td.overflow = pshard_overflow_from_name(overflow_name);
            td.strategy = LLAMA_PSHARD_STATIC_FITPARAMS_DENSEPRIO_MOEONLY;
            for (int i = 0; i < LLAMA_PSHARD_COUNT; i++) {
                if (strcmp(strat_name, llama_pshard_strategy_name((llama_pshard_strategy)i)) == 0) {
                    td.strategy = (llama_pshard_strategy)i;
                    break;
                }
            }
        } else if (s.compare(0, 3, "ot=") == 0 && !cur_variant->tiers.empty()) {
            cur_variant->tiers.back().ot_line = s.substr(3);
        }
    }
    fclose(f);

    if (variants.empty() || !in_section) return false;

    auto make_plan = [&](const tier_data & td) {
        llama_pshard_plan plan;
        plan.strategy      = td.strategy;
        plan.batch_size    = td.bs;
        plan.n_pinned      = td.n_pinned;
        plan.n_attn_pinned = td.n_attn_pinned;
        plan.overflow      = td.overflow;
        plan.tps           = td.tps;
        plan.total_vram_req = (size_t)(td.vram_mib * 1024 * 1024);
        plan.is_viable     = td.viable;
        plan.output_on_gpu = (bool)td.output_on_gpu;
        plan.pin_from_back = (bool)td.pin_from_back;

        if (!td.ot_line.empty()) {
            std::string remaining = td.ot_line;
            while (!remaining.empty()) {
                size_t comma = remaining.find(',');
                std::string token = (comma != std::string::npos) ? remaining.substr(0, comma) : remaining;
                remaining = (comma != std::string::npos) ? remaining.substr(comma + 1) : "";

                size_t eq = token.find('=');
                if (eq == std::string::npos) continue;

                std::string pattern = token.substr(0, eq);
                std::string buft_bid = token.substr(eq + 1);

                int32_t backend_id = -1;
                size_t colon = buft_bid.rfind(':');
                if (colon != std::string::npos) {
                    backend_id = atoi(buft_bid.c_str() + colon + 1);
                }

                plan.overrides.push_back({ pattern, host_buft, backend_id });
            }
        }
        return plan;
    };

    const uint32_t current_budget_mib = pshard_bytes_to_mib_ceil(current_budget);
    const uint32_t requested_cache_ubatch = registry->cache_ubatch;
    bool skipped_cache_ubatch = false;
    auto cache_ubatch_ok = [&](const variant_data & variant) {
        if (requested_cache_ubatch == 0) return true;
        if (variant.cache_ubatch == requested_cache_ubatch) return true;
        skipped_cache_ubatch = true;
        return false;
    };

    for (const auto & variant : variants) {
        if (!variant.pshard_disabled) continue;
        if (!cache_ubatch_ok(variant)) continue;
        const size_t baseline_vram = pshard_mib_to_bytes(variant.baseline_vram_mib);
        if (baseline_vram <= current_budget) {
            registry->tier_sizes.clear();
            registry->best_plans.clear();
            registry->pshard_disabled = true;
            registry->baseline_vram_req = baseline_vram;
            registry->budget_mib = variant.budget_mib;
            registry->cache_ubatch = variant.cache_ubatch;
            LLAMA_LOG_INFO("%s: loaded pshard_disabled variant budget=%u MiB cache_ubatch=%u baseline=%.1f MiB from %s\n",
                __func__, variant.budget_mib, variant.cache_ubatch, variant.baseline_vram_mib, cache_path);
            return true;
        }
    }

    const variant_data * best_whole = nullptr;
    for (const auto & variant : variants) {
        if (variant.pshard_disabled || variant.tiers.empty()) continue;
        if (!cache_ubatch_ok(variant)) continue;
        if (require_exact_budget) {
            if (variant.budget_mib != current_budget_mib) continue;
        } else if (pshard_mib_to_bytes(variant.budget_mib) > current_budget) {
            continue;
        }
        if (!best_whole || variant.budget_mib > best_whole->budget_mib) {
            best_whole = &variant;
        }
    }

    std::vector<std::pair<uint32_t, llama_pshard_plan>> selected;
    auto add_or_fill_plan = [&](const tier_data & td, bool allow_existing) {
        llama_pshard_plan plan = make_plan(td);
        auto it = std::find_if(selected.begin(), selected.end(),
            [&](const auto & p) { return p.first == td.bs; });
        if (it == selected.end()) {
            selected.push_back({td.bs, std::move(plan)});
        } else if (allow_existing || !it->second.is_viable ||
                (!best_whole && plan.is_viable && pshard_plan_is_better(plan, it->second))) {
            it->second = std::move(plan);
        }
    };

    if (best_whole) {
        for (const auto & td : best_whole->tiers) {
            add_or_fill_plan(td, true);
        }
    }

    if (!require_exact_budget) {
        for (const auto & variant : variants) {
            if (variant.pshard_disabled || pshard_mib_to_bytes(variant.budget_mib) <= current_budget) continue;
            if (!cache_ubatch_ok(variant)) continue;
            for (const auto & td : variant.tiers) {
                if (!td.viable || td.ot_line.empty()) continue;
                if (pshard_mib_to_bytes(td.vram_mib) > current_budget) continue;
                add_or_fill_plan(td, false);
            }
        }
    }

    selected.erase(std::remove_if(selected.begin(), selected.end(),
        [](const auto & p) { return p.first == 0; }), selected.end());
    if (selected.empty()) {
        if (require_exact_budget && !variants.empty()) {
            LLAMA_LOG_INFO("%s: cache miss, no exact budget=%u MiB variant in %s\n",
                __func__, current_budget_mib, cache_path);
        }
        if (skipped_cache_ubatch) {
            LLAMA_LOG_INFO("%s: cache miss, no variant with cache_ubatch=%u in %s\n",
                __func__, requested_cache_ubatch, cache_path);
        }
        return false;
    }

    std::sort(selected.begin(), selected.end(),
        [](const auto & a, const auto & b) { return a.first < b.first; });

    registry->tier_sizes.clear();
    registry->best_plans.clear();
    registry->pshard_disabled = false;
    registry->baseline_vram_req = 0;
    registry->budget_mib = best_whole ? best_whole->budget_mib : 0;
    registry->cache_ubatch = best_whole ? best_whole->cache_ubatch : requested_cache_ubatch;

    for (auto & item : selected) {
        const auto & p = item.second;
        if (p.is_viable && p.overrides.empty()) {
            LLAMA_LOG_WARN("%s: plan cache corrupt: tier bs=%u viable but has no overrides\n", __func__, item.first);
            registry->tier_sizes.clear();
            registry->best_plans.clear();
            return false;
        }
        registry->tier_sizes.push_back(item.first);
        registry->best_plans.push_back(std::move(item.second));
    }

    if (require_exact_budget) {
        LLAMA_LOG_INFO("%s: loaded %zu tier plans from exact budget=%u MiB cache_ubatch=%u variant in %s\n",
            __func__, registry->tier_sizes.size(), registry->budget_mib, registry->cache_ubatch, cache_path);
    } else if (best_whole) {
        LLAMA_LOG_INFO("%s: loaded %zu tier plans from budget=%u MiB cache_ubatch=%u variant for current budget=%u MiB in %s\n",
            __func__, registry->tier_sizes.size(), registry->budget_mib, registry->cache_ubatch, current_budget_mib, cache_path);
    } else {
        LLAMA_LOG_INFO("%s: loaded %zu salvaged tier plans for current budget=%u MiB from %s\n",
            __func__, registry->tier_sizes.size(), current_budget_mib, cache_path);
    }
    return true;
}

static bool pshard_plan_is_better(const llama_pshard_plan & candidate, const llama_pshard_plan & current) {
    if (!current.is_viable) return true;
    const bool candidate_has_tps = candidate.tps > 0.0f;
    const bool current_has_tps   = current.tps   > 0.0f;
    if (candidate_has_tps || current_has_tps) {
        if (candidate_has_tps != current_has_tps) return candidate_has_tps;
        if (candidate.tps != current.tps) return candidate.tps > current.tps;
    }
    if (candidate.n_pinned != current.n_pinned) return candidate.n_pinned > current.n_pinned;
    if (candidate.n_attn_pinned != current.n_attn_pinned) return candidate.n_attn_pinned > current.n_attn_pinned;
    if (candidate.overflow != current.overflow) return candidate.overflow > current.overflow;
    return candidate.total_vram_req < current.total_vram_req;
}

// true when every layer runs on the compute backend with no per layer override
// token_embd on host is allowed
// callers use this to skip pshard when baseline already fits
static bool pshard_plan_is_baseline_fit(
        const llama_pshard_plan & plan,
        uint32_t                  n_layers,
        int32_t                   compute_bid) {
    if (!plan.is_viable || plan.n_pinned < n_layers) return false;
    for (const auto & ov : plan.overrides) {
        // token_embd routes to CPU host buffer
        if (ov.pattern.find("token_embd") != std::string::npos) continue;
        if (ov.backend_id != compute_bid) return false;
    }
    return true;
}

// use attention priority when a forced strategy cannot fit the tier
// caller must set ctx.cparams for the target tier
static llama_pshard_plan llama_pshard_attn_pin_fallback(
        const llama_pshard_search_ctx & ctx,
        int      force_strategy,
        uint32_t hi_attn   = UINT32_MAX,
        uint32_t hi_pinned = UINT32_MAX) {
    llama_pshard_plan fallback = llama_pshard_search_attn_pin(ctx, hi_attn, hi_pinned);
    fallback.batch_size = ctx.cparams->n_batch;
    LLAMA_LOG_INFO("llama_params_fit_pshard: [bs=%-4u %-10s] forced %s non-viable, STATIC_ATTNPRIO_ALLMODELS fallback: n_pinned=%2u/%2u, %s\n",
        ctx.cparams->n_batch, llama_pshard_strategy_name(LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS),
        llama_pshard_strategy_name((llama_pshard_strategy)force_strategy),
        fallback.n_pinned, fallback.n_attn_pinned,
        fallback.is_viable ? "VIABLE" : "NOT VIABLE");
    return fallback;
}

static llama_pshard_plan llama_pshard_search_tier(
        const llama_pshard_search_ctx & ctx,
        int force_strategy,
        const std::vector<llama_device_memory_data> & dmds,
        llama_pshard_tier_prune & prune) {

    const auto * path_model = ctx.path_model;
    const auto * mparams    = ctx.mparams;
    const auto * cparams    = ctx.cparams;
    auto * tensor_buft_overrides = ctx.overrides;
    const auto   n_layers   = ctx.n_layers;
    const auto   vram_free  = ctx.vram_free;
    const auto   gpu_buft   = ctx.gpu_buft;
    const auto   host_buft  = ctx.host_buft;
    const auto & layout     = ctx.layout;
    const auto   is_moe     = ctx.is_moe;

    llama_pshard_plan best;

    if ((force_strategy < 0 || force_strategy == 0) && !prune.skip[0]) {
        llama_pshard_plan plan;
        llama_model_params mp_copy = *mparams;
        mp_copy.pshard = false;
        mp_copy.tensor_buft_overrides = nullptr;
        llama_context_params cp_copy = *cparams;
        cp_copy.pshard = false;
        float ts[16] = {};
        size_t margins[16] = {};
        margins[0] = dmds[0].free > vram_free ? dmds[0].free - vram_free : 0;
        tensor_buft_overrides[0] = { nullptr, nullptr, -1 };
        try {
            llama_params_fit_impl(path_model, &mp_copy, &cp_copy, ts, tensor_buft_overrides, margins, 0,
                GGML_LOG_LEVEL_ERROR, /*fill_front_to_back=*/true);
            plan.strategy = LLAMA_PSHARD_STATIC_FITPARAMS_DENSEPRIO_MOEONLY;
            const uint32_t ngl = (mp_copy.n_gpu_layers < 0)
                ? (n_layers + 1)
                : std::min((uint32_t)mp_copy.n_gpu_layers, n_layers + 1);
            plan.n_pinned      = (ngl > 0) ? (ngl - 1) : 0;
            plan.is_viable     = true;
            plan.pin_from_back = false;
            plan.output_on_gpu = (ngl > 0);

            if (tensor_buft_overrides[0].pattern == nullptr) {
                llama_pshard_generate_overrides(plan.n_pinned, n_layers, gpu_buft, host_buft,
                    tensor_buft_overrides, LLAMA_LAYER_FRACTION_NONE, LLAMA_PSHARD_STATIC_FITPARAMS_DENSEPRIO_MOEONLY, layout,
                    plan.pin_from_back, plan.output_on_gpu);
            } else {
                for (size_t i = 0; tensor_buft_overrides[i].pattern; i++) {
                    if (tensor_buft_overrides[i].buft == gpu_buft) {
                        tensor_buft_overrides[i].backend_id = layout.compute;
                    } else {
                        tensor_buft_overrides[i].buft       = host_buft;
                        tensor_buft_overrides[i].backend_id = layout.cpu;
                    }
                }

                {
                    thread_local std::string pat_output   = "^output";
                    thread_local std::string pat_tok_embd = "^token_embd";
                    const int32_t out_bid = plan.output_on_gpu ? layout.compute : layout.cpu;
                    size_t n_ov = 0;
                    while (tensor_buft_overrides[n_ov].pattern) n_ov++;
                    for (size_t i = n_ov + 2; i >= 2; i--) {
                        tensor_buft_overrides[i] = tensor_buft_overrides[i - 2];
                    }
                    tensor_buft_overrides[0] = { pat_output.c_str(),   host_buft, out_bid };
                    tensor_buft_overrides[1] = { pat_tok_embd.c_str(), host_buft, layout.cpu };
                }
            }

            for (const auto * ov = tensor_buft_overrides; ov->pattern; ++ov) {
                plan.overrides.push_back({ov->pattern, ov->buft, ov->backend_id});
            }

            mp_copy = *mparams;
            mp_copy.pshard = false;
            mp_copy.n_gpu_layers = plan.n_pinned + 1;
            mp_copy.tensor_buft_overrides = tensor_buft_overrides;
            llama_pshard_tps_hook_data tps_data = { ctx.predictor, layout.cpu, ctx.kv_size, (int32_t)cparams->n_batch, cparams->n_seq_max, ctx.has_rs, &plan.tps };
            auto * hook     = ctx.predictor ? pshard_tps_probe_hook : nullptr;
            auto * hookdata = ctx.predictor ? (void *)&tps_data     : nullptr;

            const auto d = llama_pshard_probe_memory(ctx, mp_copy, *cparams, GGML_LOG_LEVEL_ERROR, hook, hookdata);
            plan.total_vram_req   = d[0].mb.total();
            plan.scratch_measured = d[0].mb.compute;
            plan.cache_measured   = d[0].mb.context;
            plan.is_viable = ((int64_t)plan.total_vram_req <= (int64_t)vram_free);
        } catch (const std::exception & e) {
            LLAMA_LOG_ERROR("%s: STATIC_FITPARAMS_DENSEPRIO_MOEONLY probe failed (n_pinned=%u, bs=%u): %s\n",
                __func__, plan.n_pinned, cparams->n_batch, e.what());
            plan.is_viable = false;
        } catch (...) {
            LLAMA_LOG_ERROR("%s: STATIC_FITPARAMS_DENSEPRIO_MOEONLY probe failed (n_pinned=%u, bs=%u): unknown exception\n",
                __func__, plan.n_pinned, cparams->n_batch);
            plan.is_viable = false;
        }

        plan.batch_size = cparams->n_batch;

        {
            char tps_buf[32] = "";
            if (plan.tps > 0.0f) { snprintf(tps_buf, sizeof(tps_buf), ", tps=%.1f", plan.tps); }
            LLAMA_LOG_INFO("%s: [bs=%-4u %-10s] n_pinned=%2u, overflow=%-4s, vram=%7.1f MiB, %s%s\n",
                __func__, cparams->n_batch, llama_pshard_strategy_name(LLAMA_PSHARD_STATIC_FITPARAMS_DENSEPRIO_MOEONLY),
                plan.n_pinned, PSHARD_FRAC_NAMES[plan.overflow],
                plan.total_vram_req / (1024.0 * 1024.0),
                plan.is_viable ? "VIABLE" : "NOT VIABLE", tps_buf);
        }

        prune.update(0, plan);

        if (plan.is_viable && pshard_plan_is_better(plan, best)) {
            best = plan;
        }
        // skip the other strategies when baseline already fits
        // the caller will disable pshard for that case
        if (force_strategy < 0 && pshard_plan_is_baseline_fit(plan, n_layers, layout.compute)) {
            return best;
        }
    }

    for (int s = 1; s < LLAMA_PSHARD_COUNT; s++) {
        if (force_strategy >= 0 && force_strategy != s) continue;
        if (prune.skip[s]) continue;

        llama_pshard_strategy strategy = (llama_pshard_strategy)s;
        llama_pshard_plan plan;

        if (strategy == LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS) {
            plan = llama_pshard_search_attn_pin(ctx, prune.hi_attn, prune.hi_pinned[s]);
        } else {
            plan = llama_pshard_search_strategy(ctx, strategy, prune.hi_pinned[s]);
        }

        plan.batch_size = cparams->n_batch;

        {
            const char * status = plan.is_viable ? "VIABLE" : "NOT VIABLE";
            char tps_buf[32] = "";
            if (plan.tps > 0.0f) { snprintf(tps_buf, sizeof(tps_buf), ", tps=%.1f", plan.tps); }

            if (plan.n_attn_pinned > 0) {
                LLAMA_LOG_INFO("%s: [bs=%-4u %-10s] n_pinned=%2u (attn=%2u), overflow=%-4s, vram=%7.1f MiB, %s%s\n",
                    __func__, cparams->n_batch, llama_pshard_strategy_name(strategy),
                    plan.n_pinned, plan.n_attn_pinned, PSHARD_FRAC_NAMES[plan.overflow],
                    plan.total_vram_req / (1024.0 * 1024.0), status, tps_buf);
            } else {
                LLAMA_LOG_INFO("%s: [bs=%-4u %-10s] n_pinned=%2u, overflow=%-4s, vram=%7.1f MiB, %s%s\n",
                    __func__, cparams->n_batch, llama_pshard_strategy_name(strategy),
                    plan.n_pinned, PSHARD_FRAC_NAMES[plan.overflow],
                    plan.total_vram_req / (1024.0 * 1024.0), status, tps_buf);
            }
        }

        prune.update(s, plan);

        if (plan.is_viable && pshard_plan_is_better(plan, best)) {
            best = plan;
        }
    }

    if (!best.is_viable && force_strategy >= 0 && force_strategy != LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS) {
        llama_pshard_plan fallback = llama_pshard_attn_pin_fallback(
            ctx, force_strategy, prune.hi_attn, prune.hi_pinned[LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS]);
        prune.update(LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS, fallback);
        if (fallback.is_viable) {
            best = fallback;
        }
    }

    return best;
}

// probe one strategy across tiers
static void llama_pshard_strategy_sweep(
        int strategy,
        const llama_pshard_search_ctx & ctx_template,
        const llama_context_params   & cparams_base,
        const std::vector<llama_device_memory_data> & dmds,
        const llama_pshard_plan_registry & registry,
        int force_strategy,
        llama_pshard_plan * out_plans,   // write out_plans[tier]
        size_t n_tiers) {

    if (force_strategy >= 0 && force_strategy != strategy) return;

    llama_model_tensor_buft_override local_overrides[4096];
    llama_pshard_search_ctx ctx = ctx_template;
    ctx.overrides = local_overrides;

    llama_pshard_tier_prune prune;
    prune.init(ctx.n_layers);

    // skip the strategy if the largest tier failed
    const auto & worst = out_plans[n_tiers - 1];
    if (worst.is_viable && worst.n_pinned > 0 && worst.strategy == (llama_pshard_strategy)strategy) {
        prune.update(strategy, worst);
    }

    // smaller tiers start from the previous fit
    // smaller batches usually need less scratch
    // hybrid SSM graphs are not monotonic in batch size
    uint32_t prev_n_pinned = worst.is_viable ? worst.n_pinned : 0;

    for (int t = (int)n_tiers - 2; t >= 0; t--) {
        if (prune.skip[strategy]) break;

        llama_context_params cp_tier = cparams_base;
        cp_tier.n_batch  = registry.tier_sizes[t];
        cp_tier.n_ubatch = cp_tier.n_batch;
        ctx.cparams = &cp_tier;

        llama_pshard_strategy strat = (llama_pshard_strategy)strategy;
        llama_pshard_plan plan;

        const uint32_t lo_hint = ctx.has_rs ? 0 : prev_n_pinned;

        if (strategy == 0) {
            llama_pshard_tier_prune none_prune;
            none_prune.init(ctx.n_layers);
            for (int s = 1; s < LLAMA_PSHARD_COUNT; s++) none_prune.skip[s] = true;
            plan = llama_pshard_search_tier(ctx, 0, dmds, none_prune);
        } else if (strat == LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS) {
            plan = llama_pshard_search_attn_pin(ctx, prune.hi_attn, UINT32_MAX, lo_hint);
            plan.batch_size = cp_tier.n_batch;
        } else {
            plan = llama_pshard_search_strategy(ctx, strat, UINT32_MAX, lo_hint);
            plan.batch_size = cp_tier.n_batch;
        }

        if (plan.is_viable && plan.n_pinned > prev_n_pinned) {
            prev_n_pinned = plan.n_pinned;
        }

        {
            const char * status = plan.is_viable ? "VIABLE" : "NOT VIABLE";
            char tps_buf[32] = "";
            if (plan.tps > 0.0f) { snprintf(tps_buf, sizeof(tps_buf), ", tps=%.1f", plan.tps); }

            if (plan.n_attn_pinned > 0) {
                LLAMA_LOG_INFO("%s: [bs=%-4u %-10s] n_pinned=%2u (attn=%2u), overflow=%-4s, vram=%7.1f MiB, %s%s\n",
                    __func__, cp_tier.n_batch, llama_pshard_strategy_name(strat),
                    plan.n_pinned, plan.n_attn_pinned, PSHARD_FRAC_NAMES[plan.overflow],
                    plan.total_vram_req / (1024.0 * 1024.0), status, tps_buf);
            } else {
                LLAMA_LOG_INFO("%s: [bs=%-4u %-10s] n_pinned=%2u, overflow=%-4s, vram=%7.1f MiB, %s%s\n",
                    __func__, cp_tier.n_batch, llama_pshard_strategy_name(strat),
                    plan.n_pinned, PSHARD_FRAC_NAMES[plan.overflow],
                    plan.total_vram_req / (1024.0 * 1024.0), status, tps_buf);
            }
        }

        out_plans[t] = plan;
    }
}

static void llama_pshard_parallel_worker(
        std::atomic<int> & next_strategy,
        const llama_pshard_search_ctx & ctx_template,
        const llama_context_params   & cparams_base,
        const std::vector<llama_device_memory_data> & dmds,
        const llama_pshard_plan_registry & registry,
        int force_strategy,
        llama_pshard_plan * all_plans,
        size_t n_tiers) {

    while (true) {
        int s = next_strategy.fetch_add(1);
        if (s >= LLAMA_PSHARD_COUNT) return;

        llama_pshard_plan * out = all_plans + s * n_tiers;
        llama_pshard_strategy_sweep(s, ctx_template, cparams_base, dmds, registry, force_strategy, out, n_tiers);
    }
}

static bool llama_pshard_params_supported(
        const struct llama_model_params * mparams,
        const struct llama_context_params * cparams) {
    const llama_model_params default_mparams = llama_model_default_params();

    auto disable = [](const char * reason) {
        LLAMA_LOG_WARN("%s: %s, disabling pshard\n", "llama_params_fit_pshard", reason);
        return false;
    };

    if (!cparams->offload_kqv) {
        return disable("offload_kqv=false is not supported");
    }
    if (mparams->split_mode == LLAMA_SPLIT_MODE_TENSOR) {
        return disable("SPLIT_MODE_TENSOR is not supported");
    }
    if (mparams->split_mode == LLAMA_SPLIT_MODE_ROW) {
        return disable("SPLIT_MODE_ROW is not supported");
    }
    if (mparams->n_gpu_layers != default_mparams.n_gpu_layers) {
        return disable("n_gpu_layers is already set by the user");
    }
    if (mparams->tensor_split) {
        for (size_t i = 0; i < llama_max_devices(); i++) {
            if (mparams->tensor_split[i] != 0.0f) {
                return disable("tensor_split is already set by the user");
            }
        }
    }
    if (mparams->tensor_buft_overrides &&
        (mparams->tensor_buft_overrides->pattern || mparams->tensor_buft_overrides->buft)) {
        return disable("tensor_buft_overrides are already set by the user");
    }

    return true;
}

void llama_params_fit_pshard(
        const char                              * path_model,
        struct llama_model_params               * mparams,
        struct llama_context_params             * cparams,
        struct llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t                                    max_vram_mb) {
    const int64_t t0_us = llama_time_us();

    if (!llama_pshard_params_supported(mparams, cparams)) {
        mparams->pshard = false;
        cparams->pshard = false;
        return;
    }

    mparams->pshard = true;
    cparams->pshard = true;

    // step 1: probe device memory and model parameters
    std::vector<llama_device> devs;
    uint32_t hp_ngl = 0, hp_nct = 0, hp_nex = 0, hp_nr = 0;

    llama_model_params mparams_probe = *mparams;
    mparams_probe.pshard = false;
    const auto dmds = llama_get_device_memory_data(
        path_model, &mparams_probe, cparams, devs, hp_ngl, hp_nct, hp_nex, hp_nr, GGML_LOG_LEVEL_ERROR);

    if (devs.empty()) {
        LLAMA_LOG_ERROR("%s: no GPU devices found\n", __func__);
        return;
    }

    const uint32_t n_layers  = hp_ngl;
    const size_t   vram_free = (max_vram_mb > 0) ? max_vram_mb * 1024ULL * 1024ULL : dmds[0].free;

    if (mparams->max_vram_alloc == 0) {
        mparams->max_vram_alloc = (uint32_t)(vram_free / (1024 * 1024));
    }

    LLAMA_LOG_INFO("%s: probing pshard plans: %u layers, %.1f MiB VRAM free\n",
        __func__, n_layers, vram_free / (1024.0 * 1024.0));

    // step 2: read forced strategy from env (PSHARD_STRATEGY)
    const int force_strategy = pshard_strategy_from_env();
    if (force_strategy >= 0) {
        LLAMA_LOG_INFO("%s: forcing strategy %s (PSHARD_STRATEGY=%s)\n",
            __func__, llama_pshard_strategy_name((llama_pshard_strategy)force_strategy),
            getenv("PSHARD_STRATEGY"));
    } else if (getenv("PSHARD_STRATEGY")) {
        LLAMA_LOG_WARN("%s: invalid PSHARD_STRATEGY='%s', ignoring\n",
            __func__, getenv("PSHARD_STRATEGY"));
    }

    // step 3: derive layout, buftypes, optional benchmark predictor, search ctx
    ggml_backend_buffer_type_t gpu_buft  = ggml_backend_dev_buffer_type(devs[0].dev);
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(devs[0].dev);
    if (!host_buft) {
        host_buft = ggml_backend_cpu_buffer_type();
    }
    const int32_t           cpu_bid = pshard_dev_layout::compute_cpu_backend_id(devs.size());
    const pshard_dev_layout layout  = pshard_dev_layout::for_device(0, cpu_bid);

    std::unique_ptr<llama_benchmark_predictor> predictor;
    {
        const char * env_cpu  = getenv("PSHARD_CPU_PROFILE");
        const char * env_gpu  = getenv("PSHARD_GPU_PROFILE");
        const char * cpu_path = env_cpu ? env_cpu : "cpu_profile.txt";
        const char * gpu_path = env_gpu ? env_gpu : "gpu_profile.txt";

        auto p = std::make_unique<llama_benchmark_predictor>();
        const bool has_cpu = p->load_cpu(cpu_path, cparams->n_threads);
        const bool has_gpu = p->load_gpu(gpu_path);
        if (has_cpu || has_gpu) {
            predictor = std::move(p);
            LLAMA_LOG_INFO("%s: benchmark predictor loaded (cpu=%s gpu=%s)\n",
                __func__, has_cpu ? "yes" : "no", has_gpu ? "yes" : "no");
        }
    }

    llama_pshard_search_ctx ctx = {
        path_model, mparams, cparams, tensor_buft_overrides,
        n_layers, vram_free, gpu_buft, host_buft, layout,
        /*is_moe=*/(hp_nex > 0), /*has_rs=*/(hp_nr > 0),
        predictor.get(), cparams->n_ctx, 0,
    };

    // step 4: registry lookup -- try to load <model>.tensor_overrides.pshard_registry; merge cached plans by tier
    const std::string cache_path = std::string(path_model) + ".tensor_overrides.pshard_registry";

    int64_t model_file_size = 0;
    if (FILE * mf = fopen(path_model, "rb")) {
#ifdef _WIN32
        _fseeki64(mf, 0, SEEK_END);
        model_file_size = _ftelli64(mf);
#else
        fseeko(mf, 0, SEEK_END);
        model_file_size = ftello(mf);
#endif
        fclose(mf);
    }
    const uint64_t fp = pshard_registry_fingerprint(mparams, cparams, model_file_size);

    llama_pshard_plan_registry * registry  = mparams->pshard_registry;
    bool                         needs_probe = true;

    if (registry) {
        registry->budget_mib = pshard_bytes_to_mib_ceil(vram_free);
        registry->cache_ubatch = registry->tier_sizes.empty() ? 0 : registry->tier_sizes.back();
        ctx.cache_ubatch = registry->cache_ubatch;

        // save requested tiers before load (load overwrites tier_sizes)
        std::vector<uint32_t> requested_tiers = registry->tier_sizes;

        llama_pshard_plan_registry cached;
        cached.budget_mib = registry->budget_mib;
        cached.cache_ubatch = registry->cache_ubatch;
        if (!mparams->pshard_cache_skip_load &&
            pshard_registry_load(&cached, fp, cache_path.c_str(), host_buft, vram_free, true)) {
            // merge cached plans into registry by matching batch size
            std::unordered_map<uint32_t, llama_pshard_plan> cache_map;
            for (size_t i = 0; i < cached.tier_sizes.size(); i++) {
                cache_map[cached.tier_sizes[i]] = cached.best_plans[i];
            }

            registry->tier_sizes = requested_tiers;
            registry->best_plans.resize(requested_tiers.size());
            registry->pshard_disabled = cached.pshard_disabled;
            registry->baseline_vram_req = cached.baseline_vram_req;
            registry->budget_mib = cached.budget_mib;
            registry->cache_ubatch = cached.cache_ubatch;

            size_t n_hit = 0;
            for (size_t i = 0; i < requested_tiers.size(); i++) {
                auto it = cache_map.find(requested_tiers[i]);
                if (it != cache_map.end()) {
                    registry->best_plans[i] = it->second;
                    n_hit++;
                }
            }

            if (n_hit == requested_tiers.size()) {
                needs_probe = false;
                LLAMA_LOG_INFO("%s: loaded all %zu tiers from exact budget=%u MiB cache variant\n",
                    __func__, n_hit, registry->budget_mib);
            } else if (n_hit > 0) {
                LLAMA_LOG_INFO("%s: loaded %zu/%zu tiers from exact budget=%u MiB cache variant, %zu need probing\n",
                    __func__, n_hit, requested_tiers.size(), registry->budget_mib, requested_tiers.size() - n_hit);
            }
        }

        // common setup for cache hit and miss
    }

    // step 4b: skip pshard when a cached baseline variant fits this budget
    if (registry && registry->pshard_disabled) {
        LLAMA_LOG_INFO("%s: cache says baseline %.1f MiB fits this budget (variant budget=%u MiB), using baseline loading\n",
            __func__, registry->baseline_vram_req / (1024.0 * 1024.0), registry->budget_mib);
        mparams->pshard = false;
        cparams->pshard = false;
        mparams->n_gpu_layers = n_layers + 1;
        tensor_buft_overrides[0] = { nullptr, nullptr, -1 };
        mparams->tensor_buft_overrides = nullptr;

        const int64_t t1_us = llama_time_us();
        LLAMA_LOG_INFO("%s: best strategy: baseline (cached), all %u layers on GPU, took %.2f s\n",
            __func__, n_layers, (t1_us - t0_us) * 1e-6);
        return;
    }

    // step 5: probe tiers largest first and skip pshard when baseline already fits
    if (registry && needs_probe) {
        const size_t n_tiers = registry->tier_sizes.size();

        // step 5a: probe largest tier serially
        {
            const size_t t = n_tiers - 1;

            if (registry->best_plans[t].is_viable) {
                LLAMA_LOG_INFO("%s: === tier %zu (bs=%u) [cached] ===\n", __func__, t, registry->tier_sizes[t]);
            } else {
                llama_pshard_tier_prune prune_worst;
                prune_worst.init(n_layers);

                const llama_context_params * saved = ctx.cparams;
                llama_context_params cp_tier = *cparams;
                cp_tier.n_batch  = registry->tier_sizes[t];
                cp_tier.n_ubatch = cp_tier.n_batch;
                ctx.cparams = &cp_tier;

                LLAMA_LOG_INFO("%s: === tier %zu (bs=%u) [worst-case first] ===\n", __func__, t, registry->tier_sizes[t]);
                registry->best_plans[t] = llama_pshard_search_tier(ctx, force_strategy, dmds, prune_worst);

                ctx.cparams = saved;
            }

            // bail out when the largest tier fully fits on the compute backend
            // the standard loader is enough in that case
            const auto & worst = registry->best_plans[t];
            if (pshard_plan_is_baseline_fit(worst, n_layers, layout.compute)) {
                const size_t actual_mb = (worst.total_vram_req + 1024*1024 - 1) / (1024 * 1024);

                LLAMA_LOG_INFO("%s: all layers fit at largest tier in %zu MiB -- using baseline loading (no pshard needed)\n",
                    __func__, actual_mb);

                if (actual_mb < (size_t)mparams->max_vram_alloc) {
                    LLAMA_LOG_INFO("%s: baseline requires %zu MiB under budget %u MiB\n",
                        __func__, actual_mb, registry->budget_mib);
                }

                // save the baseline decision for later runs
                registry->pshard_disabled = true;
                registry->baseline_vram_req = worst.total_vram_req;
                pshard_registry_save(registry, fp, cache_path.c_str(), host_buft, cparams);

                mparams->pshard = false;
                cparams->pshard = false;
                mparams->n_gpu_layers = n_layers + 1;
                tensor_buft_overrides[0] = { nullptr, nullptr, -1 };
                mparams->tensor_buft_overrides = nullptr;

                const int64_t t1_us = llama_time_us();
                LLAMA_LOG_INFO("%s: best strategy: %s (baseline), all %u layers on GPU, took %.2f s\n",
                    __func__, llama_pshard_strategy_name(worst.strategy), n_layers, (t1_us - t0_us) * 1e-6);
                return;
            }
        }

        // step 5b: probe remaining tiers
        {
            std::vector<llama_pshard_plan> all_plans(LLAMA_PSHARD_COUNT * n_tiers);
            for (int s = 0; s < LLAMA_PSHARD_COUNT; s++) {
                all_plans[s * n_tiers + (n_tiers - 1)] = registry->best_plans[n_tiers - 1];
            }

            int n_workers = std::min((int)cparams->n_threads, (int)LLAMA_PSHARD_COUNT);
            n_workers = std::max(n_workers, 1);

            // baseline is serial because llama_params_fit_impl is not thread safe
            llama_pshard_strategy_sweep(0, ctx, *cparams, dmds, *registry, force_strategy,
                all_plans.data() + 0 * n_tiers, n_tiers);

            int n_pshard_strategies = LLAMA_PSHARD_COUNT - 1;
            n_workers = std::min(n_workers, n_pshard_strategies);

            LLAMA_LOG_INFO("%s: parallel planning with %d workers (%d pshard strategies)\n",
                __func__, n_workers, n_pshard_strategies);

            std::atomic<int> next_strategy{1};

            if (n_workers <= 1) {
                llama_pshard_parallel_worker(next_strategy, ctx, *cparams, dmds,
                    *registry, force_strategy, all_plans.data(), n_tiers);
            } else {
                std::vector<std::thread> threads;
                for (int w = 1; w < n_workers; w++) {
                    threads.emplace_back(llama_pshard_parallel_worker,
                        std::ref(next_strategy), std::cref(ctx), std::cref(*cparams),
                        std::cref(dmds), std::cref(*registry), force_strategy,
                        all_plans.data(), n_tiers);
                }
                llama_pshard_parallel_worker(next_strategy, ctx, *cparams, dmds,
                    *registry, force_strategy, all_plans.data(), n_tiers);

                for (auto & t : threads) t.join();
            }

            for (size_t t = 0; t < n_tiers - 1; t++) {
                llama_pshard_plan best;
                for (int s = 0; s < LLAMA_PSHARD_COUNT; s++) {
                    auto & p = all_plans[s * n_tiers + t];
                    if (p.is_viable && pshard_plan_is_better(p, best)) {
                        best = p;
                    }
                }
                registry->best_plans[t] = best;
            }

            // step 5c: try attention priority fallback for forced strategies that cannot fit
            // forced strategy sweeps can leave smaller tiers unfilled
            if (force_strategy >= 0 && force_strategy != LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS) {
                for (size_t t = 0; t < n_tiers - 1; t++) {
                    if (registry->best_plans[t].is_viable) continue;

                    const llama_context_params * saved = ctx.cparams;
                    llama_context_params cp_tier = *cparams;
                    cp_tier.n_batch  = registry->tier_sizes[t];
                    cp_tier.n_ubatch = cp_tier.n_batch;
                    ctx.cparams = &cp_tier;

                    llama_pshard_plan fallback = llama_pshard_attn_pin_fallback(ctx, force_strategy);
                    if (fallback.is_viable) {
                        registry->best_plans[t] = fallback;
                    }

                    ctx.cparams = saved;
                }
            }
        }

        pshard_registry_save(registry, fp, cache_path.c_str(), host_buft, cparams);
    }

    // step 6: pick the active plan (default tier = largest) and resolve best_plan
    if (registry) {
        const size_t default_tier = registry->tier_sizes.size() - 1;
        if (auto * best = registry->get_best(default_tier)) {
            registry->active_plan = best;
        }
    }

    llama_pshard_plan best_plan;
    if (registry && registry->active_plan) {
        best_plan = *registry->active_plan;
    } else {
        llama_pshard_tier_prune prune_single;
        prune_single.init(n_layers);
        best_plan = llama_pshard_search_tier(ctx, force_strategy, dmds, prune_single);
    }

    // step 7: fall back to all cpu when no plan is viable
    if (!best_plan.is_viable) {
        LLAMA_LOG_WARN("%s: no viable plan found, falling back to STATIC_ATTNPRIO_ALLMODELS with n_pinned=0\n", __func__);
        llama_pshard_generate_overrides(0, n_layers, gpu_buft, host_buft, tensor_buft_overrides,
            LLAMA_LAYER_FRACTION_NONE, LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS, layout,
            /*pin_from_back=*/false, /*output_on_gpu=*/false, /*n_attn_pinned=*/0);
        mparams->n_gpu_layers = n_layers + 1;
        mparams->tensor_buft_overrides = tensor_buft_overrides;
        return;
    }

    // step 8: apply best plan to tensor_buft_overrides
    if (best_plan.strategy == LLAMA_PSHARD_STATIC_FITPARAMS_DENSEPRIO_MOEONLY) {
        thread_local std::vector<std::string> shard_none_patterns;
        shard_none_patterns.clear();
        shard_none_patterns.reserve(best_plan.overrides.size());
        for (const auto & ov : best_plan.overrides) {
            shard_none_patterns.push_back(ov.pattern);
        }
        for (size_t i = 0; i < best_plan.overrides.size(); i++) {
            tensor_buft_overrides[i].pattern    = shard_none_patterns[i].c_str();
            tensor_buft_overrides[i].buft       = best_plan.overrides[i].buft;
            tensor_buft_overrides[i].backend_id = best_plan.overrides[i].backend_id;
        }
        tensor_buft_overrides[best_plan.overrides.size()] = { nullptr, nullptr, -1 };
    } else {
        llama_pshard_generate_overrides(best_plan.n_pinned, n_layers, gpu_buft, host_buft,
            tensor_buft_overrides, (llama_layer_fraction)best_plan.overflow, best_plan.strategy, layout,
            best_plan.pin_from_back, best_plan.output_on_gpu, best_plan.n_attn_pinned);
    }

    for (size_t i = 0; tensor_buft_overrides[i].pattern; i++) {
        if (tensor_buft_overrides[i].backend_id == layout.compute) {
            tensor_buft_overrides[i].buft = host_buft;
        }
    }

    mparams->n_gpu_layers = n_layers + 1;
    mparams->tensor_buft_overrides = tensor_buft_overrides;

    const int64_t t1_us = llama_time_us();
    LLAMA_LOG_INFO("%s: best strategy: %s, n_pinned=%u, n_attn=%u/%u%s%s, took %.2f s\n",
        __func__, llama_pshard_strategy_name(best_plan.strategy),
        best_plan.n_pinned, best_plan.n_attn_pinned, n_layers,
        best_plan.overflow ? " (partial: " : "",
        best_plan.overflow ? PSHARD_FRAC_NAMES[best_plan.overflow] : "",
        (t1_us - t0_us) * 1e-6);

    {
        int n_bid0 = 0, n_total = 0;
        for (const auto * ov = tensor_buft_overrides; ov->pattern; ++ov) {
            n_total++;
            if (ov->backend_id == layout.compute) n_bid0++;
            LLAMA_LOG_DEBUG("%s:   override: %-25s -> %-15s backend_id=%d\n",
                __func__, ov->pattern, ggml_backend_buft_name(ov->buft), ov->backend_id);
        }
        LLAMA_LOG_INFO("%s: %d overrides, %d with bid=%d (compute)\n",
            __func__, n_total, n_bid0, layout.compute);
    }
}
