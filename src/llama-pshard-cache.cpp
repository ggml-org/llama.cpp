#include "llama-pshard-plan.h"
#include "llama-impl.h"

#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

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

void llama_pshard_generate_overrides(
        uint32_t n_pinned,
        uint32_t n_layers,
        ggml_backend_buffer_type_t gpu_buft,
        ggml_backend_buffer_type_t host_buft,
        struct llama_model_tensor_buft_override * tensor_buft_overrides,
        llama_layer_fraction overflow_type,
        llama_pshard_strategy strategy,
        const pshard_dev_layout & layout,
        bool pin_from_back,
        bool output_on_gpu,
        uint32_t n_attn_pinned) {
    GGML_UNUSED(gpu_buft);

    thread_local std::array<std::string, 1000> patterns_layer;
    thread_local std::array<std::string, 1000> patterns_layer_attn;
    thread_local std::array<std::string, 1000> patterns_layer_ffn;
    thread_local std::string pat_output = "^output";

    const uint32_t il_pin_start = pin_from_back ? (n_layers - n_pinned) : 0;
    GGML_ASSERT(n_layers <= 1000);
    const uint32_t il_pin_end   = pin_from_back ? n_layers : n_pinned;
    const uint32_t il_boundary_raw = pin_from_back ? (il_pin_start > 0 ? il_pin_start - 1 : UINT32_MAX) : il_pin_end;
    const uint32_t il_boundary = (overflow_type != LLAMA_LAYER_FRACTION_NONE && il_boundary_raw < n_layers) ? il_boundary_raw : UINT32_MAX;
    const bool output_on_cpu = !output_on_gpu;

    size_t itbo = 0;

    auto emit = [&](const char * pat, ggml_backend_buffer_type_t buft, int32_t bid) {
        tensor_buft_overrides[itbo] = { pat, buft, bid };
        itbo++;
    };

    {
        thread_local std::string pat_tok_embd = "^token_embd";
        const int32_t out_bid = output_on_cpu ? layout.cpu : layout.compute;
        emit(pat_output.c_str(), host_buft, out_bid);
        emit(pat_tok_embd.c_str(), host_buft, layout.cpu);
    }

    for (uint32_t il = 0; il < n_layers; il++) {
        if (patterns_layer[il].empty())      { patterns_layer[il]      = "blk\\." + std::to_string(il) + "\\..*"; }
        if (patterns_layer_attn[il].empty()) { patterns_layer_attn[il] = "blk\\." + std::to_string(il) + "\\.attn_(q|k|v|output|q_norm|k_norm).*"; }
        if (patterns_layer_ffn[il].empty())  { patterns_layer_ffn[il]  = "blk\\." + std::to_string(il) + "\\.ffn_((up|gate|down)\\.|(up|down|gate|gate_up)_(ch|)exps).*"; }

        if (il == il_boundary) {
            const char * overflow_pat = llama_get_overflow_pattern(il, overflow_type);
            if (overflow_pat) {
                emit(overflow_pat, host_buft, layout.shard(il));
            }
            emit(patterns_layer[il].c_str(), host_buft, layout.compute);
        } else if (il >= il_pin_start && il < il_pin_end) {
            emit(patterns_layer[il].c_str(), host_buft, layout.compute);
        } else {
            const bool    use_alternating_shards = strategy == LLAMA_PSHARD_GPUONLY_LAYERPIN_LAYERSTREAM;
            const int32_t shard_bid = use_alternating_shards ? layout.shard(il) : layout.shard_a;
            switch (strategy) {
                case LLAMA_PSHARD_GPUONLY_LAYERPIN_LAYERSTREAM:
                    emit(patterns_layer[il].c_str(), host_buft, shard_bid);
                    break;
                case LLAMA_PSHARD_GPUONLY_ATTNPIN_FFNSTREAM:
                    emit(patterns_layer_ffn[il].c_str(), host_buft, shard_bid);
                    emit(patterns_layer[il].c_str(), host_buft, layout.compute);
                    break;
                case LLAMA_PSHARD_DYNAMIC_FFNCPU_ATTNSTREAM:
                    emit(patterns_layer_ffn[il].c_str(), host_buft, layout.cpu);
                    emit(patterns_layer[il].c_str(), host_buft, shard_bid);
                    break;
                case LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS:
                    if (n_attn_pinned > 0 && il < n_attn_pinned) {
                        emit(patterns_layer_ffn[il].c_str(), host_buft, layout.cpu);
                        emit(patterns_layer[il].c_str(), host_buft, layout.compute);
                    } else {
                        emit(patterns_layer[il].c_str(), host_buft, layout.cpu);
                    }
                    break;
                default: break;
            }
        }
    }
    tensor_buft_overrides[itbo] = { nullptr, nullptr, -1 };
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
        llama_pshard_strategy strategy = LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS;
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
            if (in_section) break;
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
            }
            continue;
        }

        if (s.compare(0, 5, "[tier") == 0) {
            tier_data td = {};
            size_t tier_idx = 0;
            if (s.find("not_viable") != std::string::npos) {
                if (sscanf(s.c_str(), "[tier %zu bs=%u]", &tier_idx, &td.bs) < 2) continue;
                td.viable = false;
            } else {
                if (sscanf(s.c_str(), "[tier %zu bs=%u]", &tier_idx, &td.bs) < 2) continue;
                td.viable = true;
            }
            cur_variant->tiers.push_back(td);
        } else if (s.compare(0, 9, "strategy=") == 0 && !cur_variant->tiers.empty()) {
            auto & td = cur_variant->tiers.back();
            char strat_name[64] = {}, overflow_name_buf[16] = {};
            if (sscanf(s.c_str(), "strategy=%63s n_pinned=%u n_attn_pinned=%u overflow=%15s tps=%f vram=%lf",
                   strat_name, &td.n_pinned, &td.n_attn_pinned, overflow_name_buf, &td.tps, &td.vram_mib) < 4) {
                td.viable = false;
                continue;
            }

            const char * ogg = strstr(s.c_str(), "output_on_gpu=");
            const char * pfb = strstr(s.c_str(), "pin_from_back=");
            if (!ogg || !pfb) {
                td.viable = false;
                continue;
            }
            td.output_on_gpu = atoi(ogg + 14);
            td.pin_from_back = atoi(pfb + 14);

            td.overflow = pshard_overflow_from_name(overflow_name_buf);
            bool found_strategy = false;
            for (int i = 0; i < LLAMA_PSHARD_COUNT; i++) {
                if (strcmp(strat_name, llama_pshard_strategy_name((llama_pshard_strategy)i)) == 0) {
                    td.strategy = (llama_pshard_strategy)i;
                    found_strategy = true;
                    break;
                }
            }
            if (!found_strategy) {
                td.viable = false;
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
    auto variant_cache_ubatch = [&](const variant_data & variant) {
        return variant.cache_ubatch ? variant.cache_ubatch : requested_cache_ubatch;
    };
    auto cache_ubatch_ok = [&](const variant_data & variant) {
        if (requested_cache_ubatch == 0) return true;
        if (variant.cache_ubatch == 0) return true;
        if (variant.cache_ubatch <= requested_cache_ubatch) return true;
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
        if (!best_whole ||
                variant_cache_ubatch(variant) > variant_cache_ubatch(*best_whole) ||
                (variant_cache_ubatch(variant) == variant_cache_ubatch(*best_whole) &&
                 variant.budget_mib > best_whole->budget_mib)) {
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

    const uint32_t selected_cache_ubatch = best_whole ? variant_cache_ubatch(*best_whole) : requested_cache_ubatch;
    selected.erase(std::remove_if(selected.begin(), selected.end(),
        [&](const auto & p) {
            if (p.first == 0) return true;
            return selected_cache_ubatch > 0 && p.first > selected_cache_ubatch;
        }), selected.end());
    if (selected.empty()) {
        if (require_exact_budget && !variants.empty()) {
            LLAMA_LOG_INFO("%s: cache miss, no exact budget=%u MiB variant in %s\n",
                __func__, current_budget_mib, cache_path);
        }
        if (skipped_cache_ubatch) {
            LLAMA_LOG_INFO("%s: cache miss, no variant with cache_ubatch <= target cache_ubatch=%u in %s\n",
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
    registry->cache_ubatch = selected_cache_ubatch;

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
        LLAMA_LOG_INFO("%s: loaded %zu tier plans from budget=%u MiB cache_ubatch=%u variant for current budget=%u MiB target cache_ubatch=%u in %s\n",
            __func__, registry->tier_sizes.size(), registry->budget_mib, registry->cache_ubatch, current_budget_mib, requested_cache_ubatch, cache_path);
    } else {
        LLAMA_LOG_INFO("%s: loaded %zu salvaged tier plans for current budget=%u MiB from %s\n",
            __func__, registry->tier_sizes.size(), current_budget_mib, cache_path);
    }
    return true;
}

llama_pshard_plan_registry * llama_pshard_registry_create(uint32_t n_tier_max, uint32_t n_seq_max) {
    auto * registry = new llama_pshard_plan_registry();
    registry->init(n_tier_max, n_seq_max);
    return registry;
}

void llama_pshard_registry_free(llama_pshard_plan_registry * registry) {
    delete registry;
}

struct llama_pshard_cache_probe {
    std::vector<llama_device> devs;
    uint32_t n_layers    = 0;
    uint32_t n_ctx_train = 0;
    uint32_t n_expert    = 0;
    size_t   vram_free   = 0;
    size_t   vram_budget = 0;
    size_t   vram_total  = 0;
    ggml_backend_buffer_type_t host_buft = nullptr;
};

static bool llama_pshard_probe_model_only(
        const char * path_model,
        const struct llama_model_params * mparams,
        size_t max_vram_mb,
        size_t fit_target_mb,
        llama_pshard_cache_probe & probe) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= GGML_LOG_LEVEL_ERROR ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    llama_model_params mparams_probe = *mparams;
    mparams_probe.no_alloc  = true;
    mparams_probe.pshard    = false;
    mparams_probe.use_mmap  = false;
    mparams_probe.use_mlock = false;

    llama_model * model = llama_model_load_from_file(path_model, mparams_probe);
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);

    if (!model) {
        return false;
    }

    probe.devs        = model->devices;
    probe.n_layers    = model->hparams.n_layer;
    probe.n_ctx_train = model->hparams.n_ctx_train;
    probe.n_expert    = model->hparams.n_expert;

    if (!probe.devs.empty()) {
        ggml_backend_dev_t dev = probe.devs[0].dev;
        ggml_backend_dev_memory(dev, &probe.vram_free, &probe.vram_total);

        const size_t mib = 1024ULL * 1024ULL;
        const size_t fit_target_bytes = fit_target_mb * mib;
        probe.vram_budget = max_vram_mb > 0
            ? max_vram_mb * mib
            : (probe.vram_free > fit_target_bytes ? probe.vram_free - fit_target_bytes : 0);

        probe.host_buft = ggml_backend_dev_host_buffer_type(dev);
        if (!probe.host_buft) {
            probe.host_buft = ggml_backend_cpu_buffer_type();
        }
    }

    llama_model_free(model);
    return true;
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
        const char * path_model,
        struct llama_model_params * mparams,
        struct llama_context_params * cparams,
        struct llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t max_vram_mb,
        size_t fit_target_mb) {
    const std::string cache_path = std::string(path_model) + ".tensor_overrides.pshard_registry";

    if (!llama_pshard_params_supported(mparams, cparams)) {
        mparams->pshard = false;
        cparams->pshard = false;
        return;
    }

    llama_pshard_cache_probe probe;
    if (!llama_pshard_probe_model_only(path_model, mparams, max_vram_mb, fit_target_mb, probe)) {
        LLAMA_LOG_WARN("%s: failed to probe model metadata, disabling pshard\n", __func__);
        mparams->pshard = false;
        cparams->pshard = false;
        return;
    }

    if (probe.devs.empty()) {
        LLAMA_LOG_WARN("%s: no GPU devices found, disabling pshard\n", __func__);
        mparams->pshard = false;
        cparams->pshard = false;
        return;
    }

    const auto &   devs      = probe.devs;
    const uint32_t n_layers  = probe.n_layers;
    const size_t   vram_free = probe.vram_budget;

    if (vram_free > 0) {
        mparams->max_vram_alloc = std::max<size_t>(1, pshard_bytes_to_mib_ceil(vram_free));
    }

    ggml_backend_buffer_type_t host_buft = probe.host_buft;

    LLAMA_LOG_INFO("%s: probe: %u layers, %.1f MiB VRAM free, %.1f MiB budget%s\n",
        __func__, n_layers,
        probe.vram_free / (1024.0 * 1024.0),
        vram_free / (1024.0 * 1024.0),
        max_vram_mb > 0 ? " (-mva)" : " (free - fit target)");

    const uint32_t n_ctx_plan = cparams->n_ctx > 0 ? cparams->n_ctx : probe.n_ctx_train;

    auto * registry = mparams->pshard_registry;
    if (!registry) {
        LLAMA_LOG_ERROR("%s: pshard_registry is null\n", __func__);
        mparams->pshard = false;
        cparams->pshard = false;
        return;
    }
    const uint32_t requested_tier_max = registry->cache_ubatch;
    const uint32_t tier_max_auto = std::min(std::max(cparams->n_batch, (uint32_t) 16384), n_ctx_plan);
    const uint32_t tier_max = std::min(requested_tier_max > 0 ? requested_tier_max : tier_max_auto, n_ctx_plan);

    if (registry->cache_ubatch != tier_max) {
        registry->init(tier_max, cparams->n_seq_max);
    }

    registry->budget_mib = pshard_bytes_to_mib_ceil(vram_free);
    const uint32_t runtime_n_batch = std::min(n_ctx_plan, cparams->n_batch);
    const uint32_t runtime_cache_ubatch = std::min(runtime_n_batch, cparams->n_ubatch == 0 ? runtime_n_batch : cparams->n_ubatch);
    registry->cache_ubatch = registry->cache_ubatch ? std::min(n_ctx_plan, registry->cache_ubatch) : runtime_cache_ubatch;

    int64_t model_file_size = 0;
    {
        FILE * mf = fopen(path_model, "rb");
        if (mf) {
#ifdef _WIN32
            _fseeki64(mf, 0, SEEK_END);
            model_file_size = _ftelli64(mf);
#else
            fseeko(mf, 0, SEEK_END);
            model_file_size = ftello(mf);
#endif
            fclose(mf);
        }
    }

    const uint64_t fp = pshard_registry_fingerprint(
        mparams, cparams, model_file_size);

    if (!pshard_registry_load(registry, fp, cache_path.c_str(), host_buft, vram_free, false)) {
        LLAMA_LOG_WARN("%s: no matching plan cache at %s (fingerprint=0x%016" PRIx64 "), disabling pshard\n",
            __func__, cache_path.c_str(), fp);
        mparams->pshard = false;
        cparams->pshard = false;
        return;
    }

    if (!registry->pshard_disabled) {
        LLAMA_LOG_INFO("%s: loaded %zu tier plans from cache (variant budget=%u MiB cache_ubatch=%u)\n",
            __func__, registry->tier_sizes.size(), registry->budget_mib, registry->cache_ubatch);
    }

    // cached baseline fit for this budget
    // use the normal load path
    if (registry->pshard_disabled) {
        LLAMA_LOG_INFO("%s: cache says baseline %.1f MiB fits this budget (variant budget=%u MiB cache_ubatch=%u), using baseline loading\n",
            __func__, registry->baseline_vram_req / (1024.0 * 1024.0), registry->budget_mib, registry->cache_ubatch);
        mparams->pshard = false;
        cparams->pshard = false;
        mparams->n_gpu_layers = n_layers + 1;
        tensor_buft_overrides[0] = { nullptr, nullptr, -1 };
        mparams->tensor_buft_overrides = nullptr;
        return;
    }

    if (registry->cache_ubatch > 0) {
        const uint32_t pshard_ubatch = std::min(n_ctx_plan, registry->cache_ubatch);
        cparams->n_batch  = pshard_ubatch;
        cparams->n_ubatch = pshard_ubatch;
    }

    // pick the highest viable tier
    size_t default_tier = registry->tier_sizes.size();
    llama_pshard_plan * best = nullptr;
    for (size_t t = registry->tier_sizes.size(); t-- > 0; ) {
        llama_pshard_plan * candidate = registry->get_best(t);
        if (candidate && candidate->is_viable) {
            default_tier = t;
            best = candidate;
            break;
        }
    }
    if (!best) {
        LLAMA_LOG_WARN("%s: no viable plan in cache, disabling pshard\n", __func__);
        mparams->pshard = false;
        cparams->pshard = false;
        return;
    }
    if (default_tier < registry->tier_sizes.size() - 1) {
        LLAMA_LOG_INFO("%s: highest tier (bs=%u) not viable, falling back to bs=%u\n",
            __func__, registry->tier_sizes.back(), registry->tier_sizes[default_tier]);
        const uint32_t tier_bs = registry->tier_sizes[default_tier];
        cparams->n_batch  = std::min(cparams->n_batch,  tier_bs);
        cparams->n_ubatch = std::min(cparams->n_ubatch, tier_bs);
        LLAMA_LOG_INFO("%s: clamped n_batch/n_ubatch to %u\n", __func__, tier_bs);
    }

    if (best->n_pinned > n_layers) {
        LLAMA_LOG_WARN("%s: cache stale: n_pinned=%u > n_layers=%u, regenerate cache\n",
            __func__, best->n_pinned, n_layers);
        mparams->pshard = false;
        cparams->pshard = false;
        return;
    }

    const int32_t cpu_bid = pshard_dev_layout::compute_cpu_backend_id(devs.size());
    const pshard_dev_layout layout = pshard_dev_layout::for_device(0, cpu_bid);
    llama_pshard_generate_overrides(
        best->n_pinned, n_layers, host_buft, host_buft,
        tensor_buft_overrides,
        (llama_layer_fraction)best->overflow,
        best->strategy, layout,
        best->pin_from_back, best->output_on_gpu, best->n_attn_pinned);

    for (size_t i = 0; tensor_buft_overrides[i].pattern; i++) {
        if (tensor_buft_overrides[i].backend_id == layout.compute) {
            tensor_buft_overrides[i].buft = host_buft;
        }
    }

    mparams->tensor_buft_overrides = tensor_buft_overrides;
    mparams->n_gpu_layers = n_layers + 1;

    LLAMA_LOG_INFO("%s: plan: %s, n_pinned=%u/%u, vram=%zu MiB, n_gpu_layers=%d\n",
        __func__, llama_pshard_strategy_name(best->strategy),
        best->n_pinned, n_layers, mparams->max_vram_alloc, mparams->n_gpu_layers);
}
