#include "llama.h"

#include "arg.h"
#include "common.h"
#include "log.h"

#include <algorithm>
#include <cstdlib>
#include <cinttypes>
#include <cstdio>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

struct bench_plan_params {
    bool enabled = false;

    std::vector<int> n_prompt;
    std::vector<int> n_gen;
    std::vector<std::pair<int, int>> n_pg;
    std::vector<int> n_depth;
    bool flash_attn_seen = false;
};

static std::vector<int> parse_int_range(const std::string & s) {
    // first[-last[(+|*)step]]
    std::regex range_regex(R"(^(\d+)(?:-(\d+)(?:([\+|\*])(\d+))?)?(?:,|$))");

    std::smatch match;
    std::string::const_iterator search_start(s.cbegin());

    std::vector<int> result;
    while (std::regex_search(search_start, s.cend(), match, range_regex)) {
        int first = std::stoi(match[1]);
        int last  = match[2].matched ? std::stoi(match[2]) : first;
        char op   = match[3].matched ? match[3].str()[0] : '+';
        int step  = match[4].matched ? std::stoi(match[4]) : 1;

        for (int i = first; i <= last;) {
            result.push_back(i);

            int prev_i = i;
            if (op == '+') {
                i += step;
            } else if (op == '*') {
                i *= step;
            } else {
                throw std::invalid_argument("invalid range format");
            }

            if (i <= prev_i) {
                throw std::invalid_argument("invalid range");
            }
        }

        search_start = match.suffix().first;
    }

    if (search_start != s.cend()) {
        throw std::invalid_argument("invalid range format");
    }

    return result;
}

static std::pair<int, int> parse_pg(const std::string & s) {
    const size_t comma = s.find(',');
    if (comma == std::string::npos || s.find(',', comma + 1) != std::string::npos) {
        throw std::invalid_argument("invalid -pg format");
    }

    return { std::stoi(s.substr(0, comma)), std::stoi(s.substr(comma + 1)) };
}

static bool is_bench_plan_arg_with_value(const std::string & arg) {
    return arg == "-p"  || arg == "--n-prompt" ||
           arg == "-n"  || arg == "--n-gen"    ||
           arg == "-pg" ||
           arg == "-d"  || arg == "--n-depth";
}

static bool parse_bench_plan_args(int argc, char ** argv, bench_plan_params & bench, std::vector<std::string> & forwarded) {
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--bench-plan") {
            bench.enabled = true;
            break;
        }
    }

    forwarded.clear();
    forwarded.push_back(argv[0]);

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];

        if (arg == "--bench-plan") {
            continue;
        }

        if (bench.enabled && (arg == "-fa" || arg == "--flash-attn")) {
            bench.flash_attn_seen = true;
        }

        if (!bench.enabled || !is_bench_plan_arg_with_value(arg)) {
            forwarded.push_back(arg);
            continue;
        }

        if (++i >= argc) {
            fprintf(stderr, "%s: missing value for %s\n", __func__, arg.c_str());
            return false;
        }

        try {
            if (arg == "-p" || arg == "--n-prompt") {
                auto vals = parse_int_range(argv[i]);
                bench.n_prompt.insert(bench.n_prompt.end(), vals.begin(), vals.end());
            } else if (arg == "-n" || arg == "--n-gen") {
                auto vals = parse_int_range(argv[i]);
                bench.n_gen.insert(bench.n_gen.end(), vals.begin(), vals.end());
            } else if (arg == "-pg") {
                bench.n_pg.push_back(parse_pg(argv[i]));
            } else if (arg == "-d" || arg == "--n-depth") {
                auto vals = parse_int_range(argv[i]);
                bench.n_depth.insert(bench.n_depth.end(), vals.begin(), vals.end());
            }
        } catch (const std::exception & e) {
            fprintf(stderr, "%s: invalid value for %s: %s (%s)\n", __func__, arg.c_str(), argv[i], e.what());
            return false;
        }
    }

    if (bench.enabled) {
        if (bench.n_prompt.empty()) {
            bench.n_prompt.push_back(512);
        }
        if (bench.n_gen.empty()) {
            bench.n_gen.push_back(128);
        }
        if (bench.n_depth.empty()) {
            bench.n_depth.push_back(0);
        }
    }

    return true;
}

static void bench_plan_add_case(std::map<uint32_t, uint32_t> & result, uint32_t n_ctx, uint32_t tier_cap) {
    if (n_ctx == 0) {
        return;
    }

    auto & max_tier_cap = result[n_ctx];
    max_tier_cap = std::max(max_tier_cap, tier_cap);
}

static std::map<uint32_t, uint32_t> bench_plan_context_tier_caps(const bench_plan_params & bench) {
    std::map<uint32_t, uint32_t> result;

    for (const int d : bench.n_depth) {
        for (const int p : bench.n_prompt) {
            if (p > 0) {
                bench_plan_add_case(result, (uint32_t) (p + d), (uint32_t) p);
            }
        }

        for (const int n : bench.n_gen) {
            if (n > 0) {
                bench_plan_add_case(result, (uint32_t) (n + d), 0);
            }
        }

        for (const auto & pg : bench.n_pg) {
            if (pg.first > 0 || pg.second > 0) {
                bench_plan_add_case(result, (uint32_t) (pg.first + pg.second + d), (uint32_t) std::max(pg.first, 0));
            }
        }
    }

    return result;
}

static void plan_pshard_context(common_params & params, uint32_t n_ctx, uint32_t bench_tier_cap = 0, bool bench_plan = false) {
    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    cparams.n_ctx = n_ctx;

    params.tensor_buft_overrides.assign(4096, {});

    const uint32_t tier_max_auto = bench_plan ?
        std::max(bench_tier_cap, (uint32_t) 16) :
        std::min(std::max(cparams.n_batch, (uint32_t) 16384), cparams.n_ctx);
    const uint32_t tier_max_user = params.pshard_tier_max > 0 ? std::min(params.pshard_tier_max, tier_max_auto) : tier_max_auto;
    const uint32_t tier_max      = bench_plan ? tier_max_user : std::min(tier_max_user, cparams.n_ctx);

    mparams.pshard_registry        = llama_pshard_registry_create(tier_max, cparams.n_seq_max);
    mparams.pshard_cache_skip_load = true;

    const size_t fit_target_mb = params.fit_params_target.empty() ? 0 : params.fit_params_target[0] / (1024 * 1024);

    if (bench_plan) {
        LOG_INF("%s: planning pshard tensor overrides for n_ctx=%u tier_cap=%u tier_max=%u...\n",
                __func__, n_ctx, bench_tier_cap, tier_max);
    } else {
        LOG_INF("%s: planning pshard tensor overrides for n_ctx=%u tier_max=%u...\n", __func__, n_ctx, tier_max);
    }
    llama_params_fit_pshard(params.model.path.c_str(), &mparams, &cparams,
        params.tensor_buft_overrides.data(), params.max_vram_alloc, fit_target_mb);

    llama_pshard_registry_free(mparams.pshard_registry);
}

int main(int argc, char ** argv) {
    common_params params;
    bench_plan_params bench;
    std::vector<std::string> forwarded;

    if (!parse_bench_plan_args(argc, argv, bench, forwarded)) {
        return 1;
    }

    std::vector<char *> forwarded_argv;
    forwarded_argv.reserve(forwarded.size());
    for (std::string & arg : forwarded) {
        forwarded_argv.push_back(arg.data());
    }

    if (!common_params_parse((int) forwarded_argv.size(), forwarded_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (bench.enabled && !bench.flash_attn_seen && getenv("LLAMA_ARG_FLASH_ATTN") == nullptr) {
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    if (bench.enabled) {
        const std::map<uint32_t, uint32_t> ctx_caps = bench_plan_context_tier_caps(bench);
        if (ctx_caps.empty()) {
            fprintf(stderr, "%s: --bench-plan produced no non-zero llama-bench contexts\n", __func__);
            return 1;
        }

        LOG_INF("%s: planning %zu unique llama-bench context(s)\n", __func__, ctx_caps.size());
        for (const auto & ctx_cap : ctx_caps) {
            plan_pshard_context(params, ctx_cap.first, ctx_cap.second, true);
        }
    } else {
        auto cparams = common_context_params_to_llama(params);
        plan_pshard_context(params, cparams.n_ctx);
    }

    LOG_INF("%s: planning complete, registry written next to model file\n", __func__);

    return 0;
}
