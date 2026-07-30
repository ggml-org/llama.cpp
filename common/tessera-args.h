#pragma once

#include <cstdint>
#include <string>

struct common_tessera_params {
    std::string mode = "default";
    std::string imatrix;
    std::string policy;
    std::string policy_out;
    std::string ga_checkpoint;
    std::string calib_corpus;
    std::string calib_corpus_out;
    uint64_t    evolve_seed = 0;
    int         evolve_iters = 8;
    int         evolve_islands = 4;
    int         evolve_population = 16;
    bool        evolve_only = false;
    bool        calibrate_only = false;
    float       outlier_frac = 0.005f;
    std::string awq_alpha = "auto";
    float       awq_clip = 1.0f;
    std::string ternary_threshold = "auto";
    std::string range_selection = "legacy";
    bool        champq = false;
    int         nthreads = 0;
};

const common_tessera_params & common_get_tessera_params();
