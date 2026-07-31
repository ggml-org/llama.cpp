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
    bool        kernel_fitness = false;
    std::string kernel_fitness_dir;
    float       kernel_fitness_blend = 1.0f;
    bool        w4a4 = false;
    float       w4a4_outlier_thresh = 6.0f;
    bool        acceptance = false;
    std::string acceptance_out;
    // self-improving capability loop: output-targeting ops that run then exit
    // without quantizing (precedent: --tessera-evolve-only / --tessera-calibrate-only)
    std::string capability_eval;
    std::string capability_out;
    std::string adapt_eval;
    std::string adapt_out;
    bool        adapt_dry_run = false;
    double      adapt_epsilon = 0.02;
    // tier-2 anonymizer: scrub a text payload then exit without quantizing
    std::string anonymize_in;
    std::string anonymize_out;
    std::string anonymize_level = "balanced";
    std::string anonymize_map;
};

const common_tessera_params & common_get_tessera_params();

// Parse one --tessera-* / --calib-* flag at argv[i] into the shared Tessera
// params (for tools that hand-roll their arg loop). Returns argv slots consumed
// (1 = switch, 2 = valued), 0 if argv[i] is not a Tessera flag, or -1 on a
// validation error (message written to err).
int common_tessera_parse_one(int argc, char ** argv, int i, std::string & err);
