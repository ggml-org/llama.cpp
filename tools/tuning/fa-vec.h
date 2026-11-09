#pragma once

#include "ggml-backend.h"

struct tuner_opts {
    const char * dtype_filter   = nullptr;  // comma-separated, e.g. "f16,q4_0"; null = all
    const char * dk_filter      = nullptr;  // comma-separated dk values, e.g. "128,192"; null = all
    int          reps           = 7;
    unsigned     seed           = 1234;
    bool         cooldown       = true;
    double       cool_drift     = 0.10;
    double       cool_eps       = 0.03;
    int          cool_max_wait  = 120;
    int          cool_max_retry = 2;
};

// Returns false only when the required Metal proc bridges are unavailable.
bool tuner_fa_vec_run(ggml_backend_t backend, ggml_backend_dev_t dev, const tuner_opts & opts);
