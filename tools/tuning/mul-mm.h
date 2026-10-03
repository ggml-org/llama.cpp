#pragma once

#include "fa-vec.h"  // tuner_opts (shared CLI options)

#include "ggml-backend.h"

// Returns false only when the required Metal proc bridges are unavailable.
bool tuner_mul_mm_run(ggml_backend_t backend, ggml_backend_dev_t dev, const tuner_opts & opts);
