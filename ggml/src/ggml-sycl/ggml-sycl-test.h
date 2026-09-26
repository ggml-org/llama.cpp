#pragma once

#include "ggml-backend.h"

// Arms one synthetic synchronization failure for the selected SYCL backend.
// Defined only when ggml-sycl is built with GGML_SYCL_TESTING.
bool ggml_backend_sycl_test_inject_sync_failure_once(ggml_backend_t backend);
