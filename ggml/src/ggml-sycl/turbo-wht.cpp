//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "turbo-wht.hpp"
#include "common.hpp"

void ggml_sycl_op_turbo_wht(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("ggml_sycl: TurboQuant KV cache (turbo2/turbo3/turbo4) is not supported on the SYCL backend - it miscompiles / produces incorrect output on Intel Arc. Use --cache-type-k/v q8_0 (or f16), or run turbo KV on CPU with -ngl 0.");
}
