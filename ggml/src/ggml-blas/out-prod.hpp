#pragma once

#include "common.hpp"

void ggml_blas_out_prod_f(
        const ggml_backend_blas_context * ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
              ggml_tensor * dst);
