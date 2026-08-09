#pragma once

// Fused cold-expert MoE kernel: down(act(gate(x)) * up(x)) for cold experts
// only, in a single CPU op. see ggml_moe_cold() in ggml.c for the op contract.

#include "ggml-cpu-impl.h"

#ifdef __cplusplus
extern "C" {
#endif

void ggml_compute_forward_moe_cold(
    const struct ggml_compute_params * params,
          struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
