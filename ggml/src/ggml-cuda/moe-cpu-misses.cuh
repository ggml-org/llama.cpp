#include "common.cuh"
#include "ggml-moe-cpu-misses.h"

// a publish, join or merge op of the MoE cache, see ggml_moe_cpu_misses_api
bool ggml_cuda_moe_cpu_misses_is_op(const ggml_tensor * op);

// the miss copy of a two pass publish runs on a side stream until its join op, a CUDA graph chunk cannot end before
bool ggml_cuda_moe_cpu_misses_fork_open();

void ggml_cuda_op_moe_cpu_misses(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// nullptr if not supported
const ggml_moe_cpu_misses_api * ggml_cuda_moe_cpu_misses_get_api();
