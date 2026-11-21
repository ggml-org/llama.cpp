#include "common.cuh"

#define DEPTH_MIX_MAX_STATES  8
#define DEPTH_MIX_MAX_QUERIES 4

// One token of attention over stored states (Inkling 2 block attention residuals):
// RMS_NORM(cur) -> TRANSPOSE -> CONCAT(blocks_t, cur_t) -> CONCAT(blocks, cur) -> MUL_MAT(states, query) -> SOFT_MAX
// -> MUL_MAT(states_t, scores). Returns true and runs it as one kernel when node i starts that pattern.
bool ggml_cuda_match_depth_mix(const ggml_cgraph * cgraph, int i, bool check_memory = true);
void ggml_cuda_op_depth_mix(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
