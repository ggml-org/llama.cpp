#include "common.cuh"

#define DEPTH_MIX_MAX_STATES  8
#define DEPTH_MIX_MAX_QUERIES 4

// One token of attention over stored states (Inkling 2 block attention residuals):
// RMS_NORM(cur) -> TRANSPOSE -> CONCAT(blocks_t, cur_t) -> CONCAT(blocks, cur) -> MUL_MAT(states, query) -> SOFT_MAX
// -> MUL_MAT(states_t, scores). Returns true and runs it as one kernel when node i starts that pattern.
bool ggml_cuda_match_depth_mix(const ggml_cgraph * cgraph, int i, bool check_memory = true);
void ggml_cuda_op_depth_mix(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i,
                            float * dst2 = nullptr, const float * w2 = nullptr, float eps2 = 0.0f);

// the depth mix (one query) followed by RMS_NORM [-> MUL(w)] of its output, the norm done by the same kernel with the
// arithmetic of the vectorized rms_norm. Returns the index of the last node, or -1.
int  ggml_cuda_match_depth_mix_norm(const ggml_cgraph * cgraph, int i, bool check_memory = true);
void ggml_cuda_op_depth_mix_norm(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int last);
