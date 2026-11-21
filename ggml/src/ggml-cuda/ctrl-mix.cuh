#include "common.cuh"

// The fused kernels below run at the first node of their pattern but write the output of the last one, which the
// allocator may have placed in the memory of an input that dies inside the pattern. They fuse only when every output
// either does not overlap an input the kernel reads, or (for element-wise inputs) is exactly that input.
static inline void ggml_cuda_fusion_span(const ggml_tensor * t, int64_t & beg, int64_t & end) {
    beg = (int64_t) t->data;
    end = beg + ggml_type_size(t->type);
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        end += (t->ne[d] - 1)*t->nb[d];
    }
}

static inline bool ggml_cuda_fusion_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    int64_t ab, ae, bb, be;
    ggml_cuda_fusion_span(a, ab, ae);
    ggml_cuda_fusion_span(b, bb, be);
    return ab < be && bb < ae;
}

// in may be read element-wise at the index out writes: no overlap, or the very same contiguous elements
static inline bool ggml_cuda_fusion_elementwise_ok(const ggml_tensor * out, const ggml_tensor * in) {
    return !ggml_cuda_fusion_overlap(out, in) ||
        (out->data == in->data && ggml_is_contiguous(out) && ggml_is_contiguous(in) && ggml_nelements(out) == ggml_nelements(in) &&
         ggml_type_size(out->type) == ggml_type_size(in->type));
}

// A gated mix of two vectors driven by two small matrix-vector products (Inkling 2 loop controllers), one token:
// MUL_MAT(W_f, g), MUL_MAT(W_g, g), TANH, MUL(., cur), SUB(cur, .), TANH, MUL(msg, .), ADD
// = cur - tanh(W_f g)*cur + msg*tanh(W_g g). Returns the index of the ADD, or -1.
int  ggml_cuda_match_ctrl_mix(const ggml_cgraph * cgraph, int i, bool check_memory = true);
void ggml_cuda_op_ctrl_mix(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int last);

// MUL(x, w) -> L2_NORM [-> SCALE], w of the shape of x, as one kernel with a warp per row (Inkling 2 clipped Q and R).
// Returns the index of the last node, or -1.
int  ggml_cuda_match_mul_l2_norm(const ggml_cgraph * cgraph, int i, bool check_memory = true);
void ggml_cuda_op_mul_l2_norm(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int last);

// A relative-position attention bias for one token (Inkling 2): MUL_MAT(r, table) -> SUB(sinks) -> GET_ROWS(distance)
// -> PERMUTE -> ADD(mask) -> CPY(F16) as one kernel that computes only the gathered entries. Returns the index of the
// CPY, or -1.
int  ggml_cuda_match_rel_bias(const ggml_cgraph * cgraph, int i, bool check_memory = true);
void ggml_cuda_op_rel_bias(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int last);
