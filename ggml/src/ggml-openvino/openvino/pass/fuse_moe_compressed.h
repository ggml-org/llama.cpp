#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

// Folds the MoE expert block emitted for MUL_MAT_ID (3 GatherMatmul + SwiGLU + routing
// weighting + expert reduction) into a single ov::op::internal::MOECompressed.
class FuseMoeCompressed : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::FuseMoeCompressed")
    FuseMoeCompressed();
};

// Folds the MoE expert block emitted for a model whose gate and up projections share one
// fused MUL_MAT_ID weight (gemma-4: one GatherMatmul + Slice/Slice split, GEGLU activation)
// into a single ov::op::internal::MOECompressed, GEMM3_SWIGLU/GEGLU_ERF. Splits the fused
// weight/scale/zp into gate/up halves so it lands on the same 3-GEMM expert-grouped kernel
// FuseMoeCompressed uses for models with separate gate/up weights.
class FuseMoeCompressedFusedGateUp : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::FuseMoeCompressedFusedGateUp")
    FuseMoeCompressedFusedGateUp();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
