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

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
