#pragma once

#include <openvino/pass/matcher_pass.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

// GPU and NPU run f32 graphs in f16, so a weight MatMul whose output exceeds 65504 becomes inf even when
// a scalar Multiply right after it would bring the value back into range (granite residual_scale after
// ffn_down: the FFN output of some tokens is far above the f16 range). Rewrites
// Multiply(MatMul(x, W), s) with a scalar constant |s| < 1 as MatMul(Multiply(x, s), W), which is exact
// and keeps the MatMul output in range.
class MoveScaleBeforeMatMul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::MoveScaleBeforeMatMul")
    MoveScaleBeforeMatMul();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
