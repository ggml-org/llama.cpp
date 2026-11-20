#pragma once

#include <openvino/pass/matcher_pass.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

// GPU runs f32 graphs in f16; ReduceSum(x*x) overflows f16 (max 65504) for large rows, e.g. the
// Gemma 3n gaussian_topk std and altup magnitudes. Keep x*x -> ReduceSum [-> * k] [-> Sqrt [-> a * s / norm]]
// in f32, fenced by Converts so the f32 region does not spread to the rest of the graph.
class KeepSumSquareFP32 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::KeepSumSquareFP32")
    KeepSumSquareFP32();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
