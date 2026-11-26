#pragma once

#include <openvino/pass/matcher_pass.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

// Give a binary eltwise op's two operands the same rank, by unsqueezing leading axes onto the
// shorter one.
//
// This is semantically a no-op -- NUMPY broadcasting already left-pads the lower-rank operand
// with 1s, and the result rank is max(rank_a, rank_b) either way. It exists purely to work
// around an OpenVINO GPU-plugin defect: an eltwise op whose operands differ in rank is computed
// incorrectly once the plugin fuses it as a post-op into an `rms` primitive. gemma-4 dense hits
// this under stateful execution, where the layer tail adds a rank-3 residual to the rank-4
// RMS-norm output; the model then degenerates into repeated tokens on GPU while CPU is correct.
// Equalising the ranks keeps the fusion and makes it compute the right answer.
//
// Deliberately a graph pass rather than something the op translators do: rank is load-bearing
// during translation (several translators and later passes read operand ranks), and rewriting
// operands mid-translate breaks the attention path. Running after the graph is complete avoids
// that entirely.
//
// Only applies when both operands are real tensors -- a tensor-vs-Constant mismatch is the
// RMS norm's own eps/rsqrt arithmetic, which folds into the `rms` primitive rather than
// becoming a fused post-op, and is not affected by the defect. The norm output itself is never
// unsqueezed: when it is the lower-rank operand (gemma-3), that breaks the fused path instead.
class AlignEltwiseOperandRanks : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::AlignEltwiseOperandRanks")
    AlignEltwiseOperandRanks();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
