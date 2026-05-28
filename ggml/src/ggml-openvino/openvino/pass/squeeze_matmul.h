#ifndef GGML_SRC_GGML_OPENVINO_OPENVINO_PASS_SQUEEZE_MATMUL_H_
#define GGML_SRC_GGML_OPENVINO_OPENVINO_PASS_SQUEEZE_MATMUL_H_

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class SqueezeMatmul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::SqueezeMatmul")
    SqueezeMatmul();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#endif  // GGML_SRC_GGML_OPENVINO_OPENVINO_PASS_SQUEEZE_MATMUL_H_
