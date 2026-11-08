#pragma once

#include <openvino/pass/matcher_pass.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class FuseArgsortTopK : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::FuseArgsortTopK")
    FuseArgsortTopK();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
