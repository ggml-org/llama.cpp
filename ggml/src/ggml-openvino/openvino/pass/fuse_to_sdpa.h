#ifndef GGML_SRC_GGML_OPENVINO_OPENVINO_PASS_FUSE_TO_SDPA_H_
#define GGML_SRC_GGML_OPENVINO_OPENVINO_PASS_FUSE_TO_SDPA_H_

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class FuseToSDPA : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::FuseToSDPA")
    FuseToSDPA();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#endif  // GGML_SRC_GGML_OPENVINO_OPENVINO_PASS_FUSE_TO_SDPA_H_
