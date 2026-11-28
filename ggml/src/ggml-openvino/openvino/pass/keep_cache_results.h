#include "openvino/pass/pass.hpp"

#include <set>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

// Keeps only the Results named in |cache_outputs| (the KV / recurrent cache writes) and drops every other Result,
// so that everything that only feeds them -- for models whose tail layers read an earlier layer's KV (Gemma 3n/4
// E-series) the KV-shared layers and the LM head -- leaves the model. Parameters that nothing reads afterwards are
// removed too. Used for KV-only prefill on the static path (GGML_OPENVINO_KV_ONLY_PREFILL); the outputs of the
// batch then come from one step of the decode model.
class KeepCacheResults : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::ggml::pass::KeepCacheResults")
    explicit KeepCacheResults(std::set<std::string> cache_outputs) : m_cache_outputs(std::move(cache_outputs)) {}

    bool run_on_model(const std::shared_ptr<ov::Model> & model) override;

private:
    std::set<std::string> m_cache_outputs;
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
