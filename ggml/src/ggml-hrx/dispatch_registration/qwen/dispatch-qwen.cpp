#include "dispatch-qwen.h"

#include "dispatch-moe-router.h"
#include "dispatch-qwen-attention-postprocess.h"
#include "dispatch-qwen-matmul.h"
#include "dispatch-qwen-rmsnorm.h"
#include "dispatch-routed-ffn.h"

namespace ggml::hrx {

void register_qwen_dispatches(DispatchRegistryBuilder & registry) {
    register_qwen_attention_postprocess_dispatches(registry);
    register_qwen_matmul_dispatches(registry);
    register_routed_ffn_dispatches(registry);
    register_qwen_rmsnorm_dispatches(registry);
    register_moe_router_dispatches(registry);
}

}  // namespace ggml::hrx
