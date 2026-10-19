#include "pipeline.h"

#include "ggml-openvino/ggml-openvino-extra.h"
#include "ggml-openvino/openvino/pass/align_eltwise_ranks.h"
#include "ggml-openvino/openvino/pass/fuse_moe_compressed.h"
#include "ggml-openvino/openvino/pass/fuse_moe_router.h"

namespace ov {
namespace frontend {
namespace ggml {
namespace pipeline {

void register_gpu_passes(ov::pass::Manager & manager, const GraphTraits & traits) {
    (void) traits;

    // MoE: fuse router + experts into MOECompressed. It has no CPU plugin implementation, so it is a GPU
    // pass. GGML_OPENVINO_MOE_OP=0 keeps the unfused GatherMatmul path for fallback/debugging.
    if (ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) != 0) {
        manager.register_pass<pass::FuseMoeRouter>();
        manager.register_pass<pass::FuseMoeCompressed>();
        // Same fusion for models whose gate and up projections share one fused expert
        // weight (gemma-4). It only matches that shape and only when the experts carry an
        // integer zero point, so it is a no-op on the separate-gate/up models above.
        manager.register_pass<pass::FuseMoeCompressedFusedGateUp>();
    }

    // Workaround for an OpenVINO GPU-plugin defect: an eltwise op whose operands differ in
    // rank is computed wrongly once the plugin fuses it as a post-op into `rms`. gemma-4
    // dense under stateful execution adds a rank-3 residual to the rank-4 norm output and
    // decodes garbage on GPU while CPU is correct. Equalising the ranks is a no-op for
    // NUMPY broadcasting and makes the fused path correct.
    // Remove once the plugin guards that fusion. Opt out with
    // GGML_OPENVINO_DISABLE_ELTWISE_RANK_ALIGN=1.
    if (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_ELTWISE_RANK_ALIGN")) {
        manager.register_pass<pass::AlignEltwiseOperandRanks>();
    }
}

}  // namespace pipeline
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
