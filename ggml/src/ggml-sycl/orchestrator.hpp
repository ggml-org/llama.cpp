// Unified matmul orchestrator for SYCL backend.
// Provides a single selection API for dispatch decisions.

#ifndef GGML_SYCL_ORCHESTRATOR_HPP
#define GGML_SYCL_ORCHESTRATOR_HPP

#include "ggml.h"
#include "kernel-selection.hpp"

#include <optional>

struct ggml_backend_sycl_context;

namespace ggml_sycl {

enum class MatmulBackend {
    UnifiedKernel,
    LegacyKernel,
};

enum class OneDnnPath {
    None,
    WoQ,
    DequantFp16,
};

struct MatmulDecision {
    bool                     valid = false;
    MatmulBackend            backend = MatmulBackend::LegacyKernel;
    ggml_sycl_mul_mat_kernel kernel = ggml_sycl_mul_mat_kernel::MMQ_AOS;
    ggml_layout_mode         layout = GGML_LAYOUT_AOS;
    OneDnnPath               onednn_path = OneDnnPath::None;
};

class UnifiedMatmulOrchestrator {
  public:
    explicit UnifiedMatmulOrchestrator(ggml_backend_sycl_context & ctx);

    MatmulDecision select(const ggml_tensor * src0,
                          const ggml_tensor * src1,
                          ggml_tensor *       dst,
                          const ggml_layout_mode * forced_layout = nullptr,
                          std::optional<ggml_sycl_mul_mat_kernel> forced_kernel = std::nullopt,
                          bool allow_unified = true) const;

  private:
    ggml_backend_sycl_context & ctx_;
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_ORCHESTRATOR_HPP
