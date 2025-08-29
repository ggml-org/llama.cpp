#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "ggml-metal-device.h"
#include "ggml-metal-impl.h"
#include "ggml-metal-context.h"

extern "C" {
  GGML_BACKEND_API void ggml_backend_metal_get_device_context(ggml_backend_dev_t dev,
							      bool *has_simdgroup_mm,
							      bool *has_simdgroup_reduction,
							      bool *use_bfloat);
  
  GGML_BACKEND_API void
  ggml_backend_metal_get_device_context(ggml_backend_dev_t dev,
					bool *has_simdgroup_mm,
					bool *has_simdgroup_reduction,
					bool *has_bfloat) {
    ggml_metal_device_t dev_ctx = (ggml_metal_device_t)dev->context;
    
    const struct ggml_metal_device_props *props = ggml_metal_device_get_props(dev_ctx);
    
    *has_bfloat = props->has_bfloat;
    *has_simdgroup_reduction = props->has_simdgroup_reduction;
    *has_simdgroup_mm = props->has_simdgroup_mm;
  }
}
