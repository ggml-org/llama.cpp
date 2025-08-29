#include <cstdint>
#include "backend-internal.h"
#include "backend-dispatched.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"

void (*ggml_backend_metal_get_device_context_fct)(ggml_backend_dev_t dev,
						  bool *has_simdgroup_mm,
						  bool *has_simdgroup_reduction,
						  bool *has_bfloat) = NULL;

uint32_t
backend_metal_get_device_context(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx) {
  UNUSED(ctx);
  UNUSED(dec);

  bool has_simdgroup_mm;
  bool has_simdgroup_reduction;
  bool has_bfloat;

  uint32_t ret = 0;
  if (ggml_backend_metal_get_device_context_fct) {

    ggml_backend_metal_get_device_context_fct(dev,
					  &has_simdgroup_mm,
					  &has_simdgroup_reduction,
					  &has_bfloat
      );
  } else {
    ERROR("ggml_backend_metal_get_device_context not available :/");
    ret = 1;
  }

  vn_encode_bool_t(enc, &has_simdgroup_mm);
  vn_encode_bool_t(enc, &has_simdgroup_reduction);
  vn_encode_bool_t(enc, &has_bfloat);

  return ret;
}
