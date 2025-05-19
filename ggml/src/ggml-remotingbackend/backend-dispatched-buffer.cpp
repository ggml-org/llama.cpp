#include <cstdint>
#include "backend-internal.h"
#include "backend-dispatched.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"

uint32_t
backend_buffer_get_base(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx) {
  UNUSED(ctx);
  ggml_backend_buffer_t buffer;
  buffer = vn_decode_ggml_buffer(dec);

  uintptr_t base = (uintptr_t) buffer->iface.get_base(buffer);
  vn_encode_uintptr_t(enc, &base);

  //INFO("%s: send base %p\n", __func__,  (void *) base);

  return 0;
}

uint32_t
backend_buffer_set_tensor(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx) {
  UNUSED(ctx);
  UNUSED(enc);

  ggml_backend_buffer_t buffer;
  buffer = vn_decode_ggml_buffer(dec);

  ggml_tensor *tensor;
  // safe to remove the const qualifier here
  tensor = (ggml_tensor *) (uintptr_t) vn_decode_ggml_tensor_inplace(dec);

  void *data;
  vn_decode_uintptr_t(dec, (uintptr_t *) &data);

  size_t offset;
  vn_decode_size_t(dec, &offset);

  size_t size;
  vn_decode_size_t(dec, &size);

  INFO("Calling (%p)->set_tensor(tensor=%p, data=%p, offset=%lu, size=%lu",
       buffer, tensor, data, offset, size);

  //buffer->iface.set_tensor(buffer, tensor, data, offset, size);

  return 0;
}
