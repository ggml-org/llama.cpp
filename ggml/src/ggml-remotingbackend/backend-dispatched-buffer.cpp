#include <cstdint>
#include "backend-internal.h"
#include "backend-dispatched.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"

uint32_t
backend_buffer_get_base(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec) {
  ggml_backend_buffer_t buffer;
  buffer = vn_decode_ggml_buffer(dec);

  uintptr_t base = (uintptr_t) buffer->iface.get_base(buffer);
  vn_encode_uintptr_t(enc, &base);

  //INFO("%s: send base %p\n", __func__,  (void *) base);

  return 0;
}
