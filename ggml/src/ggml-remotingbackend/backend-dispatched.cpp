#include <cstdint>
#include "backend-internal.h"
#include "backend-dispatched.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"

#include "ggml-metal.h"

ggml_backend_reg_t reg = NULL;
ggml_backend_dev_t dev = NULL;
ggml_backend_t bck = NULL;

long long timer_start = 0;
long long timer_total = 0;
long long timer_count = 0;

uint32_t backend_dispatch_initialize(void *ggml_backend_reg_fct_p, void *ggml_backend_init_fct_p) {
  if (reg != NULL) {
    FATAL("%s: already initialized :/", __func__);
  }
  ggml_backend_reg_t (* ggml_backend_reg_fct)(void) = (ggml_backend_reg_t (*)()) ggml_backend_reg_fct_p;

  reg = ggml_backend_reg_fct();
  if (reg == NULL) {
    FATAL("%s: backend registration failed :/", __func__);
  }

  if (reg->iface.get_device_count(reg)) {
    dev = reg->iface.get_device(reg, 0);
  }

  ggml_backend_t (* ggml_backend_fct)(int) = (ggml_backend_t (*)(int)) ggml_backend_init_fct_p;

  bck = ggml_backend_fct(0);
  if (!bck) {
    ERROR("%s: backend initialization failed :/", __func__);
    return APIR_BACKEND_INITIALIZE_BACKEND_FAILED;
  }

  size_t free, total;
  dev->iface.get_memory(dev, &free, &total);
  INFO("%s: free memory: %ld MB", __func__, (size_t) free/1024/1024);

  return APIR_BACKEND_INITIALIZE_SUCCESS;
}
