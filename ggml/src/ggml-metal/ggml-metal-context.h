#pragma once

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// MTLFunctionConstantValues wrapper
//

typedef struct ggml_metal_cv * ggml_metal_cv_t;

ggml_metal_cv_t ggml_metal_cv_init(void);
void ggml_metal_cv_free(ggml_metal_cv_t cv);

void ggml_metal_cv_set_int32(ggml_metal_cv_t cv, int32_t value, int32_t idx);
void ggml_metal_cv_set_bool (ggml_metal_cv_t cv, bool    value, int32_t idx);

//
// backend
//

typedef struct ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t ctx_dev);
void ggml_metal_free(ggml_metal_t ctx);

typedef void * ggml_metal_pipeline_t;

ggml_metal_pipeline_t ggml_metal_get_pipeline(ggml_metal_t ctx, const char * name);

ggml_metal_pipeline_t ggml_metal_compile_pipeline(ggml_metal_t ctx, const char * base, const char * name, ggml_metal_cv_t cv);

void ggml_metal_synchronize(ggml_metal_t ctx);

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);

enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf);

void ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf);

void ggml_metal_set_n_cb            (ggml_metal_t ctx, int n_cb);
void ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data);
bool ggml_metal_supports_family     (ggml_metal_t ctx, int family);
void ggml_metal_capture_next_compute(ggml_metal_t ctx);

#ifdef __cplusplus
}
#endif
