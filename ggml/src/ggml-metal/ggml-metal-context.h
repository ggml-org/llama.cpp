#pragma once

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// backend context
//

typedef struct ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
void ggml_metal_free(ggml_metal_t ctx);

void ggml_metal_synchronize(ggml_metal_t ctx);

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);

enum ggml_status ggml_metal_graph_compute (ggml_metal_t ctx, struct ggml_cgraph * gf);
void             ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf);

void ggml_metal_set_n_cb            (ggml_metal_t ctx, int n_cb);
void ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data);
bool ggml_metal_supports_family     (ggml_metal_t ctx, int family);
void ggml_metal_capture_next_compute(ggml_metal_t ctx);

//
// graph encoder
//

typedef struct ggml_metal_graph_encoder * ggml_metal_graph_encoder_t;

ggml_metal_library_t ggml_metal_graph_encoder_get_lib(ggml_metal_graph_encoder_t ctx);
ggml_metal_encoder_t ggml_metal_graph_encoder_get_enc(ggml_metal_graph_encoder_t ctx);
struct ggml_cgraph * ggml_metal_graph_encoder_get_gf (ggml_metal_graph_encoder_t ctx);

const struct ggml_metal_device_props * ggml_metal_graph_encoder_get_props_dev(ggml_metal_graph_encoder_t ctx);

int ggml_metal_graph_encoder_get_idx_start(ggml_metal_graph_encoder_t ctx);
int ggml_metal_graph_encoder_get_idx_end  (ggml_metal_graph_encoder_t ctx);

bool ggml_metal_graph_encoder_get_use_fusion(ggml_metal_graph_encoder_t ctx);

int ggml_metal_graph_encoder_get_debug_fusion(ggml_metal_graph_encoder_t ctx);
int ggml_metal_graph_encoder_get_debug_graph (ggml_metal_graph_encoder_t ctx);

bool ggml_metal_graph_encoder_concurrency_reset(ggml_metal_graph_encoder_t ctx);
bool ggml_metal_graph_encoder_concurrency_check(ggml_metal_graph_encoder_t ctx, const struct ggml_tensor * node);
bool ggml_metal_graph_encoder_concurrency_add  (ggml_metal_graph_encoder_t ctx, const struct ggml_tensor * node);

#ifdef __cplusplus
}
#endif
