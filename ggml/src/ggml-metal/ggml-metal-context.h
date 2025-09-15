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
// MTLComputePipelineState wrapper
//

typedef struct ggml_metal_pipeline * ggml_metal_pipeline_t;

ggml_metal_pipeline_t ggml_metal_pipeline_init(void);
void ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline);

void ggml_metal_pipeline_set_nsg(ggml_metal_pipeline_t pipeline, int nsg);
int  ggml_metal_pipeline_get_nsg(ggml_metal_pipeline_t pipeline);

void ggml_metal_pipeline_set_nr0(ggml_metal_pipeline_t pipeline, int nr0);
int  ggml_metal_pipeline_get_nr0(ggml_metal_pipeline_t pipeline);

void ggml_metal_pipeline_set_nr1(ggml_metal_pipeline_t pipeline, int nr1);
int  ggml_metal_pipeline_get_nr1(ggml_metal_pipeline_t pipeline);

void   ggml_metal_pipeline_set_smem(ggml_metal_pipeline_t pipeline, size_t smem);
size_t ggml_metal_pipeline_get_smem(ggml_metal_pipeline_t pipeline);

int ggml_metal_pipeline_max_theads_per_threadgroup(ggml_metal_pipeline_t pipeline);

// a collection of pipelines
typedef struct ggml_metal_pipelines * ggml_metal_pipelines_t;

ggml_metal_pipelines_t ggml_metal_pipelines_init(void);
void ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls);

void                  ggml_metal_pipelines_add(ggml_metal_pipelines_t ppls, const char * name, ggml_metal_pipeline_t pipeline);
ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, const char * name);

//
// MTLCommandBuffer wrapper
//

typedef void * ggml_metal_cmd_buf_t;

//
// MTLComputeCommandEncoder wrapper
//

typedef struct ggml_metal_encoder * ggml_metal_encoder_t;

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent);
void ggml_metal_encoder_free(ggml_metal_encoder_t encoder);

void ggml_metal_encoder_debug_group_push(ggml_metal_encoder_t encoder, const char * name);
void ggml_metal_encoder_debug_group_pop (ggml_metal_encoder_t encoder);

void ggml_metal_encoder_set_pipeline(ggml_metal_encoder_t encoder, ggml_metal_pipeline_t pipeline);

void ggml_metal_encoder_set_bytes (ggml_metal_encoder_t encoder, void * data, size_t size, int idx);
void ggml_metal_encoder_set_buffer(ggml_metal_encoder_t encoder, struct ggml_metal_buffer_id buffer, int idx);

void ggml_metal_encoder_set_threadgroup_memory_size(ggml_metal_encoder_t encoder, size_t size, int idx);

void ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, int tg0, int tg1, int tg2, int tptg0, int tptg1, int tptg2);

void ggml_metal_encoder_memory_barrier(ggml_metal_encoder_t encoder);

void ggml_metal_encoder_end_encoding(ggml_metal_encoder_t encoder);

//
// backend
//

typedef struct ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t ctx_dev);
void ggml_metal_free(ggml_metal_t ctx);

ggml_metal_pipeline_t ggml_metal_get_pipeline    (ggml_metal_t ctx, const char * name);
ggml_metal_pipeline_t ggml_metal_compile_pipeline(ggml_metal_t ctx, const char * base, const char * name, ggml_metal_cv_t cv);

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

// TODO: tmp
#include "ggml-metal-common.h"

// TODO: tmp
struct ggml_metal_graph_encoder {
    ggml_metal_t ctx;

    const struct ggml_metal_device_props * props_dev;

    ggml_metal_encoder_t encoder;

    ggml_mem_ranges_t mem_ranges;

    struct ggml_cgraph * gf;

    int idx_start;
    int idx_end;

    bool use_fusion;

    int debug_fusion;
};

bool ggml_metal_graph_encoder_concurrency_reset(ggml_metal_graph_encoder_t ctx);
bool ggml_metal_graph_encoder_concurrency_check(ggml_metal_graph_encoder_t ctx, const struct ggml_tensor * node);
bool ggml_metal_graph_encoder_concurrency_add  (ggml_metal_graph_encoder_t ctx, const struct ggml_tensor * node);

#ifdef __cplusplus
}
#endif
