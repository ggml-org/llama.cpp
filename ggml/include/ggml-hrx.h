#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_backend_hrx_cache_stats {
    uint64_t graph_program_builds;
    uint64_t graph_program_hits;
    uint64_t prepared_program_builds;
    uint64_t prepared_program_hits;
};

GGML_BACKEND_API ggml_backend_t             ggml_backend_hrx_init(size_t device);
GGML_BACKEND_API bool                       ggml_backend_is_hrx(ggml_backend_t backend);
GGML_BACKEND_API int                        ggml_backend_hrx_get_device_count(void);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_hrx_buffer_type(size_t device);
GGML_BACKEND_API bool                       ggml_backend_hrx_get_cache_stats(ggml_backend_t                        backend,
                                                                             struct ggml_backend_hrx_cache_stats * stats);
GGML_BACKEND_API ggml_backend_reg_t         ggml_backend_hrx_reg(void);

#ifdef __cplusplus
}
#endif
