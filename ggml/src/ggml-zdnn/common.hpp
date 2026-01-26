#ifndef GGML_ZDNN_COMMON_HPP
#define GGML_ZDNN_COMMON_HPP

#include "ggml.h"
#include "ggml-impl.h"

#include "zdnn.h"

#include <vector>
#include <memory>

#define GGML_ZDNN_NAME    "zDNN"
#define GGML_ZDNN_VERSION ZDNN_VERNUM

#define ZDNN_CHECK(stmt)                \
    do {                                \
        zdnn_status status = (stmt);    \
        GGML_ASSERT(status == ZDNN_OK); \
    } while (0);

struct ggml_backend_zdnn_device_context {
    int zdnn_device;
    int zdnn_device_ref_count;

    bool has_parmblkformat_0;
    bool has_parmblkformat_1;  // checks for z17

    size_t max_size;

    char name[128];
};

struct ggml_backend_zdnn_context {
    int device;
    ggml_cgraph * gf;
};

struct ggml_backend_zdnn_buffer {
    void * data;
    ggml_backend_zdnn_buffer * extra;  // for bias, etc.
    size_t size;

    zdnn_tensor_desc pre_tfm_desc;
    zdnn_tensor_desc tfm_desc;
    zdnn_ztensor     ztensor;

    char name[GGML_MAX_NAME];
};

struct ggml_backend_zdnn_buffer_context {
    void * all_data;
    size_t all_size;
    bool owned;

    int n_buffers;
    std::vector<std::unique_ptr<ggml_backend_zdnn_buffer>> buffers;
};

#endif  // GGML_ZDNN_COMMON_HPP
