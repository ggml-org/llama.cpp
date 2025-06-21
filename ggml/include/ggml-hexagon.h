 /*
 * Copyright (c) 2024-2025 The ggml authors
 */
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_HEXAGON_MAX_DEVICES    4
#define GGML_HEXAGON_BACKEND_NAME   "hexagon"

enum HEXAGONBackend {
    HEXAGON_BACKEND_QNNCPU  = 0,
    HEXAGON_BACKEND_QNNGPU  = 1,
    HEXAGON_BACKEND_QNNNPU  = 2,
    HEXAGON_BACKEND_CDSP    = 3,
    HEXAGON_BACKEND_GGML    = 4, //"fake" HEXAGON backend for compare performance between HEXAGON backend and ggml backend
};

//0: general approach through QNN:offload ggmlop to QNN(QNNCPU, QNNGPU, QNNNPU）
//1: special approach through QNN-SINGLEGRAPH:mapping entire ggml cgraph to a single QNN graph
//2: general approach through Hexagon cDSP:offload ggmlop to Hexagon cDSP directly
enum hwaccel_approach_type {
     HWACCEL_QNN            = 0,
     HWACCEL_QNN_SINGLEGRAPH= 1,
     HWACCEL_CDSP           = 2,
};

GGML_BACKEND_API ggml_backend_t     ggml_backend_hexagon_init(size_t dev_num, const char * qnn_lib_path);

GGML_BACKEND_API bool               ggml_backend_is_hexagon(ggml_backend_t backend);

GGML_BACKEND_API int                ggml_backend_hexagon_get_device_count(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_hexagon_reg(void);

GGML_BACKEND_API const char *       ggml_backend_hexagon_get_devname(size_t dev_num);

GGML_BACKEND_API void               ggml_backend_set_hexagon_cfg(int new_hexagon_backend, int new_hwaccel_approach);

#ifdef __cplusplus
}
#endif
