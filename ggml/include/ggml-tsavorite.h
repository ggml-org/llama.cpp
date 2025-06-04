// ------------------------------------------------------------------------------
// Copyright (c) 2023 Tsavorite Scalable Intelligence, Inc . All rights reserved.
//
//
// This file is the confidential and proprietary property of
// Tsavorite Scalable Intelligence, Inc
//
// Possession or use of this file requires a written license from
// Tsavorite Scalable Intelligence, Inc

/******************************************************************************
 * File: ggml-tsavorite.h
 * Author TSI Inc
 *
 * Description:
 * ***************************************************************************/

//
//
// Note: this description is outdated
//
// An interface allowing to compute ggml_cgraph with tSovrite
//
// This is a fully functional interface that extends ggml with Hardware Accelerator support for
// tSovrite devices. A similar interface can be created for other GPU backends (e.g. Vulkan, CUDA,
// etc.)
//
// How it works?
//
// As long as your program can create and evaluate a ggml_cgraph on the CPU, you can use this
// interface to evaluate the same graph on the GPU. Instead of using ggml_graph_compute(), you
// use ggml_tsavorite_graph_compute()
//
// You only need to make sure that all memory buffers that you used during the graph creation
// are mapped to the device unified memory with the ggml_tsavorite_add_buffer() function. This
// mapping is used during the graph evaluation to determine the arguments of the compute kernels.
//
// Synchronization between device and host memory (for example for input and output tensors)
// is done with the ggml_tsavorite_set_tensor() and ggml_tsavorite_get_tensor() functions.
//

#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include "TestModel.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TSAVORITE_KERNEL_SIZE 64
#define TSAVORITE_DEVICE_MAX_BUF_LEN 1024 * 1024 * 128

enum ggml_tsavorite_input_tensors_count {
  TSAVORITE_UNARY_INPUT_TENSORS = 1,
  TSAVORITE_TWO_INPUT_TENSORS = 2
};

enum ggml_tsavorite_log_type {
  GGML_TSAVORITE_LOG_NONE,
  GGML_TSAVORITE_LOG_CONT,
  GGML_TSAVORITE_LOG_ERROR,
  GGML_TSAVORITE_LOG_WARN,
  GGML_TSAVORITE_LOG_DEBUG,
  GGML_TSAVORITE_LOG_INFO,
  GGML_TSAVORITE_LOG_ALL
};

enum ggml_tsavorite_kernel_mode {
    GGML_TSAVORITE_KERNEL_MODE_CPU,
    GGML_TSAVORITE_KERNEL_MODE_MLIR
};

enum ggml_tsavorite_kernel_mode ggml_tsavorite_kernel_mode_flag = GGML_TSAVORITE_KERNEL_MODE_MLIR; 
enum ggml_tsavorite_log_type ggml_tsavorite_log_type_val = GGML_TSAVORITE_LOG_ALL;
#define GGML_TSAVORITE_LOG_INFO(...)                                                               \
  do {                                                                                             \
    if (ggml_tsavorite_log_type_val >= GGML_TSAVORITE_LOG_INFO) {                                  \
      ggml_log_internal(GGML_LOG_LEVEL_INFO, __VA_ARGS__);                                         \
    }                                                                                              \
  } while (0)
#define GGML_TSAVORITE_LOG_DEBUG(...)                                                              \
  do {                                                                                             \
    if (ggml_tsavorite_log_type_val >= GGML_TSAVORITE_LOG_DEBUG) {                                 \
      ggml_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__);                                        \
    }                                                                                              \
  } while (0)
#define GGML_TSAVORITE_LOG_WARN(...)                                                               \
  do {                                                                                             \
    if (ggml_tsavorite_log_type_val >= GGML_TSAVORITE_LOG_WARN) {                                  \
      ggml_log_internal(GGML_LOG_LEVEL_WARN, __VA_ARGS__);                                         \
    }                                                                                              \
  } while (0)
#define GGML_TSAVORITE_LOG_ERROR(...)                                                              \
  do {                                                                                             \
    if (ggml_tsavorite_log_type_val >= GGML_TSAVORITE_LOG_ERROR) {                                 \
      ggml_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__);                                        \
    }                                                                                              \
  } while (0)
#define GGML_TSAVORITE_LOG_CONT(...)                                                               \
  do {                                                                                             \
    if (ggml_tsavorite_log_type_val >= GGML_TSAVORITE_LOG_CONT) {                                  \
      ggml_log_internal(GGML_LOG_LEVEL_CONT, __VA_ARGS__);                                         \
    }                                                                                              \
  } while (0)

enum ggml_tsavorite_tensor_data_type {
  GGML_TSAVORITE_TENSOR_HEADER,
  GGML_TSAVORITE_TENSOR_LEAF1,
  GGML_TSAVORITE_TENSOR_LEAF2,
  GGML_TSAVORITE_TENSOR_NODE,
  GGML_TSAVORITE_TENSOR_END_DATA
};

enum ggml_tsavorite_kernel_type {
  GGML_TSAVORITE_KERNEL_TYPE_ADD,
  GGML_TSAVORITE_KERNEL_TYPE_SUB,
  GGML_TSAVORITE_KERNEL_TYPE_MULT,
  GGML_TSAVORITE_KERNEL_TYPE_DIV,
  GGML_TSAVORITE_KERNEL_TYPE_SQRT,
  GGML_TSAVORITE_KERNEL_TYPE_NEG,
  GGML_TSAVORITE_KERNEL_TYPE_ABS,
  GGML_TSAVORITE_KERNEL_TYPE_SIN,
  GGML_TSAVORITE_KERNEL_TYPE_SIGMOID,
  GGML_TSAVORITE_KERNEL_TYPE_SILU,

  GGML_TSAVORITE_KERNEL_TYPE_COUNT
};

// max memory buffers that can be mapped to the device
#define GGML_TSAVORITE_MAX_BUFFERS 64

// max number of TSAVORITECommandBuffer used to submit a graph for processing
#define GGML_TSAVORITE_MAX_COMMAND_BUFFERS 8
#define tsi_nil 0
#define TSI_UNUSED(x) (void)(x)

typedef struct tensor_log_ {
  uint32_t leaf1_len;
  uint32_t leaf2_len;
  uint32_t node_len;
  enum ggml_tsavorite_tensor_data_type data_type;
  enum ggml_tsavorite_kernel_type kernel_type;
  uint64_t num_of_op;
  FILE *log_file;
  const ggml_tensor *tensor;
} tensor_log;

extern void _mlir_ciface_txe_add_host(void *a, void *b, void *res);
extern void _mlir_ciface_txe_sub_host(void *a, void *b, void *res);
extern void _mlir_ciface_txe_mult_host(void *a, void *b, void *res);
extern void _mlir_ciface_txe_div_host(void *a, void *b, void *res);
extern void _mlir_ciface_txe_sqrt_host(void *a, void *res);
extern void _mlir_ciface_txe_neg_host(void *a, void *res);
extern void _mlir_ciface_txe_abs_host(void *a, void *res);
extern void _mlir_ciface_txe_sin_host(void *a, void *res);
extern void _mlir_ciface_txe_sigmoid_host(void *a, void *res);
extern void _mlir_ciface_txe_silu_host(void *a, void *res);
extern void ggml_tsi_log_tensor_data(tensor_log log_data);

#define NUM_OF_TXES 1
#define MEM_REF_DESCRIPTOR_RANK 1

//
// backend API
// user-code should use only these functions
//

GGML_BACKEND_API ggml_backend_t ggml_backend_tsavorite_init(void);

GGML_BACKEND_API bool ggml_backend_is_tsavorite(ggml_backend_t backend);

GGML_BACKEND_API void ggml_backend_tsavorite_set_abort_callback(ggml_backend_t backend,
                                                                ggml_abort_callback abort_callback,
                                                                void *user_data);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_tsavorite_buffer_type(void);

// capture all command buffers committed the next time `ggml_backend_graph_compute` is called
GGML_BACKEND_API void ggml_backend_tsavorite_capture_next_compute(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_tsavorite_reg(void);

#ifdef __cplusplus
}
#endif
