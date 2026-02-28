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
//
// An interface allowing to compute ggml_cgraph with tSavorite
//
// This is a fully functional interface that extends ggml with Hardware Accelerator support for
// tSavorite devices. A similar interface can be created for other GPU backends (e.g. Vulkan, CUDA,
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
// See TMU MUL_MAT TILE BLOB ABI below for current contract.
//

#pragma once



#include "ggml-backend.h"
#include "ggml.h"

#include "TestModel.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef TSAVORITE_GGML_ASSERT
#define TSAVORITE_GGML_ASSERT(x) do { if (!(x)) abort(); } while (0)
#endif


#define TSAVORITE_DEVICE_MAX_BUF_LEN (1024 * 1024 * 128)

enum ggml_tsavorite_input_tensors_count {
  TSAVORITE_UNARY_INPUT_TENSORS = 1,
  TSAVORITE_TWO_INPUT_TENSORS = 2,
  TSAVORITE_IGNORE_TENSORS
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

extern enum ggml_tsavorite_kernel_mode ggml_tsavorite_kernel_mode_flag; 
extern enum ggml_tsavorite_log_type ggml_tsavorite_log_type_val;

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
  GGML_TSAVORITE_KERNEL_TYPE_SQR,
  GGML_TSAVORITE_KERNEL_TYPE_NEG,
  GGML_TSAVORITE_KERNEL_TYPE_ABS,
  GGML_TSAVORITE_KERNEL_TYPE_SIN,
  GGML_TSAVORITE_KERNEL_TYPE_RMS_NORM,
  GGML_TSAVORITE_KERNEL_TYPE_SIGMOID,
  GGML_TSAVORITE_KERNEL_TYPE_SILU,
  //Below GELU Kernel
  GGML_TSAVORITE_KERNEL_TYPE_REGLU,
  GGML_TSAVORITE_KERNEL_TYPE_GEGLU,

  // Currently Below kernel Implemented
  GGML_TSAVORITE_KERNEL_TYPE_SWIGLU,

  GGML_TSAVORITE_KERNEL_TYPE_SWIGLU_OAI,
  GGML_TSAVORITE_KERNEL_TYPE_GEGLU_ERF,
  GGML_TSAVORITE_KERNEL_TYPE_GEGLU_QUICK,

  GGML_TSAVORITE_KERNEL_TYPE_SOFT_MAX,
  GGML_TSAVORITE_KERNEL_TYPE_MUL_MAT,
  GGML_TSAVORITE_KERNEL_TYPE_RESHAPE,
  GGML_TSAVORITE_KERNEL_TYPE_VIEW,
  GGML_TSAVORITE_KERNEL_TYPE_PERMUTE,
  GGML_TSAVORITE_KERNEL_TYPE_TRANSPOSE,

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
  enum ggml_op kernel_type;
  uint64_t num_of_op;
  FILE *log_file;
  const ggml_tensor *tensor;
} tensor_log;

/*
 * Data Types Enum
 */
typedef enum tsi_data_type_ {
	DATA_TYPE_F32_INDEX = 0,
	DATA_TYPE_F16_INDEX,
	DATA_TYPE_MAX_INDEX
} tsi_data_type;
/*
 * FP32
 */
extern void _mlir_ciface_txe_add_host(void *a, void *b, void *res);
extern void _mlir_ciface_txe_sub_host(void *a, void *b, void *res);
extern void _mlir_ciface_txe_mult_host(void *a, void *b, void *res);
extern void _mlir_ciface_txe_div_host(void *a, void *b, void *res);
extern void _mlir_ciface_txe_sqrt_host(void *a, void *res);
extern void _mlir_ciface_txe_sqr_host(void *a, void *res);
extern void _mlir_ciface_txe_neg_host(void *a, void *res);
extern void _mlir_ciface_txe_abs_host(void *a, void *res);
extern void _mlir_ciface_txe_sin_host(void *a, void *res);
extern void _mlir_ciface_txe_sigmoid_host(void *a, void *res);
extern void _mlir_ciface_txe_silu_host(void *a, void *res);
extern void _mlir_ciface_txe_swiglu_host(void *a, void *b, void *res);
extern void _mlir_ciface_txe_soft_max_host(void *a, void *b, void *res, void *buf);
extern void _mlir_ciface_txe_rms_norm_host(void *a, void *res, void *buf);


// ============================================================================
// TMU TILE CONSTANTS
// ============================================================================

// Fixed TMU tile geometry
// Max rows per TMU call (MAR window)
#define TMU_M_TILE_MAX   64     // rows

// TMU FP32 store granularity (floats per output cacheline)
#define TMU_N_BLOCK     32     // columns (PP width)

// K must be multiple of this
// K alignment for FP32 streaming
#define TMU_K_MULTIPLE  32


#define TMU_NUM_K_BUCKETS 7


// ============================================================================
// TMU MUL_MAT TILE BLOB ABI (STATIC AOT, SINGLE‑PHASE PER‑K)
// ============================================================================
//
// SINGLE‑PHASE CONTRACT (CURRENT DESIGN)
//
// FUNCTION SIGNATURE (C ABI):
//
//   void tmu_mul_mat_k<K>(
//       const void * A_tile,   // [1,1,64,K] FP32, packed, zero‑padded
//       const void * B_tile,   // [1,1,32,K] FP32, packed, zero‑padded
//       void       * C_tile    // [1,1,64,32] FP32 scratchpad + output
//   );
//
// SEMANTICS (per invocation):
//
//   • Blob LOADS PP from C_tile
//   • Executes PP = A × B + PP for all internal K stripes
//   • LAST internal update materializes C = A × B + PP
//   • Blob STORES result back to C_tile
//
// IMPORTANT RULES:
//
//   • C_tile acts as BOTH scratchpad and final output
//   • Host MUST zero C_tile before the first K‑chunk
//   • Host MAY call multiple K‑blobs sequentially:
//       K = 2048 + 2048 + 2048 + 2048 + 2048 + 512 + 128 + 32
//   • No BEGIN / ACCUM / FINAL phases exist in this ABI
//
// SOFTWARE RESPONSIBILITY:
//
//   • Decompose K into supported buckets
//   • Pack A/B tiles per K‑chunk
//   • Zero C_tile once per output tile
//   • Call tmu_mul_mat_k<K>() in K‑decomposition order
//   • Copy C_tile → GGML output after last K call
//
// ============================================================================

typedef void (*tmu_mul_mat_tile_fn)(
    const void * A_tile,
    const void * B_tile,
    void       * C_tile
);

/* ---- Supported K bucket values ---- */
#define TMU_K_BUCKET_32   32
#define TMU_K_BUCKET_64   64
#define TMU_K_BUCKET_128  128
#define TMU_K_BUCKET_256  256
#define TMU_K_BUCKET_512  512
#define TMU_K_BUCKET_1024 1024
#define TMU_K_BUCKET_2048 2048

/* ---- Externs generated by AOT blobs (ONE per K) ---- */

void tmu_mul_mat_k32  (const void *A, const void *B, void *C);
void tmu_mul_mat_k64  (const void *A, const void *B, void *C);
void tmu_mul_mat_k128 (const void *A, const void *B, void *C);
void tmu_mul_mat_k256 (const void *A, const void *B, void *C);
void tmu_mul_mat_k512 (const void *A, const void *B, void *C);
void tmu_mul_mat_k1024(const void *A, const void *B, void *C);
void tmu_mul_mat_k2048(const void *A, const void *B, void *C);


extern void _mlir_ciface_txe_mul_mat_tile_f32_k32_host  (void *A_tile, void *B_tile, void *C_tile);
extern void _mlir_ciface_txe_mul_mat_tile_f32_k64_host  (void *A_tile, void *B_tile, void *C_tile);
extern void _mlir_ciface_txe_mul_mat_tile_f32_k128_host (void *A_tile, void *B_tile, void *C_tile);
extern void _mlir_ciface_txe_mul_mat_tile_f32_k256_host (void *A_tile, void *B_tile, void *C_tile);
extern void _mlir_ciface_txe_mul_mat_tile_f32_k512_host (void *A_tile, void *B_tile, void *C_tile);
extern void _mlir_ciface_txe_mul_mat_tile_f32_k1024_host(void *A_tile, void *B_tile, void *C_tile);
extern void _mlir_ciface_txe_mul_mat_tile_f32_k2048_host(void *A_tile, void *B_tile, void *C_tile);



/* 
 * FP16 Kernels 
 */
extern void _mlir_ciface_txe_add_16_host(void *a, void *b, void *res);
extern void _mlir_ciface_txe_sub_16_host(void *a, void *b, void *res);
extern void _mlir_ciface_txe_mult_16_host(void *a, void *b, void *res);
extern void _mlir_ciface_txe_div_16_host(void *a, void *b, void *res);
extern void _mlir_ciface_txe_sqrt_16_host(void *a, void *res);
extern void _mlir_ciface_txe_sqr_16_host(void *a, void *res);
extern void _mlir_ciface_txe_neg_16_host(void *a, void *res);
extern void _mlir_ciface_txe_abs_16_host(void *a, void *res);
extern void _mlir_ciface_txe_sin_16_host(void *a, void *res);
extern void _mlir_ciface_txe_sigmoid_16_host(void *a, void *res);
extern void _mlir_ciface_txe_silu_16_host(void *a, void *res);
extern void _mlir_ciface_txe_swiglu_16_host(void *a, void *b, void *res);
extern void _mlir_ciface_txe_rms_norm_16_host(void *a, void *res, void *buf);

extern void ggml_tsi_log_tensor_data(tensor_log log_data);

#define NUM_OF_TXES 1

// GGML supports tensors with a maximum rank of 4
#define MEM_REF_DESCRIPTOR_RANK 4
#define TSI_TVU_MEM_ALIGN 128

void tsi_cleanup();

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
