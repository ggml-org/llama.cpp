/****************************************************************************
 *
 *    Copyright (c) 2017 - 2023 by Rockchip Corp.  All rights reserved.
 *
 *    The material in this file is confidential and contains trade secrets
 *    of Rockchip Corporation. This is proprietary information owned by
 *    Rockchip Corporation. No part of this work may be disclosed,
 *    reproduced, copied, transmitted, or used in any way for any purpose,
 *    without the express written permission of Rockchip Corporation.
 *
 *****************************************************************************/

#ifndef _RKNN_CUSTOM_OP_H
#define _RKNN_CUSTOM_OP_H

#ifdef __cplusplus
extern "C" {
#endif

#include "rknn_api.h"

#include <stdint.h>

/*
    Error code returned by the RKNN Custom Operator API.
*/
#define RKNN_WARNING_SKIP_CUSTOM_OP_COMPUTE -14 /* if custom op init callback funtion return this code and op type is supported by RKNN, it will use RKNN implementation. */

#define RKNN_CUSTOM_OP_MAX_STR_LEN 64
#define RKNN_CUSTOM_OP_MAX_VALUE_LEN 32
#define RKNN_CUSTOM_OP_EXPORT __attribute__((visibility("default")))

#ifdef __arm__
typedef uint32_t rknn_custom_op_interal_context;
#else
typedef uint64_t rknn_custom_op_interal_context;
#endif
/*
    the backend execution device of custom operator.
*/
typedef enum _rknn_target_type
{
  RKNN_TARGET_TYPE_CPU = 1, /* backend device is cpu */
  RKNN_TARGET_TYPE_GPU = 2, /* backend device is gpu */
  RKNN_TARGET_TYPE_MAX
} rknn_target_type;

typedef struct _rknn_gpu_op_context
{
  void* cl_context;
  void* cl_command_queue;
  void* cl_kernel;

} rknn_gpu_op_context;

typedef struct _rknn_custom_op_context
{
  rknn_target_type               target;       /* custom op backend target */
  rknn_custom_op_interal_context internal_ctx; /* the context of custom op*/
  rknn_gpu_op_context            gpu_ctx;      /* the gpu context of custom op */
  void*                          priv_data;    /* the private data managed by user */
} rknn_custom_op_context;

typedef struct _rknn_custom_op_tensor
{
  rknn_tensor_attr attr; /* the attribute of tensor buffer. */
  rknn_tensor_mem  mem;  /* the memory information of tensor. */
} rknn_custom_op_tensor;

typedef struct _rknn_custom_op_attr
{
  char             name[RKNN_MAX_NAME_LEN]; /* the name of operator atrributes. */
  rknn_tensor_type dtype;                   /* the data type of operator attributes, indicate the 'array' type. */
  uint32_t         n_elems;                 /* the number of 'array'. */
  void* data; /* the array pointer of operator attributes, the data type of each element is determined by type. */
} rknn_custom_op_attr;

/*
    the information of custom operator to add to the rknn_context.
*/
typedef struct _rknn_custom_op
{
  uint32_t         version;                    /* custom op version */
  rknn_target_type target;                     /* custom op backend target */
  char             op_type[RKNN_MAX_NAME_LEN]; /* custom op type */

  char  cl_kernel_name[RKNN_MAX_NAME_LEN]; /* the opencl kernel name used by custom op */
  char* cl_kernel_source;  /* if cl_source_size > 0, pointer to the cl kernel source string, if cl_source_size = 0,
                              filepath to the cl kernel file. */
  uint64_t cl_source_size; /* the size of cl_kernel_source */
  char     cl_build_options[RKNN_MAX_NAME_LEN]; /* the options for opencl to build clProgram used by custom op */

  /**
   * The callback function sets that the users need to code
   */
  int (*init)(rknn_custom_op_context* op_ctx, rknn_custom_op_tensor* inputs, uint32_t n_inputs,
              rknn_custom_op_tensor* outputs, uint32_t n_outputs); /* [optional] custom op kernel init falllback function*/
  int (*prepare)(rknn_custom_op_context* op_ctx, rknn_custom_op_tensor* inputs, uint32_t n_inputs,
                 rknn_custom_op_tensor* outputs, uint32_t n_outputs); /* [optional] custom op kernel prepare falllback function*/
  int (*compute)(rknn_custom_op_context* op_ctx, rknn_custom_op_tensor* inputs, uint32_t n_inputs,
                 rknn_custom_op_tensor* outputs, uint32_t n_outputs); /* [required] custom op kernel compute falllback function */
  int (*compute_native)(rknn_custom_op_context* op_ctx, rknn_custom_op_tensor* inputs, uint32_t n_inputs,
                        rknn_custom_op_tensor* outputs, uint32_t n_outputs); /* [optional] custom op kernel compute with native attribute falllback function */
  int (*destroy)(rknn_custom_op_context* op_ctx); /* [optional] custom op kernel compute falllback function */

} rknn_custom_op;

/**
 * dlopen custom op with so required this function
 */
typedef rknn_custom_op* (*get_custom_op_func)();

/*  rknn_register_custom_ops

    Register custom operators to rknn_context.
    Steps to use a custom op:
    1. Create a rknn_custom_op structure array and fill in it.
    2. Setup prepare/compute/compute_native/destroy callback function and add them to the
       rknn_custom_op.(compute is required and other function is optional, compute_native is not supported now, set it
       to nullptr)
    3. Call rknn_register_custom_ops to register the op type after rknn_init.
    input:
        rknn_context ctx            the handle of context.
        rknn_custom_op* op          the custom operator array, each of which contains operator information and calllback function.
        uint32_t custom_op_num      the length of rknn_custom_op array.
   return:
        int                         error code.
*/
int rknn_register_custom_ops(rknn_context ctx, rknn_custom_op* op, uint32_t custom_op_num);

/*  rknn_custom_op_get_op_attr

    input:
        rknn_custom_op_context* op_ctx  the handle of custom op context.
        const char* attr_name          the attribute name of operator.
        rknn_custom_op_attr* op_attr   the data and information of operator attributes.
*/
void rknn_custom_op_get_op_attr(rknn_custom_op_context* op_ctx, const char* attr_name, rknn_custom_op_attr* op_attr);

#ifdef __cplusplus
} // extern "C"
#endif

#endif //_RKNN_CUSTOM_OP_H
