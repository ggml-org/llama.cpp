/****************************************************************************
 *
 *    Copyright (c) 2017 - 2018 by Rockchip Corp.  All rights reserved.
 *
 *    The material in this file is confidential and contains trade secrets
 *    of Rockchip Corporation. This is proprietary information owned by
 *    Rockchip Corporation. No part of this work may be disclosed,
 *    reproduced, copied, transmitted, or used in any way for any purpose,
 *    without the express written permission of Rockchip Corporation.
 *
 *****************************************************************************/

#ifndef _RKNN_MATMUL_API_H
#define _RKNN_MATMUL_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include "rknn_api.h"

typedef rknn_context rknn_matmul_ctx;

typedef enum _rknn_matmul_quant_type
{
  RKNN_QUANT_TYPE_PER_LAYER_SYM    = 0,
  RKNN_QUANT_TYPE_PER_LAYER_ASYM   = 1,
  RKNN_QUANT_TYPE_PER_CHANNEL_SYM  = 2,
  RKNN_QUANT_TYPE_PER_CHANNEL_ASYM = 3,
  RKNN_QUANT_TYPE_PER_GROUP_SYM    = 4,
  RKNN_QUANT_TYPE_PER_GROUP_ASYM   = 5,
} rknn_matmul_quant_type;

typedef struct _rknn_quant_params
{
  char name[RKNN_MAX_NAME_LEN];

  // matmul tensor scale
  float*  scale;
  int32_t scale_len;

  // matmul tensor zero point
  int32_t* zp;
  int32_t  zp_len;

} rknn_quant_params;

typedef enum _rknn_matmul_type
{
  RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32 = 1,
  RKNN_INT8_MM_INT8_TO_INT32         = 2,
  RKNN_INT8_MM_INT8_TO_INT8          = 3,
  RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16 = 4,
  RKNN_FLOAT16_MM_INT8_TO_FLOAT32    = 5,
  RKNN_FLOAT16_MM_INT8_TO_FLOAT16    = 6,
  RKNN_FLOAT16_MM_INT4_TO_FLOAT32    = 7,
  RKNN_FLOAT16_MM_INT4_TO_FLOAT16    = 8,
  RKNN_INT8_MM_INT8_TO_FLOAT32       = 9,
  RKNN_INT4_MM_INT4_TO_INT16         = 10,
  RKNN_INT8_MM_INT4_TO_INT32         = 11,
  RKNN_FLOAT16_MM_INT4_TO_BFLOAT16   = 12,
  RKNN_INT8_MM_INT4_TO_FLOAT16       = 15,
} rknn_matmul_type;

inline static const char* get_matmul_type_string(rknn_matmul_type type)
{
  switch (type) {
  case RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32:
    return "RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32";
  case RKNN_INT8_MM_INT8_TO_INT32:
    return "RKNN_INT8_MM_INT8_TO_INT32";
  case RKNN_INT8_MM_INT8_TO_INT8:
    return "RKNN_INT8_MM_INT8_TO_INT8";
  case RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16:
    return "RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16";
  case RKNN_FLOAT16_MM_INT8_TO_FLOAT32:
    return "RKNN_FLOAT16_MM_INT8_TO_FLOAT32";
  case RKNN_FLOAT16_MM_INT8_TO_FLOAT16:
    return "RKNN_FLOAT16_MM_INT8_TO_FLOAT16";
  case RKNN_INT4_MM_INT4_TO_INT16:
    return "RKNN_INT4_MM_INT4_TO_INT16";
  case RKNN_FLOAT16_MM_INT4_TO_FLOAT32:
    return "RKNN_FLOAT16_MM_INT4_TO_FLOAT32";
  case RKNN_FLOAT16_MM_INT4_TO_FLOAT16:
    return "RKNN_FLOAT16_MM_INT4_TO_FLOAT16";
  case RKNN_INT8_MM_INT4_TO_INT32:
    return "RKNN_INT8_MM_INT4_TO_INT32";
  case RKNN_INT8_MM_INT8_TO_FLOAT32:
    return "RKNN_INT8_MM_INT8_TO_FLOAT32";
  case RKNN_FLOAT16_MM_INT4_TO_BFLOAT16:
    return "RKNN_FLOAT16_MM_INT4_TO_BFLOAT16";
  default:
    return "UNKNOW";
  }
}

typedef struct _rknn_matmul_tensor_attr
{
  char name[RKNN_MAX_NAME_LEN];

  // indicate A(M, K) or B(K, N) or C(M, N)
  uint32_t n_dims;
  uint32_t dims[RKNN_MAX_DIMS];

  // matmul tensor size
  uint32_t size;

  // matmul tensor data type
  // int8 : A, B
  // int32: C
  rknn_tensor_type type;

} rknn_matmul_tensor_attr;

typedef struct _rknn_matmul_io_attr
{
  // indicate A(M, K) or B(K, N) or C(M, N)
  rknn_matmul_tensor_attr A;
  rknn_matmul_tensor_attr B;
  rknn_matmul_tensor_attr C;
} rknn_matmul_io_attr;

/*
  matmul dynamic shape struct
*/
typedef struct _rknn_matmul_shape
{
  int32_t M;
  int32_t K;
  int32_t N;
} rknn_matmul_shape;

/*
  the layout of matmul input/output tensor.
*/
typedef enum
{
  RKNN_MM_LAYOUT_NORM    = 0,
  RKNN_MM_LAYOUT_NATIVE  = 1,
  RKNN_MM_LAYOUT_TP_NORM = 2,
} rknn_matmul_layout;

/*
  matmul information struct
 */
typedef struct rknn_matmul_info_t
{
  int32_t M;
  int32_t K; // limit: RK3566/3568: int8 type must be aligned with 32byte, float16 type must be aligned with 16byte;
             // RK3562:      int8 type must be aligned with 32byte, float16 type must be aligned with 32byte;
             // RK3588/3576: int8 type must be aligned with 32byte, float16 type must be aligned with 32byte,
             //              int4 type must be aligned with 32byte;
  int32_t N; // limit: RK3566/3568: int8 type must be aligned with 16byte, float16 type must be aligned with 8byte;
             // RK3562:      int8 type must be aligned with 16byte, float16 type must be aligned with 8byte;
             // RK3588/3576: int8 type must be aligned with 32byte, float16 type must be aligned with 16byte,
             //              int4 type must be aligned with 64byte;
  // matmul data type
  // int4: int4(A) x int4(B) -> int16(C)
  // int8: int8(A) x int8(B) -> int32(C)
  // float16: float16(A) x float16(B) -> float32(C)
  rknn_matmul_type type;

  // matmul native layout for B
  // 0: normal layout
  // 1: native layout
  int16_t B_layout;

  // matmul quant type for B
  // A and C only support per layer
  // 0: per layer
  // 1: per channel
  // 2: per group
  int16_t B_quant_type;

  // matmul native layout for A and C
  // 0: normal layout
  // 1: native layout
  int16_t AC_layout;

  // matmul quant type for A and C, only support 0
  int16_t AC_quant_type;

  // iommu domain id, each domain has 4GB of space
  int32_t iommu_domain_id;

  // B_quant_type set 2, group size is enable
  int16_t group_size;

  // reserved field
  int8_t reserved[34];
} rknn_matmul_info;

/*  rknn_matmul_create

    params:
        rknn_matmul_ctx *ctx           the handle of context.
        rknn_matmul_info *info         the matmal information.
        rknn_matmul_io_attr *io_attr   inputs/output attribute
    return:
        int                         error code
*/
int rknn_matmul_create(rknn_matmul_ctx* ctx, rknn_matmul_info* info, rknn_matmul_io_attr* io_attr);

/*  rknn_matmul_create_dynamic_shape

    params:
        rknn_matmul_ctx *ctx                the handle of context.
        rknn_matmul_info *info              the matmal information.
        int shape_num                       the supported shape number of matmul.
        rknn_matmul_shape dynamic_shapes[]  the supported M,K,N shape struct array.
        rknn_matmul_io_attr *io_attr        the array of inputs and output attribute
    return:
        int                                 error code
*/
/*
  原来的info.M, K, N无效
*/
int rknn_matmul_create_dynamic_shape(rknn_matmul_ctx* ctx, rknn_matmul_info* info, int shape_num,
                                 rknn_matmul_shape dynamic_shapes[], rknn_matmul_io_attr io_attrs[]);

/* rknn_matmul_set_io_mem

    params:
        rknn_matmul_ctx ctx            the handle of context.
        rknn_tensor_mem *mem           the pointer of tensor memory information.
        rknn_matmul_tensor_attr *attr  the attribute of input or output tensor buffer.
    return:
        int                         error code.

    formula:
      C = A * B,

    limit:
      K max:   k <= 10240
      K limit: RK3566/3568: int8 type must be aligned with 32byte, float16 type must be aligned with 16byte;
               RK3562:      int8 type must be aligned with 32byte, float16 type must be aligned with 32byte;
               RK3588/3576: int8 type must be aligned with 32byte, float16 type must be aligned with 32byte,
                            int4 type must be aligned with 32byte;
      N limit: RK3566/3568: int8 type must be aligned with 16byte, float16 type must be aligned with 8byte;
               RK3562:      int8 type must be aligned with 16byte, float16 type must be aligned with 8byte;
               RK3588/3576: int8 type must be aligned with 32byte, float16 type must be aligned with 16byte,
                            int4 type must be aligned with 64byte;
    A shape: M x K
      normal layout: (M, K)
              [M1K1, M1K2, ..., M1Kk,
               M2K1, M2K2, ..., M2Kk,
               ...
               MmK1, MmK2, ..., MmKk]
      for RK3566/3568：
      int8:
      native layout: (K / 8, M, 8)
              [K1M1, K2M1,  ..., K8M1,
               K9M2, K10M2, ..., K16M2,
               ...
               K(k-7)Mm, K(k-6)Mm, ..., KkMm]
      float16:
      native layout: (K / 4, M, 4)
              [K1M1, K2M1,  ..., K4M1,
               K9M2, K10M2, ..., K8M2,
               ...
               K(k-3)Mm, K(k-2)Mm, ..., KkMm]
      for RK3562：
      int8:
      native layout: (K / 16, M, 16)
              [K1M1, K2M1,  ..., K16M1,
               K17M2, K18M2, ..., K32M2,
               ...
               K(k-15)Mm, K(k-14)Mm, ..., KkMm]
      float16:
      native layout: (K / 8, M, 8)
              [K1M1, K2M1,  ..., K8M1,
               K9M2, K10M2, ..., K16M2,
               ...
               K(k-7)Mm, K(k-6)Mm, ..., KkMm]
      for RK3588/3576：
      int4:
      native layout: (K / 32, M, 32)
              [K1M1, K2M1,  ..., K32M1,
               K33M2, K10M2, ..., K64M2,
               ...
               K(k-31)Mm, K(k-30)Mm, ..., KkMm]
      int8:
      native layout: (K / 16, M, 16)
              [K1M1, K2M1,  ..., K16M1,
               K17M2, K18M2, ..., K32M2,
               ...
               K(k-15)Mm, K(k-14)Mm, ..., KkMm]
      float16:
      native layout: (K / 8, M, 8)
              [K1M1, K2M1,  ..., K8M1,
               K9M2, K10M2, ..., K16M2,
               ...
               K(k-7)Mm, K(k-6)Mm, ..., KkMm]
    B shape: K x N
      normal layout: (K, N)
              [K1N1, K1N2, ..., K1Nn,
               K2N1, K2N2, ..., K2Nn,
               ...
               KkN1, KkN2, ..., KkNn]
      for RK3566/3568：
      int8:
      native layout: (N / 16, K / 32, 16, 32)
              [K1N1,  K2N1,  ..., K32N1,
               K1N2,  K2N2,  ..., K32N2,
               ...
               K1N16, K2N16, ..., K32N16,
               K33N1, K34N1, ..., K64N1,
               K33N2, K34N2, ..., K64N2,
               ...
               K(k-31)N16, K(k-30)N16, ..., KkN16,
               K1N17, K2N17, ..., K32N17,
               K1N18, K2N18, ..., K32N18,
               ...
               K(k-31)Nn, K(k-30)Nn, ..., KkNn]
      float16:
      native layout: (N / 8, K / 16, 8, 16)
              [K1N1,  K2N1,  ..., K16N1,
               K1N2,  K2N2,  ..., K16N2,
               ...
               K1N8,  K2N8,  ..., K16N8,
               K17N1, K18N1, ..., K32N1,
               K17N2, K18N2, ..., K32N2,
               ...
               K(k-15)N8, K(k-30)N8, ..., KkN8,
               K1N9,  K2N9,  ..., K16N9,
               K1N10, K2N10, ..., K16N10,
               ...
               K(k-15)Nn, K(k-14)Nn, ..., KkNn]
      for RK3562：
      int8:
      native layout: (N / 16, K / 32, 16, 32)
              [K1N1,  K2N1,  ..., K32N1,
               K1N2,  K2N2,  ..., K32N2,
               ...
               K1N16, K2N16, ..., K32N16,
               K33N1, K34N1, ..., K64N1,
               K33N2, K34N2, ..., K64N2,
               ...
               K(k-31)N16, K(k-30)N16, ..., KkN16,
               K1N17, K2N17, ..., K32N17,
               K1N18, K2N18, ..., K32N18,
               ...
               K(k-31)Nn, K(k-30)Nn, ..., KkNn]
      float16:
      native layout: (N / 8, K / 32, 8, 32)
              [K1N1,  K2N1,  ..., K32N1,
               K1N2,  K2N2,  ..., K32N2,
               ...
               K1N8,  K2N8,  ..., K32N8,
               K33N1, K34N1, ..., K64N1,
               K33N2, K34N2, ..., K64N2,
               ...
               K(k-31)N8, K(k-30)N8, ..., KkN8,
               K1N9,  K2N9,  ..., K16N9,
               K1N10, K2N10, ..., K16N10,
               ...
               K(k-31)Nn, K(k-30)Nn, ..., KkNn]
      for RK3588：
      when K > 8192, the B data will be split into T segments.
      int T = std::ceil(K / 8192);
      For example:  normal layout  -> native layout
      K =  20488, N = 4096, T = 3, the data will be split into 3 segments.
      subN = rknn_matmul_io_attr.B.dims[2];
      subK = rknn_matmul_io_attr.B.dims[3];
                                      (8196, 4096)          (4096 / subN, 8196 / subK, subN, subK)
        (K, N) = (20488, 4096)  ->    (8196, 4096)    ->    (4096 / subN, 8196 / subK, subN, subK)
                 normal layout        (4096, 4096)          (4096 / subN, 4096 / subK, subN, subK)
                                     T normal layout                    T native layout
      It is recommended to use the rknn_B_normal_layout_to_native_layout interface for direct data conversion.
      for RK3576：
      when K > 4096, the B data will be split into T segments.
      int T = std::ceil(K / 4096);
      For example:  normal layout  -> native layout
      K =  10240, N = 2048, T = 3, the data will be split into 3 segments.
      subN = rknn_matmul_io_attr.B.dims[2];
      subK = rknn_matmul_io_attr.B.dims[3];
                                      (4096, 2048)          (2048 / subN, 4096 / subK, subN, subK)
        (K, N) = (10240, 2048)  ->    (4096, 2048)    ->    (2048 / subN, 4096 / subK, subN, subK)
                 normal layout        (2048, 2048)          (2048 / subN, 2048 / subK, subN, subK)
                                     T normal layout                    T native layout
      It is recommended to use the rknn_B_normal_layout_to_native_layout interface for direct data conversion.
      for RK3588/3576：
      int4:
      native layout: (N / 64, K / 32, 64, 32)
              [K1N1,  K2N1,  ..., K32N1,
               K1N2,  K2N2,  ..., K32N2,
               ...
               K1N64, K2N64, ..., K32N64,
               K33N1, K34N1, ..., K64N1,
               K33N2, K34N2, ..., K64N2,
               ...
               K(k-31)N64, K(k-30)N64, ..., KkN64,
               K1N65, K2N65, ..., K32N65,
               K1N66, K2N66, ..., K32N66,
               ...
               K(k-31)Nn, K(k-30)Nn, ..., KkNn]
      int8:
      native layout: (N / 32, K / 32, 32, 32)
              [K1N1,  K2N1,  ..., K32N1,
               K1N2,  K2N2,  ..., K32N2,
               ...
               K1N32, K2N32, ..., K32N32,
               K33N1, K34N1, ..., K64N1,
               K33N2, K34N2, ..., K64N2,
               ...
               K(k-31)N32, K(k-30)N32, ..., KkN32,
               K1N33, K2N33, ..., K32N33,
               K1N34, K2N34, ..., K32N34,
               ...
               K(k-31)Nn, K(k-30)Nn, ..., KkNn]
      float16:
      native layout: (N / 16, K / 32, 16, 32)
              [K1N1,  K2N1,  ..., K32N1,
               K1N2,  K2N2,  ..., K32N2,
               ...
               K1N16, K2N16, ..., K32N16,
               K33N1, K34N1, ..., K64N1,
               K33N2, K34N2, ..., K64N2,
               ...
               K(k-31)N16, K(k-30)N16, ..., KkN16,
               K1N17, K2N17, ..., K32N17,
               K1N18, K2N18, ..., K32N18,
               ...
               K(k-31)Nn, K(k-30)Nn, ..., KkNn]
    C shape: M x N
      normal layout: (M, N)
              [M1N1, M1N2, ..., M1Nn,
               M2N1, M2N2, ..., M2Nn,
               ...
               MmN1, MmN2, ..., MmNn]
      native layout: (N / 4, M, 4)
              [N1M1, N2M1, ..., N4M1,
               N5M2, N6M2, ..., N8M2,
               ...
               N(n-3)Mm, N(n-2)Mm, ..., NnMm]
      for RK3588：
      int4:
      native layout: (N / 8, M, 8)
              [N1M1, N2M1, ..., N8M1,
               N9M2, N10M2, ..., N16M2,
               ...
               N(n-7)Mm, N(n-6)Mm, ..., NnMm]
 */
int rknn_matmul_set_io_mem(rknn_matmul_ctx ctx, rknn_tensor_mem* mem, rknn_matmul_tensor_attr* attr);

/*  rknn_matmul_set_core_mask

    set rknn core mask.(only support RK3588 in current)

    RKNN_NPU_CORE_AUTO: auto mode, default value
    RKNN_NPU_CORE_0: core 0 mode
    RKNN_NPU_CORE_1: core 1 mode
    RKNN_NPU_CORE_2: core 2 mode
    RKNN_NPU_CORE_0_1: combine core 0/1 mode
    RKNN_NPU_CORE_0_1_2: combine core 0/1/2 mode

    input:
        rknn_matmul_ctx context     the handle of context.
        rknn_core_mask core_mask    the core mask.
    return:
        int                         error code.
*/
int rknn_matmul_set_core_mask(rknn_matmul_ctx context, rknn_core_mask core_mask);

/*  rknn_matmul_set_quant_params

    set quant params.(only support matmul type RKNN_INT8_MM_INT8_TO_INT8, RKNN_INT8_MM_INT8_TO_INT32)

    input:
        rknn_matmul_ctx context     the handle of context.
        rknn_quant_params params    quant params.
    return:
        int                         error code.
*/
int rknn_matmul_set_quant_params(rknn_matmul_ctx context, rknn_quant_params* params);

/*  rknn_matmul_get_quant_params

    get per channel quant params.(only support matmul type RKNN_INT8_MM_INT8_TO_INT32)

    input:
        rknn_matmul_ctx context     the handle of context.
        rknn_quant_params params    quant params.
        float scale    get scale for user.
    return:
        int                         error code.
*/
int rknn_matmul_get_quant_params(rknn_matmul_ctx ctx, rknn_quant_params* params, float* scale);

/*  rknn_matmul_set_dynamic_shape

    set the matmul input/output shape. matmul will run under current input shape after rknn_matmul_set_dynamic_shape,
    only support M dynamicly now.

    input:
        rknn_matmul_ctx ctx         the handle of context.
        rknn_matmul_shape* shape    the M,K,N shape of matmul currently
    return:
        int                         error code.
*/
int rknn_matmul_set_dynamic_shape(rknn_matmul_ctx ctx, rknn_matmul_shape* shape);

/*  rknn_matmul_run

    run the matmul in blocking mode

    params:
        rknn_matmul_ctx ctx         the handle of context.
    return:
        int                         error code.
 */
int rknn_matmul_run(rknn_matmul_ctx ctx);

/*  rknn_matmul_destroy

    destroy the matmul context

    params:
        rknn_matmul_ctx ctx         the handle of context.
    return:
        int                         error code.
 */
int rknn_matmul_destroy(rknn_matmul_ctx ctx);

/*  rknn_B_normal_layout_to_native_layout

    change the B normal layout buffer to native layout buffer

    params:
        void* B_input               B normal layout buffer.
        void* B_output              B native layout buffer.
        int   K                     K
        int   N                     N
        rknn_matmul_info info       matmul info
    return:
        int                         error code.
 */
int rknn_B_normal_layout_to_native_layout(void* B_input, void* B_output, int K, int N, rknn_matmul_info* info);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // _RKNN_MATMUL_API_H