/*-------------------------------------------------------------------------
* Copyright (c) 2025 Ainekko, Co.
* SPDX-License-Identifier: Apache-2.0
*-------------------------------------------------------------------------
*/
/***********************************************************************/
/*! \file tensors.h
    \brief A C header that defines the tensor service available.
*/
/***********************************************************************/

#ifndef __TENSORS_H
#define __TENSORS_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__cplusplus) && (__cplusplus >= 201103L)
#include <cinttypes>
#if (__cplusplus < 202002L)
#include <cstdbool>
#endif
#else
#include <inttypes.h>
#include <stdbool.h>
#endif

/*! \def QUANT_LAST_TRANS
    \brief Tensor Quant instruction: Do not perform any more transformations.
*/
#define QUANT_LAST_TRANS 0

/*! \def QUANT_INT32_TO_FP32
    \brief Tensor Quant instruction: Convert all elements of A from 32-bit signed integer values to single-precision
    floating-point values.
*/
#define QUANT_INT32_TO_FP32 1

/*! \def QUANT_FP32_TO_INT32
    \brief Tensor Quant instruction: Convert all elements of A from single-precision floating-point values to 32-
    bit signed integer values.
*/
#define QUANT_FP32_TO_INT32 2

/*! \def QUANT_RELU
    \brief Tensor Quant instruction: Convert all negative INT32 values in A to 0
*/
#define QUANT_RELU 3

/*! \def QUANT_INT32_ADD_ROW
    \brief Tensor Quant instruction: Read the low-order COLS+1 32-bit signed integer values from an L1
    scratchpad line, and add this vector to every row of the 32-bit signed integer
    matrix A.
*/
#define QUANT_INT32_ADD_ROW 4

/*! \def QUANT_INT32_ADD_COL
    \brief Tensor Quant instruction: Read the low-order ROWS+1 32-bit signed integer values from an L1
    scratchpad line, and add this vector to every column of the 32-bit signed
    integer matrix A.
*/
#define QUANT_INT32_ADD_COL 5

/*! \def QUANT_FP32_MUL_ROW
    \brief Tensor Quant instruction: Read the low-order COLS+1 single-precision floating-point values from an
    L1 scratchpad line, and multiply the single-precision elements of each row
    of matrix A element-wise by this vector.
*/
#define QUANT_FP32_MUL_ROW 6

/*! \def QUANT_FP32_MUL_COL
    \brief Tensor Quant instruction: Read the low-order ROWS+1 single-precision floating-point values from an
    L1 scratchpad line, and multiply the single-precision elements of each col-
    umn of matrix A element-wise by this vector.
*/
#define QUANT_FP32_MUL_COL 7

/*! \def QUANT_SATINT8
    \brief Tensor Quant instruction: Clamp all 32-bit signed integer values in A to the range [-128, 127]. 
    The values are written in bits 7:0 of each element, with bits 31:8 set to zero.
*/
#define QUANT_SATINT8 8

/*! \def QUANT_SATUINT8
    \brief Tensor Quant instruction: Clamp all 32-bit signed integer values in A to the range [0, 255]. The values
    are written in bits 7:0 of each element, with bits 31:8 set to zero.
*/
#define QUANT_SATUINT8 9

/*! \def QUANT_PACK_128B
    \brief Tensor Quant instruction: Copy the low-order byte of the n-th 32-bit value in each row of A to the n-th
    byte of the row.
*/
#define QUANT_PACK_128B 10

/*! \def TENSOR_REDUCE_OP_FADD
    \brief Tensor Reduce instruction: The result is the addition of the incoming single-precision floating-point data
    and the single-precision floating-point values in the vector register file.
*/
#define TENSOR_REDUCE_OP_FADD 0

// #define TENSOR_REDUCE_OP_FSUB 1 -- Not supported

/*! \def TENSOR_REDUCE_OP_FMAX
    \brief Tensor Reduce instruction: The result is the maximum of the incoming single-precision floating-point data
and the single-precision floating-point values in the vector register file.
*/
#define TENSOR_REDUCE_OP_FMAX 2

/*! \def TENSOR_REDUCE_OP_FMIN
    \brief Tensor Reduce instruction: The result is the minimum of the incoming single-precision floating-point data
and the single-precision floating-point values in the vector register file..
*/
#define TENSOR_REDUCE_OP_FMIN 3

/*! \def TENSOR_REDUCE_OP_IADD
    \brief Tensor Reduce instruction: The result is the addition of the incoming 32-bit integer data and the 32-bit inte-
ger values in the vector register file.
*/
#define TENSOR_REDUCE_OP_IADD 4

// #define TENSOR_REDUCE_OP_ISUB 5 -- Not supported

/*! \def TENSOR_REDUCE_OP_IMAX
    \brief Tensor Reduce instruction: The result is the maximum of the incoming 32-bit signed integer data and the
32-bit signed integer values in the vector register file.
*/
#define TENSOR_REDUCE_OP_IMAX 6

/*! \def TENSOR_REDUCE_OP_IMIN
    \brief Tensor Reduce instruction: The result is the minimum of the incoming 32-bit signed integer data and the
32-bit signed integer values in the vector register file.
*/
#define TENSOR_REDUCE_OP_IMIN 7

/*! \def TENSOR_REDUCE_OP_FGET
    \brief Tensor Reduce instruction get function to be performed
*/
#define TENSOR_REDUCE_OP_FGET 8

/*! \def TENSOR_LOAD_WAIT_0
    \brief Tensor load to L1 Scratchpad with ID = 0 is complete.
*/
#define TENSOR_LOAD_WAIT_0 0

/*! \def TENSOR_LOAD_WAIT_1
    \brief Tensor load to L1 Scratchpad with ID = 1 is complete.
*/
#define TENSOR_LOAD_WAIT_1 1

/*! \def TENSOR_FMA_WAIT
    \brief All previous tensor matrix multiplication instructions are complete.
*/
#define TENSOR_FMA_WAIT 7

/*! \def TENSOR_STORE_WAIT
    \brief All previous tensor store instructions are complete.
*/
#define TENSOR_STORE_WAIT 8

/*! \def TENSOR_REDUCE_WAIT
    \brief All previous tensor reduction instructions are complete
*/
#define TENSOR_REDUCE_WAIT 9

/*! \def TENSOR_QUANT_WAIT
    \brief TensorQuant is complete
*/
#define TENSOR_QUANT_WAIT 10

/*! \def TENSOR_ERROR_LOAD_TRANSFORM
    \brief Define for tensor load transform error.
*/
#define TENSOR_ERROR_LOAD_TRANSFORM 1

/*! \def TENSOR_ERROR_FCC_OVERFLOW
    \brief Define for tensor fcc overflow error.
*/
#define TENSOR_ERROR_FCC_OVERFLOW 3

/*! \def TENSOR_ERROR_SCP_DISABLED
    \brief Define for tensor scp disabled error.
*/
#define TENSOR_ERROR_SCP_DISABLED 4

/*! \def TENSOR_ERROR_LOCKSW
    \brief Define for tensor locksw error.
*/
#define TENSOR_ERROR_LOCKSW 5

/*! \def TENSOR_ERROR_TL1_FMA
    \brief Define for L1 FMA error.
*/
#define TENSOR_ERROR_TL1_FMA 6

/*! \def TENSOR_ERROR_MEM_FAULT
    \brief Define for Memory fault error.
*/
#define TENSOR_ERROR_MEM_FAULT 7

/*! \def TENSOR_ERROR_STORE_COOP
    \brief Define for store coop error.
*/
#define TENSOR_ERROR_STORE_COOP 8

/*! \def TENSOR_ERROR_REDUCE
    \brief Define for tensor reduce error.
*/
#define TENSOR_ERROR_REDUCE 9

/*! \struct et_tensor_load_l2scp_conf
    \brief Tensor load from scp instruction configuration structure.
*/
typedef struct et_tensor_load_l2scp_conf {
    bool use_tmask;
    uint64_t dst_start;
    uint64_t addr;
    uint64_t num_lines;
    uint64_t stride;
    uint64_t id;
} et_tensor_load_l2scp_conf_t;

/*! \enum reduce_transform_t
    \brief enum transform mode for tensor reduce.
*/
typedef enum {
    FADD = 0x0ULL,
    FSUB = 0x1ULL,
    FMAX = 0x2ULL,
    FMIN = 0x3ULL,
    IADD = 0x4ULL,
    ISUB = 0x5ULL,
    IMAX = 0x6ULL,
    IMIN = 0x7ULL,
    FGET = 0x8ULL
} reduce_transform_t;

/*! \struct et_tensor_load_conf
    \brief Tensor load instruction configuration structure.
*/
typedef struct et_tensor_load_conf {
    bool use_tmask;
    bool use_coop;
    bool use_tenb;
    uint64_t dst_start;
    uint64_t transformation;
    uint64_t rd_l2scp;
    uint64_t addr;
    uint64_t offset;
    uint64_t num_lines;
    uint64_t stride;
    uint64_t id;
} et_tensor_load_conf_t;

/*! \fn inline void tensor_wait(long id)
    \brief Tensor wait instruction, Tensor Wait can be used to stall execution until
    a previously issued tensor instruction completes.
    \param id tensor ID
    \return none
    \tensorops Implementation of tensor_wait api
*/
inline __attribute__((always_inline)) void tensor_wait(long id)
{
    __asm__ __volatile__(" csrw 0x830, %[id]\n" : : [id] "r"(id) : "memory");
}

/*! \fn inline void tensor_load (tensor_load *conf)
    \brief Tensor load instruction, it loads data from memory (bypass-ing the L1 cache) 
    into the L1 scratchpad. Input parameter defines the configuration to tensor load.
    \param use_tmask the tensor_mask register is used for this operation
    \param use_coop the operation is a cooperative tensor load.
    \param dst_start L1 Scratchpad starting cache line
    \param transformation These bits, along with bit 52, decodes the type of tensor operation.
    \param use_tenb This bit, along with transformation, decodes the type of tensor operation.
    \param addr tensor load address
    \param offset tensor load address offset
    \param num_lines tensor load number of cache lines
    \param stride tensor load stride value
    \param id tensor load id  
    \return none
    \tensorops Implementation of tensor_load api
    
*/
// 1. Load Matrix A segment (1 row x 16 cols) into SCP ID 0
                // dst_start 0 refers to the first line of L1 Scratchpad
                // tensor_load(false, false, 0, 0, 0, 
                //             (uint64_t)(src0_data + m * K + kb), 0, 1, 0, 0);


inline void __attribute__((always_inline)) tensor_load(bool use_tmask, bool use_coop,
    uint64_t dst_start, uint64_t transformation, uint64_t use_tenb, uint64_t addr, uint64_t offset,
    uint64_t num_lines, uint64_t stride, uint64_t id)
{
    uint64_t csr_enc = (((uint64_t)use_tmask & 1) << 63) | (((uint64_t)use_coop & 1) << 62) |
                       ((transformation & 0x7) << 59) | ((dst_start & 0x3F) << 53) |
                       ((use_tenb & 0x1) << 52) | ((addr & 0xFFFFFFFFFFC0ULL)) |
                       ((offset & 0x3) << 4) | ((num_lines & 0xF));
                       
    uint64_t x31_enc = (stride & 0xFFFFFFFFFFC0ULL) | (id & 0x1);

   __asm__ __volatile__(
        "mv x31, %[x31v]\n"
        "csrw 0x83f, %[csrv]\n"
        :
        : [x31v] "r"(x31_enc),
          [csrv] "r"(csr_enc)
        : "x31", "memory"
    );
}

/*! \fn inline void et_tensor_load (et_tensor_load_conf_t *conf)
    \brief Tensor load instruction, it loads data from memory (bypass-ing the L1 cache) 
    into the L1 scratchpad. Input parameter defines the configuration to tensor load.
    \param conf tensor load configuration
    \return none
    \tensorops Implementation of et_tensor_load api
*/
inline void __attribute__((always_inline)) et_tensor_load(et_tensor_load_conf_t *conf)
{
    tensor_load(conf->use_tmask, conf->use_coop, conf->dst_start, conf->transformation,
        (uint64_t)conf->use_tenb, conf->addr, conf->offset, conf->num_lines, conf->stride,
        conf->id);
}

/*! \fn inline void tensor_load_setup_b(bool use_coop, uint64_t addr, uint64_t num_lines, uint64_t stride, uint64_t id)
    \brief Tensor load instruction setup
    \param use_coop the operation is a cooperative tensor load.
    \param addr tensor load address
    \param num_lines tensor load number of cache lines
    \param stride tensor load stride value
    \param id tensor load id  
    \return none
    \tensorops Implementation of tensor_load_setup_b api
*/
inline void __attribute__((always_inline))
tensor_load_setup_b(bool use_coop, uint64_t addr, uint64_t num_lines, uint64_t stride, uint64_t id)
{
    uint64_t csr_enc = (((uint64_t)use_coop & 1) << 62) | (0x1ULL << 52) |
                       ((addr & 0xFFFFFFFFFFC0ULL)) | ((num_lines & 0xF));
    uint64_t x31_enc = (stride & 0xFFFFFFFFFFC0ULL) | (id & 0x1);

    __asm__ __volatile__(
        "mv x31, %[x31v]\n"
        "csrw 0x83f, %[csrv]\n"
        :
        : [x31v] "r"(x31_enc),
          [csrv] "r"(csr_enc)
        : "x31", "memory"
    );
}

/*! \fn inline void et_tensor_load_l2scp (et_tensor_load_l2scp_conf_t *conf)
    \brief Tensor load l2scp loads data from memory (bypassing the L1 and L2 caches) into the L2 scratchpad.
   \param conf tensor load configuration
   \return none
   \tensorops Implementation of et_tensor_load_l2scp api
*/
inline void __attribute__((always_inline)) et_tensor_load_l2scp(et_tensor_load_l2scp_conf_t *conf)
{
    uint64_t csr_enc =
        (((((uint64_t)conf->use_tmask) & 1) << 63) | ((conf->dst_start & 0x1FFFCUL) << (48 - 2)) |
            ((conf->dst_start & 0x3UL) << 4) | ((conf->addr & 0xFFFFFFFFFFC0UL)) |
            ((conf->num_lines & 0x0FUL)));
    uint64_t x31_enc = (conf->stride & 0xFFFFFFFFFFC0ULL) | (conf->id & 0x1);

    __asm__ __volatile__(
        "mv x31, %[x31v]\n"
        "csrw 0x85f, %[csrv]\n"
        :
        : [x31v] "r"(x31_enc),
          [csrv] "r"(csr_enc)
        : "x31", "memory"
    );
}

/*! \fn inline void tensor_store_scp(uint64_t entry_stride,
                                     uint64_t start_scp_entry,
                                     uint64_t Arows,
                                     uint64_t addr,
                                     uint64_t stride)
   \brief Tensor Store writes a series of 64-byte blocks of data from the L1 scratchpad into memory.  
   A matrix X can have up to 16 rows, and each row can be up to 64B in size (the number of columns depends on the type of elements of X).
   \param entry_stride Register stride
   \param start_scp_entry Start register
   \param Arows A matrix row size
   \param addr Virtual Address 
   \param stride This value is the distance in bytes between consecutive tensor rows in memory
   \return none
   \tensorops Implementation of tensor_store_scp api
*/
inline void __attribute__((always_inline)) tensor_store_scp(
    uint64_t entry_stride, uint64_t start_scp_entry, uint64_t Arows, uint64_t addr, uint64_t stride)
{
    uint64_t csr_enc = ((entry_stride & 0x3) << 62) | ((start_scp_entry & 0x3F) << 56) |
                       ((addr & 0xFFFFFFFFFFC0ULL)) | ((Arows & 0xF) << 51) | (((uint64_t)1) << 48);
    uint64_t x31_enc = (stride & 0xFFFFFFFFFFC0UL);

    __asm__ __volatile__(
        "mv x31, %[x31v]\n"
        "csrw 0x87f, %[csrv]\n"
        :
        : [x31v] "r"(x31_enc),
          [csrv] "r"(csr_enc)
        : "x31", "memory"
    );
}

/*! \fn inline void tensor_store(uint64_t reg_stride,
                                 uint64_t start_reg,
                                 uint64_t cols,
                                 uint64_t Arows,
                                 uint64_t addr,
                                 uint64_t coop_store,
                                 uint64_t stride)
   \brief The Tensor store instruction reads a tensor from the vector register files and writes it to memory,
   bypassing the L1 data cache and the L2 cache. For the purposes of this instruction the tensor has ROWS+1 rows,
   and each row is 16*SIZE+16 bytes in size.
   \param reg_stride Register stride
   \param start_reg start register address
   \param cols  matrix row size.
   \param Arows  matrix row size
   \param addr Virtual Address 
   \param coop_store Number of minions to cooperate with
   \param stride This value is the distance in bytes between consecutive tensor rows in memory
   \return none
   \tensorops Implementation of tensor_store api
*/
inline void __attribute__((always_inline)) tensor_store(uint64_t reg_stride, uint64_t start_reg,
    uint64_t cols, uint64_t Arows, uint64_t addr, uint64_t coop_store, uint64_t stride)
{
    uint64_t warl = 0;
    uint64_t csr_enc = ((reg_stride & 0x3) << 62) | ((start_reg & 0x1F) << 57) |
                       ((cols & 0x3) << 55) | ((addr & 0xFFFFFFFFFFF0)) | ((Arows & 0xF) << 51) |
                       ((coop_store & 0x3) << 49) | ((warl & 0xF));

    uint64_t x31_enc = (stride & 0xFFFFFFFFFF0UL);

    __asm__ __volatile__(
        "mv x31, %[x31v]\n"
        "csrw 0x87f, %[csrv]\n"
        :
        : [x31v] "r"(x31_enc),
          [csrv] "r"(csr_enc)
        : "x31", "memory"
    );
}

/*! \fn inline void tensor_fma(bool use_tmask,
                               uint64_t b_num_col,
                               uint64_t a_num_rows,
                               uint64_t a_num_cols, 
                               uint64_t offset, 
                               bool tenc_loc, 
                               bool tenb_unsigned, 
                               bool tena_unsigned, 
                               bool tenb_loc, 
                               uint64_t scp_loc_b, 
                               uint64_t scp_loc_a, 
                               uint64_t opcode, 
                               bool first_pass)
   \brief The Tensor FMA instruction multiplies two matrices A and B, optionally adds the resulting matrix
   to a third matrix C, and writes the result back onto matrix C
   \param use_tmask Use tensor_mask CSR to skip operations in an A row granularity.
   \param b_num_col B matrix number of columns
   \param a_num_rows A matrix number of rows
   \param a_num_cols A matrix number of columns
   \param offset A matrix starting column for the operation.
   \param tenc_loc Location of matrix C (0 = L1 scratchpad, 1 = memory).   
   \param tenb_unsigned TenB is signed (0) or unsigned (1).
   \param tena_unsigned TenA is signed (0) or unsigned (1).
   \param tenb_loc Location of matrix B (0 = L1 scratchpad, 1 = memory).
   \param scp_loc_b Starting L1 scratchpad cache line where matrix B is stored, ignored when xs[20] = 1.
   \param scp_loc_a Starting L1 scratchpad cache line where matrix A is stored, ignored when xs[20] = 1.
   \param opcode TensorType = 011
   \param first_pass if set to 0 then the initial value of TenC is added to the result
   \return none
   \tensorops Implementation of tensor_fma api
*/
inline void __attribute__((always_inline))
tensor_fma(bool use_tmask, uint64_t b_num_col, uint64_t a_num_rows, uint64_t a_num_cols,
    uint64_t offset, bool tenc_loc, bool tenb_unsigned, bool tena_unsigned, bool tenb_loc,
    uint64_t scp_loc_b, uint64_t scp_loc_a, uint64_t opcode, bool first_pass)
{
    uint64_t csr_enc =
        (((uint64_t)use_tmask & 1) << 63) | ((b_num_col & 0x3) << 55) | ((a_num_rows & 0xF) << 51) |
        ((a_num_cols & 0xF) << 47) | ((offset & 0xF) << 43) | (((uint64_t)tenc_loc & 1) << 23) |
        (((uint64_t)tena_unsigned & 1) << 22) | (((uint64_t)tenb_unsigned & 1) << 21) |
        (((uint64_t)tenb_loc & 1) << 20) | ((scp_loc_b & 0xFF) << 12) | ((scp_loc_a & 0xFF) << 4) |
        ((opcode & 0x7) << 1) | ((uint64_t)first_pass & 1);

    __asm__ __volatile__("csrw 0x801, %[csr_enc]\n" : : [csr_enc] "r"(csr_enc) :);
}

/*! \fn inline uint32_t tensor_reduce_uint32(uint32_t value, uint64_t operation, uint64_t partnerID, uint64_t action)
   \brief Tensor reduce allows a group of harts to communicate values held in floating-point registers to collectively calculate a reduction
   function. 
   \param value Register stride
   \param operation Function to be performed.
   \param partnerID Receiver minionID.
   \param action action value
   \return uint32_t value after reduction
   \tensorops Implementation of tensor_reduce_uint32 api
*/
inline uint32_t __attribute__((always_inline))
tensor_reduce_uint32(uint32_t value, uint64_t operation, uint64_t partnerID, uint64_t action)
{
    uint64_t warl = 0;
    uint32_t out;
    uint64_t csr_enc = ((warl & 0x2) << 62) | ((0ULL & 0x1F) << 57) | ((warl & 0x1FFFFFFF) << 28) |
                       ((operation & 0xF) << 24) | ((1ULL & 0xFF) << 16) |
                       ((partnerID & 0x1FFF) << 3) | ((warl & 0x1) << 2) | ((action & 0x3));

    __asm__ __volatile__("fmv.s.x     f0, %[value]\n"
                         "csrw 0x800, %[csr_enc]\n"
                         "fmv.x.s     %[out], f0\n"
                         : [out] "=r"(out)
                         : [csr_enc] "r"(csr_enc), [value] "r"(value)
                         : "f0");

    return out;
}

/*! \fn inline float tensor_reduce_float(float freg, uint64_t operation, uint64_t num_reg, uint64_t partnerID, uint64_t action) {
   \brief TensorReduce allows a group of harts to communicate values held in floating-point registers to collectively calculate a reduction
   function. 
   \param freg Freg register stride
   \param operation Function to be performed.
   \param num_reg number of registers to use
   \param partnerID Receiver minionID.
   \param action action value
   \return float value after reduction
   \tensorops Implementation of tensor_reduce_float api
*/
inline float __attribute__((always_inline)) tensor_reduce_float(
    float freg, uint64_t operation, uint64_t num_reg, uint64_t partnerID, uint64_t action)
{
    uint64_t warl = 0;
    float out;
    uint64_t csr_enc = ((warl & 0x2) << 62) | ((0ULL & 0x1F) << 57) | ((warl & 0x1FFFFFFF) << 28) |
                       ((operation & 0xF) << 24) | ((num_reg & 0xFF) << 16) |
                       ((partnerID & 0x1FFF) << 3) | ((warl & 0x1) << 2) | ((action & 0x3));

    __asm__ __volatile__("fmv.s   f0, %[freg]\n"
                         "csrw 0x800, %[csr_enc]\n"
                         "fmv.s   %[out], f0\n"
                         : [out] "=f"(out)
                         : [csr_enc] "r"(csr_enc), [freg] "f"(freg)
                         : "f0");

    return out;
}

//#define tensor_reduce_float1(fval, operation, partnerID, action) do {
//   uint64_t warl = 0;
//   float out;
//   uint64_t csr_enc = ((warl      & 0x2        ) << 62) |
//                      ((0         & 0x1F       ) << 57) |
//                      ((warl      & 0x1FFFFFFF ) << 28) |
//                      ((operation & 0xF        ) << 24) |
//                      ((1         & 0xFF       ) << 16) |
//                      ((partnerID & 0x1FFF     ) << 3 ) |
//                      ((warl      & 0x1        ) << 2 ) |
//                      ((action    & 0x3        )      );
//
//   register float asm("f0") fval;
//   __asm__ volatile (
//         "csrw 0x800, %[csr_enc]"
//         : "+r" (ftmp)
//         : [csr_enc] "r" (csr_enc)
//   );
//} while (0)
//
//
//inline float __attribute__((always_inline)) tensor_reduce_float(uint64_t fstart, uint64_t operation, uint64_t num_reg, uint64_t partnerID, uint64_t action) {
//   uint64_t warl = 0;
//   float out;
//   uint64_t csr_enc = ((warl      & 0x2        ) << 62) |
//                      ((fstart    & 0x1F       ) << 57) |
//                      ((warl      & 0x1FFFFFFF ) << 28) |
//                      ((operation & 0xF        ) << 24) |
//                      ((num_reg   & 0xFF       ) << 16) |
//                      ((partnerID & 0x1FFF     ) << 3 ) |
//                      ((warl      & 0x1        ) << 2 ) |
//                      ((action    & 0x3        )      );
//
//   __asm__ volatile (
//         "csrw 0x800, %[csr_enc]\n"
//         : /*empty*/
//	 : [csr_enc] "r" (csr_enc),
//         : /*"f0", "f1", "f2", "f3", "f4",
//           "f5", "f6", "f7", "f8", "f9",
//           "f10", "f11", "f12", "f13", "f14",
//           "f15", "f16", "f17", "f18", "f19",
//           "f20", "f21", "f22", "f23", "f24",
//           "f25", "f26", "f27", "f28", "f29",
//           "f30", "f31"*/
//   );
//
//   return out;
//}

/*! \fn inline void tensor_reduce(uint64_t start_reg, uint64_t operation, uint64_t num_reg, uint64_t partnerID, uint64_t action)
   \brief The TensorReduce instruction allows up to 216 harts to collectively calculate a reduction function. 
   \param start_reg starting register 
   \param operation Function to be performed.
   \param num_reg number of registers
   \param partnerID Receiver minionID.
   \param action action value
   \return uint32_t value after reduction
   \tensorops Implementation of tensor_reduce api
*/

inline void __attribute__((always_inline)) tensor_reduce(
    uint64_t start_reg, uint64_t operation, uint64_t num_reg, uint64_t partnerID, uint64_t action)
{
    uint64_t warl = 0;

    uint64_t csr_enc = ((warl & 0x2) << 62) | ((start_reg & 0x1F) << 57) |
                       ((warl & 0x1FFFFFFF) << 28) | ((operation & 0xF) << 24) |
                       ((num_reg & 0xFF) << 16) | ((partnerID & 0x1FFF) << 3) |
                       ((warl & 0x1) << 2) | ((action & 0x3));

    __asm__ __volatile__("csrw 0x800, %[csr_enc]\n" : : [csr_enc] "r"(csr_enc) :);
}

/*! \fn inline void tensor_reduce_send(uint64_t start_reg, uint64_t num_reg, uint64_t partnerID)
   \brief This function applies reduce instruction to function and then sends to partner minion.
   \param start_reg starting register 
   \param num_reg number of registers
   \param partnerID Receiver minionID.
   \return none
   \tensorops Implementation of tensor_reduce_send api
*/
inline void __attribute__((always_inline))
tensor_reduce_send(uint64_t start_reg, uint64_t num_reg, uint64_t partnerID)
{
    uint64_t warl = 0;
    tensor_reduce(start_reg, warl, num_reg, partnerID, 0);
}

/*! \fn inline void tensor_reduce_recv(uint64_t start_reg, uint64_t operation, uint64_t num_reg, uint64_t partnerID)
   \brief This function recieves reduce function from partner minion.
   \param start_reg starting register 
   \param operation operation to be performed
   \param num_reg number of registers
   \param partnerID Receiver minionID.
   \return none
   \tensorops Implementation of tensor_reduce_recv api
*/
inline void __attribute__((always_inline))
tensor_reduce_recv(uint64_t start_reg, uint64_t operation, uint64_t num_reg, uint64_t partnerID)
{
    tensor_reduce(start_reg, operation, num_reg, partnerID, 1);
}

/*! \fn inline void tensor_reduce_auto(uint64_t start_reg, uint64_t operation, uint64_t num_reg, uint64_t tree_depth)
   \brief The Tensor reduce instruction allows up to 216 harts to collectively calculate a reduction function.
   \param start_reg starting register 
   \param operation operation to be performed
   \param num_reg number of registers
   \param tree_depth tree depth
   \return none
   \tensorops Implementation of tensor_reduce_auto api
*/
inline void __attribute__((always_inline))
tensor_reduce_auto(uint64_t start_reg, uint64_t operation, uint64_t num_reg, uint64_t tree_depth)
{
    tensor_reduce(start_reg, operation, num_reg, (0ULL << 4) | (tree_depth & 0xF), 3);
}

/*! \fn inline void tensor_broadcast(uint64_t start_reg, uint64_t operation, uint64_t num_reg, uint64_t tree_depth) {
   \brief The Tensor broadcast instruction allows up to 216 harts to receive values held in the vector registers
   of one of the harts in the group. The broadcast operation is performed in a binary-tree fashion, where the source
   data is originally in the root node and the final result ends up in the leaf nodes.
   \param start_reg Starting floating-point register
   \param operation operation to be performed
   \param num_reg Number of floating-point registers
   \param tree_depth tree depth
   \return none
   \tensorops Implementation of tensor_broadcast api
*/
inline void __attribute__((always_inline))
tensor_broadcast(uint64_t start_reg, uint64_t operation, uint64_t num_reg, uint64_t tree_depth)
{
    tensor_reduce(start_reg, operation, num_reg, (0ULL << 4) | (tree_depth & 0xF), 2);
}

/*! \fn inline void tensor_reduce_autopair(uint64_t start_reg, uint64_t operation, uint64_t num_reg, uint64_t start_lvl, uint64_t end_lvl, uint64_t action) {
   \brief This function is wrapper of Tensor Reduce (auto-pair variant) instruction.
   \param start_reg Starting floating-point register
   \param operation Function to be performed
   \param num_reg Number of floating-point registers
   \param start_lvl starting level value
   \param end_lvl ending level value
   \param action action value
   \return none
   \tensorops Implementation of tensor_reduce_autopair api

*/
inline void __attribute__((always_inline)) tensor_reduce_autopair(uint64_t start_reg,
    uint64_t operation, uint64_t num_reg, uint64_t start_lvl, uint64_t end_lvl, uint64_t action)
{
    uint64_t partnerID;
    // PRM-10 defines the partnerID field for Tensor Reduce (auto-pair variant) as following:
    // [15:11] WARL(0)
    // [10: 7] End level for autopair
    // [ 6: 3] Start level for autopair
    uint64_t warl = 0;
    partnerID = ((warl & 0xF) << 11) | ((end_lvl & 0xF) << 7) | ((start_lvl & 0xF) << 3);
    // Operations encoding:
    // 0000=fadd, 0001=fsub, 0010=fmax, 0011=fmin, 0100=iadd, 0101=isub, 0110=imax, 0111=imin, 1000=fget
    //
    // Action encoding:
    // 00=send, 01=receive, 10=auto-pair broadcast derive from hartid,11=auto-pair reduce derive from hartid
    tensor_reduce(start_reg, operation, num_reg, (partnerID >> 3), action);
}

/*! \fn inline void tensor_quant(uint64_t start_reg, uint64_t col, uint64_t row, uint64_t scp_loc, uint64_t transf9, uint64_t transf8, uint64_t transf7, uint64_t transf6, uint64_t transf5, uint64_t transf4, uint64_t transf3, uint64_t transf2, uint64_t transf1, uint64_t transf0 )
   \brief Tensor quantization (TensorQuant) instructions are encoded as writes to the tensor_quant CSR. The TensorQuant
   instruction performs a sequence of up to 10 transformations to a matrix A
   \param start_reg Starting register
   \param col A matrix number of columns.
   \param row A matrix number of rows.
   \param scp_loc L1 scratchpad cache line where the first vector is stored.
   \param transf9 Transformation 9.
   \param transf8 Transformation 8.
   \param transf7 Transformation 7.
   \param transf6 Transformation 6.
   \param transf5 Transformation 5.
   \param transf4 Transformation 4.
   \param transf3 Transformation 3.
   \param transf2 Transformation 2.
   \param transf1 Transformation 1.
   \param transf0 Transformation 0.
   \return none
   \tensorops Implementation of tensor_quant api
*/
inline void __attribute__((always_inline))
tensor_quant(uint64_t start_reg, uint64_t col, uint64_t row, uint64_t scp_loc, uint64_t transf9,
    uint64_t transf8, uint64_t transf7, uint64_t transf6, uint64_t transf5, uint64_t transf4,
    uint64_t transf3, uint64_t transf2, uint64_t transf1, uint64_t transf0)
{
    uint64_t csr_enc = ((start_reg & 0x1F) << 57) | ((col & 0x3) << 55) | ((row & 0xF) << 51) |
                       ((scp_loc & 0x3F) << 45) | ((transf9 & 0xF) << 36) |
                       ((transf8 & 0xF) << 32) | ((transf7 & 0xF) << 28) | ((transf6 & 0xF) << 24) |
                       ((transf5 & 0xF) << 20) | ((transf4 & 0xF) << 16) | ((transf3 & 0xF) << 12) |
                       ((transf2 & 0xF) << 8) | ((transf1 & 0xF) << 4) | ((transf0 & 0xF) << 0);

    __asm__ __volatile__("csrw 0x806, %[csr_enc]\n" : : [csr_enc] "r"(csr_enc) :);
}

/*! \fn inline void tensor_mask(uint64_t zeros, uint64_t mask_bits)
   \brief The TensorLoad, TensorFMA, and CacheOp instructions can operate under the
   control of the tensor_mask CSR. The tensor_mask CSR contains one bit for each
   of the destination lines that TensorLoad can potentially write into the scratchpad
   \param zeros all zeros
   \param mask_bits tensor bit mask
   \return none
   \tensorops Implementation of tensor_mask api
*/
inline void __attribute__((always_inline)) tensor_mask(uint64_t zeros, uint64_t mask_bits)
{
    uint64_t csr_enc = ((zeros & 0x000000000000) << 16) | (mask_bits & 0xFFFF);

    __asm__ __volatile__("csrw 0x805, %[csr_enc]\n" : : [csr_enc] "r"(csr_enc) :);
}

/*! \fn inline void tensor_coop(uint64_t val)
   \brief The tensor_coop instruction specifies which harts participate in cooperative tensor load operations. Only the first hart of each
   selected Minion core participates in the cooperative operations, since the second hart cannot issue tensor load operations.
   \param val value contains encoded coop id, minion and neigh mask
   \return none
   \tensorops Implementation of tensor_coop api
*/
inline void __attribute__((always_inline)) tensor_coop(uint64_t val)
{
    __asm__ __volatile__("csrw 0x804, %[val]\n" : : [val] "r"(val) :);
}

/*! \fn inline void convolution_ctrl(uint64_t row_start, uint64_t col_start)
   \brief This function modifies the convolution control register.
   This register encodes the location of a tensor inside a larger two-dimensional array.
   \param row_start signed integer value specifying the row inside the array where the first row of the tensor resides
   \param col_start signed integer value specifying the column inside the array where the first column of the tensor resides
   \return none
   \tensorops Implementation of convolution_ctrl api
*/
inline void __attribute__((always_inline)) convolution_ctrl(uint64_t row_start, uint64_t col_start)
{
    uint64_t csr_enc = ((row_start & 0xFFFF) << 32) | (col_start & 0xFFFF);

    __asm__ __volatile__("csrw 0x803, %[csr_enc]\n" : : [csr_enc] "r"(csr_enc) :);
}

/*! \fn inline void convolution_size(uint64_t srow, uint64_t nrow, uint64_t scol, uint64_t ncol)
   \brief This function modifies the convolution size register.
   This register specifies the layout of a two-dimensional array used for convolutions.
   \param srow integer value specifying the row inside the array where the first row of the tensor resides
   \param nrow integer values specifying the number of rows of the array
   \param scol integer value specifying the distance, in number of columns, between consecutive column accesses to the array during
    convolution operations
   \param ncol integer values specifying the number of columns of the array
   \return none
   \tensorops Implementation of convolution_size api
*/
inline void __attribute__((always_inline))
convolution_size(uint64_t srow, uint64_t nrow, uint64_t scol, uint64_t ncol)
{
    uint64_t csr_enc = ((srow & 0xFF) << 56) | ((nrow & 0xFFFF) << 32) | ((scol & 0xFF) << 24) |
                       ((ncol & 0xFFFF));

    __asm__ __volatile__("csrw 0x802, %[csr_enc]\n" : : [csr_enc] "r"(csr_enc) :);
}

/*! \fn inline unsigned get_tensor_error()
   \brief This function returns tensor error register value.
   The tensor_error register accrues errors that occur during the execution of tensor instructions and cache management operations. When the tensor coprocessor or the cache management coprocessor generates an exception, the exception is recorded in
   the tensor_error register and execution does not trap. The tensor_error register is never cleared by the implementation. It is the
   responsibility of the software to clear tensor_error
   \return Tensor error value
   \tensorops Implementation of get_tensor_error api
*/
inline unsigned long __attribute__((always_inline)) get_tensor_error()
{
    unsigned long error;

    __asm__ __volatile__("csrr %0, 0x808" : "=r"(error));

    return error;
}

/*! \fn inline uint64_t get_tensor_mask()
   \brief This function returns tensor mask register value.
   \return Tensor mask value
   \tensorops Implementation of get_tensor_mask api
*/
inline uint64_t __attribute__((always_inline)) get_tensor_mask()
{
    uint64_t val;

    __asm__ __volatile__("csrr %0, 0x805" : "=r"(val));

    return val;
}

#define mask_set(msk, val)                                          \
    do                                                              \
    {                                                               \
        __asm__ volatile("mov.m.x m" #msk ", zero, %0" ::"n"(val)); \
    } while (0)

#define flw_ps(fd, ptr)                                       \
    do                                                        \
    {                                                         \
        __asm__ volatile("flw.ps f" #fd ", (%0)" ::"r"(ptr)); \
    } while (0)

#define fsw_ps(fd, ptr)                                                  \
    do                                                                   \
    {                                                                    \
        __asm__ volatile("fsw.ps f" #fd ", (%0)" ::"r"(ptr) : "memory"); \
    } while (0)

#ifdef __cplusplus
}
#endif

#endif // ! __TENSORS_H
