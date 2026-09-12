/*
 *  Copyright (C) 2019-2022, Xilinx Inc
 *
 *  This file is dual licensed.  It may be redistributed and/or modified
 *  under the terms of the Apache 2.0 License OR version 2 of the GNU
 *  General Public License.
 *
 *  Apache License Verbiage
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  GPL license Verbiage:
 *
 *  This program is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License as
 *  published by the Free Software Foundation; either version 2 of the
 *  License, or (at your option) any later version.  This program is
 *  distributed in the hope that it will be useful, but WITHOUT ANY
 *  WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
 *  License for more details.  You should have received a copy of the
 *  GNU General Public License along with this program; if not, write
 *  to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
 *  Boston, MA 02111-1307 USA
 *
 */

/**
 * DOC: XRT Embedded Runtime definition
 *
 * Header file *ert.h* defines data structures used by Emebdded Runtime (ERT) and
 * XRT xclExecBuf() API.
 */

#ifndef INCLUDE_XRT_DETAIL_ERT_H_
#define INCLUDE_XRT_DETAIL_ERT_H_

#if defined(__linux__) && defined(__KERNEL__)
# include <linux/types.h>
#elif defined(__windows__) && defined(_KERNEL_MODE)
# include <stdlib.h>
#elif defined(__cplusplus) && !defined(_KERNEL_MODE)
# include <cstdint>
# include <cstdio>
#else
# include <stdbool.h>
# include <stdint.h>
# include <stdio.h>
#endif

#ifdef _WIN32
# pragma warning( push )
# pragma warning( disable : 4200 4201 )
#endif

#if defined(__GNUC__)
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wpedantic"
#endif

#define to_cfg_pkg(pkg) \
    ((struct ert_configure_cmd *)(pkg))
#define to_start_krnl_pkg(pkg) \
    ((struct ert_start_kernel_cmd *)(pkg))
#define to_copybo_pkg(pkg) \
    ((struct ert_start_copybo_cmd *)(pkg))
#define to_cfg_sk_pkg(pkg) \
    ((struct ert_configure_sk_cmd *)(pkg))
#define to_init_krnl_pkg(pkg) \
    ((struct ert_init_kernel_cmd *)(pkg))
#define to_validate_pkg(pkg) \
    ((struct ert_validate_cmd *)(pkg))
#define to_abort_pkg(pkg) \
    ((struct ert_abort_cmd *)(pkg))


#define HOST_RW_PATTERN     0xF0F0F0F0
#define DEVICE_RW_PATTERN   0x0F0F0F0F

/**
 * struct ert_packet: ERT generic packet format
 *
 * @state:   [3-0] current state of a command
 * @custom:  [11-4] custom per specific commands
 * @count:   [22-12] number of words in payload (data)
 * @opcode:  [27-23] opcode identifying specific command
 * @type:    [31-28] type of command (currently 0)
 * @data:    count number of words representing packet payload
 */
struct ert_packet {
  union {
    struct {
      uint32_t state:4;   /* [3-0]   */
      uint32_t custom:8;  /* [11-4]  */
      uint32_t count:11;  /* [22-12] */
      uint32_t opcode:5;  /* [27-23] */
      uint32_t type:4;    /* [31-28] */
    };
    uint32_t header;
  };
#if defined(__linux__) && defined(__KERNEL__)
  uint32_t data[];   /* count number of words */
#else
  uint32_t data[1];   /* count number of words */
#endif
};

/**
 * struct ert_start_kernel_cmd: ERT start kernel command format
 *
 * @state:           [3-0]   current state of a command
 * @stat_enabled:    [4]     enabled driver to record timestamp for various
 *                           states cmd has gone through. The stat data
 *                           is appended after cmd data.
 * @extra_cu_masks:  [11-10] extra CU masks in addition to mandatory mask
 * @count:           [22-12] number of words following header for cmd data. Not
 *                           include stat data.
 * @opcode:          [27-23] 0, opcode for start_kernel
 * @type:            [31-27] 0, type of start_kernel
 *
 * @cu_mask:         first mandatory CU mask
 * @data:            count-1 number of words representing interpreted payload
 *
 * The packet payload is comprised of reserved id field, a mandatory CU mask,
 * and extra_cu_masks per header field, followed by a CU register map of size
 * (count - (1 + extra_cu_masks)) uint32_t words.
 */
struct ert_start_kernel_cmd {
  union {
    struct {
      uint32_t state:4;          /* [3-0]   */
      uint32_t stat_enabled:1;   /* [4]     */
      uint32_t unused:5;         /* [9-5]   */
      uint32_t extra_cu_masks:2; /* [11-10] */
      uint32_t count:11;         /* [22-12] */
      uint32_t opcode:5;         /* [27-23] */
      uint32_t type:4;           /* [31-27] */
    };
    uint32_t header;
  };
  /* payload */
  uint32_t cu_mask;              /* mandatory cu mask */
  #if defined(__linux__) && defined(__KERNEL__)
  uint32_t data[];   /* flexible array member*/
#else
  uint32_t data[1];   /* count -1 number of words */
#endif
};

/**
 * struct ert_dpu_data - interpretation of data payload for ERT_START_DPU
 *
 * @instruction_buffer:       address of instruction buffer
 * @instruction_buffer_size:  size of instruction buffer in bytes
 * @uc_index:                 microblaze controller index
 * @chained:                  number of following ert_dpu_data elements
 *
 * The ert_dpu_data is prepended to data payload of ert_start_kernel_cmd
 * after any extra cu masks.  The payload count of the ert packet is
 * incremented with the size (words) of ert_dpu_data elements
 * preprended to the data payload.
 *
 * The data payload for ERT_START_DPU is interpreted as fixed instruction
 * buffer address along with instruction count, followed by regular kernel
 * arguments.
 */
struct ert_dpu_data {
  uint64_t instruction_buffer;       /* buffer address 2 words */
  uint32_t instruction_buffer_size;  /* size of buffer in bytes */
  uint16_t uc_index;                 /* microblaze controller index */
  uint16_t chained;                  /* number of following ert_dpu_data elements */
};

/**
 * struct ert_npu_data - interpretation of data payload for ERT_START_NPU
 *
 * @instruction_buffer:       address of instruction buffer
 * @instruction_buffer_size:  size of instruction buffer in bytes
 * @instruction_prop_count:   WORD length of property name value pairs
 *
 * The ert_npu_data is prepended to data payload of ert_start_kernel_cmd
 * after any extra cu masks.  The payload count of the ert packet is
 * incremented with the size (words) of ert_npu_data elements
 * preprended to the data payload.
 *
 * The data payload for ERT_START_NPU is interpreted as instruction
 * buffer address, instruction count along with instruction property,
 * followed by regular kernel arguments.
 *
 * When instruction_prop_count is non-zero, it indicates the length
 * (in 32 bits WORD) of the instruction buffer properties after this
 * fields. This count is reserved for future extension. One example
 * propertiy is the number of actual columns this instruction used.
 */
struct ert_npu_data {
  uint64_t instruction_buffer;       /* buffer address 2 words */
  uint32_t instruction_buffer_size;  /* size of buffer in bytes */
  uint32_t instruction_prop_count;   /* WORD length of following properties nv pairs */
};

/**
 * struct ert_npu_preempt_data - interpretation of data payload for ERT_START_NPU_PREEMPT
 *
 * @instruction_buffer:       address of instruction buffer
 * @save_buffer:              address of save instruction buffer
 * @restore_buffer:           address of restrore instruction buffer
 * @instruction_buffer_size:  size of instruction buffer in bytes
 * @save_buffer_size:         size of save instruction buffer in bytes
 * @restore_buffer_size:      size of restore instruction buffer in bytes
 * @instruction_prop_count:   number of property name value pairs
 *
 * The ert_npu_preempt_data is prepended to data payload of ert_start_kernel_cmd
 * after any extra cu masks.  The payload count of the ert packet is
 * incremented with the size (words) of ert_npu_preempt_data elements
 * preprended to the data payload.
 *
 * The data payload for ERT_START_NPU_PREEMPT is interpreted as instruction
 * buffer, save instruction buffer, restore instruction buffer and their
 * size, along with instruction property, followed by regular kernel arguments.
 *
 * When instruction_prop_count is non-zero, it indicates the length
 * (in 32 bits WORD) of the instruction buffer properties after this
 * fields. This count is reserved for future extension. One example
 * propertiy is the number of actual columns this instruction used.
 */
struct ert_npu_preempt_data {
  uint64_t instruction_buffer;       /* buffer address 2 words */
  uint64_t save_buffer;              /* buffer address 2 words */
  uint64_t restore_buffer;           /* buffer address 2 words */
  uint32_t instruction_buffer_size;  /* size of buffer in bytes */
  uint32_t save_buffer_size;         /* size of buffer in bytes */
  uint32_t restore_buffer_size;      /* size of buffer in bytes */
  uint32_t instruction_prop_count;   /* DWORD length of following properties nv pairs */
};

/**
 * struct ert_cmd_chain_data - interpretation of data payload for ERT_CMD_CHAIN
 *
 * @command_count: number of commands in chain
 * @submit_index:  index of last successfully submitted command in chain
 * @error_index:   index of failing command if cmd status is not completed
 * @data[]:        address of each command in chain
 *
 * This is the payload of an *ert_packet* when the opcode is ERT_CMD_CHAIN
 */
struct ert_cmd_chain_data {
  uint32_t command_count;
  uint32_t submit_index;
  uint32_t error_index;
  uint32_t reserved[3];
  uint64_t data[];
};

#ifndef U30_DEBUG
#define ert_write_return_code(cmd, value) \
do { \
  struct ert_start_kernel_cmd *skcmd = (struct ert_start_kernel_cmd *)cmd; \
  int end_idx = skcmd->count - 1 - skcmd->extra_cu_masks; \
  skcmd->data[end_idx] = value; \
} while (0)

#define ert_read_return_code(cmd, ret) \
do { \
  struct ert_start_kernel_cmd *skcmd = (struct ert_start_kernel_cmd *)cmd; \
  int end_idx = skcmd->count - 1 - skcmd->extra_cu_masks; \
  ret = skcmd->data[end_idx]; \
} while (0)
#else
/* These are for debug legacy U30 firmware */
#define ert_write_return_code(cmd, value) \
do { \
  struct ert_start_kernel_cmd *skcmd = (struct ert_start_kernel_cmd *)cmd; \
  skcmd->cu_mask = value; \
} while (0)

#define ert_read_return_code(cmd, ret) \
do { \
  struct ert_start_kernel_cmd *skcmd = (struct ert_start_kernel_cmd *)cmd; \
  ret = skcmd->cu_mask; \
} while (0)
#endif

/**
 * struct ert_init_kernel_cmd: ERT initialize kernel command format
 * this command initializes CUs by writing CU registers. CUs are
 * represented by cu_mask and extra_cu_masks.
 *
 * @state:           [3-0]   current state of a command
 * @update_rtp:      [4]     command is for runtime update of cu argument
 * @extra_cu_masks:  [11-10] extra CU masks in addition to mandatory mask
 * @count:           [22-12] number of words following header
 * @opcode:          [27-23] 0, opcode for init_kernel
 * @type:            [31-27] 0, type of init_kernel
 *
 * @cu_run_timeout   the configured CU timeout value in Microseconds
 *                   setting to 0 means CU should not timeout
 * @cu_reset_timeout the configured CU reset timeout value in Microseconds
 *                   when CU timeout, CU will be reset. this indicates
 *                   CU reset should be completed within the timeout value.
 *                   if cu_run_timeout is set to 0, this field is undefined.
 *
 * @cu_mask:         first mandatory CU mask
 * @data:            count-9 number of words representing interpreted payload
 *
 * The packet payload is comprised of reserved id field, 8 reserved fields,
 * a mandatory CU mask, and extra_cu_masks per header field, followed by a
 * CU register map of size (count - (9 + extra_cu_masks)) uint32_t words.
 */
struct ert_init_kernel_cmd {
  union {
    struct {
      uint32_t state:4;          /* [3-0]   */
      uint32_t update_rtp:1;     /* [4]  */
      uint32_t unused:5;         /* [9-5]  */
      uint32_t extra_cu_masks:2; /* [11-10]  */
      uint32_t count:11;         /* [22-12] */
      uint32_t opcode:5;         /* [27-23] */
      uint32_t type:4;           /* [31-27] */
    };
    uint32_t header;
  };

  uint32_t cu_run_timeout;   /* CU timeout value in Microseconds */
  uint32_t cu_reset_timeout; /* CU reset timeout value in Microseconds */
  uint32_t reserved[6];      /* reserved for future use */

  /* payload */
  uint32_t cu_mask;          /* mandatory cu mask */
 #if defined(__linux__) && defined(__KERNEL__)
  uint32_t data[];   /* Flexible array member */
#else
  uint32_t data[1];          /* count-9 number of words */
#endif
};

#define KDMA_BLOCK_SIZE 64   /* Limited by KDMA CU */
struct ert_start_copybo_cmd {
  uint32_t state:4;          /* [3-0], must be ERT_CMD_STATE_NEW */
  uint32_t unused:6;         /* [9-4] */
  uint32_t extra_cu_masks:2; /* [11-10], = 3 */
  uint32_t count:11;         /* [22-12], = 16, exclude 'arg' */
  uint32_t opcode:5;         /* [27-23], = ERT_START_COPYBO */
  uint32_t type:4;           /* [31-27], = ERT_DEFAULT */
  uint32_t cu_mask[4];       /* mandatory cu masks */
  uint32_t reserved[4];      /* for scheduler use */
  uint32_t src_addr_lo;      /* low 32 bit of src addr */
  uint32_t src_addr_hi;      /* high 32 bit of src addr */
  uint32_t src_bo_hdl;       /* src bo handle, cleared by driver */
  uint32_t dst_addr_lo;      /* low 32 bit of dst addr */
  uint32_t dst_addr_hi;      /* high 32 bit of dst addr */
  uint32_t dst_bo_hdl;       /* dst bo handle, cleared by driver */
  uint32_t size;             /* size in bytes low 32 bit*/
  uint32_t size_hi;          /* size in bytes high 32 bit*/
  void     *arg;             /* pointer to aux data for KDS */
};

/**
 * struct ert_configure_cmd: ERT configure command format
 *
 * @state:           [3-0] current state of a command
 * @count:           [22-12] number of words in payload (5 + num_cus)
 * @opcode:          [27-23] 1, opcode for configure
 * @type:            [31-27] 0, type of configure
 *
 * @slot_size:       command queue slot size
 * @num_cus:         number of compute units in program
 * @cu_shift:        shift value to convert CU idx to CU addr
 * @cu_base_addr:    base address to add to CU addr for actual physical address
 *
 * @ert:1            enable embedded HW scheduler
 * @polling:1        poll for command completion
 * @cu_dma:1         enable CUDMA custom module for HW scheduler
 * @cu_isr:1         enable CUISR custom module for HW scheduler
 * @cq_int:1         enable interrupt from host to HW scheduler
 * @cdma:1           enable CDMA kernel
 * @unused:25
 * @dsa52:1          reserved for internal use
 *
 * @data:            addresses of @num_cus CUs
 */
struct ert_configure_cmd {
  union {
    struct {
      uint32_t state:4;          /* [3-0]   */
      uint32_t unused:8;         /* [11-4]  */
      uint32_t count:11;         /* [22-12] */
      uint32_t opcode:5;         /* [27-23] */
      uint32_t type:4;           /* [31-27] */
    };
    uint32_t header;
  };

  /* payload */
  uint32_t slot_size;
  uint32_t num_cus;
  uint32_t cu_shift;
  uint32_t cu_base_addr;

  /* features */
  uint32_t ert:1;
  uint32_t polling:1;
  uint32_t cu_dma:1;
  uint32_t cu_isr:1;
  uint32_t cq_int:1;
  uint32_t cdma:1;
  uint32_t dataflow:1;
  /* WORKAROUND: allow xclRegWrite/xclRegRead access shared CU */
  uint32_t rw_shared:1;
  uint32_t kds_30:1;
  uint32_t dmsg:1;
  uint32_t echo:1;
  uint32_t intr:1;
  uint32_t unusedf:19;
  uint32_t dsa52:1;

  /* cu address map size is num_cus */
#if defined(__linux__) && defined(__KERNEL__)
  uint32_t data[];  /* Flexible array member */
#else
  uint32_t data[1];
#endif
};

/*
 * Note: We need to put maximum 128 soft kernel image
 *       in one config command (1024 DWs including header).
 *       So each one needs to be smaller than 8 DWs.
 *
 * This data struct is obsoleted. Only used in legacy ERT firmware.
 * Use 'struct config_sk_image_uuid' instead on XGQ based ERT.
 *
 * @start_cuidx:     start index of compute units of each image
 * @num_cus:         number of compute units of each image
 * @sk_name:         symbol name of soft kernel of each image
 */
struct config_sk_image {
  uint32_t start_cuidx;
  uint32_t num_cus;
  uint32_t sk_name[5];
};

/*
 * Note: We need to put maximum 128 soft kernel image
 *       in one config command (1024 DWs including header).
 *       So each one needs to be smaller than 8 DWs.
 *
 * @start_cuidx:     start index of compute units of each image
 * @num_cus:         number of compute units of each image
 * @sk_name:         symbol name of soft kernel of each image
 * @sk_uuid:         xclbin uuid that this soft kernel image belones to
 */
struct config_sk_image_uuid {
  uint32_t start_cuidx;
  uint32_t num_cus;
  uint32_t sk_name[5];
  unsigned char     sk_uuid[16];
  uint32_t slot_id;
};

/**
 * struct ert_configure_sk_cmd: ERT configure soft kernel command format
 *
 * @state:           [3-0] current state of a command
 * @count:           [22-12] number of words in payload
 * @opcode:          [27-23] 1, opcode for configure
 * @type:            [31-27] 0, type of configure
 *
 * @num_image:       number of images
*/
struct ert_configure_sk_cmd {
  union {
    struct {
      uint32_t state:4;          /* [3-0]   */
      uint32_t unused:8;         /* [11-4]  */
      uint32_t count:11;         /* [22-12] */
      uint32_t opcode:5;         /* [27-23] */
      uint32_t type:4;           /* [31-27] */
    };
    uint32_t header;
  };

  /* payload */
  uint32_t num_image;
  struct config_sk_image image[1];
};

/**
 * struct ert_unconfigure_sk_cmd: ERT unconfigure soft kernel command format
 *
 * @state:           [3-0] current state of a command
 * @count:           [22-12] number of words in payload
 * @opcode:          [27-23] 1, opcode for configure
 * @type:            [31-27] 0, type of configure
 *
 * @start_cuidx:     start index of compute units
 * @num_cus:         number of compute units in program
 */
struct ert_unconfigure_sk_cmd {
  union {
    struct {
      uint32_t state:4;          /* [3-0]   */
      uint32_t unused:8;         /* [11-4]  */
      uint32_t count:11;         /* [22-12] */
      uint32_t opcode:5;         /* [27-23] */
      uint32_t type:4;           /* [31-27] */
    };
    uint32_t header;
  };

  /* payload */
  uint32_t start_cuidx;
  uint32_t num_cus;
};

/**
 * struct ert_abort_cmd: ERT abort command format.
 *
 * @exec_bo_handle: The bo handle of execbuf command to abort
 */
struct ert_abort_cmd {
  union {
    struct {
      uint32_t state:4;          /* [3-0]   */
      uint32_t custom:8;         /* [11-4]  */
      uint32_t count:11;         /* [22-12] */
      uint32_t opcode:5;         /* [27-23] */
      uint32_t type:4;           /* [31-27] */
    };
    uint32_t header;
  };

  /* payload */
  uint64_t exec_bo_handle;
};

/**
 * struct ert_validate_cmd: ERT BIST command format.
 *
 */
struct ert_validate_cmd {
  union {
    struct {
      uint32_t state:4;          /* [3-0]   */
      uint32_t custom:8;         /* [11-4]  */
      uint32_t count:11;         /* [22-12] */
      uint32_t opcode:5;         /* [27-23] */
      uint32_t type:4;           /* [31-27] */
    };
    uint32_t header;
  };
  uint32_t timestamp;
  uint32_t cq_read_single;
  uint32_t cq_write_single;
  uint32_t cu_read_single;
  uint32_t cu_write_single;
};

/**
 * struct ert_validate_cmd: ERT BIST command format.
 *
 */
struct ert_access_valid_cmd {
  union {
    struct {
      uint32_t state:4;          /* [3-0]   */
      uint32_t custom:8;         /* [11-4]  */
      uint32_t count:11;         /* [22-12] */
      uint32_t opcode:5;         /* [27-23] */
      uint32_t type:4;           /* [31-27] */
    };
    uint32_t header;
  };
  uint32_t h2h_access;
  uint32_t h2d_access;
  uint32_t d2h_access;
  uint32_t d2d_access;
  uint32_t d2cu_access;
  uint32_t wr_count;
  uint32_t wr_test;
};

/**
 * struct ert_ctx_health_data: interpretation of payload for an ert packet
 *                             which has context health data
 *
 * @version:                    context health data version (current 0.0)
 * @txn_op_idx:                 index of last TXN control code executed
 * @ctx_pc:                     program counter for that context
 * @fatal_error_type:           the fatal error type if context crashes
 * @fatal_error_exception_type: LX7 exception type
 * @fatal_error_exception_pc:   LX7 program counter at the time of the exception
 * @fatal_error_app_module:     module name where the exception occurred
 *
 * Field                       Default value  Comment
 * txn_op_idx:                 0xFFFFFFFF     there is no txn control code is running or the
 *                                            last txn control code op idx is not captured
 * ctx_pc:                     0              context .text program counter is not captured
 * fatal_error_type:           0              no fatal error or fatal error is not captured
 * fatal_error_exception_type: 0
 * fatal_error_exception_pc:   0
 * fatal_error_app_module:     0
 *
 * Once an ert packet completes with state ERT_CMD_STATE_TIMEOUT, the ert
 * packet starting from payload will have the following information.
 */
struct ert_ctx_health_data {
  // platform agnositic fields
  uint32_t version;
  uint32_t txn_op_idx;
  uint32_t ctx_pc;
  uint32_t fatal_error_type;
  uint32_t fatal_error_exception_type;
  uint32_t fatal_error_exception_pc;
  uint32_t fatal_error_app_module;
};

/**
 * struct ert_ctx_health_data_aie2: interpretation of payload for an ert v1 packet
 *                                  which has context health data for npu0
 *
 * @txn_op_idx:                 index of last TXN control code executed
 * @ctx_pc:                     program counter for that context
 * @fatal_error_type:           the fatal error type if context crashes
 * @fatal_error_exception_type: LX7 exception type
 * @fatal_error_exception_pc:   LX7 program counter at the time of the exception
 * @fatal_error_app_module:     module name where the exception occurred
 *
 * Field                       Default value  Comment
 * txn_op_idx:                 0xFFFFFFFF     there is no txn control code is running or the
 *                                            last txn control code op idx is not captured
 * ctx_pc:                     0              context .text program counter is not captured
 * fatal_error_type:           0              no fatal error or fatal error is not captured
 * fatal_error_exception_type: 0
 * fatal_error_exception_pc:   0
 * fatal_error_app_module:     0
 *
 * Once an ert packet completes with state ERT_CMD_STATE_TIMEOUT, the ert
 * v1 packet will have the following information for npu0 generation.
 */
struct ert_ctx_health_data_aie2 {
  uint32_t txn_op_idx;
  uint32_t ctx_pc;
  uint32_t fatal_error_type;
  uint32_t fatal_error_exception_type;
  uint32_t fatal_error_exception_pc;
  uint32_t fatal_error_app_module;
};

/**
 * struct ert_uc_health_info: Health data for each cert
 *
 * @uc_idx:            uC index in this context, 0 is the lead
 * @uc_idle_status:    valid when CERT is CTX_IDEL, represent the reason CERT is idle
 *                     hsa_lite_status register:
 *                         bit 0: HSA queue not empty
 *                         bit 1: preemption save completion
 *                         bit 2: CERT is idle
 * @misc_status:       valid when UCCTX_ERROR, represent the reason UC hangs
 *                         bit 0: uC fw exception
 *                         bit 1: control code hang
 * @fw_state:          uC FW state
 * @page_idx:          page index of the current control code
 * @offset:            bytes offset inside page
 * @restore_page:      in case context is preempted, the page index to be executed on resume
 * @restore_offset:    in case context is preempted, the bytes offset inside restore_page to be executed on resume
 * @uc_ear:            in case of uC crash, the exception address of uC
 * @uc_esr:            in case of uC crash, the exception status of uC
 * @uc_pc:             in case of uC crash, the PC of the current uC
 */
struct ert_uc_health_info {
  uint32_t uc_idx;
  uint32_t uc_idle_status;
  uint32_t misc_status;
  uint32_t fw_state;
  uint32_t page_idx;
  uint32_t offset;
  uint32_t restore_page;
  uint32_t restore_offset;
  uint32_t uc_ear;
  uint32_t uc_esr;
  uint32_t uc_pc;
};

/**
 * struct ert_ctx_health_data_aie4: interpretation of payload for an ert v1 packet 
 *                                  which has context health data for aie2ps and aie4
 *
 * @ctx_state:             context state
 * @num_ucs:               number of uC reported
 * @uc_info:               array for health data for each uC in the context.
 *                         the array size is based on num_certs.
 *
 * Once an ert packet completes with state ERT_CMD_STATE_TIMEOUT, the ert
 * v1 packet will have the following information for aie2ps/aie4 generation.
 */
struct ert_ctx_health_data_aie4 {
  uint32_t ctx_state;
  uint32_t num_uc;
  struct ert_uc_health_info uc_info[];
};

/**
 * struct ert_ctx_health_data_v1: interpretation of payload for an ert v1 packet 
 *
 * @version:               context health data version (1)
 * @npu_gen:               npu generation
 * @aie2:                  context health data for npu generation aie2/aie2p
 * @aie4:                  context health data for npu generation aie2ps/aie4
 *
 * If version is 1, we should use this data structure to parse context health data
 * starting from the ert packet payload. And use corresponding data structure based
 * on the npu generation.
 */
struct ert_ctx_health_data_v1 {
  uint32_t version;
  uint32_t npu_gen;
  union {
    struct ert_ctx_health_data_aie2 aie2;
    struct ert_ctx_health_data_aie4 aie4;
  };
};

/**
 * Enum for context health data version (between XRT shim and driver).
 * The version field in ert_ctx_health_data/_v1 will be set by the driver
 */
enum ert_ctx_health_data_version {
  ERT_CTX_HEALTH_DATA_V0 = 0,
  ERT_CTX_HEALTH_DATA_V1 = 1
};

/**
 * Enum for device generation type.
 * The npu_gen field in ert_ctx_health_data_v1 is set by the driver based on the npu device device:
 *   NPU_GEN_AIE2 -> for AIE2/AIE2P
 *   NPU_GEN_AIE4 -> for AIE4/AIE2PS
 */
enum npu_gen_type {
  NPU_GEN_AIE2,
  NPU_GEN_AIE4
};

/**
 * ERT command state
 *
 * @ERT_CMD_STATE_NEW:         Set by host before submitting a command to
 *                             scheduler
 * @ERT_CMD_STATE_QUEUED:      Internal scheduler state
 * @ERT_CMD_STATE_SUBMITTED:   Internal scheduler state
 * @ERT_CMD_STATE_RUNNING:     Internal scheduler state
 * @ERT_CMD_STATE_COMPLETED:   Set by scheduler when command completes
 * @ERT_CMD_STATE_ERROR:       Set by scheduler if command failed
 * @ERT_CMD_STATE_ABORT:       Set by scheduler if command abort
 * @ERT_CMD_STATE_TIMEOUT:     Set by scheduler if command timeout and reset
 * @ERT_CMD_STATE_NORESPONSE:  Set by scheduler if command timeout and fail to
 *                             reset
 */
enum ert_cmd_state {
  ERT_CMD_STATE_NEW = 1,
  ERT_CMD_STATE_QUEUED = 2,
  ERT_CMD_STATE_RUNNING = 3,
  ERT_CMD_STATE_COMPLETED = 4,
  ERT_CMD_STATE_ERROR = 5,
  ERT_CMD_STATE_ABORT = 6,
  ERT_CMD_STATE_SUBMITTED = 7,
  ERT_CMD_STATE_TIMEOUT = 8,
  ERT_CMD_STATE_NORESPONSE = 9,
  ERT_CMD_STATE_SKERROR = 10, //Check for error return code from Soft Kernel
  ERT_CMD_STATE_SKCRASHED = 11, //Soft kernel has crashed
  ERT_CMD_STATE_MAX, // Always the last one
};

struct cu_cmd_state_timestamps {
  uint64_t skc_timestamps[ERT_CMD_STATE_MAX]; // In nano-second
};

/**
 * Opcode types for commands
 *
 * @ERT_START_CU:          start a workgroup on a CU
 * @ERT_START_KERNEL:      currently aliased to ERT_START_CU
 * @ERT_CONFIGURE:         configure command scheduler
 * @ERT_EXEC_WRITE:        execute a specified CU after writing
 * @ERT_CU_STAT:           get stats about CU execution
 * @ERT_START_COPYBO:      start KDMA CU or P2P, may be converted to ERT_START_CU
 *                         before cmd reach to scheduler, short-term hack
 * @ERT_SK_CONFIG:         configure soft kernel
 * @ERT_SK_START:          start a soft kernel
 * @ERT_SK_UNCONFIG:       unconfigure a soft kernel
 * @ERT_START_KEY_VAL:     same as ERT_START_CU but with key-value pair flavor
 * @ERT_START_DPU:         instruction buffer command format
 * @ERT_CMD_CHAIN:         command chain
 * @ERT_START_NPU:         instruction buffer command format on NPU format
 * @ERT_START_NPU_PREEMPT: instruction buffer command with preemption format on NPU
 */
enum ert_cmd_opcode {
  ERT_START_CU              = 0,
  ERT_START_KERNEL          = 0,
  ERT_CONFIGURE             = 2,
  ERT_EXIT                  = 3,
  ERT_ABORT                 = 4,
  ERT_EXEC_WRITE            = 5,
  ERT_CU_STAT               = 6,
  ERT_START_COPYBO          = 7,
  ERT_SK_CONFIG             = 8,
  ERT_SK_START              = 9,
  ERT_SK_UNCONFIG           = 10,
  ERT_INIT_CU               = 11,
  ERT_START_FA              = 12,
  ERT_CLK_CALIB             = 13,
  ERT_MB_VALIDATE           = 14,
  ERT_START_KEY_VAL         = 15,
  ERT_ACCESS_TEST_C         = 16,
  ERT_ACCESS_TEST           = 17,
  ERT_START_DPU             = 18,
  ERT_CMD_CHAIN             = 19,
  ERT_START_NPU             = 20,
  ERT_START_NPU_PREEMPT     = 21,
  ERT_START_NPU_PREEMPT_ELF = 22,
};

/**
 * Command types
 *
 * @ERT_DEFAULT:        default command type
 * @ERT_KDS_LOCAL:      command processed by KDS locally
 * @ERT_CTRL:           control command uses reserved command queue slot
 * @ERT_CU:             compute unit command
 */
enum ert_cmd_type {
  ERT_DEFAULT = 0,
  ERT_KDS_LOCAL = 1,
  ERT_CTRL = 2,
  ERT_CU = 3,
  ERT_SCU = 4,
};

/**
 * Soft kernel types
 *
 * @SOFTKERNEL_TYPE_EXEC:       executable
 */
enum softkernel_type {
  SOFTKERNEL_TYPE_EXEC = 0,
};

/*
 * Base address GPIO per spec
 * | Offset  | Description
 * -----------------------
 * | 0x00    | ERT_MGMT_PF_base_addr (Not sure where this should be use)
 * | 0x08    | ERT_USER_PF_base_addr. The base address of ERT peripherals
 */
#if defined(ERT_BUILD_V20)
uint32_t ert_base_addr = 0;
# define ERT_BASE_ADDR                     0x01F30008
#endif

#if defined(ERT_BUILD_V30)
uint32_t ert_base_addr = 0;
# define ERT_BASE_ADDR                     0x01F30008
#endif

/**
 * Address constants per spec
 */
#define ERT_WORD_SIZE                      4          /* 4 bytes */
#define ERT_CQ_SIZE                        0x10000    /* 64K */
#if defined(ERT_BUILD_U50)
# define ERT_CQ_BASE_ADDR                  0x340000
# define ERT_CSR_ADDR                      0x360000
#elif defined(ERT_BUILD_V20)
# define ERT_CQ_BASE_ADDR                  (0x000000 + ert_base_addr)
# define ERT_CSR_ADDR                      (0x010000 + ert_base_addr)
#elif defined(ERT_BUILD_V30)
# define ERT_CQ_BASE_ADDR                  0x1F60000
# define ERT_CSR_ADDR                      (0x010000 + ert_base_addr)
#else
# define ERT_CQ_BASE_ADDR                  0x190000
# define ERT_CSR_ADDR                      0x180000
#endif

/**
 * The STATUS REGISTER is for communicating completed CQ slot indices
 * MicroBlaze write, host reads.  MB(W) / HOST(COR)
 */
#define ERT_STATUS_REGISTER_ADDR          (ERT_CSR_ADDR)
#define ERT_STATUS_REGISTER_ADDR0         (ERT_CSR_ADDR)
#define ERT_STATUS_REGISTER_ADDR1         (ERT_CSR_ADDR + 0x4)
#define ERT_STATUS_REGISTER_ADDR2         (ERT_CSR_ADDR + 0x8)
#define ERT_STATUS_REGISTER_ADDR3         (ERT_CSR_ADDR + 0xC)

/**
 * The CU DMA REGISTER is for communicating which CQ slot is to be started
 * on a specific CU.  MB selects a free CU on which the command can
 * run, then writes the 1<<CU back to the command slot CU mask and
 * writes the slot index to the CU DMA REGISTER.  HW is notified when
 * the register is written and now does the DMA transfer of CU regmap
 * map from command to CU, while MB continues its work. MB(W) / HW(R)
 */
#define ERT_CU_DMA_ENABLE_ADDR            (ERT_CSR_ADDR + 0x18)
#define ERT_CU_DMA_REGISTER_ADDR          (ERT_CSR_ADDR + 0x1C)
#define ERT_CU_DMA_REGISTER_ADDR0         (ERT_CSR_ADDR + 0x1C)
#define ERT_CU_DMA_REGISTER_ADDR1         (ERT_CSR_ADDR + 0x20)
#define ERT_CU_DMA_REGISTER_ADDR2         (ERT_CSR_ADDR + 0x24)
#define ERT_CU_DMA_REGISTER_ADDR3         (ERT_CSR_ADDR + 0x28)

/**
 * The SLOT SIZE is the size of slots in command queue, it is
 * configurable per xclbin. MB(W) / HW(R)
 */
#define ERT_CQ_SLOT_SIZE_ADDR             (ERT_CSR_ADDR + 0x2C)

/**
 * The CU_OFFSET is the size of a CU's address map in power of 2.  For
 * example a 64K regmap is 2^16 so 16 is written to the CU_OFFSET_ADDR.
 * MB(W) / HW(R)
 */
#define ERT_CU_OFFSET_ADDR                (ERT_CSR_ADDR + 0x30)

/**
 * The number of slots is command_queue_size / slot_size.
 * MB(W) / HW(R)
 */
#define ERT_CQ_NUMBER_OF_SLOTS_ADDR       (ERT_CSR_ADDR + 0x34)

/**
 * All CUs placed in same address space separated by CU_OFFSET. The
 * CU_BASE_ADDRESS is the address of the first CU. MB(W) / HW(R)
 */
#define ERT_CU_BASE_ADDRESS_ADDR          (ERT_CSR_ADDR + 0x38)

/**
 * The CQ_BASE_ADDRESS is the base address of the command queue.
 * MB(W) / HW(R)
 */
#define ERT_CQ_BASE_ADDRESS_ADDR          (ERT_CSR_ADDR + 0x3C)

/**
 * The CU_ISR_HANDLER_ENABLE (MB(W)/HW(R)) enables the HW handling of
 * CU interrupts.  When a CU interrupts (when done), hardware handles
 * the interrupt and writes the index of the CU that completed into
 * the CU_STATUS_REGISTER (HW(W)/MB(COR)) as a bitmask
 */
#define ERT_CU_ISR_HANDLER_ENABLE_ADDR    (ERT_CSR_ADDR + 0x40)
#define ERT_CU_STATUS_REGISTER_ADDR       (ERT_CSR_ADDR + 0x44)
#define ERT_CU_STATUS_REGISTER_ADDR0      (ERT_CSR_ADDR + 0x44)
#define ERT_CU_STATUS_REGISTER_ADDR1      (ERT_CSR_ADDR + 0x48)
#define ERT_CU_STATUS_REGISTER_ADDR2      (ERT_CSR_ADDR + 0x4C)
#define ERT_CU_STATUS_REGISTER_ADDR3      (ERT_CSR_ADDR + 0x50)

/**
 * The CQ_STATUS_ENABLE (MB(W)/HW(R)) enables interrupts from HOST to
 * MB to indicate the presense of a new command in some slot.  The
 * slot index is written to the CQ_STATUS_REGISTER (HOST(W)/MB(R))
 */
#define ERT_CQ_STATUS_ENABLE_ADDR         (ERT_CSR_ADDR + 0x54)
#define ERT_CQ_STATUS_REGISTER_ADDR       (ERT_CSR_ADDR + 0x58)
#define ERT_CQ_STATUS_REGISTER_ADDR0      (ERT_CSR_ADDR + 0x58)
#define ERT_CQ_STATUS_REGISTER_ADDR1      (ERT_CSR_ADDR + 0x5C)
#define ERT_CQ_STATUS_REGISTER_ADDR2      (ERT_CSR_ADDR + 0x60)
#define ERT_CQ_STATUS_REGISTER_ADDR3      (ERT_CSR_ADDR + 0x64)

/**
 * The NUMBER_OF_CU (MB(W)/HW(R) is the number of CUs per current
 * xclbin.  This is an optimization that allows HW to only check CU
 * completion on actual CUs.
 */
#define ERT_NUMBER_OF_CU_ADDR             (ERT_CSR_ADDR + 0x68)

/**
 * Enable global interrupts from MB to HOST on command completion.
 * When enabled writing to STATUS_REGISTER causes an interrupt in HOST.
 * MB(W)
 */
#define ERT_HOST_INTERRUPT_ENABLE_ADDR    (ERT_CSR_ADDR + 0x100)

/**
 * Interrupt controller base address
 * This value is per hardware BSP (XPAR_INTC_SINGLE_BASEADDR)
 */
#if defined(ERT_BUILD_U50)
# define ERT_INTC_ADDR                     0x00310000
#elif defined(ERT_BUILD_V20)
# define ERT_INTC_ADDR                     0x01F20000
#elif defined(ERT_BUILD_V30)
# define ERT_INTC_ADDR                     0x01F20000
# define ERT_INTC_CU_0_31_ADDR             (0x0000 + ert_base_addr)
# define ERT_INTC_CU_32_63_ADDR            (0x1000 + ert_base_addr)
# define ERT_INTC_CU_64_95_ADDR            (0x2000 + ert_base_addr)
# define ERT_INTC_CU_96_127_ADDR           (0x3000 + ert_base_addr)
#else
# define ERT_INTC_ADDR                     0x41200000
# define ERT_INTC_CU_0_31_ADDR             0x0000
# define ERT_INTC_CU_32_63_ADDR            0x1000
# define ERT_INTC_CU_64_95_ADDR            0x2000
# define ERT_INTC_CU_96_127_ADDR           0x3000
#endif

/**
 * Look up table for CUISR for CU addresses
 */
#define ERT_CUISR_LUT_ADDR                (ERT_CSR_ADDR + 0x400)

/**
 * ERT exit command/ack
 */
#define	ERT_EXIT_CMD			  ((ERT_EXIT << 23) | ERT_CMD_STATE_NEW)
#define	ERT_EXIT_ACK			  (ERT_CMD_STATE_COMPLETED)
#define	ERT_EXIT_CMD_OP			  (ERT_EXIT << 23)

/**
 * State machine for both CUDMA and CUISR modules
 */
#define ERT_HLS_MODULE_IDLE               0x1
#define ERT_CUDMA_STATE                   (ERT_CSR_ADDR + 0x318)
#define ERT_CUISR_STATE                   (ERT_CSR_ADDR + 0x328)

/**
 * Interrupt address masks written by MB when interrupts from
 * CU are enabled
 */
#define ERT_INTC_IPR_ADDR                 (ERT_INTC_ADDR + 0x4)  /* pending */
#define ERT_INTC_IER_ADDR                 (ERT_INTC_ADDR + 0x8)  /* enable */
#define ERT_INTC_IAR_ADDR                 (ERT_INTC_ADDR + 0x0C) /* acknowledge */
#define ERT_INTC_MER_ADDR                 (ERT_INTC_ADDR + 0x1C) /* master enable */

#define ERT_INTC_CU_0_31_IPR              (ERT_INTC_CU_0_31_ADDR + 0x4)  /* pending */
#define ERT_INTC_CU_0_31_IER              (ERT_INTC_CU_0_31_ADDR + 0x8)  /* enable */
#define ERT_INTC_CU_0_31_IAR              (ERT_INTC_CU_0_31_ADDR + 0x0C) /* acknowledge */
#define ERT_INTC_CU_0_31_MER              (ERT_INTC_CU_0_31_ADDR + 0x1C) /* master enable */

#define ERT_INTC_CU_32_63_IPR             (ERT_INTC_CU_32_63_ADDR + 0x4)  /* pending */
#define ERT_INTC_CU_32_63_IER             (ERT_INTC_CU_32_63_ADDR + 0x8)  /* enable */
#define ERT_INTC_CU_32_63_IAR             (ERT_INTC_CU_32_63_ADDR + 0x0C) /* acknowledge */
#define ERT_INTC_CU_32_63_MER             (ERT_INTC_CU_32_63_ADDR + 0x1C) /* master enable */

#define ERT_INTC_CU_64_95_IPR             (ERT_INTC_CU_64_95_ADDR + 0x4)  /* pending */
#define ERT_INTC_CU_64_95_IER             (ERT_INTC_CU_64_95_ADDR + 0x8)  /* enable */
#define ERT_INTC_CU_64_95_IAR             (ERT_INTC_CU_64_95_ADDR + 0x0C) /* acknowledge */
#define ERT_INTC_CU_64_95_MER             (ERT_INTC_CU_64_95_ADDR + 0x1C) /* master enable */

#define ERT_INTC_CU_96_127_IPR            (ERT_INTC_CU_96_127_ADDR + 0x4)  /* pending */
#define ERT_INTC_CU_96_127_IER            (ERT_INTC_CU_96_127_ADDR + 0x8)  /* enable */
#define ERT_INTC_CU_96_127_IAR            (ERT_INTC_CU_96_127_ADDR + 0x0C) /* acknowledge */
#define ERT_INTC_CU_96_127_MER            (ERT_INTC_CU_96_127_ADDR + 0x1C) /* master enable */


#if defined(ERT_BUILD_V30)
# define ERT_CLK_COUNTER_ADDR              0x1F70000
#else
# define ERT_CLK_COUNTER_ADDR              0x0
#endif

/*
* Used in driver and user space code
*/
/*
* Upper limit on number of dependencies in execBuf waitlist
*/
#define MAX_DEPS        8

/*
* Maximum size of mandatory fields in bytes for all packet type
*/
#define MAX_HEADER_SIZE 64

/*
* Maximum size of mandatory fields in bytes for all packet type
*/
#define MAX_CONFIG_PACKET_SIZE 512

/*
* Maximum size of CQ slot
*/
#define MAX_CQ_SLOT_SIZE 4096

/*
 * Helper functions to hide details of ert_start_copybo_cmd
 */
static inline void
ert_fill_copybo_cmd(struct ert_start_copybo_cmd *pkt, uint32_t src_bo,
  uint32_t dst_bo, uint64_t src_offset, uint64_t dst_offset, uint32_t size)
{
  pkt->state = ERT_CMD_STATE_NEW;
  pkt->extra_cu_masks = 3;
  pkt->count = 16;
  pkt->opcode = ERT_START_COPYBO;
  pkt->type = ERT_DEFAULT;
  pkt->cu_mask[0] = 0;
  pkt->cu_mask[1] = 0;
  pkt->cu_mask[2] = 0;
  pkt->cu_mask[3] = 0;
  pkt->src_addr_lo = (uint32_t)src_offset;
  pkt->src_addr_hi = (src_offset >> 32) & 0xFFFFFFFF;
  pkt->src_bo_hdl = src_bo;
  pkt->dst_addr_lo = (uint32_t)dst_offset;
  pkt->dst_addr_hi = (dst_offset >> 32) & 0xFFFFFFFF;
  pkt->dst_bo_hdl = dst_bo;
  pkt->size = size;
  pkt->size_hi = 0; /* set to 0 explicitly */
  pkt->arg = 0;
}
static inline uint64_t
ert_copybo_src_offset(struct ert_start_copybo_cmd *pkt)
{
  return (uint64_t)pkt->src_addr_hi << 32 | pkt->src_addr_lo;
}
static inline uint64_t
ert_copybo_dst_offset(struct ert_start_copybo_cmd *pkt)
{
  return (uint64_t)pkt->dst_addr_hi << 32 | pkt->dst_addr_lo;
}
static inline uint64_t
ert_copybo_size(struct ert_start_copybo_cmd *pkt)
{
  return pkt->size;
}

static inline bool
ert_valid_opcode(struct ert_packet *pkt)
{
  struct ert_start_kernel_cmd *skcmd;
  struct ert_init_kernel_cmd *ikcmd;
  struct ert_start_copybo_cmd *sccmd;
  struct ert_configure_cmd *ccmd;
  struct ert_configure_sk_cmd *cscmd;
  struct ert_cmd_chain_data *ccdata;
  bool valid;

  switch (pkt->opcode) {
  case ERT_START_CU:
    skcmd = to_start_krnl_pkg(pkt);
    /* 1 cu mask + 4 registers */
    valid = (skcmd->count >= skcmd->extra_cu_masks + 1 + 4);
    break;
  case ERT_START_DPU:
    skcmd = to_start_krnl_pkg(pkt);
    /* 1 mandatory cumask + extra_cu_masks + size (in words) of ert_dpu_data */
    valid = (skcmd->count >= 1+ skcmd->extra_cu_masks + sizeof(struct ert_dpu_data) / sizeof(uint32_t));
    break;
  case ERT_CMD_CHAIN:
    ccdata = (struct ert_cmd_chain_data*) pkt->data;
    /* header count must match number of commands in payload */
    valid = (pkt->count == (ccdata->command_count * sizeof(uint64_t) + sizeof(struct ert_cmd_chain_data)) / sizeof(uint32_t));
    break;
  case ERT_START_NPU:
    skcmd = to_start_krnl_pkg(pkt);
    /* 1 mandatory cumask + extra_cu_masks + ert_npu_data */
    valid = (skcmd->count >= 1+ skcmd->extra_cu_masks + sizeof(struct ert_npu_data) / sizeof(uint32_t));
    break;
  case ERT_START_NPU_PREEMPT:
    skcmd = to_start_krnl_pkg(pkt);
    /* 1 mandatory cumask + extra_cu_masks + ert_npu_preempt_data */
    valid = (skcmd->count >= 1+ skcmd->extra_cu_masks + sizeof(struct ert_npu_preempt_data) / sizeof(uint32_t));
    break;
  case ERT_START_NPU_PREEMPT_ELF:
    skcmd = to_start_krnl_pkg(pkt);
    /* 1 mandatory cumask + extra_cu_masks + ert_npu_preempt_data */
    valid = (skcmd->count >= 1+ skcmd->extra_cu_masks + sizeof(struct ert_npu_preempt_data) / sizeof(uint32_t));
    break;
  case ERT_START_KEY_VAL:
    skcmd = to_start_krnl_pkg(pkt);
    /* 1 cu mask */
    valid = (skcmd->count >= skcmd->extra_cu_masks + 1);
    break;
  case ERT_EXEC_WRITE:
    skcmd = to_start_krnl_pkg(pkt);
    /* 1 cu mask + 6 registers */
    valid = (skcmd->count >= skcmd->extra_cu_masks + 1 + 6);
    break;
  case ERT_START_FA:
    skcmd = to_start_krnl_pkg(pkt);
    /* 1 cu mask */
    valid = (skcmd->count >= skcmd->extra_cu_masks + 1);
    break;
  case ERT_SK_START:
    skcmd = to_start_krnl_pkg(pkt);
    /* 1 cu mask + 1 control word */
    valid = (skcmd->count >= skcmd->extra_cu_masks + 1 + 1);
    break;
  case ERT_CONFIGURE:
    ccmd = to_cfg_pkg(pkt);
    /* 5 mandatory fields in struct */
    valid = (ccmd->count >= 5 + ccmd->num_cus);
    break;
  case ERT_START_COPYBO:
    sccmd = to_copybo_pkg(pkt);
    valid = (sccmd->count == 16);
    break;
  case ERT_INIT_CU:
    ikcmd = to_init_krnl_pkg(pkt);
    /* 9 mandatory words in struct + 4 control registers */
    valid = (ikcmd->count >= ikcmd->extra_cu_masks + 9 + 4);
    break;
  case ERT_SK_CONFIG:
    cscmd = to_cfg_sk_pkg(pkt);
    valid = (cscmd->count == sizeof(struct config_sk_image) * cscmd->num_image / 4 + 1);
    break;
  case ERT_CLK_CALIB:
  case ERT_MB_VALIDATE:
  case ERT_ACCESS_TEST_C:
  case ERT_CU_STAT: /* TODO: Rules to validate? */
  case ERT_EXIT:
  case ERT_ABORT:
    valid = true;
    break;
  case ERT_SK_UNCONFIG: /* NOTE: obsolete */
  default:
    valid = false;
  }

  return valid;
}

static inline uint64_t
get_ert_packet_size_bytes(struct ert_packet *pkt)
{
  // header plus payload
  return sizeof(pkt->header) + pkt->count * sizeof(uint32_t);
}

static inline struct ert_dpu_data*
get_ert_dpu_data(struct ert_start_kernel_cmd* pkt)
{
  if (pkt->opcode != ERT_START_DPU)
    return NULL;

  // past extra cu_masks embedded in the packet data
  return (struct ert_dpu_data*) (pkt->data + pkt->extra_cu_masks);
}

static inline struct ert_dpu_data*
get_ert_dpu_data_next(struct ert_dpu_data* dpu_data)
{
  if (dpu_data->chained == 0)
    return NULL;

  return dpu_data + 1;
}

static inline struct ert_cmd_chain_data*
get_ert_cmd_chain_data(struct ert_packet* pkt)
{
  if (pkt->opcode != ERT_CMD_CHAIN)
    return NULL;

  return (struct ert_cmd_chain_data*) pkt->data;
}

static inline struct ert_npu_data*
get_ert_npu_data(struct ert_start_kernel_cmd* pkt)
{
  if (pkt->opcode != ERT_START_NPU)
    return NULL;

  // past extra cu_masks embedded in the packet data
  return (struct ert_npu_data*) (pkt->data + pkt->extra_cu_masks);
}

static inline struct ert_npu_preempt_data*
get_ert_npu_preempt_data(struct ert_start_kernel_cmd* pkt)
{
  if (pkt->opcode != ERT_START_NPU_PREEMPT)
    return NULL;

  // past extra cu_masks embedded in the packet data
  return (struct ert_npu_preempt_data*) (pkt->data + pkt->extra_cu_masks);
}

static inline struct ert_npu_preempt_data*
get_ert_npu_elf_data(struct ert_start_kernel_cmd* pkt)
{
  if (pkt->opcode != ERT_START_NPU_PREEMPT_ELF)
    return NULL;
  // past extra cu_masks embedded in the packet data
  return (struct ert_npu_preempt_data*) (pkt->data + pkt->extra_cu_masks);
}

static inline uint32_t*
get_ert_regmap_begin(struct ert_start_kernel_cmd* pkt)
{
  switch (pkt->opcode) {
  case ERT_START_DPU:
    return pkt->data + pkt->extra_cu_masks
      + (get_ert_dpu_data(pkt)->chained + 1) * sizeof(struct ert_dpu_data) / sizeof(uint32_t);

  case ERT_START_NPU:
    return pkt->data + pkt->extra_cu_masks
      + sizeof(struct ert_npu_data) / sizeof(uint32_t)
      + get_ert_npu_data(pkt)->instruction_prop_count;

  case ERT_START_NPU_PREEMPT:
    return pkt->data + pkt->extra_cu_masks
      + sizeof(struct ert_npu_preempt_data) / sizeof(uint32_t)
      + get_ert_npu_preempt_data(pkt)->instruction_prop_count;

  case ERT_START_NPU_PREEMPT_ELF:
    return pkt->data + pkt->extra_cu_masks
      + sizeof(struct ert_npu_preempt_data) / sizeof(uint32_t)
      + get_ert_npu_elf_data(pkt)->instruction_prop_count;

  default:
    // skip past embedded extra cu_masks
    return pkt->data + pkt->extra_cu_masks;
  }
}

static inline uint32_t*
get_ert_regmap_end(struct ert_start_kernel_cmd* pkt)
{
  // pkt->count includes the mandatory cumask which precededs data array
  return &pkt->cu_mask + pkt->count;
}

static inline uint64_t
get_ert_regmap_size_bytes(struct ert_start_kernel_cmd* pkt)
{
  return (get_ert_regmap_end(pkt) - get_ert_regmap_begin(pkt)) * sizeof(uint32_t);
}
/* ert_ctx_health_data structure is valid only if the ert opcode is START_NPU_* and cmd state is ERT_CMD_STATE_TIMEOUT*/
static inline const struct ert_ctx_health_data*
get_ert_ctx_health_data(const struct ert_packet* pkt)
{
  switch (pkt->opcode) {
  case ERT_START_CU:
  case ERT_START_NPU:
  case ERT_START_NPU_PREEMPT:
  case ERT_START_NPU_PREEMPT_ELF:
    if (pkt->state == ERT_CMD_STATE_TIMEOUT)
      return (const struct ert_ctx_health_data*) pkt->data;
  }
  return NULL;
}

static inline const struct ert_ctx_health_data_v1*
get_ert_ctx_health_data_v1(const struct ert_packet* pkt)
{
  switch (pkt->opcode) {
  case ERT_START_CU:
  case ERT_START_NPU:
  case ERT_START_DPU:
  case ERT_START_NPU_PREEMPT:
  case ERT_START_NPU_PREEMPT_ELF:
    if (pkt->state == ERT_CMD_STATE_TIMEOUT)
      return (const struct ert_ctx_health_data_v1*) pkt->data;
  }
  return NULL;
}

#ifdef __linux__
#define P2ROUNDUP(x, align)     (-(-(x) & -(align)))
static inline struct cu_cmd_state_timestamps *
ert_start_kernel_timestamps(struct ert_start_kernel_cmd *pkt)
{
  uint64_t offset = pkt->count * sizeof(uint32_t) + sizeof(pkt->header);
  /* Make sure the offset of timestamps are properly aligned. */
  return (struct cu_cmd_state_timestamps *)
    ((char *)pkt + P2ROUNDUP(offset, sizeof(uint64_t)));
}

/* Return 0 if this pkt doesn't support timestamp or disabled */
static inline int
get_size_with_timestamps_or_zero(struct ert_packet *pkt)
{
  struct ert_start_kernel_cmd *skcmd;
  int size = 0;

  switch (pkt->opcode) {
  case ERT_START_CU:
  case ERT_EXEC_WRITE:
  case ERT_START_FA:
  case ERT_SK_START:
    skcmd = to_start_krnl_pkg(pkt);
    if (skcmd->stat_enabled) {
      size = (char *)ert_start_kernel_timestamps(skcmd) - (char *)pkt;
      size += sizeof(struct cu_cmd_state_timestamps);
    }
  }

  return size;
}
#endif

#if defined(__GNUC__)
# pragma GCC diagnostic pop
#endif

#ifdef _WIN32
# pragma warning( pop )
#endif

#endif
