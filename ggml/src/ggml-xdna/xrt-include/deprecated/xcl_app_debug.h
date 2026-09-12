/**
 * Copyright (C) 2016-2021 Xilinx, Inc
 * Copyright (C) 2022 Advanced Micro Devices, Inc. - All rights reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

/**
 * Xilinx SDAccel HAL userspace driver extension APIs
 * Performance Monitoring Exposed Parameters
 * Copyright (C) 2015-2017, Xilinx Inc - All rights reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#ifndef _XCL_DEBUG_H_
#define _XCL_DEBUG_H_

// For performance counter definitions
#include "xclperf.h"

#ifdef __cplusplus
extern "C" {
#endif

/************************ AIM Debug Counters ********************************/
#define XAIM_DEBUG_SAMPLE_COUNTERS_PER_SLOT           9
#define XAIM_TOTAL_DEBUG_SAMPLE_COUNTERS_PER_SLOT     13

#define XAIM_WRITE_BYTES_INDEX         0
#define XAIM_WRITE_TRANX_INDEX         1
#define XAIM_READ_BYTES_INDEX          2
#define XAIM_READ_TRANX_INDEX          3
#define XAIM_OUTSTANDING_COUNT_INDEX   4
#define XAIM_WRITE_LAST_ADDRESS_INDEX  5
#define XAIM_WRITE_LAST_DATA_INDEX     6
#define XAIM_READ_LAST_ADDRESS_INDEX   7
#define XAIM_READ_LAST_DATA_INDEX      8

#define XAIM_IOCTL_WRITE_BYTES_INDEX         0
#define XAIM_IOCTL_WRITE_TRANX_INDEX         1
#define XAIM_IOCTL_WRITE_LATENCY_INDEX       2
#define XAIM_IOCTL_WRITE_BUSY_CYCLES_INDEX   3
#define XAIM_IOCTL_READ_BYTES_INDEX          4
#define XAIM_IOCTL_READ_TRANX_INDEX          5
#define XAIM_IOCTL_READ_LATENCY_INDEX        6
#define XAIM_IOCTL_READ_BUSY_CYCLES_INDEX    7
#define XAIM_IOCTL_OUTSTANDING_COUNT_INDEX   8
#define XAIM_IOCTL_WRITE_LAST_ADDRESS_INDEX  9
#define XAIM_IOCTL_WRITE_LAST_DATA_INDEX     10
#define XAIM_IOCTL_READ_LAST_ADDRESS_INDEX   11
#define XAIM_IOCTL_READ_LAST_DATA_INDEX      12

/************************ AM Debug Counters ********************************/
#define XAM_DEBUG_SAMPLE_COUNTERS_PER_SLOT     8
#define XAM_TOTAL_DEBUG_COUNTERS_PER_SLOT      10

#define XAM_ACCEL_EXECUTION_COUNT_INDEX       0
#define XAM_ACCEL_EXECUTION_CYCLES_INDEX      1
#define XAM_ACCEL_STALL_INT_INDEX             2
#define XAM_ACCEL_STALL_STR_INDEX             3
#define XAM_ACCEL_STALL_EXT_INDEX             4
#define XAM_ACCEL_MIN_EXECUTION_CYCLES_INDEX  5
#define XAM_ACCEL_MAX_EXECUTION_CYCLES_INDEX  6
#define XAM_ACCEL_TOTAL_CU_START_INDEX        7

#define XAM_IOCTL_EXECUTION_COUNT_INDEX       0
#define XAM_IOCTL_START_COUNT_INDEX           1
#define XAM_IOCTL_EXECUTION_CYCLES_INDEX      2
#define XAM_IOCTL_STALL_INT_INDEX             3
#define XAM_IOCTL_STALL_STR_INDEX             4
#define XAM_IOCTL_STALL_EXT_INDEX             5
#define XAM_IOCTL_BUSY_CYCLES_INDEX           6
#define XAM_IOCTL_MAX_PARALLEL_ITR_INDEX      7
#define XAM_IOCTL_MAX_EXECUTION_CYCLES_INDEX  8
#define XAM_IOCTL_MIN_EXECUTION_CYCLES_INDEX  9

/************************ ASM Debug Counters ********************************/
#define XASM_DEBUG_SAMPLE_COUNTERS_PER_SLOT    5

#define XASM_NUM_TRANX_INDEX      0
#define XASM_DATA_BYTES_INDEX     1
#define XASM_BUSY_CYCLES_INDEX    2
#define XASM_STALL_CYCLES_INDEX   3
#define XASM_STARVE_CYCLES_INDEX  4



/*
 * LAPC related defs here
 */
#define XLAPC_MAX_NUMBER_SLOTS           31
#define XLAPC_STATUS_PER_SLOT            9
#define XLAPC_STATUS_REG_NUM             4

/* Metric counters per slot */
#define XLAPC_OVERALL_STATUS                0
#define XLAPC_CUMULATIVE_STATUS_0           1
#define XLAPC_CUMULATIVE_STATUS_1           2
#define XLAPC_CUMULATIVE_STATUS_2           3
#define XLAPC_CUMULATIVE_STATUS_3           4
#define XLAPC_SNAPSHOT_STATUS_0             5
#define XLAPC_SNAPSHOT_STATUS_1             6
#define XLAPC_SNAPSHOT_STATUS_2             7
#define XLAPC_SNAPSHOT_STATUS_3             8


/*
 * AXI Streaming Protocol Checker related defs here
 */
#define XSPC_MAX_NUMBER_SLOTS 31
#define XSPC_STATUS_PER_SLOT  3

#define XSPC_PC_ASSERTED 0
#define XSPC_CURRENT_PC  1
#define XSPC_SNAPSHOT_PC 2


/********************** Definitions: Enums, Structs ***************************/
enum xclDebugReadType {
  XCL_DEBUG_READ_TYPE_APM  = 0,
  XCL_DEBUG_READ_TYPE_LAPC = 1,
  XCL_DEBUG_READ_TYPE_AIM  = 2,
  XCL_DEBUG_READ_TYPE_ASM  = 3,
  XCL_DEBUG_READ_TYPE_AM   = 4,
  XCL_DEBUG_READ_TYPE_SPC  = 5,
  XCL_DEBUG_READ_TYPE_ADD  = 6
};

/* Debug counter results */
typedef struct {
  unsigned long long int WriteBytes     [XAIM_MAX_NUMBER_SLOTS];
  unsigned long long int WriteTranx     [XAIM_MAX_NUMBER_SLOTS];
  unsigned long long int ReadBytes      [XAIM_MAX_NUMBER_SLOTS];
  unsigned long long int ReadTranx      [XAIM_MAX_NUMBER_SLOTS];

  unsigned long long int OutStandCnts   [XAIM_MAX_NUMBER_SLOTS];
  unsigned long long int LastWriteAddr  [XAIM_MAX_NUMBER_SLOTS];
  unsigned long long int LastWriteData  [XAIM_MAX_NUMBER_SLOTS];
  unsigned long long int LastReadAddr   [XAIM_MAX_NUMBER_SLOTS];
  unsigned long long int LastReadData   [XAIM_MAX_NUMBER_SLOTS];
  unsigned int           NumSlots;
  char                   DevUserName    [256];
} xclDebugCountersResults;

typedef struct {
  unsigned int           NumSlots ;
  char                   DevUserName    [256] ;

  unsigned long long int StrNumTranx    [XASM_MAX_NUMBER_SLOTS] ;
  unsigned long long int StrDataBytes   [XASM_MAX_NUMBER_SLOTS] ;
  unsigned long long int StrBusyCycles  [XASM_MAX_NUMBER_SLOTS] ;
  unsigned long long int StrStallCycles [XASM_MAX_NUMBER_SLOTS] ;
  unsigned long long int StrStarveCycles[XASM_MAX_NUMBER_SLOTS] ;
} xclStreamingDebugCountersResults ;

typedef struct {
  unsigned int           NumSlots ;
  char                   DevUserName    [256] ;

  unsigned long long CuExecCount        [XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuExecCycles       [XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuBusyCycles       [XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuMaxParallelIter  [XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuStallExtCycles   [XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuStallIntCycles   [XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuStallStrCycles   [XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuMinExecCycles    [XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuMaxExecCycles    [XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuStartCount       [XAM_MAX_NUMBER_SLOTS];
} xclAccelMonitorCounterResults;

typedef struct {
  unsigned int           Num;
  unsigned int           DeadlockStatus;
} xclAccelDeadlockDetectorResults;

enum xclCheckerType {
XCL_CHECKER_MEMORY = 0,
XCL_CHECKER_STREAM = 1
};

/* Debug checker results */
typedef struct {
  unsigned int   OverallStatus[XLAPC_MAX_NUMBER_SLOTS];
  unsigned int   CumulativeStatus[XLAPC_MAX_NUMBER_SLOTS][XLAPC_STATUS_REG_NUM];
  unsigned int   SnapshotStatus[XLAPC_MAX_NUMBER_SLOTS][XLAPC_STATUS_REG_NUM];
  unsigned int   NumSlots;
  char DevUserName[256];
} xclDebugCheckersResults;

typedef struct {
  unsigned int PCAsserted[XSPC_MAX_NUMBER_SLOTS];
  unsigned int CurrentPC [XSPC_MAX_NUMBER_SLOTS];
  unsigned int SnapshotPC[XSPC_MAX_NUMBER_SLOTS];
  unsigned int NumSlots;
  char DevUserName[256];
} xclDebugStreamingCheckersResults;

#ifdef __cplusplus
}
#endif
#endif
