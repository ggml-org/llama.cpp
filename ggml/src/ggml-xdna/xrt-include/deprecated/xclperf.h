/**
 * Copyright (C) 2016-2017 Xilinx, Inc
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
 * Copyright (C) 2015-2016, Xilinx Inc - All rights reserved
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

#ifndef _XCL_PERF_H_
#define _XCL_PERF_H_

// DSA version (e.g., XCL_PLATFORM=xilinx_adm-pcie-7v3_1ddr_1_1)
// Simply a default as its read from the device using lspci (see CR 870994)
#define DSA_MAJOR_VERSION 1
#define DSA_MINOR_VERSION 1

/************************ DEBUG IP LAYOUT ************************************/

#define IP_LAYOUT_HOST_NAME "HOST"
#define IP_LAYOUT_SEP "-"

/************************ APM 0: Monitor MIG Ports ****************************/

#define XPAR_AXI_PERF_MON_0_NUMBER_SLOTS                2

#define XPAR_AXI_PERF_MON_0_SLOT0_NAME                  "OCL Region"
#define XPAR_AXI_PERF_MON_0_SLOT1_NAME                  "Host"
#define XPAR_AXI_PERF_MON_0_OCL_REGION_SLOT             0
#define XPAR_AXI_PERF_MON_0_HOST_SLOT                   1

#define XPAR_AIM0_HOST_SLOT                             0
#define XPAR_AIM0_FIRST_KERNEL_SLOT                     1

#define XPAR_AXI_PERF_MON_0_OCL_REGION_SLOT2            2
#define XPAR_AXI_PERF_MON_0_OCL_REGION_SLOT3            3
#define XPAR_AXI_PERF_MON_0_OCL_REGION_SLOT4            4
#define XPAR_AXI_PERF_MON_0_OCL_REGION_SLOT5            5
#define XPAR_AXI_PERF_MON_0_OCL_REGION_SLOT6            6
#define XPAR_AXI_PERF_MON_0_OCL_REGION_SLOT7            7

#define XPAR_AXI_PERF_MON_0_SLOT2_NAME                  "OCL Region Master 2"
#define XPAR_AXI_PERF_MON_0_SLOT3_NAME                  "OCL Region Master 3"
#define XPAR_AXI_PERF_MON_0_SLOT4_NAME                  "OCL Region Master 4"
#define XPAR_AXI_PERF_MON_0_SLOT5_NAME                  "OCL Region Master 5"
#define XPAR_AXI_PERF_MON_0_SLOT6_NAME                  "OCL Region Master 6"
#define XPAR_AXI_PERF_MON_0_SLOT7_NAME                  "OCL Region Master 7"

#define XPAR_AXI_PERF_MON_0_SLOT0_DATA_WIDTH            512
#define XPAR_AXI_PERF_MON_0_SLOT1_DATA_WIDTH            512
#define XPAR_AXI_PERF_MON_0_SLOT2_DATA_WIDTH            512
#define XPAR_AXI_PERF_MON_0_SLOT3_DATA_WIDTH            512
#define XPAR_AXI_PERF_MON_0_SLOT4_DATA_WIDTH            512
#define XPAR_AXI_PERF_MON_0_SLOT5_DATA_WIDTH            512
#define XPAR_AXI_PERF_MON_0_SLOT6_DATA_WIDTH            512
#define XPAR_AXI_PERF_MON_0_SLOT7_DATA_WIDTH            512

/* Profile */
#define XPAR_AXI_PERF_MON_0_IS_EVENT_COUNT              1
#define XPAR_AXI_PERF_MON_0_HAVE_SAMPLED_COUNTERS       1
#define XPAR_AXI_PERF_MON_0_NUMBER_COUNTERS (XPAR_AXI_PERF_MON_0_NUMBER_SLOTS * XAPM_METRIC_COUNTERS_PER_SLOT)

/* Trace */
#define XPAR_AXI_PERF_MON_0_IS_EVENT_LOG                1
#define XPAR_AXI_PERF_MON_0_SHOW_AXI_IDS                1
#define XPAR_AXI_PERF_MON_0_SHOW_AXI_LEN                1
// 2 DDR platform
#define XPAR_AXI_PERF_MON_0_SHOW_AXI_IDS_2DDR           0
#define XPAR_AXI_PERF_MON_0_SHOW_AXI_LEN_2DDR           1

/* AXI Stream FIFOs */
#define XPAR_AXI_PERF_MON_0_TRACE_NUMBER_FIFO           3

#ifdef XRT_EDGE
#define XPAR_AXI_PERF_MON_0_TRACE_WORD_WIDTH            32
#else
#define XPAR_AXI_PERF_MON_0_TRACE_WORD_WIDTH            64
#endif

#define XPAR_AXI_PERF_MON_0_TRACE_NUMBER_SAMPLES        8192
#define MAX_TRACE_NUMBER_SAMPLES                        16384

#define XPAR_AXI_PERF_MON_0_TRACE_OFFSET_0              0x010000
#define XPAR_AXI_PERF_MON_0_TRACE_OFFSET_1              0x011000
#define XPAR_AXI_PERF_MON_0_TRACE_OFFSET_2              0x012000
// CR 877788: the extra 0x80001000 is a bug in Vivado where the AXI4 base address is not set correctly
// TODO: remove it once that bug is fixed!
//#define XPAR_AXI_PERF_MON_0_TRACE_OFFSET_AXI_FULL       (0x2000000000 + 0x80001000)
#define XPAR_AXI_PERF_MON_0_TRACE_OFFSET_AXI_FULL       0x2000000000
// Default for new monitoring
//#define XPAR_AXI_PERF_MON_0_TRACE_OFFSET_AXI_FULL2      (0x0400000000 + 0x80001000)
#define XPAR_AXI_PERF_MON_0_TRACE_OFFSET_AXI_FULL2      0x0400000000

/********************* APM 1: Monitor PCIe DMA Masters ************************/

#define XPAR_AXI_PERF_MON_1_NUMBER_SLOTS                2

#define XPAR_AXI_PERF_MON_1_SLOT0_NAME                  "DMA AXI4 Master"
#define XPAR_AXI_PERF_MON_1_SLOT1_NAME                  "DMA AXI4-Lite Master"
#define XPAR_AXI_PERF_MON_1_SLOT2_NAME                  "Null"
#define XPAR_AXI_PERF_MON_1_SLOT3_NAME                  "Null"
#define XPAR_AXI_PERF_MON_1_SLOT4_NAME                  "Null"
#define XPAR_AXI_PERF_MON_1_SLOT5_NAME                  "Null"
#define XPAR_AXI_PERF_MON_1_SLOT6_NAME                  "Null"
#define XPAR_AXI_PERF_MON_1_SLOT7_NAME                  "Null"

#define XPAR_AXI_PERF_MON_1_SLOT0_DATA_WIDTH            512
#define XPAR_AXI_PERF_MON_1_SLOT1_DATA_WIDTH            512
#define XPAR_AXI_PERF_MON_1_SLOT2_DATA_WIDTH            512
#define XPAR_AXI_PERF_MON_1_SLOT3_DATA_WIDTH            512
#define XPAR_AXI_PERF_MON_1_SLOT4_DATA_WIDTH            512
#define XPAR_AXI_PERF_MON_1_SLOT5_DATA_WIDTH            512
#define XPAR_AXI_PERF_MON_1_SLOT6_DATA_WIDTH            512
#define XPAR_AXI_PERF_MON_1_SLOT7_DATA_WIDTH            512

/* Profile */
#define XPAR_AXI_PERF_MON_1_IS_EVENT_COUNT              1
#define XPAR_AXI_PERF_MON_1_HAVE_SAMPLED_COUNTERS       1
#define XPAR_AXI_PERF_MON_1_NUMBER_COUNTERS (XPAR_AXI_PERF_MON_1_NUMBER_SLOTS * XAPM_METRIC_COUNTERS_PER_SLOT)
#define XPAR_AXI_PERF_MON_1_SCALE_FACTOR                1

/* Trace */
#define XPAR_AXI_PERF_MON_1_IS_EVENT_LOG                0
#define XPAR_AXI_PERF_MON_1_SHOW_AXI_IDS                0
#define XPAR_AXI_PERF_MON_1_SHOW_AXI_LEN                0

/* AXI Stream FIFOs */
#define XPAR_AXI_PERF_MON_1_TRACE_NUMBER_FIFO           0
#define XPAR_AXI_PERF_MON_1_TRACE_WORD_WIDTH            0
#define XPAR_AXI_PERF_MON_1_TRACE_NUMBER_SAMPLES        0

/************************ APM 2: Monitor OCL Region ***************************/

#define XPAR_AXI_PERF_MON_2_NUMBER_SLOTS                1

#define XPAR_AXI_PERF_MON_2_SLOT0_NAME                  "Kernel0"
#define XPAR_AXI_PERF_MON_2_SLOT1_NAME                  "Kernel1"
#define XPAR_AXI_PERF_MON_2_SLOT2_NAME                  "Kernel2"
#define XPAR_AXI_PERF_MON_2_SLOT3_NAME                  "Kernel3"
#define XPAR_AXI_PERF_MON_2_SLOT4_NAME                  "Kernel4"
#define XPAR_AXI_PERF_MON_2_SLOT5_NAME                  "Kernel5"
#define XPAR_AXI_PERF_MON_2_SLOT6_NAME                  "Kernel6"
#define XPAR_AXI_PERF_MON_2_SLOT7_NAME                  "Kernel7"

#define XPAR_AXI_PERF_MON_2_SLOT0_DATA_WIDTH            512
#define XPAR_AXI_PERF_MON_2_SLOT1_DATA_WIDTH            512
#define XPAR_AXI_PERF_MON_2_SLOT2_DATA_WIDTH            512
#define XPAR_AXI_PERF_MON_2_SLOT3_DATA_WIDTH            512
#define XPAR_AXI_PERF_MON_2_SLOT4_DATA_WIDTH            512
#define XPAR_AXI_PERF_MON_2_SLOT5_DATA_WIDTH            512
#define XPAR_AXI_PERF_MON_2_SLOT6_DATA_WIDTH            512
#define XPAR_AXI_PERF_MON_2_SLOT7_DATA_WIDTH            512

/* Profile */
#define XPAR_AXI_PERF_MON_2_IS_EVENT_COUNT              0
#define XPAR_AXI_PERF_MON_2_HAVE_SAMPLED_COUNTERS       0
#define XPAR_AXI_PERF_MON_2_NUMBER_COUNTERS             0
#define XPAR_AXI_PERF_MON_2_SCALE_FACTOR                1

/* Trace */
#define XPAR_AXI_PERF_MON_2_IS_EVENT_LOG                1
#define XPAR_AXI_PERF_MON_2_SHOW_AXI_IDS                0
#define XPAR_AXI_PERF_MON_2_SHOW_AXI_LEN                0

/* AXI Stream FIFOs */
/* NOTE: number of FIFOs is dependent upon the number of compute units being monitored */
//#define XPAR_AXI_PERF_MON_2_TRACE_NUMBER_FIFO           2
#define XPAR_AXI_PERF_MON_2_TRACE_WORD_WIDTH            64
#define XPAR_AXI_PERF_MON_2_TRACE_NUMBER_SAMPLES        8192

#define XPAR_AXI_PERF_MON_2_TRACE_OFFSET_0              0x01000
#define XPAR_AXI_PERF_MON_2_TRACE_OFFSET_1              0x02000
#define XPAR_AXI_PERF_MON_2_TRACE_OFFSET_2              0x03000

/************************ APM Profile Counters ********************************/

#define XAPM_MAX_NUMBER_SLOTS             8
// Max slots = floor(max slots on trace funnel / 2) = floor(63 / 2) = 31
// NOTE: AIM max slots += 3 to support XDMA/KDMA/P2P monitors on some 2018.3 platforms
#define XAIM_MAX_NUMBER_SLOTS             34
#define XAM_MAX_NUMBER_SLOTS             31
#define XASM_MAX_NUMBER_SLOTS            31
#define XAPM_METRIC_COUNTERS_PER_SLOT     8

/* Metric counters per slot */
#define XAPM_METRIC_WRITE_BYTES           0
#define XAPM_METRIC_WRITE_TRANX           1
#define XAPM_METRIC_WRITE_LATENCY         2
#define XAPM_METRIC_READ_BYTES            3
#define XAPM_METRIC_READ_TRANX            4
#define XAPM_METRIC_READ_LATENCY          5
#define XAPM_METRIC_WRITE_MIN_MAX         6
#define XAPM_METRIC_READ_MIN_MAX          7

#define XAPM_METRIC_COUNT0_NAME           "Write Byte Count"
#define XAPM_METRIC_COUNT1_NAME           "Write Transaction Count"
#define XAPM_METRIC_COUNT2_NAME           "Total Write Latency"
#define XAPM_METRIC_COUNT3_NAME           "Read Byte Count"
#define XAPM_METRIC_COUNT4_NAME           "Read Transaction Count"
#define XAPM_METRIC_COUNT5_NAME           "Total Read Latency"
#define XAPM_METRIC_COUNT6_NAME           "Min/Max Write Latency"
#define XAPM_METRIC_COUNT7_NAME           "Min/Max Read Latency"

/************************ APM Debug Counters ********************************/
#define XAPM_DEBUG_METRIC_COUNTERS_PER_SLOT     4  //debug is only interested in 4 metric counters

/************************ APM Trace Stream ************************************/

/************************ Trace IDs ************************************/

#define MIN_TRACE_ID_AIM        0
#define MAX_TRACE_ID_AIM        61
#define MIN_TRACE_ID_AM        64
#define MAX_TRACE_ID_AM        544
#define MAX_TRACE_ID_AM_HWEM   94
#define MIN_TRACE_ID_ASM       576
#define MAX_TRACE_ID_ASM       607

/* Bit locations of trace flags */
#define XAPM_READ_LAST                   6
#define XAPM_READ_FIRST                  5
#define XAPM_READ_ADDR                   4
#define XAPM_RESPONSE                    3
#define XAPM_WRITE_LAST                  2
#define XAPM_WRITE_FIRST                 1
#define XAPM_WRITE_ADDR                  0

/* Bit locations of external event flags */
#define XAPM_EXT_START                   2
#define XAPM_EXT_STOP                    1
#define XAPM_EXT_EVENT                   0

/* Total number of bits per slot */
#define FLAGS_PER_SLOT                   7
#define EXT_EVENTS_PER_SLOT              3

/* Cycles to add to timestamp if overflow occurs */
#define LOOP_ADD_TIME                    (1<<16)
#define LOOP_ADD_TIME_AIM                (1ULL<<44)

/********************** Definitions: Enums, Structs ***************************/

/* Performance monitor type or location */
enum xclPerfMonType {
	XCL_PERF_MON_MEMORY = 0,
	XCL_PERF_MON_HOST   = 1,
	XCL_PERF_MON_SHELL  = 2,
	XCL_PERF_MON_ACCEL  = 3,
	XCL_PERF_MON_STALL  = 4,
	XCL_PERF_MON_STR    = 5,
	XCL_PERF_MON_FIFO   = 6,
  XCL_PERF_MON_NOC    = 7,
	XCL_PERF_MON_TOTAL_PROFILE = 8
};

/* Performance monitor start event */
enum xclPerfMonStartEvent {
	XCL_PERF_MON_START_ADDR = 0,
	XCL_PERF_MON_START_FIRST_DATA = 1
};

/* Performance monitor end event */
enum xclPerfMonEndEvent {
	XCL_PERF_MON_END_LAST_DATA = 0,
	XCL_PERF_MON_END_RESPONSE = 1
};

/* Performance monitor counter types */
enum xclPerfMonCounterType {
  XCL_PERF_MON_WRITE_BYTES = 0,
  XCL_PERF_MON_WRITE_TRANX = 1,
  XCL_PERF_MON_WRITE_LATENCY = 2,
  XCL_PERF_MON_READ_BYTES = 3,
  XCL_PERF_MON_READ_TRANX = 4,
  XCL_PERF_MON_READ_LATENCY = 5
};

/*
 * Performance monitor event types
 * NOTE: these are the same values used by Zynq
 */
enum xclPerfMonEventType {
  XCL_PERF_MON_START_EVENT = 0x4,
  XCL_PERF_MON_END_EVENT = 0x5
};

/*
 * Xocc follows this convention
 * Even IDs are Reads
 * Odd IDs are Writes
 */
#define IS_WRITE(x) ((x) & 1)
#define IS_READ(x) (!((x) & 1))

#define XAM_TRACE_CU_MASK         0x1
#define XAM_TRACE_STALL_INT_MASK  0x2
#define XAM_TRACE_STALL_STR_MASK  0x4
#define XAM_TRACE_STALL_EXT_MASK  0x8

/*
 * Performance monitor IDs for host SW events
 * NOTE: HW events start at 0, Zynq SW events start at 4000
 */
enum xclPerfMonEventID {
  XCL_PERF_MON_HW_EVENT = 0,
  XCL_PERF_MON_GENERAL_ID = 3000,
  XCL_PERF_MON_QUEUE_ID = 3001,
  XCL_PERF_MON_READ_ID = 3002,
  XCL_PERF_MON_WRITE_ID = 3003,
  XCL_PERF_MON_API_GET_PLATFORM_ID = 3005,
  XCL_PERF_MON_API_GET_PLATFORM_INFO_ID = 3006,
  XCL_PERF_MON_API_GET_DEVICE_ID = 3007,
  XCL_PERF_MON_API_GET_DEVICE_INFO_ID = 3008,
  XCL_PERF_MON_API_BUILD_PROGRAM_ID = 3009,
  XCL_PERF_MON_API_CREATE_CONTEXT_ID = 3010,
  XCL_PERF_MON_API_CREATE_CONTEXT_TYPE_ID = 3011,
  XCL_PERF_MON_API_CREATE_COMMAND_QUEUE_ID = 3012,
  XCL_PERF_MON_API_CREATE_PROGRAM_BINARY_ID = 3013,
  XCL_PERF_MON_API_CREATE_BUFFER_ID = 3014,
  XCL_PERF_MON_API_CREATE_IMAGE_ID = 3015,
  XCL_PERF_MON_API_CREATE_KERNEL_ID = 3016,
  XCL_PERF_MON_API_KERNEL_ARG_ID = 3017,
  XCL_PERF_MON_API_WAIT_FOR_EVENTS_ID = 3018,
  XCL_PERF_MON_API_READ_BUFFER_ID = 3019,
  XCL_PERF_MON_API_WRITE_BUFFER_ID = 3020,
  XCL_PERF_MON_API_READ_IMAGE_ID = 3021,
  XCL_PERF_MON_API_WRITE_IMAGE_ID = 3022,
  XCL_PERF_MON_API_MIGRATE_MEM_ID = 3023,
  XCL_PERF_MON_API_MIGRATE_MEM_OBJECTS_ID = 3024,
  XCL_PERF_MON_API_MAP_BUFFER_ID = 3025,
  XCL_PERF_MON_API_UNMAP_MEM_OBJECT_ID = 3026,
  XCL_PERF_MON_API_NDRANGE_KERNEL_ID = 3027,
  XCL_PERF_MON_API_TASK_ID = 3028,
  XCL_PERF_MON_KERNEL0_ID = 3100,
  XCL_PERF_MON_KERNEL1_ID = 3101,
  XCL_PERF_MON_KERNEL2_ID = 3102,
  XCL_PERF_MON_KERNEL3_ID = 3103,
  XCL_PERF_MON_KERNEL4_ID = 3104,
  XCL_PERF_MON_KERNEL5_ID = 3105,
  XCL_PERF_MON_KERNEL6_ID = 3106,
  XCL_PERF_MON_KERNEL7_ID = 3107,
  XCL_PERF_MON_CU0_ID = 3200,
  XCL_PERF_MON_CU1_ID = 3201,
  XCL_PERF_MON_CU2_ID = 3202,
  XCL_PERF_MON_CU3_ID = 3203,
  XCL_PERF_MON_CU4_ID = 3204,
  XCL_PERF_MON_CU5_ID = 3205,
  XCL_PERF_MON_CU6_ID = 3206,
  XCL_PERF_MON_CU7_ID = 3207,
  XCL_PERF_MON_PROGRAM_END = 4090,
  XCL_PERF_MON_IGNORE_EVENT = 4095
};

/* Performance monitor counter results */
typedef struct {
  float              SampleIntervalUsec;
  unsigned long long WriteBytes[XAIM_MAX_NUMBER_SLOTS];
  unsigned long long WriteTranx[XAIM_MAX_NUMBER_SLOTS];
  unsigned long long WriteLatency[XAIM_MAX_NUMBER_SLOTS];
  unsigned short     WriteMinLatency[XAIM_MAX_NUMBER_SLOTS];
  unsigned short     WriteMaxLatency[XAIM_MAX_NUMBER_SLOTS];
  unsigned long long ReadBytes[XAIM_MAX_NUMBER_SLOTS];
  unsigned long long ReadTranx[XAIM_MAX_NUMBER_SLOTS];
  unsigned long long ReadLatency[XAIM_MAX_NUMBER_SLOTS];
  unsigned short     ReadMinLatency[XAIM_MAX_NUMBER_SLOTS];
  unsigned short     ReadMaxLatency[XAIM_MAX_NUMBER_SLOTS];
  unsigned long long ReadBusyCycles[XAIM_MAX_NUMBER_SLOTS];
  unsigned long long WriteBusyCycles[XAIM_MAX_NUMBER_SLOTS];
  // Accelerator Monitor
  unsigned long long CuExecCount[XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuExecCycles[XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuBusyCycles[XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuMaxParallelIter[XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuStallExtCycles[XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuStallIntCycles[XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuStallStrCycles[XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuMinExecCycles[XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuMaxExecCycles[XAM_MAX_NUMBER_SLOTS];
  // AXI Stream Monitor
  unsigned long long StrNumTranx[XASM_MAX_NUMBER_SLOTS];
  unsigned long long StrDataBytes[XASM_MAX_NUMBER_SLOTS];
  unsigned long long StrBusyCycles[XASM_MAX_NUMBER_SLOTS];
  unsigned long long StrStallCycles[XASM_MAX_NUMBER_SLOTS];
  unsigned long long StrStarveCycles[XASM_MAX_NUMBER_SLOTS];
} xclCounterResults;

/* Performance monitor trace results */
typedef struct {
  enum xclPerfMonEventID EventID;
  enum xclPerfMonEventType EventType;
  unsigned long long Timestamp;
  unsigned char  Overflow;
  unsigned int TraceID;
  unsigned char Error;
  unsigned char Reserved;
  int isClockTrain;
  // Used in HW Emulation
  unsigned long long  HostTimestamp;
  unsigned char  EventFlags;
  unsigned char  WriteAddrLen;
  unsigned char  ReadAddrLen;
  unsigned short WriteBytes;
  unsigned short ReadBytes;
} xclTraceResults;

typedef struct {
  unsigned int mLength;
  //unsigned int mNumSlots;
  xclTraceResults mArray[MAX_TRACE_NUMBER_SAMPLES];
} xclTraceResultsVector;

#define DRIVER_NAME_ROOT "/dev"
#define DEVICE_PREFIX "/dri/renderD"
#define NIFD_PREFIX "/nifd"
#define SYSFS_NAME_ROOT "/sys/bus/pci/devices/"
#define MAX_NAME_LEN 256

enum DeviceType {
  SW_EMU = 0,
  HW_EMU = 1,
  XBB = 2,
  AWS = 3
};

/**
 * \brief data structure for querying device info
 * 
 * TODO:
 * all the data for nifd won't be avaiable until nifd
 * driver is merged and scan.h is changed to recognize
 * nifd driver.
 */
typedef struct {
  enum DeviceType device_type;
  unsigned int device_index;
  unsigned int user_instance;
  unsigned int nifd_instance;
  char device_name[MAX_NAME_LEN];
  char nifd_name[MAX_NAME_LEN];
} xclDebugProfileDeviceInfo;

/**
 * hal level xdp plugin types
 * 
 * The data structures for hal level xdp plugins
 * that will be interpreted by both the shim and
 * xdp
 * 
 * custom plugin requirements:
 *  1. has a method called hal_level_xdp_cb_func
 *      that takes a enum type and a void pointer
 *      payload which can be cast to one of the
 *      structs listed below.
 *  2. config through initialization by setting the
 *      plugin path attribute to the dynamic library.
 */ 

/**
   HAL API Profiling Interface plugin types
   
   The data structure for enabling the HAL API Profiling 
   plugins that will be interpreted by both the shim and
   xdp.

   custom plugin requirements:
    1. Has an exported method called hal_api_interface_cb_func
       that takes an enum type and a void pointer payload
       which can be cast to one of the structs below.
    2. config through initialization by setting the 
        plugin path attribute to the dynamic library
 */

struct HalPluginConfig {
  int state; /** < [unused] indicates if on or off */
  char plugin_path[256]; /** < [unused] indicates which dynamic library to load */
  /**
   * The switches for what to profile and what
   * not to should go here. The attibutes will
   * be added once settle down on what kind of
   * swtiches make sense for the plugins.
   */
};

// Used in the HAL API Interface to access hardware counters in host code
enum HalInterfaceCallbackType {
  START_DEVICE_PROFILING,
  CREATE_PROFILE_RESULTS,
  GET_PROFILE_RESULTS,
  DESTROY_PROFILE_RESULTS
} ;

#ifdef __cplusplus
#include <cstdint>
#include <cstddef>
#else
#include <stdlib.h>
#include <stdint.h>
#endif

/**
 * This is an example of the struct that callback
 * functions can take. Eventually, different API
 * callbacks are likely to take different structs.
 */
typedef struct CBPayload {
  uint64_t idcode;
  void* deviceHandle;
} CBPayload;

/**
 * More callback payload struct should be declared 
 * here for the users to include.
 */

struct BOTransferCBPayload
{
  struct CBPayload basePayload ;
  uint64_t bufferTransferId;
  size_t   size;
#if 0
  uint32_t boHandle;
  uint64_t dest;
  size_t offset;
#endif
};

struct SyncBOCBPayload
{
  struct CBPayload basePayload ;
  uint64_t bufferTransferId;
  size_t   size;
  bool     isWriteToDevice;
#if 0
  uint32_t boHandle;
  size_t offset;
#endif
};

struct ReadWriteCBPayload
{
  struct CBPayload basePayload;
  size_t   size;
#if 0
  uint32_t addressSpace;
  uint64_t offset;
#endif
};

struct UnmgdPreadPwriteCBPayload
{
  struct CBPayload basePayload;
  unsigned int flags;
  size_t   count;
  uint64_t offset;
};

struct XclbinCBPayload
{
  struct CBPayload basePayload ;
  const void* binary ;
};

struct ProfileResultsCBPayload
{
  struct CBPayload basePayload;
  void* results;
};

/**
 * end hal level xdp plugin types
 */


#endif
