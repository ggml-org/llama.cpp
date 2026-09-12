/*
 * Copyright (C) 2019-2020, Xilinx Inc - All rights reserved
 * Xilinx Runtime (XRT) Experimental APIs
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

#ifndef _XRT_NEXT_APIS_H_
#define _XRT_NEXT_APIS_H_

#include "xrt/deprecated/xrt.h"

/*
 * typedef xclInterruptNotifyHandle
 *
 * Implementation specific type representing an interrupt notify handle
 * that is used by xclOpen/CloseIPInterruptNotify()
 */  
#ifdef _WIN32
typedef void* xclInterruptNotifyHandle;
# define XCL_NULL_INTC_HANDLE INVALID_HANDLE_VALUE
#else
typedef int xclInterruptNotifyHandle;
# define XCL_NULL_INTC_HANDLE -1
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Place holder for experimental APIs
 */

/**
 * xclP2pEnable() - enable or disable p2p
 *
 * @handle:        Device handle
 * @enable:        false-disable, true-enable
 * @force:         true-force to reassign bus IO memory
 * Return:         0 on success or appropriate error number
 *
 * Enable or Disable P2P feature. Warm reboot might be required.
 */
XCL_DRIVER_DLLESPEC
int
xclP2pEnable(xclDeviceHandle handle, bool enable, bool force);

/*
 * API to get number of live processes on the given device.
 * This uses kdsstat information in sysfs.
 */
XCL_DRIVER_DLLESPEC
uint32_t
xclGetNumLiveProcesses(xclDeviceHandle handle);

/**
 * xclGetSysfsPath() - Helper function to build a sysfs node full path
 *
 * @handle:              Device handle
 * @subdev:              Sub-device name
 * @entry:               Name of sysfs node
 * @sysfsPath:           Return string with full path for sysfs node
 * @size:                Length of the return string
 * Return:               0 or standard error number
 *
 * (For debug and profile usage only for now) The sysfs information is
 * not accessible above XRT layer now. However, debug/profile needs
 * information from sysfs (for example debug_ip_layout) to properly
 * initialize xdp code, so this helper API can be used
 */
XCL_DRIVER_DLLESPEC
int
xclGetSysfsPath(xclDeviceHandle handle, const char* subdev,
                const char* entry, char* sysfsPath, size_t size);

struct KernelTransferData
{
   char* cuPortName;
   char* argName;
   char* memoryName;

   uint64_t totalReadBytes;
   uint64_t totalReadTranx;
   uint64_t totalReadLatency;
   uint64_t totalReadBusyCycles;
   uint64_t minReadLatency;
   uint64_t maxReadLatency;

   uint64_t totalWriteBytes;
   uint64_t totalWriteTranx;
   uint64_t totalWriteLatency;
   uint64_t totalWriteBusyCycles;
   uint64_t minWriteLatency;
   uint64_t maxWriteLatency;
};

struct CuExecData
{
   char* cuName;
   char* kernelName;

   uint64_t cuExecCount;
   uint64_t cuExecCycles;
   uint64_t cuBusyCycles;
   uint64_t cuMaxExecCycles;
   uint64_t cuMinExecCycles;
   uint64_t cuMaxParallelIter;
   uint64_t cuStallExtCycles;
   uint64_t cuStallIntCycles;
   uint64_t cuStallStrCycles;

};

struct StreamTransferData
{
  char* masterPortName;
  char* slavePortName;

  uint64_t strmNumTranx;
  uint64_t strmBusyCycles;
  uint64_t strmDataBytes;
  uint64_t strmStallCycles;
  uint64_t strmStarveCycles;
};

struct ProfileResults
{
  char* deviceName;

  uint64_t numAIM;
  struct KernelTransferData *kernelTransferData;

  uint64_t numAM;
  struct CuExecData *cuExecData;

  uint64_t numASM;
  struct StreamTransferData  *streamData;
};

/**
 * int xclCreateProfileResults(xclDeviceHandle, ProfileResults**)
 *
 *  - Creates and initializes buffer for storing the ProfileResults from
 *    the device.
 *  - To use this API, "profile_api" configuration needs to be set to
 *    "true" in xrt.ini.  "profile_api=true" enables profiling (for
 *    profile counters on device) using XRT APIs (no OpenCL API)
 *
 * @xclDeviceHandle : Device handle
 * @ProfileResults  : Double pointer to hold the pointer to buffer created
 *                    to store profiling results (on success). This argument
 *                    remains unchanged if xclCreateProfileResults fails.
 */
XCL_DRIVER_DLLESPEC
int
xclCreateProfileResults(xclDeviceHandle, struct ProfileResults**);

/**
 * int xclGetProfileResults(xclDeviceHandle, ProfileResults*)
 *
 *  - Read the profiling counters from the hardware and populate
 *     profiling data in the “ProfileResults” structure (created using
 *     xclCreateProfileResults).
 *  - To use this API, "profile_api" configuration needs to be set to
 *    "true" in xrt.ini.  "profile_api=true" enables profiling (for
 *    profile counters on device) using XRT APIs (no OpenCL API)
 *
 * @xclDeviceHandle : Device handle
 * @ProfileResults* : Pointer to buffer to store profiling results.
 *                    This buffer should be created using previous
 *                    call to "xclCreateProfileResults"
 */
XCL_DRIVER_DLLESPEC
int
xclGetProfileResults(xclDeviceHandle, struct ProfileResults*);

/**
 * int xclDestroyProfileResults(xclDeviceHandle, ProfileResults*)
 *
 *  - Traverse the given ProfileResults structure and delete the
 *    allocated memory
 *  - To use this API, "profile_api" configuration needs to be set to
 *    "true" in xrt.ini.  "profile_api=true" enables profiling (for
 *     profile counters on device) using XRT APIs (no OpenCL API)
 *
 * @xclDeviceHandle : Device handle
 * @ProfileResults  : Pointer to buffer to be deleted
 */
XCL_DRIVER_DLLESPEC
int
xclDestroyProfileResults(xclDeviceHandle, struct ProfileResults*);

/**
 * xclRegRead() - Read register in register space of an IP
 *
 * @handle:        Device handle
 * @ipIndex:       IP index
 * @offset:        Offset in the register space
 * @datap:         Pointer to where result will be saved
 * Return:         0 or appropriate error number
 *
 * Caller should own an exclusive context on the IP obtained via
 * xclOpenContext() This API may be used to read from device registers.
 * Offset is relative to the the register map exposed by the IP.
 * Please note that the intent of this API is to support legacy RTL IPs.
 * If you are creating a new IP please use one of the well defined
 * execution models which are natively supported by XRT.
 * If you have a new execution model that you would like to be natively
 * supported by XRT please look into ert.h to define new command objects.
 */
XCL_DRIVER_DLLESPEC
int
xclRegRead(xclDeviceHandle handle, uint32_t ipIndex, uint32_t offset, uint32_t *datap);

/**
 * xclRegWRite() - Write to register in register space of an IP
 *
 * @handle:        Device handle
 * @ipIndex:       IP index
 * @offset:        Offset in the register space
 * @data:          Data to be written
 * Return:         0 or appropriate error number
 *
 * Caller should own an exclusive context on the CU obtained via
 * xclOpenContext() This API may be used to write to device registers.
 * Offset is relative to the the register map exposed by the CU.
 * Please note that the intent of this API is to support legacy RTL IPs.
 * If you are creating a new IP please use one of the well defined
 * execution models which are natively supported by XRT.
 * If you have a new execution model that you would like to be natively
 * supported by XRT please look into ert.h to define new command objects.
 */
XCL_DRIVER_DLLESPEC
int
xclRegWrite(xclDeviceHandle handle, uint32_t ipIndex, uint32_t offset, uint32_t data);

/**
 * xclOpenIPInterruptNotify() - Open a fd for IP interrupt notify
 *
 * @handle:     Device handle
 * @ipIndex:    IP index
 * @flags:      flags for the fd
 * Return:      Interrupt notify handle
 *
 * Caller should own an exclusive context on the IP obtained via
 * xclOpenContext()
 *
 * Support for non managed IP. This is the proper way to support
 * custom IPs which doesn't compliant with supported control protocol.
 *
 * This API would open a handle used for CU interrupt
 * notification. The usage is similar to open a UIO device.  Caller
 * could use standard read/poll/select system call to wait for IP
 * interrupts.  Once this API was called, xclExecBuf() could not be
 * used to schedule this specific IP.  The expectation is the caller
 * performing any necessary actions to make it work.
 *
 * Note: the IP irq would be disable after this is called. Caller
 * could manually enable the interrupt by write().
 */
XCL_DRIVER_DLLESPEC
xclInterruptNotifyHandle
xclOpenIPInterruptNotify(xclDeviceHandle handle, uint32_t ipIndex, unsigned int flags);

/**
 * xclCloseIPInterruptNotify() - Clost the interrupt notify fd
 *
 * @handle:     Device handle
 * @fd:         fd handle
 * Return:      0 or appropriate error number
 *
 * Caller should own an exclusive context on the IP obtained via xclOpenContext()
 * This API would close the file descriptor used for IP interrupt notification.
 * Once this API was called, xclExecBuf() could schedule this specific IP.
 */
XCL_DRIVER_DLLESPEC
int
xclCloseIPInterruptNotify(xclDeviceHandle handle, int fd);

/**
 * xclErrorClear() - Clear all asynchronous error records in driver
 *
 * @handle:     Device handle
 * Return:      0 or appropriate error number
 *
 * Note: this API needs root privilege
 */
XCL_DRIVER_DLLESPEC
int
xclErrorClear(xclDeviceHandle handle);

#ifdef __cplusplus
}
#endif

#endif
