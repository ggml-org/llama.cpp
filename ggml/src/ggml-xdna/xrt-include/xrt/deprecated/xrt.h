/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2020-2022, Xilinx Inc. All rights reserved
 * Copyright (C) 2022-2023, Advanced Micro Devices, Inc. All rights reserved.
 */
#ifndef _XCL_XRT_DEPRECATED_H_
#define _XCL_XRT_DEPRECATED_H_

/* This header file is included from include.xrt.h, it is not
 * Not a stand-alone header file
 *
 * All APIs in this file will be removed in later XRT releases.
 */

#ifdef __cplusplus
#include <cstdlib>
#include <cstdint>
#else
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#endif

#if defined (_WIN32)
#define NOMINMAX
#include <windows.h>
#include "xrt/detail/windows/types.h"
#endif

#include "xrt/detail/xclbin.h"
#include "xrt/deprecated/xclerr.h"
//#include "xclhal2_mem.h"

//#include "deprecated/xcl_app_debug.h"

#ifdef __GNUC__
# define XRT_DEPRECATED __attribute__ ((deprecated))
#else
# define XRT_DEPRECATED
#endif

#if defined(_WIN32)
# ifndef XRT_STATIC_BUILD
#  ifdef XCL_DRIVER_DLL_EXPORT
#   define XCL_DRIVER_DLLESPEC __declspec(dllexport)
#  else
#   define XCL_DRIVER_DLLESPEC __declspec(dllimport)
#  endif
# endif
# define XCL_DRIVER_DLLHIDDEN
#endif

#ifdef __linux__
# define XCL_DRIVER_DLLESPEC __attribute__((visibility("default")))
# define XCL_DRIVER_DLLHIDDEN __attribute__((visibility("hidden")))
#endif

#ifndef XCL_DRIVER_DLLESPEC
# define XCL_DRIVER_DLLESPEC
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * typedef xclDeviceHandle - opaque device handle
 *
 * A device handle of xclDeviceHandle kind is obtained by opening a
 * device. Clients pass this device handle to refer to the opened
 * device in all future interaction with XRT.
 */
typedef void * xclDeviceHandle;
#define XRT_NULL_HANDLE NULL

/*
 * typedef xclBufferHandle - opaque buffer handle
 *
 * A buffer handle of xclBufferHandle kind is obtained by allocating buffer
 * objects through HAL API. The buffer handle is used by XRT HAL APIs that
 * operate on on buffer objects.
 */
#ifdef _WIN32
typedef void * xclBufferHandle;
# define NULLBO	INVALID_HANDLE_VALUE
#else
typedef unsigned int xclBufferHandle;
# define NULLBO	0xffffffff
#endif
#define XRT_NULL_BO NULLBO

/*
 * typedef xclBufferExportHandle
 *
 * Implementation specific type representing an exported buffer handle
 * that can be passed between processes.
 */
#ifdef _WIN32
typedef uint64_t xclBufferExportHandle;  // TBD
#define NULLBOEXPORT -1
#else
typedef int32_t xclBufferExportHandle;
#define NULLBOEXPORT -1
#endif
#define XRT_NULL_BO_EXPORT NULLBOEXPORT

struct axlf;

/*
 * Structure used to obtain various bits of information from the device.
 */
struct xclDeviceInfo2 {
  unsigned int mMagic; // = 0X586C0C6C; XL OpenCL X->58(ASCII), L->6C(ASCII), O->0 C->C L->6C(ASCII);
  char mName[256];
  unsigned short mHALMajorVersion;
  unsigned short mHALMinorVersion;
  unsigned short mVendorId;
  unsigned short mDeviceId;
  unsigned short mSubsystemId;
  unsigned short mSubsystemVendorId;
  unsigned short mDeviceVersion;
  size_t mDDRSize;                    // Size of DDR memory
  size_t mDataAlignment;              // Minimum data alignment requirement for host buffers
  size_t mDDRFreeSize;                // Total unused/available DDR memory
  size_t mMinTransferSize;            // Minimum DMA buffer size
  unsigned short mDDRBankCount;
  unsigned short mOCLFrequency[4];
  unsigned short mPCIeLinkWidth;
  unsigned short mPCIeLinkSpeed;
  unsigned short mDMAThreads;
  unsigned short mOnChipTemp;
  unsigned short mFanTemp;
  unsigned short mVInt;
  unsigned short mVAux;
  unsigned short mVBram;
  float mCurrent;
  unsigned short mNumClocks;
  unsigned short mFanSpeed;
  bool mMigCalib;
  unsigned long long mXMCVersion;
  unsigned long long mMBVersion;
  unsigned short m12VPex;
  unsigned short m12VAux;
  unsigned long long mPexCurr;
  unsigned long long mAuxCurr;
  unsigned short mFanRpm;
  unsigned short mDimmTemp[4];
  unsigned short mSE98Temp[4];
  unsigned short m3v3Pex;
  unsigned short m3v3Aux;
  unsigned short mDDRVppBottom;
  unsigned short mDDRVppTop;
  unsigned short mSys5v5;
  unsigned short m1v2Top;
  unsigned short m1v8Top;
  unsigned short m0v85;
  unsigned short mMgt0v9;
  unsigned short m12vSW;
  unsigned short mMgtVtt;
  unsigned short m1v2Bottom;
  unsigned long long mDriverVersion;
  unsigned int mPciSlot;
  bool mIsXPR;
  unsigned long long mTimeStamp;
  char mFpga[256];
  unsigned short mPCIeLinkWidthMax;
  unsigned short mPCIeLinkSpeedMax;
  unsigned short mVccIntVol;
  unsigned short mVccIntCurr;
  unsigned short mNumCDMA;
};

/*
 *  Unused, keep for backwards compatibility
 */
enum xclBOKind {
    XCL_BO_SHARED_VIRTUAL = 0,
    XCL_BO_SHARED_PHYSICAL,
    XCL_BO_MIRRORED_VIRTUAL,
    XCL_BO_DEVICE_RAM,
    XCL_BO_DEVICE_BRAM,
    XCL_BO_DEVICE_PREALLOCATED_BRAM,
};

enum xclBOSyncDirection {
    XCL_BO_SYNC_BO_TO_DEVICE = 0,
    XCL_BO_SYNC_BO_FROM_DEVICE,
    XCL_BO_SYNC_BO_GMIO_TO_AIE,
    XCL_BO_SYNC_BO_AIE_TO_GMIO,
};

/*
 * Define address spaces on the device AXI bus. The enums are used in
 * xclRead() and xclWrite() to pass relative offsets.
 */
enum xclAddressSpace {
    XCL_ADDR_SPACE_DEVICE_FLAT = 0,     // Absolute address space
    XCL_ADDR_SPACE_DEVICE_RAM = 1,      // Address space for the DDR memory
    XCL_ADDR_KERNEL_CTRL = 2,           // Address space for the OCL Region control port
    XCL_ADDR_SPACE_DEVICE_PERFMON = 3,  // Address space for the Performance monitors
    XCL_ADDR_SPACE_DEVICE_REG     = 4,  // Address space for device registers.
    XCL_ADDR_SPACE_DEVICE_CHECKER = 5,  // Address space for protocol checker
    XCL_ADDR_SPACE_MAX = 8
};

/*
 * Defines log message severity levels for messages sent to log file
 * with xclLogMsg cmd
 */
enum xrtLogMsgLevel {
     XRT_EMERGENCY = 0,
     XRT_ALERT = 1,
     XRT_CRITICAL = 2,
     XRT_ERROR = 3,
     XRT_WARNING = 4,
     XRT_NOTICE = 5,
     XRT_INFO = 6,
     XRT_DEBUG = 7
};

/*
 * Defines verbosity levels which are passed to xclOpen during device
 * creation time
 */
enum xclVerbosityLevel {
    XCL_QUIET = 0,
    XCL_INFO = 1,
    XCL_WARN = 2,
    XCL_ERROR = 3
};

enum xclResetKind {
    XCL_RESET_KERNEL, // not implemented through xocl user pf
    XCL_RESET_FULL,   // not implemented through xocl user pf
    XCL_USER_RESET
};

#define XCL_DEVICE_USAGE_COUNT 8
struct xclDeviceUsage {
    size_t h2c[XCL_DEVICE_USAGE_COUNT];
    size_t c2h[XCL_DEVICE_USAGE_COUNT];
    size_t ddrMemUsed[XCL_DEVICE_USAGE_COUNT];
    unsigned int ddrBOAllocated[XCL_DEVICE_USAGE_COUNT];
    unsigned int totalContexts;
    uint64_t xclbinId[4];
    unsigned int dma_channel_cnt;
    unsigned int mm_channel_cnt;
    uint64_t memSize[XCL_DEVICE_USAGE_COUNT];
};

struct xclBOProperties {
    uint32_t handle;
    uint32_t flags;
    uint64_t size;
    uint64_t paddr;
    int reserved; // not implemented
};

/*
 * xclProbe() - Enumerate devices found in the system
 *
 * Return: count of devices found
 */
XCL_DRIVER_DLLHIDDEN
unsigned int
xclProbe();

/*
 * xclOpen() - Open a device and obtain its handle.
 *
 * @deviceIndex:   Slot number of device 0 for first device, 1 for the second device...
 * @logFileName:   Unused, logging is controlled via xrt.ini
 * @level:         Unused, verbosity is controlled via xrt.ini
 *
 * Return:         Device handle
 */
XCL_DRIVER_DLLHIDDEN
xclDeviceHandle
xclOpen(unsigned int deviceIndex, const char* unused1, enum xclVerbosityLevel unused2);

/*
 * xclClose() - Close an opened device
 *
 * @handle:        Device handle
 */
XCL_DRIVER_DLLHIDDEN
void
xclClose(xclDeviceHandle handle);

/*
 * xclGetDeviceInfo2() - Obtain various bits of information from the device
 *
 * @handle:        Device handle
 * @info:          Information record
 * Return:         0 on success or appropriate error number
 */
XCL_DRIVER_DLLHIDDEN
int
xclGetDeviceInfo2(xclDeviceHandle handle, struct xclDeviceInfo2 *info);

/*
 * xclGetUsageInfo() - Obtain usage information from the device
 *
 * @handle:        Device handle
 * @info:          Information record
 * Return:         0 on success or appropriate error number
 */
XCL_DRIVER_DLLHIDDEN
int
xclGetUsageInfo(xclDeviceHandle handle, struct xclDeviceUsage *info);

/*
 * xclGetErrorStatus() - Obtain error information from the device
 *
 * @handle:        Device handle
 * @info:          Information record
 * Return:         0 on success or appropriate error number
 */
XCL_DRIVER_DLLHIDDEN
int
xclGetErrorStatus(xclDeviceHandle handle, struct xclErrorStatus *info);

/*
 * xclLoadXclBin() - Download FPGA image (xclbin) to the device
 *
 * @handle:        Device handle
 * @buffer:        Pointer to device image (xclbin) in memory
 * Return:         0 on success or appropriate error number
 *
 * Download FPGA image (AXLF) to the device. The PR bitstream is
 * encapsulated inside xclbin as a section.  The xclbin may also
 * contain other sections, which are suitably handled by the driver.
 *
 * This API also downloads OVERLAY (dtbo) section (Edge only).
 */
XCL_DRIVER_DLLHIDDEN
int
xclLoadXclBin(xclDeviceHandle handle, const struct axlf *buffer);

/*
 * xclGetSectionInfo() - Get information from sysfs about the downloaded xclbin sections
 *
 * @handle:        Device handle
 * @info:          Pointer to preallocated memory for return value.
 * @size:          size of preallocated memory.
 * @kind:          axlf_section_kind for which info is being queried
 * @index:         The (sub)section index for the "kind" type.
 * Return:         0 on success or appropriate error number
 *
 * Get the section information from sysfs. The index corrresponds to
 * the (section) entry of the axlf_section_kind data being
 * queried. The info and the size contain the return binary value of
 * the subsection and its size.
 */
XCL_DRIVER_DLLHIDDEN
int
xclGetSectionInfo(xclDeviceHandle handle, void* info, size_t *size,
                  enum axlf_section_kind kind, int index);

/*
 * xclReClock2() - Configure PR region frequncies
 *
 * @handle:        Device handle
 * @region:        PR region (always 0)
 * @targetFreqMHz: Array of target frequencies in order for the Clock Wizards driving
 *                 the PR region
 * Return:         0 on success or appropriate error number
 */
XCL_DRIVER_DLLHIDDEN
int
xclReClock2(xclDeviceHandle handle, unsigned short region,
            const unsigned short *targetFreqMHz);

/*
 * xclOpenContext() - Create shared/exclusive context on compute units
 *
 * @handle:        Device handle
 * @xclbinId:      UUID of the xclbin image running on the device
 * @ipIndex:       IP index
 * @shared:        Shared access or exclusive access
 * Return:         0 on success or appropriate error number
 *
 * The context is necessary before submitting execution jobs using
 * xclExecBuf(). Contexts may be exclusive or shared. Allocation of
 * exclusive contexts on a hardware IP would succeed only if another
 * client has not already setup up a context on that hardware IP.
 * Shared contexts can be concurrently allocated by many
 * processes on the same compute units.
 */
XCL_DRIVER_DLLHIDDEN
int
xclOpenContext(xclDeviceHandle handle, const xuid_t xclbinId, unsigned int ipIndex,
               bool shared);

/*
 * xclCloseContext() - Close previously opened context
 *
 * @handle:        Device handle
 * @xclbinId:      UUID of the xclbin image running on the device
 * @ipIndex:       ipIndex
 * Return:         0 on success or appropriate error number
 *
 * Close a previously allocated shared/exclusive context for a hardware IP.
 */
XCL_DRIVER_DLLHIDDEN
int
xclCloseContext(xclDeviceHandle handle, const xuid_t xclbinId, unsigned int ipIndex);

/*
 * Get the version number. 1 => Hal1 ; 2 => Hal2
 */
XCL_DRIVER_DLLHIDDEN
unsigned int
xclVersion();

/*
 * xclLogMsg() - Send message to log file as per settings in ini file.
 *
 * @handle:        Device handle
 * @level:         Severity level of the msg
 * @tag:           Tag supplied by the client, like "OCL" etc.
 * @format:        Format of Msg string to write to log file
 * @...:           All other arguments as per the format
 *
 * Return:         0 on success or appropriate error number
 */
XCL_DRIVER_DLLHIDDEN
int
xclLogMsg(xclDeviceHandle handle, enum xrtLogMsgLevel level, const char* tag,
          const char* format, ...);

/*
 * xclAllocBO() - Allocate a BO of requested size with appropriate flags
 *
 * @handle:        Device handle
 * @size:          Size of buffer
 * @unused:        This argument is ignored
 * @flags:         Specify bank information, etc
 * Return:         BO handle
 */
XCL_DRIVER_DLLHIDDEN
xclBufferHandle
xclAllocBO(xclDeviceHandle handle, size_t size,int unused, unsigned int flags);

/*
 * xclAllocUserPtrBO() - Allocate a BO using userptr provided by the user
 *
 * @handle:        Device handle
 * @userptr:       Pointer to 4K aligned user memory
 * @size:          Size of buffer
 * @flags:         Specify bank information, etc
 * Return:         BO handle
 */
XCL_DRIVER_DLLHIDDEN
xclBufferHandle
xclAllocUserPtrBO(xclDeviceHandle handle,void *userptr, size_t size,
                  unsigned int flags);

/*
 * xclFreeBO() - Free a previously allocated BO
 *
 * @handle:        Device handle
 * @boHandle:      BO handle
 */
XCL_DRIVER_DLLHIDDEN
void
xclFreeBO(xclDeviceHandle handle, xclBufferHandle boHandle);

/*
 * xclWriteBO() - Copy-in user data to host backing storage of BO
 *
 * @handle:        Device handle
 * @boHandle:      BO handle
 * @src:           Source data pointer
 * @size:          Size of data to copy
 * @seek:          Offset within the BO
 * Return:         0 on success or appropriate error number
 *
 * Copy host buffer contents to previously allocated device
 * memory. ``seek`` specifies how many bytes to skip at the beginning
 * of the BO before copying-in ``size`` bytes of host buffer.
 */
XCL_DRIVER_DLLHIDDEN
size_t
xclWriteBO(xclDeviceHandle handle, xclBufferHandle boHandle,
           const void *src, size_t size, size_t seek);

/*
 * xclReadBO() - Copy-out user data from host backing storage of BO
 *
 * @handle:        Device handle
 * @boHandle:      BO handle
 * @dst:           Destination data pointer
 * @size:          Size of data to copy
 * @skip:          Offset within the BO
 * Return:         0 on success or appropriate error number
 *
 * Copy contents of previously allocated device memory to host
 * buffer. ``skip`` specifies how many bytes to skip from the
 * beginning of the BO before copying-out ``size`` bytes of device
 * buffer.
 */
XCL_DRIVER_DLLHIDDEN
size_t
xclReadBO(xclDeviceHandle handle, xclBufferHandle boHandle,
          void *dst, size_t size, size_t skip);

/*
 * xclMapBO() - Memory map BO into user's address space
 *
 * @handle:        Device handle
 * @boHandle:      BO handle
 * @write:         READ only or READ/WRITE mapping
 * Return:         Memory mapped buffer
 *
 * Map the contents of the buffer object into host memory
 * To unmap the buffer call xclUnmapBO().
 */
XCL_DRIVER_DLLHIDDEN
void*
xclMapBO(xclDeviceHandle handle, xclBufferHandle boHandle, bool write);

/*
 * xclUnmapBO() - Unmap a BO that was previously mapped with xclMapBO
 *
 * @handle:        Device handle
 * @boHandle:      BO handle
 * @addr:          The mapped void * pointer returned from xclMapBO()
 */
XCL_DRIVER_DLLHIDDEN
int
xclUnmapBO(xclDeviceHandle handle, xclBufferHandle boHandle, void* addr);

/*
 * xclSyncBO() - Synchronize buffer contents in requested direction
 *
 * @handle:        Device handle
 * @boHandle:      BO handle
 * @dir:           To device or from device
 * @size:          Size of data to synchronize
 * @offset:        Offset within the BO
 * Return:         0 on success or standard errno
 *
 * Synchronize the buffer contents between host and device. Depending
 * on the memory model this may require DMA to/from device or CPU
 * cache flushing/invalidation
 */
XCL_DRIVER_DLLHIDDEN
int
xclSyncBO(xclDeviceHandle handle, xclBufferHandle boHandle,
          enum xclBOSyncDirection dir, size_t size, size_t offset);

/*
 * xclCopyBO() - Copy device buffer contents to another buffer
 *
 * @handle:        Device handle
 * @dstBoHandle:   Destination BO handle
 * @srcBoHandle:   Source BO handle
 * @size:          Size of data to synchronize
 * @dst_offset:    dst  Offset within the BO
 * @src_offset:    src  Offset within the BO
 * Return:         0 on success or standard errno
 *
 * Copy from source buffer contents to destination buffer, can be
 * device to device or device to host.  Always perform WRITE to
 * achieve better performance, destination buffer can be on device or
 * host require DMA from device
 */
XCL_DRIVER_DLLHIDDEN
int
xclCopyBO(xclDeviceHandle handle, xclBufferHandle dstBoHandle,
          xclBufferHandle srcBoHandle, size_t size, size_t dst_offset,
          size_t src_offset);

/*
 * xclExportBO() - Obtain DMA-BUF file descriptor for a BO
 *
 * @handle:        Device handle
 * @boHandle:      BO handle which needs to be exported
 * Return:         File handle to the BO or standard errno
 *
 * Export a BO for import into another device or Linux subsystem which
 * accepts DMA-BUF fd This operation is backed by Linux DMA-BUF
 * framework.  The file handle must be explicitly closed when no
 * longer needed.
 */
XCL_DRIVER_DLLHIDDEN
xclBufferExportHandle
xclExportBO(xclDeviceHandle handle, xclBufferHandle boHandle);

/*
 * xclImportBO() - Obtain BO handle for a BO represented by DMA-BUF file descriptor
 *
 * @handle:        Device handle
 * @fd:            File handle to foreign BO owned by another device which needs to be imported
 * @flags:         Unused
 * Return:         BO handle of the imported BO
 *
 * Import a BO exported by another device.     *
 * This operation is backed by Linux DMA-BUF framework
 */
XCL_DRIVER_DLLHIDDEN
xclBufferHandle
xclImportBO(xclDeviceHandle handle, xclBufferExportHandle fd, unsigned int flags);

/*
 * xclGetBOProperties() - Obtain xclBOProperties struct for a BO
 *
 * @handle:        Device handle
 * @boHandle:      BO handle
 * @properties:    BO properties struct pointer
 * Return:         0 on success
 *
 * This is the prefered method for obtaining BO property information.
 */
XCL_DRIVER_DLLHIDDEN
int
xclGetBOProperties(xclDeviceHandle handle, xclBufferHandle boHandle,
                   struct xclBOProperties *properties);

/*
 * xclIPName2Index() - Obtain IP index by IP name
 *
 * @handle:        Device handle
 * @ipName:        IP name. usually "<kernel name>:<instance name>"
 * Return:         IP index or appropriate error number
 *
 * xclIPName2Index() should be used to obtain unique index of IP as understood by other XRT APIs
 * like xclOpenContext().
 *
 */
XCL_DRIVER_DLLHIDDEN
int
xclIPName2Index(xclDeviceHandle handle, const char *ipName);

/*
 * xclUnmgdPread() - Perform unmanaged device memory read operation
 *
 * @handle:        Device handle
 * @flags:         Unused
 * @buf:           Destination data pointer
 * @size:          Size of data to copy
 * @offset:        Absolute offset inside device
 * Return:         0 on success or appropriate error number
 *
 * This API may be used to perform DMA operation from absolute
 * location specified. As stated before this API is for use by
 * debuggers and profilers. Do not use it in your application.
 */
XCL_DRIVER_DLLHIDDEN
ssize_t
xclUnmgdPread(xclDeviceHandle handle, unsigned int flags, void *buf,
              size_t size, uint64_t offset);

/*
 * xclUnmgdPwrite() - Perform unmanaged device memory read operation
 *
 * @handle:        Device handle
 * @flags:         Unused
 * @buf:           Source data pointer
 * @size:          Size of data to copy
 * @offset:        Absolute offset inside device
 * Return:         0 on success or appropriate error number
 *
 * This API may be used to perform DMA operation to an absolute
 * location specified. As stated before this API is for use by
 * debuggers and profilers. Do not use it in your application.
 */
XCL_DRIVER_DLLHIDDEN
ssize_t
xclUnmgdPwrite(xclDeviceHandle handle, unsigned int flags, const void *buf,
               size_t size, uint64_t offset);

/*
 * xclExecBuf() - Submit an execution request to the embedded (or software) scheduler
 *
 * @handle:        Device handle
 * @cmdBO:         BO handle containing command packet
 * Return:         0 or standard error number
 *
 * Submit an exec buffer for execution. The exec buffer layout is
 * defined by struct ert_packet which is defined in file *ert.h*. The
 * BO should been allocated with DRM_XOCL_BO_EXECBUF flag.
 */
XCL_DRIVER_DLLHIDDEN
int
xclExecBuf(xclDeviceHandle handle, xclBufferHandle cmdBO);

/*
 * xclExecWait() - Wait for one or more execution events on the device
 *
 * @handle:                  Device handle
 * @timeoutMilliSec:         How long to wait for
 * Return:                   Same code as poll system call
 *
 * Wait for notification from the hardware. The function essentially
 * calls "poll" system call on the driver file handle. The return
 * value has same semantics as poll system call.  If return value is >
 * 0 caller should check the status of submitted exec buffers Note
 * that if you perform wait for the same handle from multiple threads,
 * you may lose wakeup for some of them. So, use different handle in
 * different threads.
 */
XCL_DRIVER_DLLHIDDEN
int
xclExecWait(xclDeviceHandle handle, int timeoutMilliSec);

XCL_DRIVER_DLLHIDDEN
const struct axlf_section_header*
wrap_get_axlf_section(const struct axlf* top, enum axlf_section_kind kind);

/* Use xrt-smi to reset device */
XRT_DEPRECATED
XCL_DRIVER_DLLHIDDEN
int
xclResetDevice(xclDeviceHandle handle, enum xclResetKind kind);

/* Not supported */
XRT_DEPRECATED
XCL_DRIVER_DLLHIDDEN
int
xclLockDevice(xclDeviceHandle handle);

/* Not supported */
XRT_DEPRECATED
XCL_DRIVER_DLLHIDDEN
int
xclUnlockDevice(xclDeviceHandle handle);

/* Use xbmgmt to flash device */
XRT_DEPRECATED
XCL_DRIVER_DLLHIDDEN
int
xclUpgradeFirmware(xclDeviceHandle handle, const char *fileName);

/* Use xbmgmt to flash device */
XRT_DEPRECATED
XCL_DRIVER_DLLHIDDEN
int
xclUpgradeFirmware2(xclDeviceHandle handle, const char *file1, const char* file2);

/* Use xbmgmt to flash device */
XRT_DEPRECATED
XCL_DRIVER_DLLHIDDEN
int
xclUpgradeFirmwareXSpi(xclDeviceHandle handle, const char *fileName, int index);

/* Not supported */
XRT_DEPRECATED
XCL_DRIVER_DLLHIDDEN
int
xclBootFPGA(xclDeviceHandle handle);

/* Not supported */
XRT_DEPRECATED
XCL_DRIVER_DLLHIDDEN
int
xclRemoveAndScanFPGA();

/* Use xclGetBOProperties */
XRT_DEPRECATED
static inline size_t
xclGetBOSize(xclDeviceHandle handle, xclBufferHandle boHandle)
{
    struct xclBOProperties p;
    return !xclGetBOProperties(handle, boHandle, &p) ? (size_t)p.size : (size_t)-1;
}

/* Use xclGetBOProperties */
XRT_DEPRECATED
static inline uint64_t
xclGetDeviceAddr(xclDeviceHandle handle, xclBufferHandle boHandle)
{
    struct xclBOProperties p;
    return !xclGetBOProperties(handle, boHandle, &p) ? p.paddr : (uint64_t)-1;
}

/* Use xclRegWrite */
XRT_DEPRECATED
XCL_DRIVER_DLLHIDDEN
size_t
xclWrite(xclDeviceHandle handle, enum xclAddressSpace space, uint64_t offset,
         const void *hostBuf, size_t size);

/* Use xclRegRead */
XRT_DEPRECATED
XCL_DRIVER_DLLHIDDEN
size_t
xclRead(xclDeviceHandle handle, enum xclAddressSpace space, uint64_t offset,
        void *hostbuf, size_t size);

/* Not supported */
XRT_DEPRECATED
XCL_DRIVER_DLLHIDDEN
int
xclRegisterInterruptNotify(xclDeviceHandle handle, unsigned int userInterrupt,
                           int fd);

/*
 * DOC: XRT Stream Queue APIs
 *
 * NOTE: ALL STREAMING APIs ARE DEPRECATED!!!! THESE WILL BE REMOVED IN
 * A FUTURE RELEASE. PLEASE PORT YOUR APPLICATION TO USE SLAVE BRIDGE
 * (ALSO KNOWN AS HOST MEMORY) FOR EQUIVALENT FUNCTIONALITY.
 */

enum xclStreamContextFlags {
	/* Enum for xclQueueContext.flags */
	XRT_QUEUE_FLAG_POLLING		= (1 << 2),
};

struct xclQueueContext {
    uint32_t	type;	   /* stream or packet Queue, read or write Queue*/
    uint32_t	state;	   /* initialized, running */
    uint64_t	route;	   /* route id from xclbin */
    uint64_t	flow;	   /* flow id from xclbin */
    uint32_t	qsize;	   /* number of descriptors */
    uint32_t	desc_size; /* this might imply max inline msg size */
    uint64_t	flags;	   /* isr en, wb en, etc */
};

struct xclReqBuffer {
    union {
	char*    buf;    // ptr or,
	uint64_t va;	 // offset
    };
    uint64_t  len;
    uint64_t  buf_hdl;   // NULL when first field is buffer pointer
};

enum xclQueueRequestKind {
    XCL_QUEUE_WRITE = 0,
    XCL_QUEUE_READ  = 1,
    //More, in-line etc.
};

enum xclQueueRequestFlag {
    XCL_QUEUE_REQ_EOT			= 1 << 0,
    XCL_QUEUE_REQ_CDH			= 1 << 1,
    XCL_QUEUE_REQ_NONBLOCKING		= 1 << 2,
    XCL_QUEUE_REQ_SILENT		= 1 << 3, /* not supp. not generate event for non-blocking req */
};

struct xclQueueRequest {
    enum xclQueueRequestKind op_code;
    struct xclReqBuffer*       bufs;
    uint32_t	        buf_num;
    char*               cdh;
    uint32_t	        cdh_len;
    uint32_t		flag;
    void*		priv_data;
    uint32_t            timeout;
};

struct xclReqCompletion {
    char			resv[64]; /* reserved for meta data */
    void			*priv_data;
    size_t			nbytes;
    int				err_code;
};

/* End XRT Stream Queue APIs */

/* Not supported */
XRT_DEPRECATED
XCL_DRIVER_DLLHIDDEN
int
xclExecBufWithWaitList(xclDeviceHandle handle, xclBufferHandle cmdBO,
                       size_t num_bo_in_wait_list, xclBufferHandle *bo_wait_list);

#if 0
/* Not supported */
XRT_DEPRECATED
XCL_DRIVER_DLLESPEC
size_t
xclDebugReadIPStatus(xclDeviceHandle handle, enum xclDebugReadType type,
                     void* debugResults);
#endif

/*
 * This function is for internal use. We don't want outside user to use it.
 * Once the internal project move to XRT APIs. Then we can create an internal function
 * and remove this one.
 */
 /*
  * @handle: Device handle
  * @ipIndex: IP index
  * @start: the start offset of the read-only register range
  * @size: the size of the read-only register range
  * Return: 0 on success or appropriate error number
  *
  * This function is to set the read-only register range on a CU. It will be system-wide impact.
  * This is used when open a CU in shared context. It allows multiple users to call xclRegRead()
  * to access CU without impact KDS/ERT scheduling. It is not able to change the range after
  * the first xclRegRead().
  * This function returns error when called in an exclusive context.
  */
XCL_DRIVER_DLLHIDDEN
int
xclIPSetReadRange(xclDeviceHandle handle, uint32_t ipIndex, uint32_t start, uint32_t size);

#ifdef __cplusplus
}
#endif

#endif
