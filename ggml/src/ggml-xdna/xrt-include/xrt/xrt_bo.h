// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2020-2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
#ifndef XRT_BO_H_
#define XRT_BO_H_

#include "xrt/deprecated/xrt.h"
#include "xrt/detail/xrt_mem.h"
#include "xrt/detail/pimpl.h"

#ifdef __cplusplus
# include <memory>
# include <type_traits>
#endif

/**
 * typedef xrtDeviceHandle - opaque device handle
 */
typedef void* xrtDeviceHandle;

/**
 * typedef xrtBufferHandle - opaque buffer handle
 */
typedef void* xrtBufferHandle;

/**
 * typedef xrtBufferFlags - flags for BO
 *
 * See ``xrt_mem.h`` for available flags
 */
typedef uint64_t xrtBufferFlags;

/**
 * typedef xrtMemoryGroup - Memory bank group for buffer
 */
typedef uint32_t xrtMemoryGroup;

/*!
 * @struct xcl_buffer_handle
 *
 * @brief
 * Typed xclBufferHandle used to prevent ambiguity
 *
 * @details
 * Use when constructing xrt::bo from xclBufferHandle
 */
struct xcl_buffer_handle { xclBufferHandle bhdl; };

#ifdef __cplusplus

namespace xrt {

using memory_group = xrtMemoryGroup;

/*!
 * @struct pid_type
 *
 * @brief
 * Typed pid_t used to prevent ambiguity when contructing
 * bo with a process id.
 *
 * @details
 * Use xrt::bo bo{..., pid_type{pid}, ...};
 */
struct pid_type { pid_t pid; };

class device;
class hw_context;
class bo_impl;
/*!
 * @class bo
 * 
 * @brief
 * xrt::bo represents a buffer object that can be used as kernel argument
 */
class bo
{
public:
  /*!
   * @class async_handle
   *
   * @brief
   * xrt::bo::async_handle represents an asynchronously operation
   *
   * @details
   * A handle object is returned from asynchronous buffer object
   * operations.  It can be used to wait for the operation to
   * complete.
   */
  class async_handle_impl;
  class async_handle : public detail::pimpl<async_handle_impl>
  {
  public:
    explicit
    async_handle(std::shared_ptr<async_handle_impl> handle)
      : detail::pimpl<async_handle_impl>(std::move(handle))
    {}

    XCL_DRIVER_DLLESPEC
    void
    wait();
  };

public:
  /**
   * @enum flags - buffer object flags
   *
   * @var normal
   *  Create normal BO with host side and device side buffers
   * @var cacheable
   *  Create cacheable BO.  Only effective on embedded platforms.
   * @var device_only
   *  Create a BO with a device side buffer only
   * @var host_only
   *  Create a BO with a host side buffer only
   * @var p2p
   *  Create a BO for peer-to-peer use
   * @var svm
   *  Create a BO for SVM (supported on specific platforms only)
   * @var carveout
   *  Create a BO from a reserved memory pool. Supported for specific
   *  platforms only. For AMD Ryzen NPU this memory is allocated from
   *  a host memory carveout pool.
   * 
   * The flags used by xrt::bo are compatible with XCL style
   * flags as define in ``xrt_mem.h``
   */
  enum class flags : uint32_t
  {
    normal       = 0,
    cacheable    = XRT_BO_FLAGS_CACHEABLE,
    device_only  = XRT_BO_FLAGS_DEV_ONLY,
    host_only    = XRT_BO_FLAGS_HOST_ONLY,
    p2p          = XRT_BO_FLAGS_P2P,
    svm          = XRT_BO_FLAGS_SVM,
    carveout     = XRT_BO_FLAGS_CARVEOUT,
  };

#ifdef _WIN32
  using export_handle = uint64_t;
#else
  using export_handle = int32_t;
#endif

  /**
   * bo() - Constructor for empty bo
   *
   * A default constructed bo can be assigned to and can be used in a
   * Boolean check along with comparison.  
   *
   * Unless otherwise noted, it is undefined behavior to use xrt::bo
   * APIs on a default constructed object.
   */
  bo()
  {}

  /**
   * bo() - Constructor with user host buffer and flags
   *
   * @param device
   *  The device on which to allocate this buffer
   * @param userptr
   *  Pointer to aligned user memory
   * @param sz
   *  Size of buffer
   * @param flags
   *  Specify type of buffer
   * @param grp
   *  Device memory group to allocate buffer in
   *
   * The device memory group depends on connectivity.  If the buffer
   * as a kernel argument, then the memory group can be obtained from
   * the xrt::kernel object.
   */
  XCL_DRIVER_DLLESPEC
  bo(const xrt::device& device, void* userptr, size_t sz, bo::flags flags, memory_group grp);

  /**
   * bo() - Constructor with user host buffer and default flags
   *
   * @param device
   *  The device on which to allocate this buffer
   * @param userptr
   *  Pointer to aligned user memory
   * @param sz
   *  Size of buffer
   * @param grp
   *  Device memory group to allocate buffer in
   *
   * The device memory group depends on connectivity.  If the buffer
   * as a kernel argument, then the memory group can be obtained from
   * the xrt::kernel object.
   */
  XCL_DRIVER_DLLESPEC
  bo(const xrt::device& device, void* userptr, size_t sz, memory_group grp);

  /**
   * bo() - Constructor where XRT manages host buffer if needed
   *
   * @param device
   *  The device on which to allocate this buffer
   * @param sz
   *  Size of buffer
   * @param flags
   *  Specify type of buffer
   * @param grp
   *  Device memory group to allocate buffer in
   *
   * The device memory group depends on connectivity.  If the buffer
   * as a kernel argument, then the memory group can be obtained from
   * the xrt::kernel object.
   */
  XCL_DRIVER_DLLESPEC
  bo(const xrt::device& device, size_t sz, bo::flags flags, memory_group grp);

  /**
   * bo() - Constructor, default flags, where XRT manages host buffer if any
   *
   * @param device
   *  The device on which to allocate this buffer
   * @param sz
   *  Size of buffer
   * @param flags
   *  Specify type of buffer
   * @param grp
   *  Device memory group to allocate buffer in
   *
   * The device memory group depends on connectivity.  If the buffer
   * as a kernel argument, then the memory group can be obtained from
   * the xrt::kernel object.
   */
  XCL_DRIVER_DLLESPEC
  bo(const xrt::device& device, size_t sz, memory_group grp);

  /**
   * bo() - Constructor to import an exported buffer
   *
   * @param device
   *  Device that imports the exported buffer
   * @param ehdl
   *  Exported buffer handle, implementation specific type
   *
   * If the exported buffer handle acquired by using the export() method is
   * from another process, then it must be transferred through proper IPC
   * mechanism translating the underlying file-descriptor asscociated with
   * the buffer, see also constructor taking process id as argument.
   */
  XCL_DRIVER_DLLESPEC
  bo(const xrt::device& device, export_handle ehdl);

  /**
   * bo() - Constructor to import an exported buffer from another process
   *
   * @param device
   *  Device that imports the exported buffer
   * @param pid
   *  Process id of exporting process
   * @param ehdl
   *  Exported buffer handle, implementation specific type
   *
   * The exported buffer handle is obtained from exporting process by
   * calling `export()`. This contructor requires that XRT is built on
   * and running on a system with pidfd support.  Also the importing
   * process must have permission to duplicate the exporting process'
   * file descriptor.  This permission is controlled by ptrace access
   * mode PTRACE_MODE_ATTACH_REALCREDS check (see ptrace(2)).
   */
  XCL_DRIVER_DLLESPEC
  bo(const xrt::device& device, pid_type pid, export_handle ehdl);

  /**
   * bo() - Constructor with user host buffer and flags
   *
   * @param hwctx
   *  The hardware context in which to allocate this buffer
   * @param userptr
   *  Pointer to aligned user memory
   * @param sz
   *  Size of buffer
   * @param flags
   *  Specify type of buffer
   * @param grp
   *  Device memory group to allocate buffer in
   *
   * The device memory group depends on connectivity.  If the buffer
   * as a kernel argument, then the memory group can be obtained from
   * the xrt::kernel object.
   */
  XCL_DRIVER_DLLESPEC
  bo(const xrt::hw_context& hwctx, void* userptr, size_t sz, bo::flags flags, memory_group grp);

  /**
   * bo() - Constructor with user host buffer and default flags
   *
   * @param hwctx
   *  The hardware context in which to allocate this buffer
   * @param userptr
   *  Pointer to aligned user memory
   * @param sz
   *  Size of buffer
   * @param grp
   *  Device memory group to allocate buffer in
   *
   * The device memory group depends on connectivity.  If the buffer
   * as a kernel argument, then the memory group can be obtained from
   * the xrt::kernel object.
   */
  XCL_DRIVER_DLLESPEC
  bo(const xrt::hw_context& hwctx, void* userptr, size_t sz, memory_group grp);

  /**
   * bo() - Constructor where XRT manages host buffer if needed
   *
   * @param hwctx
   *  The hardware context in which to allocate this buffer
   * @param sz
   *  Size of buffer
   * @param flags
   *  Specify type of buffer
   * @param grp
   *  Device memory group to allocate buffer in
   *
   * The device memory group depends on connectivity.  If the buffer
   * as a kernel argument, then the memory group can be obtained from
   * the xrt::kernel object.
   */
  XCL_DRIVER_DLLESPEC
  bo(const xrt::hw_context& hwctx, size_t sz, bo::flags flags, memory_group grp);

  /**
   * bo() - Constructor, default flags, where XRT manages host buffer if any
   *
   * @param hwctx
   *  The hardware context in which to allocate this buffer
   * @param sz
   *  Size of buffer
   * @param flags
   *  Specify type of buffer
   * @param grp
   *  Device memory group to allocate buffer in
   *
   * The device memory group depends on connectivity.  If the buffer
   * as a kernel argument, then the memory group can be obtained from
   * the xrt::kernel object.
   */
  XCL_DRIVER_DLLESPEC
  bo(const xrt::hw_context& hwctx, size_t sz, memory_group grp);

  /// @cond
  // Deprecated constructor, use xrt::device variant
  XCL_DRIVER_DLLESPEC
  bo(xclDeviceHandle dhdl, void* userptr, size_t sz, bo::flags flags, memory_group grp);
  /// @endcond

  /// @cond
  // Deprecated legacy constructor
  bo(xclDeviceHandle dhdl, void* userptr, size_t sz, xrtBufferFlags flags, memory_group grp)
    : bo(dhdl, userptr, sz, static_cast<bo::flags>(flags), grp)
  {}
  /// @endcond

  /// @cond
  // Deprecated constructor, use xrt::device variant
  bo(xclDeviceHandle dhdl, void* userptr, size_t sz, memory_group grp)
    : bo(dhdl, userptr, sz, bo::flags::normal, grp)
  {}
  /// @endcond

  /// @cond
  // Deprecated constructor, use xrt::device variant
  XCL_DRIVER_DLLESPEC
  bo(xclDeviceHandle dhdl, size_t size, bo::flags flags, memory_group grp);
  /// @endcond

  /// @cond
  // Legacy constructor, use xrt::device variant
  bo(xclDeviceHandle dhdl, size_t size, xrtBufferFlags flags, memory_group grp)
    : bo(dhdl, size, static_cast<bo::flags>(flags), grp)
  {}
  /// @endcond

  /// @cond
  // Legacy constructor, use xrt::device variant
  bo(xclDeviceHandle dhdl, size_t size, memory_group grp)
    : bo(dhdl, size, bo::flags::normal, grp)
  {}
  /// @endcond

  /// @cond
  // Legacy constructor, use xrt::device variant
  XCL_DRIVER_DLLESPEC
  bo(xclDeviceHandle dhdl, xclBufferExportHandle ehdl);
  /// @endcond

  /// @cond
  // Legacy constructor, use xrt::device variant
  XCL_DRIVER_DLLESPEC
  bo(xclDeviceHandle dhdl, pid_type pid, xclBufferExportHandle ehdl);
  /// @endcond

  /**
   * bo() - Constructor for sub-buffer
   *
   * @param parent
   *  Parent buffer
   * @param size
   *  Size of sub-buffer
   * @param offset
   *  Offset into parent buffer
   */
  XCL_DRIVER_DLLESPEC
  bo(const bo& parent, size_t size, size_t offset);

  /// @cond
  /**
   * bo() - Constructor from xclBufferHandle
   *
   * @param dhdl
   *  Device on which the buffer handle was created
   * @param xhdl
   *  Typified shim buffer handle created with xclAllocBO variants
   *
   * This function allows construction of xrt::bo object from an
   * xclBufferHandle supposedly allocated using deprecated xcl APIs.
   * The buffer handle is allocated with xclAllocBO and must be
   * freed with xclFreeBO.
   *
   * Note that argument xclBufferHandle must be wrapped as
   * an xcl_buffer_handle in order to disambiguate the untyped
   * xclBufferHandle.
   *
   * Mixing xcl style APIs and xrt APIs is discouraged and
   * as such this documentation is not included in doxygen.
   */
  XCL_DRIVER_DLLESPEC
  bo(xclDeviceHandle dhdl, xcl_buffer_handle xhdl);
  /// @endcond

  /**
   * bo() - Copy ctor
   *
   * Performs shallow copy, sharing data with the source
   */
  bo(const bo& rhs) = default;

  /**
   * operator= () - Copy assignment
   *
   * Performs shallow copy, sharing data with the source
   */
  bo&
  operator=(const bo& rhs) = default;

  /**
   * bo() - Move ctor
   */
  bo(bo&& rhs) = default;

  /**
   * operator= () - Move assignment
   */
  bo&
  operator=(bo&& rhs) = default;

  /**
   * operator== () - Compare operator
   */
  bool
  operator==(const bo& rhs) const
  {
    return handle == rhs.handle;
  }

  /**
   * operator bool() - Check if bo handle is valid
   */
  explicit
  operator bool() const
  {
    return handle != nullptr;
  }

  /**
   * size() - Get the size of this buffer
   *
   * @return
   *  Size of buffer in bytes
   *
   * Returns 0 for a default constructed xrt::bo.
   */
  XCL_DRIVER_DLLESPEC
  size_t
  size() const;

  /**
   * address() - Get the device address of this buffer
   *
   * @return
   *  Device address of buffer
   */
  XCL_DRIVER_DLLESPEC
  uint64_t
  address() const;

  /**
   * get_memory_group() - Get the memory group in which this buffer is allocated
   *
   * @return
   *  Memory group index with which the buffer was constructed
   */
  XCL_DRIVER_DLLESPEC
  memory_group
  get_memory_group() const;

  /**
   * get_flags() - Get the flags with which this buffer was constructed
   *
   * @return
   *  The xrt::bo::flags used when the buffer was contructed
   */
  XCL_DRIVER_DLLESPEC
  flags
  get_flags() const;

  /**
   * buffer_export() - Export this buffer
   *
   * @return
   *  Exported buffer handle
   *
   * An exported buffer can be imported on another device by this
   * process or another process. For multiprocess transfer, the exported
   * buffer must be transferred through a proper IPC facility to translate
   * the underlying file-descriptor properly into another process.
   *
   * The lifetime of the exported buffer handle is associated with the
   * exporting buffer (this).  The handle is disposed of when the
   * exporting buffer is destructed.
   *
   * It is undefined behavior to use the export handle after the
   * exporting buffer object has gone out of scope.
   */
  XCL_DRIVER_DLLESPEC
  export_handle
  export_buffer();

  /**
   * async() - Start buffer content txfer with device side
   *
   * @param dir
   *  To device or from device
   * @param sz
   *  Size of data to synchronize
   * @param offset
   *  Offset within the BO
   *
   * Asynchronously transfer specified size bytes of buffer
   * starting at specified offset.
   */
  XCL_DRIVER_DLLESPEC
  async_handle
  async(xclBOSyncDirection dir, size_t sz, size_t offset);

  /**
   * async() - Start buffer content txfer with device side
   *
   * @param dir
   *  To device or from device
   * @param sz
   *  Size of data to synchronize
   * @param offset
   *  Offset within the BO
   *
   * Asynchronously transfer entire buffer content in specified direction
   */
  async_handle
  async(xclBOSyncDirection dir)
  {
    return async(dir, size(), 0);
  }

  /**
   * sync() - Synchronize buffer content with device side
   *
   * @param dir
   *  To device or from device
   * @param sz
   *  Size of data to synchronize
   * @param offset
   *  Offset within the BO
   *
   * Sync specified size bytes of buffer starting at specified offset.
   */
  XCL_DRIVER_DLLESPEC
  void
  sync(xclBOSyncDirection dir, size_t sz, size_t offset);

  /**
   * sync() - Synchronize buffer content with device side
   *
   * @param dir
   *  To device or from device
   *
   * Sync entire buffer content in specified direction.
   */
  void
  sync(xclBOSyncDirection dir)
  {
    sync(dir, size(), 0);
  }

  /**
   * map() - Map the host side buffer into application
   *
   * @return
   *  Memory mapped buffer
   *
   * Map the contents of the buffer object into host memory
   */
  XCL_DRIVER_DLLESPEC
  void*
  map();

  /**
   * map() - Const variant of map
   *
   * Constness is enforce through return type
   */
  XCL_DRIVER_DLLESPEC
  const void*
  map() const;

  /**
   * map() - Map the host side buffer info application
   *
   * @tparam MapType
   *  Type of mapped data
   * @return
   *  Memory mapped buffer
   */
  template<typename MapType>
  MapType
  map()
  {
    return reinterpret_cast<MapType>(map());
  }

  /**
   * map() - Const variant of map
   *
   * Constness is enforce through return type
   */
  template<typename MapType>
  MapType
  map() const
  {
    static_assert(std::is_pointer<MapType>::value &&
                  std::is_const<std::remove_pointer_t<MapType>>::value,
                  "MapType must be a pointer and const-qualified type");

    return reinterpret_cast<MapType>(map());
  }

  /**
   * write() - Copy-in user data to host backing storage of BO
   *
   * @param src
   *  Source data pointer
   * @param size
   *  Size of data to copy
   * @param seek
   *  Offset within the BO
   *
   * Copy source data to host buffer of this buffer object.
   * ``seek`` specifies how many bytes to skip at the beginning
   * of the BO before copying-in ``size`` bytes to host buffer.
   *
   * If BO has no host backing storage, e.g. a device only buffer,
   * then write is directly to the device buffer.
   */
  XCL_DRIVER_DLLESPEC
  void
  write(const void* src, size_t size, size_t seek);

  /**
   * write() - Copy-in user data to host backing storage of BO
   *
   * @param src
   *  Source data pointer
   *
   * Copy specified source data to host buffer of this buffer object.
   *
   * If BO has no host backing storage, e.g. a device only buffer,
   * then write is directly to the device buffer.
   */
  void
  write(const void* src)
  {
    write(src, size(), 0);
  }

  /**
   * read() - Copy-out user data from host backing storage of BO
   *
   * @param dst
   *  Destination data pointer
   * @param size
   *  Size of data to copy
   * @param skip
   *  Offset within the BO
   *
   * Copy content of host buffer of this buffer object to specified
   * destination.  ``skip`` specifies how many bytes to skip from the
   * beginning of the BO before copying-out ``size`` bytes of host
   * buffer.
   *
   * If BO has no host backing storage, e.g. a device only buffer,
   * then read is directly from the device buffer.
   */
  XCL_DRIVER_DLLESPEC
  void
  read(void* dst, size_t size, size_t skip);

  /**
   * read() - Copy-out user data from host backing storage of BO
   *
   * @param dst
   *  Destination data pointer
   *
   * Copy content of host buffer of this buffer object to specified
   * destination.
   *
   * If BO has no host backing storage, e.g. a device only buffer,
   * then read is directly from the device buffer.
   */
  void
  read(void* dst)
  {
    read(dst, size(), 0);
  }

  /**
   * copy() - Deep copy BO content from another buffer
   *
   * @param src
   *  Source BO to copy from
   * @param sz
   *  Size of data to copy
   * @param src_offset
   *  Offset into src buffer copy from
   * @param dst_offset
   *  Offset into this buffer to copy to
   *
   * Throws if copy size is 0 or sz + src/dst_offset is out of bounds.
   */
  XCL_DRIVER_DLLESPEC
  void
  copy(const bo& src, size_t sz, size_t src_offset=0, size_t dst_offset=0);

  /**
   * copy() - Deep copy BO content from another buffer
   *
   * @param src
   *  Source BO to copy from
   *
   * Copy full content of specified src buffer object to this buffer object
   */
  void
  copy(const bo& src)
  {
    copy(src, src.size());
  }

  /**
   * ~bo() - Destructor for bo object
   */
  XCL_DRIVER_DLLESPEC
  ~bo();

public:
  /// @cond
  const std::shared_ptr<bo_impl>&
  get_handle() const
  {
    return handle;
  }

  // Construct bo from C-API handle
  // Throws if argument handle is not from xrtBOAlloc variant
  XCL_DRIVER_DLLESPEC
  bo(xrtBufferHandle);

  bo(std::shared_ptr<bo_impl> impl)
    : handle(std::move(impl))
  {}
  /// @endcond
private:
  std::shared_ptr<bo_impl> handle;
};

} // namespace xrt

/// @cond
extern "C" {
#endif

/**
 * xrtBOAllocUserPtr() - Allocate a BO using userptr provided by the user
 *
 * @dhdl:          Device handle
 * @userptr:       Pointer to 4K aligned user memory
 * @size:          Size of buffer
 * @flags:         Specify type of buffer
 * @grp:           Specify bank information
 * Return:         xrtBufferHandle on success or NULL
 */
XCL_DRIVER_DLLESPEC
xrtBufferHandle
xrtBOAllocUserPtr(xrtDeviceHandle dhdl, void* userptr, size_t size, xrtBufferFlags flags, xrtMemoryGroup grp);

/**
 * xrtBOAlloc() - Allocate a BO of requested size with appropriate flags
 *
 * @dhdl:          Device handle
 * @size:          Size of buffer
 * @flags:         Specify type of buffer
 * @grp:           Specify bank information
 * Return:         xrtBufferHandle on success or NULL
 */
XCL_DRIVER_DLLESPEC
xrtBufferHandle
xrtBOAlloc(xrtDeviceHandle dhdl, size_t size, xrtBufferFlags flags, xrtMemoryGroup grp);

/**
 * xrtBOImport() - Allocate a BO imported from another device
 *
 * @dhdl:     Device that imports the exported buffer
 * @ehdl:     Exported buffer handle, implementation specific type
 *
 * The exported buffer handle is acquired by using the export() method
 * and can be passed to another process.
 */
XCL_DRIVER_DLLESPEC
xrtBufferHandle
xrtBOImport(xrtDeviceHandle dhdl, xclBufferExportHandle ehdl);

/**
 * xrtBOExport() - Export this buffer
 *
 * @bhdl:   Buffer handle
 * Return:  Exported buffer handle
 *
 * An exported buffer can be imported on another device by this
 * process or another process.
 */
XCL_DRIVER_DLLESPEC
xclBufferExportHandle
xrtBOExport(xrtBufferHandle bhdl);

/**
 * xrtBOSubAlloc() - Allocate a sub buffer from a parent buffer
 *
 * @parent:        Parent buffer handle
 * @size:          Size of sub buffer
 * @offset:        Offset into parent buffer
 * Return:         xrtBufferHandle on success or NULL
 */
XCL_DRIVER_DLLESPEC
xrtBufferHandle
xrtBOSubAlloc(xrtBufferHandle parent, size_t size, size_t offset);

/*
 * xrtBOAllocFromXcl() - Undocumented allocation from an xclBufferHandle
 *
 * @dhdl:  XRT device handle on which the buffer is residing.
 * @xhdl:  The xcl buffer handle to convert to an xrtBufferHandle
 * Return: xrtBufferHandle which must be freed using `xrtBOFree`
 *
 * Note that argument xclBufferHandle must be wrapped as
 * an xcl_buffer_handle in order to disambiguate the untyped
 * xclBufferHandle.
 *
 * Please note that the device is an xrtDeviceHandle.  It is the
 * responsibility of the user to convert an xclDeviceHandle to
 * an xrtDeviceHandle before calling this API.
 *
 * The xrtBufferHandle returned by this API
 *
 * The argument xcl buffer handle must be explicitly freed using
 * xclFreeBO, in other words xclAllocBO requires xclFreeBO and
 * xrtBOAlloc requires xrtBOFree.
 */
XCL_DRIVER_DLLESPEC
xrtBufferHandle
xrtBOAllocFromXcl(xrtDeviceHandle dhdl, xclBufferHandle xhdl);

/**
 * xrtBOFree() - Free a previously allocated BO
 *
 * @bhdl:         Buffer handle
 * Return:        0 on success, or err code on error
 */
XCL_DRIVER_DLLESPEC
int
xrtBOFree(xrtBufferHandle bhdl);

/**
 * xrtBOSize() - Get the size of this buffer
 *
 * @bhdl:         Buffer handle
 * Return:        Size of buffer in bytes
 */
XCL_DRIVER_DLLESPEC
size_t
xrtBOSize(xrtBufferHandle bhdl);

/**
 * xrtBOAddr() - Get the physical address of this buffer
 *
 * @bhdl:         Buffer handle
 * Return:        Device address of this BO, or LLONG_MAX on error
 */
XCL_DRIVER_DLLESPEC
uint64_t
xrtBOAddress(xrtBufferHandle bhdl);

/**
 * xrtBOSync() - Synchronize buffer contents in requested direction
 *
 * @bhdl:          Bufferhandle
 * @dir:           To device or from device
 * @size:          Size of data to synchronize
 * @offset:        Offset within the BO
 * Return:         0 on success or error
 *
 * Synchronize the buffer contents between host and device. Depending
 * on the memory model this may require DMA to/from device or CPU
 * cache flushing/invalidation
 */
XCL_DRIVER_DLLESPEC
int
xrtBOSync(xrtBufferHandle bhdl, enum xclBOSyncDirection dir, size_t size, size_t offset);

/**
 * xrtBOMap() - Memory map BO into user's address space
 *
 * @bhdl:       Buffer handle
 * Return:      Memory mapped buffer, or NULL on error
 *
 * Map the contents of the buffer object into host memory.  The buffer
 * object is unmapped when freed.
 */
XCL_DRIVER_DLLESPEC
void*
xrtBOMap(xrtBufferHandle bhdl);

/**
 * xrtBOWrite() - Copy-in user data to host backing storage of BO
 *
 * @bhdl:          Buffer handle
 * @src:           Source data pointer
 * @size:          Size of data to copy
 * @seek:          Offset within the BO
 * Return:         0 on success or appropriate error number
 *
 * Copy host buffer contents to previously allocated device
 * memory. ``seek`` specifies how many bytes to skip at the beginning
 * of the BO before copying-in ``size`` bytes of host buffer.
 */
XCL_DRIVER_DLLESPEC
int
xrtBOWrite(xrtBufferHandle bhdl, const void* src, size_t size, size_t seek);

/**
 * xrtBORead() - Copy-out user data from host backing storage of BO
 *
 * @bhdl:          Buffer handle
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
XCL_DRIVER_DLLESPEC
int
xrtBORead(xrtBufferHandle bhdl, void* dst, size_t size, size_t skip);

/**
 * xrtBOCopy() - Deep copy BO content from another buffer
 *
 * @dst:          Destination BO to copy to
 * @src:          Source BO to copy from
 * @sz:           Size of data to copy
 * @dst_offset:   Offset into destination buffer to copy to
 * @src_offset:   Offset into src buffer to copy from
 * Return:        0 on success or appropriate error number
 *
 * It is an error if sz is 0 bytes or sz + src/dst_offset is out of bounds.
 */
XCL_DRIVER_DLLESPEC
int
xrtBOCopy(xrtBufferHandle dst, xrtBufferHandle src, size_t sz, size_t dst_offset, size_t src_offset);

/// @endcond
#ifdef __cplusplus
}
#endif

#endif
