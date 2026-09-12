// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
#ifndef XRT_EXT_H_
#define XRT_EXT_H_

// XRT extensions to XRT coreutil
// These extensions are experimental

#include "xrt/detail/config.h"
#include "xrt/detail/bitmask.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_module.h"

#ifdef __cplusplus
# include <cstdint>
#endif

#ifdef __cplusplus
namespace xrt::ext {

/*!
 * @class bo
 *
 * @brief Buffer object extension
 * xrt::ext::bo is an extension of xrt::bo with additional functionality
 *
 * @details
 * An extension buffer amends the contruction of an xrt::bo with
 * additional simplified constructors for specifying access mode of
 * host only buffers.
 *
 * Once constructed, the object must be assigned to an xrt::bo object
 * before use.  This is becayse XRT relies on templated kernel
 * argument assignment and the templated assignment operator is not
 * specialized for xrt::ext::bo.
 *
 * Ultimately the extension will be merged into class xrt::bo.
 */
class bo : public xrt::bo
{
public:

  /**
   * @enum access_mode - buffer object accessibility
   *
   * @var none
   *   No access is specified, same as read|write|local
   * @var read
   *   The buffer is read by device, the cpu writes to the buffer
   * @var write
   *   The buffer is written by device, the host reads from the buffer
   * @var local
   *   Access is local to process and device on which it is allocated
   * @var shared
   *   Access is shared between devices within process
   * @var process
   *   Access is shared between processes and devices
   * @var hybrid
   *   Access is shared between drivers (cross-adapter)
   *
   * The access mode is used to specify how the buffer is used by
   * device and process.
   *
   * A buffer can be specified as local, meaning it is only used by
   * the process and device on which it is allocated.  A buffer can
   * also be specified as shared, meaning it is shared between devices
   * within the process.  Finally a buffer can be specified as
   * process, meaning it is shared between processes and devices. If
   * neither local, shared, or process is specified, the default is
   * local.  Only one of local, shared, or process can be specified.
   *
   * A buffer can be opened for read, meaning the device will read the
   * content written by host, or it can be opened for write, meaning
   * the device will write to the buffer and the host will read. To
   * specify that a buffer is used for both read and write, the access
   * flags can be ORed.  If neither read or write is specified, the
   * default is read|write.
   *
   * The default access mode is read|write|local when no access mode
   * is specified.
   *
   * Friend operators are provided for bitwise operations on access
   * mode.  It is invalid to combine local, shared, proces, and hybrid.
   */
  enum class access_mode : uint64_t
  {
    none    = 0,

    read  = 1 << 0,
    write = 1 << 1,
    read_write = read | write,

    local   = 0,
    shared  = 1 << 2,
    process = 1 << 3,
    hybrid  = 1 << 4, 
  };

  friend constexpr access_mode operator&(access_mode lhs, access_mode rhs)
  {
    return xrt::detail::operator&(lhs, rhs);
  }

  friend constexpr access_mode operator|(access_mode lhs, access_mode rhs)
  {
    return xrt::detail::operator|(lhs, rhs);
  }

  /**
   * bo() - Constructor with user host buffer and access mode
   *
   * @param device
   *  The device on which to allocate this buffer
   * @param userptr
   *  The host buffer which must be page aligned
   * @param sz
   *  Size of buffer which must in multiple of page size
   * @param access
   *  Specific access mode for the buffer (see `enum access_mode`)
   *
   * This constructor creates a host_only buffer object with
   * specified access.
   */
  XRT_API_EXPORT
  bo(const xrt::device& device, void* userptr, size_t sz, access_mode access);

  /**
   * bo() - Constructor with user host buffer and access mode
   *
   * @param device
   *  The device on which to allocate this buffer
   * @param userptr
   *  The host buffer which must be page aligned
   * @param sz
   *  Size of buffer which must in multiple of page size
   * @param access
   *  Specific access mode for the buffer (see `enum access_mode`)
   *
   * This constructor creates a host_only buffer object with local
   * access and read|write direction.
   */
  XRT_API_EXPORT
  bo(const xrt::device& device, void* userptr, size_t sz);

  /**
   * bo() - Constructor for buffer object with specific access
   *
   * @param device
   *  The device on which to allocate this buffer
   * @param sz
   *  Size of buffer
   * @param access
   *  Specific access mode for the buffer (see `enum access_mode`)
   *
   * This constructor creates a host_only buffer object with
   * specified access.
   */
  XRT_API_EXPORT
  bo(const xrt::device& device, size_t sz, access_mode access);

  /**
   * bo() - Constructor for buffer object
   *
   * @param device
   *  The device on which to allocate this buffer
   * @param sz
   *  Size of buffer

   * This constructor creates a host_only buffer object with local
   * access and read|write direction.
   */
  XRT_API_EXPORT
  bo(const xrt::device& device, size_t sz);

  /**
   * bo() - Constructor to import an exported buffer from another process
   *
   * @param device
   *  The device that imports this buffer
   * @param pid
   *  Process id of exporting process
   * @param ehdl
   *  Exported buffer handle, implementation specific type
   *
   * The exported buffer handle is obtained from exporting process by
   * calling `export_buffer()` on the buffer to be exported.
   */
  XRT_API_EXPORT
  bo(const xrt::device& device, pid_type pid, xrt::bo::export_handle ehdl);

  /**
   * bo() - Constructor for buffer object
   *
   * @param hwctx
   *  The hardware context that this buffer object uses for queue
   *  operations such as syncing and residency operations.
   * @param sz
   *  Size of buffer
   * @param access
   *  Specific access mode for the buffer (see `enum access_mode`)
   *
   * This constructor creates a host_only buffer object with specified
   * access mode.  The hardware context is used for syncing of data
   * to from device and residency operations, which are all enqueued
   * operations that are synchronized on fence objects.
   */
  XRT_API_EXPORT
  bo(const xrt::hw_context& hwctx, size_t sz, access_mode access);

  /**
   * bo() - Constructor for buffer object
   *
   * @param hwctx
   *  The hardware context that this buffer object uses for queue
   *  operations such as syncing and residency operations.
   * @param sz
   *  Size of buffer
   *
   * This constructor creates a host_only buffer object with local
   * access and in|out direction.
   */
  XRT_API_EXPORT
  bo(const xrt::hw_context& hwctx, size_t sz);

  /// @cond
  // Deprecated.  Hardware context specific import is not supported
  XRT_API_EXPORT
  bo(const xrt::hw_context& hwctx, pid_type pid, xrt::bo::export_handle ehdl);
  /// @endcond
};


class kernel : public xrt::kernel
{
public:
  /**
   * kernel() - Constructor from module
   *
   * @param hwctx
   *   The hardware context that this kernel is created in.
   * @param module
   *   A module with elf binary instruction code which the
   *   kernel function will execute.
   * @param name
   *  Name of kernel function to construct.
   *
   * The module contains an elf binary with instructions for maybe
   * multiple functions.  When the kernel is constructed, the
   * corresponding function is located in the module.  The
   * instructions for the function are sent to the kernel mode driver
   * in an ERT packet along with the kernel function arguments.
   */
  XRT_API_EXPORT
  kernel(const xrt::hw_context& ctx, const xrt::module& mod, const std::string& name);

  /**
   * kernel() - Constructor from kernel name
   *
   * @param ctx
   *   The hardware context that this kernel is created in
   * @param name
   *   Name of kernel function to construct
   *
   * Constructs a kernel object by searching through all the ELF files
   * that are registered with the provided context. The function looks
   * for an ELF file that contains a kernel with the specified name.
   * Once a matching ELF file is found, it is used to construct the
   * kernel object.
   */
  XRT_API_EXPORT
  kernel(const xrt::hw_context& ctx, const std::string& name);
};

} // xrt::ext

#else
# error xrt::ext is only implemented for C++
#endif // __cplusplus

#endif
