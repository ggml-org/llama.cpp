// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2021-2022 Xilinx, Inc.  All rights reserved.
// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#ifndef XRT_MAILBOX_H_
#define XRT_MAILBOX_H_

#include "xrt/deprecated/xrt.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/detail/pimpl.h"

#ifdef __cplusplus
# include <cstdint>
#endif

#ifdef __cplusplus

namespace xrt {

/*!
 * @class mailbox
 *
 * @brief
 * xrt::mailbox provides access to the kernel mailbox if any
 *
 * @details
 * The mailbox extends the API of an xrt::run with mailbox specific
 * APIs to explicitly control mailbox aspects of a kernel.  It is an
 * error to construct a mailbox from a run object or kernel that
 * doesn't support mailbox.
 */
class mailbox_impl;
class mailbox : public detail::pimpl<mailbox_impl>
{
public:
  /**
   * mailbox() - Construct mailbox from a \ref xrt::run object.
   *
   * It is an error to construct a mailbox from a run object that is
   * not associated with a kernel supporting mailbox, or from a run
   * object associated with multiple compute units.  In other words a
   * mailbox is 1-1 with a compute unit that is managed by the
   * argument run object.
   */
  XCL_DRIVER_DLLESPEC
  explicit
  mailbox(const run& run);

  /**
   * read() - Read kernel arguments into mailbox copy
   *
   * This function is asynchronous, it requests the kernel to update
   * the content of the mailbox when it is safe to do so.  If the
   * kernel is in auto restart mode, then the update is delayed until
   * beginning of next iteration.  If the kernel is idle, then the
   * update is immediate.
   *
   * The mailbox is busy until the kernel has updated the content.
   * It is an error to call ``read()`` while the mailbox is busy,
   * throws std::system_error with std::errc::device_or_resource_busy.
   *
   * This function invalidates any argument data returned through
   * ``get_arg`` function.
   */
  XCL_DRIVER_DLLESPEC
  void
  read();

  /**
   * write() - Write the mailbox copy of kernel arguments to kernel
   *
   * This function is asynchronous, it requests the kernel to copy
   * the content of the mailbox when it is safe to do so.  If the
   * kernel is in auto restart mode, then the copying is delayed until
   * beginning of next iteration.  If the kernel is idle, then the
   * copying is immediate.
   *
   * The mailbox is busy until the kernel has copied the content.
   * It is an error to call ``write()`` while the mailbox is busy,
   * throws std::system_error with std::errc::device_or_resource_busy.
   */
  XCL_DRIVER_DLLESPEC
  void
  write();

  /**
   * get_arg() - Returns the mailbox copy of an argument
   *
   * @param index
   *  Index of kernel argument to read
   * @return
   *  The raw data and size in bytes of data
   *
   * The argument data returned is a direct reference to the mailbox
   * copy of the data.  It is valid only until the mailbox is updated
   * again.
   *
   * The function is synchronous and blocks if the mailbox is busy,
   * per pending ``read()`` or ``write()``.
   *
   * Subject to deprecation in favor of type safe version.
   */
  XCL_DRIVER_DLLESPEC
  std::pair<const void*, size_t>
  get_arg(int index) const;

  /**
   * set_arg() - Set a specific kernel global argument in the mailbox
   *
   * @param index
   *  Index of kernel argument to set
   * @param boh
   *  The global buffer argument value to set (lvalue).
   *
   * Use this API to queue up a new kernel argument value that can
   * be written to the kernel using ``write()``.
   *
   * The function is synchronous and blocks if the mailbox is busy,
   * per pending ``read()`` or ``write()``.
   */
  void
  set_arg(int index, xrt::bo& boh)
  {
    set_arg_at_index(index, boh);
  }

  /**
   * set_arg() - xrt::bo variant for const lvalue
   */
  void
  set_arg(int index, const xrt::bo& boh)
  {
    set_arg_at_index(index, boh);
  }

  /**
   * set_arg() - xrt::bo variant for rvalue
   */
  void
  set_arg(int index, xrt::bo&& boh)
  {
    set_arg_at_index(index, boh);
  }

  /**
   * set_arg() - Set a specific kernel scalar argument in the mailbox
   *
   * @param index
   *  Index of kernel argument to set
   * @param arg
   *  The scalar argument value to set.
   *
   * Use this API to queue up a new kernel scalar arguments value that
   * can be written to the kernel using ``write()``.
   *
   * The function is synchronous and blocks if the mailbox is busy,
   * per pending ``read()`` or ``write()``.
   */
  template <typename ArgType>
  void
  set_arg(int index, ArgType&& arg)
  {
    set_arg_at_index(index, &arg, sizeof(arg));
  }

  /**
   * set_arg - set named argument in the mailbox
   *
   * @param argnm
   *   Name of kernel argument
   * @param argvalue
   *   Argument value
   *
   * Throws if specified argument name doesn't match kernel
   * specification. Throws if argument value is incompatible with
   * specified argument
   */
  template <typename ArgType>
  void
  set_arg(const std::string& argnm, ArgType&& argvalue)
  {
    auto index = get_arg_index(argnm);
    set_arg(index, std::forward<ArgType>(argvalue));
  }


private:
  XCL_DRIVER_DLLESPEC
  int
  get_arg_index(const std::string& argnm) const;

  XCL_DRIVER_DLLESPEC
  void
  set_arg_at_index(int index, const void* value, size_t bytes);

  XCL_DRIVER_DLLESPEC
  void
  set_arg_at_index(int index, const xrt::bo&);
};

} // xrt
#endif

#endif
