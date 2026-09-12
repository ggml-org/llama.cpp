// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
#ifndef XRT_MODULE_H_
#define XRT_MODULE_H_

#include "xrt/detail/config.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_uuid.h"
#include "xrt/detail/pimpl.h"
#include "xrt/experimental/xrt_elf.h"

#ifdef __cplusplus
# include <cstdint>
# include <string>
#endif

#ifdef __cplusplus
namespace xrt {

/*!
 * @class module
 *
 * @brief
 * An xrt::module contains functions an application will execute in
 * hardware.  In Alveo the xclbin is the container that configures the
 * hardware and provides the functions.  In AIE the functions are a
 * set of instructions that are run on configured hardware, the
 * instructions are embedded in an elf file, which is parsed for meta
 * data determining how the functions are invoked.
 *
 * An xrt::module is constructed from the object that contains the
 * functions to execute.  In the case of Alveo, the module is
 * definitely constructed from an xrt::xclbin , in case of AIE the
 * module is constructed from an xrt::elf or from a user pointer.
 */
class module_impl;
class module : public detail::pimpl<module_impl>
{
public:
  /**
   */
  module()
  {}

  /**
   * module() - Constructor from elf
   *
   * @param elf
   *   An elf binary with functions to execute
   *
   * The elf binary contains instructions for functions to be executed
   * in some hardware context.  The elf binary has text segments with
   * meta data to be mined for function arguments and type.
   *
   * The constructor retains ownership of the elf object.
   */
  XRT_API_EXPORT
  explicit
  module(const xrt::elf& elf);

  /**
   * module() - Constructor from user ptr
   *
   * @param userptr
   *   A pointer to an opaque representation of the instructions
   *   to execute on hardware configured by an xclbin with uuid
   * @param sz
   *   Size in bytes of the userptr buffer
   * @param uuid
   *   Unique id of the hardware configuration.  Must match the
   *   xclbin uuid use to configure the hardware.
   *
   * The user pointer is an opaque representation of the instructions
   * to execute on hardware configured by xclbin.
   *
   * The constructor copies the content of the userptr.
   */
  XRT_API_EXPORT
  module(void* userptr, size_t sz, const xrt::uuid& uuid);

  /**
   * module() - Constructor associate module with hardware context
   *
   * @param parent
   *   Parent module with instruction buffer to move into hwctx
   * @param hwctx
   *   Hardware context to associate with module
   *
   * Copy content of existing module into an allocation associated
   * with the specified hardware context.
   *
   * Throws if module is not compatible with hardware context
   */
  XRT_API_EXPORT
  module(const xrt::module& parent, const xrt::hw_context& hwctx);

  ///@cond
  // Undocumented converting constructor using impl only
  XRT_API_EXPORT
  explicit
  module(std::shared_ptr<module_impl> impl)
    : detail::pimpl<module_impl>(std::move(impl))
  {}
  /// @endcond

  /**
   * get_cfg_uuid() - Get the uuid of the hardware configuration
   *
   * @return
   *   UUID of matching hardware configuration
   *
   * An module is associated with exactly one hardware configuration.
   * This function returns the uuid that identifies the configuration.
   */
  XRT_API_EXPORT
  xrt::uuid
  get_cfg_uuid() const;

  XRT_API_EXPORT
  xrt::hw_context
  get_hw_context() const;

private:
};

} // namespace xrt

#else
# error xrt::module is only implemented for C++
#endif // __cplusplus

#endif
