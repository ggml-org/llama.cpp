// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef XRT_ELF_H_
#define XRT_ELF_H_

#include "xrt/detail/config.h"
#include "xrt/detail/pimpl.h"
#include "xrt/xrt_uuid.h"

#ifdef __cplusplus
# include <string>
# include <istream>
#endif

#ifdef __cplusplus
namespace xrt {

/*!
 * @class elf
 *
 * @brief
 * An elf contains instructions for functions to execute in some
 * pre-configured hardware.  The xrt::elf class provides APIs to mine
 * the elf itself for relevant data.
 *
 * An xclbin is used to configure the hardware and an elf object is
 * always associated with exactly one xclbin, meaning the instructions
 * are for a specific hardware configuration.
 */
class elf_impl;
class elf : public detail::pimpl<elf_impl>
{
public:
  elf() = default;

  /**
   * elf() - Constructor from ELF file
   */
  XRT_API_EXPORT
  explicit
  elf(const std::string& fnm);

  /**
   * elf() - Constructor from raw data
   *
   * @param data
   *  Raw data of elf 
   *
   * The raw data of the elfcan be deleted after calling the
   * constructor.
   */
  XRT_API_EXPORT
  explicit
  elf(const std::string_view& data);

  /**
   * elf() - Constructor from ELF file
   *
   * Avoid ambiguity between std::string and std::string_view.
   */
  explicit
  elf(const char* fnm)
    : elf(std::string(fnm))
  {}

  /**
   * elf() - Constructor from raw ELF data stream
   *
   * @param stream
   *  Raw data stream of elf
   *
   */
  XRT_API_EXPORT
  explicit
  elf(std::istream& stream);

  /**
   * elf() - Constructor from raw ELF data
   *
   * @param data
   *  Pointer to raw elf data
   * @param size
   *  Size of raw elf data
   *
   */
  XRT_API_EXPORT
  elf(const void *data, size_t size);

  /**
   * get_cfg_uuid() - Get the configuration UUID of the elf
   *
   * @return
   *  The configuration UUID of the elf
   */
  XRT_API_EXPORT
  xrt::uuid
  get_cfg_uuid() const;

  /**
   * is_full_elf() - Check if the elf is a full ELF
   * 
   * A full ELF can be used as a replacement for xclbin, it contains
   * all the information required to create a hardware context like
   * partition size, kernel signatures, etc.
   *
   * @return
   *  True if the elf is a full ELF, false otherwise
   */
  XRT_API_EXPORT
  bool
  is_full_elf() const;
};

} // namespace xrt

#else
# error xrt::elf is only implemented for C++
#endif // __cplusplus

#endif
