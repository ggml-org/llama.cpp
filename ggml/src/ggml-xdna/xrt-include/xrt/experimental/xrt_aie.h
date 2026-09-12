// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2021 Xilinx, Inc. All rights reserved.
// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#ifndef XRT_EXPERIMENTAL_AIE_H
#define XRT_EXPERIMENTAL_AIE_H
#include "xrt/xrt_aie.h"
#include "xrt/experimental/xrt_elf.h"

#ifdef __cplusplus
#include <type_traits>

namespace xrt::aie {

/**
 * An aie::program object is a representaiton of a program to be executed
 * on the AIE.  The program is an ELF file with sections and data
 * specific to the AIE.
 *
 * A program is added to a hardware context either when the hardware
 * context is contructed or by later adding the program to an existing
 * hardware context.
 *
 * An aie::program object can be created from an existing xrt::elf object
 * provided the ELF represents an AIE program. It can also be constructed
 * from a file or stream as provided by class xrt::elf.
 *
 * The xrt::aie::program and xrt::elf are interchangable; the former
 * can be assigned to the latter, and the former can be constructed
 * from the latter without object slicing since the xrt::aie::program
 * is sharing same underlying implmenentation as xrt::elf.
 */
class program : public xrt::elf
{
  // Validate that the ELF represents an AIE program
  XRT_API_EXPORT
  void
  valid_or_error();

public:
  using size_type = uint32_t;

  /**
   * program() - Construct for empty program
   *
   * It is undefined behavior to use a default constructed program object
   * for anything but assignment.
   */
  program() = default;

  /**
   * program() - Create a program object using xrt::elf constructors.
   *
   * Construction fails if the ELF is not a valid AIE program.
   *
   * This constructor is enabled only for types that do not match
   * program, this avoids forwaring reference overload
   */
  template <typename ArgType,
            typename = std::enable_if_t<!std::is_same_v<std::decay_t<ArgType>, program>>>
  explicit
  program(ArgType&& arg)
    : xrt::elf{std::forward<ArgType>(arg)}
  {
    valid_or_error();
  }
  
  /**
   * program() - Create a program object from an existing ELF.
   *
   * Construction fails if the ELF is not a valid AIE program.
   */
  explicit
  program(xrt::elf xe)
    : xrt::elf{std::move(xe)}
  {
    valid_or_error();
  }

  /**
   * get_partition_size() - Required partition size to run the program
   *
   * The partition size is used to configure a hardware context such
   * that it spans columns sufficient to run the program.
   */
  XRT_API_EXPORT
  size_type
  get_partition_size() const;
};

} // namespace xrt::aie

#endif // __cplusplus
#endif // XRT_EXPERIMENTAL_AIE_H
