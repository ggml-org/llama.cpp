// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
#ifndef XRT_EXCEPTION_H_
#define XRT_EXCEPTION_H_

#ifdef __cplusplus
# include <exception>
#endif

#ifdef __cplusplus

namespace xrt {

/**
 * class exception -- XRT specific exceptions
 *
 * In most APIs errors are propagated as std::exception or
 * mostly as std::system_error with error codes corresponding to
 * POSIX error codes.
 *
 * Specific non standard system errors are defined as a specific
 * xrt::exception, which in turn is derived from a std::exception.
 */
class exception : public std::exception
{};

} // namespace xrt

#endif // __cplusplus

#endif
