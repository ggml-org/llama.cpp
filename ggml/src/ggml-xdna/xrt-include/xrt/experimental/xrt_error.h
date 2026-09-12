// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2020 Xilinx, Inc.  All rights reserved.
// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#ifndef xrt_error_h_
#define xrt_error_h_

#include "xrt/deprecated/xrt.h"
#include "xrt_device.h"
#include "xrt/detail/xrt_error_code.h"

#ifdef __cplusplus
# include <string>
# include <memory>
# include <cstdint>
#endif

#ifdef __cplusplus
namespace xrt {

class error_impl;
class error
{
public:
  /**
   * error() -  Constructor for last asynchronous error of a class
   *
   * @device:   Device to get last errors from
   * @ecl:      Error class to retrieve error from
   */
  XCL_DRIVER_DLLESPEC
  error(const xrt::device& device, xrtErrorClass ecl);

  /**
   * error() - Constructor from error code and time stamp
   *
   * @ecl:       Error code
   * @timestamp: Time stamp
   *
   * Allow construction of error object from manually retrieved
   * error code and timestamp
   */
  XCL_DRIVER_DLLESPEC
  error(xrtErrorCode ecl, xrtErrorTime timestamp);

  /**
   * get_timestamp() - Get the timestamp for this error
   *
   * Return:  Error timestamp
   */
  XCL_DRIVER_DLLESPEC
  xrtErrorTime
  get_timestamp() const;

  /**
   * get_error_code() - Get the error code for this error
   *
   * Return:  Underlying xrt error code
   */
  XCL_DRIVER_DLLESPEC
  xrtErrorCode
  get_error_code() const;

  /**
   * to_string() - Convert error object into a formatted string
   *
   * Return:  Formatted string for error
   */
  XCL_DRIVER_DLLESPEC
  std::string
  to_string() const;

private:
  std::shared_ptr<error_impl> handle;
};

} // xrt

#endif

/**
 * xrtErrorGetLast - Get the last error code and its timestamp of a given error class.
 *
 * @handle:       Device handle.
 * @class:        Error Class for the last error to get.
 * @error:        Returned XRT error code.
 * @timestamp:    The timestamp when the error generated
 *
 * Return:        0 on success or appropriate XRT error code.
 */
XCL_DRIVER_DLLESPEC
int
xrtErrorGetLast(xrtDeviceHandle handle, enum xrtErrorClass ecl, xrtErrorCode* error, uint64_t* timestamp);

/**
 * xrtErrorGetString - Get the description string of a given error code.
 *
 * @handle:       Device handle.
 * @error:        XRT error code.
 * @out:          Preallocated output buffer for the error string.
 * @len:          Length of output buffer.
 * @out_len:      Output of length of message, ignored if null.
 *
 * Return:        0 on success or appropriate XRT error code.
 *
 * Specifying out_len while passing nullptr for output buffer will
 * return the message length, which can then be used to allocate the
 * output buffer itself.
 */
XCL_DRIVER_DLLESPEC
int
xrtErrorGetString(xrtDeviceHandle, xrtErrorCode error, char* out, size_t len, size_t* out_len);

#endif
