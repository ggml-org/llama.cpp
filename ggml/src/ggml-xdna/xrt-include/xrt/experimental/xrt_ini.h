// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2021-2022 Xilinx, Inc.  All rights reserved.
// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#ifndef XRT_INI_H_
#define XRT_INI_H_

#include "xrt/deprecated/xrt.h"

#ifdef __cplusplus
# include <string>
#endif

#ifdef __cplusplus

/*!
 * @namespace xrt::ini
 *
 * @brief
 * APIs for XRT configuration control.
 *
 * @details
 * XRT can be configured through a json xrt.ini file co-located with
 * the host executable.  If present, XRT uses configuration options
 * from the ini file when a given option is first accessed. Without an
 * ini file, the configuration options take on default values.
 *
 * The APIs in this file allow host application to specify
 * configuration options for XRT programatically.  It is only possible
 * for the host application to change configuration options before a
 * given option is used by XRT the very first time.
 */
namespace xrt { namespace ini {

/*!
 * set() - Change xrt.ini string value for specified key
 *
 * @param key
 *  Key to change value for
 * @param value
 *  New value for key
 *
 * Throws if key value cannot be changed.
 */
XCL_DRIVER_DLLESPEC
void
set(const std::string& key, const std::string& value);

/*!
 * set() - Change xrt.ini string value for specified key
 *
 * @param key
 *  Key to change value for
 * @param value
 *  New value for key
 *
 * Throws if key value cannot be changed.
 */
inline void
set(const std::string& key, unsigned int value)
{
  set(key, std::to_string(value));
}


}} // ini, xrt

/// @cond
extern "C" {
#endif

/**
 * xrtIniSet() - Change xrt.ini string value for specified key
 *
 * @key:    Key to change value for
 * @value:  New value for key
 * Return:  0 on success, error if key value cannot be changed
 */
XCL_DRIVER_DLLESPEC
int
xrtIniStringSet(const char* key, const char* value);

/**
 * xrtIniUintSet() - Change xrt.ini unsigned int value for specified key
 *
 * @key:    Key to change value for
 * @value:  New value for key
 * Return:  0 on success, error if key value cannot be changed
 */
XCL_DRIVER_DLLESPEC
int
xrtIniUintSet(const char* key, unsigned int value);

/// @endcond
#ifdef __cplusplus
}
#endif

#endif
