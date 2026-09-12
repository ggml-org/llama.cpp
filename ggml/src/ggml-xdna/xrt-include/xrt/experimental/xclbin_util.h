// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2020 Xilinx, Inc. All rights reserved.
// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

/**
 * This file contains publically exported xclbin utilities.
 */

#ifndef xclbin_util_h_
#define xclbin_util_h_

#include "xrt/detail/xclbin.h"
#include <cerrno>
#include <cstring>

static inline const axlf*
xclbin_axlf_handle(const void *xclbin)
{
  const axlf* top = (const axlf*) xclbin;
  return (strncmp(top->m_magic,"xclbin2",7)) ? NULL : top;
}

/**
 * xclbin_uuid() - Get the uuid of an xclbin
 *
 * @xclbin:  Raw pointer the enture xclbin file content
 * @out:     A uuid to populate with the xclbin uuid
 * Return:   0 on success, errcode otherwise
 */
static inline int
xclbin_uuid(const void *xclbin, xuid_t out)
{
  const axlf* top = xclbin_axlf_handle(xclbin);
  return (top) ? uuid_copy(out, top->m_header.uuid) , 0 : -EINVAL;
}

#endif
