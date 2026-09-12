// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#ifndef XRT_VERSION_SLIM_H_
#define XRT_VERSION_SLIM_H_

// This header contains minimal version info needed for xrt/detail/abi.h header
//
// Instantiated from XRT's src/CMake/config/version-slim.h.in for the XRT release
// that the vendored headers were taken from (2.21, the runtime shipped with the
// AMD NPU driver). xrt::detail::abi carries this into every inline API call so
// the DLL can validate schema compatibility.

#define XRT_VERSION(major, minor) ((major << 16) + (minor))
#define XRT_VERSION_CODE XRT_VERSION(2, 21)
#define XRT_MAJOR(code) ((code >> 16))
#define XRT_MINOR(code) (code - ((code >> 16) << 16))

#endif
