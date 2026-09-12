// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022 Xilinx, Inc. All rights reserved.

#ifndef XRT_DETAIL_ANY_H
#define XRT_DETAIL_ANY_H

#ifdef __cplusplus

// XRT c++17 and later uses std::any in exported headers
// this avoids dependencies on boost-dev (boost::any)
// in client code that includes xrt_device.h and also
// compiles with c++17 or later.
#if (__cplusplus >= 201703L)
# include <any>
// forward declare boost::any for backwards compatibility
// with binaries compiled with xrt <= 2.12.x
namespace boost { class any; }
#else
// binaries compiled with c++14 require boost::any definition
// in xrt_device.h
# include <boost/any.hpp>
#endif

namespace xrt { namespace detail {

#if (__cplusplus >= 201703L)
using any = std::any;
#else
# define XRT_NO_STD_ANY
using any = boost::any;
#endif

}} // detail, xrt

#endif

#endif
