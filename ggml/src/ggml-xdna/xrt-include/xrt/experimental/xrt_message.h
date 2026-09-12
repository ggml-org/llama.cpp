// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2021-2022 Xilinx, Inc.  All rights reserved.
// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#ifndef xrt_message_h_
#define xrt_message_h_

#include "xrt/deprecated/xrt.h"

#ifdef __cplusplus
# include <string>
# include <boost/format.hpp>
#endif

#ifdef __cplusplus

/*!
 * @namespace xrt::message
 *
 * @brief
 * APIs for XRT messaging.
 *
 * @details
 * XRT internally uses a message system that supports dispatching of
 * messages to null, console, file, or syslog under different verbosity
 * levels.  The sink and verbosity level is controlled statically
 * through ``xrt.ini`` or at run-time using ``xrt::ini``.
 *
 * The APIs in this file allow host application to use the same
 * message dispatch mechanism as XRT is configured to use.
 */
namespace xrt { namespace message {

/*!
 * @enum level
 *
 * @brief
 * Verbosity level for messages
 *
 * @details
 * Use logging APIs to control at what verbosity level the
 * messages should be issued.  The default verbosity can be changed in
 * `xrt.ini` or programatically by using `xrt::ini::set`.
 *
 * @var emergency
 * @var alert
 * @var critical
 * @var error
 * @var warning
 * @var notice
 * @var info
 * @var debug
 */
enum class level : unsigned short
{
  emergency = xrtLogMsgLevel::XRT_EMERGENCY,
  alert     = xrtLogMsgLevel::XRT_ALERT,
  critical  = xrtLogMsgLevel::XRT_CRITICAL,
  error     = xrtLogMsgLevel::XRT_ERROR,
  warning   = xrtLogMsgLevel::XRT_WARNING,
  notice    = xrtLogMsgLevel::XRT_NOTICE,
  info      = xrtLogMsgLevel::XRT_INFO,
  debug     = xrtLogMsgLevel::XRT_DEBUG
};

/// @cond
namespace detail {

template <typename ArgType>
void
format(boost::format& fmt, ArgType&& arg)
{
  fmt % std::forward<ArgType>(arg);
}

template <typename ArgType, typename ...Args>
void
format(boost::format& fmt, ArgType&& arg, Args&&... args)
{
  fmt % std::forward<ArgType>(arg);
  format(fmt, std::forward<Args>(args)...);
}

// enabled() - check if specified level is enabled
XCL_DRIVER_DLLESPEC
bool
enabled(level lvl);

} // detail
/// @endcond

/**
 * log() - Dispatch composed log message
 *
 * @param lvl
 *  Severity level, the message is ignored if configured level
 *  is less than specified level.
 * @param tag
 *  The message tag to use.
 * @param msg
 *  A formatted composed message
 */
XCL_DRIVER_DLLESPEC
void
log(level lvl, const std::string& tag, const std::string& msg);

/**
 * logf() - Compose and dispatch formatted log message
 *
 * @param lvl
 *  Severity level, the message is ignored if configured level
 *  is less than specified level.
 * @param tag
 *  The message tag to use.
 * @param format
 *  A format string similar to printf or boost::format
 * @param args
 *  Message arguments for the placeholders used in the format string
 *
 * This log function uses boost::format to compose the message from
 * specified format string and arguments.
 */
template <typename ...Args>
void
logf(level lvl, const std::string& tag, const char* format, Args&&... args)
{
  if (!detail::enabled(lvl))
    return;

  auto fmt = boost::format(format);
  detail::format(fmt, std::forward<Args>(args)...);
  log(lvl, tag, fmt.str());
}

}}

#endif // __cplusplus

#endif
