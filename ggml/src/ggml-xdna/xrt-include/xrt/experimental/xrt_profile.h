// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2020-2022 Xilinx, Inc.  All rights reserved.
// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#ifndef XRT_PROFILE_H_
#define XRT_PROFILE_H_

#include "xrt/deprecated/xrt.h"

#ifdef __cplusplus

#include <chrono>

namespace xrt { namespace profile {

/*!
 * @class user_range
 *
 * @brief
 * xrt::profile::user_range xrt::profile::user_range is used to track start and stop times
 * between two user defined points in host program and add them to profiling output for
 * visualization using post-processing tools.
 *
 * @details
 * By turning on various trace options in the xrt.ini file,
 * XRT applications will track events and generate files, which are turned into a
 * timeline visualization and summary via post-processing tools.  A user range is used
 * to explicitly add events measured from start to stop from the host code to that
 * timeline visualization and summary.
 * Usage -
 * 1. If a user_range is instantiated using the default constructor, no time is marked
 *    until the user calls start with two strings (label and tooltip)
 * 2. The user must call start() and stop() to mark ranges of time they are interested in.
 *    If stop() is not called, then any range being tracked lasts until the user_range object is destructed
 * 3. As a shortcut, the user can instantiate a user_range object passing two strings (label and tooltip).  This starts monitoring the range immediately
 *    The user_range object can be reused any number of times, by calling start()/stop() pairs in the host code
 * 4. Multiple sequential calls to start() ignore all but the first call
 * 5. Multiple sequential calls to stop() ignore all but the first call
 */
class user_range
{
private:
  uint32_t id;
  bool active;
public:
  /**
   * user_range() - Default Constructor for user range
   *
   * If this constructor is called, the object is created but the 
   * beginning of the range time is not marked.
   */
  XCL_DRIVER_DLLESPEC
  user_range();

  /**
   * user_range() - Constructor for user range with text
   *
   * @param label
   * The string that appears embedded on the waveform for this range
   * @param tooltip
   * The string that appears on the waveform when hovering over the range
   *
   * Construct an object and start keeping track of time immediately
   * upon construction.
   */
  XCL_DRIVER_DLLESPEC
  user_range(const char* label, const char* tooltip = nullptr);

  /**
   * user_range() - Copy constructor
   *
   * Explicitly removed as ranges are actively keeping track of real time
   *  and cannot be copied.
   */
  user_range(const user_range& rhs) = delete;

  /**
   * user_range() - Move constructor
   *
   * Explicitly removed as ranges are actively keeping track of real time
   *  and cannot be moved.
   */
  user_range(user_range&& rhs) = delete;

  /**
   * operator=()
   *
   * Explicitly removed as ranges should not be assigned
   */
  user_range& operator=(const user_range& rhs) = delete;
  user_range& operator=(user_range&& rhs) = delete;

  /**
   * ~user_range() - Destructor
   *
   * If the range is still active, mark this point as the end of the range
   */
  XCL_DRIVER_DLLESPEC
  ~user_range();

  /**
   * start() - Mark the start position of a user range
   *
   * @param label
   * The string that appears embedded on the waveform for this range
   * @param tooltip
   * The string that appears on the waveform when hovering over the range
   *
   * If the range is still actively recording time, end the current
   * range and start a new one.
   */
  XCL_DRIVER_DLLESPEC
  void
  start(const char* label, const char* tooltip = nullptr);

  /**
   * end() - Mark the end position of a user range
   *
   * If the range has already been ended, do nothing.  Otherwise
   * mark the end position of the range and stop tracking time.
   */
  XCL_DRIVER_DLLESPEC
  void
  end();
};

/*!
 * @class user_event
 *
 * @brief
 * xrt::profile::user_event mark a specific point in execution with a label for later visualization
 *
 * @details
 * A user event can be generated from inside host code and optionally tagged with a label.
 * These events are post-processed and represented on summary and trace waveforms as markers.
 */
class user_event
{
public:
  /**
   * user_event() - Constructor
   *
   * Create an object, but do not mark any particular time
   */
  XCL_DRIVER_DLLESPEC
  user_event();

  /**
   * ~user_event() - Destructor
   *
   * Destroy the object, but do not mark any particular time
   */
  XCL_DRIVER_DLLESPEC
  ~user_event();
    
  /**
   * mark() - Mark the current moment in time with a marker on the waveform
   *
   * @param label
   * An optional label that will be displayed on top of marker in waveform
   *
   */
  XCL_DRIVER_DLLESPEC
  void
  mark(const char* label = nullptr);

  /**
   * mark_time_ns() - Mark a custom moment in time with a marker on the waveform
   *
   * @param time_ns
   * Time duration in nanoseconds since appliction start.
   * This must be compatible with xrt_core::time_ns() API.
   * @param label
   * An optional label that will be displayed on top of marker in waveform
   *
   */
  XCL_DRIVER_DLLESPEC
  void
  mark_time_ns(const std::chrono::nanoseconds& time_ns, const char* label = nullptr);
};

} // end namespace profile
} // end namespace xrt

extern "C" {
#endif

/**
 * xrtURStart() - Mark the start time in the user code of a range
 *
 * @id:      A user supplied id to match starts and ends of ranges
 * @label:   The text to display in the waveform for this particular range
 * @tooltip: The text to display in the waveform when hovering over this range
 * Return:   none
 *
 */
XCL_DRIVER_DLLESPEC
void
xrtURStart(unsigned int id, const char* label, const char* tooltip = nullptr);

/**
 * xrtUREnd() - Mark the end time in the user code of a range
 *
 * @id:      A user supplied id to match starts and ends of ranges
 * Return:   none
 *
 */
XCL_DRIVER_DLLESPEC
void
xrtUREnd(unsigned int id);

/**
 * xrtUEMark() - Mark the current time as when something happened
 *
 * @label:   An optional label that is added to the marker in the waveform
 * Return:   none
 *
 */
XCL_DRIVER_DLLESPEC
void
xrtUEMark(const char* label);

/**
 * xrtUEMarkTimeNs() - Mark a custom time as when something happened
 *
 * @time_ns: Time duration in nanoseconds since application start.
 * This must be compatible with xrt_core::time_ns() API.
 * @label:   An optional label that is added to the marker in the waveform
 * Return:   none
 *
 */
XCL_DRIVER_DLLESPEC
void
xrtUEMarkTimeNs(unsigned long long int time_ns, const char* label);

#ifdef __cplusplus
}
#endif

#endif
