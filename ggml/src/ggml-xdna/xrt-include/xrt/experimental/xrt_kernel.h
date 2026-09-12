// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2021 Xilinx, Inc. All rights reserved.
// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
#ifndef XRT_EXPERIMENTAL_KERNEL_H
#define XRT_EXPERIMENTAL_KERNEL_H
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_hw_context.h"

#include "xrt/detail/config.h"

#ifdef __cplusplus
# include "xrt/detail/pimpl.h"
# include <chrono>
# include <condition_variable>
#endif

#ifdef __cplusplus
namespace xrt {

/**
 * class runlist - A class to manage a list of xrt::run objects
 *
 * @brief
 * This class is used to manage a list of xrt::run objects such
 * that they can be executed atomically in the order they are added
 * to the list.
 *
 * @details
 * Run objects are added to the list using the add() method. A runlist
 * is submitted for execution using the execute() method. Once the
 * list is executing, no more run objects can be added until execution
 * of the current last run object has completed. The list can be reset
 * using the reset() method, which will clear all run objects, or the
 * list can be reused by calling execute() again maybe with additional
 * run objects.
 *
 * There is no support for removing individual run objects from the
 * list.
 */
class runlist_impl;
class runlist : public detail::pimpl<runlist_impl>
{
public:
  /**
   * class command_error - exception for abnormal runlist execution
   *
   * Captures the failing run object and the state at which it failed.
   */
  class command_error_impl;
  class command_error : public detail::pimpl<command_error_impl>, public std::exception
  {
  public:
    XRT_API_EXPORT
    command_error(const xrt::run& run, ert_cmd_state state, const std::string& what);

    /**
     * get_run() - run object that failed
     */
    XRT_API_EXPORT
    xrt::run
    get_run() const;

    /**
     * get_command_state() - command state upon completion
     */
    XRT_API_EXPORT
    ert_cmd_state
    get_command_state() const;

    XRT_API_EXPORT
    const char*
    what() const noexcept override;
  };

  /**
   * aie_error - exception for AIE abnormal command execution
   *
   * This exception provides access to context health information.
   */
  class aie_error : public command_error
  {
    using command_error::command_error;
    template <typename T> using span = xrt::detail::span<T>;
  public:
    XRT_API_EXPORT
    aie_error(const xrt::run& run, ert_cmd_state state, const std::string& what);
    /**
     * Get the raw context health data
     * The data format is not necessarily ABI compatible so should
     * not be used in deployed applications.
     */
    XRT_API_EXPORT
    span<const uint32_t>
    data() const;
  };

public:
  /**
   * runlist() - Construct empty runlist object
   *
   * Can be used as lvalue in assignment.
   *
   * It is undefined behavior to use a default constructed runlist
   * for anything but assignment.
   */
  runlist() = default;

  /**
   * runlist - Constructor
   *
   * A runlist is associated with a specific hwctx. All run objects
   * added to the list must be associated with kernel objects that are
   * created in specified hwctx.
   *
   * Throws is invariant per run object hwctx requirement is violated.
   */
  XRT_API_EXPORT
  explicit
  runlist(const xrt::hw_context& hwctx);

  /**
   * runlist - Destructor
   *
   * The destructor of the runlist clears the association with the run
   * objects, but does not check for runlist state or wait for run
   * object completion.  It is the caller's responsibility to ensure
   * that the runlist is not executing when the destructor is called
   * or beware that there may be run objects still executing.
   */
  XRT_API_EXPORT
  ~runlist();

  /**
   * add() - Add a run object to the list
   *
   * The run object is added to the end of the list.  A run object can
   * only be added to a runlist once and only a runlist which must be
   * associated with the same hwctx as the kernel from which the run
   * object was created.
   *
   * It is undefined behavior to add a run object to a runlist which
   * is currently executing or to explicitly start (xrt::run::start())
   * a run object that is part of a runlist.
   *
   * It is the caller's responsibility to ensure that the runlist is
   * not executing when this method is called.  This can be done by
   * calling the wait() method on the runlist object.
   *
   * The state of run objects in a runlist should be ignored.  The
   * `xrt::run::state()` function is not guaranteed to reflect the
   * actual run object state and cannot be called for run objects that
   * are part of a runlist. If any run object fails to complete
   * successfully, `xrt::runlist::wait()` will throw an exception with
   * the failed run object and it's fail state.
   *
   * Throws if the kernel from which the run object was created does
   * not match the hwctx from which the runlist was created.
   *
   * Throws if the run object is already part of a runlist.
   *
   * Throws if runlist is executing.
   */
  XRT_API_EXPORT
  void
  add(const xrt::run& run);

  /**
   * add() - Move a run object into the list
   *
   * Same behavior as copy add()
   */
  XRT_API_EXPORT
  void
  add(xrt::run&& run);

  /**
   * execute() - Execute the runlist
   *
   * The runlist is submitted for execution. The run objects in the
   * list are executed atomically in the order they were added to the
   * list.
   *
   * Executing an empty runlist is a no-op.
   *
   * Throws if runlist is already executing.
   */
  XRT_API_EXPORT
  void
  execute();

  /**
   * wait() - Wait for the runlist to complete
   *
   * @param timeout
   *  Timeout for wait.  A value of 0, implies block until all run
   *  objects have completed successfully.
   * @return
   *  std::cv_status::no_timeout if list has completed execution of
   *  all run objects, std::cv_status::timeout if the timeout expired
   *  prior to all run objects completing.
   *
   * Completion of a runlist execution means that all run objects have
   * completed succesfully.  If any run object in the list fails to
   * complete successfully, the function throws
   * `xrt::runlist::command_error` with the failed run object and
   * state.
   */
  XRT_API_EXPORT
  std::cv_status
  wait(const std::chrono::milliseconds& timeout) const;

  /**
   * wait() - Wait for the runlist to complete
   *
   * This is a convenience method that calls wait() with a timeout of 0.
   *
   * The function blocks until all run objects have completed or
   * throws if any run object fails to complete successfully.
   */
  void
  wait() const
  {
    wait(std::chrono::milliseconds(0));
  }

  /**
   * state() - Check the current state of a runlist object
   *
   * @return
   *  Current state of this run object.
   *
   * The state values are defined in ``include/ert.h``
   * The state of an empty runlist is ERT_CMD_STATE_COMPLETED.
   *
   * This function is the preferred way to poll a runlist for
   * for completion.
   */
  XRT_API_EXPORT
  ert_cmd_state
  state() const;

  /**
   * DEPRECATED, prefer `state()`.
   *
   * Poll the runlist for completion.
   *
   * @return 0 if command list is still running, any other value
   *  indicates completion.
   *
   * Completion of a runlist execution means that all run objects have
   * completed succesfully.  If any run object in the list fails to
   * complete successfully, the function throws
   * `xrt::runlist::command_error` with the failed run object and
   * state.
   */
  XRT_API_EXPORT
  int
  poll() const;
  
  /**
   * reset() - Reset the runlist
   *
   * The runlist is reset to its initial state. All run objects are
   * removed from the list.
   *
   * It is the caller's responsibility to ensure that the runlist is
   * not executing when this method is called.  This can be done by
   * calling the wait() method on the runlist object.
   *
   * Throws if runlist is executing.
   */
  XRT_API_EXPORT
  void
  reset();
};

} // namespace xrt

#endif // __cplusplus
#endif // XRT_EXPERIMENTAL_KERNEL_H

  
