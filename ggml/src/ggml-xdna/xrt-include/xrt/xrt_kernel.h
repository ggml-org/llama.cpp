/*
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (C) 2020-2022 Xilinx, Inc. All rights reserved.
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
 */
#ifndef XRT_KERNEL_H_
#define XRT_KERNEL_H_
#include "xrt/detail/config.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_uuid.h"

#include "xrt/detail/ert.h"
#include "xrt/detail/span.h"
#include "xrt/deprecated/xrt.h"

#ifdef __cplusplus
# include "xrt/experimental/xrt_exception.h"
# include "xrt/experimental/xrt_fence.h"
# include "xrt/experimental/xrt_hw_context.h"
# include <chrono>
# include <condition_variable>
# include <cstdint>
# include <functional>
# include <memory>
# include <vector>
#endif

/**
 * typedef xrtKernelHandle - opaque kernel handle
 *
 * A kernel handle is obtained by opening a kernel.  Clients
 * pass this kernel handle to APIs that operate on a kernel.
 */
typedef void * xrtKernelHandle;

/**
 * typedef xrtRunHandle - opaque handle to a specific kernel run
 *
 * A run handle is obtained by running a kernel.  Clients
 * use a run handle to check or wait for kernel completion.
 */
typedef void * xrtRunHandle;  // NOLINT

#ifdef __cplusplus

namespace xrt {

/*!
 * @class autostart
 *
 * @brief
 * xrt::autostart is a specific type used as template argument.
 *
 * @details
 * When implicitly starting a kernel through templated operator, the
 * first argument can be specified as xrt::autostart indicating the
 * number of iterations the kernel run should perform.
 *
 * The default iterations, or xrt::autostart constructed with the
 * value 0, represents a for-ever running kernel.
 *
 * When a kernel is auto-started, the running kernel can be manipulated
 * through a \ref xrt::mailbox object provided the kernel is synthesized
 * with mailbox.
 *
 * The counted auto-restart feature is supported only for kernels that
 * are specifically synthesized with counted auto restart.  The
 * default value of xrt::autostart is supported default for AP_CTRL_HS
 * and AP_CTRL_CHAIN.
 *
 * Currently autostart is only supported for kernels with one compute
 * unit which must be opened in exclusive mode.
 */
struct autostart
{
  unsigned int iterations = 0;
};

class kernel;

/*!
 * @class run
 *
 * @brief
 * xrt::run represents one execution of a kernel
 *
 * @details
 * The run handle can be explicitly constructed from a kernel object
 * or implicitly constructed from starting a kernel execution.
 *
 * A run handle can be re-used to execute the same kernel again.
 */
class run_impl;
class run
{
public:
  /**
   * command_error - exception for abnormal command execution
   *
   * Used by ``wait2()`` when command completes unsuccessfully.
   */
  class command_error_impl;
  class command_error : public detail::pimpl<command_error_impl>, public std::exception
  {
  public:
    XRT_API_EXPORT
    command_error(ert_cmd_state state, const std::string& what);

    XRT_API_EXPORT
    command_error(const xrt::run& run, const std::string& what);

    /**
     * get_command_state() - command state upon completion
     */
    XRT_API_EXPORT
    ert_cmd_state
    get_command_state() const;

    XRT_API_EXPORT
    const char*
    what() const noexcept override;

  private:
    // This member is a mistake, but cannot remove it without breaking ABI
    std::shared_ptr<command_error_impl> m_impl;
  };

  /**
   * aie_error - exception for AIE abnormal command execution
   *
   * This exception provides access to context health information.
   */
  class aie_error : public command_error
  {
  public:
    template <typename T> using span = xrt::detail::span<T>;
  public:
    /**
     * aie_error() - Constructor must hold on to run object
     */
    XRT_API_EXPORT
    aie_error(const xrt::run& run, const std::string& what);
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
   * run() - Construct empty run object
   *
   * Can be used as lvalue in assignment.
   *
   * It is undefined behavior to use a default constructed run object
   * for anything but assignment.
   */
  run() = default;

  /**
   * run() - Construct run object from a kernel object
   *
   * @param krnl: Kernel object representing the kernel to execute
   */
  XRT_API_EXPORT
  explicit
  run(const kernel& krnl);

  /**
   * run() - Copy ctor
   *
   * Performs shallow copy, sharing data with the source
   */
  run(const run&) = default;

  /**
   * run() - Move ctor
   */
  run(run&&) = default;

  /**
   * ~run() - Destruct run object
   */
  XRT_API_EXPORT
  ~run();

  /**
   * operator= () - Copy assignment
   *
   * Performs shallow copy assignment, sharing data with the source
   */
  run&
  operator=(const run&) = default;

  /**
   * operator= () - Move assignment
   */
  run&
  operator=(run&&) = default;

  /**
   * start() - Start one execution of a run.
   *
   * This function is asynchronous, run status must be expclicit checked
   * or ``wait()`` must be used to wait for the run to complete.
   */
  XRT_API_EXPORT
  void
  start();

  /**
   * start() - Start auto-restart execution of a run
   *
   * @param iterations
   *   Number of times to iterate the same run.
   *
   * An iteration count of zero means that the kernel should run
   * forever, or until explicitly stopped using ``stop()``.
   *
   * This function is asynchronous, run status must be expclicit
   * checked or ``wait()`` must be used to wait for the run to
   * complete.
   *
   * The kernel run object is complete only after all iterations have
   * completed, or until run object has been explicitly stopped.
   *
   * Changing kernel arguments ``set_arg()`` while the kernel is running
   * has undefined behavior.  To synchronize change of arguments, please
   * use \ref xrt::mailbox.
   *
   * Currently autostart is only supported for kernels with one
   * compute unit which must be opened in exclusive mode.
   */
  XRT_API_EXPORT
  void
  start(const autostart& iterations);

  /**
   * stop() - Stop kernel run object at next safe iteration
   *
   * If the kernel run object has been started by specifying
   * an iteration count or by specifying default iteration count,
   * then this function can be used to stop the iteration early.
   *
   * The function is synchronous and waits for the kernel
   * run object to complete.
   *
   * If the kernel is not iterating, then calling this funciton
   * is the same as calling ``wait()``.
   */
  XRT_API_EXPORT
  void
  stop();

  /**
   * abort() - Abort a run object that has been started
   *
   * @return
   *  State of aborted command
   *
   * If the run object has been sent to scheduler for execution, then
   * this function can be used to abort the scheduled command.
   *
   * The function is synchronous and will wait for abort to complete.
   * The return value is the state of the aborted command.
   */
  XRT_API_EXPORT
  ert_cmd_state
  abort();

  /**
   * wait() - Wait for a run to complete execution
   *
   * @param timeout
   *  Timeout for wait (default block till run completes)
   * @return
   *  Command state upon return of wait, or ERT_CMD_STATE_TIMEOUT
   *  if timeout exceeded.
   *
   * The default timeout of 0ms indicates blocking until run completes.
   *
   * The current thread will block until the run completes or timeout
   * expires. Completion does not guarantee success, the run status
   * should be checked by using ``state``.
   *
   * If specified time out is exceeded, the function returns with
   * ERT_CMD_STATE_TIMEOUT, it is the callers responsibility to abort
   * the run if it continues to time out.
   *
   * The current implementation of this API can mask out the timeout
   * of this run so that the call either never returns or doesn't
   * return until the run completes by itself. This can happen if
   * other runs are continuosly completing within the specified
   * timeout for this run.  If the device is otherwise idle, or if the
   * time between run completion exceeds the specified timeout, then
   * this function will identify the timeout.
   */
  XRT_API_EXPORT
  ert_cmd_state
  wait(const std::chrono::milliseconds& timeout = std::chrono::milliseconds{0}) const;

  /**
   * wait() - Wait for specified milliseconds for run to complete
   *
   * @param timeout_ms
   *  Timeout in milliseconds
   * @return
   *  Command state upon return of wait, or ERT_CMD_STATE_TIMEOUT
   *  if timeout exceeded.
   *
   * The default timeout of 0ms indicates blocking until run completes.
   *
   * The current thread will block until the run completes or timeout
   * expires. Completion does not guarantee success, the run status
   * should be checked by using ``state``.
   *
   * If specified time out is exceeded, the function returns with
   * ERT_CMD_STATE_TIMEOUT, it is the callers responsibility to abort
   * the run if it continues to time out.
   *
   * The current implementation of this API can mask out the timeout
   * of this run so that the call either never returns or doesn't
   * return until the run completes by itself. This can happen if
   * other runs are continuosly completing within the specified
   * timeout for this run.  If the device is otherwise idle, or if the
   * time between run completion exceeds the specified timeout, then
   * this function will identify the timeout.
   */
  ert_cmd_state
  wait(unsigned int timeout_ms) const
  {
    return wait(timeout_ms * std::chrono::milliseconds{1});
  }

  /**
   * wait2() - Wait for specified milliseconds for run to complete
   *
   * @param timeout
   *  Timeout for wait (default block until run completes)
   * @return
   *  std::cv_status::no_timeout when command completes successfully.
   *  std::cv_status::timeout when wait timed out without command
   *  completing.
   *
   * Successful command completion means that the command state is
   * ERT_CMD_STATE_COMPLETED.  All other command states result in this
   * function throwing ``command_error`` exception with the command
   * state embedded in the exception.
   *
   * Throws ``xrt::run::command_error`` on abnormal command termination.
   *
   * The current thread blocks until the run successfully completes or
   * timeout expires. A return code of std::cv_state::no_timeout
   * guarantees that the command completed successfully.
   *
   * If specified time out is exceeded, the function returns with
   * std::cv_status::timeout, it is the callers responsibility to abort
   * the run if it continues to time out.
   *
   * The current implementation of this API can mask out the timeout
   * of this run so that the call either never returns or doesn't
   * return until the run completes by itself. This can happen if
   * other runs are continuosly completing within the specified
   * timeout for this run.  If the device is otherwise idle, or if the
   * time between run completion exceeds the specified timeout, then
   * this function will identify the timeout.
   */
  XRT_API_EXPORT
  std::cv_status
  wait2(const std::chrono::milliseconds& timeout) const;

  /**
   * wait2() - Wait for successful command completion
   *
   * Successful command completion means that the command state is
   * ERT_CMD_STATE_COMPLETED.  All other command states result in this
   * function throwing ``command_error`` exception with the command
   * state embedded in the exception.
   *
   * Throws ``xrt::run::command_error`` on abnormal command termination.
   */
  void
  wait2() const
  {
    wait2(std::chrono::milliseconds{0});
  }

  /**
   * state() - Check the current state of a run object
   *
   * @return
   *  Current state of this run object
   *
   * The state values are defined in ``include/ert.h``
   */
  XRT_API_EXPORT
  ert_cmd_state
  state() const;

  /**
   * return_code() - Get the return code from PS kernel
   *
   * @return
   *  Return code from PS kernel run
   */
  XRT_API_EXPORT
  uint32_t
  return_code() const;

  /**
   * add_callback() - Add a callback function for run state
   *
   * @param state       State to invoke callback on
   * @param callback    Callback function
   * @param data        User data to pass to callback function
   *
   * The function is called when the run object changes state to
   * argument state or any error state.  Only
   * ``ERT_CMD_STATE_COMPLETED`` is supported currently.
   *
   * The function object's first parameter is a unique 'key'
   * for this xrt::run object implementation on which the callback
   * was added. This 'key' can be used to identify an actual run
   * object that refers to the implementaion that is maybe shared
   * by multiple xrt::run objects.
   *
   * Any number of callbacks are supported.
   *
   * Execution of a run object with callback functions is referred to
   * as managed execution.  Managed execution is supported on Alveo
   * style platforms only. If targeted platform does not support
   * managed execution, then an exception is thrown when the run
   * object is submitted for execution.
   */
  XRT_API_EXPORT
  void
  add_callback(ert_cmd_state state,
               std::function<void(const void*, ert_cmd_state, void*)> callback,
               void* data);

  /**
   * operator bool() - Check if run handle is valid
   *
   * @return
   *   True if run is associated with kernel object, false otherwise
   */
  explicit
  operator bool() const
  {
    return handle != nullptr;
  }

  /**
   * operator < () - Weak ordering
   *
   * @param rhs
   *  Object to compare with
   * @return
   *  True if object is ordered less that compared with other
   */
  bool
  operator < (const xrt::run& rhs) const
  {
    return handle < rhs.handle;
  }

  /**
   * set_arg() - Set a specific kernel scalar argument for this run
   *
   * @param index
   *  Index of kernel argument to update
   * @param arg
   *  The scalar argument value to set.
   *
   * Use this API to explicit set or change a kernel argument prior
   * to starting kernel execution.  After setting arguments, the
   * kernel can be started using ``start()`` on the run object.
   *
   * See also ``operator()`` to set all arguments and start kernel.
   */
  template <typename ArgType>
  void
  set_arg(int index, ArgType&& arg)
  {
    set_arg_at_index(index, &arg, sizeof(arg));
  }

  /**
   * set_arg() - Set a specific kernel global argument for a run
   *
   * @param index
   *  Index of kernel argument to set
   * @param boh
   *  The global buffer argument value to set (lvalue).
   *
   * Use this API to explicit set or change a kernel argument prior
   * to starting kernel execution.  After setting arguments, the
   * kernel can be started using ``start()`` on the run object.
   *
   * See also ``operator()`` to set all arguments and start kernel.
   */
  void
  set_arg(int index, xrt::bo& boh)
  {
    set_arg_at_index(index, boh);
  }

  /**
   * set_arg - xrt::bo variant for const lvalue
   */
  void
  set_arg(int index, const xrt::bo& boh)
  {
    set_arg_at_index(index, boh);
  }

  /**
   * set_arg - xrt::bo variant for rvalue
   */
  void
  set_arg(int index, xrt::bo&& boh)
  {
    set_arg_at_index(index, boh);
  }

  ///@cond
  /// Experimental in 2023.2
  XRT_API_EXPORT
  void
  submit_wait(const xrt::fence& fence);
  
  XRT_API_EXPORT
  void
  submit_signal(const xrt::fence& fence);
  ///@endcond

  /**
   * set_arg - set named argument
   *
   * @param argnm
   *   Name of kernel argument
   * @param argvalue
   *   Argument value
   *
   * Throws if specified argument name doesn't match kernel
   * specification. Throws if argument value is incompatible with
   * specified argument
   */
  template <typename ArgType>
  void
  set_arg(const std::string& argnm, ArgType&& argvalue)
  {
    auto index = get_arg_index(argnm);
    set_arg(index, std::forward<ArgType>(argvalue));
  }

  /**
   * udpdate_arg() - Asynchronous update of scalar kernel global argument
   *
   * @param index
   *  Index of kernel argument to update
   * @param arg
   *  The scalar argument value to set.
   *
   * Use this API to asynchronously update a specific scalar argument
   * of the kernel associated with the run object.
   *
   * This API is only supported on Edge.
   */
  template <typename ArgType>
  void
  update_arg(int index, ArgType&& arg)
  {
    update_arg_at_index(index, &arg, sizeof(arg));
  }

  /**
   * update_arg() - Asynchronous update of kernel global argument for a run
   *
   * @param index
   *  Index of kernel argument to update
   * @param boh
   *  The global buffer argument value to set.
   *
   * Use this API to asynchronously update a specific kernel
   * argument of an existing run.
   *
   * This API is only supported on Edge.
   */
  void
  update_arg(int index, const xrt::bo& boh)
  {
    update_arg_at_index(index, boh);
  }

  /**
   * operator() - Set all kernel arguments and start the run
   *
   * @param args
   *  Kernel arguments
   *
   * Use this API to explicitly set all kernel arguments and
   * start kernel execution.
   */
  template<typename ...Args>
  void
  operator() (Args&&... args)
  {
    set_arg(0, std::forward<Args>(args)...);
    start();
  }

  /**
   * operator() - Set all kernel arguments and start the run
   *
   * @param count
   *  Iteration count specifying number of iterations of the run
   * @param args
   *  Kernel arguments
   *
   * Use this API to explicitly set all kernel arguments and start
   * kernel execution for specified number of iterations.
   *
   * An iteration count of '1' invokes the kernel once and is the same
   * as calling the operator without specifying ``autostart``.
   *
   * The run is complete only after all iterations have completed or
   * when the kernel has been explicitly stopped using ``stop()``.
   *
   * Currently autostart is only supported for kernels with one
   * compute unit which must be opened in exclusive mode.
   */
  template<typename ...Args>
  void
  operator() (autostart&& count, Args&&... args)
  {
    set_arg(0, std::forward<Args>(args)...);
    start(count);
  }

  /**
   * get_ctrl_scratchpad_bo() - Get the ctrl scratchpad bo object
   * 
   * NPU uses ctrl scratchpad memory to store control state data.
   * This memory is created by XRT based on ELF used to create xrt::kernel
   * The API returns the buffer object (bo) created by XRT allowing
   * applications to read from or write to it.
   * This API is only valid for run objects associated with an ELF.
   * 
   * Throws if control scratchpad section is not absent in ELF or
   * if any error occurs while retrieving the bo
   */
  XRT_API_EXPORT
  xrt::bo
  get_ctrl_scratchpad_bo() const;

public:
  /// @cond
  const std::shared_ptr<run_impl>&
  get_handle() const
  {
    return handle;
  }

  // run() - Construct run object from a pimpl
  explicit
  run(std::shared_ptr<run_impl> impl)
    : handle(std::move(impl))
  {}

  // backdoor access to command packet
  XRT_API_EXPORT
  ert_packet*
  get_ert_packet() const;
  /// @endcond

public:
  // Use at your own risk, prefer documented type-safe arguments
  void
  set_arg(int index, const void* value, size_t bytes)
  {
    set_arg_at_index(index, value, bytes);
  }

  // Use at your own risk, prefer documented type-safe arguments
  void
  update_arg(int index, const void* value, size_t bytes)
  {
    update_arg_at_index(index, value, bytes);
  }

private:
  std::shared_ptr<run_impl> handle;

  XRT_API_EXPORT
  int
  get_arg_index(const std::string& argnm) const;

  XRT_API_EXPORT
  void
  set_arg_at_index(int index, const void* value, size_t bytes);

  XRT_API_EXPORT
  void
  set_arg_at_index(int index, const xrt::bo&);

  XRT_API_EXPORT
  void
  update_arg_at_index(int index, const void* value, size_t bytes);

  XRT_API_EXPORT
  void
  update_arg_at_index(int index, const xrt::bo&);

  template<typename ArgType, typename ...Args>
  void
  set_arg(int argno, ArgType&& arg, Args&&... args)
  {
    set_arg(argno, std::forward<ArgType>(arg));
    set_arg(++argno, std::forward<Args>(args)...);
  }
};


/*!
 * @class kernel
 *
 * A kernel object represents a set of instances matching a specified name.
 * The kernel is created by finding matching kernel instances in the
 * currently loaded xclbin.
 *
 * Most interaction with kernel objects are through \ref xrt::run objects created
 * from the kernel object to represent an execution of the kernel
 */
class kernel_impl;
class kernel
{
public:
  /**
   * cu_access_mode - compute unit access mode
   *
   * @var shared
   *  CUs can be shared between processes
   * @var exclusive
   *  CUs are owned exclusively by this process
   */
  enum class cu_access_mode : uint8_t { exclusive = 0, shared = 1, none = 2 };

  /**
   * kernel() - Construct for empty kernel
   *
   * It is undefined behavior to use a default constructed kernel object
   * for anything but assignment.
   */
  kernel() = default;

  /**
   * kernel() - Constructor from a device and xclbin
   *
   * @param device
   *  Device on which the kernel should execute
   * @param xclbin_id
   *  UUID of the xclbin with the kernel
   * @param name
   *  Name of kernel to construct
   * @param mode
   *  Open the kernel instances with specified access (default shared)
   *
   * The kernel name must uniquely identify compatible kernel
   * instances (compute units).  Optionally specify which kernel
   * instance(s) to open using
   * "kernelname:{instancename1,instancename2,...}" syntax.  The
   * compute units are default opened with shared access, meaning that
   * other kernels and other process will have shared access to same
   * compute units.
   */
  XRT_API_EXPORT
  kernel(const xrt::device& device, const xrt::uuid& xclbin_id, const std::string& name,
         cu_access_mode mode = cu_access_mode::shared);


  /// @cond
  /// Experimental in 2022.2, 2023.1, 2023.3
  XRT_API_EXPORT
  kernel(const xrt::hw_context& ctx, const std::string& name);
  /// @endcond

  /// @cond
  /// Deprecated construtor for exclusive access
  kernel(const xrt::device& device, const xrt::uuid& xclbin_id, const std::string& name, bool ex)
    : kernel(device, xclbin_id, name, ex ? cu_access_mode::exclusive : cu_access_mode::shared)
  {}

  /**
   * Obsoleted construction from xclDeviceHandle
   */
  XRT_API_EXPORT
  kernel(xclDeviceHandle dhdl, const xrt::uuid& xclbin_id, const std::string& name,
         cu_access_mode mode = cu_access_mode::shared);
  /// @endcond

  /**
   * kernel() - Copy ctor
   *
   * Performs shallow copy, sharing data with the source
   */
  kernel(const kernel&) = default;

  /**
   * kernel() - Move ctor
   */
  kernel(kernel&&) = default;

  /**
   * Destructor for kernel - needed for tracing
   */
  XRT_API_EXPORT
  ~kernel();

  /**
   * operator= () - Copy assignment
   *
   * Performs shallow copy assignment, sharing data with the source
   */
  kernel&
  operator=(const kernel&) = default;

  /**
   * operator= () - Move assignment
   */
  kernel&
  operator=(kernel&&) = default;

  /**
   * operator() - Invoke the kernel function
   *
   * @param args
   *  Kernel arguments
   * @return
   *  Run object representing this kernel function invocation
   */
  template<typename ...Args>
  run
  operator() (Args&&... args)
  {
    run r(*this);
    r(std::forward<Args>(args)...);
    return r;
  }

  /**
   * group_id() - Get the memory bank group id of an kernel argument
   *
   * @param argno
   *  The argument index
   * @return
   *  The memory group id to use when allocating buffers (see xrt::bo)
   *
   * The function throws if the group id is ambigious.
   */
  XRT_API_EXPORT
  int
  group_id(int argno) const;

  /**
   * offset() - Get the offset of kernel argument
   *
   * @param argno
   *  The argument index
   * @return
   *  The kernel register offset of the argument with specified index
   *
   * Use with ``read_register()`` and ``write_register()`` if manually
   * reading or writing kernel registers for explicit arguments.
   */
  XRT_API_EXPORT
  uint32_t
  offset(int argno) const;

  [[deprecated("Please use user-managed xrt::ip "
               "for read and write register functionality")]]
  XRT_API_EXPORT
  void
  write_register(uint32_t offset, uint32_t data);

  [[deprecated("It is not recommended to use read_register() with XRT "
               "managed kernels.  Please use user-managed xrt::ip for read and "
               "write register functionality")]]
  XRT_API_EXPORT
  uint32_t
  read_register(uint32_t offset) const;

  /**
   * get_name() - Return the name of the kernel
   */
  XRT_API_EXPORT
  std::string
  get_name() const;

  /**
   * get_xclbin() - Return the xclbin containing the kernel
   */
  XRT_API_EXPORT
  xrt::xclbin
  get_xclbin() const;

public:
  /// @cond
  const std::shared_ptr<kernel_impl>&
  get_handle() const
  {
    return handle;
  }

  kernel(std::shared_ptr<kernel_impl> impl)
    : handle(std::move(impl))
  {}
  /// @endcond

private:
  std::shared_ptr<kernel_impl> handle;
};

/// @cond
// Undocumented experimental API subject to be replaced
XRT_API_EXPORT
void
set_read_range(const xrt::kernel& kernel, uint32_t start, uint32_t size);
/// @endcond


} // namespace xrt

/// @cond
extern "C" {
#endif

/**
 * xrtPLKernelOpen() - Open a PL kernel and obtain its handle.
 *
 * @deviceHandle:  Handle to the device with the kernel
 * @xclbinId:      The uuid of the xclbin with the specified kernel.
 * @name:          Name of kernel to open.
 * Return:         Handle representing the opened kernel.
 *
 * The kernel name must uniquely identify compatible kernel instances
 * (compute units).  Optionally specify which kernel instance(s) to
 * open using "kernelname:{instancename1,instancename2,...}" syntax.
 * The compute units are opened with shared access, meaning that
 * other kernels and other process will have shared access to same
 * compute units.  If exclusive access is needed then open the
 * kernel using @xrtPLKernelOpenExclusve().
 *
 * An xclbin with the specified kernel must have been loaded prior
 * to calling this function. An XRT_NULL_HANDLE is returned on error
 * and errno is set accordingly.
 *
 * A kernel handle is thread safe and can be shared between threads.
 */
XRT_API_EXPORT
xrtKernelHandle
xrtPLKernelOpen(xrtDeviceHandle deviceHandle, const xuid_t xclbinId, const char *name);

/**
 * xrtPLKernelOpenExclusive() - Open a PL kernel and obtain its handle.
 *
 * @deviceHandle:  Handle to the device with the kernel
 * @xclbinId:      The uuid of the xclbin with the specified kernel.
 * @name:          Name of kernel to open.
 * Return:         Handle representing the opened kernel.
 *
 * Same as @xrtPLKernelOpen(), but opens compute units with exclusive
 * access.  Fails if any compute unit is already opened with either
 * exclusive or shared access.
 */
XRT_API_EXPORT
xrtKernelHandle
xrtPLKernelOpenExclusive(xrtDeviceHandle deviceHandle, const xuid_t xclbinId, const char *name);

/**
 * xrtKernelClose() - Close an opened kernel
 *
 * @kernelHandle: Handle to kernel previously opened with xrtKernelOpen
 * Return:        0 on success, -1 on error
 */
XRT_API_EXPORT
int
xrtKernelClose(xrtKernelHandle kernelHandle);

/**
 * xrtKernelArgGroupId() - Acquire bank group id for kernel argument
 *
 * @kernelHandle: Handle to kernel previously opened with xrtKernelOpen
 * @argno:        Index of kernel argument
 * Return:        Group id or negative error code on error
 *
 * A valid group id is a non-negative integer.  The group id is required
 * when constructing a buffer object.
 *
 * The kernel argument group id is ambigious if kernel has multiple kernel
 * with different connectivity for specified argument.  In this case the
 * API returns error.
 */
XRT_API_EXPORT
int
xrtKernelArgGroupId(xrtKernelHandle kernelHandle, int argno);

/**
 * xrtKernelArgOffset() - Get the offset of kernel argument
 *
 * @khdl:   Handle to kernel previously opened with xrtKernelOpen
 * @argno:  Index of kernel argument
 * Return:  The kernel register offset of the argument with specified index
 *
 * Use with ``xrtKernelReadRegister()`` and ``xrtKernelWriteRegister()``
 * if manually reading or writing kernel registers for explicit arguments.
 */
XRT_API_EXPORT
uint32_t
xrtKernelArgOffset(xrtKernelHandle khdl, int argno);

/**
 * xrtKernelReadRegister() - Read data from kernel address range
 *
 * @kernelHandle: Handle to kernel previously opened with xrtKernelOpen
 * @offset:       Offset in register space to read from
 * @datap:        Pointer to location where to write data
 * Return:        0 on success, errcode otherwise
 *
 * The kernel must be associated with exactly one kernel instance
 * (compute unit), which must be opened for exclusive access.
 */
XRT_API_EXPORT
int
xrtKernelReadRegister(xrtKernelHandle kernelHandle, uint32_t offset, uint32_t* datap);

/**
 * xrtKernelWriteRegister() - Write to the address range of a kernel
 *
 * @kernelHandle: Handle to kernel previously opened with xrtKernelOpen
 * @offset:       Offset in register space to write to
 * @data:         Data to write
 * Return:        0 on success, errcode otherwise
 *
 * The kernel must be associated with exactly one kernel instance
 * (compute unit), which must be opened for exclusive access.
 */
XRT_API_EXPORT
int
xrtKernelWriteRegister(xrtKernelHandle kernelHandle, uint32_t offset, uint32_t data);

/**
 * xrtKernelRun() - Start a kernel execution
 *
 * @kernelHandle: Handle to the kernel to run
 * @...:          Kernel arguments
 * Return:        Run handle which must be closed with xrtRunClose()
 *
 * A run handle is specific to one execution of a kernel.  Once
 * execution completes, the run handle can be re-used to execute the
 * same kernel again.  When no longer needed, then run handle must be
 * closed with xrtRunClose().
 */
XRT_API_EXPORT
xrtRunHandle
xrtKernelRun(xrtKernelHandle kernelHandle, ...);

/**
 * xrtRunOpen() - Open a new run handle for a kernel without starting kernel
 *
 * @kernelHandle: Handle to the kernel to associate the run handle with
 * Return:        Run handle which must be closed with xrtRunClose()
 *
 * The handle can be used repeatedly to start an execution of the
 * associated kernel.  This API allows application to manage run
 * handles without maintaining corresponding kernel handle.
 */
XRT_API_EXPORT
xrtRunHandle
xrtRunOpen(xrtKernelHandle kernelHandle);

/**
 * xrtRunSetArg() - Set a specific kernel argument for this run
 *
 * @rhdl:       Handle to the run object to modify
 * @index:      Index of kernel argument to set
 * @...:        The argument value to set.
 * Return:      0 on success, -1 on error
 *
 * Use this API to explicitly set specific kernel arguments prior
 * to starting kernel execution.  After setting all arguments, the
 * kernel execution can be start with xrtRunStart()
 */
XRT_API_EXPORT
int
xrtRunSetArg(xrtRunHandle rhdl, int index, ...);

/**
 * xrtRunUpdateArg() - Asynchronous update of kernel argument
 *
 * @rhdl:       Handle to the run object to modify
 * @index:      Index of kernel argument to update
 * @...:        The argument value to update.
 * Return:      0 on success, -1 on error
 *
 * Use this API to asynchronously update a specific kernel
 * argument of an existing run.
 *
 * This API is only supported on Edge.
 */
XRT_API_EXPORT
int
xrtRunUpdateArg(xrtRunHandle rhdl, int index, ...);

/**
 * xrtRunStart() - Start existing run handle
 *
 * @rhdl:       Handle to the run object to start
 * Return:      0 on success, -1 on error
 *
 * Use this API when re-using a run handle for more than one execution
 * of the kernel associated with the run handle.
 */
XRT_API_EXPORT
int
xrtRunStart(xrtRunHandle rhdl);

/**
 * xrtRunWait() - Wait for a run to complete
 *
 * @rhdl:       Handle to the run object to start
 * Return:      Run command state for completed run,
 *              or ERT_CMD_STATE_ABORT on error
 *
 * Blocks current thread until job has completed
 */
XRT_API_EXPORT
enum ert_cmd_state
xrtRunWait(xrtRunHandle rhdl);

/**
 * xrtRunWait() - Wait for a run to complete
 *
 * @rhdl:       Handle to the run object to start
 * @timeout_ms: Timeout in millisecond
 * Return:      Run command state for completed run, or
 *              current status if timeout.
 *
 * Blocks current thread until job has completed
 */
XRT_API_EXPORT
enum ert_cmd_state
xrtRunWaitFor(xrtRunHandle rhdl, unsigned int timeout_ms);

/**
 * xrtRunState() - Check the current state of a run
 *
 * @rhdl:       Handle to check
 * Return:      The underlying command execution state per ert.h
 */
XRT_API_EXPORT
enum ert_cmd_state
xrtRunState(xrtRunHandle rhdl);

/**
 * xrtRunSetCallback() - Set a callback function
 *
 * @rhdl:        Handle to set callback on
 * @state:       State to invoke callback on
 * @callback:    Callback function
 * @data:        User data to pass to callback function
 *
 * Register a run callback function that is invoked when the
 * run changes underlying execution state to specified state.
 * Support states are: ERT_CMD_STATE_COMPLETED (to be extended)
 */
XRT_API_EXPORT
int
xrtRunSetCallback(xrtRunHandle rhdl, enum ert_cmd_state state,
                  void (* callback)(xrtRunHandle, enum ert_cmd_state, void*),
                  void* data);

/**
 * xrtRunClose() - Close a run handle
 *
 * @rhdl:  Handle to close
 * Return:      0 on success, -1 on error
 */
XRT_API_EXPORT
int
xrtRunClose(xrtRunHandle rhdl);

/// @endcond

#ifdef __cplusplus
}
#endif

#endif
