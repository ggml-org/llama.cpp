// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
#ifndef XRT_HW_CONTEXT_H_
#define XRT_HW_CONTEXT_H_

#include "xrt/detail/config.h"
#include "xrt/detail/pimpl.h"

#include "xrt/xrt_device.h"
#include "xrt/xrt_uuid.h"

#include "xrt/experimental/xrt_elf.h"

#ifdef __cplusplus

#include <map>

// Opaque handle for internal use
namespace xrt_core {
class hwctx_handle;
}

namespace xrt {

/**
 * class hw_context -- manage hw resources
 *
 * A hardware context associates an xclbin with hardware
 * resources.  Prior to creating a context, the xclbin
 * must be registered with the device (`xrt::device::register_xclbin`)
 *
 */
class hw_context_impl;
class hw_context : public detail::pimpl<hw_context_impl>
{
public:

  /**
   * Experimental specification of Configuration Parameters which contains QoS and Communication Channel requirements
   *
   * Free formed key-value entry.
   *
   * Supported keys are:
   *  - gops                   // giga operations per second
   *  - fps                    // frames per second
   *  - dma_bandwidth          // gigabytes per second
   *  - latency                // ??
   *  - frame_execution_time   // ??
   *  - priority               // ??
   *  - enable_isp_channel     // toggle isp communication
   *  - enable_acp_channel     // toggle acp communication
   *
   * Currently ignored for legacy platforms
   */
  using cfg_param_type = std::map<std::string, uint32_t>;
  using qos_type = cfg_param_type; //alias to old type

  /**
   * @enum access_mode - legacy access mode
   *
   * @var exclusive
   *  Create a context for exclusive access to shareable resources.
   *  Legacy compute unit access control.
   * @var shared
   *  Create a context for shared access to shareable resources
   *  Legacy compute unit access control.
   *
   * Access mode is mutually exclusive with qos
   */
  enum class access_mode : uint8_t {
    exclusive = 0,
    shared = 1
  };

public:
  /**
   * hw_context() - Constructor for empty context
   *
   * It is undefined behavior to use a default constructed hw_context
   * for anything but assignment.
   */
  hw_context() = default;

  /**
   * hw_context() - Constructor with QoS control and access control
   * 
   * @param device
   *  Device where context is created
   * @param cfg_param
   *  Configuration Parameters (incl. Quality of Service)
   * @param mode
   *  Access control for the context
   * 
   * When application uses this constructor no hw resources are allocated
   * It acts as placeholder and is used for setting QoS and access control
   * Applications can later add configuration Elfs using add_config api.
   * The QoS definition is subject to change, so this API is not guaranteed
   * to be ABI compatible in future releases
   */
  XRT_API_EXPORT
  hw_context(const xrt::device& device, const cfg_param_type& cfg_param, access_mode mode);

  /**
   * hw_context() - Constructor with Elf file
   *
   * @param device
   *  Device where context is created
   * @param elf
   *  Elf configuration object
   * @param cfg_param
   *  Configuration Parameters (incl. Quality of Service)
   * @param mode
   *  Access control for the context
   *
   * The QoS definition is subject to change, so this API is not guaranteed
   * to be ABI compatible in future releases. When cfg_param and access_mode
   * are not passed hw context with shared access mode is created.
   */
  XRT_API_EXPORT
  hw_context(const xrt::device& device, const xrt::elf& elf,
             const cfg_param_type& cfg_param, access_mode mode);

  /**
    * hw_context() - Constructor with Elf file with implied qos and mode
    *
    * @param device
    *  Device where context is created
    * @param elf
    *  Elf configuration object
    *
    * This constructor defaults optional configuration parameters
    * and creates a hw context with shared access mode.
    */
  XRT_API_EXPORT
  hw_context(const xrt::device& device, const xrt::elf& elf);

  /**
   * add_config() - adds config Elf file to the context
   * 
   * @param elf
   *  XRT Elf object created from config Elf file
   * 
   * Adds config Elf to context if it is the first config added
   * If config already exists, it will be added only when configuration matches
   * with existing one else an exception is thrown
   */
  XRT_API_EXPORT
  void
  add_config(const xrt::elf& elf);

  /**
   * hw_context() - Constructor with QoS control
   *
   * @param device
   *  Device where context is created
   * @param xclbin_id
   *  UUID of xclbin that should be assigned to HW resources
   * @param cfg_param
   *  Configuration Parameters (incl. Quality of Service)
   *
   * The QoS definition is subject to change, so this API is not guaranteed
   * to be ABI compatible in future releases.
   */
  XRT_API_EXPORT
  hw_context(const xrt::device& device, const xrt::uuid& xclbin_id, const cfg_param_type& cfg_param);

  /**
   * hw_context() - Construct with specific access control
   *
   * @param device
   *  Device where context is created
   * @param xclbin_id
   *  UUID of xclbin that should be assigned to HW resources
   * @param mode
   *  Access control for the context
   */
  XRT_API_EXPORT
  hw_context(const xrt::device& device, const xrt::uuid& xclbin_id, access_mode mode);

  ///@cond
  // Undocumented construction w/o specifying qos
  // Subject to change in default qos value
  hw_context(const xrt::device& device, const xrt::uuid& xclbin_id)
    : hw_context{device, xclbin_id, access_mode::shared}
  {}
  /// @endcond

  ///@cond
  // Undocumented converting constructor using impl only
  hw_context(std::shared_ptr<hw_context_impl> impl)
    : detail::pimpl<hw_context_impl>(std::move(impl))
  {}
  /// @endcond

  /**
   * hw_context() - Copy ctor
   *
   * Performs shallow copy, sharing data with the source
   */
  hw_context(const hw_context&) = default;

  /**
   * hw_context() - Move ctor
   */
  hw_context(hw_context&&) = default;

  /**
   * ~hw_context() - Destructor
   */
  XRT_API_EXPORT
  ~hw_context();

  /**
   * operator= () - Copy assignment
   *
   * Performs shallow copy, sharing data with the source
   */
  hw_context&
  operator=(const hw_context&) = default;

  /**
   * operator= () - Move assignment
   */
  hw_context&
  operator=(hw_context&&) = default;

  ///@cond
  // Undocument experimental API to change the QoS of a hardware context
  // Subject to change or removal
  XRT_API_EXPORT
  void
  update_qos(const qos_type& qos);
  ///@endcond

  /**
   * get_device() - Device from which context was created
   */
  XRT_API_EXPORT
  xrt::device
  get_device() const;

  /**
   * get_xclbin_uuid() - UUID of xclbin from which context was created
   * Returns empty uuid if context was created without xclbin (created with Elf)
   */
  XRT_API_EXPORT
  xrt::uuid
  get_xclbin_uuid() const;

  /**
   * get_xclbin() - Retrieve underlying xclbin matching the UUID
   * Returns empty xclbin if context was created without xclbin (created with Elf)
   */
  XRT_API_EXPORT
  xrt::xclbin
  get_xclbin() const;

  /**
   * get_mode() - Get the context access mode
   */
  XRT_API_EXPORT
  access_mode
  get_mode() const;

  /**
   * get_aie_coredump() - Returns the coredump of AIE Array if available.
   * Coredump represents memory/register dump of AIE tiles in this ctx.
   * This function can throw.
   */
  XRT_API_EXPORT
  std::vector<char>
  get_aie_coredump() const;

public:
  /// @cond
  // Undocumented internal access to low level context handle
  // Subject to change without warning
  XRT_API_EXPORT
  explicit operator xrt_core::hwctx_handle* () const;
  /// @endcond
};

} // namespace xrt

#else
# error xrt_hwcontext is only implemented for C++
#endif // __cplusplus

#endif
