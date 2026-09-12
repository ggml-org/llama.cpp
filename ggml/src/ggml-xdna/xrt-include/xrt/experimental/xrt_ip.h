// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2021-2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
#ifndef XRT_IP_H_
#define XRT_IP_H_

#include "xrt/deprecated/xrt.h"
#include "xrt/xrt_uuid.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/detail/pimpl.h"

#ifdef __cplusplus
# include <condition_variable>
# include <cstdint>
# include <string>
#endif

#ifdef __cplusplus

namespace xrt {

/*!
 * @class ip
 *
 * @brief
 * xrt::ip represent the custom IP
 *
 * @details The ip can be controlled through read- and write register
 * only.  If the IP supports interrupt notification, then xrt::ip
 * objects supports enabling and control of underlying IP interrupt.
 *
 * In order to construct an ip object, the following requirements must be met:
 *
 *   - The custom IP must appear in IP_LAYOUT section of xclbin.
 *   - The custom IP must have a base address such that it can be controlled
 *     through register access at offsets from base address.
 *   - The custom IP must have an address range so that write and read access
 *     to base address offset can be validated.
 *   - XRT supports exclusive access only for the custom IP, this is to other
 *     processes from accessing the same IP at the same time.
 */
class ip_impl;
class ip : public detail::pimpl<ip_impl>
{
public:
  /*!
   * @class interrupt
   *
   * @brief
   * xrt::ip::interrupt represents an IP interrupt event.
   *
   * This class represents an IP interrupt event.  The interrupt
   * object is contructed via `xrt::ip::create_interrupt_notify()`.
   * The object can be used to enable and disable IP interrupts
   * and to wait for an interrupt to occur.
   *
   * Upon construction, the IP interrupt is automatically enabled.
   */
  class interrupt_impl;
  class interrupt : public detail::pimpl<interrupt_impl>
  {
  public:
    /// @cond
    explicit
    interrupt(std::shared_ptr<interrupt_impl> handle)
      : detail::pimpl<interrupt_impl>(std::move(handle))
    {}
    /// @endcond

    /**
     * enable() - Enable interrupt
     *
     * Enables the IP interrupt if not already enabled.  The
     * IP interrupt is automatically enabled when the interrupt
     * object is created.
     */
    XCL_DRIVER_DLLESPEC
    void
    enable();

    /**
     * disable() - Disable interrupt
     *
     * Disables the IP interrupt notification from IP.
     */
    XCL_DRIVER_DLLESPEC
    void
    disable();

    /**
     * wait() - Wait for interrupt
     *
     * Wait for interrupt from IP. Upon return, interrupt is
     * re-enabled.
     */
    XCL_DRIVER_DLLESPEC
    void
    wait();

    /**
     * wait() - Wait for interrupt or timeout to occur
     *
     * @param timeout
     *   Timout in milliseconds.
     * @return
     *   std::cv_status::timeout if the timeout specified expired,
     *   std::cv_status::no_timeout otherwise.
     *
     * Blocks the current thread until an interrupt is received from the IP,  or
     * until after the specified timeout duration
     */
    XCL_DRIVER_DLLESPEC
    std::cv_status
    wait(const std::chrono::milliseconds& timeout) const;
  };

public:
  /**
   * ip() - Construct empty ip object
   */
  ip()
  {}

  /**
   * ip() - Constructor from a device and xclbin
   *
   * @param device
   *  Device programmed with the IP
   * @param xclbin_id
   *  UUID of the xclbin with the IP
   * @param name
   *  Name of IP to construct
   *
   * The IP is opened with exclusive access meaning that no other
   * xrt::ip objects can use the same IP, nor will another process be
   * able to use the IP while one process has been granted access.
   *
   * Constructor throws on error.
   */
  XCL_DRIVER_DLLESPEC
  ip(const xrt::device& device, const xrt::uuid& xclbin_id, const std::string& name);

  /// @cond
  /// Experimental in 2022.2
  XCL_DRIVER_DLLESPEC
  ip(const xrt::hw_context& ctx, const std::string& name);
  /// @endcond

  /**
   * write_register() - Write to the address range of an ip
   *
   * @param offset
   *  Offset in register space to write to
   * @param data
   *  Data to write
   *
   * Throws std::out_or_range if offset is outside the
   * ip address space
   */
  XCL_DRIVER_DLLESPEC
  void
  write_register(uint32_t offset, uint32_t data);

  /**
   * read_register() - Read data from ip address range
   *
   * @param offset
   *  Offset in register space to read from
   * @return
   *  Value read from offset
   *
   * Throws std::out_or_range if offset is outside the
   * ip address space
   */
  XCL_DRIVER_DLLESPEC
  uint32_t
  read_register(uint32_t offset) const;

  /**
   * create_interrupt_notify() - Create xrt::ip::interrupt object
   *
   * @return
   *  xrt::ip::interrupt object can be used to control IP interrupt.
   *
   * This function creates an xrt::ip::interrupt object that can be
   * used to control and wait for IP interrupt.   On successful
   * return the IP has interrupt enabled.
   *
   * Throws if the custom IP doesn't support interrupts.
   */
  XCL_DRIVER_DLLESPEC
  interrupt
  create_interrupt_notify();
};

} // xrt
#endif

#endif
