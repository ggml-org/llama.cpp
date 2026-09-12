// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2020-2022 Xilinx, Inc.  All rights reserved.
// Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
#ifndef XRT_XCLBIN_H_
#define XRT_XCLBIN_H_

#include "xrt/xrt_uuid.h"

#include "xrt/detail/config.h"
#include "xrt/detail/pimpl.h"
#include "xrt/detail/xclbin.h"
#include "xrt/deprecated/xrt.h"

#ifdef __cplusplus
# include <iterator>
# include <utility>
# include <vector>
# include <string>
# include <string_view>
#endif

/**
 * typedef xrtXclbinHandle - opaque xclbin handle
 */
typedef void* xrtXclbinHandle; // NOLINT

#ifdef __cplusplus
namespace xrt {

/*!
 * @class xclbin
 *
 * @brief
 * xrt::xclbin represents an xclbin and provides APIs to access meta data.
 *
 * @details
 * The xclbin object is constructed by the user from a file.
 *
 * When the xclbin object is constructed from a complete xclbin, then it
 * can be used by xrt::device to program the xclbin onto the device.
 *
 * **First-class objects and class navigation**
 *
 * All meta data is rooted at xrt::xclbin.
 *
 * \image{inline} html xclbin_navigation.png "xclbin navigation"
 *
 * From the xclbin object
 * xrt::xclbin::kernel or xrt::xclbin::ip objects can be constructed.
 *
 * The xrt:xclbin::kernel is a concept modelled only in the xclbin XML
 * metadata, it corresponds to a function that can be executed by one
 * or more compute units modelled by xrt::xclbin::ip objects.  An
 * xrt::xclbin::ip object corresponds to an entry in the xclbin
 * IP_LAYOUT section, so the xrt::xclbin::kernel object is just a
 * grouping of one or more of these.
 *
 * In some cases the kernel concept is not needed, thus
 * xrt::xclbin::ip objects corresponding to entries in the xclbin
 * IP_LAYOUT sections can be accessed directly.
 *
 * An xrt::xclbin::arg object corresponds to one or more entries in
 * the xclbin CONNECTIVITY section decorated with additional meta data
 * (offset, size, type, etc) from the XML section if available.  An
 * argument object represents a specific kernel or ip argument. If
 * the argument is a global buffer, then it may connect to one or more
 * memory objects.
 *
 * Finally the xrt::xclbin::mem object corresponds to an entry in the
 * MEM_TOPOLOGY section of the xclbin.
 */
class xclbin_impl;
class xclbin : public detail::pimpl<xclbin_impl>
{
public:
  /*!
   * @enum target_type
   *
   * @brief
   * Type of xclbin
   *
   * @details
   * See `xclbin.h`
   */
  enum class target_type { hw, sw_emu, hw_emu };

public:
  /*!
   * @class mem
   *
   * @brief
   * xrt::xclbin::mem represents a physical device memory bank
   *
   * @details
   * A memory object is constructed from an entry in the MEM_TOPOLOGY
   * section of an xclbin.
   */
  class mem_impl;
  class mem : public detail::pimpl<mem_impl>
  {
  public:
    /**
     * @enum memory_type - type of memory
     *
     * @details
     * See `xclbin.h`
     */
    enum class memory_type : uint8_t {
      ddr3                 = MEM_DDR3,
      ddr4                 = MEM_DDR4,
      dram                 = MEM_DRAM,
      streaming            = MEM_STREAMING,
      preallocated_global  = MEM_PREALLOCATED_GLOB,
      are                  = MEM_ARE, //Aurora
      hbm                  = MEM_HBM,
      bram                 = MEM_BRAM,
      uram                 = MEM_URAM,
      streaming_connection = MEM_STREAMING_CONNECTION,
      host                 = MEM_HOST
    };

  public:
    mem() = default;

    explicit
    mem(std::shared_ptr<mem_impl> handle)
      : detail::pimpl<mem_impl>(std::move(handle))
    {}

    /**
     * get_tag() - Get tag name
     *
     * @return
     *   Memory tag name
     */
    XRT_API_EXPORT
    std::string
    get_tag() const;

    /**
     * get_base_address() - Get the base address of the memory bank
     *
     * @return
     *  Base address of the memory bank, or -1 for invalid base address
     */
    XRT_API_EXPORT
    uint64_t
    get_base_address() const;

    /**
     * get_size_kb() - Get the size of the memory in KB
     *
     * @return
     *  Size of memory in KB, or -1 for invalid size
     */
    XRT_API_EXPORT
    uint64_t
    get_size_kb() const;

    /**
     * get_used() - Get used status of the memory
     *
     * @return
     *  True of this memory bank is used by kernels in the xclbin
     *  or false otherwise.
     *
     * A value of false indicates that no buffer can be allocated
     * in this memory bank.
     */
    XRT_API_EXPORT
    bool
    get_used() const;

    /**
     * get_type() - Get the type of the memory
     *
     * @return
     *  Memory type
     *
     */
    XRT_API_EXPORT
    memory_type
    get_type() const;

    /**
     * get_index() - Get the index of the memory
     *
     * @return
     *  Index of the memory within the memory topology
     *
     * The returned index can be used when allocating buffers using
     * \ref xrt::bo provided the memory bank is connected / used.
     */
    XRT_API_EXPORT
    int32_t
    get_index() const;
  };

  /*!
   * @class arg
   *
   * @brief
   * class arg - xrt::xclbin::arg represents a compute unit argument
   *
   * @details
   * The argument object constructed from the xclbin connectivity
   * section.  An argument is connected to a memory bank or a memory
   * group, which dictates where in device memory a global buffer
   * used with this kernel argument must be allocated.
   */
  class arg_impl;
  class arg : public detail::pimpl<arg_impl>
  {
  public:
    arg() = default;

    explicit
    arg(std::shared_ptr<arg_impl> handle)
      : detail::pimpl<arg_impl>(std::move(handle))
    {}

    /**
     * get_name() - Get argument name
     *
     * @return
     *  Name of argument.
     *
     */
    XRT_API_EXPORT
    std::string
    get_name() const;

    /**
     * get_mems() - Get list of device memories from xclbin.
     *
     * @return
     *  A list of xrt::xclbin::mem objects to which this argument
     *  is connected.
     */
    XRT_API_EXPORT
    std::vector<mem>
    get_mems() const;

    /**
     * get_port() - Get port name of this argument
     *
     * @return
     *  Port name
     */
    XRT_API_EXPORT
    std::string
    get_port() const;

    /**
     * get_size() - Argument size in bytes
     *
     * @return
     *   Argument size
     */
    XRT_API_EXPORT
    uint64_t
    get_size() const;

    /**
     * get_offset() - Argument offset
     *
     * @return
     *   Argument offset
     */
    XRT_API_EXPORT
    uint64_t
    get_offset() const;

    /**
     * get_host_type() - Get the argument host type
     *
     * @return
     *   Argument host type
     */
    XRT_API_EXPORT
    std::string
    get_host_type() const;

    /**
     * get_index() - Get the index of this argument
     *
     * @return
     *   Argument index
     */
    XRT_API_EXPORT
    size_t
    get_index() const;
  };

  /*!
   * @class ip
   *
   * @brief
   * xrt::xclbin::ip represents a IP in an xclbin.
   *
   * @details
   * The ip corresponds to an entry in the IP_LAYOUT section of the
   * xclbin.
   */
  class ip_impl;
  class ip : public detail::pimpl<ip_impl>
  {
  public:
    /**
     * @enum control_type -
     *
     * @details
     * See `xclbin.h`
     */
    enum class control_type : uint8_t { hs = 0, chain = 1, none = 2, fa = 5 };

    /**
     * @enum ip_type
     *
     * @details
     * See `xclbin.h`
     */
    enum class ip_type : uint8_t { pl = IP_KERNEL, ps = IP_PS_KERNEL };

  public:
    ip() = default;

    explicit
    ip(std::shared_ptr<ip_impl> handle)
      : detail::pimpl<ip_impl>(std::move(handle))
    {}

    /**
     * get_name() - Get name of IP
     *
     * @return
     *  IP name.
     */
    XRT_API_EXPORT
    std::string
    get_name() const;

    /**
     * get_type() - Get the IP type
     *
     * @return
     *  IP type
     */
    XRT_API_EXPORT
    ip_type
    get_type() const;

    /**
     * get_control_type() - Get the IP control protocol
     *
     * @return
     *  Control type
     */
    XRT_API_EXPORT
    control_type
    get_control_type() const;

    /**
     * get_num_args() - Number of arguments
     *
     * @return
     *  Number of arguments for this IP per CONNECTIVITY section
     */
    XRT_API_EXPORT
    size_t
    get_num_args() const;

    /**
     * get_args() - Get list of IP arguments
     *
     * @return
     *  A list sorted of xclbin::arg sorted by argument indices
     *
     * An argument may have multiple memory connections
     */
    XRT_API_EXPORT
    std::vector<arg>
    get_args() const;

    /**
     * get_arg() - Get argument at index.
     *
     * @param index
     *  Index of argument
     * @return
     *  The argument a specified index
     *
     * The argument may have multiple memory connections
     */
    XRT_API_EXPORT
    arg
    get_arg(int32_t index) const;

    /**
     * get_base_address() - Get the base address of the cu
     *
     * @return
     *  The base address of the IP
     */
    XRT_API_EXPORT
    uint64_t
    get_base_address() const;

    /**
     * get_size() - Get the address range size of this IP.
     *
     * @return
     *  The size of this IP
     *
     * The address range is a property of the kernel and
     * as such only valid for for kernel compute units.
     *
     * For IPs that are not associated with a kernel, the
     * size return is 0.
     */
    XRT_API_EXPORT
    size_t
    get_size() const;
  };

  /*!
   * class kernel
   *
   * @brief
   * xrt::xclbin::kernel represents a kernel in an xclbin.
   *
   * @details
   * The kernel corresponds to an entry in the XML meta data section
   * of the xclbin combined with meta data from other xclbin sections.
   * The kernel object is implicitly constructed from the xclbin
   * object via APIs.
   */
  class kernel_impl;
  class kernel : public detail::pimpl<kernel_impl>
  {
  public:
    /**
     * @enum kernel_type
     *
     * The kernel type is extracted from the XML kernel meta data section
     */
    enum class kernel_type : uint8_t { none = 0, pl = 1, ps = 2, dpu = 3};
    
  public:
    kernel() = default;

    explicit
    kernel(std::shared_ptr<kernel_impl> handle)
      : detail::pimpl<kernel_impl>(std::move(handle))
    {}

    /**
     * get_name() - Get kernel name
     *
     * @return
     *  The name of the kernel
     */
    XRT_API_EXPORT
    std::string
    get_name() const;

    /**
     * get_type() - Get kernel type
     *
     * @return
     *  The type of the kernel
     */
    XRT_API_EXPORT
    kernel_type
    get_type() const;

    /**
     * get_cus() - Get list of cu from kernel.
     *
     * @return
     *  A list of xrt::xclbin::ip objects corresponding the compute units
     *  for this kernel object.
     */
    XRT_API_EXPORT
    std::vector<ip>
    get_cus() const;

    /**
     * get_cus() - Get list of compute units that matches name
     *
     * @param name
     *  Name to match against, prefixed with kernel name
     * @return
     *  A list of xrt::xclbin::ip objects that are compute units
     *  of this kernel object and matches the specified name.
     *
     * The kernel name can optionally specify which kernel instance(s) to
     * match "kernel:{cu1,cu2,...} syntax.
     */
    XRT_API_EXPORT
    std::vector<ip>
    get_cus(const std::string& name) const;

    /**
     * get_cu() - Get compute unit by name
     *
     * @return
     *  The xct::xclbin::ip object matching the specified name, or error if
     *  not present.
     */
    XRT_API_EXPORT
    ip
    get_cu(const std::string& name) const;

    /**
     * get_num_args() - Number of arguments
     *
     * @return
     *  Number of arguments for this kernel.
     */
    XRT_API_EXPORT
    size_t
    get_num_args() const;

    /**
     * get_args() - Get list of kernel arguments
     *
     * @return
     *  A list sorted of xclbin::arg sorted by argument indices
     *
     * An argument may have multiple memory connections
     */
    XRT_API_EXPORT
    std::vector<arg>
    get_args() const;

    /**
     * get_arg() - Get kernel argument at index.
     *
     * @return
     *  The xrt::xclbin::arg object at specified argument index.
     *
     * The memory connections of an argument is the union of the
     * connections for each compute unit for that particular argument.
     * In other words, for each connection of the argument returned
     * by ``get_arg()`` there is at least one compute unit that has
     * that connection.
     */
    XRT_API_EXPORT
    arg
    get_arg(int32_t index) const;
  };

  /// @cond
  /** undocumented access to aie metadata, subject to change **/
  class aie_partition_impl;
  class aie_partition : detail::pimpl<aie_partition_impl>
  {
  public:
    explicit
    aie_partition(std::shared_ptr<aie_partition_impl> handle)
      : detail::pimpl<aie_partition_impl>(std::move(handle))
    {}

    XRT_API_EXPORT
    uint64_t
    get_inference_fingerprint() const;

    XRT_API_EXPORT
    uint64_t
    get_pre_post_fingerprint() const;

    XRT_API_EXPORT
    uint32_t
    get_operations_per_cycle() const;
  };
  /// @endcond

public:
  /**
   * xclbin() - Construct empty xclbin object
   */
  xclbin() = default;

  /// @cond
  /**
   * xclbin() - Construct from handle
   */
  explicit
  xclbin(std::shared_ptr<xclbin_impl> handle)
    : detail::pimpl<xclbin_impl>(handle)
  {}
  /// @endcond

  /**
   * xclbin() - Constructor from an xclbin filename
   *
   *
   * @param filename : A path relative or absolute to an xclbin file
   *
   * If the specified path is an absolute path then the function
   * returns this path or throws if file does not exist.  If the path
   * is relative, or just a plain file name, then the function check
   * first in current directory, then in the platform specific xclbin
   * repository. 
   *
   * Throws if file could not be found.
   */
  XRT_API_EXPORT
  explicit
  xclbin(const std::string& filename);

  /**
   * xclbin() - Constructor from raw data
   *
   * @param data
   *  Raw data of xclbin
   *
   * The raw data of the xclbin can be deleted after calling the
   * constructor.
   */
  XRT_API_EXPORT
  explicit
  xclbin(const std::vector<char>& data);

  /**
   * xclbin() - Constructor from raw data
   *
   * @param data
   *  Raw data of xclbin
   *
   * The raw data of the xclbin can be deleted after calling the
   * constructor.
   */
  XRT_API_EXPORT
  explicit
  xclbin(const std::string_view& data);

  /**
   * xclbin() - Constructor from raw data
   *
   * @param top
   *  Raw data of xclbin file as axlf*
   *
   * The argument axlf is copied by the constructor.
   */
  XRT_API_EXPORT
  explicit
  xclbin(const axlf* top);

  /**
   * get_kernels() - Get list of kernels from xclbin.
   *
   * @return
   *  A list of xrt::xclbin::kernel from xclbin.
   *
   * Kernels are extracted from embedded XML metadata in the xclbin.
   * A kernel groups one or more compute units. A kernel has arguments
   * from which offset, type, etc can be retrived.
   */
  XRT_API_EXPORT
  std::vector<kernel>
  get_kernels() const;

  /**
   * get_kernel() - Get a kernel by name from xclbin
   *
   * @param name
   *  Name of kernel to get.
   * @return
   *  The matching kernel from the xclbin or error
   *  if no matching kernel is found.
   *
   * A matching kernel is extracted from embedded XML metadata in the
   * xclbin.  A kernel groups one or more compute units. A kernel has
   * arguments from which offset, type, etc can be retrived.
   */
  XRT_API_EXPORT
  kernel
  get_kernel(const std::string& name) const;

  /**
   * get_ips() - Get a list of IPs from the xclbin
   *
   * @return
   *  A list of xrt::xclbin::ip objects from xclbin.
   *
   * The returned xrt::xclbin::ip objects are extracted from the
   * IP_LAYOUT section of the xclbin.
   */
  XRT_API_EXPORT
  std::vector<ip>
  get_ips() const;

  /**
   * get_ips() - Get list of ips that matches name
   *
   * @param name
   *  Name to match against, prefixed with kernel name
   * @return
   *  A list of xrt::xclbin::ip objects that are compute units
   *  of this kernel object and matches the specified name.
   *
   * The kernel name can optionally specify which kernel instance(s) to
   * match "kernel:{ip1,ip2,...} syntax.
   */
  XRT_API_EXPORT
  std::vector<ip>
  get_ips(const std::string& name) const;

  /**
   * get_ip() - Get a specific IP from the xclbin
   *
   * @return
   *  A list of xrt::xclbin::ip objects from xclbin.
   *
   * The returned xrt::xclbin::ip object is extracted from the
   * IP_LAYOUT section of the xclbin.
   */
  XRT_API_EXPORT
  ip
  get_ip(const std::string& name) const;

  /**
   * get_mems() - Get list of memory objects
   *
   * @return
   * A list of xrt::xclbin::mem objects from xclbin
   *
   * The returned xrt::xclbin::mem objects are extracted from
   * the xclbin.
   */
  XRT_API_EXPORT
  std::vector<mem>
  get_mems() const;

  /// @cond
  XRT_API_EXPORT
  std::vector<aie_partition>
  get_aie_partitions() const;
  /// @endcond

  /**
   * get_xsa_name() - Get Xilinx Support Archive (XSA) name of xclbin
   *
   * @return
   *  Name of XSA (vbnv name).
   *
   * An exception is thrown if the data is missing.
   */
  XRT_API_EXPORT
  std::string
  get_xsa_name() const;

  /**
   * get_fpga_device_name() - Get FPGA device name
   *
   * @return
   *  Name of fpga device per XML metadata.
   */
  XRT_API_EXPORT
  std::string
  get_fpga_device_name() const;

  /**
   * get_uuid() - Get the uuid of the xclbin
   *
   * @return
   *  UUID of xclbin
   *
   * An exception is thrown if the data is missing.
   */
  XRT_API_EXPORT
  uuid
  get_uuid() const;

  /**
  * get_interface_uuid() - Get the interface uuid of the xclbin
  *
  * @return
  *  Interface uuid of the xclbin
  *
  * An exception is thrown if the data is missing.
  */
  XRT_API_EXPORT
  uuid
  get_interface_uuid() const;

  /**
   * get_target_type() - Get the type of this xclbin
   *
   * @return
   *  Target type, which can be hw, sw_emu, or hw_emu
   */
  XRT_API_EXPORT
  target_type
  get_target_type() const;

  /// @cond
  /**
   * get_axlf() - Get the axlf data of the xclbin
   *
   * @return
   *  The axlf data of the xclbin object
   *
   * An exception is thrown if the data is missing.
   */
  XRT_API_EXPORT
  const axlf*
  get_axlf() const;

  /**
   * get_axlf_section() - Retrieve specified xclbin section
   *
   * @param section
   *  The section to retrieve
   * @return
   *  The specified section if available cast to specified type.
   *  Note, that this is an unsafe cast, behavior is undefined if the
   *  specified SectionType is invalid.
   *
   * The SectionType template parameter is an axlf type from xclbin.h
   * and it much match the type of the section data retrieved.
   *
   * Throws if requested section does not exist in the xclbin.
   */
  template <typename SectionType>
  SectionType
  get_axlf_section(axlf_section_kind section) const
  {
    return reinterpret_cast<SectionType>(get_axlf_section(section).first);
  }
  /// @endcond

private:
  XRT_API_EXPORT
  std::pair<const char*, size_t>
  get_axlf_section(axlf_section_kind section) const;
};

/**
 * xclbin_repository - Repository of xclbins
 *
 * A repository of xclbins is a collection of xclbins that can be
 * searched for a specific xclbin through iteration.
 *
 * The location of a repository is specified by a directory or it
 * can be implementations and platform specific.
 */
class xclbin_repository_impl;
class xclbin_repository : public detail::pimpl<xclbin_repository_impl>
{
public:
  /**
   * xclbin_repository - Default constructor
   *
   * Create repository from builtin platform specific repository
   * path or paths.
   */
  XRT_API_EXPORT
  xclbin_repository();

  /**
   * xclbin_repository - 
   *
   * Create repository from specified path.
   *
   * The specified directory can be an absolute path to a directory or
   * it can be a relative path to a directory rooted at the current
   * working directory.
   */
  XRT_API_EXPORT
  explicit
  xclbin_repository(const std::string& dir);

  /**
   * iterator - Iterator over xclbins in repository
   *
   * The iterator is a forward iterator that iterates over the
   * xclbins in the repository.  The iterator dereferences to an
   * xrt::xclbin object by value.
   */
  class iterator_impl;
  class iterator : public detail::pimpl<iterator_impl>
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = xclbin;
    using pointer           = value_type;
    using reference         = value_type;

  public:
    /**
     * iterator - Converting default constructor from implmentation
     */
    iterator(std::shared_ptr<iterator_impl> handle)
      : detail::pimpl<iterator_impl>(std::move(handle))
    {}

    /**
     * iterator - Copy constructor
     *
     * Create a copy of an iterator
     */
    XRT_API_EXPORT
    iterator(const iterator&);

    /**
     * Advance iterator to next xclbin
     */
    XRT_API_EXPORT
    iterator&
    operator++();

    /**
     * Advance iterator to next xclbin return old iterator
     *
     * This is a relatively expensive operation that duplicates
     * the internal representation of the original iterator.
     */
    XRT_API_EXPORT
    iterator
    operator++(int);

    /**
     * Compare iterators
     */
    XRT_API_EXPORT
    bool
    operator==(const iterator& rhs) const;

    /**
     * Compare iterators
     */
    bool
    operator!=(const iterator& rhs) const
    {
      return !(*this == rhs);
    }

    /**
     * Dereference iterator
     *
     * Returns xrt::xclbin object by value.  The xclbin object
     * is constructed on the fly.
     */
    XRT_API_EXPORT
    value_type
    operator*() const;

    /**
     * Dereference iterator
     *
     * Returns xrt::xclbin object by value.  The xclbin object
     * is constructed on the fly.
     */
    XRT_API_EXPORT
    value_type
    operator->() const;

    /**
     * Get path to xclbin file path in repository for this iterator
     */
    XRT_API_EXPORT
    std::string
    path() const;
  };

  /**
   * begin() - Get iterator to first xclbin in repository
   */
  XRT_API_EXPORT
  iterator
  begin() const;

  /**
   * end() - Get iterator to end of xclbin repository
   */
  XRT_API_EXPORT
  iterator
  end() const;

  /**
   * load() - Load xclbin from repository
   */
  XRT_API_EXPORT
  xclbin
  load(const std::string& name) const;
};

} // namespace xrt

/// @cond
extern "C" {
#endif

/**
 * xrtXclbinAllocFilename() - Allocate a xclbin using xclbin filename
 *
 * @filename:      path to the xclbin file
 * Return:         xrtXclbinHandle on success or NULL with errno set
 */
XRT_API_EXPORT
xrtXclbinHandle
xrtXclbinAllocFilename(const char* filename);


/**
 * xrtXclbinAllocAxlf() - Allocate a xclbin using an axlf
 *
 * @top_axlf:      an axlf
 * Return:         xrtXclbinHandle on success or NULL with errno set
 */
XRT_API_EXPORT
xrtXclbinHandle
xrtXclbinAllocAxlf(const struct axlf* top_axlf);

/**
 * xrtXclbinAllocRawData() - Allocate a xclbin using raw data
 *
 * @data:          raw data buffer of xclbin
 * @size:          size (in bytes) of raw data buffer of xclbin
 * Return:         xrtXclbinHandle on success or NULL with errno set
 */
XRT_API_EXPORT
xrtXclbinHandle
xrtXclbinAllocRawData(const char* data, int size);

/**
 * xrtXclbinFreeHandle() - Deallocate the xclbin handle
 *
 * @xhdl:          xclbin handle
 * Return:         0 on success, -1 on error
 */
XRT_API_EXPORT
int
xrtXclbinFreeHandle(xrtXclbinHandle xhdl);

/**
 * xrtXclbinGetXSAName() - Get Xilinx Support Archive (XSA) Name of xclbin handle
 *
 * @xhdl:       Xclbin handle
 * @name:       Return name of XSA.
 *              If the value is nullptr, the content of this value will not be populated.
 *              Otherwise, the the content of this value will be populated.
 * @size:       size (in bytes) of @name.
 * @ret_size:   Return size (in bytes) of XSA name.
 *              If the value is nullptr, the content of this value will not be populated.
 *              Otherwise, the the content of this value will be populated.
 * Return:  0 on success or appropriate error number
 */
XRT_API_EXPORT
int
xrtXclbinGetXSAName(xrtXclbinHandle xhdl, char* name, int size, int* ret_size);

/**
 * xrtXclbinGetUUID() - Get UUID of xclbin handle
 *
 * @xhdl:     Xclbin handle
 * @ret_uuid: Return xclbin id in this uuid_t struct
 * Return:    0 on success or appropriate error number
 */
XRT_API_EXPORT
int
xrtXclbinGetUUID(xrtXclbinHandle xhdl, xuid_t ret_uuid);

/**
 * xrtXclbinGetNumKernels() - Get number of PL kernels in xclbin
 *
 * @xhdl:   Xclbin handle obtained from an xrtXclbinAlloc function
 * Return:  The number of PL kernels in the xclbin
 *
 * Kernels are extracted from embedded XML metadata in the xclbin.
 * A kernel groups one or more compute units. A kernel has arguments
 * from which offset, type, etc can be retrived.
 */
XRT_API_EXPORT
size_t
xrtXclbinGetNumKernels(xrtXclbinHandle xhdl);

/**
 * xrtXclbinGetNumKernelComputeUnits() - Get number of CUs in xclbin
 *
 * @xhdl:   Xclbin handle obtained from an xrtXclbinAlloc function
 * Return:  The number of compute units
 *
 * Compute units are associated with kernels.  This function returns
 * the total number of compute units as the sum of compute units over
 * all kernels.
 */
XRT_API_EXPORT
size_t
xrtXclbinGetNumKernelComputeUnits(xrtXclbinHandle xhdl);

/**
 * xrtXclbinGetData() - Get the raw data of the xclbin handle
 *
 * @xhdl:       Xclbin handle
 * @data:       Return raw data.
 *              If the value is nullptr, the content of this value will not be populated.
 *              Otherwise, the the content of this value will be populated.
 * @size:       Size (in bytes) of @data
 * @ret_size:   Return size (in bytes) of XSA name.
 *              If the value is nullptr, the content of this value will not be populated.
 *              Otherwise, the the content of this value will be populated.
 * Return:  0 on success or appropriate error number
 */
XRT_API_EXPORT
int
xrtXclbinGetData(xrtXclbinHandle xhdl, char* data, int size, int* ret_size);

/*
 * xrtGetXclbinUUID() - Get UUID of xclbin image running on device
 *
 * @dhdl:   Device handle
 * @out:    Return xclbin id in this uuid_t struct
 * Return:  0 on success or appropriate error number
 */
XRT_API_EXPORT
int
xrtXclbinUUID(xclDeviceHandle dhdl, xuid_t out);

/// @endcond
#ifdef __cplusplus
}
#endif

#endif
