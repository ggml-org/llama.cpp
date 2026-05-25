//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#ifndef AMD_VULKAN_MEMORY_ALLOCATOR_H
#define AMD_VULKAN_MEMORY_ALLOCATOR_H

/** \mainpage Vulkan Memory Allocator

<b>Version 3.3.0</b>

Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved. \n
License: MIT \n
See also: [product page on GPUOpen](https://gpuopen.com/gaming-product/vulkan-memory-allocator/),
[repository on GitHub](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)


<b>API documentation divided into groups:</b> [Topics](topics.html)

<b>General documentation chapters:</b>

- \subpage faq
- \subpage quick_start
  - [Project setup](@ref quick_start_project_setup)
  - [Initialization](@ref quick_start_initialization)
  - [Resource allocation](@ref quick_start_resource_allocation)
- \subpage choosing_memory_type
  - [Usage](@ref choosing_memory_type_usage)
  - [Required and preferred flags](@ref choosing_memory_type_required_preferred_flags)
  - [Explicit memory types](@ref choosing_memory_type_explicit_memory_types)
  - [Custom memory pools](@ref choosing_memory_type_custom_memory_pools)
  - [Dedicated allocations](@ref choosing_memory_type_dedicated_allocations)
- \subpage memory_mapping
  - [Copy functions](@ref memory_mapping_copy_functions)
  - [Mapping functions](@ref memory_mapping_mapping_functions)
  - [Persistently mapped memory](@ref memory_mapping_persistently_mapped_memory)
  - [Cache flush and invalidate](@ref memory_mapping_cache_control)
- \subpage staying_within_budget
  - [Querying for budget](@ref staying_within_budget_querying_for_budget)
  - [Controlling memory usage](@ref staying_within_budget_controlling_memory_usage)
- \subpage resource_aliasing
- \subpage custom_memory_pools
  - [Choosing memory type index](@ref custom_memory_pools_MemTypeIndex)
  - [When not to use custom pools](@ref custom_memory_pools_when_not_use)
  - [Linear allocation algorithm](@ref linear_algorithm)
    - [Free-at-once](@ref linear_algorithm_free_at_once)
    - [Stack](@ref linear_algorithm_stack)
    - [Double stack](@ref linear_algorithm_double_stack)
    - [Ring buffer](@ref linear_algorithm_ring_buffer)
- \subpage defragmentation
- \subpage statistics
  - [Numeric statistics](@ref statistics_numeric_statistics)
  - [JSON dump](@ref statistics_json_dump)
- \subpage allocation_annotation
  - [Allocation user data](@ref allocation_user_data)
  - [Allocation names](@ref allocation_names)
- \subpage virtual_allocator
- \subpage debugging_memory_usage
  - [Memory initialization](@ref debugging_memory_usage_initialization)
  - [Margins](@ref debugging_memory_usage_margins)
  - [Corruption detection](@ref debugging_memory_usage_corruption_detection)
  - [Leak detection features](@ref debugging_memory_usage_leak_detection)
- \subpage other_api_interop
- \subpage usage_patterns
    - [GPU-only resource](@ref usage_patterns_gpu_only)
    - [Staging copy for upload](@ref usage_patterns_staging_copy_upload)
    - [Readback](@ref usage_patterns_readback)
    - [Advanced data uploading](@ref usage_patterns_advanced_data_uploading)
    - [Other use cases](@ref usage_patterns_other_use_cases)
- \subpage configuration
  - [Pointers to Vulkan functions](@ref config_Vulkan_functions)
  - [Custom host memory allocator](@ref custom_memory_allocator)
  - [Device memory allocation callbacks](@ref allocation_callbacks)
  - [Device heap memory limit](@ref heap_memory_limit)
- <b>Extension support</b>
    - \subpage vk_khr_dedicated_allocation
    - \subpage enabling_buffer_device_address
    - \subpage vk_ext_memory_priority
    - \subpage vk_amd_device_coherent_memory
    - \subpage vk_khr_external_memory_win32
- \subpage general_considerations
  - [Thread safety](@ref general_considerations_thread_safety)
  - [Versioning and compatibility](@ref general_considerations_versioning_and_compatibility)
  - [Validation layer warnings](@ref general_considerations_validation_layer_warnings)
  - [Allocation algorithm](@ref general_considerations_allocation_algorithm)
  - [Features not supported](@ref general_considerations_features_not_supported)

\defgroup group_init Library initialization

\brief API elements related to the initialization and management of the entire library, especially #VmaAllocator object.

\defgroup group_alloc Memory allocation

\brief API elements related to the allocation, deallocation, and management of Vulkan memory, buffers, images.
Most basic ones being: vmaCreateBuffer(), vmaCreateImage().

\defgroup group_virtual Virtual allocator

\brief API elements related to the mechanism of \ref virtual_allocator - using the core allocation algorithm
for user-defined purpose without allocating any real GPU memory.

\defgroup group_stats Statistics

\brief API elements that query current status of the allocator, from memory usage, budget, to full dump of the internal state in JSON format.
See documentation chapter: \ref statistics.
*/


#ifdef __cplusplus
extern "C" {
#endif

#if !defined(VULKAN_H_)
#include <vulkan/vulkan.h>
#endif

#if !defined(VMA_VULKAN_VERSION)
    #if defined(VK_VERSION_1_4)
        #define VMA_VULKAN_VERSION 1004000
    #elif defined(VK_VERSION_1_3)
        #define VMA_VULKAN_VERSION 1003000
    #elif defined(VK_VERSION_1_2)
        #define VMA_VULKAN_VERSION 1002000
    #elif defined(VK_VERSION_1_1)
        #define VMA_VULKAN_VERSION 1001000
    #else
        #define VMA_VULKAN_VERSION 1000000
    #endif
#endif

#if defined(__ANDROID__) && defined(VK_NO_PROTOTYPES) && VMA_STATIC_VULKAN_FUNCTIONS
    extern PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
    extern PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr;
    extern PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties;
    extern PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties;
    extern PFN_vkAllocateMemory vkAllocateMemory;
    extern PFN_vkFreeMemory vkFreeMemory;
    extern PFN_vkMapMemory vkMapMemory;
    extern PFN_vkUnmapMemory vkUnmapMemory;
    extern PFN_vkFlushMappedMemoryRanges vkFlushMappedMemoryRanges;
    extern PFN_vkInvalidateMappedMemoryRanges vkInvalidateMappedMemoryRanges;
    extern PFN_vkBindBufferMemory vkBindBufferMemory;
    extern PFN_vkBindImageMemory vkBindImageMemory;
    extern PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements;
    extern PFN_vkGetImageMemoryRequirements vkGetImageMemoryRequirements;
    extern PFN_vkCreateBuffer vkCreateBuffer;
    extern PFN_vkDestroyBuffer vkDestroyBuffer;
    extern PFN_vkCreateImage vkCreateImage;
    extern PFN_vkDestroyImage vkDestroyImage;
    extern PFN_vkCmdCopyBuffer vkCmdCopyBuffer;
    #if VMA_VULKAN_VERSION >= 1001000
        extern PFN_vkGetBufferMemoryRequirements2 vkGetBufferMemoryRequirements2;
        extern PFN_vkGetImageMemoryRequirements2 vkGetImageMemoryRequirements2;
        extern PFN_vkBindBufferMemory2 vkBindBufferMemory2;
        extern PFN_vkBindImageMemory2 vkBindImageMemory2;
        extern PFN_vkGetPhysicalDeviceMemoryProperties2 vkGetPhysicalDeviceMemoryProperties2;
    #endif // #if VMA_VULKAN_VERSION >= 1001000
#endif // #if defined(__ANDROID__) && VMA_STATIC_VULKAN_FUNCTIONS && VK_NO_PROTOTYPES

#if !defined(VMA_DEDICATED_ALLOCATION)
    #if VK_KHR_get_memory_requirements2 && VK_KHR_dedicated_allocation
        #define VMA_DEDICATED_ALLOCATION 1
    #else
        #define VMA_DEDICATED_ALLOCATION 0
    #endif
#endif

#if !defined(VMA_BIND_MEMORY2)
    #if VK_KHR_bind_memory2
        #define VMA_BIND_MEMORY2 1
    #else
        #define VMA_BIND_MEMORY2 0
    #endif
#endif

#if !defined(VMA_MEMORY_BUDGET)
    #if VK_EXT_memory_budget && (VK_KHR_get_physical_device_properties2 || VMA_VULKAN_VERSION >= 1001000)
        #define VMA_MEMORY_BUDGET 1
    #else
        #define VMA_MEMORY_BUDGET 0
    #endif
#endif

// Defined to 1 when VK_KHR_buffer_device_address device extension or equivalent core Vulkan 1.2 feature is defined in its headers.
#if !defined(VMA_BUFFER_DEVICE_ADDRESS)
    #if VK_KHR_buffer_device_address || VMA_VULKAN_VERSION >= 1002000
        #define VMA_BUFFER_DEVICE_ADDRESS 1
    #else
        #define VMA_BUFFER_DEVICE_ADDRESS 0
    #endif
#endif

// Defined to 1 when VK_EXT_memory_priority device extension is defined in Vulkan headers.
#if !defined(VMA_MEMORY_PRIORITY)
    #if VK_EXT_memory_priority
        #define VMA_MEMORY_PRIORITY 1
    #else
        #define VMA_MEMORY_PRIORITY 0
    #endif
#endif

// Defined to 1 when VK_KHR_maintenance4 device extension is defined in Vulkan headers.
#if !defined(VMA_KHR_MAINTENANCE4)
    #if VK_KHR_maintenance4
        #define VMA_KHR_MAINTENANCE4 1
    #else
        #define VMA_KHR_MAINTENANCE4 0
    #endif
#endif

// Defined to 1 when VK_KHR_maintenance5 device extension is defined in Vulkan headers.
#if !defined(VMA_KHR_MAINTENANCE5)
    #if VK_KHR_maintenance5
        #define VMA_KHR_MAINTENANCE5 1
    #else
        #define VMA_KHR_MAINTENANCE5 0
    #endif
#endif


// Defined to 1 when VK_KHR_external_memory device extension is defined in Vulkan headers.
#if !defined(VMA_EXTERNAL_MEMORY)
    #if VK_KHR_external_memory
        #define VMA_EXTERNAL_MEMORY 1
    #else
        #define VMA_EXTERNAL_MEMORY 0
    #endif
#endif

// Defined to 1 when VK_KHR_external_memory_win32 device extension is defined in Vulkan headers.
#if !defined(VMA_EXTERNAL_MEMORY_WIN32)
    #if VK_KHR_external_memory_win32
        #define VMA_EXTERNAL_MEMORY_WIN32 1
    #else
        #define VMA_EXTERNAL_MEMORY_WIN32 0
    #endif
#endif

// Define these macros to decorate all public functions with additional code,
// before and after returned type, appropriately. This may be useful for
// exporting the functions when compiling VMA as a separate library. Example:
// #define VMA_CALL_PRE  __declspec(dllexport)
// #define VMA_CALL_POST __cdecl
#ifndef VMA_CALL_PRE
    #define VMA_CALL_PRE
#endif
#ifndef VMA_CALL_POST
    #define VMA_CALL_POST
#endif

// Define this macro to decorate pNext pointers with an attribute specifying the Vulkan
// structure that will be extended via the pNext chain.
#ifndef VMA_EXTENDS_VK_STRUCT
    #define VMA_EXTENDS_VK_STRUCT(vkStruct)
#endif

// Define this macro to decorate pointers with an attribute specifying the
// length of the array they point to if they are not null.
//
// The length may be one of
// - The name of another parameter in the argument list where the pointer is declared
// - The name of another member in the struct where the pointer is declared
// - The name of a member of a struct type, meaning the value of that member in
//   the context of the call. For example
//   VMA_LEN_IF_NOT_NULL("VkPhysicalDeviceMemoryProperties::memoryHeapCount"),
//   this means the number of memory heaps available in the device associated
//   with the VmaAllocator being dealt with.
#ifndef VMA_LEN_IF_NOT_NULL
    #define VMA_LEN_IF_NOT_NULL(len)
#endif

#ifndef _Nullable
#define _Nullable
#endif

#ifndef _Nonnull
#define _Nonnull
#endif

// The VMA_NULLABLE macro is defined to be _Nullable when compiling with Clang.
// see: https://clang.llvm.org/docs/AttributeReference.html#nullable
#ifndef VMA_NULLABLE
    #ifdef __clang__
        #define VMA_NULLABLE _Nullable
    #else
        #define VMA_NULLABLE
    #endif
#endif

// The VMA_NOT_NULL macro is defined to be _Nonnull when compiling with Clang.
// see: https://clang.llvm.org/docs/AttributeReference.html#nonnull
#ifndef VMA_NOT_NULL
    #ifdef __clang__
        #define VMA_NOT_NULL _Nonnull
    #else
        #define VMA_NOT_NULL
    #endif
#endif

// If non-dispatchable handles are represented as pointers then we can give
// then nullability annotations
#ifndef VMA_NOT_NULL_NON_DISPATCHABLE
    #if defined(__LP64__) || defined(_WIN64) || (defined(__x86_64__) && !defined(__ILP32__) ) || defined(_M_X64) || defined(__ia64) || defined (_M_IA64) || defined(__aarch64__) || defined(__powerpc64__)
        #define VMA_NOT_NULL_NON_DISPATCHABLE VMA_NOT_NULL
    #else
        #define VMA_NOT_NULL_NON_DISPATCHABLE
    #endif
#endif

#ifndef VMA_NULLABLE_NON_DISPATCHABLE
    #if defined(__LP64__) || defined(_WIN64) || (defined(__x86_64__) && !defined(__ILP32__) ) || defined(_M_X64) || defined(__ia64) || defined (_M_IA64) || defined(__aarch64__) || defined(__powerpc64__)
        #define VMA_NULLABLE_NON_DISPATCHABLE VMA_NULLABLE
    #else
        #define VMA_NULLABLE_NON_DISPATCHABLE
    #endif
#endif

#ifndef VMA_STATS_STRING_ENABLED
    #define VMA_STATS_STRING_ENABLED 1
#endif

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
//    INTERFACE
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Sections for managing code placement in file, only for development purposes e.g. for convenient folding inside an IDE.
#ifndef _VMA_ENUM_DECLARATIONS

/**
\addtogroup group_init
@{
*/

/// Flags for created #VmaAllocator.
typedef enum VmaAllocatorCreateFlagBits
{
    /** \brief Allocator and all objects created from it will not be synchronized internally, so you must guarantee they are used from only one thread at a time or synchronized externally by you.

    Using this flag may increase performance because internal mutexes are not used.
    */
    VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT = 0x00000001,
    /** \brief Enables usage of VK_KHR_dedicated_allocation extension.

    The flag works only if VmaAllocatorCreateInfo::vulkanApiVersion `== VK_API_VERSION_1_0`.
    When it is `VK_API_VERSION_1_1`, the flag is ignored because the extension has been promoted to Vulkan 1.1.

    Using this extension will automatically allocate dedicated blocks of memory for
    some buffers and images instead of suballocating place for them out of bigger
    memory blocks (as if you explicitly used #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
    flag) when it is recommended by the driver. It may improve performance on some
    GPUs.

    You may set this flag only if you found out that following device extensions are
    supported, you enabled them while creating Vulkan device passed as
    VmaAllocatorCreateInfo::device, and you want them to be used internally by this
    library:

    - VK_KHR_get_memory_requirements2 (device extension)
    - VK_KHR_dedicated_allocation (device extension)

    When this flag is set, you can experience following warnings reported by Vulkan
    validation layer. You can ignore them.

    > vkBindBufferMemory(): Binding memory to buffer 0x2d but vkGetBufferMemoryRequirements() has not been called on that buffer.
    */
    VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT = 0x00000002,
    /**
    Enables usage of VK_KHR_bind_memory2 extension.

    The flag works only if VmaAllocatorCreateInfo::vulkanApiVersion `== VK_API_VERSION_1_0`.
    When it is `VK_API_VERSION_1_1`, the flag is ignored because the extension has been promoted to Vulkan 1.1.

    You may set this flag only if you found out that this device extension is supported,
    you enabled it while creating Vulkan device passed as VmaAllocatorCreateInfo::device,
    and you want it to be used internally by this library.

    The extension provides functions `vkBindBufferMemory2KHR` and `vkBindImageMemory2KHR`,
    which allow to pass a chain of `pNext` structures while binding.
    This flag is required if you use `pNext` parameter in vmaBindBufferMemory2() or vmaBindImageMemory2().
    */
    VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT = 0x00000004,
    /**
    Enables usage of VK_EXT_memory_budget extension.

    You may set this flag only if you found out that this device extension is supported,
    you enabled it while creating Vulkan device passed as VmaAllocatorCreateInfo::device,
    and you want it to be used internally by this library, along with another instance extension
    VK_KHR_get_physical_device_properties2, which is required by it (or Vulkan 1.1, where this extension is promoted).

    The extension provides query for current memory usage and budget, which will probably
    be more accurate than an estimation used by the library otherwise.
    */
    VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT = 0x00000008,
    /**
    Enables usage of VK_AMD_device_coherent_memory extension.

    You may set this flag only if you:

    - found out that this device extension is supported and enabled it while creating Vulkan device passed as VmaAllocatorCreateInfo::device,
    - checked that `VkPhysicalDeviceCoherentMemoryFeaturesAMD::deviceCoherentMemory` is true and set it while creating the Vulkan device,
    - want it to be used internally by this library.

    The extension and accompanying device feature provide access to memory types with
    `VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD` and `VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD` flags.
    They are useful mostly for writing breadcrumb markers - a common method for debugging GPU crash/hang/TDR.

    When the extension is not enabled, such memory types are still enumerated, but their usage is illegal.
    To protect from this error, if you don't create the allocator with this flag, it will refuse to allocate any memory or create a custom pool in such memory type,
    returning `VK_ERROR_FEATURE_NOT_PRESENT`.
    */
    VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT = 0x00000010,
    /**
    Enables usage of "buffer device address" feature, which allows you to use function
    `vkGetBufferDeviceAddress*` to get raw GPU pointer to a buffer and pass it for usage inside a shader.

    You may set this flag only if you:

    1. (For Vulkan version < 1.2) Found as available and enabled device extension
    VK_KHR_buffer_device_address.
    This extension is promoted to core Vulkan 1.2.
    2. Found as available and enabled device feature `VkPhysicalDeviceBufferDeviceAddressFeatures::bufferDeviceAddress`.

    When this flag is set, you can create buffers with `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` using VMA.
    The library automatically adds `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT` to
    allocated memory blocks wherever it might be needed.

    For more information, see documentation chapter \ref enabling_buffer_device_address.
    */
    VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT = 0x00000020,
    /**
    Enables usage of VK_EXT_memory_priority extension in the library.

    You may set this flag only if you found available and enabled this device extension,
    along with `VkPhysicalDeviceMemoryPriorityFeaturesEXT::memoryPriority == VK_TRUE`,
    while creating Vulkan device passed as VmaAllocatorCreateInfo::device.

    When this flag is used, VmaAllocationCreateInfo::priority and VmaPoolCreateInfo::priority
    are used to set priorities of allocated Vulkan memory. Without it, these variables are ignored.

    A priority must be a floating-point value between 0 and 1, indicating the priority of the allocation relative to other memory allocations.
    Larger values are higher priority. The granularity of the priorities is implementation-dependent.
    It is automatically passed to every call to `vkAllocateMemory` done by the library using structure `VkMemoryPriorityAllocateInfoEXT`.
    The value to be used for default priority is 0.5.
    For more details, see the documentation of the VK_EXT_memory_priority extension.
    */
    VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT = 0x00000040,
    /**
    Enables usage of VK_KHR_maintenance4 extension in the library.

    You may set this flag only if you found available and enabled this device extension,
    while creating Vulkan device passed as VmaAllocatorCreateInfo::device.
    */
    VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT = 0x00000080,
    /**
    Enables usage of VK_KHR_maintenance5 extension in the library.

    You should set this flag if you found available and enabled this device extension,
    while creating Vulkan device passed as VmaAllocatorCreateInfo::device.
    */
    VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT = 0x00000100,

    /**
    Enables usage of VK_KHR_external_memory_win32 extension in the library.

    You should set this flag if you found available and enabled this device extension,
    while creating Vulkan device passed as VmaAllocatorCreateInfo::device.
    For more information, see \ref vk_khr_external_memory_win32.
    */
    VMA_ALLOCATOR_CREATE_KHR_EXTERNAL_MEMORY_WIN32_BIT = 0x00000200,

    VMA_ALLOCATOR_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VmaAllocatorCreateFlagBits;
/// See #VmaAllocatorCreateFlagBits.
typedef VkFlags VmaAllocatorCreateFlags;

/** @} */

/**
\addtogroup group_alloc
@{
*/

/// \brief Intended usage of the allocated memory.
typedef enum VmaMemoryUsage
{
    /** No intended memory usage specified.
    Use other members of VmaAllocationCreateInfo to specify your requirements.
    */
    VMA_MEMORY_USAGE_UNKNOWN = 0,
    /**
    \deprecated Obsolete, preserved for backward compatibility.
    Prefers `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`.
    */
    VMA_MEMORY_USAGE_GPU_ONLY = 1,
    /**
    \deprecated Obsolete, preserved for backward compatibility.
    Guarantees `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` and `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`.
    */
    VMA_MEMORY_USAGE_CPU_ONLY = 2,
    /**
    \deprecated Obsolete, preserved for backward compatibility.
    Guarantees `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`, prefers `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`.
    */
    VMA_MEMORY_USAGE_CPU_TO_GPU = 3,
    /**
    \deprecated Obsolete, preserved for backward compatibility.
    Guarantees `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`, prefers `VK_MEMORY_PROPERTY_HOST_CACHED_BIT`.
    */
    VMA_MEMORY_USAGE_GPU_TO_CPU = 4,
    /**
    \deprecated Obsolete, preserved for backward compatibility.
    Prefers not `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`.
    */
    VMA_MEMORY_USAGE_CPU_COPY = 5,
    /**
    Lazily allocated GPU memory having `VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT`.
    Exists mostly on mobile platforms. Using it on desktop PC or other GPUs with no such memory type present will fail the allocation.

    Usage: Memory for transient attachment images (color attachments, depth attachments etc.), created with `VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT`.

    Allocations with this usage are always created as dedicated - it implies #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.
    */
    VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED = 6,
    /**
    Selects best memory type automatically.
    This flag is recommended for most common use cases.

    When using this flag, if you want to map the allocation (using vmaMapMemory() or #VMA_ALLOCATION_CREATE_MAPPED_BIT),
    you must pass one of the flags: #VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or #VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
    in VmaAllocationCreateInfo::flags.

    It can be used only with functions that let the library know `VkBufferCreateInfo` or `VkImageCreateInfo`, e.g.
    vmaCreateBuffer(), vmaCreateImage(), vmaFindMemoryTypeIndexForBufferInfo(), vmaFindMemoryTypeIndexForImageInfo()
    and not with generic memory allocation functions.
    */
    VMA_MEMORY_USAGE_AUTO = 7,
    /**
    Selects best memory type automatically with preference for GPU (device) memory.

    When using this flag, if you want to map the allocation (using vmaMapMemory() or #VMA_ALLOCATION_CREATE_MAPPED_BIT),
    you must pass one of the flags: #VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or #VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
    in VmaAllocationCreateInfo::flags.

    It can be used only with functions that let the library know `VkBufferCreateInfo` or `VkImageCreateInfo`, e.g.
    vmaCreateBuffer(), vmaCreateImage(), vmaFindMemoryTypeIndexForBufferInfo(), vmaFindMemoryTypeIndexForImageInfo()
    and not with generic memory allocation functions.
    */
    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE = 8,
    /**
    Selects best memory type automatically with preference for CPU (host) memory.

    When using this flag, if you want to map the allocation (using vmaMapMemory() or #VMA_ALLOCATION_CREATE_MAPPED_BIT),
    you must pass one of the flags: #VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or #VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
    in VmaAllocationCreateInfo::flags.

    It can be used only with functions that let the library know `VkBufferCreateInfo` or `VkImageCreateInfo`, e.g.
    vmaCreateBuffer(), vmaCreateImage(), vmaFindMemoryTypeIndexForBufferInfo(), vmaFindMemoryTypeIndexForImageInfo()
    and not with generic memory allocation functions.
    */
    VMA_MEMORY_USAGE_AUTO_PREFER_HOST = 9,

    VMA_MEMORY_USAGE_MAX_ENUM = 0x7FFFFFFF
} VmaMemoryUsage;

/// Flags to be passed as VmaAllocationCreateInfo::flags.
typedef enum VmaAllocationCreateFlagBits
{
    /** \brief Set this flag if the allocation should have its own memory block.

    Use it for special, big resources, like fullscreen images used as attachments.

    If you use this flag while creating a buffer or an image, `VkMemoryDedicatedAllocateInfo`
    structure is applied if possible.
    */
    VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT = 0x00000001,

    /** \brief Set this flag to only try to allocate from existing `VkDeviceMemory` blocks and never create new such block.

    If new allocation cannot be placed in any of the existing blocks, allocation
    fails with `VK_ERROR_OUT_OF_DEVICE_MEMORY` error.

    You should not use #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT and
    #VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT at the same time. It makes no sense.
    */
    VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT = 0x00000002,
    /** \brief Set this flag to use a memory that will be persistently mapped and retrieve pointer to it.

    Pointer to mapped memory will be returned through VmaAllocationInfo::pMappedData.

    It is valid to use this flag for allocation made from memory type that is not
    `HOST_VISIBLE`. This flag is then ignored and memory is not mapped. This is
    useful if you need an allocation that is efficient to use on GPU
    (`DEVICE_LOCAL`) and still want to map it directly if possible on platforms that
    support it (e.g. Intel GPU).
    */
    VMA_ALLOCATION_CREATE_MAPPED_BIT = 0x00000004,
    /** \deprecated Preserved for backward compatibility. Consider using vmaSetAllocationName() instead.

    Set this flag to treat VmaAllocationCreateInfo::pUserData as pointer to a
    null-terminated string. Instead of copying pointer value, a local copy of the
    string is made and stored in allocation's `pName`. The string is automatically
    freed together with the allocation. It is also used in vmaBuildStatsString().
    */
    VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT = 0x00000020,
    /** Allocation will be created from upper stack in a double stack pool.

    This flag is only allowed for custom pools created with #VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT flag.
    */
    VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT = 0x00000040,
    /** Create both buffer/image and allocation, but don't bind them together.
    It is useful when you want to bind yourself to do some more advanced binding, e.g. using some extensions.
    The flag is meaningful only with functions that bind by default: vmaCreateBuffer(), vmaCreateImage().
    Otherwise it is ignored.

    If you want to make sure the new buffer/image is not tied to the new memory allocation
    through `VkMemoryDedicatedAllocateInfoKHR` structure in case the allocation ends up in its own memory block,
    use also flag #VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT.
    */
    VMA_ALLOCATION_CREATE_DONT_BIND_BIT = 0x00000080,
    /** Create allocation only if additional device memory required for it, if any, won't exceed
    memory budget. Otherwise return `VK_ERROR_OUT_OF_DEVICE_MEMORY`.
    */
    VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT = 0x00000100,
    /** \brief Set this flag if the allocated memory will have aliasing resources.

    Usage of this flag prevents supplying `VkMemoryDedicatedAllocateInfoKHR` when #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT is specified.
    Otherwise created dedicated memory will not be suitable for aliasing resources, resulting in Vulkan Validation Layer errors.
    */
    VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT = 0x00000200,
    /**
    Requests possibility to map the allocation (using vmaMapMemory() or #VMA_ALLOCATION_CREATE_MAPPED_BIT).

    - If you use #VMA_MEMORY_USAGE_AUTO or other `VMA_MEMORY_USAGE_AUTO*` value,
      you must use this flag to be able to map the allocation. Otherwise, mapping is incorrect.
    - If you use other value of #VmaMemoryUsage, this flag is ignored and mapping is always possible in memory types that are `HOST_VISIBLE`.
      This includes allocations created in \ref custom_memory_pools.

    Declares that mapped memory will only be written sequentially, e.g. using `memcpy()` or a loop writing number-by-number,
    never read or accessed randomly, so a memory type can be selected that is uncached and write-combined.

    \warning Violating this declaration may work correctly, but will likely be very slow.
    Watch out for implicit reads introduced by doing e.g. `pMappedData[i] += x;`
    Better prepare your data in a local variable and `memcpy()` it to the mapped pointer all at once.
    */
    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT = 0x00000400,
    /**
    Requests possibility to map the allocation (using vmaMapMemory() or #VMA_ALLOCATION_CREATE_MAPPED_BIT).

    - If you use #VMA_MEMORY_USAGE_AUTO or other `VMA_MEMORY_USAGE_AUTO*` value,
      you must use this flag to be able to map the allocation. Otherwise, mapping is incorrect.
    - If you use other value of #VmaMemoryUsage, this flag is ignored and mapping is always possible in memory types that are `HOST_VISIBLE`.
      This includes allocations created in \ref custom_memory_pools.

    Declares that mapped memory can be read, written, and accessed in random order,
    so a `HOST_CACHED` memory type is preferred.
    */
    VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT = 0x00000800,
    /**
    Together with #VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or #VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
    it says that despite request for host access, a not-`HOST_VISIBLE` memory type can be selected
    if it may improve performance.

    By using this flag, you declare that you will check if the allocation ended up in a `HOST_VISIBLE` memory type
    (e.g. using vmaGetAllocationMemoryProperties()) and if not, you will create some "staging" buffer and
    issue an explicit transfer to write/read your data.
    To prepare for this possibility, don't forget to add appropriate flags like
    `VK_BUFFER_USAGE_TRANSFER_DST_BIT`, `VK_BUFFER_USAGE_TRANSFER_SRC_BIT` to the parameters of created buffer or image.
    */
    VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT = 0x00001000,
    /** Allocation strategy that chooses smallest possible free range for the allocation
    to minimize memory usage and fragmentation, possibly at the expense of allocation time.
    */
    VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT = 0x00010000,
    /** Allocation strategy that chooses first suitable free range for the allocation -
    not necessarily in terms of the smallest offset but the one that is easiest and fastest to find
    to minimize allocation time, possibly at the expense of allocation quality.
    */
    VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT = 0x00020000,
    /** Allocation strategy that chooses always the lowest offset in available space.
    This is not the most efficient strategy but achieves highly packed data.
    Used internally by defragmentation, not recommended in typical usage.
    */
    VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT  = 0x00040000,
    /** Alias to #VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT.
    */
    VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT = VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT,
    /** Alias to #VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT.
    */
    VMA_ALLOCATION_CREATE_STRATEGY_FIRST_FIT_BIT = VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT,
    /** A bit mask to extract only `STRATEGY` bits from entire set of flags.
    */
    VMA_ALLOCATION_CREATE_STRATEGY_MASK =
        VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT |
        VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT |
        VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT,

    VMA_ALLOCATION_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VmaAllocationCreateFlagBits;
/// See #VmaAllocationCreateFlagBits.
typedef VkFlags VmaAllocationCreateFlags;

/// Flags to be passed as VmaPoolCreateInfo::flags.
typedef enum VmaPoolCreateFlagBits
{
    /** \brief Use this flag if you always allocate only buffers and linear images or only optimal images out of this pool and so Buffer-Image Granularity can be ignored.

    This is an optional optimization flag.

    If you always allocate using vmaCreateBuffer(), vmaCreateImage(),
    vmaAllocateMemoryForBuffer(), then you don't need to use it because allocator
    knows exact type of your allocations so it can handle Buffer-Image Granularity
    in the optimal way.

    If you also allocate using vmaAllocateMemoryForImage() or vmaAllocateMemory(),
    exact type of such allocations is not known, so allocator must be conservative
    in handling Buffer-Image Granularity, which can lead to suboptimal allocation
    (wasted memory). In that case, if you can make sure you always allocate only
    buffers and linear images or only optimal images out of this pool, use this flag
    to make allocator disregard Buffer-Image Granularity and so make allocations
    faster and more optimal.
    */
    VMA_POOL_CREATE_IGNORE_BUFFER_IMAGE_GRANULARITY_BIT = 0x00000002,

    /** \brief Enables alternative, linear allocation algorithm in this pool.

    Specify this flag to enable linear allocation algorithm, which always creates
    new allocations after last one and doesn't reuse space from allocations freed in
    between. It trades memory consumption for simplified algorithm and data
    structure, which has better performance and uses less memory for metadata.

    By using this flag, you can achieve behavior of free-at-once, stack,
    ring buffer, and double stack.
    For details, see documentation chapter \ref linear_algorithm.
    */
    VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT = 0x00000004,

    /** Bit mask to extract only `ALGORITHM` bits from entire set of flags.
    */
    VMA_POOL_CREATE_ALGORITHM_MASK =
        VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT,

    VMA_POOL_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VmaPoolCreateFlagBits;
/// Flags to be passed as VmaPoolCreateInfo::flags. See #VmaPoolCreateFlagBits.
typedef VkFlags VmaPoolCreateFlags;

/// Flags to be passed as VmaDefragmentationInfo::flags.
typedef enum VmaDefragmentationFlagBits
{
    /* \brief Use simple but fast algorithm for defragmentation.
    May not achieve best results but will require least time to compute and least allocations to copy.
    */
    VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FAST_BIT = 0x1,
    /* \brief Default defragmentation algorithm, applied also when no `ALGORITHM` flag is specified.
    Offers a balance between defragmentation quality and the amount of allocations and bytes that need to be moved.
    */
    VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT = 0x2,
    /* \brief Perform full defragmentation of memory.
    Can result in notably more time to compute and allocations to copy, but will achieve best memory packing.
    */
    VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FULL_BIT = 0x4,
    /** \brief Use the most roboust algorithm at the cost of time to compute and number of copies to make.
    Only available when bufferImageGranularity is greater than 1, since it aims to reduce
    alignment issues between different types of resources.
    Otherwise falls back to same behavior as #VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FULL_BIT.
    */
    VMA_DEFRAGMENTATION_FLAG_ALGORITHM_EXTENSIVE_BIT = 0x8,

    /// A bit mask to extract only `ALGORITHM` bits from entire set of flags.
    VMA_DEFRAGMENTATION_FLAG_ALGORITHM_MASK =
        VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FAST_BIT |
        VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT |
        VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FULL_BIT |
        VMA_DEFRAGMENTATION_FLAG_ALGORITHM_EXTENSIVE_BIT,

    VMA_DEFRAGMENTATION_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VmaDefragmentationFlagBits;
/// See #VmaDefragmentationFlagBits.
typedef VkFlags VmaDefragmentationFlags;

/// Operation performed on single defragmentation move. See structure #VmaDefragmentationMove.
typedef enum VmaDefragmentationMoveOperation
{
    /// Buffer/image has been recreated at `dstTmpAllocation`, data has been copied, old buffer/image has been destroyed. `srcAllocation` should be changed to point to the new place. This is the default value set by vmaBeginDefragmentationPass().
    VMA_DEFRAGMENTATION_MOVE_OPERATION_COPY = 0,
    /// Set this value if you cannot move the allocation. New place reserved at `dstTmpAllocation` will be freed. `srcAllocation` will remain unchanged.
    VMA_DEFRAGMENTATION_MOVE_OPERATION_IGNORE = 1,
    /// Set this value if you decide to abandon the allocation and you destroyed the buffer/image. New place reserved at `dstTmpAllocation` will be freed, along with `srcAllocation`, which will be destroyed.
    VMA_DEFRAGMENTATION_MOVE_OPERATION_DESTROY = 2,
} VmaDefragmentationMoveOperation;

/** @} */

/**
\addtogroup group_virtual
@{
*/

/// Flags to be passed as VmaVirtualBlockCreateInfo::flags.
typedef enum VmaVirtualBlockCreateFlagBits
{
    /** \brief Enables alternative, linear allocation algorithm in this virtual block.

    Specify this flag to enable linear allocation algorithm, which always creates
    new allocations after last one and doesn't reuse space from allocations freed in
    between. It trades memory consumption for simplified algorithm and data
    structure, which has better performance and uses less memory for metadata.

    By using this flag, you can achieve behavior of free-at-once, stack,
    ring buffer, and double stack.
    For details, see documentation chapter \ref linear_algorithm.
    */
    VMA_VIRTUAL_BLOCK_CREATE_LINEAR_ALGORITHM_BIT = 0x00000001,

    /** \brief Bit mask to extract only `ALGORITHM` bits from entire set of flags.
    */
    VMA_VIRTUAL_BLOCK_CREATE_ALGORITHM_MASK =
        VMA_VIRTUAL_BLOCK_CREATE_LINEAR_ALGORITHM_BIT,

    VMA_VIRTUAL_BLOCK_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VmaVirtualBlockCreateFlagBits;
/// Flags to be passed as VmaVirtualBlockCreateInfo::flags. See #VmaVirtualBlockCreateFlagBits.
typedef VkFlags VmaVirtualBlockCreateFlags;

/// Flags to be passed as VmaVirtualAllocationCreateInfo::flags.
typedef enum VmaVirtualAllocationCreateFlagBits
{
    /** \brief Allocation will be created from upper stack in a double stack pool.

    This flag is only allowed for virtual blocks created with #VMA_VIRTUAL_BLOCK_CREATE_LINEAR_ALGORITHM_BIT flag.
    */
    VMA_VIRTUAL_ALLOCATION_CREATE_UPPER_ADDRESS_BIT = VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT,
    /** \brief Allocation strategy that tries to minimize memory usage.
    */
    VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT = VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT,
    /** \brief Allocation strategy that tries to minimize allocation time.
    */
    VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT = VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT,
    /** Allocation strategy that chooses always the lowest offset in available space.
    This is not the most efficient strategy but achieves highly packed data.
    */
    VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT = VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT,
    /** \brief A bit mask to extract only `STRATEGY` bits from entire set of flags.

    These strategy flags are binary compatible with equivalent flags in #VmaAllocationCreateFlagBits.
    */
    VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MASK = VMA_ALLOCATION_CREATE_STRATEGY_MASK,

    VMA_VIRTUAL_ALLOCATION_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VmaVirtualAllocationCreateFlagBits;
/// Flags to be passed as VmaVirtualAllocationCreateInfo::flags. See #VmaVirtualAllocationCreateFlagBits.
typedef VkFlags VmaVirtualAllocationCreateFlags;

/** @} */

#endif // _VMA_ENUM_DECLARATIONS

#ifndef _VMA_DATA_TYPES_DECLARATIONS

/**
\addtogroup group_init
@{ */

/** \struct VmaAllocator
\brief Represents main object of this library initialized.

Fill structure #VmaAllocatorCreateInfo and call function vmaCreateAllocator() to create it.
Call function vmaDestroyAllocator() to destroy it.

It is recommended to create just one object of this type per `VkDevice` object,
right after Vulkan is initialized and keep it alive until before Vulkan device is destroyed.
*/
VK_DEFINE_HANDLE(VmaAllocator)

/** @} */

/**
\addtogroup group_alloc
@{
*/

/** \struct VmaPool
\brief Represents custom memory pool

Fill structure VmaPoolCreateInfo and call function vmaCreatePool() to create it.
Call function vmaDestroyPool() to destroy it.

For more information see [Custom memory pools](@ref choosing_memory_type_custom_memory_pools).
*/
VK_DEFINE_HANDLE(VmaPool)

/** \struct VmaAllocation
\brief Represents single memory allocation.

It may be either dedicated block of `VkDeviceMemory` or a specific region of a bigger block of this type
plus unique offset.

There are multiple ways to create such object.
You need to fill structure VmaAllocationCreateInfo.
For more information see [Choosing memory type](@ref choosing_memory_type).

Although the library provides convenience functions that create Vulkan buffer or image,
allocate memory for it and bind them together,
binding of the allocation to a buffer or an image is out of scope of the allocation itself.
Allocation object can exist without buffer/image bound,
binding can be done manually by the user, and destruction of it can be done
independently of destruction of the allocation.

The object also remembers its size and some other information.
To retrieve this information, use function vmaGetAllocationInfo() and inspect
returned structure VmaAllocationInfo.
*/
VK_DEFINE_HANDLE(VmaAllocation)

/** \struct VmaDefragmentationContext
\brief An opaque object that represents started defragmentation process.

Fill structure #VmaDefragmentationInfo and call function vmaBeginDefragmentation() to create it.
Call function vmaEndDefragmentation() to destroy it.
*/
VK_DEFINE_HANDLE(VmaDefragmentationContext)

/** @} */

/**
\addtogroup group_virtual
@{
*/

/** \struct VmaVirtualAllocation
\brief Represents single memory allocation done inside VmaVirtualBlock.

Use it as a unique identifier to virtual allocation within the single block.

Use value `VK_NULL_HANDLE` to represent a null/invalid allocation.
*/
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VmaVirtualAllocation)

/** @} */

/**
\addtogroup group_virtual
@{
*/

/** \struct VmaVirtualBlock
\brief Handle to a virtual block object that allows to use core allocation algorithm without allocating any real GPU memory.

Fill in #VmaVirtualBlockCreateInfo structure and use vmaCreateVirtualBlock() to create it. Use vmaDestroyVirtualBlock() to destroy it.
For more information, see documentation chapter \ref virtual_allocator.

This object is not thread-safe - should not be used from multiple threads simultaneously, must be synchronized externally.
*/
VK_DEFINE_HANDLE(VmaVirtualBlock)

/** @} */

/**
\addtogroup group_init
@{
*/

/// Callback function called after successful vkAllocateMemory.
typedef void (VKAPI_PTR* PFN_vmaAllocateDeviceMemoryFunction)(
    VmaAllocator VMA_NOT_NULL                    allocator,
    uint32_t                                     memoryType,
    VkDeviceMemory VMA_NOT_NULL_NON_DISPATCHABLE memory,
    VkDeviceSize                                 size,
    void* VMA_NULLABLE                           pUserData);

/// Callback function called before vkFreeMemory.
typedef void (VKAPI_PTR* PFN_vmaFreeDeviceMemoryFunction)(
    VmaAllocator VMA_NOT_NULL                    allocator,
    uint32_t                                     memoryType,
    VkDeviceMemory VMA_NOT_NULL_NON_DISPATCHABLE memory,
    VkDeviceSize                                 size,
    void* VMA_NULLABLE                           pUserData);

/** \brief Set of callbacks that the library will call for `vkAllocateMemory` and `vkFreeMemory`.

Provided for informative purpose, e.g. to gather statistics about number of
allocations or total amount of memory allocated in Vulkan.

Used in VmaAllocatorCreateInfo::pDeviceMemoryCallbacks.
*/
typedef struct VmaDeviceMemoryCallbacks
{
    /// Optional, can be null.
    PFN_vmaAllocateDeviceMemoryFunction VMA_NULLABLE pfnAllocate;
    /// Optional, can be null.
    PFN_vmaFreeDeviceMemoryFunction VMA_NULLABLE pfnFree;
    /// Optional, can be null.
    void* VMA_NULLABLE pUserData;
} VmaDeviceMemoryCallbacks;

/** \brief Pointers to some Vulkan functions - a subset used by the library.

Used in VmaAllocatorCreateInfo::pVulkanFunctions.
*/
typedef struct VmaVulkanFunctions
{
    /// Required when using VMA_DYNAMIC_VULKAN_FUNCTIONS.
    PFN_vkGetInstanceProcAddr VMA_NULLABLE vkGetInstanceProcAddr;
    /// Required when using VMA_DYNAMIC_VULKAN_FUNCTIONS.
    PFN_vkGetDeviceProcAddr VMA_NULLABLE vkGetDeviceProcAddr;
    PFN_vkGetPhysicalDeviceProperties VMA_NULLABLE vkGetPhysicalDeviceProperties;
    PFN_vkGetPhysicalDeviceMemoryProperties VMA_NULLABLE vkGetPhysicalDeviceMemoryProperties;
    PFN_vkAllocateMemory VMA_NULLABLE vkAllocateMemory;
    PFN_vkFreeMemory VMA_NULLABLE vkFreeMemory;
    PFN_vkMapMemory VMA_NULLABLE vkMapMemory;
    PFN_vkUnmapMemory VMA_NULLABLE vkUnmapMemory;
    PFN_vkFlushMappedMemoryRanges VMA_NULLABLE vkFlushMappedMemoryRanges;
    PFN_vkInvalidateMappedMemoryRanges VMA_NULLABLE vkInvalidateMappedMemoryRanges;
    PFN_vkBindBufferMemory VMA_NULLABLE vkBindBufferMemory;
    PFN_vkBindImageMemory VMA_NULLABLE vkBindImageMemory;
    PFN_vkGetBufferMemoryRequirements VMA_NULLABLE vkGetBufferMemoryRequirements;
    PFN_vkGetImageMemoryRequirements VMA_NULLABLE vkGetImageMemoryRequirements;
    PFN_vkCreateBuffer VMA_NULLABLE vkCreateBuffer;
    PFN_vkDestroyBuffer VMA_NULLABLE vkDestroyBuffer;
    PFN_vkCreateImage VMA_NULLABLE vkCreateImage;
    PFN_vkDestroyImage VMA_NULLABLE vkDestroyImage;
    PFN_vkCmdCopyBuffer VMA_NULLABLE vkCmdCopyBuffer;
#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
    /// Fetch "vkGetBufferMemoryRequirements2" on Vulkan >= 1.1, fetch "vkGetBufferMemoryRequirements2KHR" when using VK_KHR_dedicated_allocation extension.
    PFN_vkGetBufferMemoryRequirements2KHR VMA_NULLABLE vkGetBufferMemoryRequirements2KHR;
    /// Fetch "vkGetImageMemoryRequirements2" on Vulkan >= 1.1, fetch "vkGetImageMemoryRequirements2KHR" when using VK_KHR_dedicated_allocation extension.
    PFN_vkGetImageMemoryRequirements2KHR VMA_NULLABLE vkGetImageMemoryRequirements2KHR;
#endif
#if VMA_BIND_MEMORY2 || VMA_VULKAN_VERSION >= 1001000
    /// Fetch "vkBindBufferMemory2" on Vulkan >= 1.1, fetch "vkBindBufferMemory2KHR" when using VK_KHR_bind_memory2 extension.
    PFN_vkBindBufferMemory2KHR VMA_NULLABLE vkBindBufferMemory2KHR;
    /// Fetch "vkBindImageMemory2" on Vulkan >= 1.1, fetch "vkBindImageMemory2KHR" when using VK_KHR_bind_memory2 extension.
    PFN_vkBindImageMemory2KHR VMA_NULLABLE vkBindImageMemory2KHR;
#endif
#if VMA_MEMORY_BUDGET || VMA_VULKAN_VERSION >= 1001000
    /// Fetch from "vkGetPhysicalDeviceMemoryProperties2" on Vulkan >= 1.1, but you can also fetch it from "vkGetPhysicalDeviceMemoryProperties2KHR" if you enabled extension VK_KHR_get_physical_device_properties2.
    PFN_vkGetPhysicalDeviceMemoryProperties2KHR VMA_NULLABLE vkGetPhysicalDeviceMemoryProperties2KHR;
#endif
#if VMA_KHR_MAINTENANCE4 || VMA_VULKAN_VERSION >= 1003000
    /// Fetch from "vkGetDeviceBufferMemoryRequirements" on Vulkan >= 1.3, but you can also fetch it from "vkGetDeviceBufferMemoryRequirementsKHR" if you enabled extension VK_KHR_maintenance4.
    PFN_vkGetDeviceBufferMemoryRequirementsKHR VMA_NULLABLE vkGetDeviceBufferMemoryRequirements;
    /// Fetch from "vkGetDeviceImageMemoryRequirements" on Vulkan >= 1.3, but you can also fetch it from "vkGetDeviceImageMemoryRequirementsKHR" if you enabled extension VK_KHR_maintenance4.
    PFN_vkGetDeviceImageMemoryRequirementsKHR VMA_NULLABLE vkGetDeviceImageMemoryRequirements;
#endif
#if VMA_EXTERNAL_MEMORY_WIN32
    PFN_vkGetMemoryWin32HandleKHR VMA_NULLABLE vkGetMemoryWin32HandleKHR;
#else
    void* VMA_NULLABLE vkGetMemoryWin32HandleKHR;
#endif
} VmaVulkanFunctions;

/// Description of a Allocator to be created.
typedef struct VmaAllocatorCreateInfo
{
    /// Flags for created allocator. Use #VmaAllocatorCreateFlagBits enum.
    VmaAllocatorCreateFlags flags;
    /// Vulkan physical device.
    /** It must be valid throughout whole lifetime of created allocator. */
    VkPhysicalDevice VMA_NOT_NULL physicalDevice;
    /// Vulkan device.
    /** It must be valid throughout whole lifetime of created allocator. */
    VkDevice VMA_NOT_NULL device;
    /// Preferred size of a single `VkDeviceMemory` block to be allocated from large heaps > 1 GiB. Optional.
    /** Set to 0 to use default, which is currently 256 MiB. */
    VkDeviceSize preferredLargeHeapBlockSize;
    /// Custom CPU memory allocation callbacks. Optional.
    /** Optional, can be null. When specified, will also be used for all CPU-side memory allocations. */
    const VkAllocationCallbacks* VMA_NULLABLE pAllocationCallbacks;
    /// Informative callbacks for `vkAllocateMemory`, `vkFreeMemory`. Optional.
    /** Optional, can be null. */
    const VmaDeviceMemoryCallbacks* VMA_NULLABLE pDeviceMemoryCallbacks;
    /** \brief Either null or a pointer to an array of limits on maximum number of bytes that can be allocated out of particular Vulkan memory heap.

    If not NULL, it must be a pointer to an array of
    `VkPhysicalDeviceMemoryProperties::memoryHeapCount` elements, defining limit on
    maximum number of bytes that can be allocated out of particular Vulkan memory
    heap.

    Any of the elements may be equal to `VK_WHOLE_SIZE`, which means no limit on that
    heap. This is also the default in case of `pHeapSizeLimit` = NULL.

    If there is a limit defined for a heap:

    - If user tries to allocate more memory from that heap using this allocator,
      the allocation fails with `VK_ERROR_OUT_OF_DEVICE_MEMORY`.
    - If the limit is smaller than heap size reported in `VkMemoryHeap::size`, the
      value of this limit will be reported instead when using vmaGetMemoryProperties().

    Warning! Using this feature may not be equivalent to installing a GPU with
    smaller amount of memory, because graphics driver doesn't necessary fail new
    allocations with `VK_ERROR_OUT_OF_DEVICE_MEMORY` result when memory capacity is
    exceeded. It may return success and just silently migrate some device memory
    blocks to system RAM. This driver behavior can also be controlled using
    VK_AMD_memory_overallocation_behavior extension.
    */
    const VkDeviceSize* VMA_NULLABLE VMA_LEN_IF_NOT_NULL("VkPhysicalDeviceMemoryProperties::memoryHeapCount") pHeapSizeLimit;

    /** \brief Pointers to Vulkan functions. Can be null.

    For details see [Pointers to Vulkan functions](@ref config_Vulkan_functions).
    */
    const VmaVulkanFunctions* VMA_NULLABLE pVulkanFunctions;
    /** \brief Handle to Vulkan instance object.

    Starting from version 3.0.0 this member is no longer optional, it must be set!
    */
    VkInstance VMA_NOT_NULL instance;
    /** \brief Optional. Vulkan version that the application uses.

    It must be a value in the format as created by macro `VK_MAKE_VERSION` or a constant like: `VK_API_VERSION_1_1`, `VK_API_VERSION_1_0`.
    The patch version number specified is ignored. Only the major and minor versions are considered.
    Only versions 1.0...1.4 are supported by the current implementation.
    Leaving it initialized to zero is equivalent to `VK_API_VERSION_1_0`.
    It must match the Vulkan version used by the application and supported on the selected physical device,
    so it must be no higher than `VkApplicationInfo::apiVersion` passed to `vkCreateInstance`
    and no higher than `VkPhysicalDeviceProperties::apiVersion` found on the physical device used.
    */
    uint32_t vulkanApiVersion;
#if VMA_EXTERNAL_MEMORY
    /** \brief Either null or a pointer to an array of external memory handle types for each Vulkan memory type.

    If not NULL, it must be a pointer to an array of `VkPhysicalDeviceMemoryProperties::memoryTypeCount`
    elements, defining external memory handle types of particular Vulkan memory type,
    to be passed using `VkExportMemoryAllocateInfoKHR`.

    Any of the elements may be equal to 0, which means not to use `VkExportMemoryAllocateInfoKHR` on this memory type.
    This is also the default in case of `pTypeExternalMemoryHandleTypes` = NULL.
    */
    const VkExternalMemoryHandleTypeFlagsKHR* VMA_NULLABLE VMA_LEN_IF_NOT_NULL("VkPhysicalDeviceMemoryProperties::memoryTypeCount") pTypeExternalMemoryHandleTypes;
#endif // #if VMA_EXTERNAL_MEMORY
} VmaAllocatorCreateInfo;

/// Information about existing #VmaAllocator object.
typedef struct VmaAllocatorInfo
{
    /** \brief Handle to Vulkan instance object.

    This is the same value as has been passed through VmaAllocatorCreateInfo::instance.
    */
    VkInstance VMA_NOT_NULL instance;
    /** \brief Handle to Vulkan physical device object.

    This is the same value as has been passed through VmaAllocatorCreateInfo::physicalDevice.
    */
    VkPhysicalDevice VMA_NOT_NULL physicalDevice;
    /** \brief Handle to Vulkan device object.

    This is the same value as has been passed through VmaAllocatorCreateInfo::device.
    */
    VkDevice VMA_NOT_NULL device;
} VmaAllocatorInfo;

/** @} */

/**
\addtogroup group_stats
@{
*/

/** \brief Calculated statistics of memory usage e.g. in a specific memory type, heap, custom pool, or total.

These are fast to calculate.
See functions: vmaGetHeapBudgets(), vmaGetPoolStatistics().
*/
typedef struct VmaStatistics
{
    /** \brief Number of `VkDeviceMemory` objects - Vulkan memory blocks allocated.
    */
    uint32_t blockCount;
    /** \brief Number of #VmaAllocation objects allocated.

    Dedicated allocations have their own blocks, so each one adds 1 to `allocationCount` as well as `blockCount`.
    */
    uint32_t allocationCount;
    /** \brief Number of bytes allocated in `VkDeviceMemory` blocks.

    \note To avoid confusion, please be aware that what Vulkan calls an "allocation" - a whole `VkDeviceMemory` object
    (e.g. as in `VkPhysicalDeviceLimits::maxMemoryAllocationCount`) is called a "block" in VMA, while VMA calls
    "allocation" a #VmaAllocation object that represents a memory region sub-allocated from such block, usually for a single buffer or image.
    */
    VkDeviceSize blockBytes;
    /** \brief Total number of bytes occupied by all #VmaAllocation objects.

    Always less or equal than `blockBytes`.
    Difference `(blockBytes - allocationBytes)` is the amount of memory allocated from Vulkan
    but unused by any #VmaAllocation.
    */
    VkDeviceSize allocationBytes;
} VmaStatistics;

/** \brief More detailed statistics than #VmaStatistics.

These are slower to calculate. Use for debugging purposes.
See functions: vmaCalculateStatistics(), vmaCalculatePoolStatistics().

Previous version of the statistics API provided averages, but they have been removed
because they can be easily calculated as:

\code
VkDeviceSize allocationSizeAvg = detailedStats.statistics.allocationBytes / detailedStats.statistics.allocationCount;
VkDeviceSize unusedBytes = detailedStats.statistics.blockBytes - detailedStats.statistics.allocationBytes;
VkDeviceSize unusedRangeSizeAvg = unusedBytes / detailedStats.unusedRangeCount;
\endcode
*/
typedef struct VmaDetailedStatistics
{
    /// Basic statistics.
    VmaStatistics statistics;
    /// Number of free ranges of memory between allocations.
    uint32_t unusedRangeCount;
    /// Smallest allocation size. `VK_WHOLE_SIZE` if there are 0 allocations.
    VkDeviceSize allocationSizeMin;
    /// Largest allocation size. 0 if there are 0 allocations.
    VkDeviceSize allocationSizeMax;
    /// Smallest empty range size. `VK_WHOLE_SIZE` if there are 0 empty ranges.
    VkDeviceSize unusedRangeSizeMin;
    /// Largest empty range size. 0 if there are 0 empty ranges.
    VkDeviceSize unusedRangeSizeMax;
} VmaDetailedStatistics;

/** \brief  General statistics from current state of the Allocator -
total memory usage across all memory heaps and types.

These are slower to calculate. Use for debugging purposes.
See function vmaCalculateStatistics().
*/
typedef struct VmaTotalStatistics
{
    VmaDetailedStatistics memoryType[VK_MAX_MEMORY_TYPES];
    VmaDetailedStatistics memoryHeap[VK_MAX_MEMORY_HEAPS];
    VmaDetailedStatistics total;
} VmaTotalStatistics;

/** \brief Statistics of current memory usage and available budget for a specific memory heap.

These are fast to calculate.
See function vmaGetHeapBudgets().
*/
typedef struct VmaBudget
{
    /** \brief Statistics fetched from the library.
    */
    VmaStatistics statistics;
    /** \brief Estimated current memory usage of the program, in bytes.

    Fetched from system using VK_EXT_memory_budget extension if enabled.

    It might be different than `statistics.blockBytes` (usually higher) due to additional implicit objects
    also occupying the memory, like swapchain, pipelines, descriptor heaps, command buffers, or
    `VkDeviceMemory` blocks allocated outside of this library, if any.
    */
    VkDeviceSize usage;
    /** \brief Estimated amount of memory available to the program, in bytes.

    Fetched from system using VK_EXT_memory_budget extension if enabled.

    It might be different (most probably smaller) than `VkMemoryHeap::size[heapIndex]` due to factors
    external to the program, decided by the operating system.
    Difference `budget - usage` is the amount of additional memory that can probably
    be allocated without problems. Exceeding the budget may result in various problems.
    */
    VkDeviceSize budget;
} VmaBudget;

/** @} */

/**
\addtogroup group_alloc
@{
*/

/** \brief Parameters of new #VmaAllocation.

To be used with functions like vmaCreateBuffer(), vmaCreateImage(), and many others.
*/
typedef struct VmaAllocationCreateInfo
{
    /// Use #VmaAllocationCreateFlagBits enum.
    VmaAllocationCreateFlags flags;
    /** \brief Intended usage of memory.

    You can leave #VMA_MEMORY_USAGE_UNKNOWN if you specify memory requirements in other way. \n
    If `pool` is not null, this member is ignored.
    */
    VmaMemoryUsage usage;
    /** \brief Flags that must be set in a Memory Type chosen for an allocation.

    Leave 0 if you specify memory requirements in other way. \n
    If `pool` is not null, this member is ignored.*/
    VkMemoryPropertyFlags requiredFlags;
    /** \brief Flags that preferably should be set in a memory type chosen for an allocation.

    Set to 0 if no additional flags are preferred. \n
    If `pool` is not null, this member is ignored. */
    VkMemoryPropertyFlags preferredFlags;
    /** \brief Bitmask containing one bit set for every memory type acceptable for this allocation.

    Value 0 is equivalent to `UINT32_MAX` - it means any memory type is accepted if
    it meets other requirements specified by this structure, with no further
    restrictions on memory type index. \n
    If `pool` is not null, this member is ignored.
    */
    uint32_t memoryTypeBits;
    /** \brief Pool that this allocation should be created in.

    Leave `VK_NULL_HANDLE` to allocate from default pool. If not null, members:
    `usage`, `requiredFlags`, `preferredFlags`, `memoryTypeBits` are ignored.
    */
    VmaPool VMA_NULLABLE pool;
    /** \brief Custom general-purpose pointer that will be stored in #VmaAllocation, can be read as VmaAllocationInfo::pUserData and changed using vmaSetAllocationUserData().

    If #VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT is used, it must be either
    null or pointer to a null-terminated string. The string will be then copied to
    internal buffer, so it doesn't need to be valid after allocation call.
    */
    void* VMA_NULLABLE pUserData;
    /** \brief A floating-point value between 0 and 1, indicating the priority of the allocation relative to other memory allocations.

    It is used only when #VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT flag was used during creation of the #VmaAllocator object
    and this allocation ends up as dedicated or is explicitly forced as dedicated using #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.
    Otherwise, it has the priority of a memory block where it is placed and this variable is ignored.
    */
    float priority;
} VmaAllocationCreateInfo;

/// Describes parameter of created #VmaPool.
typedef struct VmaPoolCreateInfo
{
    /** \brief Vulkan memory type index to allocate this pool from.
    */
    uint32_t memoryTypeIndex;
    /** \brief Use combination of #VmaPoolCreateFlagBits.
    */
    VmaPoolCreateFlags flags;
    /** \brief Size of a single `VkDeviceMemory` block to be allocated as part of this pool, in bytes. Optional.

    Specify nonzero to set explicit, constant size of memory blocks used by this
    pool.

    Leave 0 to use default and let the library manage block sizes automatically.
    Sizes of particular blocks may vary.
    In this case, the pool will also support dedicated allocations.
    */
    VkDeviceSize blockSize;
    /** \brief Minimum number of blocks to be always allocated in this pool, even if they stay empty.

    Set to 0 to have no preallocated blocks and allow the pool be completely empty.
    */
    size_t minBlockCount;
    /** \brief Maximum number of blocks that can be allocated in this pool. Optional.

    Set to 0 to use default, which is `SIZE_MAX`, which means no limit.

    Set to same value as VmaPoolCreateInfo::minBlockCount to have fixed amount of memory allocated
    throughout whole lifetime of this pool.
    */
    size_t maxBlockCount;
    /** \brief A floating-point value between 0 and 1, indicating the priority of the allocations in this pool relative to other memory allocations.

    It is used only when #VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT flag was used during creation of the #VmaAllocator object.
    Otherwise, this variable is ignored.
    */
    float priority;
    /** \brief Additional minimum alignment to be used for all allocations created from this pool. Can be 0.

    Leave 0 (default) not to impose any additional alignment. If not 0, it must be a power of two.
    It can be useful in cases where alignment returned by Vulkan by functions like `vkGetBufferMemoryRequirements` is not enough,
    e.g. when doing interop with OpenGL.
    */
    VkDeviceSize minAllocationAlignment;
    /** \brief Additional `pNext` chain to be attached to `VkMemoryAllocateInfo` used for every allocation made by this pool. Optional.

    Optional, can be null. If not null, it must point to a `pNext` chain of structures that can be attached to `VkMemoryAllocateInfo`.
    It can be useful for special needs such as adding `VkExportMemoryAllocateInfoKHR`.
    Structures pointed by this member must remain alive and unchanged for the whole lifetime of the custom pool.

    Please note that some structures, e.g. `VkMemoryPriorityAllocateInfoEXT`, `VkMemoryDedicatedAllocateInfoKHR`,
    can be attached automatically by this library when using other, more convenient of its features.
    */
    void* VMA_NULLABLE VMA_EXTENDS_VK_STRUCT(VkMemoryAllocateInfo) pMemoryAllocateNext;
} VmaPoolCreateInfo;

/** @} */

/**
\addtogroup group_alloc
@{
*/

/**
Parameters of #VmaAllocation objects, that can be retrieved using function vmaGetAllocationInfo().

There is also an extended version of this structure that carries additional parameters: #VmaAllocationInfo2.
*/
typedef struct VmaAllocationInfo
{
    /** \brief Memory type index that this allocation was allocated from.

    It never changes.
    */
    uint32_t memoryType;
    /** \brief Handle to Vulkan memory object.

    Same memory object can be shared by multiple allocations.

    It can change after the allocation is moved during \ref defragmentation.
    */
    VkDeviceMemory VMA_NULLABLE_NON_DISPATCHABLE deviceMemory;
    /** \brief Offset in `VkDeviceMemory` object to the beginning of this allocation, in bytes. `(deviceMemory, offset)` pair is unique to this allocation.

    You usually don't need to use this offset. If you create a buffer or an image together with the allocation using e.g. function
    vmaCreateBuffer(), vmaCreateImage(), functions that operate on these resources refer to the beginning of the buffer or image,
    not entire device memory block. Functions like vmaMapMemory(), vmaBindBufferMemory() also refer to the beginning of the allocation
    and apply this offset automatically.

    It can change after the allocation is moved during \ref defragmentation.
    */
    VkDeviceSize offset;
    /** \brief Size of this allocation, in bytes.

    It never changes.

    \note Allocation size returned in this variable may be greater than the size
    requested for the resource e.g. as `VkBufferCreateInfo::size`. Whole size of the
    allocation is accessible for operations on memory e.g. using a pointer after
    mapping with vmaMapMemory(), but operations on the resource e.g. using
    `vkCmdCopyBuffer` must be limited to the size of the resource.
    */
    VkDeviceSize size;
    /** \brief Pointer to the beginning of this allocation as mapped data.

    If the allocation hasn't been mapped using vmaMapMemory() and hasn't been
    created with #VMA_ALLOCATION_CREATE_MAPPED_BIT flag, this value is null.

    It can change after call to vmaMapMemory(), vmaUnmapMemory().
    It can also change after the allocation is moved during \ref defragmentation.
    */
    void* VMA_NULLABLE pMappedData;
    /** \brief Custom general-purpose pointer that was passed as VmaAllocationCreateInfo::pUserData or set using vmaSetAllocationUserData().

    It can change after call to vmaSetAllocationUserData() for this allocation.
    */
    void* VMA_NULLABLE pUserData;
    /** \brief Custom allocation name that was set with vmaSetAllocationName().

    It can change after call to vmaSetAllocationName() for this allocation.

    Another way to set custom name is to pass it in VmaAllocationCreateInfo::pUserData with
    additional flag #VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT set [DEPRECATED].
    */
    const char* VMA_NULLABLE pName;
} VmaAllocationInfo;

/// Extended parameters of a #VmaAllocation object that can be retrieved using function vmaget_allocation_info2().
typedef struct VmaAllocationInfo2
{
    /** \brief Basic parameters of the allocation.
    
    If you need only these, you can use function vmaGetAllocationInfo() and structure #VmaAllocationInfo instead.
    */
    VmaAllocationInfo allocationInfo;
    /** \brief Size of the `VkDeviceMemory` block that the allocation belongs to.
    
    In case of an allocation with dedicated memory, it will be equal to `allocationInfo.size`.
    */
    VkDeviceSize blockSize;
    /** \brief `VK_TRUE` if the allocation has dedicated memory, `VK_FALSE` if it was placed as part of a larger memory block.
    
    When `VK_TRUE`, it also means `VkMemoryDedicatedAllocateInfo` was used when creating the allocation
    (if VK_KHR_dedicated_allocation extension or Vulkan version >= 1.1 is enabled).
    */
    VkBool32 dedicatedMemory;
} VmaAllocationInfo2;

/** Callback function called during vmaBeginDefragmentation() to check custom criterion about ending current defragmentation pass.

Should return true if the defragmentation needs to stop current pass.
*/
typedef VkBool32 (VKAPI_PTR* PFN_vmaCheckDefragmentationBreakFunction)(void* VMA_NULLABLE pUserData);

/** \brief Parameters for defragmentation.

To be used with function vmaBeginDefragmentation().
*/
typedef struct VmaDefragmentationInfo
{
    /// \brief Use combination of #VmaDefragmentationFlagBits.
    VmaDefragmentationFlags flags;
    /** \brief Custom pool to be defragmented.

    If null then default pools will undergo defragmentation process.
    */
    VmaPool VMA_NULLABLE pool;
    /** \brief Maximum numbers of bytes that can be copied during single pass, while moving allocations to different places.

    `0` means no limit.
    */
    VkDeviceSize maxBytesPerPass;
    /** \brief Maximum number of allocations that can be moved during single pass to a different place.

    `0` means no limit.
    */
    uint32_t maxAllocationsPerPass;
    /** \brief Optional custom callback for stopping vmaBeginDefragmentation().

    Have to return true for breaking current defragmentation pass.
    */
    PFN_vmaCheckDefragmentationBreakFunction VMA_NULLABLE pfnBreakCallback;
    /// \brief Optional data to pass to custom callback for stopping pass of defragmentation.
    void* VMA_NULLABLE pBreakCallbackUserData;
} VmaDefragmentationInfo;

/// Single move of an allocation to be done for defragmentation.
typedef struct VmaDefragmentationMove
{
    /// Operation to be performed on the allocation by vmaEndDefragmentationPass(). Default value is #VMA_DEFRAGMENTATION_MOVE_OPERATION_COPY. You can modify it.
    VmaDefragmentationMoveOperation operation;
    /// Allocation that should be moved.
    VmaAllocation VMA_NOT_NULL srcAllocation;
    /** \brief Temporary allocation pointing to destination memory that will replace `srcAllocation`.

    \warning Do not store this allocation in your data structures! It exists only temporarily, for the duration of the defragmentation pass,
    to be used for binding new buffer/image to the destination memory using e.g. vmaBindBufferMemory().
    vmaEndDefragmentationPass() will destroy it and make `srcAllocation` point to this memory.
    */
    VmaAllocation VMA_NOT_NULL dstTmpAllocation;
} VmaDefragmentationMove;

/** \brief Parameters for incremental defragmentation steps.

To be used with function vmaBeginDefragmentationPass().
*/
typedef struct VmaDefragmentationPassMoveInfo
{
    /// Number of elements in the `pMoves` array.
    uint32_t moveCount;
    /** \brief Array of moves to be performed by the user in the current defragmentation pass.

    Pointer to an array of `moveCount` elements, owned by VMA, created in vmaBeginDefragmentationPass(), destroyed in vmaEndDefragmentationPass().

    For each element, you should:

    1. Create a new buffer/image in the place pointed by VmaDefragmentationMove::dstMemory + VmaDefragmentationMove::dstOffset.
    2. Copy data from the VmaDefragmentationMove::srcAllocation e.g. using `vkCmdCopyBuffer`, `vkCmdCopyImage`.
    3. Make sure these commands finished executing on the GPU.
    4. destroy the old buffer/image.

    Only then you can finish defragmentation pass by calling vmaEndDefragmentationPass().
    After this call, the allocation will point to the new place in memory.

    Alternatively, if you cannot move specific allocation, you can set VmaDefragmentationMove::operation to #VMA_DEFRAGMENTATION_MOVE_OPERATION_IGNORE.

    Alternatively, if you decide you want to completely remove the allocation:

    1. destroy its buffer/image.
    2. Set VmaDefragmentationMove::operation to #VMA_DEFRAGMENTATION_MOVE_OPERATION_DESTROY.

    Then, after vmaEndDefragmentationPass() the allocation will be freed.
    */
    VmaDefragmentationMove* VMA_NULLABLE VMA_LEN_IF_NOT_NULL(moveCount) pMoves;
} VmaDefragmentationPassMoveInfo;

/// Statistics returned for defragmentation process in function vmaEndDefragmentation().
typedef struct VmaDefragmentationStats
{
    /// Total number of bytes that have been copied while moving allocations to different places.
    VkDeviceSize bytesMoved;
    /// Total number of bytes that have been released to the system by freeing empty `VkDeviceMemory` objects.
    VkDeviceSize bytesFreed;
    /// Number of allocations that have been moved to different places.
    uint32_t allocationsMoved;
    /// Number of empty `VkDeviceMemory` objects that have been released to the system.
    uint32_t deviceMemoryBlocksFreed;
} VmaDefragmentationStats;

/** @} */

/**
\addtogroup group_virtual
@{
*/

/// Parameters of created #VmaVirtualBlock object to be passed to vmaCreateVirtualBlock().
typedef struct VmaVirtualBlockCreateInfo
{
    /** \brief Total size of the virtual block.

    Sizes can be expressed in bytes or any units you want as long as you are consistent in using them.
    For example, if you allocate from some array of structures, 1 can mean single instance of entire structure.
    */
    VkDeviceSize size;

    /** \brief Use combination of #VmaVirtualBlockCreateFlagBits.
    */
    VmaVirtualBlockCreateFlags flags;

    /** \brief Custom CPU memory allocation callbacks. Optional.

    Optional, can be null. When specified, they will be used for all CPU-side memory allocations.
    */
    const VkAllocationCallbacks* VMA_NULLABLE pAllocationCallbacks;
} VmaVirtualBlockCreateInfo;

/// Parameters of created virtual allocation to be passed to vmaVirtualAllocate().
typedef struct VmaVirtualAllocationCreateInfo
{
    /** \brief Size of the allocation.

    Cannot be zero.
    */
    VkDeviceSize size;
    /** \brief Required alignment of the allocation. Optional.

    Must be power of two. Special value 0 has the same meaning as 1 - means no special alignment is required, so allocation can start at any offset.
    */
    VkDeviceSize alignment;
    /** \brief Use combination of #VmaVirtualAllocationCreateFlagBits.
    */
    VmaVirtualAllocationCreateFlags flags;
    /** \brief Custom pointer to be associated with the allocation. Optional.

    It can be any value and can be used for user-defined purposes. It can be fetched or changed later.
    */
    void* VMA_NULLABLE pUserData;
} VmaVirtualAllocationCreateInfo;

/// Parameters of an existing virtual allocation, returned by vmaGetVirtualAllocationInfo().
typedef struct VmaVirtualAllocationInfo
{
    /** \brief Offset of the allocation.

    Offset at which the allocation was made.
    */
    VkDeviceSize offset;
    /** \brief Size of the allocation.

    Same value as passed in VmaVirtualAllocationCreateInfo::size.
    */
    VkDeviceSize size;
    /** \brief Custom pointer associated with the allocation.

    Same value as passed in VmaVirtualAllocationCreateInfo::pUserData or to vmaSetVirtualAllocationUserData().
    */
    void* VMA_NULLABLE pUserData;
} VmaVirtualAllocationInfo;

/** @} */

#endif // _VMA_DATA_TYPES_DECLARATIONS

#ifndef _VMA_FUNCTION_HEADERS

/**
\addtogroup group_init
@{
*/

#ifdef VOLK_HEADER_VERSION
/** \brief Fully initializes `pDstVulkanFunctions` structure with Vulkan functions needed by VMA
using [volk library](https://github.com/zeux/volk).

This function is defined in VMA header only if "volk.h" was included before it.

To use this function properly:

-# Initialize volk and Vulkan:
   -# Call `volkInitialize()`
   -# Create `VkInstance` object
   -# Call `volkLoadInstance()`
   -# Create `VkDevice` object
   -# Call `volkLoadDevice()`
-# Fill in structure #VmaAllocatorCreateInfo, especially members:
   - VmaAllocatorCreateInfo::device
   - VmaAllocatorCreateInfo::vulkanApiVersion
   - VmaAllocatorCreateInfo::flags - set appropriate flags for the Vulkan extensions you enabled
-# Create an instance of the #VmaVulkanFunctions structure.
-# Call vmaImportVulkanFunctionsFromVolk().
   Parameter `pAllocatorCreateInfo` is read to find out which functions should be fetched for
   appropriate Vulkan version and extensions.
   Parameter `pDstVulkanFunctions` is filled with those function pointers, or null if not applicable.
-# Attach the #VmaVulkanFunctions structure to VmaAllocatorCreateInfo::pVulkanFunctions.
-# Call vmaCreateAllocator() to create the #VmaAllocator object.

Example:

\code
VmaAllocatorCreateInfo allocatorCreateInfo = {};
allocatorCreateInfo.physicalDevice = myPhysicalDevice;
allocatorCreateInfo.device = myDevice;
allocatorCreateInfo.instance = myInstance;
allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;
allocatorCreateInfo.flags = VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT |
    VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT |
    VMA_ALLOCATOR_CREATE_KHR_EXTERNAL_MEMORY_WIN32_BIT;

VmaVulkanFunctions vulkanFunctions;
VkResult res = vmaImportVulkanFunctionsFromVolk(&allocatorCreateInfo, &vulkanFunctions);
// Check res...
allocatorCreateInfo.pVulkanFunctions = &vulkanFunctions;

VmaAllocator allocator;
res = vmaCreateAllocator(&allocatorCreateInfo, &allocator);
// Check res...
\endcode

Internally in this function, pointers to functions related to the entire Vulkan instance are fetched using global function definitions,
while pointers to functions related to the Vulkan device are fetched using `volkLoadDeviceTable()` for given `pAllocatorCreateInfo->device`.
 */
VMA_CALL_PRE VkResult VMA_CALL_POST vmaImportVulkanFunctionsFromVolk(
    const VmaAllocatorCreateInfo* VMA_NOT_NULL pAllocatorCreateInfo,
    VmaVulkanFunctions* VMA_NOT_NULL pDstVulkanFunctions);
#endif

/// Creates #VmaAllocator object.
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAllocator(
    const VmaAllocatorCreateInfo* VMA_NOT_NULL pCreateInfo,
    VmaAllocator VMA_NULLABLE* VMA_NOT_NULL pAllocator);

/// Destroys allocator object.
VMA_CALL_PRE void VMA_CALL_POST vmaDestroyAllocator(
    VmaAllocator VMA_NULLABLE allocator);

/** \brief Returns information about existing #VmaAllocator object - handle to Vulkan device etc.

It might be useful if you want to keep just the #VmaAllocator handle and fetch other required handles to
`VkPhysicalDevice`, `VkDevice` etc. every time using this function.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetAllocatorInfo(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocatorInfo* VMA_NOT_NULL pAllocatorInfo);

/**
PhysicalDeviceProperties are fetched from physicalDevice by the allocator.
You can access it here, without fetching it again on your own.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetPhysicalDeviceProperties(
    VmaAllocator VMA_NOT_NULL allocator,
    const VkPhysicalDeviceProperties* VMA_NULLABLE* VMA_NOT_NULL ppPhysicalDeviceProperties);

/**
PhysicalDeviceMemoryProperties are fetched from physicalDevice by the allocator.
You can access it here, without fetching it again on your own.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetMemoryProperties(
    VmaAllocator VMA_NOT_NULL allocator,
    const VkPhysicalDeviceMemoryProperties* VMA_NULLABLE* VMA_NOT_NULL ppPhysicalDeviceMemoryProperties);

/**
\brief Given Memory Type Index, returns Property Flags of this memory type.

This is just a convenience function. Same information can be obtained using
vmaGetMemoryProperties().
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetMemoryTypeProperties(
    VmaAllocator VMA_NOT_NULL allocator,
    uint32_t memoryTypeIndex,
    VkMemoryPropertyFlags* VMA_NOT_NULL pFlags);

/** \brief Sets index of the current frame.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaSetCurrentFrameIndex(
    VmaAllocator VMA_NOT_NULL allocator,
    uint32_t frameIndex);

/** @} */

/**
\addtogroup group_stats
@{
*/

/** \brief Retrieves statistics from current state of the Allocator.

This function is called "calculate" not "get" because it has to traverse all
internal data structures, so it may be quite slow. Use it for debugging purposes.
For faster but more brief statistics suitable to be called every frame or every allocation,
use vmaGetHeapBudgets().

Note that when using allocator from multiple threads, returned information may immediately
become outdated.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaCalculateStatistics(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaTotalStatistics* VMA_NOT_NULL pStats);

/** \brief Retrieves information about current memory usage and budget for all memory heaps.

\param allocator
\param[out] pBudgets Must point to array with number of elements at least equal to number of memory heaps in physical device used.

This function is called "get" not "calculate" because it is very fast, suitable to be called
every frame or every allocation. For more detailed statistics use vmaCalculateStatistics().

Note that when using allocator from multiple threads, returned information may immediately
become outdated.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetHeapBudgets(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaBudget* VMA_NOT_NULL VMA_LEN_IF_NOT_NULL("VkPhysicalDeviceMemoryProperties::memoryHeapCount") pBudgets);

/** @} */

/**
\addtogroup group_alloc
@{
*/

/**
\brief Helps to find `memoryTypeIndex`, given `memoryTypeBits` and #VmaAllocationCreateInfo.

This algorithm tries to find a memory type that:

- Is allowed by `memoryTypeBits`.
- Contains all the flags from `pAllocationCreateInfo->requiredFlags`.
- Matches intended usage.
- Has as many flags from `pAllocationCreateInfo->preferredFlags` as possible.

\return Returns `VK_ERROR_FEATURE_NOT_PRESENT` if not found. Receiving such result
from this function or any other allocating function probably means that your
device doesn't support any memory type with requested features for the specific
type of resource you want to use it for. Please check parameters of your
resource, like image layout (`OPTIMAL` versus `LINEAR`) or mip level count.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaFindMemoryTypeIndex(
    VmaAllocator VMA_NOT_NULL allocator,
    uint32_t memoryTypeBits,
    const VmaAllocationCreateInfo* VMA_NOT_NULL pAllocationCreateInfo,
    uint32_t* VMA_NOT_NULL pMemoryTypeIndex);

/**
\brief Helps to find `memoryTypeIndex`, given `VkBufferCreateInfo` and #VmaAllocationCreateInfo.

It can be useful e.g. to determine value to be used as VmaPoolCreateInfo::memoryTypeIndex.
It may need to internally create a temporary, dummy buffer that never has memory bound.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaFindMemoryTypeIndexForBufferInfo(
    VmaAllocator VMA_NOT_NULL allocator,
    const VkBufferCreateInfo* VMA_NOT_NULL pBufferCreateInfo,
    const VmaAllocationCreateInfo* VMA_NOT_NULL pAllocationCreateInfo,
    uint32_t* VMA_NOT_NULL pMemoryTypeIndex);

/**
\brief Helps to find `memoryTypeIndex`, given `VkImageCreateInfo` and #VmaAllocationCreateInfo.

It can be useful e.g. to determine value to be used as VmaPoolCreateInfo::memoryTypeIndex.
It may need to internally create a temporary, dummy image that never has memory bound.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaFindMemoryTypeIndexForImageInfo(
    VmaAllocator VMA_NOT_NULL allocator,
    const VkImageCreateInfo* VMA_NOT_NULL pImageCreateInfo,
    const VmaAllocationCreateInfo* VMA_NOT_NULL pAllocationCreateInfo,
    uint32_t* VMA_NOT_NULL pMemoryTypeIndex);

/** \brief Allocates Vulkan device memory and creates #VmaPool object.

\param allocator Allocator object.
\param pCreateInfo Parameters of pool to create.
\param[out] pPool Handle to created pool.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreatePool(
    VmaAllocator VMA_NOT_NULL allocator,
    const VmaPoolCreateInfo* VMA_NOT_NULL pCreateInfo,
    VmaPool VMA_NULLABLE* VMA_NOT_NULL pPool);

/** \brief Destroys #VmaPool object and frees Vulkan device memory.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaDestroyPool(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaPool VMA_NULLABLE pool);

/** @} */

/**
\addtogroup group_stats
@{
*/

/** \brief Retrieves statistics of existing #VmaPool object.

\param allocator Allocator object.
\param pool Pool object.
\param[out] pPoolStats Statistics of specified pool.

Note that when using the pool from multiple threads, returned information may immediately
become outdated.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetPoolStatistics(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaPool VMA_NOT_NULL pool,
    VmaStatistics* VMA_NOT_NULL pPoolStats);

/** \brief Retrieves detailed statistics of existing #VmaPool object.

\param allocator Allocator object.
\param pool Pool object.
\param[out] pPoolStats Statistics of specified pool.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaCalculatePoolStatistics(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaPool VMA_NOT_NULL pool,
    VmaDetailedStatistics* VMA_NOT_NULL pPoolStats);

/** @} */

/**
\addtogroup group_alloc
@{
*/

/** \brief Checks magic number in margins around all allocations in given memory pool in search for corruptions.

Corruption detection is enabled only when `VMA_DEBUG_DETECT_CORRUPTION` macro is defined to nonzero,
`VMA_DEBUG_MARGIN` is defined to nonzero and the pool is created in memory type that is
`HOST_VISIBLE` and `HOST_COHERENT`. For more information, see [Corruption detection](@ref debugging_memory_usage_corruption_detection).

Possible return values:

- `VK_ERROR_FEATURE_NOT_PRESENT` - corruption detection is not enabled for specified pool.
- `VK_SUCCESS` - corruption detection has been performed and succeeded.
- `VK_ERROR_UNKNOWN` - corruption detection has been performed and found memory corruptions around one of the allocations.
  `VMA_ASSERT` is also fired in that case.
- Other value: Error returned by Vulkan, e.g. memory mapping failure.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCheckPoolCorruption(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaPool VMA_NOT_NULL pool);

/** \brief Retrieves name of a custom pool.

After the call `ppName` is either null or points to an internally-owned null-terminated string
containing name of the pool that was previously set. The pointer becomes invalid when the pool is
destroyed or its name is changed using vmaSetPoolName().
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetPoolName(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaPool VMA_NOT_NULL pool,
    const char* VMA_NULLABLE* VMA_NOT_NULL ppName);

/** \brief Sets name of a custom pool.

`pName` can be either null or pointer to a null-terminated string with new name for the pool.
Function makes internal copy of the string, so it can be changed or freed immediately after this call.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaSetPoolName(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaPool VMA_NOT_NULL pool,
    const char* VMA_NULLABLE pName);

/** \brief General purpose memory allocation.

\param allocator
\param pVkMemoryRequirements
\param pCreateInfo
\param[out] pAllocation Handle to allocated memory.
\param[out] pAllocationInfo Optional. Information about allocated memory. It can be later fetched using function vmaGetAllocationInfo().

You should free the memory using vmaFreeMemory() or vmaFreeMemoryPages().

It is recommended to use vmaAllocateMemoryForBuffer(), vmaAllocateMemoryForImage(),
vmaCreateBuffer(), vmaCreateImage() instead whenever possible.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemory(
    VmaAllocator VMA_NOT_NULL allocator,
    const VkMemoryRequirements* VMA_NOT_NULL pVkMemoryRequirements,
    const VmaAllocationCreateInfo* VMA_NOT_NULL pCreateInfo,
    VmaAllocation VMA_NULLABLE* VMA_NOT_NULL pAllocation,
    VmaAllocationInfo* VMA_NULLABLE pAllocationInfo);

/** \brief General purpose memory allocation for multiple allocation objects at once.

\param allocator Allocator object.
\param pVkMemoryRequirements Memory requirements for each allocation.
\param pCreateInfo Creation parameters for each allocation.
\param allocationCount Number of allocations to make.
\param[out] pAllocations Pointer to array that will be filled with handles to created allocations.
\param[out] pAllocationInfo Optional. Pointer to array that will be filled with parameters of created allocations.

You should free the memory using vmaFreeMemory() or vmaFreeMemoryPages().

Word "pages" is just a suggestion to use this function to allocate pieces of memory needed for sparse binding.
It is just a general purpose allocation function able to make multiple allocations at once.
It may be internally optimized to be more efficient than calling vmaAllocateMemory() `allocationCount` times.

All allocations are made using same parameters. All of them are created out of the same memory pool and type.
If any allocation fails, all allocations already made within this function call are also freed, so that when
returned result is not `VK_SUCCESS`, `pAllocation` array is always entirely filled with `VK_NULL_HANDLE`.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemoryPages(
    VmaAllocator VMA_NOT_NULL allocator,
    const VkMemoryRequirements* VMA_NOT_NULL VMA_LEN_IF_NOT_NULL(allocationCount) pVkMemoryRequirements,
    const VmaAllocationCreateInfo* VMA_NOT_NULL VMA_LEN_IF_NOT_NULL(allocationCount) pCreateInfo,
    size_t allocationCount,
    VmaAllocation VMA_NULLABLE* VMA_NOT_NULL VMA_LEN_IF_NOT_NULL(allocationCount) pAllocations,
    VmaAllocationInfo* VMA_NULLABLE VMA_LEN_IF_NOT_NULL(allocationCount) pAllocationInfo);

/** \brief Allocates memory suitable for given `VkBuffer`.

\param allocator
\param buffer
\param pCreateInfo
\param[out] pAllocation Handle to allocated memory.
\param[out] pAllocationInfo Optional. Information about allocated memory. It can be later fetched using function vmaGetAllocationInfo().

It only creates #VmaAllocation. To bind the memory to the buffer, use vmaBindBufferMemory().

This is a special-purpose function. In most cases you should use vmaCreateBuffer().

You must free the allocation using vmaFreeMemory() when no longer needed.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemoryForBuffer(
    VmaAllocator VMA_NOT_NULL allocator,
    VkBuffer VMA_NOT_NULL_NON_DISPATCHABLE buffer,
    const VmaAllocationCreateInfo* VMA_NOT_NULL pCreateInfo,
    VmaAllocation VMA_NULLABLE* VMA_NOT_NULL pAllocation,
    VmaAllocationInfo* VMA_NULLABLE pAllocationInfo);

/** \brief Allocates memory suitable for given `VkImage`.

\param allocator
\param image
\param pCreateInfo
\param[out] pAllocation Handle to allocated memory.
\param[out] pAllocationInfo Optional. Information about allocated memory. It can be later fetched using function vmaGetAllocationInfo().

It only creates #VmaAllocation. To bind the memory to the buffer, use vmaBindImageMemory().

This is a special-purpose function. In most cases you should use vmaCreateImage().

You must free the allocation using vmaFreeMemory() when no longer needed.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemoryForImage(
    VmaAllocator VMA_NOT_NULL allocator,
    VkImage VMA_NOT_NULL_NON_DISPATCHABLE image,
    const VmaAllocationCreateInfo* VMA_NOT_NULL pCreateInfo,
    VmaAllocation VMA_NULLABLE* VMA_NOT_NULL pAllocation,
    VmaAllocationInfo* VMA_NULLABLE pAllocationInfo);

/** \brief Frees memory previously allocated using vmaAllocateMemory(), vmaAllocateMemoryForBuffer(), or vmaAllocateMemoryForImage().

Passing `VK_NULL_HANDLE` as `allocation` is valid. Such function call is just skipped.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaFreeMemory(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NULLABLE allocation);

/** \brief Frees memory and destroys multiple allocations.

Word "pages" is just a suggestion to use this function to free pieces of memory used for sparse binding.
It is just a general purpose function to free memory and destroy allocations made using e.g. vmaAllocateMemory(),
vmaAllocateMemoryPages() and other functions.
It may be internally optimized to be more efficient than calling vmaFreeMemory() `allocationCount` times.

Allocations in `pAllocations` array can come from any memory pools and types.
Passing `VK_NULL_HANDLE` as elements of `pAllocations` array is valid. Such entries are just skipped.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaFreeMemoryPages(
    VmaAllocator VMA_NOT_NULL allocator,
    size_t allocationCount,
    const VmaAllocation VMA_NULLABLE* VMA_NOT_NULL VMA_LEN_IF_NOT_NULL(allocationCount) pAllocations);

/** \brief Returns current information about specified allocation.

Current parameters of given allocation are returned in `pAllocationInfo`.

Although this function doesn't lock any mutex, so it should be quite efficient,
you should avoid calling it too often.
You can retrieve same VmaAllocationInfo structure while creating your resource, from function
vmaCreateBuffer(), vmaCreateImage(). You can remember it if you are sure parameters don't change
(e.g. due to defragmentation).

There is also a new function vmaget_allocation_info2() that offers extended information
about the allocation, returned using new structure #VmaAllocationInfo2.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetAllocationInfo(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VmaAllocationInfo* VMA_NOT_NULL pAllocationInfo);

/** \brief Returns extended information about specified allocation.

Current parameters of given allocation are returned in `pAllocationInfo`.
Extended parameters in structure #VmaAllocationInfo2 include memory block size
and a flag telling whether the allocation has dedicated memory.
It can be useful e.g. for interop with OpenGL.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaget_allocation_info2(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VmaAllocationInfo2* VMA_NOT_NULL pAllocationInfo);

/** \brief Sets pUserData in given allocation to new value.

The value of pointer `pUserData` is copied to allocation's `pUserData`.
It is opaque, so you can use it however you want - e.g.
as a pointer, ordinal number or some handle to you own data.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaSetAllocationUserData(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    void* VMA_NULLABLE pUserData);

/** \brief Sets pName in given allocation to new value.

`pName` must be either null, or pointer to a null-terminated string. The function
makes local copy of the string and sets it as allocation's `pName`. String
passed as pName doesn't need to be valid for whole lifetime of the allocation -
you can free it after this call. String previously pointed by allocation's
`pName` is freed from memory.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaSetAllocationName(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    const char* VMA_NULLABLE pName);

/**
\brief Given an allocation, returns Property Flags of its memory type.

This is just a convenience function. Same information can be obtained using
vmaGetAllocationInfo() + vmaGetMemoryProperties().
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetAllocationMemoryProperties(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VkMemoryPropertyFlags* VMA_NOT_NULL pFlags);


#if VMA_EXTERNAL_MEMORY_WIN32
/**
\brief Given an allocation, returns Win32 handle that may be imported by other processes or APIs.

\param hTargetProcess Must be a valid handle to target process or null. If it's null, the function returns
    handle for the current process.
\param[out] pHandle Output parameter that returns the handle.

The function fills `pHandle` with handle that can be used in target process.
The handle is fetched using function `vkGetMemoryWin32HandleKHR`.
When no longer needed, you must close it using:

\code
CloseHandle(handle);
\endcode

You can close it any time, before or after destroying the allocation object.
It is reference-counted internally by Windows.

Note the handle is returned for the entire `VkDeviceMemory` block that the allocation belongs to.
If the allocation is sub-allocated from a larger block, you may need to consider the offset of the allocation
(VmaAllocationInfo::offset).

If the function fails with `VK_ERROR_FEATURE_NOT_PRESENT` error code, please double-check
that VmaVulkanFunctions::vkGetMemoryWin32HandleKHR function pointer is set, e.g. either by using `VMA_DYNAMIC_VULKAN_FUNCTIONS`
or by manually passing it through VmaAllocatorCreateInfo::pVulkanFunctions.

For more information, see chapter \ref vk_khr_external_memory_win32.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaGetMemoryWin32Handle(VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation, HANDLE VMA_NULLABLE hTargetProcess, HANDLE _Nullable * _Nonnull pHandle);
#endif // VMA_EXTERNAL_MEMORY_WIN32

/** \brief Maps memory represented by given allocation and returns pointer to it.

Maps memory represented by given allocation to make it accessible to CPU code.
When succeeded, `*ppData` contains pointer to first byte of this memory.

\warning
If the allocation is part of a bigger `VkDeviceMemory` block, returned pointer is
correctly offsetted to the beginning of region assigned to this particular allocation.
Unlike the result of `vkMapMemory`, it points to the allocation, not to the beginning of the whole block.
You should not add VmaAllocationInfo::offset to it!

Mapping is internally reference-counted and synchronized, so despite raw Vulkan
function `vkMapMemory()` cannot be used to map same block of `VkDeviceMemory`
multiple times simultaneously, it is safe to call this function on allocations
assigned to the same memory block. Actual Vulkan memory will be mapped on first
mapping and unmapped on last unmapping.

If the function succeeded, you must call vmaUnmapMemory() to unmap the
allocation when mapping is no longer needed or before freeing the allocation, at
the latest.

It also safe to call this function multiple times on the same allocation. You
must call vmaUnmapMemory() same number of times as you called vmaMapMemory().

It is also safe to call this function on allocation created with
#VMA_ALLOCATION_CREATE_MAPPED_BIT flag. Its memory stays mapped all the time.
You must still call vmaUnmapMemory() same number of times as you called
vmaMapMemory(). You must not call vmaUnmapMemory() additional time to free the
"0-th" mapping made automatically due to #VMA_ALLOCATION_CREATE_MAPPED_BIT flag.

This function fails when used on allocation made in memory type that is not
`HOST_VISIBLE`.

This function doesn't automatically flush or invalidate caches.
If the allocation is made from a memory types that is not `HOST_COHERENT`,
you also need to use vmaInvalidateAllocation() / vmaFlushAllocation(), as required by Vulkan specification.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaMapMemory(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    void* VMA_NULLABLE* VMA_NOT_NULL ppData);

/** \brief Unmaps memory represented by given allocation, mapped previously using vmaMapMemory().

For details, see description of vmaMapMemory().

This function doesn't automatically flush or invalidate caches.
If the allocation is made from a memory types that is not `HOST_COHERENT`,
you also need to use vmaInvalidateAllocation() / vmaFlushAllocation(), as required by Vulkan specification.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaUnmapMemory(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation);

/** \brief Flushes memory of given allocation.

Calls `vkFlushMappedMemoryRanges()` for memory associated with given range of given allocation.
It needs to be called after writing to a mapped memory for memory types that are not `HOST_COHERENT`.
unmap operation doesn't do that automatically.

- `offset` must be relative to the beginning of allocation.
- `size` can be `VK_WHOLE_SIZE`. It means all memory from `offset` the the end of given allocation.
- `offset` and `size` don't have to be aligned.
  They are internally rounded down/up to multiply of `nonCoherentAtomSize`.
- If `size` is 0, this call is ignored.
- If memory type that the `allocation` belongs to is not `HOST_VISIBLE` or it is `HOST_COHERENT`,
  this call is ignored.

Warning! `offset` and `size` are relative to the contents of given `allocation`.
If you mean whole allocation, you can pass 0 and `VK_WHOLE_SIZE`, respectively.
Do not pass allocation's offset as `offset`!!!

This function returns the `VkResult` from `vkFlushMappedMemoryRanges` if it is
called, otherwise `VK_SUCCESS`.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaFlushAllocation(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VkDeviceSize offset,
    VkDeviceSize size);

/** \brief Invalidates memory of given allocation.

Calls `vkInvalidateMappedMemoryRanges()` for memory associated with given range of given allocation.
It needs to be called before reading from a mapped memory for memory types that are not `HOST_COHERENT`.
Map operation doesn't do that automatically.

- `offset` must be relative to the beginning of allocation.
- `size` can be `VK_WHOLE_SIZE`. It means all memory from `offset` the the end of given allocation.
- `offset` and `size` don't have to be aligned.
  They are internally rounded down/up to multiply of `nonCoherentAtomSize`.
- If `size` is 0, this call is ignored.
- If memory type that the `allocation` belongs to is not `HOST_VISIBLE` or it is `HOST_COHERENT`,
  this call is ignored.

Warning! `offset` and `size` are relative to the contents of given `allocation`.
If you mean whole allocation, you can pass 0 and `VK_WHOLE_SIZE`, respectively.
Do not pass allocation's offset as `offset`!!!

This function returns the `VkResult` from `vkInvalidateMappedMemoryRanges` if
it is called, otherwise `VK_SUCCESS`.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaInvalidateAllocation(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VkDeviceSize offset,
    VkDeviceSize size);

/** \brief Flushes memory of given set of allocations.

Calls `vkFlushMappedMemoryRanges()` for memory associated with given ranges of given allocations.
For more information, see documentation of vmaFlushAllocation().

\param allocator
\param allocationCount
\param allocations
\param offsets If not null, it must point to an array of offsets of regions to flush, relative to the beginning of respective allocations. Null means all offsets are zero.
\param sizes If not null, it must point to an array of sizes of regions to flush in respective allocations. Null means `VK_WHOLE_SIZE` for all allocations.

This function returns the `VkResult` from `vkFlushMappedMemoryRanges` if it is
called, otherwise `VK_SUCCESS`.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaFlushAllocations(
    VmaAllocator VMA_NOT_NULL allocator,
    uint32_t allocationCount,
    const VmaAllocation VMA_NOT_NULL* VMA_NULLABLE VMA_LEN_IF_NOT_NULL(allocationCount) allocations,
    const VkDeviceSize* VMA_NULLABLE VMA_LEN_IF_NOT_NULL(allocationCount) offsets,
    const VkDeviceSize* VMA_NULLABLE VMA_LEN_IF_NOT_NULL(allocationCount) sizes);

/** \brief Invalidates memory of given set of allocations.

Calls `vkInvalidateMappedMemoryRanges()` for memory associated with given ranges of given allocations.
For more information, see documentation of vmaInvalidateAllocation().

\param allocator
\param allocationCount
\param allocations
\param offsets If not null, it must point to an array of offsets of regions to flush, relative to the beginning of respective allocations. Null means all offsets are zero.
\param sizes If not null, it must point to an array of sizes of regions to flush in respective allocations. Null means `VK_WHOLE_SIZE` for all allocations.

This function returns the `VkResult` from `vkInvalidateMappedMemoryRanges` if it is
called, otherwise `VK_SUCCESS`.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaInvalidateAllocations(
    VmaAllocator VMA_NOT_NULL allocator,
    uint32_t allocationCount,
    const VmaAllocation VMA_NOT_NULL* VMA_NULLABLE VMA_LEN_IF_NOT_NULL(allocationCount) allocations,
    const VkDeviceSize* VMA_NULLABLE VMA_LEN_IF_NOT_NULL(allocationCount) offsets,
    const VkDeviceSize* VMA_NULLABLE VMA_LEN_IF_NOT_NULL(allocationCount) sizes);

/** \brief Maps the allocation temporarily if needed, copies data from specified host pointer to it, and flushes the memory from the host caches if needed.

\param allocator
\param pSrcHostPointer Pointer to the host data that become source of the copy.
\param dstAllocation   Handle to the allocation that becomes destination of the copy.
\param dstAllocationLocalOffset  Offset within `dstAllocation` where to write copied data, in bytes.
\param size            Number of bytes to copy.

This is a convenience function that allows to copy data from a host pointer to an allocation easily.
Same behavior can be achieved by calling vmaMapMemory(), `memcpy()`, vmaUnmapMemory(), vmaFlushAllocation().

This function can be called only for allocations created in a memory type that has `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` flag.
It can be ensured e.g. by using #VMA_MEMORY_USAGE_AUTO and #VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or
#VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT.
Otherwise, the function will fail and generate a Validation Layers error.

`dstAllocationLocalOffset` is relative to the contents of given `dstAllocation`.
If you mean whole allocation, you should pass 0.
Do not pass allocation's offset within device memory block this parameter!
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCopyMemoryToAllocation(
    VmaAllocator VMA_NOT_NULL allocator,
    const void* VMA_NOT_NULL VMA_LEN_IF_NOT_NULL(size) pSrcHostPointer,
    VmaAllocation VMA_NOT_NULL dstAllocation,
    VkDeviceSize dstAllocationLocalOffset,
    VkDeviceSize size);

/** \brief Invalidates memory in the host caches if needed, maps the allocation temporarily if needed, and copies data from it to a specified host pointer.

\param allocator
\param srcAllocation   Handle to the allocation that becomes source of the copy.
\param srcAllocationLocalOffset  Offset within `srcAllocation` where to read copied data, in bytes.
\param pDstHostPointer Pointer to the host memory that become destination of the copy.
\param size            Number of bytes to copy.

This is a convenience function that allows to copy data from an allocation to a host pointer easily.
Same behavior can be achieved by calling vmaInvalidateAllocation(), vmaMapMemory(), `memcpy()`, vmaUnmapMemory().

This function should be called only for allocations created in a memory type that has `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`
and `VK_MEMORY_PROPERTY_HOST_CACHED_BIT` flag.
It can be ensured e.g. by using #VMA_MEMORY_USAGE_AUTO and #VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT.
Otherwise, the function may fail and generate a Validation Layers error.
It may also work very slowly when reading from an uncached memory.

`srcAllocationLocalOffset` is relative to the contents of given `srcAllocation`.
If you mean whole allocation, you should pass 0.
Do not pass allocation's offset within device memory block as this parameter!
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCopyAllocationToMemory(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL srcAllocation,
    VkDeviceSize srcAllocationLocalOffset,
    void* VMA_NOT_NULL VMA_LEN_IF_NOT_NULL(size) pDstHostPointer,
    VkDeviceSize size);

/** \brief Checks magic number in margins around all allocations in given memory types (in both default and custom pools) in search for corruptions.

\param allocator
\param memoryTypeBits Bit mask, where each bit set means that a memory type with that index should be checked.

Corruption detection is enabled only when `VMA_DEBUG_DETECT_CORRUPTION` macro is defined to nonzero,
`VMA_DEBUG_MARGIN` is defined to nonzero and only for memory types that are
`HOST_VISIBLE` and `HOST_COHERENT`. For more information, see [Corruption detection](@ref debugging_memory_usage_corruption_detection).

Possible return values:

- `VK_ERROR_FEATURE_NOT_PRESENT` - corruption detection is not enabled for any of specified memory types.
- `VK_SUCCESS` - corruption detection has been performed and succeeded.
- `VK_ERROR_UNKNOWN` - corruption detection has been performed and found memory corruptions around one of the allocations.
  `VMA_ASSERT` is also fired in that case.
- Other value: Error returned by Vulkan, e.g. memory mapping failure.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCheckCorruption(
    VmaAllocator VMA_NOT_NULL allocator,
    uint32_t memoryTypeBits);

/** \brief Begins defragmentation process.

\param allocator Allocator object.
\param pInfo Structure filled with parameters of defragmentation.
\param[out] pContext Context object that must be passed to vmaEndDefragmentation() to finish defragmentation.
\returns
- `VK_SUCCESS` if defragmentation can begin.
- `VK_ERROR_FEATURE_NOT_PRESENT` if defragmentation is not supported.

For more information about defragmentation, see documentation chapter:
[Defragmentation](@ref defragmentation).
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaBeginDefragmentation(
    VmaAllocator VMA_NOT_NULL allocator,
    const VmaDefragmentationInfo* VMA_NOT_NULL pInfo,
    VmaDefragmentationContext VMA_NULLABLE* VMA_NOT_NULL pContext);

/** \brief Ends defragmentation process.

\param allocator Allocator object.
\param context Context object that has been created by vmaBeginDefragmentation().
\param[out] pStats Optional stats for the defragmentation. Can be null.

Use this function to finish defragmentation started by vmaBeginDefragmentation().
*/
VMA_CALL_PRE void VMA_CALL_POST vmaEndDefragmentation(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaDefragmentationContext VMA_NOT_NULL context,
    VmaDefragmentationStats* VMA_NULLABLE pStats);

/** \brief Starts single defragmentation pass.

\param allocator Allocator object.
\param context Context object that has been created by vmaBeginDefragmentation().
\param[out] pPassInfo Computed information for current pass.
\returns
- `VK_SUCCESS` if no more moves are possible. Then you can omit call to vmaEndDefragmentationPass() and simply end whole defragmentation.
- `VK_INCOMPLETE` if there are pending moves returned in `pPassInfo`. You need to perform them, call vmaEndDefragmentationPass(),
  and then preferably try another pass with vmaBeginDefragmentationPass().
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaBeginDefragmentationPass(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaDefragmentationContext VMA_NOT_NULL context,
    VmaDefragmentationPassMoveInfo* VMA_NOT_NULL pPassInfo);

/** \brief Ends single defragmentation pass.

\param allocator Allocator object.
\param context Context object that has been created by vmaBeginDefragmentation().
\param pPassInfo Computed information for current pass filled by vmaBeginDefragmentationPass() and possibly modified by you.

Returns `VK_SUCCESS` if no more moves are possible or `VK_INCOMPLETE` if more defragmentations are possible.

Ends incremental defragmentation pass and commits all defragmentation moves from `pPassInfo`.
After this call:

- Allocations at `pPassInfo[i].srcAllocation` that had `pPassInfo[i].operation ==` #VMA_DEFRAGMENTATION_MOVE_OPERATION_COPY
  (which is the default) will be pointing to the new destination place.
- Allocation at `pPassInfo[i].srcAllocation` that had `pPassInfo[i].operation ==` #VMA_DEFRAGMENTATION_MOVE_OPERATION_DESTROY
  will be freed.

If no more moves are possible you can end whole defragmentation.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaEndDefragmentationPass(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaDefragmentationContext VMA_NOT_NULL context,
    VmaDefragmentationPassMoveInfo* VMA_NOT_NULL pPassInfo);

/** \brief Binds buffer to allocation.

Binds specified buffer to region of memory represented by specified allocation.
Gets `VkDeviceMemory` handle and offset from the allocation.
If you want to create a buffer, allocate memory for it and bind them together separately,
you should use this function for binding instead of standard `vkBindBufferMemory()`,
because it ensures proper synchronization so that when a `VkDeviceMemory` object is used by multiple
allocations, calls to `vkBind*Memory()` or `vkMapMemory()` won't happen from multiple threads simultaneously
(which is illegal in Vulkan).

It is recommended to use function vmaCreateBuffer() instead of this one.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindBufferMemory(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VkBuffer VMA_NOT_NULL_NON_DISPATCHABLE buffer);

/** \brief Binds buffer to allocation with additional parameters.

\param allocator
\param allocation
\param allocationLocalOffset Additional offset to be added while binding, relative to the beginning of the `allocation`. Normally it should be 0.
\param buffer
\param pNext A chain of structures to be attached to `VkBindBufferMemoryInfoKHR` structure used internally. Normally it should be null.

This function is similar to vmaBindBufferMemory(), but it provides additional parameters.

If `pNext` is not null, #VmaAllocator object must have been created with #VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT flag
or with VmaAllocatorCreateInfo::vulkanApiVersion `>= VK_API_VERSION_1_1`. Otherwise the call fails.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindBufferMemory2(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VkDeviceSize allocationLocalOffset,
    VkBuffer VMA_NOT_NULL_NON_DISPATCHABLE buffer,
    const void* VMA_NULLABLE VMA_EXTENDS_VK_STRUCT(VkBindBufferMemoryInfoKHR) pNext);

/** \brief Binds image to allocation.

Binds specified image to region of memory represented by specified allocation.
Gets `VkDeviceMemory` handle and offset from the allocation.
If you want to create an image, allocate memory for it and bind them together separately,
you should use this function for binding instead of standard `vkBindImageMemory()`,
because it ensures proper synchronization so that when a `VkDeviceMemory` object is used by multiple
allocations, calls to `vkBind*Memory()` or `vkMapMemory()` won't happen from multiple threads simultaneously
(which is illegal in Vulkan).

It is recommended to use function vmaCreateImage() instead of this one.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindImageMemory(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VkImage VMA_NOT_NULL_NON_DISPATCHABLE image);

/** \brief Binds image to allocation with additional parameters.

\param allocator
\param allocation
\param allocationLocalOffset Additional offset to be added while binding, relative to the beginning of the `allocation`. Normally it should be 0.
\param image
\param pNext A chain of structures to be attached to `VkBindImageMemoryInfoKHR` structure used internally. Normally it should be null.

This function is similar to vmaBindImageMemory(), but it provides additional parameters.

If `pNext` is not null, #VmaAllocator object must have been created with #VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT flag
or with VmaAllocatorCreateInfo::vulkanApiVersion `>= VK_API_VERSION_1_1`. Otherwise the call fails.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindImageMemory2(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VkDeviceSize allocationLocalOffset,
    VkImage VMA_NOT_NULL_NON_DISPATCHABLE image,
    const void* VMA_NULLABLE VMA_EXTENDS_VK_STRUCT(VkBindImageMemoryInfoKHR) pNext);

/** \brief Creates a new `VkBuffer`, allocates and binds memory for it.

\param allocator
\param pBufferCreateInfo
\param pAllocationCreateInfo
\param[out] pBuffer Buffer that was created.
\param[out] pAllocation Allocation that was created.
\param[out] pAllocationInfo Optional. Information about allocated memory. It can be later fetched using function vmaGetAllocationInfo().

This function automatically:

-# Creates buffer.
-# Allocates appropriate memory for it.
-# Binds the buffer with the memory.

If any of these operations fail, buffer and allocation are not created,
returned value is negative error code, `*pBuffer` and `*pAllocation` are null.

If the function succeeded, you must destroy both buffer and allocation when you
no longer need them using either convenience function vmaDestroyBuffer() or
separately, using `vkDestroyBuffer()` and vmaFreeMemory().

If #VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT flag was used,
VK_KHR_dedicated_allocation extension is used internally to query driver whether
it requires or prefers the new buffer to have dedicated allocation. If yes,
and if dedicated allocation is possible
(#VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT is not used), it creates dedicated
allocation for this buffer, just like when using
#VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.

\note This function creates a new `VkBuffer`. Sub-allocation of parts of one large buffer,
although recommended as a good practice, is out of scope of this library and could be implemented
by the user as a higher-level logic on top of VMA.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateBuffer(
    VmaAllocator VMA_NOT_NULL allocator,
    const VkBufferCreateInfo* VMA_NOT_NULL pBufferCreateInfo,
    const VmaAllocationCreateInfo* VMA_NOT_NULL pAllocationCreateInfo,
    VkBuffer VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pBuffer,
    VmaAllocation VMA_NULLABLE* VMA_NOT_NULL pAllocation,
    VmaAllocationInfo* VMA_NULLABLE pAllocationInfo);

/** \brief Creates a buffer with additional minimum alignment.

Similar to vmaCreateBuffer() but provides additional parameter `minAlignment` which allows to specify custom,
minimum alignment to be used when placing the buffer inside a larger memory block, which may be needed e.g.
for interop with OpenGL.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateBufferWithAlignment(
    VmaAllocator VMA_NOT_NULL allocator,
    const VkBufferCreateInfo* VMA_NOT_NULL pBufferCreateInfo,
    const VmaAllocationCreateInfo* VMA_NOT_NULL pAllocationCreateInfo,
    VkDeviceSize minAlignment,
    VkBuffer VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pBuffer,
    VmaAllocation VMA_NULLABLE* VMA_NOT_NULL pAllocation,
    VmaAllocationInfo* VMA_NULLABLE pAllocationInfo);

/** \brief Creates a new `VkBuffer`, binds already created memory for it.

\param allocator
\param allocation Allocation that provides memory to be used for binding new buffer to it.
\param pBufferCreateInfo
\param[out] pBuffer Buffer that was created.

This function automatically:

-# Creates buffer.
-# Binds the buffer with the supplied memory.

If any of these operations fail, buffer is not created,
returned value is negative error code and `*pBuffer` is null.

If the function succeeded, you must destroy the buffer when you
no longer need it using `vkDestroyBuffer()`. If you want to also destroy the corresponding
allocation you can use convenience function vmaDestroyBuffer().

\note There is a new version of this function augmented with parameter `allocationLocalOffset` - see vmaCreateAliasingBuffer2().
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAliasingBuffer(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    const VkBufferCreateInfo* VMA_NOT_NULL pBufferCreateInfo,
    VkBuffer VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pBuffer);

/** \brief Creates a new `VkBuffer`, binds already created memory for it.

\param allocator
\param allocation Allocation that provides memory to be used for binding new buffer to it.
\param allocationLocalOffset Additional offset to be added while binding, relative to the beginning of the allocation. Normally it should be 0.
\param pBufferCreateInfo 
\param[out] pBuffer Buffer that was created.

This function automatically:

-# Creates buffer.
-# Binds the buffer with the supplied memory.

If any of these operations fail, buffer is not created,
returned value is negative error code and `*pBuffer` is null.

If the function succeeded, you must destroy the buffer when you
no longer need it using `vkDestroyBuffer()`. If you want to also destroy the corresponding
allocation you can use convenience function vmaDestroyBuffer().

\note This is a new version of the function augmented with parameter `allocationLocalOffset`.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAliasingBuffer2(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VkDeviceSize allocationLocalOffset,
    const VkBufferCreateInfo* VMA_NOT_NULL pBufferCreateInfo,
    VkBuffer VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pBuffer);

/** \brief Destroys Vulkan buffer and frees allocated memory.

This is just a convenience function equivalent to:

\code
vkDestroyBuffer(device, buffer, allocationCallbacks);
vmaFreeMemory(allocator, allocation);
\endcode

It is safe to pass null as buffer and/or allocation.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaDestroyBuffer(
    VmaAllocator VMA_NOT_NULL allocator,
    VkBuffer VMA_NULLABLE_NON_DISPATCHABLE buffer,
    VmaAllocation VMA_NULLABLE allocation);

/// Function similar to vmaCreateBuffer().
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateImage(
    VmaAllocator VMA_NOT_NULL allocator,
    const VkImageCreateInfo* VMA_NOT_NULL pImageCreateInfo,
    const VmaAllocationCreateInfo* VMA_NOT_NULL pAllocationCreateInfo,
    VkImage VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pImage,
    VmaAllocation VMA_NULLABLE* VMA_NOT_NULL pAllocation,
    VmaAllocationInfo* VMA_NULLABLE pAllocationInfo);

/// Function similar to vmaCreateAliasingBuffer() but for images.
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAliasingImage(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    const VkImageCreateInfo* VMA_NOT_NULL pImageCreateInfo,
    VkImage VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pImage);

/// Function similar to vmaCreateAliasingBuffer2() but for images.
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAliasingImage2(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VkDeviceSize allocationLocalOffset,
    const VkImageCreateInfo* VMA_NOT_NULL pImageCreateInfo,
    VkImage VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pImage);

/** \brief Destroys Vulkan image and frees allocated memory.

This is just a convenience function equivalent to:

\code
vkDestroyImage(device, image, allocationCallbacks);
vmaFreeMemory(allocator, allocation);
\endcode

It is safe to pass null as image and/or allocation.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaDestroyImage(
    VmaAllocator VMA_NOT_NULL allocator,
    VkImage VMA_NULLABLE_NON_DISPATCHABLE image,
    VmaAllocation VMA_NULLABLE allocation);

/** @} */

/**
\addtogroup group_virtual
@{
*/

/** \brief Creates new #VmaVirtualBlock object.

\param pCreateInfo Parameters for creation.
\param[out] pVirtualBlock Returned virtual block object or `VMA_NULL` if creation failed.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateVirtualBlock(
    const VmaVirtualBlockCreateInfo* VMA_NOT_NULL pCreateInfo,
    VmaVirtualBlock VMA_NULLABLE* VMA_NOT_NULL pVirtualBlock);

/** \brief Destroys #VmaVirtualBlock object.

Please note that you should consciously handle virtual allocations that could remain unfreed in the block.
You should either free them individually using vmaVirtualFree() or call vmaClearVirtualBlock()
if you are sure this is what you want. If you do neither, an assert is called.

If you keep pointers to some additional metadata associated with your virtual allocations in their `pUserData`,
don't forget to free them.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaDestroyVirtualBlock(
    VmaVirtualBlock VMA_NULLABLE virtualBlock);

/** \brief Returns true of the #VmaVirtualBlock is empty - contains 0 virtual allocations and has all its space available for new allocations.
*/
VMA_CALL_PRE VkBool32 VMA_CALL_POST vmaIsVirtualBlockEmpty(
    VmaVirtualBlock VMA_NOT_NULL virtualBlock);

/** \brief Returns information about a specific virtual allocation within a virtual block, like its size and `pUserData` pointer.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetVirtualAllocationInfo(
    VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    VmaVirtualAllocation VMA_NOT_NULL_NON_DISPATCHABLE allocation, VmaVirtualAllocationInfo* VMA_NOT_NULL pVirtualAllocInfo);

/** \brief Allocates new virtual allocation inside given #VmaVirtualBlock.

If the allocation fails due to not enough free space available, `VK_ERROR_OUT_OF_DEVICE_MEMORY` is returned
(despite the function doesn't ever allocate actual GPU memory).
`pAllocation` is then set to `VK_NULL_HANDLE` and `pOffset`, if not null, it set to `UINT64_MAX`.

\param virtualBlock Virtual block
\param pCreateInfo Parameters for the allocation
\param[out] pAllocation Returned handle of the new allocation
\param[out] pOffset Returned offset of the new allocation. Optional, can be null.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaVirtualAllocate(
    VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    const VmaVirtualAllocationCreateInfo* VMA_NOT_NULL pCreateInfo,
    VmaVirtualAllocation VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pAllocation,
    VkDeviceSize* VMA_NULLABLE pOffset);

/** \brief Frees virtual allocation inside given #VmaVirtualBlock.

It is correct to call this function with `allocation == VK_NULL_HANDLE` - it does nothing.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaVirtualFree(
    VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    VmaVirtualAllocation VMA_NULLABLE_NON_DISPATCHABLE allocation);

/** \brief Frees all virtual allocations inside given #VmaVirtualBlock.

You must either call this function or free each virtual allocation individually with vmaVirtualFree()
before destroying a virtual block. Otherwise, an assert is called.

If you keep pointer to some additional metadata associated with your virtual allocation in its `pUserData`,
don't forget to free it as well.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaClearVirtualBlock(
    VmaVirtualBlock VMA_NOT_NULL virtualBlock);

/** \brief Changes custom pointer associated with given virtual allocation.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaSetVirtualAllocationUserData(
    VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    VmaVirtualAllocation VMA_NOT_NULL_NON_DISPATCHABLE allocation,
    void* VMA_NULLABLE pUserData);

/** \brief Calculates and returns statistics about virtual allocations and memory usage in given #VmaVirtualBlock.

This function is fast to call. For more detailed statistics, see vmaCalculateVirtualBlockStatistics().
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetVirtualBlockStatistics(
    VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    VmaStatistics* VMA_NOT_NULL pStats);

/** \brief Calculates and returns detailed statistics about virtual allocations and memory usage in given #VmaVirtualBlock.

This function is slow to call. Use for debugging purposes.
For less detailed statistics, see vmaGetVirtualBlockStatistics().
*/
VMA_CALL_PRE void VMA_CALL_POST vmaCalculateVirtualBlockStatistics(
    VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    VmaDetailedStatistics* VMA_NOT_NULL pStats);

/** @} */

#if VMA_STATS_STRING_ENABLED
/**
\addtogroup group_stats
@{
*/

/** \brief Builds and returns a null-terminated string in JSON format with information about given #VmaVirtualBlock.
\param virtualBlock Virtual block.
\param[out] ppStatsString Returned string.
\param detailedMap Pass `VK_FALSE` to only obtain statistics as returned by vmaCalculateVirtualBlockStatistics(). Pass `VK_TRUE` to also obtain full list of allocations and free spaces.

Returned string must be freed using vmaFreeVirtualBlockStatsString().
*/
VMA_CALL_PRE void VMA_CALL_POST vmaBuildVirtualBlockStatsString(
    VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    char* VMA_NULLABLE* VMA_NOT_NULL ppStatsString,
    VkBool32 detailedMap);

/// Frees a string returned by vmaBuildVirtualBlockStatsString().
VMA_CALL_PRE void VMA_CALL_POST vmaFreeVirtualBlockStatsString(
    VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    char* VMA_NULLABLE pStatsString);

/** \brief Builds and returns statistics as a null-terminated string in JSON format.
\param allocator
\param[out] ppStatsString Must be freed using vmaFreeStatsString() function.
\param detailedMap
*/
VMA_CALL_PRE void VMA_CALL_POST vmaBuildStatsString(
    VmaAllocator VMA_NOT_NULL allocator,
    char* VMA_NULLABLE* VMA_NOT_NULL ppStatsString,
    VkBool32 detailedMap);

VMA_CALL_PRE void VMA_CALL_POST vmaFreeStatsString(
    VmaAllocator VMA_NOT_NULL allocator,
    char* VMA_NULLABLE pStatsString);

/** @} */

#endif // VMA_STATS_STRING_ENABLED

#endif // _VMA_FUNCTION_HEADERS

#ifdef __cplusplus
}
#endif

#endif // AMD_VULKAN_MEMORY_ALLOCATOR_H

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
//    IMPLEMENTATION
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// For Visual Studio IntelliSense.
#if defined(__cplusplus) && defined(__INTELLISENSE__)
#define VMA_IMPLEMENTATION
#endif

#ifdef VMA_IMPLEMENTATION
#undef VMA_IMPLEMENTATION

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <utility>
#include <type_traits>

#if !defined(VMA_CPP20)
    #if __cplusplus >= 202002L || _MSVC_LANG >= 202002L // C++20
        #define VMA_CPP20 1
    #else
        #define VMA_CPP20 0
    #endif
#endif

#ifdef _MSC_VER
    #include <intrin.h> // For functions like __popcnt, _BitScanForward etc.
#endif
#if VMA_CPP20
    #include <bit>
#endif

#if VMA_STATS_STRING_ENABLED
    #include <cstdio> // For snprintf
#endif

/*******************************************************************************
CONFIGURATION SECTION

Define some of these macros before each #include of this header or change them
here if you need other then default behavior depending on your environment.
*/
#ifndef _VMA_CONFIGURATION

/*
Define this macro to 1 to make the library fetch pointers to Vulkan functions
internally, like:

    vulkanFunctions.vkAllocateMemory = &vkAllocateMemory;
*/
#if !defined(VMA_STATIC_VULKAN_FUNCTIONS) && !defined(VK_NO_PROTOTYPES)
    #define VMA_STATIC_VULKAN_FUNCTIONS 1
#endif

/*
Define this macro to 1 to make the library fetch pointers to Vulkan functions
internally, like:

    vulkanFunctions.vkAllocateMemory = (PFN_vkAllocateMemory)vkGetDeviceProcAddr(device, "vkAllocateMemory");

To use this feature in new versions of VMA you now have to pass
VmaVulkanFunctions::vkGetInstanceProcAddr and vkGetDeviceProcAddr as
VmaAllocatorCreateInfo::pVulkanFunctions. Other members can be null.
*/
#if !defined(VMA_DYNAMIC_VULKAN_FUNCTIONS)
    #define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#endif

#ifndef VMA_USE_STL_SHARED_MUTEX
    #if __cplusplus >= 201703L || _MSVC_LANG >= 201703L // C++17
        #define VMA_USE_STL_SHARED_MUTEX 1
    // Visual studio defines __cplusplus properly only when passed additional parameter: /Zc:__cplusplus
    // Otherwise it is always 199711L, despite shared_mutex works since Visual Studio 2015 Update 2.
    #elif defined(_MSC_FULL_VER) && _MSC_FULL_VER >= 190023918 && __cplusplus == 199711L && _MSVC_LANG >= 201703L
        #define VMA_USE_STL_SHARED_MUTEX 1
    #else
        #define VMA_USE_STL_SHARED_MUTEX 0
    #endif
#endif

/*
Define this macro to include custom header files without having to edit this file directly, e.g.:

    // Inside of "my_vma_configuration_user_includes.h":

    #include "my_custom_assert.h" // for MY_CUSTOM_ASSERT
    #include "my_custom_min.h" // for my_custom_min
    #include <algorithm>
    #include <mutex>

    // Inside a different file, which includes "vk_mem_alloc.h":

    #define VMA_CONFIGURATION_USER_INCLUDES_H "my_vma_configuration_user_includes.h"
    #define VMA_ASSERT(expr) MY_CUSTOM_ASSERT(expr)
    #define VMA_MIN(v1, v2)  (my_custom_min(v1, v2))
    #include "vk_mem_alloc.h"
    ...

The following headers are used in this CONFIGURATION section only, so feel free to
remove them if not needed.
*/
#if !defined(VMA_CONFIGURATION_USER_INCLUDES_H)
    #include <cassert> // for assert
    #include <algorithm> // for min, max, swap
    #include <mutex>
#else
    #include VMA_CONFIGURATION_USER_INCLUDES_H
#endif

#ifndef VMA_NULL
   // Value used as null pointer. Define it to e.g.: nullptr, NULL, 0, (void*)0.
   #define VMA_NULL   nullptr
#endif

#ifndef VMA_FALLTHROUGH
    #if __cplusplus >= 201703L || _MSVC_LANG >= 201703L // C++17
        #define VMA_FALLTHROUGH [[fallthrough]]
    #else
        #define VMA_FALLTHROUGH
    #endif
#endif

// Normal assert to check for programmer's errors, especially in Debug configuration.
#ifndef VMA_ASSERT
   #ifdef NDEBUG
       #define VMA_ASSERT(expr)
   #else
       #define VMA_ASSERT(expr)         assert(expr)
   #endif
#endif

// Assert that will be called very often, like inside data structures e.g. operator[].
// Making it non-empty can make program slow.
#ifndef VMA_HEAVY_ASSERT
   #ifdef NDEBUG
       #define VMA_HEAVY_ASSERT(expr)
   #else
       #define VMA_HEAVY_ASSERT(expr)   //VMA_ASSERT(expr)
   #endif
#endif

// Assert used for reporting memory leaks - unfreed allocations.
#ifndef VMA_ASSERT_LEAK
    #define VMA_ASSERT_LEAK(expr)   VMA_ASSERT(expr)
#endif

// If your compiler is not compatible with C++17 and definition of
// aligned_alloc() function is missing, uncommenting following line may help:

//#include <malloc.h>

#if defined(__ANDROID_API__) && (__ANDROID_API__ < 16)
#include <cstdlib>
static void* vma_aligned_alloc(size_t alignment, size_t size)
{
    // alignment must be >= sizeof(void*)
    if(alignment < sizeof(void*))
    {
        alignment = sizeof(void*);
    }

    return memalign(alignment, size);
}
#elif defined(__APPLE__) || defined(__ANDROID__) || (defined(__linux__) && defined(__GLIBCXX__) && !defined(_GLIBCXX_HAVE_ALIGNED_ALLOC))
#include <cstdlib>

#if defined(__APPLE__)
#include <AvailabilityMacros.h>
#endif

static void* vma_aligned_alloc(size_t alignment, size_t size)
{
    // Unfortunately, aligned_alloc causes VMA to crash due to it returning null pointers. (At least under 11.4)
    // Therefore, for now disable this specific exception until a proper solution is found.
    //#if defined(__APPLE__) && (defined(MAC_OS_X_VERSION_10_16) || defined(__IPHONE_14_0))
    //#if MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_16 || __IPHONE_OS_VERSION_MAX_ALLOWED >= __IPHONE_14_0
    //    // For C++14, usr/include/malloc/_malloc.h declares aligned_alloc()) only
    //    // with the MacOSX11.0 SDK in Xcode 12 (which is what adds
    //    // MAC_OS_X_VERSION_10_16), even though the function is marked
    //    // available for 10.15. That is why the preprocessor checks for 10.16 but
    //    // the __builtin_available checks for 10.15.
    //    // People who use C++17 could call aligned_alloc with the 10.15 SDK already.
    //    if (__builtin_available(macOS 10.15, iOS 13, *))
    //        return aligned_alloc(alignment, size);
    //#endif
    //#endif

    // alignment must be >= sizeof(void*)
    if(alignment < sizeof(void*))
    {
        alignment = sizeof(void*);
    }

    void *pointer;
    if(posix_memalign(&pointer, alignment, size) == 0)
        return pointer;
    return VMA_NULL;
}
#elif defined(_WIN32)
static void* _Nullable vma_aligned_alloc(size_t alignment, size_t size)
{
    return _aligned_malloc(size, alignment);
}
#elif __cplusplus >= 201703L || _MSVC_LANG >= 201703L // C++17
static void* _Nullable vma_aligned_alloc(size_t alignment, size_t size)
{
    return aligned_alloc(alignment, size);
}
#else
static void* vma_aligned_alloc(size_t alignment, size_t size)
{
    VMA_ASSERT(0 && "Could not implement aligned_alloc automatically. Please enable C++17 or later in your compiler or provide custom implementation of macro VMA_SYSTEM_ALIGNED_MALLOC (and VMA_SYSTEM_ALIGNED_FREE if needed) using the API of your system.");
    return VMA_NULL;
}
#endif

#if defined(_WIN32)
static void vma_aligned_free(void* VMA_NULLABLE ptr)
{
    _aligned_free(ptr);
}
#else
static void vma_aligned_free(void* VMA_NULLABLE ptr)
{
    free(ptr);
}
#endif

#ifndef VMA_ALIGN_OF
   #define VMA_ALIGN_OF(type)       (alignof(type))
#endif

#ifndef VMA_SYSTEM_ALIGNED_MALLOC
   #define VMA_SYSTEM_ALIGNED_MALLOC(size, alignment) vma_aligned_alloc((alignment), (size))
#endif

#ifndef VMA_SYSTEM_ALIGNED_FREE
   // VMA_SYSTEM_FREE is the old name, but might have been defined by the user
   #if defined(VMA_SYSTEM_FREE)
      #define VMA_SYSTEM_ALIGNED_FREE(ptr)     VMA_SYSTEM_FREE(ptr)
   #else
      #define VMA_SYSTEM_ALIGNED_FREE(ptr)     vma_aligned_free(ptr)
    #endif
#endif

#ifndef VMA_COUNT_BITS_SET
    // Returns number of bits set to 1 in (v)
    #define VMA_COUNT_BITS_SET(v) VmaCountBitsSet(v)
#endif

#ifndef VMA_BITSCAN_LSB
    // Scans integer for index of first nonzero value from the Least Significant Bit (LSB). If mask is 0 then returns UINT8_MAX
    #define VMA_BITSCAN_LSB(mask) VmaBitScanLSB(mask)
#endif

#ifndef VMA_BITSCAN_MSB
    // Scans integer for index of first nonzero value from the Most Significant Bit (MSB). If mask is 0 then returns UINT8_MAX
    #define VMA_BITSCAN_MSB(mask) VmaBitScanMSB(mask)
#endif

#ifndef VMA_MIN
   #define VMA_MIN(v1, v2)    ((std::min)((v1), (v2)))
#endif

#ifndef VMA_MAX
   #define VMA_MAX(v1, v2)    ((std::max)((v1), (v2)))
#endif

#ifndef VMA_SORT
   #define VMA_SORT(beg, end, cmp)  std::sort(beg, end, cmp)
#endif

#ifndef VMA_DEBUG_LOG_FORMAT
   #define VMA_DEBUG_LOG_FORMAT(format, ...)
   /*
   #define VMA_DEBUG_LOG_FORMAT(format, ...) do { \
       printf((format), __VA_ARGS__); \
       printf("\n"); \
   } while(false)
   */
#endif

#ifndef VMA_DEBUG_LOG
    #define VMA_DEBUG_LOG(str)   VMA_DEBUG_LOG_FORMAT("%s", (str))
#endif

#ifndef VMA_LEAK_LOG_FORMAT
    #define VMA_LEAK_LOG_FORMAT(format, ...)   VMA_DEBUG_LOG_FORMAT(format, __VA_ARGS__)
#endif

#ifndef VMA_CLASS_NO_COPY
    #define VMA_CLASS_NO_COPY(className) \
        private: \
            className(const className&) = delete; \
            className& operator=(const className&) = delete;
#endif
#ifndef VMA_CLASS_NO_COPY_NO_MOVE
    #define VMA_CLASS_NO_COPY_NO_MOVE(className) \
        private: \
            className(const className&) = delete; \
            className(className&&) = delete; \
            className& operator=(const className&) = delete; \
            className& operator=(className&&) = delete;
#endif

// Define this macro to 1 to enable functions: vmaBuildStatsString, vmaFreeStatsString.
#if VMA_STATS_STRING_ENABLED
    static inline void VmaUint32ToStr(char* VMA_NOT_NULL outStr, size_t strLen, uint32_t num)
    {
        snprintf(outStr, strLen, "%" PRIu32, num);
    }
    static inline void VmaUint64ToStr(char* VMA_NOT_NULL outStr, size_t strLen, uint64_t num)
    {
        snprintf(outStr, strLen, "%" PRIu64, num);
    }
    static inline void VmaPtrToStr(char* VMA_NOT_NULL outStr, size_t strLen, const void* VMA_NULLABLE ptr)
    {
        snprintf(outStr, strLen, "%p", ptr);
    }
#endif

#ifndef VMA_MUTEX
    class VmaMutex
    {
    VMA_CLASS_NO_COPY_NO_MOVE(VmaMutex)
    public:
        VmaMutex() = default;
        void lock() { _mutex.lock(); }
        void unlock() { _mutex.unlock(); }
        bool try_lock() { return _mutex.try_lock(); }
    private:
        std::mutex _mutex;
    };
    #define VMA_MUTEX VmaMutex
#endif

// Read-write mutex, where "read" is shared access, "write" is exclusive access.
#ifndef VMA_RW_MUTEX
    #if VMA_USE_STL_SHARED_MUTEX
        // Use std::shared_mutex from C++17.
        #include <shared_mutex>
        class VmaRWMutex
        {
        public:
            void lock_read() { _mutex.lock_shared(); }
            void unlock_read() { _mutex.unlock_shared(); }
            bool try_lock_read() { return _mutex.try_lock_shared(); }
            void lock_write() { _mutex.lock(); }
            void unlock_write() { _mutex.unlock(); }
            bool try_lock_write() { return _mutex.try_lock(); }
        private:
            std::shared_mutex _mutex;
        };
        #define VMA_RW_MUTEX VmaRWMutex
    #elif defined(_WIN32) && defined(WINVER) && defined(SRWLOCK_INIT) && WINVER >= 0x0600
        // Use SRWLOCK from WinAPI.
        // Minimum supported client = Windows Vista, server = Windows Server 2008.
        class VmaRWMutex
        {
        public:
            VmaRWMutex() { InitializeSRWLock(&_lock); }
            void lock_read() { AcquireSRWLockShared(&_lock); }
            void unlock_read() { ReleaseSRWLockShared(&_lock); }
            bool try_lock_read() { return TryAcquireSRWLockShared(&_lock) != FALSE; }
            void lock_write() { AcquireSRWLockExclusive(&_lock); }
            void unlock_write() { ReleaseSRWLockExclusive(&_lock); }
            bool try_lock_write() { return TryAcquireSRWLockExclusive(&_lock) != FALSE; }
        private:
            SRWLOCK _lock;
        };
        #define VMA_RW_MUTEX VmaRWMutex
    #else
        // Less efficient fallback: Use normal mutex.
        class VmaRWMutex
        {
        public:
            void lock_read() { _mutex.lock(); }
            void unlock_read() { _mutex.unlock(); }
            bool try_lock_read() { return _mutex.try_lock(); }
            void lock_write() { _mutex.lock(); }
            void unlock_write() { _mutex.unlock(); }
            bool try_lock_write() { return _mutex.try_lock(); }
        private:
            VMA_MUTEX _mutex;
        };
        #define VMA_RW_MUTEX VmaRWMutex
    #endif // #if VMA_USE_STL_SHARED_MUTEX
#endif // #ifndef VMA_RW_MUTEX

/*
If providing your own implementation, you need to implement a subset of std::atomic.
*/
#ifndef VMA_ATOMIC_UINT32
    #include <atomic>
    #define VMA_ATOMIC_UINT32 std::atomic<uint32_t>
#endif

#ifndef VMA_ATOMIC_UINT64
    #include <atomic>
    #define VMA_ATOMIC_UINT64 std::atomic<uint64_t>
#endif

#ifndef VMA_DEBUG_ALWAYS_DEDICATED_MEMORY
    /**
    Every allocation will have its own memory block.
    Define to 1 for debugging purposes only.
    */
    #define VMA_DEBUG_ALWAYS_DEDICATED_MEMORY (0)
#endif

#ifndef VMA_MIN_ALIGNMENT
    /**
    Minimum alignment of all allocations, in bytes.
    Set to more than 1 for debugging purposes. Must be power of two.
    */
    #ifdef VMA_DEBUG_ALIGNMENT // Old name
        #define VMA_MIN_ALIGNMENT VMA_DEBUG_ALIGNMENT
    #else
        #define VMA_MIN_ALIGNMENT (1)
    #endif
#endif

#ifndef VMA_DEBUG_MARGIN
    /**
    Minimum margin after every allocation, in bytes.
    Set nonzero for debugging purposes only.
    */
    #define VMA_DEBUG_MARGIN (0)
#endif

#ifndef VMA_DEBUG_INITIALIZE_ALLOCATIONS
    /**
    Define this macro to 1 to automatically fill new allocations and destroyed
    allocations with some bit pattern.
    */
    #define VMA_DEBUG_INITIALIZE_ALLOCATIONS (0)
#endif

#ifndef VMA_DEBUG_DETECT_CORRUPTION
    /**
    Define this macro to 1 together with non-zero value of VMA_DEBUG_MARGIN to
    enable writing magic value to the margin after every allocation and
    validating it, so that memory corruptions (out-of-bounds writes) are detected.
    */
    #define VMA_DEBUG_DETECT_CORRUPTION (0)
#endif

#ifndef VMA_DEBUG_GLOBAL_MUTEX
    /**
    Set this to 1 for debugging purposes only, to enable single mutex protecting all
    entry calls to the library. Can be useful for debugging multithreading issues.
    */
    #define VMA_DEBUG_GLOBAL_MUTEX (0)
#endif

#ifndef VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY
    /**
    Minimum value for VkPhysicalDeviceLimits::bufferImageGranularity.
    Set to more than 1 for debugging purposes only. Must be power of two.
    */
    #define VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY (1)
#endif

#ifndef VMA_DEBUG_DONT_EXCEED_MAX_MEMORY_ALLOCATION_COUNT
    /*
    Set this to 1 to make VMA never exceed VkPhysicalDeviceLimits::maxMemoryAllocationCount
    and return error instead of leaving up to Vulkan implementation what to do in such cases.
    */
    #define VMA_DEBUG_DONT_EXCEED_MAX_MEMORY_ALLOCATION_COUNT (1)
#endif

#ifndef VMA_DEBUG_DONT_EXCEED_HEAP_SIZE_WITH_ALLOCATION_SIZE
    /*
    Set this to 1 to make VMA never exceed VkPhysicalDeviceMemoryProperties::memoryHeaps[i].size
    with a single allocation size VkMemoryAllocateInfo::allocationSize
    and return error instead of leaving up to Vulkan implementation what to do in such cases.
    It protects agaist validation error VUID-vkAllocateMemory-pAllocateInfo-01713.
    On the other hand, allowing exceeding this size may result in a successful allocation despite the validation error.
    */
    #define VMA_DEBUG_DONT_EXCEED_HEAP_SIZE_WITH_ALLOCATION_SIZE (1)
#endif

#ifndef VMA_SMALL_HEAP_MAX_SIZE
   /// Maximum size of a memory heap in Vulkan to consider it "small".
   #define VMA_SMALL_HEAP_MAX_SIZE (1024ULL * 1024 * 1024)
#endif

#ifndef VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE
   /// Default size of a block allocated as single VkDeviceMemory from a "large" heap.
   #define VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE (256ULL * 1024 * 1024)
#endif

/*
Mapping hysteresis is a logic that launches when vmaMapMemory/vmaUnmapMemory is called
or a persistently mapped allocation is created and destroyed several times in a row.
It keeps additional +1 mapping of a device memory block to prevent calling actual
vkMapMemory/vkUnmapMemory too many times, which may improve performance and help
tools like RenderDoc.
*/
#ifndef VMA_MAPPING_HYSTERESIS_ENABLED
    #define VMA_MAPPING_HYSTERESIS_ENABLED 1
#endif

#define VMA_VALIDATE(cond) do { if(!(cond)) { \
        VMA_ASSERT(0 && "Validation failed: " #cond); \
        return false; \
    } } while(false)

/*******************************************************************************
END OF CONFIGURATION
*/
#endif // _VMA_CONFIGURATION


static const uint8_t VMA_ALLOCATION_FILL_PATTERN_CREATED = 0xDC;
static const uint8_t VMA_ALLOCATION_FILL_PATTERN_DESTROYED = 0xEF;
// Decimal 2139416166, float NaN, little-endian binary 66 E6 84 7F.
static const uint32_t VMA_CORRUPTION_DETECTION_MAGIC_VALUE = 0x7F84E666;

// Copy of some Vulkan definitions so we don't need to check their existence just to handle few constants.
static const uint32_t VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD_COPY = 0x00000040;
static const uint32_t VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD_COPY = 0x00000080;
static const uint32_t VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_COPY = 0x00020000;
static const uint32_t VK_IMAGE_CREATE_DISJOINT_BIT_COPY = 0x00000200;
static const int32_t VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT_COPY = 1000158000;
static const uint32_t VMA_ALLOCATION_INTERNAL_STRATEGY_MIN_OFFSET = 0x10000000U;
static const uint32_t VMA_ALLOCATION_TRY_COUNT = 32;
static const uint32_t VMA_VENDOR_ID_AMD = 4098;

// This one is tricky. Vulkan specification defines this code as available since
// Vulkan 1.0, but doesn't actually define it in Vulkan SDK earlier than 1.2.131.
// See pull request #207.
#define VK_ERROR_UNKNOWN_COPY ((VkResult)-13)


#if VMA_STATS_STRING_ENABLED
// Correspond to values of enum VmaSuballocationType.
static const char* _Nonnull const VMA_SUBALLOCATION_TYPE_NAMES[] =
{
    "FREE",
    "UNKNOWN",
    "BUFFER",
    "IMAGE_UNKNOWN",
    "IMAGE_LINEAR",
    "IMAGE_OPTIMAL",
};
#endif

static const VkAllocationCallbacks VmaEmptyAllocationCallbacks =
    { VMA_NULL, VMA_NULL, VMA_NULL, VMA_NULL, VMA_NULL, VMA_NULL };


#ifndef _VMA_ENUM_DECLARATIONS

enum VmaSuballocationType
{
    VMA_SUBALLOCATION_TYPE_FREE = 0,
    VMA_SUBALLOCATION_TYPE_UNKNOWN = 1,
    VMA_SUBALLOCATION_TYPE_BUFFER = 2,
    VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN = 3,
    VMA_SUBALLOCATION_TYPE_IMAGE_LINEAR = 4,
    VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL = 5,
    VMA_SUBALLOCATION_TYPE_MAX_ENUM = 0x7FFFFFFF
};

enum VMA_CACHE_OPERATION
{
    VMA_CACHE_FLUSH,
    VMA_CACHE_INVALIDATE
};

enum class VmaAllocationRequestType
{
    Normal,
    TLSF,
    // Used by "Linear" algorithm.
    UpperAddress,
    EndOf1st,
    EndOf2nd,
};

#endif // _VMA_ENUM_DECLARATIONS

#ifndef _VMA_FORWARD_DECLARATIONS
// Opaque handle used by allocation algorithms to identify single allocation in any conforming way.
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VmaAllocHandle);

struct VmaMutexLock;
struct VmaMutexLockRead;
struct VmaMutexLockWrite;

template<typename T>
struct AtomicTransactionalIncrement;

template<typename T>
struct VmaStlAllocator;

template<typename T, typename AllocatorT>
class VmaVector;

template<typename T, typename AllocatorT, size_t N>
class VmaSmallVector;

template<typename T>
class VmaPoolAllocator;

template<typename T>
struct VmaListItem;

template<typename T>
class VmaRawList;

template<typename T, typename AllocatorT>
class VmaList;

template<typename ItemTypeTraits>
class VmaIntrusiveLinkedList;

#if VMA_STATS_STRING_ENABLED
class VmaStringBuilder;
class VmaJsonWriter;
#endif

class VmaDeviceMemoryBlock;

struct VmaDedicatedAllocationListItemTraits;
class VmaDedicatedAllocationList;

struct VmaSuballocation;
struct VmaSuballocationOffsetLess;
struct VmaSuballocationOffsetGreater;
struct VmaSuballocationItemSizeLess;

typedef VmaList<VmaSuballocation, VmaStlAllocator<VmaSuballocation>> VmaSuballocationList;

struct VmaAllocationRequest;

class VmaBlockMetadata;
class VmaBlockMetadata_Linear;
class VmaBlockMetadata_TLSF;

class VmaBlockVector;

struct VmaPoolListItemTraits;

struct VmaCurrentBudgetData;

class VmaAllocationObjectAllocator;

#endif // _VMA_FORWARD_DECLARATIONS


#ifndef _VMA_FUNCTIONS

/*
Returns number of bits set to 1 in (v).

On specific platforms and compilers you can use intrinsics like:

Visual Studio:
    return __popcnt(v);
GCC, Clang:
    return static_cast<uint32_t>(__builtin_popcount(v));

Define macro VMA_COUNT_BITS_SET to provide your optimized implementation.
But you need to check in runtime whether user's CPU supports these, as some old processors don't.
*/
static inline uint32_t VmaCountBitsSet(uint32_t v)
{
#if VMA_CPP20
    return std::popcount(v);
#else
    uint32_t c = v - ((v >> 1) & 0x55555555);
    c = ((c >> 2) & 0x33333333) + (c & 0x33333333);
    c = ((c >> 4) + c) & 0x0F0F0F0F;
    c = ((c >> 8) + c) & 0x00FF00FF;
    c = ((c >> 16) + c) & 0x0000FFFF;
    return c;
#endif
}

static inline uint8_t VmaBitScanLSB(uint64_t mask)
{
#if defined(_MSC_VER) && defined(_WIN64)
    unsigned long pos;
    if (_BitScanForward64(&pos, mask))
        return static_cast<uint8_t>(pos);
    return UINT8_MAX;
#elif VMA_CPP20
    if(mask != 0)
        return static_cast<uint8_t>(std::countr_zero(mask));
    return UINT8_MAX;
#elif defined __GNUC__ || defined __clang__
    return static_cast<uint8_t>(__builtin_ffsll(mask)) - 1U;
#else
    uint8_t pos = 0;
    uint64_t bit = 1;
    do
    {
        if (mask & bit)
            return pos;
        bit <<= 1;
    } while (pos++ < 63);
    return UINT8_MAX;
#endif
}

static inline uint8_t VmaBitScanLSB(uint32_t mask)
{
#ifdef _MSC_VER
    unsigned long pos;
    if (_BitScanForward(&pos, mask))
        return static_cast<uint8_t>(pos);
    return UINT8_MAX;
#elif VMA_CPP20
    if(mask != 0)
        return static_cast<uint8_t>(std::countr_zero(mask));
    return UINT8_MAX;
#elif defined __GNUC__ || defined __clang__
    return static_cast<uint8_t>(__builtin_ffs(mask)) - 1U;
#else
    uint8_t pos = 0;
    uint32_t bit = 1;
    do
    {
        if (mask & bit)
            return pos;
        bit <<= 1;
    } while (pos++ < 31);
    return UINT8_MAX;
#endif
}

static inline uint8_t VmaBitScanMSB(uint64_t mask)
{
#if defined(_MSC_VER) && defined(_WIN64)
    unsigned long pos;
    if (_BitScanReverse64(&pos, mask))
        return static_cast<uint8_t>(pos);
#elif VMA_CPP20
    if(mask != 0)
        return 63 - static_cast<uint8_t>(std::countl_zero(mask));
#elif defined __GNUC__ || defined __clang__
    if (mask != 0)
        return 63 - static_cast<uint8_t>(__builtin_clzll(mask));
#else
    uint8_t pos = 63;
    uint64_t bit = 1ULL << 63;
    do
    {
        if (mask & bit)
            return pos;
        bit >>= 1;
    } while (pos-- > 0);
#endif
    return UINT8_MAX;
}

static inline uint8_t VmaBitScanMSB(uint32_t mask)
{
#ifdef _MSC_VER
    unsigned long pos;
    if (_BitScanReverse(&pos, mask))
        return static_cast<uint8_t>(pos);
#elif VMA_CPP20
    if(mask != 0)
        return 31 - static_cast<uint8_t>(std::countl_zero(mask));
#elif defined __GNUC__ || defined __clang__
    if (mask != 0)
        return 31 - static_cast<uint8_t>(__builtin_clz(mask));
#else
    uint8_t pos = 31;
    uint32_t bit = 1UL << 31;
    do
    {
        if (mask & bit)
            return pos;
        bit >>= 1;
    } while (pos-- > 0);
#endif
    return UINT8_MAX;
}

/*
Returns true if given number is a power of two.
T must be unsigned integer number or signed integer but always nonnegative.
For 0 returns true.
*/
template <typename T>
inline bool VmaIsPow2(T x)
{
    return (x & (x - 1)) == 0;
}

// Aligns given value up to nearest multiply of align value. For example: VmaAlignUp(11, 8) = 16.
// Use types like uint32_t, uint64_t as T.
template <typename T>
static inline T VmaAlignUp(T val, T alignment)
{
    VMA_HEAVY_ASSERT(VmaIsPow2(alignment));
    return (val + alignment - 1) & ~(alignment - 1);
}

// Aligns given value down to nearest multiply of align value. For example: VmaAlignDown(11, 8) = 8.
// Use types like uint32_t, uint64_t as T.
template <typename T>
static inline T VmaAlignDown(T val, T alignment)
{
    VMA_HEAVY_ASSERT(VmaIsPow2(alignment));
    return val & ~(alignment - 1);
}

// Division with mathematical rounding to nearest number.
template <typename T>
static inline T VmaRoundDiv(T x, T y)
{
    return (x + (y / (T)2)) / y;
}

// Divide by 'y' and round up to nearest integer.
template <typename T>
static inline T VmaDivideRoundingUp(T x, T y)
{
    return (x + y - (T)1) / y;
}

// Returns smallest power of 2 greater or equal to v.
static inline uint32_t VmaNextPow2(uint32_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

static inline uint64_t VmaNextPow2(uint64_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v++;
    return v;
}

// Returns largest power of 2 less or equal to v.
static inline uint32_t VmaPrevPow2(uint32_t v)
{
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v = v ^ (v >> 1);
    return v;
}

static inline uint64_t VmaPrevPow2(uint64_t v)
{
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v = v ^ (v >> 1);
    return v;
}

static inline bool VmaStrIsEmpty(const char* VMA_NULLABLE pStr)
{
    return pStr == VMA_NULL || *pStr == '\0';
}

/*
Returns true if two memory blocks occupy overlapping pages.
ResourceA must be in less memory offset than ResourceB.

Algorithm is based on "Vulkan 1.0.39 - A Specification (with all registered Vulkan extensions)"
chapter 11.6 "Resource Memory Association", paragraph "Buffer-Image Granularity".
*/
static inline bool VmaBlocksOnSamePage(
    VkDeviceSize resourceAOffset,
    VkDeviceSize resourceASize,
    VkDeviceSize resourceBOffset,
    VkDeviceSize pageSize)
{
    VMA_ASSERT(resourceAOffset + resourceASize <= resourceBOffset && resourceASize > 0 && pageSize > 0);
    VkDeviceSize resourceAEnd = resourceAOffset + resourceASize - 1;
    VkDeviceSize resourceAEndPage = resourceAEnd & ~(pageSize - 1);
    VkDeviceSize resourceBStart = resourceBOffset;
    VkDeviceSize resourceBStartPage = resourceBStart & ~(pageSize - 1);
    return resourceAEndPage == resourceBStartPage;
}

/*
Returns true if given suballocation types could conflict and must respect
VkPhysicalDeviceLimits::bufferImageGranularity. They conflict if one is buffer
or linear image and another one is optimal image. If type is unknown, behave
conservatively.
*/
static inline bool VmaIsBufferImageGranularityConflict(
    VmaSuballocationType suballocType1,
    VmaSuballocationType suballocType2)
{
    if (suballocType1 > suballocType2)
    {
        std::swap(suballocType1, suballocType2);
    }

    switch (suballocType1)
    {
    case VMA_SUBALLOCATION_TYPE_FREE:
        return false;
    case VMA_SUBALLOCATION_TYPE_UNKNOWN:
        return true;
    case VMA_SUBALLOCATION_TYPE_BUFFER:
        return
            suballocType2 == VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN ||
            suballocType2 == VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL;
    case VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN:
        return
            suballocType2 == VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN ||
            suballocType2 == VMA_SUBALLOCATION_TYPE_IMAGE_LINEAR ||
            suballocType2 == VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL;
    case VMA_SUBALLOCATION_TYPE_IMAGE_LINEAR:
        return
            suballocType2 == VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL;
    case VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL:
        return false;
    default:
        VMA_ASSERT(0);
        return true;
    }
}

static void VmaWriteMagicValue(void* VMA_NOT_NULL pData, VkDeviceSize offset)
{
#if VMA_DEBUG_MARGIN > 0 && VMA_DEBUG_DETECT_CORRUPTION
    uint32_t* pDst = (uint32_t*)((char*)pData + offset);
    const size_t numberCount = VMA_DEBUG_MARGIN / sizeof(uint32_t);
    for (size_t i = 0; i < numberCount; ++i, ++pDst)
    {
        *pDst = VMA_CORRUPTION_DETECTION_MAGIC_VALUE;
    }
#else
    // no-op
#endif
}

static bool VmaValidateMagicValue(const void* VMA_NOT_NULL pData, VkDeviceSize offset)
{
#if VMA_DEBUG_MARGIN > 0 && VMA_DEBUG_DETECT_CORRUPTION
    const uint32_t* pSrc = (const uint32_t*)((const char*)pData + offset);
    const size_t numberCount = VMA_DEBUG_MARGIN / sizeof(uint32_t);
    for (size_t i = 0; i < numberCount; ++i, ++pSrc)
    {
        if (*pSrc != VMA_CORRUPTION_DETECTION_MAGIC_VALUE)
        {
            return false;
        }
    }
#endif
    return true;
}

/*
Fills structure with parameters of an example buffer to be used for transfers
during GPU memory defragmentation.
*/
static void VmaFillGpuDefragmentationBufferCreateInfo(VkBufferCreateInfo& outBufCreateInfo)
{
    memset(&outBufCreateInfo, 0, sizeof(outBufCreateInfo));
    outBufCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    outBufCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    outBufCreateInfo.size = (VkDeviceSize)VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE; // Example size.
}


/*
Performs binary search and returns iterator to first element that is greater or
equal to (key), according to comparison (cmp).

Cmp should return true if first argument is less than second argument.

Returned value is the found element, if present in the collection or place where
new element with value (key) should be inserted.
*/
template <typename CmpLess, typename IterT, typename KeyT>
static IterT VmaBinaryFindFirstNotLess(IterT beg, IterT end, const KeyT& key, const CmpLess& cmp)
{
    size_t down = 0;
    size_t up = size_t(end - beg);
    while (down < up)
    {
        const size_t mid = down + (up - down) / 2;  // Overflow-safe midpoint calculation
        if (cmp(*(beg + mid), key))
        {
            down = mid + 1;
        }
        else
        {
            up = mid;
        }
    }
    return beg + down;
}

template<typename CmpLess, typename IterT, typename KeyT>
IterT VmaBinaryFindSorted(const IterT& beg, const IterT& end, const KeyT& value, const CmpLess& cmp)
{
    IterT it = VmaBinaryFindFirstNotLess<CmpLess, IterT, KeyT>(
        beg, end, value, cmp);
    if (it == end ||
        (!cmp(*it, value) && !cmp(value, *it)))
    {
        return it;
    }
    return end;
}

/*
Returns true if all pointers in the array are not-null and unique.
Warning! O(n^2) complexity. Use only inside VMA_HEAVY_ASSERT.
T must be pointer type, e.g. VmaAllocation, VmaPool.
*/
template<typename T>
static bool VmaValidatePointerArray(uint32_t count, const T* VMA_NULLABLE arr)
{
    for (uint32_t i = 0; i < count; ++i)
    {
        const T iPtr = arr[i];
        if (iPtr == VMA_NULL)
        {
            return false;
        }
        for (uint32_t j = i + 1; j < count; ++j)
        {
            if (iPtr == arr[j])
            {
                return false;
            }
        }
    }
    return true;
}

template<typename MainT, typename NewT>
static inline void VmaPnextChainPushFront(MainT* VMA_NOT_NULL mainStruct, NewT* VMA_NOT_NULL newStruct)
{
    newStruct->pNext = mainStruct->pNext;
    mainStruct->pNext = newStruct;
}
// Finds structure with s->sType == sType in mainStruct->pNext chain.
// Returns pointer to it. If not found, returns null.
template<typename FindT, typename MainT>
static inline const FindT* VMA_NULLABLE VmaPnextChainFind(const MainT* VMA_NOT_NULL mainStruct, VkStructureType sType)
{
    for(const VkBaseInStructure* s = (const VkBaseInStructure*)mainStruct->pNext;
        s != VMA_NULL; s = s->pNext)
    {
        if(s->sType == sType)
        {
            return (const FindT*)s;
        }
    }
    return VMA_NULL;
}

// An abstraction over buffer or image `usage` flags, depending on available extensions.
struct VmaBufferImageUsage
{
#if VMA_KHR_MAINTENANCE5
    typedef uint64_t BaseType; // VkFlags64
#else
    typedef uint32_t BaseType; // VkFlags32
#endif

    static const VmaBufferImageUsage UNKNOWN;

    BaseType Value;

    VmaBufferImageUsage() { *this = UNKNOWN; }
    explicit VmaBufferImageUsage(BaseType usage) : Value(usage) { }
    VmaBufferImageUsage(const VkBufferCreateInfo &createInfo, bool useKhrMaintenance5);
    explicit VmaBufferImageUsage(const VkImageCreateInfo &createInfo);

    bool operator==(const VmaBufferImageUsage& rhs) const { return Value == rhs.Value; }
    bool operator!=(const VmaBufferImageUsage& rhs) const { return Value != rhs.Value; }

    bool Contains(BaseType flag) const { return (Value & flag) != 0; }
    bool contains_device_access() const
    {
        // This relies on values of VK_IMAGE_USAGE_TRANSFER* being the same as VK_BUFFER_IMAGE_TRANSFER*.
        return (Value & ~BaseType(VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) != 0;
    }
};

const VmaBufferImageUsage VmaBufferImageUsage::UNKNOWN = VmaBufferImageUsage(0);

VmaBufferImageUsage::VmaBufferImageUsage(const VkBufferCreateInfo &createInfo,
    bool useKhrMaintenance5)
{
#if VMA_KHR_MAINTENANCE5
    if(useKhrMaintenance5)
    {
        // If VkBufferCreateInfo::pNext chain contains VkBufferUsageFlags2CreateInfoKHR,
        // take usage from it and ignore VkBufferCreateInfo::usage, per specification
        // of the VK_KHR_maintenance5 extension.
        const VkBufferUsageFlags2CreateInfoKHR* const usageFlags2 =
            VmaPnextChainFind<VkBufferUsageFlags2CreateInfoKHR>(&createInfo, VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR);
        if(usageFlags2 != VMA_NULL)
        {
            this->Value = usageFlags2->usage;
            return;
        }
    }
#endif

    this->Value = (BaseType)createInfo.usage;
}

VmaBufferImageUsage::VmaBufferImageUsage(const VkImageCreateInfo &createInfo)
    : Value((BaseType)createInfo.usage)
{
    // Maybe in the future there will be VK_KHR_maintenanceN extension with structure
    // VkImageUsageFlags2CreateInfoKHR, like the one for buffers...
}

// This is the main algorithm that guides the selection of a memory type best for an allocation -
// converts usage to required/preferred/not preferred flags.
static bool FindMemoryPreferences(
    bool isIntegratedGPU,
    const VmaAllocationCreateInfo& allocCreateInfo,
    VmaBufferImageUsage bufImgUsage,
    VkMemoryPropertyFlags& outRequiredFlags,
    VkMemoryPropertyFlags& outPreferredFlags,
    VkMemoryPropertyFlags& outNotPreferredFlags)
{
    outRequiredFlags = allocCreateInfo.requiredFlags;
    outPreferredFlags = allocCreateInfo.preferredFlags;
    outNotPreferredFlags = 0;

    switch(allocCreateInfo.usage)
    {
    case VMA_MEMORY_USAGE_UNKNOWN:
        break;
    case VMA_MEMORY_USAGE_GPU_ONLY:
        if(!isIntegratedGPU || (outPreferredFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0)
        {
            outPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        }
        break;
    case VMA_MEMORY_USAGE_CPU_ONLY:
        outRequiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        break;
    case VMA_MEMORY_USAGE_CPU_TO_GPU:
        outRequiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        if(!isIntegratedGPU || (outPreferredFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0)
        {
            outPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        }
        break;
    case VMA_MEMORY_USAGE_GPU_TO_CPU:
        outRequiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        outPreferredFlags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        break;
    case VMA_MEMORY_USAGE_CPU_COPY:
        outNotPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;
    case VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED:
        outRequiredFlags |= VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
        break;
    case VMA_MEMORY_USAGE_AUTO:
    case VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE:
    case VMA_MEMORY_USAGE_AUTO_PREFER_HOST:
    {
        if(bufImgUsage == VmaBufferImageUsage::UNKNOWN)
        {
            VMA_ASSERT(0 && "VMA_MEMORY_USAGE_AUTO* values can only be used with functions like vmaCreateBuffer, vmaCreateImage so that the details of the created resource are known."
                " Maybe you use VkBufferUsageFlags2CreateInfoKHR but forgot to use VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT?" );
            return false;
        }

        const bool deviceAccess = bufImgUsage.contains_device_access();
        const bool hostAccessSequentialWrite = (allocCreateInfo.flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT) != 0;
        const bool hostAccessRandom = (allocCreateInfo.flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT) != 0;
        const bool hostAccessAllowTransferInstead = (allocCreateInfo.flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT) != 0;
        const bool preferDevice = allocCreateInfo.usage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        const bool preferHost = allocCreateInfo.usage == VMA_MEMORY_USAGE_AUTO_PREFER_HOST;

        // CPU random access - e.g. a buffer written to or transferred from GPU to read back on CPU.
        if(hostAccessRandom)
        {
            // Prefer cached. Cannot require it, because some platforms don't have it (e.g. Raspberry Pi - see #362)!
            outPreferredFlags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

            if (!isIntegratedGPU && deviceAccess && hostAccessAllowTransferInstead && !preferHost)
            {
                // Nice if it will end up in HOST_VISIBLE, but more importantly prefer DEVICE_LOCAL.
                // Omitting HOST_VISIBLE here is intentional.
                // In case there is DEVICE_LOCAL | HOST_VISIBLE | HOST_CACHED, it will pick that one.
                // Otherwise, this will give same weight to DEVICE_LOCAL as HOST_VISIBLE | HOST_CACHED and select the former if occurs first on the list.
                outPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            }
            else
            {
                // Always CPU memory.
                outRequiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            }
        }
        // CPU sequential write - may be CPU or host-visible GPU memory, uncached and write-combined.
        else if(hostAccessSequentialWrite)
        {
            // Want uncached and write-combined.
            outNotPreferredFlags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

            if(!isIntegratedGPU && deviceAccess && hostAccessAllowTransferInstead && !preferHost)
            {
                outPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            }
            else
            {
                outRequiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
                // Direct GPU access, CPU sequential write (e.g. a dynamic uniform buffer updated every frame)
                if(deviceAccess)
                {
                    // Could go to CPU memory or GPU BAR/unified. Up to the user to decide. If no preference, choose GPU memory.
                    if(preferHost)
                        outNotPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                    else
                        outPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                }
                // GPU no direct access, CPU sequential write (e.g. an upload buffer to be transferred to the GPU)
                else
                {
                    // Could go to CPU memory or GPU BAR/unified. Up to the user to decide. If no preference, choose CPU memory.
                    if(preferDevice)
                        outPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                    else
                        outNotPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                }
            }
        }
        // No CPU access
        else
        {
            // if(deviceAccess)
            //
            // GPU access, no CPU access (e.g. a color attachment image) - prefer GPU memory,
            // unless there is a clear preference from the user not to do so.
            //
            // else:
            //
            // No direct GPU access, no CPU access, just transfers.
            // It may be staging copy intended for e.g. preserving image for next frame (then better GPU memory) or
            // a "swap file" copy to free some GPU memory (then better CPU memory).
            // Up to the user to decide. If no preferece, assume the former and choose GPU memory.

            if(preferHost)
                outNotPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            else
                outPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        }
        break;
    }
    default:
        VMA_ASSERT(0);
    }

    // Avoid DEVICE_COHERENT unless explicitly requested.
    if(((allocCreateInfo.requiredFlags | allocCreateInfo.preferredFlags) &
        (VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD_COPY | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD_COPY)) == 0)
    {
        outNotPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD_COPY;
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Memory allocation

static void* VMA_NULLABLE VmaMalloc(const VkAllocationCallbacks* VMA_NULLABLE pAllocationCallbacks, size_t size, size_t alignment)
{
    void* result = VMA_NULL;
    if ((pAllocationCallbacks != VMA_NULL) &&
        (pAllocationCallbacks->pfnAllocation != VMA_NULL))
    {
        result = (*pAllocationCallbacks->pfnAllocation)(
            pAllocationCallbacks->pUserData,
            size,
            alignment,
            VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
    }
    else
    {
        result = VMA_SYSTEM_ALIGNED_MALLOC(size, alignment);
    }
    VMA_ASSERT(result != VMA_NULL && "CPU memory allocation failed.");
    return result;
}

static void VmaFree(const VkAllocationCallbacks* VMA_NULLABLE pAllocationCallbacks, void* VMA_NULLABLE ptr)
{
    if ((pAllocationCallbacks != VMA_NULL) &&
        (pAllocationCallbacks->pfnFree != VMA_NULL))
    {
        (*pAllocationCallbacks->pfnFree)(pAllocationCallbacks->pUserData, ptr);
    }
    else
    {
        VMA_SYSTEM_ALIGNED_FREE(ptr);
    }
}

template<typename T>
static T* VMA_NULLABLE VmaAllocate(const VkAllocationCallbacks* VMA_NULLABLE pAllocationCallbacks)
{
    return (T*)VmaMalloc(pAllocationCallbacks, sizeof(T), VMA_ALIGN_OF(T));
}

template<typename T>
static T* VMA_NULLABLE VmaAllocateArray(const VkAllocationCallbacks* VMA_NULLABLE pAllocationCallbacks, size_t count)
{
    return (T*)VmaMalloc(pAllocationCallbacks, sizeof(T) * count, VMA_ALIGN_OF(T));
}

#define vma_new(allocator, type)   new(VmaAllocate<type>(allocator))(type)

#define vma_new_array(allocator, type, count)   new(VmaAllocateArray<type>((allocator), (count)))(type)

template<typename T>
static void vma_delete(const VkAllocationCallbacks* VMA_NULLABLE pAllocationCallbacks, T* VMA_NULLABLE ptr)
{
    ptr->~T();
    VmaFree(pAllocationCallbacks, ptr);
}

template<typename T>
static void vma_delete_array(const VkAllocationCallbacks* VMA_NULLABLE pAllocationCallbacks, T* VMA_NULLABLE ptr, size_t count)
{
    if (ptr != VMA_NULL)
    {
        for (size_t i = count; i--; )
        {
            ptr[i].~T();
        }
        VmaFree(pAllocationCallbacks, ptr);
    }
}

static char* VMA_NULLABLE VmaCreateStringCopy(const VkAllocationCallbacks* VMA_NULLABLE allocs, const char* VMA_NULLABLE srcStr)
{
    if (srcStr != VMA_NULL)
    {
        const size_t len = strlen(srcStr);
        char* const result = vma_new_array(allocs, char, len + 1);
        memcpy(result, srcStr, len + 1);
        return result;
    }
    return VMA_NULL;
}

#if VMA_STATS_STRING_ENABLED
static char* VMA_NULLABLE VmaCreateStringCopy(const VkAllocationCallbacks* VMA_NULLABLE allocs, const char* VMA_NULLABLE srcStr, size_t strLen)
{
    if (srcStr != VMA_NULL)
    {
        char* const result = vma_new_array(allocs, char, strLen + 1);
        memcpy(result, srcStr, strLen);
        result[strLen] = '\0';
        return result;
    }
    return VMA_NULL;
}
#endif // VMA_STATS_STRING_ENABLED

static void VmaFreeString(const VkAllocationCallbacks* VMA_NULLABLE allocs, char* VMA_NULLABLE str)
{
    if (str != VMA_NULL)
    {
        const size_t len = strlen(str);
        vma_delete_array(allocs, str, len + 1);
    }
}

template<typename CmpLess, typename VectorT>
size_t VmaVectorInsertSorted(VectorT& vector, const typename VectorT::value_type& value)
{
    const size_t indexToInsert = VmaBinaryFindFirstNotLess(
        vector.data(),
        vector.data() + vector.size(),
        value,
        CmpLess()) - vector.data();
    VmaVectorInsert(vector, indexToInsert, value);
    return indexToInsert;
}

template<typename CmpLess, typename VectorT>
bool VmaVectorRemoveSorted(VectorT& vector, const typename VectorT::value_type& value)
{
    CmpLess comparator;
    typename VectorT::iterator it = VmaBinaryFindFirstNotLess(
        vector.begin(),
        vector.end(),
        value,
        comparator);
    if ((it != vector.end()) && !comparator(*it, value) && !comparator(value, *it))
    {
        size_t indexToRemove = it - vector.begin();
        VmaVectorRemove(vector, indexToRemove);
        return true;
    }
    return false;
}
#endif // _VMA_FUNCTIONS

#ifndef _VMA_STATISTICS_FUNCTIONS

static void VmaClearStatistics(VmaStatistics& outStats)
{
    outStats.blockCount = 0;
    outStats.allocationCount = 0;
    outStats.blockBytes = 0;
    outStats.allocationBytes = 0;
}

static void VmaAddStatistics(VmaStatistics& inoutStats, const VmaStatistics& src)
{
    inoutStats.blockCount += src.blockCount;
    inoutStats.allocationCount += src.allocationCount;
    inoutStats.blockBytes += src.blockBytes;
    inoutStats.allocationBytes += src.allocationBytes;
}

static void VmaClearDetailedStatistics(VmaDetailedStatistics& outStats)
{
    VmaClearStatistics(outStats.statistics);
    outStats.unusedRangeCount = 0;
    outStats.allocationSizeMin = VK_WHOLE_SIZE;
    outStats.allocationSizeMax = 0;
    outStats.unusedRangeSizeMin = VK_WHOLE_SIZE;
    outStats.unusedRangeSizeMax = 0;
}

static void VmaAddDetailedStatisticsAllocation(VmaDetailedStatistics& inoutStats, VkDeviceSize size)
{
    inoutStats.statistics.allocationCount++;
    inoutStats.statistics.allocationBytes += size;
    inoutStats.allocationSizeMin = VMA_MIN(inoutStats.allocationSizeMin, size);
    inoutStats.allocationSizeMax = VMA_MAX(inoutStats.allocationSizeMax, size);
}

static void VmaAddDetailedStatisticsUnusedRange(VmaDetailedStatistics& inoutStats, VkDeviceSize size)
{
    inoutStats.unusedRangeCount++;
    inoutStats.unusedRangeSizeMin = VMA_MIN(inoutStats.unusedRangeSizeMin, size);
    inoutStats.unusedRangeSizeMax = VMA_MAX(inoutStats.unusedRangeSizeMax, size);
}

static void VmaAddDetailedStatistics(VmaDetailedStatistics& inoutStats, const VmaDetailedStatistics& src)
{
    VmaAddStatistics(inoutStats.statistics, src.statistics);
    inoutStats.unusedRangeCount += src.unusedRangeCount;
    inoutStats.allocationSizeMin = VMA_MIN(inoutStats.allocationSizeMin, src.allocationSizeMin);
    inoutStats.allocationSizeMax = VMA_MAX(inoutStats.allocationSizeMax, src.allocationSizeMax);
    inoutStats.unusedRangeSizeMin = VMA_MIN(inoutStats.unusedRangeSizeMin, src.unusedRangeSizeMin);
    inoutStats.unusedRangeSizeMax = VMA_MAX(inoutStats.unusedRangeSizeMax, src.unusedRangeSizeMax);
}

#endif // _VMA_STATISTICS_FUNCTIONS

#ifndef _VMA_MUTEX_LOCK
// Helper RAII class to lock a mutex in constructor and unlock it in destructor (at the end of scope).
struct VmaMutexLock
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaMutexLock)
public:
    explicit VmaMutexLock(VMA_MUTEX& mutex, bool useMutex = true) :
        _p_mutex(useMutex ? &mutex : VMA_NULL)
    {
        if (_p_mutex) { _p_mutex->lock(); }
    }
    ~VmaMutexLock() {  if (_p_mutex) { _p_mutex->unlock(); } }

private:
    VMA_MUTEX* _Nullable _p_mutex;
};

// Helper RAII class to lock a RW mutex in constructor and unlock it in destructor (at the end of scope), for reading.
struct VmaMutexLockRead
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaMutexLockRead)
public:
    VmaMutexLockRead(VMA_RW_MUTEX& mutex, bool useMutex) :
        _p_mutex(useMutex ? &mutex : VMA_NULL)
    {
        if (_p_mutex) { _p_mutex->lock_read(); }
    }
    ~VmaMutexLockRead() { if (_p_mutex) { _p_mutex->unlock_read(); } }

private:
    VMA_RW_MUTEX* _Nullable _p_mutex;
};

// Helper RAII class to lock a RW mutex in constructor and unlock it in destructor (at the end of scope), for writing.
struct VmaMutexLockWrite
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaMutexLockWrite)
public:
    VmaMutexLockWrite(VMA_RW_MUTEX& mutex, bool useMutex)
        : _p_mutex(useMutex ? &mutex : VMA_NULL)
    {
        if (_p_mutex) { _p_mutex->lock_write(); }
    }
    ~VmaMutexLockWrite() { if (_p_mutex) { _p_mutex->unlock_write(); } }

private:
    VMA_RW_MUTEX* _Nullable _p_mutex;
};

#if VMA_DEBUG_GLOBAL_MUTEX
    static VMA_MUTEX gDebugGlobalMutex;
    #define VMA_DEBUG_GLOBAL_MUTEX_LOCK VmaMutexLock debugGlobalMutexLock(gDebugGlobalMutex, true);
#else
    #define VMA_DEBUG_GLOBAL_MUTEX_LOCK
#endif
#endif // _VMA_MUTEX_LOCK

#ifndef _VMA_ATOMIC_TRANSACTIONAL_INCREMENT
// An object that increments given atomic but decrements it back in the destructor unless commit() is called.
template<typename AtomicT>
struct AtomicTransactionalIncrement
{
public:
    using T = decltype(AtomicT().load());

    ~AtomicTransactionalIncrement()
    {
        if(_atomic)
            --(*_atomic);
    }

    void commit() { _atomic = VMA_NULL; }
    T Increment(AtomicT* VMA_NOT_NULL atomic)
    {
        _atomic = atomic;
        return _atomic->fetch_add(1);
    }

private:
    AtomicT* _Nullable _atomic = VMA_NULL;
};
#endif // _VMA_ATOMIC_TRANSACTIONAL_INCREMENT

#ifndef _VMA_STL_ALLOCATOR
// STL-compatible allocator.
template<typename T>
struct VmaStlAllocator
{
    const VkAllocationCallbacks* _Nullable const _p_callbacks;
    typedef T value_type;

    explicit VmaStlAllocator(const VkAllocationCallbacks* VMA_NULLABLE pCallbacks) : _p_callbacks(pCallbacks) {}
    template<typename U>
    explicit VmaStlAllocator(const VmaStlAllocator<U>& src) : _p_callbacks(src._p_callbacks) {}
    VmaStlAllocator(const VmaStlAllocator&) = default;
    VmaStlAllocator& operator=(const VmaStlAllocator&) = delete;

    T* VMA_NULLABLE allocate(size_t n) { return VmaAllocateArray<T>(_p_callbacks, n); }
    void deallocate(T* VMA_NULLABLE p, size_t n) { VmaFree(_p_callbacks, p); }

    template<typename U>
    bool operator==(const VmaStlAllocator<U>& rhs) const
    {
        return _p_callbacks == rhs._p_callbacks;
    }
    template<typename U>
    bool operator!=(const VmaStlAllocator<U>& rhs) const
    {
        return _p_callbacks != rhs._p_callbacks;
    }
};
#endif // _VMA_STL_ALLOCATOR

#ifndef _VMA_VECTOR
/* Class with interface compatible with subset of std::vector.
T must be POD because constructors and destructors are not called and memcpy is
used for these objects. */
template<typename T, typename AllocatorT>
class VmaVector
{
public:
    typedef T value_type;
    typedef T* _Nullable iterator;
    typedef const T* _Nullable const_iterator;

    explicit VmaVector(const AllocatorT& allocator);
    VmaVector(size_t count, const AllocatorT& allocator);
    // This version of the constructor is here for compatibility with pre-C++14 std::vector.
    // value is unused.
    VmaVector(size_t count, const T& value, const AllocatorT& allocator) : VmaVector(count, allocator) {}
    VmaVector(const VmaVector<T, AllocatorT>& src);
    VmaVector& operator=(const VmaVector& rhs);
    ~VmaVector() { VmaFree(_allocator._p_callbacks, _p_array); }

    bool empty() const { return _count == 0; }
    size_t size() const { return _count; }
    T* _Nullable data() { return _p_array; }
    T& front() { VMA_HEAVY_ASSERT(_count > 0); return _p_array[0]; }
    T& back() { VMA_HEAVY_ASSERT(_count > 0); return _p_array[_count - 1]; }
    const T* _Nullable data() const { return _p_array; }
    const T& front() const { VMA_HEAVY_ASSERT(_count > 0); return _p_array[0]; }
    const T& back() const { VMA_HEAVY_ASSERT(_count > 0); return _p_array[_count - 1]; }

    iterator begin() { return _p_array; }
    iterator end() { return _p_array + _count; }
    const_iterator cbegin() const { return _p_array; }
    const_iterator cend() const { return _p_array + _count; }
    const_iterator begin() const { return cbegin(); }
    const_iterator end() const { return cend(); }

    void pop_front() { VMA_HEAVY_ASSERT(_count > 0); remove(0); }
    void pop_back() { VMA_HEAVY_ASSERT(_count > 0); resize(size() - 1); }
    void push_front(const T& src) { insert(0, src); }

    void push_back(const T& src);
    void reserve(size_t newCapacity, bool freeMemory = false);
    void resize(size_t newCount);
    void clear() { resize(0); }
    void shrink_to_fit();
    void insert(size_t index, const T& src);
    void remove(size_t index);

    T& operator[](size_t index) { VMA_HEAVY_ASSERT(index < _count); return _p_array[index]; }
    const T& operator[](size_t index) const { VMA_HEAVY_ASSERT(index < _count); return _p_array[index]; }

private:
    AllocatorT _allocator;
    T* _Nullable _p_array;
    size_t _count;
    size_t _capacity;
};

#ifndef _VMA_VECTOR_FUNCTIONS
template<typename T, typename AllocatorT>
VmaVector<T, AllocatorT>::VmaVector(const AllocatorT& allocator)
    : _allocator(allocator),
    _p_array(VMA_NULL),
    _count(0),
    _capacity(0) {}

template<typename T, typename AllocatorT>
VmaVector<T, AllocatorT>::VmaVector(size_t count, const AllocatorT& allocator)
    : _allocator(allocator),
    _p_array(count ? (T*)VmaAllocateArray<T>(allocator._p_callbacks, count) : VMA_NULL),
    _count(count),
    _capacity(count) {}

template<typename T, typename AllocatorT>
VmaVector<T, AllocatorT>::VmaVector(const VmaVector& src)
    : _allocator(src._allocator),
    _p_array(src._count ? (T*)VmaAllocateArray<T>(src._allocator._p_callbacks, src._count) : VMA_NULL),
    _count(src._count),
    _capacity(src._count)
{
    if (_count != 0)
    {
        memcpy(_p_array, src._p_array, _count * sizeof(T));
    }
}

template<typename T, typename AllocatorT>
VmaVector<T, AllocatorT>& VmaVector<T, AllocatorT>::operator=(const VmaVector& rhs)
{
    if (&rhs != this)
    {
        resize(rhs._count);
        if (_count != 0)
        {
            memcpy(_p_array, rhs._p_array, _count * sizeof(T));
        }
    }
    return *this;
}

template<typename T, typename AllocatorT>
void VmaVector<T, AllocatorT>::push_back(const T& src)
{
    const size_t newIndex = size();
    resize(newIndex + 1);
    _p_array[newIndex] = src;
}

template<typename T, typename AllocatorT>
void VmaVector<T, AllocatorT>::reserve(size_t newCapacity, bool freeMemory)
{
    newCapacity = VMA_MAX(newCapacity, _count);

    if ((newCapacity < _capacity) && !freeMemory)
    {
        newCapacity = _capacity;
    }

    if (newCapacity != _capacity)
    {
        T* const newArray = newCapacity ? VmaAllocateArray<T>(_allocator, newCapacity) : VMA_NULL;
        if (_count != 0)
        {
            memcpy(newArray, _p_array, _count * sizeof(T));
        }
        VmaFree(_allocator._p_callbacks, _p_array);
        _capacity = newCapacity;
        _p_array = newArray;
    }
}

template<typename T, typename AllocatorT>
void VmaVector<T, AllocatorT>::resize(size_t newCount)
{
    size_t newCapacity = _capacity;
    if (newCount > _capacity)
    {
        newCapacity = VMA_MAX(newCount, VMA_MAX(_capacity * 3 / 2, (size_t)8));
    }

    if (newCapacity != _capacity)
    {
        T* const newArray = newCapacity ? VmaAllocateArray<T>(_allocator._p_callbacks, newCapacity) : VMA_NULL;
        const size_t elementsToCopy = VMA_MIN(_count, newCount);
        if (elementsToCopy != 0)
        {
            memcpy(newArray, _p_array, elementsToCopy * sizeof(T));
        }
        VmaFree(_allocator._p_callbacks, _p_array);
        _capacity = newCapacity;
        _p_array = newArray;
    }

    _count = newCount;
}

template<typename T, typename AllocatorT>
void VmaVector<T, AllocatorT>::shrink_to_fit()
{
    if (_capacity > _count)
    {
        T* newArray = VMA_NULL;
        if (_count > 0)
        {
            newArray = VmaAllocateArray<T>(_allocator._p_callbacks, _count);
            memcpy(newArray, _p_array, _count * sizeof(T));
        }
        VmaFree(_allocator._p_callbacks, _p_array);
        _capacity = _count;
        _p_array = newArray;
    }
}

template<typename T, typename AllocatorT>
void VmaVector<T, AllocatorT>::insert(size_t index, const T& src)
{
    VMA_HEAVY_ASSERT(index <= _count);
    const size_t oldCount = size();
    resize(oldCount + 1);
    if (index < oldCount)
    {
        memmove(_p_array + (index + 1), _p_array + index, (oldCount - index) * sizeof(T));
    }
    _p_array[index] = src;
}

template<typename T, typename AllocatorT>
void VmaVector<T, AllocatorT>::remove(size_t index)
{
    VMA_HEAVY_ASSERT(index < _count);
    const size_t oldCount = size();
    if (index < oldCount - 1)
    {
        memmove(_p_array + index, _p_array + (index + 1), (oldCount - index - 1) * sizeof(T));
    }
    resize(oldCount - 1);
}
#endif // _VMA_VECTOR_FUNCTIONS

template<typename T, typename allocatorT>
static void VmaVectorInsert(VmaVector<T, allocatorT>& vec, size_t index, const T& item)
{
    vec.insert(index, item);
}

template<typename T, typename allocatorT>
static void VmaVectorRemove(VmaVector<T, allocatorT>& vec, size_t index)
{
    vec.remove(index);
}
#endif // _VMA_VECTOR

#ifndef _VMA_SMALL_VECTOR
/*
This is a vector (a variable-sized array), optimized for the case when the array is small.

It contains some number of elements in-place, which allows it to avoid heap allocation
when the actual number of elements is below that threshold. This allows normal "small"
cases to be fast without losing generality for large inputs.
*/
template<typename T, typename AllocatorT, size_t N>
class VmaSmallVector
{
public:
    typedef T value_type;
    typedef T* _Nullable iterator;

    explicit VmaSmallVector(const AllocatorT& allocator);
    VmaSmallVector(size_t count, const AllocatorT& allocator);
    template<typename SrcT, typename SrcAllocatorT, size_t SrcN>
    explicit VmaSmallVector(const VmaSmallVector<SrcT, SrcAllocatorT, SrcN>&) = delete;
    template<typename SrcT, typename SrcAllocatorT, size_t SrcN>
    VmaSmallVector<T, AllocatorT, N>& operator=(const VmaSmallVector<SrcT, SrcAllocatorT, SrcN>&) = delete;
    ~VmaSmallVector() = default;

    bool empty() const { return _count == 0; }
    size_t size() const { return _count; }
    T* _Nullable data() { return _count > N ? _dynamic_array.data() : _static_array; }
    T& front() { VMA_HEAVY_ASSERT(_count > 0); return data()[0]; }
    T& back() { VMA_HEAVY_ASSERT(_count > 0); return data()[_count - 1]; }
    const T* _Nullable data() const { return _count > N ? _dynamic_array.data() : _static_array; }
    const T& front() const { VMA_HEAVY_ASSERT(_count > 0); return data()[0]; }
    const T& back() const { VMA_HEAVY_ASSERT(_count > 0); return data()[_count - 1]; }

    iterator begin() { return data(); }
    iterator end() { return data() + _count; }

    void pop_front() { VMA_HEAVY_ASSERT(_count > 0); remove(0); }
    void pop_back() { VMA_HEAVY_ASSERT(_count > 0); resize(size() - 1); }
    void push_front(const T& src) { insert(0, src); }

    void push_back(const T& src);
    void resize(size_t newCount, bool freeMemory = false);
    void clear(bool freeMemory = false);
    void insert(size_t index, const T& src);
    void remove(size_t index);

    T& operator[](size_t index) { VMA_HEAVY_ASSERT(index < _count); return data()[index]; }
    const T& operator[](size_t index) const { VMA_HEAVY_ASSERT(index < _count); return data()[index]; }

private:
    size_t _count;
    T _static_array[N]; // Used when _size <= N
    VmaVector<T, AllocatorT> _dynamic_array; // Used when _size > N
};

#ifndef _VMA_SMALL_VECTOR_FUNCTIONS
template<typename T, typename AllocatorT, size_t N>
VmaSmallVector<T, AllocatorT, N>::VmaSmallVector(const AllocatorT& allocator)
    : _count(0),
    _dynamic_array(allocator) {}

template<typename T, typename AllocatorT, size_t N>
VmaSmallVector<T, AllocatorT, N>::VmaSmallVector(size_t count, const AllocatorT& allocator)
    : _count(count),
    _dynamic_array(count > N ? count : 0, allocator) {}

template<typename T, typename AllocatorT, size_t N>
void VmaSmallVector<T, AllocatorT, N>::push_back(const T& src)
{
    const size_t newIndex = size();
    resize(newIndex + 1);
    data()[newIndex] = src;
}

template<typename T, typename AllocatorT, size_t N>
void VmaSmallVector<T, AllocatorT, N>::resize(size_t newCount, bool freeMemory)
{
    if (newCount > N && _count > N)
    {
        // Any direction, staying in _dynamic_array
        _dynamic_array.resize(newCount);
        if (freeMemory)
        {
            _dynamic_array.shrink_to_fit();
        }
    }
    else if (newCount > N && _count <= N)
    {
        // Growing, moving from _static_array to _dynamic_array
        _dynamic_array.resize(newCount);
        if (_count > 0)
        {
            memcpy(_dynamic_array.data(), _static_array, _count * sizeof(T));
        }
    }
    else if (newCount <= N && _count > N)
    {
        // Shrinking, moving from _dynamic_array to _static_array
        if (newCount > 0)
        {
            memcpy(_static_array, _dynamic_array.data(), newCount * sizeof(T));
        }
        _dynamic_array.resize(0);
        if (freeMemory)
        {
            _dynamic_array.shrink_to_fit();
        }
    }
    else
    {
        // Any direction, staying in _static_array - nothing to do here
    }
    _count = newCount;
}

template<typename T, typename AllocatorT, size_t N>
void VmaSmallVector<T, AllocatorT, N>::clear(bool freeMemory)
{
    _dynamic_array.clear();
    if (freeMemory)
    {
        _dynamic_array.shrink_to_fit();
    }
    _count = 0;
}

template<typename T, typename AllocatorT, size_t N>
void VmaSmallVector<T, AllocatorT, N>::insert(size_t index, const T& src)
{
    VMA_HEAVY_ASSERT(index <= _count);
    const size_t oldCount = size();
    resize(oldCount + 1);
    T* const dataPtr = data();
    if (index < oldCount)
    {
        //  I know, this could be more optimal for case where memmove can be memcpy directly from _static_array to _dynamic_array.
        memmove(dataPtr + (index + 1), dataPtr + index, (oldCount - index) * sizeof(T));
    }
    dataPtr[index] = src;
}

template<typename T, typename AllocatorT, size_t N>
void VmaSmallVector<T, AllocatorT, N>::remove(size_t index)
{
    VMA_HEAVY_ASSERT(index < _count);
    const size_t oldCount = size();
    if (index < oldCount - 1)
    {
        //  I know, this could be more optimal for case where memmove can be memcpy directly from _dynamic_array to _static_array.
        T* const dataPtr = data();
        memmove(dataPtr + index, dataPtr + (index + 1), (oldCount - index - 1) * sizeof(T));
    }
    resize(oldCount - 1);
}
#endif // _VMA_SMALL_VECTOR_FUNCTIONS
#endif // _VMA_SMALL_VECTOR

#ifndef _VMA_POOL_ALLOCATOR
/*
Allocator for objects of type T using a list of arrays (pools) to speed up
allocation. Number of elements that can be allocated is not bounded because
allocator can create multiple blocks.
*/
template<typename T>
class VmaPoolAllocator
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaPoolAllocator)
public:
    VmaPoolAllocator(const VkAllocationCallbacks* pAllocationCallbacks, uint32_t firstBlockCapacity);
    ~VmaPoolAllocator();
    template<typename... Types> T* alloc(Types&&... args);
    void free(T* ptr);

private:
    union Item
    {
        uint32_t NextFreeIndex;
        alignas(T) char Value[sizeof(T)];
    };
    struct ItemBlock
    {
        Item* pItems;
        uint32_t Capacity;
        uint32_t FirstFreeIndex;
    };

    const VkAllocationCallbacks* _p_allocation_callbacks;
    const uint32_t _first_block_capacity;
    VmaVector<ItemBlock, VmaStlAllocator<ItemBlock>> _item_blocks;

    ItemBlock& create_new_block();
};

#ifndef _VMA_POOL_ALLOCATOR_FUNCTIONS
template<typename T>
VmaPoolAllocator<T>::VmaPoolAllocator(const VkAllocationCallbacks* pAllocationCallbacks, uint32_t firstBlockCapacity)
    : _p_allocation_callbacks(pAllocationCallbacks),
    _first_block_capacity(firstBlockCapacity),
    _item_blocks(VmaStlAllocator<ItemBlock>(pAllocationCallbacks))
{
    VMA_ASSERT(_first_block_capacity > 1);
}

template<typename T>
VmaPoolAllocator<T>::~VmaPoolAllocator()
{
    for (size_t i = _item_blocks.size(); i--;)
        vma_delete_array(_p_allocation_callbacks, _item_blocks[i].pItems, _item_blocks[i].Capacity);
    _item_blocks.clear();
}

template<typename T>
template<typename... Types> T* VmaPoolAllocator<T>::alloc(Types&&... args)
{
    for (size_t i = _item_blocks.size(); i--; )
    {
        ItemBlock& block = _item_blocks[i];
        // This block has some free items: Use first one.
        if (block.FirstFreeIndex != UINT32_MAX)
        {
            Item* const pItem = &block.pItems[block.FirstFreeIndex];
            block.FirstFreeIndex = pItem->NextFreeIndex;
            T* result = (T*)&pItem->Value;
            new(result)T(std::forward<Types>(args)...); // Explicit constructor call.
            return result;
        }
    }

    // No block has free item: Create new one and use it.
    ItemBlock& newBlock = create_new_block();
    Item* const pItem = &newBlock.pItems[0];
    newBlock.FirstFreeIndex = pItem->NextFreeIndex;
    T* result = (T*)&pItem->Value;
    new(result) T(std::forward<Types>(args)...); // Explicit constructor call.
    return result;
}

template<typename T>
void VmaPoolAllocator<T>::free(T* ptr)
{
    // Search all memory blocks to find ptr.
    for (size_t i = _item_blocks.size(); i--; )
    {
        ItemBlock& block = _item_blocks[i];

        // Casting to union.
        Item* pItemPtr = VMA_NULL;
        memcpy(&pItemPtr, &ptr, sizeof(pItemPtr));

        // Check if pItemPtr is in address range of this block.
        if ((pItemPtr >= block.pItems) && (pItemPtr < block.pItems + block.Capacity))
        {
            ptr->~T(); // Explicit destructor call.
            const uint32_t index = static_cast<uint32_t>(pItemPtr - block.pItems);
            pItemPtr->NextFreeIndex = block.FirstFreeIndex;
            block.FirstFreeIndex = index;
            return;
        }
    }
    VMA_ASSERT(0 && "Pointer doesn't belong to this memory pool.");
}

template<typename T>
typename VmaPoolAllocator<T>::ItemBlock& VmaPoolAllocator<T>::create_new_block()
{
    const uint32_t newBlockCapacity = _item_blocks.empty() ?
        _first_block_capacity : _item_blocks.back().Capacity * 3 / 2;

    const ItemBlock newBlock =
    {
        vma_new_array(_p_allocation_callbacks, Item, newBlockCapacity),
        newBlockCapacity,
        0
    };

    _item_blocks.push_back(newBlock);

    // Setup singly-linked list of all free items in this block.
    for (uint32_t i = 0; i < newBlockCapacity - 1; ++i)
        newBlock.pItems[i].NextFreeIndex = i + 1;
    newBlock.pItems[newBlockCapacity - 1].NextFreeIndex = UINT32_MAX;
    return _item_blocks.back();
}
#endif // _VMA_POOL_ALLOCATOR_FUNCTIONS
#endif // _VMA_POOL_ALLOCATOR

#ifndef _VMA_RAW_LIST
template<typename T>
struct VmaListItem
{
    VmaListItem* pPrev;
    VmaListItem* pNext;
    T Value;
};

// Doubly linked list.
template<typename T>
class VmaRawList
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaRawList)
public:
    typedef VmaListItem<T> ItemType;

    explicit VmaRawList(const VkAllocationCallbacks* pAllocationCallbacks);
    // Intentionally not calling Clear, because that would be unnecessary
    // computations to return all items to _item_allocator as free.
    ~VmaRawList() = default;

    size_t get_count() const { return _count; }
    bool is_empty() const { return _count == 0; }

    ItemType* front() { return _p_front; }
    ItemType* back() { return _p_back; }
    const ItemType* front() const { return _p_front; }
    const ItemType* back() const { return _p_back; }

    ItemType* push_front();
    ItemType* push_back();
    ItemType* push_front(const T& value);
    ItemType* push_back(const T& value);
    void pop_front();
    void pop_back();

    // Item can be null - it means PushBack.
    ItemType* insert_before(ItemType* pItem);
    // Item can be null - it means PushFront.
    ItemType* insert_after(ItemType* pItem);
    ItemType* insert_before(ItemType* pItem, const T& value);
    ItemType* insert_after(ItemType* pItem, const T& value);

    void clear();
    void remove(ItemType* pItem);

private:
    const VkAllocationCallbacks* const _p_allocation_callbacks;
    VmaPoolAllocator<ItemType> _item_allocator;
    ItemType* _p_front;
    ItemType* _p_back;
    size_t _count;
};

#ifndef _VMA_RAW_LIST_FUNCTIONS
template<typename T>
VmaRawList<T>::VmaRawList(const VkAllocationCallbacks* pAllocationCallbacks)
    : _p_allocation_callbacks(pAllocationCallbacks),
    _item_allocator(pAllocationCallbacks, 128),
    _p_front(VMA_NULL),
    _p_back(VMA_NULL),
    _count(0) {}

template<typename T>
VmaListItem<T>* VmaRawList<T>::push_front()
{
    ItemType* const pNewItem = _item_allocator.alloc();
    pNewItem->pPrev = VMA_NULL;
    if (is_empty())
    {
        pNewItem->pNext = VMA_NULL;
        _p_front = pNewItem;
        _p_back = pNewItem;
        _count = 1;
    }
    else
    {
        pNewItem->pNext = _p_front;
        _p_front->pPrev = pNewItem;
        _p_front = pNewItem;
        ++_count;
    }
    return pNewItem;
}

template<typename T>
VmaListItem<T>* VmaRawList<T>::push_back()
{
    ItemType* const pNewItem = _item_allocator.alloc();
    pNewItem->pNext = VMA_NULL;
    if(is_empty())
    {
        pNewItem->pPrev = VMA_NULL;
        _p_front = pNewItem;
        _p_back = pNewItem;
        _count = 1;
    }
    else
    {
        pNewItem->pPrev = _p_back;
        _p_back->pNext = pNewItem;
        _p_back = pNewItem;
        ++_count;
    }
    return pNewItem;
}

template<typename T>
VmaListItem<T>* VmaRawList<T>::push_front(const T& value)
{
    ItemType* const pNewItem = push_front();
    pNewItem->Value = value;
    return pNewItem;
}

template<typename T>
VmaListItem<T>* VmaRawList<T>::push_back(const T& value)
{
    ItemType* const pNewItem = push_back();
    pNewItem->Value = value;
    return pNewItem;
}

template<typename T>
void VmaRawList<T>::pop_front()
{
    VMA_HEAVY_ASSERT(_count > 0);
    ItemType* const pFrontItem = _p_front;
    ItemType* const pNextItem = pFrontItem->pNext;
    if (pNextItem != VMA_NULL)
    {
        pNextItem->pPrev = VMA_NULL;
    }
    _p_front = pNextItem;
    _item_allocator.free(pFrontItem);
    --_count;
}

template<typename T>
void VmaRawList<T>::pop_back()
{
    VMA_HEAVY_ASSERT(_count > 0);
    ItemType* const pBackItem = _p_back;
    ItemType* const pPrevItem = pBackItem->pPrev;
    if(pPrevItem != VMA_NULL)
    {
        pPrevItem->pNext = VMA_NULL;
    }
    _p_back = pPrevItem;
    _item_allocator.free(pBackItem);
    --_count;
}

template<typename T>
void VmaRawList<T>::clear()
{
    if (!is_empty())
    {
        ItemType* pItem = _p_back;
        while (pItem != VMA_NULL)
        {
            ItemType* const pPrevItem = pItem->pPrev;
            _item_allocator.free(pItem);
            pItem = pPrevItem;
        }
        _p_front = VMA_NULL;
        _p_back = VMA_NULL;
        _count = 0;
    }
}

template<typename T>
void VmaRawList<T>::remove(ItemType* pItem)
{
    VMA_HEAVY_ASSERT(pItem != VMA_NULL);
    VMA_HEAVY_ASSERT(_count > 0);

    if(pItem->pPrev != VMA_NULL)
    {
        pItem->pPrev->pNext = pItem->pNext;
    }
    else
    {
        VMA_HEAVY_ASSERT(_p_front == pItem);
        _p_front = pItem->pNext;
    }

    if(pItem->pNext != VMA_NULL)
    {
        pItem->pNext->pPrev = pItem->pPrev;
    }
    else
    {
        VMA_HEAVY_ASSERT(_p_back == pItem);
        _p_back = pItem->pPrev;
    }

    _item_allocator.free(pItem);
    --_count;
}

template<typename T>
VmaListItem<T>* VmaRawList<T>::insert_before(ItemType* pItem)
{
    if(pItem != VMA_NULL)
    {
        ItemType* const prevItem = pItem->pPrev;
        ItemType* const newItem = _item_allocator.alloc();
        newItem->pPrev = prevItem;
        newItem->pNext = pItem;
        pItem->pPrev = newItem;
        if(prevItem != VMA_NULL)
        {
            prevItem->pNext = newItem;
        }
        else
        {
            VMA_HEAVY_ASSERT(_p_front == pItem);
            _p_front = newItem;
        }
        ++_count;
        return newItem;
    }
    return push_back();
}

template<typename T>
VmaListItem<T>* VmaRawList<T>::insert_after(ItemType* pItem)
{
    if(pItem != VMA_NULL)
    {
        ItemType* const nextItem = pItem->pNext;
        ItemType* const newItem = _item_allocator.alloc();
        newItem->pNext = nextItem;
        newItem->pPrev = pItem;
        pItem->pNext = newItem;
        if(nextItem != VMA_NULL)
        {
            nextItem->pPrev = newItem;
        }
        else
        {
            VMA_HEAVY_ASSERT(_p_back == pItem);
            _p_back = newItem;
        }
        ++_count;
        return newItem;
    }
    return push_front();
}

template<typename T>
VmaListItem<T>* VmaRawList<T>::insert_before(ItemType* pItem, const T& value)
{
    ItemType* const newItem = insert_before(pItem);
    newItem->Value = value;
    return newItem;
}

template<typename T>
VmaListItem<T>* VmaRawList<T>::insert_after(ItemType* pItem, const T& value)
{
    ItemType* const newItem = insert_after(pItem);
    newItem->Value = value;
    return newItem;
}
#endif // _VMA_RAW_LIST_FUNCTIONS
#endif // _VMA_RAW_LIST

#ifndef _VMA_LIST
template<typename T, typename AllocatorT>
class VmaList
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaList)
public:
    class reverse_iterator;
    class const_iterator;
    class const_reverse_iterator;

    class iterator
    {
        friend class const_iterator;
        friend class VmaList<T, AllocatorT>;
    public:
        iterator() :  _p_list(VMA_NULL), _p_item(VMA_NULL) {}
        explicit iterator(const reverse_iterator& src) : _p_list(src._p_list), _p_item(src._p_item) {}

        T& operator*() const { VMA_HEAVY_ASSERT(_p_item != VMA_NULL); return _p_item->Value; }
        T* operator->() const { VMA_HEAVY_ASSERT(_p_item != VMA_NULL); return &_p_item->Value; }

        bool operator==(const iterator& rhs) const { VMA_HEAVY_ASSERT(_p_list == rhs._p_list); return _p_item == rhs._p_item; }
        bool operator!=(const iterator& rhs) const { VMA_HEAVY_ASSERT(_p_list == rhs._p_list); return _p_item != rhs._p_item; }

        const iterator operator++(int) { iterator result = *this; ++*this; return result; }
        const iterator operator--(int) { iterator result = *this; --*this; return result; }

        iterator& operator++() { VMA_HEAVY_ASSERT(_p_item != VMA_NULL); _p_item = _p_item->pNext; return *this; }
        iterator& operator--();

    private:
        VmaRawList<T>* _p_list;
        VmaListItem<T>* _p_item;

        iterator(VmaRawList<T>* pList, VmaListItem<T>* pItem) : _p_list(pList),  _p_item(pItem) {}
    };
    class reverse_iterator
    {
        friend class const_reverse_iterator;
        friend class VmaList<T, AllocatorT>;
    public:
        reverse_iterator() : _p_list(VMA_NULL), _p_item(VMA_NULL) {}
        explicit reverse_iterator(const iterator& src) : _p_list(src._p_list), _p_item(src._p_item) {}

        T& operator*() const { VMA_HEAVY_ASSERT(_p_item != VMA_NULL); return _p_item->Value; }
        T* operator->() const { VMA_HEAVY_ASSERT(_p_item != VMA_NULL); return &_p_item->Value; }

        bool operator==(const reverse_iterator& rhs) const { VMA_HEAVY_ASSERT(_p_list == rhs._p_list); return _p_item == rhs._p_item; }
        bool operator!=(const reverse_iterator& rhs) const { VMA_HEAVY_ASSERT(_p_list == rhs._p_list); return _p_item != rhs._p_item; }

        const reverse_iterator operator++(int) { reverse_iterator result = *this; ++* this; return result; }
        const reverse_iterator operator--(int) { reverse_iterator result = *this; --* this; return result; }

        reverse_iterator& operator++() { VMA_HEAVY_ASSERT(_p_item != VMA_NULL); _p_item = _p_item->pPrev; return *this; }
        reverse_iterator& operator--();

    private:
        VmaRawList<T>* _p_list;
        VmaListItem<T>* _p_item;

        reverse_iterator(VmaRawList<T>* pList, VmaListItem<T>* pItem) : _p_list(pList),  _p_item(pItem) {}
    };
    class const_iterator
    {
        friend class VmaList<T, AllocatorT>;
    public:
        const_iterator() : _p_list(VMA_NULL), _p_item(VMA_NULL) {}
        explicit const_iterator(const iterator& src) : _p_list(src._p_list), _p_item(src._p_item) {}
        explicit const_iterator(const reverse_iterator& src) : _p_list(src._p_list), _p_item(src._p_item) {}

        iterator drop_const() { return { const_cast<VmaRawList<T>*>(_p_list), const_cast<VmaListItem<T>*>(_p_item) }; }

        const T& operator*() const { VMA_HEAVY_ASSERT(_p_item != VMA_NULL); return _p_item->Value; }
        const T* operator->() const { VMA_HEAVY_ASSERT(_p_item != VMA_NULL); return &_p_item->Value; }

        bool operator==(const const_iterator& rhs) const { VMA_HEAVY_ASSERT(_p_list == rhs._p_list); return _p_item == rhs._p_item; }
        bool operator!=(const const_iterator& rhs) const { VMA_HEAVY_ASSERT(_p_list == rhs._p_list); return _p_item != rhs._p_item; }

        const const_iterator operator++(int) { const_iterator result = *this; ++* this; return result; }
        const const_iterator operator--(int) { const_iterator result = *this; --* this; return result; }

        const_iterator& operator++() { VMA_HEAVY_ASSERT(_p_item != VMA_NULL); _p_item = _p_item->pNext; return *this; }
        const_iterator& operator--();

    private:
        const VmaRawList<T>* _p_list;
        const VmaListItem<T>* _p_item;

        const_iterator(const VmaRawList<T>* pList, const VmaListItem<T>* pItem) : _p_list(pList), _p_item(pItem) {}
    };
    class const_reverse_iterator
    {
        friend class VmaList<T, AllocatorT>;
    public:
        const_reverse_iterator() : _p_list(VMA_NULL), _p_item(VMA_NULL) {}
        explicit const_reverse_iterator(const reverse_iterator& src) : _p_list(src._p_list), _p_item(src._p_item) {}
        explicit const_reverse_iterator(const iterator& src) : _p_list(src._p_list), _p_item(src._p_item) {}

        reverse_iterator drop_const() { return { const_cast<VmaRawList<T>*>(_p_list), const_cast<VmaListItem<T>*>(_p_item) }; }

        const T& operator*() const { VMA_HEAVY_ASSERT(_p_item != VMA_NULL); return _p_item->Value; }
        const T* operator->() const { VMA_HEAVY_ASSERT(_p_item != VMA_NULL); return &_p_item->Value; }

        bool operator==(const const_reverse_iterator& rhs) const { VMA_HEAVY_ASSERT(_p_list == rhs._p_list); return _p_item == rhs._p_item; }
        bool operator!=(const const_reverse_iterator& rhs) const { VMA_HEAVY_ASSERT(_p_list == rhs._p_list); return _p_item != rhs._p_item; }

        const const_reverse_iterator operator++(int) { const_reverse_iterator result = *this; ++* this; return result; }
        const const_reverse_iterator operator--(int) { const_reverse_iterator result = *this; --* this; return result; }

        const_reverse_iterator& operator++() { VMA_HEAVY_ASSERT(_p_item != VMA_NULL); _p_item = _p_item->pPrev; return *this; }
        const_reverse_iterator& operator--();

    private:
        const VmaRawList<T>* _p_list;
        const VmaListItem<T>* _p_item;

        const_reverse_iterator(const VmaRawList<T>* pList, const VmaListItem<T>* pItem) : _p_list(pList), _p_item(pItem) {}
    };

    explicit VmaList(const AllocatorT& allocator) : _raw_list(allocator._p_callbacks) {}

    bool empty() const { return _raw_list.is_empty(); }
    size_t size() const { return _raw_list.get_count(); }

    iterator begin() { return iterator(&_raw_list, _raw_list.front()); }
    iterator end() { return iterator(&_raw_list, VMA_NULL); }

    const_iterator cbegin() const { return const_iterator(&_raw_list, _raw_list.front()); }
    const_iterator cend() const { return const_iterator(&_raw_list, VMA_NULL); }

    const_iterator begin() const { return cbegin(); }
    const_iterator end() const { return cend(); }

    reverse_iterator rbegin() { return reverse_iterator(&_raw_list, _raw_list.back()); }
    reverse_iterator rend() { return reverse_iterator(&_raw_list, VMA_NULL); }

    const_reverse_iterator crbegin() const { return const_reverse_iterator(&_raw_list, _raw_list.back()); }
    const_reverse_iterator crend() const { return const_reverse_iterator(&_raw_list, VMA_NULL); }

    const_reverse_iterator rbegin() const { return crbegin(); }
    const_reverse_iterator rend() const { return crend(); }

    void push_back(const T& value) { _raw_list.push_back(value); }
    iterator insert(iterator it, const T& value) { return iterator(&_raw_list, _raw_list.insert_before(it._p_item, value)); }

    void clear() { _raw_list.clear(); }
    void erase(iterator it) { _raw_list.remove(it._p_item); }

private:
    VmaRawList<T> _raw_list;
};

#ifndef _VMA_LIST_FUNCTIONS
template<typename T, typename AllocatorT>
typename VmaList<T, AllocatorT>::iterator& VmaList<T, AllocatorT>::iterator::operator--()
{
    if (_p_item != VMA_NULL)
    {
        _p_item = _p_item->pPrev;
    }
    else
    {
        VMA_HEAVY_ASSERT(!_p_list->is_empty());
        _p_item = _p_list->back();
    }
    return *this;
}

template<typename T, typename AllocatorT>
typename VmaList<T, AllocatorT>::reverse_iterator& VmaList<T, AllocatorT>::reverse_iterator::operator--()
{
    if (_p_item != VMA_NULL)
    {
        _p_item = _p_item->pNext;
    }
    else
    {
        VMA_HEAVY_ASSERT(!_p_list->is_empty());
        _p_item = _p_list->front();
    }
    return *this;
}

template<typename T, typename AllocatorT>
typename VmaList<T, AllocatorT>::const_iterator& VmaList<T, AllocatorT>::const_iterator::operator--()
{
    if (_p_item != VMA_NULL)
    {
        _p_item = _p_item->pPrev;
    }
    else
    {
        VMA_HEAVY_ASSERT(!_p_list->is_empty());
        _p_item = _p_list->back();
    }
    return *this;
}

template<typename T, typename AllocatorT>
typename VmaList<T, AllocatorT>::const_reverse_iterator& VmaList<T, AllocatorT>::const_reverse_iterator::operator--()
{
    if (_p_item != VMA_NULL)
    {
        _p_item = _p_item->pNext;
    }
    else
    {
        VMA_HEAVY_ASSERT(!_p_list->is_empty());
        _p_item = _p_list->back();
    }
    return *this;
}
#endif // _VMA_LIST_FUNCTIONS
#endif // _VMA_LIST

#ifndef _VMA_INTRUSIVE_LINKED_LIST
/*
Expected interface of ItemTypeTraits:
struct MyItemTypeTraits
{
    typedef MyItem ItemType;
    static ItemType* get_prev(const ItemType* item) { return item->myPrevPtr; }
    static ItemType* get_next(const ItemType* item) { return item->myNextPtr; }
    static ItemType*& access_prev(ItemType* item) { return item->myPrevPtr; }
    static ItemType*& access_next(ItemType* item) { return item->myNextPtr; }
};
*/
template<typename ItemTypeTraits>
class VmaIntrusiveLinkedList
{
public:
    typedef typename ItemTypeTraits::ItemType ItemType;
    static ItemType* get_prev(const ItemType* item) { return ItemTypeTraits::get_prev(item); }
    static ItemType* get_next(const ItemType* item) { return ItemTypeTraits::get_next(item); }

    // Movable, not copyable.
    VmaIntrusiveLinkedList() = default;
    VmaIntrusiveLinkedList(VmaIntrusiveLinkedList && src) noexcept;
    VmaIntrusiveLinkedList(const VmaIntrusiveLinkedList&) = delete;
    VmaIntrusiveLinkedList& operator=(VmaIntrusiveLinkedList&& src) noexcept;
    VmaIntrusiveLinkedList& operator=(const VmaIntrusiveLinkedList&) = delete;
    ~VmaIntrusiveLinkedList() { VMA_HEAVY_ASSERT(is_empty()); }

    size_t get_count() const { return _count; }
    bool is_empty() const { return _count == 0; }
    ItemType* front() { return _front; }
    ItemType* back() { return _back; }
    const ItemType* front() const { return _front; }
    const ItemType* back() const { return _back; }

    void push_back(ItemType* item);
    void push_front(ItemType* item);
    ItemType* pop_back();
    ItemType* pop_front();

    // MyItem can be null - it means PushBack.
    void insert_before(ItemType* existingItem, ItemType* newItem);
    // MyItem can be null - it means PushFront.
    void insert_after(ItemType* existingItem, ItemType* newItem);
    void remove(ItemType* item);
    void remove_all();

private:
    ItemType* _front = VMA_NULL;
    ItemType* _back = VMA_NULL;
    size_t _count = 0;
};

#ifndef _VMA_INTRUSIVE_LINKED_LIST_FUNCTIONS
template<typename ItemTypeTraits>
VmaIntrusiveLinkedList<ItemTypeTraits>::VmaIntrusiveLinkedList(VmaIntrusiveLinkedList&& src) noexcept
    : _front(src._front), _back(src._back), _count(src._count)
{
    src._front = src._back = VMA_NULL;
    src._count = 0;
}

template<typename ItemTypeTraits>
VmaIntrusiveLinkedList<ItemTypeTraits>& VmaIntrusiveLinkedList<ItemTypeTraits>::operator=(VmaIntrusiveLinkedList&& src) noexcept
{
    if (&src != this)
    {
        VMA_HEAVY_ASSERT(is_empty());
        _front = src._front;
        _back = src._back;
        _count = src._count;
        src._front = src._back = VMA_NULL;
        src._count = 0;
    }
    return *this;
}

template<typename ItemTypeTraits>
void VmaIntrusiveLinkedList<ItemTypeTraits>::push_back(ItemType* item)
{
    VMA_HEAVY_ASSERT(ItemTypeTraits::get_prev(item) == VMA_NULL && ItemTypeTraits::get_next(item) == VMA_NULL);
    if (is_empty())
    {
        _front = item;
        _back = item;
        _count = 1;
    }
    else
    {
        ItemTypeTraits::access_prev(item) = _back;
        ItemTypeTraits::access_next(_back) = item;
        _back = item;
        ++_count;
    }
}

template<typename ItemTypeTraits>
void VmaIntrusiveLinkedList<ItemTypeTraits>::push_front(ItemType* item)
{
    VMA_HEAVY_ASSERT(ItemTypeTraits::get_prev(item) == VMA_NULL && ItemTypeTraits::get_next(item) == VMA_NULL);
    if (is_empty())
    {
        _front = item;
        _back = item;
        _count = 1;
    }
    else
    {
        ItemTypeTraits::access_next(item) = _front;
        ItemTypeTraits::access_prev(_front) = item;
        _front = item;
        ++_count;
    }
}

template<typename ItemTypeTraits>
typename VmaIntrusiveLinkedList<ItemTypeTraits>::ItemType* VmaIntrusiveLinkedList<ItemTypeTraits>::pop_back()
{
    VMA_HEAVY_ASSERT(_count > 0);
    ItemType* const backItem = _back;
    ItemType* const prevItem = ItemTypeTraits::get_prev(backItem);
    if (prevItem != VMA_NULL)
    {
        ItemTypeTraits::access_next(prevItem) = VMA_NULL;
    }
    _back = prevItem;
    --_count;
    ItemTypeTraits::access_prev(backItem) = VMA_NULL;
    ItemTypeTraits::access_next(backItem) = VMA_NULL;
    return backItem;
}

template<typename ItemTypeTraits>
typename VmaIntrusiveLinkedList<ItemTypeTraits>::ItemType* VmaIntrusiveLinkedList<ItemTypeTraits>::pop_front()
{
    VMA_HEAVY_ASSERT(_count > 0);
    ItemType* const frontItem = _front;
    ItemType* const nextItem = ItemTypeTraits::get_next(frontItem);
    if (nextItem != VMA_NULL)
    {
        ItemTypeTraits::access_prev(nextItem) = VMA_NULL;
    }
    _front = nextItem;
    --_count;
    ItemTypeTraits::access_prev(frontItem) = VMA_NULL;
    ItemTypeTraits::access_next(frontItem) = VMA_NULL;
    return frontItem;
}

template<typename ItemTypeTraits>
void VmaIntrusiveLinkedList<ItemTypeTraits>::insert_before(ItemType* existingItem, ItemType* newItem)
{
    VMA_HEAVY_ASSERT(newItem != VMA_NULL && ItemTypeTraits::get_prev(newItem) == VMA_NULL && ItemTypeTraits::get_next(newItem) == VMA_NULL);
    if (existingItem != VMA_NULL)
    {
        ItemType* const prevItem = ItemTypeTraits::get_prev(existingItem);
        ItemTypeTraits::access_prev(newItem) = prevItem;
        ItemTypeTraits::access_next(newItem) = existingItem;
        ItemTypeTraits::access_prev(existingItem) = newItem;
        if (prevItem != VMA_NULL)
        {
            ItemTypeTraits::access_next(prevItem) = newItem;
        }
        else
        {
            VMA_HEAVY_ASSERT(_front == existingItem);
            _front = newItem;
        }
        ++_count;
    }
    else
        push_back(newItem);
}

template<typename ItemTypeTraits>
void VmaIntrusiveLinkedList<ItemTypeTraits>::insert_after(ItemType* existingItem, ItemType* newItem)
{
    VMA_HEAVY_ASSERT(newItem != VMA_NULL && ItemTypeTraits::get_prev(newItem) == VMA_NULL && ItemTypeTraits::get_next(newItem) == VMA_NULL);
    if (existingItem != VMA_NULL)
    {
        ItemType* const nextItem = ItemTypeTraits::get_next(existingItem);
        ItemTypeTraits::access_next(newItem) = nextItem;
        ItemTypeTraits::access_prev(newItem) = existingItem;
        ItemTypeTraits::access_next(existingItem) = newItem;
        if (nextItem != VMA_NULL)
        {
            ItemTypeTraits::access_prev(nextItem) = newItem;
        }
        else
        {
            VMA_HEAVY_ASSERT(_back == existingItem);
            _back = newItem;
        }
        ++_count;
    }
    else
        return push_front(newItem);
}

template<typename ItemTypeTraits>
void VmaIntrusiveLinkedList<ItemTypeTraits>::remove(ItemType* item)
{
    VMA_HEAVY_ASSERT(item != VMA_NULL && _count > 0);
    if (ItemTypeTraits::get_prev(item) != VMA_NULL)
    {
        ItemTypeTraits::access_next(ItemTypeTraits::access_prev(item)) = ItemTypeTraits::get_next(item);
    }
    else
    {
        VMA_HEAVY_ASSERT(_front == item);
        _front = ItemTypeTraits::get_next(item);
    }

    if (ItemTypeTraits::get_next(item) != VMA_NULL)
    {
        ItemTypeTraits::access_prev(ItemTypeTraits::access_next(item)) = ItemTypeTraits::get_prev(item);
    }
    else
    {
        VMA_HEAVY_ASSERT(_back == item);
        _back = ItemTypeTraits::get_prev(item);
    }
    ItemTypeTraits::access_prev(item) = VMA_NULL;
    ItemTypeTraits::access_next(item) = VMA_NULL;
    --_count;
}

template<typename ItemTypeTraits>
void VmaIntrusiveLinkedList<ItemTypeTraits>::remove_all()
{
    if (!is_empty())
    {
        ItemType* item = _back;
        while (item != VMA_NULL)
        {
            ItemType* const prevItem = ItemTypeTraits::access_prev(item);
            ItemTypeTraits::access_prev(item) = VMA_NULL;
            ItemTypeTraits::access_next(item) = VMA_NULL;
            item = prevItem;
        }
        _front = VMA_NULL;
        _back = VMA_NULL;
        _count = 0;
    }
}
#endif // _VMA_INTRUSIVE_LINKED_LIST_FUNCTIONS
#endif // _VMA_INTRUSIVE_LINKED_LIST

#if !defined(_VMA_STRING_BUILDER) && VMA_STATS_STRING_ENABLED
class VmaStringBuilder
{
public:
    explicit VmaStringBuilder(const VkAllocationCallbacks* allocationCallbacks) : _data(VmaStlAllocator<char>(allocationCallbacks)) {}
    ~VmaStringBuilder() = default;

    size_t get_length() const { return _data.size(); }
    // Returned string is not null-terminated!
    const char* get_data() const { return _data.data(); }
    void add_new_line() { add('\n'); }
    void add(char ch) { _data.push_back(ch); }

    void add(const char* pStr);
    void add_number(uint32_t num);
    void add_number(uint64_t num);
    void add_pointer(const void* ptr);

private:
    VmaVector<char, VmaStlAllocator<char>> _data;
};

#ifndef _VMA_STRING_BUILDER_FUNCTIONS
void VmaStringBuilder::add(const char* pStr)
{
    const size_t strLen = strlen(pStr);
    if (strLen > 0)
    {
        const size_t oldCount = _data.size();
        _data.resize(oldCount + strLen);
        memcpy(_data.data() + oldCount, pStr, strLen);
    }
}

void VmaStringBuilder::add_number(uint32_t num)
{
    char buf[11];
    buf[10] = '\0';
    char* p = &buf[10];
    do
    {
        *--p = '0' + (char)(num % 10);
        num /= 10;
    } while (num);
    add(p);
}

void VmaStringBuilder::add_number(uint64_t num)
{
    char buf[21];
    buf[20] = '\0';
    char* p = &buf[20];
    do
    {
        *--p = '0' + (char)(num % 10);
        num /= 10;
    } while (num);
    add(p);
}

void VmaStringBuilder::add_pointer(const void* ptr)
{
    char buf[21];
    VmaPtrToStr(buf, sizeof(buf), ptr);
    add(buf);
}
#endif //_VMA_STRING_BUILDER_FUNCTIONS
#endif // _VMA_STRING_BUILDER

#if !defined(_VMA_JSON_WRITER) && VMA_STATS_STRING_ENABLED
/*
Allows to conveniently build a correct JSON document to be written to the
VmaStringBuilder passed to the constructor.
*/
class VmaJsonWriter
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaJsonWriter)
public:
    // sb - string builder to write the document to. Must remain alive for the whole lifetime of this object.
    VmaJsonWriter(const VkAllocationCallbacks* pAllocationCallbacks, VmaStringBuilder& sb);
    ~VmaJsonWriter();

    // Begins object by writing "{".
    // Inside an object, you must call pairs of write_string and a value, e.g.:
    // j.begin_object(true); j.write_string("A"); j.write_number(1); j.write_string("B"); j.write_number(2); j.end_object();
    // Will write: { "A": 1, "B": 2 }
    void begin_object(bool singleLine = false);
    // Ends object by writing "}".
    void end_object();

    // Begins array by writing "[".
    // Inside an array, you can write a sequence of any values.
    void begin_array(bool singleLine = false);
    // Ends array by writing "[".
    void end_array();

    // Writes a string value inside "".
    // pStr can contain any ANSI characters, including '"', new line etc. - they will be properly escaped.
    void write_string(const char* pStr);

    // Begins writing a string value.
    // Call begin_string, continue_string, continue_string, ..., end_string instead of
    // write_string to conveniently build the string content incrementally, made of
    // parts including numbers.
    void begin_string(const char* pStr = VMA_NULL);
    // Posts next part of an open string.
    void continue_string(const char* pStr);
    // Posts next part of an open string. The number is converted to decimal characters.
    void continue_string(uint32_t n);
    void continue_string(uint64_t n);
    // Posts next part of an open string. Pointer value is converted to characters
    // using "%p" formatting - shown as hexadecimal number, e.g.: 000000081276Ad00
    void continue_string_pointer(const void* ptr);
    // Ends writing a string value by writing '"'.
    void end_string(const char* pStr = VMA_NULL);

    // Writes a number value.
    void write_number(uint32_t n);
    void write_number(uint64_t n);
    // Writes a boolean value - false or true.
    void write_bool(bool b);
    // Writes a null value.
    void write_null();

private:
    enum COLLECTION_TYPE
    {
        COLLECTION_TYPE_OBJECT,
        COLLECTION_TYPE_ARRAY,
    };
    struct StackItem
    {
        COLLECTION_TYPE type;
        uint32_t valueCount;
        bool singleLineMode;
    };

    static const char* const INDENT;

    VmaStringBuilder& _sb;
    VmaVector< StackItem, VmaStlAllocator<StackItem> > _stack;
    bool _inside_string;

    void begin_value(bool isString);
    void write_indent(bool oneLess = false);
};
const char* const VmaJsonWriter::INDENT = "  ";

#ifndef _VMA_JSON_WRITER_FUNCTIONS
VmaJsonWriter::VmaJsonWriter(const VkAllocationCallbacks* pAllocationCallbacks, VmaStringBuilder& sb)
    : _sb(sb),
    _stack(VmaStlAllocator<StackItem>(pAllocationCallbacks)),
    _inside_string(false) {}

VmaJsonWriter::~VmaJsonWriter()
{
    VMA_ASSERT(!_inside_string);
    VMA_ASSERT(_stack.empty());
}

void VmaJsonWriter::begin_object(bool singleLine)
{
    VMA_ASSERT(!_inside_string);

    begin_value(false);
    _sb.add('{');

    StackItem item;
    item.type = COLLECTION_TYPE_OBJECT;
    item.valueCount = 0;
    item.singleLineMode = singleLine;
    _stack.push_back(item);
}

void VmaJsonWriter::end_object()
{
    VMA_ASSERT(!_inside_string);

    write_indent(true);
    _sb.add('}');

    VMA_ASSERT(!_stack.empty() && _stack.back().type == COLLECTION_TYPE_OBJECT);
    _stack.pop_back();
}

void VmaJsonWriter::begin_array(bool singleLine)
{
    VMA_ASSERT(!_inside_string);

    begin_value(false);
    _sb.add('[');

    StackItem item;
    item.type = COLLECTION_TYPE_ARRAY;
    item.valueCount = 0;
    item.singleLineMode = singleLine;
    _stack.push_back(item);
}

void VmaJsonWriter::end_array()
{
    VMA_ASSERT(!_inside_string);

    write_indent(true);
    _sb.add(']');

    VMA_ASSERT(!_stack.empty() && _stack.back().type == COLLECTION_TYPE_ARRAY);
    _stack.pop_back();
}

void VmaJsonWriter::write_string(const char* pStr)
{
    begin_string(pStr);
    end_string();
}

void VmaJsonWriter::begin_string(const char* pStr)
{
    VMA_ASSERT(!_inside_string);

    begin_value(true);
    _sb.add('"');
    _inside_string = true;
    if (pStr != VMA_NULL && pStr[0] != '\0')
    {
        continue_string(pStr);
    }
}

void VmaJsonWriter::continue_string(const char* pStr)
{
    VMA_ASSERT(_inside_string);

    const size_t strLen = strlen(pStr);
    for (size_t i = 0; i < strLen; ++i)
    {
        char ch = pStr[i];
        if (ch == '\\')
        {
            _sb.add("\\\\");
        }
        else if (ch == '"')
        {
            _sb.add("\\\"");
        }
        else if ((uint8_t)ch >= 32)
        {
            _sb.add(ch);
        }
        else switch (ch)
        {
        case '\b':
            _sb.add("\\b");
            break;
        case '\f':
            _sb.add("\\f");
            break;
        case '\n':
            _sb.add("\\n");
            break;
        case '\r':
            _sb.add("\\r");
            break;
        case '\t':
            _sb.add("\\t");
            break;
        default:
            VMA_ASSERT(0 && "Character not currently supported.");
        }
    }
}

void VmaJsonWriter::continue_string(uint32_t n)
{
    VMA_ASSERT(_inside_string);
    _sb.add_number(n);
}

void VmaJsonWriter::continue_string(uint64_t n)
{
    VMA_ASSERT(_inside_string);
    _sb.add_number(n);
}

void VmaJsonWriter::continue_string_pointer(const void* ptr)
{
    VMA_ASSERT(_inside_string);
    _sb.add_pointer(ptr);
}

void VmaJsonWriter::end_string(const char* pStr)
{
    VMA_ASSERT(_inside_string);
    if (pStr != VMA_NULL && pStr[0] != '\0')
    {
        continue_string(pStr);
    }
    _sb.add('"');
    _inside_string = false;
}

void VmaJsonWriter::write_number(uint32_t n)
{
    VMA_ASSERT(!_inside_string);
    begin_value(false);
    _sb.add_number(n);
}

void VmaJsonWriter::write_number(uint64_t n)
{
    VMA_ASSERT(!_inside_string);
    begin_value(false);
    _sb.add_number(n);
}

void VmaJsonWriter::write_bool(bool b)
{
    VMA_ASSERT(!_inside_string);
    begin_value(false);
    _sb.add(b ? "true" : "false");
}

void VmaJsonWriter::write_null()
{
    VMA_ASSERT(!_inside_string);
    begin_value(false);
    _sb.add("null");
}

void VmaJsonWriter::begin_value(bool isString)
{
    if (!_stack.empty())
    {
        StackItem& currItem = _stack.back();
        if (currItem.type == COLLECTION_TYPE_OBJECT &&
            currItem.valueCount % 2 == 0)
        {
            VMA_ASSERT(isString);
        }

        if (currItem.type == COLLECTION_TYPE_OBJECT &&
            currItem.valueCount % 2 != 0)
        {
            _sb.add(": ");
        }
        else if (currItem.valueCount > 0)
        {
            _sb.add(", ");
            write_indent();
        }
        else
        {
            write_indent();
        }
        ++currItem.valueCount;
    }
}

void VmaJsonWriter::write_indent(bool oneLess)
{
    if (!_stack.empty() && !_stack.back().singleLineMode)
    {
        _sb.add_new_line();

        size_t count = _stack.size();
        if (count > 0 && oneLess)
        {
            --count;
        }
        for (size_t i = 0; i < count; ++i)
        {
            _sb.add(INDENT);
        }
    }
}
#endif // _VMA_JSON_WRITER_FUNCTIONS

static void VmaPrintDetailedStatistics(VmaJsonWriter& json, const VmaDetailedStatistics& stat)
{
    json.begin_object();

    json.write_string("BlockCount");
    json.write_number(stat.statistics.blockCount);
    json.write_string("BlockBytes");
    json.write_number(stat.statistics.blockBytes);
    json.write_string("AllocationCount");
    json.write_number(stat.statistics.allocationCount);
    json.write_string("AllocationBytes");
    json.write_number(stat.statistics.allocationBytes);
    json.write_string("UnusedRangeCount");
    json.write_number(stat.unusedRangeCount);

    if (stat.statistics.allocationCount > 1)
    {
        json.write_string("AllocationSizeMin");
        json.write_number(stat.allocationSizeMin);
        json.write_string("AllocationSizeMax");
        json.write_number(stat.allocationSizeMax);
    }
    if (stat.unusedRangeCount > 1)
    {
        json.write_string("UnusedRangeSizeMin");
        json.write_number(stat.unusedRangeSizeMin);
        json.write_string("UnusedRangeSizeMax");
        json.write_number(stat.unusedRangeSizeMax);
    }
    json.end_object();
}
#endif // _VMA_JSON_WRITER

#ifndef _VMA_MAPPING_HYSTERESIS

class VmaMappingHysteresis
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaMappingHysteresis)
public:
    VmaMappingHysteresis() = default;

    uint32_t get_extra_mapping() const { return _extra_mapping; }

    // Call when Map was called.
    // Returns true if switched to extra +1 mapping reference count.
    bool post_map()
    {
#if VMA_MAPPING_HYSTERESIS_ENABLED
        if(_extra_mapping == 0)
        {
            ++_major_counter;
            if(_major_counter >= COUNTER_MIN_EXTRA_MAPPING)
            {
                _extra_mapping = 1;
                _major_counter = 0;
                _minor_counter = 0;
                return true;
            }
        }
        else // _extra_mapping == 1
            post_minor_counter();
#endif // #if VMA_MAPPING_HYSTERESIS_ENABLED
        return false;
    }

    // Call when unmap was called.
    void post_unmap()
    {
#if VMA_MAPPING_HYSTERESIS_ENABLED
        if(_extra_mapping == 0)
            ++_major_counter;
        else // _extra_mapping == 1
            post_minor_counter();
#endif // #if VMA_MAPPING_HYSTERESIS_ENABLED
    }

    // Call when allocation was made from the memory block.
    void post_alloc()
    {
#if VMA_MAPPING_HYSTERESIS_ENABLED
        if(_extra_mapping == 1)
            ++_major_counter;
        else // _extra_mapping == 0
            post_minor_counter();
#endif // #if VMA_MAPPING_HYSTERESIS_ENABLED
    }

    // Call when allocation was freed from the memory block.
    // Returns true if switched to extra -1 mapping reference count.
    bool post_free()
    {
#if VMA_MAPPING_HYSTERESIS_ENABLED
        if(_extra_mapping == 1)
        {
            ++_major_counter;
            if(_major_counter >= COUNTER_MIN_EXTRA_MAPPING &&
                _major_counter > _minor_counter + 1)
            {
                _extra_mapping = 0;
                _major_counter = 0;
                _minor_counter = 0;
                return true;
            }
        }
        else // _extra_mapping == 0
            post_minor_counter();
#endif // #if VMA_MAPPING_HYSTERESIS_ENABLED
        return false;
    }

private:
    static const int32_t COUNTER_MIN_EXTRA_MAPPING = 7;

    uint32_t _minor_counter = 0;
    uint32_t _major_counter = 0;
    uint32_t _extra_mapping = 0; // 0 or 1.

    void post_minor_counter()
    {
        if(_minor_counter < _major_counter)
        {
            ++_minor_counter;
        }
        else if(_major_counter > 0)
        {
            --_major_counter;
            --_minor_counter;
        }
    }
};

#endif // _VMA_MAPPING_HYSTERESIS

#if VMA_EXTERNAL_MEMORY_WIN32
class VmaWin32Handle
{
public:
    VmaWin32Handle() noexcept : _h_handle(VMA_NULL) { }
    explicit VmaWin32Handle(HANDLE hHandle) noexcept : _h_handle(hHandle) { }
    ~VmaWin32Handle() noexcept { if (_h_handle != VMA_NULL) { ::CloseHandle(_h_handle); } }
    VMA_CLASS_NO_COPY_NO_MOVE(VmaWin32Handle)

public:
    // Strengthened
    VkResult get_handle(VkDevice device, VkDeviceMemory memory, PFN_vkGetMemoryWin32HandleKHR pvkGetMemoryWin32HandleKHR, HANDLE hTargetProcess, bool useMutex, HANDLE* pHandle) noexcept
    {
        *pHandle = VMA_NULL;
        // Try to get handle first.
        if (_h_handle != VMA_NULL)
        {
            *pHandle = Duplicate(hTargetProcess);
            return VK_SUCCESS;
        }

        VkResult res = VK_SUCCESS;
        // If failed, try to create it.
        {
            VmaMutexLockWrite lock(_mutex, useMutex);
            if (_h_handle == VMA_NULL)
            {
                res = Create(device, memory, pvkGetMemoryWin32HandleKHR, &_h_handle);
            }
        }

        *pHandle = Duplicate(hTargetProcess);
        return res;
    }

    operator bool() const noexcept { return _h_handle != VMA_NULL; }
private:
    // Not atomic
    static VkResult Create(VkDevice device, VkDeviceMemory memory, PFN_vkGetMemoryWin32HandleKHR pvkGetMemoryWin32HandleKHR, HANDLE* pHandle) noexcept
    {
        VkResult res = VK_ERROR_FEATURE_NOT_PRESENT;
        if (pvkGetMemoryWin32HandleKHR != VMA_NULL)
        {
            VkMemoryGetWin32HandleInfoKHR handleInfo{ };
            handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
            handleInfo.memory = memory;
            handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
            res = pvkGetMemoryWin32HandleKHR(device, &handleInfo, pHandle);
        }
        return res;
    }
    HANDLE Duplicate(HANDLE hTargetProcess = VMA_NULL) const noexcept
    {
        if (!_h_handle)
            return _h_handle;

        HANDLE hCurrentProcess = ::GetCurrentProcess();
        HANDLE hDupHandle = VMA_NULL;
        if (!::DuplicateHandle(hCurrentProcess, _h_handle, hTargetProcess ? hTargetProcess : hCurrentProcess, &hDupHandle, 0, FALSE, DUPLICATE_SAME_ACCESS))
        {
            VMA_ASSERT(0 && "Failed to duplicate handle.");
        }
        return hDupHandle;
    }
private:
    HANDLE _h_handle;
    VMA_RW_MUTEX _mutex; // Protects access _handle
};
#else 
class VmaWin32Handle
{
    // ABI compatibility
    void* placeholder = VMA_NULL;
    VMA_RW_MUTEX placeholder2;
};
#endif // VMA_EXTERNAL_MEMORY_WIN32


#ifndef _VMA_DEVICE_MEMORY_BLOCK
/*
Represents a single block of device memory (`VkDeviceMemory`) with all the
data about its regions (aka suballocations, #VmaAllocation), assigned and free.

Thread-safety:
- Access to _p_metadata must be externally synchronized.
- Map, unmap, Bind* are synchronized internally.
*/
class VmaDeviceMemoryBlock
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaDeviceMemoryBlock)
public:
    VmaBlockMetadata* _p_metadata;

    explicit VmaDeviceMemoryBlock(VmaAllocator hAllocator);
    ~VmaDeviceMemoryBlock();

    // Always call after construction.
    void init(
        VmaAllocator hAllocator,
        VmaPool hParentPool,
        uint32_t newMemoryTypeIndex,
        VkDeviceMemory newMemory,
        VkDeviceSize newSize,
        uint32_t id,
        uint32_t algorithm,
        VkDeviceSize bufferImageGranularity);
    // Always call before destruction.
    void destroy(VmaAllocator allocator);

    VmaPool get_parent_pool() const { return _h_parent_pool; }
    VkDeviceMemory get_device_memory() const { return _h_memory; }
    uint32_t get_memory_type_index() const { return _memory_type_index; }
    uint32_t get_id() const { return _id; }
    void* get_mapped_data() const { return _p_mapped_data; }
    uint32_t get_map_ref_count() const { return _map_count; }

    // Call when allocation/free was made from _p_metadata.
    // Used for _mapping_hysteresis.
    void post_alloc(VmaAllocator hAllocator);
    void post_free(VmaAllocator hAllocator);

    // Validates all data structures inside this object. If not valid, returns false.
    bool validate() const;
    VkResult check_corruption(VmaAllocator hAllocator);

    // ppData can be null.
    VkResult Map(VmaAllocator hAllocator, uint32_t count, void** ppData);
    void unmap(VmaAllocator hAllocator, uint32_t count);

    VkResult write_magic_value_after_allocation(VmaAllocator hAllocator, VkDeviceSize allocOffset, VkDeviceSize allocSize);
    VkResult validate_magic_value_after_allocation(VmaAllocator hAllocator, VkDeviceSize allocOffset, VkDeviceSize allocSize);

    VkResult bind_buffer_memory(
        VmaAllocator hAllocator,
        VmaAllocation hAllocation,
        VkDeviceSize allocationLocalOffset,
        VkBuffer hBuffer,
        const void* pNext);
    VkResult bind_image_memory(
        VmaAllocator hAllocator,
        VmaAllocation hAllocation,
        VkDeviceSize allocationLocalOffset,
        VkImage hImage,
        const void* pNext);
#if VMA_EXTERNAL_MEMORY_WIN32
    VkResult create_win32_handle(
        const VmaAllocator hAllocator,
        PFN_vkGetMemoryWin32HandleKHR pvkGetMemoryWin32HandleKHR,
        HANDLE hTargetProcess,
        HANDLE* pHandle)noexcept;
#endif // VMA_EXTERNAL_MEMORY_WIN32
private:
    VmaPool _h_parent_pool; // VK_NULL_HANDLE if not belongs to custom pool.
    uint32_t _memory_type_index;
    uint32_t _id;
    VkDeviceMemory _h_memory;

    /*
    Protects access to _h_memory so it is not used by multiple threads simultaneously, e.g. vkMapMemory, vkBindBufferMemory.
    Also protects _map_count, _p_mapped_data.
    Allocations, deallocations, any change in _p_metadata is protected by parent's VmaBlockVector::_mutex.
    */
    VMA_MUTEX _map_and_bind_mutex;
    VmaMappingHysteresis _mapping_hysteresis;
    uint32_t _map_count;
    void* _p_mapped_data;

    VmaWin32Handle _handle;
};
#endif // _VMA_DEVICE_MEMORY_BLOCK

#ifndef _VMA_ALLOCATION_T
struct VmaAllocationExtraData
{
    void* _p_mapped_data = VMA_NULL; // Not null means memory is mapped.
    VmaWin32Handle _handle;
};

struct VmaAllocation_T
{
    friend struct VmaDedicatedAllocationListItemTraits;

    enum FLAGS
    {
        FLAG_PERSISTENT_MAP   = 0x01,
        FLAG_MAPPING_ALLOWED  = 0x02,
    };

public:
    enum ALLOCATION_TYPE
    {
        ALLOCATION_TYPE_NONE,
        ALLOCATION_TYPE_BLOCK,
        ALLOCATION_TYPE_DEDICATED,
    };

    // This struct is allocated using VmaPoolAllocator.
    explicit VmaAllocation_T(bool mappingAllowed);
    ~VmaAllocation_T();

    void init_block_allocation(
        VmaDeviceMemoryBlock* block,
        VmaAllocHandle allocHandle,
        VkDeviceSize alignment,
        VkDeviceSize size,
        uint32_t memoryTypeIndex,
        VmaSuballocationType suballocationType,
        bool mapped);
    // pMappedData not null means allocation is created with MAPPED flag.
    void init_dedicated_allocation(
        VmaAllocator allocator,
        VmaPool hParentPool,
        uint32_t memoryTypeIndex,
        VkDeviceMemory hMemory,
        VmaSuballocationType suballocationType,
        void* pMappedData,
        VkDeviceSize size);
    void destroy(VmaAllocator allocator);

    ALLOCATION_TYPE get_type() const { return (ALLOCATION_TYPE)_type; }
    VkDeviceSize get_alignment() const { return _alignment; }
    VkDeviceSize get_size() const { return _size; }
    void* get_user_data() const { return _p_user_data; }
    const char* get_name() const { return _p_name; }
    VmaSuballocationType get_suballocation_type() const { return (VmaSuballocationType)_suballocation_type; }

    VmaDeviceMemoryBlock* get_block() const { VMA_ASSERT(_type == ALLOCATION_TYPE_BLOCK); return _block_allocation._block; }
    uint32_t get_memory_type_index() const { return _memory_type_index; }
    bool is_persistent_map() const { return (_flags & FLAG_PERSISTENT_MAP) != 0; }
    bool is_mapping_allowed() const { return (_flags & FLAG_MAPPING_ALLOWED) != 0; }

    void set_user_data(VmaAllocator hAllocator, void* pUserData) { _p_user_data = pUserData; }
    void set_name(VmaAllocator hAllocator, const char* pName);
    void free_name(VmaAllocator hAllocator);
    uint8_t swap_block_allocation(VmaAllocator hAllocator, VmaAllocation allocation);
    VmaAllocHandle get_alloc_handle() const;
    VkDeviceSize get_offset() const;
    VmaPool get_parent_pool() const;
    VkDeviceMemory get_memory() const;
    void* get_mapped_data() const;

    void block_alloc_map();
    void block_alloc_unmap();
    VkResult dedicated_alloc_map(VmaAllocator hAllocator, void** ppData);
    void dedicated_alloc_unmap(VmaAllocator hAllocator);

#if VMA_STATS_STRING_ENABLED
    VmaBufferImageUsage get_buffer_image_usage() const { return _buffer_image_usage; }
    void init_buffer_usage(const VkBufferCreateInfo &createInfo, bool useKhrMaintenance5)
    {
        VMA_ASSERT(_buffer_image_usage == VmaBufferImageUsage::UNKNOWN);
        _buffer_image_usage = VmaBufferImageUsage(createInfo, useKhrMaintenance5);
    }
    void init_image_usage(const VkImageCreateInfo &createInfo)
    {
        VMA_ASSERT(_buffer_image_usage == VmaBufferImageUsage::UNKNOWN);
        _buffer_image_usage = VmaBufferImageUsage(createInfo);
    }
    void print_parameters(class VmaJsonWriter& json) const;
#endif

#if VMA_EXTERNAL_MEMORY_WIN32
    VkResult get_win32_handle(VmaAllocator hAllocator, HANDLE hTargetProcess, HANDLE* hHandle) noexcept;
#endif // VMA_EXTERNAL_MEMORY_WIN32

private:
    // Allocation out of VmaDeviceMemoryBlock.
    struct BlockAllocation
    {
        VmaDeviceMemoryBlock* _block;
        VmaAllocHandle _alloc_handle;
    };
    // Allocation for an object that has its own private VkDeviceMemory.
    struct DedicatedAllocation
    {
        VmaPool _h_parent_pool; // VK_NULL_HANDLE if not belongs to custom pool.
        VkDeviceMemory _h_memory;
        VmaAllocationExtraData* _extra_data;
        VmaAllocation_T* _prev;
        VmaAllocation_T* _next;
    };
    union
    {
        // Allocation out of VmaDeviceMemoryBlock.
        BlockAllocation _block_allocation;
        // Allocation for an object that has its own private VkDeviceMemory.
        DedicatedAllocation _dedicated_allocation;
    };

    VkDeviceSize _alignment;
    VkDeviceSize _size;
    void* _p_user_data;
    char* _p_name;
    uint32_t _memory_type_index;
    uint8_t _type; // ALLOCATION_TYPE
    uint8_t _suballocation_type; // VmaSuballocationType
    // Reference counter for vmaMapMemory()/vmaUnmapMemory().
    uint8_t _map_count;
    uint8_t _flags; // enum FLAGS
#if VMA_STATS_STRING_ENABLED
    VmaBufferImageUsage _buffer_image_usage; // 0 if unknown.
#endif

    void ensure_extra_data(VmaAllocator hAllocator);
};
#endif // _VMA_ALLOCATION_T

#ifndef _VMA_DEDICATED_ALLOCATION_LIST_ITEM_TRAITS
struct VmaDedicatedAllocationListItemTraits
{
    typedef VmaAllocation_T ItemType;

    static ItemType* get_prev(const ItemType* item)
    {
        VMA_HEAVY_ASSERT(item->get_type() == VmaAllocation_T::ALLOCATION_TYPE_DEDICATED);
        return item->_dedicated_allocation._prev;
    }
    static ItemType* get_next(const ItemType* item)
    {
        VMA_HEAVY_ASSERT(item->get_type() == VmaAllocation_T::ALLOCATION_TYPE_DEDICATED);
        return item->_dedicated_allocation._next;
    }
    static ItemType*& access_prev(ItemType* item)
    {
        VMA_HEAVY_ASSERT(item->get_type() == VmaAllocation_T::ALLOCATION_TYPE_DEDICATED);
        return item->_dedicated_allocation._prev;
    }
    static ItemType*& access_next(ItemType* item)
    {
        VMA_HEAVY_ASSERT(item->get_type() == VmaAllocation_T::ALLOCATION_TYPE_DEDICATED);
        return item->_dedicated_allocation._next;
    }
};
#endif // _VMA_DEDICATED_ALLOCATION_LIST_ITEM_TRAITS

#ifndef _VMA_DEDICATED_ALLOCATION_LIST
/*
Stores linked list of VmaAllocation_T objects.
Thread-safe, synchronized internally.
*/
class VmaDedicatedAllocationList
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaDedicatedAllocationList)
public:
    VmaDedicatedAllocationList() = default;
    ~VmaDedicatedAllocationList();

    void init(bool useMutex) { _use_mutex = useMutex; }
    bool validate();

    void add_detailed_statistics(VmaDetailedStatistics& inoutStats);
    void add_statistics(VmaStatistics& inoutStats);
#if VMA_STATS_STRING_ENABLED
    // Writes JSON array with the list of allocations.
    void build_stats_string(VmaJsonWriter& json);
#endif

    bool is_empty();
    void register_allocation(VmaAllocation alloc);
    void unregister_allocation(VmaAllocation alloc);

private:
    typedef VmaIntrusiveLinkedList<VmaDedicatedAllocationListItemTraits> DedicatedAllocationLinkedList;

    bool _use_mutex = true;
    VMA_RW_MUTEX _mutex;
    DedicatedAllocationLinkedList _allocation_list;
};

#ifndef _VMA_DEDICATED_ALLOCATION_LIST_FUNCTIONS

VmaDedicatedAllocationList::~VmaDedicatedAllocationList()
{
    VMA_HEAVY_ASSERT(validate());

    if (!_allocation_list.is_empty())
    {
        VMA_ASSERT_LEAK(false && "Unfreed dedicated allocations found!");
    }
}

bool VmaDedicatedAllocationList::validate()
{
    const size_t declaredCount = _allocation_list.get_count();
    size_t actualCount = 0;
    VmaMutexLockRead lock(_mutex, _use_mutex);
    for (VmaAllocation alloc = _allocation_list.front();
        alloc != VMA_NULL; alloc = _allocation_list.get_next(alloc))
    {
        ++actualCount;
    }
    VMA_VALIDATE(actualCount == declaredCount);

    return true;
}

void VmaDedicatedAllocationList::add_detailed_statistics(VmaDetailedStatistics& inoutStats)
{
    for(auto* item = _allocation_list.front(); item != VMA_NULL; item = DedicatedAllocationLinkedList::get_next(item))
    {
        const VkDeviceSize size = item->get_size();
        inoutStats.statistics.blockCount++;
        inoutStats.statistics.blockBytes += size;
        VmaAddDetailedStatisticsAllocation(inoutStats, item->get_size());
    }
}

void VmaDedicatedAllocationList::add_statistics(VmaStatistics& inoutStats)
{
    VmaMutexLockRead lock(_mutex, _use_mutex);

    const uint32_t allocCount = (uint32_t)_allocation_list.get_count();
    inoutStats.blockCount += allocCount;
    inoutStats.allocationCount += allocCount;

    for(auto* item = _allocation_list.front(); item != VMA_NULL; item = DedicatedAllocationLinkedList::get_next(item))
    {
        const VkDeviceSize size = item->get_size();
        inoutStats.blockBytes += size;
        inoutStats.allocationBytes += size;
    }
}

#if VMA_STATS_STRING_ENABLED
void VmaDedicatedAllocationList::build_stats_string(VmaJsonWriter& json)
{
    VmaMutexLockRead lock(_mutex, _use_mutex);
    json.begin_array();
    for (VmaAllocation alloc = _allocation_list.front();
        alloc != VMA_NULL; alloc = _allocation_list.get_next(alloc))
    {
        json.begin_object(true);
        alloc->print_parameters(json);
        json.end_object();
    }
    json.end_array();
}
#endif // VMA_STATS_STRING_ENABLED

bool VmaDedicatedAllocationList::is_empty()
{
    VmaMutexLockRead lock(_mutex, _use_mutex);
    return _allocation_list.is_empty();
}

void VmaDedicatedAllocationList::register_allocation(VmaAllocation alloc)
{
    VmaMutexLockWrite lock(_mutex, _use_mutex);
    _allocation_list.push_back(alloc);
}

void VmaDedicatedAllocationList::unregister_allocation(VmaAllocation alloc)
{
    VmaMutexLockWrite lock(_mutex, _use_mutex);
    _allocation_list.remove(alloc);
}
#endif // _VMA_DEDICATED_ALLOCATION_LIST_FUNCTIONS
#endif // _VMA_DEDICATED_ALLOCATION_LIST

#ifndef _VMA_SUBALLOCATION
/*
Represents a region of VmaDeviceMemoryBlock that is either assigned and returned as
allocated memory block or free.
*/
struct VmaSuballocation
{
    VkDeviceSize offset;
    VkDeviceSize size;
    void* userData;
    VmaSuballocationType type;
};

// Comparator for offsets.
struct VmaSuballocationOffsetLess
{
    bool operator()(const VmaSuballocation& lhs, const VmaSuballocation& rhs) const
    {
        return lhs.offset < rhs.offset;
    }
};

struct VmaSuballocationOffsetGreater
{
    bool operator()(const VmaSuballocation& lhs, const VmaSuballocation& rhs) const
    {
        return lhs.offset > rhs.offset;
    }
};

struct VmaSuballocationItemSizeLess
{
    bool operator()(const VmaSuballocationList::iterator lhs,
        const VmaSuballocationList::iterator rhs) const
    {
        return lhs->size < rhs->size;
    }

    bool operator()(const VmaSuballocationList::iterator lhs,
        VkDeviceSize rhsSize) const
    {
        return lhs->size < rhsSize;
    }
};
#endif // _VMA_SUBALLOCATION

#ifndef _VMA_ALLOCATION_REQUEST
/*
Parameters of planned allocation inside a VmaDeviceMemoryBlock.
item points to a FREE suballocation.
*/
struct VmaAllocationRequest
{
    VmaAllocHandle allocHandle;
    VkDeviceSize size;
    VmaSuballocationList::iterator item;
    void* customData;
    uint64_t algorithmData;
    VmaAllocationRequestType type;
};
#endif // _VMA_ALLOCATION_REQUEST

#ifndef _VMA_BLOCK_METADATA
/*
Data structure used for bookkeeping of allocations and unused ranges of memory
in a single VkDeviceMemory block.
*/
class VmaBlockMetadata
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaBlockMetadata)
public:
    // pAllocationCallbacks, if not null, must be owned externally - alive and unchanged for the whole lifetime of this object.
    VmaBlockMetadata(const VkAllocationCallbacks* pAllocationCallbacks,
        VkDeviceSize bufferImageGranularity, bool isVirtual);
    virtual ~VmaBlockMetadata() = default;

    virtual void init(VkDeviceSize size) { _size = size; }
    bool is_virtual() const { return _is_virtual; }
    VkDeviceSize get_size() const { return _size; }

    // Validates all data structures inside this object. If not valid, returns false.
    virtual bool validate() const = 0;
    virtual size_t get_allocation_count() const = 0;
    virtual size_t get_free_regions_count() const = 0;
    virtual VkDeviceSize get_sum_free_size() const = 0;
    // Returns true if this block is empty - contains only single free suballocation.
    virtual bool is_empty() const = 0;
    virtual void get_allocation_info(VmaAllocHandle allocHandle, VmaVirtualAllocationInfo& outInfo) = 0;
    virtual VkDeviceSize get_allocation_offset(VmaAllocHandle allocHandle) const = 0;
    virtual void* get_allocation_user_data(VmaAllocHandle allocHandle) const = 0;

    virtual VmaAllocHandle get_allocation_list_begin() const = 0;
    virtual VmaAllocHandle get_next_allocation(VmaAllocHandle prevAlloc) const = 0;
    virtual VkDeviceSize get_next_free_region_size(VmaAllocHandle alloc) const = 0;

    // Shouldn't modify blockCount.
    virtual void add_detailed_statistics(VmaDetailedStatistics& inoutStats) const = 0;
    virtual void add_statistics(VmaStatistics& inoutStats) const = 0;

#if VMA_STATS_STRING_ENABLED
    virtual void print_detailed_map(class VmaJsonWriter& json) const = 0;
#endif

    // Tries to find a place for suballocation with given parameters inside this block.
    // If succeeded, fills pAllocationRequest and returns true.
    // If failed, returns false.
    virtual bool create_allocation_request(
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        bool upperAddress,
        VmaSuballocationType allocType,
        // Always one of VMA_ALLOCATION_CREATE_STRATEGY_* or VMA_ALLOCATION_INTERNAL_STRATEGY_* flags.
        uint32_t strategy,
        VmaAllocationRequest* pAllocationRequest) = 0;

    virtual VkResult check_corruption(const void* pBlockData) = 0;

    // Makes actual allocation based on request. Request must already be checked and valid.
    virtual void alloc(
        const VmaAllocationRequest& request,
        VmaSuballocationType type,
        void* userData) = 0;

    // Frees suballocation assigned to given memory region.
    virtual void free(VmaAllocHandle allocHandle) = 0;

    // Frees all allocations.
    // Careful! Don't call it if there are VmaAllocation objects owned by userData of cleared allocations!
    virtual void clear() = 0;

    virtual void set_allocation_user_data(VmaAllocHandle allocHandle, void* userData) = 0;
    virtual void debug_log_all_allocations() const = 0;

protected:
    const VkAllocationCallbacks* get_allocation_callbacks() const { return _p_allocation_callbacks; }
    VkDeviceSize get_buffer_image_granularity() const { return _buffer_image_granularity; }
    VkDeviceSize get_debug_margin() const { return VkDeviceSize(is_virtual() ? 0 : VMA_DEBUG_MARGIN); }

    void debug_log_allocation(VkDeviceSize offset, VkDeviceSize size, void* userData) const;
#if VMA_STATS_STRING_ENABLED
    // mapRefCount == UINT32_MAX means unspecified.
    void print_detailed_map_begin(class VmaJsonWriter& json,
        VkDeviceSize unusedBytes,
        size_t allocationCount,
        size_t unusedRangeCount) const;
    void print_detailed_map_allocation(class VmaJsonWriter& json,
        VkDeviceSize offset, VkDeviceSize size, void* userData) const;
    static void print_detailed_map_unused_range(class VmaJsonWriter& json,
        VkDeviceSize offset,
        VkDeviceSize size);
    static void print_detailed_map_end(class VmaJsonWriter& json);
#endif

private:
    VkDeviceSize _size;
    const VkAllocationCallbacks* _p_allocation_callbacks;
    const VkDeviceSize _buffer_image_granularity;
    const bool _is_virtual;
};

#ifndef _VMA_BLOCK_METADATA_FUNCTIONS
VmaBlockMetadata::VmaBlockMetadata(const VkAllocationCallbacks* pAllocationCallbacks,
    VkDeviceSize bufferImageGranularity, bool isVirtual)
    : _size(0),
    _p_allocation_callbacks(pAllocationCallbacks),
    _buffer_image_granularity(bufferImageGranularity),
    _is_virtual(isVirtual) {}

void VmaBlockMetadata::debug_log_allocation(VkDeviceSize offset, VkDeviceSize size, void* userData) const
{
    if (is_virtual())
    {
        VMA_LEAK_LOG_FORMAT("UNFREED VIRTUAL ALLOCATION; Offset: %" PRIu64 "; Size: %" PRIu64 "; UserData: %p", offset, size, userData);
    }
    else
    {
        VMA_ASSERT(userData != VMA_NULL);
        VmaAllocation allocation = reinterpret_cast<VmaAllocation>(userData);

        userData = allocation->get_user_data();
        [[maybe_unused]] const char* name = allocation->get_name();

#if VMA_STATS_STRING_ENABLED
        VMA_LEAK_LOG_FORMAT("UNFREED ALLOCATION; Offset: %" PRIu64 "; Size: %" PRIu64 "; UserData: %p; Name: %s; Type: %s; Usage: %" PRIu64,
            offset, size, userData, name ? name : "vma_empty",
            VMA_SUBALLOCATION_TYPE_NAMES[allocation->get_suballocation_type()],
            (uint64_t)allocation->get_buffer_image_usage().Value);
#else
        VMA_LEAK_LOG_FORMAT("UNFREED ALLOCATION; Offset: %" PRIu64 "; Size: %" PRIu64 "; UserData: %p; Name: %s; Type: %u",
            offset, size, userData, name ? name : "vma_empty",
            (unsigned)allocation->get_suballocation_type());
#endif // VMA_STATS_STRING_ENABLED
    }

}

#if VMA_STATS_STRING_ENABLED
void VmaBlockMetadata::print_detailed_map_begin(class VmaJsonWriter& json,
    VkDeviceSize unusedBytes, size_t allocationCount, size_t unusedRangeCount) const
{
    json.write_string("TotalBytes");
    json.write_number(get_size());

    json.write_string("UnusedBytes");
    json.write_number(unusedBytes);

    json.write_string("Allocations");
    json.write_number((uint64_t)allocationCount);

    json.write_string("UnusedRanges");
    json.write_number((uint64_t)unusedRangeCount);

    json.write_string("Suballocations");
    json.begin_array();
}

void VmaBlockMetadata::print_detailed_map_allocation(class VmaJsonWriter& json,
    VkDeviceSize offset, VkDeviceSize size, void* userData) const
{
    json.begin_object(true);

    json.write_string("Offset");
    json.write_number(offset);

    if (is_virtual())
    {
        json.write_string("Size");
        json.write_number(size);
        if (userData)
        {
            json.write_string("CustomData");
            json.begin_string();
            json.continue_string_pointer(userData);
            json.end_string();
        }
    }
    else
    {
        ((VmaAllocation)userData)->print_parameters(json);
    }

    json.end_object();
}

void VmaBlockMetadata::print_detailed_map_unused_range(class VmaJsonWriter& json,
    VkDeviceSize offset, VkDeviceSize size)
{
    json.begin_object(true);

    json.write_string("Offset");
    json.write_number(offset);

    json.write_string("Type");
    json.write_string(VMA_SUBALLOCATION_TYPE_NAMES[VMA_SUBALLOCATION_TYPE_FREE]);

    json.write_string("Size");
    json.write_number(size);

    json.end_object();
}

void VmaBlockMetadata::print_detailed_map_end(class VmaJsonWriter& json)
{
    json.end_array();
}
#endif // VMA_STATS_STRING_ENABLED
#endif // _VMA_BLOCK_METADATA_FUNCTIONS
#endif // _VMA_BLOCK_METADATA

#ifndef _VMA_BLOCK_BUFFER_IMAGE_GRANULARITY
// Before deleting object of this class remember to call 'destroy()'
class VmaBlockBufferImageGranularity final
{
public:
    struct ValidationContext
    {
        const VkAllocationCallbacks* allocCallbacks;
        uint16_t* pageAllocs;
    };

    explicit VmaBlockBufferImageGranularity(VkDeviceSize bufferImageGranularity);
    ~VmaBlockBufferImageGranularity();

    bool is_enabled() const { return _buffer_image_granularity > MAX_LOW_BUFFER_IMAGE_GRANULARITY; }

    void init(const VkAllocationCallbacks* pAllocationCallbacks, VkDeviceSize size);
    // Before destroying object you must call free it's memory
    void destroy(const VkAllocationCallbacks* pAllocationCallbacks);

    void roundup_alloc_request(VmaSuballocationType allocType,
        VkDeviceSize& inOutAllocSize,
        VkDeviceSize& inOutAllocAlignment) const;

    bool check_conflict_and_align_up(VkDeviceSize& inOutAllocOffset,
        VkDeviceSize allocSize,
        VkDeviceSize blockOffset,
        VkDeviceSize blockSize,
        VmaSuballocationType allocType) const;

    void alloc_pages(uint8_t allocType, VkDeviceSize offset, VkDeviceSize size);
    void free_pages(VkDeviceSize offset, VkDeviceSize size);
    void clear();

    ValidationContext start_validation(const VkAllocationCallbacks* pAllocationCallbacks,
        bool isVirutal) const;
    bool validate(ValidationContext& ctx, VkDeviceSize offset, VkDeviceSize size) const;
    bool finish_validation(ValidationContext& ctx) const;

private:
    static const uint16_t MAX_LOW_BUFFER_IMAGE_GRANULARITY = 256;

    struct RegionInfo
    {
        uint8_t allocType;
        uint16_t allocCount;
    };

    VkDeviceSize _buffer_image_granularity;
    uint32_t _region_count;
    RegionInfo* _region_info;

    uint32_t get_start_page(VkDeviceSize offset) const { return offset_to_page_index(offset & ~(_buffer_image_granularity - 1)); }
    uint32_t get_end_page(VkDeviceSize offset, VkDeviceSize size) const { return offset_to_page_index((offset + size - 1) & ~(_buffer_image_granularity - 1)); }

    uint32_t offset_to_page_index(VkDeviceSize offset) const;
    static void alloc_page(RegionInfo& page, uint8_t allocType);
};

#ifndef _VMA_BLOCK_BUFFER_IMAGE_GRANULARITY_FUNCTIONS
VmaBlockBufferImageGranularity::VmaBlockBufferImageGranularity(VkDeviceSize bufferImageGranularity)
    : _buffer_image_granularity(bufferImageGranularity),
    _region_count(0),
    _region_info(VMA_NULL) {}

VmaBlockBufferImageGranularity::~VmaBlockBufferImageGranularity()
{
    VMA_ASSERT(_region_info == VMA_NULL && "Free not called before destroying object!");
}

void VmaBlockBufferImageGranularity::init(const VkAllocationCallbacks* pAllocationCallbacks, VkDeviceSize size)
{
    if (is_enabled())
    {
        _region_count = static_cast<uint32_t>(VmaDivideRoundingUp(size, _buffer_image_granularity));
        _region_info = vma_new_array(pAllocationCallbacks, RegionInfo, _region_count);
        memset(_region_info, 0, _region_count * sizeof(RegionInfo));
    }
}

void VmaBlockBufferImageGranularity::destroy(const VkAllocationCallbacks* pAllocationCallbacks)
{
    if (_region_info)
    {
        vma_delete_array(pAllocationCallbacks, _region_info, _region_count);
        _region_info = VMA_NULL;
    }
}

void VmaBlockBufferImageGranularity::roundup_alloc_request(VmaSuballocationType allocType,
    VkDeviceSize& inOutAllocSize,
    VkDeviceSize& inOutAllocAlignment) const
{
    if (_buffer_image_granularity > 1 &&
        _buffer_image_granularity <= MAX_LOW_BUFFER_IMAGE_GRANULARITY)
    {
        if (allocType == VMA_SUBALLOCATION_TYPE_UNKNOWN ||
            allocType == VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN ||
            allocType == VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL)
        {
            inOutAllocAlignment = VMA_MAX(inOutAllocAlignment, _buffer_image_granularity);
            inOutAllocSize = VmaAlignUp(inOutAllocSize, _buffer_image_granularity);
        }
    }
}

bool VmaBlockBufferImageGranularity::check_conflict_and_align_up(VkDeviceSize& inOutAllocOffset,
    VkDeviceSize allocSize,
    VkDeviceSize blockOffset,
    VkDeviceSize blockSize,
    VmaSuballocationType allocType) const
{
    if (is_enabled())
    {
        uint32_t startPage = get_start_page(inOutAllocOffset);
        if (_region_info[startPage].allocCount > 0 &&
            VmaIsBufferImageGranularityConflict(static_cast<VmaSuballocationType>(_region_info[startPage].allocType), allocType))
        {
            inOutAllocOffset = VmaAlignUp(inOutAllocOffset, _buffer_image_granularity);
            if (blockSize < allocSize + inOutAllocOffset - blockOffset)
                return true;
            ++startPage;
        }
        uint32_t endPage = get_end_page(inOutAllocOffset, allocSize);
        if (endPage != startPage &&
            _region_info[endPage].allocCount > 0 &&
            VmaIsBufferImageGranularityConflict(static_cast<VmaSuballocationType>(_region_info[endPage].allocType), allocType))
        {
            return true;
        }
    }
    return false;
}

void VmaBlockBufferImageGranularity::alloc_pages(uint8_t allocType, VkDeviceSize offset, VkDeviceSize size)
{
    if (is_enabled())
    {
        uint32_t startPage = get_start_page(offset);
        alloc_page(_region_info[startPage], allocType);

        uint32_t endPage = get_end_page(offset, size);
        if (startPage != endPage)
            alloc_page(_region_info[endPage], allocType);
    }
}

void VmaBlockBufferImageGranularity::free_pages(VkDeviceSize offset, VkDeviceSize size)
{
    if (is_enabled())
    {
        uint32_t startPage = get_start_page(offset);
        --_region_info[startPage].allocCount;
        if (_region_info[startPage].allocCount == 0)
            _region_info[startPage].allocType = VMA_SUBALLOCATION_TYPE_FREE;
        uint32_t endPage = get_end_page(offset, size);
        if (startPage != endPage)
        {
            --_region_info[endPage].allocCount;
            if (_region_info[endPage].allocCount == 0)
                _region_info[endPage].allocType = VMA_SUBALLOCATION_TYPE_FREE;
        }
    }
}

void VmaBlockBufferImageGranularity::clear()
{
    if (_region_info)
        memset(_region_info, 0, _region_count * sizeof(RegionInfo));
}

VmaBlockBufferImageGranularity::ValidationContext VmaBlockBufferImageGranularity::start_validation(
    const VkAllocationCallbacks* pAllocationCallbacks, bool isVirutal) const
{
    ValidationContext ctx{ pAllocationCallbacks, VMA_NULL };
    if (!isVirutal && is_enabled())
    {
        ctx.pageAllocs = vma_new_array(pAllocationCallbacks, uint16_t, _region_count);
        memset(ctx.pageAllocs, 0, _region_count * sizeof(uint16_t));
    }
    return ctx;
}

bool VmaBlockBufferImageGranularity::validate(ValidationContext& ctx,
    VkDeviceSize offset, VkDeviceSize size) const
{
    if (is_enabled())
    {
        uint32_t start = get_start_page(offset);
        ++ctx.pageAllocs[start];
        VMA_VALIDATE(_region_info[start].allocCount > 0);

        uint32_t end = get_end_page(offset, size);
        if (start != end)
        {
            ++ctx.pageAllocs[end];
            VMA_VALIDATE(_region_info[end].allocCount > 0);
        }
    }
    return true;
}

bool VmaBlockBufferImageGranularity::finish_validation(ValidationContext& ctx) const
{
    // Check proper page structure
    if (is_enabled())
    {
        VMA_ASSERT(ctx.pageAllocs != VMA_NULL && "Validation context not initialized!");

        for (uint32_t page = 0; page < _region_count; ++page)
        {
            VMA_VALIDATE(ctx.pageAllocs[page] == _region_info[page].allocCount);
        }
        vma_delete_array(ctx.allocCallbacks, ctx.pageAllocs, _region_count);
        ctx.pageAllocs = VMA_NULL;
    }
    return true;
}

uint32_t VmaBlockBufferImageGranularity::offset_to_page_index(VkDeviceSize offset) const
{
    return static_cast<uint32_t>(offset >> VMA_BITSCAN_MSB(_buffer_image_granularity));
}

void VmaBlockBufferImageGranularity::alloc_page(RegionInfo& page, uint8_t allocType)
{
    // When current alloc type is free then it can be overridden by new type
    if (page.allocCount == 0 || (page.allocCount > 0 && page.allocType == VMA_SUBALLOCATION_TYPE_FREE))
        page.allocType = allocType;

    ++page.allocCount;
}
#endif // _VMA_BLOCK_BUFFER_IMAGE_GRANULARITY_FUNCTIONS
#endif // _VMA_BLOCK_BUFFER_IMAGE_GRANULARITY

#ifndef _VMA_BLOCK_METADATA_LINEAR
/*
Allocations and their references in internal data structure look like this:

if(_2nd_vector_mode == SECOND_VECTOR_EMPTY):

        0 +-------+
          |       |
          |       |
          |       |
          +-------+
          | Alloc |  1st[_1st_null_items_begin_count]
          +-------+
          | Alloc |  1st[_1st_null_items_begin_count + 1]
          +-------+
          |  ...  |
          +-------+
          | Alloc |  1st[1st.size() - 1]
          +-------+
          |       |
          |       |
          |       |
get_size() +-------+

if(_2nd_vector_mode == SECOND_VECTOR_RING_BUFFER):

        0 +-------+
          | Alloc |  2nd[0]
          +-------+
          | Alloc |  2nd[1]
          +-------+
          |  ...  |
          +-------+
          | Alloc |  2nd[2nd.size() - 1]
          +-------+
          |       |
          |       |
          |       |
          +-------+
          | Alloc |  1st[_1st_null_items_begin_count]
          +-------+
          | Alloc |  1st[_1st_null_items_begin_count + 1]
          +-------+
          |  ...  |
          +-------+
          | Alloc |  1st[1st.size() - 1]
          +-------+
          |       |
get_size() +-------+

if(_2nd_vector_mode == SECOND_VECTOR_DOUBLE_STACK):

        0 +-------+
          |       |
          |       |
          |       |
          +-------+
          | Alloc |  1st[_1st_null_items_begin_count]
          +-------+
          | Alloc |  1st[_1st_null_items_begin_count + 1]
          +-------+
          |  ...  |
          +-------+
          | Alloc |  1st[1st.size() - 1]
          +-------+
          |       |
          |       |
          |       |
          +-------+
          | Alloc |  2nd[2nd.size() - 1]
          +-------+
          |  ...  |
          +-------+
          | Alloc |  2nd[1]
          +-------+
          | Alloc |  2nd[0]
get_size() +-------+

*/
class VmaBlockMetadata_Linear : public VmaBlockMetadata
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaBlockMetadata_Linear)
public:
    VmaBlockMetadata_Linear(const VkAllocationCallbacks* pAllocationCallbacks,
        VkDeviceSize bufferImageGranularity, bool isVirtual);
    ~VmaBlockMetadata_Linear() override = default;

    VkDeviceSize get_sum_free_size() const override { return _sum_free_size; }
    bool is_empty() const override { return get_allocation_count() == 0; }
    VkDeviceSize get_allocation_offset(VmaAllocHandle allocHandle) const override { return (VkDeviceSize)allocHandle - 1; }

    void init(VkDeviceSize size) override;
    bool validate() const override;
    size_t get_allocation_count() const override;
    size_t get_free_regions_count() const override;

    void add_detailed_statistics(VmaDetailedStatistics& inoutStats) const override;
    void add_statistics(VmaStatistics& inoutStats) const override;

#if VMA_STATS_STRING_ENABLED
    void print_detailed_map(class VmaJsonWriter& json) const override;
#endif

    bool create_allocation_request(
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        bool upperAddress,
        VmaSuballocationType allocType,
        uint32_t strategy,
        VmaAllocationRequest* pAllocationRequest) override;

    VkResult check_corruption(const void* pBlockData) override;

    void alloc(
        const VmaAllocationRequest& request,
        VmaSuballocationType type,
        void* userData) override;

    void free(VmaAllocHandle allocHandle) override;
    void get_allocation_info(VmaAllocHandle allocHandle, VmaVirtualAllocationInfo& outInfo) override;
    void* get_allocation_user_data(VmaAllocHandle allocHandle) const override;
    VmaAllocHandle get_allocation_list_begin() const override;
    VmaAllocHandle get_next_allocation(VmaAllocHandle prevAlloc) const override;
    VkDeviceSize get_next_free_region_size(VmaAllocHandle alloc) const override;
    void clear() override;
    void set_allocation_user_data(VmaAllocHandle allocHandle, void* userData) override;
    void debug_log_all_allocations() const override;

private:
    /*
    There are two suballocation vectors, used in ping-pong way.
    The one with index _1st_vector_index is called 1st.
    The one with index (_1st_vector_index ^ 1) is called 2nd.
    2nd can be non-empty only when 1st is not empty.
    When 2nd is not empty, _2nd_vector_mode indicates its mode of operation.
    */
    typedef VmaVector<VmaSuballocation, VmaStlAllocator<VmaSuballocation>> SuballocationVectorType;

    enum SECOND_VECTOR_MODE
    {
        SECOND_VECTOR_EMPTY,
        /*
        Suballocations in 2nd vector are created later than the ones in 1st, but they
        all have smaller offset.
        */
        SECOND_VECTOR_RING_BUFFER,
        /*
        Suballocations in 2nd vector are upper side of double stack.
        They all have offsets higher than those in 1st vector.
        Top of this stack means smaller offsets, but higher indices in this vector.
        */
        SECOND_VECTOR_DOUBLE_STACK,
    };

    VkDeviceSize _sum_free_size;
    SuballocationVectorType _suballocations0, _suballocations1;
    uint32_t _1st_vector_index;
    SECOND_VECTOR_MODE _2nd_vector_mode;
    // Number of items in 1st vector with hAllocation = null at the beginning.
    size_t _1st_null_items_begin_count;
    // Number of other items in 1st vector with hAllocation = null somewhere in the middle.
    size_t _1st_null_items_middle_count;
    // Number of items in 2nd vector with hAllocation = null.
    size_t _2nd_null_items_count;

    SuballocationVectorType& AccessSuballocations1st() { return _1st_vector_index ? _suballocations1 : _suballocations0; }
    SuballocationVectorType& AccessSuballocations2nd() { return _1st_vector_index ? _suballocations0 : _suballocations1; }
    const SuballocationVectorType& AccessSuballocations1st() const { return _1st_vector_index ? _suballocations1 : _suballocations0; }
    const SuballocationVectorType& AccessSuballocations2nd() const { return _1st_vector_index ? _suballocations0 : _suballocations1; }

    VmaSuballocation& find_suballocation(VkDeviceSize offset) const;
    bool should_compact_1st() const;
    void cleanup_after_free();

    bool create_allocation_request_lower_address(
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        VmaSuballocationType allocType,
        uint32_t strategy,
        VmaAllocationRequest* pAllocationRequest);
    bool create_allocation_request_upper_address(
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        VmaSuballocationType allocType,
        uint32_t strategy,
        VmaAllocationRequest* pAllocationRequest);
};

#ifndef _VMA_BLOCK_METADATA_LINEAR_FUNCTIONS
VmaBlockMetadata_Linear::VmaBlockMetadata_Linear(const VkAllocationCallbacks* pAllocationCallbacks,
    VkDeviceSize bufferImageGranularity, bool isVirtual)
    : VmaBlockMetadata(pAllocationCallbacks, bufferImageGranularity, isVirtual),
    _sum_free_size(0),
    _suballocations0(VmaStlAllocator<VmaSuballocation>(pAllocationCallbacks)),
    _suballocations1(VmaStlAllocator<VmaSuballocation>(pAllocationCallbacks)),
    _1st_vector_index(0),
    _2nd_vector_mode(SECOND_VECTOR_EMPTY),
    _1st_null_items_begin_count(0),
    _1st_null_items_middle_count(0),
    _2nd_null_items_count(0) {}

void VmaBlockMetadata_Linear::init(VkDeviceSize size)
{
    VmaBlockMetadata::init(size);
    _sum_free_size = size;
}

bool VmaBlockMetadata_Linear::validate() const
{
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    VMA_VALIDATE(suballocations2nd.empty() == (_2nd_vector_mode == SECOND_VECTOR_EMPTY));
    VMA_VALIDATE(!suballocations1st.empty() ||
        suballocations2nd.empty() ||
        _2nd_vector_mode != SECOND_VECTOR_RING_BUFFER);

    if (!suballocations1st.empty())
    {
        // Null item at the beginning should be accounted into _1st_null_items_begin_count.
        VMA_VALIDATE(suballocations1st[_1st_null_items_begin_count].type != VMA_SUBALLOCATION_TYPE_FREE);
        // Null item at the end should be just pop_back().
        VMA_VALIDATE(suballocations1st.back().type != VMA_SUBALLOCATION_TYPE_FREE);
    }
    if (!suballocations2nd.empty())
    {
        // Null item at the end should be just pop_back().
        VMA_VALIDATE(suballocations2nd.back().type != VMA_SUBALLOCATION_TYPE_FREE);
    }

    VMA_VALIDATE(_1st_null_items_begin_count + _1st_null_items_middle_count <= suballocations1st.size());
    VMA_VALIDATE(_2nd_null_items_count <= suballocations2nd.size());

    VkDeviceSize sumUsedSize = 0;
    const size_t suballoc1stCount = suballocations1st.size();
    const VkDeviceSize debugMargin = get_debug_margin();
    VkDeviceSize offset = 0;

    if (_2nd_vector_mode == SECOND_VECTOR_RING_BUFFER)
    {
        const size_t suballoc2ndCount = suballocations2nd.size();
        size_t nullItem2ndCount = 0;
        for (size_t i = 0; i < suballoc2ndCount; ++i)
        {
            const VmaSuballocation& suballoc = suballocations2nd[i];
            const bool currFree = (suballoc.type == VMA_SUBALLOCATION_TYPE_FREE);

            VmaAllocation const alloc = (VmaAllocation)suballoc.userData;
            if (!is_virtual())
            {
                VMA_VALIDATE(currFree == (alloc == VK_NULL_HANDLE));
            }
            VMA_VALIDATE(suballoc.offset >= offset);

            if (!currFree)
            {
                if (!is_virtual())
                {
                    VMA_VALIDATE((VkDeviceSize)alloc->get_alloc_handle() == suballoc.offset + 1);
                    VMA_VALIDATE(alloc->get_size() == suballoc.size);
                }
                sumUsedSize += suballoc.size;
            }
            else
            {
                ++nullItem2ndCount;
            }

            offset = suballoc.offset + suballoc.size + debugMargin;
        }

        VMA_VALIDATE(nullItem2ndCount == _2nd_null_items_count);
    }

    for (size_t i = 0; i < _1st_null_items_begin_count; ++i)
    {
        const VmaSuballocation& suballoc = suballocations1st[i];
        VMA_VALIDATE(suballoc.type == VMA_SUBALLOCATION_TYPE_FREE &&
            suballoc.userData == VMA_NULL);
    }

    size_t nullItem1stCount = _1st_null_items_begin_count;

    for (size_t i = _1st_null_items_begin_count; i < suballoc1stCount; ++i)
    {
        const VmaSuballocation& suballoc = suballocations1st[i];
        const bool currFree = (suballoc.type == VMA_SUBALLOCATION_TYPE_FREE);

        VmaAllocation const alloc = (VmaAllocation)suballoc.userData;
        if (!is_virtual())
        {
            VMA_VALIDATE(currFree == (alloc == VK_NULL_HANDLE));
        }
        VMA_VALIDATE(suballoc.offset >= offset);
        VMA_VALIDATE(i >= _1st_null_items_begin_count || currFree);

        if (!currFree)
        {
            if (!is_virtual())
            {
                VMA_VALIDATE((VkDeviceSize)alloc->get_alloc_handle() == suballoc.offset + 1);
                VMA_VALIDATE(alloc->get_size() == suballoc.size);
            }
            sumUsedSize += suballoc.size;
        }
        else
        {
            ++nullItem1stCount;
        }

        offset = suballoc.offset + suballoc.size + debugMargin;
    }
    VMA_VALIDATE(nullItem1stCount == _1st_null_items_begin_count + _1st_null_items_middle_count);

    if (_2nd_vector_mode == SECOND_VECTOR_DOUBLE_STACK)
    {
        const size_t suballoc2ndCount = suballocations2nd.size();
        size_t nullItem2ndCount = 0;
        for (size_t i = suballoc2ndCount; i--; )
        {
            const VmaSuballocation& suballoc = suballocations2nd[i];
            const bool currFree = (suballoc.type == VMA_SUBALLOCATION_TYPE_FREE);

            VmaAllocation const alloc = (VmaAllocation)suballoc.userData;
            if (!is_virtual())
            {
                VMA_VALIDATE(currFree == (alloc == VK_NULL_HANDLE));
            }
            VMA_VALIDATE(suballoc.offset >= offset);

            if (!currFree)
            {
                if (!is_virtual())
                {
                    VMA_VALIDATE((VkDeviceSize)alloc->get_alloc_handle() == suballoc.offset + 1);
                    VMA_VALIDATE(alloc->get_size() == suballoc.size);
                }
                sumUsedSize += suballoc.size;
            }
            else
            {
                ++nullItem2ndCount;
            }

            offset = suballoc.offset + suballoc.size + debugMargin;
        }

        VMA_VALIDATE(nullItem2ndCount == _2nd_null_items_count);
    }

    VMA_VALIDATE(offset <= get_size());
    VMA_VALIDATE(_sum_free_size == get_size() - sumUsedSize);

    return true;
}

size_t VmaBlockMetadata_Linear::get_allocation_count() const
{
    return AccessSuballocations1st().size() - _1st_null_items_begin_count - _1st_null_items_middle_count +
        AccessSuballocations2nd().size() - _2nd_null_items_count;
}

size_t VmaBlockMetadata_Linear::get_free_regions_count() const
{
    // Function only used for defragmentation, which is disabled for this algorithm
    VMA_ASSERT(0);
    return SIZE_MAX;
}

void VmaBlockMetadata_Linear::add_detailed_statistics(VmaDetailedStatistics& inoutStats) const
{
    const VkDeviceSize size = get_size();
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    const size_t suballoc1stCount = suballocations1st.size();
    const size_t suballoc2ndCount = suballocations2nd.size();

    inoutStats.statistics.blockCount++;
    inoutStats.statistics.blockBytes += size;

    VkDeviceSize lastOffset = 0;

    if (_2nd_vector_mode == SECOND_VECTOR_RING_BUFFER)
    {
        const VkDeviceSize freeSpace2ndTo1stEnd = suballocations1st[_1st_null_items_begin_count].offset;
        size_t nextAlloc2ndIndex = 0;
        while (lastOffset < freeSpace2ndTo1stEnd)
        {
            // Find next non-null allocation or move nextAllocIndex to the end.
            while (nextAlloc2ndIndex < suballoc2ndCount &&
                suballocations2nd[nextAlloc2ndIndex].userData == VMA_NULL)
            {
                ++nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex < suballoc2ndCount)
            {
                const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // 1. Process free space before this allocation.
                if (lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                    VmaAddDetailedStatisticsUnusedRange(inoutStats, unusedRangeSize);
                }

                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                VmaAddDetailedStatisticsAllocation(inoutStats, suballoc.size);

                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                ++nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                // There is free space from lastOffset to freeSpace2ndTo1stEnd.
                if (lastOffset < freeSpace2ndTo1stEnd)
                {
                    const VkDeviceSize unusedRangeSize = freeSpace2ndTo1stEnd - lastOffset;
                    VmaAddDetailedStatisticsUnusedRange(inoutStats, unusedRangeSize);
                }

                // End of loop.
                lastOffset = freeSpace2ndTo1stEnd;
            }
        }
    }

    size_t nextAlloc1stIndex = _1st_null_items_begin_count;
    const VkDeviceSize freeSpace1stTo2ndEnd =
        _2nd_vector_mode == SECOND_VECTOR_DOUBLE_STACK ? suballocations2nd.back().offset : size;
    while (lastOffset < freeSpace1stTo2ndEnd)
    {
        // Find next non-null allocation or move nextAllocIndex to the end.
        while (nextAlloc1stIndex < suballoc1stCount &&
            suballocations1st[nextAlloc1stIndex].userData == VMA_NULL)
        {
            ++nextAlloc1stIndex;
        }

        // Found non-null allocation.
        if (nextAlloc1stIndex < suballoc1stCount)
        {
            const VmaSuballocation& suballoc = suballocations1st[nextAlloc1stIndex];

            // 1. Process free space before this allocation.
            if (lastOffset < suballoc.offset)
            {
                // There is free space from lastOffset to suballoc.offset.
                const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                VmaAddDetailedStatisticsUnusedRange(inoutStats, unusedRangeSize);
            }

            // 2. Process this allocation.
            // There is allocation with suballoc.offset, suballoc.size.
            VmaAddDetailedStatisticsAllocation(inoutStats, suballoc.size);

            // 3. Prepare for next iteration.
            lastOffset = suballoc.offset + suballoc.size;
            ++nextAlloc1stIndex;
        }
        // We are at the end.
        else
        {
            // There is free space from lastOffset to freeSpace1stTo2ndEnd.
            if (lastOffset < freeSpace1stTo2ndEnd)
            {
                const VkDeviceSize unusedRangeSize = freeSpace1stTo2ndEnd - lastOffset;
                VmaAddDetailedStatisticsUnusedRange(inoutStats, unusedRangeSize);
            }

            // End of loop.
            lastOffset = freeSpace1stTo2ndEnd;
        }
    }

    if (_2nd_vector_mode == SECOND_VECTOR_DOUBLE_STACK)
    {
        size_t nextAlloc2ndIndex = suballocations2nd.size() - 1;
        while (lastOffset < size)
        {
            // Find next non-null allocation or move nextAllocIndex to the end.
            while (nextAlloc2ndIndex != SIZE_MAX &&
                suballocations2nd[nextAlloc2ndIndex].userData == VMA_NULL)
            {
                --nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex != SIZE_MAX)
            {
                const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // 1. Process free space before this allocation.
                if (lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                    VmaAddDetailedStatisticsUnusedRange(inoutStats, unusedRangeSize);
                }

                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                VmaAddDetailedStatisticsAllocation(inoutStats, suballoc.size);

                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                --nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                // There is free space from lastOffset to size.
                if (lastOffset < size)
                {
                    const VkDeviceSize unusedRangeSize = size - lastOffset;
                    VmaAddDetailedStatisticsUnusedRange(inoutStats, unusedRangeSize);
                }

                // End of loop.
                lastOffset = size;
            }
        }
    }
}

void VmaBlockMetadata_Linear::add_statistics(VmaStatistics& inoutStats) const
{
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    const VkDeviceSize size = get_size();
    const size_t suballoc1stCount = suballocations1st.size();
    const size_t suballoc2ndCount = suballocations2nd.size();

    inoutStats.blockCount++;
    inoutStats.blockBytes += size;
    inoutStats.allocationBytes += size - _sum_free_size;

    VkDeviceSize lastOffset = 0;

    if (_2nd_vector_mode == SECOND_VECTOR_RING_BUFFER)
    {
        const VkDeviceSize freeSpace2ndTo1stEnd = suballocations1st[_1st_null_items_begin_count].offset;
        size_t nextAlloc2ndIndex = _1st_null_items_begin_count;
        while (lastOffset < freeSpace2ndTo1stEnd)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while (nextAlloc2ndIndex < suballoc2ndCount &&
                suballocations2nd[nextAlloc2ndIndex].userData == VMA_NULL)
            {
                ++nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex < suballoc2ndCount)
            {
                const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                ++inoutStats.allocationCount;

                // Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                ++nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                // End of loop.
                lastOffset = freeSpace2ndTo1stEnd;
            }
        }
    }

    size_t nextAlloc1stIndex = _1st_null_items_begin_count;
    const VkDeviceSize freeSpace1stTo2ndEnd =
        _2nd_vector_mode == SECOND_VECTOR_DOUBLE_STACK ? suballocations2nd.back().offset : size;
    while (lastOffset < freeSpace1stTo2ndEnd)
    {
        // Find next non-null allocation or move nextAllocIndex to the end.
        while (nextAlloc1stIndex < suballoc1stCount &&
            suballocations1st[nextAlloc1stIndex].userData == VMA_NULL)
        {
            ++nextAlloc1stIndex;
        }

        // Found non-null allocation.
        if (nextAlloc1stIndex < suballoc1stCount)
        {
            const VmaSuballocation& suballoc = suballocations1st[nextAlloc1stIndex];

            // Process this allocation.
            // There is allocation with suballoc.offset, suballoc.size.
            ++inoutStats.allocationCount;

            // Prepare for next iteration.
            lastOffset = suballoc.offset + suballoc.size;
            ++nextAlloc1stIndex;
        }
        // We are at the end.
        else
        {
            // End of loop.
            lastOffset = freeSpace1stTo2ndEnd;
        }
    }

    if (_2nd_vector_mode == SECOND_VECTOR_DOUBLE_STACK)
    {
        size_t nextAlloc2ndIndex = suballocations2nd.size() - 1;
        while (lastOffset < size)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while (nextAlloc2ndIndex != SIZE_MAX &&
                suballocations2nd[nextAlloc2ndIndex].userData == VMA_NULL)
            {
                --nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex != SIZE_MAX)
            {
                const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                ++inoutStats.allocationCount;

                // Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                --nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                // End of loop.
                lastOffset = size;
            }
        }
    }
}

#if VMA_STATS_STRING_ENABLED
void VmaBlockMetadata_Linear::print_detailed_map(class VmaJsonWriter& json) const
{
    const VkDeviceSize size = get_size();
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    const size_t suballoc1stCount = suballocations1st.size();
    const size_t suballoc2ndCount = suballocations2nd.size();

    // FIRST PASS

    size_t unusedRangeCount = 0;
    VkDeviceSize usedBytes = 0;

    VkDeviceSize lastOffset = 0;

    size_t alloc2ndCount = 0;
    if (_2nd_vector_mode == SECOND_VECTOR_RING_BUFFER)
    {
        const VkDeviceSize freeSpace2ndTo1stEnd = suballocations1st[_1st_null_items_begin_count].offset;
        size_t nextAlloc2ndIndex = 0;
        while (lastOffset < freeSpace2ndTo1stEnd)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while (nextAlloc2ndIndex < suballoc2ndCount &&
                suballocations2nd[nextAlloc2ndIndex].userData == VMA_NULL)
            {
                ++nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex < suballoc2ndCount)
            {
                const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // 1. Process free space before this allocation.
                if (lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    ++unusedRangeCount;
                }

                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                ++alloc2ndCount;
                usedBytes += suballoc.size;

                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                ++nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                if (lastOffset < freeSpace2ndTo1stEnd)
                {
                    // There is free space from lastOffset to freeSpace2ndTo1stEnd.
                    ++unusedRangeCount;
                }

                // End of loop.
                lastOffset = freeSpace2ndTo1stEnd;
            }
        }
    }

    size_t nextAlloc1stIndex = _1st_null_items_begin_count;
    size_t alloc1stCount = 0;
    const VkDeviceSize freeSpace1stTo2ndEnd =
        _2nd_vector_mode == SECOND_VECTOR_DOUBLE_STACK ? suballocations2nd.back().offset : size;
    while (lastOffset < freeSpace1stTo2ndEnd)
    {
        // Find next non-null allocation or move nextAllocIndex to the end.
        while (nextAlloc1stIndex < suballoc1stCount &&
            suballocations1st[nextAlloc1stIndex].userData == VMA_NULL)
        {
            ++nextAlloc1stIndex;
        }

        // Found non-null allocation.
        if (nextAlloc1stIndex < suballoc1stCount)
        {
            const VmaSuballocation& suballoc = suballocations1st[nextAlloc1stIndex];

            // 1. Process free space before this allocation.
            if (lastOffset < suballoc.offset)
            {
                // There is free space from lastOffset to suballoc.offset.
                ++unusedRangeCount;
            }

            // 2. Process this allocation.
            // There is allocation with suballoc.offset, suballoc.size.
            ++alloc1stCount;
            usedBytes += suballoc.size;

            // 3. Prepare for next iteration.
            lastOffset = suballoc.offset + suballoc.size;
            ++nextAlloc1stIndex;
        }
        // We are at the end.
        else
        {
            if (lastOffset < freeSpace1stTo2ndEnd)
            {
                // There is free space from lastOffset to freeSpace1stTo2ndEnd.
                ++unusedRangeCount;
            }

            // End of loop.
            lastOffset = freeSpace1stTo2ndEnd;
        }
    }

    if (_2nd_vector_mode == SECOND_VECTOR_DOUBLE_STACK)
    {
        size_t nextAlloc2ndIndex = suballocations2nd.size() - 1;
        while (lastOffset < size)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while (nextAlloc2ndIndex != SIZE_MAX &&
                suballocations2nd[nextAlloc2ndIndex].userData == VMA_NULL)
            {
                --nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex != SIZE_MAX)
            {
                const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // 1. Process free space before this allocation.
                if (lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    ++unusedRangeCount;
                }

                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                ++alloc2ndCount;
                usedBytes += suballoc.size;

                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                --nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                if (lastOffset < size)
                {
                    // There is free space from lastOffset to size.
                    ++unusedRangeCount;
                }

                // End of loop.
                lastOffset = size;
            }
        }
    }

    const VkDeviceSize unusedBytes = size - usedBytes;
    print_detailed_map_begin(json, unusedBytes, alloc1stCount + alloc2ndCount, unusedRangeCount);

    // SECOND PASS
    lastOffset = 0;

    if (_2nd_vector_mode == SECOND_VECTOR_RING_BUFFER)
    {
        const VkDeviceSize freeSpace2ndTo1stEnd = suballocations1st[_1st_null_items_begin_count].offset;
        size_t nextAlloc2ndIndex = 0;
        while (lastOffset < freeSpace2ndTo1stEnd)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while (nextAlloc2ndIndex < suballoc2ndCount &&
                suballocations2nd[nextAlloc2ndIndex].userData == VMA_NULL)
            {
                ++nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex < suballoc2ndCount)
            {
                const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // 1. Process free space before this allocation.
                if (lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                    print_detailed_map_unused_range(json, lastOffset, unusedRangeSize);
                }

                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                print_detailed_map_allocation(json, suballoc.offset, suballoc.size, suballoc.userData);

                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                ++nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                if (lastOffset < freeSpace2ndTo1stEnd)
                {
                    // There is free space from lastOffset to freeSpace2ndTo1stEnd.
                    const VkDeviceSize unusedRangeSize = freeSpace2ndTo1stEnd - lastOffset;
                    print_detailed_map_unused_range(json, lastOffset, unusedRangeSize);
                }

                // End of loop.
                lastOffset = freeSpace2ndTo1stEnd;
            }
        }
    }

    nextAlloc1stIndex = _1st_null_items_begin_count;
    while (lastOffset < freeSpace1stTo2ndEnd)
    {
        // Find next non-null allocation or move nextAllocIndex to the end.
        while (nextAlloc1stIndex < suballoc1stCount &&
            suballocations1st[nextAlloc1stIndex].userData == VMA_NULL)
        {
            ++nextAlloc1stIndex;
        }

        // Found non-null allocation.
        if (nextAlloc1stIndex < suballoc1stCount)
        {
            const VmaSuballocation& suballoc = suballocations1st[nextAlloc1stIndex];

            // 1. Process free space before this allocation.
            if (lastOffset < suballoc.offset)
            {
                // There is free space from lastOffset to suballoc.offset.
                const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                print_detailed_map_unused_range(json, lastOffset, unusedRangeSize);
            }

            // 2. Process this allocation.
            // There is allocation with suballoc.offset, suballoc.size.
            print_detailed_map_allocation(json, suballoc.offset, suballoc.size, suballoc.userData);

            // 3. Prepare for next iteration.
            lastOffset = suballoc.offset + suballoc.size;
            ++nextAlloc1stIndex;
        }
        // We are at the end.
        else
        {
            if (lastOffset < freeSpace1stTo2ndEnd)
            {
                // There is free space from lastOffset to freeSpace1stTo2ndEnd.
                const VkDeviceSize unusedRangeSize = freeSpace1stTo2ndEnd - lastOffset;
                print_detailed_map_unused_range(json, lastOffset, unusedRangeSize);
            }

            // End of loop.
            lastOffset = freeSpace1stTo2ndEnd;
        }
    }

    if (_2nd_vector_mode == SECOND_VECTOR_DOUBLE_STACK)
    {
        size_t nextAlloc2ndIndex = suballocations2nd.size() - 1;
        while (lastOffset < size)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while (nextAlloc2ndIndex != SIZE_MAX &&
                suballocations2nd[nextAlloc2ndIndex].userData == VMA_NULL)
            {
                --nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex != SIZE_MAX)
            {
                const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // 1. Process free space before this allocation.
                if (lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                    print_detailed_map_unused_range(json, lastOffset, unusedRangeSize);
                }

                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                print_detailed_map_allocation(json, suballoc.offset, suballoc.size, suballoc.userData);

                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                --nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                if (lastOffset < size)
                {
                    // There is free space from lastOffset to size.
                    const VkDeviceSize unusedRangeSize = size - lastOffset;
                    print_detailed_map_unused_range(json, lastOffset, unusedRangeSize);
                }

                // End of loop.
                lastOffset = size;
            }
        }
    }

    print_detailed_map_end(json);
}
#endif // VMA_STATS_STRING_ENABLED

bool VmaBlockMetadata_Linear::create_allocation_request(
    VkDeviceSize allocSize,
    VkDeviceSize allocAlignment,
    bool upperAddress,
    VmaSuballocationType allocType,
    uint32_t strategy,
    VmaAllocationRequest* pAllocationRequest)
{
    VMA_ASSERT(allocSize > 0);
    VMA_ASSERT(allocType != VMA_SUBALLOCATION_TYPE_FREE);
    VMA_ASSERT(pAllocationRequest != VMA_NULL);
    VMA_HEAVY_ASSERT(validate());

    if(allocSize > get_size())
        return false;

    pAllocationRequest->size = allocSize;
    return upperAddress ?
        create_allocation_request_upper_address(
            allocSize, allocAlignment, allocType, strategy, pAllocationRequest) :
        create_allocation_request_lower_address(
            allocSize, allocAlignment, allocType, strategy, pAllocationRequest);
}

VkResult VmaBlockMetadata_Linear::check_corruption(const void* pBlockData)
{
    VMA_ASSERT(!is_virtual());
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    for (size_t i = _1st_null_items_begin_count, count = suballocations1st.size(); i < count; ++i)
    {
        const VmaSuballocation& suballoc = suballocations1st[i];
        if (suballoc.type != VMA_SUBALLOCATION_TYPE_FREE)
        {
            if (!VmaValidateMagicValue(pBlockData, suballoc.offset + suballoc.size))
            {
                VMA_ASSERT(0 && "MEMORY CORRUPTION DETECTED AFTER VALIDATED ALLOCATION!");
                return VK_ERROR_UNKNOWN_COPY;
            }
        }
    }

    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    for (size_t i = 0, count = suballocations2nd.size(); i < count; ++i)
    {
        const VmaSuballocation& suballoc = suballocations2nd[i];
        if (suballoc.type != VMA_SUBALLOCATION_TYPE_FREE)
        {
            if (!VmaValidateMagicValue(pBlockData, suballoc.offset + suballoc.size))
            {
                VMA_ASSERT(0 && "MEMORY CORRUPTION DETECTED AFTER VALIDATED ALLOCATION!");
                return VK_ERROR_UNKNOWN_COPY;
            }
        }
    }

    return VK_SUCCESS;
}

void VmaBlockMetadata_Linear::alloc(
    const VmaAllocationRequest& request,
    VmaSuballocationType type,
    void* userData)
{
    const VkDeviceSize offset = (VkDeviceSize)request.allocHandle - 1;
    const VmaSuballocation newSuballoc = { offset, request.size, userData, type };

    switch (request.type)
    {
    case VmaAllocationRequestType::UpperAddress:
    {
        VMA_ASSERT(_2nd_vector_mode != SECOND_VECTOR_RING_BUFFER &&
            "CRITICAL ERROR: Trying to use linear allocator as double stack while it was already used as ring buffer.");
        SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
        suballocations2nd.push_back(newSuballoc);
        _2nd_vector_mode = SECOND_VECTOR_DOUBLE_STACK;
    }
    break;
    case VmaAllocationRequestType::EndOf1st:
    {
        SuballocationVectorType& suballocations1st = AccessSuballocations1st();

        VMA_ASSERT(suballocations1st.empty() ||
            offset >= suballocations1st.back().offset + suballocations1st.back().size);
        // Check if it fits before the end of the block.
        VMA_ASSERT(offset + request.size <= get_size());

        suballocations1st.push_back(newSuballoc);
    }
    break;
    case VmaAllocationRequestType::EndOf2nd:
    {
        SuballocationVectorType& suballocations1st = AccessSuballocations1st();
        // New allocation at the end of 2-part ring buffer, so before first allocation from 1st vector.
        VMA_ASSERT(!suballocations1st.empty() &&
            offset + request.size <= suballocations1st[_1st_null_items_begin_count].offset);
        SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

        switch (_2nd_vector_mode)
        {
        case SECOND_VECTOR_EMPTY:
            // First allocation from second part ring buffer.
            VMA_ASSERT(suballocations2nd.empty());
            _2nd_vector_mode = SECOND_VECTOR_RING_BUFFER;
            break;
        case SECOND_VECTOR_RING_BUFFER:
            // 2-part ring buffer is already started.
            VMA_ASSERT(!suballocations2nd.empty());
            break;
        case SECOND_VECTOR_DOUBLE_STACK:
            VMA_ASSERT(0 && "CRITICAL ERROR: Trying to use linear allocator as ring buffer while it was already used as double stack.");
            break;
        default:
            VMA_ASSERT(0);
        }

        suballocations2nd.push_back(newSuballoc);
    }
    break;
    default:
        VMA_ASSERT(0 && "CRITICAL INTERNAL ERROR.");
    }

    _sum_free_size -= newSuballoc.size;
}

void VmaBlockMetadata_Linear::free(VmaAllocHandle allocHandle)
{
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    VkDeviceSize offset = (VkDeviceSize)allocHandle - 1;

    if (!suballocations1st.empty())
    {
        // First allocation: Mark it as next empty at the beginning.
        VmaSuballocation& firstSuballoc = suballocations1st[_1st_null_items_begin_count];
        if (firstSuballoc.offset == offset)
        {
            firstSuballoc.type = VMA_SUBALLOCATION_TYPE_FREE;
            firstSuballoc.userData = VMA_NULL;
            _sum_free_size += firstSuballoc.size;
            ++_1st_null_items_begin_count;
            cleanup_after_free();
            return;
        }
    }

    // Last allocation in 2-part ring buffer or top of upper stack (same logic).
    if (_2nd_vector_mode == SECOND_VECTOR_RING_BUFFER ||
        _2nd_vector_mode == SECOND_VECTOR_DOUBLE_STACK)
    {
        VmaSuballocation& lastSuballoc = suballocations2nd.back();
        if (lastSuballoc.offset == offset)
        {
            _sum_free_size += lastSuballoc.size;
            suballocations2nd.pop_back();
            cleanup_after_free();
            return;
        }
    }
    // Last allocation in 1st vector.
    else if (_2nd_vector_mode == SECOND_VECTOR_EMPTY)
    {
        VmaSuballocation& lastSuballoc = suballocations1st.back();
        if (lastSuballoc.offset == offset)
        {
            _sum_free_size += lastSuballoc.size;
            suballocations1st.pop_back();
            cleanup_after_free();
            return;
        }
    }

    VmaSuballocation refSuballoc;
    refSuballoc.offset = offset;
    // Rest of members stays uninitialized intentionally for better performance.

    // Item from the middle of 1st vector.
    {
        const SuballocationVectorType::iterator it = VmaBinaryFindSorted(
            suballocations1st.begin() + _1st_null_items_begin_count,
            suballocations1st.end(),
            refSuballoc,
            VmaSuballocationOffsetLess());
        if (it != suballocations1st.end())
        {
            it->type = VMA_SUBALLOCATION_TYPE_FREE;
            it->userData = VMA_NULL;
            ++_1st_null_items_middle_count;
            _sum_free_size += it->size;
            cleanup_after_free();
            return;
        }
    }

    if (_2nd_vector_mode != SECOND_VECTOR_EMPTY)
    {
        // Item from the middle of 2nd vector.
        const SuballocationVectorType::iterator it = _2nd_vector_mode == SECOND_VECTOR_RING_BUFFER ?
            VmaBinaryFindSorted(suballocations2nd.begin(), suballocations2nd.end(), refSuballoc, VmaSuballocationOffsetLess()) :
            VmaBinaryFindSorted(suballocations2nd.begin(), suballocations2nd.end(), refSuballoc, VmaSuballocationOffsetGreater());
        if (it != suballocations2nd.end())
        {
            it->type = VMA_SUBALLOCATION_TYPE_FREE;
            it->userData = VMA_NULL;
            ++_2nd_null_items_count;
            _sum_free_size += it->size;
            cleanup_after_free();
            return;
        }
    }

    VMA_ASSERT(0 && "Allocation to free not found in linear allocator!");
}

void VmaBlockMetadata_Linear::get_allocation_info(VmaAllocHandle allocHandle, VmaVirtualAllocationInfo& outInfo)
{
    outInfo.offset = (VkDeviceSize)allocHandle - 1;
    VmaSuballocation& suballoc = find_suballocation(outInfo.offset);
    outInfo.size = suballoc.size;
    outInfo.pUserData = suballoc.userData;
}

void* VmaBlockMetadata_Linear::get_allocation_user_data(VmaAllocHandle allocHandle) const
{
    return find_suballocation((VkDeviceSize)allocHandle - 1).userData;
}

VmaAllocHandle VmaBlockMetadata_Linear::get_allocation_list_begin() const
{
    // Function only used for defragmentation, which is disabled for this algorithm
    VMA_ASSERT(0);
    return VK_NULL_HANDLE;
}

VmaAllocHandle VmaBlockMetadata_Linear::get_next_allocation(VmaAllocHandle prevAlloc) const
{
    // Function only used for defragmentation, which is disabled for this algorithm
    VMA_ASSERT(0);
    return VK_NULL_HANDLE;
}

VkDeviceSize VmaBlockMetadata_Linear::get_next_free_region_size(VmaAllocHandle alloc) const
{
    // Function only used for defragmentation, which is disabled for this algorithm
    VMA_ASSERT(0);
    return 0;
}

void VmaBlockMetadata_Linear::clear()
{
    _sum_free_size = get_size();
    _suballocations0.clear();
    _suballocations1.clear();
    // Leaving _1st_vector_index unchanged - it doesn't matter.
    _2nd_vector_mode = SECOND_VECTOR_EMPTY;
    _1st_null_items_begin_count = 0;
    _1st_null_items_middle_count = 0;
    _2nd_null_items_count = 0;
}

void VmaBlockMetadata_Linear::set_allocation_user_data(VmaAllocHandle allocHandle, void* userData)
{
    VmaSuballocation& suballoc = find_suballocation((VkDeviceSize)allocHandle - 1);
    suballoc.userData = userData;
}

void VmaBlockMetadata_Linear::debug_log_all_allocations() const
{
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    for (auto it = suballocations1st.begin() + _1st_null_items_begin_count; it != suballocations1st.end(); ++it)
        if (it->type != VMA_SUBALLOCATION_TYPE_FREE)
            debug_log_allocation(it->offset, it->size, it->userData);

    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    for (auto it = suballocations2nd.begin(); it != suballocations2nd.end(); ++it)
        if (it->type != VMA_SUBALLOCATION_TYPE_FREE)
            debug_log_allocation(it->offset, it->size, it->userData);
}

VmaSuballocation& VmaBlockMetadata_Linear::find_suballocation(VkDeviceSize offset) const
{
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    VmaSuballocation refSuballoc;
    refSuballoc.offset = offset;
    // Rest of members stays uninitialized intentionally for better performance.

    // Item from the 1st vector.
    {
        SuballocationVectorType::const_iterator it = VmaBinaryFindSorted(
            suballocations1st.begin() + _1st_null_items_begin_count,
            suballocations1st.end(),
            refSuballoc,
            VmaSuballocationOffsetLess());
        if (it != suballocations1st.end())
        {
            return const_cast<VmaSuballocation&>(*it);
        }
    }

    if (_2nd_vector_mode != SECOND_VECTOR_EMPTY)
    {
        // Rest of members stays uninitialized intentionally for better performance.
        SuballocationVectorType::const_iterator it = _2nd_vector_mode == SECOND_VECTOR_RING_BUFFER ?
            VmaBinaryFindSorted(suballocations2nd.begin(), suballocations2nd.end(), refSuballoc, VmaSuballocationOffsetLess()) :
            VmaBinaryFindSorted(suballocations2nd.begin(), suballocations2nd.end(), refSuballoc, VmaSuballocationOffsetGreater());
        if (it != suballocations2nd.end())
        {
            return const_cast<VmaSuballocation&>(*it);
        }
    }

    VMA_ASSERT(0 && "Allocation not found in linear allocator!");
    return const_cast<VmaSuballocation&>(suballocations1st.back()); // Should never occur.
}

bool VmaBlockMetadata_Linear::should_compact_1st() const
{
    const size_t nullItemCount = _1st_null_items_begin_count + _1st_null_items_middle_count;
    const size_t suballocCount = AccessSuballocations1st().size();
    return suballocCount > 32 && nullItemCount * 2 >= (suballocCount - nullItemCount) * 3;
}

void VmaBlockMetadata_Linear::cleanup_after_free()
{
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    if (is_empty())
    {
        suballocations1st.clear();
        suballocations2nd.clear();
        _1st_null_items_begin_count = 0;
        _1st_null_items_middle_count = 0;
        _2nd_null_items_count = 0;
        _2nd_vector_mode = SECOND_VECTOR_EMPTY;
    }
    else
    {
        const size_t suballoc1stCount = suballocations1st.size();
        const size_t nullItem1stCount = _1st_null_items_begin_count + _1st_null_items_middle_count;
        VMA_ASSERT(nullItem1stCount <= suballoc1stCount);

        // Find more null items at the beginning of 1st vector.
        while (_1st_null_items_begin_count < suballoc1stCount &&
            suballocations1st[_1st_null_items_begin_count].type == VMA_SUBALLOCATION_TYPE_FREE)
        {
            ++_1st_null_items_begin_count;
            --_1st_null_items_middle_count;
        }

        // Find more null items at the end of 1st vector.
        while (_1st_null_items_middle_count > 0 &&
            suballocations1st.back().type == VMA_SUBALLOCATION_TYPE_FREE)
        {
            --_1st_null_items_middle_count;
            suballocations1st.pop_back();
        }

        // Find more null items at the end of 2nd vector.
        while (_2nd_null_items_count > 0 &&
            suballocations2nd.back().type == VMA_SUBALLOCATION_TYPE_FREE)
        {
            --_2nd_null_items_count;
            suballocations2nd.pop_back();
        }

        // Find more null items at the beginning of 2nd vector.
        while (_2nd_null_items_count > 0 &&
            suballocations2nd[0].type == VMA_SUBALLOCATION_TYPE_FREE)
        {
            --_2nd_null_items_count;
            VmaVectorRemove(suballocations2nd, 0);
        }

        if (should_compact_1st())
        {
            const size_t nonNullItemCount = suballoc1stCount - nullItem1stCount;
            size_t srcIndex = _1st_null_items_begin_count;
            for (size_t dstIndex = 0; dstIndex < nonNullItemCount; ++dstIndex)
            {
                while (suballocations1st[srcIndex].type == VMA_SUBALLOCATION_TYPE_FREE)
                {
                    ++srcIndex;
                }
                if (dstIndex != srcIndex)
                {
                    suballocations1st[dstIndex] = suballocations1st[srcIndex];
                }
                ++srcIndex;
            }
            suballocations1st.resize(nonNullItemCount);
            _1st_null_items_begin_count = 0;
            _1st_null_items_middle_count = 0;
        }

        // 2nd vector became empty.
        if (suballocations2nd.empty())
        {
            _2nd_vector_mode = SECOND_VECTOR_EMPTY;
        }

        // 1st vector became empty.
        if (suballocations1st.size() - _1st_null_items_begin_count == 0)
        {
            suballocations1st.clear();
            _1st_null_items_begin_count = 0;

            if (!suballocations2nd.empty() && _2nd_vector_mode == SECOND_VECTOR_RING_BUFFER)
            {
                // Swap 1st with 2nd. Now 2nd is empty.
                _2nd_vector_mode = SECOND_VECTOR_EMPTY;
                _1st_null_items_middle_count = _2nd_null_items_count;
                while (_1st_null_items_begin_count < suballocations2nd.size() &&
                    suballocations2nd[_1st_null_items_begin_count].type == VMA_SUBALLOCATION_TYPE_FREE)
                {
                    ++_1st_null_items_begin_count;
                    --_1st_null_items_middle_count;
                }
                _2nd_null_items_count = 0;
                _1st_vector_index ^= 1;
            }
        }
    }

    VMA_HEAVY_ASSERT(validate());
}

bool VmaBlockMetadata_Linear::create_allocation_request_lower_address(
    VkDeviceSize allocSize,
    VkDeviceSize allocAlignment,
    VmaSuballocationType allocType,
    uint32_t strategy,
    VmaAllocationRequest* pAllocationRequest)
{
    const VkDeviceSize blockSize = get_size();
    const VkDeviceSize debugMargin = get_debug_margin();
    const VkDeviceSize bufferImageGranularity = get_buffer_image_granularity();
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    if (_2nd_vector_mode == SECOND_VECTOR_EMPTY || _2nd_vector_mode == SECOND_VECTOR_DOUBLE_STACK)
    {
        // Try to allocate at the end of 1st vector.

        VkDeviceSize resultBaseOffset = 0;
        if (!suballocations1st.empty())
        {
            const VmaSuballocation& lastSuballoc = suballocations1st.back();
            resultBaseOffset = lastSuballoc.offset + lastSuballoc.size + debugMargin;
        }

        // Start from offset equal to beginning of free space.
        VkDeviceSize resultOffset = resultBaseOffset;

        // Apply alignment.
        resultOffset = VmaAlignUp(resultOffset, allocAlignment);

        // Check previous suballocations for BufferImageGranularity conflicts.
        // Make bigger alignment if necessary.
        if (bufferImageGranularity > 1 && bufferImageGranularity != allocAlignment && !suballocations1st.empty())
        {
            bool bufferImageGranularityConflict = false;
            for (size_t prevSuballocIndex = suballocations1st.size(); prevSuballocIndex--; )
            {
                const VmaSuballocation& prevSuballoc = suballocations1st[prevSuballocIndex];
                if (VmaBlocksOnSamePage(prevSuballoc.offset, prevSuballoc.size, resultOffset, bufferImageGranularity))
                {
                    if (VmaIsBufferImageGranularityConflict(prevSuballoc.type, allocType))
                    {
                        bufferImageGranularityConflict = true;
                        break;
                    }
                }
                else
                    // Already on previous page.
                    break;
            }
            if (bufferImageGranularityConflict)
            {
                resultOffset = VmaAlignUp(resultOffset, bufferImageGranularity);
            }
        }

        const VkDeviceSize freeSpaceEnd = _2nd_vector_mode == SECOND_VECTOR_DOUBLE_STACK ?
            suballocations2nd.back().offset : blockSize;

        // There is enough free space at the end after alignment.
        if (resultOffset + allocSize + debugMargin <= freeSpaceEnd)
        {
            // Check next suballocations for BufferImageGranularity conflicts.
            // If conflict exists, allocation cannot be made here.
            if ((allocSize % bufferImageGranularity || resultOffset % bufferImageGranularity) && _2nd_vector_mode == SECOND_VECTOR_DOUBLE_STACK)
            {
                for (size_t nextSuballocIndex = suballocations2nd.size(); nextSuballocIndex--; )
                {
                    const VmaSuballocation& nextSuballoc = suballocations2nd[nextSuballocIndex];
                    if (VmaBlocksOnSamePage(resultOffset, allocSize, nextSuballoc.offset, bufferImageGranularity))
                    {
                        if (VmaIsBufferImageGranularityConflict(allocType, nextSuballoc.type))
                        {
                            return false;
                        }
                    }
                    else
                    {
                        // Already on previous page.
                        break;
                    }
                }
            }

            // All tests passed: Success.
            pAllocationRequest->allocHandle = (VmaAllocHandle)(resultOffset + 1);
            // pAllocationRequest->item, customData unused.
            pAllocationRequest->type = VmaAllocationRequestType::EndOf1st;
            return true;
        }
    }

    // Wrap-around to end of 2nd vector. Try to allocate there, watching for the
    // beginning of 1st vector as the end of free space.
    if (_2nd_vector_mode == SECOND_VECTOR_EMPTY || _2nd_vector_mode == SECOND_VECTOR_RING_BUFFER)
    {
        VMA_ASSERT(!suballocations1st.empty());

        VkDeviceSize resultBaseOffset = 0;
        if (!suballocations2nd.empty())
        {
            const VmaSuballocation& lastSuballoc = suballocations2nd.back();
            resultBaseOffset = lastSuballoc.offset + lastSuballoc.size + debugMargin;
        }

        // Start from offset equal to beginning of free space.
        VkDeviceSize resultOffset = resultBaseOffset;

        // Apply alignment.
        resultOffset = VmaAlignUp(resultOffset, allocAlignment);

        // Check previous suballocations for BufferImageGranularity conflicts.
        // Make bigger alignment if necessary.
        if (bufferImageGranularity > 1 && bufferImageGranularity != allocAlignment && !suballocations2nd.empty())
        {
            bool bufferImageGranularityConflict = false;
            for (size_t prevSuballocIndex = suballocations2nd.size(); prevSuballocIndex--; )
            {
                const VmaSuballocation& prevSuballoc = suballocations2nd[prevSuballocIndex];
                if (VmaBlocksOnSamePage(prevSuballoc.offset, prevSuballoc.size, resultOffset, bufferImageGranularity))
                {
                    if (VmaIsBufferImageGranularityConflict(prevSuballoc.type, allocType))
                    {
                        bufferImageGranularityConflict = true;
                        break;
                    }
                }
                else
                    // Already on previous page.
                    break;
            }
            if (bufferImageGranularityConflict)
            {
                resultOffset = VmaAlignUp(resultOffset, bufferImageGranularity);
            }
        }

        size_t index1st = _1st_null_items_begin_count;

        // There is enough free space at the end after alignment.
        if ((index1st == suballocations1st.size() && resultOffset + allocSize + debugMargin <= blockSize) ||
            (index1st < suballocations1st.size() && resultOffset + allocSize + debugMargin <= suballocations1st[index1st].offset))
        {
            // Check next suballocations for BufferImageGranularity conflicts.
            // If conflict exists, allocation cannot be made here.
            if (allocSize % bufferImageGranularity || resultOffset % bufferImageGranularity)
            {
                for (size_t nextSuballocIndex = index1st;
                    nextSuballocIndex < suballocations1st.size();
                    nextSuballocIndex++)
                {
                    const VmaSuballocation& nextSuballoc = suballocations1st[nextSuballocIndex];
                    if (VmaBlocksOnSamePage(resultOffset, allocSize, nextSuballoc.offset, bufferImageGranularity))
                    {
                        if (VmaIsBufferImageGranularityConflict(allocType, nextSuballoc.type))
                        {
                            return false;
                        }
                    }
                    else
                    {
                        // Already on next page.
                        break;
                    }
                }
            }

            // All tests passed: Success.
            pAllocationRequest->allocHandle = (VmaAllocHandle)(resultOffset + 1);
            pAllocationRequest->type = VmaAllocationRequestType::EndOf2nd;
            // pAllocationRequest->item, customData unused.
            return true;
        }
    }

    return false;
}

bool VmaBlockMetadata_Linear::create_allocation_request_upper_address(
    VkDeviceSize allocSize,
    VkDeviceSize allocAlignment,
    VmaSuballocationType allocType,
    uint32_t strategy,
    VmaAllocationRequest* pAllocationRequest)
{
    const VkDeviceSize blockSize = get_size();
    const VkDeviceSize bufferImageGranularity = get_buffer_image_granularity();
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    if (_2nd_vector_mode == SECOND_VECTOR_RING_BUFFER)
    {
        VMA_ASSERT(0 && "Trying to use pool with linear algorithm as double stack, while it is already being used as ring buffer.");
        return false;
    }

    // Try to allocate before 2nd.back(), or end of block if 2nd.empty().
    if (allocSize > blockSize)
    {
        return false;
    }
    VkDeviceSize resultBaseOffset = blockSize - allocSize;
    if (!suballocations2nd.empty())
    {
        const VmaSuballocation& lastSuballoc = suballocations2nd.back();
        resultBaseOffset = lastSuballoc.offset - allocSize;
        if (allocSize > lastSuballoc.offset)
        {
            return false;
        }
    }

    // Start from offset equal to end of free space.
    VkDeviceSize resultOffset = resultBaseOffset;

    const VkDeviceSize debugMargin = get_debug_margin();

    // Apply debugMargin at the end.
    if (debugMargin > 0)
    {
        if (resultOffset < debugMargin)
        {
            return false;
        }
        resultOffset -= debugMargin;
    }

    // Apply alignment.
    resultOffset = VmaAlignDown(resultOffset, allocAlignment);

    // Check next suballocations from 2nd for BufferImageGranularity conflicts.
    // Make bigger alignment if necessary.
    if (bufferImageGranularity > 1 && bufferImageGranularity != allocAlignment && !suballocations2nd.empty())
    {
        bool bufferImageGranularityConflict = false;
        for (size_t nextSuballocIndex = suballocations2nd.size(); nextSuballocIndex--; )
        {
            const VmaSuballocation& nextSuballoc = suballocations2nd[nextSuballocIndex];
            if (VmaBlocksOnSamePage(resultOffset, allocSize, nextSuballoc.offset, bufferImageGranularity))
            {
                if (VmaIsBufferImageGranularityConflict(nextSuballoc.type, allocType))
                {
                    bufferImageGranularityConflict = true;
                    break;
                }
            }
            else
                // Already on previous page.
                break;
        }
        if (bufferImageGranularityConflict)
        {
            resultOffset = VmaAlignDown(resultOffset, bufferImageGranularity);
        }
    }

    // There is enough free space.
    const VkDeviceSize endOf1st = !suballocations1st.empty() ?
        suballocations1st.back().offset + suballocations1st.back().size :
        0;
    if (endOf1st + debugMargin <= resultOffset)
    {
        // Check previous suballocations for BufferImageGranularity conflicts.
        // If conflict exists, allocation cannot be made here.
        if (bufferImageGranularity > 1)
        {
            for (size_t prevSuballocIndex = suballocations1st.size(); prevSuballocIndex--; )
            {
                const VmaSuballocation& prevSuballoc = suballocations1st[prevSuballocIndex];
                if (VmaBlocksOnSamePage(prevSuballoc.offset, prevSuballoc.size, resultOffset, bufferImageGranularity))
                {
                    if (VmaIsBufferImageGranularityConflict(allocType, prevSuballoc.type))
                    {
                        return false;
                    }
                }
                else
                {
                    // Already on next page.
                    break;
                }
            }
        }

        // All tests passed: Success.
        pAllocationRequest->allocHandle = (VmaAllocHandle)(resultOffset + 1);
        // pAllocationRequest->item unused.
        pAllocationRequest->type = VmaAllocationRequestType::UpperAddress;
        return true;
    }

    return false;
}
#endif // _VMA_BLOCK_METADATA_LINEAR_FUNCTIONS
#endif // _VMA_BLOCK_METADATA_LINEAR

#ifndef _VMA_BLOCK_METADATA_TLSF
// To not search current larger region if first allocation won't succeed and skip to smaller range
// use with VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT as strategy in create_allocation_request().
// When fragmentation and reusal of previous blocks doesn't matter then use with
// VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT for fastest alloc time possible.
class VmaBlockMetadata_TLSF : public VmaBlockMetadata
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaBlockMetadata_TLSF)
public:
    VmaBlockMetadata_TLSF(const VkAllocationCallbacks* pAllocationCallbacks,
        VkDeviceSize bufferImageGranularity, bool isVirtual);
    ~VmaBlockMetadata_TLSF() override;

    size_t get_allocation_count() const override { return _alloc_count; }
    size_t get_free_regions_count() const override { return _blocks_free_count + 1; }
    VkDeviceSize get_sum_free_size() const override { return _blocks_free_size + _null_block->size; }
    bool is_empty() const override { return _null_block->offset == 0; }
    VkDeviceSize get_allocation_offset(VmaAllocHandle allocHandle) const override { return ((Block*)allocHandle)->offset; }

    void init(VkDeviceSize size) override;
    bool validate() const override;

    void add_detailed_statistics(VmaDetailedStatistics& inoutStats) const override;
    void add_statistics(VmaStatistics& inoutStats) const override;

#if VMA_STATS_STRING_ENABLED
    void print_detailed_map(class VmaJsonWriter& json) const override;
#endif

    bool create_allocation_request(
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        bool upperAddress,
        VmaSuballocationType allocType,
        uint32_t strategy,
        VmaAllocationRequest* pAllocationRequest) override;

    VkResult check_corruption(const void* pBlockData) override;
    void alloc(
        const VmaAllocationRequest& request,
        VmaSuballocationType type,
        void* userData) override;

    void free(VmaAllocHandle allocHandle) override;
    void get_allocation_info(VmaAllocHandle allocHandle, VmaVirtualAllocationInfo& outInfo) override;
    void* get_allocation_user_data(VmaAllocHandle allocHandle) const override;
    VmaAllocHandle get_allocation_list_begin() const override;
    VmaAllocHandle get_next_allocation(VmaAllocHandle prevAlloc) const override;
    VkDeviceSize get_next_free_region_size(VmaAllocHandle alloc) const override;
    void clear() override;
    void set_allocation_user_data(VmaAllocHandle allocHandle, void* userData) override;
    void debug_log_all_allocations() const override;

private:
    // According to original paper it should be preferable 4 or 5:
    // M. Masmano, I. Ripoll, A. Crespo, and J. Real "TLSF: a New Dynamic Memory Allocator for Real-Time Systems"
    // http://www.gii.upv.es/tlsf/files/ecrts04_tlsf.pdf
    static const uint8_t SECOND_LEVEL_INDEX = 5;
    static const uint16_t SMALL_BUFFER_SIZE = 256;
    static const uint32_t INITIAL_BLOCK_ALLOC_COUNT = 16;
    static const uint8_t MEMORY_CLASS_SHIFT = 7;
    static const uint8_t MAX_MEMORY_CLASSES = 65 - MEMORY_CLASS_SHIFT;

    class Block
    {
    public:
        VkDeviceSize offset;
        VkDeviceSize size;
        Block* prevPhysical;
        Block* nextPhysical;

        void mark_free() { prevFree = VMA_NULL; }
        void mark_taken() { prevFree = this; }
        bool is_free() const { return prevFree != this; }
        void*& user_data() { VMA_HEAVY_ASSERT(!is_free()); return userData; }
        Block*& prev_free() { return prevFree; }
        Block*& next_free() { VMA_HEAVY_ASSERT(is_free()); return nextFree; }

    private:
        Block* prevFree; // Address of the same block here indicates that block is taken
        union
        {
            Block* nextFree;
            void* userData;
        };
    };

    size_t _alloc_count;
    // Total number of free blocks besides null block
    size_t _blocks_free_count;
    // Total size of free blocks excluding null block
    VkDeviceSize _blocks_free_size;
    uint32_t _is_free_bitmap;
    uint8_t _memory_classes;
    uint32_t _inner_is_free_bitmap[MAX_MEMORY_CLASSES];
    uint32_t _lists_count;
    /*
    * 0: 0-3 lists for small buffers
    * 1+: 0-(2^SLI-1) lists for normal buffers
    */
    Block** _free_list;
    VmaPoolAllocator<Block> _block_allocator;
    Block* _null_block;
    VmaBlockBufferImageGranularity _granularity_handler;

    static uint8_t size_to_memory_class(VkDeviceSize size);
    uint16_t size_to_second_index(VkDeviceSize size, uint8_t memoryClass) const;
    uint32_t get_list_index(uint8_t memoryClass, uint16_t secondIndex) const;
    uint32_t get_list_index(VkDeviceSize size) const;

    void remove_free_block(Block* block);
    void insert_free_block(Block* block);
    void merge_block(Block* block, Block* prev);

    Block* find_free_block(VkDeviceSize size, uint32_t& listIndex) const;
    bool check_block(
        Block& block,
        uint32_t listIndex,
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        VmaSuballocationType allocType,
        VmaAllocationRequest* pAllocationRequest);
};

#ifndef _VMA_BLOCK_METADATA_TLSF_FUNCTIONS
VmaBlockMetadata_TLSF::VmaBlockMetadata_TLSF(const VkAllocationCallbacks* pAllocationCallbacks,
    VkDeviceSize bufferImageGranularity, bool isVirtual)
    : VmaBlockMetadata(pAllocationCallbacks, bufferImageGranularity, isVirtual),
    _alloc_count(0),
    _blocks_free_count(0),
    _blocks_free_size(0),
    _is_free_bitmap(0),
    _memory_classes(0),
    _lists_count(0),
    _free_list(VMA_NULL),
    _block_allocator(pAllocationCallbacks, INITIAL_BLOCK_ALLOC_COUNT),
    _null_block(VMA_NULL),
    _granularity_handler(bufferImageGranularity) {}

VmaBlockMetadata_TLSF::~VmaBlockMetadata_TLSF()
{
    if (_free_list)
        vma_delete_array(get_allocation_callbacks(), _free_list, _lists_count);
    _granularity_handler.destroy(get_allocation_callbacks());
}

void VmaBlockMetadata_TLSF::init(VkDeviceSize size)
{
    VmaBlockMetadata::init(size);

    if (!is_virtual())
        _granularity_handler.init(get_allocation_callbacks(), size);

    _null_block = _block_allocator.alloc();
    _null_block->size = size;
    _null_block->offset = 0;
    _null_block->prevPhysical = VMA_NULL;
    _null_block->nextPhysical = VMA_NULL;
    _null_block->mark_free();
    _null_block->next_free() = VMA_NULL;
    _null_block->prev_free() = VMA_NULL;
    uint8_t memoryClass = size_to_memory_class(size);
    uint16_t sli = size_to_second_index(size, memoryClass);
    _lists_count = (memoryClass == 0 ? 0 : (memoryClass - 1) * (1UL << SECOND_LEVEL_INDEX) + sli) + 1;
    if (is_virtual())
        _lists_count += 1UL << SECOND_LEVEL_INDEX;
    else
        _lists_count += 4;

    _memory_classes = memoryClass + uint8_t(2);
    memset(_inner_is_free_bitmap, 0, MAX_MEMORY_CLASSES * sizeof(uint32_t));

    _free_list = vma_new_array(get_allocation_callbacks(), Block*, _lists_count);
    memset(_free_list, 0, _lists_count * sizeof(Block*));
}

bool VmaBlockMetadata_TLSF::validate() const
{
    VMA_VALIDATE(get_sum_free_size() <= get_size());

    VkDeviceSize calculatedSize = _null_block->size;
    VkDeviceSize calculatedFreeSize = _null_block->size;
    size_t allocCount = 0;
    size_t freeCount = 0;

    // Check integrity of free lists
    for (uint32_t list = 0; list < _lists_count; ++list)
    {
        Block* block = _free_list[list];
        if (block != VMA_NULL)
        {
            VMA_VALIDATE(block->is_free());
            VMA_VALIDATE(block->prev_free() == VMA_NULL);
            while (block->next_free())
            {
                VMA_VALIDATE(block->next_free()->is_free());
                VMA_VALIDATE(block->next_free()->prev_free() == block);
                block = block->next_free();
            }
        }
    }

    VkDeviceSize nextOffset = _null_block->offset;
    auto validateCtx = _granularity_handler.start_validation(get_allocation_callbacks(), is_virtual());

    VMA_VALIDATE(_null_block->nextPhysical == VMA_NULL);
    if (_null_block->prevPhysical)
    {
        VMA_VALIDATE(_null_block->prevPhysical->nextPhysical == _null_block);
    }
    // Check all blocks
    for (Block* prev = _null_block->prevPhysical; prev != VMA_NULL; prev = prev->prevPhysical)
    {
        VMA_VALIDATE(prev->offset + prev->size == nextOffset);
        nextOffset = prev->offset;
        calculatedSize += prev->size;

        uint32_t listIndex = get_list_index(prev->size);
        if (prev->is_free())
        {
            ++freeCount;
            // Check if free block belongs to free list
            Block* freeBlock = _free_list[listIndex];
            VMA_VALIDATE(freeBlock != VMA_NULL);

            bool found = false;
            do
            {
                if (freeBlock == prev)
                    found = true;

                freeBlock = freeBlock->next_free();
            } while (!found && freeBlock != VMA_NULL);

            VMA_VALIDATE(found);
            calculatedFreeSize += prev->size;
        }
        else
        {
            ++allocCount;
            // Check if taken block is not on a free list
            Block* freeBlock = _free_list[listIndex];
            while (freeBlock)
            {
                VMA_VALIDATE(freeBlock != prev);
                freeBlock = freeBlock->next_free();
            }

            if (!is_virtual())
            {
                VMA_VALIDATE(_granularity_handler.validate(validateCtx, prev->offset, prev->size));
            }
        }

        if (prev->prevPhysical)
        {
            VMA_VALIDATE(prev->prevPhysical->nextPhysical == prev);
        }
    }

    if (!is_virtual())
    {
        VMA_VALIDATE(_granularity_handler.finish_validation(validateCtx));
    }

    VMA_VALIDATE(nextOffset == 0);
    VMA_VALIDATE(calculatedSize == get_size());
    VMA_VALIDATE(calculatedFreeSize == get_sum_free_size());
    VMA_VALIDATE(allocCount == _alloc_count);
    VMA_VALIDATE(freeCount == _blocks_free_count);

    return true;
}

void VmaBlockMetadata_TLSF::add_detailed_statistics(VmaDetailedStatistics& inoutStats) const
{
    inoutStats.statistics.blockCount++;
    inoutStats.statistics.blockBytes += get_size();
    if (_null_block->size > 0)
        VmaAddDetailedStatisticsUnusedRange(inoutStats, _null_block->size);

    for (Block* block = _null_block->prevPhysical; block != VMA_NULL; block = block->prevPhysical)
    {
        if (block->is_free())
            VmaAddDetailedStatisticsUnusedRange(inoutStats, block->size);
        else
            VmaAddDetailedStatisticsAllocation(inoutStats, block->size);
    }
}

void VmaBlockMetadata_TLSF::add_statistics(VmaStatistics& inoutStats) const
{
    inoutStats.blockCount++;
    inoutStats.allocationCount += (uint32_t)_alloc_count;
    inoutStats.blockBytes += get_size();
    inoutStats.allocationBytes += get_size() - get_sum_free_size();
}

#if VMA_STATS_STRING_ENABLED
void VmaBlockMetadata_TLSF::print_detailed_map(class VmaJsonWriter& json) const
{
    size_t blockCount = _alloc_count + _blocks_free_count;
    VmaStlAllocator<Block*> allocator(get_allocation_callbacks());
    VmaVector<Block*, VmaStlAllocator<Block*>> blockList(blockCount, allocator);

    size_t i = blockCount;
    for (Block* block = _null_block->prevPhysical; block != VMA_NULL; block = block->prevPhysical)
    {
        blockList[--i] = block;
    }
    VMA_ASSERT(i == 0);

    VmaDetailedStatistics stats;
    VmaClearDetailedStatistics(stats);
    add_detailed_statistics(stats);

    print_detailed_map_begin(json,
        stats.statistics.blockBytes - stats.statistics.allocationBytes,
        stats.statistics.allocationCount,
        stats.unusedRangeCount);

    for (; i < blockCount; ++i)
    {
        Block* block = blockList[i];
        if (block->is_free())
            print_detailed_map_unused_range(json, block->offset, block->size);
        else
            print_detailed_map_allocation(json, block->offset, block->size, block->user_data());
    }
    if (_null_block->size > 0)
        print_detailed_map_unused_range(json, _null_block->offset, _null_block->size);

    print_detailed_map_end(json);
}
#endif

bool VmaBlockMetadata_TLSF::create_allocation_request(
    VkDeviceSize allocSize,
    VkDeviceSize allocAlignment,
    bool upperAddress,
    VmaSuballocationType allocType,
    uint32_t strategy,
    VmaAllocationRequest* pAllocationRequest)
{
    VMA_ASSERT(allocSize > 0 && "Cannot allocate empty block!");
    VMA_ASSERT(!upperAddress && "VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT can be used only with linear algorithm.");

    // For small granularity round up
    if (!is_virtual())
        _granularity_handler.roundup_alloc_request(allocType, allocSize, allocAlignment);

    allocSize += get_debug_margin();
    // Quick check for too small pool
    if (allocSize > get_sum_free_size())
        return false;

    // If no free blocks in pool then check only null block
    if (_blocks_free_count == 0)
        return check_block(*_null_block, _lists_count, allocSize, allocAlignment, allocType, pAllocationRequest);

    // Round up to the next block
    VkDeviceSize sizeForNextList = allocSize;
    VkDeviceSize smallSizeStep = VkDeviceSize(SMALL_BUFFER_SIZE / (is_virtual() ? 1U << SECOND_LEVEL_INDEX : 4U));
    if (allocSize > SMALL_BUFFER_SIZE)
    {
        sizeForNextList += (1ULL << (VMA_BITSCAN_MSB(allocSize) - SECOND_LEVEL_INDEX));
    }
    else if (allocSize > SMALL_BUFFER_SIZE - smallSizeStep)
        sizeForNextList = SMALL_BUFFER_SIZE + 1;
    else
        sizeForNextList += smallSizeStep;

    uint32_t nextListIndex = _lists_count;
    uint32_t prevListIndex = _lists_count;
    Block* nextListBlock = VMA_NULL;
    Block* prevListBlock = VMA_NULL;

    // Check blocks according to strategies
    if (strategy & VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT)
    {
        // Quick check for larger block first
        nextListBlock = find_free_block(sizeForNextList, nextListIndex);
        if (nextListBlock != VMA_NULL && check_block(*nextListBlock, nextListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
            return true;

        // If not fitted then null block
        if (check_block(*_null_block, _lists_count, allocSize, allocAlignment, allocType, pAllocationRequest))
            return true;

        // Null block failed, search larger bucket
        while (nextListBlock)
        {
            if (check_block(*nextListBlock, nextListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
                return true;
            nextListBlock = nextListBlock->next_free();
        }

        // Failed again, check best fit bucket
        prevListBlock = find_free_block(allocSize, prevListIndex);
        while (prevListBlock)
        {
            if (check_block(*prevListBlock, prevListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
                return true;
            prevListBlock = prevListBlock->next_free();
        }
    }
    else if (strategy & VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT)
    {
        // Check best fit bucket
        prevListBlock = find_free_block(allocSize, prevListIndex);
        while (prevListBlock)
        {
            if (check_block(*prevListBlock, prevListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
                return true;
            prevListBlock = prevListBlock->next_free();
        }

        // If failed check null block
        if (check_block(*_null_block, _lists_count, allocSize, allocAlignment, allocType, pAllocationRequest))
            return true;

        // Check larger bucket
        nextListBlock = find_free_block(sizeForNextList, nextListIndex);
        while (nextListBlock)
        {
            if (check_block(*nextListBlock, nextListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
                return true;
            nextListBlock = nextListBlock->next_free();
        }
    }
    else if (strategy & VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT )
    {
        // Perform search from the start
        VmaStlAllocator<Block*> allocator(get_allocation_callbacks());
        VmaVector<Block*, VmaStlAllocator<Block*>> blockList(_blocks_free_count, allocator);

        size_t i = _blocks_free_count;
        for (Block* block = _null_block->prevPhysical; block != VMA_NULL; block = block->prevPhysical)
        {
            if (block->is_free() && block->size >= allocSize)
                blockList[--i] = block;
        }

        for (; i < _blocks_free_count; ++i)
        {
            Block& block = *blockList[i];
            if (check_block(block, get_list_index(block.size), allocSize, allocAlignment, allocType, pAllocationRequest))
                return true;
        }

        // If failed check null block
        if (check_block(*_null_block, _lists_count, allocSize, allocAlignment, allocType, pAllocationRequest))
            return true;

        // Whole range searched, no more memory
        return false;
    }
    else
    {
        // Check larger bucket
        nextListBlock = find_free_block(sizeForNextList, nextListIndex);
        while (nextListBlock)
        {
            if (check_block(*nextListBlock, nextListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
                return true;
            nextListBlock = nextListBlock->next_free();
        }

        // If failed check null block
        if (check_block(*_null_block, _lists_count, allocSize, allocAlignment, allocType, pAllocationRequest))
            return true;

        // Check best fit bucket
        prevListBlock = find_free_block(allocSize, prevListIndex);
        while (prevListBlock)
        {
            if (check_block(*prevListBlock, prevListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
                return true;
            prevListBlock = prevListBlock->next_free();
        }
    }

    // Worst case, full search has to be done
    while (++nextListIndex < _lists_count)
    {
        nextListBlock = _free_list[nextListIndex];
        while (nextListBlock)
        {
            if (check_block(*nextListBlock, nextListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
                return true;
            nextListBlock = nextListBlock->next_free();
        }
    }

    // No more memory sadly
    return false;
}

VkResult VmaBlockMetadata_TLSF::check_corruption(const void* pBlockData)
{
    for (Block* block = _null_block->prevPhysical; block != VMA_NULL; block = block->prevPhysical)
    {
        if (!block->is_free())
        {
            if (!VmaValidateMagicValue(pBlockData, block->offset + block->size))
            {
                VMA_ASSERT(0 && "MEMORY CORRUPTION DETECTED AFTER VALIDATED ALLOCATION!");
                return VK_ERROR_UNKNOWN_COPY;
            }
        }
    }

    return VK_SUCCESS;
}

void VmaBlockMetadata_TLSF::alloc(
    const VmaAllocationRequest& request,
    VmaSuballocationType type,
    void* userData)
{
    VMA_ASSERT(request.type == VmaAllocationRequestType::TLSF);

    // Get block and pop it from the free list
    Block* currentBlock = (Block*)request.allocHandle;
    VkDeviceSize offset = request.algorithmData;
    VMA_ASSERT(currentBlock != VMA_NULL);
    VMA_ASSERT(currentBlock->offset <= offset);

    if (currentBlock != _null_block)
        remove_free_block(currentBlock);

    VkDeviceSize debugMargin = get_debug_margin();
    VkDeviceSize missingAlignment = offset - currentBlock->offset;

    // Append missing alignment to prev block or create new one
    if (missingAlignment)
    {
        Block* prevBlock = currentBlock->prevPhysical;
        VMA_ASSERT(prevBlock != VMA_NULL && "There should be no missing alignment at offset 0!");

        if (prevBlock->is_free() && prevBlock->size != debugMargin)
        {
            uint32_t oldList = get_list_index(prevBlock->size);
            prevBlock->size += missingAlignment;
            // Check if new size crosses list bucket
            if (oldList != get_list_index(prevBlock->size))
            {
                prevBlock->size -= missingAlignment;
                remove_free_block(prevBlock);
                prevBlock->size += missingAlignment;
                insert_free_block(prevBlock);
            }
            else
                _blocks_free_size += missingAlignment;
        }
        else
        {
            Block* newBlock = _block_allocator.alloc();
            currentBlock->prevPhysical = newBlock;
            prevBlock->nextPhysical = newBlock;
            newBlock->prevPhysical = prevBlock;
            newBlock->nextPhysical = currentBlock;
            newBlock->size = missingAlignment;
            newBlock->offset = currentBlock->offset;
            newBlock->mark_taken();

            insert_free_block(newBlock);
        }

        currentBlock->size -= missingAlignment;
        currentBlock->offset += missingAlignment;
    }

    VkDeviceSize size = request.size + debugMargin;
    if (currentBlock->size == size)
    {
        if (currentBlock == _null_block)
        {
            // Setup new null block
            _null_block = _block_allocator.alloc();
            _null_block->size = 0;
            _null_block->offset = currentBlock->offset + size;
            _null_block->prevPhysical = currentBlock;
            _null_block->nextPhysical = VMA_NULL;
            _null_block->mark_free();
            _null_block->prev_free() = VMA_NULL;
            _null_block->next_free() = VMA_NULL;
            currentBlock->nextPhysical = _null_block;
            currentBlock->mark_taken();
        }
    }
    else
    {
        VMA_ASSERT(currentBlock->size > size && "Proper block already found, shouldn't find smaller one!");

        // Create new free block
        Block* newBlock = _block_allocator.alloc();
        newBlock->size = currentBlock->size - size;
        newBlock->offset = currentBlock->offset + size;
        newBlock->prevPhysical = currentBlock;
        newBlock->nextPhysical = currentBlock->nextPhysical;
        currentBlock->nextPhysical = newBlock;
        currentBlock->size = size;

        if (currentBlock == _null_block)
        {
            _null_block = newBlock;
            _null_block->mark_free();
            _null_block->next_free() = VMA_NULL;
            _null_block->prev_free() = VMA_NULL;
            currentBlock->mark_taken();
        }
        else
        {
            newBlock->nextPhysical->prevPhysical = newBlock;
            newBlock->mark_taken();
            insert_free_block(newBlock);
        }
    }
    currentBlock->user_data() = userData;

    if (debugMargin > 0)
    {
        currentBlock->size -= debugMargin;
        Block* newBlock = _block_allocator.alloc();
        newBlock->size = debugMargin;
        newBlock->offset = currentBlock->offset + currentBlock->size;
        newBlock->prevPhysical = currentBlock;
        newBlock->nextPhysical = currentBlock->nextPhysical;
        newBlock->mark_taken();
        currentBlock->nextPhysical->prevPhysical = newBlock;
        currentBlock->nextPhysical = newBlock;
        insert_free_block(newBlock);
    }

    if (!is_virtual())
        _granularity_handler.alloc_pages((uint8_t)(uintptr_t)request.customData,
            currentBlock->offset, currentBlock->size);
    ++_alloc_count;
}

void VmaBlockMetadata_TLSF::free(VmaAllocHandle allocHandle)
{
    Block* block = (Block*)allocHandle;
    Block* next = block->nextPhysical;
    VMA_ASSERT(!block->is_free() && "Block is already free!");

    if (!is_virtual())
        _granularity_handler.free_pages(block->offset, block->size);
    --_alloc_count;

    VkDeviceSize debugMargin = get_debug_margin();
    if (debugMargin > 0)
    {
        remove_free_block(next);
        merge_block(next, block);
        block = next;
        next = next->nextPhysical;
    }

    // Try merging
    Block* prev = block->prevPhysical;
    if (prev != VMA_NULL && prev->is_free() && prev->size != debugMargin)
    {
        remove_free_block(prev);
        merge_block(block, prev);
    }

    if (!next->is_free())
        insert_free_block(block);
    else if (next == _null_block)
        merge_block(_null_block, block);
    else
    {
        remove_free_block(next);
        merge_block(next, block);
        insert_free_block(next);
    }
}

void VmaBlockMetadata_TLSF::get_allocation_info(VmaAllocHandle allocHandle, VmaVirtualAllocationInfo& outInfo)
{
    Block* block = (Block*)allocHandle;
    VMA_ASSERT(!block->is_free() && "Cannot get allocation info for free block!");
    outInfo.offset = block->offset;
    outInfo.size = block->size;
    outInfo.pUserData = block->user_data();
}

void* VmaBlockMetadata_TLSF::get_allocation_user_data(VmaAllocHandle allocHandle) const
{
    Block* block = (Block*)allocHandle;
    VMA_ASSERT(!block->is_free() && "Cannot get user data for free block!");
    return block->user_data();
}

VmaAllocHandle VmaBlockMetadata_TLSF::get_allocation_list_begin() const
{
    if (_alloc_count == 0)
        return VK_NULL_HANDLE;

    for (Block* block = _null_block->prevPhysical; block; block = block->prevPhysical)
    {
        if (!block->is_free())
            return (VmaAllocHandle)block;
    }
    VMA_ASSERT(false && "If _alloc_count > 0 then should find any allocation!");
    return VK_NULL_HANDLE;
}

VmaAllocHandle VmaBlockMetadata_TLSF::get_next_allocation(VmaAllocHandle prevAlloc) const
{
    Block* startBlock = (Block*)prevAlloc;
    VMA_ASSERT(!startBlock->is_free() && "Incorrect block!");

    for (Block* block = startBlock->prevPhysical; block; block = block->prevPhysical)
    {
        if (!block->is_free())
            return (VmaAllocHandle)block;
    }
    return VK_NULL_HANDLE;
}

VkDeviceSize VmaBlockMetadata_TLSF::get_next_free_region_size(VmaAllocHandle alloc) const
{
    Block* block = (Block*)alloc;
    VMA_ASSERT(!block->is_free() && "Incorrect block!");

    if (block->prevPhysical)
        return block->prevPhysical->is_free() ? block->prevPhysical->size : 0;
    return 0;
}

void VmaBlockMetadata_TLSF::clear()
{
    _alloc_count = 0;
    _blocks_free_count = 0;
    _blocks_free_size = 0;
    _is_free_bitmap = 0;
    _null_block->offset = 0;
    _null_block->size = get_size();
    Block* block = _null_block->prevPhysical;
    _null_block->prevPhysical = VMA_NULL;
    while (block)
    {
        Block* prev = block->prevPhysical;
        _block_allocator.free(block);
        block = prev;
    }
    memset(_free_list, 0, _lists_count * sizeof(Block*));
    memset(_inner_is_free_bitmap, 0, _memory_classes * sizeof(uint32_t));
    _granularity_handler.clear();
}

void VmaBlockMetadata_TLSF::set_allocation_user_data(VmaAllocHandle allocHandle, void* userData)
{
    Block* block = (Block*)allocHandle;
    VMA_ASSERT(!block->is_free() && "Trying to set user data for not allocated block!");
    block->user_data() = userData;
}

void VmaBlockMetadata_TLSF::debug_log_all_allocations() const
{
    for (Block* block = _null_block->prevPhysical; block != VMA_NULL; block = block->prevPhysical)
        if (!block->is_free())
            debug_log_allocation(block->offset, block->size, block->user_data());
}

uint8_t VmaBlockMetadata_TLSF::size_to_memory_class(VkDeviceSize size)
{
    if (size > SMALL_BUFFER_SIZE)
        return uint8_t(VMA_BITSCAN_MSB(size) - MEMORY_CLASS_SHIFT);
    return 0;
}

uint16_t VmaBlockMetadata_TLSF::size_to_second_index(VkDeviceSize size, uint8_t memoryClass) const
{
    if (memoryClass == 0)
    {
        if (is_virtual())
            return static_cast<uint16_t>((size - 1) / 8);
        return static_cast<uint16_t>((size - 1) / 64);
    }
    return static_cast<uint16_t>((size >> (memoryClass + MEMORY_CLASS_SHIFT - SECOND_LEVEL_INDEX)) ^ (1U << SECOND_LEVEL_INDEX));
}

uint32_t VmaBlockMetadata_TLSF::get_list_index(uint8_t memoryClass, uint16_t secondIndex) const
{
    if (memoryClass == 0)
        return secondIndex;

    const uint32_t index = static_cast<uint32_t>(memoryClass - 1) * (1 << SECOND_LEVEL_INDEX) + secondIndex;
    if (is_virtual())
        return index + (1 << SECOND_LEVEL_INDEX);
    return index + 4;
}

uint32_t VmaBlockMetadata_TLSF::get_list_index(VkDeviceSize size) const
{
    uint8_t memoryClass = size_to_memory_class(size);
    return get_list_index(memoryClass, size_to_second_index(size, memoryClass));
}

void VmaBlockMetadata_TLSF::remove_free_block(Block* block)
{
    VMA_ASSERT(block != _null_block);
    VMA_ASSERT(block->is_free());

    if (block->next_free() != VMA_NULL)
        block->next_free()->prev_free() = block->prev_free();
    if (block->prev_free() != VMA_NULL)
        block->prev_free()->next_free() = block->next_free();
    else
    {
        uint8_t memClass = size_to_memory_class(block->size);
        uint16_t secondIndex = size_to_second_index(block->size, memClass);
        uint32_t index = get_list_index(memClass, secondIndex);
        VMA_ASSERT(_free_list[index] == block);
        _free_list[index] = block->next_free();
        if (block->next_free() == VMA_NULL)
        {
            _inner_is_free_bitmap[memClass] &= ~(1U << secondIndex);
            if (_inner_is_free_bitmap[memClass] == 0)
                _is_free_bitmap &= ~(1UL << memClass);
        }
    }
    block->mark_taken();
    block->user_data() = VMA_NULL;
    --_blocks_free_count;
    _blocks_free_size -= block->size;
}

void VmaBlockMetadata_TLSF::insert_free_block(Block* block)
{
    VMA_ASSERT(block != _null_block);
    VMA_ASSERT(!block->is_free() && "Cannot insert block twice!");

    uint8_t memClass = size_to_memory_class(block->size);
    uint16_t secondIndex = size_to_second_index(block->size, memClass);
    uint32_t index = get_list_index(memClass, secondIndex);
    VMA_ASSERT(index < _lists_count);
    block->prev_free() = VMA_NULL;
    block->next_free() = _free_list[index];
    _free_list[index] = block;
    if (block->next_free() != VMA_NULL)
        block->next_free()->prev_free() = block;
    else
    {
        _inner_is_free_bitmap[memClass] |= 1U << secondIndex;
        _is_free_bitmap |= 1UL << memClass;
    }
    ++_blocks_free_count;
    _blocks_free_size += block->size;
}

void VmaBlockMetadata_TLSF::merge_block(Block* block, Block* prev)
{
    VMA_ASSERT(block->prevPhysical == prev && "Cannot merge separate physical regions!");
    VMA_ASSERT(!prev->is_free() && "Cannot merge block that belongs to free list!");

    block->offset = prev->offset;
    block->size += prev->size;
    block->prevPhysical = prev->prevPhysical;
    if (block->prevPhysical)
        block->prevPhysical->nextPhysical = block;
    _block_allocator.free(prev);
}

VmaBlockMetadata_TLSF::Block* VmaBlockMetadata_TLSF::find_free_block(VkDeviceSize size, uint32_t& listIndex) const
{
    uint8_t memoryClass = size_to_memory_class(size);
    uint32_t innerFreeMap = _inner_is_free_bitmap[memoryClass] & (~0U << size_to_second_index(size, memoryClass));
    if (!innerFreeMap)
    {
        // Check higher levels for available blocks
        uint32_t freeMap = _is_free_bitmap & (~0UL << (memoryClass + 1));
        if (!freeMap)
            return VMA_NULL; // No more memory available

        // Find lowest free region
        memoryClass = VMA_BITSCAN_LSB(freeMap);
        innerFreeMap = _inner_is_free_bitmap[memoryClass];
        VMA_ASSERT(innerFreeMap != 0);
    }
    // Find lowest free subregion
    listIndex = get_list_index(memoryClass, VMA_BITSCAN_LSB(innerFreeMap));
    VMA_ASSERT(_free_list[listIndex]);
    return _free_list[listIndex];
}

bool VmaBlockMetadata_TLSF::check_block(
    Block& block,
    uint32_t listIndex,
    VkDeviceSize allocSize,
    VkDeviceSize allocAlignment,
    VmaSuballocationType allocType,
    VmaAllocationRequest* pAllocationRequest)
{
    VMA_ASSERT(block.is_free() && "Block is already taken!");

    VkDeviceSize alignedOffset = VmaAlignUp(block.offset, allocAlignment);
    if (block.size < allocSize + alignedOffset - block.offset)
        return false;

    // Check for granularity conflicts
    if (!is_virtual() &&
        _granularity_handler.check_conflict_and_align_up(alignedOffset, allocSize, block.offset, block.size, allocType))
        return false;

    // Alloc successful
    pAllocationRequest->type = VmaAllocationRequestType::TLSF;
    pAllocationRequest->allocHandle = (VmaAllocHandle)&block;
    pAllocationRequest->size = allocSize - get_debug_margin();
    pAllocationRequest->customData = (void*)allocType;
    pAllocationRequest->algorithmData = alignedOffset;

    // Place block at the start of list if it's normal block
    if (listIndex != _lists_count && block.prev_free())
    {
        block.prev_free()->next_free() = block.next_free();
        if (block.next_free())
            block.next_free()->prev_free() = block.prev_free();
        block.prev_free() = VMA_NULL;
        block.next_free() = _free_list[listIndex];
        _free_list[listIndex] = &block;
        if (block.next_free())
            block.next_free()->prev_free() = &block;
    }

    return true;
}
#endif // _VMA_BLOCK_METADATA_TLSF_FUNCTIONS
#endif // _VMA_BLOCK_METADATA_TLSF

#ifndef _VMA_BLOCK_VECTOR
/*
Sequence of VmaDeviceMemoryBlock. Represents memory blocks allocated for a specific
Vulkan memory type.

Synchronized internally with a mutex.
*/
class VmaBlockVector
{
    friend struct VmaDefragmentationContext_T;
    VMA_CLASS_NO_COPY_NO_MOVE(VmaBlockVector)
public:
    VmaBlockVector(
        VmaAllocator hAllocator,
        VmaPool hParentPool,
        uint32_t memoryTypeIndex,
        VkDeviceSize preferredBlockSize,
        size_t minBlockCount,
        size_t maxBlockCount,
        VkDeviceSize bufferImageGranularity,
        bool explicitBlockSize,
        uint32_t algorithm,
        float priority,
        VkDeviceSize minAllocationAlignment,
        void* pMemoryAllocateNext);
    ~VmaBlockVector();

    VmaAllocator get_allocator() const { return _h_allocator; }
    VmaPool get_parent_pool() const { return _h_parent_pool; }
    bool is_custom_pool() const { return _h_parent_pool != VMA_NULL; }
    uint32_t get_memory_type_index() const { return _memory_type_index; }
    VkDeviceSize get_preferred_block_size() const { return _preferred_block_size; }
    VkDeviceSize get_buffer_image_granularity() const { return _buffer_image_granularity; }
    uint32_t get_algorithm() const { return _algorithm; }
    bool has_explicit_block_size() const { return _explicit_block_size; }
    float get_priority() const { return _priority; }
    const void* get_allocation_next_ptr() const { return _p_memory_allocate_next; }
    // To be used only while the _mutex is locked. Used during defragmentation.
    size_t get_block_count() const { return _blocks.size(); }
    // To be used only while the _mutex is locked. Used during defragmentation.
    VmaDeviceMemoryBlock* get_block(size_t index) const { return _blocks[index]; }
    VMA_RW_MUTEX &get_mutex() { return _mutex; }

    VkResult create_min_blocks();
    void add_statistics(VmaStatistics& inoutStats);
    void add_detailed_statistics(VmaDetailedStatistics& inoutStats);
    bool is_empty();
    bool is_corruption_detection_enabled() const;

    VkResult Allocate(
        VkDeviceSize size,
        VkDeviceSize alignment,
        const VmaAllocationCreateInfo& createInfo,
        VmaSuballocationType suballocType,
        size_t allocationCount,
        VmaAllocation* pAllocations);

    void free(VmaAllocation hAllocation);

#if VMA_STATS_STRING_ENABLED
    void print_detailed_map(class VmaJsonWriter& json);
#endif

    VkResult check_corruption();

private:
    const VmaAllocator _h_allocator;
    const VmaPool _h_parent_pool;
    const uint32_t _memory_type_index;
    const VkDeviceSize _preferred_block_size;
    const size_t _min_block_count;
    const size_t _max_block_count;
    const VkDeviceSize _buffer_image_granularity;
    const bool _explicit_block_size;
    const uint32_t _algorithm;
    const float _priority;
    const VkDeviceSize _min_allocation_alignment;

    void* const _p_memory_allocate_next;
    VMA_RW_MUTEX _mutex;
    // Incrementally sorted by sumFreeSize, ascending.
    VmaVector<VmaDeviceMemoryBlock*, VmaStlAllocator<VmaDeviceMemoryBlock*>> _blocks;
    uint32_t _next_block_id;
    bool _incremental_sort = true;

    void set_incremental_sort(bool val) { _incremental_sort = val; }

    VkDeviceSize calc_max_block_size() const;
    // Finds and removes given block from vector.
    void remove(VmaDeviceMemoryBlock* pBlock);
    // Performs single step in sorting _blocks. They may not be fully sorted
    // after this call.
    void incrementally_sort_blocks();
    void sort_by_free_size();

    VkResult allocate_page(
        VkDeviceSize size,
        VkDeviceSize alignment,
        const VmaAllocationCreateInfo& createInfo,
        VmaSuballocationType suballocType,
        VmaAllocation* pAllocation);

    VkResult allocate_from_block(
        VmaDeviceMemoryBlock* pBlock,
        VkDeviceSize size,
        VkDeviceSize alignment,
        VmaAllocationCreateFlags allocFlags,
        void* pUserData,
        VmaSuballocationType suballocType,
        uint32_t strategy,
        VmaAllocation* pAllocation);

    VkResult commit_allocation_request(
        VmaAllocationRequest& allocRequest,
        VmaDeviceMemoryBlock* pBlock,
        VkDeviceSize alignment,
        VmaAllocationCreateFlags allocFlags,
        void* pUserData,
        VmaSuballocationType suballocType,
        VmaAllocation* pAllocation);

    VkResult create_block(VkDeviceSize blockSize, size_t* pNewBlockIndex);
    bool has_empty_block();
};
#endif // _VMA_BLOCK_VECTOR

#ifndef _VMA_DEFRAGMENTATION_CONTEXT
struct VmaDefragmentationContext_T
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaDefragmentationContext_T)
public:
    VmaDefragmentationContext_T(
        VmaAllocator hAllocator,
        const VmaDefragmentationInfo& info);
    ~VmaDefragmentationContext_T();

    void get_stats(VmaDefragmentationStats& outStats) { outStats = _global_stats; }

    VkResult defragment_pass_begin(VmaDefragmentationPassMoveInfo& moveInfo);
    VkResult defragment_pass_end(VmaDefragmentationPassMoveInfo& moveInfo);

private:
    // Max number of allocations to ignore due to size constraints before ending single pass
    static const uint8_t MAX_ALLOCS_TO_IGNORE = 16;
    enum class CounterStatus { Pass, Ignore, End };

    struct FragmentedBlock
    {
        uint32_t data;
        VmaDeviceMemoryBlock* block;
    };
    struct StateBalanced
    {
        VkDeviceSize avgFreeSize = 0;
        VkDeviceSize avgAllocSize = UINT64_MAX;
    };
    struct StateExtensive
    {
        enum class Operation : uint8_t
        {
            find_free_blockBuffer, find_free_blockTexture, find_free_blockAll,
            MoveBuffers, MoveTextures, MoveAll,
            Cleanup, Done
        };

        Operation operation = Operation::find_free_blockTexture;
        size_t firstFreeBlock = SIZE_MAX;
    };
    struct MoveAllocationData
    {
        VkDeviceSize size;
        VkDeviceSize alignment;
        VmaSuballocationType type;
        VmaAllocationCreateFlags flags;
        VmaDefragmentationMove move = {};
    };

    const VkDeviceSize _max_pass_bytes;
    const uint32_t _max_pass_allocations;
    const PFN_vmaCheckDefragmentationBreakFunction _break_callback;
    void* _break_callback_user_data;

    VmaStlAllocator<VmaDefragmentationMove> _move_allocator;
    VmaVector<VmaDefragmentationMove, VmaStlAllocator<VmaDefragmentationMove>> _moves;

    uint8_t _ignored_allocs = 0;
    uint32_t _algorithm;
    uint32_t _block_vector_count;
    VmaBlockVector* _pool_block_vector;
    VmaBlockVector** _p_block_vectors;
    size_t _immovable_block_count = 0;
    VmaDefragmentationStats _global_stats = { 0 };
    VmaDefragmentationStats _pass_stats = { 0 };
    void* _algorithm_state = VMA_NULL;

    static MoveAllocationData get_move_data(VmaAllocHandle handle, VmaBlockMetadata* metadata);
    CounterStatus check_counters(VkDeviceSize bytes);
    bool increment_counters(VkDeviceSize bytes);
    bool realloc_within_block(VmaBlockVector& vector, VmaDeviceMemoryBlock* block);
    bool alloc_in_other_block(size_t start, size_t end, MoveAllocationData& data, VmaBlockVector& vector);

    bool compute_defragmentation(VmaBlockVector& vector, size_t index);
    bool compute_defragmentation_fast(VmaBlockVector& vector);
    bool compute_defragmentation_balanced(VmaBlockVector& vector, size_t index, bool update);
    bool compute_defragmentation_full(VmaBlockVector& vector);
    bool compute_defragmentation_extensive(VmaBlockVector& vector, size_t index);

    static void update_vector_statistics(VmaBlockVector& vector, StateBalanced& state);
    bool move_data_to_free_blocks(VmaSuballocationType currentType,
        VmaBlockVector& vector, size_t firstFreeBlock,
        bool& texturePresent, bool& bufferPresent, bool& otherPresent);
};
#endif // _VMA_DEFRAGMENTATION_CONTEXT

#ifndef _VMA_POOL_T
struct VmaPool_T
{
    friend struct VmaPoolListItemTraits;
    VMA_CLASS_NO_COPY_NO_MOVE(VmaPool_T)
public:
    VmaBlockVector _block_vector;
    VmaDedicatedAllocationList _dedicated_allocations;

    VmaPool_T(
        VmaAllocator hAllocator,
        const VmaPoolCreateInfo& createInfo,
        VkDeviceSize preferredBlockSize);
    ~VmaPool_T();

    uint32_t get_id() const { return _id; }
    void set_id(uint32_t id) { VMA_ASSERT(_id == 0); _id = id; }

    const char* get_name() const { return _name; }
    void set_name(const char* pName);

#if VMA_STATS_STRING_ENABLED
    //void print_detailed_map(class VmaStringBuilder& sb);
#endif

private:
    uint32_t _id;
    char* _name;
    VmaPool_T* _prev_pool = VMA_NULL;
    VmaPool_T* _next_pool = VMA_NULL;
};

struct VmaPoolListItemTraits
{
    typedef VmaPool_T ItemType;

    static ItemType* get_prev(const ItemType* item) { return item->_prev_pool; }
    static ItemType* get_next(const ItemType* item) { return item->_next_pool; }
    static ItemType*& access_prev(ItemType* item) { return item->_prev_pool; }
    static ItemType*& access_next(ItemType* item) { return item->_next_pool; }
};
#endif // _VMA_POOL_T

#ifndef _VMA_CURRENT_BUDGET_DATA
struct VmaCurrentBudgetData
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaCurrentBudgetData)
public:

    VMA_ATOMIC_UINT32 _block_count[VK_MAX_MEMORY_HEAPS];
    VMA_ATOMIC_UINT32 _allocation_count[VK_MAX_MEMORY_HEAPS];
    VMA_ATOMIC_UINT64 _block_bytes[VK_MAX_MEMORY_HEAPS];
    VMA_ATOMIC_UINT64 _allocation_bytes[VK_MAX_MEMORY_HEAPS];

#if VMA_MEMORY_BUDGET
    VMA_ATOMIC_UINT32 _operations_since_budget_fetch;
    VMA_RW_MUTEX _budget_mutex;
    uint64_t _vulkan_usage[VK_MAX_MEMORY_HEAPS];
    uint64_t _vulkan_budget[VK_MAX_MEMORY_HEAPS];
    uint64_t _block_bytes_at_budget_fetch[VK_MAX_MEMORY_HEAPS];
#endif // VMA_MEMORY_BUDGET

    VmaCurrentBudgetData();

    void add_allocation(uint32_t heapIndex, VkDeviceSize allocationSize);
    void remove_allocation(uint32_t heapIndex, VkDeviceSize allocationSize);
};

#ifndef _VMA_CURRENT_BUDGET_DATA_FUNCTIONS
VmaCurrentBudgetData::VmaCurrentBudgetData()
{
    for (uint32_t heapIndex = 0; heapIndex < VK_MAX_MEMORY_HEAPS; ++heapIndex)
    {
        _block_count[heapIndex] = 0;
        _allocation_count[heapIndex] = 0;
        _block_bytes[heapIndex] = 0;
        _allocation_bytes[heapIndex] = 0;
#if VMA_MEMORY_BUDGET
        _vulkan_usage[heapIndex] = 0;
        _vulkan_budget[heapIndex] = 0;
        _block_bytes_at_budget_fetch[heapIndex] = 0;
#endif
    }

#if VMA_MEMORY_BUDGET
    _operations_since_budget_fetch = 0;
#endif
}

void VmaCurrentBudgetData::add_allocation(uint32_t heapIndex, VkDeviceSize allocationSize)
{
    _allocation_bytes[heapIndex] += allocationSize;
    ++_allocation_count[heapIndex];
#if VMA_MEMORY_BUDGET
    ++_operations_since_budget_fetch;
#endif
}

void VmaCurrentBudgetData::remove_allocation(uint32_t heapIndex, VkDeviceSize allocationSize)
{
    VMA_ASSERT(_allocation_bytes[heapIndex] >= allocationSize);
    _allocation_bytes[heapIndex] -= allocationSize;
    VMA_ASSERT(_allocation_count[heapIndex] > 0);
    --_allocation_count[heapIndex];
#if VMA_MEMORY_BUDGET
    ++_operations_since_budget_fetch;
#endif
}
#endif // _VMA_CURRENT_BUDGET_DATA_FUNCTIONS
#endif // _VMA_CURRENT_BUDGET_DATA

#ifndef _VMA_ALLOCATION_OBJECT_ALLOCATOR
/*
Thread-safe wrapper over VmaPoolAllocator free list, for allocation of VmaAllocation_T objects.
*/
class VmaAllocationObjectAllocator
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaAllocationObjectAllocator)
public:
    explicit VmaAllocationObjectAllocator(const VkAllocationCallbacks* pAllocationCallbacks)
        : _allocator(pAllocationCallbacks, 1024) {}

    template<typename... Types> VmaAllocation Allocate(Types&&... args);
    void free(VmaAllocation hAlloc);

private:
    VMA_MUTEX _mutex;
    VmaPoolAllocator<VmaAllocation_T> _allocator;
};

template<typename... Types>
VmaAllocation VmaAllocationObjectAllocator::Allocate(Types&&... args)
{
    VmaMutexLock mutexLock(_mutex);
    return _allocator.alloc<Types...>(std::forward<Types>(args)...);
}

void VmaAllocationObjectAllocator::free(VmaAllocation hAlloc)
{
    VmaMutexLock mutexLock(_mutex);
    _allocator.free(hAlloc);
}
#endif // _VMA_ALLOCATION_OBJECT_ALLOCATOR

#ifndef _VMA_VIRTUAL_BLOCK_T
struct VmaVirtualBlock_T
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaVirtualBlock_T)
public:
    const bool _allocation_callbacks_specified;
    const VkAllocationCallbacks _allocation_callbacks;

    explicit VmaVirtualBlock_T(const VmaVirtualBlockCreateInfo& createInfo);
    ~VmaVirtualBlock_T();

    bool is_empty() const { return _metadata->is_empty(); }
    void free(VmaVirtualAllocation allocation) { _metadata->free((VmaAllocHandle)allocation); }
    void set_allocation_user_data(VmaVirtualAllocation allocation, void* userData) { _metadata->set_allocation_user_data((VmaAllocHandle)allocation, userData); }
    void clear() { _metadata->clear(); }

    const VkAllocationCallbacks* get_allocation_callbacks() const;
    void get_allocation_info(VmaVirtualAllocation allocation, VmaVirtualAllocationInfo& outInfo);
    VkResult Allocate(const VmaVirtualAllocationCreateInfo& createInfo, VmaVirtualAllocation& outAllocation,
        VkDeviceSize* outOffset);
    void get_statistics(VmaStatistics& outStats) const;
    void calculate_detailed_statistics(VmaDetailedStatistics& outStats) const;
#if VMA_STATS_STRING_ENABLED
    void build_stats_string(bool detailedMap, VmaStringBuilder& sb) const;
#endif

private:
    VmaBlockMetadata* _metadata;
};

#ifndef _VMA_VIRTUAL_BLOCK_T_FUNCTIONS
VmaVirtualBlock_T::VmaVirtualBlock_T(const VmaVirtualBlockCreateInfo& createInfo)
    : _allocation_callbacks_specified(createInfo.pAllocationCallbacks != VMA_NULL),
    _allocation_callbacks(createInfo.pAllocationCallbacks != VMA_NULL ? *createInfo.pAllocationCallbacks : VmaEmptyAllocationCallbacks)
{
    const uint32_t algorithm = createInfo.flags & VMA_VIRTUAL_BLOCK_CREATE_ALGORITHM_MASK;
    switch (algorithm)
    {
    case 0:
        _metadata = vma_new(get_allocation_callbacks(), VmaBlockMetadata_TLSF)(VK_NULL_HANDLE, 1, true);
        break;
    case VMA_VIRTUAL_BLOCK_CREATE_LINEAR_ALGORITHM_BIT:
        _metadata = vma_new(get_allocation_callbacks(), VmaBlockMetadata_Linear)(VK_NULL_HANDLE, 1, true);
        break;
    default:
        VMA_ASSERT(0);
        _metadata = vma_new(get_allocation_callbacks(), VmaBlockMetadata_TLSF)(VK_NULL_HANDLE, 1, true);
    }

    _metadata->init(createInfo.size);
}

VmaVirtualBlock_T::~VmaVirtualBlock_T()
{
    // Define macro VMA_DEBUG_LOG_FORMAT or more specialized VMA_LEAK_LOG_FORMAT
    // to receive the list of the unfreed allocations.
    if (!_metadata->is_empty())
        _metadata->debug_log_all_allocations();
    // This is the most important assert in the entire library.
    // Hitting it means you have some memory leak - unreleased virtual allocations.
    VMA_ASSERT_LEAK(_metadata->is_empty() && "Some virtual allocations were not freed before destruction of this virtual block!");

    vma_delete(get_allocation_callbacks(), _metadata);
}

const VkAllocationCallbacks* VmaVirtualBlock_T::get_allocation_callbacks() const
{
    return _allocation_callbacks_specified ? &_allocation_callbacks : VMA_NULL;
}

void VmaVirtualBlock_T::get_allocation_info(VmaVirtualAllocation allocation, VmaVirtualAllocationInfo& outInfo)
{
    _metadata->get_allocation_info((VmaAllocHandle)allocation, outInfo);
}

VkResult VmaVirtualBlock_T::Allocate(const VmaVirtualAllocationCreateInfo& createInfo, VmaVirtualAllocation& outAllocation,
    VkDeviceSize* outOffset)
{
    VmaAllocationRequest request = {};
    if (_metadata->create_allocation_request(
        createInfo.size, // allocSize
        VMA_MAX(createInfo.alignment, (VkDeviceSize)1), // allocAlignment
        (createInfo.flags & VMA_VIRTUAL_ALLOCATION_CREATE_UPPER_ADDRESS_BIT) != 0, // upperAddress
        VMA_SUBALLOCATION_TYPE_UNKNOWN, // allocType - unimportant
        createInfo.flags & VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MASK, // strategy
        &request))
    {
        _metadata->alloc(request,
            VMA_SUBALLOCATION_TYPE_UNKNOWN, // type - unimportant
            createInfo.pUserData);
        outAllocation = (VmaVirtualAllocation)request.allocHandle;
        if(outOffset)
            *outOffset = _metadata->get_allocation_offset(request.allocHandle);
        return VK_SUCCESS;
    }
    outAllocation = (VmaVirtualAllocation)VK_NULL_HANDLE;
    if (outOffset)
        *outOffset = UINT64_MAX;
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

void VmaVirtualBlock_T::get_statistics(VmaStatistics& outStats) const
{
    VmaClearStatistics(outStats);
    _metadata->add_statistics(outStats);
}

void VmaVirtualBlock_T::calculate_detailed_statistics(VmaDetailedStatistics& outStats) const
{
    VmaClearDetailedStatistics(outStats);
    _metadata->add_detailed_statistics(outStats);
}

#if VMA_STATS_STRING_ENABLED
void VmaVirtualBlock_T::build_stats_string(bool detailedMap, VmaStringBuilder& sb) const
{
    VmaJsonWriter json(get_allocation_callbacks(), sb);
    json.begin_object();

    VmaDetailedStatistics stats;
    calculate_detailed_statistics(stats);

    json.write_string("Stats");
    VmaPrintDetailedStatistics(json, stats);

    if (detailedMap)
    {
        json.write_string("Details");
        json.begin_object();
        _metadata->print_detailed_map(json);
        json.end_object();
    }

    json.end_object();
}
#endif // VMA_STATS_STRING_ENABLED
#endif // _VMA_VIRTUAL_BLOCK_T_FUNCTIONS
#endif // _VMA_VIRTUAL_BLOCK_T


// Main allocator object.
struct VmaAllocator_T
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaAllocator_T)
public:
    const bool _use_mutex;
    const uint32_t _vulkan_api_version;
    bool _use_khr_dedicated_allocation; // Can be set only if _vulkan_api_version < VK_MAKE_VERSION(1, 1, 0).
    bool _use_khr_bind_memory2; // Can be set only if _vulkan_api_version < VK_MAKE_VERSION(1, 1, 0).
    bool _use_ext_memory_budget;
    bool _use_amd_device_coherent_memory;
    bool _use_khr_buffer_device_address;
    bool _use_ext_memory_priority;
    bool _use_khr_maintenance4;
    bool _use_khr_maintenance5;
    bool _use_khr_external_memory_win32;
    const VkDevice _h_device;
    const VkInstance _h_instance;
    const bool _allocation_callbacks_specified;
    const VkAllocationCallbacks _allocation_callbacks;
    VmaDeviceMemoryCallbacks _device_memory_callbacks;
    VmaAllocationObjectAllocator _allocation_object_allocator;

    // Each bit (1 << i) is set if HeapSizeLimit is enabled for that heap, so cannot allocate more than the heap size.
    uint32_t _heap_size_limit_mask;

    VkPhysicalDeviceProperties _physical_device_properties;
    VkPhysicalDeviceMemoryProperties _mem_props;

    // Default pools.
    VmaBlockVector* _p_block_vectors[VK_MAX_MEMORY_TYPES];
    VmaDedicatedAllocationList _dedicated_allocations[VK_MAX_MEMORY_TYPES];

    VmaCurrentBudgetData _budget;
    VMA_ATOMIC_UINT32 _device_memory_count; // Total number of VkDeviceMemory objects.

    explicit VmaAllocator_T(const VmaAllocatorCreateInfo* pCreateInfo);
    VkResult init(const VmaAllocatorCreateInfo* pCreateInfo);
    ~VmaAllocator_T();

    const VkAllocationCallbacks* get_allocation_callbacks() const
    {
        return _allocation_callbacks_specified ? &_allocation_callbacks : VMA_NULL;
    }
    const VmaVulkanFunctions& get_vulkan_functions() const
    {
        return _vulkan_functions;
    }

    VkPhysicalDevice get_physical_device() const { return _physical_device; }

    VkDeviceSize get_buffer_image_granularity() const
    {
        return VMA_MAX(
            static_cast<VkDeviceSize>(VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY),
            _physical_device_properties.limits.bufferImageGranularity);
    }

    uint32_t get_memory_heap_count() const { return _mem_props.memoryHeapCount; }
    uint32_t get_memory_type_count() const { return _mem_props.memoryTypeCount; }

    uint32_t memory_type_index_to_heap_index(uint32_t memTypeIndex) const
    {
        VMA_ASSERT(memTypeIndex < _mem_props.memoryTypeCount);
        return _mem_props.memoryTypes[memTypeIndex].heapIndex;
    }
    // True when specific memory type is HOST_VISIBLE but not HOST_COHERENT.
    bool is_memory_type_non_coherent(uint32_t memTypeIndex) const
    {
        return (_mem_props.memoryTypes[memTypeIndex].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    }
    // Minimum alignment for all allocations in specific memory type.
    VkDeviceSize get_memory_type_min_alignment(uint32_t memTypeIndex) const
    {
        return is_memory_type_non_coherent(memTypeIndex) ?
            VMA_MAX((VkDeviceSize)VMA_MIN_ALIGNMENT, _physical_device_properties.limits.nonCoherentAtomSize) :
            (VkDeviceSize)VMA_MIN_ALIGNMENT;
    }

    bool is_integrated_gpu() const
    {
        return _physical_device_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    }

    uint32_t get_global_memory_type_bits() const { return _global_memory_type_bits; }

    void get_buffer_memory_requirements(
        VkBuffer hBuffer,
        VkMemoryRequirements& memReq,
        bool& requiresDedicatedAllocation,
        bool& prefersDedicatedAllocation) const;
    void get_image_memory_requirements(
        VkImage hImage,
        VkMemoryRequirements& memReq,
        bool& requiresDedicatedAllocation,
        bool& prefersDedicatedAllocation) const;
    VkResult find_memory_type_index(
        uint32_t memoryTypeBits,
        const VmaAllocationCreateInfo* pAllocationCreateInfo,
        VmaBufferImageUsage bufImgUsage,
        uint32_t* pMemoryTypeIndex) const;

    // Main allocation function.
    VkResult allocate_memory(
        const VkMemoryRequirements& vkMemReq,
        bool requiresDedicatedAllocation,
        bool prefersDedicatedAllocation,
        VkBuffer dedicatedBuffer,
        VkImage dedicatedImage,
        VmaBufferImageUsage dedicatedBufferImageUsage,
        const VmaAllocationCreateInfo& createInfo,
        VmaSuballocationType suballocType,
        size_t allocationCount,
        VmaAllocation* pAllocations);

    // Main deallocation function.
    void free_memory(
        size_t allocationCount,
        const VmaAllocation* pAllocations);

    void calculate_statistics(VmaTotalStatistics* pStats);

    void get_heap_budgets(
        VmaBudget* outBudgets, uint32_t firstHeap, uint32_t heapCount);

#if VMA_STATS_STRING_ENABLED
    void print_detailed_map(class VmaJsonWriter& json);
#endif

    static void get_allocation_info(VmaAllocation hAllocation, VmaAllocationInfo* pAllocationInfo);
    static void get_allocation_info2(VmaAllocation hAllocation, VmaAllocationInfo2* pAllocationInfo);

    VkResult create_pool(const VmaPoolCreateInfo* pCreateInfo, VmaPool* pPool);
    void destroy_pool(VmaPool pool);
    static void get_pool_statistics(VmaPool pool, VmaStatistics* pPoolStats);
    static void calculate_pool_statistics(VmaPool pool, VmaDetailedStatistics* pPoolStats);

    void set_current_frame_index(uint32_t frameIndex);
    uint32_t get_current_frame_index() const { return _current_frame_index.load(); }

    static VkResult check_pool_corruption(VmaPool hPool);
    VkResult check_corruption(uint32_t memoryTypeBits);

    // Call to Vulkan function vkAllocateMemory with accompanying bookkeeping.
    VkResult allocate_vulkan_memory(const VkMemoryAllocateInfo* pAllocateInfo, VkDeviceMemory* pMemory);
    // Call to Vulkan function vkFreeMemory with accompanying bookkeeping.
    void free_vulkan_memory(uint32_t memoryType, VkDeviceSize size, VkDeviceMemory hMemory);
    // Call to Vulkan function vkBindBufferMemory or vkBindBufferMemory2KHR.
    VkResult bind_vulkan_buffer(
        VkDeviceMemory memory,
        VkDeviceSize memoryOffset,
        VkBuffer buffer,
        const void* pNext) const;
    // Call to Vulkan function vkBindImageMemory or vkBindImageMemory2KHR.
    VkResult bind_vulkan_image(
        VkDeviceMemory memory,
        VkDeviceSize memoryOffset,
        VkImage image,
        const void* pNext) const;

    VkResult Map(VmaAllocation hAllocation, void** ppData);
    void unmap(VmaAllocation hAllocation);

    VkResult bind_buffer_memory(
        VmaAllocation hAllocation,
        VkDeviceSize allocationLocalOffset,
        VkBuffer hBuffer,
        const void* pNext);
    VkResult bind_image_memory(
        VmaAllocation hAllocation,
        VkDeviceSize allocationLocalOffset,
        VkImage hImage,
        const void* pNext);

    VkResult flush_or_invalidate_allocation(
        VmaAllocation hAllocation,
        VkDeviceSize offset, VkDeviceSize size,
        VMA_CACHE_OPERATION op);
    VkResult flush_or_invalidate_allocations(
        uint32_t allocationCount,
        const VmaAllocation* allocations,
        const VkDeviceSize* offsets, const VkDeviceSize* sizes,
        VMA_CACHE_OPERATION op);

    VkResult copy_memory_to_allocation(
        const void* pSrcHostPointer,
        VmaAllocation dstAllocation,
        VkDeviceSize dstAllocationLocalOffset,
        VkDeviceSize size);
    VkResult copy_allocation_to_memory(
        VmaAllocation srcAllocation,
        VkDeviceSize srcAllocationLocalOffset,
        void* pDstHostPointer,
        VkDeviceSize size);

    void fill_allocation(VmaAllocation hAllocation, uint8_t pattern);

    /*
    Returns bit mask of memory types that can support defragmentation on GPU as
    they support creation of required buffer for copy operations.
    */
    uint32_t get_gpu_defragmentation_memory_type_bits();

#if VMA_EXTERNAL_MEMORY
    VkExternalMemoryHandleTypeFlagsKHR get_external_memory_handle_type_flags(uint32_t memTypeIndex) const
    {
        return _type_external_memory_handle_types[memTypeIndex];
    }
#endif // #if VMA_EXTERNAL_MEMORY

private:
    VkDeviceSize _preferred_large_heap_block_size;

    VkPhysicalDevice _physical_device;
    VMA_ATOMIC_UINT32 _current_frame_index;
    VMA_ATOMIC_UINT32 _gpu_defragmentation_memory_type_bits; // UINT32_MAX means uninitialized.
#if VMA_EXTERNAL_MEMORY
    VkExternalMemoryHandleTypeFlagsKHR _type_external_memory_handle_types[VK_MAX_MEMORY_TYPES];
#endif // #if VMA_EXTERNAL_MEMORY

    VMA_RW_MUTEX _pools_mutex;
    typedef VmaIntrusiveLinkedList<VmaPoolListItemTraits> PoolList;
    // Protected by _pools_mutex.
    PoolList _pools;
    uint32_t _next_pool_id;

    VmaVulkanFunctions _vulkan_functions;

    // Global bit mask AND-ed with any memoryTypeBits to disallow certain memory types.
    uint32_t _global_memory_type_bits;

    void import_vulkan_functions(const VmaVulkanFunctions* pVulkanFunctions);

#if VMA_STATIC_VULKAN_FUNCTIONS == 1
    void import_vulkan_functions_static();
#endif

    void import_vulkan_functions_custom(const VmaVulkanFunctions* pVulkanFunctions);

#if VMA_DYNAMIC_VULKAN_FUNCTIONS == 1
    void import_vulkan_functions_dynamic();
#endif

    void validate_vulkan_functions() const;

    VkDeviceSize calc_preferred_block_size(uint32_t memTypeIndex);

    VkResult allocate_memory_of_type(
        VmaPool pool,
        VkDeviceSize size,
        VkDeviceSize alignment,
        bool dedicatedPreferred,
        VkBuffer dedicatedBuffer,
        VkImage dedicatedImage,
        VmaBufferImageUsage dedicatedBufferImageUsage,
        const VmaAllocationCreateInfo& createInfo,
        uint32_t memTypeIndex,
        VmaSuballocationType suballocType,
        VmaDedicatedAllocationList& dedicatedAllocations,
        VmaBlockVector& blockVector,
        size_t allocationCount,
        VmaAllocation* pAllocations);

    // Helper function only to be used inside AllocateDedicatedMemory.
    VkResult allocate_dedicated_memory_page(
        VmaPool pool,
        VkDeviceSize size,
        VmaSuballocationType suballocType,
        uint32_t memTypeIndex,
        const VkMemoryAllocateInfo& allocInfo,
        bool map,
        bool isUserDataString,
        bool isMappingAllowed,
        void* pUserData,
        VmaAllocation* pAllocation);

    // Allocates and registers new VkDeviceMemory specifically for dedicated allocations.
    VkResult allocate_dedicated_memory(
        VmaPool pool,
        VkDeviceSize size,
        VmaSuballocationType suballocType,
        VmaDedicatedAllocationList& dedicatedAllocations,
        uint32_t memTypeIndex,
        bool map,
        bool isUserDataString,
        bool isMappingAllowed,
        bool canAliasMemory,
        void* pUserData,
        float priority,
        VkBuffer dedicatedBuffer,
        VkImage dedicatedImage,
        VmaBufferImageUsage dedicatedBufferImageUsage,
        size_t allocationCount,
        VmaAllocation* pAllocations,
        const void* pNextChain = VMA_NULL);

    void free_dedicated_memory(VmaAllocation allocation);

    VkResult calc_mem_type_params(
        VmaAllocationCreateInfo& outCreateInfo,
        uint32_t memTypeIndex,
        VkDeviceSize size,
        size_t allocationCount);
    static VkResult calc_allocation_params(
        VmaAllocationCreateInfo& outCreateInfo,
        bool dedicatedRequired);

    /*
    Calculates and returns bit mask of memory types that can support defragmentation
    on GPU as they support creation of required buffer for copy operations.
    */
    uint32_t calculate_gpu_defragmentation_memory_type_bits() const;
    uint32_t calculate_global_memory_type_bits() const;

    bool get_flush_or_invalidate_range(
        VmaAllocation allocation,
        VkDeviceSize offset, VkDeviceSize size,
        VkMappedMemoryRange& outRange) const;

#if VMA_MEMORY_BUDGET
    void update_vulkan_budget();
#endif // #if VMA_MEMORY_BUDGET
};


#ifndef _VMA_MEMORY_FUNCTIONS
static void* VmaMalloc(VmaAllocator hAllocator, size_t size, size_t alignment)
{
    return VmaMalloc(&hAllocator->_allocation_callbacks, size, alignment);
}

static void VmaFree(VmaAllocator hAllocator, void* ptr)
{
    VmaFree(&hAllocator->_allocation_callbacks, ptr);
}

template<typename T>
static T* VmaAllocate(VmaAllocator hAllocator)
{
    return (T*)VmaMalloc(hAllocator, sizeof(T), VMA_ALIGN_OF(T));
}

template<typename T>
static T* VmaAllocateArray(VmaAllocator hAllocator, size_t count)
{
    return (T*)VmaMalloc(hAllocator, sizeof(T) * count, VMA_ALIGN_OF(T));
}

template<typename T>
static void vma_delete(VmaAllocator hAllocator, T* ptr)
{
    if(ptr != VMA_NULL)
    {
        ptr->~T();
        VmaFree(hAllocator, ptr);
    }
}

template<typename T>
static void vma_delete_array(VmaAllocator hAllocator, T* ptr, size_t count)
{
    if(ptr != VMA_NULL)
    {
        for(size_t i = count; i--; )
            ptr[i].~T();
        VmaFree(hAllocator, ptr);
    }
}
#endif // _VMA_MEMORY_FUNCTIONS

#ifndef _VMA_DEVICE_MEMORY_BLOCK_FUNCTIONS
VmaDeviceMemoryBlock::VmaDeviceMemoryBlock(VmaAllocator hAllocator)
    : _p_metadata(VMA_NULL),
    _h_parent_pool(nullptr),
    _memory_type_index(UINT32_MAX),
    _id(0),
    _h_memory(VK_NULL_HANDLE),
    _map_count(0),
    _p_mapped_data(VMA_NULL){}

VmaDeviceMemoryBlock::~VmaDeviceMemoryBlock()
{
    VMA_ASSERT_LEAK(_map_count == 0 && "VkDeviceMemory block is being destroyed while it is still mapped.");
    VMA_ASSERT_LEAK(_h_memory == VK_NULL_HANDLE);
}

void VmaDeviceMemoryBlock::init(
    VmaAllocator hAllocator,
    VmaPool hParentPool,
    uint32_t newMemoryTypeIndex,
    VkDeviceMemory newMemory,
    VkDeviceSize newSize,
    uint32_t id,
    uint32_t algorithm,
    VkDeviceSize bufferImageGranularity)
{
    VMA_ASSERT(_h_memory == VK_NULL_HANDLE);

    _h_parent_pool = hParentPool;
    _memory_type_index = newMemoryTypeIndex;
    _id = id;
    _h_memory = newMemory;

    switch (algorithm)
    {
    case 0:
        _p_metadata = vma_new(hAllocator, VmaBlockMetadata_TLSF)(hAllocator->get_allocation_callbacks(),
            bufferImageGranularity, false); // isVirtual
        break;
    case VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT:
        _p_metadata = vma_new(hAllocator, VmaBlockMetadata_Linear)(hAllocator->get_allocation_callbacks(),
            bufferImageGranularity, false); // isVirtual
        break;
    default:
        VMA_ASSERT(0);
        _p_metadata = vma_new(hAllocator, VmaBlockMetadata_TLSF)(hAllocator->get_allocation_callbacks(),
            bufferImageGranularity, false); // isVirtual
    }
    _p_metadata->init(newSize);
}

void VmaDeviceMemoryBlock::destroy(VmaAllocator allocator)
{
    // Define macro VMA_DEBUG_LOG_FORMAT or more specialized VMA_LEAK_LOG_FORMAT
    // to receive the list of the unfreed allocations.
    if (!_p_metadata->is_empty())
        _p_metadata->debug_log_all_allocations();
    // This is the most important assert in the entire library.
    // Hitting it means you have some memory leak - unreleased VmaAllocation objects.
    VMA_ASSERT_LEAK(_p_metadata->is_empty() && "Some allocations were not freed before destruction of this memory block!");

    VMA_ASSERT_LEAK(_h_memory != VK_NULL_HANDLE);
    allocator->free_vulkan_memory(_memory_type_index, _p_metadata->get_size(), _h_memory);
    _h_memory = VK_NULL_HANDLE;

    vma_delete(allocator, _p_metadata);
    _p_metadata = VMA_NULL;
}

void VmaDeviceMemoryBlock::post_alloc(VmaAllocator hAllocator)
{
    VmaMutexLock lock(_map_and_bind_mutex, hAllocator->_use_mutex);
    _mapping_hysteresis.post_alloc();
}

void VmaDeviceMemoryBlock::post_free(VmaAllocator hAllocator)
{
    VmaMutexLock lock(_map_and_bind_mutex, hAllocator->_use_mutex);
    if(_mapping_hysteresis.post_free())
    {
        VMA_ASSERT(_mapping_hysteresis.get_extra_mapping() == 0);
        if (_map_count == 0)
        {
            _p_mapped_data = VMA_NULL;
            (*hAllocator->get_vulkan_functions().vkUnmapMemory)(hAllocator->_h_device, _h_memory);
        }
    }
}

bool VmaDeviceMemoryBlock::validate() const
{
    VMA_VALIDATE((_h_memory != VK_NULL_HANDLE) &&
        (_p_metadata->get_size() != 0));

    return _p_metadata->validate();
}

VkResult VmaDeviceMemoryBlock::check_corruption(VmaAllocator hAllocator)
{
    void* pData = VMA_NULL;
    VkResult res = Map(hAllocator, 1, &pData);
    if (res != VK_SUCCESS)
    {
        return res;
    }

    res = _p_metadata->check_corruption(pData);

    unmap(hAllocator, 1);

    return res;
}

VkResult VmaDeviceMemoryBlock::Map(VmaAllocator hAllocator, uint32_t count, void** ppData)
{
    if (count == 0)
    {
        return VK_SUCCESS;
    }

    VmaMutexLock lock(_map_and_bind_mutex, hAllocator->_use_mutex);
    const uint32_t oldTotalMapCount = _map_count + _mapping_hysteresis.get_extra_mapping();
    if (oldTotalMapCount != 0)
    {
        VMA_ASSERT(_p_mapped_data != VMA_NULL);
        _mapping_hysteresis.post_map();
        _map_count += count;
        if (ppData != VMA_NULL)
        {
            *ppData = _p_mapped_data;
        }
        return VK_SUCCESS;
    }

    VkResult result = (*hAllocator->get_vulkan_functions().vkMapMemory)(
        hAllocator->_h_device,
        _h_memory,
        0, // offset
        VK_WHOLE_SIZE,
        0, // flags
        &_p_mapped_data);
    if (result == VK_SUCCESS)
    {
        VMA_ASSERT(_p_mapped_data != VMA_NULL);
        _mapping_hysteresis.post_map();
        _map_count = count;
        if (ppData != VMA_NULL)
        {
            *ppData = _p_mapped_data;
        }
    }
    return result;
}

void VmaDeviceMemoryBlock::unmap(VmaAllocator hAllocator, uint32_t count)
{
    if (count == 0)
    {
        return;
    }

    VmaMutexLock lock(_map_and_bind_mutex, hAllocator->_use_mutex);
    if (_map_count >= count)
    {
        _map_count -= count;
        const uint32_t totalMapCount = _map_count + _mapping_hysteresis.get_extra_mapping();
        if (totalMapCount == 0)
        {
            _p_mapped_data = VMA_NULL;
            (*hAllocator->get_vulkan_functions().vkUnmapMemory)(hAllocator->_h_device, _h_memory);
        }
        _mapping_hysteresis.post_unmap();
    }
    else
    {
        VMA_ASSERT(0 && "VkDeviceMemory block is being unmapped while it was not previously mapped.");
    }
}

VkResult VmaDeviceMemoryBlock::write_magic_value_after_allocation(VmaAllocator hAllocator, VkDeviceSize allocOffset, VkDeviceSize allocSize)
{
    VMA_ASSERT(VMA_DEBUG_MARGIN > 0 && VMA_DEBUG_MARGIN % 4 == 0 && VMA_DEBUG_DETECT_CORRUPTION);

    void* pData = VMA_NULL;
    VkResult res = Map(hAllocator, 1, &pData);
    if (res != VK_SUCCESS)
    {
        return res;
    }

    VmaWriteMagicValue(pData, allocOffset + allocSize);

    unmap(hAllocator, 1);
    return VK_SUCCESS;
}

VkResult VmaDeviceMemoryBlock::validate_magic_value_after_allocation(VmaAllocator hAllocator, VkDeviceSize allocOffset, VkDeviceSize allocSize)
{
    VMA_ASSERT(VMA_DEBUG_MARGIN > 0 && VMA_DEBUG_MARGIN % 4 == 0 && VMA_DEBUG_DETECT_CORRUPTION);

    void* pData = VMA_NULL;
    VkResult res = Map(hAllocator, 1, &pData);
    if (res != VK_SUCCESS)
    {
        return res;
    }

    if (!VmaValidateMagicValue(pData, allocOffset + allocSize))
    {
        VMA_ASSERT(0 && "MEMORY CORRUPTION DETECTED AFTER FREED ALLOCATION!");
    }

    unmap(hAllocator, 1);
    return VK_SUCCESS;
}

VkResult VmaDeviceMemoryBlock::bind_buffer_memory(
    VmaAllocator hAllocator,
    VmaAllocation hAllocation,
    VkDeviceSize allocationLocalOffset,
    VkBuffer hBuffer,
    const void* pNext)
{
    VMA_ASSERT(hAllocation->get_type() == VmaAllocation_T::ALLOCATION_TYPE_BLOCK &&
        hAllocation->get_block() == this);
    VMA_ASSERT(allocationLocalOffset < hAllocation->get_size() &&
        "Invalid allocationLocalOffset. Did you forget that this offset is relative to the beginning of the allocation, not the whole memory block?");
    const VkDeviceSize memoryOffset = hAllocation->get_offset() + allocationLocalOffset;
    // This lock is important so that we don't call vkBind... and/or vkMap... simultaneously on the same VkDeviceMemory from multiple threads.
    VmaMutexLock lock(_map_and_bind_mutex, hAllocator->_use_mutex);
    return hAllocator->bind_vulkan_buffer(_h_memory, memoryOffset, hBuffer, pNext);
}

VkResult VmaDeviceMemoryBlock::bind_image_memory(
    VmaAllocator hAllocator,
    VmaAllocation hAllocation,
    VkDeviceSize allocationLocalOffset,
    VkImage hImage,
    const void* pNext)
{
    VMA_ASSERT(hAllocation->get_type() == VmaAllocation_T::ALLOCATION_TYPE_BLOCK &&
        hAllocation->get_block() == this);
    VMA_ASSERT(allocationLocalOffset < hAllocation->get_size() &&
        "Invalid allocationLocalOffset. Did you forget that this offset is relative to the beginning of the allocation, not the whole memory block?");
    const VkDeviceSize memoryOffset = hAllocation->get_offset() + allocationLocalOffset;
    // This lock is important so that we don't call vkBind... and/or vkMap... simultaneously on the same VkDeviceMemory from multiple threads.
    VmaMutexLock lock(_map_and_bind_mutex, hAllocator->_use_mutex);
    return hAllocator->bind_vulkan_image(_h_memory, memoryOffset, hImage, pNext);
}

#if VMA_EXTERNAL_MEMORY_WIN32
VkResult VmaDeviceMemoryBlock::create_win32_handle(const VmaAllocator hAllocator, PFN_vkGetMemoryWin32HandleKHR pvkGetMemoryWin32HandleKHR, HANDLE hTargetProcess, HANDLE* pHandle) noexcept
{
    VMA_ASSERT(pHandle);
    return _handle.get_handle(hAllocator->_h_device, _h_memory, pvkGetMemoryWin32HandleKHR, hTargetProcess, hAllocator->_use_mutex, pHandle);
}
#endif // VMA_EXTERNAL_MEMORY_WIN32
#endif // _VMA_DEVICE_MEMORY_BLOCK_FUNCTIONS

#ifndef _VMA_ALLOCATION_T_FUNCTIONS
VmaAllocation_T::VmaAllocation_T(bool mappingAllowed)
    : _alignment{ 1 },
    _size{ 0 },
    _p_user_data{ VMA_NULL },
    _p_name{ VMA_NULL },
    _memory_type_index{ 0 },
    _type{ (uint8_t)ALLOCATION_TYPE_NONE },
    _suballocation_type{ (uint8_t)VMA_SUBALLOCATION_TYPE_UNKNOWN },
    _map_count{ 0 },
    _flags{ 0 }
{
    if(mappingAllowed)
        _flags |= (uint8_t)FLAG_MAPPING_ALLOWED;
}

VmaAllocation_T::~VmaAllocation_T()
{
    VMA_ASSERT_LEAK(_map_count == 0 && "Allocation was not unmapped before destruction.");

    // Check if owned string was freed.
    VMA_ASSERT(_p_name == VMA_NULL);
}

void VmaAllocation_T::init_block_allocation(
    VmaDeviceMemoryBlock* block,
    VmaAllocHandle allocHandle,
    VkDeviceSize alignment,
    VkDeviceSize size,
    uint32_t memoryTypeIndex,
    VmaSuballocationType suballocationType,
    bool mapped)
{
    VMA_ASSERT(_type == ALLOCATION_TYPE_NONE);
    VMA_ASSERT(block != VMA_NULL);
    _type = (uint8_t)ALLOCATION_TYPE_BLOCK;
    _alignment = alignment;
    _size = size;
    _memory_type_index = memoryTypeIndex;
    if(mapped)
    {
        VMA_ASSERT(is_mapping_allowed() && "Mapping is not allowed on this allocation! Please use one of the new VMA_ALLOCATION_CREATE_HOST_ACCESS_* flags when creating it.");
        _flags |= (uint8_t)FLAG_PERSISTENT_MAP;
    }
    _suballocation_type = (uint8_t)suballocationType;
    _block_allocation._block = block;
    _block_allocation._alloc_handle = allocHandle;
}

void VmaAllocation_T::init_dedicated_allocation(
    VmaAllocator allocator,
    VmaPool hParentPool,
    uint32_t memoryTypeIndex,
    VkDeviceMemory hMemory,
    VmaSuballocationType suballocationType,
    void* pMappedData,
    VkDeviceSize size)
{
    VMA_ASSERT(_type == ALLOCATION_TYPE_NONE);
    VMA_ASSERT(hMemory != VK_NULL_HANDLE);
    _type = (uint8_t)ALLOCATION_TYPE_DEDICATED;
    _alignment = 0;
    _size = size;
    _memory_type_index = memoryTypeIndex;
    _suballocation_type = (uint8_t)suballocationType;
    _dedicated_allocation._extra_data = VMA_NULL;
    _dedicated_allocation._h_parent_pool = hParentPool;
    _dedicated_allocation._h_memory = hMemory;
    _dedicated_allocation._prev = VMA_NULL;
    _dedicated_allocation._next = VMA_NULL;

    if (pMappedData != VMA_NULL)
    {
        VMA_ASSERT(is_mapping_allowed() && "Mapping is not allowed on this allocation! Please use one of the new VMA_ALLOCATION_CREATE_HOST_ACCESS_* flags when creating it.");
        _flags |= (uint8_t)FLAG_PERSISTENT_MAP;
        ensure_extra_data(allocator);
        _dedicated_allocation._extra_data->_p_mapped_data = pMappedData;
    }
}

void VmaAllocation_T::destroy(VmaAllocator allocator)
{
    free_name(allocator);

    if (get_type() == ALLOCATION_TYPE_DEDICATED)
    {
        vma_delete(allocator, _dedicated_allocation._extra_data);
    }
}

void VmaAllocation_T::set_name(VmaAllocator hAllocator, const char* pName)
{
    VMA_ASSERT(pName == VMA_NULL || pName != _p_name);

    free_name(hAllocator);

    if (pName != VMA_NULL)
        _p_name = VmaCreateStringCopy(hAllocator->get_allocation_callbacks(), pName);
}

uint8_t VmaAllocation_T::swap_block_allocation(VmaAllocator hAllocator, VmaAllocation allocation)
{
    VMA_ASSERT(allocation != VMA_NULL);
    VMA_ASSERT(_type == ALLOCATION_TYPE_BLOCK);
    VMA_ASSERT(allocation->_type == ALLOCATION_TYPE_BLOCK);

    if (_map_count != 0)
        _block_allocation._block->unmap(hAllocator, _map_count);

    _block_allocation._block->_p_metadata->set_allocation_user_data(_block_allocation._alloc_handle, allocation);
    std::swap(_block_allocation, allocation->_block_allocation);
    _block_allocation._block->_p_metadata->set_allocation_user_data(_block_allocation._alloc_handle, this);

#if VMA_STATS_STRING_ENABLED
    std::swap(_buffer_image_usage, allocation->_buffer_image_usage);
#endif
    return _map_count;
}

VmaAllocHandle VmaAllocation_T::get_alloc_handle() const
{
    switch (_type)
    {
    case ALLOCATION_TYPE_BLOCK:
        return _block_allocation._alloc_handle;
    case ALLOCATION_TYPE_DEDICATED:
        return VK_NULL_HANDLE;
    default:
        VMA_ASSERT(0);
        return VK_NULL_HANDLE;
    }
}

VkDeviceSize VmaAllocation_T::get_offset() const
{
    switch (_type)
    {
    case ALLOCATION_TYPE_BLOCK:
        return _block_allocation._block->_p_metadata->get_allocation_offset(_block_allocation._alloc_handle);
    case ALLOCATION_TYPE_DEDICATED:
        return 0;
    default:
        VMA_ASSERT(0);
        return 0;
    }
}

VmaPool VmaAllocation_T::get_parent_pool() const
{
    switch (_type)
    {
    case ALLOCATION_TYPE_BLOCK:
        return _block_allocation._block->get_parent_pool();
    case ALLOCATION_TYPE_DEDICATED:
        return _dedicated_allocation._h_parent_pool;
    default:
        VMA_ASSERT(0);
        return VK_NULL_HANDLE;
    }
}

VkDeviceMemory VmaAllocation_T::get_memory() const
{
    switch (_type)
    {
    case ALLOCATION_TYPE_BLOCK:
        return _block_allocation._block->get_device_memory();
    case ALLOCATION_TYPE_DEDICATED:
        return _dedicated_allocation._h_memory;
    default:
        VMA_ASSERT(0);
        return VK_NULL_HANDLE;
    }
}

void* VmaAllocation_T::get_mapped_data() const
{
    switch (_type)
    {
    case ALLOCATION_TYPE_BLOCK:
        if (_map_count != 0 || is_persistent_map())
        {
            void* pBlockData = _block_allocation._block->get_mapped_data();
            VMA_ASSERT(pBlockData != VMA_NULL);
            return (char*)pBlockData + get_offset();
        }
        else
        {
            return VMA_NULL;
        }
        break;
    case ALLOCATION_TYPE_DEDICATED:
        VMA_ASSERT((_dedicated_allocation._extra_data != VMA_NULL && _dedicated_allocation._extra_data->_p_mapped_data != VMA_NULL) ==
            (_map_count != 0 || is_persistent_map()));
        return _dedicated_allocation._extra_data != VMA_NULL ? _dedicated_allocation._extra_data->_p_mapped_data : VMA_NULL;
    default:
        VMA_ASSERT(0);
        return VMA_NULL;
    }
}

void VmaAllocation_T::block_alloc_map()
{
    VMA_ASSERT(get_type() == ALLOCATION_TYPE_BLOCK);
    VMA_ASSERT(is_mapping_allowed() && "Mapping is not allowed on this allocation! Please use one of the new VMA_ALLOCATION_CREATE_HOST_ACCESS_* flags when creating it.");

    if (_map_count < 0xFF)
    {
        ++_map_count;
    }
    else
    {
        VMA_ASSERT(0 && "Allocation mapped too many times simultaneously.");
    }
}

void VmaAllocation_T::block_alloc_unmap()
{
    VMA_ASSERT(get_type() == ALLOCATION_TYPE_BLOCK);

    if (_map_count > 0)
    {
        --_map_count;
    }
    else
    {
        VMA_ASSERT(0 && "Unmapping allocation not previously mapped.");
    }
}

VkResult VmaAllocation_T::dedicated_alloc_map(VmaAllocator hAllocator, void** ppData)
{
    VMA_ASSERT(get_type() == ALLOCATION_TYPE_DEDICATED);
    VMA_ASSERT(is_mapping_allowed() && "Mapping is not allowed on this allocation! Please use one of the new VMA_ALLOCATION_CREATE_HOST_ACCESS_* flags when creating it.");

    ensure_extra_data(hAllocator);

    if (_map_count != 0 || is_persistent_map())
    {
        if (_map_count < 0xFF)
        {
            VMA_ASSERT(_dedicated_allocation._extra_data->_p_mapped_data != VMA_NULL);
            *ppData = _dedicated_allocation._extra_data->_p_mapped_data;
            ++_map_count;
            return VK_SUCCESS;
        }

        VMA_ASSERT(0 && "Dedicated allocation mapped too many times simultaneously.");
        return VK_ERROR_MEMORY_MAP_FAILED;
    }

    VkResult result = (*hAllocator->get_vulkan_functions().vkMapMemory)(
        hAllocator->_h_device,
        _dedicated_allocation._h_memory,
        0, // offset
        VK_WHOLE_SIZE,
        0, // flags
        ppData);
    if (result == VK_SUCCESS)
    {
        _dedicated_allocation._extra_data->_p_mapped_data = *ppData;
        _map_count = 1;
    }
    return result;
}

void VmaAllocation_T::dedicated_alloc_unmap(VmaAllocator hAllocator)
{
    VMA_ASSERT(get_type() == ALLOCATION_TYPE_DEDICATED);

    if (_map_count > 0)
    {
        --_map_count;
        if (_map_count == 0 && !is_persistent_map())
        {
            VMA_ASSERT(_dedicated_allocation._extra_data != VMA_NULL);
            _dedicated_allocation._extra_data->_p_mapped_data = VMA_NULL;
            (*hAllocator->get_vulkan_functions().vkUnmapMemory)(
                hAllocator->_h_device,
                _dedicated_allocation._h_memory);
        }
    }
    else
    {
        VMA_ASSERT(0 && "Unmapping dedicated allocation not previously mapped.");
    }
}

#if VMA_STATS_STRING_ENABLED
void VmaAllocation_T::print_parameters(class VmaJsonWriter& json) const
{
    json.write_string("Type");
    json.write_string(VMA_SUBALLOCATION_TYPE_NAMES[_suballocation_type]);

    json.write_string("Size");
    json.write_number(_size);
    json.write_string("Usage");
    json.write_number(_buffer_image_usage.Value); // It may be uint32_t or uint64_t.

    if (_p_user_data != VMA_NULL)
    {
        json.write_string("CustomData");
        json.begin_string();
        json.continue_string_pointer(_p_user_data);
        json.end_string();
    }
    if (_p_name != VMA_NULL)
    {
        json.write_string("Name");
        json.write_string(_p_name);
    }
}
#if VMA_EXTERNAL_MEMORY_WIN32
VkResult VmaAllocation_T::get_win32_handle(VmaAllocator hAllocator, HANDLE hTargetProcess, HANDLE* pHandle) noexcept
{
    auto pvkGetMemoryWin32HandleKHR = hAllocator->get_vulkan_functions().vkGetMemoryWin32HandleKHR;
    switch (_type)
    {
    case ALLOCATION_TYPE_BLOCK:
        return _block_allocation._block->create_win32_handle(hAllocator, pvkGetMemoryWin32HandleKHR, hTargetProcess, pHandle);
    case ALLOCATION_TYPE_DEDICATED:
        ensure_extra_data(hAllocator);
        return _dedicated_allocation._extra_data->_handle.get_handle(hAllocator->_h_device, _dedicated_allocation._h_memory, pvkGetMemoryWin32HandleKHR, hTargetProcess, hAllocator->_use_mutex, pHandle);
    default:
        VMA_ASSERT(0);
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
}
#endif // VMA_EXTERNAL_MEMORY_WIN32
#endif // VMA_STATS_STRING_ENABLED

void VmaAllocation_T::ensure_extra_data(VmaAllocator hAllocator)
{
    if (_dedicated_allocation._extra_data == VMA_NULL)
    {
        _dedicated_allocation._extra_data = vma_new(hAllocator, VmaAllocationExtraData)();
    }
}

void VmaAllocation_T::free_name(VmaAllocator hAllocator)
{
    if(_p_name)
    {
        VmaFreeString(hAllocator->get_allocation_callbacks(), _p_name);
        _p_name = VMA_NULL;
    }
}
#endif // _VMA_ALLOCATION_T_FUNCTIONS

#ifndef _VMA_BLOCK_VECTOR_FUNCTIONS
VmaBlockVector::VmaBlockVector(
    VmaAllocator hAllocator,
    VmaPool hParentPool,
    uint32_t memoryTypeIndex,
    VkDeviceSize preferredBlockSize,
    size_t minBlockCount,
    size_t maxBlockCount,
    VkDeviceSize bufferImageGranularity,
    bool explicitBlockSize,
    uint32_t algorithm,
    float priority,
    VkDeviceSize minAllocationAlignment,
    void* pMemoryAllocateNext)
    : _h_allocator(hAllocator),
    _h_parent_pool(hParentPool),
    _memory_type_index(memoryTypeIndex),
    _preferred_block_size(preferredBlockSize),
    _min_block_count(minBlockCount),
    _max_block_count(maxBlockCount),
    _buffer_image_granularity(bufferImageGranularity),
    _explicit_block_size(explicitBlockSize),
    _algorithm(algorithm),
    _priority(priority),
    _min_allocation_alignment(minAllocationAlignment),
    _p_memory_allocate_next(pMemoryAllocateNext),
    _blocks(VmaStlAllocator<VmaDeviceMemoryBlock*>(hAllocator->get_allocation_callbacks())),
    _next_block_id(0) {}

VmaBlockVector::~VmaBlockVector()
{
    for (size_t i = _blocks.size(); i--; )
    {
        _blocks[i]->destroy(_h_allocator);
        vma_delete(_h_allocator, _blocks[i]);
    }
}

VkResult VmaBlockVector::create_min_blocks()
{
    for (size_t i = 0; i < _min_block_count; ++i)
    {
        VkResult res = create_block(_preferred_block_size, VMA_NULL);
        if (res != VK_SUCCESS)
        {
            return res;
        }
    }
    return VK_SUCCESS;
}

void VmaBlockVector::add_statistics(VmaStatistics& inoutStats)
{
    VmaMutexLockRead lock(_mutex, _h_allocator->_use_mutex);

    const size_t blockCount = _blocks.size();
    for (uint32_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
    {
        const VmaDeviceMemoryBlock* const pBlock = _blocks[blockIndex];
        VMA_ASSERT(pBlock);
        VMA_HEAVY_ASSERT(pBlock->validate());
        pBlock->_p_metadata->add_statistics(inoutStats);
    }
}

void VmaBlockVector::add_detailed_statistics(VmaDetailedStatistics& inoutStats)
{
    VmaMutexLockRead lock(_mutex, _h_allocator->_use_mutex);

    const size_t blockCount = _blocks.size();
    for (uint32_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
    {
        const VmaDeviceMemoryBlock* const pBlock = _blocks[blockIndex];
        VMA_ASSERT(pBlock);
        VMA_HEAVY_ASSERT(pBlock->validate());
        pBlock->_p_metadata->add_detailed_statistics(inoutStats);
    }
}

bool VmaBlockVector::is_empty()
{
    VmaMutexLockRead lock(_mutex, _h_allocator->_use_mutex);
    return _blocks.empty();
}

bool VmaBlockVector::is_corruption_detection_enabled() const
{
    const uint32_t requiredMemFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    return (VMA_DEBUG_DETECT_CORRUPTION != 0) &&
        (VMA_DEBUG_MARGIN > 0) &&
        (_algorithm == 0 || _algorithm == VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT) &&
        (_h_allocator->_mem_props.memoryTypes[_memory_type_index].propertyFlags & requiredMemFlags) == requiredMemFlags;
}

VkResult VmaBlockVector::Allocate(
    VkDeviceSize size,
    VkDeviceSize alignment,
    const VmaAllocationCreateInfo& createInfo,
    VmaSuballocationType suballocType,
    size_t allocationCount,
    VmaAllocation* pAllocations)
{
    size_t allocIndex = 0;
    VkResult res = VK_SUCCESS;

    alignment = VMA_MAX(alignment, _min_allocation_alignment);

    if (is_corruption_detection_enabled())
    {
        size = VmaAlignUp<VkDeviceSize>(size, sizeof(VMA_CORRUPTION_DETECTION_MAGIC_VALUE));
        alignment = VmaAlignUp<VkDeviceSize>(alignment, sizeof(VMA_CORRUPTION_DETECTION_MAGIC_VALUE));
    }

    {
        VmaMutexLockWrite lock(_mutex, _h_allocator->_use_mutex);
        for (; allocIndex < allocationCount; ++allocIndex)
        {
            res = allocate_page(
                size,
                alignment,
                createInfo,
                suballocType,
                pAllocations + allocIndex);
            if (res != VK_SUCCESS)
            {
                break;
            }
        }
    }

    if (res != VK_SUCCESS)
    {
        // Free all already created allocations.
        while (allocIndex--)
            free(pAllocations[allocIndex]);
        memset(pAllocations, 0, sizeof(VmaAllocation) * allocationCount);
    }

    return res;
}

VkResult VmaBlockVector::allocate_page(
    VkDeviceSize size,
    VkDeviceSize alignment,
    const VmaAllocationCreateInfo& createInfo,
    VmaSuballocationType suballocType,
    VmaAllocation* pAllocation)
{
    const bool isUpperAddress = (createInfo.flags & VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT) != 0;

    VkDeviceSize freeMemory = 0;
    {
        const uint32_t heapIndex = _h_allocator->memory_type_index_to_heap_index(_memory_type_index);
        VmaBudget heapBudget = {};
        _h_allocator->get_heap_budgets(&heapBudget, heapIndex, 1);
        freeMemory = (heapBudget.usage < heapBudget.budget) ? (heapBudget.budget - heapBudget.usage) : 0;
    }

    const bool canFallbackToDedicated = !has_explicit_block_size() &&
        (createInfo.flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) == 0;
    const bool canCreateNewBlock =
        ((createInfo.flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) == 0) &&
        (_blocks.size() < _max_block_count) &&
        (freeMemory >= size || !canFallbackToDedicated);
    uint32_t strategy = createInfo.flags & VMA_ALLOCATION_CREATE_STRATEGY_MASK;

    // Upper address can only be used with linear allocator and within single memory block.
    if (isUpperAddress &&
        (_algorithm != VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT || _max_block_count > 1))
    {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    // Early reject: requested allocation size is larger that maximum block size for this block vector.
    if (size + VMA_DEBUG_MARGIN > _preferred_block_size)
    {
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    // 1. Search existing allocations. Try to allocate.
    if (_algorithm == VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT)
    {
        // Use only last block.
        if (!_blocks.empty())
        {
            VmaDeviceMemoryBlock* const pCurrBlock = _blocks.back();
            VMA_ASSERT(pCurrBlock);
            VkResult res = allocate_from_block(
                pCurrBlock, size, alignment, createInfo.flags, createInfo.pUserData, suballocType, strategy, pAllocation);
            if (res == VK_SUCCESS)
            {
                VMA_DEBUG_LOG_FORMAT("    Returned from last block #%" PRIu32, pCurrBlock->get_id());
                incrementally_sort_blocks();
                return VK_SUCCESS;
            }
        }
    }
    else
    {
        if (strategy != VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT) // MIN_MEMORY or default
        {
            const bool isHostVisible =
                (_h_allocator->_mem_props.memoryTypes[_memory_type_index].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
            if(isHostVisible)
            {
                const bool isMappingAllowed = (createInfo.flags &
                    (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0;
                /*
                For non-mappable allocations, check blocks that are not mapped first.
                For mappable allocations, check blocks that are already mapped first.
                This way, having many blocks, we will separate mappable and non-mappable allocations,
                hopefully limiting the number of blocks that are mapped, which will help tools like RenderDoc.
                */
                for(size_t mappingI = 0; mappingI < 2; ++mappingI)
                {
                    // Forward order in _blocks - prefer blocks with smallest amount of free space.
                    for (size_t blockIndex = 0; blockIndex < _blocks.size(); ++blockIndex)
                    {
                        VmaDeviceMemoryBlock* const pCurrBlock = _blocks[blockIndex];
                        VMA_ASSERT(pCurrBlock);
                        const bool isBlockMapped = pCurrBlock->get_mapped_data() != VMA_NULL;
                        if((mappingI == 0) == (isMappingAllowed == isBlockMapped))
                        {
                            VkResult res = allocate_from_block(
                                pCurrBlock, size, alignment, createInfo.flags, createInfo.pUserData, suballocType, strategy, pAllocation);
                            if (res == VK_SUCCESS)
                            {
                                VMA_DEBUG_LOG_FORMAT("    Returned from existing block #%" PRIu32, pCurrBlock->get_id());
                                incrementally_sort_blocks();
                                return VK_SUCCESS;
                            }
                        }
                    }
                }
            }
            else
            {
                // Forward order in _blocks - prefer blocks with smallest amount of free space.
                for (size_t blockIndex = 0; blockIndex < _blocks.size(); ++blockIndex)
                {
                    VmaDeviceMemoryBlock* const pCurrBlock = _blocks[blockIndex];
                    VMA_ASSERT(pCurrBlock);
                    VkResult res = allocate_from_block(
                        pCurrBlock, size, alignment, createInfo.flags, createInfo.pUserData, suballocType, strategy, pAllocation);
                    if (res == VK_SUCCESS)
                    {
                        VMA_DEBUG_LOG_FORMAT("    Returned from existing block #%" PRIu32, pCurrBlock->get_id());
                        incrementally_sort_blocks();
                        return VK_SUCCESS;
                    }
                }
            }
        }
        else // VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT
        {
            // Backward order in _blocks - prefer blocks with largest amount of free space.
            for (size_t blockIndex = _blocks.size(); blockIndex--; )
            {
                VmaDeviceMemoryBlock* const pCurrBlock = _blocks[blockIndex];
                VMA_ASSERT(pCurrBlock);
                VkResult res = allocate_from_block(pCurrBlock, size, alignment, createInfo.flags, createInfo.pUserData, suballocType, strategy, pAllocation);
                if (res == VK_SUCCESS)
                {
                    VMA_DEBUG_LOG_FORMAT("    Returned from existing block #%" PRIu32, pCurrBlock->get_id());
                    incrementally_sort_blocks();
                    return VK_SUCCESS;
                }
            }
        }
    }

    // 2. Try to create new block.
    if (canCreateNewBlock)
    {
        // Calculate optimal size for new block.
        VkDeviceSize newBlockSize = _preferred_block_size;
        uint32_t newBlockSizeShift = 0;
        const uint32_t NEW_BLOCK_SIZE_SHIFT_MAX = 3;

        if (!_explicit_block_size)
        {
            // Allocate 1/8, 1/4, 1/2 as first blocks.
            const VkDeviceSize maxExistingBlockSize = calc_max_block_size();
            for (uint32_t i = 0; i < NEW_BLOCK_SIZE_SHIFT_MAX; ++i)
            {
                const VkDeviceSize smallerNewBlockSize = newBlockSize / 2;
                if (smallerNewBlockSize > maxExistingBlockSize && smallerNewBlockSize >= size * 2)
                {
                    newBlockSize = smallerNewBlockSize;
                    ++newBlockSizeShift;
                }
                else
                {
                    break;
                }
            }
        }

        size_t newBlockIndex = 0;
        VkResult res = (newBlockSize <= freeMemory || !canFallbackToDedicated) ?
            create_block(newBlockSize, &newBlockIndex) : VK_ERROR_OUT_OF_DEVICE_MEMORY;
        // Allocation of this size failed? Try 1/2, 1/4, 1/8 of _preferred_block_size.
        if (!_explicit_block_size)
        {
            while (res < 0 && newBlockSizeShift < NEW_BLOCK_SIZE_SHIFT_MAX)
            {
                const VkDeviceSize smallerNewBlockSize = newBlockSize / 2;
                if (smallerNewBlockSize >= size)
                {
                    newBlockSize = smallerNewBlockSize;
                    ++newBlockSizeShift;
                    res = (newBlockSize <= freeMemory || !canFallbackToDedicated) ?
                        create_block(newBlockSize, &newBlockIndex) : VK_ERROR_OUT_OF_DEVICE_MEMORY;
                }
                else
                {
                    break;
                }
            }
        }

        if (res == VK_SUCCESS)
        {
            VmaDeviceMemoryBlock* const pBlock = _blocks[newBlockIndex];
            VMA_ASSERT(pBlock->_p_metadata->get_size() >= size);

            res = allocate_from_block(
                pBlock, size, alignment, createInfo.flags, createInfo.pUserData, suballocType, strategy, pAllocation);
            if (res == VK_SUCCESS)
            {
                VMA_DEBUG_LOG_FORMAT("    Created new block #%" PRIu32 " Size=%" PRIu64, pBlock->get_id(), newBlockSize);
                incrementally_sort_blocks();
                return VK_SUCCESS;
            }

            // Allocation from new block failed, possibly due to VMA_DEBUG_MARGIN or alignment.
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
    }

    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

void VmaBlockVector::free(VmaAllocation hAllocation)
{
    VmaDeviceMemoryBlock* pBlockToDelete = VMA_NULL;

    bool budgetExceeded = false;
    {
        const uint32_t heapIndex = _h_allocator->memory_type_index_to_heap_index(_memory_type_index);
        VmaBudget heapBudget = {};
        _h_allocator->get_heap_budgets(&heapBudget, heapIndex, 1);
        budgetExceeded = heapBudget.usage >= heapBudget.budget;
    }

    // Scope for lock.
    {
        VmaMutexLockWrite lock(_mutex, _h_allocator->_use_mutex);

        VmaDeviceMemoryBlock* pBlock = hAllocation->get_block();

        if (is_corruption_detection_enabled())
        {
            VkResult res = pBlock->validate_magic_value_after_allocation(_h_allocator, hAllocation->get_offset(), hAllocation->get_size());
            VMA_ASSERT(res == VK_SUCCESS && "Couldn't map block memory to validate magic value.");
        }

        if (hAllocation->is_persistent_map())
        {
            pBlock->unmap(_h_allocator, 1);
        }

        const bool hadEmptyBlockBeforeFree = has_empty_block();
        pBlock->_p_metadata->free(hAllocation->get_alloc_handle());
        pBlock->post_free(_h_allocator);
        VMA_HEAVY_ASSERT(pBlock->validate());

        VMA_DEBUG_LOG_FORMAT("  Freed from MemoryTypeIndex=%" PRIu32, _memory_type_index);

        const bool canDeleteBlock = _blocks.size() > _min_block_count;
        // pBlock became empty after this deallocation.
        if (pBlock->_p_metadata->is_empty())
        {
            // Already had empty block. We don't want to have two, so delete this one.
            if ((hadEmptyBlockBeforeFree || budgetExceeded) && canDeleteBlock)
            {
                pBlockToDelete = pBlock;
                remove(pBlock);
            }
            // else: We now have one empty block - leave it. A hysteresis to avoid allocating whole block back and forth.
        }
        // pBlock didn't become empty, but we have another empty block - find and free that one.
        // (This is optional, heuristics.)
        else if (hadEmptyBlockBeforeFree && canDeleteBlock)
        {
            VmaDeviceMemoryBlock* pLastBlock = _blocks.back();
            if (pLastBlock->_p_metadata->is_empty())
            {
                pBlockToDelete = pLastBlock;
                _blocks.pop_back();
            }
        }

        incrementally_sort_blocks();

        _h_allocator->_budget.remove_allocation(_h_allocator->memory_type_index_to_heap_index(_memory_type_index), hAllocation->get_size());
        hAllocation->destroy(_h_allocator);
        _h_allocator->_allocation_object_allocator.free(hAllocation);
    }

    // Destruction of a free block. Deferred until this point, outside of mutex
    // lock, for performance reason.
    if (pBlockToDelete != VMA_NULL)
    {
        VMA_DEBUG_LOG_FORMAT("    Deleted empty block #%" PRIu32, pBlockToDelete->get_id());
        pBlockToDelete->destroy(_h_allocator);
        vma_delete(_h_allocator, pBlockToDelete);
    }
}

VkDeviceSize VmaBlockVector::calc_max_block_size() const
{
    VkDeviceSize result = 0;
    for (size_t i = _blocks.size(); i--; )
    {
        result = VMA_MAX(result, _blocks[i]->_p_metadata->get_size());
        if (result >= _preferred_block_size)
        {
            break;
        }
    }
    return result;
}

void VmaBlockVector::remove(VmaDeviceMemoryBlock* pBlock)
{
    for (uint32_t blockIndex = 0; blockIndex < _blocks.size(); ++blockIndex)
    {
        if (_blocks[blockIndex] == pBlock)
        {
            VmaVectorRemove(_blocks, blockIndex);
            return;
        }
    }
    VMA_ASSERT(0);
}

void VmaBlockVector::incrementally_sort_blocks()
{
    if (!_incremental_sort)
        return;
    if (_algorithm != VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT)
    {
        // Bubble sort only until first swap.
        for (size_t i = 1; i < _blocks.size(); ++i)
        {
            if (_blocks[i - 1]->_p_metadata->get_sum_free_size() > _blocks[i]->_p_metadata->get_sum_free_size())
            {
                std::swap(_blocks[i - 1], _blocks[i]);
                return;
            }
        }
    }
}

void VmaBlockVector::sort_by_free_size()
{
    VMA_SORT(_blocks.begin(), _blocks.end(),
        [](VmaDeviceMemoryBlock* b1, VmaDeviceMemoryBlock* b2) -> bool
        {
            return b1->_p_metadata->get_sum_free_size() < b2->_p_metadata->get_sum_free_size();
        });
}

VkResult VmaBlockVector::allocate_from_block(
    VmaDeviceMemoryBlock* pBlock,
    VkDeviceSize size,
    VkDeviceSize alignment,
    VmaAllocationCreateFlags allocFlags,
    void* pUserData,
    VmaSuballocationType suballocType,
    uint32_t strategy,
    VmaAllocation* pAllocation)
{
    const bool isUpperAddress = (allocFlags & VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT) != 0;

    VmaAllocationRequest currRequest = {};
    if (pBlock->_p_metadata->create_allocation_request(
        size,
        alignment,
        isUpperAddress,
        suballocType,
        strategy,
        &currRequest))
    {
        return commit_allocation_request(currRequest, pBlock, alignment, allocFlags, pUserData, suballocType, pAllocation);
    }
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

VkResult VmaBlockVector::commit_allocation_request(
    VmaAllocationRequest& allocRequest,
    VmaDeviceMemoryBlock* pBlock,
    VkDeviceSize alignment,
    VmaAllocationCreateFlags allocFlags,
    void* pUserData,
    VmaSuballocationType suballocType,
    VmaAllocation* pAllocation)
{
    const bool mapped = (allocFlags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0;
    const bool isUserDataString = (allocFlags & VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT) != 0;
    const bool isMappingAllowed = (allocFlags &
        (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0;

    pBlock->post_alloc(_h_allocator);
    // Allocate from pCurrBlock.
    if (mapped)
    {
        VkResult res = pBlock->Map(_h_allocator, 1, VMA_NULL);
        if (res != VK_SUCCESS)
        {
            return res;
        }
    }

    *pAllocation = _h_allocator->_allocation_object_allocator.Allocate(isMappingAllowed);
    pBlock->_p_metadata->alloc(allocRequest, suballocType, *pAllocation);
    (*pAllocation)->init_block_allocation(
        pBlock,
        allocRequest.allocHandle,
        alignment,
        allocRequest.size, // Not size, as actual allocation size may be larger than requested!
        _memory_type_index,
        suballocType,
        mapped);
    VMA_HEAVY_ASSERT(pBlock->validate());
    if (isUserDataString)
        (*pAllocation)->set_name(_h_allocator, (const char*)pUserData);
    else
        (*pAllocation)->set_user_data(_h_allocator, pUserData);
    _h_allocator->_budget.add_allocation(_h_allocator->memory_type_index_to_heap_index(_memory_type_index), allocRequest.size);
    if (VMA_DEBUG_INITIALIZE_ALLOCATIONS)
    {
        _h_allocator->fill_allocation(*pAllocation, VMA_ALLOCATION_FILL_PATTERN_CREATED);
    }
    if (is_corruption_detection_enabled())
    {
        VkResult res = pBlock->write_magic_value_after_allocation(_h_allocator, (*pAllocation)->get_offset(), allocRequest.size);
        VMA_ASSERT(res == VK_SUCCESS && "Couldn't map block memory to write magic value.");
    }
    return VK_SUCCESS;
}

VkResult VmaBlockVector::create_block(VkDeviceSize blockSize, size_t* pNewBlockIndex)
{
    VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocInfo.pNext = _p_memory_allocate_next;
    allocInfo.memoryTypeIndex = _memory_type_index;
    allocInfo.allocationSize = blockSize;

#if VMA_BUFFER_DEVICE_ADDRESS
    // Every standalone block can potentially contain a buffer with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT - always enable the feature.
    VkMemoryAllocateFlagsInfoKHR allocFlagsInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_KHR };
    if (_h_allocator->_use_khr_buffer_device_address)
    {
        allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
        VmaPnextChainPushFront(&allocInfo, &allocFlagsInfo);
    }
#endif // VMA_BUFFER_DEVICE_ADDRESS

#if VMA_MEMORY_PRIORITY
    VkMemoryPriorityAllocateInfoEXT priorityInfo = { VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT };
    if (_h_allocator->_use_ext_memory_priority)
    {
        VMA_ASSERT(_priority >= 0.F && _priority <= 1.F);
        priorityInfo.priority = _priority;
        VmaPnextChainPushFront(&allocInfo, &priorityInfo);
    }
#endif // VMA_MEMORY_PRIORITY

#if VMA_EXTERNAL_MEMORY
    // Attach VkExportMemoryAllocateInfoKHR if necessary.
    VkExportMemoryAllocateInfoKHR exportMemoryAllocInfo = { VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR };
    exportMemoryAllocInfo.handleTypes = _h_allocator->get_external_memory_handle_type_flags(_memory_type_index);
    if (exportMemoryAllocInfo.handleTypes != 0)
    {
        VmaPnextChainPushFront(&allocInfo, &exportMemoryAllocInfo);
    }
#endif // VMA_EXTERNAL_MEMORY

    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkResult res = _h_allocator->allocate_vulkan_memory(&allocInfo, &mem);
    if (res < 0)
    {
        return res;
    }

    // New VkDeviceMemory successfully created.

    // Create new Allocation for it.
    VmaDeviceMemoryBlock* const pBlock = vma_new(_h_allocator, VmaDeviceMemoryBlock)(_h_allocator);
    pBlock->init(
        _h_allocator,
        _h_parent_pool,
        _memory_type_index,
        mem,
        allocInfo.allocationSize,
        _next_block_id++,
        _algorithm,
        _buffer_image_granularity);

    _blocks.push_back(pBlock);
    if (pNewBlockIndex != VMA_NULL)
    {
        *pNewBlockIndex = _blocks.size() - 1;
    }

    return VK_SUCCESS;
}

bool VmaBlockVector::has_empty_block()
{
    for (size_t index = 0, count = _blocks.size(); index < count; ++index)
    {
        VmaDeviceMemoryBlock* const pBlock = _blocks[index];
        if (pBlock->_p_metadata->is_empty())
        {
            return true;
        }
    }
    return false;
}

#if VMA_STATS_STRING_ENABLED
void VmaBlockVector::print_detailed_map(class VmaJsonWriter& json)
{
    VmaMutexLockRead lock(_mutex, _h_allocator->_use_mutex);


    json.begin_object();
    for (size_t i = 0; i < _blocks.size(); ++i)
    {
        json.begin_string();
        json.continue_string(_blocks[i]->get_id());
        json.end_string();

        json.begin_object();
        json.write_string("MapRefCount");
        json.write_number(_blocks[i]->get_map_ref_count());

        _blocks[i]->_p_metadata->print_detailed_map(json);
        json.end_object();
    }
    json.end_object();
}
#endif // VMA_STATS_STRING_ENABLED

VkResult VmaBlockVector::check_corruption()
{
    if (!is_corruption_detection_enabled())
    {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    VmaMutexLockRead lock(_mutex, _h_allocator->_use_mutex);
    for (uint32_t blockIndex = 0; blockIndex < _blocks.size(); ++blockIndex)
    {
        VmaDeviceMemoryBlock* const pBlock = _blocks[blockIndex];
        VMA_ASSERT(pBlock);
        VkResult res = pBlock->check_corruption(_h_allocator);
        if (res != VK_SUCCESS)
        {
            return res;
        }
    }
    return VK_SUCCESS;
}

#endif // _VMA_BLOCK_VECTOR_FUNCTIONS

#ifndef _VMA_DEFRAGMENTATION_CONTEXT_FUNCTIONS
VmaDefragmentationContext_T::VmaDefragmentationContext_T(
    VmaAllocator hAllocator,
    const VmaDefragmentationInfo& info)
    : _max_pass_bytes(info.maxBytesPerPass == 0 ? VK_WHOLE_SIZE : info.maxBytesPerPass),
    _max_pass_allocations(info.maxAllocationsPerPass == 0 ? UINT32_MAX : info.maxAllocationsPerPass),
    _break_callback(info.pfnBreakCallback),
    _break_callback_user_data(info.pBreakCallbackUserData),
    _move_allocator(hAllocator->get_allocation_callbacks()),
    _moves(_move_allocator),
    _algorithm(info.flags & VMA_DEFRAGMENTATION_FLAG_ALGORITHM_MASK)
{
    if (info.pool != VMA_NULL)
    {
        _block_vector_count = 1;
        _pool_block_vector = &info.pool->_block_vector;
        _p_block_vectors = &_pool_block_vector;
        _pool_block_vector->set_incremental_sort(false);
        _pool_block_vector->sort_by_free_size();
    }
    else
    {
        _block_vector_count = hAllocator->get_memory_type_count();
        _pool_block_vector = VMA_NULL;
        _p_block_vectors = hAllocator->_p_block_vectors;
        for (uint32_t i = 0; i < _block_vector_count; ++i)
        {
            VmaBlockVector* vector = _p_block_vectors[i];
            if (vector != VMA_NULL)
            {
                vector->set_incremental_sort(false);
                vector->sort_by_free_size();
            }
        }
    }

    switch (_algorithm)
    {
    case 0: // Default algorithm
        _algorithm = VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT;
        _algorithm_state = vma_new_array(hAllocator, StateBalanced, _block_vector_count);
        break;
    case VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT:
        _algorithm_state = vma_new_array(hAllocator, StateBalanced, _block_vector_count);
        break;
    case VMA_DEFRAGMENTATION_FLAG_ALGORITHM_EXTENSIVE_BIT:
        if (hAllocator->get_buffer_image_granularity() > 1)
        {
            _algorithm_state = vma_new_array(hAllocator, StateExtensive, _block_vector_count);
        }
        break;
    default:
        ; // Do nothing.
    }
}

VmaDefragmentationContext_T::~VmaDefragmentationContext_T()
{
    if (_pool_block_vector != VMA_NULL)
    {
        _pool_block_vector->set_incremental_sort(true);
    }
    else
    {
        for (uint32_t i = 0; i < _block_vector_count; ++i)
        {
            VmaBlockVector* vector = _p_block_vectors[i];
            if (vector != VMA_NULL)
                vector->set_incremental_sort(true);
        }
    }

    if (_algorithm_state)
    {
        switch (_algorithm)
        {
        case VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT:
            vma_delete_array(_move_allocator._p_callbacks, reinterpret_cast<StateBalanced*>(_algorithm_state), _block_vector_count);
            break;
        case VMA_DEFRAGMENTATION_FLAG_ALGORITHM_EXTENSIVE_BIT:
            vma_delete_array(_move_allocator._p_callbacks, reinterpret_cast<StateExtensive*>(_algorithm_state), _block_vector_count);
            break;
        default:
            VMA_ASSERT(0);
        }
    }
}

VkResult VmaDefragmentationContext_T::defragment_pass_begin(VmaDefragmentationPassMoveInfo& moveInfo)
{
    if (_pool_block_vector != VMA_NULL)
    {
        VmaMutexLockWrite lock(_pool_block_vector->get_mutex(), _pool_block_vector->get_allocator()->_use_mutex);

        if (_pool_block_vector->get_block_count() > 1)
            compute_defragmentation(*_pool_block_vector, 0);
        else if (_pool_block_vector->get_block_count() == 1)
            realloc_within_block(*_pool_block_vector, _pool_block_vector->get_block(0));
    }
    else
    {
        for (uint32_t i = 0; i < _block_vector_count; ++i)
        {
            if (_p_block_vectors[i] != VMA_NULL)
            {
                VmaMutexLockWrite lock(_p_block_vectors[i]->get_mutex(), _p_block_vectors[i]->get_allocator()->_use_mutex);

                if (_p_block_vectors[i]->get_block_count() > 1)
                {
                    if (compute_defragmentation(*_p_block_vectors[i], i))
                        break;
                }
                else if (_p_block_vectors[i]->get_block_count() == 1)
                {
                    if (realloc_within_block(*_p_block_vectors[i], _p_block_vectors[i]->get_block(0)))
                        break;
                }
            }
        }
    }

    moveInfo.moveCount = static_cast<uint32_t>(_moves.size());
    if (moveInfo.moveCount > 0)
    {
        moveInfo.pMoves = _moves.data();
        return VK_INCOMPLETE;
    }

    moveInfo.pMoves = VMA_NULL;
    return VK_SUCCESS;
}

VkResult VmaDefragmentationContext_T::defragment_pass_end(VmaDefragmentationPassMoveInfo& moveInfo)
{
    VMA_ASSERT(moveInfo.moveCount > 0 ? moveInfo.pMoves != VMA_NULL : true);

    VkResult result = VK_SUCCESS;
    VmaStlAllocator<FragmentedBlock> blockAllocator(_move_allocator._p_callbacks);
    VmaVector<FragmentedBlock, VmaStlAllocator<FragmentedBlock>> immovableBlocks(blockAllocator);
    VmaVector<FragmentedBlock, VmaStlAllocator<FragmentedBlock>> mappedBlocks(blockAllocator);

    VmaAllocator allocator = VMA_NULL;
    for (uint32_t i = 0; i < moveInfo.moveCount; ++i)
    {
        VmaDefragmentationMove& move = moveInfo.pMoves[i];
        size_t prevCount = 0;
        size_t currentCount = 0;
        VkDeviceSize freedBlockSize = 0;

        uint32_t vectorIndex = 0;
        VmaBlockVector* vector = VMA_NULL;
        if (_pool_block_vector != VMA_NULL)
        {
            vector = _pool_block_vector;
        }
        else
        {
            vectorIndex = move.srcAllocation->get_memory_type_index();
            vector = _p_block_vectors[vectorIndex];
            VMA_ASSERT(vector != VMA_NULL);
        }

        switch (move.operation)
        {
        case VMA_DEFRAGMENTATION_MOVE_OPERATION_COPY:
        {
            uint8_t mapCount = move.srcAllocation->swap_block_allocation(vector->_h_allocator, move.dstTmpAllocation);
            if (mapCount > 0)
            {
                allocator = vector->_h_allocator;
                VmaDeviceMemoryBlock* newMapBlock = move.srcAllocation->get_block();
                bool notPresent = true;
                for (FragmentedBlock& block : mappedBlocks)
                {
                    if (block.block == newMapBlock)
                    {
                        notPresent = false;
                        block.data += mapCount;
                        break;
                    }
                }
                if (notPresent)
                    mappedBlocks.push_back({ mapCount, newMapBlock });
            }

            // Scope for locks, Free have it's own lock
            {
                VmaMutexLockRead lock(vector->get_mutex(), vector->get_allocator()->_use_mutex);
                prevCount = vector->get_block_count();
                freedBlockSize = move.dstTmpAllocation->get_block()->_p_metadata->get_size();
            }
            vector->free(move.dstTmpAllocation);
            {
                VmaMutexLockRead lock(vector->get_mutex(), vector->get_allocator()->_use_mutex);
                currentCount = vector->get_block_count();
            }

            result = VK_INCOMPLETE;
            break;
        }
        case VMA_DEFRAGMENTATION_MOVE_OPERATION_IGNORE:
        {
            _pass_stats.bytesMoved -= move.srcAllocation->get_size();
            --_pass_stats.allocationsMoved;
            vector->free(move.dstTmpAllocation);

            VmaDeviceMemoryBlock* newBlock = move.srcAllocation->get_block();
            bool notPresent = true;
            for (const FragmentedBlock& block : immovableBlocks)
            {
                if (block.block == newBlock)
                {
                    notPresent = false;
                    break;
                }
            }
            if (notPresent)
                immovableBlocks.push_back({ vectorIndex, newBlock });
            break;
        }
        case VMA_DEFRAGMENTATION_MOVE_OPERATION_DESTROY:
        {
            _pass_stats.bytesMoved -= move.srcAllocation->get_size();
            --_pass_stats.allocationsMoved;
            // Scope for locks, Free have it's own lock
            {
                VmaMutexLockRead lock(vector->get_mutex(), vector->get_allocator()->_use_mutex);
                prevCount = vector->get_block_count();
                freedBlockSize = move.srcAllocation->get_block()->_p_metadata->get_size();
            }
            vector->free(move.srcAllocation);
            {
                VmaMutexLockRead lock(vector->get_mutex(), vector->get_allocator()->_use_mutex);
                currentCount = vector->get_block_count();
            }
            freedBlockSize *= prevCount - currentCount;

            VkDeviceSize dstBlockSize = SIZE_MAX;
            {
                VmaMutexLockRead lock(vector->get_mutex(), vector->get_allocator()->_use_mutex);
                dstBlockSize = move.dstTmpAllocation->get_block()->_p_metadata->get_size();
            }
            vector->free(move.dstTmpAllocation);
            {
                VmaMutexLockRead lock(vector->get_mutex(), vector->get_allocator()->_use_mutex);
                freedBlockSize += dstBlockSize * (currentCount - vector->get_block_count());
                currentCount = vector->get_block_count();
            }

            result = VK_INCOMPLETE;
            break;
        }
        default:
            VMA_ASSERT(0);
        }

        if (prevCount > currentCount)
        {
            size_t freedBlocks = prevCount - currentCount;
            _pass_stats.deviceMemoryBlocksFreed += static_cast<uint32_t>(freedBlocks);
            _pass_stats.bytesFreed += freedBlockSize;
        }

        if(_algorithm == VMA_DEFRAGMENTATION_FLAG_ALGORITHM_EXTENSIVE_BIT &&
            _algorithm_state != VMA_NULL)
        {
            // Avoid unnecessary tries to allocate when new free block is available
            StateExtensive& state = reinterpret_cast<StateExtensive*>(_algorithm_state)[vectorIndex];
            if (state.firstFreeBlock != SIZE_MAX)
            {
                const size_t diff = prevCount - currentCount;
                if (state.firstFreeBlock >= diff)
                {
                    state.firstFreeBlock -= diff;
                    if (state.firstFreeBlock != 0)
                        state.firstFreeBlock -= vector->get_block(state.firstFreeBlock - 1)->_p_metadata->is_empty();
                }
                else
                    state.firstFreeBlock = 0;
            }
        }
    }
    moveInfo.moveCount = 0;
    moveInfo.pMoves = VMA_NULL;
    _moves.clear();

    // Update stats
    _global_stats.allocationsMoved += _pass_stats.allocationsMoved;
    _global_stats.bytesFreed += _pass_stats.bytesFreed;
    _global_stats.bytesMoved += _pass_stats.bytesMoved;
    _global_stats.deviceMemoryBlocksFreed += _pass_stats.deviceMemoryBlocksFreed;
    _pass_stats = { 0 };

    // Move blocks with immovable allocations according to algorithm
    if (!immovableBlocks.empty())
    {
        do
        {
            if(_algorithm == VMA_DEFRAGMENTATION_FLAG_ALGORITHM_EXTENSIVE_BIT)
            {
                if (_algorithm_state != VMA_NULL)
                {
                    bool swapped = false;
                    // Move to the start of free blocks range
                    for (const FragmentedBlock& block : immovableBlocks)
                    {
                        StateExtensive& state = reinterpret_cast<StateExtensive*>(_algorithm_state)[block.data];
                        if (state.operation != StateExtensive::Operation::Cleanup)
                        {
                            VmaBlockVector* vector = _p_block_vectors[block.data];
                            VmaMutexLockWrite lock(vector->get_mutex(), vector->get_allocator()->_use_mutex);

                            for (size_t i = 0, count = vector->get_block_count() - _immovable_block_count; i < count; ++i)
                            {
                                if (vector->get_block(i) == block.block)
                                {
                                    std::swap(vector->_blocks[i], vector->_blocks[vector->get_block_count() - ++_immovable_block_count]);
                                    if (state.firstFreeBlock != SIZE_MAX)
                                    {
                                        if (i + 1 < state.firstFreeBlock)
                                        {
                                            if (state.firstFreeBlock > 1)
                                                std::swap(vector->_blocks[i], vector->_blocks[--state.firstFreeBlock]);
                                            else
                                                --state.firstFreeBlock;
                                        }
                                    }
                                    swapped = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (swapped)
                        result = VK_INCOMPLETE;
                    break;
                }
            }

            // Move to the beginning
            for (const FragmentedBlock& block : immovableBlocks)
            {
                VmaBlockVector* vector = _p_block_vectors[block.data];
                VmaMutexLockWrite lock(vector->get_mutex(), vector->get_allocator()->_use_mutex);

                for (size_t i = _immovable_block_count; i < vector->get_block_count(); ++i)
                {
                    if (vector->get_block(i) == block.block)
                    {
                        std::swap(vector->_blocks[i], vector->_blocks[_immovable_block_count++]);
                        break;
                    }
                }
            }
        } while (false);
    }

    // Bulk-map destination blocks
    for (const FragmentedBlock& block : mappedBlocks)
    {
        VkResult res = block.block->Map(allocator, block.data, VMA_NULL);
        VMA_ASSERT(res == VK_SUCCESS);
    }
    return result;
}

bool VmaDefragmentationContext_T::compute_defragmentation(VmaBlockVector& vector, size_t index)
{
    switch (_algorithm)
    {
    case VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FAST_BIT:
        return compute_defragmentation_fast(vector);
    case VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT:
        return compute_defragmentation_balanced(vector, index, true);
    case VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FULL_BIT:
        return compute_defragmentation_full(vector);
    case VMA_DEFRAGMENTATION_FLAG_ALGORITHM_EXTENSIVE_BIT:
        return compute_defragmentation_extensive(vector, index);
    default:
        VMA_ASSERT(0);
        return compute_defragmentation_balanced(vector, index, true);
    }
}

VmaDefragmentationContext_T::MoveAllocationData VmaDefragmentationContext_T::get_move_data(
    VmaAllocHandle handle, VmaBlockMetadata* metadata)
{
    MoveAllocationData moveData;
    moveData.move.srcAllocation = (VmaAllocation)metadata->get_allocation_user_data(handle);
    moveData.size = moveData.move.srcAllocation->get_size();
    moveData.alignment = moveData.move.srcAllocation->get_alignment();
    moveData.type = moveData.move.srcAllocation->get_suballocation_type();
    moveData.flags = 0;

    if (moveData.move.srcAllocation->is_persistent_map())
        moveData.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
    if (moveData.move.srcAllocation->is_mapping_allowed())
        moveData.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

    return moveData;
}

VmaDefragmentationContext_T::CounterStatus VmaDefragmentationContext_T::check_counters(VkDeviceSize bytes)
{
    // Check custom criteria if exists
    if (_break_callback && _break_callback(_break_callback_user_data))
        return CounterStatus::End;

    // Ignore allocation if will exceed max size for copy
    if (_pass_stats.bytesMoved + bytes > _max_pass_bytes)
    {
        if (++_ignored_allocs < MAX_ALLOCS_TO_IGNORE)
            return CounterStatus::Ignore;
        return CounterStatus::End;
    }

    _ignored_allocs = 0;
    return CounterStatus::Pass;
}

bool VmaDefragmentationContext_T::increment_counters(VkDeviceSize bytes)
{
    _pass_stats.bytesMoved += bytes;
    // Early return when max found
    if (++_pass_stats.allocationsMoved >= _max_pass_allocations || _pass_stats.bytesMoved >= _max_pass_bytes)
    {
        VMA_ASSERT((_pass_stats.allocationsMoved == _max_pass_allocations ||
            _pass_stats.bytesMoved == _max_pass_bytes) && "Exceeded maximal pass threshold!");
        return true;
    }
    return false;
}

bool VmaDefragmentationContext_T::realloc_within_block(VmaBlockVector& vector, VmaDeviceMemoryBlock* block)
{
    VmaBlockMetadata* metadata = block->_p_metadata;

    for (VmaAllocHandle handle = metadata->get_allocation_list_begin();
        handle != VK_NULL_HANDLE;
        handle = metadata->get_next_allocation(handle))
    {
        MoveAllocationData moveData = get_move_data(handle, metadata);
        // Ignore newly created allocations by defragmentation algorithm
        if (moveData.move.srcAllocation->get_user_data() == this)
            continue;
        switch (check_counters(moveData.move.srcAllocation->get_size()))
        {
        case CounterStatus::Ignore:
            continue;
        case CounterStatus::End:
            return true;
        case CounterStatus::Pass:
            break;
        default:
            VMA_ASSERT(0);
        }

        VkDeviceSize offset = moveData.move.srcAllocation->get_offset();
        if (offset != 0 && metadata->get_sum_free_size() >= moveData.size)
        {
            VmaAllocationRequest request = {};
            if (metadata->create_allocation_request(
                moveData.size,
                moveData.alignment,
                false,
                moveData.type,
                VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT,
                &request))
            {
                if (metadata->get_allocation_offset(request.allocHandle) < offset)
                {
                    if (vector.commit_allocation_request(
                        request,
                        block,
                        moveData.alignment,
                        moveData.flags,
                        this,
                        moveData.type,
                        &moveData.move.dstTmpAllocation) == VK_SUCCESS)
                    {
                        _moves.push_back(moveData.move);
                        if (increment_counters(moveData.size))
                            return true;
                    }
                }
            }
        }
    }
    return false;
}

bool VmaDefragmentationContext_T::alloc_in_other_block(size_t start, size_t end, MoveAllocationData& data, VmaBlockVector& vector)
{
    for (; start < end; ++start)
    {
        VmaDeviceMemoryBlock* dstBlock = vector.get_block(start);
        if (dstBlock->_p_metadata->get_sum_free_size() >= data.size)
        {
            if (vector.allocate_from_block(dstBlock,
                data.size,
                data.alignment,
                data.flags,
                this,
                data.type,
                0,
                &data.move.dstTmpAllocation) == VK_SUCCESS)
            {
                _moves.push_back(data.move);
                if (increment_counters(data.size))
                    return true;
                break;
            }
        }
    }
    return false;
}

bool VmaDefragmentationContext_T::compute_defragmentation_fast(VmaBlockVector& vector)
{
    // Move only between blocks

    // Go through allocations in last blocks and try to fit them inside first ones
    for (size_t i = vector.get_block_count() - 1; i > _immovable_block_count; --i)
    {
        VmaBlockMetadata* metadata = vector.get_block(i)->_p_metadata;

        for (VmaAllocHandle handle = metadata->get_allocation_list_begin();
            handle != VK_NULL_HANDLE;
            handle = metadata->get_next_allocation(handle))
        {
            MoveAllocationData moveData = get_move_data(handle, metadata);
            // Ignore newly created allocations by defragmentation algorithm
            if (moveData.move.srcAllocation->get_user_data() == this)
                continue;
            switch (check_counters(moveData.move.srcAllocation->get_size()))
            {
            case CounterStatus::Ignore:
                continue;
            case CounterStatus::End:
                return true;
            case CounterStatus::Pass:
                break;
            default:
                VMA_ASSERT(0);
            }

            // Check all previous blocks for free space
            if (alloc_in_other_block(0, i, moveData, vector))
                return true;
        }
    }
    return false;
}

bool VmaDefragmentationContext_T::compute_defragmentation_balanced(VmaBlockVector& vector, size_t index, bool update)
{
    // Go over every allocation and try to fit it in previous blocks at lowest offsets,
    // if not possible: realloc within single block to minimize offset (exclude offset == 0),
    // but only if there are noticeable gaps between them (some heuristic, ex. average size of allocation in block)
    VMA_ASSERT(_algorithm_state != VMA_NULL);

    StateBalanced& vectorState = reinterpret_cast<StateBalanced*>(_algorithm_state)[index];
    if (update && vectorState.avgAllocSize == UINT64_MAX)
        update_vector_statistics(vector, vectorState);

    const size_t startMoveCount = _moves.size();
    VkDeviceSize minimalFreeRegion = vectorState.avgFreeSize / 2;
    for (size_t i = vector.get_block_count() - 1; i > _immovable_block_count; --i)
    {
        VmaDeviceMemoryBlock* block = vector.get_block(i);
        VmaBlockMetadata* metadata = block->_p_metadata;
        VkDeviceSize prevFreeRegionSize = 0;

        for (VmaAllocHandle handle = metadata->get_allocation_list_begin();
            handle != VK_NULL_HANDLE;
            handle = metadata->get_next_allocation(handle))
        {
            MoveAllocationData moveData = get_move_data(handle, metadata);
            // Ignore newly created allocations by defragmentation algorithm
            if (moveData.move.srcAllocation->get_user_data() == this)
                continue;
            switch (check_counters(moveData.move.srcAllocation->get_size()))
            {
            case CounterStatus::Ignore:
                continue;
            case CounterStatus::End:
                return true;
            case CounterStatus::Pass:
                break;
            default:
                VMA_ASSERT(0);
            }

            // Check all previous blocks for free space
            const size_t prevMoveCount = _moves.size();
            if (alloc_in_other_block(0, i, moveData, vector))
                return true;

            VkDeviceSize nextFreeRegionSize = metadata->get_next_free_region_size(handle);
            // If no room found then realloc within block for lower offset
            VkDeviceSize offset = moveData.move.srcAllocation->get_offset();
            if (prevMoveCount == _moves.size() && offset != 0 && metadata->get_sum_free_size() >= moveData.size)
            {
                // Check if realloc will make sense
                if (prevFreeRegionSize >= minimalFreeRegion ||
                    nextFreeRegionSize >= minimalFreeRegion ||
                    moveData.size <= vectorState.avgFreeSize ||
                    moveData.size <= vectorState.avgAllocSize)
                {
                    VmaAllocationRequest request = {};
                    if (metadata->create_allocation_request(
                        moveData.size,
                        moveData.alignment,
                        false,
                        moveData.type,
                        VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT,
                        &request))
                    {
                        if (metadata->get_allocation_offset(request.allocHandle) < offset)
                        {
                            if (vector.commit_allocation_request(
                                request,
                                block,
                                moveData.alignment,
                                moveData.flags,
                                this,
                                moveData.type,
                                &moveData.move.dstTmpAllocation) == VK_SUCCESS)
                            {
                                _moves.push_back(moveData.move);
                                if (increment_counters(moveData.size))
                                    return true;
                            }
                        }
                    }
                }
            }
            prevFreeRegionSize = nextFreeRegionSize;
        }
    }

    // No moves performed, update statistics to current vector state
    if (startMoveCount == _moves.size() && !update)
    {
        vectorState.avgAllocSize = UINT64_MAX;
        return compute_defragmentation_balanced(vector, index, false);
    }
    return false;
}

bool VmaDefragmentationContext_T::compute_defragmentation_full(VmaBlockVector& vector)
{
    // Go over every allocation and try to fit it in previous blocks at lowest offsets,
    // if not possible: realloc within single block to minimize offset (exclude offset == 0)

    for (size_t i = vector.get_block_count() - 1; i > _immovable_block_count; --i)
    {
        VmaDeviceMemoryBlock* block = vector.get_block(i);
        VmaBlockMetadata* metadata = block->_p_metadata;

        for (VmaAllocHandle handle = metadata->get_allocation_list_begin();
            handle != VK_NULL_HANDLE;
            handle = metadata->get_next_allocation(handle))
        {
            MoveAllocationData moveData = get_move_data(handle, metadata);
            // Ignore newly created allocations by defragmentation algorithm
            if (moveData.move.srcAllocation->get_user_data() == this)
                continue;
            switch (check_counters(moveData.move.srcAllocation->get_size()))
            {
            case CounterStatus::Ignore:
                continue;
            case CounterStatus::End:
                return true;
            case CounterStatus::Pass:
                break;
            default:
                VMA_ASSERT(0);
            }

            // Check all previous blocks for free space
            const size_t prevMoveCount = _moves.size();
            if (alloc_in_other_block(0, i, moveData, vector))
                return true;

            // If no room found then realloc within block for lower offset
            VkDeviceSize offset = moveData.move.srcAllocation->get_offset();
            if (prevMoveCount == _moves.size() && offset != 0 && metadata->get_sum_free_size() >= moveData.size)
            {
                VmaAllocationRequest request = {};
                if (metadata->create_allocation_request(
                    moveData.size,
                    moveData.alignment,
                    false,
                    moveData.type,
                    VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT,
                    &request))
                {
                    if (metadata->get_allocation_offset(request.allocHandle) < offset)
                    {
                        if (vector.commit_allocation_request(
                            request,
                            block,
                            moveData.alignment,
                            moveData.flags,
                            this,
                            moveData.type,
                            &moveData.move.dstTmpAllocation) == VK_SUCCESS)
                        {
                            _moves.push_back(moveData.move);
                            if (increment_counters(moveData.size))
                                return true;
                        }
                    }
                }
            }
        }
    }
    return false;
}

bool VmaDefragmentationContext_T::compute_defragmentation_extensive(VmaBlockVector& vector, size_t index)
{
    // First free single block, then populate it to the brim, then free another block, and so on

    // Fallback to previous algorithm since without granularity conflicts it can achieve max packing
    if (vector._buffer_image_granularity == 1)
        return compute_defragmentation_full(vector);

    VMA_ASSERT(_algorithm_state != VMA_NULL);

    StateExtensive& vectorState = reinterpret_cast<StateExtensive*>(_algorithm_state)[index];

    bool texturePresent = false;
    bool bufferPresent = false;
    bool otherPresent = false;
    switch (vectorState.operation)
    {
    case StateExtensive::Operation::Done: // Vector defragmented
        return false;
    case StateExtensive::Operation::find_free_blockBuffer:
    case StateExtensive::Operation::find_free_blockTexture:
    case StateExtensive::Operation::find_free_blockAll:
    {
        // No more blocks to free, just perform fast realloc and move to cleanup
        if (vectorState.firstFreeBlock == 0)
        {
            vectorState.operation = StateExtensive::Operation::Cleanup;
            return compute_defragmentation_fast(vector);
        }

        // No free blocks, have to clear last one
        size_t last = (vectorState.firstFreeBlock == SIZE_MAX ? vector.get_block_count() : vectorState.firstFreeBlock) - 1;
        VmaBlockMetadata* freeMetadata = vector.get_block(last)->_p_metadata;

        const size_t prevMoveCount = _moves.size();
        for (VmaAllocHandle handle = freeMetadata->get_allocation_list_begin();
            handle != VK_NULL_HANDLE;
            handle = freeMetadata->get_next_allocation(handle))
        {
            MoveAllocationData moveData = get_move_data(handle, freeMetadata);
            switch (check_counters(moveData.move.srcAllocation->get_size()))
            {
            case CounterStatus::Ignore:
                continue;
            case CounterStatus::End:
                return true;
            case CounterStatus::Pass:
                break;
            default:
                VMA_ASSERT(0);
            }

            // Check all previous blocks for free space
            if (alloc_in_other_block(0, last, moveData, vector))
            {
                // Full clear performed already
                if (prevMoveCount != _moves.size() && freeMetadata->get_next_allocation(handle) == VK_NULL_HANDLE)
                    vectorState.firstFreeBlock = last;
                return true;
            }
        }

        if (prevMoveCount == _moves.size())
        {
            // Cannot perform full clear, have to move data in other blocks around
            if (last != 0)
            {
                for (size_t i = last - 1; i; --i)
                {
                    if (realloc_within_block(vector, vector.get_block(i)))
                        return true;
                }
            }

            if (prevMoveCount == _moves.size())
            {
                // No possible reallocs within blocks, try to move them around fast
                return compute_defragmentation_fast(vector);
            }
        }
        else
        {
            switch (vectorState.operation)
            {
            case StateExtensive::Operation::find_free_blockBuffer:
                vectorState.operation = StateExtensive::Operation::MoveBuffers;
                break;
            case StateExtensive::Operation::find_free_blockTexture:
                vectorState.operation = StateExtensive::Operation::MoveTextures;
                break;
            case StateExtensive::Operation::find_free_blockAll:
                vectorState.operation = StateExtensive::Operation::MoveAll;
                break;
            default:
                VMA_ASSERT(0);
                vectorState.operation = StateExtensive::Operation::MoveTextures;
            }
            vectorState.firstFreeBlock = last;
            // Nothing done, block found without reallocations, can perform another reallocs in same pass
            return compute_defragmentation_extensive(vector, index);
        }
        break;
    }
    case StateExtensive::Operation::MoveTextures:
    {
        if (move_data_to_free_blocks(VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL, vector,
            vectorState.firstFreeBlock, texturePresent, bufferPresent, otherPresent))
        {
            if (texturePresent)
            {
                vectorState.operation = StateExtensive::Operation::find_free_blockTexture;
                return compute_defragmentation_extensive(vector, index);
            }

            if (!bufferPresent && !otherPresent)
            {
                vectorState.operation = StateExtensive::Operation::Cleanup;
                break;
            }

            // No more textures to move, check buffers
            vectorState.operation = StateExtensive::Operation::MoveBuffers;
            bufferPresent = false;
            otherPresent = false;
        }
        else
            break;
        VMA_FALLTHROUGH; // Fallthrough
    }
    case StateExtensive::Operation::MoveBuffers:
    {
        if (move_data_to_free_blocks(VMA_SUBALLOCATION_TYPE_BUFFER, vector,
            vectorState.firstFreeBlock, texturePresent, bufferPresent, otherPresent))
        {
            if (bufferPresent)
            {
                vectorState.operation = StateExtensive::Operation::find_free_blockBuffer;
                return compute_defragmentation_extensive(vector, index);
            }

            if (!otherPresent)
            {
                vectorState.operation = StateExtensive::Operation::Cleanup;
                break;
            }

            // No more buffers to move, check all others
            vectorState.operation = StateExtensive::Operation::MoveAll;
            otherPresent = false;
        }
        else
            break;
        VMA_FALLTHROUGH; // Fallthrough
    }
    case StateExtensive::Operation::MoveAll:
    {
        if (move_data_to_free_blocks(VMA_SUBALLOCATION_TYPE_FREE, vector,
            vectorState.firstFreeBlock, texturePresent, bufferPresent, otherPresent))
        {
            if (otherPresent)
            {
                vectorState.operation = StateExtensive::Operation::find_free_blockBuffer;
                return compute_defragmentation_extensive(vector, index);
            }
            // Everything moved
            vectorState.operation = StateExtensive::Operation::Cleanup;
        }
        break;
    }
    case StateExtensive::Operation::Cleanup:
        // Cleanup is handled below so that other operations may reuse the cleanup code. This case is here to prevent the unhandled enum value warning (C4062).
        break;
    }

    if (vectorState.operation == StateExtensive::Operation::Cleanup)
    {
        // All other work done, pack data in blocks even tighter if possible
        const size_t prevMoveCount = _moves.size();
        for (size_t i = 0; i < vector.get_block_count(); ++i)
        {
            if (realloc_within_block(vector, vector.get_block(i)))
                return true;
        }

        if (prevMoveCount == _moves.size())
            vectorState.operation = StateExtensive::Operation::Done;
    }
    return false;
}

void VmaDefragmentationContext_T::update_vector_statistics(VmaBlockVector& vector, StateBalanced& state)
{
    size_t allocCount = 0;
    size_t freeCount = 0;
    state.avgFreeSize = 0;
    state.avgAllocSize = 0;

    for (size_t i = 0; i < vector.get_block_count(); ++i)
    {
        VmaBlockMetadata* metadata = vector.get_block(i)->_p_metadata;

        allocCount += metadata->get_allocation_count();
        freeCount += metadata->get_free_regions_count();
        state.avgFreeSize += metadata->get_sum_free_size();
        state.avgAllocSize += metadata->get_size();
    }

    state.avgAllocSize = (state.avgAllocSize - state.avgFreeSize) / allocCount;
    state.avgFreeSize /= freeCount;
}

bool VmaDefragmentationContext_T::move_data_to_free_blocks(VmaSuballocationType currentType,
    VmaBlockVector& vector, size_t firstFreeBlock,
    bool& texturePresent, bool& bufferPresent, bool& otherPresent)
{
    const size_t prevMoveCount = _moves.size();
    for (size_t i = firstFreeBlock ; i;)
    {
        VmaDeviceMemoryBlock* block = vector.get_block(--i);
        VmaBlockMetadata* metadata = block->_p_metadata;

        for (VmaAllocHandle handle = metadata->get_allocation_list_begin();
            handle != VK_NULL_HANDLE;
            handle = metadata->get_next_allocation(handle))
        {
            MoveAllocationData moveData = get_move_data(handle, metadata);
            // Ignore newly created allocations by defragmentation algorithm
            if (moveData.move.srcAllocation->get_user_data() == this)
                continue;
            switch (check_counters(moveData.move.srcAllocation->get_size()))
            {
            case CounterStatus::Ignore:
                continue;
            case CounterStatus::End:
                return true;
            case CounterStatus::Pass:
                break;
            default:
                VMA_ASSERT(0);
            }

            // Move only single type of resources at once
            if (!VmaIsBufferImageGranularityConflict(moveData.type, currentType))
            {
                // Try to fit allocation into free blocks
                if (alloc_in_other_block(firstFreeBlock, vector.get_block_count(), moveData, vector))
                    return false;
            }

            if (!VmaIsBufferImageGranularityConflict(moveData.type, VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL))
                texturePresent = true;
            else if (!VmaIsBufferImageGranularityConflict(moveData.type, VMA_SUBALLOCATION_TYPE_BUFFER))
                bufferPresent = true;
            else
                otherPresent = true;
        }
    }
    return prevMoveCount == _moves.size();
}
#endif // _VMA_DEFRAGMENTATION_CONTEXT_FUNCTIONS

#ifndef _VMA_POOL_T_FUNCTIONS
VmaPool_T::VmaPool_T(
    VmaAllocator hAllocator,
    const VmaPoolCreateInfo& createInfo,
    VkDeviceSize preferredBlockSize)
    : _block_vector(
        hAllocator,
        this, // hParentPool
        createInfo.memoryTypeIndex,
        createInfo.blockSize != 0 ? createInfo.blockSize : preferredBlockSize,
        createInfo.minBlockCount,
        createInfo.maxBlockCount,
        (createInfo.flags& VMA_POOL_CREATE_IGNORE_BUFFER_IMAGE_GRANULARITY_BIT) != 0 ? 1 : hAllocator->get_buffer_image_granularity(),
        createInfo.blockSize != 0, // explicitBlockSize
        createInfo.flags & VMA_POOL_CREATE_ALGORITHM_MASK, // algorithm
        createInfo.priority,
        VMA_MAX(hAllocator->get_memory_type_min_alignment(createInfo.memoryTypeIndex), createInfo.minAllocationAlignment),
        createInfo.pMemoryAllocateNext),
    _id(0),
    _name(VMA_NULL) {}

VmaPool_T::~VmaPool_T()
{
    VMA_ASSERT(_prev_pool == VMA_NULL && _next_pool == VMA_NULL);

    const VkAllocationCallbacks* allocs = _block_vector.get_allocator()->get_allocation_callbacks();
    VmaFreeString(allocs, _name);
}

void VmaPool_T::set_name(const char* pName)
{
    const VkAllocationCallbacks* allocs = _block_vector.get_allocator()->get_allocation_callbacks();
    VmaFreeString(allocs, _name);

    if (pName != VMA_NULL)
    {
        _name = VmaCreateStringCopy(allocs, pName);
    }
    else
    {
        _name = VMA_NULL;
    }
}
#endif // _VMA_POOL_T_FUNCTIONS

#ifndef _VMA_ALLOCATOR_T_FUNCTIONS
VmaAllocator_T::VmaAllocator_T(const VmaAllocatorCreateInfo* pCreateInfo) :
    _use_mutex((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT) == 0),
    _vulkan_api_version(pCreateInfo->vulkanApiVersion != 0 ? pCreateInfo->vulkanApiVersion : VK_API_VERSION_1_0),
    _use_khr_dedicated_allocation((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT) != 0),
    _use_khr_bind_memory2((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT) != 0),
    _use_ext_memory_budget((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT) != 0),
    _use_amd_device_coherent_memory((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT) != 0),
    _use_khr_buffer_device_address((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT) != 0),
    _use_ext_memory_priority((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT) != 0),
    _use_khr_maintenance4((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT) != 0),
    _use_khr_maintenance5((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT) != 0),
    _use_khr_external_memory_win32((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_EXTERNAL_MEMORY_WIN32_BIT) != 0),
    _h_device(pCreateInfo->device),
    _h_instance(pCreateInfo->instance),
    _allocation_callbacks_specified(pCreateInfo->pAllocationCallbacks != VMA_NULL),
    _allocation_callbacks(pCreateInfo->pAllocationCallbacks ?
        *pCreateInfo->pAllocationCallbacks : VmaEmptyAllocationCallbacks),
    _allocation_object_allocator(&_allocation_callbacks),
    _heap_size_limit_mask(0),
    _device_memory_count(0),
    _preferred_large_heap_block_size(0),
    _physical_device(pCreateInfo->physicalDevice),
    _gpu_defragmentation_memory_type_bits(UINT32_MAX),
    _next_pool_id(0),
    _global_memory_type_bits(UINT32_MAX)
{
    if(_vulkan_api_version >= VK_MAKE_VERSION(1, 1, 0))
    {
        _use_khr_dedicated_allocation = false;
        _use_khr_bind_memory2 = false;
    }

    if(VMA_DEBUG_DETECT_CORRUPTION)
    {
        // Needs to be multiply of uint32_t size because we are going to write VMA_CORRUPTION_DETECTION_MAGIC_VALUE to it.
        VMA_ASSERT(VMA_DEBUG_MARGIN % sizeof(uint32_t) == 0);
    }

    VMA_ASSERT(pCreateInfo->physicalDevice && pCreateInfo->device && pCreateInfo->instance);

    if(_vulkan_api_version < VK_MAKE_VERSION(1, 1, 0))
    {
#if !(VMA_DEDICATED_ALLOCATION)
        if((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT) != 0)
        {
            VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT set but required extensions are disabled by preprocessor macros.");
        }
#endif
#if !(VMA_BIND_MEMORY2)
        if((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT) != 0)
        {
            VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT set but required extension is disabled by preprocessor macros.");
        }
#endif
    }
#if !(VMA_MEMORY_BUDGET)
    if((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT) != 0)
    {
        VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT set but required extension is disabled by preprocessor macros.");
    }
#endif
#if !(VMA_BUFFER_DEVICE_ADDRESS)
    if(_use_khr_buffer_device_address)
    {
        VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT is set but required extension or Vulkan 1.2 is not available in your Vulkan header or its support in VMA has been disabled by a preprocessor macro.");
    }
#endif
#if VMA_VULKAN_VERSION < 1004000
    VMA_ASSERT(_vulkan_api_version < VK_MAKE_VERSION(1, 4, 0) && "vulkanApiVersion >= VK_API_VERSION_1_4 but required Vulkan version is disabled by preprocessor macros.");
#endif
#if VMA_VULKAN_VERSION < 1003000
    VMA_ASSERT(_vulkan_api_version < VK_MAKE_VERSION(1, 3, 0) && "vulkanApiVersion >= VK_API_VERSION_1_3 but required Vulkan version is disabled by preprocessor macros.");
#endif
#if VMA_VULKAN_VERSION < 1002000
    VMA_ASSERT(_vulkan_api_version < VK_MAKE_VERSION(1, 2, 0) && "vulkanApiVersion >= VK_API_VERSION_1_2 but required Vulkan version is disabled by preprocessor macros.");
#endif
#if VMA_VULKAN_VERSION < 1001000
    VMA_ASSERT(_vulkan_api_version < VK_MAKE_VERSION(1, 1, 0) && "vulkanApiVersion >= VK_API_VERSION_1_1 but required Vulkan version is disabled by preprocessor macros.");
#endif
#if !(VMA_MEMORY_PRIORITY)
    if(_use_ext_memory_priority)
    {
        VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT is set but required extension is not available in your Vulkan header or its support in VMA has been disabled by a preprocessor macro.");
    }
#endif
#if !(VMA_KHR_MAINTENANCE4)
    if(_use_khr_maintenance4)
    {
        VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT is set but required extension is not available in your Vulkan header or its support in VMA has been disabled by a preprocessor macro.");
    }
#endif
#if !(VMA_KHR_MAINTENANCE5)
    if(_use_khr_maintenance5)
    {
        VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT is set but required extension is not available in your Vulkan header or its support in VMA has been disabled by a preprocessor macro.");
    }
#endif
#if !(VMA_KHR_MAINTENANCE5)
    if(_use_khr_maintenance5)
    {
        VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT is set but required extension is not available in your Vulkan header or its support in VMA has been disabled by a preprocessor macro.");
    }
#endif

#if !(VMA_EXTERNAL_MEMORY_WIN32)
    if(_use_khr_external_memory_win32)
    {
        VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_KHR_EXTERNAL_MEMORY_WIN32_BIT is set but required extension is not available in your Vulkan header or its support in VMA has been disabled by a preprocessor macro.");
    }
#endif

    memset(&_device_memory_callbacks, 0 ,sizeof(_device_memory_callbacks));
    memset(&_physical_device_properties, 0, sizeof(_physical_device_properties));
    memset(&_mem_props, 0, sizeof(_mem_props));

    memset(&_p_block_vectors, 0, sizeof(_p_block_vectors));
    memset(&_vulkan_functions, 0, sizeof(_vulkan_functions));

#if VMA_EXTERNAL_MEMORY
    memset(&_type_external_memory_handle_types, 0, sizeof(_type_external_memory_handle_types));
#endif // #if VMA_EXTERNAL_MEMORY

    if(pCreateInfo->pDeviceMemoryCallbacks != VMA_NULL)
    {
        _device_memory_callbacks.pUserData = pCreateInfo->pDeviceMemoryCallbacks->pUserData;
        _device_memory_callbacks.pfnAllocate = pCreateInfo->pDeviceMemoryCallbacks->pfnAllocate;
        _device_memory_callbacks.pfnFree = pCreateInfo->pDeviceMemoryCallbacks->pfnFree;
    }

    import_vulkan_functions(pCreateInfo->pVulkanFunctions);

    (*_vulkan_functions.vkGetPhysicalDeviceProperties)(_physical_device, &_physical_device_properties);
    (*_vulkan_functions.vkGetPhysicalDeviceMemoryProperties)(_physical_device, &_mem_props);

    VMA_ASSERT(VmaIsPow2(VMA_MIN_ALIGNMENT));
    VMA_ASSERT(VmaIsPow2(VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY));
    VMA_ASSERT(VmaIsPow2(_physical_device_properties.limits.bufferImageGranularity));
    VMA_ASSERT(VmaIsPow2(_physical_device_properties.limits.nonCoherentAtomSize));

    _preferred_large_heap_block_size = (pCreateInfo->preferredLargeHeapBlockSize != 0) ?
        pCreateInfo->preferredLargeHeapBlockSize : static_cast<VkDeviceSize>(VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE);

    _global_memory_type_bits = calculate_global_memory_type_bits();

#if VMA_EXTERNAL_MEMORY
    if(pCreateInfo->pTypeExternalMemoryHandleTypes != VMA_NULL)
    {
        memcpy(_type_external_memory_handle_types, pCreateInfo->pTypeExternalMemoryHandleTypes,
            sizeof(VkExternalMemoryHandleTypeFlagsKHR) * get_memory_type_count());
    }
#endif // #if VMA_EXTERNAL_MEMORY

    if(pCreateInfo->pHeapSizeLimit != VMA_NULL)
    {
        for(uint32_t heapIndex = 0; heapIndex < get_memory_heap_count(); ++heapIndex)
        {
            const VkDeviceSize limit = pCreateInfo->pHeapSizeLimit[heapIndex];
            if(limit != VK_WHOLE_SIZE)
            {
                _heap_size_limit_mask |= 1U << heapIndex;
                if(limit < _mem_props.memoryHeaps[heapIndex].size)
                {
                    _mem_props.memoryHeaps[heapIndex].size = limit;
                }
            }
        }
    }

    for(uint32_t memTypeIndex = 0; memTypeIndex < get_memory_type_count(); ++memTypeIndex)
    {
        // Create only supported types
        if((_global_memory_type_bits & (1U << memTypeIndex)) != 0)
        {
            const VkDeviceSize preferredBlockSize = calc_preferred_block_size(memTypeIndex);
            _p_block_vectors[memTypeIndex] = vma_new(this, VmaBlockVector)(
                this,
                VK_NULL_HANDLE, // hParentPool
                memTypeIndex,
                preferredBlockSize,
                0,
                SIZE_MAX,
                get_buffer_image_granularity(),
                false, // explicitBlockSize
                0, // algorithm
                0.5F, // priority (0.5 is the default per Vulkan spec)
                get_memory_type_min_alignment(memTypeIndex), // minAllocationAlignment
                VMA_NULL); // // pMemoryAllocateNext
            // No need to call _p_block_vectors[memTypeIndex][blockVectorTypeIndex]->create_min_blocks here,
            // because minBlockCount is 0.
        }
    }
}

VkResult VmaAllocator_T::init(const VmaAllocatorCreateInfo* pCreateInfo)
{
    VkResult res = VK_SUCCESS;

#if VMA_MEMORY_BUDGET
    if(_use_ext_memory_budget)
    {
        update_vulkan_budget();
    }
#endif // #if VMA_MEMORY_BUDGET

    return res;
}

VmaAllocator_T::~VmaAllocator_T()
{
    VMA_ASSERT(_pools.is_empty());

    for(size_t memTypeIndex = get_memory_type_count(); memTypeIndex--; )
    {
        vma_delete(this, _p_block_vectors[memTypeIndex]);
    }
}

void VmaAllocator_T::import_vulkan_functions(const VmaVulkanFunctions* pVulkanFunctions)
{
#if VMA_STATIC_VULKAN_FUNCTIONS == 1
    import_vulkan_functions_static();
#endif

    if(pVulkanFunctions != VMA_NULL)
    {
        import_vulkan_functions_custom(pVulkanFunctions);
    }

#if VMA_DYNAMIC_VULKAN_FUNCTIONS == 1
    import_vulkan_functions_dynamic();
#endif

    validate_vulkan_functions();
}

#if VMA_STATIC_VULKAN_FUNCTIONS == 1

void VmaAllocator_T::import_vulkan_functions_static()
{
    // Vulkan 1.0
    _vulkan_functions.vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)vkGetInstanceProcAddr;
    _vulkan_functions.vkGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)vkGetDeviceProcAddr;
    _vulkan_functions.vkGetPhysicalDeviceProperties = (PFN_vkGetPhysicalDeviceProperties)vkGetPhysicalDeviceProperties;
    _vulkan_functions.vkGetPhysicalDeviceMemoryProperties = (PFN_vkGetPhysicalDeviceMemoryProperties)vkGetPhysicalDeviceMemoryProperties;
    _vulkan_functions.vkAllocateMemory = (PFN_vkAllocateMemory)vkAllocateMemory;
    _vulkan_functions.vkFreeMemory = (PFN_vkFreeMemory)vkFreeMemory;
    _vulkan_functions.vkMapMemory = (PFN_vkMapMemory)vkMapMemory;
    _vulkan_functions.vkUnmapMemory = (PFN_vkUnmapMemory)vkUnmapMemory;
    _vulkan_functions.vkFlushMappedMemoryRanges = (PFN_vkFlushMappedMemoryRanges)vkFlushMappedMemoryRanges;
    _vulkan_functions.vkInvalidateMappedMemoryRanges = (PFN_vkInvalidateMappedMemoryRanges)vkInvalidateMappedMemoryRanges;
    _vulkan_functions.vkBindBufferMemory = (PFN_vkBindBufferMemory)vkBindBufferMemory;
    _vulkan_functions.vkBindImageMemory = (PFN_vkBindImageMemory)vkBindImageMemory;
    _vulkan_functions.vkGetBufferMemoryRequirements = (PFN_vkGetBufferMemoryRequirements)vkGetBufferMemoryRequirements;
    _vulkan_functions.vkGetImageMemoryRequirements = (PFN_vkGetImageMemoryRequirements)vkGetImageMemoryRequirements;
    _vulkan_functions.vkCreateBuffer = (PFN_vkCreateBuffer)vkCreateBuffer;
    _vulkan_functions.vkDestroyBuffer = (PFN_vkDestroyBuffer)vkDestroyBuffer;
    _vulkan_functions.vkCreateImage = (PFN_vkCreateImage)vkCreateImage;
    _vulkan_functions.vkDestroyImage = (PFN_vkDestroyImage)vkDestroyImage;
    _vulkan_functions.vkCmdCopyBuffer = (PFN_vkCmdCopyBuffer)vkCmdCopyBuffer;

    // Vulkan 1.1
#if VMA_VULKAN_VERSION >= 1001000
    if(_vulkan_api_version >= VK_MAKE_VERSION(1, 1, 0))
    {
        _vulkan_functions.vkGetBufferMemoryRequirements2KHR = (PFN_vkGetBufferMemoryRequirements2)vkGetBufferMemoryRequirements2;
        _vulkan_functions.vkGetImageMemoryRequirements2KHR = (PFN_vkGetImageMemoryRequirements2)vkGetImageMemoryRequirements2;
        _vulkan_functions.vkBindBufferMemory2KHR = (PFN_vkBindBufferMemory2)vkBindBufferMemory2;
        _vulkan_functions.vkBindImageMemory2KHR = (PFN_vkBindImageMemory2)vkBindImageMemory2;
    }
#endif

#if VMA_VULKAN_VERSION >= 1001000
    if(_vulkan_api_version >= VK_MAKE_VERSION(1, 1, 0))
    {
        _vulkan_functions.vkGetPhysicalDeviceMemoryProperties2KHR = (PFN_vkGetPhysicalDeviceMemoryProperties2)vkGetPhysicalDeviceMemoryProperties2;
    }
#endif

#if VMA_VULKAN_VERSION >= 1003000
    if(_vulkan_api_version >= VK_MAKE_VERSION(1, 3, 0))
    {
        _vulkan_functions.vkGetDeviceBufferMemoryRequirements = (PFN_vkGetDeviceBufferMemoryRequirements)vkGetDeviceBufferMemoryRequirements;
        _vulkan_functions.vkGetDeviceImageMemoryRequirements = (PFN_vkGetDeviceImageMemoryRequirements)vkGetDeviceImageMemoryRequirements;
    }
#endif
}

#endif // VMA_STATIC_VULKAN_FUNCTIONS == 1

void VmaAllocator_T::import_vulkan_functions_custom(const VmaVulkanFunctions* pVulkanFunctions)
{
    VMA_ASSERT(pVulkanFunctions != VMA_NULL);

#define VMA_COPY_IF_NOT_NULL(funcName) \
    if(pVulkanFunctions->funcName != VMA_NULL) _vulkan_functions.funcName = pVulkanFunctions->funcName;

    VMA_COPY_IF_NOT_NULL(vkGetInstanceProcAddr);
    VMA_COPY_IF_NOT_NULL(vkGetDeviceProcAddr);
    VMA_COPY_IF_NOT_NULL(vkGetPhysicalDeviceProperties);
    VMA_COPY_IF_NOT_NULL(vkGetPhysicalDeviceMemoryProperties);
    VMA_COPY_IF_NOT_NULL(vkAllocateMemory);
    VMA_COPY_IF_NOT_NULL(vkFreeMemory);
    VMA_COPY_IF_NOT_NULL(vkMapMemory);
    VMA_COPY_IF_NOT_NULL(vkUnmapMemory);
    VMA_COPY_IF_NOT_NULL(vkFlushMappedMemoryRanges);
    VMA_COPY_IF_NOT_NULL(vkInvalidateMappedMemoryRanges);
    VMA_COPY_IF_NOT_NULL(vkBindBufferMemory);
    VMA_COPY_IF_NOT_NULL(vkBindImageMemory);
    VMA_COPY_IF_NOT_NULL(vkGetBufferMemoryRequirements);
    VMA_COPY_IF_NOT_NULL(vkGetImageMemoryRequirements);
    VMA_COPY_IF_NOT_NULL(vkCreateBuffer);
    VMA_COPY_IF_NOT_NULL(vkDestroyBuffer);
    VMA_COPY_IF_NOT_NULL(vkCreateImage);
    VMA_COPY_IF_NOT_NULL(vkDestroyImage);
    VMA_COPY_IF_NOT_NULL(vkCmdCopyBuffer);

#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
    VMA_COPY_IF_NOT_NULL(vkGetBufferMemoryRequirements2KHR);
    VMA_COPY_IF_NOT_NULL(vkGetImageMemoryRequirements2KHR);
#endif

#if VMA_BIND_MEMORY2 || VMA_VULKAN_VERSION >= 1001000
    VMA_COPY_IF_NOT_NULL(vkBindBufferMemory2KHR);
    VMA_COPY_IF_NOT_NULL(vkBindImageMemory2KHR);
#endif

#if VMA_MEMORY_BUDGET || VMA_VULKAN_VERSION >= 1001000
    VMA_COPY_IF_NOT_NULL(vkGetPhysicalDeviceMemoryProperties2KHR);
#endif

#if VMA_KHR_MAINTENANCE4 || VMA_VULKAN_VERSION >= 1003000
    VMA_COPY_IF_NOT_NULL(vkGetDeviceBufferMemoryRequirements);
    VMA_COPY_IF_NOT_NULL(vkGetDeviceImageMemoryRequirements);
#endif
#if VMA_EXTERNAL_MEMORY_WIN32
    VMA_COPY_IF_NOT_NULL(vkGetMemoryWin32HandleKHR);
#endif
#undef VMA_COPY_IF_NOT_NULL
}

#if VMA_DYNAMIC_VULKAN_FUNCTIONS == 1

void VmaAllocator_T::import_vulkan_functions_dynamic()
{
    VMA_ASSERT(_vulkan_functions.vkGetInstanceProcAddr && _vulkan_functions.vkGetDeviceProcAddr &&
        "To use VMA_DYNAMIC_VULKAN_FUNCTIONS in new versions of VMA you now have to pass "
        "VmaVulkanFunctions::vkGetInstanceProcAddr and vkGetDeviceProcAddr as VmaAllocatorCreateInfo::pVulkanFunctions. "
        "Other members can be null.");

#define VMA_FETCH_INSTANCE_FUNC(memberName, functionPointerType, functionNameString) \
    if(_vulkan_functions.memberName == VMA_NULL) \
        _vulkan_functions.memberName = \
            (functionPointerType)_vulkan_functions.vkGetInstanceProcAddr(_h_instance, functionNameString);
#define VMA_FETCH_DEVICE_FUNC(memberName, functionPointerType, functionNameString) \
    if(_vulkan_functions.memberName == VMA_NULL) \
        _vulkan_functions.memberName = \
            (functionPointerType)_vulkan_functions.vkGetDeviceProcAddr(_h_device, functionNameString);

    VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceProperties, PFN_vkGetPhysicalDeviceProperties, "vkGetPhysicalDeviceProperties");
    VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties, PFN_vkGetPhysicalDeviceMemoryProperties, "vkGetPhysicalDeviceMemoryProperties");
    VMA_FETCH_DEVICE_FUNC(vkAllocateMemory, PFN_vkAllocateMemory, "vkAllocateMemory");
    VMA_FETCH_DEVICE_FUNC(vkFreeMemory, PFN_vkFreeMemory, "vkFreeMemory");
    VMA_FETCH_DEVICE_FUNC(vkMapMemory, PFN_vkMapMemory, "vkMapMemory");
    VMA_FETCH_DEVICE_FUNC(vkUnmapMemory, PFN_vkUnmapMemory, "vkUnmapMemory");
    VMA_FETCH_DEVICE_FUNC(vkFlushMappedMemoryRanges, PFN_vkFlushMappedMemoryRanges, "vkFlushMappedMemoryRanges");
    VMA_FETCH_DEVICE_FUNC(vkInvalidateMappedMemoryRanges, PFN_vkInvalidateMappedMemoryRanges, "vkInvalidateMappedMemoryRanges");
    VMA_FETCH_DEVICE_FUNC(vkBindBufferMemory, PFN_vkBindBufferMemory, "vkBindBufferMemory");
    VMA_FETCH_DEVICE_FUNC(vkBindImageMemory, PFN_vkBindImageMemory, "vkBindImageMemory");
    VMA_FETCH_DEVICE_FUNC(vkGetBufferMemoryRequirements, PFN_vkGetBufferMemoryRequirements, "vkGetBufferMemoryRequirements");
    VMA_FETCH_DEVICE_FUNC(vkGetImageMemoryRequirements, PFN_vkGetImageMemoryRequirements, "vkGetImageMemoryRequirements");
    VMA_FETCH_DEVICE_FUNC(vkCreateBuffer, PFN_vkCreateBuffer, "vkCreateBuffer");
    VMA_FETCH_DEVICE_FUNC(vkDestroyBuffer, PFN_vkDestroyBuffer, "vkDestroyBuffer");
    VMA_FETCH_DEVICE_FUNC(vkCreateImage, PFN_vkCreateImage, "vkCreateImage");
    VMA_FETCH_DEVICE_FUNC(vkDestroyImage, PFN_vkDestroyImage, "vkDestroyImage");
    VMA_FETCH_DEVICE_FUNC(vkCmdCopyBuffer, PFN_vkCmdCopyBuffer, "vkCmdCopyBuffer");

#if VMA_VULKAN_VERSION >= 1001000
    if(_vulkan_api_version >= VK_MAKE_VERSION(1, 1, 0))
    {
        VMA_FETCH_DEVICE_FUNC(vkGetBufferMemoryRequirements2KHR, PFN_vkGetBufferMemoryRequirements2, "vkGetBufferMemoryRequirements2");
        VMA_FETCH_DEVICE_FUNC(vkGetImageMemoryRequirements2KHR, PFN_vkGetImageMemoryRequirements2, "vkGetImageMemoryRequirements2");
        VMA_FETCH_DEVICE_FUNC(vkBindBufferMemory2KHR, PFN_vkBindBufferMemory2, "vkBindBufferMemory2");
        VMA_FETCH_DEVICE_FUNC(vkBindImageMemory2KHR, PFN_vkBindImageMemory2, "vkBindImageMemory2");
    }
#endif

#if VMA_MEMORY_BUDGET || VMA_VULKAN_VERSION >= 1001000
    if(_vulkan_api_version >= VK_MAKE_VERSION(1, 1, 0))
    {
        VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties2KHR, PFN_vkGetPhysicalDeviceMemoryProperties2KHR, "vkGetPhysicalDeviceMemoryProperties2");
        // Try to fetch the pointer from the other name, based on suspected driver bug - see issue #410.
        VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties2KHR, PFN_vkGetPhysicalDeviceMemoryProperties2KHR, "vkGetPhysicalDeviceMemoryProperties2KHR");
    }
    else if(_use_ext_memory_budget)
    {
        VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties2KHR, PFN_vkGetPhysicalDeviceMemoryProperties2KHR, "vkGetPhysicalDeviceMemoryProperties2KHR");
        // Try to fetch the pointer from the other name, based on suspected driver bug - see issue #410.
        VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties2KHR, PFN_vkGetPhysicalDeviceMemoryProperties2KHR, "vkGetPhysicalDeviceMemoryProperties2");
    }
#endif

#if VMA_DEDICATED_ALLOCATION
    if(_use_khr_dedicated_allocation)
    {
        VMA_FETCH_DEVICE_FUNC(vkGetBufferMemoryRequirements2KHR, PFN_vkGetBufferMemoryRequirements2KHR, "vkGetBufferMemoryRequirements2KHR");
        VMA_FETCH_DEVICE_FUNC(vkGetImageMemoryRequirements2KHR, PFN_vkGetImageMemoryRequirements2KHR, "vkGetImageMemoryRequirements2KHR");
    }
#endif

#if VMA_BIND_MEMORY2
    if(_use_khr_bind_memory2)
    {
        VMA_FETCH_DEVICE_FUNC(vkBindBufferMemory2KHR, PFN_vkBindBufferMemory2KHR, "vkBindBufferMemory2KHR");
        VMA_FETCH_DEVICE_FUNC(vkBindImageMemory2KHR, PFN_vkBindImageMemory2KHR, "vkBindImageMemory2KHR");
    }
#endif // #if VMA_BIND_MEMORY2

#if VMA_MEMORY_BUDGET || VMA_VULKAN_VERSION >= 1001000
    if(_vulkan_api_version >= VK_MAKE_VERSION(1, 1, 0))
    {
        VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties2KHR, PFN_vkGetPhysicalDeviceMemoryProperties2KHR, "vkGetPhysicalDeviceMemoryProperties2");
    }
    else if(_use_ext_memory_budget)
    {
        VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties2KHR, PFN_vkGetPhysicalDeviceMemoryProperties2KHR, "vkGetPhysicalDeviceMemoryProperties2KHR");
    }
#endif // #if VMA_MEMORY_BUDGET

#if VMA_VULKAN_VERSION >= 1003000
    if(_vulkan_api_version >= VK_MAKE_VERSION(1, 3, 0))
    {
        VMA_FETCH_DEVICE_FUNC(vkGetDeviceBufferMemoryRequirements, PFN_vkGetDeviceBufferMemoryRequirements, "vkGetDeviceBufferMemoryRequirements");
        VMA_FETCH_DEVICE_FUNC(vkGetDeviceImageMemoryRequirements, PFN_vkGetDeviceImageMemoryRequirements, "vkGetDeviceImageMemoryRequirements");
    }
#endif
#if VMA_KHR_MAINTENANCE4
    if(_use_khr_maintenance4)
    {
        VMA_FETCH_DEVICE_FUNC(vkGetDeviceBufferMemoryRequirements, PFN_vkGetDeviceBufferMemoryRequirementsKHR, "vkGetDeviceBufferMemoryRequirementsKHR");
        VMA_FETCH_DEVICE_FUNC(vkGetDeviceImageMemoryRequirements, PFN_vkGetDeviceImageMemoryRequirementsKHR, "vkGetDeviceImageMemoryRequirementsKHR");
    }
#endif
#if VMA_EXTERNAL_MEMORY_WIN32
    if (_use_khr_external_memory_win32)
    {
        VMA_FETCH_DEVICE_FUNC(vkGetMemoryWin32HandleKHR, PFN_vkGetMemoryWin32HandleKHR, "vkGetMemoryWin32HandleKHR");
    }
#endif
#undef VMA_FETCH_DEVICE_FUNC
#undef VMA_FETCH_INSTANCE_FUNC
}

#endif // VMA_DYNAMIC_VULKAN_FUNCTIONS == 1

void VmaAllocator_T::validate_vulkan_functions() const
{
    VMA_ASSERT(_vulkan_functions.vkGetPhysicalDeviceProperties != VMA_NULL);
    VMA_ASSERT(_vulkan_functions.vkGetPhysicalDeviceMemoryProperties != VMA_NULL);
    VMA_ASSERT(_vulkan_functions.vkAllocateMemory != VMA_NULL);
    VMA_ASSERT(_vulkan_functions.vkFreeMemory != VMA_NULL);
    VMA_ASSERT(_vulkan_functions.vkMapMemory != VMA_NULL);
    VMA_ASSERT(_vulkan_functions.vkUnmapMemory != VMA_NULL);
    VMA_ASSERT(_vulkan_functions.vkFlushMappedMemoryRanges != VMA_NULL);
    VMA_ASSERT(_vulkan_functions.vkInvalidateMappedMemoryRanges != VMA_NULL);
    VMA_ASSERT(_vulkan_functions.vkBindBufferMemory != VMA_NULL);
    VMA_ASSERT(_vulkan_functions.vkBindImageMemory != VMA_NULL);
    VMA_ASSERT(_vulkan_functions.vkGetBufferMemoryRequirements != VMA_NULL);
    VMA_ASSERT(_vulkan_functions.vkGetImageMemoryRequirements != VMA_NULL);
    VMA_ASSERT(_vulkan_functions.vkCreateBuffer != VMA_NULL);
    VMA_ASSERT(_vulkan_functions.vkDestroyBuffer != VMA_NULL);
    VMA_ASSERT(_vulkan_functions.vkCreateImage != VMA_NULL);
    VMA_ASSERT(_vulkan_functions.vkDestroyImage != VMA_NULL);
    VMA_ASSERT(_vulkan_functions.vkCmdCopyBuffer != VMA_NULL);

#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
    if(_vulkan_api_version >= VK_MAKE_VERSION(1, 1, 0) || _use_khr_dedicated_allocation)
    {
        VMA_ASSERT(_vulkan_functions.vkGetBufferMemoryRequirements2KHR != VMA_NULL);
        VMA_ASSERT(_vulkan_functions.vkGetImageMemoryRequirements2KHR != VMA_NULL);
    }
#endif

#if VMA_BIND_MEMORY2 || VMA_VULKAN_VERSION >= 1001000
    if(_vulkan_api_version >= VK_MAKE_VERSION(1, 1, 0) || _use_khr_bind_memory2)
    {
        VMA_ASSERT(_vulkan_functions.vkBindBufferMemory2KHR != VMA_NULL);
        VMA_ASSERT(_vulkan_functions.vkBindImageMemory2KHR != VMA_NULL);
    }
#endif

#if VMA_MEMORY_BUDGET || VMA_VULKAN_VERSION >= 1001000
    if(_use_ext_memory_budget || _vulkan_api_version >= VK_MAKE_VERSION(1, 1, 0))
    {
        VMA_ASSERT(_vulkan_functions.vkGetPhysicalDeviceMemoryProperties2KHR != VMA_NULL);
    }
#endif
#if VMA_EXTERNAL_MEMORY_WIN32
    if (_use_khr_external_memory_win32)
    {
        VMA_ASSERT(_vulkan_functions.vkGetMemoryWin32HandleKHR != VMA_NULL);
    }
#endif

    // Not validating these due to suspected driver bugs with these function
    // pointers being null despite correct extension or Vulkan version is enabled.
    // See issue #397. Their usage in VMA is optional anyway.
    //
    // VMA_ASSERT(_vulkan_functions.vkGetDeviceBufferMemoryRequirements != VMA_NULL);
    // VMA_ASSERT(_vulkan_functions.vkGetDeviceImageMemoryRequirements != VMA_NULL);
}

VkDeviceSize VmaAllocator_T::calc_preferred_block_size(uint32_t memTypeIndex)
{
    const uint32_t heapIndex = memory_type_index_to_heap_index(memTypeIndex);
    const VkDeviceSize heapSize = _mem_props.memoryHeaps[heapIndex].size;
    const bool isSmallHeap = heapSize <= VMA_SMALL_HEAP_MAX_SIZE;
    return VmaAlignUp(isSmallHeap ? (heapSize / 8) : _preferred_large_heap_block_size, (VkDeviceSize)32);
}

VkResult VmaAllocator_T::allocate_memory_of_type(
    VmaPool pool,
    VkDeviceSize size,
    VkDeviceSize alignment,
    bool dedicatedPreferred,
    VkBuffer dedicatedBuffer,
    VkImage dedicatedImage,
    VmaBufferImageUsage dedicatedBufferImageUsage,
    const VmaAllocationCreateInfo& createInfo,
    uint32_t memTypeIndex,
    VmaSuballocationType suballocType,
    VmaDedicatedAllocationList& dedicatedAllocations,
    VmaBlockVector& blockVector,
    size_t allocationCount,
    VmaAllocation* pAllocations)
{
    VMA_ASSERT(pAllocations != VMA_NULL);
    VMA_DEBUG_LOG_FORMAT("  AllocateMemory: MemoryTypeIndex=%" PRIu32 ", AllocationCount=%zu, Size=%" PRIu64, memTypeIndex, allocationCount, size);

    VmaAllocationCreateInfo finalCreateInfo = createInfo;
    VkResult res = calc_mem_type_params(
        finalCreateInfo,
        memTypeIndex,
        size,
        allocationCount);
    if(res != VK_SUCCESS)
        return res;

    if((finalCreateInfo.flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT) != 0)
    {
        return allocate_dedicated_memory(
            pool,
            size,
            suballocType,
            dedicatedAllocations,
            memTypeIndex,
            (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0,
            (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT) != 0,
            (finalCreateInfo.flags &
                (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0,
            (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT) != 0,
            finalCreateInfo.pUserData,
            finalCreateInfo.priority,
            dedicatedBuffer,
            dedicatedImage,
            dedicatedBufferImageUsage,
            allocationCount,
            pAllocations,
            blockVector.get_allocation_next_ptr());
    }

    const bool canAllocateDedicated =
        (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) == 0 &&
        (pool == VK_NULL_HANDLE || !blockVector.has_explicit_block_size());

    if(canAllocateDedicated)
    {
        // Heuristics: Allocate dedicated memory if requested size if greater than half of preferred block size.
        if(size > blockVector.get_preferred_block_size() / 2)
        {
            dedicatedPreferred = true;
        }
        // Protection against creating each allocation as dedicated when we reach or exceed heap size/budget,
        // which can quickly deplete maxMemoryAllocationCount: Don't prefer dedicated allocations when above
        // 3/4 of the maximum allocation count.
        if(_physical_device_properties.limits.maxMemoryAllocationCount < UINT32_MAX / 4 &&
            _device_memory_count.load() > _physical_device_properties.limits.maxMemoryAllocationCount * 3 / 4)
        {
            dedicatedPreferred = false;
        }

        if(dedicatedPreferred)
        {
            res = allocate_dedicated_memory(
                pool,
                size,
                suballocType,
                dedicatedAllocations,
                memTypeIndex,
                (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0,
                (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT) != 0,
                (finalCreateInfo.flags &
                    (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0,
                (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT) != 0,
                finalCreateInfo.pUserData,
                finalCreateInfo.priority,
                dedicatedBuffer,
                dedicatedImage,
                dedicatedBufferImageUsage,
                allocationCount,
                pAllocations,
                blockVector.get_allocation_next_ptr());
            if(res == VK_SUCCESS)
            {
                // Succeeded: AllocateDedicatedMemory function already filled pMemory, nothing more to do here.
                VMA_DEBUG_LOG("    Allocated as DedicatedMemory");
                return VK_SUCCESS;
            }
        }
    }

    res = blockVector.Allocate(
        size,
        alignment,
        finalCreateInfo,
        suballocType,
        allocationCount,
        pAllocations);
    if(res == VK_SUCCESS)
        return VK_SUCCESS;

    // Try dedicated memory.
    if(canAllocateDedicated && !dedicatedPreferred)
    {
        res = allocate_dedicated_memory(
            pool,
            size,
            suballocType,
            dedicatedAllocations,
            memTypeIndex,
            (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0,
            (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT) != 0,
            (finalCreateInfo.flags &
                (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0,
            (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT) != 0,
            finalCreateInfo.pUserData,
            finalCreateInfo.priority,
            dedicatedBuffer,
            dedicatedImage,
            dedicatedBufferImageUsage,
            allocationCount,
            pAllocations,
            blockVector.get_allocation_next_ptr());
        if(res == VK_SUCCESS)
        {
            // Succeeded: AllocateDedicatedMemory function already filled pMemory, nothing more to do here.
            VMA_DEBUG_LOG("    Allocated as DedicatedMemory");
            return VK_SUCCESS;
        }
    }
    // Everything failed: Return error code.
    VMA_DEBUG_LOG("    vkAllocateMemory FAILED");
    return res;
}

VkResult VmaAllocator_T::allocate_dedicated_memory(
    VmaPool pool,
    VkDeviceSize size,
    VmaSuballocationType suballocType,
    VmaDedicatedAllocationList& dedicatedAllocations,
    uint32_t memTypeIndex,
    bool map,
    bool isUserDataString,
    bool isMappingAllowed,
    bool canAliasMemory,
    void* pUserData,
    float priority,
    VkBuffer dedicatedBuffer,
    VkImage dedicatedImage,
    VmaBufferImageUsage dedicatedBufferImageUsage,
    size_t allocationCount,
    VmaAllocation* pAllocations,
    const void* pNextChain)
{
    VMA_ASSERT(allocationCount > 0 && pAllocations);

    VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocInfo.memoryTypeIndex = memTypeIndex;
    allocInfo.allocationSize = size;
    allocInfo.pNext = pNextChain;

#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
    VkMemoryDedicatedAllocateInfoKHR dedicatedAllocInfo = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR };
    if(!canAliasMemory)
    {
        if(_use_khr_dedicated_allocation || _vulkan_api_version >= VK_MAKE_VERSION(1, 1, 0))
        {
            if(dedicatedBuffer != VK_NULL_HANDLE)
            {
                VMA_ASSERT(dedicatedImage == VK_NULL_HANDLE);
                dedicatedAllocInfo.buffer = dedicatedBuffer;
                VmaPnextChainPushFront(&allocInfo, &dedicatedAllocInfo);
            }
            else if(dedicatedImage != VK_NULL_HANDLE)
            {
                dedicatedAllocInfo.image = dedicatedImage;
                VmaPnextChainPushFront(&allocInfo, &dedicatedAllocInfo);
            }
        }
    }
#endif // #if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000

#if VMA_BUFFER_DEVICE_ADDRESS
    VkMemoryAllocateFlagsInfoKHR allocFlagsInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_KHR };
    if(_use_khr_buffer_device_address)
    {
        bool canContainBufferWithDeviceAddress = true;
        if(dedicatedBuffer != VK_NULL_HANDLE)
        {
            canContainBufferWithDeviceAddress = dedicatedBufferImageUsage == VmaBufferImageUsage::UNKNOWN ||
                dedicatedBufferImageUsage.Contains(VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_EXT);
        }
        else if(dedicatedImage != VK_NULL_HANDLE)
        {
            canContainBufferWithDeviceAddress = false;
        }
        if(canContainBufferWithDeviceAddress)
        {
            allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
            VmaPnextChainPushFront(&allocInfo, &allocFlagsInfo);
        }
    }
#endif // #if VMA_BUFFER_DEVICE_ADDRESS

#if VMA_MEMORY_PRIORITY
    VkMemoryPriorityAllocateInfoEXT priorityInfo = { VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT };
    if(_use_ext_memory_priority)
    {
        VMA_ASSERT(priority >= 0.F && priority <= 1.F);
        priorityInfo.priority = priority;
        VmaPnextChainPushFront(&allocInfo, &priorityInfo);
    }
#endif // #if VMA_MEMORY_PRIORITY

#if VMA_EXTERNAL_MEMORY
    // Attach VkExportMemoryAllocateInfoKHR if necessary.
    VkExportMemoryAllocateInfoKHR exportMemoryAllocInfo = { VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR };
    exportMemoryAllocInfo.handleTypes = get_external_memory_handle_type_flags(memTypeIndex);
    if(exportMemoryAllocInfo.handleTypes != 0)
    {
        VmaPnextChainPushFront(&allocInfo, &exportMemoryAllocInfo);
    }
#endif // #if VMA_EXTERNAL_MEMORY

    size_t allocIndex = 0;
    VkResult res = VK_SUCCESS;
    for(; allocIndex < allocationCount; ++allocIndex)
    {
        res = allocate_dedicated_memory_page(
            pool,
            size,
            suballocType,
            memTypeIndex,
            allocInfo,
            map,
            isUserDataString,
            isMappingAllowed,
            pUserData,
            pAllocations + allocIndex);
        if(res != VK_SUCCESS)
        {
            break;
        }
    }

    if(res == VK_SUCCESS)
    {
        for (allocIndex = 0; allocIndex < allocationCount; ++allocIndex)
        {
            dedicatedAllocations.register_allocation(pAllocations[allocIndex]);
        }
        VMA_DEBUG_LOG_FORMAT("    Allocated DedicatedMemory Count=%zu, MemoryTypeIndex=#%" PRIu32, allocationCount, memTypeIndex);
    }
    else
    {
        // Free all already created allocations.
        while(allocIndex--)
        {
            VmaAllocation currAlloc = pAllocations[allocIndex];
            VkDeviceMemory hMemory = currAlloc->get_memory();

            /*
            There is no need to call this, because Vulkan spec allows to skip vkUnmapMemory
            before vkFreeMemory.

            if(currAlloc->get_mapped_data() != VMA_NULL)
            {
                (*_vulkan_functions.vkUnmapMemory)(_h_device, hMemory);
            }
            */

            free_vulkan_memory(memTypeIndex, currAlloc->get_size(), hMemory);
            _budget.remove_allocation(memory_type_index_to_heap_index(memTypeIndex), currAlloc->get_size());
            _allocation_object_allocator.free(currAlloc);
        }

        memset(pAllocations, 0, sizeof(VmaAllocation) * allocationCount);
    }

    return res;
}

VkResult VmaAllocator_T::allocate_dedicated_memory_page(
    VmaPool pool,
    VkDeviceSize size,
    VmaSuballocationType suballocType,
    uint32_t memTypeIndex,
    const VkMemoryAllocateInfo& allocInfo,
    bool map,
    bool isUserDataString,
    bool isMappingAllowed,
    void* pUserData,
    VmaAllocation* pAllocation)
{
    VkDeviceMemory hMemory = VK_NULL_HANDLE;
    VkResult res = allocate_vulkan_memory(&allocInfo, &hMemory);
    if(res < 0)
    {
        VMA_DEBUG_LOG("    vkAllocateMemory FAILED");
        return res;
    }

    void* pMappedData = VMA_NULL;
    if(map)
    {
        res = (*_vulkan_functions.vkMapMemory)(
            _h_device,
            hMemory,
            0,
            VK_WHOLE_SIZE,
            0,
            &pMappedData);
        if(res < 0)
        {
            VMA_DEBUG_LOG("    vkMapMemory FAILED");
            free_vulkan_memory(memTypeIndex, size, hMemory);
            return res;
        }
    }

    *pAllocation = _allocation_object_allocator.Allocate(isMappingAllowed);
    (*pAllocation)->init_dedicated_allocation(this, pool, memTypeIndex, hMemory, suballocType, pMappedData, size);
    if (isUserDataString)
        (*pAllocation)->set_name(this, (const char*)pUserData);
    else
        (*pAllocation)->set_user_data(this, pUserData);
    _budget.add_allocation(memory_type_index_to_heap_index(memTypeIndex), size);
    if(VMA_DEBUG_INITIALIZE_ALLOCATIONS)
    {
        fill_allocation(*pAllocation, VMA_ALLOCATION_FILL_PATTERN_CREATED);
    }

    return VK_SUCCESS;
}

void VmaAllocator_T::get_buffer_memory_requirements(
    VkBuffer hBuffer,
    VkMemoryRequirements& memReq,
    bool& requiresDedicatedAllocation,
    bool& prefersDedicatedAllocation) const
{
#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
    if(_use_khr_dedicated_allocation || _vulkan_api_version >= VK_MAKE_VERSION(1, 1, 0))
    {
        VkBufferMemoryRequirementsInfo2KHR memReqInfo = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2_KHR };
        memReqInfo.buffer = hBuffer;

        VkMemoryDedicatedRequirementsKHR memDedicatedReq = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR };

        VkMemoryRequirements2KHR memReq2 = { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR };
        VmaPnextChainPushFront(&memReq2, &memDedicatedReq);

        (*_vulkan_functions.vkGetBufferMemoryRequirements2KHR)(_h_device, &memReqInfo, &memReq2);

        memReq = memReq2.memoryRequirements;
        requiresDedicatedAllocation = (memDedicatedReq.requiresDedicatedAllocation != VK_FALSE);
        prefersDedicatedAllocation  = (memDedicatedReq.prefersDedicatedAllocation  != VK_FALSE);
    }
    else
#endif // #if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
    {
        (*_vulkan_functions.vkGetBufferMemoryRequirements)(_h_device, hBuffer, &memReq);
        requiresDedicatedAllocation = false;
        prefersDedicatedAllocation  = false;
    }
}

void VmaAllocator_T::get_image_memory_requirements(
    VkImage hImage,
    VkMemoryRequirements& memReq,
    bool& requiresDedicatedAllocation,
    bool& prefersDedicatedAllocation) const
{
#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
    if(_use_khr_dedicated_allocation || _vulkan_api_version >= VK_MAKE_VERSION(1, 1, 0))
    {
        VkImageMemoryRequirementsInfo2KHR memReqInfo = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2_KHR };
        memReqInfo.image = hImage;

        VkMemoryDedicatedRequirementsKHR memDedicatedReq = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR };

        VkMemoryRequirements2KHR memReq2 = { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR };
        VmaPnextChainPushFront(&memReq2, &memDedicatedReq);

        (*_vulkan_functions.vkGetImageMemoryRequirements2KHR)(_h_device, &memReqInfo, &memReq2);

        memReq = memReq2.memoryRequirements;
        requiresDedicatedAllocation = (memDedicatedReq.requiresDedicatedAllocation != VK_FALSE);
        prefersDedicatedAllocation  = (memDedicatedReq.prefersDedicatedAllocation  != VK_FALSE);
    }
    else
#endif // #if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
    {
        (*_vulkan_functions.vkGetImageMemoryRequirements)(_h_device, hImage, &memReq);
        requiresDedicatedAllocation = false;
        prefersDedicatedAllocation  = false;
    }
}

VkResult VmaAllocator_T::find_memory_type_index(
    uint32_t memoryTypeBits,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    VmaBufferImageUsage bufImgUsage,
    uint32_t* pMemoryTypeIndex) const
{
    memoryTypeBits &= get_global_memory_type_bits();

    if(pAllocationCreateInfo->memoryTypeBits != 0)
    {
        memoryTypeBits &= pAllocationCreateInfo->memoryTypeBits;
    }

    VkMemoryPropertyFlags requiredFlags = 0;
    VkMemoryPropertyFlags preferredFlags = 0;
    VkMemoryPropertyFlags notPreferredFlags = 0;
    if(!FindMemoryPreferences(
        is_integrated_gpu(),
        *pAllocationCreateInfo,
        bufImgUsage,
        requiredFlags, preferredFlags, notPreferredFlags))
    {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    *pMemoryTypeIndex = UINT32_MAX;
    uint32_t minCost = UINT32_MAX;
    for(uint32_t memTypeIndex = 0, memTypeBit = 1;
        memTypeIndex < get_memory_type_count();
        ++memTypeIndex, memTypeBit <<= 1)
    {
        // This memory type is acceptable according to memoryTypeBits bitmask.
        if((memTypeBit & memoryTypeBits) != 0)
        {
            const VkMemoryPropertyFlags currFlags =
                _mem_props.memoryTypes[memTypeIndex].propertyFlags;
            // This memory type contains requiredFlags.
            if((requiredFlags & ~currFlags) == 0)
            {
                // Calculate cost as number of bits from preferredFlags not present in this memory type.
                uint32_t currCost = VMA_COUNT_BITS_SET(preferredFlags & ~currFlags) +
                    VMA_COUNT_BITS_SET(currFlags & notPreferredFlags);
                // Remember memory type with lowest cost.
                if(currCost < minCost)
                {
                    *pMemoryTypeIndex = memTypeIndex;
                    if(currCost == 0)
                    {
                        return VK_SUCCESS;
                    }
                    minCost = currCost;
                }
            }
        }
    }
    return (*pMemoryTypeIndex != UINT32_MAX) ? VK_SUCCESS : VK_ERROR_FEATURE_NOT_PRESENT;
}

VkResult VmaAllocator_T::calc_mem_type_params(
    VmaAllocationCreateInfo& inoutCreateInfo,
    uint32_t memTypeIndex,
    VkDeviceSize size,
    size_t allocationCount)
{
    // If memory type is not HOST_VISIBLE, disable MAPPED.
    if((inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0 &&
        (_mem_props.memoryTypes[memTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0)
    {
        inoutCreateInfo.flags &= ~VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    if((inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT) != 0 &&
        (inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT) != 0)
    {
        const uint32_t heapIndex = memory_type_index_to_heap_index(memTypeIndex);
        VmaBudget heapBudget = {};
        get_heap_budgets(&heapBudget, heapIndex, 1);
        if(heapBudget.usage + size * allocationCount > heapBudget.budget)
        {
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
    }
    return VK_SUCCESS;
}

VkResult VmaAllocator_T::calc_allocation_params(
    VmaAllocationCreateInfo& inoutCreateInfo,
    bool dedicatedRequired)
{
    VMA_ASSERT((inoutCreateInfo.flags &
        (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) !=
        (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT) &&
        "Specifying both flags VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT and VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT is incorrect.");
    VMA_ASSERT((((inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT) == 0 ||
        (inoutCreateInfo.flags & (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0)) &&
        "Specifying VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT requires also VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT.");
    if(inoutCreateInfo.usage == VMA_MEMORY_USAGE_AUTO || inoutCreateInfo.usage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE || inoutCreateInfo.usage == VMA_MEMORY_USAGE_AUTO_PREFER_HOST)
    {
        if((inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0)
        {
            VMA_ASSERT((inoutCreateInfo.flags & (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0 &&
                "When using VMA_ALLOCATION_CREATE_MAPPED_BIT and usage = VMA_MEMORY_USAGE_AUTO*, you must also specify VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT.");
        }
    }

    // If memory is lazily allocated, it should be always dedicated.
    if(dedicatedRequired ||
        inoutCreateInfo.usage == VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED)
    {
        inoutCreateInfo.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    }

    if(inoutCreateInfo.pool != VK_NULL_HANDLE)
    {
        if(inoutCreateInfo.pool->_block_vector.has_explicit_block_size() &&
            (inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT) != 0)
        {
            VMA_ASSERT(0 && "Specifying VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT while current custom pool doesn't support dedicated allocations.");
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        inoutCreateInfo.priority = inoutCreateInfo.pool->_block_vector.get_priority();
    }

    if((inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT) != 0 &&
        (inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) != 0)
    {
        VMA_ASSERT(0 && "Specifying VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT together with VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT makes no sense.");
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    if(VMA_DEBUG_ALWAYS_DEDICATED_MEMORY &&
        (inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) != 0)
    {
        inoutCreateInfo.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    }

    // Non-auto USAGE values imply HOST_ACCESS flags.
    // And so does VMA_MEMORY_USAGE_UNKNOWN because it is used with custom pools.
    // Which specific flag is used doesn't matter. They change things only when used with VMA_MEMORY_USAGE_AUTO*.
    // Otherwise they just protect from assert on mapping.
    if(inoutCreateInfo.usage != VMA_MEMORY_USAGE_AUTO &&
        inoutCreateInfo.usage != VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE &&
        inoutCreateInfo.usage != VMA_MEMORY_USAGE_AUTO_PREFER_HOST)
    {
        if((inoutCreateInfo.flags & (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) == 0)
        {
            inoutCreateInfo.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        }
    }

    return VK_SUCCESS;
}

VkResult VmaAllocator_T::allocate_memory(
    const VkMemoryRequirements& vkMemReq,
    bool requiresDedicatedAllocation,
    bool prefersDedicatedAllocation,
    VkBuffer dedicatedBuffer,
    VkImage dedicatedImage,
    VmaBufferImageUsage dedicatedBufferImageUsage,
    const VmaAllocationCreateInfo& createInfo,
    VmaSuballocationType suballocType,
    size_t allocationCount,
    VmaAllocation* pAllocations)
{
    memset(pAllocations, 0, sizeof(VmaAllocation) * allocationCount);

    VMA_ASSERT(VmaIsPow2(vkMemReq.alignment));

    if(vkMemReq.size == 0)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VmaAllocationCreateInfo createInfoFinal = createInfo;
    VkResult res = calc_allocation_params(createInfoFinal, requiresDedicatedAllocation);
    if(res != VK_SUCCESS)
        return res;

    if(createInfoFinal.pool != VK_NULL_HANDLE)
    {
        VmaBlockVector& blockVector = createInfoFinal.pool->_block_vector;
        return allocate_memory_of_type(
            createInfoFinal.pool,
            vkMemReq.size,
            vkMemReq.alignment,
            prefersDedicatedAllocation,
            dedicatedBuffer,
            dedicatedImage,
            dedicatedBufferImageUsage,
            createInfoFinal,
            blockVector.get_memory_type_index(),
            suballocType,
            createInfoFinal.pool->_dedicated_allocations,
            blockVector,
            allocationCount,
            pAllocations);
    }

    // Bit mask of memory Vulkan types acceptable for this allocation.
    uint32_t memoryTypeBits = vkMemReq.memoryTypeBits;
    uint32_t memTypeIndex = UINT32_MAX;
    res = find_memory_type_index(memoryTypeBits, &createInfoFinal, dedicatedBufferImageUsage, &memTypeIndex);
    // Can't find any single memory type matching requirements. res is VK_ERROR_FEATURE_NOT_PRESENT.
    if(res != VK_SUCCESS)
        return res;
    
    do
    {
        VmaBlockVector* blockVector = _p_block_vectors[memTypeIndex];
        VMA_ASSERT(blockVector && "Trying to use unsupported memory type!");
        res = allocate_memory_of_type(
            VK_NULL_HANDLE,
            vkMemReq.size,
            vkMemReq.alignment,
            requiresDedicatedAllocation || prefersDedicatedAllocation,
            dedicatedBuffer,
            dedicatedImage,
            dedicatedBufferImageUsage,
            createInfoFinal,
            memTypeIndex,
            suballocType,
            _dedicated_allocations[memTypeIndex],
            *blockVector,
            allocationCount,
            pAllocations);
        // Allocation succeeded
        if(res == VK_SUCCESS)
            return VK_SUCCESS;

        // Remove old memTypeIndex from list of possibilities.
        memoryTypeBits &= ~(1U << memTypeIndex);
        // Find alternative memTypeIndex.
        res = find_memory_type_index(memoryTypeBits, &createInfoFinal, dedicatedBufferImageUsage, &memTypeIndex);
    } while(res == VK_SUCCESS);

    // No other matching memory type index could be found.
    // Not returning res, which is VK_ERROR_FEATURE_NOT_PRESENT, because we already failed to allocate once.
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

void VmaAllocator_T::free_memory(
    size_t allocationCount,
    const VmaAllocation* pAllocations)
{
    VMA_ASSERT(pAllocations);

    for(size_t allocIndex = allocationCount; allocIndex--; )
    {
        VmaAllocation allocation = pAllocations[allocIndex];

        if(allocation != VK_NULL_HANDLE)
        {
            if(VMA_DEBUG_INITIALIZE_ALLOCATIONS)
            {
                fill_allocation(allocation, VMA_ALLOCATION_FILL_PATTERN_DESTROYED);
            }

            switch(allocation->get_type())
            {
            case VmaAllocation_T::ALLOCATION_TYPE_BLOCK:
                {
                    VmaBlockVector* pBlockVector = VMA_NULL;
                    VmaPool hPool = allocation->get_parent_pool();
                    if(hPool != VK_NULL_HANDLE)
                    {
                        pBlockVector = &hPool->_block_vector;
                    }
                    else
                    {
                        const uint32_t memTypeIndex = allocation->get_memory_type_index();
                        pBlockVector = _p_block_vectors[memTypeIndex];
                        VMA_ASSERT(pBlockVector && "Trying to free memory of unsupported type!");
                    }
                    pBlockVector->free(allocation);
                }
                break;
            case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
                free_dedicated_memory(allocation);
                break;
            default:
                VMA_ASSERT(0);
            }
        }
    }
}

void VmaAllocator_T::calculate_statistics(VmaTotalStatistics* pStats)
{
    // Initialize.
    VmaClearDetailedStatistics(pStats->total);
    for(uint32_t i = 0; i < VK_MAX_MEMORY_TYPES; ++i)
        VmaClearDetailedStatistics(pStats->memoryType[i]);
    for(uint32_t i = 0; i < VK_MAX_MEMORY_HEAPS; ++i)
        VmaClearDetailedStatistics(pStats->memoryHeap[i]);

    // Process default pools.
    for(uint32_t memTypeIndex = 0; memTypeIndex < get_memory_type_count(); ++memTypeIndex)
    {
        VmaBlockVector* const pBlockVector = _p_block_vectors[memTypeIndex];
        if (pBlockVector != VMA_NULL)
            pBlockVector->add_detailed_statistics(pStats->memoryType[memTypeIndex]);
    }

    // Process custom pools.
    {
        VmaMutexLockRead lock(_pools_mutex, _use_mutex);
        for(VmaPool pool = _pools.front(); pool != VMA_NULL; pool = _pools.get_next(pool))
        {
            VmaBlockVector& blockVector = pool->_block_vector;
            const uint32_t memTypeIndex = blockVector.get_memory_type_index();
            blockVector.add_detailed_statistics(pStats->memoryType[memTypeIndex]);
            pool->_dedicated_allocations.add_detailed_statistics(pStats->memoryType[memTypeIndex]);
        }
    }

    // Process dedicated allocations.
    for(uint32_t memTypeIndex = 0; memTypeIndex < get_memory_type_count(); ++memTypeIndex)
    {
        _dedicated_allocations[memTypeIndex].add_detailed_statistics(pStats->memoryType[memTypeIndex]);
    }

    // Sum from memory types to memory heaps.
    for(uint32_t memTypeIndex = 0; memTypeIndex < get_memory_type_count(); ++memTypeIndex)
    {
        const uint32_t memHeapIndex = _mem_props.memoryTypes[memTypeIndex].heapIndex;
        VmaAddDetailedStatistics(pStats->memoryHeap[memHeapIndex], pStats->memoryType[memTypeIndex]);
    }

    // Sum from memory heaps to total.
    for(uint32_t memHeapIndex = 0; memHeapIndex < get_memory_heap_count(); ++memHeapIndex)
        VmaAddDetailedStatistics(pStats->total, pStats->memoryHeap[memHeapIndex]);

    VMA_ASSERT(pStats->total.statistics.allocationCount == 0 ||
        pStats->total.allocationSizeMax >= pStats->total.allocationSizeMin);
    VMA_ASSERT(pStats->total.unusedRangeCount == 0 ||
        pStats->total.unusedRangeSizeMax >= pStats->total.unusedRangeSizeMin);
}

void VmaAllocator_T::get_heap_budgets(VmaBudget* outBudgets, uint32_t firstHeap, uint32_t heapCount)
{
#if VMA_MEMORY_BUDGET
    if(_use_ext_memory_budget)
    {
        if(_budget._operations_since_budget_fetch < 30)
        {
            VmaMutexLockRead lockRead(_budget._budget_mutex, _use_mutex);
            for(uint32_t i = 0; i < heapCount; ++i, ++outBudgets)
            {
                const uint32_t heapIndex = firstHeap + i;

                outBudgets->statistics.blockCount = _budget._block_count[heapIndex];
                outBudgets->statistics.allocationCount = _budget._allocation_count[heapIndex];
                outBudgets->statistics.blockBytes = _budget._block_bytes[heapIndex];
                outBudgets->statistics.allocationBytes = _budget._allocation_bytes[heapIndex];

                if(_budget._vulkan_usage[heapIndex] + outBudgets->statistics.blockBytes > _budget._block_bytes_at_budget_fetch[heapIndex])
                {
                    outBudgets->usage = _budget._vulkan_usage[heapIndex] +
                        outBudgets->statistics.blockBytes - _budget._block_bytes_at_budget_fetch[heapIndex];
                }
                else
                {
                    outBudgets->usage = 0;
                }

                // Have to take MIN with heap size because explicit HeapSizeLimit is included in it.
                outBudgets->budget = VMA_MIN(
                    _budget._vulkan_budget[heapIndex], _mem_props.memoryHeaps[heapIndex].size);
            }
        }
        else
        {
            update_vulkan_budget(); // Outside of mutex lock
            get_heap_budgets(outBudgets, firstHeap, heapCount); // Recursion
        }
    }
    else
#endif
    {
        for(uint32_t i = 0; i < heapCount; ++i, ++outBudgets)
        {
            const uint32_t heapIndex = firstHeap + i;

            outBudgets->statistics.blockCount = _budget._block_count[heapIndex];
            outBudgets->statistics.allocationCount = _budget._allocation_count[heapIndex];
            outBudgets->statistics.blockBytes = _budget._block_bytes[heapIndex];
            outBudgets->statistics.allocationBytes = _budget._allocation_bytes[heapIndex];

            outBudgets->usage = outBudgets->statistics.blockBytes;
            outBudgets->budget = _mem_props.memoryHeaps[heapIndex].size * 8 / 10; // 80% heuristics.
        }
    }
}

void VmaAllocator_T::get_allocation_info(VmaAllocation hAllocation, VmaAllocationInfo* pAllocationInfo)
{
    pAllocationInfo->memoryType = hAllocation->get_memory_type_index();
    pAllocationInfo->deviceMemory = hAllocation->get_memory();
    pAllocationInfo->offset = hAllocation->get_offset();
    pAllocationInfo->size = hAllocation->get_size();
    pAllocationInfo->pMappedData = hAllocation->get_mapped_data();
    pAllocationInfo->pUserData = hAllocation->get_user_data();
    pAllocationInfo->pName = hAllocation->get_name();
}

void VmaAllocator_T::get_allocation_info2(VmaAllocation hAllocation, VmaAllocationInfo2* pAllocationInfo)
{
    get_allocation_info(hAllocation, &pAllocationInfo->allocationInfo);

    switch (hAllocation->get_type())
    {
    case VmaAllocation_T::ALLOCATION_TYPE_BLOCK:
        pAllocationInfo->blockSize = hAllocation->get_block()->_p_metadata->get_size();
        pAllocationInfo->dedicatedMemory = VK_FALSE;
        break;
    case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
        pAllocationInfo->blockSize = pAllocationInfo->allocationInfo.size;
        pAllocationInfo->dedicatedMemory = VK_TRUE;
        break;
    default:
        VMA_ASSERT(0);
    }
}

VkResult VmaAllocator_T::create_pool(const VmaPoolCreateInfo* pCreateInfo, VmaPool* pPool)
{
    VMA_DEBUG_LOG_FORMAT("  CreatePool: MemoryTypeIndex=%" PRIu32 ", flags=%" PRIu32, pCreateInfo->memoryTypeIndex, pCreateInfo->flags);

    VmaPoolCreateInfo newCreateInfo = *pCreateInfo;

    // Protection against uninitialized new structure member. If garbage data are left there, this pointer dereference would crash.
    if(pCreateInfo->pMemoryAllocateNext)
    {
        VMA_ASSERT(((const VkBaseInStructure*)pCreateInfo->pMemoryAllocateNext)->sType != 0);
    }

    if(newCreateInfo.maxBlockCount == 0)
    {
        newCreateInfo.maxBlockCount = SIZE_MAX;
    }
    if(newCreateInfo.minBlockCount > newCreateInfo.maxBlockCount)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    // Memory type index out of range or forbidden.
    if(pCreateInfo->memoryTypeIndex >= get_memory_type_count() ||
        ((1U << pCreateInfo->memoryTypeIndex) & _global_memory_type_bits) == 0)
    {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if(newCreateInfo.minAllocationAlignment > 0)
    {
        VMA_ASSERT(VmaIsPow2(newCreateInfo.minAllocationAlignment));
    }

    const VkDeviceSize preferredBlockSize = calc_preferred_block_size(newCreateInfo.memoryTypeIndex);

    *pPool = vma_new(this, VmaPool_T)(this, newCreateInfo, preferredBlockSize);

    VkResult res = (*pPool)->_block_vector.create_min_blocks();
    if(res != VK_SUCCESS)
    {
        vma_delete(this, *pPool);
        *pPool = VMA_NULL;
        return res;
    }

    // add to _pools.
    {
        VmaMutexLockWrite lock(_pools_mutex, _use_mutex);
        (*pPool)->set_id(_next_pool_id++);
        _pools.push_back(*pPool);
    }

    return VK_SUCCESS;
}

void VmaAllocator_T::destroy_pool(VmaPool pool)
{
    // Remove from _pools.
    {
        VmaMutexLockWrite lock(_pools_mutex, _use_mutex);
        _pools.remove(pool);
    }

    vma_delete(this, pool);
}

void VmaAllocator_T::get_pool_statistics(VmaPool pool, VmaStatistics* pPoolStats)
{
    VmaClearStatistics(*pPoolStats);
    pool->_block_vector.add_statistics(*pPoolStats);
    pool->_dedicated_allocations.add_statistics(*pPoolStats);
}

void VmaAllocator_T::calculate_pool_statistics(VmaPool pool, VmaDetailedStatistics* pPoolStats)
{
    VmaClearDetailedStatistics(*pPoolStats);
    pool->_block_vector.add_detailed_statistics(*pPoolStats);
    pool->_dedicated_allocations.add_detailed_statistics(*pPoolStats);
}

void VmaAllocator_T::set_current_frame_index(uint32_t frameIndex)
{
    _current_frame_index.store(frameIndex);

#if VMA_MEMORY_BUDGET
    if(_use_ext_memory_budget)
    {
        update_vulkan_budget();
    }
#endif // #if VMA_MEMORY_BUDGET
}

VkResult VmaAllocator_T::check_pool_corruption(VmaPool hPool)
{
    return hPool->_block_vector.check_corruption();
}

VkResult VmaAllocator_T::check_corruption(uint32_t memoryTypeBits)
{
    VkResult finalRes = VK_ERROR_FEATURE_NOT_PRESENT;

    // Process default pools.
    for(uint32_t memTypeIndex = 0; memTypeIndex < get_memory_type_count(); ++memTypeIndex)
    {
        VmaBlockVector* const pBlockVector = _p_block_vectors[memTypeIndex];
        if(pBlockVector != VMA_NULL)
        {
            VkResult localRes = pBlockVector->check_corruption();
            switch(localRes)
            {
            case VK_ERROR_FEATURE_NOT_PRESENT:
                break;
            case VK_SUCCESS:
                finalRes = VK_SUCCESS;
                break;
            default:
                return localRes;
            }
        }
    }

    // Process custom pools.
    {
        VmaMutexLockRead lock(_pools_mutex, _use_mutex);
        for(VmaPool pool = _pools.front(); pool != VMA_NULL; pool = _pools.get_next(pool))
        {
            if(((1U << pool->_block_vector.get_memory_type_index()) & memoryTypeBits) != 0)
            {
                VkResult localRes = pool->_block_vector.check_corruption();
                switch(localRes)
                {
                case VK_ERROR_FEATURE_NOT_PRESENT:
                    break;
                case VK_SUCCESS:
                    finalRes = VK_SUCCESS;
                    break;
                default:
                    return localRes;
                }
            }
        }
    }

    return finalRes;
}

VkResult VmaAllocator_T::allocate_vulkan_memory(const VkMemoryAllocateInfo* pAllocateInfo, VkDeviceMemory* pMemory)
{
    const uint32_t heapIndex = memory_type_index_to_heap_index(pAllocateInfo->memoryTypeIndex);

#if VMA_DEBUG_DONT_EXCEED_HEAP_SIZE_WITH_ALLOCATION_SIZE
    if (pAllocateInfo->allocationSize > _mem_props.memoryHeaps[heapIndex].size)
    {
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
#endif

    AtomicTransactionalIncrement<VMA_ATOMIC_UINT32> deviceMemoryCountIncrement;
    const uint64_t prevDeviceMemoryCount = deviceMemoryCountIncrement.Increment(&_device_memory_count);
#if VMA_DEBUG_DONT_EXCEED_MAX_MEMORY_ALLOCATION_COUNT
    if(prevDeviceMemoryCount >= _physical_device_properties.limits.maxMemoryAllocationCount)
    {
        return VK_ERROR_TOO_MANY_OBJECTS;
    }
#endif

    // HeapSizeLimit is in effect for this heap.
    if((_heap_size_limit_mask & (1U << heapIndex)) != 0)
    {
        const VkDeviceSize heapSize = _mem_props.memoryHeaps[heapIndex].size;
        VkDeviceSize blockBytes = _budget._block_bytes[heapIndex];
        for(;;)
        {
            const VkDeviceSize blockBytesAfterAllocation = blockBytes + pAllocateInfo->allocationSize;
            if(blockBytesAfterAllocation > heapSize)
            {
                return VK_ERROR_OUT_OF_DEVICE_MEMORY;
            }
            if(_budget._block_bytes[heapIndex].compare_exchange_strong(blockBytes, blockBytesAfterAllocation))
            {
                break;
            }
        }
    }
    else
    {
        _budget._block_bytes[heapIndex] += pAllocateInfo->allocationSize;
    }
    ++_budget._block_count[heapIndex];

    // VULKAN CALL vkAllocateMemory.
    VkResult res = (*_vulkan_functions.vkAllocateMemory)(_h_device, pAllocateInfo, get_allocation_callbacks(), pMemory);

    if(res == VK_SUCCESS)
    {
#if VMA_MEMORY_BUDGET
        ++_budget._operations_since_budget_fetch;
#endif

        // Informative callback.
        if(_device_memory_callbacks.pfnAllocate != VMA_NULL)
        {
            (*_device_memory_callbacks.pfnAllocate)(this, pAllocateInfo->memoryTypeIndex, *pMemory, pAllocateInfo->allocationSize, _device_memory_callbacks.pUserData);
        }

        deviceMemoryCountIncrement.commit();
    }
    else
    {
        --_budget._block_count[heapIndex];
        _budget._block_bytes[heapIndex] -= pAllocateInfo->allocationSize;
    }

    return res;
}

void VmaAllocator_T::free_vulkan_memory(uint32_t memoryType, VkDeviceSize size, VkDeviceMemory hMemory)
{
    // Informative callback.
    if(_device_memory_callbacks.pfnFree != VMA_NULL)
    {
        (*_device_memory_callbacks.pfnFree)(this, memoryType, hMemory, size, _device_memory_callbacks.pUserData);
    }

    // VULKAN CALL vkFreeMemory.
    (*_vulkan_functions.vkFreeMemory)(_h_device, hMemory, get_allocation_callbacks());

    const uint32_t heapIndex = memory_type_index_to_heap_index(memoryType);
    --_budget._block_count[heapIndex];
    _budget._block_bytes[heapIndex] -= size;

    --_device_memory_count;
}

VkResult VmaAllocator_T::bind_vulkan_buffer(
    VkDeviceMemory memory,
    VkDeviceSize memoryOffset,
    VkBuffer buffer,
    const void* pNext) const
{
    if(pNext != VMA_NULL)
    {
#if VMA_VULKAN_VERSION >= 1001000 || VMA_BIND_MEMORY2
        if((_use_khr_bind_memory2 || _vulkan_api_version >= VK_MAKE_VERSION(1, 1, 0)) &&
            _vulkan_functions.vkBindBufferMemory2KHR != VMA_NULL)
        {
            VkBindBufferMemoryInfoKHR bindBufferMemoryInfo = { VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO_KHR };
            bindBufferMemoryInfo.pNext = pNext;
            bindBufferMemoryInfo.buffer = buffer;
            bindBufferMemoryInfo.memory = memory;
            bindBufferMemoryInfo.memoryOffset = memoryOffset;
            return (*_vulkan_functions.vkBindBufferMemory2KHR)(_h_device, 1, &bindBufferMemoryInfo);
        }
#endif // #if VMA_VULKAN_VERSION >= 1001000 || VMA_BIND_MEMORY2

        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
    else
    {
        return (*_vulkan_functions.vkBindBufferMemory)(_h_device, buffer, memory, memoryOffset);
    }
}

VkResult VmaAllocator_T::bind_vulkan_image(
    VkDeviceMemory memory,
    VkDeviceSize memoryOffset,
    VkImage image,
    const void* pNext) const
{
    if(pNext != VMA_NULL)
    {
#if VMA_VULKAN_VERSION >= 1001000 || VMA_BIND_MEMORY2
        if((_use_khr_bind_memory2 || _vulkan_api_version >= VK_MAKE_VERSION(1, 1, 0)) &&
            _vulkan_functions.vkBindImageMemory2KHR != VMA_NULL)
        {
            VkBindImageMemoryInfoKHR bindBufferMemoryInfo = { VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO_KHR };
            bindBufferMemoryInfo.pNext = pNext;
            bindBufferMemoryInfo.image = image;
            bindBufferMemoryInfo.memory = memory;
            bindBufferMemoryInfo.memoryOffset = memoryOffset;
            return (*_vulkan_functions.vkBindImageMemory2KHR)(_h_device, 1, &bindBufferMemoryInfo);
        }
#endif // #if VMA_BIND_MEMORY2

        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

    return (*_vulkan_functions.vkBindImageMemory)(_h_device, image, memory, memoryOffset);
}

VkResult VmaAllocator_T::Map(VmaAllocation hAllocation, void** ppData)
{
    switch(hAllocation->get_type())
    {
    case VmaAllocation_T::ALLOCATION_TYPE_BLOCK:
        {
            VmaDeviceMemoryBlock* const pBlock = hAllocation->get_block();
            char *pBytes = VMA_NULL;
            VkResult res = pBlock->Map(this, 1, (void**)&pBytes);
            if(res == VK_SUCCESS)
            {
                *ppData = pBytes + (ptrdiff_t)hAllocation->get_offset();
                hAllocation->block_alloc_map();
            }
            return res;
        }
    case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
        return hAllocation->dedicated_alloc_map(this, ppData);
    default:
        VMA_ASSERT(0);
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
}

void VmaAllocator_T::unmap(VmaAllocation hAllocation)
{
    switch(hAllocation->get_type())
    {
    case VmaAllocation_T::ALLOCATION_TYPE_BLOCK:
        {
            VmaDeviceMemoryBlock* const pBlock = hAllocation->get_block();
            hAllocation->block_alloc_unmap();
            pBlock->unmap(this, 1);
        }
        break;
    case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
        hAllocation->dedicated_alloc_unmap(this);
        break;
    default:
        VMA_ASSERT(0);
    }
}

VkResult VmaAllocator_T::bind_buffer_memory(
    VmaAllocation hAllocation,
    VkDeviceSize allocationLocalOffset,
    VkBuffer hBuffer,
    const void* pNext)
{
    VkResult res = VK_ERROR_UNKNOWN_COPY;
    switch(hAllocation->get_type())
    {
    case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
        res = bind_vulkan_buffer(hAllocation->get_memory(), allocationLocalOffset, hBuffer, pNext);
        break;
    case VmaAllocation_T::ALLOCATION_TYPE_BLOCK:
    {
        VmaDeviceMemoryBlock* const pBlock = hAllocation->get_block();
        VMA_ASSERT(pBlock && "Binding buffer to allocation that doesn't belong to any block.");
        res = pBlock->bind_buffer_memory(this, hAllocation, allocationLocalOffset, hBuffer, pNext);
        break;
    }
    default:
        VMA_ASSERT(0);
    }
    return res;
}

VkResult VmaAllocator_T::bind_image_memory(
    VmaAllocation hAllocation,
    VkDeviceSize allocationLocalOffset,
    VkImage hImage,
    const void* pNext)
{
    VkResult res = VK_ERROR_UNKNOWN_COPY;
    switch(hAllocation->get_type())
    {
    case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
        res = bind_vulkan_image(hAllocation->get_memory(), allocationLocalOffset, hImage, pNext);
        break;
    case VmaAllocation_T::ALLOCATION_TYPE_BLOCK:
    {
        VmaDeviceMemoryBlock* pBlock = hAllocation->get_block();
        VMA_ASSERT(pBlock && "Binding image to allocation that doesn't belong to any block.");
        res = pBlock->bind_image_memory(this, hAllocation, allocationLocalOffset, hImage, pNext);
        break;
    }
    default:
        VMA_ASSERT(0);
    }
    return res;
}

VkResult VmaAllocator_T::flush_or_invalidate_allocation(
    VmaAllocation hAllocation,
    VkDeviceSize offset, VkDeviceSize size,
    VMA_CACHE_OPERATION op)
{
    VkResult res = VK_SUCCESS;

    VkMappedMemoryRange memRange = {};
    if(get_flush_or_invalidate_range(hAllocation, offset, size, memRange))
    {
        switch(op)
        {
        case VMA_CACHE_FLUSH:
            res = (*get_vulkan_functions().vkFlushMappedMemoryRanges)(_h_device, 1, &memRange);
            break;
        case VMA_CACHE_INVALIDATE:
            res = (*get_vulkan_functions().vkInvalidateMappedMemoryRanges)(_h_device, 1, &memRange);
            break;
        default:
            VMA_ASSERT(0);
        }
    }
    // else: Just ignore this call.
    return res;
}

VkResult VmaAllocator_T::flush_or_invalidate_allocations(
    uint32_t allocationCount,
    const VmaAllocation* allocations,
    const VkDeviceSize* offsets, const VkDeviceSize* sizes,
    VMA_CACHE_OPERATION op)
{
    typedef VmaStlAllocator<VkMappedMemoryRange> RangeAllocator;
    typedef VmaSmallVector<VkMappedMemoryRange, RangeAllocator, 16> RangeVector;
    RangeVector ranges = RangeVector(RangeAllocator(get_allocation_callbacks()));

    for(uint32_t allocIndex = 0; allocIndex < allocationCount; ++allocIndex)
    {
        const VmaAllocation alloc = allocations[allocIndex];
        const VkDeviceSize offset = offsets != VMA_NULL ? offsets[allocIndex] : 0;
        const VkDeviceSize size = sizes != VMA_NULL ? sizes[allocIndex] : VK_WHOLE_SIZE;
        VkMappedMemoryRange newRange;
        if(get_flush_or_invalidate_range(alloc, offset, size, newRange))
        {
            ranges.push_back(newRange);
        }
    }

    VkResult res = VK_SUCCESS;
    if(!ranges.empty())
    {
        switch(op)
        {
        case VMA_CACHE_FLUSH:
            res = (*get_vulkan_functions().vkFlushMappedMemoryRanges)(_h_device, (uint32_t)ranges.size(), ranges.data());
            break;
        case VMA_CACHE_INVALIDATE:
            res = (*get_vulkan_functions().vkInvalidateMappedMemoryRanges)(_h_device, (uint32_t)ranges.size(), ranges.data());
            break;
        default:
            VMA_ASSERT(0);
        }
    }
    // else: Just ignore this call.
    return res;
}

VkResult VmaAllocator_T::copy_memory_to_allocation(
    const void* pSrcHostPointer,
    VmaAllocation dstAllocation,
    VkDeviceSize dstAllocationLocalOffset,
    VkDeviceSize size)
{
    void* dstMappedData = VMA_NULL;
    VkResult res = Map(dstAllocation, &dstMappedData);
    if(res == VK_SUCCESS)
    {
        memcpy((char*)dstMappedData + dstAllocationLocalOffset, pSrcHostPointer, (size_t)size);
        unmap(dstAllocation);
        res = flush_or_invalidate_allocation(dstAllocation, dstAllocationLocalOffset, size, VMA_CACHE_FLUSH);
    }
    return res;
}

VkResult VmaAllocator_T::copy_allocation_to_memory(
    VmaAllocation srcAllocation,
    VkDeviceSize srcAllocationLocalOffset,
    void* pDstHostPointer,
    VkDeviceSize size)
{
    void* srcMappedData = VMA_NULL;
    VkResult res = Map(srcAllocation, &srcMappedData);
    if(res == VK_SUCCESS)
    {
        res = flush_or_invalidate_allocation(srcAllocation, srcAllocationLocalOffset, size, VMA_CACHE_INVALIDATE);
        if(res == VK_SUCCESS)
        {
            memcpy(pDstHostPointer, (const char*)srcMappedData + srcAllocationLocalOffset, (size_t)size);
            unmap(srcAllocation);
        }
    }
    return res;
}

void VmaAllocator_T::free_dedicated_memory(VmaAllocation allocation)
{
    VMA_ASSERT(allocation && allocation->get_type() == VmaAllocation_T::ALLOCATION_TYPE_DEDICATED);

    const uint32_t memTypeIndex = allocation->get_memory_type_index();
    VmaPool parentPool = allocation->get_parent_pool();
    if(parentPool == VK_NULL_HANDLE)
    {
        // Default pool
        _dedicated_allocations[memTypeIndex].unregister_allocation(allocation);
    }
    else
    {
        // Custom pool
        parentPool->_dedicated_allocations.unregister_allocation(allocation);
    }

    VkDeviceMemory hMemory = allocation->get_memory();

    /*
    There is no need to call this, because Vulkan spec allows to skip vkUnmapMemory
    before vkFreeMemory.

    if(allocation->get_mapped_data() != VMA_NULL)
    {
        (*_vulkan_functions.vkUnmapMemory)(_h_device, hMemory);
    }
    */

    free_vulkan_memory(memTypeIndex, allocation->get_size(), hMemory);

    _budget.remove_allocation(memory_type_index_to_heap_index(allocation->get_memory_type_index()), allocation->get_size());
    allocation->destroy(this);
    _allocation_object_allocator.free(allocation);

    VMA_DEBUG_LOG_FORMAT("    Freed DedicatedMemory MemoryTypeIndex=%" PRIu32, memTypeIndex);
}

uint32_t VmaAllocator_T::calculate_gpu_defragmentation_memory_type_bits() const
{
    VkBufferCreateInfo dummyBufCreateInfo;
    VmaFillGpuDefragmentationBufferCreateInfo(dummyBufCreateInfo);

    uint32_t memoryTypeBits = 0;

    // Create buffer.
    VkBuffer buf = VK_NULL_HANDLE;
    VkResult res = (*get_vulkan_functions().vkCreateBuffer)(
        _h_device, &dummyBufCreateInfo, get_allocation_callbacks(), &buf);
    if(res == VK_SUCCESS)
    {
        // Query for supported memory types.
        VkMemoryRequirements memReq;
        (*get_vulkan_functions().vkGetBufferMemoryRequirements)(_h_device, buf, &memReq);
        memoryTypeBits = memReq.memoryTypeBits;

        // destroy buffer.
        (*get_vulkan_functions().vkDestroyBuffer)(_h_device, buf, get_allocation_callbacks());
    }

    return memoryTypeBits;
}

uint32_t VmaAllocator_T::calculate_global_memory_type_bits() const
{
    // Make sure memory information is already fetched.
    VMA_ASSERT(get_memory_type_count() > 0);

    uint32_t memoryTypeBits = UINT32_MAX;

    if(!_use_amd_device_coherent_memory)
    {
        // Exclude memory types that have VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD.
        for(uint32_t memTypeIndex = 0; memTypeIndex < get_memory_type_count(); ++memTypeIndex)
        {
            if((_mem_props.memoryTypes[memTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD_COPY) != 0)
            {
                memoryTypeBits &= ~(1U << memTypeIndex);
            }
        }
    }

    return memoryTypeBits;
}

bool VmaAllocator_T::get_flush_or_invalidate_range(
    VmaAllocation allocation,
    VkDeviceSize offset, VkDeviceSize size,
    VkMappedMemoryRange& outRange) const
{
    const uint32_t memTypeIndex = allocation->get_memory_type_index();
    if(size > 0 && is_memory_type_non_coherent(memTypeIndex))
    {
        const VkDeviceSize nonCoherentAtomSize = _physical_device_properties.limits.nonCoherentAtomSize;
        const VkDeviceSize allocationSize = allocation->get_size();
        VMA_ASSERT(offset <= allocationSize);

        outRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        outRange.pNext = VMA_NULL;
        outRange.memory = allocation->get_memory();

        switch(allocation->get_type())
        {
        case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
            outRange.offset = VmaAlignDown(offset, nonCoherentAtomSize);
            if(size == VK_WHOLE_SIZE)
            {
                outRange.size = allocationSize - outRange.offset;
            }
            else
            {
                VMA_ASSERT(offset + size <= allocationSize);
                outRange.size = VMA_MIN(
                    VmaAlignUp(size + (offset - outRange.offset), nonCoherentAtomSize),
                    allocationSize - outRange.offset);
            }
            break;
        case VmaAllocation_T::ALLOCATION_TYPE_BLOCK:
        {
            // 1. Still within this allocation.
            outRange.offset = VmaAlignDown(offset, nonCoherentAtomSize);
            if(size == VK_WHOLE_SIZE)
            {
                size = allocationSize - offset;
            }
            else
            {
                VMA_ASSERT(offset + size <= allocationSize);
            }
            outRange.size = VmaAlignUp(size + (offset - outRange.offset), nonCoherentAtomSize);

            // 2. Adjust to whole block.
            const VkDeviceSize allocationOffset = allocation->get_offset();
            VMA_ASSERT(allocationOffset % nonCoherentAtomSize == 0);
            const VkDeviceSize blockSize = allocation->get_block()->_p_metadata->get_size();
            outRange.offset += allocationOffset;
            outRange.size = VMA_MIN(outRange.size, blockSize - outRange.offset);

            break;
        }
        default:
            VMA_ASSERT(0);
        }
        return true;
    }
    return false;
}

#if VMA_MEMORY_BUDGET
void VmaAllocator_T::update_vulkan_budget()
{
    VMA_ASSERT(_use_ext_memory_budget);

    VkPhysicalDeviceMemoryProperties2KHR memProps = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR };

    VkPhysicalDeviceMemoryBudgetPropertiesEXT budgetProps = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT };
    VmaPnextChainPushFront(&memProps, &budgetProps);

    get_vulkan_functions().vkGetPhysicalDeviceMemoryProperties2KHR(_physical_device, &memProps);

    {
        VmaMutexLockWrite lockWrite(_budget._budget_mutex, _use_mutex);

        for(uint32_t heapIndex = 0; heapIndex < get_memory_heap_count(); ++heapIndex)
        {
            _budget._vulkan_usage[heapIndex] = budgetProps.heapUsage[heapIndex];
            _budget._vulkan_budget[heapIndex] = budgetProps.heapBudget[heapIndex];
            _budget._block_bytes_at_budget_fetch[heapIndex] = _budget._block_bytes[heapIndex].load();

            // Some bugged drivers return the budget incorrectly, e.g. 0 or much bigger than heap size.
            if(_budget._vulkan_budget[heapIndex] == 0)
            {
                _budget._vulkan_budget[heapIndex] = _mem_props.memoryHeaps[heapIndex].size * 8 / 10; // 80% heuristics.
            }
            else if(_budget._vulkan_budget[heapIndex] > _mem_props.memoryHeaps[heapIndex].size)
            {
                _budget._vulkan_budget[heapIndex] = _mem_props.memoryHeaps[heapIndex].size;
            }
            if(_budget._vulkan_usage[heapIndex] == 0 && _budget._block_bytes_at_budget_fetch[heapIndex] > 0)
            {
                _budget._vulkan_usage[heapIndex] = _budget._block_bytes_at_budget_fetch[heapIndex];
            }
        }
        _budget._operations_since_budget_fetch = 0;
    }
}
#endif // VMA_MEMORY_BUDGET

void VmaAllocator_T::fill_allocation(VmaAllocation hAllocation, uint8_t pattern)
{
    if(VMA_DEBUG_INITIALIZE_ALLOCATIONS &&
        hAllocation->is_mapping_allowed() &&
        (_mem_props.memoryTypes[hAllocation->get_memory_type_index()].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0)
    {
        void* pData = VMA_NULL;
        VkResult res = Map(hAllocation, &pData);
        if(res == VK_SUCCESS)
        {
            memset(pData, (int)pattern, (size_t)hAllocation->get_size());
            flush_or_invalidate_allocation(hAllocation, 0, VK_WHOLE_SIZE, VMA_CACHE_FLUSH);
            unmap(hAllocation);
        }
        else
        {
            VMA_ASSERT(0 && "VMA_DEBUG_INITIALIZE_ALLOCATIONS is enabled, but couldn't map memory to fill allocation.");
        }
    }
}

uint32_t VmaAllocator_T::get_gpu_defragmentation_memory_type_bits()
{
    uint32_t memoryTypeBits = _gpu_defragmentation_memory_type_bits.load();
    if(memoryTypeBits == UINT32_MAX)
    {
        memoryTypeBits = calculate_gpu_defragmentation_memory_type_bits();
        _gpu_defragmentation_memory_type_bits.store(memoryTypeBits);
    }
    return memoryTypeBits;
}

#if VMA_STATS_STRING_ENABLED
void VmaAllocator_T::print_detailed_map(VmaJsonWriter& json)
{
    json.write_string("DefaultPools");
    json.begin_object();
    {
        for (uint32_t memTypeIndex = 0; memTypeIndex < get_memory_type_count(); ++memTypeIndex)
        {
            VmaBlockVector* pBlockVector = _p_block_vectors[memTypeIndex];
            VmaDedicatedAllocationList& dedicatedAllocList = _dedicated_allocations[memTypeIndex];
            if (pBlockVector != VMA_NULL)
            {
                json.begin_string("Type ");
                json.continue_string(memTypeIndex);
                json.end_string();
                json.begin_object();
                {
                    json.write_string("PreferredBlockSize");
                    json.write_number(pBlockVector->get_preferred_block_size());

                    json.write_string("Blocks");
                    pBlockVector->print_detailed_map(json);

                    json.write_string("DedicatedAllocations");
                    dedicatedAllocList.build_stats_string(json);
                }
                json.end_object();
            }
        }
    }
    json.end_object();

    json.write_string("CustomPools");
    json.begin_object();
    {
        VmaMutexLockRead lock(_pools_mutex, _use_mutex);
        if (!_pools.is_empty())
        {
            for (uint32_t memTypeIndex = 0; memTypeIndex < get_memory_type_count(); ++memTypeIndex)
            {
                bool displayType = true;
                size_t index = 0;
                for (VmaPool pool = _pools.front(); pool != VMA_NULL; pool = _pools.get_next(pool))
                {
                    VmaBlockVector& blockVector = pool->_block_vector;
                    if (blockVector.get_memory_type_index() == memTypeIndex)
                    {
                        if (displayType)
                        {
                            json.begin_string("Type ");
                            json.continue_string(memTypeIndex);
                            json.end_string();
                            json.begin_array();
                            displayType = false;
                        }

                        json.begin_object();
                        {
                            json.write_string("Name");
                            json.begin_string();
                            json.continue_string((uint64_t)index++);
                            if (pool->get_name())
                            {
                                json.continue_string(" - ");
                                json.continue_string(pool->get_name());
                            }
                            json.end_string();

                            json.write_string("PreferredBlockSize");
                            json.write_number(blockVector.get_preferred_block_size());

                            json.write_string("Blocks");
                            blockVector.print_detailed_map(json);

                            json.write_string("DedicatedAllocations");
                            pool->_dedicated_allocations.build_stats_string(json);
                        }
                        json.end_object();
                    }
                }

                if (!displayType)
                    json.end_array();
            }
        }
    }
    json.end_object();
}
#endif // VMA_STATS_STRING_ENABLED
#endif // _VMA_ALLOCATOR_T_FUNCTIONS


#ifndef _VMA_PUBLIC_INTERFACE

#ifdef VOLK_HEADER_VERSION

VMA_CALL_PRE VkResult VMA_CALL_POST vmaImportVulkanFunctionsFromVolk(
    const VmaAllocatorCreateInfo* VMA_NOT_NULL pAllocatorCreateInfo,
    VmaVulkanFunctions* VMA_NOT_NULL pDstVulkanFunctions)
{
    VMA_ASSERT(pAllocatorCreateInfo != VMA_NULL);
    VMA_ASSERT(pAllocatorCreateInfo->instance != VK_NULL_HANDLE);
    VMA_ASSERT(pAllocatorCreateInfo->device != VK_NULL_HANDLE);

    memset(pDstVulkanFunctions, 0, sizeof(*pDstVulkanFunctions));
    
    VolkDeviceTable src = {};
    volkLoadDeviceTable(&src, pAllocatorCreateInfo->device);

#define COPY_GLOBAL_TO_VMA_FUNC(volkName, vmaName) if(!pDstVulkanFunctions->vmaName) pDstVulkanFunctions->vmaName = volkName;
#define COPY_DEVICE_TO_VMA_FUNC(volkName, vmaName) if(!pDstVulkanFunctions->vmaName) pDstVulkanFunctions->vmaName = src.volkName;

    COPY_GLOBAL_TO_VMA_FUNC(vkGetInstanceProcAddr, vkGetInstanceProcAddr)
    COPY_GLOBAL_TO_VMA_FUNC(vkGetDeviceProcAddr, vkGetDeviceProcAddr)
    COPY_GLOBAL_TO_VMA_FUNC(vkGetPhysicalDeviceProperties, vkGetPhysicalDeviceProperties)
    COPY_GLOBAL_TO_VMA_FUNC(vkGetPhysicalDeviceMemoryProperties, vkGetPhysicalDeviceMemoryProperties)
    COPY_DEVICE_TO_VMA_FUNC(vkAllocateMemory, vkAllocateMemory)
    COPY_DEVICE_TO_VMA_FUNC(vkFreeMemory, vkFreeMemory)
    COPY_DEVICE_TO_VMA_FUNC(vkMapMemory, vkMapMemory)
    COPY_DEVICE_TO_VMA_FUNC(vkUnmapMemory, vkUnmapMemory)
    COPY_DEVICE_TO_VMA_FUNC(vkFlushMappedMemoryRanges, vkFlushMappedMemoryRanges)
    COPY_DEVICE_TO_VMA_FUNC(vkInvalidateMappedMemoryRanges, vkInvalidateMappedMemoryRanges)
    COPY_DEVICE_TO_VMA_FUNC(vkBindBufferMemory, vkBindBufferMemory)
    COPY_DEVICE_TO_VMA_FUNC(vkBindImageMemory, vkBindImageMemory)
    COPY_DEVICE_TO_VMA_FUNC(vkGetBufferMemoryRequirements, vkGetBufferMemoryRequirements)
    COPY_DEVICE_TO_VMA_FUNC(vkGetImageMemoryRequirements, vkGetImageMemoryRequirements)
    COPY_DEVICE_TO_VMA_FUNC(vkCreateBuffer, vkCreateBuffer)
    COPY_DEVICE_TO_VMA_FUNC(vkDestroyBuffer, vkDestroyBuffer)
    COPY_DEVICE_TO_VMA_FUNC(vkCreateImage, vkCreateImage)
    COPY_DEVICE_TO_VMA_FUNC(vkDestroyImage, vkDestroyImage)
    COPY_DEVICE_TO_VMA_FUNC(vkCmdCopyBuffer, vkCmdCopyBuffer)
#if VMA_VULKAN_VERSION >= 1001000
    if (pAllocatorCreateInfo->vulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        COPY_GLOBAL_TO_VMA_FUNC(vkGetPhysicalDeviceMemoryProperties2, vkGetPhysicalDeviceMemoryProperties2KHR)
        COPY_DEVICE_TO_VMA_FUNC(vkGetBufferMemoryRequirements2, vkGetBufferMemoryRequirements2KHR)
        COPY_DEVICE_TO_VMA_FUNC(vkGetImageMemoryRequirements2, vkGetImageMemoryRequirements2KHR)
        COPY_DEVICE_TO_VMA_FUNC(vkBindBufferMemory2, vkBindBufferMemory2KHR)
        COPY_DEVICE_TO_VMA_FUNC(vkBindImageMemory2, vkBindImageMemory2KHR)
    }
#endif
#if VMA_VULKAN_VERSION >= 1003000
    if (pAllocatorCreateInfo->vulkanApiVersion >= VK_MAKE_VERSION(1, 3, 0))
    {
        COPY_DEVICE_TO_VMA_FUNC(vkGetDeviceBufferMemoryRequirements, vkGetDeviceBufferMemoryRequirements)
        COPY_DEVICE_TO_VMA_FUNC(vkGetDeviceImageMemoryRequirements, vkGetDeviceImageMemoryRequirements)
    }
#endif
#if VMA_KHR_MAINTENANCE4
    if((pAllocatorCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT) != 0)
    {
        COPY_DEVICE_TO_VMA_FUNC(vkGetDeviceBufferMemoryRequirementsKHR, vkGetDeviceBufferMemoryRequirements)
        COPY_DEVICE_TO_VMA_FUNC(vkGetDeviceImageMemoryRequirementsKHR, vkGetDeviceImageMemoryRequirements)
    }
#endif
#if VMA_DEDICATED_ALLOCATION
    if ((pAllocatorCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT) != 0)
    {
        COPY_DEVICE_TO_VMA_FUNC(vkGetBufferMemoryRequirements2KHR, vkGetBufferMemoryRequirements2KHR)
        COPY_DEVICE_TO_VMA_FUNC(vkGetImageMemoryRequirements2KHR, vkGetImageMemoryRequirements2KHR)
    }
#endif
#if VMA_BIND_MEMORY2
    if ((pAllocatorCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT) != 0)
    {
        COPY_DEVICE_TO_VMA_FUNC(vkBindBufferMemory2KHR, vkBindBufferMemory2KHR)
        COPY_DEVICE_TO_VMA_FUNC(vkBindImageMemory2KHR, vkBindImageMemory2KHR)
    }
#endif
#if VMA_MEMORY_BUDGET
    if ((pAllocatorCreateInfo->flags & VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT) != 0)
    {
        COPY_GLOBAL_TO_VMA_FUNC(vkGetPhysicalDeviceMemoryProperties2KHR, vkGetPhysicalDeviceMemoryProperties2KHR)
    }
#endif
#if VMA_EXTERNAL_MEMORY_WIN32
    if ((pAllocatorCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_EXTERNAL_MEMORY_WIN32_BIT) != 0)
    {
        COPY_DEVICE_TO_VMA_FUNC(vkGetMemoryWin32HandleKHR, vkGetMemoryWin32HandleKHR)
    }
#endif

#undef COPY_DEVICE_TO_VMA_FUNC
#undef COPY_GLOBAL_TO_VMA_FUNC

    return VK_SUCCESS;
}

#endif // #ifdef VOLK_HEADER_VERSION

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAllocator(
    const VmaAllocatorCreateInfo* pCreateInfo,
    VmaAllocator* pAllocator)
{
    VMA_ASSERT(pCreateInfo && pAllocator);
    VMA_ASSERT(pCreateInfo->vulkanApiVersion == 0 ||
        (VK_VERSION_MAJOR(pCreateInfo->vulkanApiVersion) == 1 && VK_VERSION_MINOR(pCreateInfo->vulkanApiVersion) <= 4));
    VMA_DEBUG_LOG("vmaCreateAllocator");
    *pAllocator = vma_new(pCreateInfo->pAllocationCallbacks, VmaAllocator_T)(pCreateInfo);
    VkResult result = (*pAllocator)->init(pCreateInfo);
    if(result < 0)
    {
        vma_delete(pCreateInfo->pAllocationCallbacks, *pAllocator);
        *pAllocator = VK_NULL_HANDLE;
    }
    return result;
}

VMA_CALL_PRE void VMA_CALL_POST vmaDestroyAllocator(
    VmaAllocator allocator)
{
    if(allocator != VK_NULL_HANDLE)
    {
        VMA_DEBUG_LOG("vmaDestroyAllocator");
        VkAllocationCallbacks allocationCallbacks = allocator->_allocation_callbacks; // Have to copy the callbacks when destroying.
        vma_delete(&allocationCallbacks, allocator);
    }
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetAllocatorInfo(VmaAllocator allocator, VmaAllocatorInfo* pAllocatorInfo)
{
    VMA_ASSERT(allocator && pAllocatorInfo);
    pAllocatorInfo->instance = allocator->_h_instance;
    pAllocatorInfo->physicalDevice = allocator->get_physical_device();
    pAllocatorInfo->device = allocator->_h_device;
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetPhysicalDeviceProperties(
    VmaAllocator allocator,
    const VkPhysicalDeviceProperties **ppPhysicalDeviceProperties)
{
    VMA_ASSERT(allocator && ppPhysicalDeviceProperties);
    *ppPhysicalDeviceProperties = &allocator->_physical_device_properties;
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetMemoryProperties(
    VmaAllocator allocator,
    const VkPhysicalDeviceMemoryProperties** ppPhysicalDeviceMemoryProperties)
{
    VMA_ASSERT(allocator && ppPhysicalDeviceMemoryProperties);
    *ppPhysicalDeviceMemoryProperties = &allocator->_mem_props;
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetMemoryTypeProperties(
    VmaAllocator allocator,
    uint32_t memoryTypeIndex,
    VkMemoryPropertyFlags* pFlags)
{
    VMA_ASSERT(allocator && pFlags);
    VMA_ASSERT(memoryTypeIndex < allocator->get_memory_type_count());
    *pFlags = allocator->_mem_props.memoryTypes[memoryTypeIndex].propertyFlags;
}

VMA_CALL_PRE void VMA_CALL_POST vmaSetCurrentFrameIndex(
    VmaAllocator allocator,
    uint32_t frameIndex)
{
    VMA_ASSERT(allocator);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->set_current_frame_index(frameIndex);
}

VMA_CALL_PRE void VMA_CALL_POST vmaCalculateStatistics(
    VmaAllocator allocator,
    VmaTotalStatistics* pStats)
{
    VMA_ASSERT(allocator && pStats);
    VMA_DEBUG_GLOBAL_MUTEX_LOCK
    allocator->calculate_statistics(pStats);
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetHeapBudgets(
    VmaAllocator allocator,
    VmaBudget* pBudgets)
{
    VMA_ASSERT(allocator && pBudgets);
    VMA_DEBUG_GLOBAL_MUTEX_LOCK
    allocator->get_heap_budgets(pBudgets, 0, allocator->get_memory_heap_count());
}

#if VMA_STATS_STRING_ENABLED

VMA_CALL_PRE void VMA_CALL_POST vmaBuildStatsString(
    VmaAllocator allocator,
    char** ppStatsString,
    VkBool32 detailedMap)
{
    VMA_ASSERT(allocator && ppStatsString);
    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    VmaStringBuilder sb(allocator->get_allocation_callbacks());
    {
        VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
        allocator->get_heap_budgets(budgets, 0, allocator->get_memory_heap_count());

        VmaTotalStatistics stats;
        allocator->calculate_statistics(&stats);

        VmaJsonWriter json(allocator->get_allocation_callbacks(), sb);
        json.begin_object();
        {
            json.write_string("General");
            json.begin_object();
            {
                const VkPhysicalDeviceProperties& deviceProperties = allocator->_physical_device_properties;
                const VkPhysicalDeviceMemoryProperties& memoryProperties = allocator->_mem_props;

                json.write_string("API");
                json.write_string("Vulkan");

                json.write_string("apiVersion");
                json.begin_string();
                json.continue_string(VK_VERSION_MAJOR(deviceProperties.apiVersion));
                json.continue_string(".");
                json.continue_string(VK_VERSION_MINOR(deviceProperties.apiVersion));
                json.continue_string(".");
                json.continue_string(VK_VERSION_PATCH(deviceProperties.apiVersion));
                json.end_string();

                json.write_string("GPU");
                json.write_string(deviceProperties.deviceName);
                json.write_string("deviceType");
                json.write_number(static_cast<uint32_t>(deviceProperties.deviceType));

                json.write_string("maxMemoryAllocationCount");
                json.write_number(deviceProperties.limits.maxMemoryAllocationCount);
                json.write_string("bufferImageGranularity");
                json.write_number(deviceProperties.limits.bufferImageGranularity);
                json.write_string("nonCoherentAtomSize");
                json.write_number(deviceProperties.limits.nonCoherentAtomSize);

                json.write_string("memoryHeapCount");
                json.write_number(memoryProperties.memoryHeapCount);
                json.write_string("memoryTypeCount");
                json.write_number(memoryProperties.memoryTypeCount);
            }
            json.end_object();
        }
        {
            json.write_string("Total");
            VmaPrintDetailedStatistics(json, stats.total);
        }
        {
            json.write_string("MemoryInfo");
            json.begin_object();
            {
                for (uint32_t heapIndex = 0; heapIndex < allocator->get_memory_heap_count(); ++heapIndex)
                {
                    json.begin_string("Heap ");
                    json.continue_string(heapIndex);
                    json.end_string();
                    json.begin_object();
                    {
                        const VkMemoryHeap& heapInfo = allocator->_mem_props.memoryHeaps[heapIndex];
                        json.write_string("Flags");
                        json.begin_array(true);
                        {
                            if (heapInfo.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                                json.write_string("DEVICE_LOCAL");
                        #if VMA_VULKAN_VERSION >= 1001000
                            if (heapInfo.flags & VK_MEMORY_HEAP_MULTI_INSTANCE_BIT)
                                json.write_string("MULTI_INSTANCE");
                        #endif

                            VkMemoryHeapFlags flags = heapInfo.flags &
                                ~(VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
                        #if VMA_VULKAN_VERSION >= 1001000
                                    | VK_MEMORY_HEAP_MULTI_INSTANCE_BIT
                        #endif
                                    );
                            if (flags != 0)
                                json.write_number(flags);
                        }
                        json.end_array();

                        json.write_string("Size");
                        json.write_number(heapInfo.size);

                        json.write_string("Budget");
                        json.begin_object();
                        {
                            json.write_string("BudgetBytes");
                            json.write_number(budgets[heapIndex].budget);
                            json.write_string("UsageBytes");
                            json.write_number(budgets[heapIndex].usage);
                        }
                        json.end_object();

                        json.write_string("Stats");
                        VmaPrintDetailedStatistics(json, stats.memoryHeap[heapIndex]);

                        json.write_string("MemoryPools");
                        json.begin_object();
                        {
                            for (uint32_t typeIndex = 0; typeIndex < allocator->get_memory_type_count(); ++typeIndex)
                            {
                                if (allocator->memory_type_index_to_heap_index(typeIndex) == heapIndex)
                                {
                                    json.begin_string("Type ");
                                    json.continue_string(typeIndex);
                                    json.end_string();
                                    json.begin_object();
                                    {
                                        json.write_string("Flags");
                                        json.begin_array(true);
                                        {
                                            VkMemoryPropertyFlags flags = allocator->_mem_props.memoryTypes[typeIndex].propertyFlags;
                                            if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                                                json.write_string("DEVICE_LOCAL");
                                            if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
                                                json.write_string("HOST_VISIBLE");
                                            if (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                                                json.write_string("HOST_COHERENT");
                                            if (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
                                                json.write_string("HOST_CACHED");
                                            if (flags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT)
                                                json.write_string("LAZILY_ALLOCATED");
                                        #if VMA_VULKAN_VERSION >= 1001000
                                            if (flags & VK_MEMORY_PROPERTY_PROTECTED_BIT)
                                                json.write_string("PROTECTED");
                                        #endif
                                        #if VK_AMD_device_coherent_memory
                                            if (flags & VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD_COPY)
                                                json.write_string("DEVICE_COHERENT_AMD");
                                            if (flags & VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD_COPY)
                                                json.write_string("DEVICE_UNCACHED_AMD");
                                        #endif

                                            flags &= ~(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                                        #if VMA_VULKAN_VERSION >= 1001000
                                                | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT
                                        #endif
                                        #if VK_AMD_device_coherent_memory
                                                | VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD_COPY
                                                | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD_COPY
                                        #endif
                                                | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                                | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
                                            if (flags != 0)
                                                json.write_number(flags);
                                        }
                                        json.end_array();

                                        json.write_string("Stats");
                                        VmaPrintDetailedStatistics(json, stats.memoryType[typeIndex]);
                                    }
                                    json.end_object();
                                }
                            }

                        }
                        json.end_object();
                    }
                    json.end_object();
                }
            }
            json.end_object();
        }

        if (detailedMap == VK_TRUE)
            allocator->print_detailed_map(json);

        json.end_object();
    }

    *ppStatsString = VmaCreateStringCopy(allocator->get_allocation_callbacks(), sb.get_data(), sb.get_length());
}

VMA_CALL_PRE void VMA_CALL_POST vmaFreeStatsString(
    VmaAllocator allocator,
    char* pStatsString)
{
    if(pStatsString != VMA_NULL)
    {
        VMA_ASSERT(allocator);
        VmaFreeString(allocator->get_allocation_callbacks(), pStatsString);
    }
}

#endif // VMA_STATS_STRING_ENABLED

/*
This function is not protected by any mutex because it just reads immutable data.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaFindMemoryTypeIndex(
    VmaAllocator allocator,
    uint32_t memoryTypeBits,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    uint32_t* pMemoryTypeIndex)
{
    VMA_ASSERT(allocator != VK_NULL_HANDLE);
    VMA_ASSERT(pAllocationCreateInfo != VMA_NULL);
    VMA_ASSERT(pMemoryTypeIndex != VMA_NULL);

    return allocator->find_memory_type_index(memoryTypeBits, pAllocationCreateInfo, VmaBufferImageUsage::UNKNOWN, pMemoryTypeIndex);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaFindMemoryTypeIndexForBufferInfo(
    VmaAllocator allocator,
    const VkBufferCreateInfo* pBufferCreateInfo,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    uint32_t* pMemoryTypeIndex)
{
    VMA_ASSERT(allocator != VK_NULL_HANDLE);
    VMA_ASSERT(pBufferCreateInfo != VMA_NULL);
    VMA_ASSERT(pAllocationCreateInfo != VMA_NULL);
    VMA_ASSERT(pMemoryTypeIndex != VMA_NULL);

    const VkDevice hDev = allocator->_h_device;
    const VmaVulkanFunctions* funcs = &allocator->get_vulkan_functions();
    VkResult res = VK_SUCCESS;

#if VMA_KHR_MAINTENANCE4 || VMA_VULKAN_VERSION >= 1003000
    if(funcs->vkGetDeviceBufferMemoryRequirements)
    {
        // Can query straight from VkBufferCreateInfo :)
        VkDeviceBufferMemoryRequirementsKHR devBufMemReq = {VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS_KHR};
        devBufMemReq.pCreateInfo = pBufferCreateInfo;

        VkMemoryRequirements2 memReq = {VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
        (*funcs->vkGetDeviceBufferMemoryRequirements)(hDev, &devBufMemReq, &memReq);

        res = allocator->find_memory_type_index(
            memReq.memoryRequirements.memoryTypeBits, pAllocationCreateInfo,
            VmaBufferImageUsage(*pBufferCreateInfo, allocator->_use_khr_maintenance5), pMemoryTypeIndex);
    }
    else
#endif // VMA_KHR_MAINTENANCE4 || VMA_VULKAN_VERSION >= 1003000
    {
        // Must create a dummy buffer to query :(
        VkBuffer hBuffer = VK_NULL_HANDLE;
        res = funcs->vkCreateBuffer(
            hDev, pBufferCreateInfo, allocator->get_allocation_callbacks(), &hBuffer);
        if(res == VK_SUCCESS)
        {
            VkMemoryRequirements memReq = {};
            funcs->vkGetBufferMemoryRequirements(hDev, hBuffer, &memReq);

            res = allocator->find_memory_type_index(
                memReq.memoryTypeBits, pAllocationCreateInfo,
                VmaBufferImageUsage(*pBufferCreateInfo, allocator->_use_khr_maintenance5), pMemoryTypeIndex);

            funcs->vkDestroyBuffer(
                hDev, hBuffer, allocator->get_allocation_callbacks());
        }
    }
    return res;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaFindMemoryTypeIndexForImageInfo(
    VmaAllocator allocator,
    const VkImageCreateInfo* pImageCreateInfo,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    uint32_t* pMemoryTypeIndex)
{
    VMA_ASSERT(allocator != VK_NULL_HANDLE);
    VMA_ASSERT(pImageCreateInfo != VMA_NULL);
    VMA_ASSERT(pAllocationCreateInfo != VMA_NULL);
    VMA_ASSERT(pMemoryTypeIndex != VMA_NULL);

    const VkDevice hDev = allocator->_h_device;
    const VmaVulkanFunctions* funcs = &allocator->get_vulkan_functions();
    VkResult res = VK_SUCCESS;

#if VMA_KHR_MAINTENANCE4 || VMA_VULKAN_VERSION >= 1003000
    if(funcs->vkGetDeviceImageMemoryRequirements)
    {
        // Can query straight from VkImageCreateInfo :)
        VkDeviceImageMemoryRequirementsKHR devImgMemReq = {VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS_KHR};
        devImgMemReq.pCreateInfo = pImageCreateInfo;
        VMA_ASSERT(pImageCreateInfo->tiling != VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT_COPY && (pImageCreateInfo->flags & VK_IMAGE_CREATE_DISJOINT_BIT_COPY) == 0 &&
            "Cannot use this VkImageCreateInfo with vmaFindMemoryTypeIndexForImageInfo as I don't know what to pass as VkDeviceImageMemoryRequirements::planeAspect.");

        VkMemoryRequirements2 memReq = {VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
        (*funcs->vkGetDeviceImageMemoryRequirements)(hDev, &devImgMemReq, &memReq);

        res = allocator->find_memory_type_index(
            memReq.memoryRequirements.memoryTypeBits, pAllocationCreateInfo,
            VmaBufferImageUsage(*pImageCreateInfo), pMemoryTypeIndex);
    }
    else
#endif // VMA_KHR_MAINTENANCE4 || VMA_VULKAN_VERSION >= 1003000
    {
        // Must create a dummy image to query :(
        VkImage hImage = VK_NULL_HANDLE;
        res = funcs->vkCreateImage(
            hDev, pImageCreateInfo, allocator->get_allocation_callbacks(), &hImage);
        if(res == VK_SUCCESS)
        {
            VkMemoryRequirements memReq = {};
            funcs->vkGetImageMemoryRequirements(hDev, hImage, &memReq);

            res = allocator->find_memory_type_index(
                memReq.memoryTypeBits, pAllocationCreateInfo,
                VmaBufferImageUsage(*pImageCreateInfo), pMemoryTypeIndex);

            funcs->vkDestroyImage(
                hDev, hImage, allocator->get_allocation_callbacks());
        }
    }
    return res;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreatePool(
    VmaAllocator allocator,
    const VmaPoolCreateInfo* pCreateInfo,
    VmaPool* pPool)
{
    VMA_ASSERT(allocator && pCreateInfo && pPool);

    VMA_DEBUG_LOG("vmaCreatePool");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->create_pool(pCreateInfo, pPool);
}

VMA_CALL_PRE void VMA_CALL_POST vmaDestroyPool(
    VmaAllocator allocator,
    VmaPool pool)
{
    VMA_ASSERT(allocator);

    if(pool == VK_NULL_HANDLE)
    {
        return;
    }

    VMA_DEBUG_LOG("vmaDestroyPool");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->destroy_pool(pool);
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetPoolStatistics(
    VmaAllocator allocator,
    VmaPool pool,
    VmaStatistics* pPoolStats)
{
    VMA_ASSERT(allocator && pool && pPoolStats);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->get_pool_statistics(pool, pPoolStats);
}

VMA_CALL_PRE void VMA_CALL_POST vmaCalculatePoolStatistics(
    VmaAllocator allocator,
    VmaPool pool,
    VmaDetailedStatistics* pPoolStats)
{
    VMA_ASSERT(allocator && pool && pPoolStats);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->calculate_pool_statistics(pool, pPoolStats);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCheckPoolCorruption(VmaAllocator allocator, VmaPool pool)
{
    VMA_ASSERT(allocator && pool);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    VMA_DEBUG_LOG("vmaCheckPoolCorruption");

    return allocator->check_pool_corruption(pool);
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetPoolName(
    VmaAllocator allocator,
    VmaPool pool,
    const char** ppName)
{
    VMA_ASSERT(allocator && pool && ppName);

    VMA_DEBUG_LOG("vmaGetPoolName");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    *ppName = pool->get_name();
}

VMA_CALL_PRE void VMA_CALL_POST vmaSetPoolName(
    VmaAllocator allocator,
    VmaPool pool,
    const char* pName)
{
    VMA_ASSERT(allocator && pool);

    VMA_DEBUG_LOG("vmaSetPoolName");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    pool->set_name(pName);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemory(
    VmaAllocator allocator,
    const VkMemoryRequirements* pVkMemoryRequirements,
    const VmaAllocationCreateInfo* pCreateInfo,
    VmaAllocation* pAllocation,
    VmaAllocationInfo* pAllocationInfo)
{
    VMA_ASSERT(allocator && pVkMemoryRequirements && pCreateInfo && pAllocation);

    VMA_DEBUG_LOG("vmaAllocateMemory");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    VkResult result = allocator->allocate_memory(
        *pVkMemoryRequirements,
        false, // requiresDedicatedAllocation
        false, // prefersDedicatedAllocation
        VK_NULL_HANDLE, // dedicatedBuffer
        VK_NULL_HANDLE, // dedicatedImage
        VmaBufferImageUsage::UNKNOWN, // dedicatedBufferImageUsage
        *pCreateInfo,
        VMA_SUBALLOCATION_TYPE_UNKNOWN,
        1, // allocationCount
        pAllocation);

    if(pAllocationInfo != VMA_NULL && result == VK_SUCCESS)
    {
        allocator->get_allocation_info(*pAllocation, pAllocationInfo);
    }

    return result;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemoryPages(
    VmaAllocator allocator,
    const VkMemoryRequirements* pVkMemoryRequirements,
    const VmaAllocationCreateInfo* pCreateInfo,
    size_t allocationCount,
    VmaAllocation* pAllocations,
    VmaAllocationInfo* pAllocationInfo)
{
    if(allocationCount == 0)
    {
        return VK_SUCCESS;
    }

    VMA_ASSERT(allocator && pVkMemoryRequirements && pCreateInfo && pAllocations);

    VMA_DEBUG_LOG("vmaAllocateMemoryPages");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    VkResult result = allocator->allocate_memory(
        *pVkMemoryRequirements,
        false, // requiresDedicatedAllocation
        false, // prefersDedicatedAllocation
        VK_NULL_HANDLE, // dedicatedBuffer
        VK_NULL_HANDLE, // dedicatedImage
        VmaBufferImageUsage::UNKNOWN, // dedicatedBufferImageUsage
        *pCreateInfo,
        VMA_SUBALLOCATION_TYPE_UNKNOWN,
        allocationCount,
        pAllocations);

    if(pAllocationInfo != VMA_NULL && result == VK_SUCCESS)
    {
        for(size_t i = 0; i < allocationCount; ++i)
        {
            allocator->get_allocation_info(pAllocations[i], pAllocationInfo + i);
        }
    }

    return result;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemoryForBuffer(
    VmaAllocator allocator,
    VkBuffer buffer,
    const VmaAllocationCreateInfo* pCreateInfo,
    VmaAllocation* pAllocation,
    VmaAllocationInfo* pAllocationInfo)
{
    VMA_ASSERT(allocator && buffer != VK_NULL_HANDLE && pCreateInfo && pAllocation);

    VMA_DEBUG_LOG("vmaAllocateMemoryForBuffer");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    VkMemoryRequirements vkMemReq = {};
    bool requiresDedicatedAllocation = false;
    bool prefersDedicatedAllocation = false;
    allocator->get_buffer_memory_requirements(buffer, vkMemReq,
        requiresDedicatedAllocation,
        prefersDedicatedAllocation);

    VkResult result = allocator->allocate_memory(
        vkMemReq,
        requiresDedicatedAllocation,
        prefersDedicatedAllocation,
        buffer, // dedicatedBuffer
        VK_NULL_HANDLE, // dedicatedImage
        VmaBufferImageUsage::UNKNOWN, // dedicatedBufferImageUsage
        *pCreateInfo,
        VMA_SUBALLOCATION_TYPE_BUFFER,
        1, // allocationCount
        pAllocation);

    if(pAllocationInfo && result == VK_SUCCESS)
    {
        allocator->get_allocation_info(*pAllocation, pAllocationInfo);
    }

    return result;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemoryForImage(
    VmaAllocator allocator,
    VkImage image,
    const VmaAllocationCreateInfo* pCreateInfo,
    VmaAllocation* pAllocation,
    VmaAllocationInfo* pAllocationInfo)
{
    VMA_ASSERT(allocator && image != VK_NULL_HANDLE && pCreateInfo && pAllocation);

    VMA_DEBUG_LOG("vmaAllocateMemoryForImage");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    VkMemoryRequirements vkMemReq = {};
    bool requiresDedicatedAllocation = false;
    bool prefersDedicatedAllocation  = false;
    allocator->get_image_memory_requirements(image, vkMemReq,
        requiresDedicatedAllocation, prefersDedicatedAllocation);

    VkResult result = allocator->allocate_memory(
        vkMemReq,
        requiresDedicatedAllocation,
        prefersDedicatedAllocation,
        VK_NULL_HANDLE, // dedicatedBuffer
        image, // dedicatedImage
        VmaBufferImageUsage::UNKNOWN, // dedicatedBufferImageUsage
        *pCreateInfo,
        VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN,
        1, // allocationCount
        pAllocation);

    if(pAllocationInfo && result == VK_SUCCESS)
    {
        allocator->get_allocation_info(*pAllocation, pAllocationInfo);
    }

    return result;
}

VMA_CALL_PRE void VMA_CALL_POST vmaFreeMemory(
    VmaAllocator allocator,
    VmaAllocation allocation)
{
    VMA_ASSERT(allocator);

    if(allocation == VK_NULL_HANDLE)
    {
        return;
    }

    VMA_DEBUG_LOG("vmaFreeMemory");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->free_memory(
        1, // allocationCount
        &allocation);
}

VMA_CALL_PRE void VMA_CALL_POST vmaFreeMemoryPages(
    VmaAllocator allocator,
    size_t allocationCount,
    const VmaAllocation* pAllocations)
{
    if(allocationCount == 0)
    {
        return;
    }

    VMA_ASSERT(allocator);

    VMA_DEBUG_LOG("vmaFreeMemoryPages");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->free_memory(allocationCount, pAllocations);
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetAllocationInfo(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VmaAllocationInfo* pAllocationInfo)
{
    VMA_ASSERT(allocator && allocation && pAllocationInfo);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->get_allocation_info(allocation, pAllocationInfo);
}

VMA_CALL_PRE void VMA_CALL_POST vmaget_allocation_info2(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VmaAllocationInfo2* pAllocationInfo)
{
    VMA_ASSERT(allocator && allocation && pAllocationInfo);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->get_allocation_info2(allocation, pAllocationInfo);
}

VMA_CALL_PRE void VMA_CALL_POST vmaSetAllocationUserData(
    VmaAllocator allocator,
    VmaAllocation allocation,
    void* pUserData)
{
    VMA_ASSERT(allocator && allocation);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocation->set_user_data(allocator, pUserData);
}

VMA_CALL_PRE void VMA_CALL_POST vmaSetAllocationName(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    const char* VMA_NULLABLE pName)
{
    allocation->set_name(allocator, pName);
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetAllocationMemoryProperties(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VkMemoryPropertyFlags* VMA_NOT_NULL pFlags)
{
    VMA_ASSERT(allocator && allocation && pFlags);
    const uint32_t memTypeIndex = allocation->get_memory_type_index();
    *pFlags = allocator->_mem_props.memoryTypes[memTypeIndex].propertyFlags;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaMapMemory(
    VmaAllocator allocator,
    VmaAllocation allocation,
    void** ppData)
{
    VMA_ASSERT(allocator && allocation && ppData);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->Map(allocation, ppData);
}

VMA_CALL_PRE void VMA_CALL_POST vmaUnmapMemory(
    VmaAllocator allocator,
    VmaAllocation allocation)
{
    VMA_ASSERT(allocator && allocation);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->unmap(allocation);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaFlushAllocation(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VkDeviceSize offset,
    VkDeviceSize size)
{
    VMA_ASSERT(allocator && allocation);

    VMA_DEBUG_LOG("vmaFlushAllocation");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->flush_or_invalidate_allocation(allocation, offset, size, VMA_CACHE_FLUSH);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaInvalidateAllocation(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VkDeviceSize offset,
    VkDeviceSize size)
{
    VMA_ASSERT(allocator && allocation);

    VMA_DEBUG_LOG("vmaInvalidateAllocation");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->flush_or_invalidate_allocation(allocation, offset, size, VMA_CACHE_INVALIDATE);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaFlushAllocations(
    VmaAllocator allocator,
    uint32_t allocationCount,
    const VmaAllocation* allocations,
    const VkDeviceSize* offsets,
    const VkDeviceSize* sizes)
{
    VMA_ASSERT(allocator);

    if(allocationCount == 0)
    {
        return VK_SUCCESS;
    }

    VMA_ASSERT(allocations);

    VMA_DEBUG_LOG("vmaFlushAllocations");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->flush_or_invalidate_allocations(allocationCount, allocations, offsets, sizes, VMA_CACHE_FLUSH);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaInvalidateAllocations(
    VmaAllocator allocator,
    uint32_t allocationCount,
    const VmaAllocation* allocations,
    const VkDeviceSize* offsets,
    const VkDeviceSize* sizes)
{
    VMA_ASSERT(allocator);

    if(allocationCount == 0)
    {
        return VK_SUCCESS;
    }

    VMA_ASSERT(allocations);

    VMA_DEBUG_LOG("vmaInvalidateAllocations");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->flush_or_invalidate_allocations(allocationCount, allocations, offsets, sizes, VMA_CACHE_INVALIDATE);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCopyMemoryToAllocation(
    VmaAllocator allocator,
    const void* pSrcHostPointer,
    VmaAllocation dstAllocation,
    VkDeviceSize dstAllocationLocalOffset,
    VkDeviceSize size)
{
    VMA_ASSERT(allocator && pSrcHostPointer && dstAllocation);

    if(size == 0)
    {
        return VK_SUCCESS;
    }

    VMA_DEBUG_LOG("vmaCopyMemoryToAllocation");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->copy_memory_to_allocation(pSrcHostPointer, dstAllocation, dstAllocationLocalOffset, size);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCopyAllocationToMemory(
    VmaAllocator allocator,
    VmaAllocation srcAllocation,
    VkDeviceSize srcAllocationLocalOffset,
    void* pDstHostPointer,
    VkDeviceSize size)
{
    VMA_ASSERT(allocator && srcAllocation && pDstHostPointer);

    if(size == 0)
    {
        return VK_SUCCESS;
    }

    VMA_DEBUG_LOG("vmaCopyAllocationToMemory");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->copy_allocation_to_memory(srcAllocation, srcAllocationLocalOffset, pDstHostPointer, size);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCheckCorruption(
    VmaAllocator allocator,
    uint32_t memoryTypeBits)
{
    VMA_ASSERT(allocator);

    VMA_DEBUG_LOG("vmaCheckCorruption");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->check_corruption(memoryTypeBits);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaBeginDefragmentation(
    VmaAllocator allocator,
    const VmaDefragmentationInfo* pInfo,
    VmaDefragmentationContext* pContext)
{
    VMA_ASSERT(allocator && pInfo && pContext);

    VMA_DEBUG_LOG("vmaBeginDefragmentation");

    if (pInfo->pool != VMA_NULL)
    {
        // Check if run on supported algorithms
        if (pInfo->pool->_block_vector.get_algorithm() & VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT)
            return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    *pContext = vma_new(allocator, VmaDefragmentationContext_T)(allocator, *pInfo);
    return VK_SUCCESS;
}

VMA_CALL_PRE void VMA_CALL_POST vmaEndDefragmentation(
    VmaAllocator allocator,
    VmaDefragmentationContext context,
    VmaDefragmentationStats* pStats)
{
    VMA_ASSERT(allocator && context);

    VMA_DEBUG_LOG("vmaEndDefragmentation");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    if (pStats)
        context->get_stats(*pStats);
    vma_delete(allocator, context);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaBeginDefragmentationPass(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaDefragmentationContext VMA_NOT_NULL context,
    VmaDefragmentationPassMoveInfo* VMA_NOT_NULL pPassInfo)
{
    VMA_ASSERT(context && pPassInfo);

    VMA_DEBUG_LOG("vmaBeginDefragmentationPass");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return context->defragment_pass_begin(*pPassInfo);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaEndDefragmentationPass(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaDefragmentationContext VMA_NOT_NULL context,
    VmaDefragmentationPassMoveInfo* VMA_NOT_NULL pPassInfo)
{
    VMA_ASSERT(context && pPassInfo);

    VMA_DEBUG_LOG("vmaEndDefragmentationPass");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return context->defragment_pass_end(*pPassInfo);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindBufferMemory(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VkBuffer buffer)
{
    VMA_ASSERT(allocator && allocation && buffer);

    VMA_DEBUG_LOG("vmaBindBufferMemory");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->bind_buffer_memory(allocation, 0, buffer, VMA_NULL);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindBufferMemory2(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VkDeviceSize allocationLocalOffset,
    VkBuffer buffer,
    const void* pNext)
{
    VMA_ASSERT(allocator && allocation && buffer);

    VMA_DEBUG_LOG("vmaBindBufferMemory2");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->bind_buffer_memory(allocation, allocationLocalOffset, buffer, pNext);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindImageMemory(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VkImage image)
{
    VMA_ASSERT(allocator && allocation && image);

    VMA_DEBUG_LOG("vmaBindImageMemory");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->bind_image_memory(allocation, 0, image, VMA_NULL);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindImageMemory2(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VkDeviceSize allocationLocalOffset,
    VkImage image,
    const void* pNext)
{
    VMA_ASSERT(allocator && allocation && image);

    VMA_DEBUG_LOG("vmaBindImageMemory2");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

        return allocator->bind_image_memory(allocation, allocationLocalOffset, image, pNext);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateBuffer(
    VmaAllocator allocator,
    const VkBufferCreateInfo* pBufferCreateInfo,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    VkBuffer* pBuffer,
    VmaAllocation* pAllocation,
    VmaAllocationInfo* pAllocationInfo)
{
    VMA_ASSERT(allocator && pBufferCreateInfo && pAllocationCreateInfo && pBuffer && pAllocation);

    if(pBufferCreateInfo->size == 0)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if((pBufferCreateInfo->usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_COPY) != 0 &&
        !allocator->_use_khr_buffer_device_address)
    {
        VMA_ASSERT(0 && "Creating a buffer with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT is not valid if VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT was not used.");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VMA_DEBUG_LOG("vmaCreateBuffer");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    *pBuffer = VK_NULL_HANDLE;
    *pAllocation = VK_NULL_HANDLE;

    // 1. Create VkBuffer.
    VkResult res = (*allocator->get_vulkan_functions().vkCreateBuffer)(
        allocator->_h_device,
        pBufferCreateInfo,
        allocator->get_allocation_callbacks(),
        pBuffer);
    if(res >= 0)
    {
        // 2. vkGetBufferMemoryRequirements.
        VkMemoryRequirements vkMemReq = {};
        bool requiresDedicatedAllocation = false;
        bool prefersDedicatedAllocation  = false;
        allocator->get_buffer_memory_requirements(*pBuffer, vkMemReq,
            requiresDedicatedAllocation, prefersDedicatedAllocation);

        // 3. Allocate memory using allocator.
        res = allocator->allocate_memory(
            vkMemReq,
            requiresDedicatedAllocation,
            prefersDedicatedAllocation,
            *pBuffer, // dedicatedBuffer
            VK_NULL_HANDLE, // dedicatedImage
            VmaBufferImageUsage(*pBufferCreateInfo, allocator->_use_khr_maintenance5), // dedicatedBufferImageUsage
            *pAllocationCreateInfo,
            VMA_SUBALLOCATION_TYPE_BUFFER,
            1, // allocationCount
            pAllocation);

        if(res >= 0)
        {
            // 3. Bind buffer with memory.
            if((pAllocationCreateInfo->flags & VMA_ALLOCATION_CREATE_DONT_BIND_BIT) == 0)
            {
                res = allocator->bind_buffer_memory(*pAllocation, 0, *pBuffer, VMA_NULL);
            }
            if(res >= 0)
            {
                // All steps succeeded.
                #if VMA_STATS_STRING_ENABLED
                    (*pAllocation)->init_buffer_usage(*pBufferCreateInfo, allocator->_use_khr_maintenance5);
                #endif
                if(pAllocationInfo != VMA_NULL)
                {
                    allocator->get_allocation_info(*pAllocation, pAllocationInfo);
                }

                return VK_SUCCESS;
            }
            allocator->free_memory(
                1, // allocationCount
                pAllocation);
            *pAllocation = VK_NULL_HANDLE;
            (*allocator->get_vulkan_functions().vkDestroyBuffer)(allocator->_h_device, *pBuffer, allocator->get_allocation_callbacks());
            *pBuffer = VK_NULL_HANDLE;
            return res;
        }
        (*allocator->get_vulkan_functions().vkDestroyBuffer)(allocator->_h_device, *pBuffer, allocator->get_allocation_callbacks());
        *pBuffer = VK_NULL_HANDLE;
        return res;
    }
    return res;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateBufferWithAlignment(
    VmaAllocator allocator,
    const VkBufferCreateInfo* pBufferCreateInfo,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    VkDeviceSize minAlignment,
    VkBuffer* pBuffer,
    VmaAllocation* pAllocation,
    VmaAllocationInfo* pAllocationInfo)
{
    VMA_ASSERT(allocator && pBufferCreateInfo && pAllocationCreateInfo && VmaIsPow2(minAlignment) && pBuffer && pAllocation);

    if(pBufferCreateInfo->size == 0)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if((pBufferCreateInfo->usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_COPY) != 0 &&
        !allocator->_use_khr_buffer_device_address)
    {
        VMA_ASSERT(0 && "Creating a buffer with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT is not valid if VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT was not used.");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VMA_DEBUG_LOG("vmaCreateBufferWithAlignment");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    *pBuffer = VK_NULL_HANDLE;
    *pAllocation = VK_NULL_HANDLE;

    // 1. Create VkBuffer.
    VkResult res = (*allocator->get_vulkan_functions().vkCreateBuffer)(
        allocator->_h_device,
        pBufferCreateInfo,
        allocator->get_allocation_callbacks(),
        pBuffer);
    if(res >= 0)
    {
        // 2. vkGetBufferMemoryRequirements.
        VkMemoryRequirements vkMemReq = {};
        bool requiresDedicatedAllocation = false;
        bool prefersDedicatedAllocation  = false;
        allocator->get_buffer_memory_requirements(*pBuffer, vkMemReq,
            requiresDedicatedAllocation, prefersDedicatedAllocation);

        // 2a. Include minAlignment
        vkMemReq.alignment = VMA_MAX(vkMemReq.alignment, minAlignment);

        // 3. Allocate memory using allocator.
        res = allocator->allocate_memory(
            vkMemReq,
            requiresDedicatedAllocation,
            prefersDedicatedAllocation,
            *pBuffer, // dedicatedBuffer
            VK_NULL_HANDLE, // dedicatedImage
            VmaBufferImageUsage(*pBufferCreateInfo, allocator->_use_khr_maintenance5), // dedicatedBufferImageUsage
            *pAllocationCreateInfo,
            VMA_SUBALLOCATION_TYPE_BUFFER,
            1, // allocationCount
            pAllocation);

        if(res >= 0)
        {
            // 3. Bind buffer with memory.
            if((pAllocationCreateInfo->flags & VMA_ALLOCATION_CREATE_DONT_BIND_BIT) == 0)
            {
                res = allocator->bind_buffer_memory(*pAllocation, 0, *pBuffer, VMA_NULL);
            }
            if(res >= 0)
            {
                // All steps succeeded.
                #if VMA_STATS_STRING_ENABLED
                    (*pAllocation)->init_buffer_usage(*pBufferCreateInfo, allocator->_use_khr_maintenance5);
                #endif
                if(pAllocationInfo != VMA_NULL)
                {
                    allocator->get_allocation_info(*pAllocation, pAllocationInfo);
                }

                return VK_SUCCESS;
            }
            allocator->free_memory(
                1, // allocationCount
                pAllocation);
            *pAllocation = VK_NULL_HANDLE;
            (*allocator->get_vulkan_functions().vkDestroyBuffer)(allocator->_h_device, *pBuffer, allocator->get_allocation_callbacks());
            *pBuffer = VK_NULL_HANDLE;
            return res;
        }
        (*allocator->get_vulkan_functions().vkDestroyBuffer)(allocator->_h_device, *pBuffer, allocator->get_allocation_callbacks());
        *pBuffer = VK_NULL_HANDLE;
        return res;
    }
    return res;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAliasingBuffer(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    const VkBufferCreateInfo* VMA_NOT_NULL pBufferCreateInfo,
    VkBuffer VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pBuffer)
{
    return vmaCreateAliasingBuffer2(allocator, allocation, 0, pBufferCreateInfo, pBuffer);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAliasingBuffer2(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VkDeviceSize allocationLocalOffset,
    const VkBufferCreateInfo* VMA_NOT_NULL pBufferCreateInfo,
    VkBuffer VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pBuffer)
{
    VMA_ASSERT(allocator && pBufferCreateInfo && pBuffer && allocation);
    VMA_ASSERT(allocationLocalOffset + pBufferCreateInfo->size <= allocation->get_size());

    VMA_DEBUG_LOG("vmaCreateAliasingBuffer2");

    *pBuffer = VK_NULL_HANDLE;

    if (pBufferCreateInfo->size == 0)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if ((pBufferCreateInfo->usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_COPY) != 0 &&
        !allocator->_use_khr_buffer_device_address)
    {
        VMA_ASSERT(0 && "Creating a buffer with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT is not valid if VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT was not used.");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    // 1. Create VkBuffer.
    VkResult res = (*allocator->get_vulkan_functions().vkCreateBuffer)(
        allocator->_h_device,
        pBufferCreateInfo,
        allocator->get_allocation_callbacks(),
        pBuffer);
    if (res >= 0)
    {
        // 2. Bind buffer with memory.
        res = allocator->bind_buffer_memory(allocation, allocationLocalOffset, *pBuffer, VMA_NULL);
        if (res >= 0)
        {
            return VK_SUCCESS;
        }
        (*allocator->get_vulkan_functions().vkDestroyBuffer)(allocator->_h_device, *pBuffer, allocator->get_allocation_callbacks());
    }
    return res;
}

VMA_CALL_PRE void VMA_CALL_POST vmaDestroyBuffer(
    VmaAllocator allocator,
    VkBuffer buffer,
    VmaAllocation allocation)
{
    VMA_ASSERT(allocator);

    if(buffer == VK_NULL_HANDLE && allocation == VK_NULL_HANDLE)
    {
        return;
    }

    VMA_DEBUG_LOG("vmaDestroyBuffer");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    if(buffer != VK_NULL_HANDLE)
    {
        (*allocator->get_vulkan_functions().vkDestroyBuffer)(allocator->_h_device, buffer, allocator->get_allocation_callbacks());
    }

    if(allocation != VK_NULL_HANDLE)
    {
        allocator->free_memory(
            1, // allocationCount
            &allocation);
    }
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateImage(
    VmaAllocator allocator,
    const VkImageCreateInfo* pImageCreateInfo,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    VkImage* pImage,
    VmaAllocation* pAllocation,
    VmaAllocationInfo* pAllocationInfo)
{
    VMA_ASSERT(allocator && pImageCreateInfo && pAllocationCreateInfo && pImage && pAllocation);

    VMA_ASSERT((pImageCreateInfo->flags & VK_IMAGE_CREATE_DISJOINT_BIT_COPY) == 0 &&
        "vmaCreateImage() doesn't support disjoint multi-planar images. Please allocate memory for the planes using vmaAllocateMemory() and bind them using vmaBindImageMemory2().");

    if(pImageCreateInfo->extent.width == 0 ||
        pImageCreateInfo->extent.height == 0 ||
        pImageCreateInfo->extent.depth == 0 ||
        pImageCreateInfo->mipLevels == 0 ||
        pImageCreateInfo->arrayLayers == 0)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VMA_DEBUG_LOG("vmaCreateImage");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    *pImage = VK_NULL_HANDLE;
    *pAllocation = VK_NULL_HANDLE;

    // 1. Create VkImage.
    VkResult res = (*allocator->get_vulkan_functions().vkCreateImage)(
        allocator->_h_device,
        pImageCreateInfo,
        allocator->get_allocation_callbacks(),
        pImage);
    if(res == VK_SUCCESS)
    {
        VmaSuballocationType suballocType = pImageCreateInfo->tiling == VK_IMAGE_TILING_OPTIMAL ?
            VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL :
            VMA_SUBALLOCATION_TYPE_IMAGE_LINEAR;

        // 2. Allocate memory using allocator.
        VkMemoryRequirements vkMemReq = {};
        bool requiresDedicatedAllocation = false;
        bool prefersDedicatedAllocation  = false;
        allocator->get_image_memory_requirements(*pImage, vkMemReq,
            requiresDedicatedAllocation, prefersDedicatedAllocation);

        res = allocator->allocate_memory(
            vkMemReq,
            requiresDedicatedAllocation,
            prefersDedicatedAllocation,
            VK_NULL_HANDLE, // dedicatedBuffer
            *pImage, // dedicatedImage
            VmaBufferImageUsage(*pImageCreateInfo), // dedicatedBufferImageUsage
            *pAllocationCreateInfo,
            suballocType,
            1, // allocationCount
            pAllocation);

        if(res == VK_SUCCESS)
        {
            // 3. Bind image with memory.
            if((pAllocationCreateInfo->flags & VMA_ALLOCATION_CREATE_DONT_BIND_BIT) == 0)
            {
                res = allocator->bind_image_memory(*pAllocation, 0, *pImage, VMA_NULL);
            }
            if(res == VK_SUCCESS)
            {
                // All steps succeeded.
                #if VMA_STATS_STRING_ENABLED
                    (*pAllocation)->init_image_usage(*pImageCreateInfo);
                #endif
                if(pAllocationInfo != VMA_NULL)
                {
                    allocator->get_allocation_info(*pAllocation, pAllocationInfo);
                }

                return VK_SUCCESS;
            }
            allocator->free_memory(
                1, // allocationCount
                pAllocation);
            *pAllocation = VK_NULL_HANDLE;
            (*allocator->get_vulkan_functions().vkDestroyImage)(allocator->_h_device, *pImage, allocator->get_allocation_callbacks());
            *pImage = VK_NULL_HANDLE;
            return res;
        }
        (*allocator->get_vulkan_functions().vkDestroyImage)(allocator->_h_device, *pImage, allocator->get_allocation_callbacks());
        *pImage = VK_NULL_HANDLE;
        return res;
    }
    return res;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAliasingImage(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    const VkImageCreateInfo* VMA_NOT_NULL pImageCreateInfo,
    VkImage VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pImage)
{
    return vmaCreateAliasingImage2(allocator, allocation, 0, pImageCreateInfo, pImage);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAliasingImage2(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VkDeviceSize allocationLocalOffset,
    const VkImageCreateInfo* VMA_NOT_NULL pImageCreateInfo,
    VkImage VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pImage)
{
    VMA_ASSERT(allocator && pImageCreateInfo && pImage && allocation);

    *pImage = VK_NULL_HANDLE;

    VMA_DEBUG_LOG("vmaCreateImage2");

    if (pImageCreateInfo->extent.width == 0 ||
        pImageCreateInfo->extent.height == 0 ||
        pImageCreateInfo->extent.depth == 0 ||
        pImageCreateInfo->mipLevels == 0 ||
        pImageCreateInfo->arrayLayers == 0)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    // 1. Create VkImage.
    VkResult res = (*allocator->get_vulkan_functions().vkCreateImage)(
        allocator->_h_device,
        pImageCreateInfo,
        allocator->get_allocation_callbacks(),
        pImage);
    if (res >= 0)
    {
        // 2. Bind image with memory.
        res = allocator->bind_image_memory(allocation, allocationLocalOffset, *pImage, VMA_NULL);
        if (res >= 0)
        {
            return VK_SUCCESS;
        }
        (*allocator->get_vulkan_functions().vkDestroyImage)(allocator->_h_device, *pImage, allocator->get_allocation_callbacks());
    }
    return res;
}

VMA_CALL_PRE void VMA_CALL_POST vmaDestroyImage(
    VmaAllocator VMA_NOT_NULL allocator,
    VkImage VMA_NULLABLE_NON_DISPATCHABLE image,
    VmaAllocation VMA_NULLABLE allocation)
{
    VMA_ASSERT(allocator);

    if(image == VK_NULL_HANDLE && allocation == VK_NULL_HANDLE)
    {
        return;
    }

    VMA_DEBUG_LOG("vmaDestroyImage");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    if(image != VK_NULL_HANDLE)
    {
        (*allocator->get_vulkan_functions().vkDestroyImage)(allocator->_h_device, image, allocator->get_allocation_callbacks());
    }
    if(allocation != VK_NULL_HANDLE)
    {
        allocator->free_memory(
            1, // allocationCount
            &allocation);
    }
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateVirtualBlock(
    const VmaVirtualBlockCreateInfo* VMA_NOT_NULL pCreateInfo,
    VmaVirtualBlock VMA_NULLABLE * VMA_NOT_NULL pVirtualBlock)
{
    VMA_ASSERT(pCreateInfo && pVirtualBlock);
    VMA_ASSERT(pCreateInfo->size > 0);
    VMA_DEBUG_LOG("vmaCreateVirtualBlock");
    VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    *pVirtualBlock = vma_new(pCreateInfo->pAllocationCallbacks, VmaVirtualBlock_T)(*pCreateInfo);
    return VK_SUCCESS;

    /*
    Code for the future if we ever need a separate init() method that could fail:

    VkResult res = (*pVirtualBlock)->init();
    if(res < 0)
    {
        vma_delete(pCreateInfo->pAllocationCallbacks, *pVirtualBlock);
        *pVirtualBlock = VK_NULL_HANDLE;
    }
    return res;
    */
}

VMA_CALL_PRE void VMA_CALL_POST vmaDestroyVirtualBlock(VmaVirtualBlock VMA_NULLABLE virtualBlock)
{
    if(virtualBlock != VK_NULL_HANDLE)
    {
        VMA_DEBUG_LOG("vmaDestroyVirtualBlock");
        VMA_DEBUG_GLOBAL_MUTEX_LOCK;
        VkAllocationCallbacks allocationCallbacks = virtualBlock->_allocation_callbacks; // Have to copy the callbacks when destroying.
        vma_delete(&allocationCallbacks, virtualBlock);
    }
}

VMA_CALL_PRE VkBool32 VMA_CALL_POST vmaIsVirtualBlockEmpty(VmaVirtualBlock VMA_NOT_NULL virtualBlock)
{
    VMA_ASSERT(virtualBlock != VK_NULL_HANDLE);
    VMA_DEBUG_LOG("vmaIsVirtualBlockEmpty");
    VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    return virtualBlock->is_empty() ? VK_TRUE : VK_FALSE;
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetVirtualAllocationInfo(VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    VmaVirtualAllocation VMA_NOT_NULL_NON_DISPATCHABLE allocation, VmaVirtualAllocationInfo* VMA_NOT_NULL pVirtualAllocInfo)
{
    VMA_ASSERT(virtualBlock != VK_NULL_HANDLE && pVirtualAllocInfo != VMA_NULL);
    VMA_DEBUG_LOG("vmaGetVirtualAllocationInfo");
    VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    virtualBlock->get_allocation_info(allocation, *pVirtualAllocInfo);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaVirtualAllocate(VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    const VmaVirtualAllocationCreateInfo* VMA_NOT_NULL pCreateInfo, VmaVirtualAllocation VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pAllocation,
    VkDeviceSize* VMA_NULLABLE pOffset)
{
    VMA_ASSERT(virtualBlock != VK_NULL_HANDLE && pCreateInfo != VMA_NULL && pAllocation != VMA_NULL);
    VMA_DEBUG_LOG("vmaVirtualAllocate");
    VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    return virtualBlock->Allocate(*pCreateInfo, *pAllocation, pOffset);
}

VMA_CALL_PRE void VMA_CALL_POST vmaVirtualFree(VmaVirtualBlock VMA_NOT_NULL virtualBlock, VmaVirtualAllocation VMA_NULLABLE_NON_DISPATCHABLE allocation)
{
    if(allocation != VK_NULL_HANDLE)
    {
        VMA_ASSERT(virtualBlock != VK_NULL_HANDLE);
        VMA_DEBUG_LOG("vmaVirtualFree");
        VMA_DEBUG_GLOBAL_MUTEX_LOCK;
        virtualBlock->free(allocation);
    }
}

VMA_CALL_PRE void VMA_CALL_POST vmaClearVirtualBlock(VmaVirtualBlock VMA_NOT_NULL virtualBlock)
{
    VMA_ASSERT(virtualBlock != VK_NULL_HANDLE);
    VMA_DEBUG_LOG("vmaClearVirtualBlock");
    VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    virtualBlock->clear();
}

VMA_CALL_PRE void VMA_CALL_POST vmaSetVirtualAllocationUserData(VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    VmaVirtualAllocation VMA_NOT_NULL_NON_DISPATCHABLE allocation, void* VMA_NULLABLE pUserData)
{
    VMA_ASSERT(virtualBlock != VK_NULL_HANDLE);
    VMA_DEBUG_LOG("vmaSetVirtualAllocationUserData");
    VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    virtualBlock->set_allocation_user_data(allocation, pUserData);
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetVirtualBlockStatistics(VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    VmaStatistics* VMA_NOT_NULL pStats)
{
    VMA_ASSERT(virtualBlock != VK_NULL_HANDLE && pStats != VMA_NULL);
    VMA_DEBUG_LOG("vmaGetVirtualBlockStatistics");
    VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    virtualBlock->get_statistics(*pStats);
}

VMA_CALL_PRE void VMA_CALL_POST vmaCalculateVirtualBlockStatistics(VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    VmaDetailedStatistics* VMA_NOT_NULL pStats)
{
    VMA_ASSERT(virtualBlock != VK_NULL_HANDLE && pStats != VMA_NULL);
    VMA_DEBUG_LOG("vmaCalculateVirtualBlockStatistics");
    VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    virtualBlock->calculate_detailed_statistics(*pStats);
}

#if VMA_STATS_STRING_ENABLED

VMA_CALL_PRE void VMA_CALL_POST vmaBuildVirtualBlockStatsString(VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    char* VMA_NULLABLE * VMA_NOT_NULL ppStatsString, VkBool32 detailedMap)
{
    VMA_ASSERT(virtualBlock != VK_NULL_HANDLE && ppStatsString != VMA_NULL);
    VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    const VkAllocationCallbacks* allocationCallbacks = virtualBlock->get_allocation_callbacks();
    VmaStringBuilder sb(allocationCallbacks);
    virtualBlock->build_stats_string(detailedMap != VK_FALSE, sb);
    *ppStatsString = VmaCreateStringCopy(allocationCallbacks, sb.get_data(), sb.get_length());
}

VMA_CALL_PRE void VMA_CALL_POST vmaFreeVirtualBlockStatsString(VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    char* VMA_NULLABLE pStatsString)
{
    if(pStatsString != VMA_NULL)
    {
        VMA_ASSERT(virtualBlock != VK_NULL_HANDLE);
        VMA_DEBUG_GLOBAL_MUTEX_LOCK;
        VmaFreeString(virtualBlock->get_allocation_callbacks(), pStatsString);
    }
}
#if VMA_EXTERNAL_MEMORY_WIN32
VMA_CALL_PRE VkResult VMA_CALL_POST vmaGetMemoryWin32Handle(VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation, HANDLE hTargetProcess, HANDLE* VMA_NOT_NULL pHandle)
{
    VMA_ASSERT(allocator && allocation && pHandle);
    VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    return allocation->get_win32_handle(allocator, hTargetProcess, pHandle);
}
#endif // VMA_EXTERNAL_MEMORY_WIN32 
#endif // VMA_STATS_STRING_ENABLED
#endif // _VMA_PUBLIC_INTERFACE
#endif // VMA_IMPLEMENTATION

/**
\page faq Frequently asked questions

<b>What is VMA?</b>

Vulkan(R) Memory Allocator (VMA) is a software library for developers who use the Vulkan graphics API in their code.
It is written in C++.

<b>What is the license of VMA?</b>

VMA is licensed under MIT, which means it is open source and free software.

<b>What is the purpose of VMA?</b>

VMA helps with handling one aspect of Vulkan usage, which is device memory management -
allocation of `VkDeviceMemory` objects, and creation of `VkBuffer` and `VkImage` objects.

<b>Do I need to use VMA?</b>

You don't need to, but it may be beneficial in many cases.
Vulkan is a complex and low-level API, so libraries like this that abstract certain aspects of the API
and bring them to a higher level are useful.
When developing any non-trivial Vulkan application, you likely need to use a memory allocator.
Using VMA can save time compared to implementing your own.

<b>When should I not use VMA?</b>

While VMA is useful for most applications that use the Vulkan API, there are cases
when it may be a better choice not to use it.
For example, if the application is very simple, e.g. serving as a sample or a learning exercise
to help you understand or teach others the basics of Vulkan,
and it creates only a small number of buffers and images, then including VMA may be an overkill.
Developing your own memory allocator may also be a good learning exercise.

<b>What are the benefits of using VMA?</b>

-# VMA helps in choosing the optimal memory type for your resource (buffer or image).
   In Vulkan, we have a two-level hierarchy of memory heaps and types with different flags,
   and each device can expose a different set of those.
   Implementing logic that would select the best memory type on each platform is a non-trivial task.
   VMA does that, expecting only a high-level description of the intended usage of your resource.
   For more information, see \subpage choosing_memory_type.
-# VMA allocates large blocks of `VkDeviceMemory` and sub-allocates parts of them for your resources.
   Allocating a new block of device memory may be a time-consuming operation.
   Some platforms also have a limit on the maximum number of those blocks (`VkPhysicalDeviceLimits::maxMemoryAllocationCount`)
   as low as 4096, so allocating a separate one for each resource is not an option.
   Sub-allocating parts of a memory block requires implementing an allocation algorithm,
   which is a non-trivial task.
   VMA does that, using an advanced and efficient algorithm that works well in various use cases.
-# VMA offers a simple API that allows creating buffers and textures within one function call.
   In Vulkan, the creation of a resource is a multi-step process.
   You need to create a `VkBuffer` or `VkImage`, ask it for memory requirements,
   allocate a `VkDeviceMemory` object, and finally bind the resource to the memory block.
   VMA does that automatically under a simple API within one function call: vmaCreateBuffer(), vmaCreateImage().

The library is doing much more under the hood.
For example, it respects limits like `bufferImageGranularity`, `nonCoherentAtomSize`,
and `VkMemoryDedicatedRequirements` automatically, so you don't need to think about it.

<b>Which version should I pick?</b>

You can just pick [the latest version from the "master" branch](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator).
It is kept in a good shape most of the time, compiling and working correctly,
with no compatibility-breaking changes and no unfinished code.

If you want an even more stable version, you can pick
[the latest official release](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/releases).
Current code from the master branch is occasionally tagged as a release,
with [CHANGELOG](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/blob/master/CHANGELOG.md)
carefully curated to enumerate all important changes since the previous version.

The library uses [Semantic Versioning](https://semver.org/),
which means versions that only differ in the patch number are forward and backward compatible
(e.g., only fixing some bugs), while versions that differ in the minor number are backward compatible
(e.g., only adding new functions to the API, but not removing or changing existing ones).

<b>How to integrate it with my code?</b>

VMA is an STB-style single-header C++ library.

You can pull the entire GitHub repository, e.g. using Git submodules.
The repository contains ancillary files like the Cmake script, Doxygen config file,
sample application, test suite, and others.
You can compile it as a library and link with your project.

However, a simpler way is taking the single file "include/vk_mem_alloc.h" and including it in your project.
This extensive file contains all you need: a copyright notice,
declarations of the public library interface (API), its internal implementation,
and even the documentation in form of Doxygen-style comments.

The "STB style" means not everything is implemented as inline functions in the header file.
You need to extract the internal implementation using a special macro.
This means that in every .cpp file where you need to use the library you should
`#include "vk_mem_alloc.h"` to include its public interface,
but additionally in exactly one .cpp file you should `#define VMA_IMPLEMENTATION`
before this `#include` to enable its internal implementation.
For more information, see [Project setup](@ref quick_start_project_setup).

<b>Does the library work with C or C++?</b>

The internal implementation of VMA is written in C++.
It is distributed in the source format, so you need a compiler supporting at least C++14 to build it.

However, the public interface of the library is written in C - using only enums, structs, and global functions,
in the same style as Vulkan, so you can use the library in the C code.

<b>I am not a fan of modern C++. Can I still use it?</b>

Very likely yes.
We acknowledge that many C++ developers, especially in the games industry,
do not appreciate all the latest features that the language has to offer.

- VMA doesn't throw or catch any C++ exceptions.
  It reports errors by returning a `VkResult` value instead, just like Vulkan.
  If you don't use exceptions in your project, your code is not exception-safe,
  or even if you disable exception handling in the compiler options, you can still use VMA.
- VMA doesn't use C++ run-time type information like `typeid` or `dynamic_cast`,
  so if you disable RTTI in the compiler options, you can still use the library.
- VMA uses only a limited subset of standard C and C++ library.
  It doesn't use STL containers like `std::vector`, `map`, or `string`,
  either in the public interface nor in the internal implementation.
  It implements its own containers instead.
- If you don't use the default heap memory allocator through `malloc/free` or `new/delete`
  but implement your own allocator instead, you can pass it to VMA and
  the library will use your functions for every dynamic heap allocation made internally,
  as well as passing it further to Vulkan functions. For details, see [Custom host memory allocator](@ref custom_memory_allocator).

<b>Is it available for other programming languages?</b>

VMA is a C++ library with C interface in similar style as Vulkan.
An object-oriented C++ wrapper or bindings to other programming languages are out of scope of this project,
but they are welcome as external projects.
Some of them are listed in [README.md, "See also" section](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator?tab=readme-ov-file#see-also),
including binding to C++, Python, Rust, and Haskell.
Before using any of them, please check if they are still maintained and updated to use a recent version of VMA.

<b>What platforms does it support?</b>

VMA relies only on Vulkan and some parts of the standard C and C++ library,
so it supports any platform where a C++ compiler and Vulkan are available.
It is developed mostly on Microsoft(R) Windows(R),
but it has been successfully used in Linux(R), MacOS, Android, and even FreeBSD and Raspberry Pi.

<b>Does it only work on AMD GPUs?</b>

No! While VMA is published by AMD, it works on any GPU that supports Vulkan,
whether a discrete PC graphics card, a processor integrated graphics, or a mobile SoC.
It doesn't give AMD GPUs any advantage over any other GPUs.

<b>What Vulkan versions and extensions are supported?</b>

VMA is updated to support the latest versions of Vulkan.
It currently supports Vulkan up to 1.4.
The library also supports older versions down to the first release of Vulkan 1.0.
Defining a higher minimum version support would help simplify the code,
but we acknowledge that developers on some platforms like Android still use older versions,
so the support is provided for all of them.

Among many extensions available for Vulkan, only a few interact with memory management.
VMA can automatically take advantage of them. Some of them are:
VK_EXT_memory_budget, VK_EXT_memory_priority, VK_KHR_external_memory_win32, and VK_KHR_maintenance*
extensions that are later promoted to the new versions of the core Vulkan API.

To use them, it is your responsibility to validate if they are available on the current system and if so,
enable them while creating the Vulkan device object.
You also need to pass appropriate #VmaAllocatorCreateFlagBits to inform VMA that they are enabled.
Then, the library will automatically take advantage of them.
For more information and the full list of supported extensions, see [Enabling extensions](@ref quick_start_initialization_enabling_extensions).

<b>Does it support other graphics APIs, like Microsoft DirectX(R) 12?</b>

No, but we offer an equivalent library for DirectX 12:
[D3D12 Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator).
It uses the same core allocation algorithm.
It also shares many features with VMA, like the support for custom pools and virtual allocator.
However, it is not identical in terms of the features supported.
Its API also looks different, because while the interface of VMA is similar in style to Vulkan,
the interface of D3D12MA is similar to DirectX 12.

<b>Is the library lightweight?</b>

It depends on how you define it.
VMA is implemented with high-performance and real-time applications like video games in mind.
The CPU performance overhead of using this library is low.
It uses a high-quality allocation algorithm called Two-Level Segregated Fit (TLSF),
which in most cases can find a free place for a new allocation in few steps.
The library also doesn't perform too many CPU heap allocations.
In many cases, the allocation happens with 0 new CPU heap allocations performed by the library.
Even the creation of a #VmaAllocation object doesn't typically feature an CPU allocation,
because these objects are returned out of a dedicated memory pool.

On the other hand, however, VMA needs some extra memory and extra time
to maintain the metadata about the occupied and free regions of the memory blocks,
and the algorithms and data structures used must be generic enough to work well in most cases.
If you develop your program for a very resource-constrained platform,
a custom allocator simpler than VMA may be a better choice.

<b>Does it have a documentation?</b>

Yes! VMA comes with full documentation of all elements of the API (functions, structures, enums),
as well as many generic chapters that provide an introduction,
describe core concepts of the library, good practices, etc.
The entire documentation is written in form of code comments inside "vk_mem_alloc.h", in Doxygen format.
You can access it in multiple ways:

- Browsable online: https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/
- Local HTML pages available after you clone the repository and open file "docs/html/index.html".
- You can rebuild the documentation in HTML or some other format from the source code using Doxygen.
  Configuration file "Doxyfile" is part of the repository.
- Finally, you can just read the comments preceding declarations of any public functions of the library.

<b>Is it a mature project?</b>

Yes! The library is in development since June 2017, has over 1000 commits, over 400 issue tickets
and pull requests (most of them resolved), and over 70 contributors.
It is distributed together with Vulkan SDK.
It is used by many software projects, including some large and popular ones like Qt or Blender,
as well as some AAA games.
According to the [LunarG 2024 Ecosystem Survey](https://www.lunarg.com/2024-ecosystem-survey-progress-report-released/),
it is used by over 50% of Vulkan developers.

<b>How can I contribute to the project?</b>

If you have an idea for improvement or a feature request,
you can go to [the library repository](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
and create an Issue ticket, describing your idea.
You can also implement it yourself by forking the repository, making changes to the code,
and creating a Pull request.

If you want to ask a question, you can also create a ticket the same way.
Before doing this, please make sure you read the relevant part of the Vulkan specification and VMA documentation,
where you may find the answers to your question.

If you want to report a suspected bug, you can also create a ticket the same way.
Before doing this, please put some effort into the investigation of whether the bug is really
in the library and not in your code or in the Vulkan implementation (the GPU driver) on your platform:

- Enable Vulkan validation layer and make sure it is free from any errors.
- Make sure `VMA_ASSERT` is defined to an implementation that can report a failure and not ignore it.
- Try making your allocation using pure Vulkan functions rather than VMA and see if the bug persists.

<b>I found some compilation warnings. How can we fix them?</b>

Seeing compiler warnings may be annoying to some developers,
but it is a design decision to not fix all of them.
Due to the nature of the C++ language, certain preprocessor macros can make some variables unused,
function parameters unreferenced, or conditional expressions constant in some configurations.
The code of this library should not be bigger or more complicated just to silence these warnings.
It is recommended to disable such warnings instead.
For more information, see [Features not supported](@ref general_considerations_features_not_supported).

However, if you observe a warning that is really dangerous, e.g.,
about an implicit conversion from a larger to a smaller integer type, please report it and it will be fixed ASAP.


\page quick_start Quick start

\section quick_start_project_setup Project setup

Vulkan Memory Allocator comes in form of a "stb-style" single header file.
While you can pull the entire repository e.g. as Git module, there is also Cmake script provided,
you don't need to build it as a separate library project.
You can add file "vk_mem_alloc.h" directly to your project and submit it to code repository next to your other source files.

"Single header" doesn't mean that everything is contained in C/C++ declarations,
like it tends to be in case of inline functions or C++ templates.
It means that implementation is bundled with interface in a single file and needs to be extracted using preprocessor macro.
If you don't do it properly, it will result in linker errors.

To do it properly:

-# Include "vk_mem_alloc.h" file in each CPP file where you want to use the library.
   This includes declarations of all members of the library.
-# In exactly one CPP file define following macro before this include.
   It enables also internal definitions.

\code
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
\endcode

It may be a good idea to create dedicated CPP file just for this purpose, e.g. "VmaUsage.cpp".

This library includes header `<vulkan/vulkan.h>`, which in turn
includes `<windows.h>` on Windows. If you need some specific macros defined
before including these headers (like `WIN32_LEAN_AND_MEAN` or
`WINVER` for Windows, `VK_USE_PLATFORM_WIN32_KHR` for Vulkan), you must define
them before every `#include` of this library.
It may be a good idea to create a dedicate header file for this purpose, e.g. "VmaUsage.h",
that will be included in other source files instead of VMA header directly.

This library is written in C++, but has C-compatible interface.
Thus, you can include and use "vk_mem_alloc.h" in C or C++ code, but full
implementation with `VMA_IMPLEMENTATION` macro must be compiled as C++, NOT as C.
Some features of C++14 are used and required. Features of C++20 are used optionally when available.
Some headers of standard C and C++ library are used, but STL containers, RTTI, or C++ exceptions are not used.


\section quick_start_initialization Initialization

VMA offers library interface in a style similar to Vulkan, with object handles like #VmaAllocation,
structures describing parameters of objects to be created like #VmaAllocationCreateInfo,
and errors codes returned from functions using `VkResult` type.

The first and the main object that needs to be created is #VmaAllocator.
It represents the initialization of the entire library.
Only one such object should be created per `VkDevice`.
You should create it at program startup, after `VkDevice` was created, and before any device memory allocator needs to be made.
It must be destroyed before `VkDevice` is destroyed.

At program startup:

-# Initialize Vulkan to have `VkInstance`, `VkPhysicalDevice`, `VkDevice` object.
-# Fill VmaAllocatorCreateInfo structure and call vmaCreateAllocator() to create #VmaAllocator object.

Only members `physicalDevice`, `device`, `instance` are required.
However, you should inform the library which Vulkan version do you use by setting
VmaAllocatorCreateInfo::vulkanApiVersion and which extensions did you enable
by setting VmaAllocatorCreateInfo::flags.
Otherwise, VMA would use only features of Vulkan 1.0 core with no extensions.
See below for details.

\subsection quick_start_initialization_selecting_vulkan_version Selecting Vulkan version

VMA supports Vulkan version down to 1.0, for backward compatibility.
If you want to use higher version, you need to inform the library about it.
This is a two-step process.

<b>Step 1: Compile time.</b> By default, VMA compiles with code supporting the highest
Vulkan version found in the included `<vulkan/vulkan.h>` that is also supported by the library.
If this is OK, you don't need to do anything.
However, if you want to compile VMA as if only some lower Vulkan version was available,
define macro `VMA_VULKAN_VERSION` before every `#include "vk_mem_alloc.h"`.
It should have decimal numeric value in form of ABBBCCC, where A = major, BBB = minor, CCC = patch Vulkan version.
For example, to compile against Vulkan 1.2:

\code
#define VMA_VULKAN_VERSION 1002000 // Vulkan 1.2
#include "vk_mem_alloc.h"
\endcode

<b>Step 2: Runtime.</b> Even when compiled with higher Vulkan version available,
VMA can use only features of a lower version, which is configurable during creation of the #VmaAllocator object.
By default, only Vulkan 1.0 is used.
To initialize the allocator with support for higher Vulkan version, you need to set member
VmaAllocatorCreateInfo::vulkanApiVersion to an appropriate value, e.g. using constants like `VK_API_VERSION_1_2`.
See code sample below.

\subsection quick_start_initialization_importing_vulkan_functions Importing Vulkan functions

You may need to configure importing Vulkan functions. There are 4 ways to do this:

-# **If you link with Vulkan static library** (e.g. "vulkan-1.lib" on Windows):
   - You don't need to do anything.
   - VMA will use these, as macro `VMA_STATIC_VULKAN_FUNCTIONS` is defined to 1 by default.
-# **If you want VMA to fetch pointers to Vulkan functions dynamically** using `vkGetInstanceProcAddr`,
   `vkGetDeviceProcAddr` (this is the option presented in the example below):
   - Define `VMA_STATIC_VULKAN_FUNCTIONS` to 0, `VMA_DYNAMIC_VULKAN_FUNCTIONS` to 1.
   - Provide pointers to these two functions via VmaVulkanFunctions::vkGetInstanceProcAddr,
     VmaVulkanFunctions::vkGetDeviceProcAddr.
   - The library will fetch pointers to all other functions it needs internally.
-# **If you fetch pointers to all Vulkan functions in a custom way**:
   - Define `VMA_STATIC_VULKAN_FUNCTIONS` and `VMA_DYNAMIC_VULKAN_FUNCTIONS` to 0.
   - Pass these pointers via structure #VmaVulkanFunctions.
-# **If you use [volk library](https://github.com/zeux/volk)**:
   - Define `VMA_STATIC_VULKAN_FUNCTIONS` and `VMA_DYNAMIC_VULKAN_FUNCTIONS` to 0.
   - Use function vmaImportVulkanFunctionsFromVolk() to fill in the structure #VmaVulkanFunctions.
     For more information, see the description of this function.

\subsection quick_start_initialization_enabling_extensions Enabling extensions

VMA can automatically use following Vulkan extensions.
If you found them available on the selected physical device and you enabled them
while creating `VkInstance` / `VkDevice` object, inform VMA about their availability
by setting appropriate flags in VmaAllocatorCreateInfo::flags.

Vulkan extension              | VMA flag
------------------------------|-----------------------------------------------------
VK_KHR_dedicated_allocation   | #VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT
VK_KHR_bind_memory2           | #VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT
VK_KHR_maintenance4           | #VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT
VK_KHR_maintenance5           | #VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT
VK_EXT_memory_budget          | #VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT
VK_KHR_buffer_device_address  | #VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT
VK_EXT_memory_priority        | #VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT
VK_AMD_device_coherent_memory | #VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT
VK_KHR_external_memory_win32  | #VMA_ALLOCATOR_CREATE_KHR_EXTERNAL_MEMORY_WIN32_BIT

Example with fetching pointers to Vulkan functions dynamically:

\code
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include "vk_mem_alloc.h"

...

VmaVulkanFunctions vulkanFunctions = {};
vulkanFunctions.vkGetInstanceProcAddr = &vkGetInstanceProcAddr;
vulkanFunctions.vkGetDeviceProcAddr = &vkGetDeviceProcAddr;

VmaAllocatorCreateInfo allocatorCreateInfo = {};
allocatorCreateInfo.flags = VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_2;
allocatorCreateInfo.physicalDevice = physicalDevice;
allocatorCreateInfo.device = device;
allocatorCreateInfo.instance = instance;
allocatorCreateInfo.pVulkanFunctions = &vulkanFunctions;

VmaAllocator allocator;
vmaCreateAllocator(&allocatorCreateInfo, &allocator);

// Entire program...

// At the end, don't forget to:
vmaDestroyAllocator(allocator);
\endcode


\subsection quick_start_initialization_other_config Other configuration options

There are additional configuration options available through preprocessor macros that you can define
before including VMA header and through parameters passed in #VmaAllocatorCreateInfo.
They include a possibility to use your own callbacks for host memory allocations (`VkAllocationCallbacks`),
callbacks for device memory allocations (instead of `vkAllocateMemory`, `vkFreeMemory`),
or your custom `VMA_ASSERT` macro, among others.
For more information, see: @ref configuration.


\section quick_start_resource_allocation Resource allocation

When you want to create a buffer or image:

-# Fill `VkBufferCreateInfo` / `VkImageCreateInfo` structure.
-# Fill VmaAllocationCreateInfo structure.
-# Call vmaCreateBuffer() / vmaCreateImage() to get `VkBuffer`/`VkImage` with memory
   already allocated and bound to it, plus #VmaAllocation objects that represents its underlying memory.

\code
VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
bufferInfo.size = 65536;
bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

VmaAllocationCreateInfo allocInfo = {};
allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

VkBuffer buffer;
VmaAllocation allocation;
vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr);
\endcode

Don't forget to destroy your buffer and allocation objects when no longer needed:

\code
vmaDestroyBuffer(allocator, buffer, allocation);
\endcode

If you need to map the buffer, you must set flag
#VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or #VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
in VmaAllocationCreateInfo::flags.
There are many additional parameters that can control the choice of memory type to be used for the allocation
and other features.
For more information, see documentation chapters: @ref choosing_memory_type, @ref memory_mapping.


\page choosing_memory_type Choosing memory type

Physical devices in Vulkan support various combinations of memory heaps and
types. Help with choosing correct and optimal memory type for your specific
resource is one of the key features of this library. You can use it by filling
appropriate members of VmaAllocationCreateInfo structure, as described below.
You can also combine multiple methods.

-# If you just want to find memory type index that meets your requirements, you
   can use function: vmaFindMemoryTypeIndexForBufferInfo(),
   vmaFindMemoryTypeIndexForImageInfo(), vmaFindMemoryTypeIndex().
-# If you want to allocate a region of device memory without association with any
   specific image or buffer, you can use function vmaAllocateMemory(). Usage of
   this function is not recommended and usually not needed.
   vmaAllocateMemoryPages() function is also provided for creating multiple allocations at once,
   which may be useful for sparse binding.
-# If you already have a buffer or an image created, you want to allocate memory
   for it and then you will bind it yourself, you can use function
   vmaAllocateMemoryForBuffer(), vmaAllocateMemoryForImage().
   For binding you should use functions: vmaBindBufferMemory(), vmaBindImageMemory()
   or their extended versions: vmaBindBufferMemory2(), vmaBindImageMemory2().
-# If you want to create a buffer or an image, allocate memory for it, and bind
   them together, all in one call, you can use function vmaCreateBuffer(),
   vmaCreateImage().
   <b>This is the easiest and recommended way to use this library!</b>

When using 3. or 4., the library internally queries Vulkan for memory types
supported for that buffer or image (function `vkGetBufferMemoryRequirements()`)
and uses only one of these types.

If no memory type can be found that meets all the requirements, these functions
return `VK_ERROR_FEATURE_NOT_PRESENT`.

You can leave VmaAllocationCreateInfo structure completely filled with zeros.
It means no requirements are specified for memory type.
It is valid, although not very useful.

\section choosing_memory_type_usage Usage

The easiest way to specify memory requirements is to fill member
VmaAllocationCreateInfo::usage using one of the values of enum #VmaMemoryUsage.
It defines high level, common usage types.
Since version 3 of the library, it is recommended to use #VMA_MEMORY_USAGE_AUTO to let it select best memory type for your resource automatically.

For example, if you want to create a uniform buffer that will be filled using
transfer only once or infrequently and then used for rendering every frame as a uniform buffer, you can
do it using following code. The buffer will most likely end up in a memory type with
`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` to be fast to access by the GPU device.

\code
VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
bufferInfo.size = 65536;
bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

VmaAllocationCreateInfo allocInfo = {};
allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

VkBuffer buffer;
VmaAllocation allocation;
vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr);
\endcode

If you have a preference for putting the resource in GPU (device) memory or CPU (host) memory
on systems with discrete graphics card that have the memories separate, you can use
#VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE or #VMA_MEMORY_USAGE_AUTO_PREFER_HOST.

When using `VMA_MEMORY_USAGE_AUTO*` while you want to map the allocated memory,
you also need to specify one of the host access flags:
#VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or #VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT.
This will help the library decide about preferred memory type to ensure it has `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`
so you can map it.

For example, a staging buffer that will be filled via mapped pointer and then
used as a source of transfer to the buffer described previously can be created like this.
It will likely end up in a memory type that is `HOST_VISIBLE` and `HOST_COHERENT`
but not `HOST_CACHED` (meaning uncached, write-combined) and not `DEVICE_LOCAL` (meaning system RAM).

\code
VkBufferCreateInfo stagingBufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
stagingBufferInfo.size = 65536;
stagingBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

VmaAllocationCreateInfo stagingAllocInfo = {};
stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

VkBuffer stagingBuffer;
VmaAllocation stagingAllocation;
vmaCreateBuffer(allocator, &stagingBufferInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, nullptr);
\endcode

For more examples of creating different kinds of resources, see chapter \ref usage_patterns.
See also: @ref memory_mapping.

Usage values `VMA_MEMORY_USAGE_AUTO*` are legal to use only when the library knows
about the resource being created by having `VkBufferCreateInfo` / `VkImageCreateInfo` passed,
so they work with functions like: vmaCreateBuffer(), vmaCreateImage(), vmaFindMemoryTypeIndexForBufferInfo() etc.
If you allocate raw memory using function vmaAllocateMemory(), you have to use other means of selecting
memory type, as described below.

\note
Old usage values (`VMA_MEMORY_USAGE_GPU_ONLY`, `VMA_MEMORY_USAGE_CPU_ONLY`,
`VMA_MEMORY_USAGE_CPU_TO_GPU`, `VMA_MEMORY_USAGE_GPU_TO_CPU`, `VMA_MEMORY_USAGE_CPU_COPY`)
are still available and work same way as in previous versions of the library
for backward compatibility, but they are deprecated.

\section choosing_memory_type_required_preferred_flags Required and preferred flags

You can specify more detailed requirements by filling members
VmaAllocationCreateInfo::requiredFlags and VmaAllocationCreateInfo::preferredFlags
with a combination of bits from enum `VkMemoryPropertyFlags`. For example,
if you want to create a buffer that will be persistently mapped on host (so it
must be `HOST_VISIBLE`) and preferably will also be `HOST_COHERENT` and `HOST_CACHED`,
use following code:

\code
VmaAllocationCreateInfo allocInfo = {};
allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
allocInfo.preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

VkBuffer buffer;
VmaAllocation allocation;
vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr);
\endcode

A memory type is chosen that has all the required flags and as many preferred
flags set as possible.

Value passed in VmaAllocationCreateInfo::usage is internally converted to a set of required and preferred flags,
plus some extra "magic" (heuristics).

\section choosing_memory_type_explicit_memory_types Explicit memory types

If you inspected memory types available on the physical device and <b>you have
a preference for memory types that you want to use</b>, you can fill member
VmaAllocationCreateInfo::memoryTypeBits. It is a bit mask, where each bit set
means that a memory type with that index is allowed to be used for the
allocation. Special value 0, just like `UINT32_MAX`, means there are no
restrictions to memory type index.

Please note that this member is NOT just a memory type index.
Still you can use it to choose just one, specific memory type.
For example, if you already determined that your buffer should be created in
memory type 2, use following code:

\code
uint32_t memoryTypeIndex = 2;

VmaAllocationCreateInfo allocInfo = {};
allocInfo.memoryTypeBits = 1U << memoryTypeIndex;

VkBuffer buffer;
VmaAllocation allocation;
vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr);
\endcode

You can also use this parameter to <b>exclude some memory types</b>.
If you inspect memory heaps and types available on the current physical device and
you determine that for some reason you don't want to use a specific memory type for the allocation,
you can enable automatic memory type selection but exclude certain memory type or types
by setting all bits of `memoryTypeBits` to 1 except the ones you choose.

\code
// ...
uint32_t excludedMemoryTypeIndex = 2;
VmaAllocationCreateInfo allocInfo = {};
allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
allocInfo.memoryTypeBits = ~(1U << excludedMemoryTypeIndex);
// ...
\endcode


\section choosing_memory_type_custom_memory_pools Custom memory pools

If you allocate from custom memory pool, all the ways of specifying memory
requirements described above are not applicable and the aforementioned members
of VmaAllocationCreateInfo structure are ignored. Memory type is selected
explicitly when creating the pool and then used to make all the allocations from
that pool. For further details, see \ref custom_memory_pools.

\section choosing_memory_type_dedicated_allocations Dedicated allocations

Memory for allocations is reserved out of larger block of `VkDeviceMemory`
allocated from Vulkan internally. That is the main feature of this whole library.
You can still request a separate memory block to be created for an allocation,
just like you would do in a trivial solution without using any allocator.
In that case, a buffer or image is always bound to that memory at offset 0.
This is called a "dedicated allocation".
You can explicitly request it by using flag #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.
The library can also internally decide to use dedicated allocation in some cases, e.g.:

- When the size of the allocation is large.
- When [VK_KHR_dedicated_allocation](@ref vk_khr_dedicated_allocation) extension is enabled
  and it reports that dedicated allocation is required or recommended for the resource.
- When allocation of next big memory block fails due to not enough device memory,
  but allocation with the exact requested size succeeds.


\page memory_mapping Memory mapping

To "map memory" in Vulkan means to obtain a CPU pointer to `VkDeviceMemory`,
to be able to read from it or write to it in CPU code.
Mapping is possible only of memory allocated from a memory type that has
`VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` flag.
Functions `vkMapMemory()`, `vkUnmapMemory()` are designed for this purpose.
You can use them directly with memory allocated by this library,
but it is not recommended because of following issue:
Mapping the same `VkDeviceMemory` block multiple times is illegal - only one mapping at a time is allowed.
This includes mapping disjoint regions. Mapping is not reference-counted internally by Vulkan.
It is also not thread-safe.
Because of this, Vulkan Memory Allocator provides following facilities:

\note If you want to be able to map an allocation, you need to specify one of the flags
#VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or #VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
in VmaAllocationCreateInfo::flags. These flags are required for an allocation to be mappable
when using #VMA_MEMORY_USAGE_AUTO or other `VMA_MEMORY_USAGE_AUTO*` enum values.
For other usage values they are ignored and every such allocation made in `HOST_VISIBLE` memory type is mappable,
but these flags can still be used for consistency.

\section memory_mapping_copy_functions Copy functions

The easiest way to copy data from a host pointer to an allocation is to use convenience function vmaCopyMemoryToAllocation().
It automatically maps the Vulkan memory temporarily (if not already mapped), performs `memcpy`,
and calls `vkFlushMappedMemoryRanges` (if required - if memory type is not `HOST_COHERENT`).

It is also the safest one, because using `memcpy` avoids a risk of accidentally introducing memory reads
(e.g. by doing `pMappedVectors[i] += v`), which may be very slow on memory types that are not `HOST_CACHED`.

\code
struct ConstantBuffer
{
    ...
};
ConstantBuffer constantBufferData = ...

VkBufferCreateInfo bufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
bufCreateInfo.size = sizeof(ConstantBuffer);
bufCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

VkBuffer buf;
VmaAllocation alloc;
vmaCreateBuffer(allocator, &bufCreateInfo, &allocCreateInfo, &buf, &alloc, nullptr);

vmaCopyMemoryToAllocation(allocator, &constantBufferData, alloc, 0, sizeof(ConstantBuffer));
\endcode

Copy in the other direction - from an allocation to a host pointer can be performed the same way using function vmaCopyAllocationToMemory().

\section memory_mapping_mapping_functions Mapping functions

The library provides following functions for mapping of a specific allocation: vmaMapMemory(), vmaUnmapMemory().
They are safer and more convenient to use than standard Vulkan functions.
You can map an allocation multiple times simultaneously - mapping is reference-counted internally.
You can also map different allocations simultaneously regardless of whether they use the same `VkDeviceMemory` block.
The way it is implemented is that the library always maps entire memory block, not just region of the allocation.
For further details, see description of vmaMapMemory() function.
Example:

\code
// Having these objects initialized:
struct ConstantBuffer
{
    ...
};
ConstantBuffer constantBufferData = ...

VmaAllocator allocator = ...
VkBuffer constantBuffer = ...
VmaAllocation constantBufferAllocation = ...

// You can map and fill your buffer using following code:

void* mappedData;
vmaMapMemory(allocator, constantBufferAllocation, &mappedData);
memcpy(mappedData, &constantBufferData, sizeof(constantBufferData));
vmaUnmapMemory(allocator, constantBufferAllocation);
\endcode

When mapping, you may see a warning from Vulkan validation layer similar to this one:

<i>Mapping an image with layout VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL can result in undefined behavior if this memory is used by the device. Only GENERAL or PREINITIALIZED should be used.</i>

It happens because the library maps entire `VkDeviceMemory` block, where different
types of images and buffers may end up together, especially on GPUs with unified memory like Intel.
You can safely ignore it if you are sure you access only memory of the intended
object that you wanted to map.


\section memory_mapping_persistently_mapped_memory Persistently mapped memory

Keeping your memory persistently mapped is generally OK in Vulkan.
You don't need to unmap it before using its data on the GPU.
The library provides a special feature designed for that:
Allocations made with #VMA_ALLOCATION_CREATE_MAPPED_BIT flag set in
VmaAllocationCreateInfo::flags stay mapped all the time,
so you can just access CPU pointer to it any time
without a need to call any "map" or "unmap" function.
Example:

\code
VkBufferCreateInfo bufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
bufCreateInfo.size = sizeof(ConstantBuffer);
bufCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
    VMA_ALLOCATION_CREATE_MAPPED_BIT;

VkBuffer buf;
VmaAllocation alloc;
VmaAllocationInfo allocInfo;
vmaCreateBuffer(allocator, &bufCreateInfo, &allocCreateInfo, &buf, &alloc, &allocInfo);

// Buffer is already mapped. You can access its memory.
memcpy(allocInfo.pMappedData, &constantBufferData, sizeof(constantBufferData));
\endcode

\note #VMA_ALLOCATION_CREATE_MAPPED_BIT by itself doesn't guarantee that the allocation will end up
in a mappable memory type.
For this, you need to also specify #VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or
#VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT.
#VMA_ALLOCATION_CREATE_MAPPED_BIT only guarantees that if the memory is `HOST_VISIBLE`, the allocation will be mapped on creation.
For an example of how to make use of this fact, see section \ref usage_patterns_advanced_data_uploading.

\section memory_mapping_cache_control Cache flush and invalidate

Memory in Vulkan doesn't need to be unmapped before using it on GPU,
but unless a memory types has `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT` flag set,
you need to manually **invalidate** cache before reading of mapped pointer
and **flush** cache after writing to mapped pointer.
Map/unmap operations don't do that automatically.
Vulkan provides following functions for this purpose `vkFlushMappedMemoryRanges()`,
`vkInvalidateMappedMemoryRanges()`, but this library provides more convenient
functions that refer to given allocation object: vmaFlushAllocation(),
vmaInvalidateAllocation(),
or multiple objects at once: vmaFlushAllocations(), vmaInvalidateAllocations().

Regions of memory specified for flush/invalidate must be aligned to
`VkPhysicalDeviceLimits::nonCoherentAtomSize`. This is automatically ensured by the library.
In any memory type that is `HOST_VISIBLE` but not `HOST_COHERENT`, all allocations
within blocks are aligned to this value, so their offsets are always multiply of
`nonCoherentAtomSize` and two different allocations never share same "line" of this size.

Also, Windows drivers from all 3 PC GPU vendors (AMD, Intel, NVIDIA)
currently provide `HOST_COHERENT` flag on all memory types that are
`HOST_VISIBLE`, so on PC you may not need to bother.


\page staying_within_budget Staying within budget

When developing a graphics-intensive game or program, it is important to avoid allocating
more GPU memory than it is physically available. When the memory is over-committed,
various bad things can happen, depending on the specific GPU, graphics driver, and
operating system:

- It may just work without any problems.
- The application may slow down because some memory blocks are moved to system RAM
  and the GPU has to access them through PCI Express bus.
- A new allocation may take very long time to complete, even few seconds, and possibly
  freeze entire system.
- The new allocation may fail with `VK_ERROR_OUT_OF_DEVICE_MEMORY`.
- It may even result in GPU crash (TDR), observed as `VK_ERROR_DEVICE_LOST`
  returned somewhere later.

\section staying_within_budget_querying_for_budget Querying for budget

To query for current memory usage and available budget, use function vmaGetHeapBudgets().
Returned structure #VmaBudget contains quantities expressed in bytes, per Vulkan memory heap.

Please note that this function returns different information and works faster than
vmaCalculateStatistics(). vmaGetHeapBudgets() can be called every frame or even before every
allocation, while vmaCalculateStatistics() is intended to be used rarely,
only to obtain statistical information, e.g. for debugging purposes.

It is recommended to use <b>VK_EXT_memory_budget</b> device extension to obtain information
about the budget from Vulkan device. VMA is able to use this extension automatically.
When not enabled, the allocator behaves same way, but then it estimates current usage
and available budget based on its internal information and Vulkan memory heap sizes,
which may be less precise. In order to use this extension:

1. Make sure extensions VK_EXT_memory_budget and VK_KHR_get_physical_device_properties2
   required by it are available and enable them. Please note that the first is a device
   extension and the second is instance extension!
2. Use flag #VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT when creating #VmaAllocator object.
3. Make sure to call vmaSetCurrentFrameIndex() every frame. Budget is queried from
   Vulkan inside of it to avoid overhead of querying it with every allocation.

\section staying_within_budget_controlling_memory_usage Controlling memory usage

There are many ways in which you can try to stay within the budget.

First, when making new allocation requires allocating a new memory block, the library
tries not to exceed the budget automatically. If a block with default recommended size
(e.g. 256 MB) would go over budget, a smaller block is allocated, possibly even
dedicated memory for just this resource.

If the size of the requested resource plus current memory usage is more than the
budget, by default the library still tries to create it, leaving it to the Vulkan
implementation whether the allocation succeeds or fails. You can change this behavior
by using #VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT flag. With it, the allocation is
not made if it would exceed the budget or if the budget is already exceeded.
VMA then tries to make the allocation from the next eligible Vulkan memory type.
If all of them fail, the call then fails with `VK_ERROR_OUT_OF_DEVICE_MEMORY`.
Example usage pattern may be to pass the #VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT flag
when creating resources that are not essential for the application (e.g. the texture
of a specific object) and not to pass it when creating critically important resources
(e.g. render targets).

On AMD graphics cards there is a custom vendor extension available: <b>VK_AMD_memory_overallocation_behavior</b>
that allows to control the behavior of the Vulkan implementation in out-of-memory cases -
whether it should fail with an error code or still allow the allocation.
Usage of this extension involves only passing extra structure on Vulkan device creation,
so it is out of scope of this library.

Finally, you can also use #VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT flag to make sure
a new allocation is created only when it fits inside one of the existing memory blocks.
If it would require to allocate a new block, if fails instead with `VK_ERROR_OUT_OF_DEVICE_MEMORY`.
This also ensures that the function call is very fast because it never goes to Vulkan
to obtain a new block.

\note Creating \ref custom_memory_pools with VmaPoolCreateInfo::minBlockCount
set to more than 0 will currently try to allocate memory blocks without checking whether they
fit within budget.


\page resource_aliasing Resource aliasing (overlap)

New explicit graphics APIs (Vulkan and Direct3D 12), thanks to manual memory
management, give an opportunity to alias (overlap) multiple resources in the
same region of memory - a feature not available in the old APIs (Direct3D 11, OpenGL).
It can be useful to save video memory, but it must be used with caution.

For example, if you know the flow of your whole render frame in advance, you
are going to use some intermediate textures or buffers only during a small range of render passes,
and you know these ranges don't overlap in time, you can bind these resources to
the same place in memory, even if they have completely different parameters (width, height, format etc.).

![Resource aliasing (overlap)](../gfx/Aliasing.png)

Such scenario is possible using VMA, but you need to create your images manually.
Then you need to calculate parameters of an allocation to be made using formula:

- allocation size = max(size of each image)
- allocation alignment = max(alignment of each image)
- allocation memoryTypeBits = bitwise AND(memoryTypeBits of each image)

Following example shows two different images bound to the same place in memory,
allocated to fit largest of them.

\code
// A 512x512 texture to be sampled.
VkImageCreateInfo img1CreateInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
img1CreateInfo.imageType = VK_IMAGE_TYPE_2D;
img1CreateInfo.extent.width = 512;
img1CreateInfo.extent.height = 512;
img1CreateInfo.extent.depth = 1;
img1CreateInfo.mipLevels = 10;
img1CreateInfo.arrayLayers = 1;
img1CreateInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
img1CreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
img1CreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
img1CreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
img1CreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;

// A full screen texture to be used as color attachment.
VkImageCreateInfo img2CreateInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
img2CreateInfo.imageType = VK_IMAGE_TYPE_2D;
img2CreateInfo.extent.width = 1920;
img2CreateInfo.extent.height = 1080;
img2CreateInfo.extent.depth = 1;
img2CreateInfo.mipLevels = 1;
img2CreateInfo.arrayLayers = 1;
img2CreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
img2CreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
img2CreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
img2CreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
img2CreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;

VkImage img1;
res = vkCreateImage(device, &img1CreateInfo, nullptr, &img1);
VkImage img2;
res = vkCreateImage(device, &img2CreateInfo, nullptr, &img2);

VkMemoryRequirements img1MemReq;
vkGetImageMemoryRequirements(device, img1, &img1MemReq);
VkMemoryRequirements img2MemReq;
vkGetImageMemoryRequirements(device, img2, &img2MemReq);

VkMemoryRequirements finalMemReq = {};
finalMemReq.size = std::max(img1MemReq.size, img2MemReq.size);
finalMemReq.alignment = std::max(img1MemReq.alignment, img2MemReq.alignment);
finalMemReq.memoryTypeBits = img1MemReq.memoryTypeBits & img2MemReq.memoryTypeBits;
// Validate if(finalMemReq.memoryTypeBits != 0)

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

VmaAllocation alloc;
res = vmaAllocateMemory(allocator, &finalMemReq, &allocCreateInfo, &alloc, nullptr);

res = vmaBindImageMemory(allocator, alloc, img1);
res = vmaBindImageMemory(allocator, alloc, img2);

// You can use img1, img2 here, but not at the same time!

vmaFreeMemory(allocator, alloc);
vkDestroyImage(allocator, img2, nullptr);
vkDestroyImage(allocator, img1, nullptr);
\endcode

VMA also provides convenience functions that create a buffer or image and bind it to memory
represented by an existing #VmaAllocation:
vmaCreateAliasingBuffer(), vmaCreateAliasingBuffer2(),
vmaCreateAliasingImage(), vmaCreateAliasingImage2().
Versions with "2" offer additional parameter `allocationLocalOffset`.

Remember that using resources that alias in memory requires proper synchronization.
You need to issue a memory barrier to make sure commands that use `img1` and `img2`
don't overlap on GPU timeline.
You also need to treat a resource after aliasing as uninitialized - containing garbage data.
For example, if you use `img1` and then want to use `img2`, you need to issue
an image memory barrier for `img2` with `oldLayout` = `VK_IMAGE_LAYOUT_UNDEFINED`.

Additional considerations:

- Vulkan also allows to interpret contents of memory between aliasing resources consistently in some cases.
See chapter 11.8. "Memory Aliasing" of Vulkan specification or `VK_IMAGE_CREATE_ALIAS_BIT` flag.
- You can create more complex layout where different images and buffers are bound
at different offsets inside one large allocation. For example, one can imagine
a big texture used in some render passes, aliasing with a set of many small buffers
used between in some further passes. To bind a resource at non-zero offset in an allocation,
use vmaBindBufferMemory2() / vmaBindImageMemory2().
- Before allocating memory for the resources you want to alias, check `memoryTypeBits`
returned in memory requirements of each resource to make sure the bits overlap.
Some GPUs may expose multiple memory types suitable e.g. only for buffers or
images with `COLOR_ATTACHMENT` usage, so the sets of memory types supported by your
resources may be disjoint. Aliasing them is not possible in that case.


\page custom_memory_pools Custom memory pools

A memory pool contains a number of `VkDeviceMemory` blocks.
The library automatically creates and manages default pool for each memory type available on the device.
Default memory pool automatically grows in size.
Size of allocated blocks is also variable and managed automatically.
You are using default pools whenever you leave VmaAllocationCreateInfo::pool = null.

You can create custom pool and allocate memory out of it.
It can be useful if you want to:

- Keep certain kind of allocations separate from others.
- Enforce particular, fixed size of Vulkan memory blocks.
- Limit maximum amount of Vulkan memory allocated for that pool.
- Reserve minimum or fixed amount of Vulkan memory always preallocated for that pool.
- Use extra parameters for a set of your allocations that are available in #VmaPoolCreateInfo but not in
  #VmaAllocationCreateInfo - e.g., custom minimum alignment, custom `pNext` chain.
- Perform defragmentation on a specific subset of your allocations.

To use custom memory pools:

-# Fill VmaPoolCreateInfo structure.
-# Call vmaCreatePool() to obtain #VmaPool handle.
-# When making an allocation, set VmaAllocationCreateInfo::pool to this handle.
   You don't need to specify any other parameters of this structure, like `usage`.

Example:

\code
// Find memoryTypeIndex for the pool.
VkBufferCreateInfo sampleBufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
sampleBufCreateInfo.size = 0x10000; // Doesn't matter.
sampleBufCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

VmaAllocationCreateInfo sampleAllocCreateInfo = {};
sampleAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

uint32_t memTypeIndex;
VkResult res = vmaFindMemoryTypeIndexForBufferInfo(allocator,
    &sampleBufCreateInfo, &sampleAllocCreateInfo, &memTypeIndex);
// Check res...

// Create a pool that can have at most 2 blocks, 128 MiB each.
VmaPoolCreateInfo poolCreateInfo = {};
poolCreateInfo.memoryTypeIndex = memTypeIndex;
poolCreateInfo.blockSize = 128ULL * 1024 * 1024;
poolCreateInfo.maxBlockCount = 2;

VmaPool pool;
res = vmaCreatePool(allocator, &poolCreateInfo, &pool);
// Check res...

// Allocate a buffer out of it.
VkBufferCreateInfo bufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
bufCreateInfo.size = 1024;
bufCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.pool = pool;

VkBuffer buf;
VmaAllocation alloc;
res = vmaCreateBuffer(allocator, &bufCreateInfo, &allocCreateInfo, &buf, &alloc, nullptr);
// Check res...
\endcode

You have to free all allocations made from this pool before destroying it.

\code
vmaDestroyBuffer(allocator, buf, alloc);
vmaDestroyPool(allocator, pool);
\endcode

New versions of this library support creating dedicated allocations in custom pools.
It is supported only when VmaPoolCreateInfo::blockSize = 0.
To use this feature, set VmaAllocationCreateInfo::pool to the pointer to your custom pool and
VmaAllocationCreateInfo::flags to #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.


\section custom_memory_pools_MemTypeIndex Choosing memory type index

When creating a pool, you must explicitly specify memory type index.
To find the one suitable for your buffers or images, you can use helper functions
vmaFindMemoryTypeIndexForBufferInfo(), vmaFindMemoryTypeIndexForImageInfo().
You need to provide structures with example parameters of buffers or images
that you are going to create in that pool.

\code
VkBufferCreateInfo exampleBufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
exampleBufCreateInfo.size = 1024; // Doesn't matter
exampleBufCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

uint32_t memTypeIndex;
vmaFindMemoryTypeIndexForBufferInfo(allocator, &exampleBufCreateInfo, &allocCreateInfo, &memTypeIndex);

VmaPoolCreateInfo poolCreateInfo = {};
poolCreateInfo.memoryTypeIndex = memTypeIndex;
// ...
\endcode

When creating buffers/images allocated in that pool, provide following parameters:

- `VkBufferCreateInfo`: Prefer to pass same parameters as above.
  Otherwise you risk creating resources in a memory type that is not suitable for them, which may result in undefined behavior.
  Using different `VK_BUFFER_USAGE_` flags may work, but you shouldn't create images in a pool intended for buffers
  or the other way around.
- VmaAllocationCreateInfo: You don't need to pass same parameters. Fill only `pool` member.
  Other members are ignored anyway.


\section custom_memory_pools_when_not_use When not to use custom pools

Custom pools are commonly overused by VMA users.
While it may feel natural to keep some logical groups of resources separate in memory,
in most cases it does more harm than good.
Using custom pool shouldn't be your first choice.
Instead, please make all allocations from default pools first and only use custom pools
if you can prove and measure that it is beneficial in some way,
e.g. it results in lower memory usage, better performance, etc.

Using custom pools has disadvantages:

- Each pool has its own collection of `VkDeviceMemory` blocks.
  Some of them may be partially or even completely empty.
  Spreading allocations across multiple pools increases the amount of wasted (allocated but unbound) memory.
- You must manually choose specific memory type to be used by a custom pool (set as VmaPoolCreateInfo::memoryTypeIndex).
  When using default pools, best memory type for each of your allocations can be selected automatically
  using a carefully design algorithm that works across all kinds of GPUs.
- If an allocation from a custom pool at specific memory type fails, entire allocation operation returns failure.
  When using default pools, VMA tries another compatible memory type.
- If you set VmaPoolCreateInfo::blockSize != 0, each memory block has the same size,
  while default pools start from small blocks and only allocate next blocks larger and larger
  up to the preferred block size.

Many of the common concerns can be addressed in a different way than using custom pools:

- If you want to keep your allocations of certain size (small versus large) or certain lifetime (transient versus long lived)
  separate, you likely don't need to.
  VMA uses a high quality allocation algorithm that manages memory well in various cases.
  Please measure and check if using custom pools provides a benefit.
- If you want to keep your images and buffers separate, you don't need to.
  VMA respects `bufferImageGranularity` limit automatically.
- If you want to keep your mapped and not mapped allocations separate, you don't need to.
  VMA respects `nonCoherentAtomSize` limit automatically.
  It also maps only those `VkDeviceMemory` blocks that need to map any allocation.
  It even tries to keep mappable and non-mappable allocations in separate blocks to minimize the amount of mapped memory.
- If you want to choose a custom size for the default memory block, you can set it globally instead
  using VmaAllocatorCreateInfo::preferredLargeHeapBlockSize.
- If you want to select specific memory type for your allocation,
  you can set VmaAllocationCreateInfo::memoryTypeBits to `(1U << myMemoryTypeIndex)` instead.
- If you need to create a buffer with certain minimum alignment, you can still do it
  using default pools with dedicated function vmaCreateBufferWithAlignment().


\section linear_algorithm Linear allocation algorithm

Each Vulkan memory block managed by this library has accompanying metadata that
keeps track of used and unused regions. By default, the metadata structure and
algorithm tries to find best place for new allocations among free regions to
optimize memory usage. This way you can allocate and free objects in any order.

![Default allocation algorithm](../gfx/Linear_allocator_1_algo_default.png)

Sometimes there is a need to use simpler, linear allocation algorithm. You can
create custom pool that uses such algorithm by adding flag
#VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT to VmaPoolCreateInfo::flags while creating
#VmaPool object. Then an alternative metadata management is used. It always
creates new allocations after last one and doesn't reuse free regions after
allocations freed in the middle. It results in better allocation performance and
less memory consumed by metadata.

![Linear allocation algorithm](../gfx/Linear_allocator_2_algo_linear.png)

With this one flag, you can create a custom pool that can be used in many ways:
free-at-once, stack, double stack, and ring buffer. See below for details.
You don't need to specify explicitly which of these options you are going to use - it is detected automatically.

\subsection linear_algorithm_free_at_once Free-at-once

In a pool that uses linear algorithm, you still need to free all the allocations
individually, e.g. by using vmaFreeMemory() or vmaDestroyBuffer(). You can free
them in any order. New allocations are always made after last one - free space
in the middle is not reused. However, when you release all the allocation and
the pool becomes empty, allocation starts from the beginning again. This way you
can use linear algorithm to speed up creation of allocations that you are going
to release all at once.

![Free-at-once](../gfx/Linear_allocator_3_free_at_once.png)

This mode is also available for pools created with VmaPoolCreateInfo::maxBlockCount
value that allows multiple memory blocks.

\subsection linear_algorithm_stack Stack

When you free an allocation that was created last, its space can be reused.
Thanks to this, if you always release allocations in the order opposite to their
creation (LIFO - Last In First Out), you can achieve behavior of a stack.

![Stack](../gfx/Linear_allocator_4_stack.png)

This mode is also available for pools created with VmaPoolCreateInfo::maxBlockCount
value that allows multiple memory blocks.

\subsection linear_algorithm_double_stack Double stack

The space reserved by a custom pool with linear algorithm may be used by two
stacks:

- First, default one, growing up from offset 0.
- Second, "upper" one, growing down from the end towards lower offsets.

To make allocation from the upper stack, add flag #VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT
to VmaAllocationCreateInfo::flags.

![Double stack](../gfx/Linear_allocator_7_double_stack.png)

Double stack is available only in pools with one memory block -
VmaPoolCreateInfo::maxBlockCount must be 1. Otherwise behavior is undefined.

When the two stacks' ends meet so there is not enough space between them for a
new allocation, such allocation fails with usual
`VK_ERROR_OUT_OF_DEVICE_MEMORY` error.

\subsection linear_algorithm_ring_buffer Ring buffer

When you free some allocations from the beginning and there is not enough free space
for a new one at the end of a pool, allocator's "cursor" wraps around to the
beginning and starts allocation there. Thanks to this, if you always release
allocations in the same order as you created them (FIFO - First In First Out),
you can achieve behavior of a ring buffer / queue.

![Ring buffer](../gfx/Linear_allocator_5_ring_buffer.png)

Ring buffer is available only in pools with one memory block -
VmaPoolCreateInfo::maxBlockCount must be 1. Otherwise behavior is undefined.

\note \ref defragmentation is not supported in custom pools created with #VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT.


\page defragmentation Defragmentation

Interleaved allocations and deallocations of many objects of varying size can
cause fragmentation over time, which can lead to a situation where the library is unable
to find a continuous range of free memory for a new allocation despite there is
enough free space, just scattered across many small free ranges between existing
allocations.

To mitigate this problem, you can use defragmentation feature.
It doesn't happen automatically though and needs your cooperation,
because VMA is a low level library that only allocates memory.
It cannot recreate buffers and images in a new place as it doesn't remember the contents of `VkBufferCreateInfo` / `VkImageCreateInfo` structures.
It cannot copy their contents as it doesn't record any commands to a command buffer.

Example:

\code
VmaDefragmentationInfo defragInfo = {};
defragInfo.pool = myPool;
defragInfo.flags = VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FAST_BIT;

VmaDefragmentationContext defragCtx;
VkResult res = vmaBeginDefragmentation(allocator, &defragInfo, &defragCtx);
// Check res...

for(;;)
{
    VmaDefragmentationPassMoveInfo pass;
    res = vmaBeginDefragmentationPass(allocator, defragCtx, &pass);
    if(res == VK_SUCCESS)
        break;
    else if(res != VK_INCOMPLETE)
        // Handle error...

    for(uint32_t i = 0; i < pass.moveCount; ++i)
    {
        // Inspect pass.pMoves[i].srcAllocation, identify what buffer/image it represents.
        VmaAllocationInfo allocInfo;
        vmaGetAllocationInfo(allocator, pass.pMoves[i].srcAllocation, &allocInfo);
        MyEngineResourceData* resData = (MyEngineResourceData*)allocInfo.pUserData;

        // Recreate and bind this buffer/image at: pass.pMoves[i].dstMemory, pass.pMoves[i].dstOffset.
        VkImageCreateInfo imgCreateInfo = ...
        VkImage newImg;
        res = vkCreateImage(device, &imgCreateInfo, nullptr, &newImg);
        // Check res...
        res = vmaBindImageMemory(allocator, pass.pMoves[i].dstTmpAllocation, newImg);
        // Check res...

        // Issue a vkCmdCopyBuffer/vkCmdCopyImage to copy its content to the new place.
        vkCmdCopyImage(cmdBuf, resData->img, ..., newImg, ...);
    }

    // Make sure the copy commands finished executing.
    vkWaitForFences(...);

    // destroy old buffers/images bound with pass.pMoves[i].srcAllocation.
    for(uint32_t i = 0; i < pass.moveCount; ++i)
    {
        // ...
        vkDestroyImage(device, resData->img, nullptr);
    }

    // Update appropriate descriptors to point to the new places...

    res = vmaEndDefragmentationPass(allocator, defragCtx, &pass);
    if(res == VK_SUCCESS)
        break;
    else if(res != VK_INCOMPLETE)
        // Handle error...
}

vmaEndDefragmentation(allocator, defragCtx, nullptr);
\endcode

Although functions like vmaCreateBuffer(), vmaCreateImage(), vmaDestroyBuffer(), vmaDestroyImage()
create/destroy an allocation and a buffer/image at once, these are just a shortcut for
creating the resource, allocating memory, and binding them together.
Defragmentation works on memory allocations only. You must handle the rest manually.
Defragmentation is an iterative process that should repreat "passes" as long as related functions
return `VK_INCOMPLETE` not `VK_SUCCESS`.
In each pass:

1. vmaBeginDefragmentationPass() function call:
   - Calculates and returns the list of allocations to be moved in this pass.
     Note this can be a time-consuming process.
   - Reserves destination memory for them by creating temporary destination allocations
     that you can query for their `VkDeviceMemory` + offset using vmaGetAllocationInfo().
2. Inside the pass, **you should**:
   - Inspect the returned list of allocations to be moved.
   - Create new buffers/images and bind them at the returned destination temporary allocations.
   - Copy data from source to destination resources if necessary.
   - destroy the source buffers/images, but NOT their allocations.
3. vmaEndDefragmentationPass() function call:
   - Frees the source memory reserved for the allocations that are moved.
   - Modifies source #VmaAllocation objects that are moved to point to the destination reserved memory.
   - Frees `VkDeviceMemory` blocks that became empty.

Unlike in previous iterations of the defragmentation API, there is no list of "movable" allocations passed as a parameter.
Defragmentation algorithm tries to move all suitable allocations.
You can, however, refuse to move some of them inside a defragmentation pass, by setting
`pass.pMoves[i].operation` to #VMA_DEFRAGMENTATION_MOVE_OPERATION_IGNORE.
This is not recommended and may result in suboptimal packing of the allocations after defragmentation.
If you cannot ensure any allocation can be moved, it is better to keep movable allocations separate in a custom pool.

Inside a pass, for each allocation that should be moved:

- You should copy its data from the source to the destination place by calling e.g. `vkCmdCopyBuffer()`, `vkCmdCopyImage()`.
  - You need to make sure these commands finished executing before destroying the source buffers/images and before calling vmaEndDefragmentationPass().
- If a resource doesn't contain any meaningful data, e.g. it is a transient color attachment image to be cleared,
  filled, and used temporarily in each rendering frame, you can just recreate this image
  without copying its data.
- If the resource is in `HOST_VISIBLE` and `HOST_CACHED` memory, you can copy its data on the CPU
  using `memcpy()`.
- If you cannot move the allocation, you can set `pass.pMoves[i].operation` to #VMA_DEFRAGMENTATION_MOVE_OPERATION_IGNORE.
  This will cancel the move.
  - vmaEndDefragmentationPass() will then free the destination memory
    not the source memory of the allocation, leaving it unchanged.
- If you decide the allocation is unimportant and can be destroyed instead of moved (e.g. it wasn't used for long time),
  you can set `pass.pMoves[i].operation` to #VMA_DEFRAGMENTATION_MOVE_OPERATION_DESTROY.
  - vmaEndDefragmentationPass() will then free both source and destination memory, and will destroy the source #VmaAllocation object.

You can defragment a specific custom pool by setting VmaDefragmentationInfo::pool
(like in the example above) or all the default pools by setting this member to null.

Defragmentation is always performed in each pool separately.
Allocations are never moved between different Vulkan memory types.
The size of the destination memory reserved for a moved allocation is the same as the original one.
Alignment of an allocation as it was determined using `vkGetBufferMemoryRequirements()` etc. is also respected after defragmentation.
Buffers/images should be recreated with the same `VkBufferCreateInfo` / `VkImageCreateInfo` parameters as the original ones.

You can perform the defragmentation incrementally to limit the number of allocations and bytes to be moved
in each pass, e.g. to call it in sync with render frames and not to experience too big hitches.
See members: VmaDefragmentationInfo::maxBytesPerPass, VmaDefragmentationInfo::maxAllocationsPerPass.

It is also safe to perform the defragmentation asynchronously to render frames and other Vulkan and VMA
usage, possibly from multiple threads, with the exception that allocations
returned in VmaDefragmentationPassMoveInfo::pMoves shouldn't be destroyed until the defragmentation pass is ended.

<b>Mapping</b> is preserved on allocations that are moved during defragmentation.
Whether through #VMA_ALLOCATION_CREATE_MAPPED_BIT or vmaMapMemory(), the allocations
are mapped at their new place. Of course, pointer to the mapped data changes, so it needs to be queried
using VmaAllocationInfo::pMappedData.

\note Defragmentation is not supported in custom pools created with #VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT.


\page statistics Statistics

This library contains several functions that return information about its internal state,
especially the amount of memory allocated from Vulkan.

\section statistics_numeric_statistics Numeric statistics

If you need to obtain basic statistics about memory usage per heap, together with current budget,
you can call function vmaGetHeapBudgets() and inspect structure #VmaBudget.
This is useful to keep track of memory usage and stay within budget
(see also \ref staying_within_budget).
Example:

\code
uint32_t heapIndex = ...

VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
vmaGetHeapBudgets(allocator, budgets);

printf("My heap currently has %u allocations taking %llu B,\n",
    budgets[heapIndex].statistics.allocationCount,
    budgets[heapIndex].statistics.allocationBytes);
printf("allocated out of %u Vulkan device memory blocks taking %llu B,\n",
    budgets[heapIndex].statistics.blockCount,
    budgets[heapIndex].statistics.blockBytes);
printf("Vulkan reports total usage %llu B with budget %llu B.\n",
    budgets[heapIndex].usage,
    budgets[heapIndex].budget);
\endcode

You can query for more detailed statistics per memory heap, type, and totals,
including minimum and maximum allocation size and unused range size,
by calling function vmaCalculateStatistics() and inspecting structure #VmaTotalStatistics.
This function is slower though, as it has to traverse all the internal data structures,
so it should be used only for debugging purposes.

You can query for statistics of a custom pool using function vmaGetPoolStatistics()
or vmaCalculatePoolStatistics().

You can query for information about a specific allocation using function vmaGetAllocationInfo().
It fill structure #VmaAllocationInfo.

\section statistics_json_dump JSON dump

You can dump internal state of the allocator to a string in JSON format using function vmaBuildStatsString().
The result is guaranteed to be correct JSON.
It uses ANSI encoding.
Any strings provided by user (see [Allocation names](@ref allocation_names))
are copied as-is and properly escaped for JSON, so if they use UTF-8, ISO-8859-2 or any other encoding,
this JSON string can be treated as using this encoding.
It must be freed using function vmaFreeStatsString().

The format of this JSON string is not part of official documentation of the library,
but it will not change in backward-incompatible way without increasing library major version number
and appropriate mention in changelog.

The JSON string contains all the data that can be obtained using vmaCalculateStatistics().
It can also contain detailed map of allocated memory blocks and their regions -
free and occupied by allocations.
This allows e.g. to visualize the memory or assess fragmentation.


\page allocation_annotation Allocation names and user data

\section allocation_user_data Allocation user data

You can annotate allocations with your own information, e.g. for debugging purposes.
To do that, fill VmaAllocationCreateInfo::pUserData field when creating
an allocation. It is an opaque `void*` pointer. You can use it e.g. as a pointer,
some handle, index, key, ordinal number or any other value that would associate
the allocation with your custom metadata.
It is useful to identify appropriate data structures in your engine given #VmaAllocation,
e.g. when doing \ref defragmentation.

\code
VkBufferCreateInfo bufCreateInfo = ...

MyBufferMetadata* pMetadata = CreateBufferMetadata();

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
allocCreateInfo.pUserData = pMetadata;

VkBuffer buffer;
VmaAllocation allocation;
vmaCreateBuffer(allocator, &bufCreateInfo, &allocCreateInfo, &buffer, &allocation, nullptr);
\endcode

The pointer may be later retrieved as VmaAllocationInfo::pUserData:

\code
VmaAllocationInfo allocInfo;
vmaGetAllocationInfo(allocator, allocation, &allocInfo);
MyBufferMetadata* pMetadata = (MyBufferMetadata*)allocInfo.pUserData;
\endcode

It can also be changed using function vmaSetAllocationUserData().

Values of (non-zero) allocations' `pUserData` are printed in JSON report created by
vmaBuildStatsString() in hexadecimal form.

\section allocation_names Allocation names

An allocation can also carry a null-terminated string, giving a name to the allocation.
To set it, call vmaSetAllocationName().
The library creates internal copy of the string, so the pointer you pass doesn't need
to be valid for whole lifetime of the allocation. You can free it after the call.

\code
std::string imageName = "Texture: ";
imageName += fileName;
vmaSetAllocationName(allocator, allocation, imageName.c_str());
\endcode

The string can be later retrieved by inspecting VmaAllocationInfo::pName.
It is also printed in JSON report created by vmaBuildStatsString().

\note Setting string name to VMA allocation doesn't automatically set it to the Vulkan buffer or image created with it.
You must do it manually using an extension like VK_EXT_debug_utils, which is independent of this library.


\page virtual_allocator Virtual allocator

As an extra feature, the core allocation algorithm of the library is exposed through a simple and convenient API of "virtual allocator".
It doesn't allocate any real GPU memory. It just keeps track of used and free regions of a "virtual block".
You can use it to allocate your own memory or other objects, even completely unrelated to Vulkan.
A common use case is sub-allocation of pieces of one large GPU buffer.

\section virtual_allocator_creating_virtual_block Creating virtual block

To use this functionality, there is no main "allocator" object.
You don't need to have #VmaAllocator object created.
All you need to do is to create a separate #VmaVirtualBlock object for each block of memory you want to be managed by the allocator:

-# Fill in #VmaVirtualBlockCreateInfo structure.
-# Call vmaCreateVirtualBlock(). Get new #VmaVirtualBlock object.

Example:

\code
VmaVirtualBlockCreateInfo blockCreateInfo = {};
blockCreateInfo.size = 1048576; // 1 MB

VmaVirtualBlock block;
VkResult res = vmaCreateVirtualBlock(&blockCreateInfo, &block);
\endcode

\section virtual_allocator_making_virtual_allocations Making virtual allocations

#VmaVirtualBlock object contains internal data structure that keeps track of free and occupied regions
using the same code as the main Vulkan memory allocator.
Similarly to #VmaAllocation for standard GPU allocations, there is #VmaVirtualAllocation type
that represents an opaque handle to an allocation within the virtual block.

In order to make such allocation:

-# Fill in #VmaVirtualAllocationCreateInfo structure.
-# Call vmaVirtualAllocate(). Get new #VmaVirtualAllocation object that represents the allocation.
   You can also receive `VkDeviceSize offset` that was assigned to the allocation.

Example:

\code
VmaVirtualAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.size = 4096; // 4 KB

VmaVirtualAllocation alloc;
VkDeviceSize offset;
res = vmaVirtualAllocate(block, &allocCreateInfo, &alloc, &offset);
if(res == VK_SUCCESS)
{
    // Use the 4 KB of your memory starting at offset.
}
else
{
    // Allocation failed - no space for it could be found. Handle this error!
}
\endcode

\section virtual_allocator_deallocation Deallocation

When no longer needed, an allocation can be freed by calling vmaVirtualFree().
You can only pass to this function an allocation that was previously returned by vmaVirtualAllocate()
called for the same #VmaVirtualBlock.

When whole block is no longer needed, the block object can be released by calling vmaDestroyVirtualBlock().
All allocations must be freed before the block is destroyed, which is checked internally by an assert.
However, if you don't want to call vmaVirtualFree() for each allocation, you can use vmaClearVirtualBlock() to free them all at once -
a feature not available in normal Vulkan memory allocator. Example:

\code
vmaVirtualFree(block, alloc);
vmaDestroyVirtualBlock(block);
\endcode

\section virtual_allocator_allocation_parameters Allocation parameters

You can attach a custom pointer to each allocation by using vmaSetVirtualAllocationUserData().
Its default value is null.
It can be used to store any data that needs to be associated with that allocation - e.g. an index, a handle, or a pointer to some
larger data structure containing more information. Example:

\code
struct CustomAllocData
{
    std::string _alloc_name;
};
CustomAllocData* allocData = new CustomAllocData();
allocData->_alloc_name = "My allocation 1";
vmaSetVirtualAllocationUserData(block, alloc, allocData);
\endcode

The pointer can later be fetched, along with allocation offset and size, by passing the allocation handle to function
vmaGetVirtualAllocationInfo() and inspecting returned structure #VmaVirtualAllocationInfo.
If you allocated a new object to be used as the custom pointer, don't forget to delete that object before freeing the allocation!
Example:

\code
VmaVirtualAllocationInfo allocInfo;
vmaGetVirtualAllocationInfo(block, alloc, &allocInfo);
delete (CustomAllocData*)allocInfo.pUserData;

vmaVirtualFree(block, alloc);
\endcode

\section virtual_allocator_alignment_and_units Alignment and units

It feels natural to express sizes and offsets in bytes.
If an offset of an allocation needs to be aligned to a multiply of some number (e.g. 4 bytes), you can fill optional member
VmaVirtualAllocationCreateInfo::alignment to request it. Example:

\code
VmaVirtualAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.size = 4096; // 4 KB
allocCreateInfo.alignment = 4; // Returned offset must be a multiply of 4 B

VmaVirtualAllocation alloc;
res = vmaVirtualAllocate(block, &allocCreateInfo, &alloc, nullptr);
\endcode

Alignments of different allocations made from one block may vary.
However, if all alignments and sizes are always multiply of some size e.g. 4 B or `sizeof(MyDataStruct)`,
you can express all sizes, alignments, and offsets in multiples of that size instead of individual bytes.
It might be more convenient, but you need to make sure to use this new unit consistently in all the places:

- VmaVirtualBlockCreateInfo::size
- VmaVirtualAllocationCreateInfo::size and VmaVirtualAllocationCreateInfo::alignment
- Using offset returned by vmaVirtualAllocate() or in VmaVirtualAllocationInfo::offset

\section virtual_allocator_statistics Statistics

You can obtain statistics of a virtual block using vmaGetVirtualBlockStatistics()
(to get brief statistics that are fast to calculate)
or vmaCalculateVirtualBlockStatistics() (to get more detailed statistics, slower to calculate).
The functions fill structures #VmaStatistics, #VmaDetailedStatistics respectively - same as used by the normal Vulkan memory allocator.
Example:

\code
VmaStatistics stats;
vmaGetVirtualBlockStatistics(block, &stats);
printf("My virtual block has %llu bytes used by %u virtual allocations\n",
    stats.allocationBytes, stats.allocationCount);
\endcode

You can also request a full list of allocations and free regions as a string in JSON format by calling
vmaBuildVirtualBlockStatsString().
Returned string must be later freed using vmaFreeVirtualBlockStatsString().
The format of this string differs from the one returned by the main Vulkan allocator, but it is similar.

\section virtual_allocator_additional_considerations Additional considerations

The "virtual allocator" functionality is implemented on a level of individual memory blocks.
Keeping track of a whole collection of blocks, allocating new ones when out of free space,
deleting empty ones, and deciding which one to try first for a new allocation must be implemented by the user.

Alternative allocation algorithms are supported, just like in custom pools of the real GPU memory.
See enum #VmaVirtualBlockCreateFlagBits to learn how to specify them (e.g. #VMA_VIRTUAL_BLOCK_CREATE_LINEAR_ALGORITHM_BIT).
You can find their description in chapter \ref custom_memory_pools.
Allocation strategies are also supported.
See enum #VmaVirtualAllocationCreateFlagBits to learn how to specify them (e.g. #VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT).

Following features are supported only by the allocator of the real GPU memory and not by virtual allocations:
buffer-image granularity, `VMA_DEBUG_MARGIN`, `VMA_MIN_ALIGNMENT`.


\page debugging_memory_usage Debugging incorrect memory usage

If you suspect a bug with memory usage, like usage of uninitialized memory or
memory being overwritten out of bounds of an allocation,
you can use debug features of this library to verify this.

\section debugging_memory_usage_initialization Memory initialization

If you experience a bug with incorrect and nondeterministic data in your program and you suspect uninitialized memory to be used,
you can enable automatic memory initialization to verify this.
To do it, define macro `VMA_DEBUG_INITIALIZE_ALLOCATIONS` to 1.

\code
#define VMA_DEBUG_INITIALIZE_ALLOCATIONS 1
#include "vk_mem_alloc.h"
\endcode

It makes memory of new allocations initialized to bit pattern `0xDCDCDCDC`.
Before an allocation is destroyed, its memory is filled with bit pattern `0xEFEFEFEF`.
Memory is automatically mapped and unmapped if necessary.

If you find these values while debugging your program, good chances are that you incorrectly
read Vulkan memory that is allocated but not initialized, or already freed, respectively.

Memory initialization works only with memory types that are `HOST_VISIBLE` and with allocations that can be mapped.
It works also with dedicated allocations.

\section debugging_memory_usage_margins Margins

By default, allocations are laid out in memory blocks next to each other if possible
(considering required alignment, `bufferImageGranularity`, and `nonCoherentAtomSize`).

![Allocations without margin](../gfx/Margins_1.png)

Define macro `VMA_DEBUG_MARGIN` to some non-zero value (e.g. 16) to enforce specified
number of bytes as a margin after every allocation.

\code
#define VMA_DEBUG_MARGIN 16
#include "vk_mem_alloc.h"
\endcode

![Allocations with margin](../gfx/Margins_2.png)

If your bug goes away after enabling margins, it means it may be caused by memory
being overwritten outside of allocation boundaries. It is not 100% certain though.
Change in application behavior may also be caused by different order and distribution
of allocations across memory blocks after margins are applied.

Margins work with all types of memory.

Margin is applied only to allocations made out of memory blocks and not to dedicated
allocations, which have their own memory block of specific size.
It is thus not applied to allocations made using #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT flag
or those automatically decided to put into dedicated allocations, e.g. due to its
large size or recommended by VK_KHR_dedicated_allocation extension.

Margins appear in [JSON dump](@ref statistics_json_dump) as part of free space.

Note that enabling margins increases memory usage and fragmentation.

Margins do not apply to \ref virtual_allocator.

\section debugging_memory_usage_corruption_detection Corruption detection

You can additionally define macro `VMA_DEBUG_DETECT_CORRUPTION` to 1 to enable validation
of contents of the margins.

\code
#define VMA_DEBUG_MARGIN 16
#define VMA_DEBUG_DETECT_CORRUPTION 1
#include "vk_mem_alloc.h"
\endcode

When this feature is enabled, number of bytes specified as `VMA_DEBUG_MARGIN`
(it must be multiply of 4) after every allocation is filled with a magic number.
This idea is also know as "canary".
Memory is automatically mapped and unmapped if necessary.

This number is validated automatically when the allocation is destroyed.
If it is not equal to the expected value, `VMA_ASSERT()` is executed.
It clearly means that either CPU or GPU overwritten the memory outside of boundaries of the allocation,
which indicates a serious bug.

You can also explicitly request checking margins of all allocations in all memory blocks
that belong to specified memory types by using function vmaCheckCorruption(),
or in memory blocks that belong to specified custom pool, by using function
vmaCheckPoolCorruption().

Margin validation (corruption detection) works only for memory types that are
`HOST_VISIBLE` and `HOST_COHERENT`.


\section debugging_memory_usage_leak_detection Leak detection features

At allocation and allocator destruction time VMA checks for unfreed and unmapped blocks using
`VMA_ASSERT_LEAK()`. This macro defaults to an assertion, triggering a typically fatal error in Debug
builds, and doing nothing in Release builds. You can provide your own definition of `VMA_ASSERT_LEAK()`
to change this behavior.

At memory block destruction time VMA lists out all unfreed allocations using the `VMA_LEAK_LOG_FORMAT()`
macro, which defaults to `VMA_DEBUG_LOG_FORMAT`, which in turn defaults to a no-op.
If you're having trouble with leaks - for example, the aforementioned assertion triggers, but you don't
quite know \em why -, overriding this macro to print out the the leaking blocks, combined with assigning
individual names to allocations using vmaSetAllocationName(), can greatly aid in fixing them.

\page other_api_interop Interop with other graphics APIs

VMA provides some features that help with interoperability with other graphics APIs, e.g. OpenGL.

\section opengl_interop_exporting_memory Exporting memory

If you want to attach `VkExportMemoryAllocateInfoKHR` or other structure to `pNext` chain of memory allocations made by the library:

You can create \ref custom_memory_pools for such allocations.
Define and fill in your `VkExportMemoryAllocateInfoKHR` structure and attach it to VmaPoolCreateInfo::pMemoryAllocateNext
while creating the custom pool.
Please note that the structure must remain alive and unchanged for the whole lifetime of the #VmaPool,
not only while creating it, as no copy of the structure is made,
but its original pointer is used for each allocation instead.

If you want to export all memory allocated by VMA from certain memory types,
also dedicated allocations or other allocations made from default pools,
an alternative solution is to fill in VmaAllocatorCreateInfo::pTypeExternalMemoryHandleTypes.
It should point to an array with `VkExternalMemoryHandleTypeFlagsKHR` to be automatically passed by the library
through `VkExportMemoryAllocateInfoKHR` on each allocation made from a specific memory type.
Please note that new versions of the library also support dedicated allocations created in custom pools.

You should not mix these two methods in a way that allows to apply both to the same memory type.
Otherwise, `VkExportMemoryAllocateInfoKHR` structure would be attached twice to the `pNext` chain of `VkMemoryAllocateInfo`.


\section opengl_interop_custom_alignment Custom alignment

Buffers or images exported to a different API like OpenGL may require a different alignment,
higher than the one used by the library automatically, queried from functions like `vkGetBufferMemoryRequirements`.
To impose such alignment:

You can create \ref custom_memory_pools for such allocations.
Set VmaPoolCreateInfo::minAllocationAlignment member to the minimum alignment required for each allocation
to be made out of this pool.
The alignment actually used will be the maximum of this member and the alignment returned for the specific buffer or image
from a function like `vkGetBufferMemoryRequirements`, which is called by VMA automatically.

If you want to create a buffer with a specific minimum alignment out of default pools,
use special function vmaCreateBufferWithAlignment(), which takes additional parameter `minAlignment`.

Note the problem of alignment affects only resources placed inside bigger `VkDeviceMemory` blocks and not dedicated
allocations, as these, by definition, always have alignment = 0 because the resource is bound to the beginning of its dedicated block.
You can ensure that an allocation is created as dedicated by using #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.
Contrary to Direct3D 12, Vulkan doesn't have a concept of alignment of the entire memory block passed on its allocation.

\section opengl_interop_extended_allocation_information Extended allocation information

If you want to rely on VMA to allocate your buffers and images inside larger memory blocks,
but you need to know the size of the entire block and whether the allocation was made
with its own dedicated memory, use function vmaget_allocation_info2() to retrieve
extended allocation information in structure #VmaAllocationInfo2.



\page usage_patterns Recommended usage patterns

Vulkan gives great flexibility in memory allocation.
This chapter shows the most common patterns.

See also slides from talk:
[Sawicki, Adam. Advanced Graphics Techniques Tutorial: Memory management in Vulkan and DX12. Game Developers Conference, 2018](https://www.gdcvault.com/play/1025458/Advanced-Graphics-Techniques-Tutorial-New)


\section usage_patterns_gpu_only GPU-only resource

<b>When:</b>
Any resources that you frequently write and read on GPU,
e.g. images used as color attachments (aka "render targets"), depth-stencil attachments,
images/buffers used as storage image/buffer (aka "Unordered Access View (UAV)").

<b>What to do:</b>
Let the library select the optimal memory type, which will likely have `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`.

\code
VkImageCreateInfo imgCreateInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
imgCreateInfo.imageType = VK_IMAGE_TYPE_2D;
imgCreateInfo.extent.width = 3840;
imgCreateInfo.extent.height = 2160;
imgCreateInfo.extent.depth = 1;
imgCreateInfo.mipLevels = 1;
imgCreateInfo.arrayLayers = 1;
imgCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
imgCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
imgCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
imgCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
imgCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
allocCreateInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
allocCreateInfo.priority = 1.0f;

VkImage img;
VmaAllocation alloc;
vmaCreateImage(allocator, &imgCreateInfo, &allocCreateInfo, &img, &alloc, nullptr);
\endcode

<b>Also consider:</b>
Consider creating them as dedicated allocations using #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
especially if they are large or if you plan to destroy and recreate them with different sizes
e.g. when display resolution changes.
Prefer to create such resources first and all other GPU resources (like textures and vertex buffers) later.
When VK_EXT_memory_priority extension is enabled, it is also worth setting high priority to such allocation
to decrease chances to be evicted to system memory by the operating system.

\section usage_patterns_staging_copy_upload Staging copy for upload

<b>When:</b>
A "staging" buffer than you want to map and fill from CPU code, then use as a source of transfer
to some GPU resource.

<b>What to do:</b>
Use flag #VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT.
Let the library select the optimal memory type, which will always have `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`.

\code
VkBufferCreateInfo bufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
bufCreateInfo.size = 65536;
bufCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
    VMA_ALLOCATION_CREATE_MAPPED_BIT;

VkBuffer buf;
VmaAllocation alloc;
VmaAllocationInfo allocInfo;
vmaCreateBuffer(allocator, &bufCreateInfo, &allocCreateInfo, &buf, &alloc, &allocInfo);

...

memcpy(allocInfo.pMappedData, myData, myDataSize);
\endcode

<b>Also consider:</b>
You can map the allocation using vmaMapMemory() or you can create it as persistenly mapped
using #VMA_ALLOCATION_CREATE_MAPPED_BIT, as in the example above.


\section usage_patterns_readback Readback

<b>When:</b>
Buffers for data written by or transferred from the GPU that you want to read back on the CPU,
e.g. results of some computations.

<b>What to do:</b>
Use flag #VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT.
Let the library select the optimal memory type, which will always have `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`
and `VK_MEMORY_PROPERTY_HOST_CACHED_BIT`.

\code
VkBufferCreateInfo bufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
bufCreateInfo.size = 65536;
bufCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
    VMA_ALLOCATION_CREATE_MAPPED_BIT;

VkBuffer buf;
VmaAllocation alloc;
VmaAllocationInfo allocInfo;
vmaCreateBuffer(allocator, &bufCreateInfo, &allocCreateInfo, &buf, &alloc, &allocInfo);

...

const float* downloadedData = (const float*)allocInfo.pMappedData;
\endcode


\section usage_patterns_advanced_data_uploading Advanced data uploading

For resources that you frequently write on CPU via mapped pointer and
frequently read on GPU e.g. as a uniform buffer (also called "dynamic"), multiple options are possible:

-# Easiest solution is to have one copy of the resource in `HOST_VISIBLE` memory,
   even if it means system RAM (not `DEVICE_LOCAL`) on systems with a discrete graphics card,
   and make the device reach out to that resource directly.
   - Reads performed by the device will then go through PCI Express bus.
     The performance of this access may be limited, but it may be fine depending on the size
     of this resource (whether it is small enough to quickly end up in GPU cache) and the sparsity
     of access.
-# On systems with unified memory (e.g. AMD APU or Intel integrated graphics, mobile chips),
   a memory type may be available that is both `HOST_VISIBLE` (available for mapping) and `DEVICE_LOCAL`
   (fast to access from the GPU). Then, it is likely the best choice for such type of resource.
-# Systems with a discrete graphics card and separate video memory may or may not expose
   a memory type that is both `HOST_VISIBLE` and `DEVICE_LOCAL`, also known as Base Address register_allocation (BAR).
   If they do, it represents a piece of VRAM (or entire VRAM, if ReBAR is enabled in the motherboard BIOS)
   that is available to CPU for mapping.
   - Writes performed by the host to that memory go through PCI Express bus.
     The performance of these writes may be limited, but it may be fine, especially on PCIe 4.0,
     as long as rules of using uncached and write-combined memory are followed - only sequential writes and no reads.
-# Finally, you may need or prefer to create a separate copy of the resource in `DEVICE_LOCAL` memory,
   a separate "staging" copy in `HOST_VISIBLE` memory and perform an explicit transfer command between them.

Thankfully, VMA offers an aid to create and use such resources in the the way optimal
for the current Vulkan device. To help the library make the best choice,
use flag #VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT together with
#VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT.
It will then prefer a memory type that is both `DEVICE_LOCAL` and `HOST_VISIBLE` (integrated memory or BAR),
but if no such memory type is available or allocation from it fails
(PC graphics cards have only 256 MB of BAR by default, unless ReBAR is supported and enabled in BIOS),
it will fall back to `DEVICE_LOCAL` memory for fast GPU access.
It is then up to you to detect that the allocation ended up in a memory type that is not `HOST_VISIBLE`,
so you need to create another "staging" allocation and perform explicit transfers.

\code
VkBufferCreateInfo bufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
bufCreateInfo.size = 65536;
bufCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
    VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
    VMA_ALLOCATION_CREATE_MAPPED_BIT;

VkBuffer buf;
VmaAllocation alloc;
VmaAllocationInfo allocInfo;
VkResult result = vmaCreateBuffer(allocator, &bufCreateInfo, &allocCreateInfo, &buf, &alloc, &allocInfo);
// Check result...

VkMemoryPropertyFlags memPropFlags;
vmaGetAllocationMemoryProperties(allocator, alloc, &memPropFlags);

if(memPropFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
{
    // The Allocation ended up in a mappable memory.
    // Calling vmaCopyMemoryToAllocation() does vmaMapMemory(), memcpy(), vmaUnmapMemory(), and vmaFlushAllocation().
    result = vmaCopyMemoryToAllocation(allocator, myData, alloc, 0, myDataSize);
    // Check result...

    VkBufferMemoryBarrier bufMemBarrier = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
    bufMemBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    bufMemBarrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
    bufMemBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufMemBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufMemBarrier.buffer = buf;
    bufMemBarrier.offset = 0;
    bufMemBarrier.size = VK_WHOLE_SIZE;

    // It's important to insert a buffer memory barrier here to ensure writing to the buffer has finished.
    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
        0, 0, nullptr, 1, &bufMemBarrier, 0, nullptr);
}
else
{
    // Allocation ended up in a non-mappable memory - a transfer using a staging buffer is required.
    VkBufferCreateInfo stagingBufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    stagingBufCreateInfo.size = 65536;
    stagingBufCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAllocCreateInfo = {};
    stagingAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    stagingAllocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer stagingBuf;
    VmaAllocation stagingAlloc;
    VmaAllocationInfo stagingAllocInfo;
    result = vmaCreateBuffer(allocator, &stagingBufCreateInfo, &stagingAllocCreateInfo,
        &stagingBuf, &stagingAlloc, &stagingAllocInfo);
    // Check result...

    // Calling vmaCopyMemoryToAllocation() does vmaMapMemory(), memcpy(), vmaUnmapMemory(), and vmaFlushAllocation().
    result = vmaCopyMemoryToAllocation(allocator, myData, stagingAlloc, 0, myDataSize);
    // Check result...

    VkBufferMemoryBarrier bufMemBarrier = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
    bufMemBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    bufMemBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bufMemBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufMemBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufMemBarrier.buffer = stagingBuf;
    bufMemBarrier.offset = 0;
    bufMemBarrier.size = VK_WHOLE_SIZE;

    // Insert a buffer memory barrier to make sure writing to the staging buffer has finished.
    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 1, &bufMemBarrier, 0, nullptr);

    VkBufferCopy bufCopy = {
        0, // srcOffset
        0, // dstOffset,
        myDataSize, // size
    };

    vkCmdCopyBuffer(cmdBuf, stagingBuf, buf, 1, &bufCopy);

    VkBufferMemoryBarrier bufMemBarrier2 = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
    bufMemBarrier2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bufMemBarrier2.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT; // We created a uniform buffer
    bufMemBarrier2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufMemBarrier2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufMemBarrier2.buffer = buf;
    bufMemBarrier2.offset = 0;
    bufMemBarrier2.size = VK_WHOLE_SIZE;

    // Make sure copying from staging buffer to the actual buffer has finished by inserting a buffer memory barrier.
    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
        0, 0, nullptr, 1, &bufMemBarrier2, 0, nullptr);
}
\endcode

\section usage_patterns_other_use_cases Other use cases

Here are some other, less obvious use cases and their recommended settings:

- An image that is used only as transfer source and destination, but it should stay on the device,
  as it is used to temporarily store a copy of some texture, e.g. from the current to the next frame,
  for temporal antialiasing or other temporal effects.
  - Use `VkImageCreateInfo::usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT`
  - Use VmaAllocationCreateInfo::usage = #VMA_MEMORY_USAGE_AUTO
- An image that is used only as transfer source and destination, but it should be placed
  in the system RAM despite it doesn't need to be mapped, because it serves as a "swap" copy to evict
  least recently used textures from VRAM.
  - Use `VkImageCreateInfo::usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT`
  - Use VmaAllocationCreateInfo::usage = #VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
    as VMA needs a hint here to differentiate from the previous case.
- A buffer that you want to map and write from the CPU, directly read from the GPU
  (e.g. as a uniform or vertex buffer), but you have a clear preference to place it in device or
  host memory due to its large size.
  - Use `VkBufferCreateInfo::usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT`
  - Use VmaAllocationCreateInfo::usage = #VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE or #VMA_MEMORY_USAGE_AUTO_PREFER_HOST
  - Use VmaAllocationCreateInfo::flags = #VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT


\page configuration Configuration

Please check "CONFIGURATION SECTION" in the code to find macros that you can define
before each include of this file or change directly in this file to provide
your own implementation of basic facilities like assert, `min()` and `max()` functions,
mutex, atomic etc.

For example, define `VMA_ASSERT(expr)` before including the library to provide
custom implementation of the assertion, compatible with your project.
By default it is defined to standard C `assert(expr)` in `_DEBUG` configuration
and empty otherwise.

Similarly, you can define `VMA_LEAK_LOG_FORMAT` macro to enable printing of leaked (unfreed) allocations,
including their names and other parameters. Example:

\code
#define VMA_LEAK_LOG_FORMAT(format, ...) do { \
        printf((format), __VA_ARGS__); \
        printf("\n"); \
    } while(false)
\endcode

\section config_Vulkan_functions Pointers to Vulkan functions

There are multiple ways to import pointers to Vulkan functions in the library.
In the simplest case you don't need to do anything.
If the compilation or linking of your program or the initialization of the #VmaAllocator
doesn't work for you, you can try to reconfigure it.

First, the allocator tries to fetch pointers to Vulkan functions linked statically,
like this:

\code
_vulkan_functions.vkAllocateMemory = (PFN_vkAllocateMemory)vkAllocateMemory;
\endcode

If you want to disable this feature, set configuration macro: `#define VMA_STATIC_VULKAN_FUNCTIONS 0`.

Second, you can provide the pointers yourself by setting member VmaAllocatorCreateInfo::pVulkanFunctions.
You can fetch them e.g. using functions `vkGetInstanceProcAddr` and `vkGetDeviceProcAddr` or
by using a helper library like [volk](https://github.com/zeux/volk).

Third, VMA tries to fetch remaining pointers that are still null by calling
`vkGetInstanceProcAddr` and `vkGetDeviceProcAddr` on its own.
You need to only fill in VmaVulkanFunctions::vkGetInstanceProcAddr and VmaVulkanFunctions::vkGetDeviceProcAddr.
Other pointers will be fetched automatically.
If you want to disable this feature, set configuration macro: `#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0`.

Finally, all the function pointers required by the library (considering selected
Vulkan version and enabled extensions) are checked with `VMA_ASSERT` if they are not null.


\section custom_memory_allocator Custom host memory allocator

If you use custom allocator for CPU memory rather than default operator `new`
and `delete` from C++, you can make this library using your allocator as well
by filling optional member VmaAllocatorCreateInfo::pAllocationCallbacks. These
functions will be passed to Vulkan, as well as used by the library itself to
make any CPU-side allocations.

\section allocation_callbacks Device memory allocation callbacks

The library makes calls to `vkAllocateMemory()` and `vkFreeMemory()` internally.
You can setup callbacks to be informed about these calls, e.g. for the purpose
of gathering some statistics. To do it, fill optional member
VmaAllocatorCreateInfo::pDeviceMemoryCallbacks.

\section heap_memory_limit Device heap memory limit

When device memory of certain heap runs out of free space, new allocations may
fail (returning error code) or they may succeed, silently pushing some existing_
memory blocks from GPU VRAM to system RAM (which degrades performance). This
behavior is implementation-dependent - it depends on GPU vendor and graphics
driver.

On AMD cards it can be controlled while creating Vulkan device object by using
VK_AMD_memory_overallocation_behavior extension, if available.

Alternatively, if you want to test how your program behaves with limited amount of Vulkan device
memory available without switching your graphics card to one that really has
smaller VRAM, you can use a feature of this library intended for this purpose.
To do it, fill optional member VmaAllocatorCreateInfo::pHeapSizeLimit.



\page vk_khr_dedicated_allocation VK_KHR_dedicated_allocation

VK_KHR_dedicated_allocation is a Vulkan extension which can be used to improve
performance on some GPUs. It augments Vulkan API with possibility to query
driver whether it prefers particular buffer or image to have its own, dedicated
allocation (separate `VkDeviceMemory` block) for better efficiency - to be able
to do some internal optimizations. The extension is supported by this library.
It will be used automatically when enabled.

It has been promoted to core Vulkan 1.1, so if you use eligible Vulkan version
and inform VMA about it by setting VmaAllocatorCreateInfo::vulkanApiVersion,
you are all set.

Otherwise, if you want to use it as an extension:

1 . When creating Vulkan device, check if following 2 device extensions are
supported (call `vkEnumerateDeviceExtensionProperties()`).
If yes, enable them (fill `VkDeviceCreateInfo::ppEnabledExtensionNames`).

- VK_KHR_get_memory_requirements2
- VK_KHR_dedicated_allocation

If you enabled these extensions:

2 . Use #VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT flag when creating
your #VmaAllocator to inform the library that you enabled required extensions
and you want the library to use them.

\code
allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT;

vmaCreateAllocator(&allocatorInfo, &allocator);
\endcode

That is all. The extension will be automatically used whenever you create a
buffer using vmaCreateBuffer() or image using vmaCreateImage().

When using the extension together with Vulkan Validation Layer, you will receive
warnings like this:

_vkBindBufferMemory(): Binding memory to buffer 0x33 but vkGetBufferMemoryRequirements() has not been called on that buffer._

It is OK, you should just ignore it. It happens because you use function
`vkGetBufferMemoryRequirements2KHR()` instead of standard
`vkGetBufferMemoryRequirements()`, while the validation layer seems to be
unaware of it.

To learn more about this extension, see:

- [VK_KHR_dedicated_allocation in Vulkan specification](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/chap50.html#VK_KHR_dedicated_allocation)
- [VK_KHR_dedicated_allocation unofficial manual](http://asawicki.info/articles/VK_KHR_dedicated_allocation.php5)



\page vk_ext_memory_priority VK_EXT_memory_priority

VK_EXT_memory_priority is a device extension that allows to pass additional "priority"
value to Vulkan memory allocations that the implementation may use prefer certain
buffers and images that are critical for performance to stay in device-local memory
in cases when the memory is over-subscribed, while some others may be moved to the system memory.

VMA offers convenient usage of this extension.
If you enable it, you can pass "priority" parameter when creating allocations or custom pools
and the library automatically passes the value to Vulkan using this extension.

If you want to use this extension in connection with VMA, follow these steps:

\section vk_ext_memory_priority_initialization Initialization

1) Call `vkEnumerateDeviceExtensionProperties` for the physical device.
Check if the extension is supported - if returned array of `VkExtensionProperties` contains "VK_EXT_memory_priority".

2) Call `vkGetPhysicalDeviceFeatures2` for the physical device instead of old `vkGetPhysicalDeviceFeatures`.
Attach additional structure `VkPhysicalDeviceMemoryPriorityFeaturesEXT` to `VkPhysicalDeviceFeatures2::pNext` to be returned.
Check if the device feature is really supported - check if `VkPhysicalDeviceMemoryPriorityFeaturesEXT::memoryPriority` is true.

3) While creating device with `vkCreateDevice`, enable this extension - add "VK_EXT_memory_priority"
to the list passed as `VkDeviceCreateInfo::ppEnabledExtensionNames`.

4) While creating the device, also don't set `VkDeviceCreateInfo::pEnabledFeatures`.
Fill in `VkPhysicalDeviceFeatures2` structure instead and pass it as `VkDeviceCreateInfo::pNext`.
Enable this device feature - attach additional structure `VkPhysicalDeviceMemoryPriorityFeaturesEXT` to
`VkPhysicalDeviceFeatures2::pNext` chain and set its member `memoryPriority` to `VK_TRUE`.

5) While creating #VmaAllocator with vmaCreateAllocator() inform VMA that you
have enabled this extension and feature - add #VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT
to VmaAllocatorCreateInfo::flags.

\section vk_ext_memory_priority_usage Usage

When using this extension, you should initialize following member:

- VmaAllocationCreateInfo::priority when creating a dedicated allocation with #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.
- VmaPoolCreateInfo::priority when creating a custom pool.

It should be a floating-point value between `0.0f` and `1.0f`, where recommended default is `0.5F`.
Memory allocated with higher value can be treated by the Vulkan implementation as higher priority
and so it can have lower chances of being pushed out to system memory, experiencing degraded performance.

It might be a good idea to create performance-critical resources like color-attachment or depth-stencil images
as dedicated and set high priority to them. For example:

\code
VkImageCreateInfo imgCreateInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
imgCreateInfo.imageType = VK_IMAGE_TYPE_2D;
imgCreateInfo.extent.width = 3840;
imgCreateInfo.extent.height = 2160;
imgCreateInfo.extent.depth = 1;
imgCreateInfo.mipLevels = 1;
imgCreateInfo.arrayLayers = 1;
imgCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
imgCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
imgCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
imgCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
imgCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
allocCreateInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
allocCreateInfo.priority = 1.0f;

VkImage img;
VmaAllocation alloc;
vmaCreateImage(allocator, &imgCreateInfo, &allocCreateInfo, &img, &alloc, nullptr);
\endcode

`priority` member is ignored in the following situations:

- Allocations created in custom pools: They inherit the priority, along with all other allocation parameters
  from the parameters passed in #VmaPoolCreateInfo when the pool was created.
- Allocations created in default pools: They inherit the priority from the parameters
  VMA used when creating default pools, which means `priority == 0.5F`.


\page vk_amd_device_coherent_memory VK_AMD_device_coherent_memory

VK_AMD_device_coherent_memory is a device extension that enables access to
additional memory types with `VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD` and
`VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD` flag. It is useful mostly for
allocation of buffers intended for writing "breadcrumb markers" in between passes
or draw calls, which in turn are useful for debugging GPU crash/hang/TDR cases.

When the extension is available but has not been enabled, Vulkan physical device
still exposes those memory types, but their usage is forbidden. VMA automatically
takes care of that - it returns `VK_ERROR_FEATURE_NOT_PRESENT` when an attempt
to allocate memory of such type is made.

If you want to use this extension in connection with VMA, follow these steps:

\section vk_amd_device_coherent_memory_initialization Initialization

1) Call `vkEnumerateDeviceExtensionProperties` for the physical device.
Check if the extension is supported - if returned array of `VkExtensionProperties` contains "VK_AMD_device_coherent_memory".

2) Call `vkGetPhysicalDeviceFeatures2` for the physical device instead of old `vkGetPhysicalDeviceFeatures`.
Attach additional structure `VkPhysicalDeviceCoherentMemoryFeaturesAMD` to `VkPhysicalDeviceFeatures2::pNext` to be returned.
Check if the device feature is really supported - check if `VkPhysicalDeviceCoherentMemoryFeaturesAMD::deviceCoherentMemory` is true.

3) While creating device with `vkCreateDevice`, enable this extension - add "VK_AMD_device_coherent_memory"
to the list passed as `VkDeviceCreateInfo::ppEnabledExtensionNames`.

4) While creating the device, also don't set `VkDeviceCreateInfo::pEnabledFeatures`.
Fill in `VkPhysicalDeviceFeatures2` structure instead and pass it as `VkDeviceCreateInfo::pNext`.
Enable this device feature - attach additional structure `VkPhysicalDeviceCoherentMemoryFeaturesAMD` to
`VkPhysicalDeviceFeatures2::pNext` and set its member `deviceCoherentMemory` to `VK_TRUE`.

5) While creating #VmaAllocator with vmaCreateAllocator() inform VMA that you
have enabled this extension and feature - add #VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT
to VmaAllocatorCreateInfo::flags.

\section vk_amd_device_coherent_memory_usage Usage

After following steps described above, you can create VMA allocations and custom pools
out of the special `DEVICE_COHERENT` and `DEVICE_UNCACHED` memory types on eligible
devices. There are multiple ways to do it, for example:

- You can request or prefer to allocate out of such memory types by adding
  `VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD` to VmaAllocationCreateInfo::requiredFlags
  or VmaAllocationCreateInfo::preferredFlags. Those flags can be freely mixed with
  other ways of \ref choosing_memory_type, like setting VmaAllocationCreateInfo::usage.
- If you manually found memory type index to use for this purpose, force allocation
  from this specific index by setting VmaAllocationCreateInfo::memoryTypeBits `= 1U << index`.

\section vk_amd_device_coherent_memory_more_information More information

To learn more about this extension, see [VK_AMD_device_coherent_memory in Vulkan specification](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_AMD_device_coherent_memory.html)

Example use of this extension can be found in the code of the sample and test suite
accompanying this library.


\page vk_khr_external_memory_win32 VK_KHR_external_memory_win32

On Windows, the VK_KHR_external_memory_win32 device extension allows exporting a Win32 `HANDLE`
of a `VkDeviceMemory` block, to be able to reference the memory on other Vulkan logical devices or instances,
in multiple processes, and/or in multiple APIs.
VMA offers support for it.

\section vk_khr_external_memory_win32_initialization Initialization

1) Make sure the extension is defined in the code by including following header before including VMA:

\code
#include <vulkan/vulkan_win32.h>
\endcode

2) Check if "VK_KHR_external_memory_win32" is available among device extensions.
Enable it when creating the `VkDevice` object.

3) Enable the usage of this extension in VMA by setting flag #VMA_ALLOCATOR_CREATE_KHR_EXTERNAL_MEMORY_WIN32_BIT
when calling vmaCreateAllocator().

4) Make sure that VMA has access to the `vkGetMemoryWin32HandleKHR` function by either enabling `VMA_DYNAMIC_VULKAN_FUNCTIONS` macro
or setting VmaVulkanFunctions::vkGetMemoryWin32HandleKHR explicitly.
For more information, see \ref quick_start_initialization_importing_vulkan_functions.

\section vk_khr_external_memory_win32_preparations Preparations

You can find example usage among tests, in file "Tests.cpp", function `TestWin32Handles()`.

To use the extenion, buffers need to be created with `VkExternalMemoryBufferCreateInfoKHR` attached to their `pNext` chain,
and memory allocations need to be made with `VkExportMemoryAllocateInfoKHR` attached to their `pNext` chain.
To make use of them, you need to use \ref custom_memory_pools. Example:

\code
// Define an example buffer and allocation parameters.
VkExternalMemoryBufferCreateInfoKHR externalMemBufCreateInfo = {
    VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO_KHR,
    nullptr,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
};
VkBufferCreateInfo exampleBufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
exampleBufCreateInfo.size = 0x10000; // Doesn't matter here.
exampleBufCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
exampleBufCreateInfo.pNext = &externalMemBufCreateInfo;

VmaAllocationCreateInfo exampleAllocCreateInfo = {};
exampleAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

// Find memory type index to use for the custom pool.
uint32_t memTypeIndex;
VkResult res = vmaFindMemoryTypeIndexForBufferInfo(g_Allocator,
    &exampleBufCreateInfo, &exampleAllocCreateInfo, &memTypeIndex);
// Check res...

// Create a custom pool.
constexpr static VkExportMemoryAllocateInfoKHR exportMemAllocInfo = {
    VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR,
    nullptr,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
};
VmaPoolCreateInfo poolCreateInfo = {};
poolCreateInfo.memoryTypeIndex = memTypeIndex;
poolCreateInfo.pMemoryAllocateNext = (void*)&exportMemAllocInfo;

VmaPool pool;
res = vmaCreatePool(g_Allocator, &poolCreateInfo, &pool);
// Check res...

// YOUR OTHER CODE COMES HERE....

// At the end, don't forget to destroy it!
vmaDestroyPool(g_Allocator, pool);
\endcode

Note that the structure passed as VmaPoolCreateInfo::pMemoryAllocateNext must remain alive and unchanged
for the whole lifetime of the custom pool, because it will be used when the pool allocates a new device memory block.
No copy is made internally. This is why variable `exportMemAllocInfo` is defined as `static`.

\section vk_khr_external_memory_win32_memory_allocation Memory allocation

Finally, you can create a buffer with an allocation out of the custom pool.
The buffer should use same flags as the sample buffer used to find the memory type.
It should also specify `VkExternalMemoryBufferCreateInfoKHR` in its `pNext` chain.

\code
VkExternalMemoryBufferCreateInfoKHR externalMemBufCreateInfo = {
    VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO_KHR,
    nullptr,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
};
VkBufferCreateInfo bufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
bufCreateInfo.size = // Your desired buffer size.
bufCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
bufCreateInfo.pNext = &externalMemBufCreateInfo;

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.pool = pool;  // It is enough to set this one member.

VkBuffer buf;
VmaAllocation alloc;
res = vmaCreateBuffer(g_Allocator, &bufCreateInfo, &allocCreateInfo, &buf, &alloc, nullptr);
// Check res...

// YOUR OTHER CODE COMES HERE....

// At the end, don't forget to destroy it!
vmaDestroyBuffer(g_Allocator, buf, alloc);
\endcode

If you need each allocation to have its own device memory block and start at offset 0, you can still do 
by using #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT flag. It works also with custom pools.

\section vk_khr_external_memory_win32_exporting_win32_handle Exporting Win32 handle

After the allocation is created, you can acquire a Win32 `HANDLE` to the `VkDeviceMemory` block it belongs to.
VMA function vmaGetMemoryWin32Handle() is a replacement of the Vulkan function `vkGetMemoryWin32HandleKHR`.

\code
HANDLE handle;
res = vmaGetMemoryWin32Handle(g_Allocator, alloc, nullptr, &handle);
// Check res...

// YOUR OTHER CODE COMES HERE....

// At the end, you must close the handle.
CloseHandle(handle);
\endcode

Documentation of the VK_KHR_external_memory_win32 extension states that:

> If handleType is defined as an NT handle, vkGetMemoryWin32HandleKHR must be called no more than once for each valid unique combination of memory and handleType.

This is ensured automatically inside VMA.
The library fetches the handle on first use, remembers it internally, and closes it when the memory block or dedicated allocation is destroyed.
Every time you call vmaGetMemoryWin32Handle(), VMA calls `DuplicateHandle` and returns a new handle that you need to close.

For further information, please check documentation of the vmaGetMemoryWin32Handle() function.


\page enabling_buffer_device_address Enabling buffer device address

Device extension VK_KHR_buffer_device_address
allow to fetch raw GPU pointer to a buffer and pass it for usage in a shader code.
It has been promoted to core Vulkan 1.2.

If you want to use this feature in connection with VMA, follow these steps:

\section enabling_buffer_device_address_initialization Initialization

1) (For Vulkan version < 1.2) Call `vkEnumerateDeviceExtensionProperties` for the physical device.
Check if the extension is supported - if returned array of `VkExtensionProperties` contains
"VK_KHR_buffer_device_address".

2) Call `vkGetPhysicalDeviceFeatures2` for the physical device instead of old `vkGetPhysicalDeviceFeatures`.
Attach additional structure `VkPhysicalDeviceBufferDeviceAddressFeatures*` to `VkPhysicalDeviceFeatures2::pNext` to be returned.
Check if the device feature is really supported - check if `VkPhysicalDeviceBufferDeviceAddressFeatures::bufferDeviceAddress` is true.

3) (For Vulkan version < 1.2) While creating device with `vkCreateDevice`, enable this extension - add
"VK_KHR_buffer_device_address" to the list passed as `VkDeviceCreateInfo::ppEnabledExtensionNames`.

4) While creating the device, also don't set `VkDeviceCreateInfo::pEnabledFeatures`.
Fill in `VkPhysicalDeviceFeatures2` structure instead and pass it as `VkDeviceCreateInfo::pNext`.
Enable this device feature - attach additional structure `VkPhysicalDeviceBufferDeviceAddressFeatures*` to
`VkPhysicalDeviceFeatures2::pNext` and set its member `bufferDeviceAddress` to `VK_TRUE`.

5) While creating #VmaAllocator with vmaCreateAllocator() inform VMA that you
have enabled this feature - add #VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT
to VmaAllocatorCreateInfo::flags.

\section enabling_buffer_device_address_usage Usage

After following steps described above, you can create buffers with `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT*` using VMA.
The library automatically adds `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT*` to
allocated memory blocks wherever it might be needed.

Please note that the library supports only `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT*`.
The second part of this functionality related to "capture and replay" is not supported,
as it is intended for usage in debugging tools like RenderDoc, not in everyday Vulkan usage.

\section enabling_buffer_device_address_more_information More information

To learn more about this extension, see [VK_KHR_buffer_device_address in Vulkan specification](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/chap46.html#VK_KHR_buffer_device_address)

Example use of this extension can be found in the code of the sample and test suite
accompanying this library.

\page general_considerations General considerations

\section general_considerations_thread_safety Thread safety

- The library has no global state, so separate #VmaAllocator objects can be used
  independently.
  There should be no need to create multiple such objects though - one per `VkDevice` is enough.
- By default, all calls to functions that take #VmaAllocator as first parameter
  are safe to call from multiple threads simultaneously because they are
  synchronized internally when needed.
  This includes allocation and deallocation from default memory pool, as well as custom #VmaPool.
- When the allocator is created with #VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT
  flag, calls to functions that take such #VmaAllocator object must be
  synchronized externally.
- Access to a #VmaAllocation object must be externally synchronized. For example,
  you must not call vmaGetAllocationInfo() and vmaMapMemory() from different
  threads at the same time if you pass the same #VmaAllocation object to these
  functions.
- #VmaVirtualBlock is not safe to be used from multiple threads simultaneously.

\section general_considerations_versioning_and_compatibility Versioning and compatibility

The library uses [**Semantic Versioning**](https://semver.org/),
which means version numbers follow convention: Major.Minor.Patch (e.g. 2.3.0), where:

- Incremented Patch version means a release is backward- and forward-compatible,
  introducing only some internal improvements, bug fixes, optimizations etc.
  or changes that are out of scope of the official API described in this documentation.
- Incremented Minor version means a release is backward-compatible,
  so existing code that uses the library should continue to work, while some new
  symbols could have been added: new structures, functions, new values in existing
  enums and bit flags, new structure members, but not new function parameters.
- Incrementing Major version means a release could break some backward compatibility.

All changes between official releases are documented in file "CHANGELOG.md".

\warning Backward compatibility is considered on the level of C++ source code, not binary linkage.
Adding new members to existing structures is treated as backward compatible if initializing
the new members to binary zero results in the old behavior.
You should always fully initialize all library structures to zeros and not rely on their
exact binary size.

\section general_considerations_validation_layer_warnings Validation layer warnings

When using this library, you can meet following types of warnings issued by
Vulkan validation layer. They don't necessarily indicate a bug, so you may need
to just ignore them.

- *vkBindBufferMemory(): Binding memory to buffer 0xeb8e4 but vkGetBufferMemoryRequirements() has not been called on that buffer.*
  - It happens when VK_KHR_dedicated_allocation extension is enabled.
    `vkGetBufferMemoryRequirements2KHR` function is used instead, while validation layer seems to be unaware of it.
- *Mapping an image with layout VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL can result in undefined behavior if this memory is used by the device. Only GENERAL or PREINITIALIZED should be used.*
  - It happens when you map a buffer or image, because the library maps entire
    `VkDeviceMemory` block, where different types of images and buffers may end
    up together, especially on GPUs with unified memory like Intel.
- *Non-linear image 0xebc91 is aliased with linear buffer 0xeb8e4 which may indicate a bug.*
  - It may happen when you use [defragmentation](@ref defragmentation).

\section general_considerations_allocation_algorithm Allocation algorithm

The library uses following algorithm for allocation, in order:

-# Try to find free range of memory in existing blocks.
-# If failed, try to create a new block of `VkDeviceMemory`, with preferred block size.
-# If failed, try to create such block with size / 2, size / 4, size / 8.
-# If failed, try to allocate separate `VkDeviceMemory` for this allocation,
   just like when you use #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.
-# If failed, choose other memory type that meets the requirements specified in
   VmaAllocationCreateInfo and go to point 1.
-# If failed, return `VK_ERROR_OUT_OF_DEVICE_MEMORY`.

\section general_considerations_features_not_supported Features not supported

Features deliberately excluded from the scope of this library:

-# **Data transfer.** Uploading (streaming) and downloading data of buffers and images
   between CPU and GPU memory and related synchronization is responsibility of the user.
   Defining some "texture" object that would automatically stream its data from a
   staging copy in CPU memory to GPU memory would rather be a feature of another,
   higher-level library implemented on top of VMA.
   VMA doesn't record any commands to a `VkCommandBuffer`. It just allocates memory.
-# **Recreation of buffers and images.** Although the library has functions for
   buffer and image creation: vmaCreateBuffer(), vmaCreateImage(), you need to
   recreate these objects yourself after defragmentation. That is because the big
   structures `VkBufferCreateInfo`, `VkImageCreateInfo` are not stored in
   #VmaAllocation object.
-# **Handling CPU memory allocation failures.** When dynamically creating small C++
   objects in CPU memory (not Vulkan memory), allocation failures are not checked
   and handled gracefully, because that would complicate code significantly and
   is usually not needed in desktop PC applications anyway.
   Success of an allocation is just checked with an assert.
-# **Code free of any compiler warnings.** Maintaining the library to compile and
   work correctly on so many different platforms is hard enough. Being free of
   any warnings, on any version of any compiler, is simply not feasible.
   There are many preprocessor macros that make some variables unused, function parameters unreferenced,
   or conditional expressions constant in some configurations.
   The code of this library should not be bigger or more complicated just to silence these warnings.
   It is recommended to disable such warnings instead.
-# This is a C++ library with C interface. **Bindings or ports to any other programming languages** are welcome as external projects but
   are not going to be included into this repository.
*/