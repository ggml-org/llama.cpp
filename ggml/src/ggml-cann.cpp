/*
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "ggml-cann.h"

#include <acl/acl.h>
#include <stdarg.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>

#include "ggml-backend-impl.h"
#include "ggml-cann/aclnn_ops.h"
#include "ggml-cann/common.h"

#define GGML_COMMON_DECL_C

#include "ggml-common.h"

/**
 * @brief Default logging callback for GGML.
 *
 * This function is the default logging callback that logs messages to stderr.
 *
 * @param level The log level.
 * @param msg The log message.
 * @param user_data User data passed to the callback.
 */
static void ggml_cann_default_log_callback(enum ggml_log_level level,
                                           const char* msg, void* user_data) {
    GGML_UNUSED(level);
    GGML_UNUSED(user_data);
    fprintf(stderr, "%s", msg);
}

ggml_log_callback ggml_cann_log_callback = ggml_cann_default_log_callback;
void* ggml_cann_log_user_data = NULL;

GGML_API void ggml_backend_cann_log_set_callback(ggml_log_callback log_callback,
                                                 void* user_data) {
    ggml_cann_log_callback = log_callback;
    ggml_cann_log_user_data = user_data;
}

#define GGML_CANN_LOG_INFO(...) ggml_cann_log(GGML_LOG_LEVEL_INFO, __VA_ARGS__)
#define GGML_CANN_LOG_WARN(...) ggml_cann_log(GGML_LOG_LEVEL_WARN, __VA_ARGS__)
#define GGML_CANN_LOG_ERROR(...) \
    ggml_cann_log(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)

GGML_ATTRIBUTE_FORMAT(2, 3)

/**
 * @brief Log a message using the current logging callback.
 *
 * This function formats a log message and passes it to the current logging
 * callback.
 *
 * @param level The log level.
 * @param format The format string for the log message.
 * @param ... The arguments for the format string.
 */
static void ggml_cann_log(enum ggml_log_level level, const char* format, ...) {
    if (ggml_cann_log_callback != NULL) {
        va_list args;
        va_start(args, format);
        char buffer[128];
        int len = vsnprintf(buffer, 128, format, args);
        if (len < 128) {
            ggml_cann_log_callback(level, buffer, ggml_cann_log_user_data);
        } else {
             // vsnprintf adds a null terminator
            std::vector<char> buffer2(len + 1);
            va_end(args);
            va_start(args, format);
            vsnprintf(&buffer2[0], buffer2.size(), format, args);
            ggml_cann_log_callback(level, buffer2.data(),
                                   ggml_cann_log_user_data);
        }
        va_end(args);
    }
}

/**
 * @brief Handles CANN errors by printing an error message and aborting.
 *
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred.
 * @param file The file in which the error occurred.
 * @param line The line number where the error occurred.
 * @param msg The error message.
 */
[[noreturn]] void ggml_cann_error(const char* stmt, const char* func,
                                  const char* file, int line, const char* msg) {
    int32_t id = -1;
    aclrtGetDevice(&id);

    GGML_CANN_LOG_ERROR("CANN error: %s\n", msg);
    GGML_CANN_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func,
            file, line);
    GGML_CANN_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ASSERT to get a stack trace
    GGML_ABORT("CANN error");
}

/**
 * @brief Sets the device to be used by CANN.
 *
 * @param device The device ID to set.
 */
void ggml_cann_set_device(const int32_t device) {
    // TODO: uncomment these lines after empty context has fixed.
    // int current_device;
    // ACL_CHECK(aclrtGetDevice(&current_device));

    // if (device == current_device) {
    //   return;
    // }
    ACL_CHECK(aclrtSetDevice(device));
}

/**
 * @brief Retrieves the current device ID.
 *
 * @return The current device ID.
 */
int32_t ggml_cann_get_device() {
    int32_t id;
    ACL_CHECK(aclrtGetDevice(&id));
    return id;
}

/**
 * @brief Initialize the CANN device information.
 *
 * This function initializes the CANN device information by obtaining the
 * device count and setting the memory allocation granularity for each device.
 *
 * @return A structure containing the device information.
 */
static ggml_cann_device_info ggml_cann_init() {
    ggml_cann_device_info info = {};

    aclError err = aclrtGetDeviceCount((uint32_t*)&info.device_count);

    if (err != ACL_SUCCESS) {
        GGML_CANN_LOG_ERROR("%s: failed to initialize CANN: %s\n",
                __func__, aclGetRecentErrMsg());
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_CANN_MAX_DEVICES);

    for (int id = 0; id < info.device_count; ++id) {
        aclrtPhysicalMemProp prop = {};
        prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
        prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
        prop.memAttr = ACL_HBM_MEM_HUGE;
        prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = id;
        prop.reserve = 0;
        ACL_CHECK(aclrtMemGetAllocationGranularity(
            &prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            &info.devices[id].vmm_granularity));
    }

    // TODO: add more device info later.
    return info;
}

/**
 * @brief Retrieve the CANN device information.
 *
 * This function returns a reference to a structure containing the CANN device
 * information. The device information is initialized once and reused on
 * subsequent calls.
 *
 * @return A reference to the structure containing the device information.
 */
const ggml_cann_device_info& ggml_cann_info() {
    static ggml_cann_device_info info = ggml_cann_init();
    return info;
}

//#define DEBUG_CANN_MALLOC
/**
 * @brief A pool of CANN buffers(legacy).
 *
 * This class manages a pool of CANN buffers for a specific device.
 */
struct ggml_cann_pool_leg : public ggml_cann_pool {
    /**
     * @brief The maximum number of buffers in the pool.
     */
    static const int MAX_BUFFERS = 256;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    int device;

    /**
     * @brief Structure representing a CANN buffer.
     */
    struct ggml_cann_buffer {
        void* ptr = nullptr;  ///< Pointer to the buffer memory.
        size_t size = 0;      ///< Size of the buffer.
    };

    /**
     * @brief Array of CANN buffers in the pool.
     */
    ggml_cann_buffer buffer_pool[MAX_BUFFERS] = {};

    /**
     * @brief Total size of all buffers in the pool.
     */
    size_t pool_size = 0;

    /**
     * @brief Constructor to initialize the buffer pool for a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    explicit ggml_cann_pool_leg(int device) : device(device) {}

    /**
     * @brief Destructor to free all buffers in the pool.
     */
    ~ggml_cann_pool_leg() {
        ggml_cann_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
                ACL_CHECK(aclrtFree(b.ptr));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

    /**
     * @brief Allocate a buffer of the given size.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    void* alloc(size_t size, size_t* actual_size) override {
#ifdef DEBUG_CANN_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CANN_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void* ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cann_buffer& b = buffer_pool[ibest];
            void* ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void* ptr;
        size_t look_ahead_size = (size_t)(1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255) / 256);
        ggml_cann_set_device(device);
        ACL_CHECK(
            aclrtMalloc(&ptr, look_ahead_size, ACL_MEM_MALLOC_HUGE_FIRST));
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CANN_MALLOC
        GGML_CANN_LOG_INFO(
            "%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, "
            "requested %u MB\n",
            __func__, device, nnz, (uint32_t)(max_size / 1024 / 1024),
            (uint32_t)(pool_size / 1024 / 1024),
            (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    /**
     * @brief Free a buffer and return it to the pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    void free(void* ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        // memory should always buffered. these memory may still needed by
        // tasks in stream.
        // TODO, fix me.
        GGML_ABORT("Cann buffer pool full, increase MAX_CANN_BUFFERS\n");
    }
};

/**
 * @brief A pool of CANN buffers with virtual memory.
 *
 * This class manages a pool of CANN buffers with virtual memory for a specific
 * device.
 */
struct ggml_cann_pool_vmm : public ggml_cann_pool {
    /**
     * @brief The maximum size of the virtual memory pool (32 GB).
     */
    static const size_t CANN_POOL_VMM_MAX_SIZE = 1ull << 35;  // 32 GB

    /**
     * @brief The device ID associated with this buffer pool.
     */
    int device;

    /**
     * @brief Pointer to the start of the virtual memory pool.
     */
    void* pool_addr = 0;

    /**
     * @brief Amount of virtual memory used in the pool.
     */
    size_t pool_used = 0;

    /**
     * @brief Total size of the virtual memory pool.
     */
    size_t pool_size = 0;

    /**
     * @brief Allocation granularity for the virtual memory pool.
     */
    size_t granularity;

    /**
     * @brief Handles for the physical memory allocated.
     */
    std::vector<aclrtDrvMemHandle> handles;

    /**
     * @brief Offsets for the mapped memory regions.
     */
    std::vector<void*> map_offsets;

    /**
     * @brief Constructor to initialize the buffer pool with virtual memory for
     * a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    explicit ggml_cann_pool_vmm(int device)
        : device(device),
          granularity(ggml_cann_info().devices[device].vmm_granularity) {}

    /**
     * @brief Destructor to free all buffers in the virtual memory pool.
     */
    ~ggml_cann_pool_vmm() {
        if (pool_addr != 0) {
            for (auto& offset : map_offsets) {
                ACL_CHECK(aclrtUnmapMem(offset));
            }
            for (auto& handle : handles) {
                ACL_CHECK(aclrtFreePhysical(handle));
            }
            ACL_CHECK(aclrtReleaseMemAddress(pool_addr));
        }
    }

    /**
     * @brief Allocate a buffer of the given size in the virtual memory pool.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    void* alloc(size_t size, size_t* actual_size) override {
        // round up the allocation size to the alignment to ensure that all
        // allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size =
                granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CANN_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            aclrtPhysicalMemProp prop = {};
            prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
            prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
            prop.memAttr = ACL_HBM_MEM_HUGE;
            prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = device;
            prop.reserve = 0;
            aclrtDrvMemHandle handle;
            ACL_CHECK(aclrtMallocPhysical(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                ACL_CHECK(aclrtReserveMemAddress(
                    &pool_addr, CANN_POOL_VMM_MAX_SIZE, 0, NULL, 1));
            }

            // map at the end of the pool
            ACL_CHECK(aclrtMapMem((char*)pool_addr + pool_size, reserve_size, 0,
                                  handle, 0));

            handles.push_back(handle);
            map_offsets.push_back((char*)pool_addr + pool_size);

            // add to the pool
            pool_size += reserve_size;

            // GGML_CANN_LOG_INFO("cann pool[%d]: size increased to %llu MB (
            // reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void* ptr = (void*)((char*)pool_addr + pool_used);
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CANN_MALLOC
        GGML_CANN_LOG_INFO("cann pool[%d]: allocated %llu bytes at %llx\n", device,
               (unsigned long long)size, (unsigned long long)ptr);
#endif
        return ptr;
    }

    /**
     * @brief Free a buffer and return it to the virtual memory pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    void free(void* ptr, size_t size) override {
#ifdef DEBUG_CANN_MALLOC
        GGML_CANN_LOG_INFO("cann pool[%d]: freed %llu bytes at %llx\n", device,
               (unsigned long long)size, (unsigned long long)ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void*)((char*)pool_addr + pool_used));
    }
};

/**
 * @brief Create a new CANN pool for a specific device.
 *
 * Factory method to create a new CANN pool object based on the device type.
 *
 * @param device The device ID for which to create the pool.
 * @return A unique pointer to the created CANN pool.
 */
std::unique_ptr<ggml_cann_pool> ggml_backend_cann_context::new_pool_for_device(
    int device) {
    // return std::unique_ptr<ggml_cann_pool>(new ggml_cann_pool_leg(device));
    return std::unique_ptr<ggml_cann_pool>(new ggml_cann_pool_vmm(device));
}

// cann buffer
/**
 * @brief Context for managing a CANN buffer associated with a specific device.
 *
 * This structure holds information about a CANN buffer, including the device
 * ID, device pointer, and a name derived from GGML_CANN_NAME and the device ID.
 */
struct ggml_backend_cann_buffer_context {
    int32_t device;  ///< The device ID associated with this buffer context.
    void* dev_ptr =
        nullptr;  ///< Pointer to the device memory allocated for the buffer.

    /**
     * @brief Constructor to initialize the CANN buffer context.
     *
     * @param device The device ID associated with this buffer context.
     * @param dev_ptr Pointer to the device memory allocated for the buffer.
     */
    ggml_backend_cann_buffer_context(int32_t device, void* dev_ptr)
        : device(device),
          dev_ptr(dev_ptr) {}

    /**
     * @brief Destructor to free the device memory allocated for the buffer.
     */
    ~ggml_backend_cann_buffer_context() { ACL_CHECK(aclrtFree(dev_ptr)); }
};

/**
 * @brief Retrieve the name associated with a CANN buffer.
 *
 * This function returns the name of a CANN buffer, which is stored in the
 * context of the buffer.
 *
 * @param buffer The CANN buffer whose name is to be retrieved.
 * @return A pointer to a C-string containing the name of the buffer.
 */

GGML_CALL static const char* ggml_backend_cann_buffer_get_name(
    ggml_backend_buffer_t buffer) {
    return "CANN";

    GGML_UNUSED(buffer);
}

/**
 * @brief Check if a buffer is a CANN buffer.
 *
 * This function checks if a given buffer is a CANN buffer by comparing its
 * `get_name` function pointer to `ggml_backend_cann_buffer_get_name`.
 *
 * @param buffer The buffer to check.
 * @return true if the buffer is a CANN buffer, false otherwise.
 */
GGML_CALL static bool ggml_backend_buffer_is_cann(
    ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_cann_buffer_get_name;
}

/**
 * @brief Free resources associated with a CANN buffer.
 *
 * This function frees the resources associated with a CANN buffer, including
 * its context.
 *
 * @param buffer The CANN buffer to free.
 */
GGML_CALL static void ggml_backend_cann_buffer_free_buffer(
    ggml_backend_buffer_t buffer) {
    ggml_backend_cann_buffer_context* ctx =
        (ggml_backend_cann_buffer_context*)buffer->context;
    delete ctx;
}

/**
 * @brief Retrieve the base pointer of a CANN buffer.
 *
 * This function returns the base pointer of a CANN buffer, which points to the
 * device memory allocated for the buffer.
 *
 * @param buffer The CANN buffer whose base pointer is to be retrieved.
 * @return A pointer to the base of the device memory allocated for the buffer.
 */
GGML_CALL static void* ggml_backend_cann_buffer_get_base(
    ggml_backend_buffer_t buffer) {
    ggml_backend_cann_buffer_context* ctx =
        (ggml_backend_cann_buffer_context*)buffer->context;
    return ctx->dev_ptr;
}

/**
 * @brief Transform quantized Q4.0 tensor data into a format suitable for CANN
 * processing.
 *
 * This function transforms quantized Q4.0 tensor data into a format suitable
 * for CANN processing. It extracts quantization values and scales from the
 * source data and prepares them in a format expected by CANN operations.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source data in Q4.0 format.
 * @param dst Pointer to the destination buffer where transformed data will be
 * stored.
 */
GGML_CALL static void ggml_backend_cann_transform_q4_0(ggml_tensor* tensor,
                                                       const void* src,
                                                       void* dst) {

    int64_t n_elems = ggml_nelements(tensor);
    int64_t groups = n_elems / QK4_0;
    size_t quant_bytes = n_elems * sizeof(uint8_t) / 2;

    uint8_t* quant_offset = (uint8_t*)dst;
    uint16_t* scale_offset = (uint16_t*)((char*)dst + quant_bytes);

    for (int i = 0; i < groups; i++) {
        const block_q4_0* group =
            (const block_q4_0*)((const char*)src + i * sizeof(block_q4_0));
        *scale_offset = group->d;
        scale_offset++;

        // 0-15
        for (int j = 0; j < QK4_0 / 2; j += 2) {
            (*quant_offset) = (group->qs[j] & 0x0F);
            (*quant_offset) |= ((group->qs[j + 1] << 4));
            quant_offset++;
        }

        // 16-31
        for (int j = 0; j < QK4_0 / 2; j += 2) {
            (*quant_offset) = (group->qs[j] >> 4);
            (*quant_offset) |= (group->qs[j + 1] & 0xF0);
            quant_offset++;
        }
    }

    // put (uint4b_t -8) into int4b_t
    for (quant_offset = (uint8_t*)dst;
         quant_offset < (uint8_t*)dst + quant_bytes; quant_offset++) {
        (*quant_offset) ^= 0x88;
    }
}

/**
 * @brief Transform CANN processed data back into quantized Q4.0 format.
 *
 * This function transforms CANN processed data back into quantized Q4.0 format.
 * It reverses the transformation performed by
 * ggml_backend_cann_transform_q4_0(), converting the data back into its
 * original quantized form.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source buffer containing transformed data.
 * @param dst Pointer to the destination buffer where the Q4.0 formatted data
 * will be stored.
 */
GGML_CALL static void ggml_backend_cann_transform_back_q4_0(
    const ggml_tensor* tensor, void* src, void* dst) {

    int64_t n_elems = ggml_nelements(tensor);
    int64_t groups = n_elems / QK4_0;
    size_t quant_bytes = n_elems * sizeof(uint8_t) / 2;

    uint8_t* quant_offset = (uint8_t*)src;
    uint16_t* scale_offset = (uint16_t*)((char*)src + quant_bytes);

    for (; quant_offset < (uint8_t*)src + quant_bytes; quant_offset++) {
        (*quant_offset) ^= 0x88;
    }
    quant_offset = (uint8_t*)src;

    for (int i = 0; i < groups; i++) {
        block_q4_0* group = (block_q4_0*)((char*)dst + i * sizeof(block_q4_0));
        group->d = *scale_offset;
        scale_offset++;

        // 0-15
        for (int j = 0; j < QK4_0 / 2; j += 2) {
            group->qs[j] = ((*quant_offset) & 0x0F);
            group->qs[j + 1] = ((*quant_offset) >> 4);
            quant_offset++;
        }

        // 16-31
        for (int j = 0; j < QK4_0 / 2; j += 2) {
            group->qs[j] |= ((*quant_offset) << 4);
            group->qs[j + 1] |= ((*quant_offset) & 0xF0);
            quant_offset++;
        }
    }
}

/**
 * @brief Transform quantized Q8.0 tensor data into a format suitable for CANN
 * processing.
 *
 * This function transforms quantized Q8.0 tensor data into a format suitable
 * for CANN processing. It extracts quantization values and scales from the
 * source data and prepares them in a format expected by CANN operations.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source data in Q8.0 format.
 * @param dst Pointer to the destination buffer where transformed data will be
 * stored.
 */
GGML_CALL static void ggml_backend_cann_transform_q8_0(ggml_tensor* tensor,
                                                       const void* src,
                                                       void* dst) {
    int64_t n_elems = ggml_nelements(tensor);
    int64_t groups = n_elems / QK8_0;
    size_t quant_bytes = n_elems * sizeof(uint8_t);

    uint8_t* quant_offset = (uint8_t*)dst;
    uint16_t* scale_offset = (uint16_t*)((char*)dst + quant_bytes);

    for (int i = 0; i < groups; i++) {
        const block_q8_0* group =
            (const block_q8_0*)((const char*)src + i * sizeof(block_q8_0));
        *scale_offset = group->d;
        scale_offset++;
        size_t group_quant_size = QK8_0 * sizeof(uint8_t);
        memcpy(quant_offset, group->qs, group_quant_size);
        quant_offset += group_quant_size;
    }
}

/**
 * @brief Transform CANN processed data back into quantized Q8.0 format.
 *
 * This function transforms CANN processed data back into quantized Q8.0 format.
 * It reverses the transformation performed by
 * ggml_backend_cann_transform_q8_0(), converting the data back into its
 * original quantized form.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source buffer containing transformed data.
 * @param dst Pointer to the destination buffer where the Q8.0 formatted data
 * will be stored.
 */
GGML_CALL static void ggml_backend_cann_transform_back_q8_0(
    const ggml_tensor* tensor, const void* src, void* dst) {
    int64_t n_elems = ggml_nelements(tensor);
    int64_t groups = n_elems / QK8_0;
    size_t quant_bytes = n_elems * sizeof(uint8_t);

    const uint8_t* quant_offset = (const uint8_t*)src;
    const uint16_t* scale_offset =
        (const uint16_t*)((const char*)src + quant_bytes);

    for (int i = 0; i < groups; i++) {
        block_q8_0* group = (block_q8_0*)((char*)dst + i * sizeof(block_q8_0));
        group->d = *scale_offset;
        scale_offset++;
        size_t group_quant_size = QK8_0 * sizeof(uint8_t);
        memcpy(group->qs, quant_offset, group_quant_size);
        quant_offset += group_quant_size;
    }
}

/**
 * @brief Transform tensor data based on its type for CANN processing.
 *
 * This function transforms tensor data based on its quantization type for CANN
 * processing. It dispatches the transformation based on the tensor's type to
 * specialized functions handling Q4.0 and Q8.0 formats.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source data to be transformed.
 * @param dst Pointer to the destination buffer where transformed data will be
 * stored.
 */
GGML_CALL static void ggml_backend_cann_transform(ggml_tensor* tensor,
                                                  const void* src, void* dst) {
    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            ggml_backend_cann_transform_q4_0(tensor, src, dst);
            break;
        case GGML_TYPE_Q8_0:
            ggml_backend_cann_transform_q8_0(tensor, src, dst);
            break;
        default:
            break;
    }
}

/**
 * @brief Transform CANN processed data back into tensor data based on its type.
 *
 * This function transforms CANN processed data back into tensor data based on
 * its quantization type for Q4.0 and Q8.0 formats. It dispatches the
 * transformation based on the tensor's type to specialized functions.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source data containing CANN processed data.
 * @param dst Pointer to the destination buffer where transformed tensor data
 * will be stored.
 */
GGML_CALL static void ggml_backend_cann_transform_back(
    const ggml_tensor* tensor, void* src, void* dst) {
    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            ggml_backend_cann_transform_back_q4_0(tensor, src, dst);
            break;
        case GGML_TYPE_Q8_0:
            ggml_backend_cann_transform_back_q8_0(tensor, src, dst);
            break;
        default:
            break;
    }
}

/**
 * @brief Check if transformation is needed for a given tensor type.
 *
 * This function checks if transformation is needed for a given tensor type
 * to prepare data for CANN processing.
 *
 * @param type The tensor type to check.
 * @return true if transformation is needed, false otherwise.
 */
GGML_CALL static bool need_transform(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

/**
 * @brief Initialize a tensor using data from a CANN buffer.
 *
 * This function initializes a tensor using data from a CANN buffer.
 * It handles special cases such as views and quantization.
 *
 * @param buffer The CANN buffer from which to initialize the tensor.
 * @param tensor Pointer to the tensor to be initialized.
 */
GGML_CALL static void ggml_backend_cann_buffer_init_tensor(
    ggml_backend_buffer_t buffer, ggml_tensor* tensor) {
    if (tensor->view_src != NULL && tensor->view_offs == 0) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
        return;
    }

    // TODO: can backend doesn't support quantized yet. Just leave the code
    // here.
    if (ggml_is_quantized(tensor->type)) {
        // Initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size =
            ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size && tensor->view_src == nullptr) {
            size_t memset_size = padded_size - original_size;
            ACL_CHECK(aclrtMemset((char*)tensor->data + original_size,
                                  memset_size, 0, memset_size));
        }
    }
}

// TODO: need handle tensor which has paddings.
/**
 * @brief Set tensor data in a CANN buffer.
 *
 * This function sets tensor data in a CANN buffer, handling transformations
 * if needed based on the tensor's type.
 *
 * @param buffer The CANN buffer where the tensor data will be set.
 * @param tensor Pointer to the tensor whose data will be set.
 * @param data Pointer to the source data to be copied into the tensor.
 * @param offset Offset in the source data from where to start copying.
 * @param size Size of the data to be copied, in bytes.
 */
GGML_CALL static void ggml_backend_cann_buffer_set_tensor(
    ggml_backend_buffer_t buffer, ggml_tensor *tensor, const void *data,
    size_t offset, size_t size) {
    ggml_backend_cann_buffer_context *ctx =
        (ggml_backend_cann_buffer_context *)buffer->context;

    ggml_cann_set_device(ctx->device);
    // TODO: refer to cann(#6017), it use thread's default stream.
    // For acl, synchronous functions use this default stream.
    // Why aclrtSynchronizeDevice?

    if (!need_transform(tensor->type)) {
        ACL_CHECK(aclrtMemcpy((char *)tensor->data + offset, size, data, size,
                              ACL_MEMCPY_HOST_TO_DEVICE));
    } else {
        void *transform_buffer = malloc(size);
        ggml_backend_cann_transform(tensor, data, transform_buffer);

#ifndef NDEBUG
        void *check_buffer = malloc(size);
        ggml_backend_cann_transform_back(tensor, transform_buffer,
                                         check_buffer);
        GGML_ASSERT(memcmp(data, check_buffer, size) == 0);
        free(check_buffer);
#endif
        ACL_CHECK(aclrtMemcpy((char *)tensor->data + offset, size,
                              transform_buffer, size,
                              ACL_MEMCPY_HOST_TO_DEVICE));
        free(transform_buffer);
    }
}

/**
 * @brief Get tensor data from a CANN buffer.
 *
 * This function retrieves tensor data from a CANN buffer, handling
 * transformations if needed based on the tensor's type.
 *
 * @param buffer The CANN buffer from which to retrieve tensor data.
 * @param tensor Pointer to the tensor whose data will be retrieved.
 * @param data Pointer to the destination buffer where the tensor data will be
 * copied.
 * @param offset Offset in the destination buffer where to start copying.
 * @param size Size of the data to be copied, in bytes.
 */
GGML_CALL static void ggml_backend_cann_buffer_get_tensor(
    ggml_backend_buffer_t buffer, const ggml_tensor* tensor, void* data,
    size_t offset, size_t size) {
    ggml_backend_cann_buffer_context* ctx =
        (ggml_backend_cann_buffer_context*)buffer->context;

    ggml_cann_set_device(ctx->device);

    if (!need_transform(tensor->type)) {
        ACL_CHECK(aclrtMemcpy(data, size, (char*)tensor->data + offset, size,
                              ACL_MEMCPY_DEVICE_TO_HOST));
    } else {
        void* transform_buffer = malloc(size);
        ACL_CHECK(aclrtMemcpy(transform_buffer, size,
                              (char*)tensor->data + offset, size,
                              ACL_MEMCPY_DEVICE_TO_HOST));
        ggml_backend_cann_transform_back(tensor, transform_buffer, data);
        free(transform_buffer);
    }
}

/**
 * @brief Copy tensor data between CANN buffers if possible.
 *
 * This function copies tensor data between CANN buffers if the source and
 * destination buffers are CANN buffers and they meet the necessary conditions
 * (same device or devices can access each other).
 *
 * @param buffer The destination CANN buffer where the tensor data will be
 * copied.
 * @param src Pointer to the source tensor whose data will be copied.
 * @param dst Pointer to the destination tensor where the data will be copied.
 * @return true if the copy operation succeeded, false otherwise.
 */
GGML_CALL static bool ggml_backend_cann_buffer_cpy_tensor(
    ggml_backend_buffer_t buffer, const ggml_tensor* src, ggml_tensor* dst) {
    if (ggml_backend_buffer_is_cann(src->buffer)) {
        ggml_backend_cann_buffer_context* src_ctx =
            (ggml_backend_cann_buffer_context*)src->buffer->context;
        ggml_backend_cann_buffer_context* dst_ctx =
            (ggml_backend_cann_buffer_context*)buffer->context;

        size_t memcpy_size = ggml_nbytes(src);
        // Same device.
        if (src_ctx->device == dst_ctx->device) {
            ACL_CHECK(aclrtMemcpy((char*)dst->data, memcpy_size,
                                  (const char*)src->data, memcpy_size,
                                  ACL_MEMCPY_DEVICE_TO_DEVICE));
            return true;
        } else {
            // Different device but can access by peer.
            int32_t canAccessPeer = 0;
            ACL_CHECK(aclrtDeviceCanAccessPeer(&canAccessPeer, src_ctx->device,
                                               dst_ctx->device));
            if (canAccessPeer) {
                ggml_cann_set_device(src_ctx->device);
                ACL_CHECK(aclrtDeviceEnablePeerAccess(dst_ctx->device, 0));
                ACL_CHECK(aclrtMemcpy((char*)dst->data, memcpy_size,
                                      (const char*)src->data, memcpy_size,
                                      ACL_MEMCPY_DEVICE_TO_DEVICE));
                return true;
            }
        }
    }
    return false;
}

/**
 * @brief Clear a CANN buffer by setting all its memory to a specified value.
 *
 * This function clears a CANN buffer by setting all its memory to a specified
 * value.
 *
 * @param buffer The CANN buffer to be cleared.
 * @param value The value to which each byte in the buffer will be set.
 */
GGML_CALL static void ggml_backend_cann_buffer_clear(
    ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cann_buffer_context* ctx =
        (ggml_backend_cann_buffer_context*)buffer->context;

    ggml_cann_set_device(ctx->device);
    ACL_CHECK(aclrtMemset(ctx->dev_ptr, buffer->size, value, buffer->size));
}

/**
 * @brief Interface for a CANN buffer in the backend.
 *
 * This structure defines function pointers to operations that can be performed
 * on a CANN buffer within the backend.
 */
static ggml_backend_buffer_i ggml_backend_cann_buffer_interface = {
    /* .get_name        = */ ggml_backend_cann_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_cann_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cann_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cann_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_cann_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cann_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_cann_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cann_buffer_clear,
    /* .reset           = */ NULL,
};

// cann buffer type
/**
 * @brief Structure representing context information for a specific backend
 * buffer type.
 */
struct ggml_backend_cann_buffer_type_context {
    int32_t
        device; /**< Device identifier associated with the buffer context. */
    std::string name; /**< Name associated with the buffer context. */
};

/**
 * @brief Retrieves the name associated with a CANN buffer type.
 *
 * This function returns the descriptive name associated with the specified
 * CANN buffer type context.
 *
 * @param buft Pointer to the buffer type context.
 * @return Const pointer to the C-style string containing the name.
 */
GGML_CALL static const char* ggml_backend_cann_buffer_type_name(
    ggml_backend_buffer_type_t buft) {
    return "CANN";

    GGML_UNUSED(buft);
}

/**
 * @brief Allocates a new CANN buffer of the specified type and size.
 *
 * This function allocates a new CANN buffer on the specified device with the
 * given size.
 *
 * @param buft Pointer to the buffer type context.
 * @param size Size in bytes of the buffer to allocate.
 * @return Pointer to the allocated buffer, or nullptr if allocation fails.
 */
GGML_CALL static ggml_backend_buffer_t
ggml_backend_cann_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) {
    ggml_backend_cann_buffer_type_context* buft_ctx =
        (ggml_backend_cann_buffer_type_context*)buft->context;

    ggml_cann_set_device(buft_ctx->device);

    size = std::max(size, (size_t)1);

    void* dev_ptr;
    aclError err = aclrtMalloc(&dev_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != ACL_SUCCESS) {
        GGML_CANN_LOG_ERROR(
            "%s: allocating %.2f MiB on device %d: aclrtMalloc failed: %s\n",
            __func__, size / 1024.0 / 1024.0, buft_ctx->device,
            aclGetRecentErrMsg());
        return nullptr;
    }

    ggml_backend_cann_buffer_context* ctx =
        new ggml_backend_cann_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cann_buffer_interface,
                                    ctx, size);
}

/**
 * @brief Retrieves the memory alignment requirement for CANN buffers of this
 * type.
 *
 * This function returns the alignment requirement in bytes for memory allocated
 * by the CANN buffer type.
 *
 * @param buft Pointer to the buffer type context (unused in this
 * implementation).
 * @return The alignment requirement in bytes (fixed at 128 bytes for CANN
 * buffers).
 */
GGML_CALL static size_t ggml_backend_cann_buffer_type_get_alignment(
    ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

/**
 * @brief Calculates the allocation size required for a tensor in a CANN buffer.
 *
 * Computes the total allocation size needed for storing the tensor's data in a
 * CANN buffer, considering any necessary padding or adjustments for quantized
 * types.
 *
 * @param buft Pointer to the buffer type context (unused in this
 * implementation).
 * @param tensor Pointer to the tensor for which the allocation size is
 * calculated.
 * @return The total allocation size in bytes required for the tensor in the
 * CANN buffer.
 */
GGML_CALL static size_t ggml_backend_cann_buffer_type_get_alloc_size(
    ggml_backend_buffer_type_t buft, const ggml_tensor* tensor) {
    size_t size = ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    // last line must bigger than 32, because every single op deal at
    // least 32 bytes.
    // TODO: quantized type?
    // int64_t line_size = ne0 * ggml_element_size(tensor);
    // int64_t line_size_align_32 = (line_size + 31) & ~31;
    // size += (line_size_align_32 - line_size);

    // TODO: not support quantized yet.
    // TODO: consider un-continue tensor.
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(
                tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    GGML_UNUSED(buft);
}

/**
 * @brief Interface for managing CANN buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for CANN buffer types in the GGML backend.
 */
static ggml_backend_buffer_type_i ggml_backend_cann_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cann_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_cann_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cann_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL,  // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cann_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

/**
 * @brief Retrieves the CANN buffer type for a specified device.
 *
 * This function initializes and returns the buffer type interface associated
 * with the given device. It ensures thread-safe access using a mutex.
 *
 * @param device The device index for which to retrieve the buffer type.
 * @return A pointer to the buffer type interface for the specified device, or
 * nullptr if the device index is out of range.
 */
GGML_CALL ggml_backend_buffer_type_t
ggml_backend_cann_buffer_type(int32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cann_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type
        ggml_backend_cann_buffer_types[GGML_CANN_MAX_DEVICES];

    static bool ggml_backend_cann_buffer_type_initialized = false;

    if (!ggml_backend_cann_buffer_type_initialized) {
        for (int32_t i = 0; i < GGML_CANN_MAX_DEVICES; i++) {
            ggml_backend_cann_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cann_buffer_type_interface,
                /* .context  = */
                 new ggml_backend_cann_buffer_type_context{
                    i, "CANN" + std::to_string(i)},
            };
        }
        ggml_backend_cann_buffer_type_initialized = true;
    }

    return &ggml_backend_cann_buffer_types[device];
}

/**
 * @brief Computes the forward operation for a given tensor using CANN
 * operations.
 *
 * This function selects the appropriate CANN operation based on the type of
 * operation specified in the tensor and performs the computation.
 *
 * @param ctx The CANN context containing necessary resources and
 * configurations.
 * @param dst The destination tensor where the result of the computation will be
 * stored.
 * @return true if the computation was successful; false otherwise.
 */
static bool ggml_cann_compute_forward(ggml_backend_cann_context& ctx,
                                      struct ggml_tensor* dst) {
    switch (dst->op) {
        case GGML_OP_REPEAT:
            ggml_cann_repeat(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cann_get_rows(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cann_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
            ggml_cann_add(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cann_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cann_mul_div<aclnnMulGetWorkspaceSize, aclnnMul>(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cann_mul_div<aclnnDivGetWorkspaceSize, aclnnDiv>(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_GELU:
                    ggml_cann_activation<aclnnGeluGetWorkspaceSize, aclnnGelu>(
                        ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cann_activation<aclnnSiluGetWorkspaceSize, aclnnSilu>(
                        ctx, dst);
                    break;
                // TODO: Use faster gelu??
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cann_activation<aclnnGeluGetWorkspaceSize, aclnnGelu>(
                        ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cann_activation<aclnnTanhGetWorkspaceSize, aclnnTanh>(
                        ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cann_activation<aclnnReluGetWorkspaceSize, aclnnRelu>(
                        ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cann_activation<aclnnHardsigmoidGetWorkspaceSize,
                                         aclnnHardsigmoid>(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cann_activation<aclnnHardswishGetWorkspaceSize,
                                         aclnnHardswish>(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cann_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cann_group_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cann_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cann_upsample_nearest2d(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cann_pad(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cann_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cann_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cann_leaky_relu(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cann_rms_norm(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cann_mul_mat(ctx, dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            return false;
        case GGML_OP_SCALE:
            ggml_cann_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cann_sqr(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cann_clamp(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cann_cpy(ctx, dst);
            break;
        case GGML_OP_CONT:
            ggml_cann_dup(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cann_diag_mask(ctx, dst, -INFINITY);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cann_softmax(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cann_rope(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cann_im2col(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cann_pool2d(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cann_sum_rows(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cann_argsort(ctx, dst);
            break;
        default:
            return false;
    }

    return true;
}

// backend
/**
 * @brief Retrieves the name associated with the CANN backend.
 *
 * This function returns the name assigned to the CANN backend, which is stored
 * in the context of the provided backend structure.
 *
 * @param backend Pointer to the CANN backend structure.
 * @return A pointer to a constant string representing the backend name.
 */
GGML_CALL static const char* ggml_backend_cann_name(ggml_backend_t backend) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    return cann_ctx->name.c_str();
}

/**
 * @brief Frees resources associated with the CANN backend.
 *
 * This function releases resources associated with the CANN backend context
 * and resets the device associated with the backend to its initial state.
 *
 * @param backend Pointer to the CANN backend structure to be freed.
 */
GGML_CALL static void ggml_backend_cann_free(ggml_backend_t backend) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;
    ACL_CHECK(aclrtSynchronizeDevice());
    ACL_CHECK(aclrtResetDevice(cann_ctx->device));

    // finalize when last backend freed.
    if (cann_ctx->device == ggml_backend_cann_get_device_count() - 1) {
        ACL_CHECK(aclFinalize());
    }

    delete cann_ctx;
    delete backend;
}

/**
 * @brief Retrieves the default buffer type associated with the CANN backend.
 *
 * This function returns the buffer type specific to the device associated
 * with the CANN backend. It is used to allocate buffers for computations
 * performed by the backend.
 *
 * @param backend Pointer to the CANN backend structure.
 * @return Pointer to the buffer type structure for the CANN backend.
 */
GGML_CALL static ggml_backend_buffer_type_t
ggml_backend_cann_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    return ggml_backend_cann_buffer_type(cann_ctx->device);
}

/**
 * @brief Sets tensor data asynchronously in the CANN backend.
 *
 * This function asynchronously sets tensor data in the CANN backend. Depending
 * on the tensor type, it may perform data transformations before copying data
 * to the device.
 *
 * @param backend Pointer to the CANN backend structure.
 * @param tensor Pointer to the tensor structure to set data for.
 * @param data Pointer to the host data to copy to the tensor.
 * @param offset Offset in bytes within the host data.
 * @param size Size of the data to copy in bytes.
 */
GGML_CALL static void ggml_backend_cann_set_tensor_async(ggml_backend_t backend,
                                                         ggml_tensor *tensor,
                                                         const void *data,
                                                         size_t offset,
                                                         size_t size) {
    ggml_backend_cann_context *cann_ctx =
        (ggml_backend_cann_context *)backend->context;

    if (!need_transform(tensor->type)) {
        ACL_CHECK(aclrtMemcpyAsync((char *)tensor->data + offset, size, data,
                                   size, ACL_MEMCPY_HOST_TO_DEVICE,
                                   cann_ctx->stream()));
    } else {
        void *transform_buffer = malloc(size);
        ggml_backend_cann_transform(tensor, data, transform_buffer);

#ifndef NDEBUG
        void *check_buffer = malloc(size);
        ggml_backend_cann_transform_back(tensor, transform_buffer,
                                         check_buffer);
        GGML_ASSERT(memcmp(data, check_buffer, size));
        free(check_buffer);
#endif
        ACL_CHECK(aclrtMemcpyAsync(
            (char *)tensor->data + offset, size, transform_buffer, size,
            ACL_MEMCPY_HOST_TO_DEVICE, cann_ctx->stream()));
        ACL_CHECK(aclrtSynchronizeStream(cann_ctx->stream()));
        free(transform_buffer);
    }
}

GGML_CALL static void ggml_backend_cann_get_tensor_async(
    ggml_backend_t backend, const ggml_tensor *tensor, void *data,
    size_t offset, size_t size) {
    ggml_backend_cann_context *cann_ctx =
        (ggml_backend_cann_context *)backend->context;
    ggml_backend_buffer_t buf =
        tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cann_buffer_type(cann_ctx->device) &&
                "unsupported buffer type");

    if (!need_transform(tensor->type)) {
        ACL_CHECK(aclrtMemcpyAsync(data, size, (char *)tensor->data + offset,
                                   size, ACL_MEMCPY_DEVICE_TO_HOST,
                                   cann_ctx->stream()));
    } else {
        void *transform_buffer = malloc(size);
        ACL_CHECK(aclrtMemcpyAsync(
            transform_buffer, size, (char *)tensor->data + offset, size,
            ACL_MEMCPY_DEVICE_TO_HOST, cann_ctx->stream()));
        ACL_CHECK(aclrtSynchronizeStream(cann_ctx->stream()));
        ggml_backend_cann_transform_back(tensor, transform_buffer, data);
        free(transform_buffer);
    }
}

/**
 * @brief Asynchronously copies tensor data between CANN backends.
 *
 * This function copies tensor data asynchronously between two CANN backends. It
 * checks if both tensors reside in CANN buffers and whether the devices support
 * peer-to-peer access for direct copying. If not, it returns false.
 *
 * @param backend_src Pointer to the source CANN backend structure.
 * @param backend_dst Pointer to the destination CANN backend structure.
 * @param src Pointer to the source tensor to copy data from.
 * @param dst Pointer to the destination tensor to copy data to.
 * @return true if the copy operation succeeds, false otherwise.
 */
GGML_CALL static bool ggml_backend_cann_cpy_tensor_async(
    ggml_backend_t backend_src, ggml_backend_t backend_dst,
    const ggml_tensor* src, ggml_tensor* dst) {
    GGML_ASSERT(ggml_backend_is_cann(backend_src) ||
                ggml_backend_is_cann(backend_dst));

    if (!ggml_backend_buffer_is_cann(src->buffer) ||
        !ggml_backend_buffer_is_cann(dst->buffer)) {
        return false;
    }

    ggml_backend_buffer_t buf_src =
        src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst =
        dst->view_src ? dst->view_src->buffer : dst->buffer;

    ggml_backend_cann_context* cann_ctx_src =
        (ggml_backend_cann_context*)backend_src->context;
    ggml_backend_cann_context* cann_ctx_dst =
        (ggml_backend_cann_context*)backend_dst->context;

    size_t copy_size = ggml_nbytes(dst);
    if (backend_src != backend_dst) {
        ggml_backend_cann_buffer_context* buf_ctx_src =
            (ggml_backend_cann_buffer_context*)buf_src->context;
        ggml_backend_cann_buffer_context* buf_ctx_dst =
            (ggml_backend_cann_buffer_context*)buf_dst->context;

        GGML_ASSERT(cann_ctx_src->device == buf_ctx_src->device);
        GGML_ASSERT(cann_ctx_dst->device == buf_ctx_dst->device);

        int32_t canAccessPeer = 0;
        ACL_CHECK(aclrtDeviceCanAccessPeer(&canAccessPeer, cann_ctx_src->device,
                                           cann_ctx_dst->device));
        if (!canAccessPeer) {
            return false;
        }

        // need open both directions for memcpyasync between devices.
        ggml_cann_set_device(cann_ctx_dst->device);
        ACL_CHECK(aclrtDeviceEnablePeerAccess(cann_ctx_src->device, 0));
        ggml_cann_set_device(cann_ctx_src->device);
        ACL_CHECK(aclrtDeviceEnablePeerAccess(cann_ctx_dst->device, 0));

        ACL_CHECK(aclrtMemcpyAsync(dst->data, copy_size, src->data, copy_size,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE,
                                   cann_ctx_src->stream()));

        //TODO: workaround for Event didn`t work here.
        aclrtSynchronizeStream(cann_ctx_src->stream());
    } else {
        // src and dst are on the same backend
        ACL_CHECK(aclrtMemcpyAsync(dst->data, copy_size, src->data, copy_size,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE,
                                   cann_ctx_dst->stream()));
    }

    return true;
}

/**
 * @brief Synchronizes a CANN backend.
 *
 * This function synchronizes the specified CANN backend by waiting for all
 * operations in its associated stream to complete.
 *
 * @param backend Pointer to the CANN backend structure to synchronize.
 */
GGML_CALL static void ggml_backend_cann_synchronize(ggml_backend_t backend) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    ggml_cann_set_device(cann_ctx->device);

    ACL_CHECK(aclrtSynchronizeStream(cann_ctx->stream()));
}

/**
 * @brief Computes a computational graph using a CANN backend.
 *
 * This function computes the operations defined in the computational graph
 * using the specified CANN backend.
 *
 * @param backend Pointer to the CANN backend structure to use for computation.
 * @param cgraph Pointer to the computational graph structure containing nodes
 *               representing operations to be computed.
 * @return enum ggml_status Returns GGML_STATUS_SUCCESS if computation
 *         completes successfully, otherwise an appropriate error status.
 */
GGML_CALL static enum ggml_status ggml_backend_cann_graph_compute(
    ggml_backend_t backend, ggml_cgraph* cgraph) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    ggml_cann_set_device(cann_ctx->device);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor* node = cgraph->nodes[i];

        if (ggml_is_empty(node) || node->op == GGML_OP_NONE) {
            continue;
        }

        bool ok = ggml_cann_compute_forward(*cann_ctx, node);

        if (!ok) {
            GGML_CANN_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__,
                    node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
    }

    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Checks if the CANN backend supports a specific operation.
 *
 * This function checks whether the specified operation is supported by the
 * CANN backend.
 *
 * @param backend Pointer to the CANN backend structure to check support for
 *                the operation.
 * @param op Pointer to the tensor representing the operation to check.
 * @return bool Returns true if the operation is supported by the backend,
 *              otherwise false.
 */
GGML_CALL static bool ggml_backend_cann_supports_op(ggml_backend_t backend,
                                                    const ggml_tensor* op) {
    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                    return true;
                default:
                    return false;
            }
        case GGML_OP_MUL_MAT: {
            switch (op->src[0]->type) {
                case GGML_TYPE_F16:
                case GGML_TYPE_F32:
                case GGML_TYPE_Q8_0:
                    // TODO: fix me
                    // Current groupsize should not be greater than k-1 in
                    // aclnnWeightQuantBatchMatmulV2GetWorkspaceSize().
                case GGML_TYPE_Q4_0:
                    return true;
                default:
                    return false;
            }
        }
        case GGML_OP_MUL_MAT_ID:
            return false;
        // embedding
        case GGML_OP_GET_ROWS: {
            switch (op->src[0]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q8_0:
                    return true;
                default:
                    return false;
            }
        } break;
        case GGML_OP_CPY: {
            switch (op->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_0:
                    return true;
                default:
                    return false;
            }
        }
        case GGML_OP_DUP:
        case GGML_OP_REPEAT:
        case GGML_OP_CONCAT:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_NORM:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_CLAMP:
        case GGML_OP_CONT:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ROPE:
        case GGML_OP_IM2COL:
        case GGML_OP_POOL_2D:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_ARGSORT:
        case GGML_OP_ACC:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
            return true;
        default:
            return false;
    }

    GGML_UNUSED(backend);
}

/**
 * @brief Checks if the backend buffer type is associated with the CANN backend.
 *
 * This function checks whether the provided backend buffer type is associated
 * with the CANN backend based on the comparison of its name retrieval function
 * pointer.
 *
 * @param buft Pointer to the backend buffer type to check.
 * @return bool Returns true if the buffer type is associated with the CANN
 * backend, otherwise false.
 */
static bool ggml_backend_buft_is_cann(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cann_buffer_type_name;
}

/**
 * @brief Checks if the CANN backend supports a specific backend buffer type.
 *
 * This function determines whether the CANN backend supports the given backend
 * buffer type by comparing the device context of the backend and buffer type.
 * It returns true if the devices are same between the backend context and
 * buffer type context.
 *
 * @param backend Pointer to the CANN backend.
 * @param buft Pointer to the backend buffer type to check.
 * @return bool Returns true if the CANN backend supports the buffer type,
 *              otherwise false.
 */
GGML_CALL static bool ggml_backend_cann_supports_buft(
    ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    if (ggml_backend_buft_is_cann(buft)) {
        ggml_backend_cann_context * cann_ctx =
                        (ggml_backend_cann_context *)backend->context;
        ggml_backend_cann_buffer_type_context * buft_ctx =
                        (ggml_backend_cann_buffer_type_context *)buft->context;
        return buft_ctx->device == cann_ctx->device;
    }
    return false;
}

/**
 * @brief Determines if a tensor operation should be offloaded to the CANN
 * backend.
 *
 * This function checks if a given tensor operation should be offloaded to the
 * CANN backend based on the operation type and the size of the tensor. It
 * returns true if the second dimension (ne[1]) of the tensor is greater than or
 * equal to the minimum batch size and the operation is not GGML_OP_GET_ROWS.
 *
 * @param backend Pointer to the CANN backend.
 * @param op Pointer to the tensor operation to check.
 * @return bool Returns true if the operation should be offloaded, otherwise
 * false.
 */
GGML_CALL static bool ggml_backend_cann_offload_op(ggml_backend_t backend,
                                                   const ggml_tensor* op) {
    const int min_batch_size = 32;
    GGML_UNUSED(backend);

    return op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS;
}

/**
 * @brief Creates a new event for the CANN backend.
 *
 * This function initializes a new event for the CANN backend by setting the
 * device and creating an ACL runtime event. The created event is then wrapped
 * in a ggml_backend_event structure and returned.
 *
 * @param backend Pointer to the CANN backend.
 * @return ggml_backend_event_t Returns a pointer to the new event structure.
 */
static ggml_backend_event_t ggml_backend_cann_event_new(
    ggml_backend_t backend) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    ggml_cann_set_device(cann_ctx->device);

    aclrtEvent event;
    ACL_CHECK(aclrtCreateEvent(&event));

    return new ggml_backend_event{
        /* .backend = */ backend,
        /* .context = */ event,
    };
}

/**
 * @brief Frees a CANN backend event.
 *
 * This function destroys the ACL runtime event associated with the given CANN
 * backend event and then deletes the event structure itself.
 *
 * @param event Pointer to the event structure to be freed.
 */
static void ggml_backend_cann_event_free(ggml_backend_event_t event) {
    ACL_CHECK(aclrtDestroyEvent((aclrtEvent)event->context));

    delete event;
}

/**
 * @brief Records an event on the CANN backend stream.
 *
 * This function records the given event on the ACL runtime stream associated
 * with the backend context.
 *
 * @param event Pointer to the event structure to be recorded.
 */
static void ggml_backend_cann_event_record(ggml_backend_event_t event) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)event->backend->context;

    ACL_CHECK(aclrtRecordEvent((aclrtEvent)event->context, cann_ctx->stream()));
}

/**
 * @brief Waits for a recorded event to complete on the CANN backend stream.
 *
 * This function makes the given backend wait for the event to complete on its
 * ACL runtime stream.
 *
 * @param backend Pointer to the backend structure.
 * @param event Pointer to the event structure that the backend needs to wait
 * for.
 */
static void ggml_backend_cann_event_wait(ggml_backend_t backend,
                                         ggml_backend_event_t event) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    if (ggml_backend_is_cann(event->backend)) {
        ACL_CHECK(aclrtStreamWaitEvent(cann_ctx->stream(),
                                       (aclrtEvent)event->context));
    } else {
        GGML_ABORT("fatal error");
    }
}

/**
 * @brief Synchronizes the given event on the CANN backend.
 *
 * This function waits for the specified event to complete on the ACL runtime.
 *
 * @param event Pointer to the event structure to be synchronized.
 */
static void ggml_backend_cann_event_synchronize(ggml_backend_event_t event) {
    ACL_CHECK(aclrtSynchronizeEvent((aclrtEvent)event->context));
}

/**
 * @brief Structure defining the interface for the CANN backend.
 *
 * This structure contains function pointers for various operations
 * supported by the CANN backend, including name retrieval, memory
 * management, tensor operations, synchronization, and event handling.
 */
static ggml_backend_i ggml_backend_cann_interface = {
    /* .get_name                = */ ggml_backend_cann_name,
    /* .free                    = */ ggml_backend_cann_free,
    /* .get_default_buffer_type = */ ggml_backend_cann_get_default_buffer_type,
    /* .set_tensor_async        = */ ggml_backend_cann_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cann_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_cann_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cann_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cann_graph_compute,
    /* .supports_op             = */ ggml_backend_cann_supports_op,
    /* .supports_buft           = */ ggml_backend_cann_supports_buft,
    /* .offload_op              = */ ggml_backend_cann_offload_op,
    /* .event_new               = */ ggml_backend_cann_event_new,
    /* .event_free              = */ ggml_backend_cann_event_free,
    /* .event_record            = */ ggml_backend_cann_event_record,
    /* .event_wait              = */ ggml_backend_cann_event_wait,
    /* .event_synchronize       = */ ggml_backend_cann_event_synchronize,
};

/**
 * @brief Return the hardcoded GUID for the CANN backend.
 *
 * This function returns a static GUID which uniquely identifies the CANN
 * backend.
 *
 * @return A pointer to the static GUID.
 */
static ggml_guid_t ggml_backend_cann_guid() {
    static ggml_guid guid = {0xa1, 0x94, 0xaf, 0xac, 0xbd, 0x4f, 0x47, 0x34,
                             0xbe, 0x1a, 0x9e, 0x71, 0x1f, 0x9e, 0xed, 0x64};
    return &guid;
}

GGML_CALL ggml_backend_t ggml_backend_cann_init(int32_t device) {
    aclInit(nullptr);
    if (device < 0 || device >= ggml_backend_cann_get_device_count()) {
        GGML_CANN_LOG_ERROR("%s: error: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cann_context* ctx = new ggml_backend_cann_context(device);
    if (ctx == nullptr) {
        GGML_CANN_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cann_backend =
        new ggml_backend{/* .guid      = */ ggml_backend_cann_guid(),
                         /* .interface = */ ggml_backend_cann_interface,
                         /* .context   = */ ctx};

    return cann_backend;
}

GGML_CALL bool ggml_backend_is_cann(ggml_backend_t backend) {
    return backend != NULL &&
           ggml_guid_matches(backend->guid, ggml_backend_cann_guid());
}

GGML_CALL int32_t ggml_backend_cann_get_device_count() {
    return ggml_cann_info().device_count;
}

GGML_CALL void ggml_backend_cann_get_device_description(
    int32_t device, char* description, size_t description_size) {
    ggml_cann_set_device(device);
    const char* soc_name = aclrtGetSocName();
    snprintf(description, description_size, "%s", soc_name);
}

GGML_CALL void ggml_backend_cann_get_device_memory(int32_t device, size_t* free,
                                                   size_t* total) {
    ggml_cann_set_device(device);
    ACL_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, free, total));
}

// backend registry
/**
 * @brief Initializes a CANN backend based on the provided parameters.
 *
 * This function initializes a CANN backend using the device index and then
 * initializes the backend using `ggml_backend_cann_init`.
 *
 * @param params Parameters for initialization (unused in this implementation).
 * @param user_data User data containing the device index to initialize the
 * backend.
 * @return ggml_backend_t The initialized CANN backend.
 */
GGML_CALL static ggml_backend_t ggml_backend_reg_cann_init(const char* params,
                                                           void* user_data) {
    ggml_backend_t cann_backend =
        ggml_backend_cann_init((int)(intptr_t)user_data);
    return cann_backend;

    GGML_UNUSED(params);
}

extern "C" GGML_CALL int ggml_backend_cann_reg_devices();

/**
 * @brief Registers CANN (Ascend) devices as backend options.
 *
 * This function initializes ACL, retrieves the number of available CANN
 * devices, and registers each device as a backend option using
 * `ggml_backend_register`. Each device is given a unique name based on
 * `GGML_CANN_NAME` followed by its index.
 *
 * @return int The number of CANN devices registered.
 */
GGML_CALL int ggml_backend_cann_reg_devices() {
    uint32_t device_count = ggml_backend_cann_get_device_count();
    // initialization
    for (uint32_t i = 0; i < device_count; i++) {
        char name[128];
        snprintf(name, sizeof(name), "CANN%d", i);
        ggml_backend_register(name, ggml_backend_reg_cann_init,
                              ggml_backend_cann_buffer_type(i),
                              (void*)(intptr_t)i);
    }
    return device_count;
}
