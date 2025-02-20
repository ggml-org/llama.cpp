/*
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Qualcomm QNN SDK and reference tech guides could be found at:
 * https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
 * https://developer.qualcomm.com/software/hexagon-dsp-sdk/tools
 *
 * the implementation of ggml-qnn backend has six sections:
 * section-1 does forward/external declaration,
 * section-2 defines ggml-qnn internal log function
 * section-3 does general helper macro / data structure / function
 * section-4 does QNN helper macro / data structure / function
 * section-5 does ggml-qnn backend helper macro / data structure / function / class
 * section-6 does implementation of ggml-qnn backend according to ggml's backend subsystem
 *
 * currently provide following ggml ops' QNN backend implementation:
 * - GGML_OP_ADD:    this is a simple skeleton, can expand other ggml ops according to expertise
 * - GGML_OP_MUL:    this is a simple skeleton, can expand other ggml ops according to expertise
 * - GGML_OP_MUL_MAT:this is a complicated skeleton, can expand other complex ggml ops accordingly
 *
 * of course, can porting ggml-qnn to Windows on ARM as need.
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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <unistd.h>

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <map>
#include <set>
#include <tuple>
#include <queue>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <memory>
#include <regex>
#include <random>
#include <functional>
#include <unordered_map>
#include <condition_variable>
#include <cassert>
#include <unordered_set>
#include <utility>
#include <stdatomic.h>
#if (defined __ANDROID__) || (defined ANDROID)
#include "android/log.h"
#endif

#include "QnnTypes.h"
#include "QnnCommon.h"
#include "QnnContext.h"
#include "QnnBackend.h"
#include "QnnGraph.h"
#include "QnnProperty.h"
#include "QnnTensor.h"
#include "QnnInterface.h"
#include "Saver/QnnSaver.h"
#include "System/QnnSystemInterface.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpGraph.h"

#include "ggml-qnn.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
// =================================================================================================
//  section-1: forward/external declaration
// =================================================================================================
class qnn_instance;
struct ggml_backend_qnn_context;
static int free_qnn_tensor(Qnn_Tensor_t * tensor);
static enum ggml_status ggml_backend_qnn_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph);
static void ggmlqnn_log_internal(ggml_log_level level, const char * file, const char * func, int line, const char * format, ...);

// =================================================================================================
//  section-2: ggml-qnn internal troubleshooting function
// =================================================================================================
#define GGMLQNN_DEBUG                           1  // for troubleshooting QNN backend
#define GGML_QNN_LOGBUF_LEN                     4096
#define ENABLE_QNNBACKEND_PERF                  0  // enable/disable op's perf info
#define GGMLQNN_PRINT_QNN_INTERNAL_LOG          0  // enable/disable QNN's internal log
#define GGMLQNN_PRINT_OP_ADD_LOG                0  // GGML_OP_ADD already verified with QNN-CPU / QNN-GPU / QNN-NPU
#define GGMLQNN_PRINT_OP_MUL_MAT_LOG            1

#define GGMLQNN_LOG_ERROR(...) ggmlqnn_log_internal(GGML_LOG_LEVEL_DEBUG,  __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLQNN_LOG_WARN(...)  ggmlqnn_log_internal(GGML_LOG_LEVEL_DEBUG , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLQNN_LOG_INFO(...)  ggmlqnn_log_internal(GGML_LOG_LEVEL_DEBUG , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

#if GGMLQNN_DEBUG
#define GGMLQNN_LOG_DEBUG(...) ggmlqnn_log_internal(GGML_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#else
#define GGMLQNN_LOG_DEBUG(...)
#endif
static void ggmlqnn_log_internal(ggml_log_level level, const char * file, const char * func, int line, const char * format, ...) {
    static std::mutex ggmlqnn_log_internal_mutex;
    static char s_ggmlqnn_log_internal_buf[GGML_QNN_LOGBUF_LEN];

    {
        std::lock_guard<std::mutex> lock(ggmlqnn_log_internal_mutex);
        va_list args;
        va_start(args, format);
        int len_prefix = snprintf(s_ggmlqnn_log_internal_buf, GGML_QNN_LOGBUF_LEN, "[%s, %d]: ", func, line);
        int len = vsnprintf(s_ggmlqnn_log_internal_buf + len_prefix, GGML_QNN_LOGBUF_LEN - len_prefix, format, args);
        if (len < (GGML_QNN_LOGBUF_LEN - len_prefix)) {
#if (defined __ANDROID__) || (defined ANDROID)
            //for Android application(standard APP or command line tool)
            __android_log_print(ANDROID_LOG_INFO, "ggml-qnn", "%s\n", s_ggmlqnn_log_internal_buf);
#endif
#if (defined __ANDROID__) || (defined ANDROID)
            //do nothing when running on Snapdragon based Android device
#else
            //for Snapdragon based WoA(Windows on ARM) device
            printf("%s\n", s_ggmlqnn_log_internal_buf);
#endif
        }
        va_end(args);
    }
}

// =================================================================================================
//  section-3: general helper macro / data structure / function
// =================================================================================================
#define DISABLE_COPY(class_name)                \
    class_name(const class_name &) = delete;    \
    void operator=(const class_name &) = delete

#define DISABLE_MOVE(class_name)                \
    class_name(class_name &&) = delete;         \
    void operator=(class_name &&) = delete

#define GGMLQNN_MEM_ADD(alignment)              (sizeof (size_t) + alignment)
#define GGMLQNN_MEM_MASK(alignment)             ((uintptr_t)alignment - 1)

static intptr_t ggmlqnn_align_to(size_t alignment, intptr_t offset) {
    return offset % alignment == 0 ? offset
                                   : offset +
                                     (static_cast<intptr_t>(alignment) -
                                      offset % static_cast<intptr_t>(alignment));
}

static void * ggmlqnn_mallocz_aligned(size_t size, size_t alignment) {
    uint8_t * buffer = NULL;
    size_t * sp = NULL;
    buffer = static_cast<uint8_t *>(calloc(1, size + GGMLQNN_MEM_ADD(alignment)));
    if (!buffer)
        return NULL;
    sp = (size_t *)buffer;
    *sp = size;
    buffer = (uint8_t *)(((uintptr_t) buffer + GGMLQNN_MEM_ADD(alignment)) & ~GGMLQNN_MEM_MASK(alignment));
    buffer[-1] = buffer - (uint8_t *)sp;
    return buffer;
}

static void * ggmlqnn_malloc_aligned(size_t size, size_t alignment) {
    uint8_t * buffer = NULL;
    size_t * sp = NULL;
    buffer = static_cast<uint8_t *>(malloc(size + GGMLQNN_MEM_ADD(alignment)));
    if (!buffer)
        return NULL;
    sp = (size_t *)buffer;
    *sp = size;
    buffer = (uint8_t *)(((uintptr_t) buffer + GGMLQNN_MEM_ADD(alignment)) & ~GGMLQNN_MEM_MASK(alignment));
    buffer[-1] = buffer - (uint8_t *)sp;
    return buffer;
}

static void ggmqnn_free_aligned(void * ptr) {
    uint8_t * old = (uint8_t *)ptr;
    if (!old)
        return;
    old -= old[-1];
    free(old);
}

static size_t get_system_total_memory_in_bytes() {
    struct sysinfo info = {};
    if (sysinfo(&info) == 0) {
        return (info.totalram + info.totalswap) * info.mem_unit;
    }

    auto pages = (size_t)sysconf(_SC_PHYS_PAGES);
    auto page_size = (size_t)sysconf(_SC_PAGE_SIZE);

    return pages * page_size;
}

static size_t get_system_free_memory_in_bytes() {
    struct sysinfo info = {};
    if (sysinfo(&info) == 0) {
        return (info.freeram + info.freeswap) * info.mem_unit;
    }

    auto avail_pages = (size_t)sysconf(_SC_AVPHYS_PAGES);
    auto page_size = (size_t)sysconf(_SC_PAGE_SIZE);

    return avail_pages * page_size;
}

static size_t ggmlqnn_memscpy(void * dst, size_t dst_size, const void * src, size_t copy_size) {
    if (!dst || !src || !dst_size || !copy_size)
        return 0;

    size_t min_size = dst_size < copy_size ? dst_size : copy_size;

    memcpy(dst, src, min_size);

    return min_size;
}

static char * ggmlqnn_strndup(const char * source, size_t maxlen) {
    return ::strndup(source, maxlen);
}

static void * ggmlqnn_host_malloc(size_t n) {
    void * data = NULL;
    int result = posix_memalign((void **) &data, sysconf(_SC_PAGESIZE), n);
    if (result != 0) {
        GGMLQNN_LOG_WARN("%s: error: posix_memalign failed\n", __func__);
        return NULL;
    }

    return data;
}

// =================================================================================================
//  section-4: QNN helper macro / data structure / function
// =================================================================================================
#define VALIDATE(value, status)                                                 \
  do {                                                                          \
    status = value;                                                             \
    if (status != QNN_SUCCESS) {                                                \
      GGMLQNN_LOG_WARN("%s expected QNN_SUCCESS\n", #value);                    \
      return status;                                                            \
    }                                                                           \
  } while (0)

#define CHECK_QNN_API(error, result)                                            \
    do {                                                                        \
        error = (result);                                                       \
        if (QNN_SUCCESS != error) {                                             \
            if (error == QNN_COMMON_ERROR_NOT_SUPPORTED) {                      \
                GGMLQNN_LOG_WARN("WARNING: QNN feature/API not supported\n");   \
            } else {                                                            \
                GGMLQNN_LOG_INFO("QNN API error = %d(%s)\n", error, qnn_get_error_string(error));  \
            }                                                                   \
        }                                                                       \
    } while (0)

#define VALIDATE_TENSOR_VERSION(tensor, err)            VALIDATE(validate_tensor_version(tensor), err)

#define VALIDATE_OP_CONFIG_VERSION(op, err)             VALIDATE(validate_op_config_version(op), err)

#define QNN_VER_PTR(x)                                  (&((x).v1))
#define QNN_OP_CFG_VALID(op_config)                      ((op_config).version == QNN_OPCONFIG_VERSION_1)

#define QNN_OP_CFG_GET_NAME(op_config)                   get_qnn_oponfig_name(op_config)
#define QNN_OP_CFG_GET_PACKAGE_NAME(op_config)           get_qnn_op_config_packagename(op_config)
#define QNN_OP_CFG_GET_TYPE_NAME(op_config)              get_qnn_op_config_typename(op_config)
#define QNN_OP_CFG_GET_NUM_PARAMS(op_config)             get_qnn_op_config_numparams(op_config)
#define QNN_OP_CFG_GET_PARAMS(op_config)                 get_qnn_op_config_params(op_config)
#define QNN_OP_CFG_GET_NUM_INPUTS(op_config)             get_qnn_op_config_numinputs(op_config)
#define QNN_OP_CFG_GET_INPUTS(op_config)                 get_qnn_op_config_inputs(op_config)
#define QNN_OP_CFG_GET_NUM_OUTPUTS(op_config)            get_qnn_op_config_numoutputs(op_config)
#define QNN_OP_CFG_GET_OUTPUTS(op_config)                get_qnn_op_config_outputs(op_config)

#define QNN_OP_CFG_SET_NAME(op_config, value)            set_qnn_op_config_name(op_config, value)
#define QNN_OP_CFG_SET_PACKAGE_NAME(op_config, value)    set_qnn_op_config_packagename(op_config, value)
#define QNN_OP_CFG_SET_TYPE_NAME(op_config, value)       set_qnn_op_config_typename(op_config, value)

#define QNN_OP_CFG_SET_PARAMS(op_config, num_of_params, params) \
  set_qnn_op_config_params(op_config, num_of_params, params)

#define QNN_OP_CFG_SET_INPUTS(op_config, num_of_inputs, inputTensors) \
  set_qnn_op_config_inputs(op_config, num_of_inputs, inputTensors)

#define QNN_OP_CFG_SET_OUTPUTS(op_config, num_of_outputs, output_tensors) \
  set_qnn_op_config_outputs(op_config, num_of_outputs, output_tensors)

#define QNN_TENSOR_GET_ID(tensor)                       get_qnn_tensorid(tensor)
#define QNN_TENSOR_GET_NAME(tensor)                     get_qnn_tensorname(tensor)
#define QNN_TENSOR_GET_TYPE(tensor)                     get_qnn_tensortype(tensor)
#define QNN_TENSOR_GET_DATA_FORMAT(tensor)              get_qnn_tensor_dataformat(tensor)
#define QNN_TENSOR_GET_DATA_TYPE(tensor)                get_qnn_tensor_datatype(tensor)
#define QNN_TENSOR_GET_QUANT_PARAMS(tensor)             get_qnn_tensor_quantparams(tensor)
#define QNN_TENSOR_GET_RANK(tensor)                     get_qnn_tensor_rank(tensor)
#define QNN_TENSOR_GET_DIMENSIONS(tensor)               get_qnn_tensor_dimensions(tensor)
#define QNN_TENSOR_GET_MEM_TYPE(tensor)                 get_qnn_tensor_memtype(tensor)
#define QNN_TENSOR_GET_CLIENT_BUF(tensor)               get_qnn_tensor_clientbuf(tensor)
#define QNN_TENSOR_GET_MEM_HANDLE(tensor)               get_qnn_tensor_memhandle(tensor)

#define QNN_TENSOR_SET_ID(tensor, value)                set_qnn_tensor_id(tensor, value)
#define QNN_TENSOR_SET_NAME(tensor, value)              set_qnn_tensor_name(tensor, value)
#define QNN_TENSOR_SET_TYPE(tensor, value)              set_qnn_tensor_type(tensor, value)
#define QNN_TENSOR_SET_DATA_FORMAT(tensor, value)       set_qnn_tensor_dataformat(tensor, value)
#define QNN_TENSOR_SET_DATA_TYPE(tensor, value)         set_qnn_tensor_datatype(tensor, value)
#define QNN_TENSOR_SET_QUANT_PARAMS(tensor, value)      set_qnn_tensor_quantparams(tensor, value)
#define QNN_TENSOR_SET_RANK(tensor, value)              set_qnn_tensor_rank(tensor, value)
#define QNN_TENSOR_SET_DIMENSIONS(tensor, value)        set_qnn_tensor_dimensions(tensor, value)
#define QNN_TENSOR_SET_MEM_TYPE(tensor, value)          set_qnn_tensor_memtype(tensor, value)
#define QNN_TENSOR_SET_CLIENT_BUF(tensor, value)        set_qnn_tensor_clientbuf(tensor, value)
#define QNN_TENSOR_SET_MEM_HANDLE(tensor, value)        set_qnn_tensor_memhandle(tensor, value)

static inline int validate_tensor_version(Qnn_Tensor_t tensor) {
    if (tensor.version != QNN_TENSOR_VERSION_1) {
        GGMLQNN_LOG_WARN("validate_tensor_version() tensor %s, got unsupported version %d\n",
              tensor.v1.name,
              tensor.version);
        return 1;
    }
    return 0;
}

[[maybe_unused]] static inline int validate_op_config_version(Qnn_OpConfig_t op_config) {
    if (op_config.version != QNN_OPCONFIG_VERSION_1) {
        GGMLQNN_LOG_WARN("validate_op_config_version() op %s, got unsupported version %d\n",
              op_config.v1.name,
              op_config.version);
        return 1;
    }
    return 0;
}

static inline const char * get_qnn_oponfig_name(const Qnn_OpConfig_t & op_config) {
    if (op_config.version == QNN_OPCONFIG_VERSION_1) {
        return op_config.v1.name;
    }
    return nullptr;
}

[[maybe_unused]] static inline const char * get_qnn_oponfig_name(const Qnn_OpConfig_t * op_config) {
    return get_qnn_oponfig_name(*op_config);
}

static inline const char * get_qnn_op_config_packagename(const Qnn_OpConfig_t & op_config) {
    if (op_config.version == QNN_OPCONFIG_VERSION_1) {
        return op_config.v1.packageName;
    }
    return nullptr;
}

[[maybe_unused]] static inline const char * get_qnn_op_config_packagename(const Qnn_OpConfig_t * op_config) {
    return get_qnn_op_config_packagename(*op_config);
}

static inline const char * get_qnn_op_config_typename(const Qnn_OpConfig_t & op_config) {
    if (op_config.version == QNN_OPCONFIG_VERSION_1) {
        return op_config.v1.typeName;
    }
    return nullptr;
}

[[maybe_unused]] static inline const char * get_qnn_op_config_typename(const Qnn_OpConfig_t * op_config) {
    return get_qnn_op_config_typename(*op_config);
}

static inline uint32_t get_qnn_op_config_numparams(const Qnn_OpConfig_t & op_config) {
    if (op_config.version == QNN_OPCONFIG_VERSION_1) {
        return op_config.v1.numOfParams;
    }
    return 0u;
}

[[maybe_unused]] static inline uint32_t get_qnn_op_config_numparams(const Qnn_OpConfig_t * op_config) {
    return get_qnn_op_config_numparams(*op_config);
}

static inline const Qnn_Param_t * get_qnn_op_config_params(const Qnn_OpConfig_t & op_config) {
    if (op_config.version == QNN_OPCONFIG_VERSION_1) {
        return op_config.v1.params;
    }
    return nullptr;
}

[[maybe_unused]] static inline const Qnn_Param_t * get_qnn_op_config_params(const Qnn_OpConfig_t * op_config) {
    return get_qnn_op_config_params(*op_config);
}

static inline uint32_t get_qnn_op_config_numinputs(const Qnn_OpConfig_t & op_config) {
    if (op_config.version == QNN_OPCONFIG_VERSION_1) {
        return op_config.v1.numOfInputs;
    }
    return 0u;
}

[[maybe_unused]] static inline uint32_t get_qnn_op_config_numinputs(const Qnn_OpConfig_t * op_config) {
    return get_qnn_op_config_numinputs(*op_config);
}

static inline const Qnn_Tensor_t * get_qnn_op_config_inputs(const Qnn_OpConfig_t & op_config) {
    if (op_config.version == QNN_OPCONFIG_VERSION_1) {
        return op_config.v1.inputTensors;
    }
    return nullptr;
}

[[maybe_unused]] static inline const Qnn_Tensor_t * get_qnn_op_config_inputs(const Qnn_OpConfig_t * op_config) {
    return get_qnn_op_config_inputs(*op_config);
}

static inline uint32_t get_qnn_op_config_numoutputs(const Qnn_OpConfig_t & op_config) {
    if (op_config.version == QNN_OPCONFIG_VERSION_1) {
        return op_config.v1.numOfOutputs;
    }
    return 0u;
}

[[maybe_unused]] static inline uint32_t get_qnn_op_config_numoutputs(const Qnn_OpConfig_t * op_config) {
    return get_qnn_op_config_numoutputs(*op_config);
}

static inline const Qnn_Tensor_t * get_qnn_op_config_outputs(const Qnn_OpConfig_t & op_config) {
    if (op_config.version == QNN_OPCONFIG_VERSION_1) {
        return op_config.v1.outputTensors;
    }
    return nullptr;
}

[[maybe_unused]] static inline const Qnn_Tensor_t * get_qnn_op_config_outputs(const Qnn_OpConfig_t * op_config) {
    return get_qnn_op_config_outputs(*op_config);
}

static inline void set_qnn_op_config_name(Qnn_OpConfig_t & op_config, const char * name) {
    if (op_config.version == QNN_OPCONFIG_VERSION_1) {
        op_config.v1.name = name;
    }
}

[[maybe_unused]] static inline void set_qnn_op_config_name(Qnn_OpConfig_t * op_config, const char * name) {
    set_qnn_op_config_name(*op_config, name);
}

static inline void set_qnn_op_config_packagename(Qnn_OpConfig_t & op_config, const char * package_name) {
    if (op_config.version == QNN_OPCONFIG_VERSION_1) {
        op_config.v1.packageName = package_name;
    }
}

[[maybe_unused]] static inline void set_qnn_op_config_packagename(Qnn_OpConfig_t * op_config, const char * package_name) {
    set_qnn_op_config_packagename(*op_config, package_name);
}

static inline void set_qnn_op_config_typename(Qnn_OpConfig_t & op_config, const char * type_name) {
    if (op_config.version == QNN_OPCONFIG_VERSION_1) {
        op_config.v1.typeName = type_name;
    }
}

[[maybe_unused]] static inline void set_qnn_op_config_typename(Qnn_OpConfig_t * op_config, const char * type_name) {
    set_qnn_op_config_typename(*op_config, type_name);
}

static inline void set_qnn_op_config_params(Qnn_OpConfig_t & op_config,
                                 uint32_t num_of_params,
                                 Qnn_Param_t * params) {
    if (op_config.version == QNN_OPCONFIG_VERSION_1) {
        op_config.v1.numOfParams = num_of_params;
        op_config.v1.params      = params;
    }
}

[[maybe_unused]] static inline void set_qnn_op_config_params(Qnn_OpConfig_t * op_config,
                                 uint32_t num_of_params,
                                 Qnn_Param_t * params) {
    set_qnn_op_config_params(*op_config, num_of_params, params);
}

static inline void set_qnn_op_config_inputs(Qnn_OpConfig_t & op_config,
                                 uint32_t num_of_inputs,
                                 Qnn_Tensor_t * input_tensors) {
    if (op_config.version == QNN_OPCONFIG_VERSION_1) {
        op_config.v1.numOfInputs  = num_of_inputs;
        op_config.v1.inputTensors = input_tensors;
    }
}

[[maybe_unused]] static inline void set_qnn_op_config_inputs(Qnn_OpConfig_t * op_config,
                                 uint32_t num_of_inputs,
                                 Qnn_Tensor_t * input_tensors) {
    set_qnn_op_config_inputs(*op_config, num_of_inputs, input_tensors);
}

static inline void set_qnn_op_config_outputs(Qnn_OpConfig_t & op_config,
                                  uint32_t num_of_outputs,
                                  Qnn_Tensor_t * output_tensors) {
    if (op_config.version == QNN_OPCONFIG_VERSION_1) {
        op_config.v1.numOfOutputs  = num_of_outputs;
        op_config.v1.outputTensors = output_tensors;
    }
}

[[maybe_unused]] static inline void set_qnn_op_config_outputs(Qnn_OpConfig_t * op_config,
                                  uint32_t num_of_outputs,
                                  Qnn_Tensor_t * output_tensors) {
    set_qnn_op_config_outputs(*op_config, num_of_outputs, output_tensors);
}

static inline uint32_t get_qnn_tensorid(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.id;
    }

    return 0u;
}

[[maybe_unused]] static inline uint32_t get_qnn_tensorid(const Qnn_Tensor_t * tensor) {
    return get_qnn_tensorid(*tensor);
}

static inline const char * get_qnn_tensorname(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.name;
    }
    return nullptr;
}

static inline const char * get_qnn_tensorname(const Qnn_Tensor_t * tensor) {
    return get_qnn_tensorname(*tensor);
}

static inline Qnn_TensorType_t get_qnn_tensortype(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.type;
    }
    return QNN_TENSOR_TYPE_UNDEFINED;
}

[[maybe_unused]] static inline Qnn_TensorType_t get_qnn_tensortype(const Qnn_Tensor_t * tensor) {
    return get_qnn_tensortype(*tensor);
}

static inline Qnn_TensorDataFormat_t get_qnn_tensor_dataformat(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.dataFormat;
    }
    return QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
}

[[maybe_unused]] static inline Qnn_TensorDataFormat_t get_qnn_tensor_dataformat(const Qnn_Tensor_t * tensor) {
    return get_qnn_tensor_dataformat(*tensor);
}

static inline Qnn_DataType_t get_qnn_tensor_datatype(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.dataType;
    }
    return QNN_DATATYPE_UNDEFINED;
}

[[maybe_unused]] static inline Qnn_DataType_t get_qnn_tensor_datatype(const Qnn_Tensor_t * tensor) {
    return get_qnn_tensor_datatype(*tensor);
}

static inline Qnn_QuantizeParams_t get_qnn_tensor_quantparams(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.quantizeParams;
    }
    return QNN_QUANTIZE_PARAMS_INIT;
}

[[maybe_unused]] static inline Qnn_QuantizeParams_t get_qnn_tensor_quantparams(const Qnn_Tensor_t * tensor) {
    return get_qnn_tensor_quantparams(*tensor);
}

static inline uint32_t get_qnn_tensor_rank(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.rank;
    }
    return 0u;
}

[[maybe_unused]] static inline uint32_t get_qnn_tensor_rank(const Qnn_Tensor_t * tensor) {
    return get_qnn_tensor_rank(*tensor);
}

static inline uint32_t * get_qnn_tensor_dimensions(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.dimensions;
    }
    return nullptr;
}

[[maybe_unused]] static inline uint32_t * get_qnn_tensor_dimensions(const Qnn_Tensor_t * tensor) {
    return get_qnn_tensor_dimensions(*tensor);
}

static inline Qnn_TensorMemType_t get_qnn_tensor_memtype(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.memType;
    }
    return QNN_TENSORMEMTYPE_UNDEFINED;
}

[[maybe_unused]] static inline Qnn_TensorMemType_t get_qnn_tensor_memtype(const Qnn_Tensor_t * tensor) {
    return get_qnn_tensor_memtype(*tensor);
}

static inline Qnn_ClientBuffer_t get_qnn_tensor_clientbuf(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.clientBuf;
    }
    return QNN_CLIENT_BUFFER_INIT;
}

[[maybe_unused]] static inline Qnn_ClientBuffer_t get_qnn_tensor_clientbuf(const Qnn_Tensor_t * tensor) {
    return get_qnn_tensor_clientbuf(*tensor);
}

static inline Qnn_MemHandle_t get_qnn_tensor_memhandle(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.memHandle;
    }
    return nullptr;
}

[[maybe_unused]] static inline Qnn_MemHandle_t get_qnn_tensor_memhandle(const Qnn_Tensor_t * tensor) {
    return get_qnn_tensor_memhandle(*tensor);
}

static inline void set_qnn_tensor_id(Qnn_Tensor_t & tensor, uint32_t id) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.id = id;
    }
}

[[maybe_unused]] static inline void set_qnn_tensor_id(Qnn_Tensor_t * tensor, uint32_t id) {
    set_qnn_tensor_id(*tensor, id);
}

static inline void set_qnn_tensor_name(Qnn_Tensor_t & tensor, const char * name) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.name = name;
    }
}

[[maybe_unused]] static inline void set_qnn_tensor_name(Qnn_Tensor_t * tensor, const char * name) {
    set_qnn_tensor_name(*tensor, name);
}

static inline void set_qnn_tensor_type(Qnn_Tensor_t & tensor, Qnn_TensorType_t type) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.type = type;
    }
}

[[maybe_unused]] static inline void set_qnn_tensor_type(Qnn_Tensor_t * tensor, Qnn_TensorType_t type) {
    set_qnn_tensor_type(*tensor, type);
}

static inline void set_qnn_tensor_dataformat(Qnn_Tensor_t & tensor, Qnn_TensorDataFormat_t format) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.dataFormat = format;
    }
}

[[maybe_unused]] static inline void set_qnn_tensor_dataformat(Qnn_Tensor_t * tensor, Qnn_TensorDataFormat_t format) {
    set_qnn_tensor_dataformat(*tensor, format);
}

static inline void set_qnn_tensor_datatype(Qnn_Tensor_t & tensor, Qnn_DataType_t dataType) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.dataType = dataType;
    }
}

[[maybe_unused]] static inline void set_qnn_tensor_datatype(Qnn_Tensor_t * tensor, Qnn_DataType_t dataType) {
    set_qnn_tensor_datatype(*tensor, dataType);
}

static inline void set_qnn_tensor_quantparams(Qnn_Tensor_t & tensor, Qnn_QuantizeParams_t params) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.quantizeParams = params;
    }
}

[[maybe_unused]] static inline void set_qnn_tensor_quantparams(Qnn_Tensor_t * tensor, Qnn_QuantizeParams_t params) {
    set_qnn_tensor_quantparams(*tensor, params);
}

static inline void set_qnn_tensor_rank(Qnn_Tensor_t & tensor, uint32_t rank) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.rank = rank;
    }
}

[[maybe_unused]] static inline void set_qnn_tensor_rank(Qnn_Tensor_t * tensor, uint32_t rank) {
    set_qnn_tensor_rank(*tensor, rank);
}

static inline void set_qnn_tensor_dimensions(Qnn_Tensor_t & tensor, uint32_t * dims) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.dimensions = dims;
    }
}

[[maybe_unused]] static inline void set_qnn_tensor_dimensions(Qnn_Tensor_t * tensor, uint32_t * dims) {
    set_qnn_tensor_dimensions(*tensor, dims);
}

static inline void set_qnn_tensor_memtype(Qnn_Tensor_t & tensor, Qnn_TensorMemType_t memType) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.memType = memType;
    }
}

[[maybe_unused]] static inline void set_qnn_tensor_memtype(Qnn_Tensor_t * tensor, Qnn_TensorMemType_t memType) {
    set_qnn_tensor_memtype(*tensor, memType);
}

static inline void set_qnn_tensor_clientbuf(Qnn_Tensor_t & tensor, Qnn_ClientBuffer_t clientBuf) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.clientBuf = clientBuf;
    }
}

[[maybe_unused]] static inline void set_qnn_tensor_clientbuf(Qnn_Tensor_t * tensor, Qnn_ClientBuffer_t clientBuf) {
    set_qnn_tensor_clientbuf(*tensor, clientBuf);
}

static inline void set_qnn_tensor_memhandle(Qnn_Tensor_t & tensor, Qnn_MemHandle_t handle) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.memHandle = handle;
    }
}

[[maybe_unused]] static inline void set_qnn_tensor_memhandle(Qnn_Tensor_t * tensor, Qnn_MemHandle_t handle) {
    set_qnn_tensor_memhandle(*tensor, handle);
}

inline static Qnn_Tensor_t qnn_tensor_init(Qnn_TensorVersion_t version) {
    Qnn_Tensor_t tensor;
    tensor.version = version;
    if (version == QNN_TENSOR_VERSION_1) {
        tensor.v1 = QNN_TENSOR_V1_INIT;
    } else if (version == QNN_TENSOR_VERSION_2) {
        tensor.v2 = QNN_TENSOR_V2_INIT;
    }
    return tensor;
}

static int deep_copy_qnn_tensors(Qnn_Tensor_t & src, Qnn_Tensor_t & dst) {
    int err = 0;
    VALIDATE_TENSOR_VERSION(src, err);

    dst.version = src.version;
    QNN_TENSOR_SET_NAME(
            dst, ggmlqnn_strndup(QNN_TENSOR_GET_NAME(src), std::string(QNN_TENSOR_GET_NAME(src)).size()));
    if (QNN_TENSOR_GET_NAME(dst) == nullptr) {
        return 1;
    }
    QNN_TENSOR_SET_ID(dst, QNN_TENSOR_GET_ID(src));
    QNN_TENSOR_SET_TYPE(dst, QNN_TENSOR_GET_TYPE(src));
    QNN_TENSOR_SET_DATA_FORMAT(dst, QNN_TENSOR_GET_DATA_FORMAT(src));
    QNN_TENSOR_SET_DATA_TYPE(dst, QNN_TENSOR_GET_DATA_TYPE(src));
    QNN_TENSOR_SET_MEM_TYPE(dst, QNN_TENSOR_GET_MEM_TYPE(src));

    if (QNN_TENSOR_GET_MEM_TYPE(src) == QNN_TENSORMEMTYPE_RAW) {
        Qnn_ClientBuffer_t client_buf = {nullptr, 0};
        QNN_TENSOR_SET_CLIENT_BUF(dst, client_buf);
    } else if (QNN_TENSOR_GET_MEM_TYPE(src) == QNN_TENSORMEMTYPE_MEMHANDLE) {
        QNN_TENSOR_SET_MEM_HANDLE(dst, nullptr);
    } else {
        return 1;
    }

    Qnn_QuantizeParams_t src_qparam      = QNN_TENSOR_GET_QUANT_PARAMS(src);
    Qnn_QuantizationEncoding_t encoding = src_qparam.quantizationEncoding;
    if (encoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
        Qnn_QuantizeParams_t src_qparam_cpy      = src_qparam;
        Qnn_AxisScaleOffset_t & axis_scale_offset = src_qparam_cpy.axisScaleOffsetEncoding;
        Qnn_ScaleOffset_t ** scale_offset          = &axis_scale_offset.scaleOffset;
        size_t scale_offset_size = axis_scale_offset.numScaleOffsets * sizeof(Qnn_ScaleOffset_t);
        *scale_offset           = (Qnn_ScaleOffset_t *)malloc(scale_offset_size);
        ggmlqnn_memscpy(*scale_offset,
                        scale_offset_size,
                        src_qparam.axisScaleOffsetEncoding.scaleOffset,
                        scale_offset_size);
        QNN_TENSOR_SET_QUANT_PARAMS(dst, src_qparam_cpy);
    } else if (encoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
        Qnn_QuantizeParams_t src_qparam_cpy          = src_qparam;
        Qnn_BwAxisScaleOffset_t & bwaxis_scale_offset = src_qparam_cpy.bwAxisScaleOffsetEncoding;
        size_t scale_size                          = bwaxis_scale_offset.numElements * sizeof(float);
        float ** scales                            = &bwaxis_scale_offset.scales;
        int32_t ** offsets                         = &bwaxis_scale_offset.offsets;
        *scales                                    = (float *)malloc(scale_size);
        ggmlqnn_memscpy(*scales, scale_size, src_qparam.bwAxisScaleOffsetEncoding.scales, scale_size);

        if (bwaxis_scale_offset.offsets != nullptr) {
            size_t offset_size = bwaxis_scale_offset.numElements * sizeof(int32_t);
            *offsets           = (int32_t *)malloc(offset_size);
            ggmlqnn_memscpy(*offsets, offset_size, src_qparam.bwAxisScaleOffsetEncoding.offsets, offset_size);
        }
        QNN_TENSOR_SET_QUANT_PARAMS(dst, src_qparam_cpy);
    } else {
        QNN_TENSOR_SET_QUANT_PARAMS(dst, src_qparam);
    }

    uint32_t rank = QNN_TENSOR_GET_RANK(src);
    QNN_TENSOR_SET_RANK(dst, rank);
    size_t dim_size       = GGML_MAX_DIMS * sizeof(uint32_t);
    uint32_t * dimensions = (uint32_t *)malloc(dim_size);
    if (dimensions == nullptr) {
        GGMLQNN_LOG_WARN("deep_copy_qnn_tensors() allocation error while copying tensor %s\n", QNN_TENSOR_GET_NAME(src));
        return 1;
    }
    ggmlqnn_memscpy(dimensions, dim_size, QNN_TENSOR_GET_DIMENSIONS(src), dim_size);
    QNN_TENSOR_SET_DIMENSIONS(dst, dimensions);

    return err;
}

static int free_qnn_tensor(Qnn_Tensor_t * tensor) {
    int err = 0;
    VALIDATE_TENSOR_VERSION(*tensor, err);
    free((void *) QNN_TENSOR_GET_NAME(*tensor));

    Qnn_QuantizeParams_t src_qparam      = QNN_TENSOR_GET_QUANT_PARAMS(*tensor);
    Qnn_QuantizationEncoding_t encoding = src_qparam.quantizationEncoding;
    if (encoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
        free(src_qparam.axisScaleOffsetEncoding.scaleOffset);
    } else if (encoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
        free(src_qparam.bwAxisScaleOffsetEncoding.scales);
        if (src_qparam.bwAxisScaleOffsetEncoding.offsets != nullptr) {
            free(src_qparam.bwAxisScaleOffsetEncoding.offsets);
        }
    }
    free(QNN_TENSOR_GET_DIMENSIONS(*tensor));
    free(tensor);

    return err;
}


static size_t qnn_datatype_size(Qnn_DataType_t qnn_type) {
    switch (qnn_type) {
        case QNN_DATATYPE_FLOAT_32:
            return sizeof(float);
        case QNN_DATATYPE_FLOAT_16:
            return sizeof(uint16_t);
        case QNN_DATATYPE_UINT_32:
        case QNN_DATATYPE_INT_32:
            return sizeof(int32_t);
        case QNN_DATATYPE_INT_16:
            return sizeof(int16_t);
        case QNN_DATATYPE_INT_8:
            return sizeof(int8_t);
        case QNN_DATATYPE_SFIXED_POINT_8:
            return sizeof(int8_t);
        case QNN_DATATYPE_SFIXED_POINT_4:
            return sizeof(int8_t);
        default:
            break;
    }
    return 0;
}

static const char * qnn_datatype_to_string(Qnn_DataType_t qnn_type) {
    switch (qnn_type) {
        case QNN_DATATYPE_FLOAT_32:
            return "QNN_DATATYPE_FLOAT_32";
        case QNN_DATATYPE_FLOAT_16:
            return "QNN_DATATYPE_FLOAT_16";
        case QNN_DATATYPE_UINT_32:
            return "QNN_DATATYPE_UINT_32";
        case QNN_DATATYPE_INT_32:
            return "QNN_DATATYPE_INT_32";
        case QNN_DATATYPE_INT_16:
            return "QNN_DATATYPE_INT_16";
        case QNN_DATATYPE_INT_8:
            return "QNN_DATATYPE_INT_8";
        case QNN_DATATYPE_SFIXED_POINT_8:
            return "QNN_DATATYPE_SFIXED_POINT_8";
        case QNN_DATATYPE_SFIXED_POINT_4:
            return "QNN_DATATYPE_SFIXED_POINT_4";
        default:
            break;
    }
    return "QNN_DATATYPE_UNDEFINED";
}

static const char * qnn_get_error_string(Qnn_ErrorHandle_t qnn_error_code) {
    // file:///opt/qcom/aistack/qairt/2.31.0.250130/docs/QNN/general/api_error_codes.html
    switch (qnn_error_code) {
        case QNN_SUCCESS:
            return "QNN_SUCCESS";
        case QNN_COMMON_ERROR_GENERAL:
            return "QNN_COMMON_ERROR_GENERAL";

            // QnnGraph_Error_t
        case QNN_GRAPH_ERROR_UNSUPPORTED_FEATURE:
            return "QNN_GRAPH_ERROR_UNSUPPORTED_FEATURE";
        case QNN_GRAPH_ERROR_MEM_ALLOC:
            return "QNN_GRAPH_ERROR_MEM_ALLOC";
        case QNN_GRAPH_ERROR_INVALID_ARGUMENT:
            return "QNN_GRAPH_ERROR_INVALID_ARGUMENT";
        case QNN_GRAPH_ERROR_INVALID_HANDLE:
            return "QNN_GRAPH_ERROR_INVALID_HANDLE";
        case QNN_GRAPH_ERROR_GRAPH_DOES_NOT_EXIST:
            return "QNN_GRAPH_ERROR_GRAPH_DOES_NOT_EXIST";
        case QNN_GRAPH_ERROR_INVALID_NAME:
            return "QNN_GRAPH_ERROR_INVALID_NAME";
        case QNN_GRAPH_ERROR_INVALID_TENSOR:
            return "QNN_GRAPH_ERROR_INVALID_TENSOR";
        case QNN_GRAPH_ERROR_INVALID_OP_CONFIG:
            return "QNN_GRAPH_ERROR_INVALID_OP_CONFIG";
        case QNN_GRAPH_ERROR_SET_PROFILE:
            return "QNN_GRAPH_ERROR_SET_PROFILE";
        case QNN_GRAPH_ERROR_UNCONNECTED_NODE:
            return "QNN_GRAPH_ERROR_UNCONNECTED_NODE";
        case QNN_GRAPH_ERROR_CREATE_FAILED:
            return "QNN_GRAPH_ERROR_CREATE_FAILED";
        case QNN_GRAPH_ERROR_OPTIMIZATION_FAILED:
            return "QNN_GRAPH_ERROR_OPTIMIZATION_FAILED";
        case QNN_GRAPH_ERROR_FINALIZE_FAILED:
            return "QNN_GRAPH_ERROR_FINALIZE_FAILED";
        case QNN_GRAPH_ERROR_GRAPH_NOT_FINALIZED:
            return "QNN_GRAPH_ERROR_GRAPH_NOT_FINALIZED";
        case QNN_GRAPH_ERROR_GRAPH_FINALIZED:
            return "QNN_GRAPH_ERROR_GRAPH_FINALIZED";
        case QNN_GRAPH_ERROR_EXECUTION_ASYNC_FIFO_FULL:
            return "QNN_GRAPH_ERROR_EXECUTION_ASYNC_FIFO_FULL";
        case QNN_GRAPH_ERROR_SIGNAL_IN_USE:
            return "QNN_GRAPH_ERROR_SIGNAL_IN_USE";
        case QNN_GRAPH_ERROR_ABORTED:
            return "QNN_GRAPH_ERROR_ABORTED";
        case QNN_GRAPH_ERROR_PROFILE_IN_USE:
            return "QNN_GRAPH_ERROR_PROFILE_IN_USE";
        case QNN_GRAPH_ERROR_TIMED_OUT:
            return "QNN_GRAPH_ERROR_TIMED_OUT";
        case QNN_GRAPH_ERROR_SUBGRAPH:
            return "QNN_GRAPH_ERROR_SUBGRAPH";
        case QNN_GRAPH_ERROR_DISABLED:
            return "QNN_GRAPH_ERROR_DISABLED";
        case QNN_GRAPH_ERROR_DYNAMIC_TENSOR_SHAPE:
            return "QNN_GRAPH_ERROR_DYNAMIC_TENSOR_SHAPE";
        case QNN_GRAPH_ERROR_TENSOR_SPARSITY:
            return "QNN_GRAPH_ERROR_TENSOR_SPARSITY";
        case QNN_GRAPH_ERROR_EARLY_TERMINATION:
            return "QNN_GRAPH_ERROR_EARLY_TERMINATION";
        case QNN_GRAPH_ERROR_INVALID_CONTEXT:
            return "QNN_GRAPH_ERROR_INVALID_CONTEXT";

            //QQnnTensor_Error_t
            //Invalid context/graph handle in creating tensor
        case QNN_TENSOR_ERROR_INVALID_HANDLE:
            return "QNN_TENSOR_ERROR_INVALID_HANDLE";
            //Tensor with specified credentials not registered with a context/graph
        case QNN_TENSOR_ERROR_DOES_NOT_EXIST:
            return "QNN_TENSOR_ERROR_DOES_NOT_EXIST";
            // (deprecated) Tensor has already been registered with backend
        case QNN_TENSOR_ERROR_ALREADY_EXISTS:
            return "QNN_TENSOR_ERROR_ALREADY_EXISTS";
            // Invalid tensor param.
        case QNN_TENSOR_ERROR_INVALID_TENSOR_PARAM:
            return "QNN_TENSOR_ERROR_INVALID_TENSOR_PARAM";
            // This tensor param is currently unsupported
        case QNN_TENSOR_ERROR_UNSUPPORTED_TENSOR_PARAM:
            return "QNN_TENSOR_ERROR_UNSUPPORTED_TENSOR_PARAM";
            // Tensor provided for update is invalid
        case QNN_TENSOR_ERROR_INCOMPATIBLE_TENSOR_UPDATE:
            return "QNN_TENSOR_ERROR_INCOMPATIBLE_TENSOR_UPDATE";

            // QnnOpPackage_Error_t
        case QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED:
            return "QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED";
        case QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED:
            return "QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED";
        case QNN_OP_PACKAGE_ERROR_INVALID_HANDLE:
            return "QNN_OP_PACKAGE_ERROR_INVALID_HANDLE";
        case QNN_OP_PACKAGE_ERROR_INVALID_INFRASTRUCTURE:
            return "QNN_OP_PACKAGE_ERROR_INVALID_INFRASTRUCTURE";
        case QNN_OP_PACKAGE_ERROR_INVALID_INFO:
            return "QNN_OP_PACKAGE_ERROR_INVALID_INFO";
        case QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE:
            return "QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE";
        case QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT:
            return "QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT";

        default:
            return "unknown QNN error";
    }
}

// =================================================================================================
//  section-5:ggml-qnn backend helper macro / data structure / function / class
// =================================================================================================
#define RPCMEM_DEFAULT_FLAGS                    1
#define RPCMEM_HEAP_ID_SYSTEM                   25

typedef void (* ggmlqnn_op_func_t)(ggml_backend_t backend, ggml_tensor * op);

using pfn_rpc_mem_init                                  = void (*)(void);
using pfn_rpc_mem_deinit                                = void (*)(void);
using pfn_rpc_mem_alloc                                 = void *(*)(int, uint32_t, int);
using pfn_rpc_mem_free                                  = void (*)(void *);
using pfn_rpc_mem_to_fd                                 = int (*)(void *);
using _pfn_QnnSaver_initialize                          = decltype(QnnSaver_initialize);
using _pfn_QnnInterface_getProviders                    = decltype(QnnInterface_getProviders);
using _pfn_QnnSystemInterface_getProviders              = decltype(QnnSystemInterface_getProviders);

using qnn_res_t                                         = std::tuple<Qnn_GraphHandle_t, std::vector< Qnn_Tensor_t *>>;
using qnn_tensors_t                                     = std::vector< Qnn_Tensor_t *>;

enum class ggml_qnn_profile_level {
    profile_off     = 0,
    profile_basic   = 1,
    profile_detail  = 2
};

enum qcom_htp_arch {
    NONE = 0,
    V68 = 68,
    V69 = 69,
    V73 = 73,
    V75 = 75,
    V79 = 79,
};

enum qcom_chipset_soc_model {
    UNKNOWN_SM = 0,
    SM7450 = 41,  // v69, 7 Gen1
    SM8350 = 30,  // v68, 888
    SM8450 = 36,  // v69, SD 8 Gen 1
    SM8475 = 42,  // v69, SD 8+ Gen 1
    SM8550 = 43,  // v73, SD 8 Gen 2
    SM8650 = 57,  // v75, SD 8 Gen 3
    SM8750 = 69,  // v79, SD 8 Gen 4
};

struct qcom_socinfo {
    uint32_t soc_model;
    size_t htp_arch;
    size_t vtcm_size_in_mb;
    char soc_desc[GGML_MAX_NAME];
};

//file:///opt/qcom/aistack/qairt/2.31.0.250130/docs/QNN/general/overview.html#tbl-supported-snapdragon-devices
static struct qcom_socinfo g_qnn_soc_info_table[] = {
        /* Qualcomm SnapDragon 7 Gen 1 */
        [SM7450] = {
                .soc_model         = SM7450,
                .htp_arch          = V69,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 7 Gen 1"},

        /* Qualcomm SnapDragon 888 */
        [SM8350] = {
                .soc_model         = SM8350,
                .htp_arch          = V68,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 888 "},

        /* Qualcomm SnapDragon 8 Gen 1 */
        [SM8450] = {
                .soc_model         = SM8450,
                .htp_arch          = V69,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 8 Gen 1"},

        /* Qualcomm SnapDragon 8 Gen 1+ */
        [SM8475] = {
                .soc_model         = SM8475,
                .htp_arch          = V69,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 8 Gen 1+"},

        /* Qualcomm SnapDragon 8 Gen 2 */
        [SM8550] = {
                .soc_model         = SM8550,
                .htp_arch          = V73,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 8 Gen 2"},

        /* Qualcomm SnapDragon 8 Gen 3 */
        [SM8650] = {
                .soc_model         = SM8650,
                .htp_arch          = V75,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 8 Gen 3 "},

        /* Qualcomm SnapDragon 8 Gen 4 */
        [SM8750] = {
                .soc_model         = SM8750,
                .htp_arch          = V79,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 8 Gen 4"},

};

struct ggml_backend_qnn_context {
    int device;
    int threads;
    char name[GGML_MAX_NAME];
    char desc[GGML_MAX_NAME];
    char lib[GGML_MAX_NAME];
    qnn_instance * instance;
    struct ggml_backend * backend;
    QNN_INTERFACE_VER_TYPE raw_interface;
    QNN_SYSTEM_INTERFACE_VER_TYPE raw_system_interface;
    struct qcom_socinfo           socinfo;
} ;

//TODO: the following global vars and three helper funcs should be removed in the future
static int32_t  g_ggmltensor_idx    = 0;
static void reset_idx() {
    g_ggmltensor_idx = 0;
}

static void inc_idx() {
    g_ggmltensor_idx++;
}

static int32_t get_idx() {
    return g_ggmltensor_idx;
}

// file:///opt/qcom/aistack/qairt/2.31.0.250130/docs/QNN/general/quantization.html
// CPU - Choose a non-quantized model.Quantized models are currently incompatible with the CPU backend
// GPU - Choose a non-quantized model.Quantized models are currently incompatible with the GPU backend
// HTP - Choose a quantized model. Quantized models are required when running on the HTP backend
// DSP - Choose a quantized model. Quantized models are required when running on the DSP backend
// HTA - Choose a quantized model. Quantized models are required when running on the HTA backend
static struct ggml_backend_qnn_context g_qnn_mgr[GGML_QNN_MAX_DEVICES] = {
        [QNN_BACKEND_CPU] = {.device               = 0,
                .threads              = 1,
                .name                 = "qnn-cpu",
                .desc                 = "Qualcomm Kryo CPU",
                .lib                  = "libQnnCpu.so",
                .instance             = nullptr,
                .backend              = nullptr,
                .raw_interface        = {},
                .raw_system_interface = {},
                .socinfo              = {}},

        [QNN_BACKEND_GPU] = {.device               = 1,
                .threads              = 1,
                .name                 = "qnn-gpu",
                .desc                 = "Qualcomm Adreno GPU",
                .lib                  = "libQnnGpu.so",
                .instance             = nullptr,
                .backend              = nullptr,
                .raw_interface        = {},
                .raw_system_interface = {},
                .socinfo              = {}},

        [QNN_BACKEND_NPU] = {.device               = 2,
                .threads              = 1,
                .name                 = "qnn-npu",
                .desc                 = "Qualcomm NPU(Hexagon Tensor Processor)",
                .lib                  = "libQnnHtp.so",
                .instance             = nullptr,
                .backend              = nullptr,
                .raw_interface        = {},
                .raw_system_interface = {},
                .socinfo              = {}},
};

struct qnn_op_caps_t {
    const char * qnn_op_name = nullptr;
    const size_t input_param_count = 0;
    const char * qnn_param_name = nullptr;
};

static const qnn_op_caps_t k_op_caps[] = {
        {}, // GGML_OP_NONE
        {}, // GGML_OP_DUP
        {
                // GGML_OP_ADD
                QNN_OP_ELEMENT_WISE_ADD,
                2,
        },
        {}, // GGML_OP_ADD1
        {}, // GGML_OP_ACC
        {}, // GGML_OP_SUB
        {
                // GGML_OP_MUL
                QNN_OP_ELEMENT_WISE_MULTIPLY,
                2,
        },
        {}, // GGML_OP_DIV
        {}, // GGML_OP_SQR
        {}, // GGML_OP_SQRT
        {}, // GGML_OP_LOG
        {}, // GGML_OP_SIN
        {}, // GGML_OP_COS
        {}, // GGML_OP_SUM
        {}, // GGML_OP_SUM_ROWS
        {}, // GGML_OP_MEAN
        {}, // GGML_OP_ARGMAX
        {}, // GGML_OP_COUNT_EQUAL
        {}, // GGML_OP_REPEAT
        {}, // GGML_OP_REPEAT_BACK
        {}, // GGML_OP_CONCAT
        {}, // GGML_OP_SILU_BACK
        {}, // GGML_OP_NORM
        {}, // GGML_OP_RMS_NORM
        {}, // GGML_OP_RMS_NORM_BACK
        {}, // GGML_OP_GROUP_NORM
        {
                // GGML_OP_MUL_MAT
                QNN_OP_MAT_MUL,
                2,
        },
        {}, // GGML_OP_MUL_MAT_ID
        {}, // GGML_OP_OUT_PROD
        {}, // GGML_OP_SCALE
        {}, // GGML_OP_SET
        {}, // GGML_OP_CPY
        {}, // GGML_OP_CONT
        {}, // GGML_OP_RESHAPE
        {}, // GGML_OP_VIEW
        {}, // GGML_OP_PERMUTE
        {}, // GGML_OP_TRANSPOSE
        {}, // GGML_OP_GET_ROWS
        {}, // GGML_OP_GET_ROWS_BACK
        {}, // GGML_OP_DIAG
        {}, // GGML_OP_DIAG_MASK_INF
        {}, // GGML_OP_DIAG_MASK_ZERO
        {}, // GGML_OP_SOFT_MAX
        {}, // GGML_OP_SOFT_MAX_BACK
        {}, // GGML_OP_ROPE
        {}, // GGML_OP_ROPE_BACK
        {}, // GGML_OP_CLAMP
        {}, // GGML_OP_CONV_TRANSPOSE_1D
        {}, // GGML_OP_IM2COL
        {}, // GGML_OP_IM2COL_BACK
        {}, // GGML_OP_CONV_TRANSPOSE_2D
        {}, // GGML_OP_POOL_1D
        {}, // GGML_OP_POOL_2D
        {}, // GGML_OP_POOL_2D_BACK
        {}, // GGML_OP_UPSCALE
        {}, // GGML_OP_PAD
        {}, // GGML_OP_PAD_REFLECT_1D
        {}, // GGML_OP_ARANGE
        {}, // GGML_OP_TIMESTEP_EMBEDDING
        {}, // GGML_OP_ARGSORT
        {}, // GGML_OP_LEAKY_RELU
        {}, // GGML_OP_FLASH_ATTN_EXT
        {}, // GGML_OP_FLASH_ATTN_BACK
        {}, // GGML_OP_SSM_CONV
        {}, // GGML_OP_SSM_SCAN
        {}, // GGML_OP_WIN_PART
        {}, // GGML_OP_WIN_UNPART
        {}, // GGML_OP_GET_REL_POS
        {}, // GGML_OP_ADD_REL_POS
        {}, // GGML_OP_RWKV_WKV6
        {}, // GGML_OP_GATED_LINEAR_ATTN
        {}, // GGML_OP_UNARY
        {}, // GGML_OP_MAP_UNARY
        {}, // GGML_OP_MAP_BINARY
        {}, // GGML_OP_MAP_CUSTOM1_F32
        {}, // GGML_OP_MAP_CUSTOM2_F32
        {}, // GGML_OP_MAP_CUSTOM3_F32
        {}, // GGML_OP_MAP_CUSTOM1
        {}, // GGML_OP_MAP_CUSTOM2
        {}, // GGML_OP_MAP_CUSTOM3
        {}, // GGML_OP_CROSS_ENTROPY_LOSS
        {}, // GGML_OP_CROSS_ENTROPY_LOSS_BACK
        {}, // GGML_OP_OPT_STEP_ADAMW
        {}, // GGML_UNARY_OP_ABS
        {}, // GGML_UNARY_OP_SGN
        {}, // GGML_UNARY_OP_NEG
        {}, // GGML_UNARY_OP_STEP
        {}, // GGML_UNARY_OP_TANH
        {}, // GGML_UNARY_OP_ELU
        {}, // GGML_UNARY_OP_RELU
        {}, // GGML_UNARY_OP_SIGMOID
        {}, // GGML_UNARY_OP_GELU
        {}, // GGML_UNARY_OP_GELU_QUICK
        {}, // GGML_UNARY_OP_SILU
        {}, // GGML_UNARY_OP_HARDSWISH
        {}, // GGML_UNARY_OP_HARDSIGMOID
        {}, // GGML_UNARY_OP_EXP
};

static const char * qnn_get_socmodel_desc(uint32_t soc_model) {
    switch (soc_model) {
        case SM7450:
            return "SM7450";
        case SM8350:
            return "SM8350";
        case SM8450:
            return "SM8450";
        case SM8475:
            return "SM8475";
        case SM8550:
            return "SM8550";
        case SM8650:
            return "SM8650";
        case SM8750:
            return "SM8750";
        default:
            return "unknown";
    }
}

static const char * qnn_get_htparch_desc(size_t htp_arch) {
    switch (htp_arch) {
        case V68:
            return "QCOM_HTP_V68";
        case V69:
            return "QCOM_HTP_V69";
        case V73:
            return "QCOM_HTP_V73";
        case V75:
            return "QCOM_HTP_V75";
        case V79:
            return "QCOM_HTP_V79";
        default:
            return "unknown";
    }
}

static struct qcom_socinfo * qnn_get_socinfo_from_socmodel(uint32_t soc_model) {
    size_t items = sizeof(g_qnn_soc_info_table) / sizeof(g_qnn_soc_info_table[0]);
    for (size_t idx = 0; idx < items; idx++) {
        if (soc_model == g_qnn_soc_info_table[idx].soc_model) {
            return &g_qnn_soc_info_table[idx];
        }
    }
    return nullptr;
}

static bool ggmlqnn_is_valid_params(ggml_backend_qnn_context * ctx, const ggml_tensor * src0,
                                    const ggml_tensor * src1, ggml_tensor * dst) {
    if ((nullptr == ctx) || (nullptr == src0) || (nullptr == src1) || (nullptr == dst)) {
        GGMLQNN_LOG_WARN("invalid params\n");
        return false;
    }

    qnn_instance * instance = ctx->instance;
    if (nullptr == instance) {
        GGMLQNN_LOG_WARN("invalid params\n");
        return false;
    }

    return true;
}

#define GGMLQNN_CHECK_PARAMS(ctx, src0, src1, dst)                          \
    do {                                                            \
        if (!ggmlqnn_is_valid_params((ctx), (src0), (src1), (dst))) {   \
            return;                                                 \
        }                                                           \
    } while (0)

static uint32_t ggml_get_tensor_rank(const ggml_tensor * tensor) {
    /*
    uint32_t rank = 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if ((0 != tensor->ne[i]) && (1 != tensor->ne[i])) {
            rank++;
        }
    }
    return rank;
    */
    return ggml_n_dims(tensor);
}

static uint32_t ggml_get_tensor_data_size(const ggml_tensor * tensor) {
    /*
    size_t data_size = ggml_row_size(tensor->type, tensor->ne[0]);
    size_t n_dims = ggml_get_tensor_rank(tensor);
    for (int i = 1; i < n_dims; i++) {
        data_size *= tensor->ne[i];
    }

    return data_size;
    */
    return ggml_nbytes(tensor);
}

static const char * ggml_get_type_name(ggml_type type) {
    const struct ggml_type_traits * traits = ggml_get_type_traits(type);
    return traits->type_name;
}

static const char * get_ggml_type_name(ggml_type type) {
    const auto * traits = ggml_get_type_traits(type);
    return traits->type_name;
}

//TODO:
// ref:explanation of k-quants, https://github.com/ggerganov/llama.cpp/pull/1684
static Qnn_DataType_t qnn_datatype_from_ggml_datatype(enum ggml_type ggmltype) {
    switch (ggmltype) {
        case GGML_TYPE_F16:
            return QNN_DATATYPE_FLOAT_16;
        case GGML_TYPE_F32:
            return QNN_DATATYPE_FLOAT_32;
        case GGML_TYPE_I8:
            return QNN_DATATYPE_INT_8;
        case GGML_TYPE_Q8_0:
            return QNN_DATATYPE_SFIXED_POINT_8;
        case GGML_TYPE_Q4_0:
            return QNN_DATATYPE_SFIXED_POINT_4;
        default:
            break;
    }
    return QNN_DATATYPE_UNDEFINED;
}

//TODO:
static ggml_type ggml_datatype_from_qnn_datatype(Qnn_DataType_t qnn_type) {
    switch (qnn_type) {
        case QNN_DATATYPE_FLOAT_32:
            return GGML_TYPE_F32;
        case QNN_DATATYPE_FLOAT_16:
            return GGML_TYPE_F16;
        case QNN_DATATYPE_UINT_32:
        case QNN_DATATYPE_INT_32:
            return GGML_TYPE_I32;
        case QNN_DATATYPE_INT_16:
            return GGML_TYPE_I16;
        case QNN_DATATYPE_INT_8:
            return GGML_TYPE_I8;
        case QNN_DATATYPE_SFIXED_POINT_8:
            return GGML_TYPE_Q8_0;
        case QNN_DATATYPE_SFIXED_POINT_4:
            return GGML_TYPE_Q4_0;
        default:
            break;
    }
    return GGML_TYPE_COUNT;
}

//TODO: add more ops
static const char * qnn_opname_from_ggmlop(enum ggml_op ggmlop) {
    switch (ggmlop) {
        case GGML_OP_ADD:
            return QNN_OP_ELEMENT_WISE_ADD;
        case GGML_OP_MUL_MAT:
            return QNN_OP_MAT_MUL;
        default:
            break;
    }
    return nullptr;
}

static Qnn_Tensor_t * ggml_qnn_create_general_tensor(const ggml_tensor * tensor, const char * name, Qnn_TensorType_t qnn_tensor_type,
                                                    Qnn_DataType_t qnn_data_type, uint32_t rank, uint32_t * dims, void * data, uint32_t data_size) {
    Qnn_ErrorHandle_t error = QNN_SUCCESS;
    char tensor_name[GGML_MAX_NAME] = {0};

    //TODO:remove get_idx() and inc_idx() in the future but ensure the tensor name is unique
    if (nullptr != name) {
        snprintf(tensor_name, GGML_MAX_NAME, "tensor_%-8d", get_idx());
    } else {
        snprintf(tensor_name, GGML_MAX_NAME, "tensor_%s%-8d", name, get_idx());
    }
    GGMLQNN_LOG_DEBUG("init_tensor %d", get_idx());
    inc_idx();

    uint32_t dimensions_transpose[GGML_MAX_DIMS] = {};
    uint32_t * tensor_dims = nullptr;
    if (nullptr != tensor) {
        //there are different dimension order between ggml tensor and qnn tensor
        for (size_t idx = 0; idx < rank; idx++) {
            dimensions_transpose[idx] = (uint32_t)tensor->ne[rank - 1 - idx];
        }
        tensor_dims = dimensions_transpose;
    }
    //re-assign tensor_dims
    if (nullptr != dims) {
        tensor_dims = dims;
    }

    Qnn_Tensor_t qnn_tensor = {
            .version= QNN_TENSOR_VERSION_1,
            {.v1= {
                    .id = 0,
                    .name = tensor_name,
                    .type = qnn_tensor_type,
                    .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                    .dataType = qnn_data_type,
                    .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                       QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                       {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                    .rank = rank,
                    .dimensions = tensor_dims,
                    .memType = QNN_TENSORMEMTYPE_RAW,
                    {.clientBuf = {nullptr, 0}
                    }
            }
            }
    };
    if (nullptr != name) {
        QNN_VER_PTR(qnn_tensor)->name = name;
    }
    Qnn_Tensor_t * p_qnn_tensor = (Qnn_Tensor_t *)calloc(1, sizeof(Qnn_Tensor_t));
    if (nullptr == p_qnn_tensor) {
        GGMLQNN_LOG_WARN("calloc failed");
        return nullptr;
    }
    error = deep_copy_qnn_tensors(qnn_tensor, * p_qnn_tensor);
    if (error != QNN_SUCCESS) {
        free(p_qnn_tensor);
        GGMLQNN_LOG_WARN("init tensor failed");
        return  nullptr;
    }
    QNN_VER_PTR(*p_qnn_tensor)->clientBuf = {data, data_size};

    return p_qnn_tensor;
}

static Qnn_Tensor_t * ggml_qnn_create_compute_tensor(const ggml_tensor * tensor) {
    uint32_t dimensions[] = {(uint32_t) tensor->ne[0], (uint32_t) tensor->ne[1],
                             (uint32_t) tensor->ne[2], (uint32_t) tensor->ne[3]};
    Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
    Qnn_TensorType_t qnn_tensor_type = QNN_TENSOR_TYPE_APP_WRITE;

    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        qnn_tensor_type = QNN_TENSOR_TYPE_APP_WRITE;
    } else if (tensor->flags & GGML_TENSOR_FLAG_OUTPUT) {
        qnn_tensor_type = QNN_TENSOR_TYPE_APP_READ;
    }

    qnn_data_type = qnn_datatype_from_ggml_datatype(tensor->type);
    Qnn_Tensor_t * p_qnn_tensor = ggml_qnn_create_general_tensor(tensor, nullptr,
                                  qnn_tensor_type, qnn_data_type,
                             ggml_n_dims(tensor), dimensions,
                             nullptr, 0);

    return p_qnn_tensor;
}

static void append_tensor_dimensions(const ggml_tensor * tensor, std::string & output) {
    char buffer[256] = {};
    const char * type_name = get_ggml_type_name(tensor->type);
    int len = 0;
    switch (ggml_n_dims(tensor)) {
        case 1:
            len = snprintf(buffer, sizeof(buffer), "%ldx1%s", (long)tensor->ne[0], type_name);
            break;
        case 2:
            len = snprintf(buffer, sizeof(buffer), "%ldx%ld%s", (long)tensor->ne[0], (long)tensor->ne[1], type_name);
            break;
        case 3:
            len = snprintf(buffer, sizeof(buffer), "%ldx%ldx%ld%s", (long)tensor->ne[0], (long)tensor->ne[1],
                           (long)tensor->ne[2], type_name);
            break;
        case 4:
        default:
            len = snprintf(buffer, sizeof(buffer), "%ldx%ldx%ldx%ld%s", (long)tensor->ne[0], (long)tensor->ne[1],
                           (long)tensor->ne[2], (long)tensor->ne[3], type_name);
            break;
    }
    GGML_ASSERT(len > 0 && len < (int)sizeof(buffer));
    output.append(buffer, len);
}

static size_t get_qnn_op_index(const ggml_tensor * tensor) {
    if (tensor->op == GGML_OP_UNARY) {
        return GGML_OP_COUNT + ggml_get_unary_op(tensor);
    }

    return tensor->op;
}

static size_t get_qnn_op_input_param_count(const ggml_tensor * op) {
    auto op_index = get_qnn_op_index(op);
    GGML_ASSERT(op_index < std::size(k_op_caps));
    return k_op_caps[op_index].input_param_count;
}

static void get_graph_key_from_op(const ggml_tensor * op, std::string & output) {
    GGML_ASSERT(op->op != GGML_OP_NONE);
    output += ggml_op_desc(op);
    output += get_ggml_type_name(op->type);
    size_t param_count = get_qnn_op_input_param_count(op);
    for (size_t i = 0; i < param_count; ++i) {
        auto * input = op->src[i];
        if (!input) {
            break;
        }
        output += '_';
        append_tensor_dimensions(input, output);
    }
}

#if ENABLE_QNNBACKEND_PERF
class qnn_perf {
public:
    qnn_perf(const std::string & perf_name) : _perf_name(std::move(perf_name)) {};
    qnn_perf() = delete;
    qnn_perf(const qnn_perf & ) = delete;
    qnn_perf & operator= (const qnn_perf & ) = delete;

    void start() {
        _begin_time = ggml_time_us();
    }

    void info() {
        _end_time = ggml_time_us();
        _duration = (_end_time - _begin_time);
        GGMLQNN_LOG_DEBUG("duration of %s : %lld microseconds\n", _perf_name.c_str(), _duration);
    }

private:
    int64_t _begin_time = 0LL;
    int64_t _end_time   = 0LL;
    int64_t _duration   = 0LL;
    std::string _perf_name;
};
#else
class qnn_perf {
public:
    qnn_perf(const std::string & perf_name) {}
    qnn_perf() = delete;
    qnn_perf(const qnn_perf & ) = delete;
    qnn_perf & operator= (const qnn_perf & ) = delete;

    void start() {}
    void info() {}
};
#endif

template<typename Fn>
Fn load_qnn_functionpointers(void * handle, const char * function_name) {
    return reinterpret_cast<Fn>(dlsym(handle, function_name));
}

class qnn_interface {

#define DEFINE_SHIM_FUNCTION_INTERFACE(F, pointer_name)           \
  template <typename... Args>                                     \
  inline auto qnn_##F(Args... args) const {                       \
    return (_qnn_interface->QNN_INTERFACE_VER_NAME.pointer_name)( \
        std::forward<Args>(args)...);                             \
  }


#define DEFINE_SHIM_FUNCTION_SYS_INTERFACE(F, pointer_name)                  \
  template <typename... Args>                                                \
  inline auto qnn_##F(Args... args) const {                                  \
    return (_qnn_sys_interface->QNN_SYSTEM_INTERFACE_VER_NAME.pointer_name)( \
        std::forward<Args>(args)...);                                        \
  }

    friend class qnn_instance;

public:
    qnn_interface() = default;

    // QnnBackend
    DEFINE_SHIM_FUNCTION_INTERFACE(backend_create, backendCreate);

    DEFINE_SHIM_FUNCTION_INTERFACE(backend_free, backendFree);

    DEFINE_SHIM_FUNCTION_INTERFACE(backend_register_op_package, backendRegisterOpPackage);

    DEFINE_SHIM_FUNCTION_INTERFACE(backend_validate_op_config, backendValidateOpConfig);

    DEFINE_SHIM_FUNCTION_INTERFACE(backend_get_api_version, backendGetApiVersion);

    // QnnDevice
    DEFINE_SHIM_FUNCTION_INTERFACE(device_create, deviceCreate);

    DEFINE_SHIM_FUNCTION_INTERFACE(device_free, deviceFree);

    DEFINE_SHIM_FUNCTION_INTERFACE(device_get_infrastructure, deviceGetInfrastructure);

    DEFINE_SHIM_FUNCTION_INTERFACE(device_get_platform_info, deviceGetPlatformInfo);

    DEFINE_SHIM_FUNCTION_INTERFACE(device_get_info, deviceGetInfo);

    // QnnContext
    DEFINE_SHIM_FUNCTION_INTERFACE(context_create, contextCreate);

    DEFINE_SHIM_FUNCTION_INTERFACE(context_get_binary_size, contextGetBinarySize);

    DEFINE_SHIM_FUNCTION_INTERFACE(context_get_binary, contextGetBinary);

    DEFINE_SHIM_FUNCTION_INTERFACE(context_create_from_binary, contextCreateFromBinary);

    DEFINE_SHIM_FUNCTION_INTERFACE(context_free, contextFree);

    // QnnGraph
    DEFINE_SHIM_FUNCTION_INTERFACE(graph_create, graphCreate);

    DEFINE_SHIM_FUNCTION_INTERFACE(graph_add_node, graphAddNode);

    DEFINE_SHIM_FUNCTION_INTERFACE(graph_finalize, graphFinalize);

    DEFINE_SHIM_FUNCTION_INTERFACE(graph_execute, graphExecute);

    DEFINE_SHIM_FUNCTION_INTERFACE(graph_retrieve, graphRetrieve);

    // QnnLog
    DEFINE_SHIM_FUNCTION_INTERFACE(log_create, logCreate);

    DEFINE_SHIM_FUNCTION_INTERFACE(log_free, logFree);

    DEFINE_SHIM_FUNCTION_INTERFACE(log_set_log_level, logSetLogLevel);

    // QnnProfile
    DEFINE_SHIM_FUNCTION_INTERFACE(profile_create, profileCreate);

    DEFINE_SHIM_FUNCTION_INTERFACE(profile_get_events, profileGetEvents);

    DEFINE_SHIM_FUNCTION_INTERFACE(profile_get_sub_events, profileGetSubEvents);

    DEFINE_SHIM_FUNCTION_INTERFACE(profile_get_event_data, profileGetEventData);

    DEFINE_SHIM_FUNCTION_INTERFACE(profile_free, profileFree);

    // QnnMem
    DEFINE_SHIM_FUNCTION_INTERFACE(mem_register, memRegister);

    DEFINE_SHIM_FUNCTION_INTERFACE(mem_de_register, memDeRegister);

    // QnnProperty
    DEFINE_SHIM_FUNCTION_INTERFACE(property_has_capability, propertyHasCapability);

    // QnnTensor
    DEFINE_SHIM_FUNCTION_INTERFACE(tensor_create_context_tensor, tensorCreateContextTensor);

    DEFINE_SHIM_FUNCTION_INTERFACE(tensor_create_graph_tensor, tensorCreateGraphTensor);

    // QnnSystem
    DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_create, systemContextCreate);

    DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_get_binary_info, systemContextGetBinaryInfo);

    DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_free, systemContextFree);

    void set_qnn_interface(const QnnInterface_t * qnn_interface) {
        _qnn_interface = qnn_interface;
    }

    void set_qnn_system_interface(const QnnSystemInterface_t * qnn_sys_interface) {
        _qnn_sys_interface = qnn_sys_interface;
    }

    uint32_t get_backend_id() const {
        return _qnn_interface->backendId;
    }

    bool is_loaded() const {
        return ((_qnn_sys_interface != nullptr) && (_qnn_interface != nullptr));
    }

private:
    const QnnInterface_t *_qnn_interface = nullptr;

    const QnnSystemInterface_t *_qnn_sys_interface = nullptr;
};

class qnn_instance {
public:
    using BackendIdType = decltype(QnnInterface_t{}.backendId);

    explicit qnn_instance(const std::string & lib_path, const std::string & backend_name,
                                const std::string & model_name) :
            _lib_path(std::move(lib_path)),
            _backend_name(std::move(backend_name)),
            _model_name(std::move(model_name)) {};

    ~qnn_instance() {
    }

    int qnn_init(const QnnSaver_Config_t ** saver_config);

    int qnn_finalize();

    const qnn_interface & get_qnn_interface() {
        if (!_qnn_interface.is_loaded()) {
            GGMLQNN_LOG_WARN("pls check why _qnn_interface is not loaded\n");
        }
        return _qnn_interface;
    }

    const QNN_INTERFACE_VER_TYPE & get_qnn_raw_interface() {
        if (!_qnn_interface.is_loaded()) {
            GGMLQNN_LOG_WARN("pls check why _qnn_interface is not loaded\n");
        }
        return _qnn_raw_interface;
    }

    const QNN_SYSTEM_INTERFACE_VER_TYPE & get_qnn_raw_system_interface() {
        if (!_qnn_interface.is_loaded()) {
            GGMLQNN_LOG_WARN("pls check why _qnn_interface is not loaded\n");
        }
        return _qnn_raw_system_interface;
    }

    const Qnn_LogHandle_t get_qnn_log_handle() { return _qnn_log_handle; }

    const Qnn_ProfileHandle_t get_qnn_profile_handle() { return _qnn_profile_handle; }

    const Qnn_DeviceHandle_t get_qnn_device_handle() { return _qnn_device_handle; }

    const Qnn_BackendHandle_t get_qnn_backend_handle() { return _qnn_backend_handle; }

    const Qnn_ContextHandle_t get_qnn_context_handle() { return _qnn_context_handle; }

    const QnnSystemContext_Handle_t get_qnn_system_handle() { return _qnn_system_handle; }

    const Qnn_GraphHandle_t get_qnn_graph_handle() { return _qnn_graph_handle; }

    int init_qnn_graph(const char * graph_name,
                       bool debug,
                       uint8_t do_node_validation = 1,
                       const QnnGraph_Config_t ** graph_configs = nullptr
    );
    int init_qnn_graph(const std::string & graph_name, QNNBackend device, size_t vtcm_size_in_mb = 8, size_t hvx_threads = 8);

    int finalize_qnn_graph();

    bool is_valid_graph() const { return _qnn_graph_handle != nullptr; }

    int init_htp_perfinfra() {
        QnnDevice_Infrastructure_t device_infra = nullptr;
        int error = _qnn_raw_interface.deviceGetInfrastructure(&device_infra);
        if (error != QNN_SUCCESS) {
            GGMLQNN_LOG_WARN("failed to get qnn device infra\n");
            return 1;
        }

        QnnHtpDevice_Infrastructure_t * htp_infra = static_cast<QnnHtpDevice_Infrastructure_t *>(device_infra);
        QnnHtpDevice_PerfInfrastructure_t * htp_perfinfra = &htp_infra->perfInfra;
        uint32_t power_configid = 1;
        uint32_t device_id = 0;
        uint32_t core_id = 0;
        htp_perfinfra->createPowerConfigId(device_id, core_id, &power_configid);
        _qnn_htp_perfinfra = htp_perfinfra;
        _qnn_power_configid = power_configid;

        return 0;
    }

    int set_rpc_polling() {
        if (_qnn_rpc_pollingtime > 0) {
            QnnHtpPerfInfrastructure_PowerConfig_t rpc_pollingtime;
            memset(&rpc_pollingtime, 0, sizeof(rpc_pollingtime));
            rpc_pollingtime.option =
                    QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
            rpc_pollingtime.rpcPollingTimeConfig = _qnn_rpc_pollingtime;
            const QnnHtpPerfInfrastructure_PowerConfig_t * power_configs[] = {&rpc_pollingtime, nullptr};
            if (_qnn_htp_perfinfra) {
                _qnn_htp_perfinfra->setPowerConfig(_qnn_power_configid, power_configs);
            }
        }
        return 0;
    }

    int set_high_performance_mode() {
        if (nullptr == _qnn_htp_perfinfra) {
            GGMLQNN_LOG_DEBUG("perf intra is null\n");
            return 1;
        }

        QnnHtpPerfInfrastructure_PowerConfig_t power_config;
        memset(&power_config, 0, sizeof(power_config));
        power_config.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
        power_config.dcvsV3Config.dcvsEnable = 0;
        power_config.dcvsV3Config.setDcvsEnable = 1;
        power_config.dcvsV3Config.contextId = _qnn_power_configid;
        power_config.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
        power_config.dcvsV3Config.setSleepLatency = 1; // True to consider Latency parameter otherwise False
        power_config.dcvsV3Config.setBusParams = 1; // True to consider Bus parameter otherwise False
        power_config.dcvsV3Config.setCoreParams = 1; // True to consider Core parameter otherwise False
        power_config.dcvsV3Config.sleepDisable = 0; // True to consider sleep/LPM modes, False to enable
        power_config.dcvsV3Config.setSleepDisable = 0; // True to consider sleep disable/enable parameter otherwise False
        // set Sleep latency parameter
        uint32_t latencyValue = 40;
        power_config.dcvsV3Config.sleepLatency = latencyValue; // range 40-2000 micro sec
        // set Bus Clock Parameters (refer QnnHtpPerfInfrastructure_VoltageCorner_t enum)
        power_config.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        power_config.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        power_config.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        // set Core Clock Parameters (refer QnnHtpPerfInfrastructure_VoltageCorner_t enum)
        power_config.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        power_config.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        power_config.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        // set power config with different performance parameters
        const QnnHtpPerfInfrastructure_PowerConfig_t * power_configs[] = {&power_config, nullptr};

        _qnn_htp_perfinfra->setPowerConfig(_qnn_power_configid, power_configs);

        return 0;
    }

    std::string & get_qnn_graph_name() { return _graph_name; }

    bool is_rpcmem_initialized() {
        return _rpcmem_initialized;
    }

    void set_rpcmem_initialized(bool initialized) {
        _rpcmem_initialized = initialized;
    }

    size_t get_rpcmem_capacity() { return _rpcmem_capacity; }
    size_t get_rpcmem_usage() { return _rpcmem_usage; }

    int32_t rpcmem_to_fd(void * buf);

    int register_rpcmem(void * p_data, Qnn_Tensor_t * p_tensor);
    Qnn_MemHandle_t  register_rpcmem(void * p_data, const uint32_t rank, uint32_t * dimensions, Qnn_DataType_t data_type);

    void unregister_rpcmem();
    void unregister_rpcmem(Qnn_MemHandle_t mem_handle);

    void * alloc_rpcmem(size_t bytes, size_t alignment);
    void * get_rpcmem_from_memhandle(Qnn_MemHandle_t mem_handle);

    void free_rpcmem(void * buf);
    void free_rpcmem();

    bool is_rpcmem_allocated(void * buf);

    bool is_rpcmem_registered(Qnn_MemHandle_t handle) {
        return _qnn_mem_set.count(handle) != 0U;
    }

    bool enable_qnn_rpc() {
        return _enable_qnn_rpc;
    }

    void probe_device_meminfo() {
        size_t candidate_size = 0;
        uint8_t *rpc_buffer = nullptr;
        const int SIZE_IN_MB = (1 << 20);
        size_t probe_slots[] = {1024, 1536, 2048 - 48, 2048};
        size_t probe_counts = sizeof(probe_slots) / sizeof(size_t);
        for (size_t idx = 0; idx < probe_counts; idx++) {
            rpc_buffer = static_cast<uint8_t *>(alloc_rpcmem_internal(probe_slots[idx] * SIZE_IN_MB, 4));
            if (nullptr == rpc_buffer) {
                GGMLQNN_LOG_DEBUG("alloc rpcmem %d (MB) failure, %s\n", probe_slots[idx],
                                  strerror(errno));
                break;
            } else {
                candidate_size = probe_slots[idx];
                free_rpcmem(rpc_buffer);
                rpc_buffer = nullptr;
            }
        }
        if (candidate_size > _rpcmem_capacity)
            _rpcmem_capacity = candidate_size;

        free_rpcmem();
        _rpcmem_usage = 0;
        GGMLQNN_LOG_INFO("capacity of rpc ion memory %d MB\n", _rpcmem_capacity);
    }

public:
    std::map<std::string, std::tuple<Qnn_GraphHandle_t, std::vector< Qnn_Tensor_t *>>> _qnn_graph_map;

private:
    int load_system();

    int unload_system();

    int load_backend(std::string & lib_path, const QnnSaver_Config_t ** saver_config);

    int unload_backend();

    void set_qnn_raw_interface(QNN_INTERFACE_VER_TYPE & raw_interface) {
        _qnn_raw_interface = raw_interface;
    }

    void set_qnn_raw_system_interface(QNN_SYSTEM_INTERFACE_VER_TYPE & raw_interface) {
        _qnn_raw_system_interface = raw_interface;
    }
    
    void * alloc_rpcmem_internal(size_t bytes, size_t alignment);

private:
    static constexpr const int _required_num_providers = 1;

private:
    std::string _lib_path;
    std::string _backend_name;
    std::string _model_name;               // name of prebuilt QNN model, might be used in the future
    BackendIdType _backend_id;

    bool _debug_tensor                      = false; // flag to indicate if requested graph is to be run in debug mode
    bool _do_node_validations               = true;  // flag to indicate whether all add_node calls need to be validated
    QnnLog_Level_t _qnn_log_level           = QNN_LOG_LEVEL_DEBUG;

    ggml_qnn_profile_level _profile_level   = ggml_qnn_profile_level::profile_detail;

    qnn_interface _qnn_interface;

    void * _system_lib_handle = nullptr;

    Qnn_GraphHandle_t _qnn_graph_handle = nullptr;

    Qnn_LogHandle_t _qnn_log_handle = nullptr;

    Qnn_ProfileHandle_t _qnn_profile_handle = nullptr;

    Qnn_DeviceHandle_t _qnn_device_handle = nullptr;

    Qnn_BackendHandle_t _qnn_backend_handle = nullptr;

    Qnn_ContextHandle_t _qnn_context_handle = nullptr;

    QnnSystemContext_Handle_t _qnn_system_handle = nullptr;

    QnnHtpDevice_PerfInfrastructure_t *_qnn_htp_perfinfra = nullptr;
    uint32_t _qnn_power_configid = 1;
    uint32_t _qnn_rpc_pollingtime = 9999; // 0-10000 us for high performing

    QNN_INTERFACE_VER_TYPE _qnn_raw_interface;
    QNN_SYSTEM_INTERFACE_VER_TYPE _qnn_raw_system_interface;

    std::unordered_map<void *, Qnn_MemHandle_t> _qnn_mem_set;
    std::unordered_map<void *, Qnn_MemHandle_t> _qnn_rpc_buffer_to_handles;

    static std::mutex _init_mutex;
    static std::unordered_map<BackendIdType, void *> _loaded_lib_handle;
    static std::unordered_map<std::string, BackendIdType> _lib_path_to_backend_id;
    static std::unordered_map<BackendIdType, const QnnInterface_t *> _loaded_backend;

    void * _rpc_lib_handle = nullptr;
    std::atomic_bool _rpcmem_initialized{false};
    pfn_rpc_mem_alloc _pfn_rpc_mem_alloc;
    pfn_rpc_mem_free _pfn_rpc_mem_free;
    pfn_rpc_mem_to_fd _pfn_rpc_mem_to_fd;
    pfn_rpc_mem_init  _pfn_rpc_mem_init;
    pfn_rpc_mem_deinit _pfn_rpc_mem_deinit;
    std::unordered_map<void *, void *> _rpcmem_store_map;
    std::unordered_map<void *, size_t> _rpcmem_usage_map;
    size_t                             _rpcmem_capacity = 512; // mempool size  in Mbytes
    size_t                             _rpcmem_usage    = 0;   // mempool usage in Mbytes

    std::string _graph_name;
    QNNBackend _device_id;
    bool       _enable_qnn_rpc = false; //TODO:unknown issue with QNN RPC feature

    DISABLE_COPY(qnn_instance);
    DISABLE_MOVE(qnn_instance);
};

std::mutex qnn_instance::_init_mutex;
std::unordered_map<qnn_instance::BackendIdType, void *> qnn_instance::_loaded_lib_handle;
std::unordered_map<std::string, qnn_instance::BackendIdType> qnn_instance::_lib_path_to_backend_id;
std::unordered_map<qnn_instance::BackendIdType, const QnnInterface_t *> qnn_instance::_loaded_backend;

void * qnn_instance::alloc_rpcmem_internal(size_t bytes, size_t alignment) {
    if (!_rpcmem_initialized) {
        GGMLQNN_LOG_WARN("rpc memory not initialized\n");
        return nullptr;
    }

    auto allocate_bytes = static_cast<int32_t>(bytes + alignment);
    void * buf = _pfn_rpc_mem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, allocate_bytes);
    if (buf == nullptr) {
        GGMLQNN_LOG_WARN("failed to allocate rpc memory\n");
        return nullptr;
    }

    auto aligned_buf = reinterpret_cast<void *>(ggmlqnn_align_to(alignment,
                                                         reinterpret_cast<intptr_t>(buf)));
    bool status = _rpcmem_store_map.insert(std::pair<void *, void *>(aligned_buf, buf)).second;
    if (!status) {
        GGMLQNN_LOG_WARN("failed to allocate rpc memory\n");
        _pfn_rpc_mem_free(buf);
    }
    return aligned_buf;
}

void * qnn_instance::alloc_rpcmem(size_t bytes, size_t alignment) {
    if (_rpcmem_usage > (_rpcmem_capacity - 8)) { // reserve 8Mbytes in rpc mempool
        GGMLQNN_LOG_WARN("rpc mempool capcaity: %d MB, usage: %d MB", _rpcmem_capacity, _rpcmem_usage);
        return nullptr;
    }

    auto aligned_buf = alloc_rpcmem_internal(bytes, alignment);
    if (nullptr == aligned_buf)
        return nullptr;
    _rpcmem_usage_map.insert(std::pair<void *, size_t>(aligned_buf, bytes));

    size_t rpcmem_usage_in_bytes = _rpcmem_usage * (1 << 20);
    rpcmem_usage_in_bytes += bytes;
    _rpcmem_usage = rpcmem_usage_in_bytes / ( 1 << 20);
    return aligned_buf;
}

void qnn_instance::free_rpcmem(void * buf) {
    size_t rpcbuffer_size = 0;
    if (!_rpcmem_initialized) {
        GGMLQNN_LOG_WARN("rpc memory not initialized\n");
    } else if (0 == _rpcmem_store_map.count(buf)) {
        GGMLQNN_LOG_WARN("no allocated tensor\n");
    } else {
        GGMLQNN_LOG_DEBUG("free rpc mem %p", _rpcmem_store_map[buf]);
        for (std::unordered_map<void *, size_t>::iterator it = _rpcmem_usage_map.begin();
             it != _rpcmem_usage_map.end();
             it++) {
            void * rpcbuffer = it->first;
            if (buf == rpcbuffer) {
                rpcbuffer_size = it->second;
                size_t rpcmem_usage_in_bytes = _rpcmem_usage * (1 << 20);
                rpcmem_usage_in_bytes -= rpcbuffer_size;
                _rpcmem_usage = rpcmem_usage_in_bytes / ( 1 << 20);
            }
        }
        if (rpcbuffer_size != 0) {
            _rpcmem_usage_map.erase(buf);
        } else {
            GGMLQNN_LOG_WARN("it shouldn't happen, pls check why?");
        }
        _pfn_rpc_mem_free(_rpcmem_store_map[buf]);
        _rpcmem_store_map.erase(buf);
    }
}

void qnn_instance::free_rpcmem() {
    Qnn_ErrorHandle_t error = QNN_SUCCESS;

    if (_rpcmem_store_map.empty()) {
        GGMLQNN_LOG_WARN("no rpcmem allocated\n");
        return;
    }

    for (std::unordered_map<void *, void *>::iterator it = _rpcmem_store_map.begin();
         it != _qnn_mem_set.end();
         it++) {
        void * rpcbuffer = it->second;
        GGMLQNN_LOG_DEBUG("free rpc buffer %p", rpcbuffer);
        _pfn_rpc_mem_free(rpcbuffer);
    }
    _rpcmem_store_map.clear();
    _rpcmem_usage_map.clear();
    _rpcmem_usage = 0;
}

int32_t qnn_instance::rpcmem_to_fd(void * buf) {
    int32_t mem_fd = -1;
    if (!is_rpcmem_initialized()) {
        GGMLQNN_LOG_WARN("rpc memory not initialized\n");
    } else {
        mem_fd = _pfn_rpc_mem_to_fd(buf);
    }

    return mem_fd;
}

int qnn_instance::register_rpcmem(void * p_data, Qnn_Tensor_t * p_tensor) {
    if (nullptr == p_data || (nullptr == p_tensor)) {
        GGMLQNN_LOG_WARN("invalid param\n");
        return 1;
    }

    if (!is_rpcmem_initialized()) {
        GGMLQNN_LOG_WARN("rpc memory not initialized\n");
        return 2;
    }

    if (is_rpcmem_registered((QNN_VER_PTR(*p_tensor)->memHandle))) {
        GGMLQNN_LOG_WARN("tensor %s has been registered shared memory\n", (QNN_VER_PTR(*p_tensor)->name));
        return 4;
    }

    int32_t mem_fd = rpcmem_to_fd(p_data);
    if (-1 == mem_fd) {
        GGMLQNN_LOG_WARN("failed to get file descriptor\n");
        return 5;
    }
    GGMLQNN_LOG_DEBUG("mem_fd %d\n", mem_fd);
    Qnn_MemDescriptor_t descriptor = {
            {QNN_VER_PTR(*p_tensor)->rank, QNN_VER_PTR(*p_tensor)->dimensions, nullptr},
            QNN_VER_PTR(*p_tensor)->dataType,
            QNN_MEM_TYPE_ION,
            {{mem_fd}}};
    Qnn_MemHandle_t handle = nullptr;
    int error = QNN_SUCCESS;
    error = _qnn_interface.qnn_mem_register(
            _qnn_context_handle,
            &descriptor,
            /*numDescriptors=*/1,
            &handle);
    if (error != QNN_SUCCESS) {
        GGMLQNN_LOG_WARN("failed to register shared memory, error %d, %s\n", QNN_GET_ERROR_CODE(error),
              strerror(error));
        return 6;
    } else {
        GGMLQNN_LOG_INFO("tensor %s successfully register shared memory\n", (QNN_VER_PTR(*p_tensor)->name));
    }
    QNN_VER_PTR(*p_tensor)->memHandle = handle;
    _qnn_mem_set.insert((std::pair<void*, Qnn_MemHandle_t>(p_data, handle)));

    return 0;
}

Qnn_MemHandle_t  qnn_instance::register_rpcmem(void * p_data, const uint32_t rank, uint32_t * dimensions, Qnn_DataType_t data_type) {
    if (!p_data) {
        GGMLQNN_LOG_WARN("invalid param");
        return nullptr;
    }

    if (!is_rpcmem_initialized()) {
        GGMLQNN_LOG_WARN("rpc memory not initialized");
        return nullptr;
    }

    if (is_rpcmem_registered(p_data)) {
        GGMLQNN_LOG_WARN("rpc memory already registered");
        return _qnn_rpc_buffer_to_handles[p_data];
    }

    auto mem_fd = rpcmem_to_fd(p_data);
    if (mem_fd == -1) {
        GGMLQNN_LOG_WARN("failed to get file descriptor");
        return nullptr;
    }

    GGMLQNN_LOG_DEBUG("mem_fd %d", mem_fd);
    Qnn_MemDescriptor_t descriptor = {
            {rank, dimensions, nullptr},
            data_type, QNN_MEM_TYPE_ION,
            {{mem_fd}}
    };
    Qnn_MemHandle_t handle = nullptr;
    auto error = _qnn_interface.qnn_mem_register(_qnn_context_handle, &descriptor,
            /*numDescriptors=*/1, &handle);
    if (error != QNN_SUCCESS) {
        GGMLQNN_LOG_WARN("failed to register shared memory, error %d, %s", QNN_GET_ERROR_CODE(error), strerror(error));
        return nullptr;
    }

    _qnn_rpc_buffer_to_handles.insert({p_data, handle});
    GGMLQNN_LOG_DEBUG("successfully register shared memory handler: %p", handle);
    return handle;
}

void * qnn_instance::get_rpcmem_from_memhandle(Qnn_MemHandle_t mem_handle) {
    for (std::unordered_map<void *, Qnn_MemHandle_t>::iterator it = _qnn_mem_set.begin();
         it != _qnn_mem_set.end();
         it++) {
        Qnn_MemHandle_t mem_handle = it->second;
        if (it->second == mem_handle) {
            return it->first;
        }
    }
    GGMLQNN_LOG_WARN("can't find rpcmem from qnn mem handle %p", mem_handle);
    return nullptr;
}

void qnn_instance::unregister_rpcmem() {
    Qnn_ErrorHandle_t error = QNN_SUCCESS;

    if (_qnn_mem_set.empty()) {
        GGMLQNN_LOG_WARN("no rpcmem registered\n");
    }

    for (std::unordered_map<void *, Qnn_MemHandle_t>::iterator it = _qnn_mem_set.begin();
         it != _qnn_mem_set.end();
         it++) {
        Qnn_MemHandle_t mem_handle = it->second;
        error = _qnn_interface.qnn_mem_de_register(&mem_handle, 1);
        if (error != QNN_SUCCESS) {
            GGMLQNN_LOG_WARN("failed to unregister shared memory, error %d\n",
                         QNN_GET_ERROR_CODE(error));
        } else {
            GGMLQNN_LOG_DEBUG("unregister shared memory ok");
        }
    }
    _qnn_mem_set.clear();
}

void qnn_instance::unregister_rpcmem(Qnn_MemHandle_t mem_handle) {
    Qnn_ErrorHandle_t error = _qnn_interface.qnn_mem_de_register(&mem_handle, 1);
    if (error != QNN_SUCCESS) {
        GGMLQNN_LOG_WARN("failed to unregister shared memory, error %d", QNN_GET_ERROR_CODE(error));
    }

    auto it = std::find_if(_qnn_mem_set.begin(), _qnn_mem_set.end(),
                           [mem_handle](const auto &kv) { return kv.second == mem_handle; });
    if (it == _qnn_mem_set.end()) {
        GGMLQNN_LOG_WARN("failed to find shared memory handler: %p", mem_handle);
        return;
    }

    _qnn_mem_set.erase(it);
}

bool qnn_instance::is_rpcmem_allocated(void * buf) {
    return _rpcmem_store_map.count(buf) != 0U;
}

int qnn_instance::load_backend(std::string & lib_path, const QnnSaver_Config_t ** saver_config) {
    Qnn_ErrorHandle_t error = QNN_SUCCESS;
    GGMLQNN_LOG_DEBUG("lib_path:%s\n", lib_path.c_str());

    void * lib_handle = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (nullptr == lib_handle) {
        GGMLQNN_LOG_WARN("can not open QNN library %s, with error: %s", lib_path.c_str(), dlerror());
        return 1;
    }

    auto get_providers =
            load_qnn_functionpointers<_pfn_QnnInterface_getProviders *>(lib_handle,
                                                          "QnnInterface_getProviders");
    if (nullptr == get_providers) {
        GGMLQNN_LOG_WARN("can not load symbol QnnInterface_getProviders : %s", dlerror());
        return 2;
    }

    // get QnnInterface Providers
    std::uint32_t num_providers = 0;
    const QnnInterface_t ** provider_list = nullptr;
    error = get_providers(&provider_list, &num_providers);
    if (error != QNN_SUCCESS) {
        GGMLQNN_LOG_WARN("failed to get providers, error %d", QNN_GET_ERROR_CODE(error));
        return 3;
    }
    GGMLQNN_LOG_DEBUG("num_providers=%d\n", num_providers);
    if (num_providers != _required_num_providers) {
        GGMLQNN_LOG_WARN("providers is %d instead of required %d", num_providers, _required_num_providers);
        return 4;
    }

    if (nullptr == provider_list) {
        GGMLQNN_LOG_WARN("failed to get qnn interface providers\n");
        return 5;
    }
    bool found_valid_interface = false;
    QNN_INTERFACE_VER_TYPE qnn_interface;
    for (size_t idx = 0; idx < num_providers; idx++) {
        if (QNN_API_VERSION_MAJOR == provider_list[idx]->apiVersion.coreApiVersion.major &&
            QNN_API_VERSION_MINOR <= provider_list[idx]->apiVersion.coreApiVersion.minor) {
            found_valid_interface = true;
            qnn_interface = provider_list[idx]->QNN_INTERFACE_VER_NAME;
            break;
        }
    }

    if (!found_valid_interface) {
        GGMLQNN_LOG_WARN("unable to find a valid qnn interface\n");
        return 6;
    } else {
        GGMLQNN_LOG_INFO("find a valid qnn interface\n");
    }
    set_qnn_raw_interface(qnn_interface);

    BackendIdType backend_id = provider_list[0]->backendId;
    _lib_path_to_backend_id[lib_path] = backend_id;
    if (_loaded_backend.count(backend_id) > 0) {
        GGMLQNN_LOG_WARN("lib_path %s is loaded, but backend %d already exists\n",
              lib_path.c_str(), backend_id);
    }
    _loaded_backend[backend_id] = provider_list[0];
    if (_loaded_lib_handle.count(backend_id) > 0) {
        GGMLQNN_LOG_WARN("closing %p\n", _loaded_lib_handle[backend_id]);
        int dlclose_error = dlclose(_loaded_lib_handle[backend_id]);
        if (dlclose_error != 0) {
            GGMLQNN_LOG_WARN("fail to close %p with error %s\n", _loaded_lib_handle[backend_id], dlerror());
        }
    }
    _loaded_lib_handle[backend_id] = lib_handle;
    _backend_id = backend_id;

#if 0 // leave them here for further use
    QnnSaver_Config_t outputdir_cfg;
    outputdir_cfg.option = QNN_SAVER_CONFIG_OPTION_OUTPUT_DIRECTORY;
    outputdir_cfg.outputDirectory = "/data/local/tmp/";
    QnnSaver_Config_t backendid_cfg;
    backendid_cfg.option = QNN_SAVER_CONFIG_OPTION_BACKEND_ID;
    backendid_cfg.backendId = _backend_id;

    const QnnSaver_Config_t * saver_cfg[] = {&outputdir_cfg, &backendid_cfg, nullptr};
    if (0 == QnnSaver_initialize(saver_cfg)) {
        GGMLQNN_LOG_INFO("QnnSaver_initialize successfully");
    } else {
        GGMLQNN_LOG_WARN("QnnSaver_initialize failure");
    }
#endif
    auto saver_initialize =
            load_qnn_functionpointers<_pfn_QnnSaver_initialize *>(
            _loaded_lib_handle[backend_id], "QnnSaver_initialize");
    if (nullptr != saver_initialize) {
        error = saver_initialize(saver_config);
        if (error != QNN_SUCCESS) {
            GGMLQNN_LOG_WARN("failed to saver_initialize，error %d", QNN_GET_ERROR_CODE(error));
            return 7;
        }
    } else {
        GGMLQNN_LOG_WARN("saver_initialize is null\n");
    }

    return 0;
}

int qnn_instance::unload_backend() {
    int dlclose_error = 0;
    for (auto &it : _loaded_lib_handle) {
        dlclose_error = dlclose(it.second);
        if (dlclose_error != 0) {
            GGMLQNN_LOG_WARN("failed to close QNN backend %d, error %s\n", it.first, dlerror());
        }
    }

    _loaded_lib_handle.clear();
    _lib_path_to_backend_id.clear();
    _loaded_backend.clear();

    return 0;
}

int qnn_instance::load_system() {
    Qnn_ErrorHandle_t error = QNN_SUCCESS;

    std::string system_lib_path = _lib_path + "libQnnSystem.so";
    GGMLQNN_LOG_DEBUG("system_lib_path:%s\n", system_lib_path.c_str());

    _system_lib_handle = dlopen(system_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (nullptr == _system_lib_handle) {
        GGMLQNN_LOG_WARN("can not open QNN library %s, error: %s\n", system_lib_path.c_str(), dlerror());
        //re-try with default path of QNN binary runtime lib
        _lib_path = "/data/local/tmp/";
        system_lib_path = _lib_path + "libQnnSystem.so";
        _system_lib_handle = dlopen(system_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (nullptr == _system_lib_handle) {
            GGMLQNN_LOG_WARN("can not open QNN library %s, error: %s\n", system_lib_path.c_str(), dlerror());
            return 1;
        }
    }

    auto * get_providers = reinterpret_cast<_pfn_QnnSystemInterface_getProviders *>(dlsym(
            _system_lib_handle, "QnnSystemInterface_getProviders"));
    if (nullptr == get_providers) {
        GGMLQNN_LOG_WARN("can not load QNN symbol QnnSystemInterface_getProviders: %s\n", dlerror());
        return 2;
    }

    uint32_t num_providers = 0;
    const QnnSystemInterface_t ** provider_list = nullptr;
    error = get_providers(&provider_list, &num_providers);
    if (error != QNN_SUCCESS) {
        GGMLQNN_LOG_WARN("failed to get providers, error %d\n", QNN_GET_ERROR_CODE(error));
        return 3;
    }

    if (num_providers != _required_num_providers) {
        GGMLQNN_LOG_WARN("providers is %d instead of required %d\n", num_providers, _required_num_providers);
        return 4;
    }

    if (nullptr == provider_list) {
        GGMLQNN_LOG_WARN("can not get providers\n");
        return 5;
    }

    QNN_SYSTEM_INTERFACE_VER_TYPE qnn_system_interface;
    bool found_valid_system_interface = false;
    for (size_t idx = 0; idx < num_providers; idx++) {
        if (QNN_SYSTEM_API_VERSION_MAJOR ==
            provider_list[idx]->systemApiVersion.major &&
            QNN_SYSTEM_API_VERSION_MINOR <=
            provider_list[idx]->systemApiVersion.minor) {
            found_valid_system_interface = true;
            qnn_system_interface = provider_list[idx]->QNN_SYSTEM_INTERFACE_VER_NAME;
            break;
        }
    }
    if (!found_valid_system_interface) {
        GGMLQNN_LOG_WARN("unable to find a valid qnn system interface\n");
        return 6;
    } else {
        GGMLQNN_LOG_INFO("find a valid qnn system interface\n");
    }
    set_qnn_raw_system_interface(qnn_system_interface);

    _qnn_interface.set_qnn_system_interface(provider_list[0]);

    _qnn_interface.qnn_system_context_create(&_qnn_system_handle);
    if (nullptr == _qnn_system_handle) {
        GGMLQNN_LOG_WARN("can not create QNN system contenxt\n");
    } else {
        GGMLQNN_LOG_INFO("initialize qnn system successfully\n");
    }

    return 0;
}

int qnn_instance::unload_system() {
    int result = 0;

    if (nullptr == _system_lib_handle) {
        GGMLQNN_LOG_DEBUG("system lib handle is null\n");
        return 1;
    }

    if (nullptr != _qnn_system_handle) {
        result = _qnn_interface.qnn_system_context_free(_qnn_system_handle);
        if (result != QNN_SUCCESS) {
            GGMLQNN_LOG_WARN("failed to free QNN system context\n");
        }
        _qnn_system_handle = nullptr;
    }

    int dlclose_error = dlclose(_system_lib_handle);
    if (dlclose_error != 0) {
        GGMLQNN_LOG_WARN("failed to close QnnSystem library, error %s\n", dlerror());
        return 2;
    }

    _system_lib_handle = nullptr;

    return result;
}

#if GGMLQNN_PRINT_QNN_INTERNAL_LOG
static void ggml_qnn_logcallback(const char * fmt,
                                 QnnLog_Level_t level,
                                 uint64_t timestamp,
                                 va_list argp) {

    static std::mutex log_mutex;
    static unsigned char s_ggml_qnn_logbuf[GGML_QNN_LOGBUF_LEN];

    const char * log_level_desc = "";
    switch (level) {
        case QNN_LOG_LEVEL_ERROR:
            log_level_desc = " ERROR ";
            break;
        case QNN_LOG_LEVEL_WARN:
            log_level_desc = "WARNING";
            break;
        case QNN_LOG_LEVEL_INFO:
            log_level_desc = "  INFO ";
            break;
        case QNN_LOG_LEVEL_DEBUG:
            log_level_desc = " DEBUG ";
            break;
        case QNN_LOG_LEVEL_VERBOSE:
            log_level_desc = "VERBOSE";
            break;
        case QNN_LOG_LEVEL_MAX:
            log_level_desc = "UNKNOWN";
            break;
    }

    double ms = (double) timestamp / 1000000.0;
    {
        std::lock_guard<std::mutex> lock(log_mutex);
        memset(s_ggml_qnn_logbuf, 0, GGML_QNN_LOGBUF_LEN);
        vsnprintf(reinterpret_cast<char *const>(s_ggml_qnn_logbuf), GGML_QNN_LOGBUF_LEN, fmt, argp);
        GGMLQNN_LOG_INFO("%8.1fms [%-7s] %s\n", ms, log_level_desc, s_ggml_qnn_logbuf);
    }
}
#else
static void ggml_qnn_logcallback(const char * fmt,
                                 QnnLog_Level_t level,
                                 uint64_t timestamp,
                                 va_list argp) {
}
#endif

int qnn_instance::qnn_init(const QnnSaver_Config_t ** saver_config) {
    BackendIdType backend_id = QNN_BACKEND_ID_NULL;
    GGMLQNN_LOG_DEBUG("enter qni_init\n");
    const std::lock_guard<std::mutex> lock(_init_mutex);
    if (0 != load_system()) {
        GGMLQNN_LOG_WARN("can not load QNN system lib, pls check why?\n");
        return 1;
    } else {
        GGMLQNN_LOG_DEBUG("load QNN system lib successfully\n");
    }

    std::string bakend_lib_path = _lib_path + _backend_name;
    if (0 == _lib_path_to_backend_id.count(bakend_lib_path)) {
        int is_load_ok = load_backend(bakend_lib_path, saver_config);
        if (0 != is_load_ok) {
            GGMLQNN_LOG_WARN("failed to load QNN backend\n");
            return 2;
        }
    }

    backend_id = _lib_path_to_backend_id[bakend_lib_path];
    if (0 == _loaded_backend.count(backend_id) ||
        0 == _loaded_lib_handle.count(backend_id)) {
        GGMLQNN_LOG_WARN("library %s is loaded but loaded backend count=%zu, loaded lib_handle count=%zu\n",
              bakend_lib_path.c_str(),
              _loaded_backend.count(backend_id),
              _loaded_lib_handle.count(backend_id));
        return 3;
    }
    _qnn_interface.set_qnn_interface(_loaded_backend[backend_id]);
#if 1
    _qnn_interface.qnn_log_create(ggml_qnn_logcallback, _qnn_log_level, &_qnn_log_handle);
#else
    _qnn_raw_interface.logCreate(ggml_qnn_logcallback, _qnn_log_level, &_qnn_log_handle);
#endif
    if (nullptr == _qnn_log_handle) {
        GGMLQNN_LOG_WARN("why failed to initialize qnn log\n"); //NPU backend not work on Qualcomm SoC based low-end phone
        return 4;
    } else {
        GGMLQNN_LOG_DEBUG("initialize qnn log successfully\n");
    }

    std::vector<const QnnBackend_Config_t *> temp_backend_config;
    _qnn_interface.qnn_backend_create(_qnn_log_handle,
                      temp_backend_config.empty() ? nullptr : temp_backend_config.data(),
                      &_qnn_backend_handle);
    if (nullptr == _qnn_backend_handle) {
        GGMLQNN_LOG_WARN("why failed to initialize qnn backend\n");
        return 5;
    } else {
        GGMLQNN_LOG_DEBUG("initialize qnn backend successfully\n");
    }

    if (nullptr != _qnn_raw_interface.propertyHasCapability) {
        auto qnnstatus = _qnn_raw_interface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
        if (QNN_PROPERTY_NOT_SUPPORTED == qnnstatus) {
            GGMLQNN_LOG_WARN("device property is not supported\n");
        }
        if (QNN_PROPERTY_ERROR_UNKNOWN_KEY == qnnstatus) {
            GGMLQNN_LOG_WARN("device property is not known to backend\n");
        }
    }

    auto qnnstatus = _qnn_raw_interface.deviceCreate(
            _qnn_log_handle, nullptr, &_qnn_device_handle);
    if (QNN_SUCCESS != qnnstatus && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnnstatus) {
        GGMLQNN_LOG_WARN("failed to create QNN device\n");
    } else {
        GGMLQNN_LOG_INFO("create device successfully\n");
    }

    if (ggml_qnn_profile_level::profile_off != _profile_level) {
        GGMLQNN_LOG_INFO("profiling turned on; level = %d", _profile_level);
        if (ggml_qnn_profile_level::profile_basic == _profile_level) {
            GGMLQNN_LOG_INFO("basic profiling requested. creating Qnn Profile object\n");
            if (QNN_PROFILE_NO_ERROR != _qnn_raw_interface.profileCreate(
                    _qnn_backend_handle, QNN_PROFILE_LEVEL_BASIC, &_qnn_profile_handle)) {
                GGMLQNN_LOG_WARN("unable to create profile handle in the backend\n");
                return 7;
            } else {
                GGMLQNN_LOG_DEBUG("initialize qnn profile successfully\n");
            }
        } else if (ggml_qnn_profile_level::profile_detail == _profile_level) {
            GGMLQNN_LOG_INFO("detailed profiling requested. Creating Qnn Profile object\n");
            if (QNN_PROFILE_NO_ERROR != _qnn_raw_interface.profileCreate(
                    _qnn_backend_handle, QNN_PROFILE_LEVEL_DETAILED, &_qnn_profile_handle)) {
                GGMLQNN_LOG_WARN("unable to create profile handle in the backend\n");
                return 7;
            } else {
                GGMLQNN_LOG_DEBUG("initialize qnn profile successfully\n");
            }
        }
    }

    _rpc_lib_handle = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    if (nullptr == _rpc_lib_handle) {
        GGMLQNN_LOG_WARN("failed to load qualcomm's rpc lib, error:%s\n", dlerror());
        return 9;
    } else {
        GGMLQNN_LOG_DEBUG("load rpcmem lib successfully\n");
        set_rpcmem_initialized(true);
    }
    _pfn_rpc_mem_init   = reinterpret_cast<pfn_rpc_mem_init>(dlsym(_rpc_lib_handle, "rpcmem_init"));
    _pfn_rpc_mem_deinit = reinterpret_cast<pfn_rpc_mem_deinit>(dlsym(_rpc_lib_handle, "rpcmem_deinit"));
    _pfn_rpc_mem_alloc  = reinterpret_cast<pfn_rpc_mem_alloc>(dlsym(_rpc_lib_handle,"rpcmem_alloc"));
    _pfn_rpc_mem_free   = reinterpret_cast<pfn_rpc_mem_free>(dlsym(_rpc_lib_handle, "rpcmem_free"));
    _pfn_rpc_mem_to_fd  = reinterpret_cast<pfn_rpc_mem_to_fd>(dlsym(_rpc_lib_handle,"rpcmem_to_fd"));
    if (nullptr == _pfn_rpc_mem_alloc || nullptr == _pfn_rpc_mem_free
        || nullptr == _pfn_rpc_mem_to_fd) {
        GGMLQNN_LOG_WARN("unable to access symbols in QNN RPC lib. dlerror(): %s", dlerror());
        dlclose(_rpc_lib_handle);
        return 10;
    }

    if (nullptr != _pfn_rpc_mem_init) // make Qualcomm's SoC based low-end phone happy
        _pfn_rpc_mem_init();

    std::vector<const QnnContext_Config_t *> temp_context_config;
    _qnn_interface.qnn_context_create(_qnn_backend_handle, _qnn_device_handle,
                               temp_context_config.empty() ? nullptr : temp_context_config.data(),
                               &_qnn_context_handle);
    if (nullptr == _qnn_context_handle) {
        GGMLQNN_LOG_WARN("why failed to initialize qnn context, error:%s\n", strerror(errno));
        return 8;
    } else {
        GGMLQNN_LOG_DEBUG("initialize qnn context successfully\n");
    }

    if (_backend_name.find("Htp") != std::variant_npos) {
        const QnnDevice_PlatformInfo_t * p_info = nullptr;
        _qnn_raw_interface.deviceGetPlatformInfo(nullptr, &p_info);
        GGMLQNN_LOG_INFO("device counts %d", p_info->v1.numHwDevices);
        QnnDevice_HardwareDeviceInfo_t * infos = p_info->v1.hwDevices;
        for (int i = 0; i < p_info->v1.numHwDevices; i++) {
            GGMLQNN_LOG_INFO("deviceID:%d, deviceType:%d, numCores %d", infos[i].v1.deviceId,
                         infos[i].v1.deviceType, infos[i].v1.numCores);
            QnnDevice_DeviceInfoExtension_t devinfo = infos[i].v1.deviceInfoExtension;
            QnnHtpDevice_OnChipDeviceInfoExtension_t chipinfo = devinfo->onChipDevice;
            QnnHtpDevice_Arch_t htp_arch = chipinfo.arch;
            GGMLQNN_LOG_INFO("htp_type:%d(%s)", devinfo->devType,
                             (devinfo->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP) ? "QNN_HTP_DEVICE_TYPE_ON_CHIP" : "QNN_HTP_DEVICE_TYPE_UNKNOWN");
            GGMLQNN_LOG_INFO("qualcomm soc_model:%d(%s), htp_arch:%d(%s), vtcm_size:%d MB", \
                             chipinfo.socModel, qnn_get_socmodel_desc(chipinfo.socModel), \
                             htp_arch, qnn_get_htparch_desc(htp_arch), chipinfo.vtcmSize);
            struct qcom_socinfo * socinfo = qnn_get_socinfo_from_socmodel(chipinfo.socModel);
            g_qnn_mgr[QNN_BACKEND_NPU].socinfo = { chipinfo.socModel, htp_arch, chipinfo.vtcmSize };
            if (nullptr != socinfo) {
                memcpy(g_qnn_mgr[QNN_BACKEND_NPU].socinfo.soc_desc, socinfo->soc_desc, sizeof(socinfo->soc_desc));
                GGMLQNN_LOG_INFO("soc info:%s", socinfo->soc_desc);
            } else {
                memcpy(g_qnn_mgr[QNN_BACKEND_NPU].socinfo.soc_desc, "unknown", 7);
                GGMLQNN_LOG_INFO("soc info:unknown");
            }
        }
        _qnn_raw_interface.deviceFreePlatformInfo(nullptr, p_info);

        probe_device_meminfo();

        if (0 != init_htp_perfinfra()) {
            GGMLQNN_LOG_WARN("initialize HTP performance failure");
        }
        if (0 != set_rpc_polling()) {
            GGMLQNN_LOG_WARN("set RPC polling failure");
        }
        if (0 != set_high_performance_mode()) {
            GGMLQNN_LOG_WARN("set HTP high performance mode failure");
        }
    }

    GGMLQNN_LOG_DEBUG("leave qni_init\n");

    return 0;
}

int qnn_instance::qnn_finalize() {
    int ret_status = 0;
    Qnn_ErrorHandle_t error = QNN_SUCCESS;

    GGMLQNN_LOG_DEBUG("enter %s\n", __func__);
    //TODO:should be removed in the future
    reset_idx();

    free_rpcmem();
    unregister_rpcmem();

    if (nullptr != _pfn_rpc_mem_deinit)
        _pfn_rpc_mem_deinit();

    if (dlclose(_rpc_lib_handle) != 0) {
        GGMLQNN_LOG_WARN("failed to unload qualcomm's rpc lib, error:%s\n", dlerror());
    } else {
        GGMLQNN_LOG_DEBUG("succeed to close rpcmem lib\n");
    }

    if (nullptr != _qnn_context_handle) {
        error = _qnn_interface.qnn_context_free(_qnn_context_handle, _qnn_profile_handle);
        if (error != QNN_SUCCESS) {
            GGMLQNN_LOG_WARN("failed to free QNN context_handle: ID %u, error %d\n",
                  _qnn_interface.get_backend_id(), QNN_GET_ERROR_CODE(error));

        }
        _qnn_context_handle = nullptr;
    }

    if (nullptr != _qnn_profile_handle) {
        error = _qnn_interface.qnn_profile_free(_qnn_profile_handle);
        if (error != QNN_SUCCESS) {
            GGMLQNN_LOG_WARN("failed to free QNN profile_handle: ID %u, error %d\n",
                  _qnn_interface.get_backend_id(), QNN_GET_ERROR_CODE(error));

        }
        _qnn_profile_handle = nullptr;
    }

    if (nullptr != _qnn_device_handle) {
        error = _qnn_interface.qnn_device_free(_qnn_device_handle);
        if (error != QNN_SUCCESS) {
            GGMLQNN_LOG_WARN("failed to free QNN device_handle: ID %u, error %d\n",
                  _qnn_interface.get_backend_id(), QNN_GET_ERROR_CODE(error));

        }
        _qnn_device_handle = nullptr;
    }

    if (nullptr != _qnn_backend_handle) {
        error = _qnn_interface.qnn_backend_free(_qnn_backend_handle);
        if (error != QNN_SUCCESS) {
            GGMLQNN_LOG_WARN("failed to free QNN backend_handle: ID %u, error %d\n",
                  _qnn_interface.get_backend_id(), QNN_GET_ERROR_CODE(error));
        }
        _qnn_backend_handle = nullptr;

    }

    if (nullptr != _qnn_log_handle) {
        error = _qnn_interface.qnn_log_free(_qnn_log_handle);
        if (error != QNN_SUCCESS) {
            GGMLQNN_LOG_WARN("failed to free QNN log_handle: ID %u, error %d\n",
                  _qnn_interface.get_backend_id(), QNN_GET_ERROR_CODE(error));
        }
        _qnn_log_handle = nullptr;
    }

    unload_backend();

    unload_system();
    GGMLQNN_LOG_DEBUG("leave %s\n", __func__);

    return ret_status;
}

int qnn_instance::init_qnn_graph(const std::string & graph_name, QNNBackend device, size_t vtcm_size_in_mb, size_t hvx_threads) {
    _graph_name = graph_name;
    _device_id = device;

    GGMLQNN_LOG_DEBUG("[%s][%s]created", ggml_backend_qnn_get_devname(device), graph_name.c_str());

    Qnn_ErrorHandle_t error = QNN_SUCCESS;
    Qnn_GraphHandle_t graph_handle = nullptr;
    if (device == QNN_BACKEND_NPU) {
        QnnHtpGraph_CustomConfig_t hvx_config;
        hvx_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
        hvx_config.numHvxThreads = hvx_threads;
        QnnGraph_Config_t graph_hvx_config;
        graph_hvx_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        graph_hvx_config.customConfig = &hvx_config;

        QnnHtpGraph_CustomConfig_t dlbc_config;
        dlbc_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
        dlbc_config.optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC;
        dlbc_config.optimizationOption.floatValue = 1.0; // set to 0.0 to turn off DLBC
        QnnGraph_Config_t graph_dlbc_config;
        graph_dlbc_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        graph_dlbc_config.customConfig = &dlbc_config;

        QnnHtpGraph_CustomConfig_t opt_config;
        opt_config.optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
        opt_config.optimizationOption.floatValue = 1; // 1 / 3
        QnnGraph_Config_t graph_opt_config;
        graph_opt_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        graph_opt_config.customConfig = &opt_config;

        QnnHtpGraph_CustomConfig_t vtcm_config;
        vtcm_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
        vtcm_config.vtcmSizeInMB = vtcm_size_in_mb;
        QnnGraph_Config_t graph_vtcm_config;
        graph_vtcm_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        graph_vtcm_config.customConfig = &vtcm_config;

        const QnnGraph_Config_t * graph_configs[] = {&graph_hvx_config, &graph_dlbc_config, &graph_vtcm_config,
                                                    &graph_opt_config, nullptr};
        error = _qnn_interface.qnn_graph_create(_qnn_context_handle, graph_name.c_str(), graph_configs, &graph_handle);
    } else {
        error = _qnn_interface.qnn_graph_create(_qnn_context_handle, graph_name.c_str(), nullptr, &graph_handle);
    }

    if (error != QNN_SUCCESS) {
        GGMLQNN_LOG_ERROR("[%s][%s]failed to create qnn graph, error: %s",
                      ggml_backend_qnn_get_devname(device), graph_name.c_str(),
                      qnn_get_error_string(error));
        return error;
    }

    GGMLQNN_LOG_INFO("[%s]create graph %s succeed", ggml_backend_qnn_get_devname(device), graph_name.c_str());
    _qnn_graph_handle = graph_handle;
    return QNN_SUCCESS;
}

int qnn_instance::init_qnn_graph(const char * graph_name, bool debug, uint8_t do_node_validation,
                                   const QnnGraph_Config_t ** graph_configs) {
    int result = 0;

    if (nullptr == graph_name) {
        GGMLQNN_LOG_WARN("graph name is null\n");
        return 1;
    }

    if (!_graph_name.empty()) {
        GGMLQNN_LOG_WARN("qnn model for graph %s already initialized\n", graph_name);
        return 2;
    }

    if (!do_node_validation) {
        GGMLQNN_LOG_WARN("node validation disabled, backend will not perform op validation prior to adding node\n");
    }

    _graph_name = graph_name;
    _debug_tensor = debug;
    _do_node_validations = do_node_validation;

    result = _qnn_raw_interface.graphCreate(_qnn_context_handle,
                                            graph_name,
                                            graph_configs,
                                            &_qnn_graph_handle);
    if (result != QNN_GRAPH_NO_ERROR || nullptr == _qnn_graph_handle) {
        GGMLQNN_LOG_WARN("failed to create graph in qnn context\n");
        return 3;
    } else {
        GGMLQNN_LOG_INFO("succeed to create graph %s, %p\n", graph_name, _qnn_graph_handle);
    }

    return 0;
}

int qnn_instance::finalize_qnn_graph() {
    if (nullptr != _qnn_graph_handle) {
        if (_qnn_raw_interface.graphFinalize(_qnn_graph_handle,
                                             _qnn_profile_handle, nullptr)
                                             != QNN_GRAPH_NO_ERROR) {
            GGMLQNN_LOG_WARN("finalizing graph failure\n");
            return 1;
        }
    } else {
        GGMLQNN_LOG_DEBUG("qnn graph handle is null\n");
    }

    return 0;
}

static uint8_t * create_rpc_buffer(qnn_instance * instance, const ggml_tensor * ggml_tensor, Qnn_Tensor_t * qnn_tensor, bool b_copydata) {
    if (nullptr == instance || nullptr == ggml_tensor || nullptr == qnn_tensor) {
        GGMLQNN_LOG_WARN("invalid params\n");
        return nullptr;
    }

    uint8_t * qnn_rpcbuffer = static_cast<uint8_t *>(instance->alloc_rpcmem(ggml_nbytes(ggml_tensor), 4));
    if (nullptr == qnn_rpcbuffer) {
        GGMLQNN_LOG_WARN("alloc rpcmem failure, %s\n", strerror(errno));
        return nullptr;
    } else {
        GGMLQNN_LOG_DEBUG("alloc rpcmem %p successfully\n", qnn_rpcbuffer);
    }
    if (b_copydata)
        memcpy(qnn_rpcbuffer, ggml_tensor->data, ggml_nbytes(ggml_tensor));
    instance->register_rpcmem(qnn_rpcbuffer, qnn_tensor);
    return qnn_rpcbuffer;
}

static void print_tensors_info(const char * func_name, ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    //skip sanity check of params
    if (nullptr != func_name && nullptr != ctx) {
        GGMLQNN_LOG_DEBUG("call %s in dev %s\n", func_name, ctx->name);
    }
    GGMLQNN_LOG_DEBUG("%-6s: type = %i (%s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi, %5zi)",
                      src0->name,
                      src0->type, ggml_type_name(src0->type), src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                      src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    GGMLQNN_LOG_DEBUG("%-6s: type = %i (%s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi, %5zi)",
                      src1->name,
                      src1->type, ggml_type_name(src1->type), src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                      src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
    GGMLQNN_LOG_DEBUG("%-6s: type = %i (%s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi, %5zi)",
                      dst->name,
                      dst->type, ggml_type_name(dst->type), dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                      dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
    GGMLQNN_LOG_DEBUG("\n");
}

static void dump_op_info(const struct ggml_tensor * tensor) {
    //skip sanity check of params
    const struct ggml_tensor * src0 = tensor->src[0];
    struct ggml_tensor       * src1 = tensor->src[1];
    struct ggml_tensor       * dst  = const_cast<ggml_tensor *>(tensor);
    GGMLQNN_LOG_DEBUG("op name:%s, tensor type:%s", ggml_op_name(tensor->op), ggml_type_name(tensor->type));
    print_tensors_info(nullptr, nullptr, src0, src1, dst);
}

//TODO: currently only support offloading 2D matrix to QNN backend
static void get_qnn_dimensions_from_ggml_dimensions(uint32_t * qnn_dimensions, uint32_t * ggml_dimensions, uint32_t rank) {
    if (rank > GGML_MAX_DIMS) {
        GGMLQNN_LOG_WARN("invalid params");
        return;
    }
    if (nullptr == qnn_dimensions || nullptr == ggml_dimensions) {
        GGMLQNN_LOG_WARN("invalid params");
        return;
    }
    for (size_t idx = 0; idx < GGML_MAX_DIMS; idx++)
        qnn_dimensions[idx] = ggml_dimensions[idx];

    if (rank >= 2) {
        qnn_dimensions[rank - 1] = ggml_dimensions[rank - 2];
        qnn_dimensions[rank - 2] = ggml_dimensions[rank - 1];
    }
}

// =================================================================================================
//  section-6: implementation of ggml-qnn backend
// =================================================================================================
//TODO: refine this function as it is a performance hotspot/bottleneck function
static bool ggml_qnn_can_handle_op(const struct ggml_tensor * tensor) {
    if (tensor->op == GGML_OP_NONE) {
        return true;
    }
    if (ggml_is_empty(tensor) || tensor->op == GGML_OP_RESHAPE
    || tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_VIEW
    || tensor->op == GGML_OP_PERMUTE) {
        return false;
    }

    //TODO: support other op
    bool supported_op = ((tensor->op == GGML_OP_ADD) || (tensor->op == GGML_OP_MUL_MAT)
            || (tensor->op == GGML_OP_MUL));
    if (!supported_op) {
        return false;
    }

    struct ggml_tensor * src0 = tensor->src[0];
    struct ggml_tensor * src1 = tensor->src[1];

    const int64_t ne00 = tensor->src[0]->ne[0];
    const int64_t ne01 = tensor->src[0]->ne[1];

    const int64_t ne10 = tensor->src[1]->ne[0];
    const int64_t ne11 = tensor->src[1]->ne[1];

    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];

    const uint32_t src0_rank = ggml_get_tensor_rank(src0);
    const uint32_t src1_rank = ggml_get_tensor_rank(src1);

    if (tensor->op == GGML_OP_ADD) {
        //dump_tensors_info(tensor);
        if (!ggml_are_same_shape(src0, src1)) {
            return false;
        }
        if (ne00 < 32)
            return false;
        return (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)
               && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
    }

    if (tensor->op == GGML_OP_MUL_MAT) {
        dump_op_info(tensor);
        if (src0_rank != src1_rank) // make QNN SDK happy
            return false;
        if (src0_rank < 2) // make QNN SDK happy
            return false;
        if (src0_rank > 3) //TODO: 4D matrix
            return false;
        if ((src1->ne[2] != src0->ne[2]) || (src1->ne[3] != src0->ne[3])) // make QNN SDK happy
            return false;

        //TODO: support more data type in func ggml_qnn_mul_mat(...)
        //src0: q4_0, q6_k, ...
        //src1: f32
        //dst : f32
        return  (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)
                && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16)
                && (src0->type == src1->type) && (src0->type == tensor->type);
    }

    if (tensor->op == GGML_OP_MUL) {
        //dump_tensors_info(tensor);
        if ((src0_rank != 2) || (src1_rank != 2)) //TODO: 3D and 4D matrix
            return false;
        return  (src0->type == GGML_TYPE_F32)
                && (src1->type == GGML_TYPE_F32)
                && (tensor->type == src1->type);
    }

    return  (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)
            && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16)
            && (src0->type == src1->type) && (src0->type == tensor->type);
}

/*
 * provide a general skeleton to offload ggml op to QNN backend: a single node contains 2 input
 * tensor and 1 output tensor
*/
static void ggml_qnn_general_node(ggml_backend_t backend, ggml_tensor * op) {
    Qnn_ErrorHandle_t error                     = QNN_SUCCESS;
    enum ggml_status result                     = GGML_STATUS_SUCCESS;
    bool graph_initialized                      = false;
    qnn_instance * instance                     = nullptr;
    ggml_backend_qnn_context * ctx              = (ggml_backend_qnn_context *)backend->context;
    Qnn_GraphHandle_t graph_handle              = nullptr;
    Qnn_Tensor_t * p_tensor0                    = nullptr;
    Qnn_Tensor_t * p_tensor1                    = nullptr;
    Qnn_Tensor_t * p_tensor2                    = nullptr;
    Qnn_Param_t qnn_params[]                    = {};
    const ggml_tensor * src0                    = op->src[0];
    const ggml_tensor * src1                    = op->src[1];
    ggml_tensor * dst                           = op;

    GGMLQNN_CHECK_PARAMS(ctx, src0, src1, dst);
    instance                                    = ctx->instance;
    QNN_INTERFACE_VER_TYPE qnn_raw_interface    = ctx->raw_interface;

    size_t qnn_op_index = get_qnn_op_index(op);
    GGML_ASSERT(qnn_op_index < std::size(k_op_caps));
    const char * qnn_op_name = k_op_caps[qnn_op_index].qnn_op_name;
    std::string ggml_op_name_string = std::string("ggml_") + ggml_op_name(op->op);
    const char * ggml_op_name = ggml_op_name_string.c_str();

    qnn_perf op_perf                            = qnn_perf(ggml_op_name);
    op_perf.start();

    std::string graph_name;
    get_graph_key_from_op(op, graph_name);
    if (instance->_qnn_graph_map.find(graph_name) != instance->_qnn_graph_map.end()) {
        graph_initialized = true;
        qnn_res_t & graph_item = instance->_qnn_graph_map[graph_name];
        graph_handle = std::get<0>(graph_item);
        qnn_tensors_t & tensor = std::get<1>(graph_item);
        p_tensor0     = tensor[0];
        p_tensor1     = tensor[1];
        p_tensor2     = tensor[2];
    } else {
        p_tensor0 = ggml_qnn_create_compute_tensor(src0);
        p_tensor1 = ggml_qnn_create_compute_tensor(src1);
        p_tensor2 = ggml_qnn_create_compute_tensor(dst);
    }
    print_tensors_info(__func__, ctx, src0, src1, dst);

    //ensure QNN tensor has correct tensor type
    QNN_VER_PTR(*p_tensor0)->type = QNN_TENSOR_TYPE_APP_WRITE;
    QNN_VER_PTR(*p_tensor1)->type = QNN_TENSOR_TYPE_APP_WRITE;
    QNN_VER_PTR(*p_tensor2)->type = QNN_TENSOR_TYPE_APP_READ;

    //save the original dimensions of qnn tensors
    uint32_t * tensor_0_dimensions = QNN_VER_PTR(*p_tensor0)->dimensions;
    uint32_t * tensor_1_dimensions = QNN_VER_PTR(*p_tensor1)->dimensions;
    uint32_t * tensor_2_dimensions = QNN_VER_PTR(*p_tensor2)->dimensions;

    bool enable_npu_rpc = instance->enable_qnn_rpc() && ctx->device == QNN_BACKEND_NPU;

    if (!graph_initialized) {
        GGMLQNN_LOG_DEBUG("graph name %s", graph_name.c_str());
        error = instance->init_qnn_graph(graph_name, static_cast<QNNBackend>(ctx->device), 8);
        if (QNN_SUCCESS != error) {
            GGMLQNN_LOG_INFO("can't create qnn graph handle with graph name %s, error = %d\n", graph_name.c_str(), error);
            return;
        }
        graph_handle = instance->get_qnn_graph_handle();

        if (enable_npu_rpc) {
            QNN_VER_PTR(*p_tensor0)->memType = QNN_TENSORMEMTYPE_MEMHANDLE;
            QNN_VER_PTR(*p_tensor0)->clientBuf = {.data=nullptr, .dataSize=0};

            QNN_VER_PTR(*p_tensor1)->memType = QNN_TENSORMEMTYPE_MEMHANDLE;
            QNN_VER_PTR(*p_tensor1)->clientBuf = {.data=nullptr, .dataSize=0};

            QNN_VER_PTR(*p_tensor2)->memType = QNN_TENSORMEMTYPE_MEMHANDLE;
            QNN_VER_PTR(*p_tensor2)->clientBuf = {.data=nullptr, .dataSize=0};
        }

        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor0));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor1));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor2));

        if (enable_npu_rpc) {
            uint8_t * qnn_rpcbuffer_0 = create_rpc_buffer(instance, src0, p_tensor0, true);
            uint8_t * qnn_rpcbuffer_1 = create_rpc_buffer(instance, src1, p_tensor1, true);
            uint8_t * qnn_rpcbuffer_2 = create_rpc_buffer(instance, dst, p_tensor2, false);
            if (nullptr == qnn_rpcbuffer_0 || nullptr == qnn_rpcbuffer_1 || nullptr == qnn_rpcbuffer_2) {
                GGMLQNN_LOG_INFO("create rpc buffer failure\n");
                //TODO: potential memory leak although it shouldn't happen
                return;
            }
        } else {
            QNN_VER_PTR(*p_tensor0)->clientBuf = {src0->data, ggml_get_tensor_data_size(src0)};
            QNN_VER_PTR(*p_tensor1)->clientBuf = {src1->data, ggml_get_tensor_data_size(src1)};
            QNN_VER_PTR(*p_tensor2)->clientBuf = {dst->data, ggml_get_tensor_data_size(dst)};
        }

        Qnn_Tensor_t tensor_inputs[] = {
                *p_tensor0,
                *p_tensor1
        };
        Qnn_Tensor_t tensor_outputs[] = {
                *p_tensor2
        };
        Qnn_OpConfig_t op_config = {
                QNN_OPCONFIG_VERSION_1, .v1 = {
                        ggml_op_name,
                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                        qnn_op_name,
                        0,
                        qnn_params,
                        2,
                        tensor_inputs,
                        1,
                        tensor_outputs
                }
        };
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, op_config));
        CHECK_QNN_API(error, qnn_raw_interface.graphFinalize(graph_handle, nullptr, nullptr));
        CHECK_QNN_API(error, qnn_raw_interface.graphExecute(graph_handle,
                                                            tensor_inputs, 2,
                                                            tensor_outputs, 1,
                                                            nullptr, nullptr));

        if (enable_npu_rpc) {
            uint8_t * qnn_rpcbuffer = static_cast<uint8_t *>(instance->get_rpcmem_from_memhandle(QNN_VER_PTR(*p_tensor2)->memHandle));
            GGMLQNN_LOG_INFO("qnn_rpcbuffer = %p\n", qnn_rpcbuffer);
            if (nullptr != qnn_rpcbuffer) {
                memcpy(dst->data, qnn_rpcbuffer, ggml_nbytes(dst));
            }
        }

        qnn_tensors_t ggml_op_add_tensors;
        ggml_op_add_tensors.reserve(3);
        ggml_op_add_tensors.push_back(p_tensor0);
        ggml_op_add_tensors.push_back(p_tensor1);
        ggml_op_add_tensors.push_back(p_tensor2);

        auto  graph_item = std::make_tuple(graph_handle, ggml_op_add_tensors);
        instance->_qnn_graph_map[graph_name] = graph_item;

    } else {
        Qnn_DataType_t src0_qnn_type    = QNN_DATATYPE_FLOAT_32;
        Qnn_DataType_t src1_qnn_type    = QNN_DATATYPE_FLOAT_32;
        Qnn_DataType_t dst_qnn_type     = QNN_DATATYPE_FLOAT_32;

        src0_qnn_type                   = qnn_datatype_from_ggml_datatype(src0->type);
        src1_qnn_type                   = qnn_datatype_from_ggml_datatype(src1->type);
        dst_qnn_type                    = qnn_datatype_from_ggml_datatype(dst->type);

        uint32_t dimensions_input_0[] = {(uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                         (uint32_t) src0->ne[2], (uint32_t) src0->ne[3]};
        uint32_t dimensions_input_1[] = {(uint32_t) src1->ne[0], (uint32_t) src1->ne[1],
                                         (uint32_t) src1->ne[2], (uint32_t) src1->ne[3]};
        uint32_t dimensions_output[]  = {(uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
                                         (uint32_t) dst->ne[2], (uint32_t) dst->ne[3]};

        QNN_VER_PTR(*p_tensor0)->dimensions  = dimensions_input_0;
        QNN_VER_PTR(*p_tensor0)->rank        = ggml_get_tensor_rank(src0);
        QNN_VER_PTR(*p_tensor0)->dataType    = src0_qnn_type;

        QNN_VER_PTR(*p_tensor1)->dimensions  = dimensions_input_1;
        QNN_VER_PTR(*p_tensor1)->rank        = ggml_get_tensor_rank(src1);
        QNN_VER_PTR(*p_tensor1)->dataType    = src1_qnn_type;

        QNN_VER_PTR(*p_tensor2)->dimensions  = dimensions_output;
        QNN_VER_PTR(*p_tensor2)->rank        = ggml_get_tensor_rank(dst);
        QNN_VER_PTR(*p_tensor2)->dataType    = dst_qnn_type;

        if (enable_npu_rpc) {
            //TODO: NPU RPC feature will failed with test-backend-ops
            uint8_t * qnn_buffer_0 = static_cast<uint8_t *>(instance->get_rpcmem_from_memhandle(QNN_VER_PTR(*p_tensor0)->memHandle));
            GGMLQNN_LOG_INFO("qnn_rpcbuffer_0 = %p\n", qnn_buffer_0);
            if (nullptr != qnn_buffer_0) {
                memcpy(qnn_buffer_0, src0->data, ggml_nbytes(src0));
            }

            uint8_t * qnn_buffer_1 = static_cast<uint8_t *>(instance->get_rpcmem_from_memhandle(QNN_VER_PTR(*p_tensor1)->memHandle));
            GGMLQNN_LOG_INFO("qnn_rpcbuffer_1 = %p\n", qnn_buffer_1);
            if (nullptr != qnn_buffer_1) {
                memcpy(qnn_buffer_1, src1->data, ggml_nbytes(src1));
            }
        } else {
            QNN_VER_PTR(*p_tensor0)->clientBuf = {src0->data, ggml_get_tensor_data_size(src0)};
            QNN_VER_PTR(*p_tensor1)->clientBuf = {src1->data, ggml_get_tensor_data_size(src1)};
            QNN_VER_PTR(*p_tensor2)->clientBuf = {dst->data, ggml_get_tensor_data_size(dst)};
        }

        Qnn_Tensor_t tensor_inputs[] = {
                *p_tensor0,
                *p_tensor1
        };
        Qnn_Tensor_t tensor_outputs[] = {
                *p_tensor2
        };
        CHECK_QNN_API(error, qnn_raw_interface.graphExecute(graph_handle,
                                                            tensor_inputs, 2,
                                                            tensor_outputs, 1,
                                                            nullptr, nullptr));

        if (enable_npu_rpc) {
            //TODO:NPU RPC feature will failed with test-backend-ops
            uint8_t * qnn_buffer_2 = static_cast<uint8_t *>(instance->get_rpcmem_from_memhandle(QNN_VER_PTR(*p_tensor2)->memHandle));
            if (nullptr != qnn_buffer_2) {
                memcpy(dst->data, qnn_buffer_2, ggml_nbytes(dst));
            }
        }
    }

    // restore the original dimensions of qnn tensors to avoid memory leak in func free_qnn_tensor
    QNN_VER_PTR(*p_tensor0)->dimensions = tensor_0_dimensions;
    QNN_VER_PTR(*p_tensor1)->dimensions = tensor_1_dimensions;
    QNN_VER_PTR(*p_tensor2)->dimensions = tensor_2_dimensions;
    op_perf.info();
}

/*
 * the logic of ggml_qnn_mul_mat is similar to ggml_qnn_general_node but much more complicated
 * than ggml_qnn_general_node.
 * matrix transpose and type trait are required for offload mulmat to QNN backend,
 * so it's a standalone function. accordingly, this is another typical skeleton for offload other
 * ggml ops to QNN backend
 *
 * MUL_MAT take most of the compute time (about 95%).so to speed up llama inference, should focus on MUL_MAT.
 *
 * have three kinds of MUL_MAT to compute:
 * mul_mat_f32:     both src0 and src1 are F32, this will be naturally handled in QNN backend
 * mul_mat_f16_f32: src0 is F16 and src1 is F32, f16 in src0 -> f32 in src0', then src0' * src1
 * mul_mat_q_f32:   src0 is quantized (Q4_0, Q4_1, ...) and src1 is F32, src0 -> f32 in src0', then src0' * src1
*/
static void ggml_qnn_mul_mat(ggml_backend_t backend, ggml_tensor * op) {
    Qnn_ErrorHandle_t error                     = QNN_SUCCESS;
    bool graph_initialized                      = false;
    qnn_perf op_perf                            = qnn_perf("ggml_qnn_mul_mat");
    qnn_instance * instance                     = nullptr;
    ggml_backend_qnn_context * ctx              = (ggml_backend_qnn_context *) backend->context;
    Qnn_GraphHandle_t graph_handle              = nullptr;
    Qnn_Tensor_t * p_tensor0                    = nullptr;
    Qnn_Tensor_t * p_tensor1                    = nullptr;
    Qnn_Tensor_t * p_tensor2                    = nullptr;
    Qnn_Tensor_t * p_param_tensor               = nullptr;
    Qnn_Tensor_t * p_tensor2_transpose          = nullptr;
    const ggml_tensor * src0                    = op->src[0];
    const ggml_tensor * src1                    = op->src[1];
    ggml_tensor       * dst                     = op;

    GGMLQNN_CHECK_PARAMS(ctx, src0, src1, dst);
    instance                                    = ctx->instance;
    QNN_INTERFACE_VER_TYPE qnn_raw_interface    = ctx->raw_interface;
    op_perf.start();

    uint32_t src0_rank = ggml_get_tensor_rank(src0);
    uint32_t src1_rank = ggml_get_tensor_rank(src1);
    GGML_ASSERT(src0_rank == src1_rank);
    GGML_ASSERT(src0_rank >= 2); //QNN SDK's limitation

    std::string graph_name;
    get_graph_key_from_op(op, graph_name);
    if (instance->_qnn_graph_map.find(graph_name) != instance->_qnn_graph_map.end()) {
        graph_initialized       = true;
        qnn_res_t & graph_item  = instance->_qnn_graph_map[graph_name];
        graph_handle            = std::get<0>(graph_item);
        qnn_tensors_t & tensors = std::get<1>(graph_item);
        p_tensor0               = tensors[0];
        p_tensor1               = tensors[1];
        p_tensor2               = tensors[2];
        p_param_tensor          = tensors[3];
        p_tensor2_transpose     = tensors[4];
    } else {
        p_tensor0 = ggml_qnn_create_general_tensor(src0, nullptr, QNN_TENSOR_TYPE_APP_WRITE,QNN_DATATYPE_FLOAT_32, src0_rank, nullptr, nullptr, 0);
        p_tensor1 = ggml_qnn_create_general_tensor(src1, nullptr, QNN_TENSOR_TYPE_APP_WRITE,QNN_DATATYPE_FLOAT_32, src0_rank, nullptr, nullptr, 0);
        p_tensor2 = ggml_qnn_create_general_tensor(dst, nullptr, QNN_TENSOR_TYPE_APP_READ,QNN_DATATYPE_FLOAT_32, src0_rank, nullptr, nullptr, 0);
    }

    print_tensors_info(__func__, ctx, src0, src1, dst);

    //ensure QNN tensor has correct tensor type
    QNN_VER_PTR(*p_tensor0)->type = QNN_TENSOR_TYPE_APP_WRITE;
    QNN_VER_PTR(*p_tensor1)->type = QNN_TENSOR_TYPE_APP_WRITE;
    QNN_VER_PTR(*p_tensor2)->type = QNN_TENSOR_TYPE_APP_READ;

    //save the original dimensions of qnn tensors
    uint32_t * tensor_0_dimensions = QNN_VER_PTR(*p_tensor0)->dimensions;
    uint32_t * tensor_1_dimensions = QNN_VER_PTR(*p_tensor1)->dimensions;
    uint32_t * tensor_2_dimensions = QNN_VER_PTR(*p_tensor2)->dimensions;

    if (!graph_initialized) {
        GGMLQNN_LOG_DEBUG("graph name %s", graph_name.c_str());
        /*
         there are two key-points in properly handling how to offload mulmat to the QNN backend in ggml-qnn
         1. transpose
            a 3x2 f32 matrix which means 3 rows and 2 columns. in ggml, it could be created from:
            struct ggml_tensor* matrix = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
            which like this:
            +---+---+
            | 0 | 1 |
            +---+---+
            | 2 | 3 |
            +---+---+
            | 4 | 5 |
            +---+---+
            with
                ne[0] = 2
                ne[1] = 3
            there are different dimension order between ggml tensor and qnn tensor

          2. QNN's MatMul can only support input tensors with rank >= 2

        there is gap between ggml mulmat and QNN mulmat,we need to perform a transpose operation when offloading mulmat to QNN backend.
        */

        //step-1: create qnn graph
        error = qnn_raw_interface.graphCreate(instance->get_qnn_context_handle(),
                                              graph_name.c_str(), nullptr, &graph_handle);
        if (QNN_SUCCESS != error) {
            GGMLQNN_LOG_INFO("can't create qnn graph handle with graph name %s, error = %d\n", graph_name.c_str(), error);
            return;
        }
        //step-2: create param tensor for mulmat of 2d matrix
        const uint32_t param_tensor_data[GGML_MAX_DIMS][GGML_MAX_DIMS] = {
                {0},
                {1, 0},
                {0, 2, 1},
                {0, 1, 3, 2},
        };
        uint32_t param_tensor_dims[1]   = {src0_rank};
        p_param_tensor = ggml_qnn_create_general_tensor(nullptr, "param", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32,
                                                        1, param_tensor_dims,
                                                        (void *) (param_tensor_data[src0_rank - 1]), src0_rank * sizeof(uint32_t));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_param_tensor));

        //step-3: create compute tensor from ggml tensor
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor0));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor1));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor2));

        QNN_VER_PTR(*p_tensor0)->clientBuf = {src0->data, ggml_get_tensor_data_size(src0)};
        QNN_VER_PTR(*p_tensor1)->clientBuf = {src1->data, ggml_get_tensor_data_size(src1)};
        QNN_VER_PTR(*p_tensor2)->clientBuf = {dst->data, ggml_get_tensor_data_size(dst)};

        //step-4: create a transpose tensor
        uint32_t tensor2_transpose_dims[GGML_MAX_DIMS] = {};
        p_tensor2_transpose = ggml_qnn_create_general_tensor(dst, "transpose", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, src0_rank, nullptr, nullptr, 0);
        get_qnn_dimensions_from_ggml_dimensions(tensor2_transpose_dims, tensor_2_dimensions, ggml_get_tensor_rank(dst));
        //save pointer because the dimensions of tensor p_tensor2_transpose will be changed later
        uint32_t * tensor2_dimensions_transpose = QNN_VER_PTR(*p_tensor2_transpose)->dimensions;
        //update dimensions of tensor p_tensor2_transpose to make QNN SDK happy
        QNN_VER_PTR(*p_tensor2_transpose)->dimensions = tensor2_transpose_dims;
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor2_transpose));

        //step-5: compose qnn graph: add mat_mul node
        Qnn_Param_t out_0_params[] = {
                {QNN_PARAMTYPE_SCALAR,
                           QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1,
                             .scalarParam = {QNN_DATATYPE_BOOL_8, .bool8Value = 1}
                }
        };

        Qnn_Tensor_t out_0_inputs[]  = {*p_tensor0,*p_tensor1};
        Qnn_Tensor_t out_0_outputs[] = {*p_tensor2_transpose};
        Qnn_OpConfig_t out_0 = {
                QNN_OPCONFIG_VERSION_1, .v1 =
                        {"ggmlqnn_mulmat_opconfig", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL,
                         1,
                         out_0_params,
                         2,
                         out_0_inputs,
                         1,
                         out_0_outputs}
        };
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle,out_0));

        //step-5: compose qnn graph: add transpose node
        Qnn_Param_t out_trans1_0_params[] = {
                {(Qnn_ParamType_t) 1,
                 "perm", .tensorParam = *p_param_tensor
                }
        };
        Qnn_Tensor_t out_trans1_0_inputs[]  = {*p_tensor2_transpose};
        Qnn_Tensor_t out_trans1_0_outputs[] = {*p_tensor2};
        Qnn_OpConfig_t out_trans1_0 = {
                QNN_OPCONFIG_VERSION_1,
                .v1 =  {"ggmlqnn_mulmat_transpose_opconfig",
                        "qti.aisw",
                        QNN_OP_TRANSPOSE, 1,
                        out_trans1_0_params,
                        1,
                        out_trans1_0_inputs,
                        1,
                        out_trans1_0_outputs}
        };
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle,out_trans1_0));

        //step-6: finalize qnn graph and execute qnn graph
        CHECK_QNN_API(error, qnn_raw_interface.graphFinalize(graph_handle, NULL, NULL));
        Qnn_Tensor_t input_tensors_0[]  = {*p_tensor0,*p_tensor1};
        Qnn_Tensor_t output_tensors_0[] = {*p_tensor2};
        CHECK_QNN_API(error, qnn_raw_interface.graphExecute(graph_handle,
                                               input_tensors_0, 2,
                                               output_tensors_0, 1,
                                               NULL, NULL));

        qnn_tensors_t ggml_op_mulmat_tensors;
        ggml_op_mulmat_tensors.reserve(5);
        ggml_op_mulmat_tensors.push_back(p_tensor0);
        ggml_op_mulmat_tensors.push_back(p_tensor1);
        ggml_op_mulmat_tensors.push_back(p_tensor2);
        ggml_op_mulmat_tensors.push_back(p_param_tensor);
        ggml_op_mulmat_tensors.push_back(p_tensor2_transpose);

        auto  graph_item = std::make_tuple(graph_handle, ggml_op_mulmat_tensors);
        instance->_qnn_graph_map[graph_name] = graph_item;

        //avoid cleanup these resource to make test_backend_ops happy
        //free_qnn_tensor(p_param_tensor);
        //restore pointer to avoid memory leak
        QNN_VER_PTR(*p_tensor2_transpose)->dimensions = tensor2_dimensions_transpose;
        //free_qnn_tensor(p_tensor2_transpose);
    } else {
        QNN_VER_PTR(*p_tensor0)->clientBuf = {src0->data, ggml_get_tensor_data_size(src0)};
        QNN_VER_PTR(*p_tensor1)->clientBuf = {src1->data, ggml_get_tensor_data_size(src1)};
        QNN_VER_PTR(*p_tensor2)->clientBuf = {dst->data, ggml_get_tensor_data_size(dst)};

        Qnn_Tensor_t tensor_inputs[] = {
                *p_tensor0,
                *p_tensor1
        };
        Qnn_Tensor_t tensor_outputs[] = {
                *p_tensor2
        };
        //attention:
        // this is the second technical approach of "how to utilize the Hexagon NPU maximally" through
        // QNN SDK, details could be found at
        // https://github.com/kantv-ai/llama.cpp/wiki/mapping-ggml-compute-graph-to-QNN-compute-graph
        CHECK_QNN_API(error, qnn_raw_interface.graphExecute(graph_handle,
                                         tensor_inputs, 2,
                                         tensor_outputs, 1,
                                         nullptr, nullptr));
    }

    // restore the original dimensions of qnn tensors to avoid memory leak in func free_qnn_tensor
    QNN_VER_PTR(*p_tensor0)->dimensions = tensor_0_dimensions;
    QNN_VER_PTR(*p_tensor1)->dimensions = tensor_1_dimensions;
    QNN_VER_PTR(*p_tensor2)->dimensions = tensor_2_dimensions;
    op_perf.info();
}

static bool ggml_qnn_compute_forward(ggml_backend_t backend, struct ggml_tensor * tensor) {
    ggmlqnn_op_func_t func                = nullptr;

    switch (tensor->op) {
        case GGML_OP_ADD:
            func = ggml_qnn_general_node;
            break;

        case GGML_OP_MUL_MAT:
            func = ggml_qnn_mul_mat;
            break;

        case GGML_OP_MUL:
            func = ggml_qnn_general_node;
            break;

        default:
            return false;
    }

    if (nullptr != func)
        func(backend, tensor);

    return true;
}

struct ggml_backend_qnn_buffer_context {
    ~ggml_backend_qnn_buffer_context() {
        if (buffer) {
            free(buffer);
        }

        for (auto * sub_buffer : sub_buffers) {
            free(sub_buffer);
        }

        for (auto * qnn_tensor : qnn_tensors) {
            free_qnn_tensor(qnn_tensor);
        }

        sub_buffers.clear();
        qnn_tensors.clear();
    }
    void * buffer       = nullptr;

    struct ggml_backend_qnn_context * backend_ctx = nullptr;

    size_t buffer_size  = 0;
    std::vector<void *> sub_buffers;
    std::vector<Qnn_Tensor_t *> qnn_tensors;
};

static void ggml_backend_qnn_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_qnn_buffer_context * ctx = (ggml_backend_qnn_buffer_context *)buffer->context;
    delete ctx;
}

static void * ggml_backend_qnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_qnn_buffer_context * ctx = (ggml_backend_qnn_buffer_context *)buffer->context;

    return ctx->buffer;
}

static void ggml_backend_qnn_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    Qnn_ErrorHandle_t error = QNN_SUCCESS;
    ggml_backend_qnn_buffer_context * ctx = (ggml_backend_qnn_buffer_context *)buffer->context;
    GGML_UNUSED(error);
    GGML_UNUSED(ctx);
    return;
}

static void ggml_backend_qnn_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                               ggml_tensor * tensor, const void * data,
                                               size_t offset, size_t size) {
    GGML_UNUSED(buffer);

    memcpy((char *)tensor->data + offset, data, size);
}

static void ggml_backend_qnn_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                  struct ggml_tensor * tensor,
                                                  uint8_t value, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memset((char *)tensor->data + offset, value, size);
}

static void ggml_backend_qnn_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                               const ggml_tensor * tensor,
                                               void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memcpy(data, (const char *)tensor->data + offset, size);
}

static bool ggml_backend_qnn_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                               const struct ggml_tensor * src,
                                               struct ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }

    return false;
}

static void ggml_backend_qnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_qnn_buffer_context * ctx = (ggml_backend_qnn_buffer_context *)buffer->context;
    memset(ctx->buffer, value, ctx->buffer_size);
}

[[maybe_unused]]static void ggml_backend_qnn_buffer_reset(ggml_backend_buffer_t buffer) {
    ggml_backend_qnn_buffer_context * ctx = (ggml_backend_qnn_buffer_context *)buffer->context;
    for (auto * sub_buffer : ctx->sub_buffers) {
        free(sub_buffer);
    }
    ctx->sub_buffers.clear();
}

static ggml_backend_buffer_i ggml_backend_qnn_buffer_interface = {
        /* .free_buffer     = */ ggml_backend_qnn_buffer_free_buffer,
        /* .get_base        = */ ggml_backend_qnn_buffer_get_base,
        /* .init_tensor     = */ ggml_backend_qnn_buffer_init_tensor,
        /* .memset_tensor   = */ ggml_backend_qnn_buffer_memset_tensor,
        /* .set_tensor      = */ ggml_backend_qnn_buffer_set_tensor,
        /* .get_tensor      = */ ggml_backend_qnn_buffer_get_tensor,
        /* .cpy_tensor      = */ ggml_backend_qnn_buffer_cpy_tensor,
        /* .clear           = */ ggml_backend_qnn_buffer_clear,
        /* .reset           = */ NULL,
};

static const char * ggml_backend_qnn_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return "qnn-buffer";
}

static ggml_backend_buffer_t ggml_backend_qnn_buffer_type_alloc_buffer(
                                  ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_qnn_buffer_context * ctx = new ggml_backend_qnn_buffer_context;

    size_t size_page = sysconf(_SC_PAGESIZE);
    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }
    ctx->buffer         = ggmlqnn_host_malloc(size_aligned);
    ctx->buffer_size    = size_aligned;
    if (nullptr == ctx->buffer) {
        GGMLQNN_LOG_WARN("%s: failed to allocate %.2f MiB\n", __func__, size / (1 << 20));
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_qnn_buffer_interface, ctx, size);
}

static size_t ggml_backend_qnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 32;
}

//FIXME: this value is an experimental value on Xiaomi14
static size_t ggml_backend_qnn_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);

    return (2 * (1 << 30));
}

static bool ggml_backend_qnn_buffer_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

static const char * ggml_backend_qnn_name(ggml_backend_t backend) {
    ggml_backend_qnn_context * ctx = (ggml_backend_qnn_context *) backend->context;
    return g_qnn_mgr[ctx->device].name;
}

static void ggml_backend_qnn_free(ggml_backend_t backend) {
    GGMLQNN_LOG_DEBUG("enter %s", __func__ );
    ggml_backend_qnn_context * ctx = (ggml_backend_qnn_context *) backend->context;
    GGMLQNN_LOG_DEBUG("idx %d, name:%s", ctx->device, g_qnn_mgr[ctx->device].name);

    qnn_instance * instance = (qnn_instance*)g_qnn_mgr[ctx->device].instance;
    if (instance != nullptr) {
        std::map<std::string, std::tuple<Qnn_GraphHandle_t, std::vector<Qnn_Tensor_t*>>>::iterator graph_it;

        for (graph_it = instance->_qnn_graph_map.begin();
             graph_it != instance->_qnn_graph_map.end(); graph_it++) {
            auto & graph_item = graph_it->second;
            Qnn_GraphHandle_t & graph_handle = std::get<0>(graph_item);
            qnn_tensors_t &  tensors = std::get<1>(graph_item);
            for (auto tensor_it = tensors.begin(); tensor_it != tensors.end(); ++tensor_it) {
                free_qnn_tensor(*tensor_it);
            }
            GGML_UNUSED(graph_handle);
            GGMLQNN_LOG_DEBUG("graph type:%s", graph_it->first.c_str());
        }
        instance->_qnn_graph_map.clear();

        instance->qnn_finalize();
        delete instance;
        g_qnn_mgr[ctx->device].instance = nullptr;
    }

    if (g_qnn_mgr[ctx->device].backend != nullptr) {
        delete backend;
        g_qnn_mgr[ctx->device].backend = nullptr;
    }
    GGMLQNN_LOG_DEBUG("leave %s", __func__ );
}

static enum ggml_status ggml_backend_qnn_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status result         = GGML_STATUS_SUCCESS;
    ggml_backend_qnn_context * ctx  = (ggml_backend_qnn_context *) backend->context;
    GGML_UNUSED(ctx);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE
        || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW
        || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }
        bool ok = ggml_qnn_compute_forward(backend, node);
        if (!ok) {
            GGMLQNN_LOG_DEBUG("%s: error: op not supported %s (%s)\n",
                              __func__, node->name, ggml_op_name(node->op));
        }
    }

    return result;
}

static const char * ggml_backend_qnn_device_get_name(ggml_backend_dev_t dev) {
    struct ggml_backend_qnn_context *ctx = static_cast<ggml_backend_qnn_context *>(dev->context);
    if (nullptr == ctx) {
        GGMLQNN_LOG_ERROR("pls check why ctx is null");
        return "unknown";
    }
    return ctx->name;

    GGML_UNUSED(dev);
}

static const char * ggml_backend_qnn_device_get_description(ggml_backend_dev_t dev) {
    struct ggml_backend_qnn_context * ctx = static_cast<ggml_backend_qnn_context *>(dev->context);
    if (nullptr == ctx) {
        GGMLQNN_LOG_ERROR("pls check why ctx is null");
        return "unknown";
    }
    if (0 == strncmp(ctx->name, "qnn-npu", 7)) {
        const char * soc_info = qnn_get_socmodel_desc(ctx->socinfo.soc_model);
        const char * htp_arch = qnn_get_htparch_desc(ctx->socinfo.htp_arch);
        std::string dev_desc = std::string(ctx->desc)
                + std::string(soc_info) + "_" + std::string(htp_arch)
                + "," + std::string(ctx->socinfo.soc_desc);
        return dev_desc.c_str();
    } else {
        return ctx->desc;
    }
}

static void ggml_backend_qnn_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    struct ggml_backend_qnn_context * ctx = static_cast<ggml_backend_qnn_context *>(dev->context);
    if ((nullptr == ctx) || (ctx->device > QNN_BACKEND_GGML)) {
        GGMLQNN_LOG_ERROR("pls check params");
        *free = 0;
        *total = 0;
    }

    if (QNN_BACKEND_CPU == ctx->device || QNN_BACKEND_GGML == ctx->device) {
        *total = get_system_total_memory_in_bytes();
        *free = get_system_free_memory_in_bytes();
    } else if (QNN_BACKEND_GPU == ctx->device) {
        //TODO: probe GPU info in Qualcomm Adreno GPU
        *total = get_system_total_memory_in_bytes();
        *free = get_system_free_memory_in_bytes();
    } else if (QNN_BACKEND_NPU == ctx->device) {
        size_t rpc_ion_memsize = ctx->instance->get_rpcmem_capacity();
        size_t rpc_ion_usage = ctx->instance->get_rpcmem_usage();
        GGMLQNN_LOG_DEBUG("rpc memsize %d", rpc_ion_memsize);
        GGMLQNN_LOG_DEBUG("rpc usage %d", rpc_ion_usage);
        *total = rpc_ion_memsize * (1 << 20);
        *free = (rpc_ion_memsize - rpc_ion_usage) * (1 << 20);
    }
}

static enum ggml_backend_dev_type ggml_backend_qnn_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_qnn_device_get_props(ggml_backend_dev_t dev,
                                              struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_qnn_device_get_name(dev);
    props->description = ggml_backend_qnn_device_get_description(dev);
    props->type        = ggml_backend_qnn_device_get_type(dev);
    ggml_backend_qnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
            /* .async                 = */ false,
            /* .host_buffer           = */ false,
            /* .buffer_from_host_ptr  = */ true,
            /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_qnn_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev);
    if (nullptr == params) {
        params = 0;
    }
    ggml_backend_t qnn_backend = ggml_backend_qnn_init((int) (intptr_t) params,
                                                       "/data/local/tmp/");

    return qnn_backend;

}

ggml_backend_buffer_type_t ggml_backend_qnn_buffer_type(size_t device_index) {
    if (device_index >= GGML_QNN_MAX_DEVICES) {
        GGMLQNN_LOG_DEBUG("ggml_backend_qnn_buffer_type error: device_index:%d is out of range [0, %d]\n",
                      device_index, GGML_QNN_MAX_DEVICES - 1);
        return nullptr;
    }

    static struct ggml_backend_buffer_type ggml_backend_buffer_type_qnn = {
            /* .iface   = */ {
                                     /* .get_name         = */ ggml_backend_qnn_buffer_type_name,
                                     /* .alloc_buffer     = */ ggml_backend_qnn_buffer_type_alloc_buffer,
                                     /* .get_alignment    = */ ggml_backend_qnn_buffer_type_get_alignment,
                                     /* .get_max_size     = */ ggml_backend_qnn_buffer_type_get_max_size,
                                     /* .get_alloc_size   = */ NULL,// defaults to ggml_nbytes
                                     /* .is_host          = */ ggml_backend_qnn_buffer_is_host
                             },
            /* .context = */ NULL,
    };

    return &ggml_backend_buffer_type_qnn;
}

static ggml_backend_buffer_type_t ggml_backend_qnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_qnn_context * ctx = (ggml_backend_qnn_context *) dev->context;
    return ggml_backend_qnn_buffer_type(ctx->device);
}

static ggml_backend_buffer_t ggml_backend_qnn_device_buffer_from_host_ptr(ggml_backend_dev_t dev,
                                                void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}


static bool ggml_backend_qnn_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    ggml_backend_qnn_context * ctx = (ggml_backend_qnn_context *) dev->context;
    return (ggml_qnn_can_handle_op(op));
}

static bool ggml_backend_qnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft);
}

static struct ggml_backend_device_i ggml_backend_qnn_device_interface = {
        /* .get_name             = */ ggml_backend_qnn_device_get_name,
        /* .get_description      = */ ggml_backend_qnn_device_get_description,
        /* .get_memory           = */ ggml_backend_qnn_device_get_memory,
        /* .get_type             = */ ggml_backend_qnn_device_get_type,
        /* .get_props            = */ ggml_backend_qnn_device_get_props,
        /* .init_backend         = */ ggml_backend_qnn_device_init_backend,
        /* .get_buffer_type      = */ ggml_backend_qnn_device_get_buffer_type,
        /* .get_host_buffer_type = */ NULL,
        /* .buffer_from_host_ptr = */ ggml_backend_qnn_device_buffer_from_host_ptr,
        /* .supports_op          = */ ggml_backend_qnn_device_supports_op,
        /* .supports_buft        = */ ggml_backend_qnn_device_supports_buft,
        /* .offload_op           = */ NULL,
        /* .event_new            = */ NULL,
        /* .event_free           = */ NULL,
        /* .event_synchronize    = */ NULL,
};

static ggml_backend_i ggml_backend_qnn_interface = {
        /* .get_name                = */ ggml_backend_qnn_name,
        /* .free                    = */ ggml_backend_qnn_free,
        /* .set_tensor_async        = */ nullptr,
        /* .get_tensor_async        = */ nullptr,
        /* .cpy_tensor_async        = */ nullptr,
        /* .synchronize             = */ nullptr,
        /* .graph_plan_create       = */ nullptr,
        /* .graph_plan_free         = */ nullptr,
        /* .graph_plan_update       = */ nullptr,
        /* .graph_plan_compute      = */ nullptr,
        /* .graph_compute           = */ ggml_backend_qnn_graph_compute,
        /* .event_record            = */ nullptr,
        /* .event_wait              = */ nullptr,
};

//FIXME: this guid is not make sense
static ggml_guid_t ggml_backend_qnn_guid() {
    static ggml_guid guid = {
            0x1a, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f, 0x70, 0x81,
            0x92, 0xa3, 0xb4, 0xc5, 0xd6, 0xe7, 0xf8, 0x09
    };
    return &guid;
}

bool ggml_backend_is_qnn(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_qnn_guid());
}

void ggml_backend_qnn_set_n_threads(ggml_backend_t backend, int n_threads) {
    GGML_ASSERT(ggml_backend_is_qnn(backend));

    struct ggml_backend_qnn_context * ctx = (struct ggml_backend_qnn_context *)backend->context;
    ctx->threads = n_threads;
}

int ggml_backend_qnn_get_device_count() {
    return GGML_QNN_MAX_DEVICES;
}

struct ggml_backend_qnn_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_qnn_reg_get_name(ggml_backend_reg_t reg) {
    return "ggml-qnn";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_qnn_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_QNN_MAX_DEVICES;
}

static ggml_backend_dev_t ggml_backend_qnn_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_UNUSED(reg);
    GGML_UNUSED(index);

    GGMLQNN_LOG_DEBUG("index %d", index);
    ggml_backend_qnn_reg_context * ctx = (ggml_backend_qnn_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static void * ggml_backend_qnn_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);

    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *)ggml_backend_qnn_set_n_threads;
    }
    return NULL;
}

static const ggml_backend_reg_i ggml_backend_qnn_reg_interface = {
        /* .get_name          = */ ggml_backend_qnn_reg_get_name,
        /* .get_device_count  = */ ggml_backend_qnn_reg_get_device_count,
        /* .get_device        = */ ggml_backend_qnn_reg_get_device,
        /* .get_proc_address  = */ ggml_backend_qnn_reg_get_proc_address,
};

ggml_backend_reg_t ggml_backend_qnn_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;
    GGMLQNN_LOG_DEBUG("enter ggml_backend_qnn_reg");
    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_qnn_reg_context * ctx = new ggml_backend_qnn_reg_context;

            for (int i = 0; i < ggml_backend_qnn_get_device_count(); i++) {
                ggml_backend_dev_t dev = new ggml_backend_device {
                        /* .iface       = */ ggml_backend_qnn_device_interface,
                        /* .reg         = */ &reg,
                        /* .context     = */ &g_qnn_mgr[i]
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                    /* .api_version = */ GGML_BACKEND_API_VERSION,
                    /* .iface       = */ ggml_backend_qnn_reg_interface,
                    /* .context     = */ ctx
            };
        }

        initialized = true;
    }
    GGMLQNN_LOG_DEBUG("leave ggml_backend_qnn_reg");

    return &reg;
}

/**
 *
 * @param device            0: QNN_BACKEND_CPU 1: QNN_BACKEND_GPU 2: QNN_BACKEND_NPU
 * @param qnn_lib_path      QNN binrary runtime library path, such as "/data/local/tmp/" on Android or specified in JNI layer
 * @return
 */
ggml_backend_t ggml_backend_qnn_init(size_t device, const char * qnn_lib_path) {
    int result = 0;

    if (nullptr == qnn_lib_path)
        return nullptr;

    GGMLQNN_LOG_DEBUG("device %d", device);
    GGMLQNN_LOG_DEBUG("qnn_lib_path %s", qnn_lib_path);
    if (device >= GGML_QNN_MAX_DEVICES) {
        GGMLQNN_LOG_ERROR("invalid device %d", device);
        return nullptr;
    }

    if (nullptr != g_qnn_mgr[device].backend) {
        GGMLQNN_LOG_WARN("qnn backend %d(%s) already loaded", device, ggml_backend_qnn_get_devname(device));
        return g_qnn_mgr[device].backend;
    }

    std::string path = qnn_lib_path;
    if (QNN_BACKEND_NPU == device) {
        if (0 == setenv("LD_LIBRARY_PATH",
                        (path +
                         ":/vendor/dsp/cdsp:/vendor/lib64:/vendor/dsp/dsp:/vendor/dsp/images").c_str(),
                        1)) {
            GGMLQNN_LOG_INFO("QNN NPU backend setenv successfully");
        } else {
            GGMLQNN_LOG_ERROR("QNN NPU backend setenv failure");
        }
        if (0 == setenv("ADSP_LIBRARY_PATH",
                        (path +
                         ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/vendor/dsp/dsp;/vendor/dsp/images;/dsp").c_str(),
                        1)) {
            GGMLQNN_LOG_INFO("QNN NPU backend setenv successfully");
        } else {
            GGMLQNN_LOG_ERROR("QNN NPU backend setenv failure");
        }
    } else {
        if (0 == setenv("LD_LIBRARY_PATH",
                        (path +
                         ":/vendor/dsp/cdsp:/vendor/lib64:/vendor/dsp/dsp:/vendor/dsp/images").c_str(),
                        1)) {
            GGMLQNN_LOG_INFO("%s backend setenv successfully\n", ggml_backend_qnn_get_devname(device));
        } else {
            GGMLQNN_LOG_ERROR("%s backend setenv failure\n", ggml_backend_qnn_get_devname(device));
        }
    }

    qnn_instance * instance = nullptr;
    instance = new qnn_instance(qnn_lib_path, g_qnn_mgr[device].lib, "");
    result = instance->qnn_init(nullptr);
    if (0 != result) {
        GGMLQNN_LOG_WARN("init qnn subsystem failed with qnn backend %s, pls check why\n", ggml_backend_qnn_get_devname(device));
        delete instance;
        return nullptr;
    }
    qnn_interface qnn_interface                             = instance->get_qnn_interface();
    if (!qnn_interface.is_loaded()) {
        GGMLQNN_LOG_WARN("qnn subsystem failure\n");
        delete instance;
        return nullptr;
    }

    std::string device_name = ggml_backend_qnn_get_devname(device);
    GGMLQNN_LOG_INFO("qnn device name %s", device_name.c_str());
    g_qnn_mgr[device].instance                  = instance;
    g_qnn_mgr[device].raw_interface             = instance->get_qnn_raw_interface();
    g_qnn_mgr[device].raw_system_interface      = instance->get_qnn_raw_system_interface();

    ggml_backend_t qnn_backend = new ggml_backend{
            /* .guid      = */ ggml_backend_qnn_guid(),
            /* .iface     = */ ggml_backend_qnn_interface,
            /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_qnn_reg(), device),
            /* .context   = */ &g_qnn_mgr[device]
    };
    g_qnn_mgr[device].backend   = qnn_backend;

    return qnn_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_qnn_reg)
