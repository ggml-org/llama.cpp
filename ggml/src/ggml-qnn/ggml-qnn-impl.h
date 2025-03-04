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
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>
#if defined(__ANDROID__) || defined(__linux__)
#include <unistd.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

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
#include <future>
#if (defined __ANDROID__) || (defined ANDROID)
#include "android/log.h"
#endif

#if defined(_WIN32)
#include <wchar.h>
#include <Windows.h>
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

class  qnn_instance;
struct ggml_backend_qnn_context;
void   ggmlqnn_log_internal(ggml_log_level level, const char * file, const char * func, int line, const char * format, ...);

#if 0//def NDEBUG
#define GGMLQNN_DEBUG                           0
#define ENABLE_QNNBACKEND_PERF                  0  // enable/disable op's perf info
#define GGMLQNN_PRINT_QNN_INTERNAL_LOG          0  // enable/disable QNN's internal log
#define GGMLQNN_PRINT_OP_ADD_LOG                0  // GGML_OP_ADD already verified with QNN-CPU / QNN-GPU / QNN-NPU
#define GGMLQNN_PRINT_OP_MUL_MAT_LOG            0
#else
#define GGMLQNN_DEBUG                           1  // for troubleshooting QNN backend
#define ENABLE_QNNBACKEND_PERF                  0  // enable/disable op's perf info
#define GGMLQNN_PRINT_QNN_INTERNAL_LOG          0  // enable/disable QNN's internal log
#define GGMLQNN_PRINT_OP_ADD_LOG                0  // GGML_OP_ADD already verified with QNN-CPU / QNN-GPU / QNN-NPU
#define GGMLQNN_PRINT_OP_MUL_MAT_LOG            1
#endif
#define GGML_QNN_LOGBUF_LEN                     4096

#define GGMLQNN_LOG_ERROR(...) ggmlqnn_log_internal(GGML_LOG_LEVEL_ERROR, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLQNN_LOG_WARN(...)  ggmlqnn_log_internal(GGML_LOG_LEVEL_WARN , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLQNN_LOG_INFO(...)  ggmlqnn_log_internal(GGML_LOG_LEVEL_INFO , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

#if GGMLQNN_DEBUG
#define GGMLQNN_LOG_DEBUG(...) ggmlqnn_log_internal(GGML_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#else
#define GGMLQNN_LOG_DEBUG(...)
#endif

#define CHECK_QNN_API(error, result)                                            \
    do {                                                                        \
        error = (result);                                                       \
        if (QNN_SUCCESS != error) {                                             \
            if (error == QNN_COMMON_ERROR_NOT_SUPPORTED) {                      \
                GGMLQNN_LOG_WARN("WARNING: QNN feature/API not supported\n");   \
            } else {                                                            \
                GGMLQNN_LOG_INFO("QNN API error = %d(%s)\n", error, ggmlqnn_get_error_string(error));  \
            }                                                                   \
        }                                                                       \
    } while (0)

#define QNN_VER_PTR(x)                          (&((x).v1))
#define RPCMEM_DEFAULT_FLAGS                    1
#define RPCMEM_HEAP_ID_SYSTEM                   25

#define DISABLE_COPY(class_name)                \
    class_name(const class_name &) = delete;    \
    void operator=(const class_name &) = delete

#define DISABLE_MOVE(class_name)                \
    class_name(class_name &&) = delete;         \
    void operator=(class_name &&) = delete

#define GQCGT                                   ggmlqnn_create_general_tensor

#if defined(_WIN32)
#define RTLD_GLOBAL 0x100
#define RTLD_LOCAL  0x000
#define RTLD_LAZY   0x000
#define RTLD_NOW    0x001
void *              dlopen(const char * filename, int flag);
int                 dlclose(void * handle);
void *              dlsym(void* handle, const char* name);
const char *        dlerror(void);
#endif

using pfn_rpc_mem_init                          = void (*)(void);
using pfn_rpc_mem_deinit                        = void (*)(void);
using pfn_rpc_mem_alloc                         = void *(*)(int, uint32_t, int);
using pfn_rpc_mem_free                          = void (*)(void *);
using pfn_rpc_mem_to_fd                         = int (*)(void *);
using _pfn_QnnSaver_initialize                  = decltype(QnnSaver_initialize);
using _pfn_QnnInterface_getProviders            = decltype(QnnInterface_getProviders);
using _pfn_QnnSystemInterface_getProviders      = decltype(QnnSystemInterface_getProviders);

using qnn_res_t                                 = std::tuple<Qnn_GraphHandle_t, std::vector< Qnn_Tensor_t *>>;
using qnn_tensors_t                             = std::vector< Qnn_Tensor_t *>;

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
#if defined(_MSC_VER)
    SC7280X     = 44,
    SC8280X     = 37,
    SC8380XP    = 60,
#endif
};

struct qcom_socinfo {
    uint32_t soc_model;
    size_t htp_arch;
    size_t vtcm_size_in_mb;
    char soc_desc[GGML_MAX_NAME];
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

    std::unique_ptr<char[]> work_data;
    std::vector<std::future<void>> tasks;
    size_t work_size    = 0;
    size_t desired_size = 0;
    int n_threads       = GGML_DEFAULT_N_THREADS;
};

struct qnn_op_caps_t {
    const char * qnn_op_name        = nullptr;
    const size_t input_param_count  = 0;
    const char * qnn_param_name     = nullptr;
};
extern const qnn_op_caps_t ggmlqnn_k_op_caps[];

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
    qnn_perf(const std::string & perf_name) {
        GGML_UNUSED(perf_name);
    }
    qnn_perf() = delete;
    qnn_perf(const qnn_perf & ) = delete;
    qnn_perf & operator= (const qnn_perf & ) = delete;

    void start() {}
    void info() {}
};
#endif

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
    DEFINE_SHIM_FUNCTION_INTERFACE(backend_create, backendCreate)

    DEFINE_SHIM_FUNCTION_INTERFACE(backend_free, backendFree)

    DEFINE_SHIM_FUNCTION_INTERFACE(backend_register_op_package, backendRegisterOpPackage)

    DEFINE_SHIM_FUNCTION_INTERFACE(backend_validate_op_config, backendValidateOpConfig)

    DEFINE_SHIM_FUNCTION_INTERFACE(backend_get_api_version, backendGetApiVersion)

    // QnnDevice
    DEFINE_SHIM_FUNCTION_INTERFACE(device_create, deviceCreate)

    DEFINE_SHIM_FUNCTION_INTERFACE(device_free, deviceFree)

    DEFINE_SHIM_FUNCTION_INTERFACE(device_get_infrastructure, deviceGetInfrastructure)

    DEFINE_SHIM_FUNCTION_INTERFACE(device_get_platform_info, deviceGetPlatformInfo)

    DEFINE_SHIM_FUNCTION_INTERFACE(device_get_info, deviceGetInfo)

    // QnnContext
    DEFINE_SHIM_FUNCTION_INTERFACE(context_create, contextCreate)

    DEFINE_SHIM_FUNCTION_INTERFACE(context_get_binary_size, contextGetBinarySize)

    DEFINE_SHIM_FUNCTION_INTERFACE(context_get_binary, contextGetBinary)

    DEFINE_SHIM_FUNCTION_INTERFACE(context_create_from_binary, contextCreateFromBinary)

    DEFINE_SHIM_FUNCTION_INTERFACE(context_free, contextFree)

    // QnnGraph
    DEFINE_SHIM_FUNCTION_INTERFACE(graph_create, graphCreate)

    DEFINE_SHIM_FUNCTION_INTERFACE(graph_add_node, graphAddNode)

    DEFINE_SHIM_FUNCTION_INTERFACE(graph_finalize, graphFinalize)

    DEFINE_SHIM_FUNCTION_INTERFACE(graph_execute, graphExecute)

    DEFINE_SHIM_FUNCTION_INTERFACE(graph_retrieve, graphRetrieve)

    // QnnLog
    DEFINE_SHIM_FUNCTION_INTERFACE(log_create, logCreate)

    DEFINE_SHIM_FUNCTION_INTERFACE(log_free, logFree)

    DEFINE_SHIM_FUNCTION_INTERFACE(log_set_log_level, logSetLogLevel)

    // QnnProfile
    DEFINE_SHIM_FUNCTION_INTERFACE(profile_create, profileCreate)

    DEFINE_SHIM_FUNCTION_INTERFACE(profile_get_events, profileGetEvents)

    DEFINE_SHIM_FUNCTION_INTERFACE(profile_get_sub_events, profileGetSubEvents)

    DEFINE_SHIM_FUNCTION_INTERFACE(profile_get_event_data, profileGetEventData)

    DEFINE_SHIM_FUNCTION_INTERFACE(profile_free, profileFree)

    // QnnMem
    DEFINE_SHIM_FUNCTION_INTERFACE(mem_register, memRegister)

    DEFINE_SHIM_FUNCTION_INTERFACE(mem_de_register, memDeRegister)

    // QnnProperty
    DEFINE_SHIM_FUNCTION_INTERFACE(property_has_capability, propertyHasCapability)

    // QnnTensor
    DEFINE_SHIM_FUNCTION_INTERFACE(tensor_create_context_tensor, tensorCreateContextTensor)

    DEFINE_SHIM_FUNCTION_INTERFACE(tensor_create_graph_tensor, tensorCreateGraphTensor)

    // QnnSystem
    DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_create, systemContextCreate)

    DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_get_binary_info, systemContextGetBinaryInfo)

    DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_free, systemContextFree)

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
    const QnnInterface_t * _qnn_interface           = nullptr;

    const QnnSystemInterface_t * _qnn_sys_interface = nullptr;
};

class qnn_instance {
public:
    using BackendIdType = decltype(QnnInterface_t{}.backendId);

    explicit qnn_instance(const std::string & lib_path, const std::string & backend_name,
                          const std::string & model_name) :
            _lib_path(std::move(lib_path)),
            _backend_name(std::move(backend_name)),
            _model_name(std::move(model_name)) {}

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

    Qnn_LogHandle_t get_qnn_log_handle() { return _qnn_log_handle; }

    Qnn_ProfileHandle_t get_qnn_profile_handle() { return _qnn_profile_handle; }

    Qnn_DeviceHandle_t get_qnn_device_handle() { return _qnn_device_handle; }

    Qnn_BackendHandle_t get_qnn_backend_handle() { return _qnn_backend_handle; }

    Qnn_ContextHandle_t get_qnn_context_handle() { return _qnn_context_handle; }

    QnnSystemContext_Handle_t get_qnn_system_handle() { return _qnn_system_handle; }

    Qnn_GraphHandle_t get_qnn_graph_handle() { return _qnn_graph_handle; }

    int init_qnn_graph(const char * graph_name,
                       bool debug,
                       uint8_t do_node_validation = 1,
                       const QnnGraph_Config_t ** graph_configs = nullptr
    );
    int init_qnn_graph(const std::string & graph_name, QNNBackend device, size_t vtcm_size_in_mb = 8, size_t hvx_threads = 8);

    int finalize_qnn_graph();

    bool is_valid_graph() const { return _qnn_graph_handle != nullptr; }

    int init_htp_perfinfra();

    int set_rpc_polling();

    int set_high_performance_mode();

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

    QNNBackend get_device_id() {
        return _device_id;
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

    void probe_device_meminfo();

private:
    static constexpr const int _required_num_providers = 1;

private:
    std::string     _lib_path;
    std::string     _backend_name;
    std::string     _model_name; // name of prebuilt QNN model, might be used in the future
    BackendIdType   _backend_id;

    bool _debug_tensor                      = false; // flag to indicate if requested graph is to be run in debug mode
    bool _do_node_validations               = true;  // flag to indicate whether all add_node calls need to be validated
    QnnLog_Level_t _qnn_log_level           = QNN_LOG_LEVEL_DEBUG;

    ggml_qnn_profile_level _profile_level   = ggml_qnn_profile_level::profile_detail;

    void * _system_lib_handle               = nullptr;

    Qnn_GraphHandle_t _qnn_graph_handle     = nullptr;

    Qnn_LogHandle_t _qnn_log_handle         = nullptr;

    Qnn_ProfileHandle_t _qnn_profile_handle = nullptr;

    Qnn_DeviceHandle_t _qnn_device_handle   = nullptr;

    Qnn_BackendHandle_t _qnn_backend_handle = nullptr;

    Qnn_ContextHandle_t _qnn_context_handle = nullptr;

    QnnSystemContext_Handle_t _qnn_system_handle = nullptr;

    QnnHtpDevice_PerfInfrastructure_t * _qnn_htp_perfinfra = nullptr;
    uint32_t _qnn_power_configid            = 1;
    uint32_t _qnn_rpc_pollingtime           = 9999; // 0-10000 us for high performing

    qnn_interface _qnn_interface;
    QNN_INTERFACE_VER_TYPE _qnn_raw_interface;
    QNN_SYSTEM_INTERFACE_VER_TYPE _qnn_raw_system_interface;

    std::unordered_map<void *, Qnn_MemHandle_t> _qnn_mem_set;
    std::unordered_map<void *, Qnn_MemHandle_t> _qnn_rpc_buffer_to_handles;

    static std::mutex _init_mutex;
    static std::unordered_map<BackendIdType, void *> _loaded_lib_handle;
    static std::unordered_map<std::string, BackendIdType> _lib_path_to_backend_id;
    static std::unordered_map<BackendIdType, const QnnInterface_t *> _loaded_backend;

    std::atomic_bool _rpcmem_initialized{false};
    pfn_rpc_mem_alloc _pfn_rpc_mem_alloc;
    pfn_rpc_mem_free _pfn_rpc_mem_free;
    pfn_rpc_mem_to_fd _pfn_rpc_mem_to_fd;
    pfn_rpc_mem_init  _pfn_rpc_mem_init;
    pfn_rpc_mem_deinit _pfn_rpc_mem_deinit;
    std::unordered_map<void *, void *> _rpcmem_store_map;
    std::unordered_map<void *, size_t> _rpcmem_usage_map;
    size_t                             _rpcmem_usage    = 0;   // mempool usage in Mbytes
    size_t                             _rpcmem_capacity = 512; // mempool size  in Mbytes

    std::string _graph_name;
    QNNBackend _device_id;
    void * _rpc_lib_handle      = nullptr;
    bool       _enable_qnn_rpc  = false; //TODO:unknown issue with QNN RPC feature

    DISABLE_COPY(qnn_instance);
    DISABLE_MOVE(qnn_instance);
};

size_t         ggmlqnn_get_opcaps_size(void);
size_t         ggmlqnn_get_op_index(const ggml_tensor * tensor);
Qnn_Tensor_t * ggmlqnn_create_compute_tensor(const ggml_tensor * tensor);
const char   * ggmlqnn_get_error_string(Qnn_ErrorHandle_t qnn_error_code);
Qnn_DataType_t ggmlqnn_datatype_from_ggml_datatype(enum ggml_type ggmltype);
void         * ggmlqnn_type_trait(ggml_backend_qnn_context * ctx, ggml_tensor * op);
void           ggmlqnn_get_graphkey_from_op(const ggml_tensor * op, std::string & output);
uint8_t      * ggmlqnn_create_rpc_buffer(qnn_instance * instance, const ggml_tensor * ggml_tensor, Qnn_Tensor_t * qnn_tensor, bool b_copydata);
void           ggmlqnn_print_tensors_info(const char * func_name, ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

Qnn_OpConfig_t ggmlqnn_create_op_config(const char * name, const char * package, const char * type,
                                Qnn_Param_t * params, uint32_t num_params,
                                Qnn_Tensor_t * inputs, uint32_t num_inputs,
                                Qnn_Tensor_t * outputs, uint32_t num_outputs);
Qnn_Tensor_t * ggmlqnn_create_general_tensor(const ggml_tensor * tensor, const char * name,
                                Qnn_TensorType_t qnn_tensor_type,
                                Qnn_DataType_t qnn_data_type,
                                uint32_t rank, uint32_t * dims,
                                void * data, uint32_t data_size,
                                bool b_transpose = false);
