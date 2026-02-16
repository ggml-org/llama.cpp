
#include "qnn-lib.hpp"

#include <filesystem>

#include "common.hpp"
#include "rpc-mem.hpp"

#if defined(__linux__)
#    include <unistd.h>
#endif

namespace {

#ifdef _WIN32
#    define PLATFORM_LIB_FILENAME(name) (name ".dll")
#else
#    define PLATFORM_LIB_FILENAME(name) ("lib" name ".so")
#endif

#if defined(__aarch64__) || defined(_M_ARM64)  // TODO: check for other platforms
#    define PLATFORM_LIB_POSFIX "_aarch64"
#else
#    define PLATFORM_LIB_POSFIX "_x64"
#endif

constexpr const char * kQnnSystemLibName     = PLATFORM_LIB_FILENAME("QnnSystem");
constexpr const char * kQnnCpuLibName        = PLATFORM_LIB_FILENAME("QnnCpu");
constexpr const char * kQnnGpuLibName        = PLATFORM_LIB_FILENAME("QnnGpu");
constexpr const char * kQnnNpuLibName        = PLATFORM_LIB_FILENAME("QnnHtp");
constexpr const char * kQnnCpuPackageLibName = PLATFORM_LIB_FILENAME("QnnGgmlOpPackage" PLATFORM_LIB_POSFIX);

constexpr const qnn::device_caps kDeviceCaps[] = {
    {
     // https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/CpuOpDefSupplement.html#matmul
        kQnnCpuLibName,                                                                   GGML_BACKEND_DEVICE_TYPE_ACCEL, (1L << GGML_TYPE_I8) | (1L << GGML_TYPE_F32),
#ifdef GGML_HEXAGON_ENABLE_QUANTIZED_TENSORS
     // all quantized types can be offload to CPU, at current implementation, those types will be dequantized into float32 on cpu
        0xFFFFFE,
#else
        (1L << GGML_TYPE_F32),
#endif

     0,                                                                     // 0 for no limitation
    },
    {
     // https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/GpuOpDefSupplement.html#matmul
        kQnnGpuLibName,                                                                                    GGML_BACKEND_DEVICE_TYPE_GPU,                                                                                                   (1L << GGML_TYPE_F32) | (1L << GGML_TYPE_F16),
#ifdef GGML_HEXAGON_ENABLE_QUANTIZED_TENSORS
     // all quantized types can be offload to GPU, at current implementation, those types will be dequantized into float32 on cpu
        0xFFFFFE,
#else
        (1L << GGML_TYPE_F32) | (1L << GGML_TYPE_F16),
#endif
     (128256L * 4096 *
         sizeof(float)), // tested on 8 gen 2, failed to allocate tensor with size 128256x4096 and float32
    },
    {
     // https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/HtpOpDefSupplement.html#matmul
        kQnnNpuLibName, GGML_BACKEND_DEVICE_TYPE_ACCEL,
        (1L << GGML_TYPE_F32) | (1L << GGML_TYPE_F16) | (1L << GGML_TYPE_I16),
#ifdef GGML_HEXAGON_ENABLE_QUANTIZED_TENSORS
        (1L << GGML_TYPE_Q2_K) | (1L << GGML_TYPE_Q3_K) | (1L << GGML_TYPE_Q4_K) | (1L << GGML_TYPE_Q8_K),
#else
        (1L << GGML_TYPE_F32) | (1L << GGML_TYPE_F16),
#endif
     (8192L * 2048 + 8192 * 512 + 2048 * 512) * sizeof(float),  // TODO: should have a better way to get this value
    },
};

static_assert(sizeof(kDeviceCaps) / sizeof(kDeviceCaps[0]) == QNN_BACKEND_COUNT,
              "The number of qnn devices should be equal to QNN_BACKEND_COUNT");
static_assert(kDeviceCaps[QNN_BACKEND_NPU].type == GGML_BACKEND_DEVICE_TYPE_ACCEL,
              "The NPU device should be an accelerator device");
static_assert(kDeviceCaps[QNN_BACKEND_GPU].type == GGML_BACKEND_DEVICE_TYPE_GPU,
              "The GPU device should be an GPU device");
static_assert(
    kDeviceCaps[QNN_BACKEND_CPU].type == GGML_BACKEND_DEVICE_TYPE_ACCEL,
    "The CPU device should be an accelerator device");  // we treat qnn-cpu as a supplementary accelerator device
static_assert(GGML_TYPE_Q4_0 == 2 && GGML_TYPE_Q8_K == 15, "The quantized type order is not correct");

void insert_path(std::string & path, std::string insert_path, const char separator = ':') {
    if (!insert_path.empty() && !path.empty()) {
        insert_path += separator;
    }

    path.insert(0, insert_path);
}

// TODO: Fix this for other platforms, or use a more portable way to set the library search path
bool set_qnn_lib_search_path(const std::string & custom_lib_search_path) {
#if defined(__linux__)
    {
        auto *      original        = getenv("LD_LIBRARY_PATH");
        std::string lib_search_path = original ? original : "";
        insert_path(lib_search_path,
                    "/vendor/dsp/cdsp:/vendor/lib64:"
                    "/vendor/dsp/dsp:/vendor/dsp/images");
        insert_path(lib_search_path, custom_lib_search_path);
        if (setenv("LD_LIBRARY_PATH", lib_search_path.c_str(), 1)) {
            return false;
        }
    }

#    if defined(__ANDROID__) || defined(ANDROID)
    {
        // See also: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/dsp_runtime.html
        std::string adsp_lib_search_path = custom_lib_search_path +
                                           ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/"
                                           "rfsa/adsp;/vendor/dsp/dsp;/vendor/dsp/images;/dsp";
        if (setenv("ADSP_LIBRARY_PATH", adsp_lib_search_path.c_str(), 1)) {
            return false;
        }

        QNN_LOG_DEBUG("ADSP_LIBRARY_PATH=%s", getenv("ADSP_LIBRARY_PATH\n"));
    }
#    endif

    QNN_LOG_DEBUG("LD_LIBRARY_PATH=%s", getenv("LD_LIBRARY_PATH\n"));
#else
    (void) custom_lib_search_path;
#endif

    return true;
}

common::dl_handler_t load_lib_with_fallback(const std::string & lib_path, const std::string & load_directory) {
    std::filesystem::path full_path(load_directory);
    full_path /= std::filesystem::path(lib_path).filename();
    auto handle = common::dl_load(full_path.string());
    if (!handle) {
        QNN_LOG_WARN("failed to load %s, fallback to %s\n", full_path.c_str(), lib_path.c_str());
        handle = common::dl_load(lib_path);
    }

    return handle;
}

struct op_package_lib_info {
    const char * lib_name;
    const char * interface;
    const char * type;
    size_t       htp_arch;
    const char * extra_lib_name = nullptr;
};

const op_package_lib_info & get_op_package_lib_info(uint32_t soc_model, size_t htp_arch) {
    constexpr static const op_package_lib_info kOpPackageLibInfo[] = {
        { kQnnCpuPackageLibName, "GgmlOpPackageInterfaceProvider", "CPU", qnn::NONE,
         PLATFORM_LIB_FILENAME("HtpPrepare") },
        { PLATFORM_LIB_FILENAME("QnnGgmlOpPackage_v68"), "GgmlOpPackageInterfaceProvider", "HTP", qnn::V68 },
        { PLATFORM_LIB_FILENAME("QnnGgmlOpPackage_v69"), "GgmlOpPackageInterfaceProvider", "HTP", qnn::V69 },
        { PLATFORM_LIB_FILENAME("QnnGgmlOpPackage_v73"), "GgmlOpPackageInterfaceProvider", "HTP", qnn::V73 },
        { PLATFORM_LIB_FILENAME("QnnGgmlOpPackage_v75"), "GgmlOpPackageInterfaceProvider", "HTP", qnn::V75 },
        { PLATFORM_LIB_FILENAME("QnnGgmlOpPackage_v79"), "GgmlOpPackageInterfaceProvider", "HTP", qnn::V79 },
    };

    if (soc_model == qnn::UNKNOWN || soc_model == qnn::EMULATOR_X64 || soc_model == qnn::EMULATOR_AARCH64) {
        return kOpPackageLibInfo[0];
    }

    switch (htp_arch) {
        case qnn::V68:
            static_assert(kOpPackageLibInfo[1].htp_arch == qnn::V68);
            return kOpPackageLibInfo[1];
        case qnn::V69:
            static_assert(kOpPackageLibInfo[2].htp_arch == qnn::V69);
            return kOpPackageLibInfo[2];
        case qnn::V73:
            static_assert(kOpPackageLibInfo[3].htp_arch == qnn::V73);
            return kOpPackageLibInfo[3];
        case qnn::V75:
            static_assert(kOpPackageLibInfo[4].htp_arch == qnn::V75);
            return kOpPackageLibInfo[4];
        case qnn::V79:
        default:
            static_assert(kOpPackageLibInfo[5].htp_arch == qnn::V79);
            return kOpPackageLibInfo[5];
    }
}

}  // namespace

namespace qnn {

qnn_system_interface::qnn_system_interface(const QnnSystemInterface_t & qnn_sys_interface,
                                           common::dl_handler_t         lib_handle) :
    _qnn_sys_interface(qnn_sys_interface),
    _lib_handle(lib_handle) {
    qnn_system_context_create(&_qnn_system_handle);
    if (_qnn_system_handle) {
        QNN_LOG_INFO("initialize qnn system successfully\n");
    } else {
        QNN_LOG_WARN("can not create QNN system contenxt\n");
    }
}

qnn_system_interface::~qnn_system_interface() {
    if (_qnn_system_handle) {
        if (qnn_system_context_free(_qnn_system_handle) != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to free QNN system context\n");
        }
    } else {
        QNN_LOG_WARN("system handle is null\n");
    }

    if (_lib_handle) {
        if (!common::dl_unload(_lib_handle)) {
            QNN_LOG_WARN("failed to close QnnSystem library, error %s\n", common::dl_error());
        }
    } else {
        QNN_LOG_WARN("system lib handle is null\n");
    }
}

qnn_instance::qnn_instance(const std::string & lib_path, backend_index_type device) :
    _additional_lib_load_path(lib_path) {
    _backend_lib_name = kDeviceCaps[device].lib_name;
    if (set_qnn_lib_search_path(lib_path)) {
        QNN_LOG_DEBUG("[%s] set_qnn_lib_search_path succeed\n", _backend_lib_name.c_str());
    } else {
        QNN_LOG_ERROR("[%s] set_qnn_lib_search_path failed\n", _backend_lib_name.c_str());
    }
}

bool qnn_instance::qnn_init(const QnnSaver_Config_t ** saver_config) {
    BackendIdType backend_id = QNN_BACKEND_ID_NULL;
    QNN_LOG_DEBUG("enter qnn_init\n");

    std::lock_guard<std::mutex> lock(_init_mutex);
    if (load_system() != 0) {
        QNN_LOG_WARN("failed to load QNN system lib\n");
        return false;
    } else {
        QNN_LOG_DEBUG("load QNN system lib successfully\n");
    }

    std::string backend_lib_path = _backend_lib_name;
    if (_lib_path_to_backend_id.count(backend_lib_path) == 0) {
        if (!load_backend(backend_lib_path, saver_config)) {
            QNN_LOG_WARN("failed to load QNN backend\n");
            return false;
        }
    }

    backend_id = _lib_path_to_backend_id[backend_lib_path];
    if (_loaded_backend.count(backend_id) == 0 || _loaded_lib_handle.count(backend_id) == 0) {
        QNN_LOG_WARN(
            "library %s is loaded but loaded backend count=%zu, "
            "loaded lib_handle count=%zu",
            backend_lib_path.c_str(), _loaded_backend.count(backend_id), _loaded_lib_handle.count(backend_id));
        return false;
    }

    _qnn_interface = std::make_shared<qnn_interface>(*_loaded_backend[backend_id]);
    _qnn_interface->qnn_log_create(qnn::sdk_logcallback, _qnn_log_level, &_qnn_log_handle);
    if (!_qnn_log_handle) {
        // NPU backend not work on Qualcomm SoC equipped low-end phone
        QNN_LOG_WARN("failed to initialize qnn log\n");
        return false;
    } else {
        QNN_LOG_DEBUG("initialize qnn log successfully\n");
    }

    std::vector<const QnnBackend_Config_t *> temp_backend_config;
    _qnn_interface->qnn_backend_create(
        _qnn_log_handle, temp_backend_config.empty() ? nullptr : temp_backend_config.data(), &_qnn_backend_handle);
    if (!_qnn_backend_handle) {
        QNN_LOG_WARN("failed to initialize qnn backend\n");
        return false;
    } else {
        QNN_LOG_DEBUG("initialize qnn backend successfully\n");
    }

    auto qnn_status = _qnn_interface->qnn_property_has_capability(QNN_PROPERTY_GROUP_DEVICE);
    switch (qnn_status) {
        case QNN_PROPERTY_NOT_SUPPORTED:
            QNN_LOG_WARN("device property is not supported\n");
            break;
        case QNN_PROPERTY_ERROR_UNKNOWN_KEY:
            QNN_LOG_WARN("device property is unknown\n");
            break;
    }

    {
        const QnnDevice_PlatformInfo_t * p_info = nullptr;
        qnn_status                              = _qnn_interface->qnn_device_get_platform_info(nullptr, &p_info);
        if (qnn_status == QNN_SUCCESS) {
            QNN_LOG_INFO("device counts %d\n", p_info->v1.numHwDevices);
            QnnDevice_HardwareDeviceInfo_t * infos = p_info->v1.hwDevices;
            for (uint32_t i = 0; i < p_info->v1.numHwDevices; i++) {
                QNN_LOG_INFO("deviceID:%d, deviceType:%d, numCores %d\n", (int) infos[i].v1.deviceId,
                             (int) infos[i].v1.deviceType, (int) infos[i].v1.numCores);
                QnnDevice_DeviceInfoExtension_t          devinfo  = infos[i].v1.deviceInfoExtension;
                QnnHtpDevice_OnChipDeviceInfoExtension_t chipinfo = devinfo->onChipDevice;
                size_t                                   htp_arch = (size_t) chipinfo.arch;
                QNN_LOG_INFO("htp_type:%d(%s)\n", devinfo->devType,
                             (devinfo->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP) ? "ON_CHIP" : "");
                QNN_LOG_INFO("soc_model:%s(%s), htp_arch:%s(%d), vtcm_size:%d MB\n",
                             get_chipset_desc(chipinfo.socModel), get_chipset_model(chipinfo.socModel),
                             get_htparch_desc(htp_arch), (int) htp_arch, (int) chipinfo.vtcmSize);
            }

            if (p_info->v1.numHwDevices) {
                QnnDevice_DeviceInfoExtension_t devinfo = infos[p_info->v1.numHwDevices - 1].v1.deviceInfoExtension;
                QnnHtpDevice_OnChipDeviceInfoExtension_t chipinfo = devinfo->onChipDevice;
                size_t                                   htp_arch = (size_t) chipinfo.arch;
                _soc_info                                         = { chipinfo.socModel, htp_arch, chipinfo.vtcmSize };
            }

            _qnn_interface->qnn_device_free_platform_info(nullptr, p_info);
        } else {
            // For emulator, we can't get platform info
            QNN_LOG_INFO("failed to get platform info, emulator or cpu backend?\n");
#if defined(__aarch64__) || defined(_M_ARM64)
            _soc_info = { EMULATOR_AARCH64, NONE, 0 };
#elif defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
            _soc_info = { EMULATOR_X64, NONE, 0 };
#else
            _soc_info = { UNKNOWN, NONE, 0 };
#endif
        }
    }

    {
        qnn_status = _qnn_interface->qnn_device_create(_qnn_log_handle, nullptr, &_qnn_device_handle);
        if (QNN_SUCCESS != qnn_status && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnn_status) {
            QNN_LOG_WARN("failed to create QNN device\n");
        } else {
            QNN_LOG_INFO("create QNN device successfully\n");
        }
    }

    {
        auto rpc_mem = std::make_unique<common::rpc_mem>();
        if (rpc_mem->is_valid()) {
            _rpc_mem = std::move(rpc_mem);
        }
    }

    _qnn_interface->qnn_context_create(_qnn_backend_handle, _qnn_device_handle, nullptr, &_qnn_context_handle);
    if (!_qnn_context_handle) {
        QNN_LOG_WARN("failed to initialize qnn context\n");
        return false;
    } else {
        QNN_LOG_DEBUG("initialize qnn context successfully\n");
    }

    if (_backend_lib_name.find("Htp") != _backend_lib_name.npos) {
        if (init_htp_perfinfra() != 0) {
            QNN_LOG_WARN("initialize HTP performance failure\n");
        }
        if (set_rpc_polling() != 0) {
            QNN_LOG_WARN("set RPC polling failure\n");
        }
        if (set_high_performance_mode() != 0) {
            QNN_LOG_WARN("set HTP high performance mode failure\n");
        }
    }

    QNN_LOG_DEBUG("leave qnn_init\n");
    return true;
}

bool qnn_instance::qnn_finalize() {
    if (_backend_lib_name.find("Htp") != _backend_lib_name.npos) {
        _qnn_htp_perfinfra->destroyPowerConfigId(_qnn_power_configid);
    }

    if (_qnn_context_handle) {
        auto error = _qnn_interface->qnn_context_free(_qnn_context_handle, nullptr);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to free QNN context_handle: ID %u, error %d\n", _qnn_interface->get_backend_id(),
                         (int) QNN_GET_ERROR_CODE(error));
        }
        _qnn_context_handle = nullptr;
    }

    if (_qnn_device_handle) {
        auto error = _qnn_interface->qnn_device_free(_qnn_device_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to free QNN device_handle: ID %u, error %d\n", _qnn_interface->get_backend_id(),
                         (int) QNN_GET_ERROR_CODE(error));
        }
        _qnn_device_handle = nullptr;
    }

    if (_qnn_backend_handle) {
        auto error = _qnn_interface->qnn_backend_free(_qnn_backend_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to free QNN backend_handle: ID %u, error %d\n", _qnn_interface->get_backend_id(),
                         (int) QNN_GET_ERROR_CODE(error));
        }
        _qnn_backend_handle = nullptr;
    }

    if (_qnn_log_handle) {
        auto error = _qnn_interface->qnn_log_free(_qnn_log_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to free QNN log_handle: ID %u, error %d\n", _qnn_interface->get_backend_id(),
                         (int) QNN_GET_ERROR_CODE(error));
        }
        _qnn_log_handle = nullptr;
    }

    if (_custom_op_extra_lib_handle) {
        common::dl_unload(_custom_op_extra_lib_handle);
    }

    unload_backend();

    _qnn_sys_interface.reset();

    _rpc_mem.reset();

    return true;
}

int qnn_instance::load_system() {
    QNN_LOG_DEBUG("[%s]lib: %s\n", _backend_lib_name.c_str(), kQnnSystemLibName);
    auto system_lib_handle = load_lib_with_fallback(kQnnSystemLibName, _additional_lib_load_path);
    if (!system_lib_handle) {
        QNN_LOG_WARN("can not load QNN library %s, error: %s\n", kQnnSystemLibName, common::dl_error());
        return 1;
    }

    auto * get_providers = common::dl_sym_typed<qnn::pfn_qnnsysteminterface_getproviders *>(
        system_lib_handle, "QnnSystemInterface_getProviders");
    if (!get_providers) {
        QNN_LOG_WARN("can not load QNN symbol QnnSystemInterface_getProviders: %s\n", common::dl_error());
        return 2;
    }

    uint32_t                      num_providers = 0;
    const QnnSystemInterface_t ** provider_list = nullptr;
    Qnn_ErrorHandle_t             error         = get_providers(&provider_list, &num_providers);
    if (error != QNN_SUCCESS) {
        QNN_LOG_WARN("failed to get providers, error %d\n", (int) QNN_GET_ERROR_CODE(error));
        return 3;
    }

    QNN_LOG_DEBUG("num_providers: %d\n", num_providers);
    if (num_providers != _required_num_providers) {
        QNN_LOG_WARN("providers is %d instead of required %d\n", (int) num_providers, (int) _required_num_providers);
        return 4;
    }

    if (!provider_list) {
        QNN_LOG_WARN("can not get providers\n");
        return 5;
    }

    QNN_SYSTEM_INTERFACE_VER_TYPE qnn_system_interface;
    bool                          found_valid_system_interface = false;
    for (size_t idx = 0; idx < num_providers; idx++) {
        if (QNN_SYSTEM_API_VERSION_MAJOR == provider_list[idx]->systemApiVersion.major &&
            QNN_SYSTEM_API_VERSION_MINOR <= provider_list[idx]->systemApiVersion.minor) {
            found_valid_system_interface = true;
            qnn_system_interface         = provider_list[idx]->QNN_SYSTEM_INTERFACE_VER_NAME;
            break;
        }
    }

    if (!found_valid_system_interface) {
        QNN_LOG_WARN("unable to find a valid qnn system interface\n");
        return 6;
    } else {
        QNN_LOG_DEBUG("find a valid qnn system interface\n");
    }

    auto qnn_sys_interface = std::make_shared<qnn::qnn_system_interface>(*provider_list[0], system_lib_handle);
    if (!qnn_sys_interface->is_valid()) {
        QNN_LOG_WARN("failed to create QNN system interface\n");
        return 7;
    }

    _qnn_sys_interface = qnn_sys_interface;
    return 0;
}

bool qnn_instance::load_backend(std::string & lib_path, const QnnSaver_Config_t ** /*saver_config*/) {
    QNN_LOG_DEBUG("lib_path:%s\n", lib_path.c_str());

    auto lib_handle = load_lib_with_fallback(lib_path, _additional_lib_load_path);
    if (!lib_handle) {
        QNN_LOG_WARN("can not open QNN library %s, with error: %s\n", lib_path.c_str(), common::dl_error());
        return false;
    }

    auto get_providers =
        common::dl_sym_typed<qnn::pfn_qnninterface_getproviders *>(lib_handle, "QnnInterface_getProviders");
    if (!get_providers) {
        QNN_LOG_WARN("can not load symbol QnnInterface_getProviders : %s\n", common::dl_error());
        common::dl_unload(lib_handle);
        return false;
    }

    std::uint32_t           num_providers = 0;
    const QnnInterface_t ** provider_list = nullptr;
    auto                    error         = get_providers(&provider_list, &num_providers);
    if (error != QNN_SUCCESS) {
        QNN_LOG_WARN("failed to get providers, error %d\n", (int) QNN_GET_ERROR_CODE(error));
        common::dl_unload(lib_handle);
        return false;
    }
    QNN_LOG_DEBUG("num_providers=%d\n", num_providers);
    if (num_providers != _required_num_providers) {
        QNN_LOG_WARN("providers is %d instead of required %d\n", num_providers, _required_num_providers);
        common::dl_unload(lib_handle);
        return false;
    }

    if (!provider_list) {
        QNN_LOG_WARN("failed to get qnn interface providers\n");
        common::dl_unload(lib_handle);
        return false;
    }
    bool                   found_valid_interface = false;
    QNN_INTERFACE_VER_TYPE qnn_interface;
    for (size_t idx = 0; idx < num_providers; idx++) {
        if (QNN_API_VERSION_MAJOR == provider_list[idx]->apiVersion.coreApiVersion.major &&
            QNN_API_VERSION_MINOR <= provider_list[idx]->apiVersion.coreApiVersion.minor) {
            found_valid_interface = true;
            qnn_interface         = provider_list[idx]->QNN_INTERFACE_VER_NAME;
            break;
        }
    }

    if (!found_valid_interface) {
        QNN_LOG_WARN("unable to find a valid qnn interface\n");
        common::dl_unload(lib_handle);
        return false;
    } else {
        QNN_LOG_DEBUG("find a valid qnn interface\n");
    }

    BackendIdType backend_id          = provider_list[0]->backendId;
    _lib_path_to_backend_id[lib_path] = backend_id;
    if (_loaded_backend.count(backend_id) > 0) {
        QNN_LOG_WARN("lib_path %s is loaded, but backend %d already exists\n", lib_path.c_str(), backend_id);
    }
    _loaded_backend[backend_id] = provider_list[0];
    if (_loaded_lib_handle.count(backend_id) > 0) {
        QNN_LOG_WARN("closing %p\n", _loaded_lib_handle[backend_id]);
        if (!common::dl_unload(_loaded_lib_handle[backend_id])) {
            QNN_LOG_WARN("fail to close %p with error %s\n", _loaded_lib_handle[backend_id], common::dl_error());
        }
    }
    _loaded_lib_handle[backend_id] = lib_handle;
    _backend_id                    = backend_id;

    return true;
}

void qnn_instance::unload_backend() {
    for (auto & it : _loaded_lib_handle) {
        if (!common::dl_unload(it.second)) {
            QNN_LOG_WARN("failed to close QNN backend %d, error %s\n", it.first, common::dl_error());
        }
    }

    _loaded_lib_handle.clear();
    _lib_path_to_backend_id.clear();
    _loaded_backend.clear();
}

const device_caps & get_device_caps(backend_index_type device) {
    return kDeviceCaps[device];
}

}  // namespace qnn
