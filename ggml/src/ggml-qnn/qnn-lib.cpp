
#include "qnn-lib.hpp"

#include <filesystem>

#if defined(__linux__)
#    include <unistd.h>
#endif

namespace {

#ifdef _WIN32
constexpr const char * kQnnSystemLibName = "QnnSystem.dll";
constexpr const char * kQnnRpcLibName    = "libcdsprpc.dll";
#else
constexpr const char * kQnnSystemLibName = "libQnnSystem.so";
constexpr const char * kQnnRpcLibName    = "libcdsprpc.so";

#endif

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

qnn::dl_handler_t load_lib_with_fallback(const std::string & lib_path, const std::string & load_directory) {
    std::filesystem::path full_path(load_directory);
    full_path /= std::filesystem::path(lib_path).filename();
    auto handle = qnn::dl_load(full_path.string());
    if (!handle) {
        QNN_LOG_WARN("failed to load %s, fallback to %s\n", full_path.c_str(), lib_path.c_str());
        handle = qnn::dl_load(lib_path);
    }

    return handle;
}

}  // namespace

namespace qnn {

qnn_system_interface::qnn_system_interface(const QnnSystemInterface_t & qnn_sys_interface, dl_handler_t lib_handle) :
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
        if (!dl_unload(_lib_handle)) {
            QNN_LOG_WARN("failed to close QnnSystem library, error %s\n", dl_error());
        }
    } else {
        QNN_LOG_WARN("system lib handle is null\n");
    }
}

qnn_instance::qnn_instance(const std::string & lib_path, const std::string & backend_lib_name) :
    _additional_lib_load_path(lib_path),
    _backend_lib_name(std::move(backend_lib_name)) {
    if (set_qnn_lib_search_path(lib_path)) {
        QNN_LOG_DEBUG("[%s] set_qnn_lib_search_path succeed\n", _backend_lib_name.c_str());
    } else {
        QNN_LOG_ERROR("[%s] set_qnn_lib_search_path failed\n", _backend_lib_name.c_str());
    }
}

int qnn_instance::qnn_init(const QnnSaver_Config_t ** saver_config) {
    BackendIdType backend_id = QNN_BACKEND_ID_NULL;
    QNN_LOG_DEBUG("enter qnn_init\n");

    std::lock_guard<std::mutex> lock(_init_mutex);
    if (load_system() != 0) {
        QNN_LOG_WARN("failed to load QNN system lib\n");
        return 1;
    } else {
        QNN_LOG_DEBUG("load QNN system lib successfully\n");
    }

    std::string backend_lib_path = _backend_lib_name;
    if (_lib_path_to_backend_id.count(backend_lib_path) == 0) {
        if (load_backend(backend_lib_path, saver_config) != 0) {
            QNN_LOG_WARN("failed to load QNN backend\n");
            return 2;
        }
    }

    backend_id = _lib_path_to_backend_id[backend_lib_path];
    if (_loaded_backend.count(backend_id) == 0 || _loaded_lib_handle.count(backend_id) == 0) {
        QNN_LOG_WARN(
            "library %s is loaded but loaded backend count=%zu, "
            "loaded lib_handle count=%zu",
            backend_lib_path.c_str(), _loaded_backend.count(backend_id), _loaded_lib_handle.count(backend_id));
        return 3;
    }

    _qnn_interface = std::make_shared<qnn_interface>(*_loaded_backend[backend_id]);
    _qnn_interface->qnn_log_create(qnn::sdk_logcallback, _qnn_log_level, &_qnn_log_handle);
    if (!_qnn_log_handle) {
        // NPU backend not work on Qualcomm SoC equipped low-end phone
        QNN_LOG_WARN("why failed to initialize qnn log\n");
        return 4;
    } else {
        QNN_LOG_DEBUG("initialize qnn log successfully\n");
    }

    std::vector<const QnnBackend_Config_t *> temp_backend_config;
    _qnn_interface->qnn_backend_create(
        _qnn_log_handle, temp_backend_config.empty() ? nullptr : temp_backend_config.data(), &_qnn_backend_handle);
    if (!_qnn_backend_handle) {
        QNN_LOG_WARN("why failed to initialize qnn backend\n");
        return 5;
    } else {
        QNN_LOG_DEBUG("initialize qnn backend successfully\n");
    }

    auto qnn_status = _qnn_interface->qnn_property_has_capability(QNN_PROPERTY_GROUP_DEVICE);
    if (QNN_PROPERTY_NOT_SUPPORTED == qnn_status) {
        QNN_LOG_WARN("device property is not supported\n");
    }
    if (QNN_PROPERTY_ERROR_UNKNOWN_KEY == qnn_status) {
        QNN_LOG_WARN("device property is not known to backend\n");
    }

    qnn_status = QNN_SUCCESS;
    if (_backend_lib_name.find("Htp") != _backend_lib_name.npos) {
        const QnnDevice_PlatformInfo_t * p_info = nullptr;
        qnn_status                              = _qnn_interface->qnn_device_get_platform_info(nullptr, &p_info);
        if (qnn_status == QNN_SUCCESS) {
            QNN_LOG_INFO("device counts %d\n", p_info->v1.numHwDevices);
            QnnDevice_HardwareDeviceInfo_t *         infos    = p_info->v1.hwDevices;
            QnnHtpDevice_OnChipDeviceInfoExtension_t chipinfo = {};
            for (uint32_t i = 0; i < p_info->v1.numHwDevices; i++) {
                QNN_LOG_INFO("deviceID:%d, deviceType:%d, numCores %d\n", (int) infos[i].v1.deviceId,
                             (int) infos[i].v1.deviceType, (int) infos[i].v1.numCores);
                QnnDevice_DeviceInfoExtension_t devinfo = infos[i].v1.deviceInfoExtension;
                chipinfo                                = devinfo->onChipDevice;
                size_t htp_arch                         = (size_t) chipinfo.arch;
                QNN_LOG_INFO("htp_type:%d(%s)\n", devinfo->devType,
                             (devinfo->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP) ? "ON_CHIP" : "");
                QNN_LOG_INFO("qualcomm soc_model:%d(%s), htp_arch:%d(%s), vtcm_size:%d MB\n", (int) chipinfo.socModel,
                             qnn::get_chipset_desc(chipinfo.socModel), (int) htp_arch, qnn::get_htparch_desc(htp_arch),
                             (int) chipinfo.vtcmSize);
                _soc_info = { chipinfo.socModel, htp_arch, chipinfo.vtcmSize };
            }
            _qnn_interface->qnn_device_free_platform_info(nullptr, p_info);
        } else {
            // For emulator, we can't get platform info
            QNN_LOG_WARN("failed to get platform info, are we in emulator?\n");
            _soc_info = { NONE, UNKNOWN_SM, 0 };
        }

        QnnHtpDevice_CustomConfig_t soc_customconfig;
        soc_customconfig.option   = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
        soc_customconfig.socModel = _soc_info.soc_model;
        QnnDevice_Config_t soc_devconfig;
        soc_devconfig.option       = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
        soc_devconfig.customConfig = &soc_customconfig;

        QnnHtpDevice_CustomConfig_t arch_customconfig;
        arch_customconfig.option        = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
        arch_customconfig.arch.arch     = (QnnHtpDevice_Arch_t) _soc_info.htp_arch;
        arch_customconfig.arch.deviceId = 0;  // Id of device to be used. 0 will use by default.
        QnnDevice_Config_t arch_devconfig;
        arch_devconfig.option       = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
        arch_devconfig.customConfig = &arch_customconfig;

        const QnnDevice_Config_t * p_deviceconfig[] = { &soc_devconfig, &arch_devconfig, nullptr };
        qnn_status = _qnn_interface->qnn_device_create(_qnn_log_handle, p_deviceconfig, &_qnn_device_handle);
    } else {
        qnn_status = _qnn_interface->qnn_device_create(_qnn_log_handle, nullptr, &_qnn_device_handle);
    }
    if (QNN_SUCCESS != qnn_status && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnn_status) {
        QNN_LOG_WARN("failed to create QNN device\n");
    } else {
        QNN_LOG_INFO("create QNN device successfully\n");
    }

    if (_profile_level != sdk_profile_level::profile_off) {
        QNN_LOG_INFO("profiling turned on; level = %d\n", _profile_level);
        auto profile_level =
            _profile_level == sdk_profile_level::profile_detail ? QNN_PROFILE_LEVEL_DETAILED : QNN_PROFILE_LEVEL_BASIC;

        if (QNN_PROFILE_NO_ERROR !=
            _qnn_interface->qnn_profile_create(_qnn_backend_handle, profile_level, &_qnn_profile_handle)) {
            QNN_LOG_WARN("unable to create profile handle in the backend\n");
            return 6;
        } else {
            QNN_LOG_DEBUG("initialize qnn profile successfully\n");
        }
    }

    _rpc_lib_handle = load_lib_with_fallback(kQnnRpcLibName, _additional_lib_load_path);
    if (_rpc_lib_handle) {
        _pfn_rpc_mem_alloc = reinterpret_cast<qnn::pfn_rpc_mem_alloc>(dl_sym(_rpc_lib_handle, "rpcmem_alloc"));
        _pfn_rpc_mem_free  = reinterpret_cast<qnn::pfn_rpc_mem_free>(dl_sym(_rpc_lib_handle, "rpcmem_free"));
        _pfn_rpc_mem_to_fd = reinterpret_cast<qnn::pfn_rpc_mem_to_fd>(dl_sym(_rpc_lib_handle, "rpcmem_to_fd"));
        if (!_pfn_rpc_mem_alloc || !_pfn_rpc_mem_free || !_pfn_rpc_mem_to_fd) {
            QNN_LOG_WARN("unable to access symbols in QNN RPC lib. error: %s\n", dl_error());
            dl_unload(_rpc_lib_handle);
            return 9;
        }

        _pfn_rpc_mem_init   = reinterpret_cast<qnn::pfn_rpc_mem_init>(dl_sym(_rpc_lib_handle, "rpcmem_init"));
        _pfn_rpc_mem_deinit = reinterpret_cast<qnn::pfn_rpc_mem_deinit>(dl_sym(_rpc_lib_handle, "rpcmem_deinit"));
        if (_pfn_rpc_mem_init) {
            _pfn_rpc_mem_init();
        }

        _rpcmem_initialized = true;
        QNN_LOG_DEBUG("load rpcmem lib successfully\n");
    } else {
        QNN_LOG_WARN("failed to load qualcomm rpc lib, skipping, error:%s\n", dl_error());
    }

    /* TODO: not used, keep it for further usage
             QnnContext_Config_t qnn_context_config = QNN_CONTEXT_CONFIG_INIT;
             qnn_context_config.priority = QNN_PRIORITY_DEFAULT;
             const QnnContext_Config_t * context_configs[] = {&qnn_context_config, nullptr};
    */
    _qnn_interface->qnn_context_create(_qnn_backend_handle, _qnn_device_handle, nullptr, &_qnn_context_handle);
    if (nullptr == _qnn_context_handle) {
        QNN_LOG_WARN("why failed to initialize qnn context\n");
        return 10;
    } else {
        QNN_LOG_DEBUG("initialize qnn context successfully\n");
    }

    if (_backend_lib_name.find("Htp") != _backend_lib_name.npos) {
        // TODO: faster approach to probe the accurate capacity of rpc ion memory
        size_t    candidate_size = 0;
        uint8_t * rpc_buffer     = nullptr;
        const int size_in_mb     = (1 << 20);
        size_t    probe_slots[]  = { 1024, 1536, 2048 - 48, 2048 };
        size_t    probe_counts   = sizeof(probe_slots) / sizeof(size_t);
        for (size_t idx = 0; idx < probe_counts; idx++) {
            rpc_buffer = static_cast<uint8_t *>(alloc_rpcmem(probe_slots[idx] * size_in_mb, sizeof(void *)));
            if (!rpc_buffer) {
                QNN_LOG_DEBUG("alloc rpcmem %d (MB) failure, %s\n", (int) probe_slots[idx], strerror(errno));
                break;
            } else {
                candidate_size = probe_slots[idx];
                free_rpcmem(rpc_buffer);
                rpc_buffer = nullptr;
            }
        }

        _rpcmem_capacity = std::max(candidate_size, _rpcmem_capacity);
        QNN_LOG_INFO("capacity of QNN rpc ion memory is about %d MB\n", (int) _rpcmem_capacity);

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

    return 0;
}

int qnn_instance::qnn_finalize() {
    int               ret_status = 0;
    Qnn_ErrorHandle_t error      = QNN_SUCCESS;

    if (_rpc_lib_handle) {
        if (_pfn_rpc_mem_deinit) {
            _pfn_rpc_mem_deinit();
            _pfn_rpc_mem_deinit = nullptr;
        }

        if (dl_unload(_rpc_lib_handle)) {
            QNN_LOG_DEBUG("succeed to close rpcmem lib\n");
        } else {
            QNN_LOG_WARN("failed to unload qualcomm's rpc lib, error:%s\n", dl_error());
        }
    }

    if (_backend_lib_name.find("Htp") != _backend_lib_name.npos) {
        _qnn_htp_perfinfra->destroyPowerConfigId(_qnn_power_configid);
    }

    if (_qnn_context_handle) {
        error = _qnn_interface->qnn_context_free(_qnn_context_handle, _qnn_profile_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to free QNN context_handle: ID %u, error %d\n", _qnn_interface->get_backend_id(),
                         (int) QNN_GET_ERROR_CODE(error));
        }
        _qnn_context_handle = nullptr;
    }

    if (_qnn_profile_handle) {
        error = _qnn_interface->qnn_profile_free(_qnn_profile_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to free QNN profile_handle: ID %u, error %d\n", _qnn_interface->get_backend_id(),
                         (int) QNN_GET_ERROR_CODE(error));
        }
        _qnn_profile_handle = nullptr;
    }

    if (_qnn_device_handle) {
        error = _qnn_interface->qnn_device_free(_qnn_device_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to free QNN device_handle: ID %u, error %d\n", _qnn_interface->get_backend_id(),
                         (int) QNN_GET_ERROR_CODE(error));
        }
        _qnn_device_handle = nullptr;
    }

    if (_qnn_backend_handle) {
        error = _qnn_interface->qnn_backend_free(_qnn_backend_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to free QNN backend_handle: ID %u, error %d\n", _qnn_interface->get_backend_id(),
                         (int) QNN_GET_ERROR_CODE(error));
        }
        _qnn_backend_handle = nullptr;
    }

    if (_qnn_log_handle) {
        error = _qnn_interface->qnn_log_free(_qnn_log_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to free QNN log_handle: ID %u, error %d\n", _qnn_interface->get_backend_id(),
                         (int) QNN_GET_ERROR_CODE(error));
        }
        _qnn_log_handle = nullptr;
    }

    unload_backend();

    _qnn_sys_interface.reset();

    return ret_status;
}

int qnn_instance::load_system() {
    QNN_LOG_DEBUG("[%s]lib: %s\n", _backend_lib_name.c_str(), kQnnSystemLibName);
    auto system_lib_handle = load_lib_with_fallback(kQnnSystemLibName, _additional_lib_load_path);
    if (!system_lib_handle) {
        QNN_LOG_WARN("can not load QNN library %s, error: %s\n", kQnnSystemLibName, dl_error());
        return 1;
    }

    auto * get_providers =
        dl_sym_typed<qnn::pfn_qnnsysteminterface_getproviders *>(system_lib_handle, "QnnSystemInterface_getProviders");
    if (!get_providers) {
        QNN_LOG_WARN("can not load QNN symbol QnnSystemInterface_getProviders: %s\n", dl_error());
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

int qnn_instance::load_backend(std::string & lib_path, const QnnSaver_Config_t ** /*saver_config*/) {
    Qnn_ErrorHandle_t error = QNN_SUCCESS;
    QNN_LOG_DEBUG("lib_path:%s\n", lib_path.c_str());

    auto lib_handle = load_lib_with_fallback(lib_path, _additional_lib_load_path);
    if (!lib_handle) {
        QNN_LOG_WARN("can not open QNN library %s, with error: %s\n", lib_path.c_str(), dl_error());
        return 1;
    }

    auto get_providers = dl_sym_typed<qnn::pfn_qnninterface_getproviders *>(lib_handle, "QnnInterface_getProviders");
    if (!get_providers) {
        QNN_LOG_WARN("can not load symbol QnnInterface_getProviders : %s\n", dl_error());
        return 2;
    }

    std::uint32_t           num_providers = 0;
    const QnnInterface_t ** provider_list = nullptr;
    error                                 = get_providers(&provider_list, &num_providers);
    if (error != QNN_SUCCESS) {
        QNN_LOG_WARN("failed to get providers, error %d\n", (int) QNN_GET_ERROR_CODE(error));
        return 3;
    }
    QNN_LOG_DEBUG("num_providers=%d\n", num_providers);
    if (num_providers != _required_num_providers) {
        QNN_LOG_WARN("providers is %d instead of required %d\n", num_providers, _required_num_providers);
        return 4;
    }

    if (!provider_list) {
        QNN_LOG_WARN("failed to get qnn interface providers\n");
        return 5;
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
        return 6;
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
        if (!dl_unload(_loaded_lib_handle[backend_id])) {
            QNN_LOG_WARN("fail to close %p with error %s\n", _loaded_lib_handle[backend_id], dl_error());
        }
    }
    _loaded_lib_handle[backend_id] = lib_handle;
    _backend_id                    = backend_id;

    return 0;
}

int qnn_instance::unload_backend() {
    for (auto & it : _loaded_lib_handle) {
        if (!dl_unload(it.second)) {
            QNN_LOG_WARN("failed to close QNN backend %d, error %s\n", it.first, dl_error());
        }
    }

    _loaded_lib_handle.clear();
    _lib_path_to_backend_id.clear();
    _loaded_backend.clear();

    return 0;
}

}  // namespace qnn
