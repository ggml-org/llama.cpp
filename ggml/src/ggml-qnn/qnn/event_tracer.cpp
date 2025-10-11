
#include "event_tracer.hpp"

#include <HTP/QnnHtpProfile.h>
#include <QnnProfile.h>

#include "logger.hpp"
#include "qnn-lib.hpp"

namespace {

std::string get_duration_string(const QnnProfile_EventData_t & event_data) {
    char time_str[128] = {};
    switch (event_data.unit) {
        case QNN_PROFILE_EVENTUNIT_CYCLES:
            snprintf(time_str, sizeof(time_str), "cycles: %lld", (long long int) event_data.value);
            break;
        case QNN_PROFILE_EVENTUNIT_COUNT:
            snprintf(time_str, sizeof(time_str), "count: %lld", (long long int) event_data.value);
            break;
        case QNN_PROFILE_EVENTUNIT_BYTES:
            snprintf(time_str, sizeof(time_str), "size: %lld bytes", (long long int) event_data.value);
            break;
        case QNN_PROFILE_EVENTUNIT_MICROSEC:
            {
                double duration_ms = event_data.value / 1000.0;
                snprintf(time_str, sizeof(time_str), "duration: %.3f ms", duration_ms);
            }
            break;
        default:
            break;
    }

    return time_str;
}

}  // namespace

namespace qnn {

qnn_event_tracer::qnn_event_tracer(const std::string & prefix, std::shared_ptr<qnn_interface> interface,
                                   Qnn_BackendHandle_t backend_handle, sdk_profile_level level) :
    _interface(interface),
    _prefix(prefix) {
    QnnProfile_Level_t qnn_profile_level = 0;
    switch (level) {
        case sdk_profile_level::PROFILE_BASIC:
            qnn_profile_level = QNN_PROFILE_LEVEL_BASIC;
            break;
        case sdk_profile_level::PROFILE_OP_TRACE:
        case sdk_profile_level::PROFILE_DETAIL:
            qnn_profile_level = QNN_PROFILE_LEVEL_DETAILED;
            break;
        case sdk_profile_level::PROFILE_OFF:
        default:
            QNN_LOG_WARN("[profiler][%s]invalid profile level %d, using PROFILE_OFF\n", _prefix.c_str(), level);
            return;
    }

    auto error = _interface->qnn_profile_create(backend_handle, qnn_profile_level, &_handle);
    if (error != QNN_SUCCESS) {
        QNN_LOG_ERROR("[profiler][%s]failed to create QNN profile_handle. Backend ID %u, error %ld\n", _prefix.c_str(),
                      _interface->get_backend_id(), (long) QNN_GET_ERROR_CODE(error));
        _handle = nullptr;
        return;
    }

    if (level == sdk_profile_level::PROFILE_OP_TRACE) {
        QnnProfile_Config_t qnn_profile_config                     = QNN_PROFILE_CONFIG_INIT;
        qnn_profile_config.option                                  = QNN_PROFILE_CONFIG_OPTION_ENABLE_OPTRACE;
        std::array<const QnnProfile_Config_t *, 2> profile_configs = { &qnn_profile_config, nullptr };
        error = _interface->qnn_profile_set_config(_handle, profile_configs.data());
        if (error != QNN_SUCCESS) {
            QNN_LOG_ERROR("[profiler][%s]failed to set QNN profile event. Backend ID %u, error %ld\n", _prefix.c_str(),
                          _interface->get_backend_id(), (long) QNN_GET_ERROR_CODE(error));
            _interface->qnn_profile_free(_handle);
            _handle = nullptr;
            return;
        }
    }

    QNN_LOG_DEBUG("[profiler][%s]created, Backend ID %u, level %d\n", _prefix.c_str(), _interface->get_backend_id(),
                  level);
}

qnn_event_tracer::~qnn_event_tracer() {
    if (_handle) {
        Qnn_ErrorHandle_t error = _interface->qnn_profile_free(_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_ERROR("[profiler][%s]failed to free QNN profile_handle. Backend ID %u, error %ld\n",
                          _prefix.c_str(), _interface->get_backend_id(), (long) QNN_GET_ERROR_CODE(error));
        }
        _handle = nullptr;
    }
}

void qnn_event_tracer::print_profile_events() {
    const QnnProfile_EventId_t * events_ptr = nullptr;
    uint32_t                     num_events = 0;
    auto                         error      = _interface->qnn_profile_get_events(_handle, &events_ptr, &num_events);
    if (error != QNN_SUCCESS) {
        QNN_LOG_ERROR("[profiler][%s]failed to get QNN profile events. Backend ID %u, error %ld\n", _prefix.c_str(),
                      _interface->get_backend_id(), (long) QNN_GET_ERROR_CODE(error));
        return;
    }

    if (!num_events) {
        QNN_LOG_INFO("[profiler][%s]no QNN profile events\n", _prefix.c_str());
        return;
    }

    QNN_LOG_INFO("[profiler][%s]print_profile_events start ----------------\n", _prefix.c_str());
    // see also: https://github.com/pytorch/executorch/blob/0ccf5093823761cf8ad98c75e5fe81f15ea42366/backends/qualcomm/runtime/backends/QnnProfiler.cpp#L73
    QnnProfile_EventData_t event_data;
    for (uint32_t i = 0; i < num_events; ++i) {
        error = _interface->qnn_profile_get_event_data(events_ptr[i], &event_data);
        if (error != QNN_SUCCESS) {
            QNN_LOG_ERROR("[profiler][%s]failed to get QNN profile event data. Backend ID %u, event[%d], error: %ld\n",
                          _prefix.c_str(), _interface->get_backend_id(), i, (long) QNN_GET_ERROR_CODE(error));
            continue;
        }

        const QnnProfile_EventId_t * sub_events_ptr = nullptr;
        uint32_t                     num_sub_events = 0;
        error = _interface->qnn_profile_get_sub_events(events_ptr[i], &sub_events_ptr, &num_sub_events);
        if (error != QNN_SUCCESS) {
            QNN_LOG_ERROR("[profiler][%s]failed to get QNN profile sub events. Backend ID %u, event[%d], error: %ld\n",
                          _prefix.c_str(), _interface->get_backend_id(), i, (long) QNN_GET_ERROR_CODE(error));
            continue;
        }

        auto duration = get_duration_string(event_data);
        if (!num_sub_events) {
            QNN_LOG_INFO("[profiler][%s]event[%d]: %s, %s\n", _prefix.c_str(), i, event_data.identifier,
                         duration.c_str());
            continue;
        }

        QNN_LOG_INFO("[profiler][%s]event[%d]: %s, sub_count: %d, start -------------\n", _prefix.c_str(), i,
                     event_data.identifier, num_sub_events);
        QnnProfile_EventData_t sub_event_data;
        for (std::uint32_t j = 0; j < num_sub_events; ++j) {
            error = _interface->qnn_profile_get_event_data(sub_events_ptr[j], &sub_event_data);
            if (error != QNN_SUCCESS) {
                QNN_LOG_ERROR(
                    "[profiler][%s]failed to get QNN profile sub event data. Backend ID %u, event[%d], sub_event[%d], "
                    "error: %ld\n",
                    _prefix.c_str(), _interface->get_backend_id(), i, j, (long) QNN_GET_ERROR_CODE(error));
                continue;
            }

            if (sub_event_data.type != QNN_PROFILE_EVENTTYPE_NODE) {
                QNN_LOG_DEBUG("[profiler][%s]sub_event[%d]%s, type %d, skipping\n", _prefix.c_str(), j,
                              sub_event_data.identifier, sub_event_data.type);
                continue;
            }

            auto sub_duration = get_duration_string(sub_event_data);
            QNN_LOG_INFO("[profiler][%s]sub_event[%d]: %s, %s\n", _prefix.c_str(), j, sub_event_data.identifier,
                         sub_duration.c_str());
        }

        QNN_LOG_INFO("[profiler][%s]event[%d]: %s, %s, end --------------\n", _prefix.c_str(), i, event_data.identifier,
                     duration.c_str());
    }

    QNN_LOG_INFO("[profiler][%s]print_profile_events end -----------------\n", _prefix.c_str());
}

}  // namespace qnn
