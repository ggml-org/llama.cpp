#pragma once

#include <QnnCommon.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "logger.hpp"
#include "profiler.hpp"
#include "qnn-types.hpp"

namespace qnn {

// forward declaration of qnn_interface
class qnn_interface;

class qnn_event_tracer {
  public:
    // ref:
    //   https://github.com/pytorch/executorch/blob/ae3d558d5e6aa04fc52a3065399fe6a773702f52/backends/qualcomm/serialization/qc_schema.py#L53
    //   https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/overview.html#supported-snapdragon-devices
    enum sdk_profile_level { PROFILE_OFF = 0, PROFILE_BASIC, PROFILE_DETAIL, PROFILE_OP_TRACE };

    explicit qnn_event_tracer(const std::string & prefix, std::shared_ptr<qnn_interface> interface,
                              Qnn_BackendHandle_t backend_handle, sdk_profile_level level);
    ~qnn_event_tracer();

    Qnn_ProfileHandle_t get_handle() const { return _handle; }

    void print_profile_events();

  private:
    std::shared_ptr<qnn_interface> _interface;
    Qnn_ProfileHandle_t            _handle = nullptr;
    std::string                    _prefix;

    DISABLE_COPY(qnn_event_tracer);
    DISABLE_MOVE(qnn_event_tracer);
};

using qnn_event_tracer_ptr = std::shared_ptr<qnn_event_tracer>;

}  // namespace qnn
