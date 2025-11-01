
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "buffer.hpp"
#include "ggml-qnn.h"
#include "logger.hpp"
#include "qnn-lib.hpp"
#include "utils.hpp"

namespace qnn {

static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS should be 4");

class ggml_qnn_tensor : public std::enable_shared_from_this<ggml_qnn_tensor> {
  public:
    typedef enum _tensor_type { INPUT, OUTPUT, INTERMEDIATE, PARAMETER, BIDIRECTION } tensor_type_t;

    explicit ggml_qnn_tensor(tensor_type_t tensor_type, const std::string & name,
                             const qnn_dimension_array_t & dimensions, Qnn_DataType_t data_type, int rank,
                             backend_index_type device, Qnn_GraphHandle_t graph_handle, qnn_instance_ptr qnn_instance) :
        _tensor_name(name),
        _device(device),
        _qnn_instance(qnn_instance),
        _graph_handle(graph_handle) {
        if (!_tensor_name.empty()) {
            QNN_TENSOR_SET_NAME(_qnn_tensor, _tensor_name.c_str());
        }

        _dimensions = dimensions;
        QNN_TENSOR_SET_DIMENSIONS(_qnn_tensor, _dimensions.data());
        QNN_TENSOR_SET_DATA_FORMAT(_qnn_tensor, QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER);
        update_params_from_ggml_tensor(tensor_type, data_type, rank);
        QNN_LOG_DEBUG("[%s][%s]created, rank: %d, dims: [%d, %d, %d, %d], type: %s\n", get_backend_name(device),
                      _tensor_name.c_str(), rank, (int) _dimensions[0], (int) _dimensions[1], (int) _dimensions[2],
                      (int) _dimensions[3], qnn_datatype_to_string(data_type));
    }

    explicit ggml_qnn_tensor(tensor_type_t tensor_type, const std::string & name,
                             const ggml_dimension_array_t & dimensions, ggml_type data_type, int rank,
                             backend_index_type device, Qnn_GraphHandle_t graph_handle, qnn_instance_ptr qnn_instance) :
        ggml_qnn_tensor(tensor_type, name, get_internal_dimension(dimensions, rank),
                        qnn_datatype_from_ggml_datatype(data_type), rank, device, graph_handle, qnn_instance) {}

    ~ggml_qnn_tensor() {
        _rpc_buffer.reset();
        unbind();
    }

    bool set_data_buffer(const uint8_t * buffer, const size_t buffer_size) {
        auto qnn_buffer = std::make_shared<qnn_mem_buffer>(buffer, buffer_size);
        if (bind_buffer_impl(qnn_buffer)) {
            return true;
        }

        _can_unbind = false;
        return false;
    }

    bool set_data_buffer(qnn_buffer_ptr buffer) {
        if (bind_buffer_impl(buffer)) {
            return true;
        }

        _can_unbind = false;
        return false;
    }

    bool alloc_qnn_tensor_id() {
        if (QNN_TENSOR_GET_ID(_qnn_tensor)) {
            QNN_LOG_DEBUG("[%s]tensor already has a id: %d\n", _tensor_name.c_str(), QNN_TENSOR_GET_ID(_qnn_tensor));
            return true;
        }

        Qnn_Tensor_t qnn_tensor    = _qnn_tensor;
        auto         qnn_interface = _qnn_instance->get_qnn_interface();
        auto         error         = qnn_interface->qnn_tensor_create_graph_tensor(_graph_handle, &qnn_tensor);
        if (error != QNN_SUCCESS) {
            QNN_LOG_ERROR("[%s]allocate id failed, error: %s\n", _tensor_name.c_str(), get_qnn_error_string(error));
            return false;
        }

        QNN_TENSOR_SET_ID(_qnn_tensor, QNN_TENSOR_GET_ID(qnn_tensor));
        QNN_LOG_DEBUG("[%s][%s]allocated id: %d, rank: %d\n", get_backend_name(_device), _tensor_name.c_str(),
                      QNN_TENSOR_GET_ID(qnn_tensor), QNN_TENSOR_GET_RANK(qnn_tensor));
        return true;
    }

    bool bind_ggml_tensor(ggml_tensor * tensor, qnn_buffer_ptr buffer) {
        if (!_can_unbind) {
            QNN_LOG_DEBUG("[%s]already has buffer storage, skip bind\n", _tensor_name.c_str());
            return true;
        }

#ifndef NDEBUG
        if (tensor->view_src) {
            auto * src = tensor->view_src;
            QNN_LOG_DEBUG("[%s]tensor(%s_%dx%dx%dx%d) is a view, src: %s_%dx%dx%dx%d\n", get_backend_name(_device),
                          tensor->name, (int) tensor->ne[0], (int) tensor->ne[1], (int) tensor->ne[2],
                          (int) tensor->ne[3], src->name, (int) src->ne[0], (int) src->ne[1], (int) src->ne[2],
                          (int) src->ne[3]);
        }
#endif

        if (!buffer) {
            buffer =
                std::make_shared<qnn_mem_buffer_slice>(reinterpret_cast<uint8_t *>(tensor->data), ggml_nbytes(tensor));
            QNN_LOG_DEBUG("[%s][%s]attach buffer to tensor(%s), size: %d\n", get_backend_name(_device),
                          _tensor_name.c_str(), tensor->name, (int) buffer->get_size());
        }
        if (!bind_buffer_impl(buffer)) {
            QNN_LOG_WARN("[%s]failed to bind ggml tensor(%s)\n", _tensor_name.c_str(), ggml_get_name(tensor));
            return false;
        }

        QNN_LOG_DEBUG("[%s][%s]bind to ggml tensor(%s)\n", get_backend_name(_device), _tensor_name.c_str(),
                      ggml_get_name(tensor));
        tensor->extra = this;
        _ggml_tensor  = tensor;
        return true;
    }

    bool unbind() {
        if (!_graph_handle) {
            QNN_LOG_WARN("[%s]not bound to any graph\n", _tensor_name.c_str());
            return false;
        }

        if (!_buffer) {
            QNN_LOG_DEBUG("[%s]unbind to ggml tensor\n", _tensor_name.c_str());
            return true;
        }

        if (!read_from_qnn_tensor()) {
            QNN_LOG_WARN("[%s]read from qnn tensor failed\n", _tensor_name.c_str());
            return false;
        }

        if (!_can_unbind) {
            QNN_LOG_DEBUG("[%s]already has buffer storage, stop unbind\n", _tensor_name.c_str());
            return true;
        }

        if (!should_use_mem_handle()) {
            QNN_TENSOR_SET_MEM_TYPE(_qnn_tensor, QNN_TENSORMEMTYPE_RAW);
            Qnn_ClientBuffer_t client_buf = {};
            QNN_TENSOR_SET_CLIENT_BUF(_qnn_tensor, client_buf);
            QNN_LOG_DEBUG("[%s]clear client buffer\n", _tensor_name.c_str());
        }

        QNN_LOG_DEBUG("[%s][%s]unbind from buffer: %p, size: %d\n", get_backend_name(_device), _tensor_name.c_str(),
                      (void *) _buffer->get_buffer(), (int) _buffer->get_size());
        _buffer.reset();

        if (_ggml_tensor) {
            _ggml_tensor->extra = nullptr;
            _ggml_tensor        = nullptr;
        }

        return true;
    }

    const Qnn_Tensor_t & get_qnn_tensor() const { return _qnn_tensor; }

    Qnn_DataType_t get_data_type() const { return QNN_TENSOR_GET_DATA_TYPE(_qnn_tensor); }

    const qnn_dimension_array_t & get_dimensions() const { return _dimensions; }

    uint32_t get_rank() const { return QNN_TENSOR_GET_RANK(_qnn_tensor); }

    uint32_t get_qnn_tensor_id() const { return QNN_TENSOR_GET_ID(_qnn_tensor); }

    const std::string & get_tensor_name() const { return _tensor_name; }

  private:
    bool bind_buffer_impl(qnn_buffer_ptr buffer) {
        if (_buffer) {
            if (_buffer != buffer) {
                QNN_LOG_WARN("[%s]has been bound to another buffer %p\n", _tensor_name.c_str(),
                             (void *) _buffer->get_buffer());
                return false;
            }

            QNN_LOG_DEBUG("[%s]already bound to same ggml tensor %p\n", _tensor_name.c_str(),
                          (void *) _buffer->get_buffer());
            return true;
        }

        if (QNN_TENSOR_GET_TYPE(_qnn_tensor) == QNN_TENSOR_TYPE_NATIVE) {
            QNN_LOG_DEBUG("[%s]tensor type(%d) not READ/WRITE, skipping\n", _tensor_name.c_str(),
                          (int) QNN_TENSOR_TYPE_NATIVE);
            return true;
        }

        if (should_use_mem_handle()) {
            if (!_rpc_buffer) {
                auto rpc_buffer = std::make_shared<qnn_rpc_buffer>(
                    _qnn_instance, buffer->get_size(), QNN_TENSOR_GET_RANK(_qnn_tensor),
                    QNN_TENSOR_GET_DIMENSIONS(_qnn_tensor), QNN_TENSOR_GET_DATA_TYPE(_qnn_tensor));
                if (!rpc_buffer->is_valid()) {
                    QNN_LOG_WARN("[%s][%s]alloc rpc mem failed\n", get_backend_name(_device), _tensor_name.c_str());
                    return false;
                }

                _rpc_buffer = std::move(rpc_buffer);
            }

            QNN_TENSOR_SET_MEM_TYPE(_qnn_tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
            auto mem_handle = _rpc_buffer->get_mem_handle();
            if (!mem_handle) {
                QNN_LOG_WARN("[%s][%s]can't find rpcmem from qnn mem handle\n", get_backend_name(_device),
                             _tensor_name.c_str());
                return false;
            }

            QNN_TENSOR_SET_MEM_HANDLE(_qnn_tensor, mem_handle);
            QNN_LOG_DEBUG("[%s][%s]use mem handle %p\n", get_backend_name(_device), _tensor_name.c_str(),
                          QNN_TENSOR_GET_MEM_HANDLE(_qnn_tensor));
        } else {
            QNN_TENSOR_SET_MEM_TYPE(_qnn_tensor, QNN_TENSORMEMTYPE_RAW);
            Qnn_ClientBuffer_t client_buf = { buffer->get_buffer(), (uint32_t) buffer->get_size() };
            QNN_TENSOR_SET_CLIENT_BUF(_qnn_tensor, client_buf);
            QNN_LOG_DEBUG("[%s][%s]use client buffer %p size %d\n", get_backend_name(_device), _tensor_name.c_str(),
                          client_buf.data, (int) client_buf.dataSize);
        }

        _buffer = buffer;

        if (!write_to_qnn_tensor()) {
            QNN_LOG_WARN("[%s]write to qnn tensor failed\n", _tensor_name.c_str());
            return false;
        }

        QNN_LOG_DEBUG("[%s][%s]bind to buffer: %p, size: %d\n", get_backend_name(_device), _tensor_name.c_str(),
                      (void *) buffer->get_buffer(), (int) buffer->get_size());
        return true;
    }

    bool write_to_qnn_tensor() {
        auto tensor_type = QNN_TENSOR_GET_TYPE(_qnn_tensor);
        if (tensor_type != QNN_TENSOR_TYPE_APP_WRITE && tensor_type != QNN_TENSOR_TYPE_APP_READWRITE) {
            QNN_LOG_DEBUG("[%s]tensor type(%d) not WRITE\n", _tensor_name.c_str(), (int) tensor_type);
            return true;
        }

        if (_rpc_buffer) {
            memcpy(_rpc_buffer->get_buffer(), _buffer->get_buffer(), _buffer->get_size());
            // For CPU and GPU, the data is already in the tensor.
            QNN_LOG_DEBUG("[%s][%s]write buffer(%p) to rpc buffer(%p)\n", get_backend_name(_device),
                          _tensor_name.c_str(), (void *) _buffer->get_buffer(), (void *) _rpc_buffer->get_buffer());
        }

        return true;
    }

    bool read_from_qnn_tensor() {
        auto tensor_type = QNN_TENSOR_GET_TYPE(_qnn_tensor);
        if (tensor_type != QNN_TENSOR_TYPE_APP_READ && tensor_type != QNN_TENSOR_TYPE_APP_READWRITE) {
            QNN_LOG_DEBUG("[%s]tensor type(%d) not READ\n", _tensor_name.c_str(), (int) tensor_type);
            return true;
        }

        if (_rpc_buffer) {
            memcpy(_buffer->get_buffer(), _rpc_buffer->get_buffer(), _buffer->get_size());
            // For CPU and GPU, the data is already in the tensor.
            QNN_LOG_DEBUG("[%s][%s]read buffer(%p) from rpc buffer(%p)\n", get_backend_name(_device),
                          _tensor_name.c_str(), (void *) _buffer->get_buffer(), (void *) _rpc_buffer->get_buffer());
        }

        return true;
    }

    void update_params_from_ggml_tensor(tensor_type_t tensor_type, Qnn_DataType_t data_type, int rank) {
        QNN_TENSOR_SET_DATA_TYPE(_qnn_tensor, data_type);
        // TODO: set the quantizeParams base on the tensor type

        QNN_TENSOR_SET_RANK(_qnn_tensor, (uint32_t) rank);
        QNN_TENSOR_SET_MEM_TYPE(_qnn_tensor, QNN_TENSORMEMTYPE_RAW);
        Qnn_ClientBuffer_t client_buf = {};
        QNN_TENSOR_SET_CLIENT_BUF(_qnn_tensor, client_buf);

        Qnn_TensorType_t new_tensor_type;
        switch (tensor_type) {
            case INPUT:
                new_tensor_type = QNN_TENSOR_TYPE_APP_WRITE;
                break;
            case OUTPUT:
                new_tensor_type = QNN_TENSOR_TYPE_APP_READ;
                break;
            case PARAMETER:
                new_tensor_type = QNN_TENSOR_TYPE_STATIC;
                break;
            case BIDIRECTION:
                new_tensor_type = QNN_TENSOR_TYPE_APP_READWRITE;
                break;
            case INTERMEDIATE:
            default:
                new_tensor_type = QNN_TENSOR_TYPE_NATIVE;
                break;
        }
        QNN_TENSOR_SET_TYPE(_qnn_tensor, new_tensor_type);
        QNN_LOG_DEBUG("[%s][%s]new_tensor_type %s\n", get_backend_name(_device), _tensor_name.c_str(),
                      get_qnn_tensor_type_name(new_tensor_type));
    }

    bool should_use_mem_handle() const {
        // TODO: figure out how to set rpc mem to multiple tensor
        return false;
    }

    std::string           _tensor_name;
    qnn_buffer_ptr        _buffer;
    bool                  _can_unbind = true;
    backend_index_type    _device;
    qnn_instance_ptr      _qnn_instance;
    Qnn_Tensor_t          _qnn_tensor   = qnn_tensor_init(kDefaultQnnTensorVersion);
    qnn_dimension_array_t _dimensions   = {};
    Qnn_GraphHandle_t     _graph_handle = nullptr;
    qnn_buffer_ptr        _rpc_buffer;
    ggml_tensor *         _ggml_tensor = nullptr;

    DISABLE_COPY(ggml_qnn_tensor);
    DISABLE_MOVE(ggml_qnn_tensor);
};

using qnn_tensor_ptr_t    = std::shared_ptr<ggml_qnn_tensor>;
using qnn_tensor_array_t  = std::vector<qnn_tensor_ptr_t>;
using ggml_tensor_array_t = std::vector<ggml_tensor *>;

inline qnn_tensor_ptr_t get_qnn_tensor_ptr(ggml_tensor * ggml_tensor) {
    return ggml_tensor->extra ? reinterpret_cast<ggml_qnn_tensor *>(ggml_tensor->extra)->shared_from_this() :
                                qnn_tensor_ptr_t();
}

inline int get_ggml_tensors_max_rank(const qnn::ggml_tensor_array_t & tensors) {
    int max_rank = 0;
    for (auto tensor : tensors) {
        max_rank = std::max(max_rank, ggml_n_dims(tensor));
    }

    return max_rank;
}

inline bool bind_tensors_with_custom_buffers(const ggml_tensor_array_t &   ggml_tensors,
                                             std::vector<qnn_buffer_ptr> & buffers,
                                             qnn_tensor_array_t &          tensor_wrappers,
                                             std::vector<Qnn_Tensor_t> &   qnn_tensors) {
    GGML_ASSERT(tensor_wrappers.size() == ggml_tensors.size());
    GGML_ASSERT(buffers.size() == ggml_tensors.size());
    qnn_tensors.resize(ggml_tensors.size());
    for (size_t i = 0; i < ggml_tensors.size(); i++) {
        auto * ggml_tensor = ggml_tensors[i];
        if (!tensor_wrappers[i]->bind_ggml_tensor(ggml_tensor, buffers[i])) {
            QNN_LOG_ERROR("bind tensor %s failed\n", ggml_get_name(ggml_tensor));
            return false;
        }

        qnn_tensors[i] = tensor_wrappers[i]->get_qnn_tensor();
    }

    return true;
}

inline bool bind_tensors(const ggml_tensor_array_t & ggml_tensors, qnn_tensor_array_t & tensor_wrappers,
                         std::vector<Qnn_Tensor_t> & qnn_tensors) {
    GGML_ASSERT(tensor_wrappers.size() == ggml_tensors.size());
    qnn_tensors.resize(ggml_tensors.size());
    for (size_t i = 0; i < ggml_tensors.size(); i++) {
        auto * ggml_tensor = ggml_tensors[i];
        if (!tensor_wrappers[i]->bind_ggml_tensor(ggml_tensor, qnn_buffer_ptr())) {
            QNN_LOG_ERROR("bind tensor %s failed\n", ggml_get_name(ggml_tensor));
            return false;
        }

        qnn_tensors[i] = tensor_wrappers[i]->get_qnn_tensor();
    }

    return true;
}

inline bool bind_tensors(const ggml_tensor_array_t & ggml_tensors, qnn_tensor_array_t & tensor_wrappers) {
    GGML_ASSERT(tensor_wrappers.size() == ggml_tensors.size());
    for (size_t i = 0; i < ggml_tensors.size(); i++) {
        auto * ggml_tensor = ggml_tensors[i];
        if (!tensor_wrappers[i]->bind_ggml_tensor(ggml_tensor, qnn_buffer_ptr())) {
            QNN_LOG_ERROR("bind tensor %s failed\n", ggml_get_name(ggml_tensor));
            return false;
        }
    }

    return true;
}

inline void unbind_tensors(qnn_tensor_array_t & tensor_wrappers) {
    for (auto & tensor : tensor_wrappers) {
        tensor->unbind();
    }
}

struct tensor_create_common_params {
    const char *                       name_prefix;
    int                                tensor_rank;
    bool                               is_input;
    backend_index_type                 device;
    Qnn_GraphHandle_t                  graph_handle;
    std::shared_ptr<qnn::qnn_instance> qnn_instance;
};

inline void create_tensors_from_ggml_tensor(const tensor_create_common_params & params,
                                            const ggml_tensor_array_t &         ggml_tensors,
                                            qnn_tensor_array_t *                tensor_wrappers,
                                            std::vector<Qnn_Tensor_t> *         qnn_tensors) {
    if (qnn_tensors) {
        qnn_tensors->resize(ggml_tensors.size());
    }

    if (!tensor_wrappers->empty()) {
        QNN_LOG_DEBUG("tensor_wrappers is not empty, skip create tensors\n");
        GGML_ASSERT(tensor_wrappers->size() == ggml_tensors.size());
        return;
    }

    tensor_wrappers->resize(ggml_tensors.size());

    char buffer[GGML_MAX_NAME] = {};
    auto tensor_type           = params.is_input ? ggml_qnn_tensor::INPUT : ggml_qnn_tensor::OUTPUT;
    for (size_t i = 0; i < ggml_tensors.size(); i++) {
        snprintf(buffer, GGML_MAX_NAME, "%s%d", params.name_prefix, (int) i);
        auto * ggml_tensor    = ggml_tensors[i];
        (*tensor_wrappers)[i] = std::make_shared<ggml_qnn_tensor>(tensor_type, std::string(buffer), ggml_tensor->ne,
                                                                  ggml_tensor->type, params.tensor_rank, params.device,
                                                                  params.graph_handle, params.qnn_instance);
    }
}

}  // namespace qnn
