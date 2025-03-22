#pragma once

#include <cstdint>
#include <memory>

#include "logger.hpp"
#include "qnn-lib.hpp"

namespace qnn {

/**
 * @brief An interface for managing generic QNN buffers.
 *
 * This abstract class defines the interface for managing generic memory buffers in a QNN context.
 */
class qnn_buffer_interface {
  public:
    virtual ~qnn_buffer_interface() = default;

    /**
     * @brief Checks if the buffer is valid.
     *
     * This pure virtual function must be implemented by derived classes to check
     * the validity of the buffer.
     *
     * @return true if the buffer is valid, false otherwise.
     */
    virtual bool is_valid() const = 0;

    /**
     * @brief Gets the buffer pointer.
     *
     * This pure virtual function must be implemented by derived classes to return
     * a pointer to the buffer.
     *
     * @return A pointer to the buffer.
     */
    virtual uint8_t * get_buffer() = 0;

    /**
     * @brief Gets the buffer pointer.
     *
     * This pure virtual function must be implemented by derived classes to return
     * a pointer to the buffer.
     *
     * @return A pointer to the buffer.
     */
    virtual size_t get_size() const = 0;

    /**
     * @brief Gets the QNN memory handle associated with the buffer.
     *
     * This pure virtual function must be implemented by derived classes to return
     * the memory handle associated with the buffer.
     *
     * @return The memory handle, or null if no valid QNN memory handle is attached.
     */
    virtual Qnn_MemHandle_t get_mem_handle() const = 0;
};

using qnn_buffer_ptr = std::shared_ptr<qnn_buffer_interface>;

/**
 * @brief A class for managing QNN RPC memory buffers.
 *
 * This class is responsible for allocating, registering, and managing a buffer in RPC memory.
 * It ensures that the buffer is properly allocated and registered with the QNN instance, and
 * handles cleanup of the buffer and its associated memory handle upon destruction.
 */
class qnn_rpc_buffer : public qnn_buffer_interface {
  public:
    qnn_rpc_buffer(qnn_instance_ptr qnn_instance, const size_t size, const uint32_t rank, uint32_t * dimensions,
                   Qnn_DataType_t data_type) :
        _size(size),
        _qnn_instance(qnn_instance) {
        _qnn_rpc_buffer     = static_cast<uint8_t *>(qnn_instance->alloc_rpcmem(size, alignof(uint8_t *)));
        _qnn_rpc_mem_handle = qnn_instance->register_rpcmem(_qnn_rpc_buffer, rank, dimensions, data_type);
        if (!_qnn_rpc_buffer || !_qnn_rpc_mem_handle) {
            QNN_LOG_WARN("Failed to register RPC memory: buffer or memory handle is null\n");
            // let the destructor free the buffer
            return;
        }

        QNN_LOG_DEBUG("alloc rpcmem(%p) successfully, size %d\n", (void *) _qnn_rpc_buffer, (int) size);
    }

    ~qnn_rpc_buffer() {
        if (_qnn_instance) {
            if (_qnn_rpc_mem_handle) {
                _qnn_instance->unregister_rpcmem(_qnn_rpc_mem_handle);
            }

            if (_qnn_rpc_buffer) {
                _qnn_instance->free_rpcmem(_qnn_rpc_buffer);
            }
        }
    }

    bool is_valid() const override { return _qnn_rpc_buffer && _qnn_rpc_mem_handle; }

    uint8_t * get_buffer() override { return _qnn_rpc_buffer; }

    size_t get_size() const override { return _size; }

    Qnn_MemHandle_t get_mem_handle() const override { return _qnn_rpc_mem_handle; }

  private:
    size_t           _size               = 0;
    uint8_t *        _qnn_rpc_buffer     = nullptr;
    Qnn_MemHandle_t  _qnn_rpc_mem_handle = nullptr;
    qnn_instance_ptr _qnn_instance;

    DISABLE_COPY(qnn_rpc_buffer);
    DISABLE_MOVE(qnn_rpc_buffer);
};

/**
 * @brief A class for managing QNN memory buffers allocated in regular memory.
 *
 * This class is responsible for allocating, managing, and freeing memory buffers
 * in regular (non-RPC) memory. It implements the qnn_buffer_interface to provide
 * a consistent interface for buffer management.
 */
class qnn_mem_buffer : public qnn_buffer_interface {
  public:
    explicit qnn_mem_buffer(const uint8_t * data, size_t size) {
        _buffer = reinterpret_cast<uint8_t *>(qnn::page_align_alloc(size));

        if (!_buffer) {
            QNN_LOG_WARN("failed to allocate %.2f MiB\n", float(size / (1 << 20)));
            return;
        }

        _size = size;

        if (data) {
            memcpy(_buffer, data, size);
        }

        QNN_LOG_DEBUG("alloc buffer: %p, size: %ld\n", (void *) _buffer, (long) size);
    }

    explicit qnn_mem_buffer(size_t size) : qnn_mem_buffer(nullptr, size) {}

    ~qnn_mem_buffer() {
        QNN_LOG_DEBUG("free buffer: %p, size: %ld\n", (void *) _buffer, (long) _size);
        // the free will do nothing if the _buffer is nullptr
        qnn::align_free(_buffer);
    }

    bool is_valid() const override { return _buffer != nullptr; }

    uint8_t * get_buffer() override { return _buffer; }

    size_t get_size() const override { return _size; }

    Qnn_MemHandle_t get_mem_handle() const override { return nullptr; }

  private:
    size_t    _size   = 0;
    uint8_t * _buffer = nullptr;

    DISABLE_COPY(qnn_mem_buffer);
    DISABLE_MOVE(qnn_mem_buffer);
};

class qnn_mem_buffer_slice : public qnn_buffer_interface {
  public:
    qnn_mem_buffer_slice(const uint8_t * buffer, size_t size) : _buffer(const_cast<uint8_t *>(buffer)), _size(size) {}

    bool is_valid() const override { return _buffer && _size; }

    uint8_t * get_buffer() override { return _buffer; }

    size_t get_size() const override { return _size; }

    Qnn_MemHandle_t get_mem_handle() const override { return nullptr; }

  private:
    uint8_t * _buffer = nullptr;
    size_t    _size   = 0;

    DISABLE_COPY(qnn_mem_buffer_slice);
    DISABLE_MOVE(qnn_mem_buffer_slice);
};

}  // namespace qnn
