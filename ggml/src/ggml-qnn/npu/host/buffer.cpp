#include "buffer.hpp"

#include "host_device.hpp"
#include "profiler.hpp"
#include "tensor.hpp"

#include <rpcmem.h>

namespace {

constexpr const int      kRpcMemDefaultHeapId = RPCMEM_HEAP_ID_SYSTEM;
constexpr const uint32_t kRpcMemDefaultFlags  = RPCMEM_DEFAULT_FLAGS;  // TODO: should we use a different flag?

static hexagon::host_buffer * get_buffer_object(ggml_backend_buffer_t buffer) {
    return reinterpret_cast<hexagon::host_buffer *>(buffer->context);
}

static hexagon::host_buffer_type * get_buffer_type_object(ggml_backend_buffer_type_t buft) {
    return reinterpret_cast<hexagon::host_buffer_type *>(buft->context);
}

void backend_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    SCOPED_PERFORMANCE_TRACKER("[hexagon-npu][%p]backend_buffer_free_buffer", (void *) get_buffer_object(buffer));
    delete get_buffer_object(buffer);
}

void * backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * buffer_obj = get_buffer_object(buffer);
    GGML_ASSERT(buffer_obj != nullptr);
    return buffer_obj->get_buffer();
}

ggml_status backend_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto * buffer_type_obj = get_buffer_type_object(buffer->buft);
    GGML_ASSERT(buffer_type_obj != nullptr);

    auto * device_object = buffer_type_obj->get_device();
    GGML_ASSERT(device_object != nullptr);

    auto * buffer_obj = get_buffer_object(buffer);
    GGML_ASSERT(buffer_obj != nullptr);

    SCOPED_PERFORMANCE_TRACKER("[hexagon-npu][%p]backend_buffer_init_tensor", (void *) buffer_obj);
    auto tensor_object = buffer_obj->init_tensor(tensor, device_object->get_device_handle());
    if (!tensor_object) {
        LOG_ERROR("Failed to init tensor\n");
        return GGML_STATUS_ALLOC_FAILED;
    }

    return GGML_STATUS_SUCCESS;
}

void backend_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                  ggml_tensor *         tensor,
                                  uint8_t               value,
                                  size_t                offset,
                                  size_t                size) {
    SCOPED_PERFORMANCE_TRACKER("[hexagon-npu][%p]backend_buffer_memset_tensor.size.%zu",
                               (void *) get_buffer_object(buffer), size);
    memset((char *) tensor->data + offset, value, size);
}

void backend_buffer_set_tensor(ggml_backend_buffer_t buffer,
                               ggml_tensor *         tensor,
                               const void *          data,
                               size_t                offset,
                               size_t                size) {
    SCOPED_PERFORMANCE_TRACKER("[hexagon-npu][%p]backend_buffer_set_tensor.size.%zu",
                               (void *) get_buffer_object(buffer), size);

    // TODO: use DMA instead of memcpy?
    memcpy((char *) tensor->data + offset, data, size);
}

void backend_buffer_get_tensor(ggml_backend_buffer_t buffer,
                               const ggml_tensor *   tensor,
                               void *                data,
                               size_t                offset,
                               size_t                size) {
    SCOPED_PERFORMANCE_TRACKER("[hexagon-npu][%p]backend_buffer_get_tensor", (void *) get_buffer_object(buffer));

    // TODO: use DMA instead of memcpy?
    memcpy(data, (const char *) tensor->data + offset, size);
}

bool backend_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    SCOPED_PERFORMANCE_TRACKER("[hexagon-npu][%p]backend_buffer_cpy_tensor", (void *) get_buffer_object(buffer));
    if (ggml_backend_buffer_is_host(src->buffer)) {
        // TODO: use DMA instead of memcpy?
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }

    LOG_DEBUG("[hexagon-npu][%p]backend_buffer_cpy_tensor: copy from non-host buffer not supported\n",
              (void *) get_buffer_object(buffer));
    return false;
}

void backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * buffer_obj = get_buffer_object(buffer);
    GGML_ASSERT(buffer_obj != nullptr);

    SCOPED_PERFORMANCE_TRACKER("[hexagon-npu][%p]backend_buffer_clear", (void *) buffer_obj);
    memset(buffer_obj->get_buffer(), value, buffer_obj->get_size());
}

void backend_buffer_reset(ggml_backend_buffer_t buffer) {
    auto * buffer_obj = get_buffer_object(buffer);
    GGML_ASSERT(buffer_obj != nullptr);

    SCOPED_PERFORMANCE_TRACKER("[hexagon-npu][%p]backend_buffer_reset", (void *) buffer_obj);
    buffer_obj->clear_tensors();
}

constexpr const ggml_backend_buffer_i backend_buffer_interface = {
    /* .free_buffer     = */ backend_buffer_free_buffer,
    /* .get_base        = */ backend_buffer_get_base,
    /* .init_tensor     = */ backend_buffer_init_tensor,
    /* .memset_tensor   = */ backend_buffer_memset_tensor,
    /* .set_tensor      = */ backend_buffer_set_tensor,
    /* .get_tensor      = */ backend_buffer_get_tensor,
    /* .cpy_tensor      = */ backend_buffer_cpy_tensor,
    /* .clear           = */ backend_buffer_clear,
    /* .reset           = */ backend_buffer_reset,
};

const char * backend_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    auto * buffer_type_obj = get_buffer_type_object(buft);
    GGML_ASSERT(buffer_type_obj != nullptr);
    return buffer_type_obj->get_name();
}

ggml_backend_buffer_t backend_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * buffer_type_obj = get_buffer_type_object(buft);
    GGML_ASSERT(buffer_type_obj != nullptr);
    return buffer_type_obj->allocate_buffer(size);
}

size_t backend_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    auto * buffer_type_obj = get_buffer_type_object(buft);
    GGML_ASSERT(buffer_type_obj != nullptr);
    return buffer_type_obj->get_buffer_alignment();
}

size_t backend_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    auto * buffer_type_obj = get_buffer_type_object(buft);
    GGML_ASSERT(buffer_type_obj != nullptr);
    auto size = buffer_type_obj->get_max_buffer_size();
    LOG_DEBUG("[hexagon-npu][%s]max_buffer_size: %zu\n", buffer_type_obj->get_name(), size);
    return size;
}

bool backend_buffer_is_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == backend_buffer_type_get_name;
}

}  // namespace

namespace hexagon {

host_buffer::host_buffer(common::rpc_mem_ptr allocator, size_t size, uint32_t domain_id) :
    _allocator(allocator),
    _size(size),
    _domain_id(domain_id) {
    if (!_allocator->is_valid()) {
        LOG_ERROR("[hexagon-npu]rpc memory not initialized\n");
        return;
    }

    if (size > _allocator->get_max_alloc_size()) {
        LOG_ERROR("[hexagon-npu]rpc memory size %zu exceeds max alloc size %zu\n", size,
                  _allocator->get_max_alloc_size());
        return;
    }

    _data = _allocator->alloc(kRpcMemDefaultHeapId, kRpcMemDefaultFlags, size);
    if (!_data) {
        LOG_ERROR("[hexagon-npu]failed to allocate rpc memory, size: %d MB\n", (int) (size / (1 << 20)));
        return;
    }

    LOG_DEBUG("[hexagon-npu]create host_buffer(%p), size: %zu, domain_id: %d\n", (void *) _data, size, (int) domain_id);
}

host_buffer::~host_buffer() {
    LOG_DEBUG("[hexagon-npu]destroy host_buffer(%p), size: %zu, domain_id: %d\n", (void *) _data, _size,
              (int) _domain_id);
    _tensors.clear();
    if (_buffer_fd != -1) {
        auto ret = _allocator->fastrpc_munmap((int) _domain_id, _buffer_fd, nullptr, 0);
        if (ret != AEE_SUCCESS) {
            LOG_ERROR("[hexagon-npu]failed to munmap rpc memory, fd: %d, ret: %d\n", _buffer_fd, ret);
            return;
        }
    }

    _allocator->free(_data);
}

std::shared_ptr<host_tensor> host_buffer::init_tensor(ggml_tensor * tensor, remote_handle64 device_handle) {
    if (!_data) {
        LOG_ERROR("[hexagon-npu]failed to init tensor, rpc memory not initialized\n");
        return std::shared_ptr<host_tensor>();
    }

    if (_buffer_fd == -1) {
        _buffer_fd = _allocator->to_fd(_data);
        if (_buffer_fd < 0) {
            LOG_ERROR("[hexagon-npu]failed to get fd from rpc memory\n");
            return std::shared_ptr<host_tensor>();
        }

        auto ret = _allocator->fastrpc_mmap((int) _domain_id, _buffer_fd, _data, 0, _size, FASTRPC_MAP_FD);
        if (ret != AEE_SUCCESS) {
            LOG_ERROR("[hexagon-npu]failed to mmap rpc memory, fd: %d, size: %zu, ret: %d\n", _buffer_fd, _size, ret);
            return std::shared_ptr<host_tensor>();
        }

        LOG_DEBUG("[hexagon-npu]mmap rpc memory(%p), fd: %d, addr: %p, size: %zu\n", (void *) _data, _buffer_fd, _data,
                  _size);
    }

    auto tensor_object = std::make_shared<host_tensor>(
        tensor, _buffer_fd, (uint64_t) (reinterpret_cast<uint8_t *>(tensor->data) - reinterpret_cast<uint8_t *>(_data)),
        device_handle);
    if (!tensor_object->is_valid()) {
        LOG_ERROR("[hexagon-npu]failed to init tensor, device handle: %p\n", (void *) device_handle);
        return std::shared_ptr<host_tensor>();
    }

    _tensors.push_back(tensor_object);
    return tensor_object;
}

void host_buffer::clear_tensors() {
    LOG_DEBUG("[hexagon-npu]clear host_buffer(%p) tensors\n", (void *) _data);
    host_tensor::destroy_tensors(_tensors);
}

host_buffer_type::host_buffer_type(ggml_backend_dev_t dev, const std::string & name, common::rpc_mem_ptr rpc_mem) :
    _name(name),
    _rpc_mem(rpc_mem) {
    iface = {
        /* .get_name       = */ backend_buffer_type_get_name,
        /* .alloc_buffer   = */ backend_buffer_type_alloc_buffer,
        /* .get_alignment  = */ backend_buffer_type_get_alignment,
        /* .get_max_size   = */ backend_buffer_type_get_max_size,
        /* .get_alloc_size = */ nullptr,  // defaults to ggml_nbytes
        /* .is_host = */ backend_buffer_is_host,
    };
    device  = dev;
    context = this;

    _device = reinterpret_cast<npu_device *>(device->context);
    LOG_DEBUG("[%s]create host_buffer_type %s\n", _device->get_name(), _name.c_str());
}

size_t host_buffer_type::get_buffer_alignment() const {
    return _device->is_device_initialized() ? _device->get_alignment() : 128;
}

size_t host_buffer_type::get_max_buffer_size() const {
    if (!_rpc_mem) {
        LOG_ERROR("[%s]rpc memory not initialized\n", _device->get_name());
        return 0;
    }

    return _rpc_mem->get_max_alloc_size();
}

ggml_backend_buffer_t host_buffer_type::allocate_buffer(size_t size) {
    if (!_rpc_mem) {
        LOG_ERROR("[%s]rpc memory not initialized\n", _device->get_name());
        return nullptr;
    }

    if (!_device->is_device_initialized()) {
        LOG_ERROR("[%s]device is not initialized\n", _device->get_name());
        return nullptr;
    }

    auto * buffer = new host_buffer(_rpc_mem, size, _device->get_dsp_domain_id());
    if (!buffer->is_valid()) {
        delete buffer;
        LOG_ERROR("[%s]Failed to allocate buffer of size %zu\n", _device->get_name(), size);
        return nullptr;
    }

    LOG_DEBUG("[%s]allocate buffer %p, size: %zu\n", _device->get_name(), buffer->get_buffer(), size);
    return ggml_backend_buffer_init(this, backend_buffer_interface, buffer, size);
}

}  // namespace hexagon
