#include "../ggml/src/ggml-backend-impl.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

struct test_device_context {
    int metadata[2] = {};
    size_t generation = 0;
    size_t init_count = 0;
    size_t set_count = 0;
    size_t get_count = 0;
    size_t reset_count = 0;
    bool fail_init = false;
    ggml_backend_buffer_type buft = {};
    ggml_backend_device device = {};

    void * current_metadata() {
        return &metadata[generation % 2];
    }
};

struct test_buffer_context {
    test_device_context * owner;
    uint8_t * data;
};

static const char * test_buft_name(ggml_backend_buffer_type_t) {
    return "MetaInitTest";
}

static void test_buffer_free(ggml_backend_buffer_t buffer) {
    auto * context = static_cast<test_buffer_context *>(buffer->context);
    delete[] context->data;
    delete context;
}

static void * test_buffer_base(ggml_backend_buffer_t buffer) {
    return static_cast<test_buffer_context *>(buffer->context)->data;
}

static ggml_status test_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto * owner = static_cast<test_buffer_context *>(buffer->context)->owner;
    owner->init_count++;
    if (owner->fail_init) {
        return GGML_STATUS_FAILED;
    }
    tensor->extra = owner->current_metadata();
    return GGML_STATUS_SUCCESS;
}

static void test_buffer_set_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor,
        const void * data, size_t offset, size_t size) {
    auto * owner = static_cast<test_buffer_context *>(buffer->context)->owner;
    GGML_ASSERT(tensor->extra == owner->current_metadata());
    GGML_ASSERT(offset <= ggml_nbytes(tensor) && size <= ggml_nbytes(tensor) - offset);
    std::memcpy(static_cast<uint8_t *>(tensor->data) + offset, data, size);
    owner->set_count++;
}

static void test_buffer_get_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
        void * data, size_t offset, size_t size) {
    auto * owner = static_cast<test_buffer_context *>(buffer->context)->owner;
    GGML_ASSERT(tensor->extra == owner->current_metadata());
    GGML_ASSERT(offset <= ggml_nbytes(tensor) && size <= ggml_nbytes(tensor) - offset);
    std::memcpy(data, static_cast<const uint8_t *>(tensor->data) + offset, size);
    owner->get_count++;
}

static void test_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    std::memset(test_buffer_base(buffer), value, buffer->size);
}

static void test_buffer_reset(ggml_backend_buffer_t buffer) {
    auto * owner = static_cast<test_buffer_context *>(buffer->context)->owner;
    owner->generation++;
    owner->reset_count++;
}

static ggml_backend_buffer_t test_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto * owner = static_cast<test_device_context *>(buft->context);
    auto * context = new test_buffer_context {
        owner,
        new uint8_t[size],
    };
    const ggml_backend_buffer_i iface = {
        /* .free_buffer   = */ test_buffer_free,
        /* .get_base      = */ test_buffer_base,
        /* .init_tensor   = */ test_buffer_init_tensor,
        /* .memset_tensor = */ nullptr,
        /* .set_tensor    = */ test_buffer_set_tensor,
        /* .get_tensor    = */ test_buffer_get_tensor,
        /* .set_tensor_2d = */ nullptr,
        /* .get_tensor_2d = */ nullptr,
        /* .cpy_tensor    = */ nullptr,
        /* .clear         = */ test_buffer_clear,
        /* .reset         = */ test_buffer_reset,
    };
    return ggml_backend_buffer_init(buft, iface, context, size);
}

static size_t test_buft_alignment(ggml_backend_buffer_type_t) {
    return 16;
}

static bool test_buft_is_host(ggml_backend_buffer_type_t) {
    return false;
}

static const char * test_device_name(ggml_backend_dev_t) {
    return "META_INIT_TEST";
}

static const char * test_device_description(ggml_backend_dev_t) {
    return "Meta tensor initialization test device";
}

static void test_device_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    *free = 0;
    *total = 0;
}

static enum ggml_backend_dev_type test_device_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void test_device_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = test_device_name(dev);
    props->description = test_device_description(dev);
    props->type = test_device_type(dev);
}

static ggml_backend_buffer_type_t test_device_buft(ggml_backend_dev_t dev) {
    return &static_cast<test_device_context *>(dev->context)->buft;
}

static bool test_device_supports_op(ggml_backend_dev_t, const ggml_tensor *) {
    return true;
}

static bool test_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == test_device_buft(dev);
}

static ggml_backend_meta_split_state test_split_state(const ggml_tensor *, void *) {
    return {
        GGML_BACKEND_SPLIT_AXIS_MIRRORED,
        {0},
        {1},
        1,
    };
}

static test_device_context make_test_device() {
    test_device_context result;
    result.buft.iface = {
        /* .get_name       = */ test_buft_name,
        /* .alloc_buffer   = */ test_buft_alloc,
        /* .get_alignment  = */ test_buft_alignment,
        /* .get_max_size   = */ nullptr,
        /* .get_alloc_size = */ nullptr,
        /* .is_host        = */ test_buft_is_host,
    };
    result.device.iface = {
        /* .get_name             = */ test_device_name,
        /* .get_description      = */ test_device_description,
        /* .get_memory           = */ test_device_memory,
        /* .get_type             = */ test_device_type,
        /* .get_props            = */ test_device_props,
        /* .init_backend         = */ nullptr,
        /* .get_buffer_type      = */ test_device_buft,
        /* .get_host_buffer_type = */ nullptr,
        /* .buffer_from_host_ptr = */ nullptr,
        /* .supports_op          = */ test_device_supports_op,
        /* .supports_buft        = */ test_device_supports_buft,
        /* .offload_op           = */ nullptr,
        /* .event_new            = */ nullptr,
        /* .event_free           = */ nullptr,
        /* .event_synchronize    = */ nullptr,
    };
    result.buft.device = &result.device;
    result.buft.context = &result;
    result.device.context = &result;
    return result;
}

int main() {
    test_device_context device = make_test_device();
    device.buft.device = &device.device;
    device.buft.context = &device;
    device.device.context = &device;

    ggml_backend_dev_t simple_devices[] = {&device.device};
    ggml_backend_dev_t meta_device = ggml_backend_meta_device(
        simple_devices, 1, test_split_state, nullptr);
    ggml_backend_buffer_type_t meta_buft = ggml_backend_dev_buffer_type(meta_device);
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(meta_buft, 256));
    GGML_ASSERT(buffer);

    ggml_init_params params = {
        /* .mem_size   = */ 8*ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    GGML_ASSERT(ctx);

    ggml_tensor * parent = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    GGML_ASSERT(ggml_backend_tensor_alloc(
        buffer.get(), parent, ggml_backend_buffer_get_base(buffer.get())) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(device.init_count == 1);

    const float parent_data[] = {1, 2, 3, 4};
    ggml_backend_tensor_set(parent, parent_data, 0, sizeof(parent_data));
    float parent_result[4] = {};
    ggml_backend_tensor_get(parent, parent_result, 0, sizeof(parent_result));
    GGML_ASSERT(std::memcmp(parent_data, parent_result, sizeof(parent_data)) == 0);

    ggml_tensor * view = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 2);
    view->view_src = parent;
    view->view_offs = 0;
    GGML_ASSERT(ggml_backend_view_init(view) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(device.init_count == 2);

    const float view_data[] = {5, 6};
    ggml_backend_tensor_set(view, view_data, 0, sizeof(view_data));
    float view_result[2] = {};
    ggml_backend_tensor_get(view, view_result, 0, sizeof(view_result));
    GGML_ASSERT(std::memcmp(view_data, view_result, sizeof(view_data)) == 0);

    ggml_backend_buffer_reset(buffer.get());
    GGML_ASSERT(device.reset_count == 1);

    ggml_tensor * next = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    GGML_ASSERT(ggml_backend_tensor_alloc(
        buffer.get(), next,
        static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffer.get())) + 64) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(device.init_count == 3);
    ggml_backend_tensor_set(next, parent_data, 0, sizeof(parent_data));

    ggml_context_ptr static_ctx(ggml_init(params));
    GGML_ASSERT(static_ctx);
    ggml_tensor * static_tensor = ggml_new_tensor_1d(static_ctx.get(), GGML_TYPE_F32, 4);
    ggml_backend_buffer_ptr static_buffer(
        ggml_backend_meta_alloc_ctx_tensors_from_buft(static_ctx.get(), meta_buft));
    GGML_ASSERT(static_buffer);
    GGML_ASSERT(device.init_count == 4);
    ggml_backend_tensor_set(static_tensor, parent_data, 0, sizeof(parent_data));

    device.fail_init = true;
    ggml_tensor * failed = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    GGML_ASSERT(ggml_backend_tensor_alloc(
        buffer.get(), failed,
        static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffer.get())) + 128) == GGML_STATUS_FAILED);

    GGML_ASSERT(device.init_count == 5);
    GGML_ASSERT(device.set_count == 4);
    GGML_ASSERT(device.get_count == 2);
    std::puts("test_backend_meta_tensor_initialization PASSED");
    return 0;
}
