#include "ggml-remoting.h"

extern ggml_backend_buffer_i ggml_backend_remoting_buffer_interface;

struct ggml_backend_remoting_buffer_type_context {
    std::string name;
};


static const char * ggml_backend_remoting_buffer_type_name(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);

    return "Remoting buffer";
}

static ggml_backend_buffer_t ggml_backend_remoting_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_remoting_buffer_type_context * ctx = (ggml_backend_remoting_buffer_type_context *) buft->context;


    return ggml_backend_buffer_init(buft, ggml_backend_remoting_buffer_interface, ctx, size);
}

static size_t ggml_backend_remoting_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return 4096;
}

static size_t ggml_backend_remoting_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return 40960;
}

static size_t ggml_backend_remoting_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    UNUSED(buft);
    UNUSED(tensor);
    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_remoting_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_remoting_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_remoting_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_remoting_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_remoting_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_remoting_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_remoting_device_get_buffer_type(ggml_backend_dev_t dev) {

    static struct ggml_backend_buffer_type buft {
      /* .iface    = */ ggml_backend_remoting_buffer_type_interface,
      /* .device   = */ dev,
      /* .context  = */ new ggml_backend_remoting_buffer_type_context{ "device_name"},
    };

    return & buft;
}

static void ggml_backend_remoting_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_remoting_buffer_context * ctx = (ggml_backend_remoting_buffer_context *)buffer->context;
    ggml_remoting_destroy_buffer(ctx->dev_buffer);
    delete ctx;
}

static enum ggml_status ggml_backend_remoting_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    if (tensor->view_src != nullptr) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
    }
    return GGML_STATUS_SUCCESS;
}

static void * ggml_backend_remoting_buffer_get_base(ggml_backend_buffer_t buffer) {
    return (void *) 4096;

    UNUSED(buffer);
}

static void ggml_backend_remoting_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
  UNUSED(buffer);
  UNUSED(tensor);
  UNUSED(value);
  UNUSED(offset);
  UNUSED(size);
}


static void ggml_backend_remoting_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
#if 0
    ggml_backend_remoting_buffer_context * buf_ctx = (ggml_backend_remoting_buffer_context *)buffer->context;
    remoting_buffer buf = buf_ctx->dev_buffer;

    ggml_remoting_buffer_write(buf, remoting_tensor_offset(tensor) + tensor->view_offs + offset, data, size);
#else
    UNUSED(buffer);
    UNUSED(tensor);
    UNUSED(data);
    UNUSED(offset);
    UNUSED(size);
#endif
}

static void ggml_backend_remoting_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
#if 0
    ggml_backend_remoting_buffer_context * buf_ctx = (ggml_backend_remoting_buffer_context *)buffer->context;

    remoting_buffer buf = buf_ctx->dev_buffer;

    ggml_remoting_buffer_read(buf, remoting_tensor_offset(tensor) + tensor->view_offs + offset, data, size);
#else
    UNUSED(buffer);
    UNUSED(tensor);
    UNUSED(data);
    UNUSED(offset);
    UNUSED(size);
#endif
}


static bool ggml_backend_remoting_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
  return true;

  UNUSED(buffer);
  UNUSED(src);
  UNUSED(dst);
}

static void ggml_remoting_buffer_memset(remoting_buffer& dst, size_t offset, uint32_t c, size_t size) {
  UNUSED(dst);
  UNUSED(c);
  UNUSED(size);
  UNUSED(offset);
}

static void ggml_remoting_buffer_memset_async(remoting_context& ctx, remoting_buffer& dst, size_t offset, uint32_t c, size_t size) {
  UNUSED(ctx);
  UNUSED(dst);
  UNUSED(c);
  UNUSED(size);
  UNUSED(offset);
}

static void ggml_backend_remoting_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_remoting_buffer_context * ctx = (ggml_backend_remoting_buffer_context *)buffer->context;

    ggml_remoting_buffer_memset(ctx->dev_buffer, 0, value, buffer->size);
}

ggml_backend_buffer_i ggml_backend_remoting_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_remoting_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_remoting_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_remoting_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_remoting_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_remoting_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_remoting_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_buffer_clear,
    /* .reset           = */ NULL,
};
