#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct ggml_backend_meta_device;
struct ggml_backend_meta_buffer_type;
struct ggml_backend_meta_buffer;
struct ggml_backend_meta;

//
// meta backend device
//

struct ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    void *                              get_split_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_split_state_t get_splite_state, void * get_split_state_ud) :
            simple_devs(std::move(simple_devs)), get_split_state(get_splite_state), get_split_state_ud(get_split_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        for (size_t i = 0; i < simple_devs.size(); i++) {
            if (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (simple_devs[i]);
            description += ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    bool operator<(const ggml_backend_meta_device_context & other) const {
        return std::tie(simple_devs, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud);
    }
};

static const char * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->name.c_str();
}

static const char * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->description.c_str();
}

static void ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    *free  = 0;
    *total = 0;
    for (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_total;
        ggml_backend_dev_memory(dev, &tmp_free, &tmp_total);
        *free  += tmp_free;
        *total += tmp_total;
    }
}

static enum ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_META;

    GGML_UNUSED(dev);
}

static void ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false, // Not implemented.
        /* .buffer_from_host_ptr  = */ false, // Not implemented.
        /* .events                = */ false, // Not implemented.
    };
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_buffer          = props->caps.host_buffer          && tmp_props.caps.host_buffer;
        props->caps.buffer_from_host_ptr = props->caps.buffer_from_host_ptr && tmp_props.caps.buffer_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
    }
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

static bool ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_supports_op(simple_dev, op); });
}

static bool ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    if (!ggml_backend_dev_is_meta(dev_buft)) {
        return false;
    }
    const ggml_backend_meta_device_context * meta_dev_ctx      = (const ggml_backend_meta_device_context *) dev->context;
    const ggml_backend_meta_device_context * meta_buft_dev_ctx = (const ggml_backend_meta_device_context *) dev_buft->context;
    if (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        return false;
    }
    for (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        if (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            return false;
        }
    }
    return true;
}

static const ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_memory,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    return dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    return meta_dev_ctx->simple_devs.size();
}

ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    return meta_dev_ctx->simple_devs[index];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
    GGML_ASSERT(n_devs == 1 || n_devs == 2 || n_devs == 4 || n_devs == 8);
    static std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        if (it != meta_devs.end()) {
            return &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    struct ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    return &result.first->second;
}

//
// meta backend buffer type
//

struct ggml_backend_meta_buffer_type_context {
    std::vector<ggml_backend_buffer_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_buffer_type_context(std::vector<ggml_backend_buffer_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        for (size_t i = 0; i < simple_bufts.size(); i++) {
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    bool operator<(const ggml_backend_meta_buffer_type_context & other) const {
        return simple_bufts < other.simple_bufts;
    }
};

static const char * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) buft->context;
    return meta_buft_ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

static size_t ggml_backend_meta_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    return max_alignment;
}

static size_t ggml_backend_meta_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_size(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    return max_size;
}

static size_t ggml_backend_meta_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_size = 0;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alloc_size = ggml_backend_buft_get_alloc_size(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }
    return max_alloc_size;
}

static bool ggml_backend_meta_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        if (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            return false;
        }
    }
    return true;
}

static const struct ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name         = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_meta_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_meta_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_meta_buffer_type_is_host,
};

bool ggml_backend_buft_is_meta(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev) {
    static std::map<ggml_backend_dev_t, struct ggml_backend_buffer_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        if (it != meta_bufts.end()) {
            return &it->second;
        }
    }

    const size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_buffer_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_buffer_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_buffer_type_context * buft_ctx = new ggml_backend_meta_buffer_type_context(simple_bufts);

    struct ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    return &result.first->second;
}

size_t ggml_backend_meta_buft_n_bufts(ggml_backend_buffer_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    return meta_buft_ctx->simple_bufts.size();
}

ggml_backend_buffer_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_buffer_type_t meta_buft, size_t index) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    return meta_buft_ctx->simple_bufts[index];
}

//
// meta backend buffer
//

struct ggml_backend_meta_buffer_context {
    std::map<std::pair<const ggml_tensor *, bool>, ggml_backend_meta_split_state> split_state_cache;
    std::map<          const ggml_tensor *,        std::vector<ggml_tensor *>>    simple_tensors;

    struct buffer_config {
        ggml_context          * ctx;
        ggml_backend_buffer_t   buf;

        buffer_config(ggml_context * ctx, ggml_backend_buffer_t buf) : ctx(ctx), buf(buf) {}
    };
    std::vector<buffer_config> buf_configs;
};

static void ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (auto & [ctx, buf] : buf_ctx->buf_configs) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    delete buf_ctx;
}

static void * ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return (void *) 0x1000000000000000; // FIXME
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    const size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(buffer);

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ true);
    GGML_ASSERT(split_state != GGML_BACKEND_SPLIT_STATE_UNKNOWN);

    int split_dim = split_state;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }
    if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
        GGML_ASSERT(ne[split_dim] % (n_simple_bufs*ggml_blck_size(tensor->type)) == 0);
        ne[split_dim] /= n_simple_bufs;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            if (tensor->nb[i] > tensor->nb[split_dim]) {
                GGML_ASSERT(nb[i] % (n_simple_bufs*ggml_element_size(tensor)) == 0);
                nb[i] /= n_simple_bufs;
            }
        }
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(buf_ctx->buf_configs.size());
    for (size_t j = 0; j < buf_ctx->buf_configs.size(); j++) {
        ggml_context          * simple_ctx = buf_ctx->buf_configs[j].ctx;
        ggml_backend_buffer_t   simple_buf = buf_ctx->buf_configs[j].buf;

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, sizeof(tensor->op_params));
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_offs = tensor->view_offs;
        t_ij->view_src = tensor->view_src;
        if (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
        }
        if (t_ij->view_src != nullptr) {
            t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
        } else if (simple_buf != nullptr) {
            t_ij->data = (char *) ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(buffer));
        }
        t_ij->extra = tensor->extra;
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            if (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } else if (t_ij->src[i] != nullptr && ggml_backend_buffer_is_meta(t_ij->src[i]->buffer)) {
                t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }
    buf_ctx->simple_tensors[tensor] = simple_tensors;

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));
    const ggml_backend_meta_buffer_context * buf_ctx = (const ggml_backend_meta_buffer_context *) buffer->context;

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    std::vector<ggml_tensor *> simple_tensors;
    {
        auto it = buf_ctx->simple_tensors.find(tensor);
        assert(it != buf_ctx->simple_tensors.end());
        simple_tensors = it->second;
    }

    switch (split_state) {
        case GGML_BACKEND_SPLIT_STATE_BY_NE0:
        case GGML_BACKEND_SPLIT_STATE_BY_NE1:
        case GGML_BACKEND_SPLIT_STATE_BY_NE2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[int(split_state) + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (ggml_tensor * t : simple_tensors) {
                const size_t chunk_size_j = t->nb[int(split_state) + 1];
                for (int64_t i1 = i_start; i1 < i_stop; i1++) {
                    ggml_backend_tensor_set(t, (const char *) data + i1*chunk_size_full + offset_j, i1*chunk_size_j, chunk_size_j);
                }
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_STATE_MIRRORED: {
            for (ggml_tensor * t : simple_tensors) {
                ggml_backend_tensor_set(t, data, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        } break;
    }
}

static void ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));
    const ggml_backend_meta_buffer_context * buf_ctx = (const ggml_backend_meta_buffer_context *) buffer->context;

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    std::vector<ggml_tensor *> simple_tensors;
    {
        auto it = buf_ctx->simple_tensors.find(tensor);
        assert(it != buf_ctx->simple_tensors.end());
        simple_tensors = it->second;
    }

    switch (split_state) {
        case GGML_BACKEND_SPLIT_STATE_BY_NE0:
        case GGML_BACKEND_SPLIT_STATE_BY_NE1:
        case GGML_BACKEND_SPLIT_STATE_BY_NE2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[int(split_state) + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (ggml_tensor * t : simple_tensors) {
                const size_t chunk_size_j = t->nb[int(split_state) + 1];
                for (int64_t i1 = i_start; i1 < i_stop; i1++) {
                    ggml_backend_tensor_get(t, (char *) data + i1*chunk_size_full + offset_j, i1*chunk_size_j, chunk_size_j);
                }
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_STATE_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_tensor_get(simple_tensors[0], data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        } break;
    }
}

static void ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

static void ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_reset(ggml_backend_meta_buffer_simple_buffer(buffer, i));
    }
}

static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_meta_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_meta_buffer_init_tensor,
    /* .memset_tensor   = */ nullptr, // TODO implement
    /* .set_tensor      = */ ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_get_tensor,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_meta_buffer_clear,
    /* .reset           = */ ggml_backend_meta_buffer_reset,
};

bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    return buf != nullptr && buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

size_t ggml_backend_meta_buffer_n_bufs(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    return buf_ctx->buf_configs.size();
}

ggml_backend_buffer_t ggml_backend_meta_buffer_simple_buffer(ggml_backend_buffer_t meta_buf, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    GGML_ASSERT(index < buf_ctx->buf_configs.size());
    return buf_ctx->buf_configs[index].buf;
}

struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(const struct ggml_tensor * tensor, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(index < buf_ctx->buf_configs.size());

    auto it = buf_ctx->simple_tensors.find(tensor);
    if (it == buf_ctx->simple_tensors.end()) {
        return nullptr;
    }
    return it->second[index];
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*1024, // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_backend_meta_buffer_context * buf_ctx = new ggml_backend_meta_buffer_context;
    size_t max_size = 0;
    buf_ctx->buf_configs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_backend_buffer_t simple_buf = ggml_backend_buft_alloc_buffer(ggml_backend_meta_buft_simple_buft(buft, i), size);
        max_size = std::max(max_size, ggml_backend_buffer_get_size(simple_buf));
        buf_ctx->buf_configs.emplace_back(ggml_init(params), simple_buf);
    }

    return ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

struct ggml_backend_buffer * ggml_backend_meta_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*1024, // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_backend_meta_buffer_context * meta_buf_ctx = new ggml_backend_meta_buffer_context;
    meta_buf_ctx->buf_configs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        meta_buf_ctx->buf_configs.emplace_back(ggml_init(params), nullptr);
    }

    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor(meta_buf, t);
        t->data = (void *) 0x2000000000000000; // FIXME
    }
    for (size_t i = 0; i < n_simple_bufts; i++) {
        meta_buf_ctx->buf_configs[i].buf = ggml_backend_alloc_ctx_tensors_from_buft(
            meta_buf_ctx->buf_configs[i].ctx, ggml_backend_meta_buft_simple_buft(buft, i));
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->buf_configs[i].buf));
    }
    return meta_buf;
}

//
// meta backend
//

static ggml_guid_t ggml_backend_meta_guid() {
    static ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    return &guid;
}

struct ggml_backend_meta_context {
    struct cgraph_config {
        ggml_cgraph cgraph_main;
        int         offset; // Node offset vs. original graph, only used for debugging.

        std::vector<ggml_cgraph>   cgraphs_aux;
        std::vector<ggml_tensor *> nodes_aux;

        cgraph_config(ggml_cgraph cgraph_main, int offset) : cgraph_main(cgraph_main), offset(offset) {}
    };
    struct backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        ggml_context                       * ctx = nullptr;
        std::vector<ggml_backend_buffer_t>   bufs; // Multiple buffers to reduce synchronizations.

        backend_config(ggml_backend_t backend) : backend(backend) {}

        ~backend_config() {
            for (ggml_backend_buffer_t buf : bufs) {
                ggml_backend_buffer_free(buf);
            }
            ggml_free(ctx);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    size_t                      max_tmp_size  = 0;
    size_t                      max_subgraphs = 0;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, const char * params) {
        const size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        name = "Meta(";
        backend_configs.reserve(n_devs);
        for (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            backend_configs.emplace_back(ggml_backend_dev_init(simple_dev, params));
        }
        name += ")";
    }

    ~ggml_backend_meta_context() {
        for (auto & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }

    size_t n_reduce_steps() const {
        return std::ceil(std::log2(backend_configs.size()));
    }

    ggml_tensor * get_next_tensor(size_t j, std::vector<ggml_tensor *> & tensors, ggml_tensor * node) {
        ggml_tensor * next = tensors[j] == nullptr ? ggml_get_first_tensor(backend_configs[j].ctx)
            : ggml_get_next_tensor(backend_configs[j].ctx, tensors[j]);
        if (next == nullptr) {
            next = ggml_new_tensor_1d(backend_configs[j].ctx, GGML_TYPE_F32, 1);
        }
        memset(next, 0, sizeof(ggml_tensor));
        next->op   = GGML_OP_NONE;
        next->type = node->type;
        for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
            next->ne[dim] = node->ne[dim];
            next->nb[dim] = node->nb[dim];
        }
        tensors[j] = next;
        return next;
    }
};

static const char * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) backend->context;
    return backend_ctx->name.c_str();
}

static void ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    delete backend_ctx;
    delete backend;
}

static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(ggml_backend_meta_get_split_state(tensor, false) == GGML_BACKEND_SPLIT_STATE_MIRRORED);
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_tensor_set_async(
            ggml_backend_meta_simple_backend(backend, i), ggml_backend_meta_buffer_simple_tensor(tensor, i), data, offset, size);
    }
}

static void ggml_backend_meta_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(ggml_backend_meta_get_split_state(tensor, false) == GGML_BACKEND_SPLIT_STATE_MIRRORED);
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(n_backends >= 1);
    ggml_backend_tensor_get_async( // TODO other backends may be more optimal
        ggml_backend_meta_simple_backend(backend, 0), ggml_backend_meta_buffer_simple_tensor(tensor, 0), data, offset, size);
}

static void ggml_backend_meta_synchronize(ggml_backend_t backend) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    const size_t n_reduce_steps = backend_ctx->n_reduce_steps();

    for (size_t j = 0; j < n_backends; j++) {
        auto & bcj = backend_ctx->backend_configs[j];
        bcj.cgraphs.clear();
        bcj.nodes.clear();
        bcj.nodes.reserve(cgraph->n_nodes*n_reduce_steps);

        for (int i = 0; i < cgraph->n_nodes; i++) {
            bcj.nodes.push_back(ggml_backend_meta_buffer_simple_tensor(cgraph->nodes[i], j));
            GGML_ASSERT(bcj.nodes[i]);
        }
    }

    size_t n_subgraphs  = 0;
    size_t max_tmp_size = 0;
    {
        int i_start = 0;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];
            const bool partial = ggml_backend_meta_get_split_state(node, /*assume_sync =*/ false) == GGML_BACKEND_SPLIT_STATE_PARTIAL;
            if (partial) {
                max_tmp_size = std::max(max_tmp_size, ggml_nbytes(node));
            }
            const bool new_subgraph = i + 1 == cgraph->n_nodes || partial;
            if (!new_subgraph) {
                continue;
            }

            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                bcj.cgraphs.emplace_back(*cgraph, i_start);
                bcj.cgraphs.back().cgraph_main.nodes = bcj.nodes.data() + i_start;
                bcj.cgraphs.back().cgraph_main.n_nodes = i + 1 - i_start;
            }
            n_subgraphs++;
            i_start = i + 1;
        }
        GGML_ASSERT(i_start == cgraph->n_nodes);
    }

    if (max_tmp_size > backend_ctx->max_tmp_size) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            for (ggml_backend_buffer_t buf : bcj.bufs) {
                ggml_backend_buffer_free(buf);
            }
            bcj.bufs.clear();
            for (size_t k = 0; k < n_reduce_steps + 1; k++) {
                bcj.bufs.push_back(ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
            }
        }
        backend_ctx->max_tmp_size = max_tmp_size;
    }
    if (n_subgraphs > backend_ctx->max_subgraphs) {
        ggml_init_params params = {
            /*.mem_size   =*/ n_subgraphs*n_reduce_steps*2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_free(bcj.ctx);
            bcj.ctx = ggml_init(params);
        }
        backend_ctx->max_subgraphs = n_subgraphs;
    }

    size_t i_buf = 0; // Alternate between tmp buffers per simple backend to reduce synchronizations.
    std::vector<ggml_tensor *> tensors(n_backends, nullptr);

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    auto allreduce_fallback = [&](size_t i) -> ggml_status {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];

            bcj.cgraphs[i].cgraphs_aux.clear();
            bcj.cgraphs[i].cgraphs_aux.reserve(n_reduce_steps);
            bcj.cgraphs[i].nodes_aux.clear();
            bcj.cgraphs[i].nodes_aux.reserve(n_reduce_steps*2);
        }

        for (size_t offset_j = 1; offset_j < n_backends; offset_j *= 2) {
            for (size_t j = 0; j < n_backends; j++) {
                const size_t j_other = j ^ offset_j;
                if (j_other > j) {
                    continue;
                }

                auto & bcj1 = backend_ctx->backend_configs[j];
                auto & bcj2 = backend_ctx->backend_configs[j_other];

                ggml_tensor * node1 = bcj1.cgraphs[i].cgraph_main.nodes[bcj1.cgraphs[i].cgraph_main.n_nodes-1];
                ggml_tensor * node2 = bcj2.cgraphs[i].cgraph_main.nodes[bcj2.cgraphs[i].cgraph_main.n_nodes-1];
                GGML_ASSERT(ggml_is_contiguous(node1));
                GGML_ASSERT(ggml_is_contiguous(node2));

                ggml_tensor * node_tmp_1 = backend_ctx->get_next_tensor(j,       tensors, node1);
                ggml_tensor * node_tmp_2 = backend_ctx->get_next_tensor(j_other, tensors, node2);
                node_tmp_1->buffer = bcj1.bufs[i_buf];
                node_tmp_2->buffer = bcj2.bufs[i_buf];
                node_tmp_1->data = ggml_backend_buffer_get_base(bcj1.bufs[i_buf]);
                node_tmp_2->data = ggml_backend_buffer_get_base(bcj2.bufs[i_buf]);
                bcj1.cgraphs[i].nodes_aux.push_back(node_tmp_1);
                bcj2.cgraphs[i].nodes_aux.push_back(node_tmp_2);

                ggml_backend_tensor_shfl_async(bcj1.backend, bcj2.backend, node1, node2, node_tmp_1, node_tmp_2);

                ggml_tensor * node_red_1 = backend_ctx->get_next_tensor(j,       tensors, node1);
                ggml_tensor * node_red_2 = backend_ctx->get_next_tensor(j_other, tensors, node2);
                node_red_1->view_src = node1->view_src == nullptr ? node1 : node1->view_src;
                node_red_2->view_src = node2->view_src == nullptr ? node2 : node2->view_src;
                node_red_1->view_offs = node1->view_offs;
                node_red_2->view_offs = node2->view_offs;
                node_red_1->op = GGML_OP_ADD;
                node_red_2->op = GGML_OP_ADD;
                node_red_1->src[0] = node1;
                node_red_2->src[0] = node2;
                node_red_1->src[1] = node_tmp_1;
                node_red_2->src[1] = node_tmp_2;
                node_red_1->flags |= GGML_TENSOR_FLAG_COMPUTE;
                node_red_2->flags |= GGML_TENSOR_FLAG_COMPUTE;
                ggml_backend_view_init(node_red_1);
                ggml_backend_view_init(node_red_2);
                bcj1.cgraphs[i].nodes_aux.push_back(node_red_1);
                bcj2.cgraphs[i].nodes_aux.push_back(node_red_2);

                bcj1.cgraphs[i].cgraphs_aux.push_back(*cgraph);
                bcj2.cgraphs[i].cgraphs_aux.push_back(*cgraph);
                bcj1.cgraphs[i].cgraphs_aux.back().nodes = &bcj1.cgraphs[i].nodes_aux.back();
                bcj2.cgraphs[i].cgraphs_aux.back().nodes = &bcj2.cgraphs[i].nodes_aux.back();
                bcj1.cgraphs[i].cgraphs_aux.back().n_nodes = 1;
                bcj2.cgraphs[i].cgraphs_aux.back().n_nodes = 1;

                i_buf = (i_buf + 1) % (n_reduce_steps + 1);
            }

            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, &bcj.cgraphs[i].cgraphs_aux.back());
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    };


    for (size_t i = 0; i < n_subgraphs; i++) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, &bcj.cgraphs[i].cgraph_main);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }

        if (i < n_subgraphs - 1) {
            bool backend_allreduce_success = false;
            if (backend_ctx->backend_configs[0].backend->iface.allreduce_tensor_async) {
                std::vector<ggml_backend_t> backends;
                backends.reserve(n_backends);
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    backends.push_back(bcj.backend);
                    nodes.push_back(bcj.cgraphs[i].cgraph_main.nodes[bcj.cgraphs[i].cgraph_main.n_nodes-1]);
                    GGML_ASSERT(nodes.back()->type == GGML_TYPE_F32);
                    GGML_ASSERT(ggml_is_contiguous(nodes.back()));
                }
                backend_allreduce_success = backend_ctx->backend_configs[0].backend->iface.allreduce_tensor_async(
                    backends.data(), nodes.data(), n_backends);
            }

            if (!backend_allreduce_success) {
                const ggml_status status = allreduce_fallback(i);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
        }
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .shfl_tensor_async       = */ nullptr,
    /* .allreduce_tensor_async  = */ nullptr,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_meta_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

bool ggml_backend_is_meta(ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_meta_context * backend_ctx = new ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = new struct ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    return backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs[index].backend;
}

enum ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;

    auto handle_generic = [&](const std::vector<ggml_backend_meta_split_state> & src_split_states, bool scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state homogeneous_src_split_state = GGML_BACKEND_SPLIT_STATE_NONE;
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                continue;
            }
            if (homogeneous_src_split_state == GGML_BACKEND_SPLIT_STATE_NONE) {
                homogeneous_src_split_state = src_split_states[i];
            } else if (src_split_states[i] != homogeneous_src_split_state) {
                homogeneous_src_split_state = GGML_BACKEND_SPLIT_STATE_UNKNOWN;
            }
        }
        if (homogeneous_src_split_state == GGML_BACKEND_SPLIT_STATE_NONE) {
            homogeneous_src_split_state = GGML_BACKEND_SPLIT_STATE_UNKNOWN;
        }
        if (scalar_only && homogeneous_src_split_state >= 0 && homogeneous_src_split_state < GGML_MAX_DIMS) {
            homogeneous_src_split_state = GGML_BACKEND_SPLIT_STATE_UNKNOWN;
        }
        GGML_ASSERT(homogeneous_src_split_state != GGML_BACKEND_SPLIT_STATE_UNKNOWN);
        return homogeneous_src_split_state;
    };

    // Some ops process data on a per-row bases:
    auto handle_per_row = [&](const std::vector<ggml_backend_meta_split_state> & src_split_states) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_split_states[0] != GGML_BACKEND_SPLIT_STATE_BY_NE0);
        return src_split_states[0];
    };

    // Some ops broadcast the src1 data across src0:
    auto handle_bin_bcast = [&](const std::vector<ggml_backend_meta_split_state> & src_split_states) -> ggml_backend_meta_split_state {
        if (src_split_states[0] >= 0 && src_split_states[0] < GGML_MAX_DIMS &&
                tensor->src[1]->ne[int(src_split_states[0])] == 1 && src_split_states[1] == GGML_BACKEND_SPLIT_STATE_MIRRORED) {
            return src_split_states[0];
        }
        if (src_split_states[0] == src_split_states[1] && src_split_states[2] == GGML_BACKEND_SPLIT_STATE_MIRRORED) {
            return src_split_states[0]; // GGML_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == nullptr || src_split_states[2] == GGML_BACKEND_SPLIT_STATE_MIRRORED);
        return handle_generic(src_split_states, /*scalar_only =*/ false);
    };

    auto handle_mul_mat = [&](const std::vector<ggml_backend_meta_split_state> & src_split_states) -> ggml_backend_meta_split_state {
        if (src_split_states[0] == GGML_BACKEND_SPLIT_STATE_MIRRORED && src_split_states[1] == GGML_BACKEND_SPLIT_STATE_MIRRORED) {
            return GGML_BACKEND_SPLIT_STATE_MIRRORED;
        }
        if (src_split_states[0] == GGML_BACKEND_SPLIT_STATE_BY_NE1 && src_split_states[1] == GGML_BACKEND_SPLIT_STATE_MIRRORED) {
            return GGML_BACKEND_SPLIT_STATE_BY_NE0;
        }
        if (src_split_states[0] == GGML_BACKEND_SPLIT_STATE_BY_NE0 && src_split_states[1] == GGML_BACKEND_SPLIT_STATE_BY_NE0) {
            return assume_sync ? GGML_BACKEND_SPLIT_STATE_MIRRORED : GGML_BACKEND_SPLIT_STATE_PARTIAL;
        }
        GGML_ABORT("fatal error");
        return GGML_BACKEND_SPLIT_STATE_UNKNOWN;
    };

    auto handle_reshape = [&](const std::vector<ggml_backend_meta_split_state> & src_split_states) -> ggml_backend_meta_split_state {
        switch (src_split_states[0]) {
            case GGML_BACKEND_SPLIT_STATE_BY_NE0:
            case GGML_BACKEND_SPLIT_STATE_BY_NE1:
            case GGML_BACKEND_SPLIT_STATE_BY_NE2:
            case GGML_BACKEND_SPLIT_STATE_BY_NE3: {
                GGML_ASSERT(ggml_is_contiguous(tensor));
                int64_t base_ne_in = 1;
                for (int dim = 0; dim <= int(src_split_states[0]); dim++) {
                    base_ne_in *= tensor->src[0]->ne[dim];
                }
                int64_t base_ne_out = 1;
                for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    const int64_t base_ne_out_next = base_ne_out *= tensor->ne[dim];
                    if (base_ne_out_next == base_ne_in) {
                        return ggml_backend_meta_split_state(dim);
                    }
                    base_ne_out = base_ne_out_next;
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            case GGML_BACKEND_SPLIT_STATE_MIRRORED:
            case GGML_BACKEND_SPLIT_STATE_PARTIAL: {
                return src_split_states[0];
            }
            default: {
                GGML_ABORT("fatal error");
                return GGML_BACKEND_SPLIT_STATE_UNKNOWN;
            }
        }
    };

    auto handle_view = [&](const std::vector<ggml_backend_meta_split_state> & src_split_states) -> ggml_backend_meta_split_state {
        if (ggml_is_contiguous(tensor)) {
            return handle_reshape(src_split_states);
        }
        if (src_split_states[0] == GGML_BACKEND_SPLIT_STATE_MIRRORED || src_split_states[0] == GGML_BACKEND_SPLIT_STATE_PARTIAL) {
            return src_split_states[0];
        }
        GGML_ABORT("non-contioguos view not implemented");
        return GGML_BACKEND_SPLIT_STATE_UNKNOWN;
    };

    auto handle_permute = [&](const std::vector<ggml_backend_meta_split_state> & src_split_states) -> ggml_backend_meta_split_state {
        switch (src_split_states[0]) {
            case GGML_BACKEND_SPLIT_STATE_BY_NE0:
            case GGML_BACKEND_SPLIT_STATE_BY_NE1:
            case GGML_BACKEND_SPLIT_STATE_BY_NE2:
            case GGML_BACKEND_SPLIT_STATE_BY_NE3: {
                return ggml_backend_meta_split_state(tensor->op_params[int(src_split_states[0])]);
            }
            case GGML_BACKEND_SPLIT_STATE_MIRRORED:
            case GGML_BACKEND_SPLIT_STATE_PARTIAL: {
                return src_split_states[0];
            }
            default: {
                GGML_ABORT("fatal error");
                return GGML_BACKEND_SPLIT_STATE_UNKNOWN;
            }
        }
    };

    auto handle_set_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_split_states) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_split_states[0] == GGML_BACKEND_SPLIT_STATE_BY_NE0);
        GGML_ASSERT(src_split_states[1] == GGML_BACKEND_SPLIT_STATE_MIRRORED);
        GGML_ASSERT(src_split_states[0] == GGML_BACKEND_SPLIT_STATE_BY_NE0);
        return src_split_states[0];
    };

    auto handle_rope = [&](const std::vector<ggml_backend_meta_split_state> & src_split_states) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_split_states[1] == GGML_BACKEND_SPLIT_STATE_MIRRORED);
        return src_split_states[0];
    };

    auto handle_flash_attn_ext = [&](const std::vector<ggml_backend_meta_split_state> & src_split_states) -> ggml_backend_meta_split_state {
        GGML_ASSERT(                             src_split_states[0] == GGML_BACKEND_SPLIT_STATE_BY_NE2);
        GGML_ASSERT(                             src_split_states[1] == GGML_BACKEND_SPLIT_STATE_BY_NE2);
        GGML_ASSERT(                             src_split_states[2] == GGML_BACKEND_SPLIT_STATE_BY_NE2);
        GGML_ASSERT(tensor->src[4] == nullptr || src_split_states[3] == GGML_BACKEND_SPLIT_STATE_MIRRORED);
        GGML_ASSERT(tensor->src[4] == nullptr || src_split_states[4] == GGML_BACKEND_SPLIT_STATE_BY_NE0);
        return GGML_BACKEND_SPLIT_STATE_BY_NE1;
    };

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        if (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
            return dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
        }

        std::vector<ggml_backend_meta_split_state> src_split_states(GGML_MAX_SRC, GGML_BACKEND_SPLIT_STATE_NONE);
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_split_states[i] = GGML_BACKEND_SPLIT_STATE_UNKNOWN;
                continue;
            }
            src_split_states[i] = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
        }

        switch (tensor->op) {
            case GGML_OP_NONE: {
                return GGML_BACKEND_SPLIT_STATE_MIRRORED;
            }
            case GGML_OP_DUP: {
                return handle_generic(src_split_states, /*scalar_only =*/ true);
            }
            case GGML_OP_ADD:
            case GGML_OP_ADD_ID: {
                return handle_bin_bcast(src_split_states);
            }
            case GGML_OP_ADD1:
            case GGML_OP_ACC: {
                return handle_generic(src_split_states, /*scalar_only =*/ true);
            }
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV: {
                return handle_bin_bcast(src_split_states);
            }
            case GGML_OP_SQR:
            case GGML_OP_SQRT:
            case GGML_OP_LOG:
            case GGML_OP_SIN:
            case GGML_OP_COS: {
                return handle_generic(src_split_states, /*scalar_only =*/ false);
            }
            case GGML_OP_SUM: {
                return handle_generic(src_split_states, /*scalar_only =*/ true);
            }
            case GGML_OP_SUM_ROWS:
            case GGML_OP_CUMSUM:
            case GGML_OP_MEAN:
            case GGML_OP_ARGMAX:
            case GGML_OP_COUNT_EQUAL: {
                return handle_per_row(src_split_states);
            }
            case GGML_OP_REPEAT:
            case GGML_OP_REPEAT_BACK:
            case GGML_OP_CONCAT: {
                return handle_generic(src_split_states, /*scalar_only =*/ true);
            }
            case GGML_OP_SILU_BACK: {
                return handle_generic(src_split_states, /*scalar_only =*/ false);
            }
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_RMS_NORM_BACK:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM: {
                return handle_per_row(src_split_states);
            }
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID: {
                return handle_mul_mat(src_split_states);
            }
            case GGML_OP_OUT_PROD: {
                return handle_generic(src_split_states, /*scalar_only =*/ true);
            }
            case GGML_OP_SCALE: {
                return handle_generic(src_split_states, /*scalar_only =*/ false);
            }
            case GGML_OP_SET:
            case GGML_OP_CPY:
            case GGML_OP_CONT: {
                return handle_generic(src_split_states, /*scalar_only =*/ true);
            }
            case GGML_OP_RESHAPE: {
                return handle_reshape(src_split_states);
            }
            case GGML_OP_VIEW: {
                return handle_view(src_split_states);
            }
            case GGML_OP_PERMUTE: {
                return handle_permute(src_split_states);
            }
            case GGML_OP_TRANSPOSE:
            case GGML_OP_GET_ROWS:
            case GGML_OP_GET_ROWS_BACK: {
                return handle_generic(src_split_states, /*scalar_only =*/ true);
            }
            case GGML_OP_SET_ROWS: {
                return handle_set_rows(src_split_states);
            }
            case GGML_OP_DIAG:
            case GGML_OP_DIAG_MASK_INF:
            case GGML_OP_DIAG_MASK_ZERO: {
                return handle_generic(src_split_states, /*scalar_only =*/ true);
            }
            case GGML_OP_SOFT_MAX:
            case GGML_OP_SOFT_MAX_BACK: {
                return handle_generic(src_split_states, /*scalar_only =*/ false);
            }
            case GGML_OP_ROPE: {
                return handle_rope(src_split_states);
            }
            case GGML_OP_ROPE_BACK: {
                return handle_generic(src_split_states, /*scalar_only =*/ true);
            }
            case GGML_OP_CLAMP: {
                return handle_generic(src_split_states, /*scalar_only =*/ false);
            }
            case GGML_OP_CONV_TRANSPOSE_1D:
            case GGML_OP_IM2COL:
            case GGML_OP_IM2COL_BACK:
            case GGML_OP_IM2COL_3D:
            case GGML_OP_CONV_2D:
            case GGML_OP_CONV_3D:
            case GGML_OP_CONV_2D_DW:
            case GGML_OP_CONV_TRANSPOSE_2D:
            case GGML_OP_POOL_1D:
            case GGML_OP_POOL_2D:
            case GGML_OP_POOL_2D_BACK:
            case GGML_OP_UPSCALE:
            case GGML_OP_PAD:
            case GGML_OP_PAD_REFLECT_1D:
            case GGML_OP_ROLL:
            case GGML_OP_ARANGE:
            case GGML_OP_TIMESTEP_EMBEDDING: {
                return handle_generic(src_split_states, /*scalar_only =*/ true);
            }
            case GGML_OP_ARGSORT:
            case GGML_OP_TOP_K: {
                return handle_per_row(src_split_states);
            }
            case GGML_OP_LEAKY_RELU: {
                return handle_generic(src_split_states, /*scalar_only =*/ false);
            }
            case GGML_OP_TRI:
            case GGML_OP_FILL: {
                return handle_generic(src_split_states, /*scalar_only =*/ true);
            }
            case GGML_OP_FLASH_ATTN_EXT: {
                return handle_flash_attn_ext(src_split_states);
            }
            case GGML_OP_FLASH_ATTN_BACK:
            case GGML_OP_SSM_CONV:
            case GGML_OP_SSM_SCAN:
            case GGML_OP_WIN_PART:
            case GGML_OP_WIN_UNPART:
            case GGML_OP_GET_REL_POS:
            case GGML_OP_ADD_REL_POS:
            case GGML_OP_RWKV_WKV6:
            case GGML_OP_GATED_LINEAR_ATTN:
            case GGML_OP_RWKV_WKV7:
            case GGML_OP_SOLVE_TRI: {
                return handle_generic(src_split_states, /*scalar_only =*/ true);
            }
            case GGML_OP_UNARY: {
                return handle_generic(src_split_states, /*scalar_only =*/ false);
            }
            case GGML_OP_MAP_CUSTOM1:
            case GGML_OP_MAP_CUSTOM2:
            case GGML_OP_MAP_CUSTOM3:
            case GGML_OP_CUSTOM: {
                return handle_generic(src_split_states, /*scalar_only =*/ true);
            }
            case GGML_OP_CROSS_ENTROPY_LOSS:
            case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                return handle_per_row(src_split_states);
            }
            case GGML_OP_OPT_STEP_ADAMW:
            case GGML_OP_OPT_STEP_SGD:
            case GGML_OP_GLU: {
                return handle_generic(src_split_states, /*scalar_only =*/ false);
            }
            default: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                return GGML_BACKEND_SPLIT_STATE_UNKNOWN;
            }
        }

    };

    const std::pair key = std::make_pair(tensor, assume_sync);

    if (buf_ctx->split_state_cache.find(key) == buf_ctx->split_state_cache.end()) {
        buf_ctx->split_state_cache[key] = calculate_split_state();
    }

    ggml_backend_meta_split_state ret = buf_ctx->split_state_cache[key];
    GGML_ASSERT(ret != GGML_BACKEND_SPLIT_STATE_NONE);
    if (assume_sync && ret == GGML_BACKEND_SPLIT_STATE_UNKNOWN) {
        GGML_ABORT("fatal error");
        ret = GGML_BACKEND_SPLIT_STATE_MIRRORED;
    }
    return ret;
}
