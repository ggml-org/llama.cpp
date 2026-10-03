#include "mtmd-projector-storage.h"
#include "ggml-alloc.h"
#include "gguf.h"

#include <algorithm>
#include <limits>
#include <map>
#include <mutex>
#include <unordered_set>

struct projector_file_deleter {
    void operator()(FILE * file) const { if (file) std::fclose(file); }
};
using projector_file = std::unique_ptr<FILE,projector_file_deleter>;

struct mtmd_projector_source::implementation {
    projector_file file;
    std::map<std::string,mtmd_projector_tensor_source,std::less<>> tensors;
    mutable std::mutex cursor;
};

mtmd_projector_source::mtmd_projector_source(std::unique_ptr<implementation> impl) : impl(std::move(impl)) {}
mtmd_projector_source::~mtmd_projector_source() = default;
std::shared_ptr<const mtmd_projector_source> mtmd_projector_source::open(const char * path,const gguf_context * metadata) {
    return path && *path && metadata ? from_file(ggml_fopen(path,"rb"),metadata) : nullptr;
}

// Copy file-relative descriptions while retaining the original file, not the loader's GGUF context.
std::shared_ptr<const mtmd_projector_source> mtmd_projector_source::from_file(FILE * file,const gguf_context * metadata) {
    projector_file owned(file);
    if (!owned || !metadata) return {};
    auto state = std::make_unique<implementation>();
    state->file = std::move(owned);
    const size_t data = gguf_get_data_offset(metadata);
    if (!data) return {};
    for (int64_t i = 0; i < gguf_get_n_tensors(metadata); ++i) {
        const char * name = gguf_get_tensor_name(metadata,i);
        const size_t offset = gguf_get_tensor_offset(metadata,i), bytes = gguf_get_tensor_size(metadata,i);
        if (!name || !*name || !bytes || offset > SIZE_MAX-data || bytes > SIZE_MAX-data-offset) return {};
        mtmd_projector_tensor_source entry{name,gguf_get_tensor_type(metadata,i),{},data+offset,bytes};
        if (entry.offset > uint64_t(INT64_MAX) || entry.bytes > uint64_t(INT64_MAX)-entry.offset) return {};
        std::copy_n(gguf_get_tensor_ne(metadata,i),GGML_MAX_DIMS,entry.shape.begin());
        if (!state->tensors.emplace(entry.name,entry).second) return {};
    }
    return std::shared_ptr<const mtmd_projector_source>(new mtmd_projector_source(std::move(state)));
}

const mtmd_projector_tensor_source * mtmd_projector_source::find(const char * name) const noexcept {
    if (!name) return nullptr;
    const auto found = impl->tensors.find(name);
    return found != impl->tensors.end() ? &found->second : nullptr;
}

// Serialize only the file cursor; independent modality owners can share the same immutable manifest.
bool mtmd_projector_source::read(const char * name,void * output,size_t bytes) const {
    const auto * entry = find(name);
    if (!entry || !output || bytes != entry->bytes) return false;
    std::lock_guard<std::mutex> guard(impl->cursor);
    auto * file = impl->file.get();
    std::clearerr(file);
#ifdef _WIN32
    if (_fseeki64(file,int64_t(entry->offset),SEEK_SET)) return false;
#else
    if (entry->offset > uint64_t(std::numeric_limits<off_t>::max()) || fseeko(file,off_t(entry->offset),SEEK_SET)) return false;
#endif
    return std::fread(output,1,bytes,file) == bytes;
}

// Adopt descriptors only after every selected tensor matches an unbound file entry.
std::shared_ptr<mtmd_projector_metadata> mtmd_projector_metadata::create(ggml_context_ptr context,
        std::shared_ptr<const mtmd_projector_source> source) {
    if (!context || !source || !ggml_get_no_alloc(context.get()) || !ggml_get_first_tensor(context.get())) return {};
    std::unordered_set<std::string> names;
    for (auto * tensor = ggml_get_first_tensor(context.get()); tensor; tensor = ggml_get_next_tensor(context.get(),tensor)) {
        const auto * entry = source->find(tensor->name);
        if (!entry || tensor->buffer || tensor->data || tensor->extra || tensor->view_src ||
                tensor->op != GGML_OP_NONE || tensor->type != entry->type ||
                !std::equal(entry->shape.begin(),entry->shape.end(),tensor->ne) ||
                !ggml_is_contiguous(tensor) || ggml_nbytes(tensor) != entry->bytes || !names.emplace(tensor->name).second) return {};
    }
    auto result = std::shared_ptr<mtmd_projector_metadata>(new mtmd_projector_metadata);
    result->tensors = std::move(context);
    result->reload = std::move(source);
    return result;
}
ggml_context * mtmd_projector_metadata::context() const noexcept { return tensors.get(); }
const std::shared_ptr<const mtmd_projector_source> & mtmd_projector_metadata::source() const noexcept { return reload; }

// Keep a failed candidate private and clear its bindings before another attempt can use the descriptors.
std::shared_ptr<mtmd_projector_weights> mtmd_projector_weights::allocate(std::shared_ptr<mtmd_projector_metadata> metadata,
        ggml_backend_buffer_type_t type,bool skip_upload,mtmd_progress_callback progress,void * user_data) {
    if (!metadata || !type || metadata->loading) return {};
    size_t total = 0;
    for (auto * tensor = ggml_get_first_tensor(metadata->context()); tensor; tensor = ggml_get_next_tensor(metadata->context(),tensor)) {
        if (tensor->data || tensor->buffer || tensor->extra || ggml_nbytes(tensor) > SIZE_MAX-total) return {};
        total += ggml_nbytes(tensor);
    }
    struct loading_guard {
        bool & loading;
        explicit loading_guard(bool & loading) : loading(loading) { loading = true; }
        ~loading_guard() { loading = false; }
    } guard(metadata->loading);
    if (progress && !progress(0.0f,user_data)) return {};
    auto result = std::shared_ptr<mtmd_projector_weights>(new mtmd_projector_weights);
    result->descriptors = metadata;
    result->storage.reset(ggml_backend_alloc_ctx_tensors_from_buft(result->descriptors->context(),type));
    if (!result->storage) return {};
    ggml_backend_buffer_set_usage(result->storage.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (!skip_upload) {
        size_t loaded = 0;
        std::vector<uint8_t> staging;
        for (auto * tensor = ggml_get_first_tensor(result->descriptors->context()); tensor;
                tensor = ggml_get_next_tensor(result->descriptors->context(),tensor)) {
            const size_t bytes = ggml_nbytes(tensor);
            void * destination = tensor->data;
            if (!ggml_backend_buft_is_host(type)) { staging.resize(bytes); destination = staging.data(); }
            if (!result->descriptors->source()->read(tensor->name,destination,bytes)) return {};
            if (!ggml_backend_buft_is_host(type)) ggml_backend_tensor_set(tensor,destination,0,bytes);
            loaded += bytes;
            if (progress && !progress(float(loaded)/float(total),user_data)) return {};
        }
    }
    return result;
}

// Backends may inspect descriptors during buffer release; clear raw bindings after that release completes.
mtmd_projector_weights::~mtmd_projector_weights() {
    storage.reset();
    if (!descriptors) return;
    for (auto * tensor = ggml_get_first_tensor(descriptors->context()); tensor; tensor = ggml_get_next_tensor(descriptors->context(),tensor)) {
        tensor->buffer = nullptr; tensor->data = nullptr; tensor->extra = nullptr;
    }
}
ggml_backend_buffer_t mtmd_projector_weights::buffer() const noexcept { return storage.get(); }
const std::shared_ptr<mtmd_projector_metadata> & mtmd_projector_weights::metadata() const noexcept { return descriptors; }
