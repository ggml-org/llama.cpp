#ifndef GGML_SYCL_TENSOR_HPP
#define GGML_SYCL_TENSOR_HPP

namespace ggml_sycl {

class sycl_tensor {
public:
    sycl_tensor() = default;
    sycl_tensor(const ggml_tensor * tensor, int device) : tensor_(tensor), device_(device) {}

    explicit operator bool() const {
        return tensor_ != nullptr;
    }

    const ggml_tensor * raw() const {
        return tensor_;
    }

    int device() const {
        return device_;
    }

    ggml_type type() const {
        return tensor_ ? tensor_->type : GGML_TYPE_COUNT;
    }

    ggml_op op() const {
        return tensor_ ? tensor_->op : GGML_OP_NONE;
    }

    const void * op_params() const {
        return tensor_ ? tensor_->op_params : nullptr;
    }

    int32_t flags() const {
        return tensor_ ? tensor_->flags : 0;
    }

    const char * name() const {
        return tensor_ ? tensor_->name : nullptr;
    }

    int64_t ne(int index) const {
        return tensor_ ? tensor_->ne[index] : 0;
    }

    size_t nb(int index) const {
        return tensor_ ? tensor_->nb[index] : 0;
    }

    const int64_t * ne_arr() const {
        return tensor_ ? tensor_->ne : nullptr;
    }

    const size_t * nb_arr() const {
        return tensor_ ? tensor_->nb : nullptr;
    }

    int64_t nelements() const {
        return tensor_ ? ggml_nelements(tensor_) : 0;
    }

    size_t nbytes() const {
        return tensor_ ? ggml_nbytes(tensor_) : 0;
    }

    bool is_contiguous() const {
        return tensor_ ? ggml_is_contiguous(tensor_) : false;
    }

    sycl_tensor src(int index) const {
        return tensor_ ? sycl_tensor(tensor_->src[index], device_) : sycl_tensor();
    }

    sycl_tensor view_src() const {
        return tensor_ ? sycl_tensor(tensor_->view_src, device_) : sycl_tensor();
    }

    size_t view_offs() const {
        return tensor_ ? tensor_->view_offs : 0;
    }

    ggml_backend_buffer * buffer() const {
        return tensor_ ? tensor_->buffer : nullptr;
    }

    ggml_tensor_extra_gpu * extra_gpu() const {
        return tensor_ ? static_cast<ggml_tensor_extra_gpu *>(tensor_->extra) : nullptr;
    }

    void * resolve_ptr() const {
        return tensor_ ? ggml_sycl_resolve_tensor_ptr(tensor_, device_) : nullptr;
    }

    template <typename T>
    T * resolve_as() const {
        return static_cast<T *>(resolve_ptr());
    }

private:
    const ggml_tensor * tensor_ = nullptr;
    int                 device_ = 0;
};

static_assert(sizeof(sycl_tensor) == 16, "sycl_tensor must stay lightweight");

}  // namespace ggml_sycl

#endif  // GGML_SYCL_TENSOR_HPP
