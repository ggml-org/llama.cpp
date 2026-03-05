#pragma once
#include "ggml-backend-impl.h"
#include "ggml-cpu-impl.h"
#include "ggml.h"

#ifdef __cplusplus
#    include "ggml_cpp_wrapper.h"
#    include <vector>
#    include <string>
extern "C" {
#endif

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op);
bool ggml_cpu_extra_work_size(int n_threads, const struct ggml_tensor * op, size_t * size);

#ifdef __cplusplus
}

namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  public:
    virtual ~tensor_traits();
    virtual bool work_size(int n_threads, const struct ggml_tensor * op, size_t & size)        = 0;
    virtual bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) = 0;
};

// a simple buffer for cpu
class buffer : public ggml::cpp::backend::buffer {
public:
    buffer(std::size_t size);
    virtual ~buffer();
    void* get_base() override;
    void memset_tensor(ggml_tensor & tensor, uint8_t value, std::size_t offset, std::size_t size) override;
    void get_tensor(const ggml_tensor &, void *, std::size_t, std::size_t size) override;
    void clear(uint8_t value) override;
protected:
    struct alignas(TENSOR_ALIGNMENT) aligned_uint8_t { uint8_t val; };
    const std::size_t m_size;
    aligned_uint8_t* m_data;
};

class extra_buffer_type {
  public:
    virtual ~extra_buffer_type();
    // the base buffer_type fct
    virtual const std::string& get_name() = 0;
    virtual ggml::cpp::backend::buffer* alloc_buffer(std::size_t size) = 0;
    virtual std::size_t get_alignment() { return TENSOR_ALIGNMENT; }
    virtual std::size_t get_max_size() { return SIZE_MAX; }
    virtual std::size_t get_alloc_size(const ggml_tensor& tensor) { return ggml_nbytes(&tensor); }

    // the extra fct
    virtual bool            supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) = 0;
    virtual tensor_traits * get_tensor_traits(const struct ggml_tensor * op)                   = 0;
};

ggml_backend_buffer_type_t c_wrapper(extra_buffer_type* ctx);

}  // namespace ggml::cpu

// implemented in ggml-cpu.cpp.
std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types();

#endif
