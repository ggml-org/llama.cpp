#pragma once
#ifndef __cplusplus
#error "This header is for C++ only"
#endif

#include "ggml.h"

#include "ggml-impl.h"
#include "ggml-backend.h"

#include <string>
#include <cstddef>

namespace ggml::cpp::backend {

    class GGML_API buffer { // ggml_backend_buffer_t
    public:
        virtual ~buffer();

        virtual void* get_base() = 0;
        virtual ggml_status init_tensor(ggml_tensor& /*tensor*/) { return GGML_STATUS_SUCCESS; }

        virtual void memset_tensor(      ggml_tensor & tensor, uint8_t value, std::size_t offset, std::size_t size) = 0;
        virtual void set_tensor   (      ggml_tensor & tensor, const void * data, std::size_t offset, std::size_t size) = 0;
        virtual void get_tensor   (const ggml_tensor & tensor,       void * data, std::size_t offset, std::size_t size) = 0;

        virtual bool cpy_tensor   (const ggml_tensor & /*src*/, ggml_tensor & /*dst*/) { return false; }

        virtual void clear        (uint8_t value) = 0;
        virtual void reset        () {}
    };

    class GGML_API buffer_type { // ggml_backend_buffer_type_t
    public:
        virtual ~buffer_type();

        virtual const std::string& get_name() = 0;
        virtual buffer* alloc_buffer(std::size_t size) = 0;
        virtual std::size_t get_alignment() { return TENSOR_ALIGNMENT; }
        virtual std::size_t get_max_size() { return SIZE_MAX; }
        virtual std::size_t get_alloc_size(const ggml_tensor& tensor) { return ggml_nbytes(&tensor); }
        virtual bool is_host() { return false; }
        // for pointer from memory pointer:
        virtual buffer* register_buffer(void * /*ptr*/, std::size_t /*size*/, std::size_t /*max_tensor_size*/) { return nullptr; }
    };

    // TODO: manage event
    class GGML_API event {
    public:
        virtual ~event();
    };

    // TODO: manage graph
    //class graph_plan {
    //public:
    //    virtual ~graph_plan();
    //};

    class device;

    class GGML_API backend { // ggml_backend_t
        backend() = delete;
    public:
        backend(device& dev);
        virtual ~backend();

        virtual const std::string& get_name() = 0;
        virtual const ggml_guid* get_guid() = 0;

        // need => device::caps_async() {return true;}
        virtual void   set_tensor_async(      ggml_tensor & tensor, const void * data, size_t offset, size_t size) { ggml_backend_tensor_set(&tensor, data, offset, size); }
        virtual void   get_tensor_async(const ggml_tensor & tensor,       void * data, size_t offset, size_t size) { ggml_backend_tensor_get(&tensor, data, offset, size); }
        virtual bool   cpy_tensor_async(ggml_backend_t /*backend_src*/,/* ggml_backend_t backend_dst==this,*/ const ggml_tensor & /*src*/, ggml_tensor & /*dst*/) { return false; }
        virtual void   synchronize() {}

        // TODO: manage graph
        //virtual graph_plan&        graph_plan_create(const ggml_cgraph & cgraph);
        //virtual void               graph_plan_free(graph_plan& plan);
        //virtual void               graph_plan_update(graph_plan& plan, const ggml_cgraph & cgraph);
        //virtual enum ggml_status   graph_plan_compute(graph_plan& plan);

        virtual enum ggml_status graph_compute(ggml_cgraph & cgraph) = 0;

        // need => device::caps_events() { return true; }
        virtual void event_record (event & /*event*/) { GGML_ASSERT(false); }
        virtual void event_wait   (event & /*event*/) { GGML_ASSERT(false); }

        // the extra functions:
        virtual void set_n_threads(int /*n_threads*/) { }

    protected:
        device& m_device;
    };

    class GGML_API device { // ggml_backend_dev_t
    protected:
        friend ggml_backend_buffer_type_t* backend_dev_get_extra_bufts(ggml_backend_dev_t device);
        std::vector<buffer_type*> m_extra_buffers_type;
        std::vector<ggml_backend_buffer_type_t> m_ggml_extra_buffers_type;

    public:
        virtual ~device();

        virtual const std::string&         get_name() = 0;
        virtual const std::string&         get_description() = 0;
        virtual void                       get_memory(std::size_t & free, std::size_t & total) = 0;
        virtual enum ggml_backend_dev_type get_type() = 0;
        virtual backend&                   init_backend(const std::string& params) = 0;
        virtual buffer_type&               get_buffer_type() = 0;
        virtual buffer_type*               get_host_buffer_type() { return nullptr; }
        virtual buffer_type*               get_from_host_ptr_buffer_type() { return nullptr; }

        virtual bool                       supports_op(const ggml_tensor & op) = 0;
        virtual bool                       supports_buft(ggml_backend_buffer_type_t buft) = 0;
        virtual bool                       offload_op(const ggml_tensor & /*op*/) { return false; }

        // event => caps_events() { return true; }
        virtual event*    event_new() { return nullptr; }
        virtual void      event_synchronize(event& /*event*/) { GGML_ASSERT(false); }

        //void get_props(struct ggml_backend_dev_props * props);  ggml_backend_dev_caps
        virtual bool caps_async()                { return false; }
        virtual bool caps_host_buffer()          { return get_host_buffer_type() != nullptr; }
        virtual bool caps_buffer_from_host_ptr() { return get_from_host_ptr_buffer_type() != nullptr; }
        virtual bool caps_events()               { return false; }

    protected:
        // have to be call by the device at init.
        void register_extra_buffer_type(buffer_type* buft);

    };

    class GGML_API reg { // ggml_backend_reg_t
    public:
        virtual ~reg();

        virtual const std::string& get_name() = 0;
        virtual std::size_t        get_device_count() = 0;
        virtual device&            get_device(std::size_t index) = 0;
    };

    GGML_API ggml_backend_buffer_t      c_wrapper(ggml_backend_buffer_type_t buft, buffer* ctx, std::size_t size);
    GGML_API ggml_backend_buffer_type_t c_wrapper(ggml_backend_dev_t device, buffer_type* ctx);
    GGML_API ggml_backend_t             c_wrapper(ggml_backend_dev_t device, backend* ctx);
    GGML_API ggml_backend_dev_t         c_wrapper(ggml_backend_reg_t reg, device* ctx);
    GGML_API ggml_backend_reg_t         c_wrapper(reg* ctx);

    // helper for simple cpu buffer type:
    GGML_API buffer_type* new_cpu_buffer_type(
        const std::string& name,
        bool from_ptr=false,
        std::size_t alignment = TENSOR_ALIGNMENT
    );

}
