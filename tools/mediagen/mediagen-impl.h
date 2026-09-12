#pragma once

#include "mediagen.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"

#include <cstdio>
#include <cstdarg>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

//
// logging
//

struct mediagen_logger {
    ggml_log_level verbosity = GGML_LOG_LEVEL_INFO;
    static mediagen_logger & get() { static mediagen_logger l; return l; }
    GGML_ATTRIBUTE_FORMAT(3, 4)
    void log(ggml_log_level level, const char * fmt, ...);
};

#define MG_LOG(lvl, ...) mediagen_logger::get().log(lvl, __VA_ARGS__)
#define MG_INF(...) MG_LOG(GGML_LOG_LEVEL_INFO,  __VA_ARGS__)
#define MG_WRN(...) MG_LOG(GGML_LOG_LEVEL_WARN,  __VA_ARGS__)
#define MG_ERR(...) MG_LOG(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define MG_DBG(...) MG_LOG(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)

GGML_ATTRIBUTE_FORMAT(1, 2)
std::string mg_format(const char * fmt, ...);

//
// weight storage
//

// tensor description read from a model file (safetensors or gguf)
struct mg_tensor_desc {
    std::string          name;
    ggml_type            type;
    std::vector<int64_t> ne;      // ggml order
    size_t               offset;  // byte offset in file
    size_t               nbytes;  // byte size in file
};

// a set of weights living on one backend buffer
struct mg_weights {
    ggml_context_ptr           ctx;
    ggml_backend_buffer_ptr    buf;
    std::map<std::string, ggml_tensor *> tensors;

    ggml_tensor * get(const std::string & name, bool required = true) const;
    bool has(const std::string & name) const { return tensors.count(name) > 0; }
    size_t size() const { return buf ? ggml_backend_buffer_get_size(buf.get()) : 0; }
};

// read the first n values of a 1-D f32 / f16 tensor
bool mg_tensor_to_f32(const ggml_tensor * t, int64_t n, std::vector<float> & out);

bool mg_safetensors_read_header(const std::string & path, std::vector<mg_tensor_desc> & out, std::string * metadata, size_t * data_start);

// own gguf header reader: diffusion GGUFs have tensor names longer than GGML_MAX_NAME
bool mg_gguf_read_header(const std::string & path, std::vector<mg_tensor_desc> & out, std::map<std::string, std::string> * kv_str);

struct mg_load_opts {
    std::function<std::string(const std::string &)> rename; // return "" to drop a tensor
    std::function<bool(const mg_tensor_desc &)>     keep;
    bool   bf16_to_f16 = true;
    bool   f32_from_bf16_small = true; // 1-D bf16 tensors become f32
    bool   split_conv3d = true;        // [KW,KH,KT,IC,OC] -> KT tensors "<name>.t<k>" of [KW,KH,IC,OC]
    std::function<void(const std::string &, float *, int64_t)> transform; // applied to f32 tensors while loading
};
bool mg_weights_load(mg_weights & w, const std::string & path, ggml_backend_buffer_type_t buft, const mg_load_opts & opts);

//
// graph execution
//

struct mg_backend {
    ggml_backend_t                 gpu = nullptr;
    ggml_backend_t                 cpu = nullptr;
    std::vector<ggml_backend_t>    ptrs;
    std::vector<ggml_backend_buffer_type_t> bufts;
    ggml_backend_sched_ptr         sched;
    int                            n_threads = 4;
    size_t                         max_nodes = 0;

    mg_backend() = default;
    mg_backend(const mg_backend &) = delete;
    mg_backend & operator=(const mg_backend &) = delete;
    ~mg_backend() { free(); }

    bool init(bool use_gpu, int n_threads, size_t max_nodes);
    void free();
    void release_compute(); // free the scheduler's compute buffers
    ggml_backend_buffer_type_t weight_buft() const { return ggml_backend_get_default_buffer_type(gpu ? gpu : cpu); }
    ggml_backend_buffer_type_t cpu_buft() const { return ggml_backend_get_default_buffer_type(cpu); }
    bool supports_op(const ggml_tensor * op) const;
};

// build and run one graph, sized to the scheduler of the backend
struct mg_graph {
    ggml_context_ptr ctx;
    ggml_cgraph *    gf = nullptr;
    struct input { ggml_tensor * t; std::vector<uint8_t> data; };
    std::vector<input> inputs;
    std::vector<ggml_tensor *> outputs;

    explicit mg_graph(const mg_backend & be);
    ggml_context * get() { return ctx.get(); }

    ggml_tensor * new_input(ggml_type type, const std::vector<int64_t> & ne, const void * data, const char * name);
    ggml_tensor * new_input_f32(const std::vector<int64_t> & ne, const std::vector<float> & data, const char * name) {
        return new_input(GGML_TYPE_F32, ne, data.data(), name);
    }
    void mark_output(ggml_tensor * t);

    bool compute(mg_backend & be);
    void get_output(ggml_tensor * t, std::vector<float> & out) const;
};

//
// small helpers
//

// x * (1 + scale) + shift
ggml_tensor * mg_modulate(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale, ggml_tensor * shift);
ggml_tensor * mg_linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b);

// debug: with MEDIAGEN_DUMP=<prefix> set, write <prefix>.<name>.bin
void mg_dump(const char * name, const std::vector<float> & data);
