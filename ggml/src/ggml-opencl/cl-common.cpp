#include "cl-common.h"
#include "ops.h"

std::string read_file(const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs) {
        return "";
    }
    std::string text;
    ifs.seekg(0, std::ios::end);
    text.resize(ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    ifs.read(&text[0], text.size());
    return text;
}

// fatal=false returns NULL on compile failure instead of aborting; used for
// optional FA variants that may exhaust the Adreno compiler at large DK.
// when the compiler returns CL_OUT_OF_HOST_MEMORY/CL_OUT_OF_RESOURCES (seen with DK>=256/512)
// for FA programs, do clFinish the queue to free up resources, then rebuild (up to 3x)
// if retry_queue is provided
cl_program build_program_from_source_ex(cl_context ctx, cl_device_id dev, const char* program_buffer, const std::string &compile_opts, bool fatal, const char *tag, cl_command_queue retry_queue) {
    if (tag) { GGML_LOG_INFO("ggml_opencl: compiling %s\n", tag); }
    cl_program p;
    char *program_log;
    size_t program_size;
    size_t log_size;
    int err;

    program_size = strlen(program_buffer);

    const int max_attempts = retry_queue ? 3 : 1;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        p = clCreateProgramWithSource(ctx, 1, (const char**)&program_buffer, &program_size, &err);
        if(err < 0) {
            GGML_LOG_ERROR("OpenCL error creating program");
            if (fatal) exit(1);
            return NULL;
        }

        err = clBuildProgram(p, 0, NULL, compile_opts.c_str(), NULL, NULL);
        if (err == CL_SUCCESS) {
            return p;
        }

        const bool transient = (err == CL_OUT_OF_HOST_MEMORY || err == CL_OUT_OF_RESOURCES);
        if (retry_queue && transient && attempt + 1 < max_attempts) {
            clReleaseProgram(p);
            GGML_LOG_WARN("ggml_opencl: transient compile failure (err=%d)%s%s — clFinish + retry (%d/%d)\n",
                err, tag ? " building " : "", tag ? tag : "", attempt + 2, max_attempts);
            clFinish(retry_queue);  // drain in-flight ops holding driver host-heap
            continue;
        }

        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char*) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        GGML_LOG_ERROR("ggml_opencl: kernel compile error (err=%d)%s%s:\n\n%s\n", err, tag ? " building " : "", tag ? tag : "", program_log);
        free(program_log);
        clReleaseProgram(p);
        if (fatal) {
            exit(1);
        }
        return nullptr;
    }
    return NULL;
}

cl_program build_program_from_source(ggml_backend_opencl_context * backend_ctx, const char* program_buffer, const std::string &compile_opts) {
    cl_context   ctx = backend_ctx->context;
    cl_device_id dev = backend_ctx->device;

    // Try the on-disk binary cache first. Falls through silently on miss or
    // any failure; never blocks the build path. Disabled cache => nullptr.
    cl_program p_cached = cl_program_cache_try_load(
        backend_ctx->program_cache, ctx, dev, program_buffer, compile_opts);
    if (p_cached != nullptr) {
        return p_cached;
    }

    cl_program p = build_program_from_source_ex(ctx, dev, program_buffer, compile_opts, /*fatal=*/true);

    // Best-effort save of the freshly-built binary (no-op if cache disabled).
    if (p != nullptr) {
        cl_program_cache_try_save(backend_ctx->program_cache, p, dev, program_buffer, compile_opts);
    }
    return p;
}

cl_program build_program_from_binary(cl_context ctx, cl_device_id dev, const char* program_buffer, const std::string &compile_opts, size_t bin_size) {
    cl_program p;
    char *program_log;
    size_t log_size;
    int err;

    p = clCreateProgramWithBinary(ctx, 1, &dev, &bin_size, (const unsigned char**)&program_buffer, NULL, &err);
    if(err < 0) {
        GGML_LOG_ERROR("OpenCL error creating program from binary");
        exit(1);
    }

    err = clBuildProgram(p, 0, NULL, compile_opts.c_str(), NULL, NULL);
    if(err < 0) {
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char*) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        GGML_LOG_ERROR("ggml_opencl: kernel compile error:\n\n%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return p;
}

bool use_adreno_bin_kernels(ggml_backend_opencl_context * backend_ctx) {
#ifndef GGML_OPENCL_USE_ADRENO_BIN_KERNELS
    GGML_UNUSED(backend_ctx);
    return false;
#else
    if (backend_ctx->gpu_family != GPU_FAMILY::ADRENO) {
        return false;
    }
    return backend_ctx->adreno_use_bin_kernels;
#endif // GGML_OPENCL_USE_ADRENO_BIN_KERNELS
}

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
static void transpose_2d(
    ggml_backend_opencl_context * backend_ctx,
    cl_kernel kernel,
    cl_mem src, cl_mem dst, size_t size,
    cl_int stride, cl_int rows,
    bool blocking = true,
    bool auto_local = false // let driver pick local size for non-uniform workgroups
) {
    static ggml_cl_buffer buf;

    cl_event evt;
    cl_int err;

    buf.allocate(backend_ctx->context, size);

    cl_mem trans;
    cl_buffer_region region;

    region.origin = 0;
    region.size = size;
    CL_CHECK((trans = clCreateSubBuffer(
        buf.buffer, CL_MEM_READ_WRITE,
        CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &src));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &trans));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), &stride));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), &rows));

    size_t local_size[3] = {64, 1, 1};
    size_t global_size[3] = {(size_t)stride, (size_t)rows, 1};;
    CL_CHECK(clEnqueueNDRangeKernel(backend_ctx->queue, kernel, 3, NULL,
        global_size, auto_local ? NULL : local_size, 0, NULL, NULL));

    if (blocking) {
        CL_CHECK(clEnqueueCopyBuffer(backend_ctx->queue, trans, dst, 0, 0, size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clReleaseEvent(evt));
    } else {
        CL_CHECK(clEnqueueCopyBuffer(backend_ctx->queue, trans, dst, 0, 0, size, 0, NULL, NULL));
    }

    CL_CHECK(clReleaseMemObject(trans));
}

void transpose_2d_as_8b(
    ggml_backend_opencl_context * backend_ctx,
    cl_mem src, cl_mem dst, size_t size,
    cl_int stride, cl_int rows,
    bool blocking,
    bool auto_local
) {
    transpose_2d(backend_ctx, backend_ctx->mul_mat.kernel_transpose_8_buf,
        src, dst, size, stride, rows, blocking, auto_local);
}

void transpose_2d_as_16b(
    ggml_backend_opencl_context * backend_ctx,
    cl_mem src, cl_mem dst, size_t size,
    cl_int stride, cl_int rows,
    bool blocking
) {
    transpose_2d(backend_ctx, backend_ctx->mul_mat.kernel_transpose_16_buf,
        src, dst, size, stride, rows, blocking);
}

void transpose_2d_as_32b(
    ggml_backend_opencl_context * backend_ctx,
    cl_mem src, cl_mem dst, size_t size,
    cl_int stride, cl_int rows,
    bool blocking
) {
    transpose_2d(backend_ctx, backend_ctx->mul_mat.kernel_transpose_32_buf,
        src, dst, size, stride, rows, blocking);
}
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

std::vector<ggml_backend_device> g_ggml_backend_opencl_devices;
std::vector<std::unique_ptr<ggml_backend_opencl_device_context>> g_ggml_backend_opencl_dev_ctxs;

void sync_with_other_backends(ggml_backend_opencl_context * backend_ctx) {
    if (g_ggml_backend_opencl_devices.size() < 2) {
        return; // No other devices to synchronize with.
    }

    std::vector<cl_event> events;
    events.reserve(g_ggml_backend_opencl_devices.size());

    for (ggml_backend_device & backend_dev : g_ggml_backend_opencl_devices) {
        ggml_backend_opencl_device_context * dev_ctx = (ggml_backend_opencl_device_context *) backend_dev.context;
        auto * other_backend_ctx = dev_ctx->backend_ctx;

        if (backend_ctx != other_backend_ctx) {
            cl_event ev;
            CL_CHECK(clEnqueueMarkerWithWaitList(other_backend_ctx->queue, 0, nullptr, &ev));
            CL_CHECK(clFlush(other_backend_ctx->queue));
            events.push_back(ev);
        }
    }

    CL_CHECK(clEnqueueBarrierWithWaitList(backend_ctx->queue, events.size(), events.data(), nullptr));
    for (auto ev : events) {
        CL_CHECK(clReleaseEvent(ev));
    }
}

void sync_with_other_backends(ggml_backend_t backend) {
    auto * backend_ctx = static_cast<ggml_backend_opencl_context *>(backend->context);
    sync_with_other_backends(backend_ctx);
}

// look up or create a pooled image1d_buffer over a KV-cache view.
cl_mem ggml_cl_img_pool_get_or_create(
    ggml_backend_opencl_context * backend_ctx,
    std::map<ggml_backend_opencl_context::ImagePoolKey,
             ggml_backend_opencl_context::ImagePoolEntry> & pool,
    cl_mem data_device,
    cl_ulong offset0,
    size_t required_bytes,
    cl_channel_type channel_data_type
) {
    ggml_backend_opencl_context::ImagePoolKey key{(uintptr_t)data_device, (uint64_t)offset0};
    auto it = pool.find(key);
    if (it != pool.end()
        && it->second.k_bytes >= required_bytes
        && it->second.channel_data_type == channel_data_type
        && it->second.image != nullptr) {
        return it->second.image;
    }

    // need to create or recreate and release any stale entry first.
    if (it != pool.end()) {
        if (it->second.image)      { CL_CHECK(clReleaseMemObject(it->second.image)); }
        if (it->second.sub_buffer) {CL_CHECK(clReleaseMemObject(it->second.sub_buffer)); }
        pool.erase(it);
    }

    cl_int status = CL_SUCCESS;
    cl_buffer_region region = {};
    region.origin = (size_t)offset0;
    region.size   = required_bytes;
    cl_mem sub = clCreateSubBuffer(data_device, 0,
                                   CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
    if (status != CL_SUCCESS) {
        return nullptr;
    }

    const size_t pixel_size = (channel_data_type == CL_HALF_FLOAT) ? 8 : 16;
    cl_image_format fmt = {CL_RGBA, channel_data_type};
    cl_image_desc   desc = {};
    desc.image_type   = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    desc.image_width  = required_bytes / pixel_size;
    desc.buffer       = sub;
    cl_mem img = clCreateImage(backend_ctx->context, CL_MEM_READ_ONLY,
                               &fmt, &desc, NULL, &status);
    if (status != CL_SUCCESS) {
        CL_CHECK(clReleaseMemObject(sub));
        return nullptr;
    }

    ggml_backend_opencl_context::ImagePoolEntry entry;
    entry.sub_buffer = sub;
    entry.image      = img;
    entry.k_bytes    = required_bytes;
    entry.channel_data_type = channel_data_type;
    pool[key] = entry;
    return img;
}

// True if two tensors share a device buffer with overlapping byte ranges. The pool
// allocator may place a fused op's output over a sequentially-dead input (safe for the
// original separate kernels, but a read/write race inside one fused kernel).
bool use_adreno_kernels(const ggml_backend_opencl_context *backend_ctx, const ggml_tensor *tensor) {
    int64_t threshold_ne0 = 512;
    int64_t threshold_ne1 = 512;
    if (!backend_ctx->adreno_cl_compiler_version.newer_than_or_same(E031, 38, 11, 0) &&
         backend_ctx->adreno_cl_compiler_version.type != DX) {
        threshold_ne0 = 128;
        threshold_ne1 = 128;
    }
    bool threashold_ok = tensor->ne[0] >= threshold_ne0 && tensor->ne[1] >= threshold_ne1 &&
            tensor->ne[2] == 1 && tensor->ne[3] == 1;

    // The noshuffle layout packs 2 rows per 32-bit texel and the GEMV reads it at an
    // ne1/2 texel stride with an exact-cover dispatch, so it is only addressable when
    // ne1 is a multiple of 64; an unaligned ne1 truncates the stride and the weight is
    // read misaligned. That is a property of the layout, not of one quant -- q4_K, q5_K
    // and q8_0 read the same packing as q6_K. The bound is 64, not 128: a q8_0 attention
    // weight of ne1 = 2880 is a multiple of 64 but not 128 and is correct.
    switch (tensor->type) {
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
            return threashold_ok && tensor->ne[1] % 64 == 0;
        default:
            break;
    }
    return threashold_ok;
}

bool adreno_e17_compiler_quirks(const ggml_backend_opencl_context *backend_ctx) {
    if (!backend_ctx || backend_ctx->gpu_family != GPU_FAMILY::ADRENO ||
        backend_ctx->adreno_cl_compiler_version.type != ADRENO_CL_COMPILER_TYPE::E17) {
        return false;
    }
    const char * env = getenv("GGML_OPENCL_ADRENO_E17_QUIRKS");
    return !(env && env[0] == '0');
}

bool use_adreno_moe_kernels(const ggml_backend_opencl_context *backend_ctx, const ggml_tensor *tensor) {
    // The moe weight repack kernels *_trans4_ns alias a private ushort8 through a uchar*.
    // Certain compilers (found with some A7x and A6x) miscompiles this, corrupting the weights.
    // So, exclude A6x and A7x from using Adreno MoE kernels for now.
    // The quants that have a general mul_mat_id kernel fallback to the general version; the
    // rest fallback to CPU.
    if (backend_ctx && (backend_ctx->adreno_gen == ADRENO_GPU_GEN::A6X ||
                        backend_ctx->adreno_gen == ADRENO_GPU_GEN::A7X ||
                        backend_ctx->adreno_gen == ADRENO_GPU_GEN::ADRENO_UNKNOWN)) {
        return false;
    }

    if (adreno_e17_compiler_quirks(backend_ctx)) {
        return false;
    }

    int ne01 = tensor->ne[1];
    return (((strstr(tensor->name, "ffn") != NULL) && (strstr(tensor->name, "exps") != NULL)) || (strstr(tensor->name, "as") != NULL)) && (ne01 % 32 == 0);
}

bool enable_adreno_trans_weight(const ggml_backend_opencl_context *backend_ctx, const ggml_tensor *tensor) {

    bool adreno_kernel = use_adreno_kernels(backend_ctx, tensor);

    size_t elem_num = tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];

    // The 2D weight transpose (transpose_2d_as_*) tiles rows by 4 over a 2D matrix,
    // so it requires K(ne0)%32==0, M(ne1)%4==0 and ne2==ne3==1.
    const bool shape_ok = (tensor->ne[0] % 32 == 0) && (tensor->ne[1] % 4 == 0) &&
                          (tensor->ne[2] == 1) && (tensor->ne[3] == 1);

    return ((elem_num < 128 * 1024 * 1024) && adreno_kernel && shape_ok);  // max element num: 2**27
}

inline bool tiled_gemv_default_on(const ggml_backend_opencl_context *backend_ctx) {
    return backend_ctx && (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E ||
                           backend_ctx->adreno_gen == ADRENO_GPU_GEN::A8X);
}
inline bool q6k_gemv_tiled_enabled(const ggml_backend_opencl_context *backend_ctx) {
    static const char * e = std::getenv("GGML_OPENCL_Q6K_GEMV_TILED");
    if (e && e[0] != '\0') {
        return e[0] != '0';
    }
    return tiled_gemv_default_on(backend_ctx);
}
bool use_q6k_tiled(const ggml_backend_opencl_context *backend_ctx, const ggml_tensor *tensor) {
    return q6k_gemv_tiled_enabled(backend_ctx) && tensor->type == GGML_TYPE_Q6_K &&
           tensor->ne[1] >= 32768 && tensor->ne[1] % 64 == 0 &&
           use_adreno_kernels(backend_ctx, tensor);
}
inline bool q4k_gemv_tiled_enabled(const ggml_backend_opencl_context *backend_ctx) {
    static const char * e = std::getenv("GGML_OPENCL_Q4K_GEMV_TILED");
    if (e && e[0] != '\0') {
        return e[0] != '0';
    }
    return tiled_gemv_default_on(backend_ctx);
}
bool use_q4k_tiled(const ggml_backend_opencl_context *backend_ctx, const ggml_tensor *tensor) {
    return q4k_gemv_tiled_enabled(backend_ctx) && tensor->type == GGML_TYPE_Q4_K &&
           tensor->ne[1] >= 32768 && tensor->ne[1] % 64 == 0 &&
           use_adreno_kernels(backend_ctx, tensor);
}

static inline bool flat_large_m_enabled() {
    static const char * e = getenv("GGML_OPENCL_FLAT_LARGE_M");
    static const bool en = e != nullptr && atoi(e) != 0;
    return en;
}
bool enable_adreno_trans_weight_q5_K(const ggml_backend_opencl_context *backend_ctx, const ggml_tensor *tensor) {
    if (!use_adreno_kernels(backend_ctx, tensor)) {
        return false;
    }

    const size_t elem_num = ggml_nelements(tensor);
    const size_t q_img_width = elem_num / 8;
    const size_t qh_img_width = elem_num / 16;
    const bool shape_ok = tensor->ne[0] % 32 == 0 && tensor->ne[1] % 4 == 0 &&
                          tensor->ne[2] == 1 && tensor->ne[3] == 1;

    return shape_ok && q_img_width <= backend_ctx->image_max_buffer_size &&
           qh_img_width <= backend_ctx->image_max_buffer_size;
}

bool use_flat_gemv_for_large_m_q4_K(const ggml_backend_opencl_context *backend_ctx, const ggml_tensor *tensor) {
    if (tensor->ne[1] % 4 != 0 && tensor->ne[2] == 1 && tensor->ne[3] == 1) {
        return true;
    }

    if (!flat_large_m_enabled()) {
        return false;
    }
    // gemv_noshuffle variant perf drops for large M, use flat variant for large M.
    // threshold is well above typical hidden/FFN dims, but below typical vocab sizes.
    // note that this forces large M weights to use LM GEMM.
    // EXCEPT when this branch's tiled-canonical lm_head/embed layout is active: the
    // weight is converted to the 64-row tiled layout, which the flat gemv would
    // misread as garbage. use_q4k_tiled owns these large-M weights, so defer to it.
    return tensor->ne[1] >= 32768 && tensor->ne[2] == 1 && tensor->ne[3] == 1
           && !use_q4k_tiled(backend_ctx, tensor);
}

bool use_flat_gemv_for_large_m_q6_K(const ggml_backend_opencl_context *backend_ctx, const ggml_tensor *tensor) {
    // NOTE on ordering: the ne01 % 128 escape below is a CORRECTNESS guard, not a
    // performance one, so it must be reachable regardless of flat_large_m_enabled().
    // The opt-in gate therefore sits after it, and after the tiled deferral.
    // gemv_noshuffle variant perf drops for large M, use flat variant for large M.
    // threshold is well above typical hidden/FFN dims, but below typical vocab sizes.
    // q6_K flat gemv is worse for smaller K; 2048 seems to be a reasonable threshold.
    // note that this forces large M weights to use LM GEMM.
    // When this branch's tiled-canonical lm_head/embed layout is active, the weight is
    // converted to the 64-row tiled layout, which the flat gemv would misread as
    // garbage. use_q6k_tiled owns these large-M weights (it requires ne01 % 64 == 0,
    // so it never claims an odd-vocab weight), so defer to it first.
    if (use_q6k_tiled(backend_ctx, tensor)) {
        return false;
    }
    // The noshuffle (transposed-weight) layout packs 2 rows per 32-bit texel and the
    // gemv reads it with a ne01/2 texel stride and an exact-cover dispatch of
    // ceil(ne01/2 / 64)*64 work-items with no store guard; the gemm uses 4-row tiles.
    // It is therefore only correct for ne01 % 128 == 0: an odd ne01 (e.g. granitemoe
    // lm_head [1536, 49155] -- odd vocab) truncates the texel stride, misaligning every
    // odd column of the transposed layout (gross garbage) and dropping the last row;
    // other non-multiples over-dispatch and write past the end of dst. Route such
    // tensors to the flat GEMV + regular convert; the matching GEMM (ne1>1) falls back
    // to CPU (see supports_op). All standard even-vocab/hidden dims are multiples of
    // 128 and keep the noshuffle path.
    if ((tensor->ne[1] % 128 != 0) && tensor->ne[2] == 1 && tensor->ne[3] == 1) {
        return true;
    }

    if (!flat_large_m_enabled()) {
        return false;
    }

    // The gemv_noshuffle slowdown tracks TOTAL weight size, not ne0 alone; ne0 >= 2048 is a
    // proxy for "large weight" that misses a narrow-hidden vocab-scale lm_head.
    // Add a direct size escape so such weights also take the flat path, without changing
    // which weights ne0 >= 2048 already routes there.
    // The size escape is not taken on the A7X since its compiler miscompiles the flat K-quant GEMV
    return tensor->ne[1] >= 32768
        && (tensor->ne[0] >= 2048 || (backend_ctx->adreno_gen != ADRENO_GPU_GEN::A7X && ggml_nbytes(tensor) >= (256ull << 20)))
        && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

// check if a Q8_0 tensor has been SOA'ed in set_tensor
// we store SOA'ed tensors in a map in set_tensor, check against that map
bool use_q4_0_bin_kernels(const ggml_backend_opencl_context *backend_ctx, const ggml_tensor *tensor) {
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    if (!backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_0_f32_32b_trans ||
        !backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_0_f32_32b_trans_ila_a8_bin) {
        return false;
    }
    return (tensor->ne[0] % 32 == 0) && (tensor->ne[1] % 64 == 0);
#else
    GGML_UNUSED(backend_ctx);
    GGML_UNUSED(tensor);
    return false;
#endif
}

bool use_q4_k_bin_kernels(const ggml_backend_opencl_context *backend_ctx, const ggml_tensor *tensor) {
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    if (!backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_k_f32_32b_trans ||
        !backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_k_f32_32b_trans_ila_a8_bin) {
        return false;
    }
    return (tensor->ne[0] % 256 == 0) && (tensor->ne[1] % 64 == 0) &&
           !use_q4k_tiled(backend_ctx, tensor) && !use_flat_gemv_for_large_m_q4_K(backend_ctx, tensor);
#else
    GGML_UNUSED(backend_ctx);
    GGML_UNUSED(tensor);
    return false;
#endif
}

bool use_q6_k_bin_kernels(const ggml_backend_opencl_context * backend_ctx, const ggml_tensor * tensor) {
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    if (!backend_ctx->mul_mat.kernel_gemv_noshuffle_q6_k_f32_32b_trans ||
        !backend_ctx->mul_mat.kernel_gemm_noshuffle_q6_k_f32_32b_trans_ila_a8_bin) {
        return false;
    }
    return (tensor->ne[0] % 256 == 0) && (tensor->ne[1] % 64 == 0) &&
           !use_q6k_tiled(backend_ctx, tensor) && !use_flat_gemv_for_large_m_q6_K(backend_ctx, tensor);
#else
    GGML_UNUSED(backend_ctx);
    GGML_UNUSED(tensor);
    return false;
#endif
}

bool ggml_cl_is_q8_0_soa(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_Q8_0 || tensor->buffer == nullptr) {
        return false;
    }
    auto * ctx = (ggml_backend_opencl_buffer_context *) tensor->buffer->context;
    if (ctx == nullptr) {
        return false;
    }
    const ggml_tensor * key = tensor->view_src != nullptr ? tensor->view_src : tensor;
    return ctx->q8_0_soa_tensors.count(key) > 0;
}

// check if a Q4_0 tensor has been SOA'ed in set_tensor
// we store SOA'ed tensors in a map in set_tensor, check against that map
bool ggml_cl_is_q4_0_soa(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_Q4_0 || tensor->buffer == nullptr) {
        return false;
    }
    auto * ctx = (ggml_backend_opencl_buffer_context *) tensor->buffer->context;
    if (ctx == nullptr) {
        return false;
    }
    const ggml_tensor * key = tensor->view_src != nullptr ? tensor->view_src : tensor;
    return ctx->q4_0_soa_tensors.count(key) > 0;
}


void load_cl_kernels(ggml_backend_opencl_context * backend_ctx) {
    if (backend_ctx->kernels_loaded) {
        return;
    }

    const auto opencl_c_std =
        std::string("CL") + std::to_string(backend_ctx->opencl_c_version.major) + "." +
        std::to_string(backend_ctx->opencl_c_version.minor);
    backend_ctx->kernel_compile_opts =
        std::string("-cl-std=") + opencl_c_std +
        " -cl-mad-enable -cl-unsafe-math-optimizations"
        " -cl-finite-math-only -cl-fast-relaxed-math";
    if (backend_ctx->adreno_use_large_buffer) {
        backend_ctx->kernel_compile_opts += " -qcom-enable-large-buffer ";
    }

    GGML_LOG_INFO("ggml_opencl: loading OpenCL kernels");
    ggml_cl_load_kernels_add(backend_ctx);
    ggml_cl_load_kernels_add_id(backend_ctx);
    ggml_cl_load_kernels_tri(backend_ctx);
    ggml_cl_load_kernels_fill(backend_ctx);
    ggml_cl_load_kernels_clamp(backend_ctx);
    ggml_cl_load_kernels_cpy(backend_ctx);
    ggml_cl_load_kernels_repack(backend_ctx);
    ggml_cl_load_kernels_diag_mask_inf(backend_ctx);
    ggml_cl_load_kernels_diag(backend_ctx);
    ggml_cl_load_kernels_gelu(backend_ctx);
    ggml_cl_load_kernels_glu(backend_ctx);
    ggml_cl_load_kernels_get_rows(backend_ctx);
    ggml_cl_load_kernels_solve_tri(backend_ctx);
    ggml_cl_load_kernels_im2col(backend_ctx);
    ggml_cl_load_kernels_mul_mat(backend_ctx);
    ggml_cl_load_kernels_mul(backend_ctx);
    ggml_cl_load_kernels_norm(backend_ctx);
    ggml_cl_load_kernels_relu(backend_ctx);
    ggml_cl_load_kernels_rms_norm(backend_ctx);
    ggml_cl_load_kernels_l2_norm(backend_ctx);
    ggml_cl_load_kernels_rope(backend_ctx);
    ggml_cl_load_kernels_scale(backend_ctx);
    ggml_cl_load_kernels_silu(backend_ctx);
    ggml_cl_load_kernels_soft_max(backend_ctx);
    ggml_cl_load_kernels_div(backend_ctx);
    ggml_cl_load_kernels_sqr(backend_ctx);
    ggml_cl_load_kernels_sqrt(backend_ctx);
    ggml_cl_load_kernels_mean(backend_ctx);
    ggml_cl_load_kernels_sub(backend_ctx);
    ggml_cl_load_kernels_sum_rows(backend_ctx);
    ggml_cl_load_kernels_cumsum(backend_ctx);
    ggml_cl_load_kernels_sigmoid(backend_ctx);
    ggml_cl_load_kernels_group_norm(backend_ctx);
    ggml_cl_load_kernels_repeat(backend_ctx);
    ggml_cl_load_kernels_pad(backend_ctx);
    ggml_cl_load_kernels_tanh(backend_ctx);
    ggml_cl_load_kernels_neg(backend_ctx);
    ggml_cl_load_kernels_exp(backend_ctx);
    ggml_cl_load_kernels_expm1(backend_ctx);
    ggml_cl_load_kernels_abs(backend_ctx);
    ggml_cl_load_kernels_softplus(backend_ctx);
    ggml_cl_load_kernels_upscale(backend_ctx);
    ggml_cl_load_kernels_concat(backend_ctx);
    ggml_cl_load_kernels_timestep_embedding(backend_ctx);
    ggml_cl_load_kernels_set_rows(backend_ctx);
    ggml_cl_load_kernels_conv_2d(backend_ctx);
    ggml_cl_load_kernels_ssm_conv(backend_ctx);
    ggml_cl_load_kernels_ssm_scan(backend_ctx);
    ggml_cl_load_kernels_gated_delta_net(backend_ctx);
    ggml_cl_load_kernels_mul_mat_id(backend_ctx);
    ggml_cl_load_kernels_mul_mat_adreno(backend_ctx);
    ggml_cl_load_kernels_unary_ext(backend_ctx);
    ggml_cl_load_kernels_nop(backend_ctx);
    ggml_cl_load_kernels_dup(backend_ctx);
    ggml_cl_load_kernels_cont(backend_ctx);
    ggml_cl_load_kernels_reshape(backend_ctx);
    ggml_cl_load_kernels_view(backend_ctx);
    ggml_cl_load_kernels_permute(backend_ctx);
    ggml_cl_load_kernels_transpose(backend_ctx);
    ggml_cl_load_kernels_set(backend_ctx);
    ggml_cl_load_kernels_flash_attn(backend_ctx);
    ggml_cl_load_kernels_unpack(backend_ctx);
    GGML_LOG_CONT("\n");
    backend_ctx->kernels_loaded = true;
}
