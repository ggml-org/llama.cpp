#pragma once

#include "ggml.h"

#if defined(GGML_OPENCL_USE_ADRENO_KERNELS) && defined(GGML_OPENCL_USE_ADRENO_BIN_KERNELS) && GGML_OPENCL_TARGET_VERSION >= 300
#include "opencl_profiling_info.h"
#include "cl-recordable-queue.h"

class Conv3D_GMEM {
public:
    static size_t get_hash(cl_device_id device, const ggml_tensor * op);
    static std::string program_name(const ggml_tensor * op);
    static Conv3D_GMEM * create(cl_context context, cl_device_id device, cl_command_queue queue, cl_program program, ProfilingInfo & profiling_info, const ggml_tensor * op);

    ~Conv3D_GMEM();

    void enqueue(cl_mem weight, cl_ulong weight_offset, cl_mem input, cl_ulong input_offset, cl_mem output, cl_ulong output_offset,
        cl_uint od, cl_uint oh, cl_uint ow, cl_uint id, cl_uint ih, cl_uint iw,
        cl_uint stride, cl_uint dilation, cl_uint pad_w, cl_uint pad_h, cl_uint pad_d, const ggml_tensor * tensor);

private:
    Conv3D_GMEM(cl_context context, cl_device_id device, cl_command_queue queue, cl_kernel kernel_conv, cl_kernel kernel_relayout, ProfilingInfo & profiling_info, const ggml_tensor * op);
    static bool has_extension(cl_device_id device, const char * extension);
    static bool multiply(size_t lhs, size_t rhs, size_t * result);
    static cl_uint tile_n_for(cl_uint output_channels);

    cl_context context;
    const cl_uint oc, ic, fd, fh, fw, tile_n, padded_oc;
    const ggml_type weight_type;
    cl_recordable_queue recordable_queue;
#ifdef GGML_OPENCL_PROFILING
    ProfilingInfo & profiling_info;
#endif
    cl_kernel kernel_conv = nullptr;
    cl_kernel kernel_relayout = nullptr;
    cl_mem gmem_weight = nullptr;
    cl_mem gmem_weight_image = nullptr;
};
#endif
