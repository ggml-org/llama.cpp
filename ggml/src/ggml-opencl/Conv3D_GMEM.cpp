#include "Conv3D_GMEM.h"

#include "ggml-impl.h"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if defined(GGML_OPENCL_USE_ADRENO_KERNELS) && defined(GGML_OPENCL_USE_ADRENO_BIN_KERNELS) && GGML_OPENCL_TARGET_VERSION >= 300

bool Conv3D_GMEM::has_extension(cl_device_id device, const char * extension) {
    size_t size = 0;
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &size));
    std::vector<char> extensions(size);
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, extensions.size(), extensions.data(), nullptr));
    const std::string extension_list(extensions.data());
    size_t begin = 0;
    while ((begin = extension_list.find(extension, begin)) != std::string::npos) {
        const size_t end = begin + strlen(extension);
        if ((begin == 0 || extension_list[begin - 1] == ' ') &&
            (end == extension_list.size() || extension_list[end] == ' ')) {
            return true;
        }
        begin = end;
    }
    return false;
}

bool Conv3D_GMEM::multiply(size_t lhs, size_t rhs, size_t * result) {
    if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

cl_uint Conv3D_GMEM::tile_n_for(cl_uint output_channels) {
    return output_channels >= 64 ? 64 : output_channels >= 32 ? 32 : 16;
}

size_t Conv3D_GMEM::get_hash(cl_device_id device, const ggml_tensor * op) {
    static const bool disabled = getenv("GGML_OPENCL_DISABLE_CONV_GMEM") != nullptr;
    if (disabled || !has_extension(device, "cl_qcom_onchip_global_memory") || !cl_recordable_queue::supported(device)) {
        return 0;
    }

    const ggml_tensor * weight = op->src[0];
    const ggml_tensor * input = op->src[1];
    if ((weight->type != GGML_TYPE_F32 && weight->type != GGML_TYPE_F16) || input->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(weight) || !ggml_is_contiguous(input) || !ggml_is_contiguous(op)) {
        return 0;
    }

    const int32_t stride0 = op->op_params[0], stride1 = op->op_params[1], stride2 = op->op_params[2];
    const int32_t dilation0 = op->op_params[6], dilation1 = op->op_params[7], dilation2 = op->op_params[8];
    const int32_t ic = op->op_params[9], batch = op->op_params[10], oc = op->op_params[11];
    if (ic <= 0 || oc <= 0 || batch != 1 || stride0 <= 0 || dilation0 <= 0 || stride0 != stride1 || stride1 != stride2 || dilation0 != dilation1 || dilation1 != dilation2) {
        return 0;
    }
    // Channel-tail address checks reduce register-limited concurrency on Adreno.
    // Use the generic kernel for IC that is not eight-aligned.
    if (ic % 8 != 0) {
        return 0;
    }

    const int64_t dimensions[] = { weight->ne[0], weight->ne[1], weight->ne[2], input->ne[0], input->ne[1], input->ne[2], op->ne[0], op->ne[1], op->ne[2] };
    for (int64_t dimension : dimensions) {
        if (dimension <= 0 || (uint64_t) dimension > std::numeric_limits<cl_uint>::max()) {
            return 0;
        }
    }

    const cl_uint tile_n = tile_n_for((cl_uint) oc);
    const size_t padded_oc = ((size_t) oc + tile_n - 1) / tile_n * tile_n;
    size_t weight_elements = padded_oc;
    for (int64_t dimension : { weight->ne[0], weight->ne[1], weight->ne[2], (int64_t) ic }) {
        if (!multiply(weight_elements, (size_t) dimension, &weight_elements)) {
            return 0;
        }
    }

    size_t input_elements = 1;
    for (int64_t dimension : { input->ne[0], input->ne[1], input->ne[2], (int64_t) ic }) {
        if (!multiply(input_elements, (size_t) dimension, &input_elements)) {
            return 0;
        }
    }
    if (input_elements > std::numeric_limits<cl_uint>::max() / sizeof(float)) {
        return 0;
    }

    size_t output_elements = 1;
    for (int64_t dimension : { op->ne[0], op->ne[1], op->ne[2] }) {
        if (!multiply(output_elements, (size_t) dimension, &output_elements)) {
            return 0;
        }
    }
    if (output_elements > std::numeric_limits<cl_uint>::max() - 63 || !multiply(output_elements, (size_t) oc * sizeof(float), &output_elements)) {
        return 0;
    }

    size_t gmem_size = 0;
    size_t image_max_elements = 0;
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_ONCHIP_GLOBAL_MEM_SIZE_QCOM, sizeof(gmem_size), &gmem_size, nullptr));
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, sizeof(image_max_elements), &image_max_elements, nullptr));
    if (weight_elements > image_max_elements || input_elements > image_max_elements ||
        weight_elements > gmem_size / ggml_type_size(weight->type) ||
        weight_elements > std::numeric_limits<cl_uint>::max() / ggml_type_size(weight->type) ||
        weight_elements > (size_t) std::numeric_limits<int>::max()) {
        return 0;
    }

    size_t hash = 1469598103934665603ULL;
    for (cl_uint field : { (cl_uint) oc, (cl_uint) ic, (cl_uint) weight->ne[2], (cl_uint) weight->ne[1], (cl_uint) weight->ne[0], (cl_uint) weight->type }) {
        hash ^= field;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string Conv3D_GMEM::program_name(const ggml_tensor * op) {
    return std::string("conv3d_gmem_") + (op->src[0]->type == GGML_TYPE_F16 ? "f16_n" : "f32_n") + std::to_string(tile_n_for((cl_uint) op->op_params[11]));
}

Conv3D_GMEM * Conv3D_GMEM::create(cl_context context, cl_device_id device, cl_command_queue queue, cl_program program, ProfilingInfo & profiling_info, const ggml_tensor * op) {
    cl_int status = CL_SUCCESS;
    cl_kernel conv = clCreateKernel(program, "kernel_conv3d_gmem", &status);
    if (status != CL_SUCCESS) {
        return nullptr;
    }
    cl_kernel relayout = clCreateKernel(program, "conv3d_gmem_relayout_weight", &status);
    bool valid = status == CL_SUCCESS;
    if (valid) {
        for (cl_kernel kernel : { conv, relayout }) {
            size_t max_work_group_size = 0;
            status = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, nullptr);
            valid = valid && status == CL_SUCCESS && max_work_group_size >= 64;
        }
    }
    if (!valid) {
        if (relayout) {
            CL_CHECK(clReleaseKernel(relayout));
        }
        CL_CHECK(clReleaseKernel(conv));
        return nullptr;
    }
    return new Conv3D_GMEM(context, device, queue, conv, relayout, profiling_info, op);
}

Conv3D_GMEM::Conv3D_GMEM(cl_context context, cl_device_id device, cl_command_queue queue, cl_kernel kernel_conv, cl_kernel kernel_relayout, ProfilingInfo & profiling_info, const ggml_tensor * op)
    : context(context),
      oc((cl_uint) op->op_params[11]), ic((cl_uint) op->op_params[9]),
      fd((cl_uint) op->src[0]->ne[2]), fh((cl_uint) op->src[0]->ne[1]), fw((cl_uint) op->src[0]->ne[0]),
      tile_n(tile_n_for(oc)), padded_oc((oc + tile_n - 1) / tile_n * tile_n), weight_type(op->src[0]->type),
      recordable_queue(context, device, queue)
#ifdef GGML_OPENCL_PROFILING
      , profiling_info(profiling_info)
#endif
      , kernel_conv(kernel_conv), kernel_relayout(kernel_relayout) {
#ifndef GGML_OPENCL_PROFILING
    GGML_UNUSED(profiling_info);
#endif
    cl_int status = CL_SUCCESS;

    const size_t element_size = ggml_type_size(weight_type);
    const size_t weight_size = (size_t) padded_oc * fd * fh * fw * ic * element_size;
    const cl_mem_properties properties[] = { CL_MEM_ONCHIP_GLOBAL_QCOM, 1, CL_MEM_ONCHIP_GLOBAL_OFFSET_QCOM, 0, 0 };
    CL_CHECK((gmem_weight = clCreateBufferWithProperties(context, properties, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, weight_size, nullptr, &status), status));

    const cl_image_format format = { CL_R, weight_type == GGML_TYPE_F16 ? (cl_channel_type) CL_HALF_FLOAT : (cl_channel_type) CL_FLOAT };
    cl_image_desc description = {};
    description.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    description.image_width = (size_t) padded_oc * fd * fh * fw * ic;
    description.buffer = gmem_weight;
    CL_CHECK((gmem_weight_image = clCreateImage(context, CL_MEM_READ_ONLY, &format, &description, nullptr, &status), status));
}

void Conv3D_GMEM::enqueue(cl_mem weight, cl_ulong weight_offset, cl_mem input, cl_ulong input_offset, cl_mem output, cl_ulong output_offset,
    cl_uint od, cl_uint oh, cl_uint ow, cl_uint id, cl_uint ih, cl_uint iw,
    cl_uint stride, cl_uint dilation, cl_uint pad_w, cl_uint pad_h, cl_uint pad_d, const ggml_tensor * tensor) {
    cl_int status = CL_SUCCESS;
    const size_t weight_element_size = ggml_type_size(weight_type);
    const size_t real_m = (size_t) od * oh * ow;
    const cl_uint padded_m = (cl_uint) ((real_m + 63) / 64 * 64);
    const size_t input_elements = (size_t) id * ih * iw * ic;
    const size_t input_size = input_elements * sizeof(float);
    const size_t weight_size = (size_t) oc * fd * fh * fw * ic * weight_element_size;
    const size_t output_size = (size_t) oc * real_m * sizeof(float);
    const size_t line_stride_a_size = (size_t) sizeof(float) * id * ih * iw;
    const size_t line_stride_b_size = weight_element_size * fd * fh * fw * ic;
    GGML_ASSERT(line_stride_a_size <= std::numeric_limits<cl_uint>::max());
    GGML_ASSERT(line_stride_b_size <= std::numeric_limits<cl_uint>::max());

    cl_mem input_sub = input;
    if (input_offset != 0) {
        const cl_buffer_region region = { (size_t) input_offset, input_size };
        CL_CHECK((input_sub = clCreateSubBuffer(input, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &status), status));
    }
    const cl_image_format input_format = { CL_R, CL_FLOAT };
    cl_image_desc input_description = {};
    input_description.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    input_description.image_width = input_elements;
    input_description.buffer = input_sub;
    cl_mem input_image = clCreateImage(context, CL_MEM_READ_ONLY, &input_format, &input_description, nullptr, &status);
    CL_CHECK(status);

    cl_mem weight_sub = weight;
    if (weight_offset != 0) {
        const cl_buffer_region region = { (size_t) weight_offset, weight_size };
        CL_CHECK((weight_sub = clCreateSubBuffer(weight, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &status), status));
    }
    const int relayout_args[] = { (int) fd, (int) fh, (int) fw, (int) ic, (int) oc, (int) ((size_t) padded_oc * fd * fh * fw * ic) };
    cl_uint arg = 0;
    recordable_queue.set_arg(kernel_relayout, arg++, sizeof(weight_sub), &weight_sub);
    recordable_queue.set_arg(kernel_relayout, arg++, sizeof(gmem_weight), &gmem_weight);
    for (const int & value : relayout_args) {
        recordable_queue.set_arg(kernel_relayout, arg++, sizeof(value), &value);
    }
    const size_t relayout_global = ((size_t) relayout_args[5] + 63) / 64 * 64;
    recordable_queue.record(kernel_relayout, { relayout_global }, { 64 });

    cl_mem output_sub = output;
    if (output_offset != 0) {
        const cl_buffer_region region = { (size_t) output_offset, output_size };
        CL_CHECK((output_sub = clCreateSubBuffer(output, CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &status), status));
    }

    const cl_uint line_stride_a = (cl_uint) line_stride_a_size;
    const cl_uint line_stride_b = (cl_uint) line_stride_b_size;
    const cl_uint args[] = { padded_m, padded_oc, ic, line_stride_a, line_stride_b, fd, fh, fw, id, ih, iw, ic, oc, od, oh, ow, dilation, stride, pad_w, pad_h, pad_d };
    arg = 0;
    recordable_queue.set_arg(kernel_conv, arg++, sizeof(input_image), &input_image);
    recordable_queue.set_arg(kernel_conv, arg++, sizeof(gmem_weight_image), &gmem_weight_image);
    recordable_queue.set_arg(kernel_conv, arg++, sizeof(output_sub), &output_sub);
    for (const cl_uint & value : args) {
        recordable_queue.set_arg(kernel_conv, arg++, sizeof(value), &value);
    }
    const size_t global_size[] = { 64, padded_m / 64, padded_oc / tile_n };
    const size_t local_size[] = { 64, 1, 1 };
    recordable_queue.record(kernel_conv, { global_size[0], global_size[1], global_size[2] }, { local_size[0], local_size[1], local_size[2] });

#ifdef GGML_OPENCL_PROFILING
    cl_event event = nullptr;
    recordable_queue.enqueue(&event);
    if (event) {
        profiling_info.populate(event, kernel_conv, 3, global_size, local_size, tensor);
        CL_CHECK(clReleaseEvent(event));
    }
#else
    recordable_queue.enqueue();
    GGML_UNUSED(tensor);
#endif

    CL_CHECK(clReleaseMemObject(input_image));
    if (input_sub != input) {
        CL_CHECK(clReleaseMemObject(input_sub));
    }
    if (weight_sub != weight) {
        CL_CHECK(clReleaseMemObject(weight_sub));
    }
    if (output_sub != output) {
        CL_CHECK(clReleaseMemObject(output_sub));
    }
}

Conv3D_GMEM::~Conv3D_GMEM() {
    CL_CHECK(clReleaseKernel(kernel_relayout));
    CL_CHECK(clReleaseKernel(kernel_conv));
    CL_CHECK(clReleaseMemObject(gmem_weight_image));
    CL_CHECK(clReleaseMemObject(gmem_weight));
}
#endif
