#pragma once

#include <CL/cl.h>
#include <CL/cl_ext_qcom.h>

#include <string>
#include <vector>

class cl_recordable_queue {
public:
    cl_recordable_queue(const cl_recordable_queue &) = delete;
    cl_recordable_queue & operator=(const cl_recordable_queue &) = delete;

    static bool supported(cl_device_id device) {
        size_t size = 0;
        CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &size));
        std::vector<char> extensions(size);
        CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, extensions.size(), extensions.data(), nullptr));
        const std::string extension_list(extensions.data());
        const std::string extension("cl_qcom_recordable_queues");
        size_t begin = 0;
        while ((begin = extension_list.find(extension, begin)) != std::string::npos) {
            const size_t end = begin + extension.size();
            if ((begin == 0 || extension_list[begin - 1] == ' ') &&
                (end == extension_list.size() || extension_list[end] == ' ')) {
                return true;
            }
            begin = end;
        }
        return false;
    }

    cl_recordable_queue(cl_context context, cl_device_id device, cl_command_queue queue)
        : queue(queue) {
        cl_platform_id platform = nullptr;
        CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr));

        new_recording = (clNewRecordingQCOM_fn) clGetExtensionFunctionAddressForPlatform(platform, "clNewRecordingQCOM");
        end_recording = (clEndRecordingQCOM_fn) clGetExtensionFunctionAddressForPlatform(platform, "clEndRecordingQCOM");
        release_recording = (clReleaseRecordingQCOM_fn) clGetExtensionFunctionAddressForPlatform(platform, "clReleaseRecordingQCOM");
        enqueue_recording = (clEnqueueRecordingQCOM_fn) clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueRecordingQCOM");
        GGML_ASSERT(new_recording && end_recording && release_recording && enqueue_recording);

        const cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_RECORDABLE_QCOM, 0 };
        CL_CHECK((recording_queue = clCreateCommandQueueWithProperties(context, device, properties, &status), status));
        CL_CHECK((recording = new_recording(recording_queue, &status), status));
        recording_in_progress = true;
    }

    void set_arg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void * arg_value) {
        if (recording_in_progress) {
            CL_CHECK(clSetKernelArg(kernel, arg_index, arg_size, arg_value));
        } else {
            args.push_back({ dispatch_index, arg_index, arg_size, arg_value });
        }
    }

    void record(cl_kernel kernel, const std::vector<size_t> & global, const std::vector<size_t> & local) {
        if (recording_in_progress) {
            CL_CHECK(clEnqueueNDRangeKernel(recording_queue, kernel, global.size(), nullptr, global.data(), local.data(), 0, nullptr, nullptr));
        } else {
            global_sizes.push_back(global);
            local_sizes.push_back(local);
            ++dispatch_index;
        }
    }

    void enqueue(cl_event * event = nullptr) {
        if (recording_in_progress) {
            CL_CHECK(end_recording(recording));
            recording_in_progress = false;
            CL_CHECK(enqueue_recording(queue, recording, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, event));
            return;
        }

        global_updates.resize(global_sizes.size());
        local_updates.resize(local_sizes.size());
        for (size_t i = 0; i < global_sizes.size(); ++i) {
            global_updates[i] = { (cl_uint) i, global_sizes[i].data() };
            local_updates[i] = { (cl_uint) i, local_sizes[i].data() };
        }
        CL_CHECK(enqueue_recording(queue, recording,
            args.size(), args.data(), 0, nullptr,
            global_updates.size(), global_updates.data(), local_updates.size(), local_updates.data(),
            0, nullptr, event));
        args.clear();
        global_sizes.clear();
        local_sizes.clear();
        global_updates.clear();
        local_updates.clear();
        dispatch_index = 0;
    }

    ~cl_recordable_queue() {
        if (recording) {
            CL_CHECK(release_recording(recording));
        }
        if (recording_queue) {
            CL_CHECK(clReleaseCommandQueue(recording_queue));
        }
    }

private:
    cl_command_queue queue = nullptr;
    cl_command_queue recording_queue = nullptr;
    cl_recording_qcom recording = nullptr;
    bool recording_in_progress = false;
    cl_int status = CL_SUCCESS;

    clNewRecordingQCOM_fn new_recording = nullptr;
    clEndRecordingQCOM_fn end_recording = nullptr;
    clReleaseRecordingQCOM_fn release_recording = nullptr;
    clEnqueueRecordingQCOM_fn enqueue_recording = nullptr;

    cl_uint dispatch_index = 0;
    std::vector<cl_array_arg_qcom> args;
    std::vector<std::vector<size_t>> global_sizes;
    std::vector<std::vector<size_t>> local_sizes;
    std::vector<cl_workgroup_qcom> global_updates;
    std::vector<cl_workgroup_qcom> local_updates;
};
