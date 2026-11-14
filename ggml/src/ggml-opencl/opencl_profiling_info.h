#pragma once

#include "ggml-impl.h"

#include <CL/cl.h>

#include <cinttypes>
#include <cstdio>
#include <iterator>
#include <string>
#include <vector>

#ifndef CL_CHECK
#define CL_CHECK(err)                                               \
    do {                                                            \
        cl_int err_ = (err);                                        \
        if (err_ != CL_SUCCESS) {                                   \
            GGML_LOG_ERROR("ggml_opencl: %s error %d at %s:%d\n",  \
                #err, err_, __FILE__, __LINE__);                    \
            GGML_ASSERT(0);                                         \
        }                                                           \
    } while (0)
#endif

class ProfilingInfo {
    struct ProfilingEvent {
        ProfilingEvent(cl_event event, cl_kernel kernel, cl_uint work_dim, const size_t * global_size, const size_t * local_size, const ggml_tensor * tensor)
            : kernel(kernel), event(event) {
            CL_CHECK(clRetainEvent(event));
            populate(work_dim, global_size, local_size, tensor);
        }

        void populate(cl_uint work_dim, const size_t * global_size, const size_t * local_size, const ggml_tensor * tensor) {
            op_name = tensor->name;

            for (cl_uint i = 0; i < work_dim; ++i) {
                this->global_size[i] = global_size[i];
                this->local_size[i] = local_size ? local_size[i] : 0;
            }

            output_size[0] = tensor->ne[0];
            output_size[1] = tensor->ne[1];
            output_size[2] = tensor->ne[2];
            output_size[3] = tensor->ne[3];
        }

        std::string op_name;
        std::string kernel_name;
        cl_kernel kernel = nullptr;
        cl_event event = nullptr;
        cl_ulong cmd_queued = 0;
        cl_ulong cmd_submit = 0;
        cl_ulong cmd_start = 0;
        cl_ulong cmd_end = 0;
        cl_ulong cmd_duration_ns = 0;
        size_t global_size[3] = { 0, 0, 0 };
        size_t local_size[3] = { 0, 0, 0 };
        size_t output_size[4] = { 0, 0, 0, 0 };
    };

public:
    void populate(cl_event event, cl_kernel kernel, cl_uint work_dim, const size_t * global_size, const size_t * local_size, const ggml_tensor * tensor) {
        profiling_info.emplace_back(event, kernel, work_dim, global_size, local_size, tensor);
        if (profiling_info.size() >= 2048) {
            flush();
        }
    }

    void flush() {
        for (ProfilingEvent & info : profiling_info) {
            cl_ulong cmd_complete;
            CL_CHECK(clWaitForEvents(1, &info.event));
            CL_CHECK(clGetEventProfilingInfo(info.event, CL_PROFILING_COMMAND_QUEUED, sizeof(info.cmd_queued), &info.cmd_queued, nullptr));
            CL_CHECK(clGetEventProfilingInfo(info.event, CL_PROFILING_COMMAND_SUBMIT, sizeof(info.cmd_submit), &info.cmd_submit, nullptr));
            CL_CHECK(clGetEventProfilingInfo(info.event, CL_PROFILING_COMMAND_START, sizeof(info.cmd_start), &info.cmd_start, nullptr));
            CL_CHECK(clGetEventProfilingInfo(info.event, CL_PROFILING_COMMAND_END, sizeof(info.cmd_end), &info.cmd_end, nullptr));
            CL_CHECK(clGetEventProfilingInfo(info.event, CL_PROFILING_COMMAND_COMPLETE, sizeof(cmd_complete), &cmd_complete, nullptr));
            CL_CHECK(clReleaseEvent(info.event));
            info.event = nullptr;

            char kernel_name[512];
            CL_CHECK(clGetKernelInfo(info.kernel, CL_KERNEL_FUNCTION_NAME, sizeof(kernel_name), kernel_name, nullptr));
            info.kernel_name = kernel_name;
            info.cmd_duration_ns = info.cmd_end - info.cmd_start;
        }
        profiling_results.insert(profiling_results.end(), std::make_move_iterator(profiling_info.begin()), std::make_move_iterator(profiling_info.end()));
        profiling_info.clear();
    }

    void write() {
        flush();
        if (profiling_results.empty()) {
            return;
        }

        FILE * fperf = fopen("cl_profiling.csv", "w");
        if (!fperf) {
            GGML_LOG_ERROR("Failed to open cl_profiling.csv\n");
            return;
        }
        fprintf(fperf, "op name, kernel name, exec duration (ms), global size, local size, output size\n");
        for (const ProfilingEvent & info : profiling_results) {
            fprintf(fperf, "%s,%s,%f,%zux%zux%zu,%zux%zux%zu,%zux%zux%zux%zu\n", info.op_name.c_str(), info.kernel_name.c_str(),
                info.cmd_duration_ns/1.e6f, info.global_size[0], info.global_size[1], info.global_size[2], info.local_size[0], info.local_size[1], info.local_size[2],
                info.output_size[0], info.output_size[1], info.output_size[2], info.output_size[3]);
        }
        fclose(fperf);

        FILE * ftrace = fopen("cl_trace.json", "w");
        if (!ftrace) {
            GGML_LOG_ERROR("Failed to open cl_trace.json\n");
            return;
        }
        fprintf(ftrace, "[\n");
        bool first = true;
        for (const ProfilingEvent & info : profiling_results) {
            const cl_ulong timestamps[] = { info.cmd_queued, info.cmd_submit, info.cmd_start, info.cmd_end };
            const char * phases[] = { "B", "E", "B", "E" };
            const char * threads[] = { "Host", "Host", "Device", "Device" };
            for (size_t i = 0; i < 4; ++i) {
                fprintf(ftrace, "%s{\"name\": \"%s\", \"cat\": \"OpenCL\", \"ph\": \"%s\", \"ts\": %" PRIu64 ", \"pid\": \"\", \"tid\": \"%s\"}", first ? "" : ",\n", info.kernel_name.c_str(), phases[i], timestamps[i]/1000, threads[i]);
                first = false;
            }
        }
        fprintf(ftrace, "\n]\n");
        fclose(ftrace);
    }

    void clear() {
        for (ProfilingEvent & info : profiling_info) {
            if (info.event) {
                CL_CHECK(clReleaseEvent(info.event));
            }
        }
        profiling_info.clear();
        profiling_results.clear();
    }

private:
    std::vector<ProfilingEvent> profiling_info;
    std::vector<ProfilingEvent> profiling_results;
};
