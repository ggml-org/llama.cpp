#include <level_zero/ze_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifndef MUTABLE_PROBE_SPIRV_PATH
#define MUTABLE_PROBE_SPIRV_PATH "sycl-mutable-command-list-probe.spv"
#endif

namespace {

using clock_type = std::chrono::steady_clock;

const char * ze_result_name(ze_result_t result) {
    switch (result) {
        case ZE_RESULT_SUCCESS: return "ZE_RESULT_SUCCESS";
        case ZE_RESULT_ERROR_UNINITIALIZED: return "ZE_RESULT_ERROR_UNINITIALIZED";
        case ZE_RESULT_ERROR_DEVICE_LOST: return "ZE_RESULT_ERROR_DEVICE_LOST";
        case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY: return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
        case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY: return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
        case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE: return "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE";
        case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE: return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
        case ZE_RESULT_ERROR_INVALID_ARGUMENT: return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
        case ZE_RESULT_ERROR_INVALID_NULL_HANDLE: return "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
        case ZE_RESULT_ERROR_INVALID_NULL_POINTER: return "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
        case ZE_RESULT_ERROR_INVALID_SIZE: return "ZE_RESULT_ERROR_INVALID_SIZE";
        default: return "ZE_RESULT_OTHER";
    }
}

bool check(ze_result_t result, const char * operation, std::string & error) {
    if (result == ZE_RESULT_SUCCESS) {
        return true;
    }
    error = std::string(operation) + ": " + ze_result_name(result) + " (" + std::to_string(result) + ")";
    return false;
}

double elapsed_us(clock_type::time_point start, clock_type::time_point end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if ((values.size() & 1U) != 0U) {
        return values[middle];
    }
    return (values[middle - 1] + values[middle]) * 0.5;
}

std::vector<uint8_t> read_binary(const std::string & path, std::string & error) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        error = "cannot open SPIR-V module: " + path;
        return {};
    }
    const std::streamsize size = input.tellg();
    if (size <= 0) {
        error = "empty SPIR-V module: " + path;
        return {};
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    input.seekg(0, std::ios::beg);
    if (!input.read(reinterpret_cast<char *>(bytes.data()), size)) {
        error = "cannot read SPIR-V module: " + path;
        return {};
    }
    return bytes;
}

struct Handles {
    ze_context_handle_t context = nullptr;
    ze_command_queue_handle_t queue = nullptr;
    ze_command_list_handle_t mutable_list = nullptr;
    ze_module_handle_t module = nullptr;
    ze_kernel_handle_t kernel = nullptr;
    ze_event_pool_handle_t event_pool = nullptr;
    std::array<ze_event_handle_t, 4> events = {};
    std::array<int *, 2> output = {};

    ~Handles() {
        if (queue != nullptr) {
            zeCommandQueueSynchronize(queue, std::numeric_limits<uint64_t>::max());
        }
        for (auto event : events) {
            if (event != nullptr) {
                zeEventDestroy(event);
            }
        }
        if (event_pool != nullptr) {
            zeEventPoolDestroy(event_pool);
        }
        if (mutable_list != nullptr) {
            zeCommandListDestroy(mutable_list);
        }
        if (kernel != nullptr) {
            zeKernelDestroy(kernel);
        }
        if (module != nullptr) {
            zeModuleDestroy(module);
        }
        for (auto pointer : output) {
            if (pointer != nullptr && context != nullptr) {
                zeMemFree(context, pointer);
            }
        }
        if (queue != nullptr) {
            zeCommandQueueDestroy(queue);
        }
        if (context != nullptr) {
            zeContextDestroy(context);
        }
    }
};

struct Measurement {
    uint32_t depth = 0;
    uint32_t group_count = 0;
    double update_us = 0.0;
    double replay_us = 0.0;
    double mutable_total_us = 0.0;
    double rebuild_total_us = 0.0;
    double savings_pct = 0.0;
    bool exact = false;
};

bool validate_output(const int * output, uint32_t count, int value) {
    for (uint32_t index = 0; index < count; ++index) {
        if (output[index] != value + static_cast<int>(index)) {
            return false;
        }
    }
    return true;
}

bool create_regular_list(
        ze_context_handle_t context,
        ze_device_handle_t device,
        uint32_t ordinal,
        ze_kernel_handle_t kernel,
        int * output,
        int value,
        uint32_t groups,
        ze_event_handle_t wait_event,
        ze_event_handle_t signal_event,
        ze_command_list_handle_t & list,
        std::string & error) {
    ze_command_list_desc_t list_desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    list_desc.commandQueueGroupOrdinal = ordinal;
    if (!check(zeCommandListCreate(context, device, &list_desc, &list), "zeCommandListCreate", error)) {
        return false;
    }
    if (!check(zeKernelSetArgumentValue(kernel, 0, sizeof(output), &output), "zeKernelSetArgumentValue(output)", error) ||
        !check(zeKernelSetArgumentValue(kernel, 1, sizeof(value), &value), "zeKernelSetArgumentValue(value)", error)) {
        return false;
    }
    ze_group_count_t group_count = {groups, 1, 1};
    if (!check(zeCommandListAppendLaunchKernel(list, kernel, &group_count, signal_event, 1, &wait_event),
               "zeCommandListAppendLaunchKernel", error) ||
        !check(zeCommandListClose(list), "zeCommandListClose", error)) {
        return false;
    }
    return true;
}

bool mutate_list(
        ze_command_list_handle_t list,
        uint64_t command_id,
        int * output,
        int value,
        uint32_t groups,
        ze_event_handle_t wait_event,
        ze_event_handle_t signal_event,
        std::string & error) {
    ze_mutable_kernel_argument_exp_desc_t value_desc = {ZE_STRUCTURE_TYPE_MUTABLE_KERNEL_ARGUMENT_EXP_DESC};
    value_desc.commandId = command_id;
    value_desc.argIndex = 1;
    value_desc.argSize = sizeof(value);
    value_desc.pArgValue = &value;

    ze_mutable_kernel_argument_exp_desc_t output_desc = {ZE_STRUCTURE_TYPE_MUTABLE_KERNEL_ARGUMENT_EXP_DESC};
    output_desc.commandId = command_id;
    output_desc.argIndex = 0;
    output_desc.argSize = sizeof(output);
    output_desc.pArgValue = &output;
    output_desc.pNext = &value_desc;

    ze_group_count_t group_count = {groups, 1, 1};
    ze_mutable_group_count_exp_desc_t group_desc = {ZE_STRUCTURE_TYPE_MUTABLE_GROUP_COUNT_EXP_DESC};
    group_desc.commandId = command_id;
    group_desc.pGroupCount = &group_count;
    group_desc.pNext = &output_desc;

    ze_mutable_commands_exp_desc_t mutable_desc = {ZE_STRUCTURE_TYPE_MUTABLE_COMMANDS_EXP_DESC};
    mutable_desc.pNext = &group_desc;

    if (!check(zeCommandListUpdateMutableCommandsExp(list, &mutable_desc),
               "zeCommandListUpdateMutableCommandsExp", error) ||
        !check(zeCommandListUpdateMutableCommandWaitEventsExp(list, command_id, 1, &wait_event),
               "zeCommandListUpdateMutableCommandWaitEventsExp", error) ||
        !check(zeCommandListUpdateMutableCommandSignalEventExp(list, command_id, signal_event),
               "zeCommandListUpdateMutableCommandSignalEventExp", error) ||
        !check(zeCommandListClose(list), "zeCommandListClose(mutable)", error)) {
        return false;
    }
    return true;
}

bool run_probe(const std::string & spirv_path, uint32_t iterations, std::vector<Measurement> & measurements,
               std::string & device_name, uint32_t & extension_version, uint32_t & mutable_flags, std::string & error) {
    if (!check(zeInit(ZE_INIT_FLAG_GPU_ONLY), "zeInit", error)) {
        return false;
    }

    uint32_t driver_count = 0;
    if (!check(zeDriverGet(&driver_count, nullptr), "zeDriverGet(count)", error) || driver_count == 0) {
        if (error.empty()) error = "no Level Zero drivers";
        return false;
    }
    std::vector<ze_driver_handle_t> drivers(driver_count);
    if (!check(zeDriverGet(&driver_count, drivers.data()), "zeDriverGet", error)) {
        return false;
    }

    ze_driver_handle_t driver = nullptr;
    ze_device_handle_t device = nullptr;
    for (auto candidate_driver : drivers) {
        uint32_t device_count = 0;
        if (zeDeviceGet(candidate_driver, &device_count, nullptr) != ZE_RESULT_SUCCESS) continue;
        std::vector<ze_device_handle_t> devices(device_count);
        if (zeDeviceGet(candidate_driver, &device_count, devices.data()) != ZE_RESULT_SUCCESS) continue;
        for (auto candidate_device : devices) {
            ze_device_properties_t properties = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
            if (zeDeviceGetProperties(candidate_device, &properties) == ZE_RESULT_SUCCESS &&
                properties.type == ZE_DEVICE_TYPE_GPU && properties.vendorId == 0x8086) {
                driver = candidate_driver;
                device = candidate_device;
                device_name = properties.name;
                break;
            }
        }
        if (device != nullptr) break;
    }
    if (device == nullptr) {
        error = "no Intel Level Zero GPU";
        return false;
    }

    uint32_t extension_count = 0;
    if (!check(zeDriverGetExtensionProperties(driver, &extension_count, nullptr),
               "zeDriverGetExtensionProperties(count)", error)) {
        return false;
    }
    std::vector<ze_driver_extension_properties_t> extensions(extension_count);
    if (!check(zeDriverGetExtensionProperties(driver, &extension_count, extensions.data()),
               "zeDriverGetExtensionProperties", error)) {
        return false;
    }
    for (const auto & extension : extensions) {
        if (std::strcmp(extension.name, ZE_MUTABLE_COMMAND_LIST_EXP_NAME) == 0) {
            extension_version = extension.version;
            break;
        }
    }
    if (extension_version == 0) {
        error = "ZE_experimental_mutable_command_list is not advertised";
        return false;
    }

    ze_mutable_command_list_exp_properties_t mutable_properties = {
        ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_PROPERTIES};
    ze_device_properties_t properties = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
    properties.pNext = &mutable_properties;
    if (!check(zeDeviceGetProperties(device, &properties), "zeDeviceGetProperties(mutable)", error)) {
        return false;
    }
    mutable_flags = mutable_properties.mutableCommandFlags;
    constexpr uint32_t required_flags = ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_ARGUMENTS |
                                        ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_COUNT |
                                        ZE_MUTABLE_COMMAND_EXP_FLAG_SIGNAL_EVENT |
                                        ZE_MUTABLE_COMMAND_EXP_FLAG_WAIT_EVENTS;
    if ((mutable_flags & required_flags) != required_flags) {
        error = "driver lacks required argument, group-count, or event mutation flags";
        return false;
    }

    Handles handles;
    ze_context_desc_t context_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    if (!check(zeContextCreate(driver, &context_desc, &handles.context), "zeContextCreate", error)) {
        return false;
    }

    uint32_t group_count = 0;
    if (!check(zeDeviceGetCommandQueueGroupProperties(device, &group_count, nullptr),
               "zeDeviceGetCommandQueueGroupProperties(count)", error)) {
        return false;
    }
    std::vector<ze_command_queue_group_properties_t> queue_groups(group_count);
    for (auto & group : queue_groups) group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    if (!check(zeDeviceGetCommandQueueGroupProperties(device, &group_count, queue_groups.data()),
               "zeDeviceGetCommandQueueGroupProperties", error)) {
        return false;
    }
    uint32_t ordinal = std::numeric_limits<uint32_t>::max();
    for (uint32_t index = 0; index < group_count; ++index) {
        if ((queue_groups[index].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) != 0) {
            ordinal = index;
            break;
        }
    }
    if (ordinal == std::numeric_limits<uint32_t>::max()) {
        error = "no Level Zero compute queue group";
        return false;
    }

    ze_command_queue_desc_t queue_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
    queue_desc.ordinal = ordinal;
    queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    if (!check(zeCommandQueueCreate(handles.context, device, &queue_desc, &handles.queue),
               "zeCommandQueueCreate", error)) {
        return false;
    }

    const std::vector<uint8_t> module_bytes = read_binary(spirv_path, error);
    if (module_bytes.empty()) return false;
    ze_module_desc_t module_desc = {ZE_STRUCTURE_TYPE_MODULE_DESC};
    module_desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    module_desc.inputSize = module_bytes.size();
    module_desc.pInputModule = module_bytes.data();
    ze_module_build_log_handle_t build_log = nullptr;
    const ze_result_t module_result = zeModuleCreate(handles.context, device, &module_desc, &handles.module, &build_log);
    if (module_result != ZE_RESULT_SUCCESS) {
        if (build_log != nullptr) {
            size_t log_size = 0;
            zeModuleBuildLogGetString(build_log, &log_size, nullptr);
            std::string log(log_size, '\0');
            zeModuleBuildLogGetString(build_log, &log_size, log.data());
            error = std::string("zeModuleCreate: ") + ze_result_name(module_result) + ": " + log;
        } else {
            check(module_result, "zeModuleCreate", error);
        }
        if (build_log != nullptr) zeModuleBuildLogDestroy(build_log);
        return false;
    }
    if (build_log != nullptr) zeModuleBuildLogDestroy(build_log);

    ze_kernel_desc_t kernel_desc = {ZE_STRUCTURE_TYPE_KERNEL_DESC};
    kernel_desc.pKernelName = "mutable_probe";
    if (!check(zeKernelCreate(handles.module, &kernel_desc, &handles.kernel), "zeKernelCreate", error) ||
        !check(zeKernelSetGroupSize(handles.kernel, 1, 1, 1), "zeKernelSetGroupSize", error)) {
        return false;
    }

    constexpr uint32_t max_groups = 128;
    ze_device_mem_alloc_desc_t device_alloc_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    ze_host_mem_alloc_desc_t host_alloc_desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    for (auto & pointer : handles.output) {
        if (!check(zeMemAllocShared(handles.context, &device_alloc_desc, &host_alloc_desc,
                                    max_groups * sizeof(int), alignof(int), device, reinterpret_cast<void **>(&pointer)),
                   "zeMemAllocShared", error)) {
            return false;
        }
        std::fill(pointer, pointer + max_groups, -1);
    }

    ze_event_pool_desc_t event_pool_desc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
    event_pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    event_pool_desc.count = handles.events.size();
    if (!check(zeEventPoolCreate(handles.context, &event_pool_desc, 1, &device, &handles.event_pool),
               "zeEventPoolCreate", error)) {
        return false;
    }
    for (uint32_t index = 0; index < handles.events.size(); ++index) {
        ze_event_desc_t event_desc = {ZE_STRUCTURE_TYPE_EVENT_DESC};
        event_desc.index = index;
        event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
        event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
        if (!check(zeEventCreate(handles.event_pool, &event_desc, &handles.events[index]), "zeEventCreate", error)) {
            return false;
        }
    }
    if (!check(zeEventHostSignal(handles.events[0]), "zeEventHostSignal(wait0)", error) ||
        !check(zeEventHostSignal(handles.events[1]), "zeEventHostSignal(wait1)", error)) {
        return false;
    }

    ze_mutable_command_list_exp_desc_t mutable_list_desc = {ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_DESC};
    ze_command_list_desc_t list_desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    list_desc.pNext = &mutable_list_desc;
    list_desc.commandQueueGroupOrdinal = ordinal;
    if (!check(zeCommandListCreate(handles.context, device, &list_desc, &handles.mutable_list),
               "zeCommandListCreate(mutable)", error)) {
        return false;
    }

    int initial_value = 17;
    int * initial_output = handles.output[0];
    if (!check(zeKernelSetArgumentValue(handles.kernel, 0, sizeof(initial_output), &initial_output),
               "zeKernelSetArgumentValue(initial output)", error) ||
        !check(zeKernelSetArgumentValue(handles.kernel, 1, sizeof(initial_value), &initial_value),
               "zeKernelSetArgumentValue(initial value)", error)) {
        return false;
    }
    ze_mutable_command_id_exp_desc_t command_desc = {ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_ID_EXP_DESC};
    command_desc.flags = required_flags;
    uint64_t command_id = 0;
    if (!check(zeCommandListGetNextCommandIdExp(handles.mutable_list, &command_desc, &command_id),
               "zeCommandListGetNextCommandIdExp", error)) {
        return false;
    }
    ze_group_count_t initial_groups = {8, 1, 1};
    if (!check(zeCommandListAppendLaunchKernel(handles.mutable_list, handles.kernel, &initial_groups,
                                               handles.events[2], 1, &handles.events[0]),
               "zeCommandListAppendLaunchKernel(mutable)", error) ||
        !check(zeCommandListClose(handles.mutable_list), "zeCommandListClose(initial mutable)", error) ||
        !check(zeCommandQueueExecuteCommandLists(handles.queue, 1, &handles.mutable_list, nullptr),
               "zeCommandQueueExecuteCommandLists(initial mutable)", error) ||
        !check(zeCommandQueueSynchronize(handles.queue, std::numeric_limits<uint64_t>::max()),
               "zeCommandQueueSynchronize(initial mutable)", error)) {
        return false;
    }
    if (!validate_output(initial_output, initial_groups.groupCountX, initial_value) ||
        zeEventQueryStatus(handles.events[2]) != ZE_RESULT_SUCCESS) {
        error = "initial mutable launch produced incorrect output or signal event";
        return false;
    }

    const std::array<uint32_t, 4> depths = {0, 4096, 8192, 16384};
    for (uint32_t depth : depths) {
        const uint32_t groups = depth == 0 ? 8 : depth / 128;
        std::vector<double> update_samples;
        std::vector<double> replay_samples;
        std::vector<double> mutable_total_samples;
        std::vector<double> rebuild_total_samples;
        update_samples.reserve(iterations);
        replay_samples.reserve(iterations);
        mutable_total_samples.reserve(iterations);
        rebuild_total_samples.reserve(iterations);
        bool exact = true;

        for (uint32_t iteration = 0; iteration <= iterations; ++iteration) {
            const uint32_t slot = iteration & 1U;
            const uint32_t active_groups = slot == 0 ? groups : std::max(1U, groups / 2U);
            const int value = static_cast<int>(depth + iteration + 101);
            int * output = handles.output[slot];
            std::fill(output, output + max_groups, -1);
            zeEventHostReset(handles.events[2 + slot]);

            const auto total_start = clock_type::now();
            const auto update_start = clock_type::now();
            if (!mutate_list(handles.mutable_list, command_id, output, value, active_groups,
                             handles.events[slot], handles.events[2 + slot], error)) {
                return false;
            }
            const auto update_end = clock_type::now();
            if (!check(zeCommandQueueExecuteCommandLists(handles.queue, 1, &handles.mutable_list, nullptr),
                       "zeCommandQueueExecuteCommandLists(mutable replay)", error)) {
                return false;
            }
            const auto replay_start = clock_type::now();
            if (!check(zeCommandQueueSynchronize(handles.queue, std::numeric_limits<uint64_t>::max()),
                       "zeCommandQueueSynchronize(mutable replay)", error)) {
                return false;
            }
            const auto replay_end = clock_type::now();
            exact = exact && validate_output(output, active_groups, value) &&
                    zeEventQueryStatus(handles.events[2 + slot]) == ZE_RESULT_SUCCESS;
            if (iteration != 0) {
                update_samples.push_back(elapsed_us(update_start, update_end));
                replay_samples.push_back(elapsed_us(replay_start, replay_end));
                mutable_total_samples.push_back(elapsed_us(total_start, replay_end));
            }
        }

        for (uint32_t iteration = 0; iteration <= iterations; ++iteration) {
            const uint32_t slot = iteration & 1U;
            const uint32_t active_groups = slot == 0 ? groups : std::max(1U, groups / 2U);
            const int value = static_cast<int>(depth + iteration + 10001);
            int * output = handles.output[slot];
            std::fill(output, output + max_groups, -1);
            zeEventHostReset(handles.events[2 + slot]);
            ze_command_list_handle_t list = nullptr;
            const auto rebuild_start = clock_type::now();
            if (!create_regular_list(handles.context, device, ordinal, handles.kernel, output, value, active_groups,
                                     handles.events[slot], handles.events[2 + slot], list, error)) {
                if (list != nullptr) zeCommandListDestroy(list);
                return false;
            }
            if (!check(zeCommandQueueExecuteCommandLists(handles.queue, 1, &list, nullptr),
                       "zeCommandQueueExecuteCommandLists(rebuild)", error) ||
                !check(zeCommandQueueSynchronize(handles.queue, std::numeric_limits<uint64_t>::max()),
                       "zeCommandQueueSynchronize(rebuild)", error)) {
                zeCommandListDestroy(list);
                return false;
            }
            const auto rebuild_end = clock_type::now();
            exact = exact && validate_output(output, active_groups, value) &&
                    zeEventQueryStatus(handles.events[2 + slot]) == ZE_RESULT_SUCCESS;
            zeCommandListDestroy(list);
            if (iteration != 0) rebuild_total_samples.push_back(elapsed_us(rebuild_start, rebuild_end));
        }

        Measurement measurement;
        measurement.depth = depth;
        measurement.group_count = groups;
        measurement.update_us = median(update_samples);
        measurement.replay_us = median(replay_samples);
        measurement.mutable_total_us = median(mutable_total_samples);
        measurement.rebuild_total_us = median(rebuild_total_samples);
        measurement.savings_pct = 100.0 * (1.0 - measurement.mutable_total_us / measurement.rebuild_total_us);
        measurement.exact = exact;
        measurements.push_back(measurement);
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    const std::string spirv_path = argc > 1 ? argv[1] : MUTABLE_PROBE_SPIRV_PATH;
    const uint32_t iterations = argc > 2 ? static_cast<uint32_t>(std::stoul(argv[2])) : 100;
    std::vector<Measurement> measurements;
    std::string device_name;
    std::string error;
    uint32_t extension_version = 0;
    uint32_t mutable_flags = 0;
    const bool pass = run_probe(spirv_path, iterations, measurements, device_name, extension_version, mutable_flags, error);

    std::cout << "{\n"
              << "  \"pass\": " << (pass ? "true" : "false") << ",\n"
              << "  \"device\": \"" << device_name << "\",\n"
              << "  \"extension_version\": " << extension_version << ",\n"
              << "  \"mutable_command_flags\": " << mutable_flags << ",\n"
              << "  \"iterations\": " << iterations << ",\n"
              << "  \"error\": \"" << error << "\",\n"
              << "  \"measurements\": [\n";
    for (size_t index = 0; index < measurements.size(); ++index) {
        const auto & measurement = measurements[index];
        std::cout << "    {\"depth\": " << measurement.depth
                  << ", \"group_count\": " << measurement.group_count
                  << ", \"update_us\": " << measurement.update_us
                  << ", \"replay_us\": " << measurement.replay_us
                  << ", \"mutable_total_us\": " << measurement.mutable_total_us
                  << ", \"rebuild_total_us\": " << measurement.rebuild_total_us
                  << ", \"savings_pct\": " << measurement.savings_pct
                  << ", \"exact\": " << (measurement.exact ? "true" : "false") << "}"
                  << (index + 1 == measurements.size() ? "\n" : ",\n");
    }
    std::cout << "  ]\n}\n";
    return pass && std::all_of(measurements.begin(), measurements.end(), [](const Measurement & value) {
        return value.exact;
    }) ? 0 : 1;
}
