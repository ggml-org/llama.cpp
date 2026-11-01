#pragma once

#include <cstddef>
#include <cstdint>

struct webkit_adapter_info {
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t adapter_type;
    uint32_t backend_type;
    uint32_t subgroup_min_size;
    uint32_t subgroup_max_size;
    const char * vendor;
    const char * architecture;
    const char * device;
    const char * description;
};

struct webkit_limits {
    uint64_t max_buffer_size;
    uint64_t max_storage_buffer_binding_size;
    uint32_t max_compute_invocations_per_workgroup;
    uint32_t max_compute_workgroup_storage_size;
    uint32_t max_compute_workgroups_per_dimension;
    uint32_t min_storage_buffer_offset_alignment;
    uint32_t min_uniform_buffer_offset_alignment;
};

struct webkit_constant {
    const char * key;
    size_t key_size;
    double value;
};

struct webkit_bind_entry {
    uint32_t binding;
    void * buffer;
    uint64_t offset;
    uint64_t size;
};

enum webkit_feature {
    WEBKIT_FEATURE_SHADER_F16,
    WEBKIT_FEATURE_SUBGROUPS,
    WEBKIT_FEATURE_TIMESTAMP_QUERY,
    WEBKIT_FEATURE_IMPLICIT_DEVICE_SYNCHRONIZATION,
};

using webkit_adapter_callback = void (*)(int status, void * adapter, const char * message, void * userdata);
using webkit_device_callback = void (*)(int status, void * device, const char * message, void * userdata);
using webkit_status_callback = void (*)(int status, void * userdata);

extern "C" {
void * webkit_create_instance();
void webkit_instance_process_events(void * instance);
void webkit_instance_request_adapter(void * instance, int power_preference, webkit_adapter_callback callback, void * userdata);
void webkit_adapter_get_info(void * adapter, webkit_adapter_info * info);
bool webkit_adapter_get_limits(void * adapter, webkit_limits * limits);
bool webkit_adapter_has_feature(void * adapter, int feature);
void webkit_adapter_request_device(void * adapter, const int * features, size_t feature_count, webkit_device_callback callback, void * userdata);
void * webkit_device_create_buffer(void * device, const char * label, size_t label_size, uint64_t usage, uint64_t size, bool mapped_at_creation);
void webkit_buffer_map_async(void * buffer, uint64_t mode, size_t offset, size_t size, webkit_status_callback callback, void * userdata);
void webkit_queue_on_submitted_work_done(void * queue, webkit_status_callback callback, void * userdata);
void * webkit_device_create_shader_module(void * device, const char * code, size_t code_size);
void * webkit_device_create_compute_pipeline(void * device, const char * label, size_t label_size, void * shader_module,
                                             const char * entry_point, size_t entry_point_size,
                                             const webkit_constant * constants, size_t constant_count);
void * webkit_device_create_bind_group(void * device, void * layout, const webkit_bind_entry * entries, size_t entry_count);
void * webkit_device_create_command_encoder(void * device, const char * label, size_t label_size);
void * webkit_command_encoder_begin_compute_pass(void * encoder);
void * webkit_command_encoder_finish(void * encoder);
void webkit_compute_pass_set_bind_group(void * pass, uint32_t group_index, void * bind_group,
                                        const uint32_t * offsets, size_t offset_count);
}
