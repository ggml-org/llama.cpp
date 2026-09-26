#pragma once
#include <cstdint>
#include <cstring>

// Compatibility proposal for the observed proprietary Adreno 750 compiler.
// This identifies the GPU/driver tuple, not a specific Android phone or firmware.
inline bool ggml_vk_adreno_750_matvec_shmem(
        uint32_t vendor, uint32_t driver_id, uint32_t driver_version,
        const char * device_name, bool f32_subgroup_no_shmem,
        uint32_t block, uint32_t rows, uint32_t columns,
        uint32_t required_subgroup, bool full_subgroups, uint32_t shared_bytes) {
    return vendor == 0x5143u && driver_id == 8u &&
           driver_version == 2150604839u && device_name != nullptr &&
           std::strcmp(device_name, "Adreno (TM) 750") == 0 &&
           f32_subgroup_no_shmem && block == 64 && rows == 1 && columns == 5 &&
           required_subgroup == 64 && full_subgroups && shared_bytes >= 1280;
}
