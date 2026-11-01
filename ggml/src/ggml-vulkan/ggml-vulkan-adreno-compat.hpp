#pragma once
#include <cstdint>
#include <cstring>

// Compatibility proposal for the observed proprietary Adreno 750 compiler.
// This identifies the GPU/driver tuple, not a specific Android phone or firmware.
inline bool ggml_vk_adreno_750_dmmv_shmem(
        uint32_t vendor, uint32_t driver_id, uint32_t driver_version, const char * device_name) {
    return vendor == 0x5143u && driver_id == 8u &&
           driver_version == 2150604839u && device_name != nullptr &&
           std::strcmp(device_name, "Adreno (TM) 750") == 0;
}
