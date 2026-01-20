#pragma once

#include <stdint.h>

struct virgl_renderer_capset_apir {
   uint32_t apir_version;
   uint32_t supports_blob_resources;
   uint32_t reserved[4];           // For future expansion
};
