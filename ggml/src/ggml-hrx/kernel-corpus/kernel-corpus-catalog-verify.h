#pragma once

#include "kernel-corpus-catalog.h"

namespace ggml::hrx {

#include "kernel-corpus-catalog.inc"

}  // namespace ggml::hrx

#define GGML_HRX_KERNEL_REF(family_literal, name_literal)                                     \
    ([] {                                                                                     \
        static_assert(::ggml::hrx::kernel_catalog_entry_exists(family_literal, name_literal), \
                      "unknown HRX kernel catalog entry");                                    \
        return ::ggml::hrx::kernel_catalog_ref(family_literal, name_literal);                 \
    }())
