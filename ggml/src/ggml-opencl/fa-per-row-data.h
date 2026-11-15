#pragma once

#include <CL/cl_platform.h>

// Match the source fa_per_row_data.cl layout.
struct fa_per_row_data {
    cl_float4 m_i;
    cl_float4 d_i;
    cl_float4 m_i_minus_1;
    cl_float4 d_i_minus_1;
};

static_assert(sizeof(fa_per_row_data) == 64, "FA per-row data layout");
