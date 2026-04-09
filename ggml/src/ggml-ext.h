#pragma once

#include "ggml.h"
#include "ggml-backend.h"

// This is a "staging" header for new ggml API
// It is not publicly available and it should not be used by 3rd party projects
//
// When the API matures enough, it will be moved to the official public API

//
// Meta backend
//

#define GGML_BACKEND_META_MAX_DEVICES 16

enum ggml_backend_meta_split_axis {
    // tensor split by tensor dimensions:
    GGML_BACKEND_SPLIT_AXIS_0   =  0,
    GGML_BACKEND_SPLIT_AXIS_1   =  1,
    GGML_BACKEND_SPLIT_AXIS_2   =  2,
    GGML_BACKEND_SPLIT_AXIS_3   =  3,

    GGML_BACKEND_SPLIT_AXIS_MIRRORED = 10, // all values on all backends
    GGML_BACKEND_SPLIT_AXIS_PARTIAL  = 11, // each backend has a partial sum

    // for internal bookkeeping only:
    GGML_BACKEND_SPLIT_AXIS_NONE     = 98,
    GGML_BACKEND_SPLIT_AXIS_UNKNOWN  = 99,
};
GGML_API const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis);

struct ggml_backend_meta_split_state {
    enum ggml_backend_meta_split_axis axis;

    // for tensors with axis >= 0 && axis < GGML_MAX_DIMS:
    //   - each device has a slice of the tensor along the split axis
    //   - most tensors have n_segments == 1 and a contiguous slice of the tensor data
    //   - some tensors have an inhomogenenous data layout along the split axis,
    //     those tensors are divided into segments which are each individually split across devices
    //   - ne has one entry per segment and device that add up to ggml_tensor::ne for that axis,
    //     the outer/inner loops are over segments/devices like [seg0_dev0, seg0_dev1, seg1_dev0, seg1_dev1],
    //   - for example, a transformer may have a fused QKV matrix rather than 3 matrices, those would be 3 separate segments
    //     that each need to be split individually across devices so that each device gets a slice of Q, K, and V
    int64_t  ne[16*GGML_BACKEND_META_MAX_DEVICES];
    uint32_t n_segments;
};

// function to assign split states for statically allocated tensors, compute tensor split states will be assigned to be compatible:
typedef struct ggml_backend_meta_split_state(*ggml_backend_meta_get_split_state_t)(const struct ggml_tensor * tensor, void * userdata);

// create a new meta device from "simple" devices, meta buffer type/buffer/backend is then derived from this:
// TODO: this looks a bit strange - a backend API creates a device. I think we should try
//       express this as a backend registry functionality instead
GGML_API ggml_backend_dev_t ggml_backend_meta_device(
    ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud);
