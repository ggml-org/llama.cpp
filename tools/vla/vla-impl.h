#pragma once

#include "vla.h"

#include <cstdint>
#include <memory>
#include <string>

class vla_model {
public:
    virtual ~vla_model() = default;

    virtual const char * model_type() const = 0;

    virtual int64_t state_dim() const = 0;
    virtual int64_t control_dim() const = 0;
    virtual int64_t control_horizon() const = 0;
    virtual int64_t conditioning_dim() const = 0;
    virtual int64_t n_embodiments() const = 0;

    virtual bool predict(const vla_input & input, vla_output & output) = 0;
};

struct vla_metadata {
    std::string architecture;
    std::string model_type;
    int64_t     state_dim;
    int64_t     control_dim;
    int64_t     control_horizon;
    int64_t     conditioning_dim;
    int64_t     n_embodiments;
};

bool vla_metadata_load(const char * path, vla_metadata & metadata);
