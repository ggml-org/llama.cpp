#pragma once

#include "../vla-impl.h"

#include <memory>

std::unique_ptr<vla_model> vla_model_minicpm_robot_create(
        const char *                    path,
        const vla_context_params &      params);
