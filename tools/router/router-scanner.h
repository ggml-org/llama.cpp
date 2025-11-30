#pragma once

#include "router-config.h"

#include <string>
#include <vector>

std::vector<ModelConfig> scan_default_models();
std::vector<ModelConfig> scan_custom_dir(const std::string & path, const std::string & state);
