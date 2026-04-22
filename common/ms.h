#pragma once

#include <string>
#include <vector>

// Ref: https://www.modelscope.cn/docs

namespace ms {

struct ms_file {
    std::string path;
    std::string url;
    std::string local_path;
    std::string repo_id;
    size_t size = 0;
};

using ms_files = std::vector<ms_file>;

// Result structure for model downloading (supports multimodal)
struct download_result {
    std::string model_path;
    std::string mmproj_path;
};

// Download a model from ModelScope
// clean_repo_id: format "owner/repo" (without quantization tag)
// filename: specific filename to download (optional)
// offline: if true, only check local cache without network requests
// quant_tag: quantization tag extracted from original repo ID (e.g., "Q8_0")
// token: authentication token for private repositories
std::string download_model(const std::string & clean_repo_id, const std::string & filename = "", bool offline = false, const std::string & quant_tag = "", const std::string & token = "");

// Download a model from ModelScope with multimodal support
// Returns both model path and mmproj path if available
download_result download_model_with_mmproj(const std::string & clean_repo_id, const std::string & filename = "", bool offline = false, const std::string & quant_tag = "", const std::string & token = "");

} // namespace ms
