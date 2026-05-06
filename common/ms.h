#pragma once

#include <string>

// Ref: https://www.modelscope.cn/docs

namespace ms {

struct download_result {
    std::string model_path;
    std::string mmproj_path;
};

// Download a model (and optional mmproj) from ModelScope
// clean_repo_id: format "owner/repo" (without quantization tag)
// filename: specific filename to download (optional, auto-selects best GGUF if empty)
// offline: if true, only check local cache without network requests
// quant_tag: quantization tag extracted from original repo ID (e.g., "Q8_0")
// token: authentication token for private repositories (via MS_TOKEN env or -hft flag)
download_result download_model(const std::string & clean_repo_id, const std::string & filename = "", bool offline = false, const std::string & quant_tag = "", const std::string & token = "");

} // namespace ms
