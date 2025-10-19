#pragma once

#ifdef LLAMA_USE_OCI

#include <string>

// Structure to hold OCI pull results
struct oci_pull_result {
    std::string local_path;
    std::string digest;
    int         error_code;
    std::string error_message;

    bool success() const { return error_code == 0; }
};

// Pull a model from an OCI registry
// imageRef: full image reference (e.g., "ai/smollm2:135M-Q4_0", "registry.io/user/model:tag")
// cacheDir: directory to cache downloaded models
oci_pull_result oci_pull_model(const std::string & imageRef, const std::string & cacheDir);

#endif  // LLAMA_USE_OCI
