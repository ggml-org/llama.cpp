#ifdef LLAMA_USE_OCI

#include "oci.h"

#include "log.h"

#include <nlohmann/json.hpp>

// Include the Go-generated header
#include "../oci-go/liboci.h"

using json = nlohmann::ordered_json;

oci_pull_result oci_pull_model(const std::string & imageRef, const std::string & cacheDir) {
    oci_pull_result result;
    result.error_code = 0;

    // Call the Go function
    char * json_result = PullOCIModel(const_cast<char *>(imageRef.c_str()), const_cast<char *>(cacheDir.c_str()));

    if (json_result == nullptr) {
        result.error_code    = 1;
        result.error_message = "Failed to call OCI pull function";
        return result;
    }

    try {
        // Parse the JSON result
        std::string json_str(json_result);
        auto        j = json::parse(json_str);

        if (j.contains("LocalPath")) {
            result.local_path = j["LocalPath"].get<std::string>();
        }
        if (j.contains("Digest")) {
            result.digest = j["Digest"].get<std::string>();
        }
        if (j.contains("Error") && !j["Error"].is_null()) {
            auto err = j["Error"];
            if (err.contains("Code")) {
                result.error_code = err["Code"].get<int>();
            }
            if (err.contains("Message")) {
                result.error_message = err["Message"].get<std::string>();
            }
        }
    } catch (const std::exception & e) {
        result.error_code    = 1;
        result.error_message = std::string("Failed to parse result: ") + e.what();
    }

    // Free the Go-allocated string
    FreeString(json_result);

    return result;
}

#endif  // LLAMA_USE_OCI
