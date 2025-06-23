// Function to load and parse the default.smarterquant.json file specifically for model loading.
// It populates a SmarterQuantConfigMap with tensor-specific quantization instructions found in the JSON.
// This configuration is later augmented/overridden by GGUF metadata for each tensor.
// - fname: Path to the JSON configuration file.
// - Returns: A SmarterQuantConfigMap. If the file cannot be opened or parsed,
//            an empty map is returned and warnings/errors are logged.
static SmarterQuantConfigMap load_smarter_quant_config_for_model(const std::string & fname) {
    SmarterQuantConfigMap config;
    std::ifstream ifs(fname);
    if (!ifs.is_open()) {
        // It's not an error if the file doesn't exist; SmarterQuant is optional.
        LLAMA_LOG_INFO("%s: Smarterquant config file '%s' not found. Continuing without it.\n", __func__, fname.c_str());
        return config;
    }

    nlohmann::json parsed_json;
    try {
        parsed_json = nlohmann::json::parse(ifs);
    } catch (const nlohmann::json::parse_error& e) {
        LLAMA_LOG_ERROR("%s: Failed to parse smarterquant config file '%s': %s\n", __func__, fname.c_str(), e.what());
        // Return empty config, effectively disabling the feature if JSON is malformed
        return config;
    }

    if (!parsed_json.is_object()) {
        LLAMA_LOG_ERROR("%s: Smarterquant config file '%s' must contain a top-level JSON object.\n", __func__, fname.c_str());
        return config;
    }

    for (auto it = parsed_json.begin(); it != parsed_json.end(); ++it) {
        const std::string& tensor_name = it.key();
        const nlohmann::json& tensor_data_json = it.value();

        if (!tensor_data_json.is_array() || tensor_data_json.size() != 2) {
            LLAMA_LOG_WARN("%s: Entry for tensor '%s' in '%s' is not a 2-element array. Skipping.\n", __func__, tensor_name.c_str(), fname.c_str());
            continue;
        }

        SmarterQuantTensorInfo tensor_info;
        tensor_info.enabled = false; // Will be set to true if GGUF metadata confirms

        try {
            const nlohmann::json& compression_types_json = tensor_data_json[0];
            if (!compression_types_json.is_array() || compression_types_json.size() != 4) {
                throw std::runtime_error("Compression types must be an array of 4 integers.");
            }
            for (const auto& type_json : compression_types_json) {
                if (!type_json.is_number_integer()) { // Ensure it's an integer before getting int8_t
                    throw std::runtime_error("Compression type element is not an integer.");
                }
                tensor_info.compression_types.push_back(type_json.get<int8_t>());
            }

            const nlohmann::json& column_permutation_json = tensor_data_json[1];
            if (!column_permutation_json.is_array()) {
                throw std::runtime_error("Column permutation must be an array.");
            }
            tensor_info.column_permutation.reserve(column_permutation_json.size());
            for (const auto& col_json : column_permutation_json) {
                 if (!col_json.is_number_integer()) { // Ensure it's an integer
                    throw std::runtime_error("Column permutation element is not an integer.");
                }
                tensor_info.column_permutation.push_back(col_json.get<int>());
            }
            config[tensor_name] = tensor_info;
        } catch (const std::exception& e) {
            LLAMA_LOG_WARN("%s: Error parsing SmarterQuant info for tensor '%s' in '%s': %s. Skipping.\n", __func__, tensor_name.c_str(), fname.c_str(), e.what());
            continue;
        }
    }
    LLAMA_LOG_INFO("%s: Successfully loaded smarterquant JSON config from '%s' for %zu tensors.\n", __func__, fname.c_str(), config.size());
    return config;
}
