#pragma once

#include <string>

struct common_params_model;

//
// download functionalities
//

struct common_hf_file_res {
    std::string repo; // repo name with ":tag" removed
    std::string ggufFile;
    std::string mmprojFile;
};

// resolve and download model from Docker registry
// return local path to downloaded model file
std::string common_docker_resolve_model(const std::string & docker);

/**
 * Allow getting the HF file from the HF repo with tag (like ollama), for example:
 * - bartowski/Llama-3.2-3B-Instruct-GGUF:q4
 * - bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
 * - bartowski/Llama-3.2-3B-Instruct-GGUF:q5_k_s
 * Tag is optional, default to "latest" (meaning it checks for Q4_K_M first, then Q4, then if not found, return the first GGUF file in repo)
 *
 * Return pair of <repo, file> (with "repo" already having tag removed)
 *
 * Note: we use the Ollama-compatible HF API, but not using the blobId. Instead, we use the special "ggufFile" field which returns the value for "hf_file". This is done to be backward-compatible with existing cache files.
 */
common_hf_file_res common_get_hf_file(
    const std::string & hf_repo_with_tag,
    const std::string & bearer_token,
    bool offline);

// returns true if download succeeded
bool common_download_model(
    const common_params_model & model,
    const std::string & bearer_token,
    bool offline);
