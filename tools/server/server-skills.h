#pragma once

#include "server-context.h"
#include "server-http.h"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct server_skills_config {
    bool enabled = false;
    bool trust_project_skills = false;
    std::vector<std::string> providers;
};

// Measures the instruction text of a skill with the direct model tokenizer.
// Returns nullopt when the direct tokenizer is unavailable (router mode,
// model unloaded, sleeping), in which case server_skills estimates
// ceil(bytes / 4) and marks the count estimated.
using token_count_callback = std::function<std::optional<server_token_count_snapshot>(const std::string &)>;

struct skill_catalog_cache; // path-free bounded LRU catalog cache (server-skills.cpp)

struct server_skills {
    server_http_context::handler_t handle_get;
    server_http_context::handler_t handle_post;

    server_skills(server_skills_config config, token_count_callback count_tokens);
    server_skills(server_skills &&) noexcept;
    server_skills & operator=(server_skills &&) noexcept;
    ~server_skills();

private:
    server_skills_config config;
    token_count_callback count_tokens;
    std::filesystem::path process_cwd;
    std::filesystem::path process_home;
    std::unique_ptr<skill_catalog_cache> catalog_cache;
};
