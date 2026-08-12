#pragma once

#include "server-http.h"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

struct server_skills_config {
    bool enabled = false;
    bool trust_project_skills = false;
    std::vector<std::string> providers;
};

using token_count_callback = std::function<std::size_t(const std::string &)>;

struct server_skills {
    server_http_context::handler_t handle_get;
    server_http_context::handler_t handle_post;

    server_skills(server_skills_config config, token_count_callback count_tokens);

private:
    server_skills_config config;
    token_count_callback count_tokens;
    std::filesystem::path process_cwd;
    std::filesystem::path process_home;
};
