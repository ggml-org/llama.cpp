#include "agent-session.h"
#include "../skills/skills-manager.h"
#include "../agents-md/agents-md-manager.h"

#include <cstdlib>
#include <iomanip>
#include <sstream>

// Get the user config directory for llama-agent
static std::string get_config_dir() {
#ifdef _WIN32
    const char * appdata = std::getenv("APPDATA");
    if (appdata) {
        return std::string(appdata) + "\\llama-agent";
    }
    return "";
#else
    const char * home = std::getenv("HOME");
    if (home) {
        return std::string(home) + "/.llama-agent";
    }
    return "";
#endif
}

// agent_session implementation

agent_session::agent_session(const std::string & id,
                             server_context & server_ctx,
                             const common_params & params,
                             const agent_session_config & config)
    : id_(id)
    , server_ctx_(server_ctx)
    , params_(params)
    , config_(config)
    , created_at_(std::chrono::steady_clock::now())
    , last_activity_(created_at_) {

    // Set up permission manager
    if (!config_.working_dir.empty()) {
        permissions_.set_project_root(config_.working_dir);
    }
    permissions_.set_yolo_mode(config_.yolo_mode);

    std::string config_dir = get_config_dir();

    // Discover Skills (agentskills.io spec)
    if (config_.enable_skills) {
        skills_manager skills_mgr;
        std::vector<std::string> skill_paths;

        // Project-local skills (highest priority)
        // Default to "." if working_dir not set, matching CLI behavior
        std::string skills_working_dir = config_.working_dir.empty() ? "." : config_.working_dir;
        skill_paths.push_back(skills_working_dir + "/.llama-agent/skills");
        skill_paths.push_back(skills_working_dir + "/.agent/skills");

        // User-global skills
        if (!config_dir.empty()) {
            skill_paths.push_back(config_dir + "/skills");
        }

        // Extra paths from config
        for (const auto & path : config_.extra_skills_paths) {
            skill_paths.push_back(path);
        }

        skills_mgr.discover(skill_paths);
        skills_prompt_section_ = skills_mgr.generate_prompt_section();
    }

    // Discover AGENTS.md files (agents.md spec)
    if (config_.enable_agents_md) {
        agents_md_manager agents_md_mgr;
        std::string working_dir = config_.working_dir.empty() ? "." : config_.working_dir;
        agents_md_mgr.discover(working_dir, config_dir);
        agents_md_prompt_section_ = agents_md_mgr.generate_prompt_section();
    }
}

agent_session::~agent_session() {
    cancel();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

agent_session_info agent_session::info() const {
    agent_session_info info;
    info.id = id_;
    info.state = state_.load();
    info.created_at = created_at_;
    info.last_activity = last_activity_;
    info.message_count = loop_ ? static_cast<int>(loop_->get_messages().size()) : 0;
    info.stats = loop_ ? loop_->get_stats() : session_stats{};
    return info;
}

void agent_session::send_message(const std::string & content,
                                  agent_event_callback on_event) {
    // Wait for any previous operation to complete
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    last_activity_ = std::chrono::steady_clock::now();
    is_running_.store(true);
    is_interrupted_.store(false);
    state_.store(agent_session_state::RUNNING);

    // Create agent_loop if it doesn't exist
    if (!loop_) {
        agent_config agent_cfg;
        agent_cfg.max_iterations = config_.max_iterations;
        agent_cfg.tool_timeout_ms = config_.tool_timeout_ms;
        agent_cfg.working_dir = config_.working_dir;
        agent_cfg.yolo_mode = config_.yolo_mode;

        // Skills configuration
        agent_cfg.enable_skills = config_.enable_skills;
        agent_cfg.skills_search_paths = config_.extra_skills_paths;
        agent_cfg.skills_prompt_section = skills_prompt_section_;

        // AGENTS.md configuration
        agent_cfg.enable_agents_md = config_.enable_agents_md;
        agent_cfg.agents_md_prompt_section = agents_md_prompt_section_;

        loop_ = std::make_unique<agent_loop>(
            server_ctx_,
            params_,
            agent_cfg,
            is_interrupted_
        );
    }

    // Run in background thread
    // Pass permissions_ for async permission handling (non-blocking)
    worker_thread_ = std::thread([this, content, on_event]() {
        auto should_stop = [this]() {
            return is_interrupted_.load();
        };

        // Pass async permission manager to avoid blocking on console prompts
        agent_loop_result result = loop_->run_streaming(content, on_event, should_stop, &permissions_);

        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            last_result_ = result;
        }

        state_.store(agent_session_state::IDLE);
        is_running_.store(false);
        last_activity_ = std::chrono::steady_clock::now();
    });
}

std::optional<agent_loop_result> agent_session::get_result() {
    std::lock_guard<std::mutex> lock(result_mutex_);
    return last_result_;
}

void agent_session::cancel() {
    is_interrupted_.store(true);
}

std::vector<permission_request_async> agent_session::pending_permissions() {
    return permissions_.pending();
}

bool agent_session::respond_permission(const std::string & request_id, bool allowed, permission_scope scope) {
    bool result = permissions_.respond(request_id, allowed, scope);
    if (result && state_.load() == agent_session_state::WAITING_PERMISSION) {
        state_.store(agent_session_state::RUNNING);
    }
    return result;
}

json agent_session::get_messages() const {
    if (loop_) {
        return loop_->get_messages();
    }
    return json::array();
}

session_stats agent_session::get_stats() const {
    if (loop_) {
        return loop_->get_stats();
    }
    return session_stats{};
}

void agent_session::clear() {
    if (loop_) {
        loop_->clear();
    }
    permissions_.clear_session();
    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        last_result_.reset();
    }
}

// agent_session_manager implementation

agent_session_manager::agent_session_manager(server_context & server_ctx, const common_params & params)
    : server_ctx_(server_ctx)
    , params_(params) {
}

agent_session_manager::~agent_session_manager() {
    std::lock_guard<std::mutex> lock(mutex_);
    sessions_.clear();
}

std::string agent_session_manager::generate_session_id() {
    uint64_t counter = session_counter_.fetch_add(1);
    std::stringstream ss;
    ss << "sess_" << std::hex << std::setfill('0') << std::setw(8) << counter;
    return ss.str();
}

std::string agent_session_manager::create_session(const agent_session_config & config) {
    std::string id = generate_session_id();

    auto session = std::make_unique<agent_session>(id, server_ctx_, params_, config);

    std::lock_guard<std::mutex> lock(mutex_);
    sessions_[id] = std::move(session);

    return id;
}

agent_session * agent_session_manager::get_session(const std::string & id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sessions_.find(id);
    if (it != sessions_.end()) {
        return it->second.get();
    }
    return nullptr;
}

bool agent_session_manager::delete_session(const std::string & id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sessions_.find(id);
    if (it != sessions_.end()) {
        sessions_.erase(it);
        return true;
    }
    return false;
}

std::vector<agent_session_info> agent_session_manager::list_sessions() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<agent_session_info> result;
    result.reserve(sessions_.size());
    for (const auto & [id, session] : sessions_) {
        result.push_back(session->info());
    }
    return result;
}

size_t agent_session_manager::session_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sessions_.size();
}

void agent_session_manager::cleanup(int idle_timeout_seconds) {
    auto now = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(idle_timeout_seconds);

    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = sessions_.begin(); it != sessions_.end(); ) {
        auto info = it->second->info();
        auto idle_duration = now - info.last_activity;
        if (idle_duration > timeout && info.state == agent_session_state::IDLE) {
            it = sessions_.erase(it);
        } else {
            ++it;
        }
    }
}
