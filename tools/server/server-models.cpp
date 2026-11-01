#include "server-common.h"
#include "http.h"
#include "server-models.h"
#include "server-context.h"
#include "server-stream.h"

#include "build-info.h"
#include "preset.h"
#include "download.h"
#include "fit.h"
#include "hf-cache.h"
#include "http.h"
#include "subproc.h"

#include <cpp-httplib/httplib.h> // TODO: remove this once we use HTTP client from download.h
#include <optional>

#include <functional>
#include <optional>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <deque>
#include <queue>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <cstring>

#ifndef _WIN32
extern char **environ;
#endif

#define DEFAULT_STOP_TIMEOUT 10 // seconds

// note: SIGPIPE is ignored by the server
static void request_child_exit(server_subproc & proc) {
    FILE * stdin_file = proc.sproc.stdin_file();
    if (stdin_file) {
        fprintf(stdin_file, "%s\n", CMD_ROUTER_TO_CHILD_EXIT);
        fflush(stdin_file);
    }
}

// single-threaded, watching all child processes at once
struct server_monitor {
    server_monitor(server_models & models) : models(models) {
        th = std::thread([this]() { run(); });
    }

    ~server_monitor() {
        push({ cmd_t::QUIT, {}, "", 0, false });
        th.join();
    }

    // thread-safe
    // kill_after > 0: force-kill the child after that many seconds
    void watch(const std::string & name, std::shared_ptr<server_subproc> proc, server_role_type role, int port, int kill_after = 0) {
        child_t c;
        c.name = name;
        c.proc = std::move(proc);
        c.role = role;
        c.port = port;
        if (kill_after > 0) {
            c.deadline = ggml_time_ms() + (int64_t) kill_after * 1000;
        }
        if (!c.proc->has_output()) {
            SRV_ERR("failed to get stdout/stderr of child process for name=%s\n", name.c_str());
            c.eof = true;
        }
        push({ cmd_t::WATCH, std::move(c), "", 0, false });
    }

    // thread-safe
    void stop(const std::string & name, int stop_timeout, bool send_exit) {
        push({ cmd_t::STOP, {}, name, stop_timeout, send_exit });
    }

private:
    struct child_t {
        std::string name;
        std::shared_ptr<server_subproc> proc;
        server_role_type role = SERVER_ROLE_INSTANCE;
        int port = 0;
        std::string buf[SERVER_SUBPROC_STREAMS]; // partial line of each pipe
        bool eof = false;     // output closed, waiting for the process to be reaped
        int64_t deadline = 0; // force-kill time in ms, 0 when no stop is pending
    };

    struct cmd_t {
        enum { WATCH, STOP, QUIT } type;
        child_t child;
        std::string name;
        int  stop_timeout;
        bool send_exit;
    };

    void push(cmd_t && cmd) {
        {
            std::lock_guard<std::mutex> lk(mu);
            cmds.push_back(std::move(cmd));
        }
        waiter.wake();
    }

    // returns true if the loop should exit
    bool handle_commands() {
        std::deque<cmd_t> batch;
        {
            std::lock_guard<std::mutex> lk(mu);
            batch.swap(cmds);
        }
        for (auto & cmd : batch) {
            switch (cmd.type) {
                case cmd_t::WATCH:
                    children.push_back(std::move(cmd.child));
                    break;
                case cmd_t::STOP:
                    // the newest child with this name is the one the registry knows
                    for (auto it = children.rbegin(); it != children.rend(); ++it) {
                        // a measure child has the same name, skip it
                        if (it->name != cmd.name || it->role == SERVER_ROLE_MEASURE) {
                            continue;
                        }
                        if (cmd.send_exit && !it->eof) {
                            request_child_exit(*it->proc);
                        }
                        it->deadline = ggml_time_ms() + (int64_t) cmd.stop_timeout * 1000;
                        break;
                    }
                    break;
                case cmd_t::QUIT:
                    return true;
            }
        }
        return false;
    }

    // read what the child wrote, handle its commands and forward its logs, line by line
    void read_output(child_t & c) {
        for (int i = 0; i < SERVER_SUBPROC_STREAMS; i++) {
            read_stream(c, (server_subproc_stream) i);
        }
        c.eof = c.proc->output_closed();
    }

    void read_stream(child_t & c, server_subproc_stream stream) {
        char chunk[4096];
        std::string & buf = c.buf[stream];
        bool closed = false;
        while (true) {
            int n = c.proc->read_output(stream, chunk, sizeof(chunk));
            if (n < 0) {
                closed = true;
                break;
            }
            if (n == 0) {
                break;
            }
            buf.append(chunk, (size_t) n);
            size_t start = 0;
            while (true) {
                size_t nl = buf.find('\n', start);
                if (nl == std::string::npos) {
                    break;
                }
                on_line(c, stream, buf.substr(start, nl + 1 - start));
                start = nl + 1;
            }
            buf.erase(0, start);
            if (buf.size() > max_line) {
                buf.clear(); // a child that never writes a newline must not grow this without bound
            }
        }
        if (closed && !buf.empty()) {
            on_line(c, stream, buf);
            buf.clear();
        }
    }

    void on_line(child_t & c, server_subproc_stream stream, const std::string & line) {
        if (stream == SERVER_SUBPROC_STDERR) {
            LOG("[%5d] %s", c.port, line.c_str()); // forward log
        } else if (string_starts_with(line, CMD_CHILD_TO_ROUTER_STATE)) {
            LOG_DBG("[%5d] %s", c.port, line.c_str()); // prevent spamming the log
            models.handle_child_state(c.name, line);
        } else {
            SRV_WRN("[%5d] unexpected output on the command pipe: %s", c.port, line.c_str());
        }
    }

    void run() {
        while (true) {
            if (handle_commands()) {
                return;
            }

            // wait for output, a wakeup, or the next deadline;
            // a child whose output closed is polled for its exit every 50 ms
            int64_t now     = ggml_time_ms();
            int64_t timeout = -1;
            for (const auto & c : children) {
                if (c.eof) {
                    timeout = timeout < 0 ? 50 : std::min<int64_t>(timeout, 50);
                }
                if (c.deadline) {
                    int64_t d = std::max<int64_t>(0, c.deadline - now);
                    timeout = timeout < 0 ? d : std::min(timeout, d);
                }
            }
            std::vector<server_subproc *> procs;
            std::vector<child_t *>        owners;
            for (auto & c : children) {
                if (!c.eof) {
                    procs.push_back(c.proc.get());
                    owners.push_back(&c);
                }
            }
            std::vector<bool> ready;
            waiter.wait(procs, ready, timeout);
            for (size_t i = 0; i < owners.size(); i++) {
                if (ready[i]) {
                    read_output(*owners[i]);
                }
            }

            // deadlines and exits
            now = ggml_time_ms();
            for (auto it = children.begin(); it != children.end();) {
                if (it->deadline && now >= it->deadline && !it->proc->stopped.load(std::memory_order_acquire)) {
                    SRV_WRN("force-killing child process for name=%s after timeout\n", it->name.c_str());
                    it->proc->terminate();
                    it->deadline = 0;
                }
                if (it->eof && !it->proc->is_alive()) {
                    int exit_code = it->proc->join();
                    it->proc->stopped.store(true, std::memory_order_release);
                    models.on_child_exit(it->name, it->proc, it->role, exit_code);
                    SRV_INF("instance name=%s exited with status %d\n", it->name.c_str(), exit_code);
                    it = children.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }

    static constexpr size_t max_line = 1024 * 1024;

    server_models & models;
    std::mutex mu;
    std::deque<cmd_t> cmds;
    std::vector<child_t> children; // monitor thread only
    server_subproc::waiter waiter;
    std::thread th;
};

// short loopback budget for the resumable stream router to child JSON calls (probe, lookup,
// delete). distinct from params.timeout_read/write which only applies to the generation proxy
static constexpr int STREAM_LOOKUP_TIMEOUT_MS = 250;

static void unset_reserved_args(common_preset & preset, bool unset_model_args) {
    preset.unset_option("LLAMA_ARG_SSL_KEY_FILE");
    preset.unset_option("LLAMA_ARG_SSL_CERT_FILE");
    preset.unset_option("LLAMA_API_KEY");
    preset.unset_option("LLAMA_ARG_API_KEY_FILE");
    preset.unset_option("LLAMA_ARG_MODELS_DIR");
    preset.unset_option("LLAMA_ARG_MODELS_MAX");
    preset.unset_option("LLAMA_ARG_MODELS_MEMORY_MARGIN");
    preset.unset_option("LLAMA_ARG_MODELS_PRESET");
    preset.unset_option("LLAMA_ARG_MODELS_AUTOLOAD");
    preset.unset_option("LLAMA_ARG_CONNECT");
    preset.unset_option("LLAMA_ARG_CONNECT_CODE");
    if (unset_model_args) {
        preset.unset_option("LLAMA_ARG_MODEL");
        preset.unset_option("LLAMA_ARG_MMPROJ");
        preset.unset_option("LLAMA_ARG_ALIAS");
        preset.unset_option("LLAMA_ARG_HF_REPO");
        preset.unset_option("LLAMA_ARG_HF_FILE");
    }
}

static std::vector<std::string> get_environment() {
    std::vector<std::string> env;

#ifdef _WIN32
    LPWCH env_block = GetEnvironmentStringsW();
    if (!env_block) {
        return env;
    }
    for (LPWCH e = env_block; *e; e += wcslen(e) + 1) {
        env.emplace_back(wstring_to_utf8(e));
    }
    FreeEnvironmentStringsW(env_block);
#else
    if (environ == nullptr) {
        return env;
    }
    for (char ** e = environ; *e != nullptr; e++) {
        env.emplace_back(*e);
    }
#endif

    return env;
}

void server_model_meta::update_args(common_preset_context & ctx_preset, std::string bin_path) {
    // update params
    unset_reserved_args(preset, false);
    preset.set_option(ctx_preset, "LLAMA_ARG_HOST",  CHILD_ADDR);
    preset.set_option(ctx_preset, "LLAMA_ARG_PORT",  std::to_string(port));
    preset.set_option(ctx_preset, "LLAMA_ARG_ALIAS", name);
    // the child output goes through the router to its terminal, so it follows the router colors
    preset.set_option(ctx_preset, "LLAMA_ARG_LOG_COLORS", common_log_get_colors(common_log_main()) ? "on" : "off");
    // TODO: maybe validate preset before rendering ?
    // render args
    args = preset.to_args(bin_path);

    // unified binary dispatches by subcommand, re-inject it right after the
    // binary path so the child starts as 'llama serve ...' not 'llama ...'
    const char * app_cmd = std::getenv("LLAMA_APP_CMD");
    if (app_cmd != nullptr && app_cmd[0] != '\0' && !bin_path.empty()) {
        args.insert(args.begin() + 1, app_cmd);
    }
}

void server_model_meta::update_caps(const common_params & base) {
    // reset to the default so a failed refresh cannot keep old values
    architecture = server_model_architecture_json(false, false, false, {"text"});
    n_ctx_train  = 0;

    // resolve the model file offline; do not download
    common_params params;
    params.model = base.model;
    // --no-mmproj applies to child models and blocks auto-attached projectors
    params.no_mmproj = base.no_mmproj;
    try {
        preset.apply_to_params(params, {
            "LLAMA_ARG_MODEL",
            "LLAMA_ARG_MODEL_URL",
            "LLAMA_ARG_MMPROJ",
            "LLAMA_ARG_MMPROJ_URL",
            "LLAMA_ARG_MMPROJ_AUTO",
            "LLAMA_ARG_HF_REPO",
            "LLAMA_ARG_HF_FILE",
        });
        params.offline = true;
        common_models_handler handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
        common_models_handler_apply(handler, params);
    } catch (const std::exception & e) {
        LOG_WRN("failed to resolve the model of '%s': %s\n", name.c_str(), e.what());
        return;
    }

    bool inp_image = false;
    bool inp_audio = false;
    bool gen_audio = false;
    try {
        if (!params.no_mmproj && !params.mmproj.path.empty()) {
            mtmd_caps caps = mtmd_get_cap_from_file(params.mmproj.path.c_str());
            inp_image = caps.inp_vision;
            inp_audio = caps.inp_audio;
            gen_audio = caps.gen_audio;
        }
    } catch (const std::exception & e) {
        LOG_WRN("failed to read the multimodal capabilities of '%s': %s\n", name.c_str(), e.what());
        // keep the output modalities from the GGUF metadata
    }

    // read the output modalities and the trained context from the GGUF metadata
    std::vector<std::string> output_modalities = {"text"};
    if (!params.model.path.empty()) {
        const common_gguf_info info = common_get_gguf_info(params.model.path);
        // same rule as mtmd_helper_get_capabilities()
        const bool can_chat = !gen_audio || info.has_chat_tmpl;
        output_modalities = server_model_output_modalities(info.decision_type, can_chat, gen_audio);
        n_ctx_train       = info.n_ctx_train;
    }

    // offline discovery cannot see video; a loaded model reports it
    architecture = server_model_architecture_json(inp_image, inp_audio, false, output_modalities);
}

//
// server_models
//

server_models::server_models(
        const common_params & params,
        int argc,
        char ** argv)
            : ctx_preset(LLAMA_EXAMPLE_SERVER),
              base_params(params),
              base_env(get_environment()),
              base_preset(ctx_preset.load_from_args(argc, argv)),
              sched(std::make_unique<server_models_sched>(*this)),
              monitor(std::make_unique<server_monitor>(*this)) {
    // propagate base params to child
    unset_reserved_args(base_preset, true);

    // do not propagate these options, but allow preset to explicitly set them
    base_preset.unset_option("LLAMA_ARG_LOG_FILE");

    // set binary path
    try {
        bin_path = get_server_exec_path().string();
    } catch (const std::exception & e) {
        bin_path = argv[0];
        SRV_WRN("failed to get server executable path: %s\n", e.what());
        SRV_WRN("using original argv[0] as fallback: %s\n", argv[0]);
    }

    unfinished_downloads.load();
    load_models();
    debug_fake_timing = !common_get_env("LLAMA_SERVER_DEBUG_FAKE_TIMING").empty();
}

server_models::~server_models() = default;

void server_models::instance_t::request_exit() const {
    request_child_exit(*subproc);
}

void server_models::add_model(server_model_meta && meta) {
    if (mapping.find(meta.name) != mapping.end()) {
        throw std::runtime_error(string_format("model '%s' appears multiple times", meta.name.c_str()));
    }

    // check model name does not conflict with existing aliases
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(meta.name)) {
            throw std::runtime_error(string_format("model name '%s' conflicts with alias of model '%s'",
                meta.name.c_str(), key.c_str()));
        }
    }

    // parse aliases from preset's --alias option (comma-separated)
    std::string alias_str;
    if (meta.preset.get_option("LLAMA_ARG_ALIAS", alias_str) && !alias_str.empty()) {
        for (auto & alias : string_split<std::string>(alias_str, ',')) {
            alias = string_strip(alias);
            if (!alias.empty()) {
                meta.aliases.insert(alias);
            }
        }
    }

    // parse tags from preset's --tags option (comma-separated)
    std::string tags_str;
    if (meta.preset.get_option("LLAMA_ARG_TAGS", tags_str) && !tags_str.empty()) {
        for (auto & tag : string_split<std::string>(tags_str, ',')) {
            tag = string_strip(tag);
            if (!tag.empty()) {
                meta.tags.insert(tag);
            }
        }
    }

    // validate aliases do not conflict with existing names or aliases
    for (const auto & alias : meta.aliases) {
        if (mapping.find(alias) != mapping.end()) {
            throw std::runtime_error(string_format("alias '%s' for model '%s' conflicts with existing model name",
                alias.c_str(), meta.name.c_str()));
        }
        for (const auto & [key, inst] : mapping) {
            if (inst.meta.aliases.count(alias)) {
                throw std::runtime_error(string_format("alias '%s' for model '%s' conflicts with alias of model '%s'",
                    alias.c_str(), meta.name.c_str(), key.c_str()));
            }
        }
    }

    meta.update_args(ctx_preset, bin_path); // render args
    meta.update_caps(base_params);
    std::string name = meta.name;
    mapping[name] = instance_t{
        /* subproc */ std::make_shared<server_subproc>(),
        /* meta    */ std::move(meta)
    };
}

static bool has_sidecar(const std::vector<std::string> & sidecars, const std::string & kind) {
    return std::find(sidecars.begin(), sidecars.end(), kind) != sidecars.end();
}

// reject a sidecar kind no download can pull
static void check_sidecar(const std::string & kind) {
    std::vector<std::string> kinds;
    for (const auto & s : common_download_sidecars()) {
        kinds.emplace_back(s.name);
    }
    if (!has_sidecar(kinds, kind)) {
        throw std::invalid_argument("unknown sidecar '" + kind + "', expected one of " + string_join(kinds, ", "));
    }
}

// the sidecars a download asks for; by default the ones -hf pulls without a draft type
static std::vector<std::string> parse_sidecars(const json & body) {
    std::vector<std::string> defaults;
    for (const auto & s : common_download_sidecars()) {
        if (!s.draft) {
            defaults.emplace_back(s.name);
        }
    }
    std::vector<std::string> sidecars = json_value(body, "sidecars", defaults);
    for (const auto & kind : sidecars) {
        check_sidecar(kind);
    }
    return sidecars;
}

// the download options of a model with its sidecars, the plan a removal purges partials from
static common_download_opts sidecar_download_opts(const common_params & params, const std::vector<std::string> & sidecars) {
    common_download_opts opts;
    opts.bearer_token = params.hf_token;
    for (const auto & kind : sidecars) {
        common_download_opts_add_sidecar(opts, kind);
    }
    return opts;
}

// the HF repo a model resolves from: a preset names its own, a cached model is
// listed under it, and a local file or a models dir entry has none
static std::string model_hf_repo(const server_model_meta & meta) {
    std::string hf_repo;
    if (meta.preset.get_option("LLAMA_ARG_HF_REPO", hf_repo) && !hf_repo.empty()) {
        return hf_repo;
    }
    return meta.source == SERVER_MODEL_SOURCE_CACHE ? meta.name : "";
}

// the sidecar kinds of a HF repo present on disk, as the offline plan resolves them
static json cached_sidecars(const std::string & hf_repo) {
    common_params_model model;
    model.hf_repo = hf_repo;

    common_download_opts opts;
    opts.offline = true;
    for (const auto & s : common_download_sidecars()) {
        opts.*s.opt = true;
    }

    return common_download_plan_sidecars(common_download_get_hf_plan(model, opts));
}

// replace any inherited value of `key` in a child environment
static void set_child_env(std::vector<std::string> & env, const std::string & key, const std::string & value) {
    env.erase(std::remove_if(env.begin(), env.end(), [&](const std::string & e) {
        return e.rfind(key + "=", 0) == 0;
    }), env.end());
    env.push_back(key + "=" + value);
}

// where the unfinished downloads are remembered, inside the model cache
static std::string unfinished_downloads_path() {
    return hf_cache::get_cache_path() + "/unfinished-downloads.json";
}

void server_unfinished_downloads::load() {
    std::ifstream in(unfinished_downloads_path());
    if (!in) {
        return;
    }
    std::string raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::lock_guard<std::mutex> lk(mutex);
    try {
        json data = json::parse(raw);
        for (const auto & d : data.value("downloads", json::array())) {
            server_unfinished_download download;
            download.model    = json_value(d, "model", std::string());
            download.sidecars = json_value(d, "sidecars", std::vector<std::string>());
            for (const auto & p : d.value("partials", json::array())) {
                download.partials.push_back({ json_value(p, "kind", std::string()), json_value(p, "path", std::string()) });
            }
            if (!download.model.empty()) {
                downloads.push_back(std::move(download));
            }
        }
    } catch (const std::exception & e) {
        LOG_WRN("failed to read unfinished downloads: %s\n", e.what());
    }
}

void server_unfinished_downloads::save() {
    json data;
    data["downloads"] = json::array();
    for (const auto & d : downloads) {
        json partials = json::array();
        for (const auto & p : d.partials) {
            partials.push_back({ {"kind", p.kind}, {"path", p.path} });
        }
        data["downloads"].push_back({ {"model", d.model}, {"sidecars", d.sidecars}, {"partials", partials} });
    }
    try {
        fs_write_atomic(std::filesystem::u8path(unfinished_downloads_path()), data.dump());
    } catch (const std::exception & e) {
        LOG_WRN("failed to record unfinished downloads: %s\n", e.what());
    }
}

void server_unfinished_downloads::delete_partials(const std::vector<common_download_partial> & parts) {
    for (const auto & p : parts) {
        const bool shared = std::any_of(downloads.begin(), downloads.end(), [&](const server_unfinished_download & d) {
            return std::any_of(d.partials.begin(), d.partials.end(), [&](const common_download_partial & q) {
                return q.path == p.path;
            });
        });
        if (shared) {
            continue;
        }
        std::error_code ec;
        std::filesystem::remove(std::filesystem::u8path(p.path), ec);
    }
}

void server_unfinished_downloads::add(const server_unfinished_download & download) {
    std::lock_guard<std::mutex> lk(mutex);
    std::vector<common_download_partial> stale;
    auto it = std::find_if(downloads.begin(), downloads.end(), [&](const auto & d) { return d.model == download.model; });
    if (it != downloads.end()) {
        stale = std::move(it->partials);
        *it = download;
    } else {
        downloads.push_back(download);
    }
    save();
    delete_partials(stale);
}

void server_unfinished_downloads::forget(const std::string & model) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = std::find_if(downloads.begin(), downloads.end(), [&](const auto & d) { return d.model == model; });
    if (it != downloads.end()) {
        downloads.erase(it);
        save();
    }
}

bool server_unfinished_downloads::remove(const std::string & model) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = std::find_if(downloads.begin(), downloads.end(), [&](const auto & d) { return d.model == model; });
    if (it == downloads.end()) {
        return false;
    }
    auto partials = std::move(it->partials);
    downloads.erase(it);
    save();
    delete_partials(partials);
    return true;
}

bool server_unfinished_downloads::remove_sidecar(const std::string & model, const std::string & kind) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = std::find_if(downloads.begin(), downloads.end(), [&](const auto & d) { return d.model == model; });
    if (it == downloads.end() || !has_sidecar(it->sidecars, kind)) {
        return false;
    }
    it->sidecars.erase(std::remove(it->sidecars.begin(), it->sidecars.end(), kind), it->sidecars.end());

    std::vector<common_download_partial> dropped;
    auto & partials = it->partials;
    auto pos = std::stable_partition(partials.begin(), partials.end(), [&](const common_download_partial & p) {
        return p.kind != kind;
    });
    dropped.assign(pos, partials.end());
    partials.erase(pos, partials.end());

    save();
    delete_partials(dropped);
    return true;
}

std::vector<server_unfinished_download> server_unfinished_downloads::get_all() {
    std::lock_guard<std::mutex> lk(mutex);
    return downloads;
}

std::vector<server_unfinished_download> server_models::unfinished_downloads_get() {
    return unfinished_downloads.get_all();
}

void server_models::unfinished_downloads_remove(const std::string & name) {
    // under the lock, so a download of the same tag cannot start while its files go
    std::lock_guard<std::mutex> lk(mutex);
    if (mapping.find(name) != mapping.end()) {
        throw std::invalid_argument("model '" + name + "' is downloading");
    }
    // a name with no record is no model: its files belong to the cached tags of the repo
    if (!unfinished_downloads.remove(name)) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    bool ok = common_download_remove(name);
    SRV_INF("removing unfinished download name=%s (%s)\n", name.c_str(), ok ? "succeeded" : "nothing cached");
}

bool server_models::remove_sidecar(const std::string & name, const std::string & kind) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end() && (it->second.meta.is_running() || it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADING)) {
        throw std::invalid_argument("model '" + name + "' is in use, unload it first");
    }
    // a sidecar that only ever landed as a partial is still something removed
    bool ok = unfinished_downloads.remove_sidecar(name, kind);
    ok = common_download_remove_sidecars(name, sidecar_download_opts(base_params, { kind })) || ok;
    SRV_INF("removing sidecar %s of model name=%s (%s)\n", kind.c_str(), name.c_str(), ok ? "succeeded" : "nothing cached");
    return ok;
}

void server_models::notify_sse(const std::string & event, const std::string & model_id, const json & data) {
    std::unique_ptr<server_task_result_router> result = std::make_unique<server_task_result_router>();
    result->data = {
        {"model", model_id},
        {"event", event},
    };
    if (!data.is_null()) {
        result->data["data"] = data;
    }
    SRV_DBG("notifying SSE clients about event '%s' for model '%s': %s\n", event.c_str(), model_id.c_str(), safe_json_to_str(result->data).c_str());
    sse.broadcast(std::move(result));
}

void server_models::load_models() {
    // Phase 1: load presets from all sources - pure I/O, no lock needed
    // 1. cached models
    common_presets cached_models = ctx_preset.load_from_cache();
    SRV_TRC("Loaded %zu cached model presets from %s\n", cached_models.size(), hf_cache::get_cache_path().c_str());
    // 2. local models from --models-dir
    common_presets local_models;
    if (!base_params.models_dir.empty()) {
        local_models = ctx_preset.load_from_models_dir(base_params.models_dir);
        SRV_TRC("Loaded %zu local model presets from %s\n", local_models.size(), base_params.models_dir.c_str());
    }
    // 3. custom-path models from presets
    common_preset global = {};
    common_presets custom_presets = {};
    if (!base_params.models_preset.empty()) {
        custom_presets = ctx_preset.load_from_ini(base_params.models_preset, global);
        SRV_TRC("Loaded %zu custom model presets from %s\n", custom_presets.size(), base_params.models_preset.c_str());
    }

    // cascade, apply global preset first
    cached_models  = ctx_preset.cascade(global, cached_models);
    local_models   = ctx_preset.cascade(global, local_models);
    custom_presets = ctx_preset.cascade(global, custom_presets);

    // note: if a model exists in both cached and local, local takes precedence
    common_presets final_presets;
    std::unordered_map<std::string, server_model_source> source_map;
    for (const auto & [name, preset] : cached_models) {
        final_presets[name] = preset;
        source_map[name] = SERVER_MODEL_SOURCE_CACHE;
    }
    for (const auto & [name, preset] : local_models)  {
        final_presets[name] = preset;
        source_map[name] = SERVER_MODEL_SOURCE_MODELS_DIR;
    }
    for (const auto & [name, custom] : custom_presets) {
        if (final_presets.find(name) != final_presets.end()) {
            final_presets[name].merge(custom);
        } else {
            final_presets[name] = custom;
        }
        source_map[name] = SERVER_MODEL_SOURCE_PRESET;
    }

    // overlay router's own CLI args on top of every model preset so that
    // e.g. `llama-server --temp 0` is honoured by all child processes
    for (auto & [name, preset] : final_presets) {
        preset.merge(base_preset);
    }

    auto get_source = [&](const std::string & name) {
        return source_map.count(name) ? source_map.at(name) : SERVER_MODEL_SOURCE_PRESET;
    };

    // hide cache models whose resolved file is already used by a preset with dedup-cache-models enabled
    std::set<std::string> hidden_models;
    {
        std::set<std::string> preset_paths;
        auto add_hf_path = [&preset_paths](const common_preset & preset, const char * repo_key, const char * file_key) {
            std::string hf_repo;
            if (!preset.get_option(repo_key, hf_repo) || hf_repo.empty()) {
                return;
            }
            std::string hf_file;
            preset.get_option(file_key, hf_file);
            std::string path = common_download_resolve_path(hf_repo, hf_file);
            if (!path.empty()) {
                preset_paths.insert(path);
            }
        };
        for (const auto & [name, preset] : custom_presets) {
            std::string val;
            if (!preset.get_option(COMMON_ARG_PRESET_DEDUP_CACHE_MODELS, val) || !common_arg_utils::is_truthy(val)) {
                continue;
            }
            add_hf_path(preset, "LLAMA_ARG_HF_REPO", "LLAMA_ARG_HF_FILE");
            add_hf_path(preset, "LLAMA_ARG_SPEC_DRAFT_HF_REPO", "LLAMA_ARG_SPEC_DRAFT_MODEL");
        }
        if (!preset_paths.empty()) {
            for (const auto & [name, preset] : cached_models) {
                if (get_source(name) != SERVER_MODEL_SOURCE_CACHE) {
                    continue; // merged with another source, not a pure cache entry
                }
                std::string path = common_download_resolve_path(name);
                if (!path.empty() && preset_paths.count(path)) {
                    SRV_INF("hiding cache model name=%s (deduplicated by a preset)\n", name.c_str());
                    hidden_models.insert(name);
                }
            }
        }
    }

    // Helpers that read `mapping` - must be called while holding the lock.
    auto join_set = [](const std::set<std::string> & s) {
        std::string result;
        for (const auto & v : s) {
            if (!result.empty()) result += ", ";
            result += v;
        }
        return result;
    };
    auto log_available_models = [&]() {
        SRV_INF("Available models (%zu):\n", mapping.size());
        if (mapping.empty()) {
            SRV_INF("%s", "  no models found on the system (visit https://llama.app/models for suggestions)\n");
        } else {
            for (const auto & [name, inst] : mapping) {
                const std::string source = server_model_source_to_string(inst.meta.source);

                std::string info;
                if (!inst.meta.aliases.empty()) info += " (aliases: " + join_set(inst.meta.aliases) + ")";
                if (!inst.meta.tags.empty())    info += " [tags: "    + join_set(inst.meta.tags)    + "]";

                SRV_INF("  [%10s] %s%s\n", source.c_str(), name.c_str(), info.c_str());
            }
        }
    };
    auto apply_stop_timeout = [&]() {
        for (auto & [name, inst] : mapping) {
            std::string val;
            if (inst.meta.preset.get_option(COMMON_ARG_PRESET_STOP_TIMEOUT, val)) {
                try {
                    inst.meta.stop_timeout = std::stoi(val);
                } catch (...) {
                    SRV_WRN("invalid stop-timeout value '%s' for model '%s', using default %d seconds\n",
                        val.c_str(), name.c_str(), DEFAULT_STOP_TIMEOUT);
                    inst.meta.stop_timeout = DEFAULT_STOP_TIMEOUT;
                }
            }
        }
    };
    auto apply_hidden = [&]() {
        for (auto & [name, inst] : mapping) {
            inst.meta.hidden = hidden_models.count(name) > 0;
        }
    };
    // update_args() injects HOST/PORT/ALIAS/LOG_COLORS, so strip them before comparing presets
    auto preset_options_for_compare = [](common_preset p) {
        p.unset_option("LLAMA_ARG_HOST");
        p.unset_option("LLAMA_ARG_PORT");
        p.unset_option("LLAMA_ARG_ALIAS");
        p.unset_option("LLAMA_ARG_LOG_COLORS");
        return p.options;
    };

    // Phase 2: acquire the lock once for all mapping mutations.
    // We temporarily release it only when calling functions that acquire it internally (unload)
    std::unique_lock<std::mutex> lk(mutex);

    need_reload = false;
    bool is_first_load = mapping.empty();

    if (is_first_load) {
        // FIRST LOAD: add all models, then unlock for autoloading
        for (const auto & [name, preset] : final_presets) {
            server_model_meta meta{
                /* source        */ get_source(name),
                /* preset        */ preset,
                /* name          */ name,
                /* aliases       */ {},
                /* tags          */ {},
                /* port          */ 0,
                /* status        */ SERVER_MODEL_STATUS_UNLOADED,
                /* last_used     */ 0,
                /* args          */ std::vector<std::string>(),
                /* loaded_info   */ {},
                /* progress      */ {},
                /* exit_code     */ 0,
                /* stop_timeout  */ DEFAULT_STOP_TIMEOUT,
                // /* need_download */ false,
            };
            add_model(std::move(meta));
        }
        apply_stop_timeout();
        apply_hidden();
        log_available_models();
        sched->measure_unmeasured(lk);

        // skipped on reload, see startup_models
        if (startup_models.has_value()) {
            std::vector<std::string> models_to_load;
            for (const auto & [name, inst] : mapping) {
                std::string val;
                if (inst.meta.preset.get_option(COMMON_ARG_PRESET_LOAD_ON_STARTUP, val) && common_arg_utils::is_truthy(val)) {
                    models_to_load.push_back(name);
                }
            }
            if ((int)models_to_load.size() > base_params.models_max) {
                throw std::runtime_error(string_format(
                    "number of models to load on startup (%zu) exceeds models_max (%d)",
                    models_to_load.size(), base_params.models_max));
            }

            // to be lazy-loaded after main() setup phase is completed
            startup_models = std::move(models_to_load);
        }

        lk.unlock();
    } else {
        // RELOAD: diff the new preset list against the current mapping and reconcile
        is_reloading = true;

        // find running models whose source was removed or whose preset changed
        std::vector<std::string> to_unload;
        for (const auto & [name, inst] : mapping) {
            if (!inst.meta.is_running()) continue;
            auto it = final_presets.find(name);
            if (it == final_presets.end()) {
                to_unload.push_back(name); // removed from source
            } else if (preset_options_for_compare(inst.meta.preset) != preset_options_for_compare(it->second)) {
                to_unload.push_back(name); // preset changed
            }
        }

        // unload() acquires the lock internally, so release before each call
        for (const auto & name : to_unload) {
            SRV_INF("(reload) unloading model name=%s (source updated or removed)\n", name.c_str());
            lk.unlock();
            unload(name);
            lk.lock();
        }

        // wait for all targeted models to reach UNLOADED; cv.wait handles unlock/relock
        cv.wait(lk, [&]() {
            for (const auto & name : to_unload) {
                auto it = mapping.find(name);
                if (it != mapping.end() && it->second.meta.is_running()) return false;
            }
            return true;
        });

        // erase models no longer in any source
        for (auto it = mapping.begin(); it != mapping.end(); ) {
            if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
                ++it; // download thread is still busy, skip
            } else if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADED) {
                // download finished, safe to erase
                it = mapping.erase(it);
            } else if (final_presets.find(it->first) == final_presets.end()) {
                SRV_INF("(reload) removing model name=%s (no longer in source)\n", it->first.c_str());
                it = mapping.erase(it);
            } else {
                ++it;
            }
        }

        // update presets for non-running models still in source
        for (auto & [name, inst] : mapping) {
            if (inst.meta.is_running()) continue;
            auto it = final_presets.find(name);
            if (it == final_presets.end()) continue; // erased above

            if (preset_options_for_compare(inst.meta.preset) != preset_options_for_compare(it->second)) {
                // different args, different memory: measure again
                inst.meta.memory = {};
            }
            inst.meta.preset = it->second;

            // re-parse aliases, then validate against other models
            std::set<std::string> new_aliases;
            std::string alias_str;
            if (inst.meta.preset.get_option("LLAMA_ARG_ALIAS", alias_str) && !alias_str.empty()) {
                for (auto & alias : string_split<std::string>(alias_str, ',')) {
                    alias = string_strip(alias);
                    if (!alias.empty()) new_aliases.insert(alias);
                }
            }
            inst.meta.aliases.clear();
            for (const auto & alias : new_aliases) {
                bool conflict = false;
                for (const auto & [other_name, other_inst] : mapping) {
                    if (other_name == name) continue;
                    if (other_name == alias || other_inst.meta.aliases.count(alias)) {
                        SRV_WRN("(reload) alias '%s' for model '%s' conflicts with model '%s', skipping\n",
                            alias.c_str(), name.c_str(), other_name.c_str());
                        conflict = true;
                        break;
                    }
                }
                if (!conflict) inst.meta.aliases.insert(alias);
            }

            // re-parse tags
            inst.meta.tags.clear();
            std::string tags_str;
            if (inst.meta.preset.get_option("LLAMA_ARG_TAGS", tags_str) && !tags_str.empty()) {
                for (auto & tag : string_split<std::string>(tags_str, ',')) {
                    tag = string_strip(tag);
                    if (!tag.empty()) inst.meta.tags.insert(tag);
                }
            }

            inst.meta.exit_code = 0; // clear failed state so the model can be reloaded
            inst.meta.update_args(ctx_preset, bin_path);
            inst.meta.update_caps(base_params);
        }

        // add models that are new in this reload, load-on-startup is not honored here since a
        // reload never spawns an instance
        for (const auto & [name, preset] : final_presets) {
            if (mapping.find(name) == mapping.end()) {
                server_model_meta meta{
                    /* source        */ get_source(name),
                    /* preset        */ preset,
                    /* name          */ name,
                    /* aliases       */ {},
                    /* tags          */ {},
                    /* port          */ 0,
                    /* status        */ SERVER_MODEL_STATUS_UNLOADED,
                    /* last_used     */ 0,
                    /* args          */ std::vector<std::string>(),
                    /* loaded_info   */ {},
                    /* progress      */ {},
                    /* exit_code     */ 0,
                    /* stop_timeout  */ DEFAULT_STOP_TIMEOUT,
                    // /* need_download */ false,
                };
                add_model(std::move(meta));
            }
        }

        apply_stop_timeout();
        apply_hidden();
        sched->measure_unmeasured(lk);

        // clear reload flag under the lock, this releases the load() calls waiting on !is_reloading
        is_reloading = false;
        cv.notify_all();

        log_available_models();

        lk.unlock();

        notify_sse("models_reload", "*");
    }
}

void server_models::load_startup_models() {
    std::vector<std::string> to_load;
    {
        std::lock_guard<std::mutex> lk(mutex);
        if (!startup_models.has_value()) {
            return; // already drained
        }
        to_load = std::move(*startup_models);
        startup_models.reset();
    }
    for (const auto & name : to_load) {
        SRV_INF("(startup) loading model %s\n", name.c_str());
        load(name);
    }
}

void server_models::update_meta(const std::string & name, const server_model_meta & meta) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        it->second.meta = meta;
    }
    cv.notify_all(); // notify wait_until_loading_finished
}

bool server_models::has_model(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    if (mapping.find(name) != mapping.end()) {
        return true;
    }
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(name)) {
            return true;
        }
    }
    return false;
}

std::optional<server_model_meta> server_models::get_meta(const std::string & name) {
    std::unique_lock<std::mutex> lk(mutex);
    if (need_reload) {
        lk.unlock();
        load_models();
        lk.lock();
    }

    auto it = mapping.find(name);
    if (it != mapping.end()) {
        return it->second.meta;
    }
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(name)) {
            return inst.meta;
        }
    }
    return std::nullopt;
}

std::vector<server_model_meta> server_models::get_all_meta() {
    std::unique_lock<std::mutex> lk(mutex);
    if (need_reload) {
        lk.unlock();
        load_models();
        lk.lock();
    }

    std::vector<server_model_meta> result;
    result.reserve(mapping.size());
    for (const auto & [name, inst] : mapping) {
        result.push_back(inst.meta);
    }
    return result;
}

void server_models::load(const std::string & name) {
    load(name, load_options{});
}

void server_models::load(const std::string & name, const load_options & opts) {
    // the fake load time of a model; a download worker loads none
    if (debug_fake_timing && opts.role == SERVER_ROLE_INSTANCE) {
        // do not hold the mutex here, other requests must keep making progress
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    if (!opts.custom_meta.has_value()) {
        if (!has_model(name)) {
            throw std::runtime_error("model name=" + name + " is not found");
        }
        if (opts.role == SERVER_ROLE_INSTANCE) {
            sched->make_room(name);
        }
    }

    std::unique_lock<std::mutex> lk(mutex);
    // edge case: block until any in-progress reload has finished so we always load
    // against the freshest preset and a consistent mapping state
    cv.wait(lk, [this]() { return !is_reloading; });

    // a download over an existing entry is decided under the lock, so a load of the
    // same model cannot start in between: a paused download or an idle cached model
    // is downloaded again with its files kept, a download on its way and a running
    // model are refused, and a preset or local model is not a download
    if (opts.role == SERVER_ROLE_DOWNLOAD) {
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            const auto & cur = it->second.meta;
            if (cur.status == SERVER_MODEL_STATUS_DOWNLOADING) {
                throw std::invalid_argument("model '" + name + "' is already downloading");
            }
            if (cur.is_running()) {
                throw std::invalid_argument("model '" + name + "' is running, unload it first");
            }
            if (cur.source != SERVER_MODEL_SOURCE_CACHE) {
                throw std::invalid_argument("model '" + name + "' already exists");
            }
            SRV_INF("resuming download for model '%s'\n", name.c_str());
        }

        // two downloads writing the same partial would interleave their bytes
        for (const auto & d : unfinished_downloads.get_all()) {
            auto other = mapping.find(d.model);
            if (d.model == name || other == mapping.end() || other->second.meta.status != SERVER_MODEL_STATUS_DOWNLOADING) {
                continue;
            }
            for (const auto & p : d.partials) {
                const bool shared = std::any_of(opts.partials.begin(), opts.partials.end(), [&](const common_download_partial & q) {
                    return q.path == p.path;
                });
                if (shared) {
                    throw std::invalid_argument("model '" + name + "' shares files with the download of '" + d.model + "' on its way, retry once it is done");
                }
            }
        }
    }

    auto meta = opts.custom_meta.has_value() ? *opts.custom_meta : mapping[name].meta;
    if (meta.status != SERVER_MODEL_STATUS_UNLOADED) {
        SRV_INF("model %s is not ready\n", name.c_str());
        return;
    }

    // Re-check capacity under the lock to prevent concurrent loads from
    // exceeding models_max. Without this, the window between make_room()
    // releasing its lock and this lock_guard acquiring allows multiple
    // threads to each observe capacity and all proceed to load.
    // Download workers do not use models_max slots.
    if (opts.role == SERVER_ROLE_INSTANCE && sched->limits_exceeded(lk, name, /* verbose */ false)) {
        throw std::runtime_error("model limit reached, try again later");
    }

    // prepare new instance info
    instance_t inst;
    inst.meta             = meta;
    inst.meta.port        = common_http_get_free_port();
    inst.meta.status      = SERVER_MODEL_STATUS_LOADING;
    inst.meta.loaded_info = json{};
    inst.meta.last_used   = ggml_time_ms();

    if (inst.meta.port <= 0) {
        throw std::runtime_error("failed to get a port number");
    }

    inst.subproc = std::make_shared<server_subproc>();
    {
        SRV_INF("spawning server instance with name=%s on port %d\n", inst.meta.name.c_str(), inst.meta.port);

        inst.meta.update_args(ctx_preset, bin_path); // render args

        std::vector<std::string> child_args = inst.meta.args; // copy
        std::vector<std::string> child_env  = base_env; // copy
        child_env.push_back(std::string(SERVER_ENV_ROUTER_PORT) + "=" + std::to_string(base_params.port));

        if (opts.role == SERVER_ROLE_DOWNLOAD) {
            inst.meta.status = SERVER_MODEL_STATUS_DOWNLOADING;
            child_env.push_back(std::string(SERVER_ENV_CHILD_ROLE) + "=" + server_role_type_to_str(SERVER_ROLE_DOWNLOAD));
            child_env.push_back("LLAMA_ARG_HF_REPO=" + name);

            // the sidecars as -hf would pull them: the draft types, and the mmproj switch
            // for the one that is not a draft
            bool        mmproj = false;
            std::string spec_types;
            for (const auto & s : common_download_sidecars()) {
                if (!has_sidecar(opts.sidecars, s.name)) {
                    continue;
                }
                if (s.draft) {
                    spec_types += (spec_types.empty() ? "" : ",") + std::string("draft-") + s.name;
                } else {
                    mmproj = true;
                }
            }
            set_child_env(child_env, "LLAMA_ARG_MMPROJ_AUTO", mmproj ? "1" : "0");
            set_child_env(child_env, "LLAMA_ARG_SPEC_TYPE", spec_types.empty() ? "none" : spec_types);
        }

        SRV_INF("%s", "spawning server instance with args:\n");
        for (const auto & arg : child_args) {
            SRV_INF("  %s\n", arg.c_str());
        }
        inst.meta.args = child_args; // save for debugging

        // the child writes its commands to stdout and its logs to stderr
        int options = subprocess_option_no_window;
        if (!inst.subproc->sproc.create(child_args, options, child_env)) {
            throw std::runtime_error("failed to spawn server instance");
        }
    }

    // remember the download: the partial files carry a content hash, not the name
    if (opts.role == SERVER_ROLE_DOWNLOAD) {
        unfinished_downloads.add({ name, opts.sidecars, opts.partials });
    }

    // old process should have exited already, but just in case, we clean it up here
    {
        auto it = mapping.find(name);
        if (it != mapping.end() && it->second.subproc && it->second.subproc->is_alive()) {
            SRV_WRN("old process for model name=%s is still alive, this is unexpected\n", name.c_str());
            it->second.subproc->terminate(); // force kill
        }
    }

    notify_sse("model_status", name, {
        {"status", server_model_status_to_string(inst.meta.status)},
    });

    auto proc = inst.subproc;
    int  port = inst.meta.port;
    mapping[name] = std::move(inst);
    if (opts.role == SERVER_ROLE_INSTANCE) {
        sched->on_spawned(lk, name);
    }
    monitor->watch(name, proc, opts.role, port);
    cv.notify_all();
}

void server_models::request_stop(const std::string & name, bool send_exit) {
    auto it = mapping.find(name);
    if (it == mapping.end() || stopping_models.count(name)) {
        return;
    }
    stopping_models.insert(name);
    monitor->stop(name, it->second.meta.stop_timeout, send_exit);
}

void server_models::watch_child(const std::string & name, std::shared_ptr<server_subproc> proc, server_role_type role, int port, int kill_after) {
    monitor->watch(name, std::move(proc), role, port, kill_after);
}

void server_models::on_child_exit(const std::string & name, const std::shared_ptr<server_subproc> & proc, server_role_type role, int exit_code) {
    if (role == SERVER_ROLE_MEASURE) {
        sched->on_measure_exit(name, proc, exit_code);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it == mapping.end() || it->second.subproc != proc) {
            stopping_models.erase(name);
            return; // entry erased, or a newer instance took the name
        }
    }
    if (role == SERVER_ROLE_DOWNLOAD) {
        // instance will be cleaned up on next load_models() call
        std::lock_guard<std::mutex> lk(mutex);
        stopping_models.erase(name);
        cv.notify_all();
    } else {
        update_status(name, {
            SERVER_MODEL_STATUS_UNLOADED,
            exit_code
        });
    }
}

void server_models::unload(const std::string & name) {
    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
            SRV_INF("cancelling download for model name=%s\n", name.c_str());
            it->second.request_exit();
            // for convenience, we wait the status change here
            wait(lk, name, [](const server_model_meta & new_meta) {
                return new_meta.status != SERVER_MODEL_STATUS_DOWNLOADING;
            });
        } else if (it->second.meta.is_running()) {
            SRV_INF("stopping model instance name=%s\n", name.c_str());
            bool loading = it->second.meta.status == SERVER_MODEL_STATUS_LOADING;
            if (loading) {
                // special case: if model is in loading state, unloading means force-killing it
                SRV_WRN("model name=%s is still loading, force-killing\n", name.c_str());
                it->second.subproc->terminate();
            }
            request_stop(name, !loading);
            // status change will be handled by the monitor
        } else {
            SRV_WRN("model instance name=%s is not running\n", name.c_str());
        }
    }
}

void server_models::unload_all() {
    std::unique_lock<std::mutex> lk(mutex);
    sched->cancel_measurements(lk);
    for (auto & [name, inst] : mapping) {
        if (inst.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
            SRV_INF("cancelling download for model name=%s\n", name.c_str());
            inst.request_exit();
        } else if (inst.meta.is_running()) {
            SRV_INF("stopping model instance name=%s\n", name.c_str());
            bool loading = inst.meta.status == SERVER_MODEL_STATUS_LOADING;
            if (loading) {
                inst.subproc->terminate();
            }
            request_stop(name, !loading);
        }
    }
    // wait for every child to exit, the monitor force-kills the ones that ignore the exit command
    cv.wait(lk, [this, &lk]() {
        if (sched->is_measuring(lk)) {
            return false;
        }
        for (const auto & [name, inst] : mapping) {
            if (inst.meta.is_running() || inst.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
                return false;
            }
        }
        return true;
    });
}

void server_models::update_status(const std::string & name, const update_status_args & args) {
    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        auto & meta = it->second.meta;
        meta.status      = args.status;
        meta.exit_code   = args.exit_code;
        if (args.status == SERVER_MODEL_STATUS_UNLOADED) {
            stopping_models.erase(name);
        }
        if (!args.loaded_info.is_null()) {
            meta.loaded_info = args.loaded_info;
            // the child replaces both arrays in full; a bad or missing value changes nothing
            if (args.loaded_info.contains("architecture") && args.loaded_info.at("architecture").is_object()) {
                const json & child_arch = args.loaded_info.at("architecture");
                for (const char * key : { "input_modalities", "output_modalities" }) {
                    if (!child_arch.contains(key) || !child_arch.at(key).is_array()) {
                        continue;
                    }
                    std::vector<std::string> modalities;
                    bool valid = true;
                    for (const auto & m : child_arch.at(key)) {
                        if (!m.is_string()) {
                            valid = false;
                            break;
                        }
                        modalities.push_back(m.get<std::string>());
                    }
                    if (valid) {
                        meta.architecture[key] = std::move(modalities);
                    }
                }
            }
        }
        if (!args.progress.is_null()) {
            meta.progress = args.progress;
        }
        switch (args.status) {
            case SERVER_MODEL_STATUS_LOADED:   sched->on_ready(lk, name, args.memory); break;
            case SERVER_MODEL_STATUS_SLEEPING: sched->on_sleeping(lk, name);           break;
            case SERVER_MODEL_STATUS_UNLOADED: sched->on_unloaded(lk, name);           break;
            default:                                                                   break;
        }
        // a model that comes up idle or goes down changes the slot count for queued requests
        sched->tick(lk);
    }
    // broadcast status change to SSE
    {
        json data = {
            {"status", server_model_status_to_string(args.status)},
        };
        if (args.status == SERVER_MODEL_STATUS_UNLOADED) {
            data["exit_code"] = args.exit_code;
        }
        if (!args.loaded_info.is_null()) {
            data["info"] = args.loaded_info;
        }
        if (!args.progress.is_null()) {
            data["progress"] = args.progress;
        }
        if (args.memory.has_value()) {
            data["memory"] = args.memory->to_json();
        }
        // note: notify_sse doesn't acquire the lock, so no deadlock here
        notify_sse("status_change", name, data);
    }
    cv.notify_all();
}

void server_models::update_download_progress(const std::string & name, const common_download_progress & progress, bool done, bool ok) {
    json curr;
    {
        std::lock_guard<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            if (done) {
                // mark the instance to be erased on next load_models() call
                it->second.meta.status = SERVER_MODEL_STATUS_DOWNLOADED;
                need_reload = true;

                // a download that failed keeps its tag: its partial files are still
                // there and re-posting the tag resumes them
                if (ok) {
                    unfinished_downloads.forget(name);
                }
            } else {
                json & info = it->second.meta.loaded_info;
                if (!info.contains("progress")) {
                    info["progress"] = json{};
                }
                info["progress"][progress.url] = {
                    {"done",  progress.downloaded},
                    {"total", progress.total},
                };
                curr = it->second.meta.loaded_info; // copy
            }
        }
    }
    if (done) {
        cv.notify_all(); // notify in case unload() is waiting for download to be cancelled
        notify_sse(ok ? "download_finished" : "download_failed", name, {});
    } else {
        notify_sse("download_progress", name, curr);
    }
}

bool server_models::remove(const std::string & name) {
    // do everything under one lock acquisition; avoid get_meta() /
    // unload() because they can trigger load_models() which erases
    // transient DOWNLOADING / DOWNLOADED entries as a side-effect
    std::unique_lock<std::mutex> lk(mutex);

    auto it = mapping.find(name);
    if (it == mapping.end()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (it->second.meta.source != SERVER_MODEL_SOURCE_CACHE) {
        throw std::runtime_error("model name=" + name + " is not removable (not from cache)");
    }

    if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
        // cancel in-flight download
        SRV_INF("cancelling download for model name=%s\n", name.c_str());
        it->second.request_exit();
    } else if (it->second.meta.is_running()) {
        // stop running instance
        SRV_INF("stopping model instance name=%s\n", name.c_str());
        bool loading = it->second.meta.status == SERVER_MODEL_STATUS_LOADING;
        if (loading) {
            it->second.subproc->terminate();
        }
        request_stop(name, !loading);
    }

    // wait until the child is gone
    wait(lk, name, [](const server_model_meta & meta) {
        return meta.status == SERVER_MODEL_STATUS_UNLOADED
            || meta.status == SERVER_MODEL_STATUS_DOWNLOADED;
    });

    // load_models() may have erased the entry during the wait; the files go under the
    // lock, so a download of the same tag cannot start meanwhile
    mapping.erase(name);
    unfinished_downloads.remove(name);

    // best-effort: a cancelled download may have no cached files
    bool ok = common_download_remove(name);
    SRV_INF("removing model name=%s from cache (%s)\n", name.c_str(), ok ? "succeeded" : "nothing cached");
    notify_sse("model_remove", name, {});
    return true;
}

void server_models::wait(const std::string & name, std::function<bool(const server_model_meta &)> predicate) {
    std::unique_lock<std::mutex> lk(mutex);
    wait(lk, name, predicate);
}

void server_models::wait(std::unique_lock<std::mutex> & lk, const std::string & name, std::function<bool(const server_model_meta &)> predicate) {
    cv.wait(lk, [this, &name, &predicate]() {
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            return predicate(it->second.meta);

        }
        // model was removed from mapping by another code path (e.g. load_models()).
        // nothing left to wait for - tell the caller to proceed.
        return true;
    });
}

bool server_models::ensure_model_ready(const std::string & name, const std::function<bool()> & should_stop) {
    auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    bool stopping;
    {
        std::lock_guard<std::mutex> lk(mutex);
        stopping = stopping_models.count(name) > 0;
    }
    if (!stopping && meta->is_ready()) {
        return false; // ready for taking requests
    }
    if (!stopping && meta->status == SERVER_MODEL_STATUS_SLEEPING) {
        return false; // child is sleeping but still running; new request will wake it up
    }

    bool queued   = false;
    bool did_load = false;
    {
        std::unique_lock<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it != mapping.end() && it->second.meta.status == SERVER_MODEL_STATUS_UNLOADED) {
            // the queue entry protects the model from eviction until its waiters leave
            sched->join(lk, name);
            sched->tick(lk);
            queued = true;
        }
    }

    // while queued, this is also where the load happens: the head of the queue does it
    SRV_INF("waiting until model name=%s is fully loaded...\n", name.c_str());
    std::unique_lock<std::mutex> lk(mutex);
    auto leave_queue = [this, &queued, &lk, &name]() {
        if (queued) {
            sched->leave(lk, name);
            queued = false;
        }
    };

    try {
        bool saw_loading = false;
        while (true) {
            auto it = mapping.find(name);
            if (it == mapping.end()) {
                break; // removed by another code path, nothing to wait for
            }
            if (stopping_models.count(name)) {
                // a stopping instance takes no new request, the next instance serves it
                if (!queued) {
                    sched->join(lk, name);
                    sched->tick(lk);
                    queued = true;
                }
                if (should_stop && should_stop()) {
                    throw std::runtime_error("request cancelled while waiting for model name=" + name);
                }
                cv.wait_for(lk, std::chrono::milliseconds(200));
                continue;
            }
            const server_model_status status = it->second.meta.status;

            if (status == SERVER_MODEL_STATUS_LOADED || status == SERVER_MODEL_STATUS_SLEEPING) {
                break;
            }
            if (status == SERVER_MODEL_STATUS_DOWNLOADING || status == SERVER_MODEL_STATUS_DOWNLOADED) {
                break; // do not wait on a download child
            }
            if (status == SERVER_MODEL_STATUS_LOADING) {
                saw_loading = true;
            } else if (status == SERVER_MODEL_STATUS_UNLOADED) {
                if (did_load || saw_loading) {
                    // a spawn happened and the instance came back down
                    if (it->second.meta.is_failed()) {
                        throw std::runtime_error("model name=" + name + " failed to load");
                    }
                    break; // unloaded by another code path, caller reports "not running"
                }
                if (!queued) {
                    break; // not queued, and the load someone else started fell over
                }
            }

            if (should_stop && should_stop()) {
                // if a model was evicted for us, the free slot goes to the next waiter
                throw std::runtime_error("request cancelled while waiting for model name=" + name);
            }

            // our turn: our model is at the head, and a slot really did free up
            if (status == SERVER_MODEL_STATUS_UNLOADED && sched->try_claim(lk, name)) {
                lk.unlock();
                bool ok = true;
                try {
                    SRV_INF("slot available, loading queued model name=%s\n", name.c_str());
                    load(name);
                    did_load = true;
                } catch (const std::exception & e) {
                    // lost a race for the slot, stay in line and retry
                    SRV_WRN("queued load of name=%s did not go through: %s\n", name.c_str(), e.what());
                    ok = false;
                }
                lk.lock();
                sched->claim_done(lk, name, ok);
                sched->tick(lk);
                continue;
            }

            cv.wait_for(lk, std::chrono::milliseconds(200));
        }
    } catch (...) {
        leave_queue();
        sched->tick(lk); // a slot freed for this waiter goes to the next one
        throw;
    }
    leave_queue();

    return true;
}

server_http_res_ptr server_models::proxy_request(const server_http_req & req, const std::string & method, const std::string & name, bool update_last_used, bool detached) {
    auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (!meta->is_running()) {
        throw std::invalid_argument("model name=" + name + " is not running");
    }
    {
        std::unique_lock<std::mutex> lk(mutex);
        if (update_last_used) {
            mapping[name].meta.last_used = ggml_time_ms();
        }
        mapping[name].req_count++;
    }
    if (debug_fake_timing) {
        // sleep after req_count++, so the model counts as busy while we wait here
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    SRV_INF("proxying request to model %s on port %d\n", name.c_str(), meta->port);
    std::string proxy_path = req.path;
    if (!req.query_string.empty()) {
        proxy_path += '?' + req.query_string;
    }
    auto proxy = std::make_unique<server_http_proxy>(
            method,
            "http",
            CHILD_ADDR,
            meta->port,
            proxy_path,
            req.headers,
            req.body,
            req.files,
            // a detached request belongs to a replay session
            detached
                ? std::function<bool()>([]() { return false; })
                : req.should_stop,
            base_params.timeout_read,
            base_params.timeout_write
            );

    proxy->cleanup = [this, name]() {
        std::unique_lock<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it != mapping.end() && it->second.req_count > 0) {
            it->second.req_count--;
            if (it->second.req_count == 0) {
                sched->tick(lk);
            }
        }
    };

    return proxy;
}

void server_models::proxy_ws(const server_http_req & req, server_http_context::ws_conn & conn, const std::string & name) {
    auto send_error = [&conn](const std::string & message) {
        conn.send(safe_json_to_str(json {
            {"type", "error"},
            {"error", {{"type", "server_error"}, {"code", nullptr}, {"message", message}, {"param", nullptr}}},
        }));
        conn.close();
    };

    auto meta = get_meta(name);
    if (!meta.has_value() || !meta->is_running()) {
        send_error("model name=" + name + " is not running");
        return;
    }
    {
        std::unique_lock<std::mutex> lk(mutex);
        mapping[name].meta.last_used = ggml_time_ms();
        mapping[name].req_count++;
    }
    SRV_INF("proxying websocket to model %s on port %d\n", name.c_str(), meta->port);

    std::string url = std::string("ws://") + CHILD_ADDR + ":" + std::to_string(meta->port) + req.path;
    if (!req.query_string.empty()) {
        url += '?' + req.query_string;
    }
    // the client library adds its own upgrade headers
    httplib::Headers headers;
    for (const auto & [key, value] : req.headers) {
        std::string lower = key;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower == "host" || lower == "connection" || lower == "upgrade" || lower.compare(0, 14, "sec-websocket-") == 0) {
            continue;
        }
        headers.emplace(key, value);
    }

    {
        httplib::ws::WebSocketClient upstream(url, headers);
        if (!upstream.connect()) {
            send_error("failed to connect to model name=" + name);
        } else {
            // client to child
            std::thread pump([&]() {
                std::string msg;
                while (true) {
                    const int res = conn.read(msg);
                    if (res == 0 || !(res == 1 ? upstream.send(msg) : upstream.send(msg.data(), msg.size()))) {
                        break;
                    }
                }
                upstream.close();
            });
            // child to client
            std::string msg;
            while (true) {
                const auto res = upstream.read(msg);
                if (res == httplib::ws::Fail || !(res == httplib::ws::Text ? conn.send(msg) : conn.send_binary(msg))) {
                    break;
                }
            }
            conn.close();
            pump.join();
        }
    }

    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end() && it->second.req_count > 0) {
        it->second.req_count--;
        if (it->second.req_count == 0) {
            sched->tick(lk);
        }
    }
}

void server_models::handle_child_state(const std::string & name, const std::string & raw_input) {
    server_state state;
    json payload;

    try {
        json data = json::parse(raw_input.substr(strlen(CMD_CHILD_TO_ROUTER_STATE)));
        state = server_state_from_str(json_value(data, "state", std::string()));
        payload = json_value(data, "payload", json{});
    } catch (const std::exception & e) {
        SRV_ERR("failed to parse child state update for name=%s: %s\n", name.c_str(), e.what());
        return;
    }

    switch (state) {
        case SERVER_STATE_DOWNLOADING:
            {
                std::string result = json_value(payload, "result", std::string());
                std::string url    = json_value(payload, "url",    std::string());
                auto request_exit = [&]() {
                    std::lock_guard<std::mutex> lk(mutex);
                    auto it = mapping.find(name);
                    if (it != mapping.end()) {
                        return it->second.request_exit();
                    }
                };
                if (result == "download_finished") {
                    update_download_progress(name, {}, true, true);
                    request_exit();
                } else if (result == "download_failed") {
                    update_download_progress(name, {}, true, false);
                    request_exit();
                } else if (!url.empty()) {
                    common_download_progress p;
                    p.url        = url;
                    p.downloaded = json_value(payload, "downloaded", (size_t)0);
                    p.total      = json_value(payload, "total", (size_t)0);
                    update_download_progress(name, p, false);
                }
            } break;
        case SERVER_STATE_LOADING:
            {
                update_status(name, {
                    SERVER_MODEL_STATUS_LOADING,
                    0,
                    nullptr, // no loaded_info yet
                    payload,
                });
            } break;
        case SERVER_STATE_MEASURING:
            {
                sched->handle_measured(name, payload);
            } break;
        case SERVER_STATE_READY:
            {
                std::optional<server_model_memory> memory;
                json info = payload;
                if (info.contains("memory")) {
                    try {
                        memory = server_model_memory::from_json(info.at("memory"));
                    } catch (const std::exception & e) {
                        SRV_WRN("malformed memory report from model name=%s: %s\n", name.c_str(), e.what());
                    }
                    info.erase("memory");
                }
                update_status(name, {
                    SERVER_MODEL_STATUS_LOADED,
                    0,
                    // note: payload can be empty if this is a wakeup from sleep
                    info.size() > 0 ? info : nullptr,
                    {}, // reset progress info
                    memory,
                });
            } break;
        case SERVER_STATE_SLEEPING:
            {
                update_status(name, { SERVER_MODEL_STATUS_SLEEPING });
            } break;
        default:
            // should never happen, but just in case
            GGML_ASSERT(false && "unexpected state from child server");
    }
}

//
// server_models_routes
//

// RAII wrapper similar to server_response_reader, but doesn't use server_queue
static std::atomic<int> sse_client_id_counter = 0;
struct server_models_sse_client {
    server_response & queue_results;
    int client_id;
    server_models_sse_client(server_response & q)
            : queue_results(q), client_id(sse_client_id_counter.fetch_add(1, std::memory_order_relaxed)) {
        SRV_DBG("new SSE client connected, assigned client_id=%d\n", client_id);
        queue_results.add_waiting_task_id(client_id);
    }
    ~server_models_sse_client() {
        SRV_DBG("SSE client disconnected, removing client_id=%d\n", client_id);
        queue_results.remove_waiting_task_id(client_id);
    }

    // return nullptr if should_stop() is true before receiving a result
    // note: if one error is received, it will stop further processing and return error result
    server_task_result_ptr next(const std::function<bool()> & should_stop) {
        while (true) {
            static const int http_polling_seconds = 1; // check should_stop every 1 second
            server_task_result_ptr result = queue_results.recv_with_timeout({client_id}, http_polling_seconds);
            if (result == nullptr) {
                // timeout, check stop condition
                if (should_stop()) {
                    return nullptr;
                }
                // continue waiting otherwise
            } else {
                SRV_DBG("recv result for client_id=%d: %s\n", client_id, safe_json_to_str(result->to_json()).c_str());
                return result;
            }
        }
        // should not reach here
    }
};

static void res_ok(std::unique_ptr<server_http_res> & res, const json & response_data) {
    res->status = 200;
    res->data = safe_json_to_str(response_data);
}

static void res_err(std::unique_ptr<server_http_res> & res, const json & error_data) {
    res->status = json_value(error_data, "code", 500);
    res->data = safe_json_to_str({{ "error", error_data }});
}

static bool router_validate_model(std::string & name, server_models & models, bool models_autoload, std::unique_ptr<server_http_res> & res) {
    if (name.empty()) {
        res_err(res, format_error_response("model name is missing from the request", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    auto meta = models.get_meta(name);
    if (!meta.has_value()) {
        res_err(res, format_error_response(string_format("model '%s' not found", name.c_str()), ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    // resolve alias to canonical model name
    name = meta->name;
    if (!models_autoload && !meta->is_running()) {
        res_err(res, format_error_response("model is not loaded", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    return true;
}

static bool is_autoload(const common_params & params, const server_http_req & req) {
    std::string autoload = req.get_param("autoload");
    if (autoload.empty()) {
        return params.models_autoload;
    } else {
        return autoload == "true" || autoload == "1";
    }
}

// percent encode one query or path component, covers reserved chars without pulling in
// httplib::detail. used by the stream routes to forward conversation_id to children safely
static std::string encode_qs(const std::string & in) {
    std::string out;
    out.reserve(in.size() * 3);
    for (unsigned char c : in) {
        bool safe = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
                 || c == '-' || c == '_' || c == '.' || c == '~';
        if (safe) {
            out.push_back(char(c));
        } else {
            char buf[4];
            std::snprintf(buf, sizeof(buf), "%%%02X", c);
            out.append(buf, 3);
        }
    }
    return out;
}

// resolve the child that owns a conversation's stream session via the conv_id -> model map
// populated when the POST was routed. single map lookup then a meta lookup, no polling, no
// parsing of the conv id. returns nullopt when nothing maps, the caller answers not found and
// the client recovers
static std::optional<server_model_meta> resolve_child_for_conv(
        server_models & models, const std::string & conversation_id) {
    if (conversation_id.empty()) {
        return std::nullopt;
    }
    auto tracked = models.conv_models.lookup(conversation_id);
    if (!tracked.has_value()) {
        return std::nullopt;
    }
    auto meta = models.get_meta(*tracked);
    if (meta.has_value() && meta->is_ready()) {
        return meta;
    }
    return std::nullopt;
}

void server_models_routes::init_routes() {
    if (!common_subproc::is_supported()) {
        throw std::runtime_error("subprocess is not enabled on this build");
    }

    this->get_router_props = [this](const server_http_req & req) {
        std::string name = req.get_param("model");
        if (name.empty()) {
            // main instance
            auto res = std::make_unique<server_http_res>();
            json props = {
                // TODO: add support for this on web UI
                {"role",                 "router"},
                {"max_instances",        params.models_max},
                {"models_autoload",      params.models_autoload},
                // this is a dummy response to make sure the UI doesn't break
                {"model_alias", "llama-server"},
                {"model_path",  "none"},
                {"default_generation_settings", {
                    {"params", json{}},
                    {"n_ctx",  0},
                }},
                // New key
                {"ui_settings",          ui_settings},
                {"build_info",           std::string(llama_build_info())},
                {"cors_proxy_enabled",   params.ui_mcp_proxy},
            };
            if (!params.server_connect_code.empty()) {
                props["connect_code"] = params.server_connect_code;
            }
            res_ok(res, props);
            return res;
        }
        return proxy_get(req);
    };

    this->proxy_get = [this](const server_http_req & req) {
        std::string method = "GET";
        std::string name = req.get_param("model");
        bool autoload = is_autoload(params, req);
        auto error_res = std::make_unique<server_http_res>();
        if (!router_validate_model(name, models, autoload, error_res)) {
            return error_res;
        }
        if (autoload) {
            models.ensure_model_ready(name, req.should_stop);
        }
        return models.proxy_request(req, method, name, false);
    };

    this->proxy_ws = [this](const server_http_req & req, server_http_context::ws_conn & conn) {
        std::string name = req.get_param("model");
        bool autoload = is_autoload(params, req);
        auto error_res = std::make_unique<server_http_res>();
        if (!router_validate_model(name, models, autoload, error_res)) {
            // error_res holds {"error": {...}}, send it as a websocket error event
            json err = json::parse(error_res->data);
            err["type"] = "error";
            conn.send(safe_json_to_str(err));
            conn.close();
            return;
        }
        if (autoload) {
            models.ensure_model_ready(name, req.should_stop);
        }
        models.proxy_ws(req, conn, name);
    };

    this->proxy_post = [this](const server_http_req & req) {
        std::string method = "POST";
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        bool autoload = is_autoload(params, req);
        auto error_res = std::make_unique<server_http_res>();
        if (!router_validate_model(name, models, autoload, error_res)) {
            return error_res;
        }
        // remember which child serves this conversation so the stream routes can route straight
        // to it without polling, keyed on the exact conv id from the header. registered before
        // the load wait so a stop issued while the model loads can erase the entry and cancel
        // this request instead of leaving an orphan generation
        std::string conv_id = server_stream_conv_id_from_headers(req.headers);
        uint64_t ticket = models.conv_models.remember(conv_id, name);
        // a dead socket must not cancel a session request, only a stop does (checked right below)
        auto should_stop = ticket == 0 ? req.should_stop : nullptr;
        bool waited = autoload && models.ensure_model_ready(name, should_stop);
        if (ticket != 0 && !models.conv_models.alive(conv_id, ticket)) {
            SRV_INF("request for conv_id=%s cancelled while model name=%s was loading\n",
                    conv_id.c_str(), name.c_str());
            res_err(error_res, format_error_response(
                    "request cancelled by a stop while the model was loading", ERROR_TYPE_INVALID_REQUEST));
            return error_res;
        }
        // a session request that waited for a load detaches from the client socket: the
        // client may have dropped during the wait (page reload) and the session buffer must
        // still receive the generation for a later resume
        return models.proxy_request(req, method, name, true, waited && ticket != 0); // update last usage for POST request only
    };

    this->post_router_models_load = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        auto meta = models.get_meta(name);
        if (!meta.has_value()) {
            res_err(res, format_error_response("model is not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        if (meta->is_running()) {
            res_err(res, format_error_response("model is already running", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        models.load(meta->name);
        res_ok(res, {{"success", true}});
        return res;
    };

    this->get_router_models = [this](const server_http_req & req) {
        bool reload = !req.get_param("reload", "").empty();
        if (reload) {
            models.load_models();
        }
        auto res = std::make_unique<server_http_res>();
        json models_json = json::array();
        auto all_models = models.get_all_meta();
        std::time_t t = std::time(0);
        // a download on its way has no files to resolve yet, only the kinds it pulls
        const auto unfinished = models.unfinished_downloads_get();

        for (const auto & meta : all_models) {
            if (meta.hidden) {
                continue; // cache model deduplicated by a preset
            }
            json status {
                {"value",  server_model_status_to_string(meta.status)},
                {"args",   meta.args},
            };
            if (!meta.preset.name.empty()) {
                common_preset preset_copy = meta.preset;
                unset_reserved_args(preset_copy, false);
                preset_copy.unset_option("LLAMA_ARG_HOST");
                preset_copy.unset_option("LLAMA_ARG_PORT");
                preset_copy.unset_option("LLAMA_ARG_ALIAS");
                preset_copy.unset_option("LLAMA_ARG_TAGS");
                status["preset"] = preset_copy.to_ini();
            }
            if (meta.is_failed()) {
                status["exit_code"] = meta.exit_code;
                status["failed"]    = true;
            }

            json model_info = json {
                {"id",            meta.name},
                {"aliases",       meta.aliases},
                {"tags",          meta.tags},
                {"object",        "model"},    // for OAI-compat
                {"owned_by",      "llamacpp"}, // for OAI-compat
                {"created",       t},          // for OAI-compat
                {"status",        status},
                {"architecture",  meta.architecture},
                {"source",        server_model_source_to_string(meta.source)},
                {"can_remove",    meta.source == SERVER_MODEL_SOURCE_CACHE},
                // {"need_download", meta.need_download},
                // TODO: add other fields from the GGUF metadata
            };

            if (meta.n_ctx_train > 0) {
                model_info["context_length"] = meta.n_ctx_train;
            }

            if (models.sched->memory_enabled()) {
                model_info["memory"] = meta.memory.to_json();
            }

            // the sidecars a model has on disk, resolved as -hf would load them, for a
            // model that resolves from a HF repo. A download on its way reports the
            // kinds it pulls, as an unfinished one does
            const auto it_unfinished = std::find_if(unfinished.begin(), unfinished.end(),
                [&](const server_unfinished_download & d) { return d.model == meta.name; });
            const std::string hf_repo = model_hf_repo(meta);

            if (it_unfinished != unfinished.end() && meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
                model_info["sidecars"] = it_unfinished->sidecars;
            } else if (!hf_repo.empty()) {
                const json cached = cached_sidecars(hf_repo);
                model_info["sidecars"] = cached;

                // the sidecars of an unfinished download of this model that are not on
                // disk yet: a POST with the kind resumes one, a DELETE with it drops it
                if (it_unfinished != unfinished.end()) {
                    std::vector<std::string> incomplete;
                    for (const auto & p : it_unfinished->partials) {
                        if (!p.kind.empty() && !has_sidecar(incomplete, p.kind) &&
                            std::find(cached.begin(), cached.end(), p.kind) == cached.end()) {
                            incomplete.push_back(p.kind);
                        }
                    }
                    if (!incomplete.empty()) {
                        model_info["incomplete_sidecars"] = incomplete;
                    }
                }
            }

            // merge with loaded_info from the child process if available
            if (meta.is_running()) {
                for (auto it = meta.loaded_info.begin(); it != meta.loaded_info.end(); ++it) {
                    if (!model_info.contains(it.key())) {
                        model_info[it.key()] = it.value();
                    }
                }
            }
            models_json.push_back(model_info);
        }

        const json architecture_default = json {
            {"input_modalities",  json::array({"text"})},
            {"output_modalities", json::array({"text"})},
        };

        // a download that never finished has no instance: report it as incomplete so
        // the UI can offer to resume it, without starting anything
        for (const auto & d : unfinished) {
            if (models.has_model(d.model)) {
                continue; // complete, or downloading right now
            }
            models_json.push_back(json {
                {"id",           d.model},
                {"aliases",      json::array()},
                {"tags",         json::array()},
                {"object",       "model"},
                {"owned_by",     "llamacpp"},
                {"created",      t},
                {"status",       json { {"value", server_model_status_to_string(SERVER_MODEL_STATUS_INCOMPLETE)}, {"args", json::array()} }},
                {"architecture", architecture_default},
                {"source",       server_model_source_to_string(SERVER_MODEL_SOURCE_CACHE)},
                {"can_remove",   true},
                {"sidecars",     d.sidecars},
            });
        }

        res_ok(res, {
            {"data", models_json},
            {"object", "list"},
        });
        return res;
    };

    this->post_router_models_unload = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        auto model = models.get_meta(name);
        if (!model.has_value()) {
            res_err(res, format_error_response("model is not found", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!model->is_running() && model->status != SERVER_MODEL_STATUS_DOWNLOADING) {
            res_err(res, format_error_response("model is not running", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        models.unload(model->name);
        res_ok(res, {{"success", true}});
        return res;
    };

    this->get_router_models_sse = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        res->status = 200;
        res->content_type = "text/event-stream";
        auto sse_client = std::make_shared<server_models_sse_client>(models.sse);
        res->next = [this, sse_client, &req](std::string & output) -> bool {
            auto result = sse_client->next([&]() {
                return stopping.load(std::memory_order_relaxed) || req.should_stop();
            });
            if (result == nullptr) {
                return false; // client disconnected or should_stop
            }
            output = "data: " + safe_json_to_str(result->to_json()) + "\n\n";
            return true; // listen for the next event
        };
        return res;
    };

    this->post_router_models = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();

        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        if (name.empty()) {
            throw std::invalid_argument("model must be a non-empty string");
        }
        const auto sidecars = parse_sidecars(body);

        common_params_model model;
        model.hf_repo = name;

        // validate by fetching metadata, through the plan the download follows
        common_download_hf_plan plan;
        try {
            plan = common_download_get_hf_plan(model, sidecar_download_opts(params, sidecars));
        } catch (...) {
            SRV_ERR("unknown error while validating model '%s'\n", name.c_str());
            // other exceptions will be handled by the outer ex_wrapper()
            throw;
        }

        // a tag that resolves no file has nothing to download
        if (plan.preset.path.empty() && plan.primary.path.empty() && common_download_plan_sidecars(plan).empty()) {
            throw std::invalid_argument("model '" + name + "' resolves no file to download");
        }

        // then, proceed with the actual download; load() decides over an existing entry
        SRV_INF("starting download for model '%s'\n", name.c_str());
        {
            server_models::load_options load_opts;
            load_opts.role = SERVER_ROLE_DOWNLOAD;
            load_opts.custom_meta = server_model_meta{};
            load_opts.custom_meta->source = SERVER_MODEL_SOURCE_CACHE;
            load_opts.custom_meta->name   = name;
            load_opts.sidecars            = sidecars;
            load_opts.partials            = common_download_plan_partials(plan);
            models.load(name, load_opts);
        }

        res_ok(res, {{"success", true}});
        return res;
    };

    this->del_router_models = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();

        std::string name = req.get_param("model");
        if (name.empty()) {
            throw std::invalid_argument("model must be a non-empty string");
        }

        // a sidecar alone goes, the model keeps its files
        std::string sidecar = req.get_param("sidecar");
        if (!sidecar.empty()) {
            check_sidecar(sidecar);
            res_ok(res, {{"success", models.remove_sidecar(name, sidecar)}});
            return res;
        }

        // a download listed from an earlier run has no instance, only its tag and
        // its partial files
        if (!models.has_model(name)) {
            models.unfinished_downloads_remove(name);
            res_ok(res, {{"success", true}});
            return res;
        }

        models.remove(name); // throws on error

        res_ok(res, {{"success", true}});
        return res;
    };

    this->router_stream_get = [this](const server_http_req & req) {
        // GET /v1/stream?conv_id=<id>&from=N. resolve the owning child from the conv_id -> model
        // map, 404 when nothing maps
        auto res = std::make_unique<server_http_res>();
        std::string conv_id = req.get_param("conv_id");
        if (conv_id.empty()) {
            res_err(res, format_error_response("Missing conversation id in path", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        std::optional<server_model_meta> owner = resolve_child_for_conv(models, conv_id);
        if (!owner.has_value()) {
            // a registered conv whose model is still loading earns a retry: the session appears
            // once the load ends and the pending request reaches the child
            auto tracked = models.conv_models.lookup(conv_id);
            auto meta = tracked.has_value() ? models.get_meta(*tracked) : std::nullopt;
            bool transient = meta.has_value() && (meta->status == SERVER_MODEL_STATUS_LOADING ||
                                                  meta->status == SERVER_MODEL_STATUS_DOWNLOADING ||
                                                  meta->status == SERVER_MODEL_STATUS_DOWNLOADED);
            if (transient) {
                res_err(res, format_error_response("Stream owner model is loading, retry later", ERROR_TYPE_UNAVAILABLE));
            } else {
                res_err(res, format_error_response("Stream not found or expired", ERROR_TYPE_NOT_FOUND));
            }
            return res;
        }
        std::string from = req.get_param("from");
        std::string child_path = "/v1/stream?conv_id=" + encode_qs(conv_id);
        if (!from.empty()) {
            child_path += "&from=" + from;
        }
        SRV_TRC("proxying stream resume to model %s on port %d, path=%s\n",
                owner->name.c_str(), owner->port, child_path.c_str());
        auto proxy = std::make_unique<server_http_proxy>(
                "GET",
                "http",
                CHILD_ADDR,
                owner->port,
                child_path,
                req.headers,
                req.body,
                req.files,
                req.should_stop,
                params.timeout_read,
                params.timeout_write);
        return std::unique_ptr<server_http_res>(std::move(proxy));
    };

    this->router_streams_lookup = [this](const server_http_req & req) {
        // POST /v1/streams/lookup. resolve each requested conv id to its owning child via the
        // map, group the ids per child, and query only the children that actually own some of
        // them instead of fanning out to every ready child. a child only answers for the ids
        // it owns, never lists anything else
        auto res = std::make_unique<server_http_res>();
        std::vector<std::string> requested;
        try {
            json body = json::parse(req.body);
            if (body.contains("conversation_ids") && body["conversation_ids"].is_array()) {
                for (const auto & v : body["conversation_ids"]) {
                    if (v.is_string() && !v.get<std::string>().empty()) {
                        requested.push_back(v.get<std::string>());
                    }
                }
            }
        } catch (const std::exception &) {
            res_ok(res, json::array());
            return res;
        }

        // group requested ids by the child port that owns them, drop ids that map to nothing
        std::unordered_map<int, json> per_child;
        for (const auto & cid : requested) {
            auto owner = resolve_child_for_conv(models, cid);
            if (!owner.has_value()) {
                continue;
            }
            per_child[owner->port].push_back(cid);
        }

        json aggregated = json::array();
        for (auto & [port, ids] : per_child) {
            json child_body = {{"conversation_ids", ids}};
            httplib::Client cli(CHILD_ADDR, port);
            cli.set_connection_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_read_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_write_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            auto resp = cli.Post("/v1/streams/lookup", child_body.dump(), "application/json");
            if (!resp || resp->status != 200) {
                continue;
            }
            try {
                json child_arr = json::parse(resp->body);
                if (!child_arr.is_array()) {
                    continue;
                }
                for (auto & entry : child_arr) {
                    if (entry.is_object()) {
                        aggregated.push_back(entry);
                    }
                }
            } catch (const std::exception &) {
                continue;
            }
        }
        res_ok(res, aggregated);
        return res;
    };

    this->router_stream_delete = [this](const server_http_req & req) {
        // DELETE /v1/stream?conv_id=<id>. resolve the owning child via the map and forward only to
        // it, evict_and_cancel is idempotent on the child
        auto res = std::make_unique<server_http_res>();
        std::string conv_id = req.get_param("conv_id");
        if (conv_id.empty()) {
            res_err(res, format_error_response("Missing conversation id in path", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        std::string child_path = "/v1/stream?conv_id=" + encode_qs(conv_id);
        auto owner = resolve_child_for_conv(models, conv_id);
        if (owner.has_value()) {
            httplib::Client cli(CHILD_ADDR, owner->port);
            cli.set_connection_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_read_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_write_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            auto resp = cli.Delete(child_path.c_str());
            (void) resp; // the child logs its own miss when the session is unknown there
        } else if (auto tracked = models.conv_models.lookup(conv_id); tracked.has_value()) {
            // the entry exists but its model is still loading: the forget below erases it,
            // which cancels the request parked in proxy_post before the generation starts
            SRV_INF("router stop for conv_id=%s while model name=%s is loading, cancelling the pending request\n",
                    conv_id.c_str(), tracked->c_str());
        } else {
            SRV_WRN("router stop for unknown conv_id=%s, no owning child in the conv map\n",
                    conv_id.c_str());
        }
        // drop the tracking entry, the session is being torn down
        models.conv_models.forget(conv_id);
        res->status = 204;
        res->content_type = "application/json";
        return res;
    };
}



//
// server_http_proxy
//

static std::string to_lower_copy(const std::string & value) {
    std::string lowered(value.size(), '\0');
    std::transform(value.begin(), value.end(), lowered.begin(), [](unsigned char c) { return std::tolower(c); });
    return lowered;
}

static bool should_strip_proxy_header(const std::string & header_name) {
    // Headers that get duplicated when router forwards child responses
    if (header_name == "server" ||
        header_name == "transfer-encoding" ||
        header_name == "content-length" || // quick fix for https://github.com/ggml-org/llama.cpp/issues/17710
        header_name == "keep-alive") {
        return true;
    }

    // Router injects CORS, child also sends them: duplicate
    if (header_name.rfind("access-control-", 0) == 0) {
        return true;
    }

    return false;
}

static std::string generate_multipart_boundary() {
    thread_local std::mt19937 gen(std::random_device{}());
    static const char chars[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    std::uniform_int_distribution<> dis(0, sizeof(chars) - 2);
    std::string boundary = "----llama-cpp-proxy-";
    for (int i = 0; i < 16; i++) {
        boundary += chars[dis(gen)];
    }
    return boundary;
}

static std::string build_multipart_body(
        const json & form_fields,
        const std::map<std::string, uploaded_file> & files,
        const std::string & boundary) {
    static auto sanitize_field = [](const std::string & text) {
        std::string result;
        result.reserve(text.size());
        for (char c : text) {
            if (c != '\n' && c != '\r' && c != '"') {
                result += c;
            }
        }
        return result;
    };

    std::ostringstream body;

    for (const auto & [key, value] : form_fields.items()) {
        if (value.is_array()) {
            for (const auto & item : value) {
                body << "--" << boundary << "\r\n";
                body << "Content-Disposition: form-data; name=\"" << sanitize_field(key) << "\"\r\n";
                body << "\r\n";
                if (!item.is_string()) {
                    throw std::invalid_argument("expected string");
                }
                body << item.get<std::string>() << "\r\n";
            }
        } else {
            body << "--" << boundary << "\r\n";
            body << "Content-Disposition: form-data; name=\"" << sanitize_field(key) << "\"\r\n";
            body << "\r\n";
            if (!value.is_string()) {
                throw std::invalid_argument("expected string");
            }
            body << value.get<std::string>() << "\r\n";
        }
    }

    for (const auto & [key, file] : files) {
        body << "--" << boundary << "\r\n";
        body << "Content-Disposition: form-data; name=\"" << sanitize_field(key) << "\"";
        if (!file.filename.empty()) {
            body << "; filename=\"" << sanitize_field(file.filename) << "\"";
        }
        body << "\r\n";
        if (!file.content_type.empty()) {
            body << "Content-Type: " << sanitize_field(file.content_type) << "\r\n";
        } else {
            body << "Content-Type: application/octet-stream\r\n";
        }
        body << "\r\n";
        body.write(reinterpret_cast<const char*>(file.data.data()), file.data.size());
        body << "\r\n";
    }

    body << "--" << boundary << "--\r\n";
    return body.str();
}

server_http_proxy::server_http_proxy(
        const std::string & method,
        const std::string & scheme,
        const std::string & host,
        int port,
        const std::string & path,
        const std::map<std::string, std::string> & headers,
        const std::string & body,
        const std::map<std::string, uploaded_file> & files,
        const std::function<bool()> should_stop,
        int32_t timeout_read,
        int32_t timeout_write
        ) {
    // shared between reader and writer threads
    auto cli  = std::make_shared<httplib::ClientImpl>(host, port);
    auto pipe = std::make_shared<server_pipe<msg_t>>();

    if (scheme == "https") {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        cli.reset(new httplib::SSLClient(host, port));
#else
        throw std::runtime_error("HTTPS requested but CPPHTTPLIB_OPENSSL_SUPPORT is not defined");
#endif
    }

    // setup Client
    cli->set_follow_location(true);
    cli->set_connection_timeout(timeout_read, 0); // use --timeout value instead of hardcoded 5 s
    cli->set_write_timeout(timeout_read, 0); // reversed for cli (client) vs srv (server)
    cli->set_read_timeout(timeout_write, 0);
    this->status = 500; // to be overwritten upon response
    this->cleanup_pipes = [pipe, cli]() {
        pipe->close_read();
        pipe->close_write();
        cli->stop();
    };

    // wire up the receive end of the pipe
    this->next = [pipe, should_stop](std::string & out) -> bool {
        msg_t msg;
        bool has_next = pipe->read(msg, should_stop);
        if (!msg.data.empty()) {
            out = std::move(msg.data);
        }
        return has_next; // false if EOF or pipe broken
    };

    // build the header message forwarded to the reader thread, stripping internal proxy headers
    auto make_header_msg = [](const httplib::Response & response) {
        msg_t msg;
        msg.status = response.status;
        for (const auto & [key, value] : response.headers) {
            const auto lowered = to_lower_copy(key);
            if (should_strip_proxy_header(lowered)) {
                continue;
            }
            if (lowered == "content-type") {
                msg.content_type = value;
                continue;
            }
            msg.headers[key] = value;
        }
        return msg;
    };

    // true once response_handler has already forwarded the headers
    auto headers_sent = std::make_shared<std::atomic<bool>>(false);

    // wire up the HTTP client
    // note: do NOT capture `this` pointer, as it may be destroyed before the thread ends
    httplib::ResponseHandler response_handler = [pipe, headers_sent, make_header_msg](const httplib::Response & response) {
        headers_sent->store(true);
        return pipe->write(make_header_msg(response)); // send headers first
    };
    httplib::ContentReceiverWithProgress content_receiver = [pipe](const char * data, size_t data_length, size_t, size_t) {
        // send data chunks
        // returns false if pipe is closed / broken (signal to stop receiving)
        return pipe->write({{}, 0, std::string(data, data_length), ""});
    };

    // when files are present, the body was converted from multipart form data to JSON
    // we need to reconstruct the multipart body for the downstream server
    std::string effective_body = body;
    std::string override_content_type;
    bool has_files = !files.empty();

    if (has_files) {
        json form_fields = json::parse_no_throw(body);
        if (!form_fields.is_discarded()) {
            auto boundary = generate_multipart_boundary();
            effective_body = build_multipart_body(form_fields, files, boundary);
            override_content_type = "multipart/form-data; boundary=" + boundary;
        } else {
            throw std::runtime_error("failed to parse multipart form fields JSON");
        }
    }

    // prepare the request to destination server
    httplib::Request req;
    {
        req.method = method;
        req.path = path;
        for (const auto & [key, value] : headers) {
            const auto lowered = to_lower_copy(key);
            if (lowered == "accept-encoding") {
                // disable Accept-Encoding to avoid compressed responses
                continue;
            }
            if (lowered == "transfer-encoding") {
                // the body is already decoded
                continue;
            }
            if (lowered == "content-length") {
                // let httplib calculate Content-Length from the actual body
                continue;
            }
            if (lowered == "content-type") {
                if (has_files) {
                    // we set our own Content-Type with the new boundary
                    continue;
                }
                // when no files but the original request was multipart,
                // the body is now JSON, so correct the Content-Type
                if (value.find("multipart/form-data") != std::string::npos) {
                    override_content_type = "application/json; charset=utf-8";
                    continue;
                }
            }
            if (lowered == "host") {
                bool is_default_port = (scheme == "https" && port == 443) || (scheme == "http" && port == 80);
                const std::string url_host = common_http_format_host(host);
                req.set_header(key, is_default_port ? url_host : url_host + ":" + std::to_string(port));
            } else {
                req.set_header(key, value);
            }
        }
        req.body = effective_body;
        if (!override_content_type.empty()) {
            req.set_header("Content-Type", override_content_type);
        }
        req.response_handler = response_handler;
        req.content_receiver = content_receiver;
    }

    // start the proxy thread
    SRV_DBG("start proxy thread %s %s\n", req.method.c_str(), req.path.c_str());
    this->thread = std::thread([cli, pipe, req, headers_sent, make_header_msg]() {
        auto result = cli->send(std::move(req));
        if (result.error() != httplib::Error::Success) {
            auto err_str = httplib::to_string(result.error());
            SRV_ERR("http client error: %s\n", err_str.c_str());
            pipe->write({{}, 500, "", ""}); // header
            pipe->write({{}, 0, "proxy error: " + err_str, ""}); // body
        } else if (!headers_sent->load()) {
            // httplib skips response_handler for bodyless statuses like 204, send headers here instead
            pipe->write(make_header_msg(*result));
        }
        pipe->close_write(); // signal EOF to reader
        SRV_DBG("%s", "client request thread ended\n");
    });
    this->thread.detach();

    // wait for the first chunk (headers)
    {
        msg_t header;
        if (pipe->read(header, should_stop)) {
            SRV_DBG("%s", "received response headers\n");
            this->status  = header.status;
            this->headers = std::move(header.headers);
            if (!header.content_type.empty()) {
                this->content_type = std::move(header.content_type);
            }
        } else {
            SRV_DBG("%s", "no response headers received (request cancelled?)\n");
        }
    }
}
