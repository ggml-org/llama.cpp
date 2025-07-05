#include "server-core.hpp"
#include "utils.hpp"

// mime type for sending response
#define MIMETYPE_JSON "application/json; charset=utf-8"

// auto generated files (see README.md for details)
#include "index.html.gz.hpp"
#include "loading.html.hpp"


static void log_server_request(const httplib::Request & req, const httplib::Response & res) {
    // skip GH copilot requests when using default port
    if (req.path == "/v1/health" || req.path == "/v1/completions") {
        return;
    }

    // reminder: this function is not covered by httplib's exception handler; if someone does more complicated stuff, think about wrapping it in try-catch

    SRV_INF("request: %s %s %s %d\n", req.method.c_str(), req.path.c_str(), req.remote_addr.c_str(), res.status);

    SRV_DBG("request:  %s\n", req.body.c_str());
    SRV_DBG("response: %s\n", res.body.c_str());
}

std::function<void(int)> shutdown_handler;
std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

int main(int argc, char ** argv) {
    // own arguments required by this example
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    common_init();

    // struct that contains llama context and inference
    server_context ctx_server;

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INF("system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n", params.cpuparams.n_threads, params.cpuparams_batch.n_threads, std::thread::hardware_concurrency());
    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("\n");

    std::unique_ptr<httplib::Server> svr;
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if (params.ssl_file_key != "" && params.ssl_file_cert != "") {
        LOG_INF("Running with SSL: key = %s, cert = %s\n", params.ssl_file_key.c_str(), params.ssl_file_cert.c_str());
        svr.reset(
            new httplib::SSLServer(params.ssl_file_cert.c_str(), params.ssl_file_key.c_str())
        );
    } else {
        LOG_INF("Running without SSL\n");
        svr.reset(new httplib::Server());
    }
#else
    if (params.ssl_file_key != "" && params.ssl_file_cert != "") {
        LOG_ERR("Server is built without SSL support\n");
        return 1;
    }
    svr.reset(new httplib::Server());
#endif

    std::atomic<server_state> state{SERVER_STATE_LOADING_MODEL};

    svr->set_default_headers({{"Server", "llama.cpp"}});
    svr->set_logger(log_server_request);

    auto res_error = [](httplib::Response & res, const json & error_data) {
        json final_response {{"error", error_data}};
        res.set_content(safe_json_to_str(final_response), MIMETYPE_JSON);
        res.status = json_value(error_data, "code", 500);
    };

    auto res_ok = [](httplib::Response & res, const json & data) {
        res.set_content(safe_json_to_str(data), MIMETYPE_JSON);
        res.status = 200;
    };

    svr->set_exception_handler([&res_error](const httplib::Request &, httplib::Response & res, const std::exception_ptr & ep) {
        std::string message;
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception & e) {
            message = e.what();
        } catch (...) {
            message = "Unknown Exception";
        }

        try {
            json formatted_error = format_error_response(message, ERROR_TYPE_SERVER);
            LOG_WRN("got exception: %s\n", formatted_error.dump().c_str());
            res_error(res, formatted_error);
        } catch (const std::exception & e) {
            LOG_ERR("got another exception: %s | while hanlding exception: %s\n", e.what(), message.c_str());
        }
    });

    svr->set_error_handler([&res_error](const httplib::Request &, httplib::Response & res) {
        if (res.status == 404) {
            res_error(res, format_error_response("File Not Found", ERROR_TYPE_NOT_FOUND));
        }
        // for other error codes, we skip processing here because it's already done by res_error()
    });

    // set timeouts and change hostname and port
    svr->set_read_timeout (params.timeout_read);
    svr->set_write_timeout(params.timeout_write);

    std::unordered_map<std::string, std::string> log_data;

    log_data["hostname"] = params.hostname;
    log_data["port"]     = std::to_string(params.port);

    if (params.api_keys.size() == 1) {
        auto key = params.api_keys[0];
        log_data["api_key"] = "api_key: ****" + key.substr(std::max((int)(key.length() - 4), 0));
    } else if (params.api_keys.size() > 1) {
        log_data["api_key"] = "api_key: " + std::to_string(params.api_keys.size()) + " keys loaded";
    }

    // Necessary similarity of prompt for slot selection
    ctx_server.slot_prompt_similarity = params.slot_prompt_similarity;

    //
    // Middlewares
    //

    auto middleware_validate_api_key = [&params, &res_error](const httplib::Request & req, httplib::Response & res) {
        static const std::unordered_set<std::string> public_endpoints = {
            "/health",
            "/models",
            "/v1/models",
            "/api/tags"
        };

        // If API key is not set, skip validation
        if (params.api_keys.empty()) {
            return true;
        }

        // If path is public or is static file, skip validation
        if (public_endpoints.find(req.path) != public_endpoints.end() || req.path == "/") {
            return true;
        }

        // Check for API key in the header
        auto auth_header = req.get_header_value("Authorization");

        std::string prefix = "Bearer ";
        if (auth_header.substr(0, prefix.size()) == prefix) {
            std::string received_api_key = auth_header.substr(prefix.size());
            if (std::find(params.api_keys.begin(), params.api_keys.end(), received_api_key) != params.api_keys.end()) {
                return true; // API key is valid
            }
        }

        // API key is invalid or not provided
        res_error(res, format_error_response("Invalid API Key", ERROR_TYPE_AUTHENTICATION));

        LOG_WRN("Unauthorized: Invalid API Key\n");

        return false;
    };

    auto middleware_server_state = [&res_error, &state](const httplib::Request & req, httplib::Response & res) {
        server_state current_state = state.load();
        if (current_state == SERVER_STATE_LOADING_MODEL) {
            auto tmp = string_split<std::string>(req.path, '.');
            if (req.path == "/" || tmp.back() == "html") {
                res.set_content(reinterpret_cast<const char*>(loading_html), loading_html_len, "text/html; charset=utf-8");
                res.status = 503;
            } else if (req.path == "/models" || req.path == "/v1/models" || req.path == "/api/tags") {
                // allow the models endpoint to be accessed during loading
                return true;
            } else {
                res_error(res, format_error_response("Loading model", ERROR_TYPE_UNAVAILABLE));
            }
            return false;
        }
        return true;
    };

    // register server middlewares
    svr->set_pre_routing_handler([&middleware_validate_api_key, &middleware_server_state](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        // If this is OPTIONS request, skip validation because browsers don't include Authorization header
        if (req.method == "OPTIONS") {
            res.set_header("Access-Control-Allow-Credentials", "true");
            res.set_header("Access-Control-Allow-Methods",     "GET, POST");
            res.set_header("Access-Control-Allow-Headers",     "*");
            res.set_content("", "text/html"); // blank response, no data
            return httplib::Server::HandlerResponse::Handled; // skip further processing
        }
        if (!middleware_server_state(req, res)) {
            return httplib::Server::HandlerResponse::Handled;
        }
        if (!middleware_validate_api_key(req, res)) {
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    //
    // Route handlers (or controllers)
    //

    const auto handle_health = [&](const httplib::Request &, httplib::Response & res) {
        // error and loading states are handled by middleware
        json health = {{"status", "ok"}};
        res_ok(res, health);
    };

    const auto handle_slots = [&](const httplib::Request & req, httplib::Response & res) {
        if (!params.endpoint_slots) {
            res_error(res, format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        // request slots data using task queue
        int task_id = ctx_server.queue_tasks.get_new_id();
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = task_id;
            ctx_server.queue_results.add_waiting_task_id(task_id);
            ctx_server.queue_tasks.post(std::move(task), true); // high-priority task
        }

        // get the result
        server_task_result_ptr result = ctx_server.queue_results.recv(task_id);
        ctx_server.queue_results.remove_waiting_task_id(task_id);

        if (result->is_error()) {
            res_error(res, result->to_json());
            return;
        }

        // TODO: get rid of this dynamic_cast
        auto res_metrics = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_metrics != nullptr);

        // optionally return "fail_on_no_slot" error
        if (req.has_param("fail_on_no_slot")) {
            if (res_metrics->n_idle_slots == 0) {
                res_error(res, format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return;
            }
        }

        res_ok(res, res_metrics->slots_data);
    };

    const auto handle_metrics = [&](const httplib::Request &, httplib::Response & res) {
        if (!params.endpoint_metrics) {
            res_error(res, format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        // request slots data using task queue
        int task_id = ctx_server.queue_tasks.get_new_id();
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = task_id;
            ctx_server.queue_results.add_waiting_task_id(task_id);
            ctx_server.queue_tasks.post(std::move(task), true); // high-priority task
        }

        // get the result
        server_task_result_ptr result = ctx_server.queue_results.recv(task_id);
        ctx_server.queue_results.remove_waiting_task_id(task_id);

        if (result->is_error()) {
            res_error(res, result->to_json());
            return;
        }

        // TODO: get rid of this dynamic_cast
        auto res_metrics = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_metrics != nullptr);

        // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
        json all_metrics_def = json {
            {"counter", {{
                    {"name",  "prompt_tokens_total"},
                    {"help",  "Number of prompt tokens processed."},
                    {"value",  (uint64_t) res_metrics->n_prompt_tokens_processed_total}
            }, {
                    {"name",  "prompt_seconds_total"},
                    {"help",  "Prompt process time"},
                    {"value",  (uint64_t) res_metrics->t_prompt_processing_total / 1.e3}
            }, {
                    {"name",  "tokens_predicted_total"},
                    {"help",  "Number of generation tokens processed."},
                    {"value",  (uint64_t) res_metrics->n_tokens_predicted_total}
            }, {
                    {"name",  "tokens_predicted_seconds_total"},
                    {"help",  "Predict process time"},
                    {"value",  (uint64_t) res_metrics->t_tokens_generation_total / 1.e3}
            }, {
                    {"name",  "n_decode_total"},
                    {"help",  "Total number of llama_decode() calls"},
                    {"value",  res_metrics->n_decode_total}
            }, {
                    {"name",  "n_busy_slots_per_decode"},
                    {"help",  "Average number of busy slots per llama_decode() call"},
                    {"value",  (float) res_metrics->n_busy_slots_total / std::max((float) res_metrics->n_decode_total, 1.f)}
            }}},
            {"gauge", {{
                    {"name",  "prompt_tokens_seconds"},
                    {"help",  "Average prompt throughput in tokens/s."},
                    {"value",  res_metrics->n_prompt_tokens_processed ? 1.e3 / res_metrics->t_prompt_processing * res_metrics->n_prompt_tokens_processed : 0.}
            },{
                    {"name",  "predicted_tokens_seconds"},
                    {"help",  "Average generation throughput in tokens/s."},
                    {"value",  res_metrics->n_tokens_predicted ? 1.e3 / res_metrics->t_tokens_generation * res_metrics->n_tokens_predicted : 0.}
            },{
                    {"name",  "requests_processing"},
                    {"help",  "Number of requests processing."},
                    {"value",  (uint64_t) res_metrics->n_processing_slots}
            },{
                    {"name",  "requests_deferred"},
                    {"help",  "Number of requests deferred."},
                    {"value",  (uint64_t) res_metrics->n_tasks_deferred}
            }}}
        };

        std::stringstream prometheus;

        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();

            for (const auto & metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");

                auto value = json_value(metric_def, "value", 0.);
                prometheus << "# HELP llamacpp:" << name << " " << help  << "\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\n"
                            << "llamacpp:"        << name << " " << value << "\n";
            }
        }

        res.set_header("Process-Start-Time-Unix", std::to_string(res_metrics->t_start));

        res.set_content(prometheus.str(), "text/plain; version=0.0.4");
        res.status = 200; // HTTP OK
    };

    const auto handle_slots_save = [&ctx_server, &res_error, &res_ok, &params](const httplib::Request & req, httplib::Response & res, int id_slot) {
        json request_data = json::parse(req.body);
        std::string filename = request_data.at("filename");
        if (!fs_validate_filename(filename)) {
            res_error(res, format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        std::string filepath = params.slot_save_path + filename;

        int task_id = ctx_server.queue_tasks.get_new_id();
        {
            server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
            task.id = task_id;
            task.slot_action.slot_id  = id_slot;
            task.slot_action.filename = filename;
            task.slot_action.filepath = filepath;

            ctx_server.queue_results.add_waiting_task_id(task_id);
            ctx_server.queue_tasks.post(std::move(task));
        }

        server_task_result_ptr result = ctx_server.queue_results.recv(task_id);
        ctx_server.queue_results.remove_waiting_task_id(task_id);

        if (result->is_error()) {
            res_error(res, result->to_json());
            return;
        }

        res_ok(res, result->to_json());
    };

    const auto handle_slots_restore = [&ctx_server, &res_error, &res_ok, &params](const httplib::Request & req, httplib::Response & res, int id_slot) {
        json request_data = json::parse(req.body);
        std::string filename = request_data.at("filename");
        if (!fs_validate_filename(filename)) {
            res_error(res, format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        std::string filepath = params.slot_save_path + filename;

        int task_id = ctx_server.queue_tasks.get_new_id();
        {
            server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
            task.id = task_id;
            task.slot_action.slot_id  = id_slot;
            task.slot_action.filename = filename;
            task.slot_action.filepath = filepath;

            ctx_server.queue_results.add_waiting_task_id(task_id);
            ctx_server.queue_tasks.post(std::move(task));
        }

        server_task_result_ptr result = ctx_server.queue_results.recv(task_id);
        ctx_server.queue_results.remove_waiting_task_id(task_id);

        if (result->is_error()) {
            res_error(res, result->to_json());
            return;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
        res_ok(res, result->to_json());
    };

    const auto handle_slots_erase = [&ctx_server, &res_error, &res_ok](const httplib::Request & /* req */, httplib::Response & res, int id_slot) {
        int task_id = ctx_server.queue_tasks.get_new_id();
        {
            server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
            task.id = task_id;
            task.slot_action.slot_id = id_slot;

            ctx_server.queue_results.add_waiting_task_id(task_id);
            ctx_server.queue_tasks.post(std::move(task));
        }

        server_task_result_ptr result = ctx_server.queue_results.recv(task_id);
        ctx_server.queue_results.remove_waiting_task_id(task_id);

        if (result->is_error()) {
            res_error(res, result->to_json());
            return;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
        res_ok(res, result->to_json());
    };

    const auto handle_slots_action = [&params, &res_error, &handle_slots_save, &handle_slots_restore, &handle_slots_erase](const httplib::Request & req, httplib::Response & res) {
        if (params.slot_save_path.empty()) {
            res_error(res, format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        std::string id_slot_str = req.path_params.at("id_slot");
        int id_slot;

        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res_error(res, format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        std::string action = req.get_param_value("action");

        if (action == "save") {
            handle_slots_save(req, res, id_slot);
        } else if (action == "restore") {
            handle_slots_restore(req, res, id_slot);
        } else if (action == "erase") {
            handle_slots_erase(req, res, id_slot);
        } else {
            res_error(res, format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        }
    };

    const auto handle_props = [&ctx_server, &res_ok](const httplib::Request &, httplib::Response & res) {
        // this endpoint is publicly available, please only return what is safe to be exposed
        json data = {
            { "default_generation_settings", ctx_server.default_generation_settings_for_props },
            { "total_slots",                 ctx_server.params_base.n_parallel },
            { "model_path",                  ctx_server.params_base.model.path },
            { "modalities",                  json{
                {"vision", ctx_server.oai_parser_opt.allow_image},
                {"audio",  ctx_server.oai_parser_opt.allow_audio},
            } },
            { "chat_template",               common_chat_templates_source(ctx_server.chat_templates.get()) },
            { "bos_token",                   common_token_to_piece(ctx_server.ctx, llama_vocab_bos(ctx_server.vocab), /* special= */ true)},
            { "eos_token",                   common_token_to_piece(ctx_server.ctx, llama_vocab_eos(ctx_server.vocab), /* special= */ true)},
            { "build_info",                  build_info },
        };
        if (ctx_server.params_base.use_jinja) {
            if (auto tool_use_src = common_chat_templates_source(ctx_server.chat_templates.get(), "tool_use")) {
                data["chat_template_tool_use"] = tool_use_src;
            }
        }

        res_ok(res, data);
    };

    const auto handle_props_change = [&ctx_server, &res_error, &res_ok](const httplib::Request & req, httplib::Response & res) {
        if (!ctx_server.params_base.endpoint_props) {
            res_error(res, format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        json data = json::parse(req.body);

        // update any props here

        res_ok(res, {{ "success", true }});
    };

    const auto handle_api_show = [&ctx_server, &res_ok](const httplib::Request &, httplib::Response & res) {
        json data = {
            {
                "template", common_chat_templates_source(ctx_server.chat_templates.get()),
            },
            {
                "model_info", {
                    { "llama.context_length", ctx_server.slots.back().n_ctx, },
                }
            },
            {"modelfile", ""},
            {"parameters", ""},
            {"template", common_chat_templates_source(ctx_server.chat_templates.get())},
            {"details", {
                {"parent_model", ""},
                {"format", "gguf"},
                {"family", ""},
                {"families", {""}},
                {"parameter_size", ""},
                {"quantization_level", ""}
            }},
            {"model_info", ""},
            {"capabilities", {"completion"}}
        };

        res_ok(res, data);
    };

    // handle completion-like requests (completion, chat, infill)
    // we can optionally provide a custom format for partial results and final results
    const auto handle_completions_impl = [&ctx_server, &res_error, &res_ok](
            server_task_type type,
            json & data,
            const std::vector<raw_buffer> & files,
            const std::function<bool()> & is_connection_closed,
            httplib::Response & res,
            oaicompat_type oaicompat) -> void {
        GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

        auto completion_id = gen_chatcmplid();
        std::unordered_set<int> task_ids;
        try {
            std::vector<server_task> tasks;

            const auto & prompt = data.at("prompt");
            // TODO: this log can become very long, put it behind a flag or think about a more compact format
            //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

            // process files
            mtmd::bitmaps bitmaps;
            const bool has_mtmd = ctx_server.mctx != nullptr;
            {
                if (!has_mtmd && !files.empty()) {
                    throw std::runtime_error("This server does not support multimodal");
                }
                for (auto & file : files) {
                    mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(ctx_server.mctx, file.data(), file.size()));
                    if (!bmp.ptr) {
                        throw std::runtime_error("Failed to load image or audio file");
                    }
                    // calculate bitmap hash (for KV caching)
                    std::string hash = fnv_hash(bmp.data(), bmp.n_bytes());
                    bmp.set_id(hash.c_str());
                    bitmaps.entries.push_back(std::move(bmp));
                }
            }

            // process prompt
            std::vector<server_tokens> inputs;
            if (oaicompat && !prompt.is_string()) {
                throw std::runtime_error("prompt must be a string");
            }

            if (oaicompat && has_mtmd) {
                // multimodal
                std::string prompt_str = prompt.get<std::string>();
                mtmd_input_text inp_txt = {
                    prompt_str.c_str(),
                    /* add_special */   true,
                    /* parse_special */ true,
                };
                mtmd::input_chunks chunks(mtmd_input_chunks_init());
                auto bitmaps_c_ptr = bitmaps.c_ptr();
                int32_t tokenized = mtmd_tokenize(ctx_server.mctx,
                                                    chunks.ptr.get(),
                                                    &inp_txt,
                                                    bitmaps_c_ptr.data(),
                                                    bitmaps_c_ptr.size());
                if (tokenized != 0) {
                    throw std::runtime_error("Failed to tokenize prompt");
                }

                server_tokens tmp(chunks, true);
                inputs.push_back(std::move(tmp));
            } else {
                // non-multimodal version
                auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, prompt, true, true);
                for (auto & p : tokenized_prompts) {
                    auto tmp = server_tokens(p, ctx_server.mctx != nullptr);
                    inputs.push_back(std::move(tmp));
                }
            }

            tasks.reserve(inputs.size());
            for (size_t i = 0; i < inputs.size(); i++) {
                server_task task = server_task(type);

                task.id    = ctx_server.queue_tasks.get_new_id();
                task.index = i;

                task.prompt_tokens    = std::move(inputs[i]);
                task.params           = server_task::params_from_json_cmpl(
                        ctx_server.ctx,
                        ctx_server.params_base,
                        data);
                task.id_selected_slot = json_value(data, "id_slot", -1);

                // OAI-compat
                task.params.oaicompat                 = oaicompat;
                task.params.oaicompat_cmpl_id         = completion_id;
                // oaicompat_model is already populated by params_from_json_cmpl

                tasks.push_back(std::move(task));
            }

            task_ids = server_task::get_list_id(tasks);
            ctx_server.queue_results.add_waiting_tasks(tasks);
            ctx_server.queue_tasks.post(std::move(tasks));
        } catch (const std::exception & e) {
            res_error(res, format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        bool stream = json_value(data, "stream", false);

        if (!stream) {
            ctx_server.receive_multi_results(task_ids, [&](std::vector<server_task_result_ptr> & results) {
                if (results.size() == 1) {
                    // single result
                    res_ok(res, results[0]->to_json());
                } else {
                    // multiple results (multitask)
                    json arr = json::array();
                    for (auto & res : results) {
                        arr.push_back(res->to_json());
                    }
                    res_ok(res, arr);
                }
            }, [&](const json & error_data) {
                res_error(res, error_data);
            }, is_connection_closed);

            ctx_server.queue_results.remove_waiting_task_ids(task_ids);
        } else {
            const auto chunked_content_provider = [task_ids, &ctx_server, oaicompat](size_t, httplib::DataSink & sink) {
                ctx_server.receive_cmpl_results_stream(task_ids, [&](server_task_result_ptr & result) -> bool {
                    json res_json = result->to_json();
                    if (res_json.is_array()) {
                        for (const auto & res : res_json) {
                            if (!server_sent_event(sink, "data", res)) {
                                // sending failed (HTTP connection closed), cancel the generation
                                return false;
                            }
                        }
                        return true;
                    } else {
                        return server_sent_event(sink, "data", res_json);
                    }
                }, [&](const json & error_data) {
                    server_sent_event(sink, "error", error_data);
                }, [&sink]() {
                    // note: do not use req.is_connection_closed here because req is already destroyed
                    return !sink.is_writable();
                });
                if (oaicompat != OAICOMPAT_TYPE_NONE) {
                    static const std::string ev_done = "data: [DONE]\n\n";
                    sink.write(ev_done.data(), ev_done.size());
                }
                sink.done();
                return false;
            };

            auto on_complete = [task_ids, &ctx_server] (bool) {
                ctx_server.queue_results.remove_waiting_task_ids(task_ids);
            };

            res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
        }
    };

    const auto handle_completions = [&handle_completions_impl](const httplib::Request & req, httplib::Response & res) {
        json data = json::parse(req.body);
        std::vector<raw_buffer> files; // dummy
        handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            data,
            files,
            req.is_connection_closed,
            res,
            OAICOMPAT_TYPE_NONE);
    };

    const auto handle_completions_oai = [&handle_completions_impl](const httplib::Request & req, httplib::Response & res) {
        json data = oaicompat_completion_params_parse(json::parse(req.body));
        std::vector<raw_buffer> files; // dummy
        handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            data,
            files,
            req.is_connection_closed,
            res,
            OAICOMPAT_TYPE_COMPLETION);
    };

    const auto handle_infill = [&ctx_server, &res_error, &handle_completions_impl](const httplib::Request & req, httplib::Response & res) {
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res_error(res, format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        json data = json::parse(req.body);

        // validate input
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res_error(res, format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res_error(res, format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res_error(res, format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res_error(res, format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res_error(res, format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res_error(res, format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, prompt, false, true);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            ctx_server.params_base.n_batch,
            ctx_server.params_base.n_predict,
            ctx_server.slots[0].n_ctx, // TODO: there should be a better way
            ctx_server.params_base.spm_infill,
            tokenized_prompts[0]
        );

        std::vector<raw_buffer> files; // dummy
        handle_completions_impl(
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            req.is_connection_closed,
            res,
            OAICOMPAT_TYPE_NONE); // infill is not OAI compatible
    };

    const auto handle_chat_completions = [&ctx_server, &handle_completions_impl](const httplib::Request & req, httplib::Response & res) {
        LOG_DBG("request: %s\n", req.body.c_str());

        auto body = json::parse(req.body);
        std::vector<raw_buffer> files;
        json data = oaicompat_chat_params_parse(
            body,
            ctx_server.oai_parser_opt,
            files);

        handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            data,
            files,
            req.is_connection_closed,
            res,
            OAICOMPAT_TYPE_CHAT);
    };

    // same with handle_chat_completions, but without inference part
    const auto handle_apply_template = [&ctx_server, &res_ok](const httplib::Request & req, httplib::Response & res) {
        auto body = json::parse(req.body);
        std::vector<raw_buffer> files; // dummy, unused
        json data = oaicompat_chat_params_parse(
            body,
            ctx_server.oai_parser_opt,
            files);
        res_ok(res, {{ "prompt", std::move(data.at("prompt")) }});
    };

    const auto handle_models = [&params, &ctx_server, &state, &res_ok](const httplib::Request &, httplib::Response & res) {
        server_state current_state = state.load();
        json model_meta = nullptr;
        if (current_state == SERVER_STATE_READY) {
            model_meta = ctx_server.model_meta();
        }

        json models = {
            {"models", {
                {
                    {"name", params.model_alias.empty() ? params.model.path : params.model_alias},
                    {"model", params.model_alias.empty() ? params.model.path : params.model_alias},
                    {"modified_at", ""},
                    {"size", ""},
                    {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                    {"type", "model"},
                    {"description", ""},
                    {"tags", {""}},
                    {"capabilities", {"completion"}},
                    {"parameters", ""},
                    {"details", {
                        {"parent_model", ""},
                        {"format", "gguf"},
                        {"family", ""},
                        {"families", {""}},
                        {"parameter_size", ""},
                        {"quantization_level", ""}
                    }}
                }
            }},
            {"object", "list"},
            {"data", {
                {
                    {"id",       params.model_alias.empty() ? params.model.path : params.model_alias},
                    {"object",   "model"},
                    {"created",  std::time(0)},
                    {"owned_by", "llamacpp"},
                    {"meta",     model_meta},
                },
            }}
        };

        res_ok(res, models);
    };

    const auto handle_tokenize = [&ctx_server, &res_ok](const httplib::Request & req, httplib::Response & res) {
        const json body = json::parse(req.body);

        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, true);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.ctx, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        const json data = format_tokenizer_response(tokens_response);
        res_ok(res, data);
    };

    const auto handle_detokenize = [&ctx_server, &res_ok](const httplib::Request & req, httplib::Response & res) {
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.ctx, tokens.cbegin(), tokens.cend());
        }

        const json data = format_detokenized_response(content);
        res_ok(res, data);
    };

    const auto handle_embeddings_impl = [&ctx_server, &res_error, &res_ok](const httplib::Request & req, httplib::Response & res, oaicompat_type oaicompat) {
        if (!ctx_server.params_base.embedding) {
            res_error(res, format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        if (oaicompat != OAICOMPAT_TYPE_NONE && llama_pooling_type(ctx_server.ctx) == LLAMA_POOLING_TYPE_NONE) {
            res_error(res, format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        const json body = json::parse(req.body);

        // for the shape of input/content, see tokenize_input_prompts()
        json prompt;
        if (body.count("input") != 0) {
            prompt = body.at("input");
        } else if (body.contains("content")) {
            oaicompat = OAICOMPAT_TYPE_NONE; // "content" field is not OAI compatible
            prompt = body.at("content");
        } else {
            res_error(res, format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        bool use_base64 = false;
        if (body.count("encoding_format") != 0) {
            const std::string& format = body.at("encoding_format");
            if (format == "base64") {
                use_base64 = true;
            } else if (format != "float") {
                res_error(res, format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
                return;
            }
        }

        auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, prompt, true, true);
        for (const auto & tokens : tokenized_prompts) {
            // this check is necessary for models that do not add BOS token to the input
            if (tokens.empty()) {
                res_error(res, format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
                return;
            }
        }

        // create and queue the task
        json responses = json::array();
        bool error = false;
        std::unordered_set<int> task_ids;
        {
            std::vector<server_task> tasks;
            for (size_t i = 0; i < tokenized_prompts.size(); i++) {
                server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

                task.id            = ctx_server.queue_tasks.get_new_id();
                task.index         = i;
                task.prompt_tokens = server_tokens(tokenized_prompts[i], ctx_server.mctx != nullptr);

                // OAI-compat
                task.params.oaicompat = oaicompat;

                tasks.push_back(std::move(task));
            }

            task_ids = server_task::get_list_id(tasks);
            ctx_server.queue_results.add_waiting_tasks(tasks);
            ctx_server.queue_tasks.post(std::move(tasks));
        }

        // get the result
        ctx_server.receive_multi_results(task_ids, [&](std::vector<server_task_result_ptr> & results) {
            for (auto & res : results) {
                GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }, [&](const json & error_data) {
            res_error(res, error_data);
            error = true;
        }, req.is_connection_closed);

        ctx_server.queue_results.remove_waiting_task_ids(task_ids);

        if (error) {
            return;
        }

        // write JSON response
        json root = oaicompat == OAICOMPAT_TYPE_EMBEDDING
            ? format_embeddings_response_oaicompat(body, responses, use_base64)
            : json(responses);
        res_ok(res, root);
    };

    const auto handle_embeddings = [&handle_embeddings_impl](const httplib::Request & req, httplib::Response & res) {
        handle_embeddings_impl(req, res, OAICOMPAT_TYPE_NONE);
    };

    const auto handle_embeddings_oai = [&handle_embeddings_impl](const httplib::Request & req, httplib::Response & res) {
        handle_embeddings_impl(req, res, OAICOMPAT_TYPE_EMBEDDING);
    };

    const auto handle_rerank = [&ctx_server, &res_error, &res_ok](const httplib::Request & req, httplib::Response & res) {
        if (!ctx_server.params_base.embedding || ctx_server.params_base.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res_error(res, format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        const json body = json::parse(req.body);

        // TODO: implement
        //int top_n = 1;
        //if (body.count("top_n") != 1) {
        //    top_n = body.at("top_n");
        //} else {
        //    res_error(res, format_error_response("\"top_n\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        //    return;
        //}

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res_error(res, format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return;
            }
        } else {
            res_error(res, format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res_error(res, format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        llama_tokens tokenized_query = tokenize_input_prompts(ctx_server.vocab, query, /* add_special */ false, true)[0];

        // create and queue the task
        json responses = json::array();
        bool error = false;
        std::unordered_set<int> task_ids;
        {
            std::vector<server_task> tasks;
            auto tokenized_docs = tokenize_input_prompts(ctx_server.vocab, documents, /* add_special */ false, true);
            tasks.reserve(tokenized_docs.size());
            for (size_t i = 0; i < tokenized_docs.size(); i++) {
                auto tmp = format_rerank(ctx_server.vocab, tokenized_query, tokenized_docs[i]);
                server_task task   = server_task(SERVER_TASK_TYPE_RERANK);
                task.id            = ctx_server.queue_tasks.get_new_id();
                task.index         = i;
                task.prompt_tokens = server_tokens(tmp, ctx_server.mctx != nullptr);
                tasks.push_back(std::move(task));
            }

            task_ids = server_task::get_list_id(tasks);
            ctx_server.queue_results.add_waiting_tasks(tasks);
            ctx_server.queue_tasks.post(std::move(tasks));
        }

        ctx_server.receive_multi_results(task_ids, [&](std::vector<server_task_result_ptr> & results) {
            for (auto & res : results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }, [&](const json & error_data) {
            res_error(res, error_data);
            error = true;
        }, req.is_connection_closed);

        if (error) {
            return;
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            responses,
            is_tei_format,
            documents);

        res_ok(res, root);
    };

    const auto handle_lora_adapters_list = [&](const httplib::Request &, httplib::Response & res) {
        json result = json::array();
        const auto & loras = ctx_server.params_base.lora_adapters;
        for (size_t i = 0; i < loras.size(); ++i) {
            auto & lora = loras[i];
            result.push_back({
                {"id", i},
                {"path", lora.path},
                {"scale", lora.scale},
            });
        }
        res_ok(res, result);
        res.status = 200; // HTTP OK
    };

    const auto handle_lora_adapters_apply = [&](const httplib::Request & req, httplib::Response & res) {
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res_error(res, format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        int task_id = ctx_server.queue_tasks.get_new_id();
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = task_id;
            task.set_lora = parse_lora_request(ctx_server.params_base.lora_adapters, body);
            ctx_server.queue_results.add_waiting_task_id(task_id);
            ctx_server.queue_tasks.post(std::move(task));
        }

        // get the result
        server_task_result_ptr result = ctx_server.queue_results.recv(task_id);
        ctx_server.queue_results.remove_waiting_task_id(task_id);

        if (result->is_error()) {
            res_error(res, result->to_json());
            return;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res_ok(res, result->to_json());
    };

    //
    // Router
    //

    if (!params.webui) {
        LOG_INF("Web UI is disabled\n");
    } else {
        // register static assets routes
        if (!params.public_path.empty()) {
            // Set the base directory for serving static files
            bool is_found = svr->set_mount_point("/", params.public_path);
            if (!is_found) {
                LOG_ERR("%s: static assets path not found: %s\n", __func__, params.public_path.c_str());
                return 1;
            }
        } else {
            // using embedded static index.html
            svr->Get("/", [](const httplib::Request & req, httplib::Response & res) {
                if (req.get_header_value("Accept-Encoding").find("gzip") == std::string::npos) {
                    res.set_content("Error: gzip is not supported by this browser", "text/plain");
                } else {
                    res.set_header("Content-Encoding", "gzip");
                    // COEP and COOP headers, required by pyodide (python interpreter)
                    res.set_header("Cross-Origin-Embedder-Policy", "require-corp");
                    res.set_header("Cross-Origin-Opener-Policy", "same-origin");
                    res.set_content(reinterpret_cast<const char*>(index_html_gz), index_html_gz_len, "text/html; charset=utf-8");
                }
                return false;
            });
        }
    }

    // register API routes
    svr->Get ("/health",              handle_health); // public endpoint (no API key check)
    svr->Get ("/metrics",             handle_metrics);
    svr->Get ("/props",               handle_props);
    svr->Post("/props",               handle_props_change);
    svr->Post("/api/show",            handle_api_show);
    svr->Get ("/models",              handle_models); // public endpoint (no API key check)
    svr->Get ("/v1/models",           handle_models); // public endpoint (no API key check)
    svr->Get ("/api/tags",            handle_models); // ollama specific endpoint. public endpoint (no API key check)
    svr->Post("/completion",          handle_completions); // legacy
    svr->Post("/completions",         handle_completions);
    svr->Post("/v1/completions",      handle_completions_oai);
    svr->Post("/chat/completions",    handle_chat_completions);
    svr->Post("/v1/chat/completions", handle_chat_completions);
    svr->Post("/api/chat",            handle_chat_completions); // ollama specific endpoint
    svr->Post("/infill",              handle_infill);
    svr->Post("/embedding",           handle_embeddings); // legacy
    svr->Post("/embeddings",          handle_embeddings);
    svr->Post("/v1/embeddings",       handle_embeddings_oai);
    svr->Post("/rerank",              handle_rerank);
    svr->Post("/reranking",           handle_rerank);
    svr->Post("/v1/rerank",           handle_rerank);
    svr->Post("/v1/reranking",        handle_rerank);
    svr->Post("/tokenize",            handle_tokenize);
    svr->Post("/detokenize",          handle_detokenize);
    svr->Post("/apply-template",      handle_apply_template);
    // LoRA adapters hotswap
    svr->Get ("/lora-adapters",       handle_lora_adapters_list);
    svr->Post("/lora-adapters",       handle_lora_adapters_apply);
    // Save & load slots
    svr->Get ("/slots",               handle_slots);
    svr->Post("/slots/:id_slot",      handle_slots_action);

    //
    // Start the server
    //
    if (params.n_threads_http < 1) {
        // +2 threads for monitoring endpoints
        params.n_threads_http = std::max(params.n_parallel + 2, (int32_t) std::thread::hardware_concurrency() - 1);
    }
    log_data["n_threads_http"] =  std::to_string(params.n_threads_http);
    svr->new_task_queue = [&params] { return new httplib::ThreadPool(params.n_threads_http); };

    // clean up function, to be called before exit
    auto clean_up = [&svr, &ctx_server]() {
        SRV_INF("%s: cleaning up before exit...\n", __func__);
        svr->stop();
        ctx_server.queue_results.terminate();
        llama_backend_free();
    };

    bool was_bound = false;
    bool is_sock = false;
    if (string_ends_with(std::string(params.hostname), ".sock")) {
        is_sock = true;
        LOG_INF("%s: setting address family to AF_UNIX\n", __func__);
        svr->set_address_family(AF_UNIX);
        // bind_to_port requires a second arg, any value other than 0 should
        // simply get ignored
        was_bound = svr->bind_to_port(params.hostname, 8080);
    } else {
        LOG_INF("%s: binding port with default address family\n", __func__);
        // bind HTTP listen port
        if (params.port == 0) {
            int bound_port = svr->bind_to_any_port(params.hostname);
            if ((was_bound = (bound_port >= 0))) {
                params.port = bound_port;
            }
        } else {
            was_bound = svr->bind_to_port(params.hostname, params.port);
        }
    }

    if (!was_bound) {
        LOG_ERR("%s: couldn't bind HTTP server socket, hostname: %s, port: %d\n", __func__, params.hostname.c_str(), params.port);
        clean_up();
        return 1;
    }

    // run the HTTP server in a thread
    std::thread t([&]() { svr->listen_after_bind(); });
    svr->wait_until_ready();

    LOG_INF("%s: HTTP server is listening, hostname: %s, port: %d, http threads: %d\n", __func__, params.hostname.c_str(), params.port, params.n_threads_http);

    // load the model
    LOG_INF("%s: loading model\n", __func__);

    if (!ctx_server.load_model(params)) {
        clean_up();
        t.join();
        LOG_ERR("%s: exiting due to model loading error\n", __func__);
        return 1;
    }

    ctx_server.init();
    state.store(SERVER_STATE_READY);

    LOG_INF("%s: model loaded\n", __func__);

    // print sample chat example to make it clear which template is used
    LOG_INF("%s: chat template, chat_template: %s, example_format: '%s'\n", __func__,
        common_chat_templates_source(ctx_server.chat_templates.get()),
        common_chat_format_example(ctx_server.chat_templates.get(), ctx_server.params_base.use_jinja).c_str());

    ctx_server.queue_tasks.on_new_task([&ctx_server](server_task && task) {
        ctx_server.process_single_task(std::move(task));
    });

    ctx_server.queue_tasks.on_update_slots([&ctx_server]() {
        ctx_server.update_slots();
    });

    shutdown_handler = [&](int) {
        // this will unblock start_loop()
        ctx_server.queue_tasks.terminate();
    };

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    LOG_INF("%s: server is listening on %s - starting the main loop\n", __func__,
            is_sock ? string_format("unix://%s", params.hostname.c_str()).c_str() :
                      string_format("http://%s:%d", params.hostname.c_str(), params.port).c_str());

    // this call blocks the main thread until queue_tasks.terminate() is called
    ctx_server.queue_tasks.start_loop();

    clean_up();
    t.join();

    return 0;
}
