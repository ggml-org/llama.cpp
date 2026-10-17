#include "server-routes.h"
#include "engine-chat.h"
#include "engine-common.h"
#include "engine-decision.h"
#include "server-http.h"
#include "engine-task.h"
#include "engine-queue.h"
#include "engine-schema.h"
#include "server-stream.h"

#include "build-info.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <utility>

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
    engine_response_reader rd;
    server_res_generator(engine_queue & queue_tasks, engine_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            engine_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type,
            const common_chat_session & chat_session) {
    GGML_ASSERT(type == ENGINE_TASK_TYPE_COMPLETION || type == ENGINE_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    res->set_req(&req); // will also set spipe if needed

    int32_t sse_ping_interval = params.sse_ping_interval;

    try {
        std::vector<engine_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<engine_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.get_mtmd_context() != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.get_mtmd_context(), prompt.get<std::string>(), files, ctx_server.get_mtmd_init_opt()));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.get_vocab(), ctx_server.get_mtmd_context(), prompt, true, true, ctx_server.get_mtmd_init_opt());
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        for (size_t i = 0; i < inputs.size(); i++) {
            engine_task task = engine_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = engine_schema::eval_llama_cmpl_schema(
                    ctx_server.get_vocab(),
                    params,
                    meta->logit_bias_eog,
                    data);

            task.apply_chat_session(chat_session);

            task.id_slot = json_value(data, "id_slot", -1);
            sse_ping_interval = task.params.sse_ping_interval;

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks), chat_session);
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<engine_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<engine_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<engine_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            auto effective_should_stop = [&res_this]() {
                return res_this->should_stop();
            };

            try {
                if (effective_should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                engine_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true; // should_stop condition met
                    } else if (sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(effective_should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<engine_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<engine_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, engine_context & ctx_server)
        : params(params),
          ctx_server(ctx_server),
          queue_tasks(ctx_server.get_queue_tasks()),
          queue_results(ctx_server.get_queue_results()) {
    init_routes();

    // note: this must be registered before load_model()
    //       so that on sleep phase, the callback is called before ctx is destroyed
    queue_tasks.on_sleeping_state([this](bool is_sleeping) {
        update_cached_responses(is_sleeping);
    });
}

static json get_res_model_info(const engine_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return {
        {"id",       meta.model_name},
        {"aliases",  meta.model_aliases},
        {"tags",     meta.model_tags},
        {"object",   "model"},
        {"architecture", engine_model_architecture_json(
            meta.has_inp_image,
            meta.has_inp_audio,
            meta.has_inp_video,
            meta.model_output_modalities)},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta.model_vocab_type},
            {"n_vocab",     meta.model_vocab_n_tokens},
            {"n_ctx",       meta.slot_n_ctx},
            {"n_ctx_train", meta.model_n_ctx_train},
            {"n_embd",      meta.model_n_embd_inp},
            {"n_params",    meta.model_n_params},
            {"size",        meta.model_size},
            {"ftype",       meta.model_ftype},
        }},
    };
}

static json get_res_models(const engine_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return json{
        {"models", json::array({
            {
                {"name",  meta.model_name},
                {"model", meta.model_name},
                {"modified_at", ""},
                {"size", ""},
                {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                {"type", "model"},
                {"description", ""},
                {"tags", json::array({""})},
                {"capabilities", meta.has_mtmd ? json::array({"completion","multimodal"}) : json::array({"completion"})},
                {"parameters", ""},
                {"details", {
                    {"parent_model", ""},
                    {"format", "gguf"},
                    {"family", ""},
                    {"families", json::array({""})},
                    {"parameter_size", ""},
                    {"quantization_level", ""}
                }}
            }
        })},
        {"object", "list"},
        {"data", json::array({
            get_res_model_info(meta),
        })}
    };
}

static json get_res_props(const engine_context_meta & meta, const common_params & params, bool is_sleeping) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    task_params tparams;
    tparams.sampling = params.sampling;
    json default_generation_settings_for_props = json {
        { "params", tparams.to_json(true) },
        { "n_ctx",  meta.slot_n_ctx },
    };

    std::string tmpl_default = common_chat_templates_source(meta.chat_params.tmpls.get(), "");
    std::string tmpl_tools   = common_chat_templates_source(meta.chat_params.tmpls.get(), "tool_use");

    json props = {
        { "default_generation_settings", default_generation_settings_for_props },
        { "total_slots",                 params.n_parallel },
        { "model_alias",                 meta.model_name },
        { "model_ftype",                 meta.model_ftype },
        { "model_path",                  meta.model_path },
        { "modalities",                  json {
            {"vision", meta.has_inp_image},
            {"video",  meta.has_inp_video},
            {"audio",  meta.has_inp_audio},
        } },
        { "media_marker",                get_media_marker() },
        { "endpoint_slots",              params.endpoint_slots },
        { "endpoint_props",              params.endpoint_props },
        { "endpoint_metrics",            params.endpoint_metrics },
        { "ui",                          params.ui },
        { "ui_settings",                 meta.json_ui_settings },
        { "chat_template",               tmpl_default },
        { "chat_template_caps",          meta.chat_template_caps },
        { "bos_token",                   meta.bos_token_str },
        { "eos_token",                   meta.eos_token_str },
        { "build_info",                  meta.build_info },
        { "is_sleeping",                 is_sleeping },
        { "cors_proxy_enabled",          params.ui_mcp_proxy },
    };
    if (params.use_jinja) {
        if (!tmpl_tools.empty()) {
            props["chat_template_tool_use"] = tmpl_tools;
        }
    }

    return props;
}

json server_routes::get_model_info() const {
    return get_res_model_info(*meta);
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response(true);
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // render response using cached_metrics
        auto use_cached_metrics = [&]() {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->headers["Process-Start-Time-Unix"] = std::to_string(cached_metrics.t_start);
            engine_task_result_metrics tmp;
            tmp.metrics = cached_metrics;
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = tmp.to_metrics();
            // the gauges are averaged over the window between two scrapes
            cached_metrics.reset_bucket();
            should_reset_buckets = true;
        };

        if (queue_tasks.is_sleeping()) {
            use_cached_metrics();

        } else {
            // request slots data using task queue
            {
                engine_task task(ENGINE_TASK_TYPE_METRICS);
                task.id = res->rd.get_new_id();
                // the gauges are averaged over the window between two scrapes
                task.metrics_reset_bucket = true;
                res->rd.post_task(std::move(task), true); // high-priority task
            }

            // a task posted right before sleeping is never processed, do not wait for it
            auto result = res->rd.next([&]{
                return req.should_stop() || queue_tasks.is_sleeping();
            });
            if (!result) {
                if (!req.should_stop()) {
                    use_cached_metrics();
                }
                return res;
            }

            if (result->is_error()) {
                res->error(result->to_json());
                return res;
            }

            auto res_task = dynamic_cast<engine_task_result_metrics*>(result.get());
            GGML_ASSERT(res_task != nullptr);

            res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->metrics.t_start);
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = res_task->to_metrics();
        }

        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            engine_task task(ENGINE_TASK_TYPE_SLOT_GET);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto * res_task = dynamic_cast<engine_task_result_slots*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->to_json());
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_props);
        } else {
            res->ok(get_res_props(*meta, params, false));
        }
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.get_vocab()) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.get_vocab()) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.get_vocab()) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<engine_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.get_vocab(), ctx_server.get_mtmd_context(), prompt, false, true, ctx_server.get_mtmd_init_opt());
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.get_vocab(),
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            ENGINE_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            ENGINE_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            ENGINE_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.get_vocab(),
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            ENGINE_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT,
            session);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            engine_task task(ENGINE_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = action;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = engine_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.get_vocab(),
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            ENGINE_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP,
            session);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.get_vocab(),
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            ENGINE_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR,
            session);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = engine_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.get_vocab(),
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            ENGINE_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC,
            session);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        common_chat_session session;   // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            ctx_server.get_vocab(),
            body,
            meta->chat_params,
            files,
            session);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_models);
        } else {
            res->ok(get_res_models(*meta));
        }
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.get_vocab(), body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.get_vocab(), token);
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

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens").get<llama_tokens>();
            content = tokens_to_str(ctx_server.get_vocab(), tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<engine_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.get_model(), ctx_server.get_vocab(), ctx_server.get_mtmd_context(), query, documents[i], ctx_server.get_mtmd_init_opt());
                engine_task task = engine_task(ENGINE_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<engine_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->post_systemone = [this](const server_http_req & req) {
        auto res = create_response();
        const auto & decision = ctx_server.get_decision();
        if (decision.type == COMMON_DECISION_TYPE_NONE) {
            res->error(format_error_response("This model is not a decision model", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);
        const auto questions = decision.parse_questions(body);

        std::vector<raw_buffer> files;
        const json state = decision.parse_state(body, files);
        if (!files.empty() && (!decision.can_use_images() || !meta->has_inp_image)) {
            res->error(format_error_response("This server does not support image input for decisions. For a model that supports it, start it with `--mmproj`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // one task per variant of each question, or one task for all the questions
        auto & rd = res->rd;
        {
            std::vector<engine_task> tasks;
            if (decision.is_joint()) {
                engine_task task = engine_task(ENGINE_TASK_TYPE_DECISION);
                task.id = rd.get_new_id();
                decision.fill_task_joint(state, questions, files, ctx_server.get_mtmd_context(), ctx_server.get_mtmd_init_opt(), task);
                tasks.push_back(std::move(task));
            } else {
                for (const auto & question : questions) {
                    for (size_t variant = 0; variant < decision.n_variants(question); variant++) {
                        engine_task task = engine_task(ENGINE_TASK_TYPE_DECISION);
                        task.id = rd.get_new_id();
                        decision.fill_task(state, questions, question, variant, files, ctx_server.get_mtmd_context(), ctx_server.get_mtmd_init_opt(), task);
                        tasks.push_back(std::move(task));
                    }
                }
            }
            if (decision.can_share_prompt()) {
                tasks = engine_decision_group_tasks(std::move(tasks), params.n_parallel);
            }
            rd.post_tasks(std::move(tasks));
        }

        auto all_results = rd.wait_for_all(req.should_stop);

        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        }

        json answers = json::object();
        int32_t n_tokens = 0;
        size_t i_result = 0;
        size_t i_score  = 0;
        for (const auto & question : questions) {
            std::vector<std::vector<float>> scores;
            if (decision.is_joint()) {
                // one result with the scores of all the questions, in order
                auto * result = dynamic_cast<engine_task_result_decision *>(all_results.results[0].get());
                GGML_ASSERT(result != nullptr && i_score + question.options.size() <= result->scores.size());
                scores.emplace_back(result->scores.begin() + i_score, result->scores.begin() + i_score + question.options.size());
                i_score += question.options.size();
                n_tokens = result->n_tokens;
            }
            for (size_t variant = 0; !decision.is_joint() && variant < decision.n_variants(question); variant++) {
                auto * result = dynamic_cast<engine_task_result_decision *>(all_results.results[i_result++].get());
                GGML_ASSERT(result != nullptr);
                scores.push_back(result->scores);
                n_tokens += result->n_tokens;
            }
            answers[question.id] = decision.format_answer(question, scores, !files.empty());
        }

        res->ok(json{
            {"model",   meta->model_name},
            {"answers", answers},
            {"usage",   {
                {"input_tokens",  n_tokens},
                {"output_tokens", 0},
            }},
        });
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            engine_task task(ENGINE_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<engine_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            engine_task task(ENGINE_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<engine_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        engine_task task(ENGINE_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        engine_task task(ENGINE_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<engine_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        engine_task task(ENGINE_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<engine_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    // same shapes as tokenize_input_prompts(), plus OAI content: { "content": [ { "type": "text"|"image_url"|"input_audio"|"input_video", ... } ] }
    auto tokenize_entry = [&](const json & p) {
        if (p.is_object() && p.contains("content")) {
            return tokenize_oai_content_array(ctx_server.get_vocab(), ctx_server.get_mtmd_context(), meta->chat_params, p.at("content"), true, true, ctx_server.get_mtmd_init_opt());
        }
        return tokenize_input_subprompt(ctx_server.get_vocab(), ctx_server.get_mtmd_context(), p, true, true, ctx_server.get_mtmd_init_opt());
    };

    std::vector<engine_tokens> tokenized_prompts;
    if (prompt.is_array() && !json_is_array_and_contains_numbers(prompt)) {
        for (const auto & p : prompt) {
            tokenized_prompts.push_back(tokenize_entry(p));
        }
    } else {
        tokenized_prompts.push_back(tokenize_entry(prompt));
    }
    if (tokenized_prompts.empty()) {
        res->error(format_error_response("\"input\" must not be empty", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize").get<int>();
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<engine_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            engine_task task = engine_task(ENGINE_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<engine_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = engine_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = engine_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    common_chat_session session; // dummy, unused
    json body_parsed = oaicompat_chat_params_parse(
            ctx_server.get_vocab(),
            body,
            meta->chat_params,
            files,
            session);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (ctx_server.get_mtmd_context() != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(ctx_server.get_mtmd_context(), prompt.get<std::string>(), files, ctx_server.get_mtmd_init_opt(), true).size();
    } else {
        n_tokens = tokenize_mixed(ctx_server.get_vocab(), prompt, true, true).size();
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}

void server_routes::update_cached_responses(bool is_sleeping) {
    // caller is task_queue, so ctx_server can be accessed without holding locks
    std::unique_lock<std::mutex> lock(mutex_cache);

    if (is_sleeping) {
        cached_models  = get_res_models(*meta);
        cached_props   = get_res_props(*meta, params, true);
        cached_metrics = ctx_server.get_metrics();

        should_reset_buckets = false;

        SRV_DBG("%s\n", "cached responses updated");

    } else if (should_reset_buckets) {
        // a scrape during sleep already reported these buckets
        ctx_server.reset_metrics_bucket();

        should_reset_buckets = false;
    }
}
