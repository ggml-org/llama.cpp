#include "mtmd.h"
#include "runner.h"
//

#include "arg.h"
#include "base64.hpp"
#include "common.h"
#include "ggml.h"
#include "log.h"

#include <cpp-httplib/httplib.h>
#include <signal.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;

#define MIMETYPE_JSON "application/json; charset=utf-8"

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#    include <unistd.h>
#elif defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

static std::function<void()> g_shutdown;  // Assigned in main()

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || defined(_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT && g_shutdown) {
        g_shutdown();
    }
}
#endif

static void show_additional_info(int /*argc*/, char ** argv) {
    LOG("CLI for LFM2.5-Audio-1.5B\n\n"
        "Usage: %s [options] -m <model.gguf> --mmproj <mmproj.gguf> -mv <vocoder.gguf> --tts-speaker-file "
        "<tokenizer.gguf>\n",
        argv[0]);
}

// Stream state shared between generation thread and content provider
struct StreamState {
    std::unique_lock<std::mutex> request_lock;  // Held until generation completes
    std::mutex                   mutex;
    std::condition_variable      cv;
    std::deque<std::string>      chunks;
    std::atomic<bool>            done{ false };
    std::atomic<bool>            aborted{ false };
    std::thread                  thread;

    explicit StreamState(std::unique_lock<std::mutex> && lock) : request_lock(std::move(lock)) {}

    ~StreamState() {
        if (thread.joinable()) {
            thread.join();
        }
    }

    void push(const std::string & chunk) {
        std::lock_guard<std::mutex> lock(mutex);
        chunks.push_back(chunk);
        cv.notify_one();
    }

    void finish() {
        std::lock_guard<std::mutex> lock(mutex);
        done = true;
        cv.notify_one();
    }
};

int main(int argc, char ** argv) {
    ggml_time_init();

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LIQUID_AUDIO, show_additional_info)) {
        return 1;
    }

    common_init();

    if (params.n_ctx == 0) {
        params.n_ctx = 4096;
    }

    LOG_INF("Loading model\n");
    liquid::audio::Runner runner;
    if (0 != runner.init(params)) {
        return 1;
    }
    LOG_INF("Model loaded successfully!\n");

    httplib::Server svr;
    svr.set_default_headers({
        { "Server", "lfm2-audio-server" }
    });

    std::mutex        request_mutex;  // Serializes requests
    std::atomic<bool> is_server_running(true);

    // Set up shutdown handler (captures locals, called from signal handler)
    g_shutdown = [&]() {
        is_server_running = false;
        runner.stop();
        svr.stop();
    };

    auto res_error = [](httplib::Response & res, const std::string & message, int code = 500) {
        json error_response = {
            { "error", { { "message", message }, { "type", "server_error" }, { "code", code } } }
        };
        res.set_content(error_response.dump(), MIMETYPE_JSON);
        res.status = code;
    };

    // Signal handling
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = sigint_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
#elif defined(_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    // CORS
    svr.set_pre_routing_handler([](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        if (req.method == "OPTIONS") {
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "*");
            res.set_content("", "text/html");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    // Chat completions endpoint
    svr.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        // Acquire lock - will be moved to StreamState to hold until generation completes
        std::unique_lock<std::mutex> request_lock(request_mutex);

        if (!is_server_running.load()) {
            res_error(res, "Server is shutting down", 503);
            return;
        }

        try {
            json body = json::parse(req.body);

            int  n_predict     = body.value("max_tokens", 2048);
            bool stream        = body.value("stream", false);
            bool reset_context = body.value("reset_context", true);

            std::vector<liquid::audio::Runner::Message> messages;

            if (body.contains("messages") && body["messages"].is_array()) {
                for (const auto & msg : body["messages"]) {
                    std::string role    = msg["role"];
                    auto        content = msg["content"];

                    if (role == "system") {
                        messages.push_back({ role, content, {} });
                        continue;
                    }

                    if (role != "user") {
                        res_error(res, "role must be system or user", 400);
                        return;
                    }

                    if (content.is_string()) {
                        messages.push_back({ role, content, {} });
                        continue;
                    }

                    if (!content.is_array()) {
                        res_error(res, "content must be string or array", 400);
                        return;
                    }

                    for (const auto & part : content) {
                        std::string type = part["type"];
                        if (type == "text") {
                            messages.push_back({ role, part["text"], {} });
                            continue;
                        }

                        if (type != "input_audio") {
                            res_error(res, "content type must be either text or input_audio", 400);
                            return;
                        }

                        if (part["input_audio"]["format"] != "wav") {
                            res_error(res, "input_audio format must be wav", 400);
                            return;
                        }

                        std::string          data = part["input_audio"]["data"];
                        std::vector<uint8_t> data_buf;
                        base64::decode(begin(data), end(data), std::back_inserter(data_buf));
                        auto wav_data = std::vector<std::byte>(data_buf.size());
                        memcpy(wav_data.data(), data_buf.data(), data_buf.size());
                        messages.push_back({ role, mtmd_default_marker(), wav_data });
                    }
                }
            }

            if (!stream) {
                res_error(res, "non streaming API is not implemented", 400);
                return;
            }

            // Create shared stream state - takes ownership of the request lock
            auto state = std::make_shared<StreamState>(std::move(request_lock));

            auto check_abort = [&req, state, &runner, &is_server_running]() {
                if (state->aborted.load()) {
                    return;
                }
                bool should_abort = !is_server_running.load();
                if (!should_abort && req.is_connection_closed) {
                    should_abort = req.is_connection_closed();
                }
                if (should_abort && !state->aborted.exchange(true)) {
                    LOG_INF("Aborting generation\n");
                    runner.stop();
                }
            };

            auto text_cb = [state, check_abort, &runner](const std::string & text) {
                check_abort();
                if (state->aborted.load()) {
                    return;
                }
                json chunk = {
                    { "object",  "chat.completion.chunk"                                                              },
                    { "created", std::time(0)                                                                         },
                    { "choices",
                     json::array(
                          { { { "index", 0 }, { "delta", { { "content", text } } }, { "finish_reason", nullptr } } }) }
                };
                state->push("data: " + chunk.dump() + "\n\n");
            };

            auto audio_cb = [state, check_abort, &runner](const std::vector<float> & audio) {
                check_abort();
                if (state->aborted.load()) {
                    return;
                }
                std::string audio_base64 =
                    base64::encode(reinterpret_cast<const char *>(audio.data()), audio.size() * sizeof(audio.front()));
                json chunk = {
                    { "object",  "chat.completion.chunk"                                                           },
                    { "created", std::time(0)                                                                      },
                    { "choices", json::array({ { { "index", 0 },
                                                 { "delta",
                                                   { { "audio_chunk",
                                                       { { "data", audio_base64 },
                                                         { "format", "pcm" },
                                                         { "sample_rate", runner.get_output_sample_rate() } } } } },
                                                 { "finish_reason", nullptr } } }) }
                };
                state->push("data: " + chunk.dump() + "\n\n");
            };

            // Start generation in background thread
            state->thread = std::thread([state, &runner, messages, n_predict, reset_context, text_cb, audio_cb]() {
                if (reset_context) {
                    LOG_INF("Resetting model context\n");
                    runner.reset();
                }

                std::optional<std::string> err;
                if (runner.generate(messages, n_predict, text_cb, audio_cb)) {
                    err = runner.get_last_error();
                }

                if (!state->aborted.load()) {
                    if (err) {
                        json error_chunk = {
                            { "error", { { "message", *err }, { "type", "server_error" } } }
                        };
                        state->push("data: " + error_chunk.dump() + "\n\n");
                    } else {
                        json final_chunk = {
                            { "object",  "chat.completion.chunk"                                                    },
                            { "created", std::time(0)                                                               },
                            { "choices",
                             json::array(
                                  { { { "index", 0 }, { "delta", json::object() }, { "finish_reason", "stop" } } }) }
                        };
                        state->push("data: " + final_chunk.dump() + "\n\n");
                        state->push("data: [DONE]\n\n");
                    }
                }

                state->finish();
            });

            // Stream chunks as they arrive
            res.set_content_provider(
                "text/event-stream", [state, &is_server_running](size_t, httplib::DataSink & sink) {
                    std::unique_lock<std::mutex> lock(state->mutex);

                    // Wait for data or completion
                    state->cv.wait_for(lock, std::chrono::milliseconds(100), [&state, &is_server_running]() {
                        return !state->chunks.empty() || state->done.load() || state->aborted.load() ||
                               !is_server_running.load();
                    });

                    // Check if we should abort
                    if (state->aborted.load() || !is_server_running.load()) {
                        return false;
                    }

                    // Send all available chunks
                    while (!state->chunks.empty()) {
                        const std::string & data = state->chunks.front();
                        if (!sink.write(data.c_str(), data.size())) {
                            state->aborted = true;
                            return false;
                        }
                        state->chunks.pop_front();
                    }

                    // Done?
                    if (state->done.load() && state->chunks.empty()) {
                        sink.done();
                        return false;
                    }

                    return true;
                });

            res.status = 200;

        } catch (const std::exception & e) {
            res_error(res, std::string("Error processing request: ") + e.what(), 500);
        }
    });

    LOG_INF("Starting HTTP server on %s:%d\n", params.hostname.c_str(), params.port);

    if (!svr.bind_to_port(params.hostname, params.port)) {
        LOG_ERR("Failed to bind to %s:%d\n", params.hostname.c_str(), params.port);
        return 1;
    }

    LOG_INF("Server ready at http://%s:%d\n", params.hostname.c_str(), params.port);
    svr.listen_after_bind();

    LOG_INF("\nShutting down...\n");
    g_shutdown = nullptr;  // Clear before locals go out of scope

    return 0;
}
