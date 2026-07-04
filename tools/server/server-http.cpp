#include "common.h"
#include "http.h"
#include "server-http.h"
#include "server-stream.h"
#include "server-common.h"
#include "ui.h"

#include <cpp-httplib/httplib.h>

#include <functional>
#include <future>
#include <memory>
#include <string>
#include <thread>

//
// HTTP implementation using cpp-httplib
//

class server_http_context::Impl {
public:
    std::unique_ptr<httplib::Server> srv;
};

server_http_context::server_http_context()
    : pimpl(std::make_unique<Impl>())
{}

server_http_context::~server_http_context() = default;

static void log_server_request(const httplib::Request & req, const httplib::Response & res) {
    // skip logging requests that are regularly sent, to avoid log spam
    if (req.path == "/health"
        || req.path == "/v1/health"
        || req.path == "/models"
        || req.path == "/v1/models"
        || req.path == "/props"
        || req.path == "/metrics"
    ) {
        return;
    }

    // reminder: this function is not covered by httplib's exception handler; if someone does more complicated stuff, think about wrapping it in try-catch

    SRV_TRC("done request: %s %s %s %d\n", req.method.c_str(), req.path.c_str(), req.remote_addr.c_str(), res.status);

    SRV_DBG("request:  %s\n", req.body.c_str());
    SRV_DBG("response: %s\n", res.body.c_str());
}

// For Google Cloud Platform deployment compatibility
struct gcp_params {
    bool enabled;
    std::string path_health;
    std::string path_predict;
    int port;

    // Ref: https://docs.cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#aip-variables
    gcp_params() {
        enabled = getenv("AIP_MODE", "") == "PREDICTION";
        path_health = getenv("AIP_HEALTH_ROUTE", "", true); // default: using the route defined in server.cpp
        path_predict = getenv("AIP_PREDICT_ROUTE", "/predict", true);
        port = std::stoi(getenv("AIP_HTTP_PORT", "8080"));
    }

    static std::string getenv(const char * name, const std::string & default_value, bool ensure_leading_slash = false) {
        const auto * value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') {
            return default_value;
        }
        std::string val = value;
        if (ensure_leading_slash && !val.empty() && val[0] != '/') {
            val.insert(val.begin(), '/');
        }
        return val;
    }
};

bool server_http_context::init(const common_params & params) {
    const gcp_params gcp;

    path_prefix = params.api_prefix;
    port = params.port;
    hostname = params.hostname;

    if (gcp.enabled) {
        SRV_TRC("Google Cloud Platform compat: health route = %s, predict route = %s, port = %d\n", gcp.path_health.c_str(), gcp.path_predict.c_str(), gcp.port);

        if (port != gcp.port) {
            SRV_WRN("Google Cloud Platform compat: overriding server port %d with AIP_HTTP_PORT %d\n", port, gcp.port);
        }

        port = gcp.port;
    }

    auto & srv = pimpl->srv;

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if (!params.ssl_file_key.empty() && !params.ssl_file_cert.empty()) {
        SRV_TRC("running with SSL: key = %s, cert = %s\n", params.ssl_file_key.c_str(), params.ssl_file_cert.c_str());
        srv = std::make_unique<httplib::SSLServer>(
            params.ssl_file_cert.c_str(), params.ssl_file_key.c_str()
        );
        is_ssl = true;
    } else {
        SRV_TRC("%s", "running without SSL\n");
        srv = std::make_unique<httplib::Server>();
    }
#else
    if (params.ssl_file_key != "" && params.ssl_file_cert != "") {
        SRV_ERR("%s", "the server is built without SSL support\n");
        return false;
    }
    srv.reset(new httplib::Server());
#endif

    srv->set_default_headers({{"Server", "llama.cpp"}});
    // srv->set_logger(log_server_request); // TODO @ngxson : this is too spamy, no very useful; improve it in the future
    srv->set_exception_handler([](const httplib::Request &, httplib::Response & res, const std::exception_ptr & ep) {
        // this is fail-safe; exceptions should already handled by `ex_wrapper`

        std::string message;
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception & e) {
            message = e.what();
        } catch (...) {
            message = "Unknown Exception";
        }

        res.status = 500;
        res.set_content(message, "text/plain");
        SRV_ERR("got exception: %s\n", message.c_str());
    });

    srv->set_error_handler([](const httplib::Request &, httplib::Response & res) {
        if (res.status == 404) {
            res.set_content(
                safe_json_to_str(json {
                    {"error", {
                        {"message", "File Not Found"},
                        {"type", "not_found_error"},
                        {"code", 404}
                    }}
                }),
                "application/json; charset=utf-8"
            );
        }
        // for other error codes, we skip processing here because it's already done by res->error()
    });

    // set timeouts and change hostname and port
    srv->set_read_timeout (params.timeout_read);
    srv->set_write_timeout(params.timeout_write);
    srv->set_socket_options([reuse_port = params.reuse_port](const socket_t sock) {
        httplib::set_socket_opt(sock, SOL_SOCKET, SO_REUSEADDR, 1);
        if (reuse_port) {
#ifdef SO_REUSEPORT
            httplib::set_socket_opt(sock, SOL_SOCKET, SO_REUSEPORT, 1);
#else
            SRV_WRN("%s", "SO_REUSEPORT is not supported\n");
#endif
        }
    });

    if (params.api_keys.size() == 1) {
        const auto key = params.api_keys[0];
        const std::string substr = key.substr(std::max(static_cast<int>(key.length() - 4), 0));
        SRV_TRC("api_keys: ****%s\n", substr.c_str());
    } else if (params.api_keys.size() > 1) {
        SRV_TRC("api_keys: %zu keys loaded\n", params.api_keys.size());
    }

    //
    // Middlewares
    //

    // Public endpoints - API routes plus all embedded UI assets
    static const std::unordered_set<std::string> get_public_endpoints = []() {
        std::unordered_set<std::string> endpoints {
            "/health",
            "/v1/health",
            "/models",
            "/v1/models",
            "/",
        };
        for (const llama_ui_asset & a : llama_ui_get_assets()) {
            endpoints.insert("/" + a.name);
        }
        return endpoints;
    }();

    auto middleware_validate_api_key = [api_keys = params.api_keys](const httplib::Request & req, httplib::Response & res) {
        // If API key is not set, skip validation
        if (api_keys.empty()) {
            return true;
        }

        // If path is public or a UI asset, skip validation
        if (get_public_endpoints.count(req.path)) {
            return true;
        }

        // Check for API key in the Authorization header
        std::string req_api_key = req.get_header_value("Authorization");
        if (req_api_key.empty()) {
            // retry with anthropic header
            req_api_key = req.get_header_value("X-Api-Key");
        }

        // remove the "Bearer " prefix if needed
        static std::string prefix = "Bearer ";
        if (req_api_key.substr(0, prefix.size()) == prefix) {
            req_api_key = req_api_key.substr(prefix.size());
        }

        // validate the API key
        if (std::find(api_keys.begin(), api_keys.end(), req_api_key) != api_keys.end()) {
            return true; // API key is valid
        }

        // API key is invalid or not provided
        res.status = 401;
        res.set_content(
            safe_json_to_str(json {
                {"error", {
                    {"message", "Invalid API Key"},
                    {"type", "authentication_error"},
                    {"code", 401}
                }}
            }),
            "application/json; charset=utf-8"
        );

        SRV_WRN("%s", "unauthorized: Invalid API Key\n");

        return false;
    };

    auto middleware_server_state = [this](const httplib::Request & req, httplib::Response & res) {
        if (!is_ready.load()) {
#if defined(LLAMA_UI_HAS_ASSETS)
            if (const auto tmp = string_split<std::string>(req.path, '.');
                req.path == "/" || (!tmp.empty() && tmp.back() == "html")) {
                if (const llama_ui_asset * a = llama_ui_find_asset("loading.html")) {
                    res.status = 503;
                    res.set_content(reinterpret_cast<const char*>(a->data), a->size, "text/html; charset=utf-8");
                    return false;
                }
            }
#else
            (void)req;
#endif
            // no endpoints are allowed to be accessed when the server is not ready
            // this is to prevent any data races or inconsistent states
            res.status = 503;
            res.set_content(
                safe_json_to_str(json {
                    {"error", {
                        {"message", "Loading model"},
                        {"type", "unavailable_error"},
                        {"code", 503}
                    }}
                }),
                "application/json; charset=utf-8"
            );
            return false;
        }
        return true;
    };

    // register server middlewares
    srv->set_pre_routing_handler([middleware_validate_api_key, middleware_server_state](const httplib::Request & req, httplib::Response & res) {
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

    auto n_threads_http = params.n_threads_http;
    if (n_threads_http < 1) {
        // +4 threads for monitoring, health and some threads reserved for MCP and other tasks in the future
        n_threads_http = std::max(params.n_parallel + 4, static_cast<int32_t>(std::thread::hardware_concurrency() - 1));
    }
    SRV_TRC("using %d threads for HTTP server\n", n_threads_http);
    srv->new_task_queue = [n_threads_http] {
        // spawn n_threads_http fixed thread (always alive), while allow up to 1024 max possible additional threads
        // when n_threads_http is used, server will create new "dynamic" threads that will be destroyed after processing each request
        // ref: https://github.com/yhirose/cpp-httplib/pull/2368
        const auto max_threads = static_cast<size_t>(n_threads_http + 1024);
        return new httplib::ThreadPool(n_threads_http, max_threads);
    };

    //
    // Web UI setup
    //

    // Use new `params.ui` field (backed by old `params.webui` for compat)
    if (!params.ui) {
        SRV_INF("%s", "The UI is disabled\n");
        SRV_INF("%s", "Use --ui/--no-ui (or deprecated --webui/--no-webui) to enable/disable\n");
    } else {
        // register static assets routes
        if (!params.public_path.empty()) {
            // Set the base directory for serving static files
            if (const auto is_found = srv->set_mount_point(params.api_prefix + "/", params.public_path); !is_found) {
                SRV_ERR("static assets path not found: %s\n", params.public_path.c_str());
                return false;
            }
        } else {
#if defined(LLAMA_UI_HAS_ASSETS)
            static auto handle_gzip_header = [](const httplib::Request & req, httplib::Response & res) {
                if (!llama_ui_use_gzip()) {
                    // no gzip build, skip
                    return true;
                }
                if (req.get_header_value("Accept-Encoding").find("gzip") == std::string::npos) {
                    res.status = 415; // unsupported media type
                    res.set_content("Error: gzip is not supported by this browser", "text/plain");
                    return false;
                } else {
                    res.set_header("Content-Encoding", "gzip");
                }
                return true;
            };

            auto serve_asset_cached = [](const std::string & name, bool isolation) {
                return [name, isolation](const httplib::Request & req, httplib::Response & res) {
                    if (!handle_gzip_header(req, res)) {
                        return true; // returns error message
                    }
                    const llama_ui_asset * a = llama_ui_find_asset(name);
                    if (!a) { res.status = 404; return false; }
                    res.set_header("ETag", a->etag);
                    if (const std::string & inm = req.get_header_value("If-None-Match");
                        !inm.empty() && (inm == a->etag || inm == std::string("W/") + a->etag)) {
                        res.status = 304;
                        return false;
                    }
                    if (isolation) {
                        res.set_header("Cross-Origin-Embedder-Policy", "require-corp");
                        res.set_header("Cross-Origin-Opener-Policy",   "same-origin");
                    }
                    res.set_header("Cache-Control", "public, max-age=31536000, immutable");
                    res.set_content(reinterpret_cast<const char*>(a->data), a->size, a->type.c_str());
                    return false;
                };
            };

            auto serve_asset_nocache = [](const std::string & name) {
                return [name](const httplib::Request & req, httplib::Response & res) {
                    if (!handle_gzip_header(req, res)) {
                        return true; // returns error message
                    }
                    const llama_ui_asset * a = llama_ui_find_asset(name);
                    if (!a) {
                        res.status = 404;
                        return false;
                    }
                    res.set_header("Cache-Control", "no-cache");
                    res.set_content(reinterpret_cast<const char*>(a->data), a->size, a->type.c_str());
                    return false;
                };
            };

            // main index file
            srv->Get(params.api_prefix + "/",           serve_asset_cached("index.html", true));
            srv->Get(params.api_prefix + "/index.html", serve_asset_cached("index.html", true));

            // All remaining assets registered directly from the embedded asset table.
            // PWA revalidation files (sw.js, manifest, version.json) use no-cache;
            // everything else is immutable.
            static const std::unordered_set<std::string> no_cache_names = {
                "sw.js",
                "manifest.webmanifest",
                "_app/version.json",
                "build.json"
            };

            for (const auto & a : llama_ui_get_assets()) {
                if (a.name == "index.html") continue;  // served at "/" and "/index.html" above
                if (no_cache_names.count(a.name)) {
                    SRV_DBG("serve nocache for %s\n", a.name.c_str());
                    srv->Get(params.api_prefix + "/" + a.name, serve_asset_nocache(a.name));
                } else {
                    srv->Get(params.api_prefix + "/" + a.name, serve_asset_cached(a.name, false));
                }
            }

#endif
        }
    }
    return true;
}

bool server_http_context::start() {
    // Bind and listen

    const auto & srv = pimpl->srv;
    auto was_bound = false;
    auto is_sock = false;
    if (string_ends_with(std::string(hostname), ".sock")) {
        is_sock = true;
        SRV_TRC("%s", "setting address family to AF_UNIX\n");
        srv->set_address_family(AF_UNIX);
        // bind_to_port requires a second arg, any value other than 0 should
        // simply get ignored
        was_bound = srv->bind_to_port(hostname, 8080);
    } else {
        SRV_TRC("%s", "binding port with default address family\n");
        // bind HTTP listen port
        if (port == 0) {
            const auto bound_port = srv->bind_to_any_port(hostname);
            was_bound = (bound_port >= 0);
            if (was_bound) {
                port = bound_port;
            }
        } else {
            was_bound = srv->bind_to_port(hostname, port);
        }
    }

    if (!was_bound) {
        SRV_ERR("couldn't bind HTTP server socket, hostname: %s, port: %d\n", hostname.c_str(), port);
        return false;
    }

    // run the HTTP server in a thread
    thread = std::thread([this] { pimpl->srv->listen_after_bind(); });
    srv->wait_until_ready();

    listening_address = is_sock ? string_format("unix://%s", hostname.c_str())
                                : string_format("%s://%s:%d", is_ssl ? "https" : "http", common_http_format_host(hostname).c_str(), port);
    return true;
}

void server_http_context::stop() const {
    if (pimpl->srv) {
        pimpl->srv->stop();
    }
}

static void set_headers(httplib::Response & res, const std::map<std::string, std::string> & headers) {
    for (const auto & [key, value] : headers) {
        res.set_header(key, value);
    }
}

// percent-decode a path component (%XX). path params arrive raw from httplib, unlike query
// params, so a conv id like "conv::model" sent as "conv%3A%3Amodel" must be decoded here to
// match the value the client put in the X-Conversation-Id header
static std::string decode_path_component(const std::string & in) {
    std::string out;
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); i++) {
        if (in[i] == '%' && i + 2 < in.size()) {
            auto hex = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                return -1;
            };
            int hi = hex(in[i + 1]);
            int lo = hex(in[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out.push_back(char((hi << 4) | lo));
                i += 2;
                continue;
            }
        }
        out.push_back(in[i]);
    }
    return out;
}

static std::map<std::string, std::string> get_params(const httplib::Request & req) {
    std::map<std::string, std::string> params;
    for (const auto & [key, value] : req.params) {
        params[key] = value;
    }
    for (const auto & [key, value] : req.path_params) {
        params[key] = decode_path_component(value);
    }
    return params;
}

static std::map<std::string, std::string> get_headers(const httplib::Request & req) {
    std::map<std::string, std::string> headers;
    for (const auto & [key, value] : req.headers) {
        headers[key] = value;
    }
    return headers;
}

static std::string build_query_string(const httplib::Request & req) {
    std::string qs;
    for (const auto & [key, value] : req.params) {
        if (!qs.empty()) {
            qs += '&';
        }
        qs += httplib::encode_query_component(key) + "=" + httplib::encode_query_component(value);
    }
    return qs;
}

// using unique_ptr for request to allow safe capturing in lambdas
using server_http_req_ptr = std::unique_ptr<server_http_req>;

static void process_handler_response(server_http_req_ptr && request, server_http_res_ptr & response, httplib::Response & res) {
    if (response->is_stream()) {
        res.status = response->status;
        // Tell Nginx to not buffer any streamed response
        response->headers["X-Accel-Buffering"] = "no";
        set_headers(res, response->headers);
        const std::string content_type = response->content_type;
        // convert to shared_ptr as both chunked_content_provider() and on_complete() need to use it
        std::shared_ptr<server_http_req> q_ptr = std::move(request);
        std::shared_ptr<server_http_res> r_ptr = std::move(response);

        const auto chunked_content_provider = [response = r_ptr](size_t, httplib::DataSink & sink) -> bool {
            std::string chunk;
            const bool has_next = response->next(chunk);
            if (!chunk.empty()) {
                // mirror into the ring buffer first, the session must reflect every SSE chunk
                // whether or not the wire write below succeeds
                if (response->spipe) {
                    response->spipe->write(chunk.data(), chunk.size());
                }
                if (!sink.write(chunk.data(), chunk.size())) {
                    // peer is gone, stop the wire path here
                    return false;
                }
                SRV_DBG("http: streamed chunk: %s\n", chunk.c_str());
            }
            if (!has_next) {
                // producer reached its natural end on the wire, a later close() skips the drain
                if (response->spipe) {
                    response->spipe->done();
                }
                sink.done();
                SRV_DBG("%s", "http: stream ended\n");
            }
            return has_next;
        };
        const auto on_complete = [request = q_ptr, response = r_ptr](bool) mutable {
            // on a dropped peer, close() drains the rest of the generation into the ring buffer
            if (response->spipe) {
                response->spipe->close();
            }
            response.reset(); // spipe destructor finalizes the session if attached
            request.reset();
        };
        res.set_chunked_content_provider(content_type, chunked_content_provider, on_complete);
    } else {
        res.status = response->status;
        set_headers(res, response->headers);
        res.set_content(response->data, response->content_type);
    }
}

void server_http_context::get(const std::string & path, const server_http_context::handler_t & handler) const {
    handlers.emplace(path, handler);
    pimpl->srv->Get(path_prefix + path, [handler](const httplib::Request & req, httplib::Response & res) {
        server_http_req_ptr request = std::make_unique<server_http_req>(server_http_req{
            get_params(req),
            get_headers(req),
            req.path,
            build_query_string(req),
            req.body,
            {},
            req.is_connection_closed
        });
        server_http_res_ptr response = handler(*request);
        process_handler_response(std::move(request), response, res);
    });
}

void server_http_context::post(const std::string & path, const server_http_context::handler_t & handler) const {
    handlers.emplace(path, handler);
    pimpl->srv->Post(path_prefix + path, [handler](const httplib::Request & req, httplib::Response & res) {
        std::string body = req.body;
        std::map<std::string, uploaded_file> files;

        if (req.is_multipart_form_data()) {
            // translate text fields to a JSON object and use it as the body
            json form_json = json::object();
            for (const auto & [key, field] : req.form.fields) {
                if (form_json.contains(key)) {
                    // if the key already exists, convert it to an array
                    if (!form_json[key].is_array()) {
                        json existing_value = form_json[key];
                        form_json[key] = json::array({existing_value});
                    }
                    form_json[key].push_back(field.content);
                } else {
                    form_json[key] = field.content;
                }
            }
            body = form_json.dump();

            // populate files from multipart form
            for (const auto & [key, file] : req.form.files) {
                files[key] = uploaded_file{
                    raw_buffer(file.content.begin(), file.content.end()),
                    file.filename,
                    file.content_type,
                };
            }
        }

        server_http_req_ptr request = std::make_unique<server_http_req>(server_http_req{
            get_params(req),
            get_headers(req),
            req.path,
            build_query_string(req),
            body,
            std::move(files),
            req.is_connection_closed
        });
        server_http_res_ptr response = handler(*request);
        process_handler_response(std::move(request), response, res);
    });
}

void server_http_context::del(const std::string & path, const server_http_context::handler_t & handler) const {
    handlers.emplace(path, handler);
    pimpl->srv->Delete(path_prefix + path, [handler](const httplib::Request & req, httplib::Response & res) {
        server_http_req_ptr request = std::make_unique<server_http_req>(server_http_req{
            get_params(req),
            get_headers(req),
            req.path,
            build_query_string(req),
            req.body,
            {},
            req.is_connection_closed
        });
        server_http_res_ptr response = handler(*request);
        process_handler_response(std::move(request), response, res);
    });
}

//
// Vertex AI Prediction protocol (AIP_PREDICT_ROUTE)
// https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements
//

// Derives the camelCase @requestFormat alias for a registered path.
// e.g. "/v1/chat/completions" -> "chatCompletions", "/apply-template" -> "applyTemplate"
static std::string path_to_gcp_format(const std::string & path) {
    std::string s = path;
    if (s.size() > 3 && s[0] == '/' && s[1] == 'v' && s[2] == '1') {
        s = s.substr(3);
    }
    if (!s.empty() && s[0] == '/') {
        s = s.substr(1);
    }
    std::string result;
    bool cap = false;
    for (unsigned char c : s) {
        if (c == ':') break; // stop before path parameters
        if (c == '/' || c == '-' || c == '_') {
            cap = true;
        } else {
            result += static_cast<char>(cap ? std::toupper(c) : c);
            cap = false;
        }
    }
    return result;
}

static json parse_gcp_predict_response(const server_http_res_ptr & res) {
    if (res == nullptr) {
        throw std::runtime_error("empty response from internal handler");
    }
    if (res->is_stream()) {
        throw std::invalid_argument("predict route does not support streaming responses");
    }
    if (res->data.empty()) {
        return nullptr;
    }
    try {
        return json::parse(res->data);
    } catch (...) {
        return res->data;
    }
}

void server_http_context::register_gcp_compat() const {
    const gcp_params gcp;

    if (!gcp.enabled) {
        // do nothing
        return;
    }

    if (handlers.count(gcp.path_predict)) {
        SRV_ERR("AIP_PREDICT_ROUTE=%s conflicts with an existing llama-server route\n", gcp.path_predict.c_str());
        exit(1);
    }

    // camelCase alias -> canonical path (first registration wins on collision)
    // e.g. "chatCompletions" -> "/v1/chat/completions"
    std::unordered_map<std::string, std::string> alias_to_path;
    for (const auto & [path, _] : handlers) {
        alias_to_path.emplace(path_to_gcp_format(path), path);
    }

    if (!gcp.path_health.empty()) {
        const auto health_handler = handlers.find("/health");
        GGML_ASSERT(health_handler != handlers.end());
        get(gcp.path_health, health_handler->second);
    }

    post(gcp.path_predict, [this, alias_to_path = std::move(alias_to_path)](const server_http_req & req) -> server_http_res_ptr {
        static const auto build_error = [](const std::string & message, error_type type) -> json {
            return json {{"error", format_error_response(message, type)}};
        };

        json data;
        try {
            data = json::parse(req.body);
        } catch (const std::exception & e) {
            auto res = std::make_unique<server_http_res>();
            res->status = 400;
            res->data = safe_json_to_str({{"error", format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST)}});
            return res;
        }
        if (!data.is_object()) {
            auto res = std::make_unique<server_http_res>();
            res->status = 400;
            res->data = safe_json_to_str({{"error", format_error_response("request body must be a JSON object", ERROR_TYPE_INVALID_REQUEST)}});
            return res;
        }
        if (!data.contains("instances") || !data.at("instances").is_array()) {
            auto res = std::make_unique<server_http_res>();
            res->status = 400;
            res->data = safe_json_to_str({{"error", format_error_response("request body must include an array field named instances", ERROR_TYPE_INVALID_REQUEST)}});
            return res;
        }

        const json & instances = data.at("instances");
        static const size_t MAX_INSTANCES = 128;
        if (instances.size() > MAX_INSTANCES) {
            auto res = std::make_unique<server_http_res>();
            res->status = 400;
            res->data = safe_json_to_str({{"error", format_error_response("instances array exceeds maximum size of " + std::to_string(MAX_INSTANCES), ERROR_TYPE_INVALID_REQUEST)}});
            return res;
        }

        std::vector<std::future<json>> futures;
        futures.reserve(instances.size());

        for (const auto & instance : instances) {
            futures.push_back(std::async(std::launch::async, [this, &req, &alias_to_path, instance]() -> json {
                if (!instance.is_object()) {
                    return build_error("each instance must be a JSON object", ERROR_TYPE_INVALID_REQUEST);
                }
                if (!instance.contains("@requestFormat") || !instance.at("@requestFormat").is_string()) {
                    return build_error("each instance must include a string @requestFormat", ERROR_TYPE_INVALID_REQUEST);
                }

                try {
                    json payload = instance;
                    const std::string format = payload.at("@requestFormat").get<std::string>();
                    payload.erase("@requestFormat");

                    if (payload.contains("stream")) {
                        SRV_WRN("%s", "ignoring client-provided stream field in instance, streaming is not supported in predict route\n");
                        payload["stream"] = false;
                    }

                    // accept both camelCase aliases (e.g. "chatCompletions") and direct paths
                    std::string dispatch_path;
                    auto it_alias = alias_to_path.find(format);
                    if (it_alias != alias_to_path.end()) {
                        dispatch_path = it_alias->second;
                    } else if (handlers.count(format)) {
                        dispatch_path = format;
                    } else {
                        return build_error("no handler registered for @requestFormat: " + format, ERROR_TYPE_INVALID_REQUEST);
                    }

                    const server_http_req internal_req {
                        req.params,
                        req.headers,
                        path_prefix + dispatch_path,
                        req.query_string,
                        payload.dump(),
                        {},
                        req.should_stop,
                    };

                    server_http_res_ptr internal_res = handlers.at(dispatch_path)(internal_req);
                    return parse_gcp_predict_response(internal_res);
                } catch (const std::invalid_argument & e) {
                    return build_error(e.what(), ERROR_TYPE_INVALID_REQUEST);
                } catch (const std::exception & e) {
                    return build_error(e.what(), ERROR_TYPE_SERVER);
                } catch (...) {
                    return build_error("unknown error", ERROR_TYPE_SERVER);
                }
            }));
        }

        json predictions = json::array();
        for (auto & future : futures) {
            predictions.push_back(future.get());
        }

        auto res = std::make_unique<server_http_res>();
        res->data = safe_json_to_str({{"predictions", predictions}});
        return res;
    });
}
