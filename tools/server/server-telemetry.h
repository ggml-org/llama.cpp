#pragma once

#include <map>
#include <memory>
#include <string>

class server_telemetry_span;
using server_telemetry_span_ptr = std::shared_ptr<server_telemetry_span>;

class server_telemetry_span {
  public:
    ~server_telemetry_span();

    void set_attribute(const std::string & name, const std::string & value);
    void set_attribute(const std::string & name, int64_t value);
    void set_attribute(const std::string & name, double value);
    void set_http_status(int status_code);
    void set_error(const std::string & error_type);
    void end();

  private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;

    explicit server_telemetry_span(std::unique_ptr<Impl> impl);

    friend server_telemetry_span_ptr server_telemetry_start_server_span(
        const std::string &                        method,
        const std::string &                        route,
        const std::string &                        path,
        const std::string &                        scheme,
        const std::string &                        server_address,
        int                                        server_port,
        const std::string &                        client_address,
        const std::map<std::string, std::string> & headers);
    friend server_telemetry_span_ptr server_telemetry_start_client_span(const std::string & method,
                                                                        const std::string & scheme,
                                                                        const std::string & server_address,
                                                                        int                 server_port,
                                                                        const std::string & path,
                                                                        const server_telemetry_span_ptr & parent);
    friend void                      server_telemetry_inject(std::map<std::string, std::string> & headers,
                                                             const server_telemetry_span_ptr &    span);
};

// Returns false only when tracing was requested but is unavailable in this build.
bool server_telemetry_init(bool enabled, const std::string & service_version, std::string & error);
void server_telemetry_shutdown();

server_telemetry_span_ptr server_telemetry_start_server_span(const std::string &                        method,
                                                             const std::string &                        route,
                                                             const std::string &                        path,
                                                             const std::string &                        scheme,
                                                             const std::string &                        server_address,
                                                             int                                        server_port,
                                                             const std::string &                        client_address,
                                                             const std::map<std::string, std::string> & headers);

server_telemetry_span_ptr server_telemetry_start_client_span(const std::string &               method,
                                                             const std::string &               scheme,
                                                             const std::string &               server_address,
                                                             int                               server_port,
                                                             const std::string &               path,
                                                             const server_telemetry_span_ptr & parent);

void server_telemetry_inject(std::map<std::string, std::string> & headers, const server_telemetry_span_ptr & span);
