#pragma once

#include "mcp_transport.h"
#include <condition_variable>
#include <mutex>
#include <thread>
#include <curl/curl.h>

namespace toolcall
{
    class mcp_sse_transport : public mcp_transport {
    public:
        ~mcp_sse_transport();

        mcp_sse_transport(std::string server_uri);

        virtual void start() override;
        virtual void stop()  override;
        virtual bool send(const std::string & request_json) override;

        size_t sse_read(const char * data, size_t len);

    private:
        static const int EndpointReceivedTimoutSeconds;
        static const int StartTimeoutSeconds;

        void sse_run();
        void parse_field_value(std::string field, std::string value);
        void on_endpoint_event();
        void on_message_event();

        std::string server_uri_;
        bool running_;
        std::thread sse_thread_;
        CURL * endpoint_;
        struct curl_slist * endpoint_headers_;
        std::vector<char> endpoint_errbuf_;

        struct sse_event {
            std::string type;
            std::string data;
            std::string id;
        } event_;

        std::string sse_buffer_;
        size_t sse_cursor_;
        std::string sse_last_id_;

        std::mutex mutex_;
        std::condition_variable cv_;
    };
}
