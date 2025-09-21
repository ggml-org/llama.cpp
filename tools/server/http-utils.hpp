
// increase max payload length to allow use of larger context size
#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH 1048576
// increase backlog size to avoid connection resets for >> 1 slots
#define CPPHTTPLIB_LISTEN_BACKLOG 512
// disable Nagle's algorithm
#define CPPHTTPLIB_TCP_NODELAY true
#include <cpp-httplib/httplib.h>

static bool server_sent_event(httplib::DataSink & sink, const json & data) {
    const std::string str =
        "data: " +
        data.dump(-1, ' ', false, json::error_handler_t::replace) +
        "\n\n"; // required by RFC 8895 - A message is terminated by a blank line (two line terminators in a row).

    LOG_DBG("data stream, to_send: %s", str.c_str());

    return sink.write(str.c_str(), str.size());
}
