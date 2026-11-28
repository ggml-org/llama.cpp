#include "server-wisp.h"
#include "server-common.h"

#include <cpp-httplib/httplib.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

enum wisp_packet_type : uint8_t {
    WISP_PACKET_CONNECT  = 0x01,
    WISP_PACKET_DATA     = 0x02,
    WISP_PACKET_CONTINUE = 0x03,
    WISP_PACKET_CLOSE    = 0x04,
};

enum wisp_stream_type : uint8_t {
    WISP_STREAM_TCP = 0x01,
};

enum wisp_close_reason : uint8_t {
    WISP_CLOSE_VOLUNTARY     = 0x02,
    WISP_CLOSE_NETWORK_ERROR = 0x03,
    WISP_CLOSE_INVALID_INFO  = 0x41,
    WISP_CLOSE_UNREACHABLE   = 0x42,
    WISP_CLOSE_REFUSED       = 0x44,
    WISP_CLOSE_HOST_BLOCKED  = 0x48,
    WISP_CLOSE_THROTTLED     = 0x49,
};

// packets a stream may queue toward its host; the client waits for a CONTINUE once it has sent that many
static constexpr uint32_t WISP_BUFFER_PACKETS = 128;
static constexpr size_t   WISP_HEADER_SIZE    = 5;
static constexpr size_t   WISP_CONNECT_SIZE   = 3;
static constexpr size_t   WISP_READ_SIZE      = 64 * 1024;
static constexpr size_t   WISP_MAX_STREAMS    = 64;
static constexpr time_t   WISP_CONNECT_TIMEOUT_SEC = 10;

static uint32_t wisp_read_u32(const std::string & s, size_t pos) {
    return (uint32_t) (uint8_t) s[pos] | (uint32_t) (uint8_t) s[pos + 1] << 8 | (uint32_t) (uint8_t) s[pos + 2] << 16 | (uint32_t) (uint8_t) s[pos + 3] << 24;
}

static void wisp_put_u32(std::string & s, uint32_t v) {
    for (int i = 0; i < 4; i++) {
        s.push_back((char) (v >> (8 * i)));
    }
}

static void wisp_shutdown_socket(socket_t sock) {
#ifdef _WIN32
    ::shutdown(sock, SD_BOTH);
#else
    ::shutdown(sock, SHUT_RDWR);
#endif
}

struct wisp_stream {
    uint32_t id;
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::string> queue; // DATA payloads not yet written to the host
    socket_t sock = INVALID_SOCKET;
    bool closed = false;
    std::thread worker; // resolves, connects, then writes the queue to the host
    std::thread reader; // reads the host and sends DATA packets
};

struct wisp_session {
    server_http_context::ws_conn & conn;
    const bool allow_private_urls;
    std::mutex mutex;
    std::map<uint32_t, std::shared_ptr<wisp_stream>> streams;
    std::vector<std::shared_ptr<wisp_stream>> ended; // closed streams whose threads are joined by the session thread

    wisp_session(server_http_context::ws_conn & conn, bool allow_private_urls) : conn(conn), allow_private_urls(allow_private_urls) {}

    void send_packet(uint8_t type, uint32_t id, const char * payload, size_t size) {
        std::string packet;
        packet.reserve(WISP_HEADER_SIZE + size);
        packet.push_back((char) type);
        wisp_put_u32(packet, id);
        packet.append(payload, size);
        conn.send_binary(packet);
    }

    void send_continue(uint32_t id, uint32_t remaining) {
        std::string payload;
        wisp_put_u32(payload, remaining);
        send_packet(WISP_PACKET_CONTINUE, id, payload.data(), payload.size());
    }

    void send_close(uint32_t id, uint8_t reason) {
        const char payload = (char) reason;
        send_packet(WISP_PACKET_CLOSE, id, &payload, 1);
    }

    // callable from any thread: the client hears of it when notify is set
    void end_stream(uint32_t id, uint8_t reason, bool notify) {
        std::shared_ptr<wisp_stream> st;
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = streams.find(id);
            if (it == streams.end()) {
                return;
            }
            st = it->second;
            streams.erase(it);
            ended.push_back(st);
        }
        {
            std::lock_guard<std::mutex> lock(st->mutex);
            st->closed = true;
            if (st->sock != INVALID_SOCKET) {
                wisp_shutdown_socket(st->sock);
            }
        }
        st->cv.notify_all();
        if (notify) {
            send_close(id, reason);
        }
    }

    void reap() {
        std::vector<std::shared_ptr<wisp_stream>> done;
        {
            std::lock_guard<std::mutex> lock(mutex);
            done.swap(ended);
        }
        for (auto & st : done) {
            if (st->worker.joinable()) {
                st->worker.join();
            }
            if (st->reader.joinable()) {
                st->reader.join();
            }
            if (st->sock != INVALID_SOCKET) {
                httplib::detail::close_socket(st->sock);
            }
        }
    }

    void read_host(std::shared_ptr<wisp_stream> st) {
        std::vector<char> buf(WISP_READ_SIZE);
        while (true) {
            const auto n = httplib::detail::read_socket(st->sock, buf.data(), buf.size(), 0);
            if (n <= 0) {
                break;
            }
            send_packet(WISP_PACKET_DATA, st->id, buf.data(), (size_t) n);
        }
        end_stream(st->id, WISP_CLOSE_VOLUNTARY, true);
    }

    void run_stream(std::shared_ptr<wisp_stream> st, std::string host, int port) {
        std::string addr;
        try {
            addr = server_resolve_public_addr(host, allow_private_urls);
        } catch (const std::exception & e) {
            SRV_WRN("wisp: refused %s:%d: %s\n", host.c_str(), port, e.what());
            end_stream(st->id, WISP_CLOSE_HOST_BLOCKED, true);
            return;
        }

        httplib::Error error = httplib::Error::Success;
        const socket_t sock = httplib::detail::create_client_socket(
            host, addr, port, AF_UNSPEC, true, false, nullptr,
            WISP_CONNECT_TIMEOUT_SEC, 0, 0, 0, 0, 0, std::string(), error,
            [&st]() {
                std::lock_guard<std::mutex> lock(st->mutex);
                return st->closed;
            });
        if (sock == INVALID_SOCKET) {
            end_stream(st->id, error == httplib::Error::Connection ? WISP_CLOSE_REFUSED : WISP_CLOSE_UNREACHABLE, true);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(st->mutex);
            st->sock = sock;
            if (st->closed) {
                wisp_shutdown_socket(sock);
                return;
            }
            st->reader = std::thread([this, st]() { read_host(st); });
        }

        uint32_t written = 0;
        while (true) {
            std::string data;
            size_t queued;
            {
                std::unique_lock<std::mutex> lock(st->mutex);
                st->cv.wait(lock, [&st]() { return st->closed || !st->queue.empty(); });
                if (st->closed) {
                    return;
                }
                data = std::move(st->queue.front());
                st->queue.pop_front();
                queued = st->queue.size();
            }
            for (size_t off = 0; off < data.size();) {
                const auto n = httplib::detail::send_socket(sock, data.data() + off, data.size() - off, 0);
                if (n <= 0) {
                    end_stream(st->id, WISP_CLOSE_NETWORK_ERROR, true);
                    return;
                }
                off += (size_t) n;
            }
            if (++written % (WISP_BUFFER_PACKETS / 2) == 0) {
                send_continue(st->id, WISP_BUFFER_PACKETS - (uint32_t) queued);
            }
        }
    }

    void on_connect(uint32_t id, const std::string & payload) {
        if (payload.size() <= WISP_CONNECT_SIZE || (uint8_t) payload[0] != WISP_STREAM_TCP) {
            send_close(id, WISP_CLOSE_INVALID_INFO);
            return;
        }
        const int port = (uint8_t) payload[1] | (uint8_t) payload[2] << 8;
        std::string host = payload.substr(WISP_CONNECT_SIZE);
        if (port == 0) {
            send_close(id, WISP_CLOSE_INVALID_INFO);
            return;
        }

        auto st = std::make_shared<wisp_stream>();
        st->id = id;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (streams.count(id)) {
                return;
            }
            if (streams.size() >= WISP_MAX_STREAMS) {
                send_close(id, WISP_CLOSE_THROTTLED);
                return;
            }
            streams[id] = st;
        }
        SRV_DBG("wisp: stream %u to %s:%d\n", id, host.c_str(), port);
        st->worker = std::thread([this, st, host = std::move(host), port]() { run_stream(st, host, port); });
    }

    void on_data(uint32_t id, std::string data) {
        std::shared_ptr<wisp_stream> st;
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = streams.find(id);
            if (it == streams.end()) {
                return;
            }
            st = it->second;
        }
        bool overflow;
        {
            std::lock_guard<std::mutex> lock(st->mutex);
            overflow = st->queue.size() >= WISP_BUFFER_PACKETS;
            if (!overflow) {
                st->queue.push_back(std::move(data));
            }
        }
        if (overflow) {
            end_stream(id, WISP_CLOSE_THROTTLED, true);
            return;
        }
        st->cv.notify_one();
    }

    void run() {
        send_continue(0, WISP_BUFFER_PACKETS);

        std::string msg;
        while (conn.read(msg) != 0) {
            reap();
            if (msg.size() < WISP_HEADER_SIZE) {
                continue;
            }
            const uint8_t  type = (uint8_t) msg[0];
            const uint32_t id   = wisp_read_u32(msg, 1);
            if (id == 0) {
                continue;
            }
            switch (type) {
                case WISP_PACKET_CONNECT: on_connect(id, msg.substr(WISP_HEADER_SIZE));          break;
                case WISP_PACKET_DATA:    on_data(id, msg.substr(WISP_HEADER_SIZE));             break;
                case WISP_PACKET_CLOSE:   end_stream(id, WISP_CLOSE_VOLUNTARY, false);           break;
                default:                                                                          break;
            }
        }

        std::vector<uint32_t> ids;
        {
            std::lock_guard<std::mutex> lock(mutex);
            for (const auto & it : streams) {
                ids.push_back(it.first);
            }
        }
        for (const auto id : ids) {
            end_stream(id, WISP_CLOSE_VOLUNTARY, false);
        }
        reap();
    }
};

server_http_context::ws_handler_t wisp_make_handler(bool allow_private_urls) {
    return [allow_private_urls](const server_http_req &, server_http_context::ws_conn & conn) {
        wisp_session session(conn, allow_private_urls);
        session.run();
    };
}
