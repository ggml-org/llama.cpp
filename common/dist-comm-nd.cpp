#include "dist-comm-nd.h"

#ifndef LLAMA_DIST_TP_ND

// NetworkDirect not compiled in: every entry point reports "unavailable" so dist-comm.cpp
// keeps using TCP without any further #ifdef in the call sites.

bool dist_nd_available() { return false; }

dist_nd_conn * dist_nd_establish(uintptr_t, uintptr_t, bool, std::string & err, bool) {
    err = "this binary was built without NetworkDirect support";
    return nullptr;
}

bool dist_nd_send2(dist_nd_conn *, const void *, size_t, const void *, size_t, std::string & err) {
    err = "no ND";
    return false;
}
bool dist_nd_recv(dist_nd_conn *, void *, size_t, std::string & err) { err = "no ND"; return false; }
bool dist_nd_send_drain(dist_nd_conn *, std::string &) { return true; }
void dist_nd_free(dist_nd_conn *) {}

#else // LLAMA_DIST_TP_ND

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#  define NOMINMAX
#endif
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <ndsupport.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <future>
#include <malloc.h>
#include <memory>
#include <mutex>
#include <vector>

// ---------------------------------------------------------------------------------------
// Tunables, lifted unchanged from the proven ggml-rpc NetworkDirect transport.
//
// The provider implements only a subset of ND SPI v2 - no RDMA Read, no shared receive
// queues, no usable memory windows, no inline sends, and only the first SGE of a request is
// honoured - so everything below stays inside that subset.
//
// Flow control: ND does not retry when the peer has no receive posted, so a sender may never
// have more than ND_WINDOW chunks outstanding. After ND_WINDOW chunks it blocks for an ack,
// which the receiver emits once it has consumed and reposted the same number of chunks. Both
// counters are cumulative over the connection so they stay in lockstep, and at most one ack is
// ever in flight - that is why ND_WINDOW leaves one receive slot spare.
//
// Here both ranks send and receive on the same connection, so each side runs both halves of
// that protocol; the two directions are independent and the invariant holds per direction.
//
// The receiver acks only while it consumes, inside nd_recv_raw. When both ends of a link send a
// message longer than the window before either receives (full duplex, or the edge an odd ring
// cannot 2-colour), both block at the window for an ack neither will send. That is why payloads
// above one chunk stayed on TCP. LLAMA_DIST_TP_RDMA_MULTICHUNK=1 lets a sender blocked at the
// window keep taking chunks off every connection's receive ring into a side buffer and acking
// them, so a message may span any number of 256 KiB chunks (see nd_wait_ack_progressing).
// ---------------------------------------------------------------------------------------

static constexpr uint32_t ND_MSG_MAGIC   = 0x4E444D31; // 'NDM1'
static constexpr uint32_t ND_HELLO_MAGIC = 0x444E4452; // 'DNDR'
static constexpr uint32_t ND_HELLO_VER   = 1;

static constexpr size_t   ND_PAYLOAD    = 256 * 1024;      // provider caps a transfer at 1 MiB
static constexpr int      ND_RX_DEPTH   = 24;              // pre-posted receives: 24 x 256 KiB = 6 MiB
static constexpr int      ND_WINDOW     = ND_RX_DEPTH - 1; // chunks in flight before an ack is required
// Staging slots for outbound chunks. ONE slot meant one outstanding send, which forced the whole
// exchange to serialize; a small ring lets a send stay on the wire while this rank receives. Kept
// well under ND_WINDOW so flow control, not the ring, stays what bounds messages in flight.
static constexpr int      ND_TX_DEPTH   = 4;
static constexpr int      ND_CONNECT_TIMEOUT_S = 30;
static constexpr uint64_t ND_POLL_SPIN  = 4096;
static constexpr uint32_t ND_POLL_TICK_MS = 200;   // how long to sleep on the CQ between peer liveness checks
static constexpr uint32_t ND_REAP_MS      = 200;   // bounded reap of a pending CQ notify during teardown
static constexpr uint32_t ND_MAX_EMPTY_NOTIFY = 64;      // CQ claims ready but yields nothing this many times
static constexpr uint32_t ND_MAX_IDLE_STEPS   = 100000;  // completions consumed while no payload arrives
static constexpr uint16_t ND_CM_PORT    = 23517;   // fixed by the provider's connection manager

enum nd_msg_type : uint32_t {
    ND_MSG_DATA = 1,
    ND_MSG_ACK  = 2,
    ND_MSG_BYE  = 3,
};

struct nd_msg_hdr {
    uint32_t magic;
    uint32_t type;
    uint32_t len;
    uint32_t seq;
};

static constexpr size_t ND_SLOT = ND_PAYLOAD + sizeof(nd_msg_hdr);

// out-of-band negotiation record, exchanged over the TCP ring sockets
struct nd_hello {
    uint32_t magic;
    uint32_t version;
    uint32_t ipv4;   // listener's ND address in network byte order, 0 from the dialer
    uint32_t ok;     // 1 when this side has its half ready
};

// receive contexts are 1-based so that a null context never looks like slot 0
static inline void * nd_ctx_recv(int slot) { return (void *)(uintptr_t)(slot + 1); }
static inline int    nd_ctx_slot(void * ctx) { return (int)(uintptr_t)ctx - 1; }
static void * const  ND_CTX_SEND = (void *)(uintptr_t)(ND_RX_DEPTH + 2);
// an ack posted from the dedicated ack slot (multi-chunk mode), retired outside the tx ring
static void * const  ND_CTX_ACK  = (void *)(uintptr_t)(ND_RX_DEPTH + 3);
// the dedicated ack slot sits after the tx ring in the same registered buffer
static constexpr size_t ND_ACK_SLOT_BYTES = 64;

static bool nd_debug() {
    static const bool on = []() {
        const char * v = std::getenv("LLAMA_DIST_TP_RDMA_DEBUG");
        return v && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

// Opt-in, LLAMA_DIST_TP_RDMA_MULTICHUNK=1: see the flow-control note above. Unset, nothing changes.
static bool nd_multichunk() {
    static const bool on = []() {
        const char * v = std::getenv("LLAMA_DIST_TP_RDMA_MULTICHUNK");
        const bool r = v && v[0] != '\0' && v[0] != '0';
        if (r) {
            std::fprintf(stderr, "[dist-tp][nd] multi-chunk messages: a blocked sender keeps receiving\n");
        }
        return r;
    }();
    return on;
}

#define ND_LOG_ERR(...) std::fprintf(stderr, "[dist-tp][nd] " __VA_ARGS__)
#define ND_LOG_DBG(...) do { if (nd_debug()) std::fprintf(stderr, "[dist-tp][nd] " __VA_ARGS__); } while (0)

struct nd_ov {
    OVERLAPPED ov = {};

    bool init() {
        ov.hEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        return ov.hEvent != nullptr;
    }

    ~nd_ov() {
        if (ov.hEvent) {
            CloseHandle(ov.hEvent);
        }
    }
};

static bool nd_done(HRESULT hr, IND2Overlapped * obj, OVERLAPPED * ov) {
    if (hr == ND_PENDING) {
        hr = obj->GetOverlappedResult(ov, TRUE);
    }
    return SUCCEEDED(hr);
}

struct nd_rx_item {
    int      slot;
    uint32_t len;
    uint32_t off;
};

struct nd_addr_buf { char s[32]; };

static nd_addr_buf nd_addr_str(const struct sockaddr_in & sa) {
    nd_addr_buf b;
    char ip[INET_ADDRSTRLEN] = "?";
    InetNtopA(AF_INET, (void *)&sa.sin_addr, ip, sizeof(ip));
    snprintf(b.s, sizeof(b.s), "%s:%u", ip, (unsigned)ntohs(sa.sin_port));
    return b;
}

struct dist_nd_conn {
    IND2Adapter         * adapter   = nullptr;
    IND2CompletionQueue * cq        = nullptr;
    IND2QueuePair       * qp        = nullptr;
    IND2Connector       * connector = nullptr;
    IND2Listener        * listener  = nullptr;
    IND2MemoryRegion    * tx_mr     = nullptr;
    IND2MemoryRegion    * rx_mr     = nullptr;
    HANDLE                file      = nullptr;

    nd_ov ov;
    nd_ov ov_listen;
    nd_ov ov_cq;      // kept separate: a notify can still be pending while teardown reuses ov

    uint8_t * tx_buf   = nullptr;
    uint8_t * rx_buf   = nullptr;
    UINT32    tx_token = 0;
    UINT32    rx_token = 0;

    struct sockaddr_in local = {};
    struct sockaddr_in peer  = {};

    // the TCP ring sockets, kept only for peer-liveness checks once ND carries the payload
    SOCKET tcp_out = INVALID_SOCKET;
    SOCKET tcp_in  = INVALID_SOCKET;

    std::deque<nd_rx_item> rx_ready;
    bool     connected    = false;
    bool     peer_gone    = false;
    bool     notify_armed = false;
    int      ack_pending  = 0;
    int      send_done    = 0;
    int      tx_since_ack = 0;
    int      rx_since_ack = 0;
    uint32_t tx_seq       = 0;
    size_t   tx_fill      = 0;   // payload bytes staged in the current tx slot behind the header
    int      corked       = 0;
    int      tx_slot_cur  = 0;   // ring index currently being staged into
    int      tx_inflight  = 0;   // sends posted whose completion has not been reaped yet

    // Multi-chunk mode: payload taken off the receive ring while this rank was blocked at its
    // own send window. It precedes everything still in rx_ready, and nd_recv_raw reads it first.
    std::vector<uint8_t> rx_spill;
    size_t   rx_spill_off = 0;
    int      ack_inflight = 0;   // acks posted from the dedicated slot and not yet retired

    uint8_t * tx_slot(int i) const {
        return tx_buf + (size_t)i * ND_SLOT;
    }

    uint8_t * rx_slot(int i) const {
        return rx_buf + (size_t)i * ND_SLOT;
    }

    uint8_t * ack_slot() const {
        return tx_buf + (size_t)ND_TX_DEPTH * ND_SLOT;
    }

    bool post_rx(int i) {
        ND2_SGE sge = {};
        sge.Buffer            = rx_slot(i);
        sge.BufferLength      = (ULONG)ND_SLOT;
        sge.MemoryRegionToken = rx_token;
        return SUCCEEDED(qp->Receive(nd_ctx_recv(i), &sge, 1));
    }

    ~dist_nd_conn() {
        if (connected) {
            ND_LOG_DBG("tearing down connection (%u chunks sent)\n", tx_seq);
        }
        if (notify_armed && cq) {
            if (WaitForSingleObject(ov_cq.ov.hEvent, ND_REAP_MS) == WAIT_OBJECT_0) {
                cq->GetOverlappedResult(&ov_cq.ov, FALSE);
            }
            notify_armed = false;
        }
        if (connector) {
            HRESULT hr = connector->Disconnect(&ov.ov);
            if (hr == ND_PENDING) {
                connector->GetOverlappedResult(&ov.ov, TRUE);
            }
        }
        if (qp)        { qp->Release(); }
        if (tx_mr) {
            HRESULT hr = tx_mr->Deregister(&ov.ov);
            if (hr == ND_PENDING) { tx_mr->GetOverlappedResult(&ov.ov, TRUE); }
            tx_mr->Release();
        }
        if (rx_mr) {
            HRESULT hr = rx_mr->Deregister(&ov.ov);
            if (hr == ND_PENDING) { rx_mr->GetOverlappedResult(&ov.ov, TRUE); }
            rx_mr->Release();
        }
        if (cq)        { cq->Release(); }
        if (listener)  { listener->Release(); }
        if (connector) { connector->Release(); }
        if (adapter)   { adapter->Release(); }
        if (file && file != INVALID_HANDLE_VALUE) {
            CloseHandle(file);
        }
        _aligned_free(tx_buf);
        _aligned_free(rx_buf);
    }
};

// ---------------------------------------------------------------------------------------
// runtime init
// ---------------------------------------------------------------------------------------

static bool nd_startup() {
    static std::once_flag once;
    static bool ok = false;
    std::call_once(once, []() {
        const HRESULT hr = NdStartup();
        if (FAILED(hr)) {
            ND_LOG_ERR("NdStartup failed: 0x%08lx, NetworkDirect unavailable\n", (unsigned long)hr);
            return;
        }
        ND_LOG_DBG("NdStartup ok\n");
        ok = true;
    });
    return ok;
}

// ---------------------------------------------------------------------------------------
// peer liveness
// ---------------------------------------------------------------------------------------

// Once ND is active no collective payload travels over the TCP sockets, so readability that
// turns out to be end-of-stream is the only way to notice a peer that vanished without an
// orderly shutdown. Pending real bytes (a broadcast that is still on TCP) read as "alive".
static bool nd_tcp_peer_closed(SOCKET fd) {
    if (fd == INVALID_SOCKET) {
        return false;
    }
    fd_set rd;
    FD_ZERO(&rd);
    FD_SET(fd, &rd);
    timeval tv = { 0, 0 };
    if (select(0, &rd, nullptr, nullptr, &tv) <= 0) {
        return false;
    }
    char b;
    const int r = recv(fd, &b, 1, MSG_PEEK);
    if (r == 0) {
        return true;
    }
    return r < 0 && WSAGetLastError() != WSAEWOULDBLOCK;
}

// ---------------------------------------------------------------------------------------
// adapter / queue pair setup
// ---------------------------------------------------------------------------------------

static bool nd_local_addr(SOCKET fd, struct sockaddr_in * out) {
    // an explicit override lets the RDMA NIC differ from the one carrying the ring sockets
    if (const char * env = std::getenv("LLAMA_DIST_TP_RDMA_ADDR")) {
        memset(out, 0, sizeof(*out));
        out->sin_family = AF_INET;
        if (InetPtonA(AF_INET, env, &out->sin_addr) != 1) {
            ND_LOG_ERR("LLAMA_DIST_TP_RDMA_ADDR is not a valid IPv4 address: %s\n", env);
            return false;
        }
        ND_LOG_DBG("local address %s (from LLAMA_DIST_TP_RDMA_ADDR)\n", nd_addr_str(*out).s);
        return true;
    }
    int len = sizeof(*out);
    if (getsockname(fd, (struct sockaddr *)out, &len) != 0 || out->sin_family != AF_INET) {
        ND_LOG_DBG("getsockname on the ring socket failed, no local address\n");
        return false;
    }
    out->sin_port = 0;
    ND_LOG_DBG("local address %s (from the ring socket)\n", nd_addr_str(*out).s);
    return true;
}

static bool nd_setup(std::unique_ptr<dist_nd_conn> & conn, const struct sockaddr_in & local_addr) {
    auto c = std::make_unique<dist_nd_conn>();
    if (!c->ov.init() || !c->ov_listen.init() || !c->ov_cq.init()) {
        return false;
    }
    c->local = local_addr;

    HRESULT hr = NdOpenAdapter(IID_IND2Adapter, (const struct sockaddr *)&c->local, sizeof(c->local),
                               (void **)&c->adapter);
    if (FAILED(hr)) {
        ND_LOG_DBG("NdOpenAdapter(%s) failed: 0x%08lx\n", nd_addr_str(c->local).s, (unsigned long)hr);
        return false;
    }
    ND_LOG_DBG("adapter opened on %s\n", nd_addr_str(c->local).s);
    hr = c->adapter->CreateOverlappedFile(&c->file);
    if (FAILED(hr)) {
        ND_LOG_DBG("CreateOverlappedFile failed: 0x%08lx\n", (unsigned long)hr);
        return false;
    }

    ND2_ADAPTER_INFO info = {};
    info.InfoVersion = ND_VERSION_2;
    ULONG cb_info = sizeof(info);
    hr = c->adapter->Query(&info, &cb_info);
    if (FAILED(hr)) {
        ND_LOG_DBG("adapter Query failed: 0x%08lx\n", (unsigned long)hr);
        return false;
    }
    ND_LOG_DBG("adapter limits: transfer=%lu reg=%zu depths rq=%lu iq=%lu cq=%lu\n",
               (unsigned long)info.MaxTransferLength, (size_t)info.MaxRegistrationSize,
               (unsigned long)info.MaxReceiveQueueDepth, (unsigned long)info.MaxInitiatorQueueDepth,
               (unsigned long)info.MaxCompletionQueueDepth);
    if (info.MaxTransferLength < ND_SLOT || info.MaxReceiveQueueDepth < ND_RX_DEPTH) {
        ND_LOG_ERR("adapter limits too small (transfer=%lu, rq=%lu)\n",
                   (unsigned long)info.MaxTransferLength, (unsigned long)info.MaxReceiveQueueDepth);
        return false;
    }

    c->tx_buf = (uint8_t *)_aligned_malloc(ND_SLOT * ND_TX_DEPTH + ND_ACK_SLOT_BYTES, 4096);
    c->rx_buf = (uint8_t *)_aligned_malloc(ND_SLOT * ND_RX_DEPTH, 4096);
    if (!c->tx_buf || !c->rx_buf) {
        return false;
    }

    hr = c->adapter->CreateMemoryRegion(IID_IND2MemoryRegion, c->file, (void **)&c->tx_mr);
    if (FAILED(hr)) {
        return false;
    }
    hr = c->adapter->CreateMemoryRegion(IID_IND2MemoryRegion, c->file, (void **)&c->rx_mr);
    if (FAILED(hr)) {
        return false;
    }
    hr = c->tx_mr->Register(c->tx_buf, ND_SLOT * ND_TX_DEPTH + ND_ACK_SLOT_BYTES, ND_MR_FLAG_ALLOW_LOCAL_WRITE, &c->ov.ov);
    if (!nd_done(hr, c->tx_mr, &c->ov.ov)) {
        ND_LOG_DBG("tx Register failed: 0x%08lx\n", (unsigned long)hr);
        return false;
    }
    hr = c->rx_mr->Register(c->rx_buf, ND_SLOT * ND_RX_DEPTH, ND_MR_FLAG_ALLOW_LOCAL_WRITE, &c->ov.ov);
    if (!nd_done(hr, c->rx_mr, &c->ov.ov)) {
        ND_LOG_DBG("rx Register failed: 0x%08lx\n", (unsigned long)hr);
        return false;
    }
    c->tx_token = c->tx_mr->GetLocalToken();
    c->rx_token = c->rx_mr->GetLocalToken();

    const ULONG cq_depth = std::min<ULONG>(info.MaxCompletionQueueDepth, 2 * ND_RX_DEPTH + 8);
    hr = c->adapter->CreateCompletionQueue(IID_IND2CompletionQueue, c->file, cq_depth, 0, 0, (void **)&c->cq);
    if (FAILED(hr)) {
        ND_LOG_DBG("CreateCompletionQueue(depth=%lu) failed: 0x%08lx\n", (unsigned long)cq_depth, (unsigned long)hr);
        return false;
    }
    const ULONG iq_depth = std::min<ULONG>(info.MaxInitiatorQueueDepth, ND_RX_DEPTH);
    hr = c->adapter->CreateQueuePair(IID_IND2QueuePair, c->cq, c->cq, nullptr,
                                     ND_RX_DEPTH, iq_depth,
                                     1, 1, 0, (void **)&c->qp);
    if (FAILED(hr)) {
        ND_LOG_DBG("CreateQueuePair(rq=%d iq=%lu) failed: 0x%08lx\n",
                   ND_RX_DEPTH, (unsigned long)iq_depth, (unsigned long)hr);
        return false;
    }
    hr = c->adapter->CreateConnector(IID_IND2Connector, c->file, (void **)&c->connector);
    if (FAILED(hr)) {
        ND_LOG_DBG("CreateConnector failed: 0x%08lx\n", (unsigned long)hr);
        return false;
    }

    // receives must be posted before the connection is accepted or completed
    for (int i = 0; i < ND_RX_DEPTH; i++) {
        if (!c->post_rx(i)) {
            ND_LOG_DBG("initial Receive post failed at slot %d\n", i);
            return false;
        }
    }
    ND_LOG_DBG("posted %d receives of %zu B each\n", ND_RX_DEPTH, ND_SLOT);

    conn = std::move(c);
    return true;
}

// tries the adapter behind the ring socket first, then any other ND capable local address
static bool nd_setup_any(std::unique_ptr<dist_nd_conn> & conn, SOCKET fd, struct sockaddr_in * chosen,
                         bool strict_local_addr) {
    struct sockaddr_in addr = {};
    if (nd_local_addr(fd, &addr) && nd_setup(conn, addr)) {
        *chosen = addr;
        return true;
    }
    conn.reset();
    if (std::getenv("LLAMA_DIST_TP_RDMA_ADDR")) {
        // an explicit choice must not be silently overridden
        return false;
    }
    if (strict_local_addr) {
        // This edge is not on an ND capable address. Scanning would find an adapter facing some
        // other peer and advertise or dial an address unreachable across this edge, which shows
        // up as a connect timeout rather than a clean fallback. Give up now and let it run TCP.
        ND_LOG_DBG("ring-local address is not ND capable and scanning is disabled for this edge\n");
        return false;
    }
    ND_LOG_DBG("setup on the ring-local address failed, scanning ND capable addresses\n");

    SIZE_T cb_list = 0;
    NdQueryAddressList(0, nullptr, &cb_list);
    if (cb_list == 0) {
        ND_LOG_DBG("NdQueryAddressList reports no ND capable local addresses\n");
        return false;
    }
    std::vector<uint8_t> raw(cb_list);
    SOCKET_ADDRESS_LIST * list = (SOCKET_ADDRESS_LIST *)raw.data();
    if (FAILED(NdQueryAddressList(0, list, &cb_list))) {
        ND_LOG_DBG("NdQueryAddressList failed\n");
        return false;
    }
    for (int i = 0; i < list->iAddressCount; i++) {
        const struct sockaddr * sa = list->Address[i].lpSockaddr;
        if (!sa || sa->sa_family != AF_INET) {
            continue;
        }
        memset(&addr, 0, sizeof(addr));
        memcpy(&addr, sa, sizeof(addr));
        addr.sin_port = 0;
        ND_LOG_DBG("trying local address %s\n", nd_addr_str(addr).s);
        if (nd_setup(conn, addr)) {
            *chosen = addr;
            return true;
        }
        conn.reset();
    }
    return false;
}

static bool nd_listen(dist_nd_conn * c) {
    // the provider binds the CM port with SO_REUSEADDR, so a second listener on this host would
    // bind it again and inbound connections would be split between the two at random. A plain
    // bind without SO_REUSEADDR is rejected while another listener holds the port.
    {
        SOCKET probe = socket(AF_INET, SOCK_STREAM, 0);
        if (probe == INVALID_SOCKET) {
            return false;
        }
        struct sockaddr_in probe_addr = c->local;
        probe_addr.sin_port = htons(ND_CM_PORT);
        const bool taken = bind(probe, (const struct sockaddr *)&probe_addr, sizeof(probe_addr)) != 0;
        closesocket(probe);
        if (taken) {
            ND_LOG_ERR("CM port %u already in use on this host, staying on TCP\n", ND_CM_PORT);
            return false;
        }
    }

    HRESULT hr = c->adapter->CreateListener(IID_IND2Listener, c->file, (void **)&c->listener);
    if (FAILED(hr)) {
        ND_LOG_DBG("CreateListener failed: 0x%08lx\n", (unsigned long)hr);
        return false;
    }
    hr = c->listener->Bind((const struct sockaddr *)&c->local, sizeof(c->local));
    if (FAILED(hr)) {
        ND_LOG_DBG("listener Bind failed: 0x%08lx\n", (unsigned long)hr);
        return false;
    }
    hr = c->listener->Listen(1);
    if (FAILED(hr)) {
        ND_LOG_DBG("Listen failed: 0x%08lx\n", (unsigned long)hr);
        return false;
    }
    ND_LOG_DBG("listening on %s (CM port %u)\n", nd_addr_str(c->local).s, ND_CM_PORT);
    return true;
}

static bool nd_accept(dist_nd_conn * c) {
    ND_LOG_DBG("waiting for a connection request (timeout %ds)\n", ND_CONNECT_TIMEOUT_S);
    // GetConnectionRequest blocks, so bound the wait in case the peer never dials in
    auto pending = std::async(std::launch::async, [c]() {
        return c->listener->GetConnectionRequest(c->connector, &c->ov_listen.ov);
    });
    if (pending.wait_for(std::chrono::seconds(ND_CONNECT_TIMEOUT_S)) != std::future_status::ready) {
        c->listener->CancelOverlappedRequests();
        pending.wait();
        ND_LOG_ERR("connection request timed out\n");
        return false;
    }
    HRESULT hr = pending.get();
    if (!nd_done(hr, c->listener, &c->ov_listen.ov)) {
        ND_LOG_DBG("GetConnectionRequest failed: 0x%08lx\n", (unsigned long)hr);
        return false;
    }
    struct sockaddr_in peer = {};
    ULONG cb_peer = sizeof(peer);
    if (SUCCEEDED(c->connector->GetPeerAddress((struct sockaddr *)&peer, &cb_peer))) {
        ND_LOG_DBG("connection request from %s\n", nd_addr_str(peer).s);
        c->peer = peer;
    }
    hr = c->connector->Accept(c->qp, 0, 0, nullptr, 0, &c->ov.ov);
    if (!nd_done(hr, c->connector, &c->ov.ov)) {
        ND_LOG_DBG("Accept failed: 0x%08lx\n", (unsigned long)hr);
        return false;
    }
    c->connected = true;
    return true;
}

static bool nd_connect(std::unique_ptr<dist_nd_conn> & conn, const struct sockaddr_in & peer_addr) {
    // the NIC that reaches the peer's RDMA address may not be the one carrying the ring socket
    struct sockaddr_in resolved = {};
    SIZE_T cb_resolved = sizeof(resolved);
    HRESULT hr = NdResolveAddress((const struct sockaddr *)&peer_addr, sizeof(peer_addr),
                                  (struct sockaddr *)&resolved, &cb_resolved);
    if (SUCCEEDED(hr) && resolved.sin_family == AF_INET &&
        resolved.sin_addr.s_addr != conn->local.sin_addr.s_addr) {
        resolved.sin_port = 0;
        ND_LOG_DBG("peer %s is reached via %s, reopening the adapter\n",
                   nd_addr_str(peer_addr).s, nd_addr_str(resolved).s);
        const SOCKET keep_out = conn->tcp_out;
        const SOCKET keep_in  = conn->tcp_in;
        conn.reset();
        if (!nd_setup(conn, resolved)) {
            return false;
        }
        conn->tcp_out = keep_out;
        conn->tcp_in  = keep_in;
    }

    dist_nd_conn * c = conn.get();
    c->peer = peer_addr;

    hr = c->connector->Bind((const struct sockaddr *)&c->local, sizeof(c->local));
    if (FAILED(hr)) {
        ND_LOG_DBG("connector Bind failed: 0x%08lx\n", (unsigned long)hr);
        return false;
    }
    ND_LOG_DBG("connecting %s -> %s (CM port %u, timeout %ds)\n",
               nd_addr_str(c->local).s, nd_addr_str(c->peer).s, ND_CM_PORT, ND_CONNECT_TIMEOUT_S);
    // Connect blocks in the provider's TCP-based CM, so bound it the same way nd_accept is bounded
    auto pending = std::async(std::launch::async, [c]() {
        return c->connector->Connect(c->qp, (const struct sockaddr *)&c->peer, sizeof(c->peer),
                                     0, 0, nullptr, 0, &c->ov.ov);
    });
    if (pending.wait_for(std::chrono::seconds(ND_CONNECT_TIMEOUT_S)) != std::future_status::ready) {
        c->connector->CancelOverlappedRequests();
        pending.wait();
        ND_LOG_ERR("connect timed out\n");
        return false;
    }
    hr = pending.get();
    if (!nd_done(hr, c->connector, &c->ov.ov)) {
        ND_LOG_DBG("Connect failed: 0x%08lx\n", (unsigned long)hr);
        return false;
    }
    hr = c->connector->CompleteConnect(&c->ov.ov);
    if (!nd_done(hr, c->connector, &c->ov.ov)) {
        ND_LOG_DBG("CompleteConnect failed: 0x%08lx\n", (unsigned long)hr);
        return false;
    }
    ND_LOG_DBG("connection established with %s\n", nd_addr_str(c->peer).s);
    c->connected = true;
    return true;
}

// ---------------------------------------------------------------------------------------
// data path
// ---------------------------------------------------------------------------------------

static bool nd_completion_ok(const ND2_RESULT * res) {
    if (FAILED(res->Status)) {
        // the provider reports every error event as a failed send, so the request
        // context is the only reliable way to tell what actually failed
        const int slot = nd_ctx_slot(res->RequestContext);
        ND_LOG_ERR("completion failed: status=0x%08lx type=%d on %s (bytes=%lu)\n",
                   (unsigned long)res->Status, (int)res->RequestType,
                   (res->RequestContext == ND_CTX_SEND || res->RequestContext == ND_CTX_ACK) ? "send" :
                   (slot >= 0 && slot < ND_RX_DEPTH) ? "recv" : "unknown request",
                   (unsigned long)res->BytesTransferred);
        if (slot >= 0 && slot < ND_RX_DEPTH) {
            ND_LOG_ERR("  failing recv slot %d of %d\n", slot, ND_RX_DEPTH);
        }
        return false;
    }
    return true;
}

static int64_t nd_qpc() {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return t.QuadPart;
}

// ND_POLL_TICK_MS in QueryPerformanceCounter ticks
static int64_t nd_tick_ticks() {
    static const int64_t ticks = []() {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return (int64_t)f.QuadPart * ND_POLL_TICK_MS / 1000;
    }();
    return ticks;
}

static bool nd_poll(dist_nd_conn * c, ND2_RESULT * res) {
    uint32_t ready_no_result = 0;
    for (uint64_t s = 0; ; s++) {
        if (c->cq->GetResults(res, 1) == 1) {
            return nd_completion_ok(res);
        }
        if (s < ND_POLL_SPIN) {
            YieldProcessor();
            continue;
        }
        if (!c->notify_armed) {
            HRESULT hr = c->cq->Notify(ND_CQ_NOTIFY_ANY, &c->ov_cq.ov);
            // ND_PENDING has success severity, so it must be tested before SUCCEEDED/FAILED
            if (hr == ND_PENDING) {
                c->notify_armed = true;
            } else if (FAILED(hr)) {
                ND_LOG_ERR("CQ Notify failed: 0x%08lx\n", (unsigned long)hr);
                return false;
            } else {
                // the queue claims to be non-empty yet yields nothing, so spinning here would
                // never end; bail out rather than burn a core forever
                if (++ready_no_result > ND_MAX_EMPTY_NOTIFY) {
                    ND_LOG_ERR("CQ reports ready but returns no completion, giving up\n");
                    return false;
                }
                s = 0;
                continue;
            }
        }
        // the provider cannot cancel a pending notify, so keep waiting on the same request and
        // only bound how long we sleep before checking whether the peer is still there
        const DWORD w = WaitForSingleObject(c->ov_cq.ov.hEvent, ND_POLL_TICK_MS);
        if (w == WAIT_TIMEOUT) {
            if (nd_tcp_peer_closed(c->tcp_in) || nd_tcp_peer_closed(c->tcp_out)) {
                ND_LOG_DBG("peer dropped the ring socket\n");
                c->peer_gone = true;
                return false;
            }
            continue;
        }
        if (w != WAIT_OBJECT_0) {
            ND_LOG_ERR("CQ wait failed: %lu\n", (unsigned long)GetLastError());
            return false;
        }
        c->notify_armed = false;
        HRESULT hr = c->cq->GetOverlappedResult(&c->ov_cq.ov, FALSE);
        if (FAILED(hr)) {
            ND_LOG_ERR("CQ notify result failed: 0x%08lx\n", (unsigned long)hr);
            return false;
        }
        s = 0;
    }
}

// routes one completion to the send, ack or data path
static bool nd_route(dist_nd_conn * c, const ND2_RESULT & res) {
    if (res.RequestContext == ND_CTX_ACK) {
        c->ack_inflight--;   // the dedicated ack slot is free again
        return true;
    }
    if (res.RequestContext == ND_CTX_SEND) {
        c->send_done++;
        if (c->tx_inflight > 0) {
            c->tx_inflight--;   // the slot that send occupied is free to stage into again
        }
        return true;
    }
    const int slot = nd_ctx_slot(res.RequestContext);
    if (slot < 0 || slot >= ND_RX_DEPTH) {
        ND_LOG_ERR("completion with unknown context\n");
        return false;
    }
    nd_msg_hdr hdr = {};
    memcpy(&hdr, c->rx_slot(slot), sizeof(hdr));
    if (hdr.magic != ND_MSG_MAGIC) {
        ND_LOG_ERR("framing lost (magic 0x%08x)\n", hdr.magic);
        return false;
    }
    if (hdr.type == ND_MSG_BYE) {
        ND_LOG_DBG("peer closed the connection\n");
        c->peer_gone = true;
        return false;
    }
    if (hdr.type == ND_MSG_ACK) {
        c->ack_pending++;
        return c->post_rx(slot);
    }
    if (hdr.type != ND_MSG_DATA || hdr.len == 0 || hdr.len > ND_PAYLOAD) {
        ND_LOG_ERR("bad message (type=%u len=%u)\n", hdr.type, hdr.len);
        return false;
    }
    c->rx_ready.push_back({ slot, hdr.len, 0 });
    return true;
}

// consumes exactly one completion, waiting for it
static bool nd_step(dist_nd_conn * c) {
    ND2_RESULT res = {};
    if (!nd_poll(c, &res)) {
        return false;
    }
    return nd_route(c, res);
}

// Consumes one completion if one is ready: 1 consumed, 0 none ready, -1 failed.
static int nd_try_step(dist_nd_conn * c) {
    ND2_RESULT res = {};
    if (c->cq->GetResults(&res, 1) != 1) {
        return 0;
    }
    if (!nd_completion_ok(&res)) {
        return -1;
    }
    return nd_route(c, res) ? 1 : -1;
}

static bool nd_flush(dist_nd_conn * c);

// Opt-in. Unset, every send still retires before dist_nd_send2 returns, which is exactly what the
// committed known-good build does.
static bool nd_full_duplex() {
    static const bool on = []() {
        const char * v = std::getenv("LLAMA_DIST_TP_FULL_DUPLEX");
        return v && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

// The slot about to be staged into must not still be on the wire. Sends are posted round-robin and
// retire in order, so while fewer than ND_TX_DEPTH are outstanding the current slot is the free
// one; that invariant is why this is the only place a staging write can block.
static bool nd_tx_slot_ready(dist_nd_conn * c) {
    while (c->tx_inflight >= ND_TX_DEPTH) {
        if (!nd_step(c)) {
            return false;
        }
    }
    return true;
}

// Post WITHOUT waiting for the completion. This is the change that lets an exchange overlap: this
// function used to spin on its own send completion, so a full-duplex link carried one direction at
// a time and every collective paid two serial transfers instead of one.
static bool nd_post_send(dist_nd_conn * c, int slot, size_t len) {
    ND2_SGE sge = {};
    sge.Buffer            = c->tx_slot(slot);
    sge.BufferLength      = (ULONG)len;
    sge.MemoryRegionToken = c->tx_token;

    if (FAILED(c->qp->Send(ND_CTX_SEND, &sge, 1, 0))) {
        ND_LOG_ERR("Send post failed\n");
        return false;
    }
    c->tx_inflight++;
    return true;
}

// Wait until every posted send has retired, so a staging slot is never recycled or freed while a
// transfer still references it.
static bool nd_send_drain_local(dist_nd_conn * c) {
    while (c->tx_inflight > 0) {
        if (!nd_step(c)) {
            return false;
        }
    }
    return true;
}

// A bare header (ack, bye) takes its own slot so it cannot overwrite a staged data chunk. Anything
// still staged is flushed first, which also keeps tx_seq ordering intact.
static bool nd_post_hdr_only(dist_nd_conn * c, const nd_msg_hdr & hdr) {
    if (c->tx_fill != 0 && !nd_flush(c)) {
        return false;
    }
    if (!nd_tx_slot_ready(c)) {
        return false;
    }
    const int slot = c->tx_slot_cur;
    memcpy(c->tx_slot(slot), &hdr, sizeof(hdr));
    if (!nd_post_send(c, slot, sizeof(hdr))) {
        return false;
    }
    c->tx_slot_cur = (c->tx_slot_cur + 1) % ND_TX_DEPTH;
    return true;
}

static bool nd_send_ack(dist_nd_conn * c) {
    const nd_msg_hdr hdr = { ND_MSG_MAGIC, ND_MSG_ACK, 0, c->tx_seq++ };
    return nd_post_hdr_only(c, hdr);
}

// The provider raises no completion and no notification when a peer disconnects, so an
// explicit bye is what lets the other side leave its completion queue wait promptly. It rides
// in the slot ND_WINDOW keeps spare, so it needs no flow control of its own.
static bool nd_send_bye(dist_nd_conn * c) {
    const nd_msg_hdr hdr = { ND_MSG_MAGIC, ND_MSG_BYE, 0, c->tx_seq++ };
    return nd_post_hdr_only(c, hdr);
}

// ---------------------------------------------------------------------------------------
// multi-chunk messages (opt-in, LLAMA_DIST_TP_RDMA_MULTICHUNK)
// ---------------------------------------------------------------------------------------

// Every live connection of this process. A rank blocked at one edge's window must keep the other
// edge's receive side moving too: on a 3-rank ring each rank waits for an ack from next while
// prev waits for an ack from it, which is the same deadlock one hop around.
static std::mutex                   g_nd_conns_mu;
static std::vector<dist_nd_conn *>  g_nd_conns;

static void nd_register(dist_nd_conn * c) {
    std::lock_guard<std::mutex> lk(g_nd_conns_mu);
    g_nd_conns.push_back(c);
}

static void nd_unregister(dist_nd_conn * c) {
    std::lock_guard<std::mutex> lk(g_nd_conns_mu);
    g_nd_conns.erase(std::remove(g_nd_conns.begin(), g_nd_conns.end(), c), g_nd_conns.end());
}

// An ack from its own slot, so it can leave while a data chunk is staged in the ring slot, which
// is exactly the state of a sender blocked at its window. Acks still go out one per ND_WINDOW
// chunks freed, so at most one is in flight per direction and the spare receive slot covers it.
static bool nd_post_ack_direct(dist_nd_conn * c) {
    while (c->ack_inflight > 0) {   // the previous ack still owns the slot
        if (!nd_step(c)) {
            return false;
        }
    }
    const nd_msg_hdr hdr = { ND_MSG_MAGIC, ND_MSG_ACK, 0, c->tx_seq++ };
    memcpy(c->ack_slot(), &hdr, sizeof(hdr));
    ND2_SGE sge = {};
    sge.Buffer            = c->ack_slot();
    sge.BufferLength      = (ULONG)sizeof(hdr);
    sge.MemoryRegionToken = c->tx_token;
    if (FAILED(c->qp->Send(ND_CTX_ACK, &sge, 1, 0))) {
        ND_LOG_ERR("ack post failed\n");
        return false;
    }
    c->ack_inflight++;
    return true;
}

// Moves every chunk that has arrived into rx_spill, reposts its receive and acks as nd_recv_raw
// would have. rx_ready only ever holds bytes that come after rx_spill, so order is kept.
static bool nd_spill_rx(dist_nd_conn * c) {
    while (!c->rx_ready.empty()) {
        const nd_rx_item it = c->rx_ready.front();
        c->rx_ready.pop_front();
        const uint8_t * p = c->rx_slot(it.slot) + sizeof(nd_msg_hdr) + it.off;
        c->rx_spill.insert(c->rx_spill.end(), p, p + (it.len - it.off));
        if (!c->post_rx(it.slot)) {
            return false;
        }
        if (++c->rx_since_ack >= ND_WINDOW) {
            c->rx_since_ack = 0;
            if (!nd_post_ack_direct(c)) {
                return false;
            }
        }
    }
    return true;
}

// Waits for an ack on c while draining every connection's receive side. With one connection
// (2 ranks) everything arrives on c's queue, so a blocking step is enough; with more, no single
// queue can be slept on, so the queues are polled in turn.
static bool nd_wait_ack_progressing(dist_nd_conn * c) {
    std::vector<dist_nd_conn *> all;
    {
        std::lock_guard<std::mutex> lk(g_nd_conns_mu);
        all = g_nd_conns;
    }
    if (std::find(all.begin(), all.end(), c) == all.end()) {
        all.push_back(c);
    }
    const int64_t tick = nd_tick_ticks();
    int64_t last_check = nd_qpc();
    for (;;) {
        for (dist_nd_conn * o : all) {
            if (!nd_spill_rx(o)) {
                return false;
            }
        }
        if (c->ack_pending > 0) {
            return true;
        }
        if (all.size() == 1) {
            if (!nd_step(c)) {
                return false;
            }
            continue;
        }
        bool any = false;
        for (dist_nd_conn * o : all) {
            const int r = nd_try_step(o);
            if (r < 0) {
                return false;
            }
            any = any || r > 0;
        }
        if (any) {
            continue;
        }
        YieldProcessor();
        const int64_t now = nd_qpc();
        if (now - last_check >= tick) {
            last_check = now;
            for (dist_nd_conn * o : all) {
                if (nd_tcp_peer_closed(o->tcp_in) || nd_tcp_peer_closed(o->tcp_out)) {
                    ND_LOG_DBG("peer dropped the ring socket\n");
                    o->peer_gone = true;
                    c->peer_gone = true;
                    return false;
                }
            }
        }
    }
}

static bool nd_flush(dist_nd_conn * c) {
    if (c->tx_fill == 0) {
        return true;
    }
    if (c->tx_since_ack >= ND_WINDOW) {
        if (nd_multichunk()) {
            if (!nd_wait_ack_progressing(c)) {
                return false;
            }
        } else {
            while (c->ack_pending == 0) {
                if (!nd_step(c)) {
                    return false;
                }
            }
        }
        c->ack_pending--;
        c->tx_since_ack = 0;
    }
    const int slot = c->tx_slot_cur;
    const nd_msg_hdr hdr = { ND_MSG_MAGIC, ND_MSG_DATA, (uint32_t)c->tx_fill, c->tx_seq++ };
    memcpy(c->tx_slot(slot), &hdr, sizeof(hdr));
    const size_t len = sizeof(hdr) + c->tx_fill;
    c->tx_fill = 0;
    if (!nd_post_send(c, slot, len)) {
        return false;
    }
    c->tx_slot_cur = (c->tx_slot_cur + 1) % ND_TX_DEPTH;
    c->tx_since_ack++;
    return true;
}

static bool nd_send_raw(dist_nd_conn * c, const void * data, size_t size) {
    const uint8_t * src = (const uint8_t *)data;
    size_t rem = size;
    while (rem > 0) {
        if (c->tx_fill == ND_PAYLOAD && !nd_flush(c)) {
            return false;
        }
        // Starting a fresh chunk: the slot must not still be on the wire before we write it.
        if (c->tx_fill == 0 && !nd_tx_slot_ready(c)) {
            return false;
        }
        const size_t take = std::min(rem, ND_PAYLOAD - c->tx_fill);
        memcpy(c->tx_slot(c->tx_slot_cur) + sizeof(nd_msg_hdr) + c->tx_fill, src, take);
        c->tx_fill += take;
        src += take;
        rem -= take;
    }
    if (!c->corked) {
        return nd_flush(c);
    }
    return true;
}

static bool nd_recv_raw(dist_nd_conn * c, void * data, size_t size) {
    // a staged send has to reach the peer before we block waiting for its reply
    if (!nd_flush(c)) {
        return false;
    }
    uint8_t * dst = (uint8_t *)data;
    size_t rem = size;
    while (rem > 0) {
        // bytes spilled while this rank was blocked sending come before anything in rx_ready
        if (c->rx_spill_off < c->rx_spill.size()) {
            const size_t take = std::min(c->rx_spill.size() - c->rx_spill_off, rem);
            memcpy(dst, c->rx_spill.data() + c->rx_spill_off, take);
            dst += take;
            rem -= take;
            c->rx_spill_off += take;
            if (c->rx_spill_off == c->rx_spill.size()) {
                c->rx_spill.clear();   // keeps the capacity for the next prompt batch
                c->rx_spill_off = 0;
            }
            continue;
        }
        uint32_t idle_steps = 0;
        while (c->rx_ready.empty()) {
            if (!nd_step(c)) {
                return false;
            }
            // completions that never turn into payload mean the queue is replaying something
            // stale; without this the loop would spin until the peer gives up
            if (++idle_steps > ND_MAX_IDLE_STEPS) {
                ND_LOG_ERR("consumed %u completions without receiving data\n", idle_steps);
                return false;
            }
        }
        nd_rx_item & it = c->rx_ready.front();
        const size_t take = std::min((size_t)(it.len - it.off), rem);
        memcpy(dst, c->rx_slot(it.slot) + sizeof(nd_msg_hdr) + it.off, take);
        dst += take;
        rem -= take;
        it.off += (uint32_t)take;
        if (it.off == it.len) {
            const int slot = it.slot;
            c->rx_ready.pop_front();
            if (!c->post_rx(slot)) {
                return false;
            }
            if (++c->rx_since_ack >= ND_WINDOW) {
                c->rx_since_ack = 0;
                if (!nd_send_ack(c)) {
                    return false;
                }
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------
// out-of-band negotiation over the TCP ring sockets
// ---------------------------------------------------------------------------------------

static bool tcp_write_all(SOCKET fd, const void * data, size_t size) {
    const char * p = (const char *)data;
    size_t sent = 0;
    while (sent < size) {
        const int n = send(fd, p + sent, (int)(size - sent), 0);
        if (n == SOCKET_ERROR || n <= 0) {
            return false;
        }
        sent += (size_t)n;
    }
    return true;
}

static bool tcp_read_all(SOCKET fd, void * data, size_t size) {
    char * p = (char *)data;
    size_t got = 0;
    while (got < size) {
        const int n = recv(fd, p + got, (int)(size - got), 0);
        if (n <= 0) {
            return false;
        }
        got += (size_t)n;
    }
    return true;
}

// ---------------------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------------------

bool dist_nd_available() { return true; }

dist_nd_conn * dist_nd_establish(uintptr_t tcp_out_raw, uintptr_t tcp_in_raw, bool listen_side, std::string & err,
                                 bool strict_local_addr) {
    const SOCKET tcp_out = (SOCKET)tcp_out_raw;
    const SOCKET tcp_in  = (SOCKET)tcp_in_raw;

    if (!nd_startup()) {
        err = "NdStartup failed";
        return nullptr;
    }

    std::unique_ptr<dist_nd_conn> conn;
    struct sockaddr_in addr = {};
    bool local_ok = nd_setup_any(conn, tcp_out, &addr, strict_local_addr);
    if (!local_ok) {
        conn.reset();
        err = "no usable NetworkDirect adapter on this host";
    } else {
        conn->tcp_out = tcp_out;
        conn->tcp_in  = tcp_in;
    }

    // Rank 0 listens and advertises, rank 1 dials in. Rank 0 owns the final decision so the two
    // ranks can never disagree about which transport is live.
    if (listen_side) {
        if (local_ok && !nd_listen(conn.get())) {
            conn.reset();
            local_ok = false;
            err = "NetworkDirect listener setup failed";
        }

        nd_hello out = { ND_HELLO_MAGIC, ND_HELLO_VER, local_ok ? (uint32_t)addr.sin_addr.s_addr : 0u,
                         local_ok ? 1u : 0u };
        if (!tcp_write_all(tcp_out, &out, sizeof(out))) {
            err = "failed to advertise the NetworkDirect address to the peer";
            return nullptr;
        }

        if (local_ok) {
            ND_LOG_DBG("advertised %s, accepting\n", nd_addr_str(addr).s);
            if (!nd_accept(conn.get())) {
                conn.reset();
                local_ok = false;
                err = "NetworkDirect accept failed";
            }
        }

        nd_hello in = {};
        if (!tcp_read_all(tcp_in, &in, sizeof(in))) {
            err = "failed to read the peer's NetworkDirect result";
            return nullptr;
        }
        if (in.magic != ND_HELLO_MAGIC || in.version != ND_HELLO_VER) {
            err = "peer does not speak this NetworkDirect negotiation (different build?)";
            return nullptr;
        }

        const uint32_t verdict = (local_ok && in.ok) ? 1u : 0u;
        if (!tcp_write_all(tcp_out, &verdict, sizeof(verdict))) {
            err = "failed to send the NetworkDirect verdict to the peer";
            return nullptr;
        }
        if (!verdict) {
            if (local_ok) {
                err = "peer could not bring up NetworkDirect";
            }
            return nullptr;
        }
    } else {
        nd_hello in = {};
        if (!tcp_read_all(tcp_in, &in, sizeof(in))) {
            err = "failed to read the peer's NetworkDirect address";
            return nullptr;
        }
        if (in.magic != ND_HELLO_MAGIC || in.version != ND_HELLO_VER) {
            err = "peer does not speak this NetworkDirect negotiation (different build?)";
            return nullptr;
        }
        if (!in.ok || in.ipv4 == 0) {
            err = "peer could not bring up a NetworkDirect listener";
            local_ok = false;
            conn.reset();
        }

        if (local_ok) {
            struct sockaddr_in peer_addr = {};
            peer_addr.sin_family      = AF_INET;
            peer_addr.sin_addr.s_addr = in.ipv4;
            if (!nd_connect(conn, peer_addr)) {
                conn.reset();
                local_ok = false;
                err = "NetworkDirect connect failed";
            }
        }

        const nd_hello out = { ND_HELLO_MAGIC, ND_HELLO_VER, 0u, local_ok ? 1u : 0u };
        if (!tcp_write_all(tcp_out, &out, sizeof(out))) {
            err = "failed to report the NetworkDirect result to the peer";
            return nullptr;
        }

        uint32_t verdict = 0;
        if (!tcp_read_all(tcp_in, &verdict, sizeof(verdict))) {
            err = "failed to read the NetworkDirect verdict from the peer";
            return nullptr;
        }
        if (!verdict) {
            if (local_ok) {
                err = "peer could not bring up NetworkDirect";
            }
            return nullptr;
        }
    }

    err.clear();
    nd_register(conn.get());
    return conn.release();
}

bool dist_nd_send2(dist_nd_conn * c, const void * hdr, size_t hdr_size,
                   const void * payload, size_t payload_size, std::string & err) {
    // cork across both writes so the header and its payload leave in one chunk
    c->corked++;
    bool ok = nd_send_raw(c, hdr, hdr_size);
    if (ok && payload_size > 0) {
        ok = nd_send_raw(c, payload, payload_size);
    }
    c->corked--;
    if (ok) {
        ok = nd_flush(c);
    }
    // Default retires the send before returning, identical to the previous behaviour. Full-duplex
    // mode drains after the matching receive instead, which is the whole point: both directions
    // are then on the wire at once.
    if (ok && !nd_full_duplex()) {
        ok = nd_send_drain_local(c);
    }
    if (!ok) {
        c->tx_fill = 0;   // drop the half-staged chunk; the connection is finished either way
        err = c->peer_gone ? "peer closed connection" : "NetworkDirect send failed";
        return false;
    }
    return true;
}

bool dist_nd_send_drain(dist_nd_conn * c, std::string & err) {
    if (!c) {
        return true;
    }
    if (!nd_send_drain_local(c)) {
        err = c->peer_gone ? "peer closed connection" : "NetworkDirect send drain failed";
        return false;
    }
    return true;
}

bool dist_nd_recv(dist_nd_conn * c, void * data, size_t size, std::string & err) {
    if (!nd_recv_raw(c, data, size)) {
        err = c->peer_gone ? "peer closed connection" : "NetworkDirect recv failed";
        return false;
    }
    return true;
}

void dist_nd_free(dist_nd_conn * c) {
    if (!c) {
        return;
    }
    nd_unregister(c);
    if (c->connected && !c->peer_gone) {
        // Retire outstanding sends around the bye, so teardown never frees a staging slot that
        // still has a transfer against it.
        (void) nd_send_drain_local(c);
        (void) nd_send_bye(c);
        (void) nd_send_drain_local(c);
    }
    delete c;
}

#endif // LLAMA_DIST_TP_ND
