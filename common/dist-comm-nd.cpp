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
bool dist_nd_exchange_pair(dist_nd_conn *, dist_nd_conn *, const void *, size_t, const void *, size_t,
                           void *, void *, void *, void *, std::string & err) {
    err = "no ND";
    return false;
}
bool dist_nd_exchange_n(dist_nd_conn * const *, int, const void *, size_t, const void *, size_t,
                        void * const *, void * const *, std::string & err) {
    err = "no ND";
    return false;
}
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

// LLAMA_DIST_TP_RDMA_FIXED_SLOT=1 (opt-in): for providers that need page-aligned buffers and the
// same length on both sides of every transfer. A slot is then exactly 256 KiB with the header
// inside it (262,128 B of payload), every buffer, slot and registration is page aligned, and every
// send - data, ack and bye alike - is one whole slot into a receive posted at one whole slot. The
// header still carries the real length. Unset, the layout and lengths are the ones above.
static constexpr size_t ND_FIXED_SLOT = 256 * 1024;
static constexpr size_t ND_PAGE       = 4096;
static_assert(ND_FIXED_SLOT % ND_PAGE == 0, "fixed slot must be a whole number of pages");

// LLAMA_DIST_TP_RDMA_SMALL_LANE=<bytes> (opt-in, a whole number of pages below 256 KiB; implies the
// fixed slots above): a second connection on every edge whose slots are that size. A message
// (header and payload) that fits in one of its slots goes there, everything else on the 256 KiB
// connection. Each connection keeps its own flow control and acks, and both share one completion
// queue. Data chunks carry one sequence number across the two, and the receiver takes them in that
// order, so the byte stream is the same whichever connection a chunk rode. On a provider that
// moves whole slots, a decode message of a few KiB then costs a few KiB on the wire, not 256 KiB.
static constexpr uintptr_t ND_CTX_LANE1 = 1024;   // request contexts of the small connection start here

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

static bool nd_fixed_slot() {
    static const bool on = []() {
        const char * v = std::getenv("LLAMA_DIST_TP_RDMA_FIXED_SLOT");
        const bool r = v && v[0] != '\0' && v[0] != '0';
        if (r) {
            std::fprintf(stderr, "[dist-tp][nd] fixed slots: every send and receive is one page-aligned %zu B slot\n",
                         ND_FIXED_SLOT);
        }
        return r;
    }();
    return on;
}

// slot size of the small connection, 0 when there is none
static size_t nd_small_lane() {
    static const size_t sz = []() -> size_t {
        const char * v = std::getenv("LLAMA_DIST_TP_RDMA_SMALL_LANE");
        if (!v || v[0] == '\0' || v[0] == '0') {
            return 0;
        }
        const long long b = std::strtoll(v, nullptr, 10);
        if (b < (long long)ND_PAGE || b >= (long long)ND_FIXED_SLOT || b % (long long)ND_PAGE != 0) {
            std::fprintf(stderr, "[dist-tp][nd] LLAMA_DIST_TP_RDMA_SMALL_LANE=%s ignored: needs a whole number of "
                                 "%zu B pages below %zu B\n", v, ND_PAGE, ND_FIXED_SLOT);
            return 0;
        }
        std::fprintf(stderr, "[dist-tp][nd] small connection: messages up to %zu B ride a second connection with "
                             "%lld B slots, the rest %zu B slots\n",
                     (size_t)b - sizeof(nd_msg_hdr), b, ND_FIXED_SLOT);
        return (size_t)b;
    }();
    return sz;
}

// the negotiation version: a peer with a different small connection setting must not connect
static uint32_t nd_hello_ver() {
    const size_t s = nd_small_lane();
    return s ? (uint32_t)(0x100u | (s / ND_PAGE)) : ND_HELLO_VER;
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
    uint32_t seq;   // the chunk header's sequence number
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

    // slot layout, fixed at setup: the defaults above, or whole 256 KiB slots (nd_fixed_slot)
    bool   fixed    = false;
    size_t slot_sz  = ND_SLOT;
    size_t pay_max  = ND_PAYLOAD;          // payload bytes one chunk carries behind its header
    size_t ack_sz   = ND_ACK_SLOT_BYTES;   // the dedicated ack slot after the tx ring

    // Small connection (nd_small_lane): root is the connection the caller holds and sm its small
    // twin. The twin shares root's adapter, overlapped file and completion queue, so every poll goes
    // through root, and its request contexts start at ND_CTX_LANE1 so root can route them.
    dist_nd_conn * root     = this;
    dist_nd_conn * sm       = nullptr;
    bool           shared   = false;   // adapter, file and cq belong to root
    uintptr_t      ctx_base = 0;
    uint32_t       tx_dseq  = 0;       // root only, with sm: next data sequence number out, over both
    uint32_t       rx_dseq  = 0;       // root only, with sm: next data sequence number in

    // bytes a send of len puts on the wire: a whole slot in fixed mode, else exactly len
    ULONG wire_len(size_t len) const {
        return (ULONG)(fixed ? slot_sz : len);
    }

    uint8_t * tx_slot(int i) const {
        return tx_buf + (size_t)i * slot_sz;
    }

    uint8_t * rx_slot(int i) const {
        return rx_buf + (size_t)i * slot_sz;
    }

    uint8_t * ack_slot() const {
        return tx_buf + (size_t)ND_TX_DEPTH * slot_sz;
    }

    bool post_rx(int i) {
        ND2_SGE sge = {};
        sge.Buffer            = rx_slot(i);
        sge.BufferLength      = (ULONG)slot_sz;
        sge.MemoryRegionToken = rx_token;
        return SUCCEEDED(qp->Receive((void *)(ctx_base + (uintptr_t)nd_ctx_recv(i)), &sge, 1));
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
        delete sm;   // the twin goes first: it uses this connection's adapter and completion queue
        sm = nullptr;
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
        if (cq && !shared) { cq->Release(); }
        if (listener)  { listener->Release(); }
        if (connector) { connector->Release(); }
        if (adapter && !shared) { adapter->Release(); }
        if (file && file != INVALID_HANDLE_VALUE && !shared) {
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
    if (nd_fixed_slot() || nd_small_lane() != 0) {
        c->fixed   = true;
        c->slot_sz = ND_FIXED_SLOT;
        c->pay_max = ND_FIXED_SLOT - sizeof(nd_msg_hdr);
        c->ack_sz  = ND_FIXED_SLOT;
    }
    if (info.MaxTransferLength < c->slot_sz || info.MaxReceiveQueueDepth < ND_RX_DEPTH) {
        ND_LOG_ERR("adapter limits too small (transfer=%lu, rq=%lu)\n",
                   (unsigned long)info.MaxTransferLength, (unsigned long)info.MaxReceiveQueueDepth);
        return false;
    }

    const size_t tx_bytes = c->slot_sz * ND_TX_DEPTH + c->ack_sz;
    const size_t rx_bytes = c->slot_sz * ND_RX_DEPTH;
    c->tx_buf = (uint8_t *)_aligned_malloc(tx_bytes, ND_PAGE);
    c->rx_buf = (uint8_t *)_aligned_malloc(rx_bytes, ND_PAGE);
    if (!c->tx_buf || !c->rx_buf) {
        return false;
    }
    if (c->fixed) {
        // whole slots go on the wire, so never send whatever the allocator left behind
        memset(c->tx_buf, 0, tx_bytes);
        memset(c->rx_buf, 0, rx_bytes);
    }

    hr = c->adapter->CreateMemoryRegion(IID_IND2MemoryRegion, c->file, (void **)&c->tx_mr);
    if (FAILED(hr)) {
        return false;
    }
    hr = c->adapter->CreateMemoryRegion(IID_IND2MemoryRegion, c->file, (void **)&c->rx_mr);
    if (FAILED(hr)) {
        return false;
    }
    hr = c->tx_mr->Register(c->tx_buf, tx_bytes, ND_MR_FLAG_ALLOW_LOCAL_WRITE, &c->ov.ov);
    if (!nd_done(hr, c->tx_mr, &c->ov.ov)) {
        ND_LOG_DBG("tx Register failed: 0x%08lx\n", (unsigned long)hr);
        return false;
    }
    hr = c->rx_mr->Register(c->rx_buf, rx_bytes, ND_MR_FLAG_ALLOW_LOCAL_WRITE, &c->ov.ov);
    if (!nd_done(hr, c->rx_mr, &c->ov.ov)) {
        ND_LOG_DBG("rx Register failed: 0x%08lx\n", (unsigned long)hr);
        return false;
    }
    c->tx_token = c->tx_mr->GetLocalToken();
    c->rx_token = c->rx_mr->GetLocalToken();

    // with a small connection the queue also takes its completions (nd_add_small_lane)
    const ULONG cq_depth = std::min<ULONG>(info.MaxCompletionQueueDepth,
                                           nd_small_lane() ? 4 * ND_RX_DEPTH + 16 : 2 * ND_RX_DEPTH + 8);
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
    ND_LOG_DBG("posted %d receives of %zu B each\n", ND_RX_DEPTH, c->slot_sz);

    conn = std::move(c);
    return true;
}

// Adds the small connection to c (nd_small_lane) before c listens or dials: its own page-aligned
// buffers, registrations, queue pair and connector, on c's adapter, file and completion queue.
static bool nd_add_small_lane(dist_nd_conn * c) {
    const size_t sz = nd_small_lane();
    auto s = std::make_unique<dist_nd_conn>();
    if (sz == 0 || !s->ov.init() || !s->ov_listen.init() || !s->ov_cq.init()) {
        return false;
    }
    s->shared   = true;
    s->root     = c;
    s->ctx_base = ND_CTX_LANE1;
    s->adapter  = c->adapter;
    s->file     = c->file;
    s->cq       = c->cq;
    s->local    = c->local;
    s->tcp_out  = c->tcp_out;
    s->tcp_in   = c->tcp_in;
    s->fixed    = true;
    s->slot_sz  = sz;
    s->pay_max  = sz - sizeof(nd_msg_hdr);
    s->ack_sz   = sz;

    ND2_ADAPTER_INFO info = {};
    info.InfoVersion = ND_VERSION_2;
    ULONG cb_info = sizeof(info);
    HRESULT hr = c->adapter->Query(&info, &cb_info);
    if (FAILED(hr)) {
        ND_LOG_ERR("small connection: adapter Query failed: 0x%08lx\n", (unsigned long)hr);
        return false;
    }
    const size_t tx_bytes = s->slot_sz * ND_TX_DEPTH + s->ack_sz;
    const size_t rx_bytes = s->slot_sz * ND_RX_DEPTH;
    s->tx_buf = (uint8_t *)_aligned_malloc(tx_bytes, ND_PAGE);
    s->rx_buf = (uint8_t *)_aligned_malloc(rx_bytes, ND_PAGE);
    if (!s->tx_buf || !s->rx_buf) {
        return false;
    }
    memset(s->tx_buf, 0, tx_bytes);
    memset(s->rx_buf, 0, rx_bytes);

    hr = c->adapter->CreateMemoryRegion(IID_IND2MemoryRegion, c->file, (void **)&s->tx_mr);
    if (FAILED(hr)) {
        return false;
    }
    hr = c->adapter->CreateMemoryRegion(IID_IND2MemoryRegion, c->file, (void **)&s->rx_mr);
    if (FAILED(hr)) {
        return false;
    }
    hr = s->tx_mr->Register(s->tx_buf, tx_bytes, ND_MR_FLAG_ALLOW_LOCAL_WRITE, &s->ov.ov);
    if (!nd_done(hr, s->tx_mr, &s->ov.ov)) {
        ND_LOG_ERR("small connection: tx Register failed: 0x%08lx\n", (unsigned long)hr);
        return false;
    }
    hr = s->rx_mr->Register(s->rx_buf, rx_bytes, ND_MR_FLAG_ALLOW_LOCAL_WRITE, &s->ov.ov);
    if (!nd_done(hr, s->rx_mr, &s->ov.ov)) {
        ND_LOG_ERR("small connection: rx Register failed: 0x%08lx\n", (unsigned long)hr);
        return false;
    }
    s->tx_token = s->tx_mr->GetLocalToken();
    s->rx_token = s->rx_mr->GetLocalToken();

    const ULONG iq_depth = std::min<ULONG>(info.MaxInitiatorQueueDepth, ND_RX_DEPTH);
    hr = c->adapter->CreateQueuePair(IID_IND2QueuePair, c->cq, c->cq, nullptr,
                                     ND_RX_DEPTH, iq_depth, 1, 1, 0, (void **)&s->qp);
    if (FAILED(hr)) {
        ND_LOG_ERR("small connection: CreateQueuePair on the shared completion queue failed: 0x%08lx\n",
                   (unsigned long)hr);
        return false;
    }
    hr = c->adapter->CreateConnector(IID_IND2Connector, c->file, (void **)&s->connector);
    if (FAILED(hr)) {
        ND_LOG_ERR("small connection: CreateConnector failed: 0x%08lx\n", (unsigned long)hr);
        return false;
    }
    for (int i = 0; i < ND_RX_DEPTH; i++) {
        if (!s->post_rx(i)) {
            ND_LOG_ERR("small connection: initial Receive post failed at slot %d\n", i);
            return false;
        }
    }
    ND_LOG_DBG("small connection: posted %d receives of %zu B each\n", ND_RX_DEPTH, s->slot_sz);
    c->sm = s.release();
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

// Takes the next connection request on c's listener into on (c itself, or then its small twin:
// the peer dials the two in that order).
static bool nd_accept(dist_nd_conn * c, dist_nd_conn * on) {
    ND_LOG_DBG("waiting for a connection request (timeout %ds)\n", ND_CONNECT_TIMEOUT_S);
    // GetConnectionRequest blocks, so bound the wait in case the peer never dials in
    auto pending = std::async(std::launch::async, [c, on]() {
        return c->listener->GetConnectionRequest(on->connector, &c->ov_listen.ov);
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
    if (SUCCEEDED(on->connector->GetPeerAddress((struct sockaddr *)&peer, &cb_peer))) {
        ND_LOG_DBG("connection request from %s\n", nd_addr_str(peer).s);
        on->peer = peer;
    }
    hr = on->connector->Accept(on->qp, 0, 0, nullptr, 0, &on->ov.ov);
    if (!nd_done(hr, on->connector, &on->ov.ov)) {
        ND_LOG_DBG("Accept failed: 0x%08lx\n", (unsigned long)hr);
        return false;
    }
    on->connected = true;
    return true;
}

// Bind, Connect and CompleteConnect for c's queue pair towards c->peer.
static bool nd_dial(dist_nd_conn * c) {
    HRESULT hr = c->connector->Bind((const struct sockaddr *)&c->local, sizeof(c->local));
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
    // the small twin goes on the adapter finally chosen above, before either connection dials
    if (nd_small_lane() && !c->sm && !nd_add_small_lane(c)) {
        ND_LOG_ERR("small connection setup failed\n");
        return false;
    }
    if (!nd_dial(c)) {
        return false;
    }
    if (c->sm) {
        // the listener takes requests in order, so the second one it accepts is this one
        c->sm->peer = peer_addr;
        if (!nd_dial(c->sm)) {
            ND_LOG_ERR("small connection connect failed\n");
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------
// data path
// ---------------------------------------------------------------------------------------

static bool nd_completion_ok(const ND2_RESULT * res) {
    if (FAILED(res->Status)) {
        // the provider reports every error event as a failed send, so the request
        // context is the only reliable way to tell what actually failed
        void * ctx = res->RequestContext;
        const bool lane1 = (uintptr_t)ctx >= ND_CTX_LANE1;
        if (lane1) {
            ctx = (void *)((uintptr_t)ctx - ND_CTX_LANE1);
        }
        const int slot = nd_ctx_slot(ctx);
        ND_LOG_ERR("completion failed: status=0x%08lx type=%d on %s%s (bytes=%lu)\n",
                   (unsigned long)res->Status, (int)res->RequestType,
                   (ctx == ND_CTX_SEND || ctx == ND_CTX_ACK) ? "send" :
                   (slot >= 0 && slot < ND_RX_DEPTH) ? "recv" : "unknown request",
                   lane1 ? " of the small connection" : "", (unsigned long)res->BytesTransferred);
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
    c = c->root;   // a small twin's completions land on its root's queue
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

// routes one completion to the send, ack or data path, of c or of its small twin
static bool nd_route(dist_nd_conn * c, const ND2_RESULT & res) {
    c = c->root;
    void * ctx = res.RequestContext;
    if (c->sm && (uintptr_t)ctx >= ND_CTX_LANE1) {
        ctx = (void *)((uintptr_t)ctx - ND_CTX_LANE1);
        c   = c->sm;
    }
    if (ctx == ND_CTX_ACK) {
        c->ack_inflight--;   // the dedicated ack slot is free again
        return true;
    }
    if (ctx == ND_CTX_SEND) {
        c->send_done++;
        if (c->tx_inflight > 0) {
            c->tx_inflight--;   // the slot that send occupied is free to stage into again
        }
        return true;
    }
    const int slot = nd_ctx_slot(ctx);
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
        c->peer_gone       = true;
        c->root->peer_gone = true;
        return false;
    }
    if (hdr.type == ND_MSG_ACK) {
        c->ack_pending++;
        return c->post_rx(slot);
    }
    if (hdr.type != ND_MSG_DATA || hdr.len == 0 || hdr.len > c->pay_max) {
        ND_LOG_ERR("bad message (type=%u len=%u)\n", hdr.type, hdr.len);
        return false;
    }
    c->rx_ready.push_back({ slot, hdr.len, 0, hdr.seq });
    return true;
}

// consumes exactly one completion, waiting for it
static bool nd_step(dist_nd_conn * c) {
    c = c->root;
    ND2_RESULT res = {};
    if (!nd_poll(c, &res)) {
        return false;
    }
    return nd_route(c, res);
}

// Consumes one completion if one is ready: 1 consumed, 0 none ready, -1 failed.
static int nd_try_step(dist_nd_conn * c) {
    c = c->root;
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
    sge.BufferLength      = c->wire_len(len);
    sge.MemoryRegionToken = c->tx_token;

    if (FAILED(c->qp->Send((void *)(c->ctx_base + (uintptr_t)ND_CTX_SEND), &sge, 1, 0))) {
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
    sge.BufferLength      = c->wire_len(sizeof(hdr));
    sge.MemoryRegionToken = c->tx_token;
    if (FAILED(c->qp->Send((void *)(c->ctx_base + (uintptr_t)ND_CTX_ACK), &sge, 1, 0))) {
        ND_LOG_ERR("ack post failed\n");
        return false;
    }
    c->ack_inflight++;
    return true;
}

// Which connection holds the next chunk to read, for c's byte stream (c a root): *out is c or its
// small twin, or nullptr while that chunk has not arrived. Without a twin it is c whenever anything
// is queued. With one, chunks are read in data sequence order; each connection delivers its own in
// the order they were posted, so the next chunk can only be at the front of one of the two rings,
// and two fronts that both miss it mean the stream is broken (false).
static bool nd_rx_lane(dist_nd_conn * c, dist_nd_conn ** out) {
    *out = nullptr;
    if (!c->sm) {
        if (!c->rx_ready.empty()) {
            *out = c;
        }
        return true;
    }
    for (dist_nd_conn * l : { c, c->sm }) {
        if (!l->rx_ready.empty() && l->rx_ready.front().seq == c->rx_dseq) {
            *out = l;
            return true;
        }
    }
    if (!c->rx_ready.empty() && !c->sm->rx_ready.empty()) {
        ND_LOG_ERR("data sequence lost: expected %u, fronts %u and %u (small)\n",
                   c->rx_dseq, c->rx_ready.front().seq, c->sm->rx_ready.front().seq);
        return false;
    }
    return true;
}

// The chunk at the front of l's ring (l is c or its twin) has been read to its end: repost its
// receive and ack once per window as before, directly from the ack slot (multi-chunk paths) or
// through the tx ring (nd_recv_raw).
static bool nd_rx_retire(dist_nd_conn * c, dist_nd_conn * l, bool direct_ack) {
    const int slot = l->rx_ready.front().slot;
    l->rx_ready.pop_front();
    if (c->sm) {
        c->rx_dseq++;
    }
    if (!l->post_rx(slot)) {
        return false;
    }
    if (++l->rx_since_ack >= ND_WINDOW) {
        l->rx_since_ack = 0;
        return direct_ack ? nd_post_ack_direct(l) : nd_send_ack(l);
    }
    return true;
}

// Moves every chunk that has arrived into rx_spill, reposts its receive and acks as nd_recv_raw
// would have. rx_ready only ever holds bytes that come after rx_spill, so order is kept. With a
// small twin, chunks move in data sequence order and stop at the first one still missing.
static bool nd_spill_rx(dist_nd_conn * c) {
    c = c->root;
    for (;;) {
        dist_nd_conn * l = nullptr;
        if (!nd_rx_lane(c, &l)) {
            return false;
        }
        if (l == nullptr) {
            return true;
        }
        const nd_rx_item it = l->rx_ready.front();
        const uint8_t * p = l->rx_slot(it.slot) + sizeof(nd_msg_hdr) + it.off;
        c->rx_spill.insert(c->rx_spill.end(), p, p + (it.len - it.off));
        if (!nd_rx_retire(c, l, true)) {
            return false;
        }
    }
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
    // the list holds roots; c may be a small twin, whose completions and spill go through its root
    if (std::find(all.begin(), all.end(), c->root) == all.end()) {
        all.push_back(c->root);
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
                    o->peer_gone       = true;
                    c->peer_gone       = true;
                    c->root->peer_gone = true;
                    return false;
                }
            }
        }
    }
}

// Posts the chunk staged in the current slot. The caller has made sure the window allows it: either
// fewer than ND_WINDOW chunks are unacknowledged, or an ack is in hand.
static bool nd_post_staged(dist_nd_conn * c) {
    if (c->tx_since_ack >= ND_WINDOW) {
        c->ack_pending--;
        c->tx_since_ack = 0;
    }
    const int slot = c->tx_slot_cur;
    // with a small twin the data sequence runs over both connections, and the receiver reads by it
    const uint32_t seq = c->root->sm ? c->root->tx_dseq++ : c->tx_seq++;
    const nd_msg_hdr hdr = { ND_MSG_MAGIC, ND_MSG_DATA, (uint32_t)c->tx_fill, seq };
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
    }
    return nd_post_staged(c);
}

static bool nd_send_raw(dist_nd_conn * c, const void * data, size_t size) {
    const uint8_t * src = (const uint8_t *)data;
    size_t rem = size;
    while (rem > 0) {
        if (c->tx_fill == c->pay_max && !nd_flush(c)) {
            return false;
        }
        // Starting a fresh chunk: the slot must not still be on the wire before we write it.
        if (c->tx_fill == 0 && !nd_tx_slot_ready(c)) {
            return false;
        }
        const size_t take = std::min(rem, c->pay_max - c->tx_fill);
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
    if (!nd_flush(c) || (c->sm && !nd_flush(c->sm))) {
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
        dist_nd_conn * l = nullptr;
        for (;;) {
            if (!nd_rx_lane(c, &l)) {
                return false;
            }
            if (l != nullptr) {
                break;
            }
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
        nd_rx_item & it = l->rx_ready.front();
        const size_t take = std::min((size_t)(it.len - it.off), rem);
        memcpy(dst, l->rx_slot(it.slot) + sizeof(nd_msg_hdr) + it.off, take);
        dst += take;
        rem -= take;
        it.off += (uint32_t)take;
        if (it.off == it.len && !nd_rx_retire(c, l, false)) {
            return false;
        }
    }
    return true;
}

// The connection a message of total bytes (header and payload) goes out on: c's small twin when
// the whole message fits in one of its chunks, else c.
static dist_nd_conn * nd_tx_lane(dist_nd_conn * c, size_t total) {
    return (c->sm && total <= c->sm->pay_max) ? c->sm : c;
}

// every posted send of c and of its twin retired
static bool nd_drain_all(dist_nd_conn * c) {
    return nd_send_drain_local(c) && (!c->sm || nd_send_drain_local(c->sm));
}

// drop any half-staged chunk, of c and of its twin; the connection is finished either way
static void nd_drop_staged(dist_nd_conn * c) {
    c->tx_fill = 0;
    if (c->sm) {
        c->sm->tx_fill = 0;
    }
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
        if (local_ok && nd_small_lane() && !nd_add_small_lane(conn.get())) {
            conn.reset();
            local_ok = false;
            err = "NetworkDirect small connection setup failed";
        }
        if (local_ok && !nd_listen(conn.get())) {
            conn.reset();
            local_ok = false;
            err = "NetworkDirect listener setup failed";
        }

        nd_hello out = { ND_HELLO_MAGIC, nd_hello_ver(), local_ok ? (uint32_t)addr.sin_addr.s_addr : 0u,
                         local_ok ? 1u : 0u };
        if (!tcp_write_all(tcp_out, &out, sizeof(out))) {
            err = "failed to advertise the NetworkDirect address to the peer";
            return nullptr;
        }

        if (local_ok) {
            ND_LOG_DBG("advertised %s, accepting\n", nd_addr_str(addr).s);
            if (!nd_accept(conn.get(), conn.get())) {
                conn.reset();
                local_ok = false;
                err = "NetworkDirect accept failed";
            } else if (conn->sm && !nd_accept(conn.get(), conn->sm)) {
                // the peer dials its small connection right after the first one completes
                conn.reset();
                local_ok = false;
                err = "NetworkDirect accept of the small connection failed";
            }
        }

        nd_hello in = {};
        if (!tcp_read_all(tcp_in, &in, sizeof(in))) {
            err = "failed to read the peer's NetworkDirect result";
            return nullptr;
        }
        if (in.magic != ND_HELLO_MAGIC || in.version != nd_hello_ver()) {
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
        if (in.magic != ND_HELLO_MAGIC || in.version != nd_hello_ver()) {
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

        const nd_hello out = { ND_HELLO_MAGIC, nd_hello_ver(), 0u, local_ok ? 1u : 0u };
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
    // the whole message goes out on one connection: the small twin when it fits in one of its chunks
    dist_nd_conn * l = nd_tx_lane(c, hdr_size + payload_size);
    // cork across both writes so the header and its payload leave in one chunk
    l->corked++;
    bool ok = nd_send_raw(l, hdr, hdr_size);
    if (ok && payload_size > 0) {
        ok = nd_send_raw(l, payload, payload_size);
    }
    l->corked--;
    if (ok) {
        ok = nd_flush(l);
    }
    // Default retires the send before returning, identical to the previous behaviour. Full-duplex
    // mode drains after the matching receive instead, which is the whole point: both directions
    // are then on the wire at once.
    if (ok && !nd_full_duplex()) {
        ok = nd_send_drain_local(l);
    }
    if (!ok) {
        nd_drop_staged(c);
        err = c->peer_gone ? "peer closed connection" : "NetworkDirect send failed";
        return false;
    }
    return true;
}

bool dist_nd_send_drain(dist_nd_conn * c, std::string & err) {
    if (!c) {
        return true;
    }
    if (!nd_drain_all(c)) {
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

// ---------------------------------------------------------------------------------------
// two-peer exchange for the 3-rank full mesh (multi-chunk mode only)
//
// One message, a header followed by its payload, leaves for both peers while one message of the
// same size arrives from each, all four directions at once. Nothing here waits on one connection
// while the other could move: the two halves below stage, post and consume only what is possible
// right now, and the loop in dist_nd_exchange_pair polls both completion queues between rounds.
// Received bytes go straight into the caller's buffers, so a payload of many chunks costs no copy
// beyond the staging one, and the acks for them leave as soon as a window's worth is consumed, so
// neither peer stalls at its send window while this rank is still busy sending.
// ---------------------------------------------------------------------------------------

// Stages and posts as much of [hdr | payload] on c as it can take without waiting. pos counts the
// message bytes already staged. Stops at the send window (an ack has to arrive first) and when
// every staging slot is still on the wire (a send completion has to retire one first).
static bool nd_tx_pump(dist_nd_conn * c, const uint8_t * hdr, size_t hdr_size,
                       const uint8_t * payload, size_t size, size_t & pos, bool & moved) {
    const size_t total = hdr_size + size;
    c = nd_tx_lane(c, total);   // the connection this whole message goes out on
    for (;;) {
        // a full chunk, or the last bytes of the message, leaves as soon as the window allows
        if (c->tx_fill == c->pay_max || (pos == total && c->tx_fill > 0)) {
            if (c->tx_since_ack >= ND_WINDOW && c->ack_pending == 0) {
                return true;
            }
            if (!nd_post_staged(c)) {
                return false;
            }
            moved = true;
            continue;
        }
        if (pos == total) {
            return true;
        }
        // a fresh chunk: the slot must not still be on the wire before it is written
        if (c->tx_fill == 0 && c->tx_inflight >= ND_TX_DEPTH) {
            return true;
        }
        size_t take = c->pay_max - c->tx_fill;
        const uint8_t * src;
        if (pos < hdr_size) {
            take = std::min(take, hdr_size - pos);
            src  = hdr + pos;
        } else {
            take = std::min(take, total - pos);
            src  = payload + (pos - hdr_size);
        }
        memcpy(c->tx_slot(c->tx_slot_cur) + sizeof(nd_msg_hdr) + c->tx_fill, src, take);
        c->tx_fill += take;
        pos += take;
        moved = true;
    }
}

// Copies whatever has arrived on c into [hdr | payload], without waiting: bytes spilled earlier
// first, then the receive ring, exactly the order nd_recv_raw reads them in. pos counts the
// message bytes received so far. Bytes past the end of this message stay where they are, for the
// next one.
static bool nd_rx_pump(dist_nd_conn * c, uint8_t * hdr, size_t hdr_size,
                       uint8_t * payload, size_t size, size_t & pos, bool & moved) {
    const size_t total = hdr_size + size;
    auto put = [&](const uint8_t * src, size_t n) {
        while (n > 0) {
            size_t    take = n;
            uint8_t * dst;
            if (pos < hdr_size) {
                take = std::min(take, hdr_size - pos);
                dst  = hdr + pos;
            } else {
                dst  = payload + (pos - hdr_size);
            }
            memcpy(dst, src, take);
            src += take;
            n   -= take;
            pos += take;
        }
    };
    while (pos < total && c->rx_spill_off < c->rx_spill.size()) {
        const size_t take = std::min(c->rx_spill.size() - c->rx_spill_off, total - pos);
        put(c->rx_spill.data() + c->rx_spill_off, take);
        c->rx_spill_off += take;
        if (c->rx_spill_off == c->rx_spill.size()) {
            c->rx_spill.clear();
            c->rx_spill_off = 0;
        }
        moved = true;
    }
    while (pos < total && c->rx_spill.empty()) {
        dist_nd_conn * l = nullptr;
        if (!nd_rx_lane(c, &l)) {
            return false;
        }
        if (l == nullptr) {
            break;
        }
        nd_rx_item & it = l->rx_ready.front();
        const size_t take = std::min((size_t)(it.len - it.off), total - pos);
        put(l->rx_slot(it.slot) + sizeof(nd_msg_hdr) + it.off, take);
        it.off += (uint32_t)take;
        moved = true;
        if (it.off == it.len && !nd_rx_retire(c, l, true)) {
            return false;
        }
    }
    return true;
}

bool dist_nd_exchange_pair(dist_nd_conn * a, dist_nd_conn * b, const void * hdr, size_t hdr_size,
                           const void * payload, size_t size, void * hdr_from_a, void * from_a,
                           void * hdr_from_b, void * from_b, std::string & err) {
    if (a == nullptr || b == nullptr || a == b) {
        err = "two-peer exchange needs two distinct ND connections";
        return false;
    }
    if (!nd_multichunk()) {
        err = "two-peer exchange needs LLAMA_DIST_TP_RDMA_MULTICHUNK";
        return false;
    }
    const uint8_t * h   = (const uint8_t *)hdr;
    const uint8_t * p   = (const uint8_t *)payload;
    const size_t    total = hdr_size + size;
    size_t tx_a = 0, tx_b = 0, rx_a = 0, rx_b = 0;

    const int64_t tick       = nd_tick_ticks();
    int64_t       last_check = nd_qpc();
    uint32_t      idle_steps = 0;
    bool ok = true;
    for (;;) {
        bool moved = false;
        ok = nd_tx_pump(a, h, hdr_size, p, size, tx_a, moved) &&
             nd_tx_pump(b, h, hdr_size, p, size, tx_b, moved) &&
             nd_rx_pump(a, (uint8_t *)hdr_from_a, hdr_size, (uint8_t *)from_a, size, rx_a, moved) &&
             nd_rx_pump(b, (uint8_t *)hdr_from_b, hdr_size, (uint8_t *)from_b, size, rx_b, moved);
        if (!ok) {
            break;
        }
        if (tx_a == total && tx_b == total && nd_tx_lane(a, total)->tx_fill == 0 &&
            nd_tx_lane(b, total)->tx_fill == 0 && rx_a == total && rx_b == total) {
            break;
        }
        int stepped = 0;
        for (dist_nd_conn * c : { a, b }) {
            const int r = nd_try_step(c);
            if (r < 0) {
                ok = false;
                break;
            }
            stepped += r;
        }
        if (!ok) {
            break;
        }
        if (moved) {
            idle_steps = 0;
            continue;
        }
        if (stepped > 0) {
            // completions that never let anything move mean a queue is replaying something stale
            if (++idle_steps > ND_MAX_IDLE_STEPS) {
                ND_LOG_ERR("two-peer exchange consumed %u completions without moving data\n", idle_steps);
                ok = false;
                break;
            }
            continue;
        }
        YieldProcessor();
        const int64_t now = nd_qpc();
        if (now - last_check >= tick) {
            last_check = now;
            for (dist_nd_conn * c : { a, b }) {
                if (nd_tcp_peer_closed(c->tcp_in) || nd_tcp_peer_closed(c->tcp_out)) {
                    ND_LOG_DBG("peer dropped the ring socket\n");
                    c->peer_gone = true;
                    ok = false;
                }
            }
            if (!ok) {
                break;
            }
        }
    }
    if (ok && !nd_full_duplex()) {
        ok = nd_drain_all(a) && nd_drain_all(b);
    }
    if (!ok) {
        nd_drop_staged(a);   // drop any half-staged chunk; the connections are finished either way
        nd_drop_staged(b);
        err = (a->peer_gone || b->peer_gone) ? "peer closed connection" : "NetworkDirect two-peer exchange failed";
        return false;
    }
    return true;
}

// n-peer exchange for the 4-rank full mesh: dist_nd_exchange_pair's loop over an array of
// connections. Nothing waits on one connection while another could move.
bool dist_nd_exchange_n(dist_nd_conn * const * conns, int n, const void * hdr, size_t hdr_size,
                        const void * payload, size_t size, void * const * hdr_from, void * const * from,
                        std::string & err) {
    constexpr int ND_XN_MAX = 4;
    if (n < 1 || n > ND_XN_MAX) {
        err = "n-peer exchange takes 1 to 4 ND connections";
        return false;
    }
    for (int i = 0; i < n; ++i) {
        if (conns[i] == nullptr) {
            err = "n-peer exchange needs every ND connection up";
            return false;
        }
        for (int j = 0; j < i; ++j) {
            if (conns[j] == conns[i]) {
                err = "n-peer exchange needs distinct ND connections";
                return false;
            }
        }
    }
    if (!nd_multichunk()) {
        err = "n-peer exchange needs LLAMA_DIST_TP_RDMA_MULTICHUNK";
        return false;
    }
    const uint8_t * h     = (const uint8_t *)hdr;
    const uint8_t * p     = (const uint8_t *)payload;
    const size_t    total = hdr_size + size;
    size_t tx[ND_XN_MAX] = {};
    size_t rx[ND_XN_MAX] = {};

    const int64_t tick       = nd_tick_ticks();
    int64_t       last_check = nd_qpc();
    uint32_t      idle_steps = 0;
    bool ok = true;
    for (;;) {
        bool moved = false;
        for (int i = 0; ok && i < n; ++i) {
            ok = nd_tx_pump(conns[i], h, hdr_size, p, size, tx[i], moved);
        }
        for (int i = 0; ok && i < n; ++i) {
            ok = nd_rx_pump(conns[i], (uint8_t *)hdr_from[i], hdr_size, (uint8_t *)from[i], size, rx[i], moved);
        }
        if (!ok) {
            break;
        }
        bool done = true;
        for (int i = 0; i < n; ++i) {
            if (tx[i] != total || nd_tx_lane(conns[i], total)->tx_fill != 0 || rx[i] != total) {
                done = false;
            }
        }
        if (done) {
            break;
        }
        int stepped = 0;
        for (int i = 0; i < n; ++i) {
            const int r = nd_try_step(conns[i]);
            if (r < 0) {
                ok = false;
                break;
            }
            stepped += r;
        }
        if (!ok) {
            break;
        }
        if (moved) {
            idle_steps = 0;
            continue;
        }
        if (stepped > 0) {
            // completions that never let anything move mean a queue is replaying something stale
            if (++idle_steps > ND_MAX_IDLE_STEPS) {
                ND_LOG_ERR("n-peer exchange consumed %u completions without moving data\n", idle_steps);
                ok = false;
                break;
            }
            continue;
        }
        YieldProcessor();
        const int64_t now = nd_qpc();
        if (now - last_check >= tick) {
            last_check = now;
            for (int i = 0; i < n; ++i) {
                dist_nd_conn * c = conns[i];
                if (nd_tcp_peer_closed(c->tcp_in) || nd_tcp_peer_closed(c->tcp_out)) {
                    ND_LOG_DBG("peer dropped the ring socket\n");
                    c->peer_gone = true;
                    ok = false;
                }
            }
            if (!ok) {
                break;
            }
        }
    }
    if (ok && !nd_full_duplex()) {
        for (int i = 0; ok && i < n; ++i) {
            ok = nd_drain_all(conns[i]);
        }
    }
    if (!ok) {
        bool gone = false;
        for (int i = 0; i < n; ++i) {
            nd_drop_staged(conns[i]);   // drop any half-staged chunk; the connections are finished either way
            gone = gone || conns[i]->peer_gone;
        }
        err = gone ? "peer closed connection" : "NetworkDirect n-peer exchange failed";
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
        // still has a transfer against it. The bye goes on c; the peer polls both connections.
        (void) nd_drain_all(c);
        (void) nd_send_bye(c);
        (void) nd_drain_all(c);
    }
    delete c;
}

#endif // LLAMA_DIST_TP_ND
