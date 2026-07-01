#include "llama-tp-net.h"
#include <stdlib.h>
#include <string.h>

// Process-wide CPU-TP config, set once from llama_model_params at model-load time
// (llama_tp_set_config). Left at defaults (size <= 1), the accessors fall back to the legacy
// LLAMA_TP_* environment variables so existing env-driven launches keep working. moe_mode uses the
// same 0/1/2 encoding as llama_moe_parallel_mode / tp_moe_mode. peer/port feed the all-reduce bootstrap.
static struct { int size, rank, moe_mode, attn, ssm, port; const char * peer; } g_tp = { 1, 0, 0, 0, 0, 0, NULL };
static char g_tp_peer[256];

void llama_tp_set_config(int size, int rank, int moe_mode, int attn, int ssm, const char * peer, int port) {
    g_tp.size = size; g_tp.rank = rank; g_tp.moe_mode = moe_mode;
    g_tp.attn = attn; g_tp.ssm = ssm; g_tp.port = port;
    // copy peer: it is consumed at inference time (the lazy all-reduce bootstrap), after the caller's
    // llama_model_params may be gone.
    if (peer && peer[0]) {
        strncpy(g_tp_peer, peer, sizeof(g_tp_peer) - 1);
        g_tp_peer[sizeof(g_tp_peer) - 1] = '\0';
        g_tp.peer = g_tp_peer;
    } else {
        g_tp.peer = NULL;
    }
}

// the API config takes precedence once it actually requests TP (size > 1); otherwise read the env.
static int tp_cfg_active(void) { return g_tp.size > 1; }

int llama_tp_size(void) {
    if (tp_cfg_active()) return g_tp.size;
    const char * s = getenv("LLAMA_TP_SIZE");
    return s ? atoi(s) : 1;
}

int llama_tp_rank(void) {
    if (tp_cfg_active()) return g_tp.rank;
    const char * r = getenv("LLAMA_TP_RANK");
    return r ? atoi(r) : 0;
}

int llama_tp_enabled(void) {
    return llama_tp_size() > 1;
}

int llama_tp_attn_enabled(void) {
    if (!llama_tp_enabled()) return 0;
    if (tp_cfg_active()) return g_tp.attn != 0;
    const char * a = getenv("LLAMA_TP_ATTN");
    return a && atoi(a) != 0;
}

// SSM/Mamba-2 mixer sharding: when set, the recurrent SSM layers shard their heads/d_inner across
// ranks (channel-parallel) with one all-reduce after ssm_out. Separate from attn so a hybrid can
// shard SSM without standard attention and vice-versa.
int llama_tp_ssm_enabled(void) {
    if (!llama_tp_enabled()) return 0;
    if (tp_cfg_active()) return g_tp.ssm != 0;
    const char * s = getenv("LLAMA_TP_SSM");
    return s && atoi(s) != 0;
}

// MoE-parallel mode (matches tp_moe_mode): 0=off, 1=expert-parallel, 2=tensor-parallel(experts).
// env LLAMA_TP_MOE = "tp"/"tensor" -> 2, "ep"/"expert"/"1" -> 1 (back-compat), else -> 0.
int llama_tp_moe_mode(void) {
    if (!llama_tp_enabled()) return 0;
    if (tp_cfg_active()) return g_tp.moe_mode;
    const char * m = getenv("LLAMA_TP_MOE");
    if (!m) return 0;
    if (strcmp(m, "tp") == 0 || strcmp(m, "tensor") == 0) return 2;
    if (strcmp(m, "ep") == 0 || strcmp(m, "expert") == 0 || atoi(m) != 0) return 1;
    return 0;
}

int llama_tp_moe_enabled(void) {
    return llama_tp_moe_mode() != 0;
}

const char * llama_tp_peer(void) {
    if (tp_cfg_active() && g_tp.peer) return g_tp.peer;
    return getenv("LLAMA_TP_PEER");
}

int llama_tp_port(void) {
    if (tp_cfg_active() && g_tp.port > 0) return g_tp.port;
    const char * p = getenv("LLAMA_TP_PORT");
    return p ? atoi(p) : 13700;
}

#ifdef LLAMA_TP_UCX

#define _GNU_SOURCE
#include <ucp/api/ucp.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

// N-way (power-of-2) recursive-doubling all-reduce over UCX/IB. rank = a TP shard (one process
// per node, or one per socket for NUMA-local rank=socket). At step s a rank exchanges + sums its
// vector with partner (rank XOR (1<<s)); after log2(N) steps every rank holds the global sum.
// Rank layout is chosen by the launcher so step 0 pairs are INTRA-node (UCX picks sm -> cheap) and
// later steps are inter-node (rc) -> the hierarchy comes for free with no special-casing here.
// Bootstrap: rank 0 is a TCP coordinator (on the IB subnet) that gathers every rank's UCX worker
// address and broadcasts the full list; each rank then opens eps to its log2(N) partners.

#define TP_TAG 0x7A1C
#define TP_TMP_FLOATS (4u << 20)   // 16 MB scratch (max all-reduce length = n_embd * n_tokens)
#define TP_MAXR 16                 // max ranks (=> up to 4 recursive-doubling steps)

struct tp_net {
    ucp_context_h ctx;
    ucp_worker_h  worker;
    ucp_ep_h      eps[8];   // one ep per recursive-doubling step (partner rank^(1<<s))
    int           nsteps;   // log2(size)
    int           rank;
    int           size;
    float *       tmp;
};

static int writen(int fd, const void * b, size_t n) {
    size_t o = 0; while (o < n) { ssize_t k = write(fd, (const char *)b + o, n - o); if (k <= 0) return -1; o += (size_t)k; } return 0;
}
static int readn(int fd, void * b, size_t n) {
    size_t o = 0; while (o < n) { ssize_t k = read(fd, (char *)b + o, n - o); if (k <= 0) return -1; o += (size_t)k; } return 0;
}

static int tcp_listen(int port) {
    int s = socket(AF_INET, SOCK_STREAM, 0); int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a; memset(&a, 0, sizeof a);
    a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = INADDR_ANY;
    if (bind(s, (void *)&a, sizeof a)) { perror("tp bind"); return -1; }
    listen(s, TP_MAXR); return s;
}
static int tcp_client(const char * ip, int port) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a; memset(&a, 0, sizeof a);
    a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = inet_addr(ip);
    for (int t = 0; connect(s, (void *)&a, sizeof a); t++) { if (t > 1200) return -1; usleep(100000); }
    return s;
}

// Gather all ranks' worker addresses at rank 0 and broadcast the full array to everyone.
// addrs[i]/lens[i] are filled for every rank i on return (caller frees the non-self entries).
static int tp_bootstrap(int rank, int size, const char * server_ip, int port,
                        void * myaddr, size_t mylen, void ** addrs, size_t * lens) {
    addrs[rank] = myaddr; lens[rank] = mylen;
    if (rank == 0) {
        int srv = tcp_listen(port);
        if (srv < 0) return -1;
        int fds[TP_MAXR];
        for (int i = 0; i < size; i++) fds[i] = -1;
        for (int i = 0; i < size - 1; i++) {
            int c = accept(srv, 0, 0);
            uint32_t pr = 0; uint64_t l = 0;
            if (readn(c, &pr, 4) || readn(c, &l, 8)) { close(c); close(srv); return -1; }
            void * a = malloc(l);
            if (readn(c, a, l)) { free(a); close(c); close(srv); return -1; }
            if (pr >= (uint32_t)size) { free(a); close(c); close(srv); return -1; }
            addrs[pr] = a; lens[pr] = l; fds[pr] = c;
        }
        close(srv);
        for (int pr = 1; pr < size; pr++) {
            int c = fds[pr];
            uint32_t n = (uint32_t)size; writen(c, &n, 4);
            for (int j = 0; j < size; j++) { uint64_t l = lens[j]; writen(c, &l, 8); writen(c, addrs[j], l); }
            close(c);
        }
    } else {
        int c = tcp_client(server_ip, port);
        if (c < 0) return -1;
        uint32_t pr = (uint32_t)rank; uint64_t l = mylen;
        if (writen(c, &pr, 4) || writen(c, &l, 8) || writen(c, myaddr, mylen)) { close(c); return -1; }
        uint32_t n = 0;
        if (readn(c, &n, 4) || n != (uint32_t)size) { close(c); return -1; }
        for (int j = 0; j < size; j++) {
            uint64_t ll = 0; if (readn(c, &ll, 8)) { close(c); return -1; }
            void * a = malloc(ll);                       // consume this entry off the socket
            if (readn(c, a, ll)) { free(a); close(c); return -1; }
            if (j == rank) { free(a); addrs[j] = myaddr; lens[j] = mylen; }  // keep our own addr
            else { addrs[j] = a; lens[j] = ll; }
        }
        close(c);
    }
    return 0;
}

static void wait_req(ucp_worker_h w, void * req) {
    if (req == NULL) return;
    if (UCS_PTR_IS_ERR(req)) { fprintf(stderr, "tp req error\n"); abort(); }
    while (ucp_request_check_status(req) == UCS_INPROGRESS) ucp_worker_progress(w);
    ucp_request_free(req);
}

static int ilog2(int n) { int k = 0; while ((1 << k) < n) k++; return k; }

static struct tp_net * tp_net_init(int rank, int size, const char * server_ip, int port) {
    if (size < 2 || size > TP_MAXR || (size & (size - 1)) != 0) {
        fprintf(stderr, "llama-tp: size %d must be a power of 2 in [2,%d]\n", size, TP_MAXR);
        return NULL;
    }
    struct tp_net * net = calloc(1, sizeof(struct tp_net));
    net->rank = rank; net->size = size; net->nsteps = ilog2(size);
    net->tmp = aligned_alloc(64, TP_TMP_FLOATS * sizeof(float));
    ucp_config_t * cfg; ucp_config_read(NULL, NULL, &cfg);
    ucp_params_t up; up.field_mask = UCP_PARAM_FIELD_FEATURES; up.features = UCP_FEATURE_TAG;
    if (ucp_init(&up, cfg, &net->ctx) != UCS_OK) { fprintf(stderr, "tp ucp_init fail\n"); return NULL; }
    ucp_config_release(cfg);
    ucp_worker_params_t wp; wp.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE; wp.thread_mode = UCS_THREAD_MODE_SINGLE;
    ucp_worker_create(net->ctx, &wp, &net->worker);
    ucp_address_t * myaddr; size_t myaddr_len; ucp_worker_get_address(net->worker, &myaddr, &myaddr_len);

    void * addrs[TP_MAXR]; size_t lens[TP_MAXR];
    for (int i = 0; i < size; i++) { addrs[i] = NULL; lens[i] = 0; }
    if (tp_bootstrap(rank, size, server_ip, port, myaddr, myaddr_len, addrs, lens) != 0) {
        fprintf(stderr, "tp bootstrap fail\n"); return NULL;
    }
    for (int s = 0; s < net->nsteps; s++) {
        int partner = rank ^ (1 << s);
        ucp_ep_params_t ep_p; ep_p.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
        ep_p.address = (ucp_address_t *)addrs[partner];
        if (ucp_ep_create(net->worker, &ep_p, &net->eps[s]) != UCS_OK) {
            fprintf(stderr, "tp ep fail (step %d -> rank %d)\n", s, partner); return NULL;
        }
    }
    for (int i = 0; i < size; i++) if (i != rank && addrs[i]) free(addrs[i]);
    ucp_worker_release_address(net->worker, myaddr);
    fprintf(stderr, "llama-tp: inter-node transport up (rank %d/%d, %d steps)\n", rank, size, net->nsteps);
    return net;
}

static struct tp_net * g_net = NULL;
static int g_init_done = 0;

// Optional in-inference all-reduce instrumentation (LLAMA_TP_TIME) and a comms-off timing mode
// (LLAMA_TP_NOCOMMS, which skips the network exchange -> WRONG output, RIGHT compute-only timing,
// to isolate the comms fraction of decode). Stats printed at process exit.
static int      g_time     = 0;
static int      g_nocomms  = 0;
static uint64_t g_ar_calls = 0;   // number of all-reduce ops driven
static uint64_t g_ar_ns    = 0;   // total ns spent in the exchange+sum
static uint64_t g_ar_floats = 0;  // total floats reduced (for avg size)

static uint64_t now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
static void tp_print_stats(void) {
    if (!g_ar_calls) return;
    fprintf(stderr, "llama-tp: all-reduce stats: calls=%llu  total=%.2f ms  avg=%.2f us  "
            "avg_floats=%llu  (%s)\n",
            (unsigned long long)g_ar_calls, g_ar_ns / 1e6, (g_ar_ns / 1e3) / (double)g_ar_calls,
            (unsigned long long)(g_ar_floats / g_ar_calls), g_nocomms ? "NOCOMMS" : "comms");
}

void llama_tp_allreduce_op(struct ggml_tensor * dst, const struct ggml_tensor * a,
                           int ith, int nth, void * userdata) {
    (void)a; (void)nth; (void)userdata;
    if (ith != 0) return;                       // single thread drives the UCX worker
    if (!g_init_done) {                          // lazy bootstrap on the worker thread, first call
        g_init_done = 1;
        g_time    = getenv("LLAMA_TP_TIME")    && atoi(getenv("LLAMA_TP_TIME"))    != 0;
        g_nocomms = getenv("LLAMA_TP_NOCOMMS") && atoi(getenv("LLAMA_TP_NOCOMMS")) != 0;
        if (g_time) atexit(tp_print_stats);
        if (llama_tp_size() > 1) {  // config (llama_model_params) or LLAMA_TP_* env fallback
            g_net = tp_net_init(llama_tp_rank(), llama_tp_size(), llama_tp_peer(), llama_tp_port());
        }
    }
    if (!g_net) return;
    float * buf = (float *) dst->data;
    size_t n = (size_t) ggml_nelements(dst);
    if (n > TP_TMP_FLOATS) { fprintf(stderr, "tp: count %zu too big\n", n); abort(); }
    const uint64_t t0 = g_time ? now_ns() : 0;
    if (!g_nocomms) {
        // recursive-doubling: log2(N) exchange+sum steps, distinct tag per step so a fast rank's
        // step-(s+1) message can't be grabbed by a slow rank's step-s recv.
        for (int s = 0; s < g_net->nsteps; s++) {
            ucp_request_param_t p; p.op_attr_mask = 0;
            ucp_tag_t tag = TP_TAG + s;
            void * sreq = ucp_tag_send_nbx(g_net->eps[s], buf, n * sizeof(float), tag, &p);
            void * rreq = ucp_tag_recv_nbx(g_net->worker, g_net->tmp, n * sizeof(float), tag, (ucp_tag_t)-1, &p);
            wait_req(g_net->worker, sreq);
            wait_req(g_net->worker, rreq);
            for (size_t i = 0; i < n; i++) buf[i] += g_net->tmp[i];
        }
    }
    if (g_time) { g_ar_ns += now_ns() - t0; g_ar_calls++; g_ar_floats += n; }
}

#else  // no UCX: op is a no-op

void llama_tp_allreduce_op(struct ggml_tensor * dst, const struct ggml_tensor * a,
                           int ith, int nth, void * userdata) {
    (void)dst; (void)a; (void)ith; (void)nth; (void)userdata;
}

#endif // LLAMA_TP_UCX
