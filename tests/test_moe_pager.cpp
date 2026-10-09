// test_moe_pager.cpp -- standalone verification for the O_DIRECT expert pager.
//
// Proves the thing that matters before it is wired into the graph: the bytes
// the pager DMAs for a given (layer, expert) are byte-identical to the bytes
// sitting at that expert's slice in the GGUF.
//
// It does this by reading the same slice twice through two independent paths
// (a plain buffered pread, and the pager's O_DIRECT+io_uring path) and comparing.
#include "moe_pager.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>

static const char * MODEL =
    "/mnt/windows/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS/IQ2_XS/"
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf";

int main() {
    const int NL = 48, NE = 512, NU = 10;

    moe_pager pager;
    std::string err;
    if (!pager.init(MODEL, NL, NE, NU, err)) {
        fprintf(stderr, "FAIL: pager.init: %s\n", err.c_str());
        return 1;
    }
    printf("pager: layers=%d experts=%d used=%d bytes/expert=%zu (%.3f MiB)\n",
           pager.n_layer(), pager.n_expert(), pager.n_used(),
           pager.bytes_per_expert(), pager.bytes_per_expert() / 1048576.0);

    // reference: plain buffered read of the fused gate_up slice for a few
    // (layer, expert) pairs, using the same offsets the pager derived.
    int fd = open(MODEL, O_RDONLY);
    if (fd < 0) { perror("open model"); return 1; }

    // separate gate/up path: per-role want sizes come from the pager's own
    // index (len minus pre-pad), which also tracks per-layer mixed quants
    const int probes[][2] = { {0,0}, {0,1}, {7,123}, {23,511}, {47,255} };
    int bad = 0;
    size_t biggest = 0;
    for (auto & pr : probes) {
        const moe_expert_slice * sl = pager.slices(pr[0], pr[1]);
        for (int r = 0; r < 3; r++) biggest = std::max(biggest, (size_t) sl[r].len - sl[r].pre);
    }
    char *h_ref = (char *) malloc(biggest);
    char *h_pag = (char *) malloc(biggest);
    char *d_dst = nullptr;
    cudaMalloc(&d_dst, 3 * biggest);
    cudaMemset(d_dst, 0, 3 * biggest);
    for (auto & pr : probes) {
        const int il = pr[0], e = pr[1];
        const moe_expert_slice * sl = pager.slices(il, e);
        size_t want[3];
        for (int r = 0; r < 3; r++) want[r] = (size_t) sl[r].len - sl[r].pre;
        void *dg = d_dst, *du = d_dst + want[0], *dd = d_dst + want[0] + want[1];
        if (!pager.fetch(il, e, dg, want[0], du, want[1], dd, want[2])) {
            fprintf(stderr, "FAIL: fetch(%d,%d)\n", il, e); bad = 1; continue;
        }
        bool ok = true;
        const void *dsts[3] = { dg, du, dd };
        const char *roles[3] = { "gate", "up", "down" };
        for (int r = 0; r < 3 && ok; r++) {
            if (pread(fd, h_ref, want[r], (off_t)(sl[r].off + sl[r].pre)) != (ssize_t) want[r]) {
                fprintf(stderr, "FAIL: reference pread L%d E%d %s\n", il, e, roles[r]); bad = 1; ok = false; break;
            }
            cudaMemcpy(h_pag, dsts[r], want[r], cudaMemcpyDeviceToHost);
            if (memcmp(h_ref, h_pag, want[r]) != 0) ok = false;
        }
        printf("  fetch(L%2d,E%3d): compared %zu bytes  %s\n",
               il, e, want[0] + want[1] + want[2], ok ? "EXACT MATCH" : "*** MISMATCH ***");
        if (!ok) bad = 1;
    }
    // Async path: queue prefetches, then consume them via commit and verify
    // byte-exactness. This is the exact sequence the prefetch pipeline uses:
    // non-blocking reads issued ahead, demand commits that find them in flight.
    // Use per-layer pool geometry (gate/up/down byte sizes differ by layer).
    {
        const int aprobes[][2] = { {1,19}, {8,44}, {11,300}, {37,7} };
        // issue all prefetches first so reads overlap in the ring
        for (auto & pr : aprobes) {
            if (!pager.prefetch(pr[0], pr[1])) {
                fprintf(stderr, "FAIL: prefetch(%d,%d) refused\n", pr[0], pr[1]); bad = 1;
            }
        }
        for (auto & pr : aprobes) {
            const int il = pr[0], e = pr[1];
            const moe_expert_slice * sl = pager.slices(il, e);
            // per-role want sizes from the slice lengths minus pre-pad
            // (pool geometry matches the file's per-expert bytes here)
            size_t want[3];
            for (int r = 0; r < 3; r++) want[r] = (size_t) sl[r].len - sl[r].pre;
            // destination sized generously; commit only writes want[] bytes
            static char *d_dst = nullptr;
            static size_t d_cap = 0;
            size_t need = want[0] + want[1] + want[2];
            if (need > d_cap) {
                if (d_dst) cudaFree(d_dst);
                cudaMalloc(&d_dst, need);
                d_cap = need;
            }
            void *dg = d_dst, *du = d_dst + want[0], *dd = d_dst + want[0] + want[1];
            if (!pager.commit(il, e, dg, want[0], du, want[1], dd, want[2])) {
                fprintf(stderr, "FAIL: commit(%d,%d)\n", il, e); bad = 1; continue;
            }
            bool ok = true;
            const void *dsts[3] = { dg, du, dd };
            for (int r = 0; r < 3 && ok; r++) {
                std::vector<char> ref(want[r]), got(want[r]);
                if (pread(fd, ref.data(), want[r], (off_t)(sl[r].off + sl[r].pre)) != (ssize_t) want[r]) {
                    fprintf(stderr, "FAIL: reference pread async L%d E%d r%d\n", il, e, r); bad = 1; ok = false; break;
                }
                cudaMemcpy(got.data(), dsts[r], want[r], cudaMemcpyDeviceToHost);
                if (memcmp(ref.data(), got.data(), want[r]) != 0) { ok = false; }
            }
            printf("  async(L%2d,E%3d): %zu bytes via prefetch+commit  %s (pf_hit=%ld)\n",
                   il, e, need, ok ? "EXACT MATCH" : "*** MISMATCH ***", pager.stats_pf_hit());
            if (!ok) bad = 1;
        }
    }
    close(fd);
    free(h_ref); free(h_pag);
    cudaFree(d_dst);

    printf("\nresult: %s   io_ops=%ld io_bytes=%ld pf_hit=%ld\n",
           bad ? "FAIL" : "PASS", pager.stats_io_ops(), pager.stats_io_bytes(), pager.stats_pf_hit());
    return bad;
}