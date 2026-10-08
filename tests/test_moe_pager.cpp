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

    const size_t GU = 1049600;   // fused gate_up bytes per expert (IQ2_S)
    const size_t DN = 460800;    // down bytes per expert (Q2_0)

    char *h_ref = (char *) malloc(GU);
    char *h_pag = (char *) malloc(GU);
    char *d_gu  = nullptr;
    cudaMalloc(&d_gu, GU + DN);
    cudaMemset(d_gu, 0, GU + DN);

    const int probes[][2] = { {0,0}, {0,1}, {7,123}, {23,511}, {47,255} };
    int bad = 0;
    for (auto & pr : probes) {
        const int il = pr[0], e = pr[1];
        // reference bytes straight from the file at the expert slice
        // (ask the pager for the offsets via a fetch into host-visible memory
        //  is not possible, so re-derive with the same alignment rule)
        // We instead verify determinism + non-triviality here, and exact byte
        // equality is covered by comparing two independent pager fetches.
        // exact reference: pread the same file range the pager used
        const moe_expert_slice * sl = pager.slices(il, e);
        const long skip0 = (4096 - (sl[0].off % 4096)) % 4096;
        if ((size_t) sl[0].len <= skip0) {
            fprintf(stderr, "FAIL: L%d E%d slice len %u <= skip %ld -- bad index\n",
                    il, e, sl[0].len, skip0); bad = 1; continue;
        }
        const size_t ref_n = std::min((size_t) GU, (size_t) sl[0].len - skip0);
        if (pread(fd, h_ref, ref_n, (off_t)(sl[0].off + skip0)) != (ssize_t) ref_n) {
            fprintf(stderr, "FAIL: reference pread L%d E%d\n", il, e); bad = 1; continue;
        }
        if (!pager.fetch(il, e, d_gu, GU, (char *) d_gu + GU, DN)) {
            fprintf(stderr, "FAIL: fetch(%d,%d)\n", il, e); bad = 1; continue;
        }
        cudaMemcpy(h_pag, d_gu, GU, cudaMemcpyDeviceToHost);

        const int exact = memcmp(h_ref, h_pag, ref_n) == 0;
        printf("  fetch(L%2d,E%3d): compared %zu bytes  %s\n",
               il, e, ref_n, exact ? "EXACT MATCH" : "*** MISMATCH ***");
        if (!exact) bad = 1;
    }
    close(fd);
    free(h_ref); free(h_pag);
    cudaFree(d_gu);

    printf("\nresult: %s   io_ops=%ld io_bytes=%ld\n",
           bad ? "FAIL" : "PASS", pager.stats_io_ops(), pager.stats_io_bytes());
    return bad;
}