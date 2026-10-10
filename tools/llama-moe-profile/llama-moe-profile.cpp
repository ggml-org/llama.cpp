// llama-moe-profile -- rank routed experts per layer by decode frequency.
//
// Reads a routing trace (uint8/uint16 [T, n_layer, n_used] expert ids) and
// writes a compact binary profile: u32 n_layer, u32 per_layer, then
// n_layer*per_layer u16 expert ids in rank order. The engine's hot seeder
// memory-maps this file instead of a checked-in C table, so one binary
// serves every model by regenerating the profile per workload.
//
// Usage:
//   llama-moe-profile --trace traces/qwen38/q38.ids.u8 --layers 48 \
//       --experts 512 --used 10 --top 72 -o data/profile-qwen4exp.bin
//   llama-moe-profile --check data/profile-qwen4exp.bin   (validate + print)
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

namespace {

void usage() {
    fprintf(stderr,
        "usage:\n"
        "  llama-moe-profile --trace IDS --layers N --experts N --used K --top T -o OUT\n"
        "  llama-moe-profile --check PROFILE\n");
}

struct opts {
    std::string trace, out;
    int layers = 0, experts = 0, used = 0, top = 72, width = 0; // width: 1=u8, 2=u16, 0=auto
    bool check = false;
};

bool parse(int argc, char ** argv, opts & o) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](std::string & dst) {
            if (++i >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); return false; }
            dst = argv[i]; return true;
        };
        std::string s;
        auto needi = [&](int & dst) {
            if (!need(s)) return false; dst = atoi(s.c_str()); return true;
        };
        if (a == "--trace") { if (!need(o.trace)) return false; }
        else if (a == "-o") { if (!need(o.out)) return false; }
        else if (a == "--layers") { if (!needi(o.layers)) return false; }
        else if (a == "--experts") { if (!needi(o.experts)) return false; }
        else if (a == "--used") { if (!needi(o.used)) return false; }
        else if (a == "--top") { if (!needi(o.top)) return false; }
        else if (a == "--check") { o.check = true; }
        else if (a[0] != '-') {
            if (o.check && o.out.empty()) { o.out = a; }
            else { fprintf(stderr, "unexpected arg %s\n", a.c_str()); return false; }
        }
        else if (a == "-h" || a == "--help") { usage(); exit(0); }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return false; }
    }
    return true;
}

bool build(const opts & o) {
    if (o.trace.empty() || o.out.empty() || o.layers <= 0 || o.experts <= 0 || o.used <= 0 || o.top <= 0) {
        usage(); return false;
    }
    FILE * f = fopen(o.trace.c_str(), "rb");
    if (!f) { perror("open trace"); return false; }
    fseek(f, 0, SEEK_END);
    const long nb = ftell(f);
    fseek(f, 0, SEEK_SET);
    const long row = (long) o.layers * o.used;
    // element width: explicit --width wins, else u16 when n_expert > 256
    int elem = o.width == 1 ? 1 : o.width == 2 ? 2 : (o.experts > 256 ? 2 : 1);
    if (nb % (row * elem) != 0) { fprintf(stderr, "trace size %ld not divisible by %ld\n", nb, row * elem); fclose(f); return false; }
    long nsteps = nb / (row * elem);
    std::vector<uint16_t> ids((size_t) nsteps * row);
    if (elem == 1) {
        std::vector<uint8_t> raw((size_t) nsteps * row);
        if (fread(raw.data(), 1, raw.size(), f) != raw.size()) { perror("read"); fclose(f); return false; }
        for (size_t i = 0; i < raw.size(); i++) ids[i] = raw[i];
    } else {
        if (fread(ids.data(), 2, ids.size(), f) != ids.size()) { perror("read"); fclose(f); return false; }
    }
    fclose(f);

    std::vector<uint64_t> freq((size_t) o.layers * o.experts, 0);
    for (long t = 0; t < nsteps; t++) {
        for (int l = 0; l < o.layers; l++) {
            for (int k = 0; k < o.used; k++) {
                const int e = ids[((size_t) t * o.layers + l) * o.used + k];
                if (e >= 0 && e < o.experts) freq[(size_t) l * o.experts + e]++;
            }
        }
    }
    const int top = std::min(o.top, o.experts);
    std::vector<uint16_t> out((size_t) o.layers * top);
    uint64_t covered = 0, total = 0;
    for (int l = 0; l < o.layers; l++) {
        std::vector<int> order(o.experts);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return freq[(size_t) l * o.experts + a] > freq[(size_t) l * o.experts + b];
        });
        for (int k = 0; k < top; k++) {
            out[(size_t) l * top + k] = (uint16_t) order[k];
            covered += freq[(size_t) l * o.experts + order[k]];
        }
        total += std::accumulate(freq.begin() + (size_t) l * o.experts,
                                 freq.begin() + (size_t) (l + 1) * o.experts, (uint64_t) 0);
    }
    FILE * o_ = fopen(o.out.c_str(), "wb");
    if (!o_) { perror("open out"); return false; }
    const uint32_t nl = (uint32_t) o.layers, nt = (uint32_t) top;
    fwrite(&nl, 4, 1, o_);
    fwrite(&nt, 4, 1, o_);
    fwrite(out.data(), 2, out.size(), o_);
    fclose(o_);
    printf("profile: %d layers x top-%d from %ld steps -> %s (coverage %.1f%%)\n",
           o.layers, top, nsteps, o.out.c_str(), total ? 100.0 * covered / total : 0.0);
    return true;
}

bool check(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { perror("open"); return false; }
    uint32_t nl = 0, nt = 0;
    if (fread(&nl, 4, 1, f) != 1 || fread(&nt, 4, 1, f) != 1) { fprintf(stderr, "short header\n"); fclose(f); return false; }
    if (nl == 0 || nl > 1024 || nt == 0 || nt > 4096) { fprintf(stderr, "implausible header %u x %u\n", nl, nt); fclose(f); return false; }
    std::vector<uint16_t> ids((size_t) nl * nt);
    const bool ok = fread(ids.data(), 2, ids.size(), f) == ids.size();
    // strict: file must end exactly here (catches truncation AND trailing garbage)
    bool exact = false;
    if (ok) { int c = fgetc(f); exact = (c == EOF); }
    fclose(f);
    if (!ok || !exact) { fprintf(stderr, "size mismatch for %u x %u\n", nl, nt); return false; }
    printf("profile ok: %u layers x top-%u (%zu bytes payload)\n", nl, nt, ids.size() * 2);
    printf("layer0 head: %u %u %u %u %u %u %u %u\n",
           ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7]);
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    opts o;
    if (!parse(argc, argv, o)) { usage(); return 1; }
    if (o.check) return check(o.out) ? 0 : 1;
    return build(o) ? 0 : 1;
}
