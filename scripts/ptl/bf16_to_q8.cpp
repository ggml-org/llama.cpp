// Rewrite one GGUF split: selected BF16 tensors become Q8_0, everything else is copied byte for byte.
// Selected: blk.N.hc_{attn,ffn}_{down,up}.weight and blk.N.ffn_gate_inp.weight (2D, k multiple of 32).
// usage: bf16_to_q8 --dry-run|--stats|--write IN.gguf [OUT.gguf]
//   --dry-run: list what changes and the output size, reads no tensor data
//   --stats:   also convert in memory and print the round trip error per tensor group, writes nothing
//   --write:   write OUT.gguf, refuses to overwrite and checks free disk space first
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <regex>
#include <string>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <vector>

static bool selected(const std::string & name) {
    static const std::regex re(R"(^blk\.\d+\.(hc_(attn|ffn)_(down|up)|ffn_gate_inp)\.weight$)");
    return std::regex_match(name, re);
}

struct err_stat { double se = 0, sr = 0, max_abs = 0; size_t n = 0; };

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s --dry-run|--stats|--write IN.gguf [OUT.gguf]\n", argv[0]); return 1; }
    const std::string mode = argv[1];
    const bool write = mode == "--write", stats = mode == "--stats";
    if (!write && !stats && mode != "--dry-run") { fprintf(stderr, "unknown mode\n"); return 1; }
    if (write && argc < 4) { fprintf(stderr, "--write needs OUT.gguf\n"); return 1; }
    const char * in_path = argv[2];
    const char * out_path = write ? argv[3] : nullptr;

    ggml_context * meta = nullptr;
    gguf_init_params ip = { true, &meta };
    gguf_context * in = gguf_init_from_file(in_path, ip);
    if (!in) { fprintf(stderr, "cannot read %s\n", in_path); return 1; }
    const size_t in_data = gguf_get_data_offset(in);

    gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, in);
    const int64_t n = gguf_get_n_tensors(in);
    size_t n_conv = 0, bytes_before = 0, bytes_after = 0;
    std::vector<bool> conv(n, false);
    for (int64_t i = 0; i < n; i++) {
        ggml_tensor * t = ggml_get_tensor(meta, gguf_get_tensor_name(in, i));
        gguf_add_tensor(out, t);
    }
    for (int64_t i = 0; i < n; i++) {
        const char * name = gguf_get_tensor_name(in, i);
        ggml_tensor * t = ggml_get_tensor(meta, name);
        if (!selected(name)) continue;
        if (t->type != GGML_TYPE_BF16) { fprintf(stderr, "%s is %s, expected bf16\n", name, ggml_type_name(t->type)); return 1; }
        if (t->ne[0] % ggml_blck_size(GGML_TYPE_Q8_0) != 0 || t->ne[2] != 1) { fprintf(stderr, "%s: bad shape\n", name); return 1; }
        conv[i] = true;
        n_conv++;
        bytes_before += gguf_get_tensor_size(in, i);
        gguf_set_tensor_type(out, name, GGML_TYPE_Q8_0);
        bytes_after += ggml_row_size(GGML_TYPE_Q8_0, t->ne[0]) * t->ne[1];
    }
    size_t out_size = gguf_get_meta_size(out);
    for (int64_t i = 0; i < n; i++) out_size += GGML_PAD(gguf_get_tensor_size(out, i), gguf_get_alignment(out));
    printf("tensors %lld, converted %zu: %.3f GiB BF16 -> %.3f GiB Q8_0; output about %.2f GiB\n", (long long) n, n_conv,
           bytes_before / 1073741824.0, bytes_after / 1073741824.0, out_size / 1073741824.0);
    if (mode == "--dry-run") return 0;

    FILE * fi = fopen(in_path, "rb");
    if (!fi) { perror(in_path); return 1; }
    FILE * fo = nullptr;
    if (write) {
        struct stat st;
        if (stat(out_path, &st) == 0) { fprintf(stderr, "%s exists, not overwriting\n", out_path); return 1; }
        std::string dir(out_path);
        const size_t slash = dir.rfind('/');
        dir = slash == std::string::npos ? "." : dir.substr(0, slash);
        struct statvfs vfs;
        if (statvfs(dir.c_str(), &vfs) != 0) { perror(dir.c_str()); return 1; }
        const double free_b = (double) vfs.f_bavail * vfs.f_frsize;
        if (free_b < out_size + 2.0 * (1ull << 30)) { fprintf(stderr, "not enough disk: %.1f GiB free\n", free_b / 1073741824.0); return 1; }
        fo = fopen(out_path, "wb");
        if (!fo) { perror(out_path); return 1; }
        std::vector<uint8_t> m(gguf_get_meta_size(out));
        gguf_get_meta_data(out, m.data());
        fwrite(m.data(), 1, m.size(), fo);
    }

    std::map<std::string, err_stat> errs;
    const std::regex group(R"(blk\.\d+\.)");
    std::vector<uint8_t> src, dst, zeros(gguf_get_alignment(out), 0);
    std::vector<float> f32, back;
    for (int64_t i = 0; i < n; i++) {
        const char * name = gguf_get_tensor_name(in, i);
        ggml_tensor * t = ggml_get_tensor(meta, name);
        if (!write && !conv[i]) continue;
        const size_t in_size = gguf_get_tensor_size(in, i);
        src.resize(in_size);
        if (fseeko(fi, (off_t) (in_data + gguf_get_tensor_offset(in, i)), SEEK_SET) != 0 || fread(src.data(), 1, in_size, fi) != in_size) {
            fprintf(stderr, "read failed: %s\n", name);
            return 1;
        }
        const uint8_t * data = src.data();
        size_t size = in_size;
        if (conv[i]) {
            const int64_t nx = t->ne[0], ny = t->ne[1], ne = nx * ny;
            f32.resize(ne);
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) src.data(), f32.data(), ne);
            size = ggml_row_size(GGML_TYPE_Q8_0, nx) * ny;
            dst.resize(size);
            const size_t got = ggml_quantize_chunk(GGML_TYPE_Q8_0, f32.data(), dst.data(), 0, ny, nx, nullptr);
            if (got != size) { fprintf(stderr, "%s: size %zu != %zu\n", name, got, size); return 1; }
            data = dst.data();
            if (stats) {
                back.resize(ne);
                ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(dst.data(), back.data(), ne);
                err_stat & e = errs[std::regex_replace(std::string(name), group, "blk.N.")];
                for (int64_t j = 0; j < ne; j++) {
                    const double d = (double) back[j] - f32[j];
                    e.se += d * d;
                    e.sr += (double) f32[j] * f32[j];
                    e.max_abs = std::max(e.max_abs, std::fabs(d));
                }
                e.n += ne;
            }
        }
        if (write) {
            if (fwrite(data, 1, size, fo) != size) { fprintf(stderr, "write failed: %s\n", name); return 1; }
            const size_t pad = GGML_PAD(size, gguf_get_alignment(out)) - size;
            if (pad) fwrite(zeros.data(), 1, pad, fo);
        }
    }
    for (const auto & kv : errs) {
        const err_stat & e = kv.second;
        printf("%-30s elements %11zu  NMSE %.3e  max abs %.4f  rms %.4f\n", kv.first.c_str(), e.n, e.sr > 0 ? e.se / e.sr : 0.0, e.max_abs, std::sqrt(e.sr / e.n));
    }
    fclose(fi);
    if (fo && fclose(fo) != 0) { perror("close"); return 1; }
    if (write) printf("wrote %s\n", out_path);
    gguf_free(out);
    gguf_free(in);
    ggml_free(meta);
    return 0;
}
