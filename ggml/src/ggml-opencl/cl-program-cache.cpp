// Match the version setup ggml-opencl.cpp uses, so any cl.h declarations we
// touch are consistent across this backend's translation units.
#define CL_TARGET_OPENCL_VERSION GGML_OPENCL_TARGET_VERSION
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "cl-program-cache.h"

#include "ggml-impl.h"  // GGML_LOG_INFO / WARN

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <process.h>
#  define ggml_getpid()    ((int) GetCurrentProcessId())
#else
#  include <unistd.h>
#  define ggml_getpid()    ((int) getpid())
#endif

namespace fs = std::filesystem;

// ----------------------------------------------------------------------------
// SHA-256 (FIPS 180-4). Self-contained, ~80 lines, public-domain reference.
// Hot path is a few KB of source per kernel ⇒ <1 ms total per process init.
// ----------------------------------------------------------------------------

namespace {

struct sha256_ctx {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t  buf[64];
    size_t   buf_len;
};

const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};

inline uint32_t rotr32(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }

void sha256_compress(uint32_t state[8], const uint8_t block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = ((uint32_t)block[i*4    ] << 24) |
               ((uint32_t)block[i*4 + 1] << 16) |
               ((uint32_t)block[i*4 + 2] <<  8) |
               ((uint32_t)block[i*4 + 3]      );
    }
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2],  19) ^ (w[i-2]  >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint32_t a = state[0],b = state[1],c = state[2],d = state[3],e = state[4],f = state[5],g = state[6],h = state[7];

    for (int i = 0; i < 64; ++i) {
        uint32_t S1   = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch   = (e & f) ^ ((~e) & g);
        uint32_t t1   = h + S1 + ch + K256[i] + w[i];
        uint32_t S0   = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj  = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2   = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

void sha256_init(sha256_ctx & c) {
    c.state[0]=0x6a09e667; c.state[1]=0xbb67ae85; c.state[2]=0x3c6ef372; c.state[3]=0xa54ff53a;
    c.state[4]=0x510e527f; c.state[5]=0x9b05688c; c.state[6]=0x1f83d9ab; c.state[7]=0x5be0cd19;
    c.bitlen  = 0;
    c.buf_len = 0;
}

void sha256_update(sha256_ctx & c, const void * data, size_t len) {
    const uint8_t * p = (const uint8_t *) data;
    c.bitlen += (uint64_t) len * 8;
    if (c.buf_len > 0) {
        size_t n = 64 - c.buf_len;
        if (n > len) { n = len; }
        memcpy(c.buf + c.buf_len, p, n);
        c.buf_len += n;
        p   += n;
        len -= n;
        if (c.buf_len == 64) {
            sha256_compress(c.state, c.buf);
            c.buf_len = 0;
        }
    }
    while (len >= 64) {
        sha256_compress(c.state, p);
        p   += 64;
        len -= 64;
    }
    if (len > 0) {
        memcpy(c.buf, p, len);
        c.buf_len = len;
    }
}

void sha256_final(sha256_ctx & c, uint8_t out[32]) {
    uint64_t bitlen = c.bitlen;
    c.buf[c.buf_len++] = 0x80;
    if (c.buf_len > 56) {
        while (c.buf_len < 64) { c.buf[c.buf_len++] = 0; }
        sha256_compress(c.state, c.buf);
        c.buf_len = 0;
    }
    while (c.buf_len < 56) { c.buf[c.buf_len++] = 0; }
    for (int i = 7; i >= 0; --i) { c.buf[c.buf_len++] = (uint8_t) (bitlen >> (i * 8)); }
    sha256_compress(c.state, c.buf);
    for (int i = 0; i < 8; ++i) {
        out[i*4    ] = (uint8_t) (c.state[i] >> 24);
        out[i*4 + 1] = (uint8_t) (c.state[i] >> 16);
        out[i*4 + 2] = (uint8_t) (c.state[i] >>  8);
        out[i*4 + 3] = (uint8_t) (c.state[i]      );
    }
}

std::string sha256_hex(const uint8_t digest[32]) {
    static const char hex[] = "0123456789abcdef";
    std::string s(64, '0');
    for (int i = 0; i < 32; ++i) {
        s[i*2    ] = hex[digest[i] >> 4];
        s[i*2 + 1] = hex[digest[i] & 0xf];
    }
    return s;
}

std::string compute_key(const std::string & key_suffix,
                        const char *        source,
                        const std::string & compile_opts) {
    sha256_ctx c;
    sha256_init(c);

    static const uint8_t sep = 0;
    sha256_update(c, source,             strlen(source));
    sha256_update(c, &sep,               1);
    sha256_update(c, compile_opts.data(), compile_opts.size());
    sha256_update(c, &sep,               1);
    sha256_update(c, key_suffix.data(),  key_suffix.size());

    uint8_t digest[32];
    sha256_final(c, digest);
    return sha256_hex(digest);
}

bool make_dir_recursive(const std::string & path) {
    if (path.empty()) { return false; }
    // create_directories() already creates missing parents. It returns false
    // (with ec clear) when the directory is already there, so re-check.
    const fs::path p = fs::u8path(path);
    std::error_code ec;
    if (fs::create_directories(p, ec)) { return true; }
    std::error_code ec_stat;
    return fs::is_directory(p, ec_stat);
}

std::string default_cache_dir() {
#if defined(_WIN32)
    const char * base = std::getenv("LOCALAPPDATA");
    if (!base || !*base) { base = std::getenv("APPDATA"); }
    if (!base || !*base) { base = std::getenv("TEMP"); }
    if (!base || !*base) { base = "."; }
    return std::string(base) + "\\llama.cpp\\cl-cache";
#elif defined(__APPLE__)
    const char * home = std::getenv("HOME");
    if (!home || !*home) { home = "."; }
    return std::string(home) + "/Library/Caches/llama.cpp/cl-cache";
#else
    // The throwing overload aborts the process when no usable temp directory
    // exists (e.g. Android app contexts with TMPDIR unset); an empty return
    // here just disables the cache instead.
    std::error_code ec;
    const fs::path tmp_path = fs::temp_directory_path(ec);
    if (ec || tmp_path.empty()) { return {}; }
    return tmp_path.string() + "/llama.cpp/cl-cache";
#endif
}

// Query a NUL-terminated string from clGetDeviceInfo / clGetPlatformInfo.
template <typename GetInfoFn, typename Object>
std::string query_string(GetInfoFn fn, Object obj, cl_uint name) {
    size_t sz = 0;
    if (fn(obj, name, 0, nullptr, &sz) != CL_SUCCESS || sz == 0) {
        return {};
    }
    std::string s(sz, '\0');
    if (fn(obj, name, sz, &s[0], nullptr) != CL_SUCCESS) {
        return {};
    }
    if (!s.empty() && s.back() == '\0') {
        s.pop_back();
    }
    return s;
}

std::string compute_key_suffix(cl_device_id device) {
    cl_platform_id platform = nullptr;
    clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr);

    std::string s;
    s.reserve(512);
    s += query_string(clGetDeviceInfo,   device,   CL_DEVICE_NAME);       s.push_back('\0');
    s += query_string(clGetDeviceInfo,   device,   CL_DRIVER_VERSION);    s.push_back('\0');
    s += query_string(clGetDeviceInfo,   device,   CL_DEVICE_VERSION);    s.push_back('\0');
    if (platform) {
        s += query_string(clGetPlatformInfo, platform, CL_PLATFORM_VERSION); s.push_back('\0');
    }
    s += "fmt=" + std::to_string(CL_PROGRAM_CACHE_FORMAT_VERSION);
    return s;
}

const uint8_t MAGIC[8] = { 'G','G','M','L','C','L','B','C' };

bool read_all(const std::string & path, std::vector<uint8_t> & out) {
    std::ifstream f(fs::u8path(path), std::ios::binary);
    if (!f) { return false; }
    f.seekg(0, std::ios::end);
    std::streamsize sz = f.tellg();
    if (sz < 0) { return false; }
    f.seekg(0, std::ios::beg);
    out.resize((size_t) sz);
    if (sz > 0) { f.read((char *) out.data(), sz); }
    return f.good() || f.eof();
}

bool write_atomic(const std::string & path, const uint8_t * data, size_t len) {
    const fs::path dst = fs::u8path(path);
    const fs::path tmp = fs::u8path(path + ".tmp." + std::to_string(ggml_getpid()));
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) { return false; }
        f.write((const char *) data, (std::streamsize) len);
        if (!f.good()) {
            std::error_code ec_rm;
            fs::remove(tmp, ec_rm);
            return false;
        }
    }

    std::error_code ec;
    fs::rename(tmp, dst, ec);
    if (ec) {
        std::error_code ec_rm;
        fs::remove(tmp, ec_rm);
        return false;
    }
    return true;
}

}  // namespace

static bool cache_debug_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * e = std::getenv("GGML_OPENCL_KERNEL_CACHE_DEBUG");
        cached = (e && *e) ? 1 : 0;
    }
    return cached != 0;
}

static std::string opts_preview(const std::string & opts, size_t n = 120) {
    if (opts.size() <= n) { return opts; }
    return opts.substr(0, n) + "...";
}

// Running cache tally (diagnostic; plain ints — a benign race in the rare
// multi-threaded lazy-compile case at worst miscounts by one).
static int g_cache_hits = 0, g_cache_misses = 0, g_cache_saves = 0;

// Debug trace directly to stderr
static void cache_debug_line(const char * kind, const std::string & key,
                             const char * source, const std::string & opts) {
    if (!cache_debug_enabled()) { return; }
    fprintf(stderr, "ggml_opencl: cache %-4s [h=%d m=%d s=%d] key=%s src=%zuB opts='%s'\n",
            kind, g_cache_hits, g_cache_misses, g_cache_saves,
            key.substr(0, 16).c_str(), strlen(source), opts_preview(opts).c_str());
    fflush(stderr);
}

cl_program_cache_state cl_program_cache_init(cl_device_id device) {
    cl_program_cache_state st;

    const char * env = std::getenv("GGML_OPENCL_KERNEL_CACHE_DIR");
    if (env && (!std::strcmp(env, "0")    || !std::strcmp(env, "off")  ||
                !std::strcmp(env, "none") || !std::strcmp(env, "disable") ||
                !std::strcmp(env, "disabled"))) {
        if (cache_debug_enabled()) {
            fprintf(stderr, "ggml_opencl: kernel cache disabled by GGML_OPENCL_KERNEL_CACHE_DIR=%s\n", env);
            fflush(stderr);
        }
        return st;
    }

    std::string dir;
    if (!env || !*env || !std::strcmp(env, "1") || !std::strcmp(env, "default")) {
        dir = default_cache_dir();
        if (dir.empty()) {
            GGML_LOG_INFO("ggml_opencl: kernel cache disabled (no usable default cache directory)\n");
            return st;
        }
    } else {
        dir = env;
    }

    if (!make_dir_recursive(dir)) {
        GGML_LOG_INFO("ggml_opencl: kernel cache disabled (cannot create directory '%s')\n", dir.c_str());
        return st;
    }

    st.dir        = dir;
    st.key_suffix = compute_key_suffix(device);
    GGML_LOG_INFO("ggml_opencl: kernel cache enabled at '%s'\n", st.dir.c_str());
    if (cache_debug_enabled()) {
        fprintf(stderr, "ggml_opencl: kernel cache enabled at '%s' "
                        "(GGML_OPENCL_KERNEL_CACHE_DIR=off to disable)\n", st.dir.c_str());
        fflush(stderr);
    }
    return st;
}

cl_program cl_program_cache_try_load(
    const cl_program_cache_state & state,
    cl_context                     context,
    cl_device_id                   device,
    const char *                   source,
    const std::string &            compile_opts) {

    if (state.dir.empty() || !source) { return nullptr; }

    const std::string key  = compute_key(state.key_suffix, source, compile_opts);
    const std::string path = state.dir + "/" + key + ".clbin";

    std::vector<uint8_t> file;
    if (!read_all(path, file)) {
        ++g_cache_misses;
        cache_debug_line("MISS", key, source, compile_opts);
        return nullptr;
    }
    if (file.size() < 16 || std::memcmp(file.data(), MAGIC, 8) != 0) { return nullptr; }

    uint32_t fmt =
        ((uint32_t) file[ 8]) | ((uint32_t) file[ 9] <<  8) |
        ((uint32_t) file[10] << 16) | ((uint32_t) file[11] << 24);
    if (fmt != CL_PROGRAM_CACHE_FORMAT_VERSION) { return nullptr; }

    const size_t hdr_len = 16;
    const unsigned char * bin     = file.data() + hdr_len;
    const size_t          bin_len = file.size() - hdr_len;

    cl_int err     = CL_SUCCESS;
    cl_int bin_err = CL_SUCCESS;
    cl_program p = clCreateProgramWithBinary(context, 1, &device, &bin_len, &bin, &bin_err, &err);
    if (err != CL_SUCCESS || bin_err != CL_SUCCESS || p == nullptr) {
        if (p) { clReleaseProgram(p); }
        return nullptr;
    }

    err = clBuildProgram(p, 0, nullptr, compile_opts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseProgram(p);
        return nullptr;
    }
    ++g_cache_hits;
    cache_debug_line("HIT", key, source, compile_opts);
    return p;
}

void cl_program_cache_try_save(
    const cl_program_cache_state & state,
    cl_program                     program,
    cl_device_id                   /*device*/,
    const char *                   source,
    const std::string &            compile_opts) {

    if (state.dir.empty() || !program || !source) {
        return;
    }

    cl_uint n_dev = 0;
    if (clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(n_dev), &n_dev, nullptr) != CL_SUCCESS || n_dev == 0) {
        return;
    }

    std::vector<size_t> sizes(n_dev);
    if (clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t) * n_dev, sizes.data(), nullptr) != CL_SUCCESS) {
        return;
    }
    if (sizes.empty() || sizes[0] == 0) {
        return;
    }

    std::vector<std::vector<uint8_t>> binaries(n_dev);
    std::vector<unsigned char *>      bin_ptrs(n_dev);
    for (cl_uint i = 0; i < n_dev; ++i) {
        binaries[i].resize(sizes[i]);
        bin_ptrs[i] = binaries[i].data();
    }
    if (clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char *) * n_dev, bin_ptrs.data(), nullptr) != CL_SUCCESS) {
        return;
    }

    // We only care about the first device's binary — that's the one we'd
    // re-load with on a future cache hit. Multi-device contexts aren't a
    // pattern this backend uses today.
    const std::vector<uint8_t> & bin = binaries[0];

    std::vector<uint8_t> file;
    file.reserve(16 + bin.size());
    file.insert(file.end(), MAGIC, MAGIC + 8);
    uint32_t fmt = CL_PROGRAM_CACHE_FORMAT_VERSION;
    file.push_back((uint8_t)  (fmt        & 0xff));
    file.push_back((uint8_t) ((fmt >>  8) & 0xff));
    file.push_back((uint8_t) ((fmt >> 16) & 0xff));
    file.push_back((uint8_t) ((fmt >> 24) & 0xff));
    file.push_back(0); file.push_back(0); file.push_back(0); file.push_back(0); // reserved
    file.insert(file.end(), bin.begin(), bin.end());

    const std::string key  = compute_key(state.key_suffix, source, compile_opts);
    const std::string path = state.dir + "/" + key + ".clbin";
    if (!write_atomic(path, file.data(), file.size())) {
        GGML_LOG_INFO("ggml_opencl: kernel cache: failed to write '%s'\n", path.c_str());
    } else {
        ++g_cache_saves;
        cache_debug_line("SAVE", key, source, compile_opts);
    }
}
