#include "llama-moe-file-source.h"

#include "ggml.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

llm_moe_file_source & moe_file_source() {
    static llm_moe_file_source inst;
    return inst;
}

llm_moe_file_source::~llm_moe_file_source() {
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

int llm_moe_file_source::fd() {
    std::lock_guard<std::mutex> g(mtx);
    return fd_;
}

bool llm_moe_file_source::is_open() {
    std::lock_guard<std::mutex> g(mtx);
    return fd_ >= 0;
}

// Minimal GGUF reader. Layout (little-endian):
//   magic   uint32  "GGUF"
//   version uint32  (we accept v2/v3)
//   n_tensors uint64
//   n_kv      uint64
//   kv entries (key=string, type=uint32, value=typed)
//   tensor entries: name=string, n_dims=uint32, dims[n_dims]=uint64,
//                   type=uint32, offset=uint64
//   alignment-padded
//   data: tensor blobs
//
// We only need tensor name, type, dims (for nbytes), and offset
// relative to data section start.

namespace {

constexpr uint32_t GGUF_MAGIC = 0x46554747;  // 'GGUF' little-endian

// GGUF metadata value types.
enum gguf_kv_type : uint32_t {
    KV_UINT8   = 0,
    KV_INT8    = 1,
    KV_UINT16  = 2,
    KV_INT16   = 3,
    KV_UINT32  = 4,
    KV_INT32   = 5,
    KV_FLOAT32 = 6,
    KV_BOOL    = 7,
    KV_STRING  = 8,
    KV_ARRAY   = 9,
    KV_UINT64  = 10,
    KV_INT64   = 11,
    KV_FLOAT64 = 12,
};

size_t kv_scalar_size(uint32_t t) {
    switch (t) {
        case KV_UINT8:
        case KV_INT8:
        case KV_BOOL:
            return 1;
        case KV_UINT16:
        case KV_INT16:
            return 2;
        case KV_UINT32:
        case KV_INT32:
        case KV_FLOAT32:
            return 4;
        case KV_UINT64:
        case KV_INT64:
        case KV_FLOAT64:
            return 8;
        default:
            return 0;
    }
}

bool read_exact(int fd, void * buf, size_t n) {
    uint8_t * p   = (uint8_t *) buf;
    size_t    got = 0;
    while (got < n) {
        ssize_t r = read(fd, p + got, n - got);
        if (r <= 0) {
            return false;
        }
        got += (size_t) r;
    }
    return true;
}

bool read_u32(int fd, uint32_t & v) {
    return read_exact(fd, &v, 4);
}

bool read_u64(int fd, uint64_t & v) {
    return read_exact(fd, &v, 8);
}

bool read_string(int fd, std::string & out) {
    uint64_t n;
    if (!read_u64(fd, n)) {
        return false;
    }
    out.resize((size_t) n);
    if (n == 0) {
        return true;
    }
    return read_exact(fd, &out[0], (size_t) n);
}

bool skip_kv_value(int fd, uint32_t type);

bool skip_kv_value(int fd, uint32_t type) {
    if (type == KV_STRING) {
        std::string tmp;
        return read_string(fd, tmp);
    }
    if (type == KV_ARRAY) {
        uint32_t arr_type;
        uint64_t arr_n;
        if (!read_u32(fd, arr_type) || !read_u64(fd, arr_n)) {
            return false;
        }
        if (arr_type == KV_STRING) {
            for (uint64_t i = 0; i < arr_n; ++i) {
                std::string tmp;
                if (!read_string(fd, tmp)) {
                    return false;
                }
            }
            return true;
        }
        if (arr_type == KV_ARRAY) {
            // nested arrays not supported in GGUF
            return false;
        }
        const size_t sz = kv_scalar_size(arr_type);
        if (sz == 0) {
            return false;
        }
        return lseek(fd, (off_t) (arr_n * sz), SEEK_CUR) >= 0;
    }
    const size_t sz = kv_scalar_size(type);
    if (sz == 0) {
        return false;
    }
    return lseek(fd, (off_t) sz, SEEK_CUR) >= 0;
}

// Compute on-disk byte size of a quantized/dense tensor given its
// ggml type and dims. ggml_row_size handles all known types.
size_t tensor_nbytes(uint32_t ggml_type_, const std::vector<uint64_t> & dims) {
    if (dims.empty()) {
        return 0;
    }
    const ggml_type type     = (ggml_type) ggml_type_;
    const size_t    row_size = ggml_row_size(type, (int64_t) dims[0]);
    size_t          rows     = 1;
    for (size_t i = 1; i < dims.size(); ++i) {
        rows *= (size_t) dims[i];
    }
    return row_size * rows;
}

}  // namespace

bool llm_moe_file_source::open(const std::string & path) {
    std::lock_guard<std::mutex> g(mtx);
    if (fd_ >= 0 && path_ == path) {
        return true;
    }
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
        tensors.clear();
    }
    return parse_locked(path);
}

bool llm_moe_file_source::parse_locked(const std::string & path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        fprintf(stderr, "moe_file_source: open failed: %s (%s)\n", path.c_str(), strerror(errno));
        return false;
    }

    uint32_t magic;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;

    if (!read_u32(fd_, magic) || magic != GGUF_MAGIC) {
        fprintf(stderr, "moe_file_source: bad magic\n");
        close(fd_);
        fd_ = -1;
        return false;
    }
    if (!read_u32(fd_, version) || (version < 2 || version > 3)) {
        fprintf(stderr, "moe_file_source: unsupported GGUF version %u\n", version);
        close(fd_);
        fd_ = -1;
        return false;
    }
    if (!read_u64(fd_, n_tensors) || !read_u64(fd_, n_kv)) {
        close(fd_);
        fd_ = -1;
        return false;
    }

    // Find alignment from KV (general.alignment), default 32.
    uint32_t alignment = 32;

    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        uint32_t    type;
        if (!read_string(fd_, key) || !read_u32(fd_, type)) {
            close(fd_);
            fd_ = -1;
            return false;
        }
        if (key == "general.alignment" && type == KV_UINT32) {
            if (!read_u32(fd_, alignment)) {
                close(fd_);
                fd_ = -1;
                return false;
            }
        } else {
            if (!skip_kv_value(fd_, type)) {
                fprintf(stderr, "moe_file_source: failed to skip kv '%s' type=%u\n", key.c_str(), type);
                close(fd_);
                fd_ = -1;
                return false;
            }
        }
    }

    // Read tensor info table.
    struct tinfo {
        std::string           name;
        uint32_t              ggml_type;
        std::vector<uint64_t> dims;
        uint64_t              offset;
    };
    std::vector<tinfo> infos;
    infos.reserve((size_t) n_tensors);

    for (uint64_t i = 0; i < n_tensors; ++i) {
        tinfo ti;
        uint32_t n_dims;
        if (!read_string(fd_, ti.name) || !read_u32(fd_, n_dims)) {
            close(fd_);
            fd_ = -1;
            return false;
        }
        ti.dims.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            if (!read_u64(fd_, ti.dims[d])) {
                close(fd_);
                fd_ = -1;
                return false;
            }
        }
        if (!read_u32(fd_, ti.ggml_type) || !read_u64(fd_, ti.offset)) {
            close(fd_);
            fd_ = -1;
            return false;
        }
        infos.push_back(std::move(ti));
    }

    // Compute data section start: aligned current position.
    off_t cur = lseek(fd_, 0, SEEK_CUR);
    if (cur < 0) {
        close(fd_);
        fd_ = -1;
        return false;
    }
    const uint64_t data_start = ((uint64_t) cur + alignment - 1) & ~((uint64_t) alignment - 1);

    // Build map.
    for (auto & ti : infos) {
        tensor_info entry;
        entry.file_offset = data_start + ti.offset;
        entry.nbytes      = tensor_nbytes(ti.ggml_type, ti.dims);
        tensors.emplace(std::move(ti.name), entry);
    }

    path_ = path;
    fprintf(stderr, "moe_file_source: opened %s tensors=%zu data_start=%llu alignment=%u\n", path.c_str(),
            tensors.size(), (unsigned long long) data_start, alignment);

    return true;
}

const llm_moe_file_source::tensor_info * llm_moe_file_source::lookup(const std::string & name) {
    std::lock_guard<std::mutex> g(mtx);
    if (fd_ < 0) {
        const char * env = std::getenv("LLAMA_MOE_GGUF_PATH");
        if (!env) {
            return nullptr;
        }
        if (!parse_locked(env)) {
            return nullptr;
        }
    }
    auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : &it->second;
}

bool llm_moe_file_source::pread_into(void * dst, uint64_t file_offset, size_t nbytes) {
    std::lock_guard<std::mutex> g(mtx);
    if (fd_ < 0) {
        return false;
    }
    uint8_t * p   = (uint8_t *) dst;
    size_t    got = 0;
    while (got < nbytes) {
        ssize_t r = pread(fd_, p + got, nbytes - got, (off_t) (file_offset + got));
        if (r < 0 && errno == EINTR) {
            continue;
        }
        if (r <= 0) {
            return false;
        }
        got += (size_t) r;
    }
    return true;
}