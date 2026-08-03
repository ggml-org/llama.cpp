//
// tessera-matmul-output.cpp
//
// Tessera Layer-2 matmul-output sidecar writer. See tessera-matmul-output.h.
//
// This is a thin mirror of the L1 dequant sidecar (tessera-debug.cpp)
// but captures the F32 matmul OUTPUT instead of the dequantized weight.
// The on-disk layout uses a distinct magic ("TPMO") so the two sidecar
// kinds can coexist in one directory and be routed by their magic.
//
// Audit-mandated safety properties (see agent-c-l2-forward audit):
//   1. Default capture stride is 32 (was 1), reducing the worst-case
//      per-forward-pass disk write by 32x.
//   2. A hard cap of MATMUL_OUTPUT_MAX_ROWS (4096) rows per file is
//      enforced in the CPU hook; tensors that would exceed it are
//      silently dropped with a one-time-per-tensor warning.
//   3. The CPU hook filters tensors through matmul_output_tensor_allowed();
//      only the linear-layer matmuls of the per-block transformer and
//      the model head are captured.
//   4. The second-and-onward open for the same tensor name appends to
//      the existing file rather than truncating; only the first open
//      truncates. (This preserves multi-invocation data: prefill +
//      decode of the same tensor land in one file.)
//   5. The writer tracks next_row and skips the redundant
//      seek_to_data_start() call when the caller writes rows in
//      sequential order (the typical pattern), reducing syscalls from
//      O(R^2) to O(R) per file.
//

#include "tessera-matmul-output.h"

// Pull the L1 sidecar v3 header constants so the v3 shape stays in
// lockstep between the two file kinds. The L1 sidecar owns the schema
// versioning (DEQUANT_FILE_VERSION) and the per-row v3 strip shape.
#include "tessera-debug.h"

// Intentionally NOT including common/log.h: this TU is compiled into
// the llama-tessera-debug static library, which sits below common/ in
// the layering hierarchy. Diagnostic messages go straight to stderr.

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

namespace tessera_matmul_output {

// Recursive so the public open/write/close entry points can each acquire
// the lock without deadlocking when one sequence nests another (it does
// not today, but the L1 sidecar uses the same pattern).
static std::recursive_mutex g_writer_mutex;

namespace {

// One-time env-var snapshot. The output dir and stride can also be set
// programmatically via set_*; the env vars are only read on the first
// call. Default stride is MATMUL_OUTPUT_DEFAULT_STRIDE (32), the
// audit-mandated safe default.
struct EnvState {
    std::string dir;
    int64_t     stride = MATMUL_OUTPUT_DEFAULT_STRIDE;
    bool        initialized = false;
};

EnvState & env_state() {
    static EnvState s;
    return s;
}

void ensure_env_loaded() {
    EnvState & s = env_state();
    if (s.initialized) {
        return;
    }
    s.initialized = true;
    if (const char * v = std::getenv("LLAMA_TILE640_DEBUG_MATMUL_OUTPUT_DIR"); v != nullptr) {
        s.dir = v;
    }
    if (const char * v = std::getenv("LLAMA_TILE640_DEBUG_MATMUL_OUTPUT_STRIDE"); v != nullptr) {
        int64_t st = (int64_t) atoll(v);
        s.stride = (st < 1) ? 1 : st;
    }
}

// Per-row v3 metadata. Same shape as the L1 sidecar (8 + 4 + 4 + 8 = 24
// bytes per row) so the v3 reader can parse both kinds uniformly. The
// outlier-count strip (int32 per row) is reused as a "samples seen"
// count so a downstream L2 reader can tell whether the file was
// truncated mid-row.
struct RowV3Meta {
    uint64_t timing_ns      = 0;
    uint32_t kernel_id      = 0;
    uint32_t dispatch_count = 0;
    uint64_t reserved       = 0;
};

// State for a single open sidecar file. The post-audit fields:
//   next_row           : the row index the writer expects next;
//                        write_matmul_output_row() skips the
//                        seek_to_data_start() call when the caller
//                        hands us rows in sequential order
//                        (audit fix 5).
//   file_data_offset   : the absolute byte position of the start of
//                        the data block; cached at open time so the
//                        sequential-write path does not have to
//                        recompute it on every row.
//   last_open_appended : true iff the most recent open of this file
//                        appended to an existing on-disk file (audit
//                        fix 4). First open for a tensor name always
//                        truncates.
//   row_offset         : number of rows already on disk from prior
//                        invocations of the same tensor. The CPU
//                        hook passes per-invocation row_idx in
//                        [0, rows-1]; the writer translates this to
//                        row_offset + row_idx for the on-disk
//                        position. row_samples / total_samples are
//                        indexed by the per-invocation row_idx
//                        (so they record "how many invocations
//                        wrote each per-invocation row") - the v2
//                        strip is sealed at close using these
//                        per-invocation counters, not the absolute
//                        position.
struct SidecarStream {
    std::ofstream ofs;
    std::string   tensor_name;
    int64_t       rows       = 0;     // captured rows (= logical rows / stride)
    int64_t       cols       = 0;     // elements per row (= n_embd for attn, etc.)
    int64_t       next_row   = 0;     // next absolute row expected (sequential optimization)
    int64_t       file_data_offset = 0;  // absolute byte offset of the data block
    int64_t       row_offset = 0;        // number of rows already on disk
    std::vector<RowV3Meta>    row_meta;
    std::vector<int32_t>      row_samples;   // per-row samples seen
    int64_t                   total_samples  = 0;
    bool                      last_open_appended = false;
};

static SidecarStream g_stream;

// One-time-per-tensor warning dedup set. A tensor name that was
// rejected (allowlist or row cap) is recorded here so the warning is
// logged at most once per process. The set is bounded by the number
// of distinct matmul tensors in the graph (a few hundred for a 12B
// model) so the memory cost is negligible.
static std::unordered_set<std::string> g_warned;

void warn_once(const std::string & key, const std::string & message) {
    if (g_warned.insert(key).second) {
        std::fprintf(stderr, "%s", message.c_str());
    }
}

// --- Tensor-name allowlist (audit fix 3) -------------------------------
//
// Only the linear layer matmuls of the per-block transformer (and the
// model head) are L2-relevant. Token embedding, RoPE rotation matrices,
// normalization gamma vectors, attention bias vectors, etc. are
// silently skipped. The matcher is a hand-written scanner rather than
// <regex> to keep this TU header-only and avoid pulling in libstdc++
// regex (a 1 MB static cost).
//
// Accepted forms:
//   blk.<digits>.attn_(q|k|v|output).weight
//   blk.<digits>.ffn_(gate|up|down).weight
//   output.weight
//
// Anything else returns false.
bool tensor_name_passes_allowlist(const char * name) {
    if (name == nullptr || name[0] == '\0') {
        return false;
    }
    // output.weight
    if (std::strcmp(name, "output.weight") == 0) {
        return true;
    }
    // Must start with "blk.".
    static constexpr const char * PREFIX = "blk.";
    const size_t PREFIX_LEN = 4;
    if (std::strncmp(name, PREFIX, PREFIX_LEN) != 0) {
        return false;
    }
    const char * p = name + PREFIX_LEN;
    // Skip digits (the layer number).
    if (!std::isdigit((unsigned char) *p)) {
        return false;
    }
    while (std::isdigit((unsigned char) *p)) {
        p++;
    }
    // Expect ".attn_X.weight" or ".ffn_X.weight" where X is one of
    // the allowed short names.
    if (*p != '.') {
        return false;
    }
    p++;
    if (std::strncmp(p, "attn_", 5) == 0) {
        p += 5;
        if (std::strcmp(p, "q.weight") == 0) return true;
        if (std::strcmp(p, "k.weight") == 0) return true;
        if (std::strcmp(p, "v.weight") == 0) return true;
        if (std::strcmp(p, "output.weight") == 0) return true;
        return false;
    }
    if (std::strncmp(p, "ffn_", 4) == 0) {
        p += 4;
        if (std::strcmp(p, "gate.weight") == 0) return true;
        if (std::strcmp(p, "up.weight") == 0) return true;
        if (std::strcmp(p, "down.weight") == 0) return true;
        return false;
    }
    return false;
}

// --- File I/O helpers (audit fix 4 + 5) --------------------------------

void write_v3_header(SidecarStream & s) {
    // v1 header (28 bytes): magic(4) version(4) rows(8) cols(8) dtype(4)
    s.ofs.write(MATMUL_OUTPUT_FILE_MAGIC, 4);
    uint32_t version = tessera_debug::DEQUANT_FILE_VERSION;   // 3, shared with L1
    s.ofs.write((const char *) &version, sizeof(version));
    s.ofs.write((const char *) &s.rows, sizeof(s.rows));
    s.ofs.write((const char *) &s.cols, sizeof(s.cols));
    uint32_t dtype = tessera_debug::DEQUANT_DTYPE_F32;
    s.ofs.write((const char *) &dtype, sizeof(dtype));

    // v2 header (12 bytes): outlier_threshold(4) outlier_count_total(8)
    // The L2 matmul-output sidecar reuses the v2 outlier-count strip as
    // a per-row sample counter (see SidecarStream::row_samples). The
    // threshold field is set to 0.0 (no outlier count semantics here)
    // and the total is repurposed as the total-sample count so a v3
    // reader can still validate the file.
    float outlier_threshold = 0.0f;
    s.ofs.write((const char *) &outlier_threshold, sizeof(outlier_threshold));
    s.ofs.write((const char *) &s.total_samples, sizeof(s.total_samples));

    // v2 per-row strip (R * 4 bytes) - the per-row sample counter
    std::vector<int32_t> zeros((size_t) s.rows, 0);
    s.ofs.write((const char *) zeros.data(), sizeof(int32_t) * (size_t) s.rows);

    // v3 per-row strip (R * 24 bytes) - reserved for now, finalized at close
    uint8_t row_meta_zero[24];
    std::memset(row_meta_zero, 0, sizeof(row_meta_zero));
    for (int64_t r = 0; r < s.rows; r++) {
        s.ofs.write((const char *) row_meta_zero, sizeof(row_meta_zero));
    }
    s.ofs.flush();
}

int64_t compute_data_offset(int64_t rows) {
    // data starts after v1 header (28) + v2 header (12) + v2 per-row
    // strip (R*4) + v3 per-row strip (R*24) = 40 + R*4 + R*24 = 40 + R*28
    return 40 + rows * 28;
}

void seek_to_v2_strip_start(SidecarStream & s) {
    if (!s.ofs.is_open()) {
        return;
    }
    // v2 per-row strip starts at offset 40 (v1 header = 28, v2 header = 12).
    s.ofs.seekp((std::streamoff) 40, std::ios::beg);
}

void seek_to_v3_strip_start(SidecarStream & s) {
    if (!s.ofs.is_open()) {
        return;
    }
    // v3 per-row strip starts at offset 40 + R*4 (after the v2 per-row strip).
    s.ofs.seekp((std::streamoff)(40 + s.rows * 4), std::ios::beg);
}

} // namespace

bool matmul_output_capture_enabled() {
    std::lock_guard<std::recursive_mutex> lock(g_writer_mutex);
    ensure_env_loaded();
    return !env_state().dir.empty() || !g_stream.tensor_name.empty();
}

bool matmul_output_tensor_allowed(const char * tensor_name) {
    return tensor_name_passes_allowlist(tensor_name);
}

bool matmul_output_last_open_appended() {
    std::lock_guard<std::recursive_mutex> lock(g_writer_mutex);
    return g_stream.last_open_appended;
}

int64_t matmul_output_stream_rows() {
    std::lock_guard<std::recursive_mutex> lock(g_writer_mutex);
    if (!g_stream.ofs.is_open()) {
        return -1;
    }
    return g_stream.rows;
}

int64_t matmul_output_stream_cols() {
    std::lock_guard<std::recursive_mutex> lock(g_writer_mutex);
    if (!g_stream.ofs.is_open()) {
        return -1;
    }
    return g_stream.cols;
}

void set_matmul_output_dir(const std::string & path) {
    std::lock_guard<std::recursive_mutex> lock(g_writer_mutex);
    ensure_env_loaded();
    if (path.empty()) {
        // closing existing writer
        if (g_stream.ofs.is_open()) {
            try { g_stream.ofs.flush(); } catch (...) {}
            g_stream.ofs.close();
        }
        g_stream = SidecarStream();
        env_state().dir.clear();
        // Also clear the process env var so any other static instance of
        // the sidecar state (e.g. one statically linked into libggml-cpu.dylib
        // as well as the one in libllama-common.dylib) sees the empty
        // state on its next ensure_env_loaded() call.
        ::setenv("LLAMA_TILE640_DEBUG_MATMUL_OUTPUT_DIR", "", 1);
        return;
    }
    env_state().dir = path;
    // Also write the process env var so any other static instance of
    // the sidecar state (e.g. one statically linked into
    // libggml-cpu.dylib) sees the same value via ensure_env_loaded().
    // This is a no-op when this is the only instance in the process.
    ::setenv("LLAMA_TILE640_DEBUG_MATMUL_OUTPUT_DIR", path.c_str(), 1);
    // We do NOT pre-create the directory here; it is created on the
    // first open_matmul_output_writer call (mirroring the L1 sidecar
    // contract). This avoids surprising the user with an empty dir.
}

const std::string & matmul_output_dir() {
    std::lock_guard<std::recursive_mutex> lock(g_writer_mutex);
    ensure_env_loaded();
    return env_state().dir;
}

void set_matmul_output_stride(int64_t stride) {
    std::lock_guard<std::recursive_mutex> lock(g_writer_mutex);
    ensure_env_loaded();
    env_state().stride = (stride < 1) ? 1 : stride;
}

int64_t matmul_output_stride() {
    std::lock_guard<std::recursive_mutex> lock(g_writer_mutex);
    ensure_env_loaded();
    return env_state().stride;
}

void open_matmul_output_writer(const char * tensor_name,
                               int64_t rows, int64_t cols) {
    if (tensor_name == nullptr || tensor_name[0] == '\0' || rows <= 0 || cols <= 0) {
        return;
    }
    std::lock_guard<std::recursive_mutex> lock(g_writer_mutex);
    ensure_env_loaded();
    if (env_state().dir.empty()) {
        return;
    }
    // Audit fix 2: hard cap on rows. A tensor whose captured row count
    // exceeds the cap is silently rejected (with a one-time-per-tensor
    // warning). This bounds the worst-case file size to
    // MATMUL_OUTPUT_MAX_ROWS * max(cols) * 4 bytes, which keeps the
    // total sidecar footprint well within the disk budget even for
    // pathological 32k-context forward passes on 12B models.
    if (rows > MATMUL_OUTPUT_MAX_ROWS) {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
                      "tessera-matmul-output: rejecting '%s': captured rows %lld > max %lld; "
                      "increase --tessera-matmul-output-stride to fit within the cap\n",
                      tensor_name,
                      (long long) rows, (long long) MATMUL_OUTPUT_MAX_ROWS);
        warn_once(std::string("rows:") + tensor_name, std::string(buf));
        return;
    }
    if (g_stream.tensor_name == tensor_name && g_stream.ofs.is_open()) {
        // Same tensor re-opened. The audit fix 4 design:
        //   * same shape -> append (preserves prefill + multi-decode
        //     invocations and multi-prompt imatrix chunks)
        //   * different shape -> SILENTLY DROP. Reasoning: the L2
        //     forward-pass differential is dominated by the prefill
        //     matmul (one big call with rows=n_tokens). The decode
        //     phase is a sequence of small calls (rows=1) that would
        //     otherwise TRUNCATE the prefill data and lose the bulk
        //     of the L2 signal. We keep the prefill and skip the
        //     decode calls. The skip is silent and one-time-per-shape
        //     so the user is not spammed with warnings.
        //
        // We do NOT reset g_stream on the drop: the close() at the
        // end of the rejected decode call must still seal the v2/v3
        // strips for the prefill data. Resets happen only on a
        // proper close.
        if (g_stream.rows != rows || g_stream.cols != cols) {
            char buf[512];
            std::snprintf(buf, sizeof(buf),
                          "tessera-matmul-output: skipping '%s' reopen with shape (%lld, %lld) "
                          "(keeping the first call's shape (%lld, %lld)); "
                          "the L2 differential needs the prefill data, not per-decode-token data\n",
                          tensor_name,
                          (long long) rows, (long long) cols,
                          (long long) g_stream.rows, (long long) g_stream.cols);
            warn_once(std::string("shape:") + tensor_name, std::string(buf));
            return;
        }
        // Same shape, same tensor - the new rows append. We do
        // NOT reset next_row, row_samples, or total_samples: the
        // per-row sample counter records that this row was
        // written multiple times (one per invocation), and
        // total_samples is the cumulative count.
        return;   // already open with the same shape
    } else if (g_stream.ofs.is_open()) {
        // Different tensor, close the previous one.
        g_stream.ofs.flush();
        g_stream.ofs.close();
        g_stream = SidecarStream();
    }

    std::filesystem::path dir_path(env_state().dir);
    std::error_code ec;
    std::filesystem::create_directories(dir_path, ec);
    if (ec) {
        std::fprintf(stderr, "tessera-matmul-output: failed to create dir '%s': %s\n",
                     env_state().dir.c_str(), ec.message().c_str());
        return;
    }

    g_stream.tensor_name = tensor_name;
    g_stream.rows        = rows;
    g_stream.cols        = cols;
    g_stream.row_meta.assign((size_t) rows, RowV3Meta());
    g_stream.row_samples.assign((size_t) rows, 0);
    g_stream.total_samples = 0;
    g_stream.next_row      = 0;
    g_stream.row_offset    = 0;
    g_stream.file_data_offset = compute_data_offset(rows);

    const std::string file_path = (dir_path / (std::string(tensor_name) + MATMUL_OUTPUT_FILE_SUFFIX)).string();
    // Check if a file already exists from a prior invocation. If so,
    // we append (audit fix 4). If not, we create + truncate.
    //
    // Mode selection: we need seekable read+write (NOT std::ios::app,
    // which forces every write to EOF regardless of seekp - that
    // would corrupt the v2/v3 strips and the v2 header when the
    // close function seeks back to write them). For an existing
    // file we open in std::ios::binary | std::ios::in | std::ios::out
    // (which requires the file to exist). For a new file we open
    // with std::ios::trunc to create + truncate, then reopen in
    // the in|out mode for the data writes.
    bool file_existed = std::filesystem::exists(file_path);
    if (file_existed) {
        g_stream.ofs.open(file_path, std::ios::binary | std::ios::in | std::ios::out);
        g_stream.last_open_appended = true;
    } else {
        // First call: create + truncate, write the v3 header, then
        // we are done. Subsequent writes (via write_matmul_output_row)
        // will land at the position after the header.
        g_stream.ofs.open(file_path, std::ios::binary | std::ios::out | std::ios::trunc);
        g_stream.last_open_appended = false;
    }
    if (!g_stream.ofs.is_open()) {
        std::fprintf(stderr, "tessera-matmul-output: failed to open '%s'\n", file_path.c_str());
        g_stream = SidecarStream();
        return;
    }
    if (g_stream.last_open_appended) {
        // Append path (audit fix 4): the file already has a v3
        // header and possibly prior data. We READ BACK the existing
        // per-row sample counts from the sealed v2 strip and the v2
        // header's total_samples, so the close() at the end of THIS
        // invocation writes the CUMULATIVE count, not just the count
        // from this invocation. Without this, a 3-invocation
        // sequence would leave the strip with [2, 2] (last invocation
        // only) instead of the correct [3, 3] (cumulative).
        //
        // We also compute the existing data byte count from the file
        // size and use it to set g_stream.next_row = (existing data
        // bytes / cols * 4). This makes the sequential-write
        // optimization work correctly across appends: the 2nd
        // invocation's row 0 lands AFTER the 1st invocation's data
        // rather than OVERWRITING it. Without this, every append
        // would clobber the previous data because both invocations
        // start writing at row_idx=0.
        int64_t existing_data_bytes = 0;
        {
            std::ifstream rdr(file_path, std::ios::binary);
            if (rdr) {
                rdr.seekg((std::streamoff) 32, std::ios::beg);
                rdr.read((char *) &g_stream.total_samples, sizeof(g_stream.total_samples));
                if (g_stream.total_samples < 0) g_stream.total_samples = 0;
                rdr.seekg((std::streamoff) 40, std::ios::beg);
                for (int64_t r = 0; r < g_stream.rows; r++) {
                    int32_t cnt = 0;
                    rdr.read((char *) &cnt, sizeof(cnt));
                    g_stream.row_samples[(size_t) r] = cnt;
                }
                rdr.seekg(0, std::ios::end);
                std::streamoff end = rdr.tellg();
                rdr.seekg(0, std::ios::beg);
                std::streamoff begin = rdr.tellg();
                const std::streamoff total = end - begin;
                if (total > 0) {
                    existing_data_bytes = (int64_t) total - 40;
                }
            }
        }
        // Position the writer at the END of the file so the new
        // data rows are appended after the existing data block.
        g_stream.ofs.seekp(0, std::ios::end);
        // Audit fix 5 (cross-invocation): set next_row and
        // row_offset so the sequential-write path picks up where
        // the previous invocation left off. The previous
        // invocation wrote (existing_data_bytes / (cols * 4))
        // rows; the next sequential write should land at that
        // absolute row index. The CPU hook still passes
        // per-invocation row_idx in [0, rows-1]; the writer
        // translates via g_stream.row_offset.
        if (existing_data_bytes > 0) {
            const int64_t bytes_per_row = cols * 4;
            g_stream.row_offset = existing_data_bytes / bytes_per_row;
            g_stream.next_row   = g_stream.row_offset;
        }
        g_stream.file_data_offset = compute_data_offset(rows);
    } else {
        // Fresh file: write the v3 header (header + per-row zero
        // strips), then position the writer at the start of the
        // data block so the first sequential write can go straight
        // in without an extra seek.
        write_v3_header(g_stream);
        g_stream.ofs.seekp((std::streamoff) g_stream.file_data_offset, std::ios::beg);
    }
}

void write_matmul_output_row(int64_t row_idx, const float * data, int64_t n) {
    if (data == nullptr || n <= 0) {
        return;
    }
    std::lock_guard<std::recursive_mutex> lock(g_writer_mutex);
    if (!g_stream.ofs.is_open() || row_idx < 0 || row_idx >= g_stream.rows) {
        if (g_stream.tensor_name.empty()) {
            return;   // sidecar disabled; no warning
        }
        std::fprintf(stderr, "tessera-matmul-output: row %lld out of range for '%s' (rows=%lld)\n",
                     (long long) row_idx, g_stream.tensor_name.c_str(),
                     (long long) g_stream.rows);
        return;
    }
    const int64_t effective_n = (n < g_stream.cols) ? n : g_stream.cols;
    if (n != g_stream.cols) {
        std::fprintf(stderr, "tessera-matmul-output: row %lld n=%lld != cols=%lld for '%s'; truncating\n",
                     (long long) row_idx, (long long) n, (long long) g_stream.cols,
                     g_stream.tensor_name.c_str());
    }
    // The CALLER's row_idx is in [0, rows-1] (the row within the
    // current matmul invocation). For an append, the on-disk
    // position is row_offset + row_idx, where row_offset is the
    // number of rows already on disk from prior invocations. The
    // g_stream.row_samples / total_samples vectors are also indexed
    // by the per-invocation row_idx, so they are NOT shifted; the
    // row_samples counter correctly records how many invocations
    // wrote each per-invocation row.
    const int64_t absolute_row = g_stream.row_offset + row_idx;
    // Audit fix 5: skip the redundant seek_to_data_start() when the
    // caller hands us rows in sequential order. The writer is
    // already positioned at the end of the previous row's data, so
    // the next write can go in directly. We only seek when the
    // caller jumps to a non-sequential row (e.g. out-of-order writes
    // from a parallel matmul chunk).
    if (absolute_row == g_stream.next_row) {
        // Sequential: no seek needed, the write pointer is already
        // at the right position (the end of the previous row's
        // data, or at the data block start if this is the first
        // row). The data is contiguous so a single write is enough.
        g_stream.ofs.write((const char *) data, sizeof(float) * (size_t) effective_n);
    } else {
        // Non-sequential: seek to the row's position in the data
        // block and write there. The seek is O(1) regardless of
        // how many rows have been written.
        const std::streamoff row_off = g_stream.file_data_offset +
            (std::streamoff) absolute_row * (std::streamoff) g_stream.cols * 4;
        g_stream.ofs.seekp(row_off, std::ios::beg);
        g_stream.ofs.write((const char *) data, sizeof(float) * (size_t) effective_n);
    }
    g_stream.row_samples[(size_t) row_idx] += 1;
    g_stream.total_samples += 1;
    g_stream.next_row = (absolute_row + 1 > g_stream.next_row) ? (absolute_row + 1) : g_stream.next_row;
}

void set_matmul_output_row_meta(int64_t row_idx,
                                uint64_t timing_ns,
                                uint32_t kernel_id,
                                uint32_t dispatch_count) {
    std::lock_guard<std::recursive_mutex> lock(g_writer_mutex);
    if (!g_stream.ofs.is_open() || row_idx < 0 || row_idx >= g_stream.rows) {
        return;
    }
    g_stream.row_meta[(size_t) row_idx].timing_ns      = timing_ns;
    g_stream.row_meta[(size_t) row_idx].kernel_id      = kernel_id;
    g_stream.row_meta[(size_t) row_idx].dispatch_count = dispatch_count;
}

void close_matmul_output_writer() {
    std::lock_guard<std::recursive_mutex> lock(g_writer_mutex);
    if (!g_stream.ofs.is_open()) {
        return;
    }

    // The v2/v3 per-row strips were written once at open time. We
    // re-seal them here to capture the per-row sample counts and
    // the per-row v3 metadata. The strips live at fixed offsets
    // (40 and 40 + R*4), and the data block starts at 40 + R*28,
    // so writing the strips cannot overwrite any data. We seek to
    // the exact v2 start, write the strip, then seek to the v3
    // start and write that. The v2 header total is also updated
    // to the cumulative total_samples.
    seek_to_v2_strip_start(g_stream);
    g_stream.ofs.write((const char *) g_stream.row_samples.data(),
                       sizeof(int32_t) * (size_t) g_stream.rows);
    seek_to_v3_strip_start(g_stream);
    g_stream.ofs.write((const char *) g_stream.row_meta.data(),
                       sizeof(RowV3Meta) * (size_t) g_stream.rows);

    // Update the v2 header total (repurposed as total_samples).
    const std::streamoff v2_off = 28 + 4;   // 32
    g_stream.ofs.seekp(v2_off, std::ios::beg);
    g_stream.ofs.write((const char *) &g_stream.total_samples, sizeof(g_stream.total_samples));

    g_stream.ofs.flush();
    g_stream.ofs.close();
    g_stream = SidecarStream();
}

} // namespace tessera_matmul_output
