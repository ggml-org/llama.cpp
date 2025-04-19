// merged_fuse.cpp
#define FUSE_USE_VERSION 31
#include <fuse.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <glob.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <vector>
#include <string>
#include <algorithm>

// Include the llama.cpp GGUF/ggml headers
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

// For path buffer sizes
#include <limits.h>

// Default alignment (from gguf code)
#define GGUF_DEFAULT_ALIGNMENT 32
// Use existing GGML_PAD from ggml.h

//--------------------------------------------------------------------
// Data structures to hold “merged file” layout information
//--------------------------------------------------------------------
enum SourceType { METADATA, SPLIT_FILE, PADDING };

struct Segment {
    off_t merged_offset; // starting offset in merged virtual file
    off_t length;        // segment length in bytes
    SourceType source;   // where the data comes from
    int split_index;     // valid if source==SPLIT_FILE
    off_t src_offset;    // offset in the split file (if SPLIT_FILE)
};

struct MergedFile {
    std::vector<Segment> segments;     // segments that, concatenated, form the merged file
    off_t size;                        // total merged file size
    std::vector<uint8_t> metadata;     // merged metadata bytes (from GGUF)
    std::vector<int> split_fds;        // open file descriptors for each split file
};

static MergedFile *g_merged_file = nullptr; // global merged file

//--------------------------------------------------------------------
// Helpers for naming split files
//--------------------------------------------------------------------
static bool fuse_split_path(char *path, size_t path_size, const char *prefix, int i_split, int n_split) {
    snprintf(path, path_size, "%s-%05d-of-%05d.gguf", prefix, i_split + 1, n_split);
    return access(path, F_OK) == 0;
}

static bool fuse_split_prefix(char *prefix, size_t prefix_size, const char *split_path, int /*i_split*/, int /*n_split*/) {
    // Handle HuggingFace-style split files
    strncpy(prefix, split_path, prefix_size);
    char *pos = strstr(prefix, "-of-");
    if (pos) {
        // Go back to find the start of the split number
        while (pos > prefix && *(pos-1) != '-') {
            pos--;
        }
        *pos = '\0';
        return true;
    }
    return false;
}

//--------------------------------------------------------------------
// Build merged file “map” from split files (in memory, no output file)
//--------------------------------------------------------------------
bool build_merged_file(const std::string &input_prefix) {
    // Find all split files using glob first
    char pattern[PATH_MAX] = {0};
    snprintf(pattern, sizeof(pattern), "%s*-of-*.gguf", input_prefix.c_str());
    fprintf(stderr, "Searching with glob pattern: %s\n", pattern);
    
    glob_t glob_result;
    int ret = glob(pattern, GLOB_NOSORT, NULL, &glob_result);
    if (ret != 0) {
        fprintf(stderr, "Failed to find split files using pattern: %s\n", pattern);
        return false;
    }
    
    int n_split = glob_result.gl_pathc;
    if (n_split < 1) {
        fprintf(stderr, "No split files found using pattern: %s\n", pattern);
        globfree(&glob_result);
        return false;
    }
    
    fprintf(stderr, "Detected %d splits\n", n_split);
    
    // Try to open first split file
    char split_path[PATH_MAX] = {0};
    fuse_split_path(split_path, sizeof(split_path), input_prefix.c_str(), 0, n_split);
    fprintf(stderr, "Trying to open: %s\n", split_path);
    
    int fd0 = open(split_path, O_RDONLY);
    if (fd0 < 0) {
        perror("open first split file");
        globfree(&glob_result);
        return false;
    }
    globfree(&glob_result);
    
    // Initialize GGUF context for first split (no allocation for meta data)
    struct ggml_context *ctx_meta = nullptr;
    gguf_init_params init_params = { .no_alloc = true, .ctx = &ctx_meta };
    gguf_context *ctx_gguf = gguf_init_from_file(split_path, init_params);
    if (!ctx_gguf) {
        fprintf(stderr, "Failed to load GGUF from %s\n", split_path);
        close(fd0);
        return false;
    }
    
    // We already detected the number of splits
    if (n_split < 1) {
        fprintf(stderr, "Invalid split count: %d\n", n_split);
        gguf_free(ctx_gguf);
        ggml_free(ctx_meta);
        close(fd0);
        return false;
    }
    
    // We now create a “merged” GGUF context.
    gguf_context *ctx_out = gguf_init_empty();
    // Copy metadata from the first split and update keys as in a merge:
    gguf_set_kv(ctx_out, ctx_gguf);
    
    // We will accumulate tensor information from all splits.
    struct TensorInfo {
        int split_index;
        off_t file_offset;
        size_t n_bytes;
        size_t padding;
    };
    std::vector<TensorInfo> tensors;
    int n_tensors_total = 0;
    
    // We will also open each split file and keep its fd.
    std::vector<int> fds;
    std::vector<gguf_context*> ctxs;
    std::vector<ggml_context*> metas;
    
    for (int i = 0; i < n_split; i++) {
        char spath[PATH_MAX] = {0};
        if (i == 0) {
            strncpy(spath, split_path, sizeof(spath));
        } else {
            llama_split_path(spath, sizeof(spath), input_prefix.c_str(), i, n_split);
        }
        int fd = open(spath, O_RDONLY);
        if (fd < 0) {
            perror("open split file");
            for (auto d : fds) close(d);
            gguf_free(ctx_out);
            gguf_free(ctx_gguf);
            ggml_free(ctx_meta);
            return false;
        }
        fds.push_back(fd);
        
        ggml_context *meta_ctx = nullptr;
        gguf_init_params ip = { .no_alloc = true, .ctx = &meta_ctx };
        gguf_context *ctx = gguf_init_from_file(spath, ip);
        if (!ctx) {
            fprintf(stderr, "Failed to load GGUF from %s\n", spath);
            for (auto d : fds) close(d);
            gguf_free(ctx_out);
            gguf_free(ctx_gguf);
            ggml_free(ctx_meta);
            return false;
        }
        ctxs.push_back(ctx);
        metas.push_back(meta_ctx);
        
        int n_tensors = gguf_get_n_tensors(ctx);
        n_tensors_total += n_tensors;
        for (int j = 0; j < n_tensors; j++) {
            const char *t_name = gguf_get_tensor_name(ctx, j);
            struct ggml_tensor *t = ggml_get_tensor(meta_ctx, t_name);
            size_t n_bytes = ggml_nbytes(t);
            size_t padded = GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT);
            size_t pad = padded - n_bytes;
            
            TensorInfo info;
            info.split_index = i;
            info.n_bytes = n_bytes;
            info.padding = pad;
            // Compute where tensor data is stored in the split file:
            off_t offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, j);
            info.file_offset = offset;
            tensors.push_back(info);
            
            // Add tensor to merged GGUF context
            gguf_add_tensor(ctx_out, t);
        }
    }
    
    // Compute merged metadata size and get the data into memory.
    size_t meta_size = gguf_get_meta_size(ctx_out);
    std::vector<uint8_t> merged_meta(meta_size);
    gguf_get_meta_data(ctx_out, merged_meta.data());
    
    // Now build our segments for the merged file.
    std::vector<Segment> segments;
    // First segment: the metadata (stored in memory).
    Segment seg;
    seg.merged_offset = 0;
    seg.length = meta_size;
    seg.source = METADATA;
    seg.split_index = -1;
    seg.src_offset = 0;
    segments.push_back(seg);
    
    off_t current_offset = meta_size;
    // For each tensor from each split (in order), record a data segment and (if needed) a padding segment.
    for (const auto &tinfo : tensors) {
        // Tensor data segment.
        Segment data_seg;
        data_seg.merged_offset = current_offset;
        data_seg.length = tinfo.n_bytes;
        data_seg.source = SPLIT_FILE;
        data_seg.split_index = tinfo.split_index;
        data_seg.src_offset = tinfo.file_offset;
        segments.push_back(data_seg);
        current_offset += tinfo.n_bytes;
        // Padding segment if needed.
        if (tinfo.padding > 0) {
            Segment pad_seg;
            pad_seg.merged_offset = current_offset;
            pad_seg.length = tinfo.padding;
            pad_seg.source = PADDING;
            pad_seg.split_index = -1;
            pad_seg.src_offset = 0;
            segments.push_back(pad_seg);
            current_offset += tinfo.padding;
        }
    }
    
    // Save all the information into our global structure.
    g_merged_file = new MergedFile();
    g_merged_file->segments = segments;
    g_merged_file->size = current_offset;
    g_merged_file->metadata = merged_meta;
    g_merged_file->split_fds = fds;
    
    // Cleanup the GGUF contexts (we keep the fds open)
    for (int i = 0; i < n_split; i++) {
        gguf_free(ctxs[i]);
        ggml_free(metas[i]);
    }
    gguf_free(ctx_gguf);
    ggml_free(ctx_meta);
    gguf_free(ctx_out);
    
    fprintf(stderr, "Merged file built: %ld bytes, %zu segments, %d split files, %d tensors\n",
            (long)g_merged_file->size, g_merged_file->segments.size(), n_split, n_tensors_total);
    return true;
}

//--------------------------------------------------------------------
// FUSE callbacks
//--------------------------------------------------------------------
static int merged_getattr(const char *path, struct stat *stbuf, struct fuse_file_info *fi) {
    (void) fi;
    memset(stbuf, 0, sizeof(struct stat));
    if (strcmp(path, "/") == 0) {
        stbuf->st_mode = S_IFDIR | 0755;
        stbuf->st_nlink = 2;
    } else if (strcmp(path, "/merged.gguf") == 0) {
        stbuf->st_mode = S_IFREG | 0444;
        stbuf->st_nlink = 1;
        stbuf->st_size = g_merged_file->size;
    } else {
        return -ENOENT;
    }
    return 0;
}

static int merged_readdir(const char *path, void *buf, fuse_fill_dir_t filler,
                            off_t offset, struct fuse_file_info *fi, enum fuse_readdir_flags flags) {
    (void) offset; (void) fi; (void) flags;
    if (strcmp(path, "/") != 0)
        return -ENOENT;
    filler(buf, ".", nullptr, 0, static_cast<fuse_fill_dir_flags>(0));
    filler(buf, "..", nullptr, 0, static_cast<fuse_fill_dir_flags>(0));
    filler(buf, "merged.gguf", nullptr, 0, static_cast<fuse_fill_dir_flags>(0));
    return 0;
}

static int merged_open(const char *path, struct fuse_file_info *fi) {
    if (strcmp(path, "/merged.gguf") != 0)
        return -ENOENT;
    return 0;
}

static int merged_read(const char *path, char *buf, size_t size, off_t offset, struct fuse_file_info *fi) {
    (void) fi;
    if (strcmp(path, "/merged.gguf") != 0)
        return -ENOENT;
    if (offset >= g_merged_file->size)
        return 0;
    if (static_cast<off_t>(offset + size) > g_merged_file->size)
        size = g_merged_file->size - offset;
    
    size_t bytes_read = 0;
    // Iterate over segments and copy the requested bytes.
    for (const auto &seg : g_merged_file->segments) {
        // If the segment ends before our offset, skip.
        if (seg.merged_offset + seg.length <= offset)
            continue;
        // If the segment starts after the requested region, we are done.
        if (seg.merged_offset >= static_cast<off_t>(offset + size))
            break;
        // Determine overlap.
        off_t seg_start = std::max(offset, seg.merged_offset);
        off_t seg_end = std::min(static_cast<off_t>(offset + size), seg.merged_offset + seg.length);
        size_t seg_offset = seg_start - seg.merged_offset;
        size_t to_copy = seg_end - seg_start;
        if (seg.source == METADATA) {
            memcpy(buf + bytes_read, g_merged_file->metadata.data() + seg_offset, to_copy);
        } else if (seg.source == SPLIT_FILE) {
            int fd = g_merged_file->split_fds[seg.split_index];
            ssize_t res = pread(fd, buf + bytes_read, to_copy, seg.src_offset + seg_offset);
            if (res < 0)
                return -errno;
        } else if (seg.source == PADDING) {
            memset(buf + bytes_read, 0, to_copy);
        }
        bytes_read += to_copy;
    }
    return bytes_read;
}

static const struct fuse_operations merged_oper = {
    merged_getattr,  /* getattr */
    NULL,           /* readlink */
    NULL,           /* mknod */
    NULL,           /* mkdir */
    NULL,           /* unlink */
    NULL,           /* rmdir */
    NULL,           /* symlink */
    NULL,           /* rename */
    NULL,           /* link */
    NULL,           /* chmod */
    NULL,           /* chown */
    NULL,           /* truncate */
    merged_open,    /* open */
    merged_read,    /* read */
    NULL,           /* write */
    NULL,           /* statfs */
    NULL,           /* flush */
    NULL,           /* release */
    NULL,           /* fsync */
    NULL,           /* setxattr */
    NULL,           /* getxattr */
    NULL,           /* listxattr */
    NULL,           /* removexattr */
    NULL,           /* opendir */
    merged_readdir, /* readdir */
    NULL,           /* releasedir */
    NULL,           /* fsyncdir */
    NULL,           /* init */
    NULL,           /* destroy */
    NULL,           /* access */
    NULL,           /* create */
    NULL,           /* lock */
    NULL,           /* utimens */
    NULL,           /* bmap */
    NULL,           /* ioctl */
    NULL,           /* poll */
    NULL,           /* write_buf */
    NULL,           /* read_buf */
    NULL,           /* flock */
    NULL,           /* fallocate */
    NULL,           /* copy_file_range */
    NULL            /* lseek */
};

//--------------------------------------------------------------------
// Main: parse command-line and launch FUSE
//--------------------------------------------------------------------
int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <split-file-prefix> <mountpoint>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    std::string input_prefix = argv[1];
    // Build the in-memory merged file representation
    if (!build_merged_file(input_prefix)) {
        fprintf(stderr, "Failed to build merged file\n");
        exit(EXIT_FAILURE);
    }
    // Adjust FUSE arguments (remove our extra argument)
    int fuse_argc = argc - 1;
    char **fuse_argv = &argv[1];
    return fuse_main(fuse_argc, fuse_argv, &merged_oper, nullptr);
}
