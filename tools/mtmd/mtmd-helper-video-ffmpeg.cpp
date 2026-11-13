// mtmd_helper_video_decoder backed by ffmpeg/ffprobe subprocesses

#ifdef MTMD_VIDEO

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "mtmd-helper.h"
#include "mtmd-helper-common.h"

#include "sheredom/subprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <csignal>
#include <fcntl.h>
#include <pthread.h>
#endif

#ifdef MTMD_INTERNAL_HEADER
#error "mtmd-helper is a public library outside of mtmd. it must not include internal headers"
#endif

namespace {

// RAII wrapper for managing subprocess
struct subprocess_handle {
    struct subprocess_s proc = {};
    bool created = false; // process exists and must be cleaned up
    bool alive   = false; // process can still give us data
    std::thread feeder;

    subprocess_handle() = default;
    subprocess_handle(const subprocess_handle &) = delete;
    subprocess_handle & operator=(const subprocess_handle &) = delete;
    ~subprocess_handle() { stop(); }

    void stop() {
        // note: alive becomes false on stdout EOF, but the process still needs cleanup
        if (!created) {
            return;
        }
        subprocess_terminate(&proc);
#ifdef _WIN32
        // no SIGPIPE on windows: a blocked feeder only gets a broken pipe once we close our read end of the child stdin
        if (proc.hStdInput) {
            CloseHandle(proc.hStdInput);
            proc.hStdInput = nullptr;
        }
#endif
        // join before destroy: feeder holds a FILE* from subprocess_stdin;
        // subprocess_destroy closes it, so the thread must finish first
        if (feeder.joinable()) {
            feeder.join();
        }
        subprocess_join(&proc, nullptr); // reap the child, or else it stays a zombie
        subprocess_destroy(&proc);
        created = false;
        alive   = false;
    }

    FILE * stdout_pipe() {
        return subprocess_stdout(&proc);
    }

    // buf must outlive the feeder thread, i.e. this handle
    void start_feeder(const std::vector<uint8_t> & buf) {
        feeder = std::thread([this, &buf]() {
#ifndef _WIN32
            // ffmpeg can exit before it reads all the input, for example when ffprobe already got the metadata.
            // the write below must then fail with EPIPE, instead of killing the process with SIGPIPE
            sigset_t sigpipe_set;
            sigemptyset(&sigpipe_set);
            sigaddset(&sigpipe_set, SIGPIPE);
            pthread_sigmask(SIG_BLOCK, &sigpipe_set, nullptr); // linux sends the signal to the writing thread
#endif
            FILE * f = subprocess_stdin(&proc);
            if (!f) {
                return;
            }
#ifdef F_SETNOSIGPIPE
            fcntl(fileno(f), F_SETNOSIGPIPE, 1); // macos/bsd send it to the process, so turn it off per fd
#endif
            fwrite(buf.data(), 1, buf.size(), f);
            fclose(f);
            proc.stdin_file = nullptr; // prevent double-close in subprocess_destroy
        });
    }
};

std::string resolve_bin(const char * bin_dir, const char * name) {
    if (!bin_dir || bin_dir[0] == '\0') {
        return name; // rely on PATH
    }
    std::string result = bin_dir;
    char last = result.back();
    if (last != '/' && last != '\\') {
#ifdef _WIN32
        result += '\\';
#else
        result += '/';
#endif
    }
    result += name;
#ifdef _WIN32
    result += ".exe";
#endif
    return result;
}

float parse_rational(const char * s) {
    int num = 0, den = 1;
    if (sscanf(s, "%d/%d", &num, &den) == 2 && den > 0) {
        return (float)num / (float)den;
    }
    float val;
    if (sscanf(s, "%f", &val) == 1) {
        return val;
    }
    return 0.0f;
}

struct ffmpeg_video {
    std::string path;
    std::vector<uint8_t> input_buf; // non-empty when opened from a buffer
    std::string ffmpeg_bin;
    std::string ffprobe_bin;

    uint32_t src_width     = 0;
    uint32_t src_height    = 0;
    double   video_start_s = 0.0;

    std::vector<float>   pcm;
    size_t               pcm_start = 0;
    std::vector<uint8_t> skip_buf;

    // must be destroyed before input_buf, which its feeder thread reads
    subprocess_handle sp;

    bool is_buf_input() const {
        return !input_buf.empty();
    }

    bool probe(mtmd_helper_video_stream_info & info) {
        // cache: makes stdin seekable; without it ffprobe reports width/height 0
        // for mp4 files whose moov atom sits at the end
        const char * input_arg = is_buf_input() ? "cache:pipe:0" : path.c_str();
        std::vector<const char *> cmd = {
            ffprobe_bin.c_str(),
            "-v", "quiet",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration,start_time",
            "-select_streams", "v:0",
            "-of", "default=noprint_wrappers=1",
        };
        if (is_buf_input()) {
            // remove the 64KB read-ahead limit of cache:, or else ffprobe cannot reach a moov atom at end of file
            cmd.insert(cmd.end(), {"-read_ahead_limit", "-1"});
        }
        cmd.insert(cmd.end(), {input_arg, nullptr});

        LOG_DBG("%s: launching:", __func__);
        for (size_t i = 0; cmd[i]; i++) { LOG_DBG(" %s", cmd[i]); }
        LOG_DBG("\n");

        subprocess_handle probe_sp;
        if (subprocess_create(cmd.data(),
                subprocess_option_search_user_path | subprocess_option_inherit_environment,
                &probe_sp.proc) != 0) {
            LOG_ERR("%s: failed to launch ffprobe\n", __func__);
            return false;
        }
        probe_sp.created = true;
        probe_sp.alive   = true;

        if (is_buf_input()) {
            probe_sp.start_feeder(input_buf);
        }

        char line[256];
        FILE * fp = probe_sp.stdout_pipe();

        while (fgets(line, sizeof(line), fp)) {
            char * eq = strchr(line, '=');
            if (!eq) continue;
            *eq = '\0';
            const char * key = line;
            const char * val = eq + 1;
            char * nl = (char *)strchr(val, '\n');
            if (nl) *nl = '\0';

            if (strcmp(key, "width") == 0) {
                info.width = (uint32_t)atoi(val);
            } else if (strcmp(key, "height") == 0) {
                info.height = (uint32_t)atoi(val);
            } else if (strcmp(key, "r_frame_rate") == 0) {
                info.fps = parse_rational(val);
            } else if (strcmp(key, "nb_frames") == 0 && strcmp(val, "N/A") != 0) {
                info.n_frames = atoi(val);
            } else if (strcmp(key, "duration") == 0 && strcmp(val, "N/A") != 0) {
                info.duration = (float)atof(val);
            } else if (strcmp(key, "start_time") == 0 && strcmp(val, "N/A") != 0) {
                video_start_s = atof(val);
            }
        }

        probe_sp.stop();

        src_width  = info.width;
        src_height = info.height;
        return info.width > 0 && info.height > 0 && info.fps > 0.0f;
    }

    void load_soundtrack(int rate) {
        const std::string rate_str = std::to_string(rate);
        std::vector<const char *> cmd = {ffmpeg_bin.c_str(), "-nostdin"};
        cmd.reserve(20); // GCC 11 -Wstringop-overflow false positive on the inserts below otherwise
        if (is_buf_input()) {
            cmd.insert(cmd.end(), {"-read_ahead_limit", "-1"});
        }
        cmd.insert(cmd.end(), {"-i", is_buf_input() ? "cache:pipe:0" : path.c_str(), "-map", "0:a:0", "-vn", "-ac", "1",
                               "-ar", rate_str.c_str(), "-f", "f32le", "pipe:1", "-loglevel", "error", nullptr});
        subprocess_handle audio_sp;
        if (subprocess_create(cmd.data(), subprocess_option_search_user_path | subprocess_option_inherit_environment,
                              &audio_sp.proc) != 0) {
            LOG_WRN("%s: failed to launch ffmpeg for the soundtrack\n", __func__);
            return;
        }
        audio_sp.created = true;
        audio_sp.alive   = true;
        if (is_buf_input()) {
            audio_sp.start_feeder(input_buf);
        }
        float block[4096];
        size_t n;
        while ((n = fread(block, sizeof(float), 4096, audio_sp.stdout_pipe())) > 0) {
            pcm.insert(pcm.end(), block, block + n);
        }
        audio_sp.stop();

        // align to the first video frame
        pcm_start = std::min(pcm.size(), (size_t) std::max(0.0, std::nearbyint(video_start_s * rate)));
    }

    bool start(uint32_t width, uint32_t height, float fps) {
        std::vector<const char *> cmd;
        cmd.push_back(ffmpeg_bin.c_str());

        cmd.push_back("-nostdin");
        if (is_buf_input()) {
            // remove the 64KB read-ahead limit of cache:, or else ffmpeg cannot reach a moov atom at end of file
            cmd.push_back("-read_ahead_limit");
            cmd.push_back("-1");
        }
        cmd.push_back("-i");
        // cache:pipe:0 wraps stdin with a seekable in-memory cache, letting ffmpeg seek
        // backwards for container headers (e.g. MP4 moov atom at end of file)
        cmd.push_back(is_buf_input() ? "cache:pipe:0" : path.c_str());

        const bool scale = width != src_width || height != src_height;
        char vf[160] = "";
        if (fps > 0.0f) {
            if (scale) {
                snprintf(vf, sizeof(vf), "fps=%.6f,scale=%u:%u", fps, width, height);
            } else {
                snprintf(vf, sizeof(vf), "fps=%.6f", fps);
            }
        } else if (scale) {
            // B=0 C=0.5 bicubic on RGB is closest to Pillow
            snprintf(vf, sizeof(vf),
                    "format=rgb24,scale=%u:%u:flags=bicubic+accurate_rnd+full_chroma_int:param0=0:param1=0.5",
                    width, height);
        }
        if (vf[0]) {
            cmd.push_back("-vf");
            cmd.push_back(vf);
        }

        cmd.push_back("-f");
        cmd.push_back("rawvideo");
        cmd.push_back("-pix_fmt");
        cmd.push_back("rgb24");
        cmd.push_back("pipe:1");
        cmd.push_back("-loglevel");
        cmd.push_back("error");
        cmd.push_back(nullptr);

        LOG_DBG("%s: launching:", __func__);
        for (size_t i = 0; cmd[i]; i++) {
            LOG_DBG(" %s", cmd[i]);
        }
        LOG_DBG("\n");

        int ret = subprocess_create(
            cmd.data(),
            subprocess_option_search_user_path | subprocess_option_inherit_environment,
            &sp.proc);

        sp.created = (ret == 0);
        sp.alive   = (ret == 0);
        LOG_DBG("%s: subprocess_create ret=%d proc_alive=%d\n", __func__, ret, (int)sp.alive);
        if (!sp.alive) {
            LOG_ERR("%s: failed to start ffmpeg (is ffmpeg in PATH?)\n", __func__);
            return false;
        }

        if (is_buf_input()) {
            LOG_DBG("%s: starting feeder thread for %zu-byte buffer\n", __func__, input_buf.size());
            sp.start_feeder(input_buf);
        }
        return true;
    }

    bool read(unsigned char * dst, size_t n) {
        FILE * fp = sp.stdout_pipe();
        if (!dst) {
            skip_buf.resize(n);
            dst = skip_buf.data();
        }
        size_t total_read = 0;
        while (total_read < n) {
            const size_t got = fread(dst + total_read, 1, n - total_read, fp);
            if (got == 0) {
                // clean EOF only if no bytes read yet; partial frame is an error
                LOG_DBG("%s: fread returned 0 after %zu/%zu bytes (ferror=%d)\n",
                        __func__, total_read, n, ferror(fp));
                return false;
            }
            total_read += got;
        }
        return true;
    }
};

void * ffmpeg_open(void * user_data, const char * path, const unsigned char * buf, size_t len,
                   mtmd_helper_video_stream_info * info) {
    auto v = std::make_unique<ffmpeg_video>();
    const char * bin_dir = static_cast<const char *>(user_data);
    v->ffmpeg_bin  = resolve_bin(bin_dir, "ffmpeg");
    v->ffprobe_bin = resolve_bin(bin_dir, "ffprobe");
    if (buf && len > 0) {
        v->input_buf.assign(buf, buf + len);
    } else if (path) {
        v->path = path;
    }
    if (!v->probe(*info)) {
        LOG_ERR("%s: ffprobe failed for '%s' (is ffprobe in PATH?)\n", __func__, v->is_buf_input() ? "<buffer>" : v->path.c_str());
        return nullptr;
    }
    return v.release();
}

bool ffmpeg_start(void * state, uint32_t width, uint32_t height, float fps, bool * fps_picked) {
    *fps_picked = fps > 0.0f;
    return static_cast<ffmpeg_video *>(state)->start(width, height, fps);
}

bool ffmpeg_read_frame(void * state, unsigned char * rgb, uint32_t width, uint32_t height) {
    auto * v = static_cast<ffmpeg_video *>(state);
    if (!v->sp.alive) {
        return false;
    }
    if (!v->pcm.empty()) {
        std::vector<float>().swap(v->pcm); // no longer needed
    }
    LOG_DBG("%s: reading a frame, expecting %zu bytes (%ux%u)\n", __func__, (size_t) width * height * 3, width, height);
    return v->read(rgb, (size_t) width * height * 3);
}

const float * ffmpeg_read_audio(void * state, int sample_rate, size_t * n_samples) {
    auto * v = static_cast<ffmpeg_video *>(state);
    v->pcm.clear();
    v->pcm_start = 0;
    v->load_soundtrack(sample_rate);
    *n_samples = v->pcm.size() - v->pcm_start;
    return *n_samples > 0 ? v->pcm.data() + v->pcm_start : nullptr;
}

void ffmpeg_close(void * state) {
    delete static_cast<ffmpeg_video *>(state);
}

} // namespace

const mtmd_helper_video_decoder * mtmd_helper_video_decoder_ffmpeg() {
    static const mtmd_helper_video_decoder decoder = {
        /* open       */ ffmpeg_open,
        /* read_frame */ ffmpeg_read_frame,
        /* read_audio */ ffmpeg_read_audio,
        /* close      */ ffmpeg_close,
        /* user_data  */ nullptr,
        /* start      */ ffmpeg_start,
    };
    return &decoder;
}

#endif // MTMD_VIDEO
