#include "mtmd-video.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <system_error>
#include <vector>

#ifdef _WIN32
#    include <process.h>  // for _getpid
#    define DEV_NULL "NUL"
#else
#    include <unistd.h>  // for getpid
#    define DEV_NULL "/dev/null"
#endif

static std::string quote(const std::string & s) {
#ifdef _WIN32
    // Simple conservative quoting for Windows
    std::string q = "\"";
    for (char c : s) {
        if (c == '"') {
            q += "\\\"";
        } else {
            q += c;
        }
    }
    q += "\"";
    return q;
#else
    std::string q = "'";
    for (char c : s) {
        if (c == '\'') {
            q += "'\\''";
        } else {
            q += c;
        }
    }
    q += "'";
    return q;
#endif
}

bool mtmd_video_ffmpeg_available(std::string & err) {
#ifdef _WIN32
    const std::string cmd = "ffmpeg -version > " DEV_NULL " 2>&1";
#else
    const std::string cmd = "command -v ffmpeg > " DEV_NULL " 2>&1";
#endif
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        err = "ffmpeg not found in PATH. Please install ffmpeg or add it to PATH.";
        return false;
    }
    return true;
}

static std::filesystem::path make_unique_temp_dir() {
    auto base = std::filesystem::temp_directory_path();
    // unique name: mtmd_video_<epoch_s>_<pid>
    auto now  = static_cast<long long>(std::time(nullptr));
#ifdef _WIN32
    int pid = _getpid();
#else
    int pid = getpid();
#endif
    std::ostringstream oss;
    oss << "mtmd_video_" << now << "_" << pid;
    return base / oss.str();
}

bool mtmd_video_extract_frames(const std::string &     video_path,
                               const mtmd_video_opts & opts,
                               mtmd_video_result &     out,
                               std::string &           err) {
    out = mtmd_video_result{};
    if (!mtmd_video_ffmpeg_available(err)) {
        return false;
    }

    std::error_code ec;
    auto            tmp = make_unique_temp_dir();
    if (!std::filesystem::create_directories(tmp, ec)) {
        err = "Failed to create temp dir: " + ec.message();
        return false;
    }

    const double fps      = (opts.fps > 0.f ? opts.fps : 1.0);
    out.seconds_per_frame = 1.0 / fps;

    // First, get video metadata to match vLLM's frame selection
    std::ostringstream probe_cmd;
    probe_cmd << "ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate,nb_frames -of csv=p=0 "
              << quote(video_path);

    FILE * probe_pipe = nullptr;
#ifdef _WIN32
    probe_pipe = _popen(probe_cmd.str().c_str(), "r");
#else
    probe_pipe = popen(probe_cmd.str().c_str(), "r");
#endif

    float original_fps  = 30.0f;  // default
    int   total_frames  = 180;    // default for 6 seconds at 30fps
    bool  probe_success = false;

    if (probe_pipe) {
        char buffer[256];
        if (fgets(buffer, sizeof(buffer), probe_pipe)) {
            // Parse "30/1,180" format (fps as fraction, total frames)
            int fps_num, fps_den;
            if (sscanf(buffer, "%d/%d,%d", &fps_num, &fps_den, &total_frames) == 3) {
                original_fps  = (float) fps_num / fps_den;
                probe_success = true;
            }
        }
#ifdef _WIN32
        _pclose(probe_pipe);
#else
        pclose(probe_pipe);
#endif
    }

    // Calculate frame indices using vLLM's algorithm
    const float      duration = total_frames / original_fps;
    const int        n        = opts.max_frames > 0 ? opts.max_frames : (int) std::floor(duration * fps);
    std::vector<int> frame_indices;

    for (int i = 0; i < n; ++i) {
        // Match vLLM's formula: ceil(i * original_fps / fps)
        int frame_idx = (int) std::ceil(i * original_fps / fps);
        frame_idx     = std::min(frame_idx, total_frames - 1);
        frame_indices.push_back(frame_idx);
    }

    // Build ffmpeg select filter to extract specific frames
    std::ostringstream select_expr;
    select_expr << "select='";
    for (size_t i = 0; i < frame_indices.size(); ++i) {
        if (i > 0) {
            select_expr << "+";
        }
        select_expr << "eq(n\\," << frame_indices[i] << ")";
    }
    select_expr << "'";

    const std::string  pattern = (tmp / "frame_%06d.png").string();
    std::ostringstream cmd;
    // Use PNG for lossless compression and extract specific frames
    cmd << "ffmpeg -hide_banner -loglevel error -y -i " << quote(video_path) << " -vf \"" << select_expr.str()
        << ",format=rgb24\" ";                             // Ensure RGB format
    cmd << " -vsync 0 -pix_fmt rgb24 " << quote(pattern);  // Force RGB pixel format, vsync 0 for exact frames

    if (std::system(cmd.str().c_str()) != 0) {
        err = "ffmpeg extraction failed";
        std::filesystem::remove_all(tmp, ec);
        return false;
    }

    std::vector<std::filesystem::path> frames;
    for (auto & p : std::filesystem::directory_iterator(tmp)) {
        if (!p.is_regular_file()) {
            continue;
        }
        auto ext = p.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
            frames.push_back(std::filesystem::absolute(p.path()));
        }
    }
    std::sort(frames.begin(), frames.end());
    if (frames.empty()) {
        err = "No frames extracted";
        std::filesystem::remove_all(tmp, ec);
        return false;
    }

    out.temp_dir = tmp;
    out.frames   = std::move(frames);
    return true;
}

void mtmd_video_cleanup(const mtmd_video_result & res) {
    std::error_code ec;
    if (!res.temp_dir.empty() && std::filesystem::exists(res.temp_dir, ec)) {
        std::filesystem::remove_all(res.temp_dir, ec);
    }
}
