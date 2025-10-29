// mtmd-video helpers (experimental)
// Minimal C++ helpers to load video frames (as image files) from a directory
// and append them to mtmd::bitmaps for downstream tokenization.

#ifndef MTMD_VIDEO_H
#define MTMD_VIDEO_H

#include "mtmd.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mtmd_video {

struct LoadVideoOptions {
    uint32_t max_frames = 32;   // maximum frames to load (<= 0 means no limit)
    uint32_t stride     = 1;    // take every N-th frame
    bool recursive = false;// scan subdirectories
};

struct VideoInfo {
    double fps = 0.0;           // frames per second
    int64_t total_frames = 0;   // total number of frames
};

// Check if a path is a video file based on its extension
bool is_video_file(const std::string & path);

// get video info (fps, total frames) via FFmpeg (mp4/mov/mkv/avi/webm...).
// Returns true on success, false otherwise.
bool get_video_info_ffmpeg(const std::string &file, VideoInfo &info);

// Append frames loaded from a file or directory (auto-detect).
// Returns a mtmd_bitmap containing all frames in RGB format.
mtmd_bitmap* init_video_bitmap_from_path(mtmd_context * ctx,
                               const std::string & path);

} // namespace mtmd_video

#endif // MTMD_VIDEO_H


