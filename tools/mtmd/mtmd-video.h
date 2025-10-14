// mtmd-video helpers (experimental)
// Minimal C++ helpers to load video frames (as image files) from a directory
// and append them to mtmd::bitmaps for downstream tokenization.

#ifndef MTMD_VIDEO_H
#define MTMD_VIDEO_H

#include "mtmd.h"

#include <string>
#include <vector>

namespace mtmd_video {

struct LoadVideoOptions {
    int max_frames = 32;   // maximum frames to load (<= 0 means no limit)
    int stride     = 1;    // take every N-th frame
    bool recursive = false;// scan subdirectories
};

// Load frames from a directory of images (jpg/png/bmp/webp etc.)
// Returns true on success (>=1 frame loaded), false otherwise.
bool load_frames_from_dir(mtmd_context * ctx,
                          const std::string & dir_path,
                          std::vector<mtmd::bitmap> & out_frames,
                          const LoadVideoOptions & opts = {});

// Append frames loaded from a directory into mtmd::bitmaps container.
// Returns number of frames appended (0 on failure).
size_t append_frames_from_dir(mtmd_context * ctx,
                              const std::string & dir_path,
                              mtmd::bitmaps & dst,
                              const LoadVideoOptions & opts = {});

// Load frames from a video file via FFmpeg (mp4/mov/mkv/avi/webm...).
// Returns true on success (>=1 frame loaded), false otherwise.
bool load_frames_from_file(mtmd_context * ctx,
                           const std::string & file_path,
                           std::vector<mtmd::bitmap> & out_frames,
                           const LoadVideoOptions & opts = {});

// Append frames loaded from a file or directory (auto-detect).
// Returns number of frames appended (0 on failure or unsupported input).
size_t append_frames_from_path(mtmd_context * ctx,
                               const std::string & path,
                               mtmd::bitmaps & dst,
                               const LoadVideoOptions & opts = {});

} // namespace mtmd_video

#endif // MTMD_VIDEO_H


