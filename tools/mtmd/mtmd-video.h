#pragma once

#include <filesystem>
#include <string>
#include <vector>

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define MTMD_VIDEO_API __declspec(dllexport)
#        else
#            define MTMD_VIDEO_API __declspec(dllimport)
#        endif
#    else
#        define MTMD_VIDEO_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define MTMD_VIDEO_API
#endif

// Video frame extraction options
struct mtmd_video_opts {
    float fps        = 1.0f;  // frames per second to extract
    int   max_frames = 0;     // maximum number of frames (0 = unlimited)
};

// Result of video frame extraction
struct mtmd_video_result {
    std::filesystem::path              temp_dir;           // temporary directory where frames are stored
    std::vector<std::filesystem::path> frames;             // absolute paths to extracted frames (sorted)
    double                             seconds_per_frame = 1.0;  // interval between frames in seconds
};

// Check if ffmpeg is available in PATH
// Returns false and fills error message in err on failure, else returns true
MTMD_VIDEO_API bool mtmd_video_ffmpeg_available(std::string & err);

// Extract frames from video using ffmpeg
// On success fills result and returns true, else returns false and fills error message in err
MTMD_VIDEO_API bool mtmd_video_extract_frames(
    const std::string &     video_path,
    const mtmd_video_opts & opts,
    mtmd_video_result &     out,
    std::string &           err);

// Delete the temp directory created by mtmd_video_extract_frames
MTMD_VIDEO_API void mtmd_video_cleanup(const mtmd_video_result & res);

// A simple helper to append frames into a string vector
inline void mtmd_video_append_frames(std::vector<std::string> & dst, const mtmd_video_result & res) {
    for (const auto & frame : res.frames) {
        dst.push_back(frame.string());
    }
}

