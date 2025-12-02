#pragma once
#include <filesystem>
#include <string>
#include <vector>

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define MTMD_API __declspec(dllexport)
#        else
#            define MTMD_API __declspec(dllimport)
#        endif
#    else
#        define MTMD_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define MTMD_API
#endif

struct mtmd_video_opts {
    float fps      = 1.0f;   
    int   max_frames = 0;    // 0 = unlimited
};

struct mtmd_video_result {
    std::filesystem::path              temp_dir;  // where the frames are dumped using ffmpeg and then read in mtmd::bitmaps
    std::vector<std::filesystem::path> frames;    // absolute path to the extracted frames in sorted order
    double                             seconds_per_frame = 1.0;  // 1.0 fps is default value
};

// Returns false and fills error message in err on failure, else returns true
MTMD_API bool mtmd_video_ffmpeg_available(std::string & err);

// Extract frames with ffmpeg -vf fps=<fps> into a unique temp dir. On success fills result out and returns true, else returns false and fills error message in err on failure
MTMD_API bool mtmd_video_extract_frames(const std::string & video_path, const mtmd_video_opts & opts, mtmd_video_result & out, std::string & err);

// Delete the temp directory in which mtmd_video_extract_frames extracts frames from the input video
MTMD_API void mtmd_video_cleanup(const mtmd_video_result & res);

// A simple helper to append frames into the args.images vector.
inline void mtmd_append_images(std::vector<std::string> & dst, const std::vector<std::string> & src) {
    dst.insert(dst.end(), src.begin(), src.end());
}
