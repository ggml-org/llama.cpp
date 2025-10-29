#include "mtmd-video.h"
#include "mtmd-helper.h"
#include "clip.h"

#include <algorithm>
#include <string>
#include <vector>
#include <cstdio>
#include <cerrno>
#include <cstring>
#include <memory>
#include <cmath>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace {

static bool has_image_ext(const std::string & name) {
    auto lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c){ return (char)std::tolower(c); });
    return lower.rfind(".jpg")  != std::string::npos ||
           lower.rfind(".jpeg") != std::string::npos ||
           lower.rfind(".png")  != std::string::npos ||
           lower.rfind(".bmp")  != std::string::npos ||
           lower.rfind(".gif")  != std::string::npos ||
           lower.rfind(".webp") != std::string::npos;
}

static bool is_dir(const std::string & path) {
#if defined(_WIN32)
    DWORD attrs = GetFileAttributesA(path.c_str());
    return (attrs != INVALID_FILE_ATTRIBUTES) && (attrs & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return false;
    return S_ISDIR(st.st_mode);
#endif
}

static void list_files(const std::string & dir, std::vector<std::string> & out, bool recursive) {
#if defined(_WIN32)
    std::string pattern = dir;
    if (!pattern.empty() && pattern.back() != '/' && pattern.back() != '\\') pattern += "\\";
    pattern += "*";
    WIN32_FIND_DATAA ffd;
    HANDLE hFind = FindFirstFileA(pattern.c_str(), &ffd);
    if (hFind == INVALID_HANDLE_VALUE) return;
    do {
        std::string name = ffd.cFileName;
        if (name == "." || name == "..") continue;
        std::string path = dir;
        if (!path.empty() && path.back() != '/' && path.back() != '\\') path += "\\";
        path += name;
        if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            if (recursive) list_files(path, out, recursive);
        } else {
            out.push_back(path);
        }
    } while (FindNextFileA(hFind, &ffd) != 0);
    FindClose(hFind);
#else
    DIR * dp = opendir(dir.c_str());
    if (!dp) return;
    struct dirent * de;
    while ((de = readdir(dp)) != nullptr) {
        std::string name = de->d_name;
        if (name == "." || name == "..") continue;
        std::string path = dir + "/" + name;
        if (is_dir(path)) {
            if (recursive) list_files(path, out, recursive);
        } else {
            out.push_back(path);
        }
    }
    closedir(dp);
#endif
}

} // namespace

namespace mtmd_video {

bool is_video_file(const std::string & path){
    auto lower = path;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c){ return (char)std::tolower(c); });
    return lower.rfind(".mp4")  != std::string::npos ||
           lower.rfind(".mov")  != std::string::npos ||
           lower.rfind(".mkv")  != std::string::npos ||
           lower.rfind(".avi")  != std::string::npos ||
           lower.rfind(".webm") != std::string::npos;
}

// untested
static mtmd_bitmap* load_frames_from_dir(mtmd_context * ctx,
                          const std::string & dir_path,
                          const LoadVideoOptions & opts) {
    if (!ctx || dir_path.empty() || !is_dir(dir_path) || opts.max_frames < 1) {
        return nullptr;
    }
    // note: hparam-based control is applied inside clip.cpp; nothing to set globally here

    std::vector<std::string> files;
    list_files(dir_path, files, opts.recursive);
    std::sort(files.begin(), files.end());

    auto stride = std::max(1u, opts.stride);
    size_t loaded = 0;
    unsigned char* dest = nullptr;
    mtmd_bitmap* out_frames = nullptr;

    uint32_t w=0, h=0;
    for (size_t i = 0; i < files.size(); i++) {
        if (i % stride != 0) continue;
        const std::string & f = files[i];
        if (!has_image_ext(f)) continue;
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx, f.c_str()));
        if (!bmp.ptr) continue;
        if(loaded==0){
            w = bmp.nx();
            h = bmp.ny();
            out_frames = mtmd_bitmap_init_from_video(w, h, loaded, nullptr);
            dest = mtmd_bitmap_get_data_mutable(out_frames);
        }else if(bmp.nx() != w || bmp.ny() != h){
            return nullptr; // all frames must have the same size
        }
        std::memcpy(dest,
                    bmp.data(),
                    bmp.n_bytes());
        dest += bmp.n_bytes();
        loaded++;
        if (loaded >= opts.max_frames) break;
    }
    
    return out_frames;
}

// --- FFmpeg-based file decoding (optional) ---

#ifdef MTMD_WITH_FFMPEG
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}
#pragma GCC diagnostic pop

struct DecodedFrameRGBA {
    int width;
    int height;
    std::vector<unsigned char> rgba; // size = width * height * 4
};

bool get_video_info_ffmpeg(const std::string &file, VideoInfo &info) {
    AVFormatContext *fmt = nullptr;
    if (avformat_open_input(&fmt, file.c_str(), nullptr, nullptr) < 0) {
        return false;
    }

    std::unique_ptr<AVFormatContext, void(*)(AVFormatContext*)> fmt_guard(fmt, 
        [](AVFormatContext *f){ if (f) {avformat_close_input(&f);} });

    if (avformat_find_stream_info(fmt, nullptr) < 0) {
        return false;
    }

    // find video stream
    int vstream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (vstream < 0) {
        return false;
    }

    AVStream *st = fmt->streams[vstream];

    // get fps
    if (st->avg_frame_rate.num > 0 && st->avg_frame_rate.den > 0){
        info.fps = av_q2d(st->avg_frame_rate);
    }else if (st->r_frame_rate.num > 0 && st->r_frame_rate.den > 0){
        info.fps = av_q2d(st->r_frame_rate);
    }
    // get total frames
    if (st->nb_frames > 0){
        info.total_frames = st->nb_frames;
    }else if (fmt->duration > 0 && info.fps > 0.0){
        // estimate total frames if nb_frames is not available
        info.total_frames = std::llround((fmt->duration / (double)AV_TIME_BASE) * info.fps);
    }

    return true;
}

static bool decode_video_ffmpeg_to_rgba(const std::string & file,
                                        std::vector<DecodedFrameRGBA> & frames,
                                        int max_frames,
                                        int stride) {
    if(stride <= 0 || max_frames <= 0) return false;
    AVFormatContext * fmt = nullptr;
    if (avformat_open_input(&fmt, file.c_str(), nullptr, nullptr) < 0) return false;
    std::unique_ptr<AVFormatContext, void(*)(AVFormatContext*)> fmt_guard(fmt, [](AVFormatContext *f){ if (f) avformat_close_input(&f); });
    if (avformat_find_stream_info(fmt, nullptr) < 0) return false;
    int vstream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (vstream < 0) return false;
    AVStream * st = fmt->streams[vstream];
    const AVCodec * dec = avcodec_find_decoder(st->codecpar->codec_id);
    if (!dec) return false;
    AVCodecContext * ctx = avcodec_alloc_context3(dec);
    if (!ctx) return false;
    std::unique_ptr<AVCodecContext, void(*)(AVCodecContext*)> ctx_guard(ctx, [](AVCodecContext *c){ if (c) avcodec_free_context(&c); });
    if (avcodec_parameters_to_context(ctx, st->codecpar) < 0) return false;
    if (avcodec_open2(ctx, dec, nullptr) < 0) return false;

    AVFrame * frame = av_frame_alloc();
    AVPacket * pkt  = av_packet_alloc();
    std::unique_ptr<AVFrame, void(*)(AVFrame*)> frame_guard(frame, [](AVFrame *f){ if (f) av_frame_free(&f); });
    std::unique_ptr<AVPacket, void(*)(AVPacket*)> pkt_guard(pkt, [](AVPacket *p){ if (p) av_packet_free(&p); });

    SwsContext * sws = nullptr;
    int idx = 0;
    int taken = 0;
    while (av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index != vstream) { av_packet_unref(pkt); continue; }
        if (avcodec_send_packet(ctx, pkt) < 0) { av_packet_unref(pkt); break; }
        av_packet_unref(pkt);
        while (avcodec_receive_frame(ctx, frame) == 0) {
            if (idx++ % stride != stride/2) continue;
            if (!sws) {
                sws = sws_getContext(frame->width, frame->height, (AVPixelFormat)frame->format,
                                     frame->width, frame->height, AV_PIX_FMT_RGBA,
                                     SWS_BILINEAR, nullptr, nullptr, nullptr);
                if (!sws) return false;
            }
            DecodedFrameRGBA out;
            out.width = frame->width;
            out.height = frame->height;
            out.rgba.resize((size_t)frame->width * frame->height * 4);
            uint8_t * dst_data[4] = { out.rgba.data(), nullptr, nullptr, nullptr };
            int dst_linesize[4] = { frame->width * 4, 0, 0, 0 };
            sws_scale(sws, frame->data, frame->linesize, 0, frame->height, dst_data, dst_linesize);
            frames.push_back(std::move(out));
            taken++;
            if (taken >= max_frames) break;
        }
        if (taken >= max_frames) break;
    }
    if (sws) sws_freeContext(sws);
    return taken > 0;
}

static mtmd_bitmap* load_frames_from_file(mtmd_context * ctx,
                           const std::string & file_path,
                           const LoadVideoOptions & opts) {
    if (!ctx) return nullptr;
    std::vector<DecodedFrameRGBA> decoded;
    if (!decode_video_ffmpeg_to_rgba(file_path, decoded, opts.max_frames, std::max(1u, opts.stride))) {
        return nullptr;
    }
    const size_t nframes = decoded.size();
    if(nframes < 1){
        return nullptr;
    }
    const int w = decoded[0].width;
    const int h = decoded[0].height;
    mtmd_bitmap* out_frames = mtmd_bitmap_init_from_video(uint32_t(w), uint32_t(h), uint32_t(nframes), nullptr);
    unsigned char * dst = mtmd_bitmap_get_data_mutable(out_frames);

    for (auto & fr : decoded) {
        GGML_ASSERT(w == fr.width && h == fr.height);
        const unsigned char * src = fr.rgba.data();
        for (int i = 0; i < w * h; ++i) {
            dst[0] = src[0]; // R
            dst[1] = src[1]; // G
            dst[2] = src[2]; // B
            dst += 3;
            src += 4; // skip A
        }
    }

    return out_frames;
}
#else
static mtmd_bitmap* load_frames_from_file(mtmd_context * /*ctx*/,
                           const std::string & /*file_path*/,
                           const LoadVideoOptions & /*opts*/) {
    return nullptr;
}
bool get_video_info_ffmpeg(const std::string &file, VideoInfo &info) {
    LOG_ERR("FFmpeg support is not enabled in this build\n");
    return false;
}
#endif

mtmd_bitmap* init_video_bitmap_from_path(mtmd_context * ctx,
                               const std::string & path) {
    mtmd_video::LoadVideoOptions opts;
    opts.max_frames = 32;
    opts.stride     = 1;
    opts.recursive  = false;

    auto info = mtmd_video::VideoInfo{};
    if(is_dir(path)) {
        info.fps = 1;
        std::vector<std::string> files;
        list_files(path, files, opts.recursive);
        info.total_frames = files.size();
    } else {
        if(!mtmd_video::get_video_info_ffmpeg(path, info)) {
            return nullptr;
        }
    }

    // minicpm normal speed
    const int32_t minicpmv_max_video_frames = 64;
    opts.max_frames = minicpmv_max_video_frames;
    if(info.total_frames > minicpmv_max_video_frames) {
        // uniform sample
        opts.stride = (int)std::ceil((double)info.total_frames / minicpmv_max_video_frames);
    } else {
        // 1 frame per second
        opts.stride = (info.fps > 1.0) ? (int)std::ceil(info.fps) : 1;
    }

    if (is_dir(path)) {
        return load_frames_from_dir(ctx, path, opts);
    }

    return load_frames_from_file(ctx, path, opts);
}

} // namespace mtmd_video


