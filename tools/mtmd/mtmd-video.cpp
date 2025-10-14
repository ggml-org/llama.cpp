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

bool load_frames_from_dir(mtmd_context * ctx,
                          const std::string & dir_path,
                          std::vector<mtmd::bitmap> & out_frames,
                          const LoadVideoOptions & opts) {
    if (!ctx || dir_path.empty() || !is_dir(dir_path)) {
        return false;
    }
    // note: hparam-based control is applied inside clip.cpp; nothing to set globally here

    std::vector<std::string> files;
    list_files(dir_path, files, opts.recursive);
    std::sort(files.begin(), files.end());

    int stride = std::max(1, opts.stride);
    int loaded = 0;
    for (size_t i = 0; i < files.size(); i++) {
        if ((int)i % stride != 0) continue;
        const std::string & f = files[i];
        if (!has_image_ext(f)) continue;
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx, f.c_str()));
        if (!bmp.ptr) continue;
        out_frames.push_back(std::move(bmp));
        loaded++;
        if (opts.max_frames > 0 && loaded >= opts.max_frames) break;
    }
    return loaded > 0;
}

size_t append_frames_from_dir(mtmd_context * ctx,
                              const std::string & dir_path,
                              mtmd::bitmaps & dst,
                              const LoadVideoOptions & opts) {
    std::vector<mtmd::bitmap> frames;
    if (!load_frames_from_dir(ctx, dir_path, frames, opts)) {
        return 0;
    }
    size_t before = dst.entries.size();
    for (auto & f : frames) dst.entries.push_back(std::move(f));
    return dst.entries.size() - before;
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

static bool decode_video_ffmpeg_to_rgba(const std::string & file,
                                        std::vector<DecodedFrameRGBA> & frames,
                                        int max_frames,
                                        int stride) {
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
            if (stride > 1 && (idx++ % stride != 0)) continue;
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
            if (max_frames > 0 && taken >= max_frames) break;
        }
        if (max_frames > 0 && taken >= max_frames) break;
    }
    if (sws) sws_freeContext(sws);
    return taken > 0;
}

bool load_frames_from_file(mtmd_context * ctx,
                           const std::string & file_path,
                           std::vector<mtmd::bitmap> & out_frames,
                           const LoadVideoOptions & opts) {
    if (!ctx) return false;
    std::vector<DecodedFrameRGBA> decoded;
    if (!decode_video_ffmpeg_to_rgba(file_path, decoded, opts.max_frames, std::max(1, opts.stride))) {
        return false;
    }
    for (auto & fr : decoded) {
        const int w = fr.width;
        const int h = fr.height;
        std::vector<unsigned char> rgb;
        rgb.resize((size_t)w * h * 3);
        const unsigned char * src = fr.rgba.data();
        unsigned char * dst = rgb.data();
        for (int i = 0; i < w * h; ++i) {
            dst[0] = src[0]; // R
            dst[1] = src[1]; // G
            dst[2] = src[2]; // B
            dst += 3;
            src += 4; // skip A
        }
        mtmd::bitmap bmp(mtmd_bitmap_init((uint32_t)w, (uint32_t)h, rgb.data()));
        if (bmp.ptr) out_frames.push_back(std::move(bmp));
    }
    return !out_frames.empty();
}
#else
bool load_frames_from_file(mtmd_context * /*ctx*/,
                           const std::string & /*file_path*/,
                           std::vector<mtmd::bitmap> & /*out_frames*/,
                           const LoadVideoOptions & /*opts*/) {
    return false;
}
#endif

size_t append_frames_from_path(mtmd_context * ctx,
                               const std::string & path,
                               mtmd::bitmaps & dst,
                               const LoadVideoOptions & opts) {
    if (is_dir(path)) {
        return append_frames_from_dir(ctx, path, dst, opts);
    } else {
        std::vector<mtmd::bitmap> frames;
        if (!load_frames_from_file(ctx, path, frames, opts)) return 0;
        size_t before = dst.entries.size();
        for (auto & f : frames) dst.entries.push_back(std::move(f));
        return dst.entries.size() - before;
    }
}

} // namespace mtmd_video


