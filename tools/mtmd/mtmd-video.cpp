#include <cstdint>
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

#include "mtmd-video.h"
#include "ggml.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <string>
#include <vector>
#include <cstdio>
#include <cerrno>
#include <cstring>
#include <memory>
#include <cmath>
#include <fstream>

#define LOG_INF(...) fprintf(stdout, __VA_ARGS__)
#define LOG_WRN(...) fprintf(stdout, __VA_ARGS__)
#define LOG_ERR(...) fprintf(stderr, __VA_ARGS__)

static bool is_image_file(const std::string & path) {
    // check extension first
    std::string lower = path;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c){ return (char)std::tolower(c); });

    bool has_ext = lower.rfind(".jpg")  != std::string::npos ||
                   lower.rfind(".jpeg") != std::string::npos ||
                   lower.rfind(".png")  != std::string::npos ||
                   lower.rfind(".bmp")  != std::string::npos ||
                   lower.rfind(".gif")  != std::string::npos ||
                   lower.rfind(".webp") != std::string::npos;

    if (!has_ext) return false;

    // check magic bytes then
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    std::vector<unsigned char> header(16);
    f.read(reinterpret_cast<char*>(header.data()), header.size());
    size_t n = f.gcount();
    if (n < 4) return false; // too small to be an image

    // JPEG: FF D8 FF
    if (n >= 3 && header[0] == 0xFF && header[1] == 0xD8 && header[2] == 0xFF)
        return true;

    // PNG: 89 50 4E 47 0D 0A 1A 0A
    if (n >= 8 &&
        header[0] == 0x89 && header[1] == 0x50 &&
        header[2] == 0x4E && header[3] == 0x47 &&
        header[4] == 0x0D && header[5] == 0x0A &&
        header[6] == 0x1A && header[7] == 0x0A)
        return true;

    // BMP: 42 4D
    if (n >= 2 && header[0] == 0x42 && header[1] == 0x4D)
        return true;

    // GIF: "GIF87a" or "GIF89a"
    if (n >= 6 &&
        header[0] == 'G' && header[1] == 'I' && header[2] == 'F' &&
        header[3] == '8' &&
        (header[4] == '7' || header[4] == '9') &&
        header[5] == 'a')
        return true;

    // WEBP: "RIFF....WEBP"
    if (n >= 12 &&
        header[0] == 'R' && header[1] == 'I' && header[2] == 'F' && header[3] == 'F' &&
        header[8] == 'W' && header[9] == 'E' && header[10] == 'B' && header[11] == 'P')
        return true;

    return false;
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

namespace mtmd_video {

bool is_video_file(const std::string & path){
    if(is_dir(path)) return true; // a directory containing frames images is also considered a video
    auto lower = path;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c){ return (char)std::tolower(c); });
    return lower.rfind(".mp4")  != std::string::npos ||
           lower.rfind(".mov")  != std::string::npos ||
           lower.rfind(".mkv")  != std::string::npos ||
           lower.rfind(".avi")  != std::string::npos ||
           lower.rfind(".webm") != std::string::npos;
}

static void get_video_info_from_dir(const std::string &path, VideoInfo &info){
    info.fps = 1; // do not care
    std::vector<std::string> files;
    list_files(path, files, true); // recursive
    info.total_frames = files.size();
}
// untested
static mtmd_bitmap* load_frames_from_dir(mtmd_context * ctx,
                          const std::string & dir_path,
                          const LoadVideoOptions & opts) {
    if (!ctx || dir_path.empty() || !is_dir(dir_path) || opts.max_frames < 1) {
        return nullptr;
    }
    std::vector<std::string> files;
    list_files(dir_path, files, opts.recursive);
    std::sort(files.begin(), files.end());

    auto stride = std::max(1u, opts.stride);
    size_t loaded = 0;
    mtmd_bitmap* out_frames = nullptr;
    const auto nframes = std::min(files.size() / stride, (size_t)opts.max_frames);

    uint32_t w=0, h=0;
    for (size_t i = 0; i < files.size(); i++) {
        if (i % stride != 0) continue;
        const std::string & f = files[i];
        if (!is_image_file(f)) continue;
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx, f.c_str()));
        if (!bmp.ptr) continue;
        if(loaded==0){
            w = bmp.nx();
            h = bmp.ny();
            out_frames = mtmd_bitmap_init_from_video(w, h, nframes, nullptr);
        }
        GGML_ASSERT(bmp.nx() == w && bmp.ny() == h); // all frames must have the same size
        mtmd_bitmap_set_frame(out_frames, loaded, bmp.data());
        loaded++;
        if (loaded >= nframes) break;
    }
    
    return out_frames;
}

// This Frames Queue is useful for streaming use case, 
// This frame queue is designed for streaming scenarios,
// where an FFmpeg-related thread acts as the producer generating frames,
// and the main program (the LLM inference thread) acts as the consumer processing them.
struct DecodedFramesQueue {
    int width;
    int height;
    std::queue<std::vector<unsigned char>> frames; // frame size = width * height * 3
};

static mtmd_video::LoadVideoOptions get_video_sample_options(mtmd_video::VideoInfo info){
    mtmd_video::LoadVideoOptions opts;
    opts.max_frames = 32;
    opts.stride     = 1;
    opts.recursive  = false;

    /* MiniCPM-V normal-speed video frames sample method */
    bool debug_use_small_frames_for_video = std::getenv("MTMD_MAX_VIDEO_FRAMES_SMALL") != nullptr;
    int32_t minicpmv_max_video_frames = 64;
    if(debug_use_small_frames_for_video){
        // set a small number of frames for fast test locally
        minicpmv_max_video_frames = 4;
    }
    opts.max_frames = minicpmv_max_video_frames;
    if(info.total_frames > minicpmv_max_video_frames) {
        // uniform sample
        opts.stride = (int)std::ceil((double)info.total_frames / minicpmv_max_video_frames);
    } else {
        // 1 frame per second
        opts.stride = (info.fps > 1.0) ? (int)std::ceil(info.fps) : 1;
    }
    
    return opts;
}

// --- FFmpeg-based file decoding (optional) ---

#ifdef MTMD_WITH_FFMPEG
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}
#pragma GCC diagnostic pop

bool is_video_buffer(const uint8_t *data, size_t size){
    if (!data || size < 16) return false; // too short

    AVProbeData probe;
    probe.filename = "";

    // ffmpeg requires that the last AVPROBE_PADDING_SIZE bytes of the buffer must be 0
    auto * padded = new uint8_t[size + AVPROBE_PADDING_SIZE];
    memcpy(padded, data, size);
    memset(padded + size, 0, AVPROBE_PADDING_SIZE);
    probe.buf = padded;
    probe.buf_size = (int)size;

    int score = 0;
    const AVInputFormat *fmt = av_probe_input_format2(&probe, 1, &score);

    delete [] padded;
    
    if (!fmt || score < 25) return false;
    if (fmt->flags & AVFMT_NOFILE) return false;

    static const char *image_formats[] = {
        "jpeg_pipe", "png_pipe", "bmp_pipe", "gif_pipe", "webp_pipe",
        "tiff_pipe", "image2", "image2pipe", "mjpeg"
    };
    for (const auto* name : image_formats)
        if (fmt->name && strstr(fmt->name, name))
            return false;

    return true;
}

struct BufferData {
    const uint8_t* base;
    size_t size;
    size_t pos;
    BufferData(const uint8_t* b, size_t s) : base(b), size(s), pos(0) {}
};

static int read_packet(void* opaque, uint8_t* buf, int buf_size) {
    BufferData* bd = static_cast<BufferData*>(opaque);
    if (!bd || !bd->base) return AVERROR(EIO);
    if (bd->pos >= bd->size) return AVERROR_EOF;
    size_t rem = bd->size - bd->pos;
    int to_read = (int)(rem < (size_t)buf_size ? rem : (size_t)buf_size);
    if (to_read == 0) return AVERROR_EOF;
    memcpy(buf, bd->base + bd->pos, to_read);
    bd->pos += to_read;
    return to_read;
}

static int64_t seek_packet(void* opaque, int64_t offset, int whence) {
    BufferData* bd = static_cast<BufferData*>(opaque);
    if (!bd) return -1;
    if (whence == AVSEEK_SIZE) return (int64_t)bd->size;
    size_t newpos = bd->pos;
    if (whence == SEEK_SET) {
        if (offset < 0 || (size_t)offset > bd->size) return -1;
        newpos = (size_t)offset;
    } else if (whence == SEEK_CUR) {
        if (offset < 0 && (size_t)(-offset) > bd->pos) return -1;
        newpos = bd->pos + (size_t)offset;
        if (newpos > bd->size) return -1;
    } else if (whence == SEEK_END) {
        if (offset > 0 || (size_t)(-offset) > bd->size) return -1;
        newpos = bd->size + (size_t)offset;
    } else return -1;
    bd->pos = newpos;
    return (int64_t)bd->pos;
}

static bool create_format_context_from_buffer(const uint8_t* buffer, size_t size,
                                       AVFormatContext*& fmt,
                                       AVIOContext*& avio_ctx) {
    fmt = nullptr;
    avio_ctx = nullptr;
    uint8_t* avio_ctx_buffer = nullptr;

    if (!buffer || size == 0) return false;

    // allocate BufferData
    BufferData* bd = new (std::nothrow) BufferData(buffer, size);
    if (!bd) return false;

    const int AVIO_BUF_SIZE = 4096;
    avio_ctx_buffer = static_cast<uint8_t*>(av_malloc(AVIO_BUF_SIZE));
    if (!avio_ctx_buffer) {
        delete bd;
        return false;
    }

    avio_ctx = avio_alloc_context(
        avio_ctx_buffer, AVIO_BUF_SIZE,
        0, // read only
        bd,
        &read_packet,
        nullptr,
        &seek_packet
    );

    if (!avio_ctx) {
        av_free(avio_ctx_buffer);
        delete bd;
        avio_ctx_buffer = nullptr;
        return false;
    }

    fmt = avformat_alloc_context();
    if (!fmt) {
        // avio_context_free frees ctx->buffer but NOT opaque
        if (avio_ctx->opaque) delete static_cast<BufferData*>(avio_ctx->opaque);
        avio_context_free(&avio_ctx);
        avio_ctx_buffer = nullptr;
        return false;
    }

    fmt->pb = avio_ctx;
    fmt->flags |= AVFMT_FLAG_CUSTOM_IO;

    // increase probing - optional but helpful for truncated/streamed files
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "probesize", "5000000", 0);
    av_dict_set(&opts, "analyzeduration", "5000000", 0);

    int ret = avformat_open_input(&fmt, "stream", nullptr, &opts);
    av_dict_free(&opts);

    if (ret < 0) {
        // Clean up carefully
        // If fmt exists and has pb, free pb and opaque appropriately
        if (fmt) {
            AVIOContext* pb = fmt->pb;
            BufferData* bd_from_fmt = pb ? static_cast<BufferData*>(pb->opaque) : nullptr;
            avformat_free_context(fmt);
            if (pb) {
                delete bd_from_fmt;
                avio_context_free(&pb); // frees pb->buffer
            }
            fmt = nullptr;
        } else {
            // fmt null: free avio_ctx and opaque
            if (avio_ctx) {
                if (avio_ctx->opaque) delete static_cast<BufferData*>(avio_ctx->opaque);
                avio_context_free(&avio_ctx);
                avio_ctx = nullptr;
            }
        }
        avio_ctx_buffer = nullptr;
        return false;
    }

    // success: avformat_open_input succeeded, fmt and pb are owned by caller,
    // but opaque (BufferData) must be deleted by us later (avformat_close_input won't delete opaque).
    return true;
}

static void free_format_context_from_buffer(AVFormatContext* fmt,
                                     AVIOContext* avio_ctx) {
    if (fmt) {
        // capture pb->opaque BEFORE closing
        AVIOContext* pb = fmt->pb;
        BufferData* bd = nullptr;
        if (pb) bd = static_cast<BufferData*>(pb->opaque);

        // this closes fmt and frees pb (and pb->buffer)
        avformat_close_input(&fmt);

        // avformat_close_input does not free opaque, so free it now
        if (bd) {
            delete bd;
            bd = nullptr;
        }
        // do NOT av_free(avio_ctx_buffer) here - it was freed with pb->buffer
        return;
    }

    // partial failure case: fmt is null but avio_ctx may still be valid
    if (avio_ctx) {
        BufferData* bd = static_cast<BufferData*>(avio_ctx->opaque);
        delete bd;
        avio_context_free(&avio_ctx); // frees avio_ctx->buffer
        // avio_ctx_buffer already freed by avio_context_free
        return;
    }
}


static bool get_video_info_from_format_ctx(AVFormatContext *fmt, VideoInfo &info) {
    if (!fmt) return false;
    
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

static bool decode_video_ffmpeg_to_rgba_from_format_ctx(
    AVFormatContext* fmt,
    DecodedFramesQueue& out_q,
    mtmd_video::LoadVideoOptions opts) 
{
    const auto stride = opts.stride;
    const auto max_frames = opts.max_frames;
    if(!fmt || stride <= 0 || max_frames <= 0) return false;
    // vstream checked before in get_video_info_from_format_ctx
    int vstream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (vstream < 0) return false;
    AVStream * st = fmt->streams[vstream];
    const AVCodec * dec = avcodec_find_decoder(st->codecpar->codec_id);
    if (!dec){
        LOG_ERR("%s: Could not find decoder for codec id %d\n", __func__, st->codecpar->codec_id);
        return false;
    }
    AVCodecContext * ctx = avcodec_alloc_context3(dec);
    if (!ctx){
        LOG_ERR("%s: Could not allocate codec context\n", __func__);
        return false;
    }
    std::unique_ptr<AVCodecContext, void(*)(AVCodecContext*)> ctx_guard(ctx, [](AVCodecContext *c){ if (c) avcodec_free_context(&c); });
    if (avcodec_parameters_to_context(ctx, st->codecpar) < 0){
        LOG_ERR("%s: Could not copy codec parameters to codec context\n", __func__);
        return false;
    }
    if (avcodec_open2(ctx, dec, nullptr) < 0){
        LOG_ERR("%s: Could not open codec2\n", __func__);
        return false;
    }

    AVFrame * frame = av_frame_alloc();
    AVPacket * pkt  = av_packet_alloc();
    std::unique_ptr<AVFrame, void(*)(AVFrame*)> frame_guard(frame, [](AVFrame *f){ if (f) av_frame_free(&f); });
    std::unique_ptr<AVPacket, void(*)(AVPacket*)> pkt_guard(pkt, [](AVPacket *p){ if (p) av_packet_free(&p); });

    SwsContext * sws = nullptr;
    uint32_t idx = 0;
    uint32_t taken = 0;
    const int w = ctx->width;
    const int h = ctx->height;

    if(!out_q.frames.empty() && (out_q.height!=h || out_q.width!=w)){
        LOG_ERR("Different size frames remain in decoded frames queue");
        return false;
    }
    out_q.width = w;
    out_q.height = h;

    while (av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index != vstream) { av_packet_unref(pkt); continue; }
        if (avcodec_send_packet(ctx, pkt) < 0) { av_packet_unref(pkt); break; }
        av_packet_unref(pkt);
        while (avcodec_receive_frame(ctx, frame) == 0) {
            if (idx++ % stride != stride/2) { av_frame_unref(frame); continue; }
            if (!sws) {
                sws = sws_getContext(frame->width, frame->height, (AVPixelFormat)frame->format,
                                        frame->width, frame->height, AV_PIX_FMT_RGB24,
                                        SWS_BILINEAR, nullptr, nullptr, nullptr);
                if (frame->format == AV_PIX_FMT_NONE) { av_frame_unref(frame); continue; }
                if (!sws){
                    LOG_ERR("%s: Could not initialize sws context\n", __func__);
                    av_frame_unref(frame);
                    return false;
                }
            }

            GGML_ASSERT(w==frame->width && h==frame->height);

            std::vector<uint8_t> out;
            out.resize((size_t)frame->width * frame->height * 3);

            uint8_t *dst_data[4] = { out.data(), nullptr, nullptr, nullptr };
            int dst_linesize[4] = { frame->width * 3, 0, 0, 0 };
            sws_scale(sws, frame->data, frame->linesize, 0, frame->height, dst_data, dst_linesize);
        
            out_q.frames.emplace(out);
            taken++;
            av_frame_unref(frame);
            if (max_frames > 0 && taken >= max_frames) break;
        }
    }
    if (sws) sws_freeContext(sws);
    return taken > 0;
}

static mtmd_bitmap* convert_frames_to_bitmap(mtmd_context * ctx, DecodedFramesQueue& decoded) {
    if (!ctx) return nullptr;
    if(decoded.frames.empty()) return nullptr;
    const size_t nframes = decoded.frames.size();
    if(nframes < 1){
        return nullptr;
    }
    const int w = decoded.width;
    const int h = decoded.height;
    mtmd_bitmap* out_frames = mtmd_bitmap_init_from_video(uint32_t(w), uint32_t(h), uint32_t(nframes), nullptr);
    for(size_t i=0;i<nframes;++i){
        mtmd_bitmap_set_frame(out_frames, i, decoded.frames.front().data());
        decoded.frames.pop();
    }

    return out_frames;
}

mtmd_bitmap* init_video_bitmap(mtmd_context * ctx, const uint8_t* buffer, size_t size){
    auto info = mtmd_video::VideoInfo{};
    AVFormatContext* fmt = nullptr;
    AVIOContext* avio_ctx = nullptr;
    GGML_ASSERT(create_format_context_from_buffer(buffer, size, fmt, avio_ctx));

    if(!get_video_info_from_format_ctx(fmt, info)) {
        LOG_ERR("Unable to get video info from buffer\n");
        free_format_context_from_buffer(fmt, avio_ctx);
        return nullptr;
    }

    const auto opts = get_video_sample_options(info);

    DecodedFramesQueue frames;
    if(!decode_video_ffmpeg_to_rgba_from_format_ctx(fmt, frames, opts)){
        LOG_ERR("Unable to decode video from buffer\n");
        free_format_context_from_buffer(fmt, avio_ctx);
        return nullptr;
    }

    auto * res = convert_frames_to_bitmap(ctx, frames);
    free_format_context_from_buffer(fmt, avio_ctx);
    return res;
}
#else
bool is_video_buffer(const uint8_t */*data*/, size_t /*size*/){
    LOG_WRN("FFmpeg support is not enabled in this build, can not check it\n");
    return false;
}
mtmd_bitmap* init_video_bitmap(mtmd_context * /*ctx*/, const uint8_t* /*buffer*/, size_t /*size*/){
    LOG_ERR("FFmpeg support is not enabled in this build, can not load video from buffer\n");
    return nullptr;
}
#endif

mtmd_bitmap* init_video_bitmap(mtmd_context * ctx, const std::string & path) {
    auto info = mtmd_video::VideoInfo{};
    
    if(is_dir(path)){
        get_video_info_from_dir(path, info);
        const auto opts = get_video_sample_options(info);
        return load_frames_from_dir(ctx, path, opts);
    }

    // handle file otherwise

    #ifdef MTMD_WITH_FFMPEG
    AVFormatContext* fmt = nullptr;
    LOG_INF("Loading video from path: %s\n", path.c_str());
    if (avformat_open_input(&fmt, path.c_str(), nullptr, nullptr) < 0){
        LOG_ERR("Unable to open video from path: %s\n", path.c_str());
        if(fmt) avformat_close_input(&fmt);
        return nullptr;
    }
    LOG_INF("get_video_info_from_format_ctx...\n");
    if(!get_video_info_from_format_ctx(fmt, info)) {
        LOG_ERR("Unable to get video info from path: %s\n", path.c_str());
        if(fmt) avformat_close_input(&fmt);
        return nullptr;
    }

    const auto opts = get_video_sample_options(info);

    LOG_INF("decode_video_ffmpeg_to_rgba_from_format_ctx...\n");

    DecodedFramesQueue frames;
    if(!decode_video_ffmpeg_to_rgba_from_format_ctx(fmt, frames, opts)){
        LOG_ERR("Unable to decode video from path: %s\n", path.c_str());
        if(fmt) avformat_close_input(&fmt);
        return nullptr;
    }

    auto * res = convert_frames_to_bitmap(ctx, frames);
    if(fmt) avformat_close_input(&fmt);
    return res;
    #else
    LOG_ERR("FFmpeg support is not enabled in this build, can not load video from file\n");
    return nullptr;
    #endif
}

} // namespace mtmd_video


