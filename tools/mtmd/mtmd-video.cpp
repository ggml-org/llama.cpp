#include "mtmd-video.h"
#include "clip-impl.h"
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

static bool get_video_info_from_dir(const std::string &path, VideoInfo &info){
    info.fps = 1; // do not care
    std::vector<std::string> files;
    mtmd_helper::list_files(path, files, true); // recursive
    info.total_frames = files.size();
    return true;
}
// untested
static mtmd_bitmap* load_frames_from_dir(mtmd_context * ctx,
                          const std::string & dir_path,
                          const LoadVideoOptions & opts) {
    if (!ctx || dir_path.empty() || !mtmd_helper::is_dir(dir_path) || opts.max_frames < 1) {
        return nullptr;
    }
    std::vector<std::string> files;
    mtmd_helper::list_files(dir_path, files, opts.recursive);
    std::sort(files.begin(), files.end());

    auto stride = std::max(1u, opts.stride);
    size_t loaded = 0;
    unsigned char* dest = nullptr;
    mtmd_bitmap* out_frames = nullptr;
    const auto nframes = std::min(files.size() / stride, (size_t)opts.max_frames);

    uint32_t w=0, h=0;
    for (size_t i = 0; i < files.size(); i++) {
        if (i % stride != 0) continue;
        const std::string & f = files[i];
        if (!mtmd_helper::has_image_ext(f)) continue;
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx, f.c_str()));
        if (!bmp.ptr) continue;
        if(loaded==0){
            w = bmp.nx();
            h = bmp.ny();
            out_frames = mtmd_bitmap_init_from_video(w, h, nframes, nullptr);
            dest = mtmd_bitmap_get_data_mutable(out_frames);
        }
        GGML_ASSERT(bmp.nx() == w && bmp.ny() == h); // all frames must have the same size
        std::memcpy(dest,
                    bmp.data(),
                    bmp.n_bytes());
        dest += bmp.n_bytes();
        loaded++;
        if (loaded >= nframes) break;
    }
    
    return out_frames;
}

struct DecodedFrameRGBA {
    int width;
    int height;
    std::vector<unsigned char> rgba; // size = width * height * 4
};

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
                                       AVIOContext*& avio_ctx,
                                       uint8_t*& avio_ctx_buffer) {
    fmt = nullptr;
    avio_ctx = nullptr;
    avio_ctx_buffer = nullptr;

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

// from buffer
bool get_video_info(const uint8_t* buffer, size_t size, VideoInfo &info) {
    AVFormatContext* fmt = nullptr;
    AVIOContext* avio_ctx = nullptr;
    uint8_t* avio_ctx_buffer = nullptr;

    GGML_ASSERT(create_format_context_from_buffer(buffer, size, fmt, avio_ctx, avio_ctx_buffer));
    bool ok = get_video_info_from_format_ctx(fmt, info);
    free_format_context_from_buffer(fmt, avio_ctx);
    return ok;
}

// from file
bool get_video_info(const std::string &path, VideoInfo &info) {
    if(mtmd_helper::is_dir(path)) return get_video_info_from_dir(path, info);

    AVFormatContext* fmt = nullptr;
    if (avformat_open_input(&fmt, path.c_str(), nullptr, nullptr) < 0)
        return false;

    std::unique_ptr<AVFormatContext, void(*)(AVFormatContext*)> fmt_guard(fmt, [](AVFormatContext* f){
        if (f) avformat_close_input(&f);
    });

    return get_video_info_from_format_ctx(fmt, info);
}

static bool decode_video_ffmpeg_to_rgba_from_format_ctx(
    AVFormatContext* fmt,
    std::vector<DecodedFrameRGBA>& frames,
    int max_frames,
    int stride) 
{
    if(!fmt || stride <= 0 || max_frames <= 0) return false;
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
    }
    if (sws) sws_freeContext(sws);
    return taken > 0;
}

// from file
static bool decode_video_ffmpeg_to_rgba(
    const std::string& file,
    std::vector<DecodedFrameRGBA>& frames,
    int max_frames,
    int stride)
{
    AVFormatContext* fmt = nullptr;
    if (avformat_open_input(&fmt, file.c_str(), nullptr, nullptr) < 0)
        return false;

    std::unique_ptr<AVFormatContext, void(*)(AVFormatContext*)> fmt_guard(fmt, [](AVFormatContext* f){
        if (f) avformat_close_input(&f);
    });

    return decode_video_ffmpeg_to_rgba_from_format_ctx(fmt, frames, max_frames, stride);
}

// from buffer
static bool decode_video_ffmpeg_to_rgba(
    const uint8_t* buffer,
    size_t size,
    std::vector<DecodedFrameRGBA>& frames,
    int max_frames,
    int stride)
{
    if (!buffer || size == 0) return false;
    AVFormatContext* fmt = nullptr;
    AVIOContext* avio_ctx = nullptr;
    uint8_t* avio_ctx_buffer = nullptr;

    GGML_ASSERT(create_format_context_from_buffer(buffer, size, fmt, avio_ctx, avio_ctx_buffer));
    
    bool ok = decode_video_ffmpeg_to_rgba_from_format_ctx(fmt, frames, max_frames, stride);

    free_format_context_from_buffer(fmt, avio_ctx);
    return ok;
}
#else
bool get_video_info(const std::string &path, VideoInfo &info){
    if(mtmd_helper::is_dir(path)) return get_video_info_from_dir(path, info);
    LOG_ERR("FFmpeg support is not enabled in this build\n");
    return false;
}
bool get_video_info(const uint8_t* /*buffer*/, size_t /*size*/, VideoInfo &/*info*/){
    LOG_ERR("FFmpeg support is not enabled in this build\n");
    return false;
}
bool is_video_buffer(const uint8_t */*data*/, size_t /*size*/){
    LOG_ERR("FFmpeg support is not enabled in this build\n");
    return false;
}
static bool decode_video_ffmpeg_to_rgba(
    const std::string& /*file*/,
    std::vector<DecodedFrameRGBA>& /*frames*/,
    int /*max_frames*/,
    int /*stride*/)
{
    return false;   
}
static bool decode_video_ffmpeg_to_rgba(
    const uint8_t* /*buffer*/,
    size_t /*size*/,
    std::vector<DecodedFrameRGBA>& /*frames*/,
    int /*max_frames*/,
    int /*stride*/)
{
    return false;   
}
#endif

static mtmd_bitmap* convert_frames_to_bitmap(mtmd_context * ctx, const std::vector<DecodedFrameRGBA>& decoded) {
    if (!ctx) return nullptr;
    if(decoded.empty()) return nullptr;
    const size_t nframes = decoded.size();
    if(nframes < 1){
        return nullptr;
    }
    const int w = decoded[0].width;
    const int h = decoded[0].height;
    mtmd_bitmap* out_frames = mtmd_bitmap_init_from_video(uint32_t(w), uint32_t(h), uint32_t(nframes), nullptr);
    unsigned char * dst = mtmd_bitmap_get_data_mutable(out_frames);

    for (const auto & fr : decoded) {
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

static mtmd_video::LoadVideoOptions get_video_sample_options(mtmd_video::VideoInfo info){
    mtmd_video::LoadVideoOptions opts;
    opts.max_frames = 32;
    opts.stride     = 1;
    opts.recursive  = false;

    /* MiniCPM-V normal-speed video frames sample method */

#ifdef MTMD_MAX_VIDEO_FRAMES_SMALL
    // set a small number of frames for fast test locally
    const int32_t minicpmv_max_video_frames = 4;
#else
    const int32_t minicpmv_max_video_frames = 64;
#endif
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

mtmd_bitmap* init_video_bitmap(mtmd_context * ctx, const std::string & path) {
    auto info = mtmd_video::VideoInfo{};
    if(!mtmd_video::get_video_info(path, info)) {
        LOG_ERR("Unable to get video info from path: %s\n", path.c_str());
        return nullptr;
    }

    const auto opts = get_video_sample_options(info);

    if (mtmd_helper::is_dir(path)) {
        return load_frames_from_dir(ctx, path, opts);
    }

    std::vector<DecodedFrameRGBA> frames;
    if(!decode_video_ffmpeg_to_rgba(path, frames, opts.max_frames, std::max(1u, opts.stride))){
        LOG_ERR("Unable to decode video from path: %s\n", path.c_str());
        return nullptr;
    }

    return convert_frames_to_bitmap(ctx, frames);
}

mtmd_bitmap* init_video_bitmap(mtmd_context * ctx, const uint8_t* buffer, size_t size){
    auto info = mtmd_video::VideoInfo{};
    if(!mtmd_video::get_video_info(buffer, size, info)) {
        LOG_ERR("Unable to get video info from buffer\n");
        return nullptr;
    }

    const auto opts = get_video_sample_options(info);

    std::vector<DecodedFrameRGBA> frames;
    if(!decode_video_ffmpeg_to_rgba(buffer, size, frames, opts.max_frames, std::max(1u, opts.stride))){
        LOG_ERR("Unable to decode video from buffer\n");
        return nullptr;
    }

    return convert_frames_to_bitmap(ctx, frames);
}

} // namespace mtmd_video


