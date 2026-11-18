#include "mtmd-cli-audio.h"

#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>

// libmtmd builds miniaudio without device IO, the CLI needs its own copy
#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_DECODING
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_API static
#include "miniaudio/miniaudio.h"

using clock_type = std::chrono::steady_clock;

static void sleep_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

//
// mtmd_cli_audio_in
//

struct mtmd_cli_audio_in::impl {
    // mic
    bool       has_device = false;
    ma_device  device;
    ma_pcm_rb  rb;
    std::atomic<int64_t> n_dropped{0};

    // samples of the caller
    const float * samples   = nullptr;
    size_t        n_samples = 0;
    size_t        pos       = 0;
    int           rate      = 0;
    bool          realtime  = false;
    size_t        n_chunk   = 0;
    clock_type::time_point t_start;

    ~impl() {
        if (has_device) {
            ma_device_uninit(&device);
            ma_pcm_rb_uninit(&rb);
        }
    }

    static void on_capture(ma_device * dev, void * output, const void * input, ma_uint32 n_frames) {
        (void) output;
        auto * self = (impl *) dev->pUserData;
        const float * in = (const float *) input;
        while (n_frames > 0) {
            ma_uint32 n = n_frames;
            void *    dst;
            if (ma_pcm_rb_acquire_write(&self->rb, &n, &dst) != MA_SUCCESS || n == 0) {
                self->n_dropped += n_frames;
                return;
            }
            memcpy(dst, in, n * sizeof(float));
            ma_pcm_rb_commit_write(&self->rb, n);
            in       += n;
            n_frames -= n;
        }
    }

    // samples of a realtime input that are due by now
    size_t pos_due() const {
        const double t = std::chrono::duration<double>(clock_type::now() - t_start).count();
        return std::min(n_samples, (size_t) (t * rate));
    }
};

mtmd_cli_audio_in::mtmd_cli_audio_in() : p(new impl()) {}
mtmd_cli_audio_in::~mtmd_cli_audio_in() = default;

bool mtmd_cli_audio_in::init_mic(int sample_rate) {
    // the reader can be late by this much before samples are lost
    const ma_uint32 n_buf = (ma_uint32) sample_rate * 10;
    if (ma_pcm_rb_init(ma_format_f32, 1, n_buf, nullptr, nullptr, &p->rb) != MA_SUCCESS) {
        return false;
    }
    ma_device_config cfg = ma_device_config_init(ma_device_type_capture);
    cfg.capture.format   = ma_format_f32;
    cfg.capture.channels = 1;
    cfg.sampleRate       = (ma_uint32) sample_rate;
    cfg.dataCallback     = impl::on_capture;
    cfg.pUserData        = p.get();
    if (ma_device_init(nullptr, &cfg, &p->device) != MA_SUCCESS) {
        LOG_ERR("%s: failed to open the capture device\n", __func__);
        ma_pcm_rb_uninit(&p->rb);
        return false;
    }
    p->has_device = true;
    if (ma_device_start(&p->device) != MA_SUCCESS) {
        LOG_ERR("%s: failed to start the capture device\n", __func__);
        return false;
    }
    LOG_INF("%s: listening on '%s'\n", __func__, p->device.capture.name);
    return true;
}

void mtmd_cli_audio_in::init_samples(const float * samples, size_t n_samples, int sample_rate, bool realtime, size_t n_chunk) {
    p->samples   = samples;
    p->n_samples = n_samples;
    p->rate      = sample_rate;
    p->realtime  = realtime;
    p->n_chunk   = n_chunk;
    p->t_start   = clock_type::now();
}

size_t mtmd_cli_audio_in::read(std::vector<float> & out) {
    size_t n_read = 0;
    if (p->has_device) {
        while (true) {
            ma_uint32 n = ma_pcm_rb_available_read(&p->rb);
            void *    src;
            if (n == 0 || ma_pcm_rb_acquire_read(&p->rb, &n, &src) != MA_SUCCESS || n == 0) {
                break;
            }
            out.insert(out.end(), (const float *) src, (const float *) src + n);
            ma_pcm_rb_commit_read(&p->rb, n);
            n_read += n;
        }
    } else {
        const size_t end = p->realtime ? p->pos_due() : std::min(p->n_samples, p->pos + p->n_chunk);
        if (end > p->pos) {
            out.insert(out.end(), p->samples + p->pos, p->samples + end);
            n_read = end - p->pos;
            p->pos = end;
        }
    }
    return n_read;
}

void mtmd_cli_audio_in::discard() {
    if (p->has_device) {
        std::vector<float> tmp;
        read(tmp);
    } else if (p->realtime) {
        p->pos = std::max(p->pos, p->pos_due());
    }
}

bool mtmd_cli_audio_in::is_end() const {
    return !p->has_device && p->pos >= p->n_samples;
}

int64_t mtmd_cli_audio_in::n_dropped() const {
    return p->n_dropped.load();
}

//
// mtmd_cli_audio_out
//

struct mtmd_cli_audio_out::impl {
    int n_channels = 1;

    // speaker
    bool       has_device = false;
    ma_device  device;
    ma_pcm_rb  rb;
    std::atomic<bool> drop{false}; // the callback throws the buffered audio away

    // file
    bool       has_encoder = false;
    ma_encoder encoder;

    ~impl() {
        if (has_device) {
            ma_device_uninit(&device);
            ma_pcm_rb_uninit(&rb);
        }
        if (has_encoder) {
            ma_encoder_uninit(&encoder);
        }
    }

    static void on_playback(ma_device * dev, void * output, const void * input, ma_uint32 n_frames) {
        (void) input;
        auto * self = (impl *) dev->pUserData;
        const bool drop = self->drop.load();
        float * out = (float *) output;
        // the output is silence where the buffer has nothing
        while (n_frames > 0 || drop) {
            ma_uint32 n = drop ? ma_pcm_rb_available_read(&self->rb) : n_frames;
            void *    src;
            if (ma_pcm_rb_acquire_read(&self->rb, &n, &src) != MA_SUCCESS || n == 0) {
                return;
            }
            if (!drop) {
                memcpy(out, src, n * self->n_channels * sizeof(float));
                out      += n * self->n_channels;
                n_frames -= n;
            }
            ma_pcm_rb_commit_read(&self->rb, n);
        }
    }
};

mtmd_cli_audio_out::mtmd_cli_audio_out() : p(new impl()) {}
mtmd_cli_audio_out::~mtmd_cli_audio_out() = default;

bool mtmd_cli_audio_out::init_speaker(int sample_rate, int n_channels) {
    p->n_channels = n_channels;
    // write() blocks once this much audio waits to be played
    const ma_uint32 n_buf = (ma_uint32) sample_rate * 2;
    if (ma_pcm_rb_init(ma_format_f32, (ma_uint32) n_channels, n_buf, nullptr, nullptr, &p->rb) != MA_SUCCESS) {
        return false;
    }
    ma_device_config cfg  = ma_device_config_init(ma_device_type_playback);
    cfg.playback.format   = ma_format_f32;
    cfg.playback.channels = (ma_uint32) n_channels;
    cfg.sampleRate        = (ma_uint32) sample_rate;
    cfg.dataCallback      = impl::on_playback;
    cfg.pUserData         = p.get();
    if (ma_device_init(nullptr, &cfg, &p->device) != MA_SUCCESS) {
        LOG_ERR("%s: failed to open the playback device\n", __func__);
        ma_pcm_rb_uninit(&p->rb);
        return false;
    }
    p->has_device = true;
    if (ma_device_start(&p->device) != MA_SUCCESS) {
        LOG_ERR("%s: failed to start the playback device\n", __func__);
        return false;
    }
    LOG_INF("%s: playing on '%s'\n", __func__, p->device.playback.name);
    return true;
}

bool mtmd_cli_audio_out::init_file(const std::string & fname, int sample_rate, int n_channels) {
    p->n_channels = n_channels;
    const ma_encoder_config cfg = ma_encoder_config_init(ma_encoding_format_wav, ma_format_s16, (ma_uint32) n_channels, (ma_uint32) sample_rate);
    if (ma_encoder_init_file(fname.c_str(), &cfg, &p->encoder) != MA_SUCCESS) {
        LOG_ERR("%s: failed to open %s\n", __func__, fname.c_str());
        return false;
    }
    p->has_encoder = true;
    return true;
}

void mtmd_cli_audio_out::write(const float * samples, size_t n_samples) {
    size_t n_frames = n_samples / p->n_channels;
    if (p->has_encoder) {
        std::vector<int16_t> pcm(n_frames * p->n_channels);
        ma_pcm_f32_to_s16(pcm.data(), samples, pcm.size(), ma_dither_mode_none);
        ma_encoder_write_pcm_frames(&p->encoder, pcm.data(), n_frames, nullptr);
        return;
    }
    while (p->has_device && n_frames > 0) {
        ma_uint32 n = (ma_uint32) std::min<size_t>(n_frames, UINT32_MAX);
        void *    dst;
        if (ma_pcm_rb_acquire_write(&p->rb, &n, &dst) != MA_SUCCESS) {
            return;
        }
        if (n == 0) {
            sleep_ms(5);
            continue;
        }
        memcpy(dst, samples, n * p->n_channels * sizeof(float));
        ma_pcm_rb_commit_write(&p->rb, n);
        samples  += n * p->n_channels;
        n_frames -= n;
    }
}

bool mtmd_cli_audio_out::drain(const volatile bool & stop) {
    if (!p->has_device) {
        return !stop;
    }
    bool stopped = false;
    while (ma_pcm_rb_available_read(&p->rb) > 0) {
        if (stop && !stopped) {
            stopped = true;
            p->drop = true;
        }
        sleep_ms(5);
    }
    p->drop = false;
    return !stopped;
}

//
// mtmd_cli_vad
//

static constexpr int   vad_frame_ms    = 20;
static constexpr int   vad_start_ms    = 100;    // loud for this long: the speech starts
static constexpr int   vad_pre_roll_ms = 300;    // sound kept from before the start
static constexpr float vad_ratio       = 3.0f;   // speech is this many times louder than the background
static constexpr float vad_level_min   = 0.005f; // RMS below this is never speech

mtmd_cli_vad::mtmd_cli_vad(int sample_rate, int silence_ms) {
    n_frame   = sample_rate * vad_frame_ms / 1000;
    n_silence = std::max(1, silence_ms / vad_frame_ms);
}

void mtmd_cli_vad::reset() {
    speaking = false;
    ended    = false;
    n_loud   = 0;
    n_quiet  = 0;
    frame.clear();
    pre_roll.clear();
}

bool mtmd_cli_vad::turn_ended() {
    const bool res = ended;
    ended = false;
    return res;
}

std::vector<float> mtmd_cli_vad::process(const float * samples, size_t n_samples) {
    std::vector<float> out;
    frame.insert(frame.end(), samples, samples + n_samples);

    const size_t n_pre_roll = (size_t) n_frame * vad_pre_roll_ms / vad_frame_ms;
    size_t off = 0;
    for (; off + n_frame <= frame.size(); off += n_frame) {
        const float * f = frame.data() + off;
        float sum = 0.0f;
        for (int i = 0; i < n_frame; i++) {
            sum += f[i] * f[i];
        }
        const float rms  = std::sqrt(sum / n_frame);
        const bool  loud = rms > std::max(vad_level_min, noise * vad_ratio);

        if (speaking) {
            out.insert(out.end(), f, f + n_frame);
            n_quiet = loud ? 0 : n_quiet + 1;
            if (n_quiet >= n_silence) {
                speaking = false;
                ended    = true;
                n_loud   = 0;
            }
            continue;
        }

        // follow the background: down at once, up slowly
        noise = rms < noise || noise == 0.0f ? rms : noise * 0.95f + rms * 0.05f;

        pre_roll.insert(pre_roll.end(), f, f + n_frame);
        if (pre_roll.size() > n_pre_roll) {
            pre_roll.erase(pre_roll.begin(), pre_roll.end() - n_pre_roll);
        }
        n_loud = loud ? n_loud + 1 : 0;
        if (n_loud * vad_frame_ms >= vad_start_ms) {
            speaking = true;
            n_quiet  = 0;
            out.insert(out.end(), pre_roll.begin(), pre_roll.end());
            pre_roll.clear();
        }
    }
    frame.erase(frame.begin(), frame.begin() + off);
    return out;
}
