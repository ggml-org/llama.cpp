#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// audio devices of llama-mtmd-cli, they are not part of libmtmd

// mono F32 PCM input: the default capture device, or samples given by the caller (an audio file)
struct mtmd_cli_audio_in {
    mtmd_cli_audio_in();
    ~mtmd_cli_audio_in();

    bool init_mic(int sample_rate);

    // realtime: the samples become available at the pace of a live input
    // otherwise read() gives n_chunk samples per call
    void init_samples(const float * samples, size_t n_samples, int sample_rate, bool realtime, size_t n_chunk);

    // appends the samples available now to out, returns their count (0: none yet, or the end)
    size_t read(std::vector<float> & out);

    // drops the samples not read yet
    void discard();

    bool is_end() const;

    // samples lost because read() was not called in time
    int64_t n_dropped() const;

private:
    struct impl;
    std::unique_ptr<impl> p;
};

// interleaved F32 PCM output: the default playback device, or a WAV file
struct mtmd_cli_audio_out {
    mtmd_cli_audio_out();
    ~mtmd_cli_audio_out();

    bool init_speaker(int sample_rate, int n_channels);
    bool init_file(const std::string & fname, int sample_rate, int n_channels);

    // n_samples is the total over all channels; with a speaker, blocks while its buffer is full
    void write(const float * samples, size_t n_samples);

    // waits until the speaker has played everything, returns false if stop becomes true first (the rest is dropped)
    bool drain(const volatile bool & stop);

private:
    struct impl;
    std::unique_ptr<impl> p;
};

// energy-based detection of the end of a turn, fed with mono PCM
struct mtmd_cli_vad {
    mtmd_cli_vad(int sample_rate, int silence_ms);

    // returns the samples that belong to the turn (none before the speech starts)
    // the sound right before the speech is kept, so that its start is not cut
    std::vector<float> process(const float * samples, size_t n_samples);

    bool in_speech() const { return speaking; }

    // true once: the speech stopped for silence_ms
    bool turn_ended();

    void reset();

private:
    int    n_frame;      // samples per analysis frame
    int    n_silence;    // frames of silence that end a turn
    float  noise = 0.0f; // running level of the background
    bool   speaking  = false;
    bool   ended     = false;
    int    n_loud    = 0;
    int    n_quiet   = 0;
    std::vector<float> frame;
    std::vector<float> pre_roll;
};
