#pragma once

#include "miniaudio/miniaudio.h"
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <thread>
#include <chrono>

class AudioPlayback {
public:
    AudioPlayback(int sample_rate) : sample_rate_(sample_rate) {}
    ~AudioPlayback() { stop(); }

    bool start() {
        if (running_) return true;

        ma_device_config config = ma_device_config_init(ma_device_type_playback);
        config.playback.format = ma_format_s16;
        config.playback.channels = 1;
        config.sampleRate = sample_rate_;
        config.dataCallback = data_callback;
        config.pUserData = this;
        // set small chunk size
        config.periodSizeInFrames = 1024;

        if (ma_device_init(NULL, &config, &device_) != MA_SUCCESS) {
            return false;
        }

        if (ma_device_start(&device_) != MA_SUCCESS) {
            ma_device_uninit(&device_);
            return false;
        }

        running_ = true;
        return true;
    }

    void stop() {
        if (running_) {
            running_ = false;
            cv_.notify_all();
            ma_device_uninit(&device_);
        }
    }

    void add_samples(const std::vector<int16_t>& samples) {
        std::unique_lock<std::mutex> lock(mutex_);

        // Wait until queue is small enough for backpressure (bounded queue max 100 chunks ~ 1 sec)
        // 16000 sample rate / 1024 frames = ~15 chunks per second
        cv_.wait(lock, [this]() { return !running_ || buffer_.size() < 16000 * 2; });

        if (!running_) return;

        buffer_.insert(buffer_.end(), samples.begin(), samples.end());
    }

    void wait_until_done() {
        while (running_) {
            bool done = false;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (buffer_.empty()) {
                    done = true;
                }
            }
            if (done) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

private:
    static void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
        AudioPlayback* player = static_cast<AudioPlayback*>(pDevice->pUserData);
        int16_t* out = static_cast<int16_t*>(pOutput);

        std::lock_guard<std::mutex> lock(player->mutex_);

        size_t samples_to_read = frameCount;
        if (player->buffer_.size() < samples_to_read) {
            samples_to_read = player->buffer_.size();
        }

        if (samples_to_read > 0) {
            std::copy(player->buffer_.begin(), player->buffer_.begin() + samples_to_read, out);
            player->buffer_.erase(player->buffer_.begin(), player->buffer_.begin() + samples_to_read);
        }

        // Fill remainder with silence
        if (samples_to_read < frameCount) {
            std::fill(out + samples_to_read, out + frameCount, 0);
        }

        // Notify producer that space is available
        player->cv_.notify_one();

        (void)pInput; // Unused
    }

    int sample_rate_;
    ma_device device_;
    std::vector<int16_t> buffer_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> running_{false};
};
