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
    AudioPlayback(int sample_rate) : sample_rate_(sample_rate) {
        buffer_.resize(max_capacity_);
    }
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

        size_t samples_added = 0;
        while (samples_added < samples.size() && running_) {
            // Wait until queue is small enough for backpressure (bounded queue max 2 sec)
            cv_.wait(lock, [this]() { return !running_ || count_ < max_capacity_; });

            if (!running_) break;

            size_t available = max_capacity_ - count_;
            size_t to_write = std::min(available, samples.size() - samples_added);

            // Write in two parts if it wraps around
            size_t space_until_end = max_capacity_ - tail_;
            size_t first_write = std::min(to_write, space_until_end);

            std::copy(samples.begin() + samples_added, samples.begin() + samples_added + first_write, buffer_.begin() + tail_);

            if (first_write < to_write) {
                std::copy(samples.begin() + samples_added + first_write, samples.begin() + samples_added + to_write, buffer_.begin());
            }

            tail_ = (tail_ + to_write) % max_capacity_;
            count_ += to_write;
            samples_added += to_write;
        }
    }

    void wait_until_done() {
        while (running_) {
            bool done = false;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (count_ == 0) {
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
        if (player->count_ < samples_to_read) {
            samples_to_read = player->count_;
        }

        if (samples_to_read > 0) {
            size_t space_until_end = player->max_capacity_ - player->head_;
            size_t first_read = std::min(samples_to_read, space_until_end);

            std::copy(player->buffer_.begin() + player->head_, player->buffer_.begin() + player->head_ + first_read, out);

            if (first_read < samples_to_read) {
                std::copy(player->buffer_.begin(), player->buffer_.begin() + (samples_to_read - first_read), out + first_read);
            }

            player->head_ = (player->head_ + samples_to_read) % player->max_capacity_;
            player->count_ -= samples_to_read;
        }

        // Fill remainder with silence
        if (samples_to_read < frameCount) {
            std::fill(out + samples_to_read, out + frameCount, 0);
        }

        // Notify producer that space is available
        player->cv_.notify_all();

        (void)pInput; // Unused
    }

    int sample_rate_;
    ma_device device_;
    std::vector<int16_t> buffer_;
    size_t head_{0};
    size_t tail_{0};
    size_t count_{0};
    size_t max_capacity_{32000}; // 2 seconds at 16kHz
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> running_{false};
};
