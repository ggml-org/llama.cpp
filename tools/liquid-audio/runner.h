#pragma once

#include "common.h"

#include <functional>
#include <string>

namespace liquid {
namespace audio {

using generated_audio_t = std::vector<float>;
using text_callback_t   = std::function<void(const std::string &)>;
using audio_callback_t  = std::function<void(const std::vector<float> &)>;

class Runner {
  public:
    struct Message {
        std::string            role;
        std::string            content;
        std::vector<std::byte> wav;
    };

    Runner();
    ~Runner();

    void reset();

    int  init(common_params params);
    void stop();
    int  generate(const std::vector<Message> & messages,
                  int                          n_predict,
                  const text_callback_t &      text_callback,
                  const audio_callback_t &     audio_callback);

    int          get_output_sample_rate() const;
    const char * get_last_error() const;
  private:
    class RunnerImpl;
    std::unique_ptr<RunnerImpl> impl_;
};

}  // namespace audio
}  // namespace liquid
