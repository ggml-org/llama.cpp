#pragma once

#include "readline/buffer.h"
#include "readline/history.h"
#include "readline/terminal.h"
#include "readline/errors.h"
#include <memory>
#include <string>

namespace readline {

class Readline {
public:
    explicit Readline(const Prompt& prompt);
    ~Readline() = default;

    std::string readline();
    void history_enable() { history_->enabled = true; }
    void history_disable() { history_->enabled = false; }
    bool check_interrupt();

    History* history() { return history_.get(); }
    Terminal* terminal() { return terminal_.get(); }
    bool is_pasting() const { return pasting_; }

private:
    void history_prev(Buffer* buf, std::u32string& current_line_buf);
    void history_next(Buffer* buf, std::u32string& current_line_buf);
    std::u32string utf8_to_utf32(const std::string& str);
    std::string utf32_to_utf8(const std::u32string& str);

    Prompt prompt_;
    std::unique_ptr<Terminal> terminal_;
    std::unique_ptr<History> history_;
    bool pasting_ = false;
};

} // namespace readline
