#include "readline/history.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>

namespace readline {

History::History() {
    init();
}

void History::init() {
#ifdef _WIN32
    const char* env_var = "USERPROFILE";
#else
    const char* env_var = "HOME";
#endif

    const char* home = std::getenv(env_var);
    if (!home) {
        return;
    }

    std::filesystem::path history_dir = std::filesystem::path(home) / ".readline";
    filename_ = history_dir / "history";

    // Create directory if it doesn't exist
    if (!std::filesystem::exists(history_dir)) {
        std::filesystem::create_directories(history_dir);
    }

    // Read existing history file
    std::ifstream file(filename_);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\n\r"));
            line.erase(line.find_last_not_of(" \t\n\r") + 1);

            if (!line.empty()) {
                add(line);
            }
        }
        file.close();
    }
}

void History::add(const std::string& line) {
    buffer_.push_back(line);
    compact();
    pos = size();
    if (autosave) {
        save();
    }
}

void History::compact() {
    while (buffer_.size() > limit) {
        buffer_.erase(buffer_.begin());
    }
}

void History::clear() {
    buffer_.clear();
}

std::string History::prev() {
    if (pos > 0) {
        pos--;
    }
    if (pos < buffer_.size()) {
        return buffer_[pos];
    }
    return "";
}

std::string History::next() {
    if (pos < buffer_.size()) {
        pos++;
        if (pos < buffer_.size()) {
            return buffer_[pos];
        }
    }
    return "";
}

void History::save() {
    if (!enabled) {
        return;
    }

    std::filesystem::path tmp_file = filename_;
    tmp_file += ".tmp";

    std::ofstream file(tmp_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open history file for writing");
    }

    for (const auto& line : buffer_) {
        file << line << '\n';
    }
    file.close();

    // Atomic rename
    std::filesystem::rename(tmp_file, filename_);
}

} // namespace readline
