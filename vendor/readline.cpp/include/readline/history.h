#pragma once

#include <string>
#include <vector>
#include <filesystem>

namespace readline {

class History {
public:
    History();
    ~History() = default;

    void init();
    void add(const std::string& line);
    void compact();
    void clear();
    std::string prev();
    std::string next();
    size_t size() const { return buffer_.size(); }
    void save();

    bool enabled = true;
    bool autosave = true;
    size_t pos = 0;
    size_t limit = 100;

private:
    std::vector<std::string> buffer_;
    std::filesystem::path filename_;
};

} // namespace readline
