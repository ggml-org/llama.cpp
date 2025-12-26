#pragma once

#include <exception>
#include <string>

namespace readline {

class interrupt_error : public std::exception {
public:
    interrupt_error() = default;

    const char* what() const noexcept override {
        return "Interrupt";
    }
};

class eof_error : public std::exception {
public:
    eof_error() = default;

    const char* what() const noexcept override {
        return "EOF";
    }
};

} // namespace readline
