# readline.cpp

A readline implementation providing an interactive line editing interface with history support.

## Features

- Interactive line editing
- Command history with navigation (up/down arrows)
- Word-based navigation (Alt+B/Alt+F)
- Line editing commands (Ctrl+A, Ctrl+E, Ctrl+K, etc.)
- Bracket paste support
- Customizable prompts
- History persistence

## Building

readline.cpp uses CMake. To build:

```bash
# Create build directory
mkdir build
cd build

# Configure
cmake ..

# Build
cmake --build .

# Run the example
./simple_example
```

## Requirements

- C++17 compiler (GCC 7+, Clang 5+, or MSVC 2017+)
- CMake 3.14 or higher
- OSes: Linux, macOS, Windows (open to others)

## Using the Library

```cpp
#include "readline/readline.h"
#include "readline/errors.h"
#include <iostream>

int main() {
    readline::Prompt prompt;
    prompt.prompt = "> ";
    prompt.alt_prompt = ". ";
    prompt.placeholder = "Enter a command";

    try {
        readline::Readline rl(prompt);
        rl.history_enable();

        while (true) {
            try {
                std::string line = rl.readline();
                std::cout << "You entered: " << line << "\n";
            } catch (const readline::eof_error&) {
                break;
            } catch (const readline::interrupt_error&) {
                std::cout << "^C\n";
                continue;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
```

