#include "test-log.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace {

// a log is printed after the row of its model, what does not fit stays in the file
const size_t LOG_DUMP_MAX_BYTES = 64 * 1024;

// one worker thread runs one model and the log callback fires on the thread that logged the
//   message, so the file of a model is held by the thread that runs it
thread_local FILE * t_file;

char level_char(enum ggml_log_level level) {
    switch (level) {
        case GGML_LOG_LEVEL_DEBUG: return 'D';
        case GGML_LOG_LEVEL_INFO:  return 'I';
        case GGML_LOG_LEVEL_WARN:  return 'W';
        case GGML_LOG_LEVEL_ERROR: return 'E';
        case GGML_LOG_LEVEL_CONT:  return 'C';
        default:                   return '?';
    }
}

// every line of a model log carries its level, so that a failure can be looked up with a grep
void test_log_level(enum ggml_log_level level) {
    if (level != GGML_LOG_LEVEL_NONE) {
        fprintf(t_file, "%c ", level_char(level));
    }
}

} // namespace

void test_log_capture(const std::string & path) {
    if (t_file) {
        fclose(t_file);
        t_file = nullptr;
    }

    if (path.empty()) {
        return;
    }

    t_file = fopen(path.c_str(), "w");

    if (!t_file) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, path.c_str());
    }
}

void test_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    if (!t_file) {
        common_log_default_callback(level, text, user_data);
        return;
    }

    // gate the capture the way the console would be - the debug chatter of the graph builder is
    //   orders of magnitude more than everything else, a capture never holds less than info
    if (common_log_get_verbosity(level) <= std::max(common_log_get_verbosity_thold(), LOG_LEVEL_INFO)) {
        test_log_level(level);
        fputs(text, t_file);
    }
}

void test_log_add(enum ggml_log_level level, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);

    if (t_file) {
        test_log_level(level);
        vfprintf(t_file, fmt, args);
    } else if (level == GGML_LOG_LEVEL_ERROR) {
        // without a capture the run is followed on the console, a detail line has no place there
        vfprintf(stderr, fmt, args);
        fflush(stderr);
    }

    va_end(args);
}

bool test_log_dump(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "failed to read the captured log '%s'\n", path.c_str());
        return false;
    }

    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();

    size_t skip = 0;
    if (text.size() > LOG_DUMP_MAX_BYTES) {
        skip = text.find('\n', text.size() - LOG_DUMP_MAX_BYTES);
        skip = skip == std::string::npos ? 0 : skip + 1;
    }

    if (skip > 0) {
        const size_t n_all   = std::count(text.begin(), text.end(), '\n');
        const size_t n_shown = std::count(text.begin() + skip, text.end(), '\n');

        fprintf(stdout, "... %zu of %zu lines omitted, full log: %s\n", n_all - n_shown, n_all, path.c_str());
    }

    fwrite(text.data() + skip, 1, text.size() - skip, stdout);
    fflush(stdout);

    return skip > 0;
}

std::string test_log_path(const std::string & logs_dir, const std::string & model_name) {
    const std::string file = "test-llama-context." + model_name + ".log";

    return logs_dir.empty() ? file : (std::filesystem::path(logs_dir) / file).string();
}
