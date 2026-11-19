#pragma once

#include "log.h"

#include <string>

//
// the log of a model goes to a file of its own, so that a model which failed can be read on its
// own. a worker thread runs one model and the log callback fires on the thread that produced the
// message, so a capture belongs to the thread that opened it
//

// open the log of this thread, an empty path closes it - a capture holds what the verbosity of
//   the process allows, never less than info, the details of a failure are looked for in the file
void test_log_capture(const std::string & path);

// llama, ggml and the backends log through this: into the file of this thread when it captures,
//   into the default callback otherwise
void test_log_callback(enum ggml_log_level level, const char * text, void * user_data);

// the lines of the test itself go to the file of this thread, an error goes to stderr as well when
//   there is no capture and the run is followed on the console
void test_log_add(enum ggml_log_level level, const char * fmt, ...) LOG_ATTRIBUTE_FORMAT(2, 3);

// print a captured log, false when it was printed in full and the file can be removed
bool test_log_dump(const std::string & path);

std::string test_log_path(const std::string & logs_dir, const std::string & model_name);

// a trace is a detail of a case, it is there to read in the file of the model
#define TLOG_ERR(...) test_log_add(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define TLOG_INF(...) test_log_add(GGML_LOG_LEVEL_INFO,  __VA_ARGS__)
#define TLOG_TRC(...) test_log_add(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
