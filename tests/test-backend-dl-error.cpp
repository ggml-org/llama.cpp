#include "ggml-backend.h"
#include "ggml-backend-dl.h"
#include "ggml.h"

#include <cstdio>
#include <string>
#include <thread>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: %s failed\n", __func__, __LINE__, #condition); \
        return false; \
    } \
} while (0)

static bool has_error(const std::string & text, DWORD code) {
    const std::string prefix = "Windows error " + std::to_string(code);
    const size_t pos = text.find(prefix);
    if (pos == std::string::npos) {
        return false;
    }
    const size_t end = pos + prefix.size();
    return end == text.size() || text[end] == ':' || text[end] == '\n';
}

static bool test_state(const fs::path & missing, const fs::path & system_dll, HMODULE module, DWORD load_code, DWORD symbol_code) {
    CHECK(dl_load_library(missing) == nullptr);
    SetLastError(ERROR_ACCESS_DENIED);
    const char * message = dl_error();
    const std::string saved = message;
    CHECK(has_error(saved, load_code));
    CHECK(saved.find('\r') == std::string::npos && saved.find('\n') == std::string::npos);
    CHECK(MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, saved.c_str(), -1, nullptr, 0) > 0);
    CHECK(dl_get_sym(module, "GetLastError") != nullptr);
    CHECK(saved == message);
    CHECK(std::string(dl_error()).empty());

    CHECK(dl_get_sym(module, "ggml_backend_init") == nullptr);
    SetLastError(ERROR_ACCESS_DENIED);
    CHECK(has_error(dl_error(), symbol_code));
    CHECK(std::string(dl_error()).empty());

    CHECK(dl_load_library(missing) == nullptr);
    CHECK(dl_get_sym(module, "ggml_backend_init") == nullptr);
    CHECK(has_error(dl_error(), symbol_code));
    CHECK(dl_get_sym(module, "ggml_backend_init") == nullptr);
    CHECK(dl_load_library(missing) == nullptr);
    CHECK(has_error(dl_error(), load_code));

    CHECK(dl_load_library(missing) == nullptr);
    dl_handle_ptr loaded { dl_load_library(system_dll) };
    CHECK(loaded != nullptr);
    CHECK(std::string(dl_error()).empty());
    CHECK(dl_get_sym(module, "ggml_backend_init") == nullptr);
    CHECK(dl_get_sym(module, "GetLastError") != nullptr);
    CHECK(std::string(dl_error()).empty());
    return true;
}

static bool test_threads(const fs::path & missing, HMODULE module, DWORD load_code, DWORD symbol_code) {
    CHECK(dl_load_library(missing) == nullptr);
    bool worker_ok = false;
    std::thread worker([&] {
        worker_ok = dl_get_sym(module, "ggml_backend_init") == nullptr;
        SetLastError(ERROR_ACCESS_DENIED);
        worker_ok = worker_ok && has_error(dl_error(), symbol_code);
    });
    worker.join();
    CHECK(worker_ok);
    const char * message = dl_error();
    const std::string saved = message;
    CHECK(has_error(saved, load_code));

    std::thread reader([&] {
        dl_get_sym(module, "ggml_backend_init");
        dl_error();
    });
    reader.join();
    CHECK(saved == message);
    CHECK(std::string(dl_error()).empty());
    return true;
}

struct test_log {
    std::string text;
    size_t errors = 0;
};

static void log_callback(ggml_log_level level, const char * text, void * user_data) {
    auto & log = *static_cast<test_log *>(user_data);
    log.text += text;
    log.errors += level == GGML_LOG_LEVEL_ERROR;
    fputs(text, stderr);
}

static bool test_backend(const fs::path & missing, const fs::path & system_dll, const fs::path & directory,
                         DWORD load_code, DWORD symbol_code, test_log & log) {
    log = {};
    // The explicit loader uses native narrow paths on Windows.
    CHECK(ggml_backend_load(missing.string().c_str()) == nullptr);
    CHECK(log.text.find(missing.u8string()) != std::string::npos);
    CHECK(has_error(log.text, load_code));

    log = {};
    CHECK(ggml_backend_load(system_dll.string().c_str()) == nullptr);
    CHECK(log.text.find("failed to find ggml_backend_init") != std::string::npos);
    CHECK(has_error(log.text, symbol_code));

    const fs::path fixture = directory / GGML_TEST_DL_ERROR_FIXTURE;
    dl_handle_ptr handle { dl_load_library(fixture) };
    CHECK(handle != nullptr);
    CHECK(dl_get_sym(handle.get(), "ggml_backend_score") == nullptr);
    CHECK(dl_get_sym(handle.get(), "ggml_backend_init") != nullptr);
    CHECK(std::string(dl_error()).empty());
    log = {};
    ggml_backend_reg_t reg = ggml_backend_load(fixture.string().c_str());
    CHECK(reg != nullptr);
    CHECK(ggml_backend_reg_by_name("DL_TEST") == reg);
    CHECK(log.errors == 0);
    ggml_backend_unload(reg);

#ifdef GGML_TEST_CPU
    ggml_backend_load_all();
    reg = ggml_backend_reg_by_name("CPU");
    CHECK(reg != nullptr);
    CHECK(ggml_backend_reg_dev_count(reg) > 0);
#endif
    return true;
}

int main() {
    wchar_t path[32768];
    const DWORD length = GetModuleFileNameW(nullptr, path, sizeof(path) / sizeof(path[0]));
    if (length == 0 || length == sizeof(path) / sizeof(path[0])) {
        return 1;
    }
    const fs::path directory = fs::path(path).parent_path();
    const fs::path missing = directory / (L"missing \u6d4b\u8bd5 " + std::to_wstring(GetCurrentProcessId()) + L".dll");
    const fs::path missing_api = directory / ("missing backend " + std::to_string(GetCurrentProcessId()) + ".dll");
    const UINT system_length = GetSystemDirectoryW(path, sizeof(path) / sizeof(path[0]));
    if (system_length == 0 || system_length >= sizeof(path) / sizeof(path[0]) || fs::exists(missing) || fs::exists(missing_api)) {
        return 1;
    }
    const fs::path system_dll = fs::path(path) / "kernel32.dll";
    const UINT old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    dl_handle_ptr module { LoadLibraryW(system_dll.c_str()) };
    if (!module || LoadLibraryW(missing.c_str()) != nullptr) {
        return 1;
    }
    const DWORD load_code = GetLastError();
    if (GetProcAddress(module.get(), "ggml_backend_init") != nullptr) {
        return 1;
    }
    const DWORD symbol_code = GetLastError();
    printf("Native errors: load=%lu symbol=%lu\n", load_code, symbol_code);

    ggml_log_callback old_callback;
    void * old_user_data;
    ggml_log_get(&old_callback, &old_user_data);
    test_log log;
    ggml_log_set(log_callback, &log);
    bool ok = test_state(missing, system_dll, module.get(), load_code, symbol_code);
    ok = test_threads(missing, module.get(), load_code, symbol_code) && ok;
    ok = test_backend(missing_api, system_dll, directory, load_code, symbol_code, log) && ok;
    ggml_log_set(old_callback, old_user_data);
    SetErrorMode(old_mode);
    printf("Windows loader diagnostics: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
