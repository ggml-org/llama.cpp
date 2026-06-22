//
// Created by Morgan Funtowicz on 6/19/2026.
//
#include <server-process.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#include <locale>
#include <numeric>

static std::wstring utf8_to_wide(const std::string & s) {
    if (s.empty()) return {};
    const int len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), nullptr, 0);
    if (len == 0) return {};
    std::wstring ws(len, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), ws.data(), len);
    return ws;
}

static std::wstring build_env_block(const std::vector<std::string> & env) {
    const size_t total = 1 + std::transform_reduce(
        env.cbegin(),
        env.cend(),
        static_cast<size_t>(0),
        std::plus<size_t>{},
        [](const std::string & s) { return s.size() + 1; }
    );

    std::wstring block;
    block.reserve(total);

    for (const auto & e : env) {
        block += utf8_to_wide(e);
        block += L'\0';
    }
    block += L'\0'; // double null terminator
    return block;
}

// Build a Windows command-line wstring from argv, escaping as needed.
static std::wstring escape_cmdline(const std::vector<std::string> & args) {
    const size_t total = args.size() + std::transform_reduce(
        args.cbegin(),
        args.cend(),
        static_cast<size_t>(0),
        std::plus<size_t>{},
        [](const std::string & s) { return s.size() * 2 + 2; }
    );

    std::wstring result;
    result.reserve(total);

    for (size_t i = 0; i < args.size(); i++) {
        if (i > 0) result += L' ';

        const std::wstring arg = utf8_to_wide(args[i]);
        if (const bool needs_quote = arg.empty() || arg.find_first_of(L" \t\"") != std::wstring::npos; !needs_quote) {
            result += arg;
            continue;
        }

        result += L'"';
        for (size_t j = 0; j < arg.size(); j++) {
            unsigned num_backslash = 0;
            while (j < arg.size() && arg[j] == L'\\') {
                num_backslash++;
                j++;
            }
            if (j == arg.size()) {
                result.append(num_backslash * 2, L'\\');
                break;
            }

            if (arg[j] == L'"') {
                result.append(num_backslash * 2 + 1, L'\\');
                result += L'"';
            } else {
                result.append(num_backslash, L'\\');
                result += arg[j];
            }
        }
        result += L'"';
    }
    return result;
}

FILE * server_process::get_stdin() const { return fStdin; }
FILE * server_process::get_stdout() const { return fStdout; }

bool server_process::is_alive() const {
    if (!hHandle) return false;

    DWORD exit_code = 0;
    if (!GetExitCodeProcess(hHandle, &exit_code)) return false;

    return exit_code == STILL_ACTIVE;
}

void server_process::terminate() const {
    if (hHandle) TerminateProcess(hHandle, 0);
}

int server_process::join() {
    if (!hHandle) return -1;

    DWORD exit_code = 0;

    WaitForSingleObject(hHandle, INFINITE);
    GetExitCodeProcess(hHandle, &exit_code);
    CloseHandle(hHandle);
    hHandle = nullptr;

    if (fStdin)  { fclose(fStdin);  fStdin  = nullptr; }
    if (fStdout) { fclose(fStdout); fStdout = nullptr; }

    return static_cast<int>(exit_code);
}

int server_process::run(const std::vector<std::string> & args, const std::vector<std::string> & env) {
    if (hHandle) return -1; // already running

    std::wstring wcmdline = escape_cmdline(args);

    // Use argv[0] as the application name if it contains a path separator,
    // otherwise pass NULL to let CreateProcessW search via PATH.
    std::wstring wexe;
    if (!args.empty()) {
        if (args[0].find('/') != std::string::npos || args[0].find('\\') != std::string::npos) {
            wexe = utf8_to_wide(args[0]);
        }
    }

    // Create pipes for stdin/stdout
    SECURITY_ATTRIBUTES sa = {sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};

    HANDLE stdin_read, stdin_write;
    if (!CreatePipe(&stdin_read, &stdin_write, &sa, 0)) return -1;
    SetHandleInformation(stdin_write, HANDLE_FLAG_INHERIT, 0);

    HANDLE stdout_read, stdout_write;
    if (!CreatePipe(&stdout_read, &stdout_write, &sa, 0)) {
        CloseHandle(stdin_read);
        CloseHandle(stdin_write);
        return -1;
    }
    SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si = {};
    si.cb = sizeof(si);
    si.dwFlags   = STARTF_USESTDHANDLES;
    si.hStdInput  = stdin_read;
    si.hStdOutput = stdout_write;
    si.hStdError  = stdout_write;

    PROCESS_INFORMATION pi = {nullptr};

    std::wstring env_block;
    LPWSTR env_ptr = nullptr;
    if (!env.empty()) {
        env_block = build_env_block(env);
        env_ptr = env_block.data();
    }

    constexpr DWORD flags = CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT;
    const auto ok = CreateProcessW(
        wexe.empty() ? nullptr : wexe.c_str(),
        wcmdline.data(),
        nullptr, nullptr,
        TRUE, // inherit handles
        flags,
        env_ptr,
        nullptr,
        &si, &pi
    );

    // Close the child-side handles (inherited by the child)
    CloseHandle(stdin_read);
    CloseHandle(stdout_write);

    if (!ok) {
        CloseHandle(stdin_write);
        CloseHandle(stdout_read);
        return -1;
    }

    // Close the thread handle immediately — we don't need it
    CloseHandle(pi.hThread);

    // Open C FILEs for the parent-side handles
    fStdin  = _fdopen(_open_osfhandle(reinterpret_cast<intptr_t>(stdin_write),  _O_WRONLY | _O_TEXT), "w");
    if (!fStdin) {
        CloseHandle(stdin_write);
        CloseHandle(stdout_read);
        TerminateProcess(pi.hProcess, 1);
        WaitForSingleObject(pi.hProcess, INFINITE);
        CloseHandle(pi.hProcess);
        return -1;
    }

    fStdout = _fdopen(_open_osfhandle(reinterpret_cast<intptr_t>(stdout_read),  _O_RDONLY | _O_TEXT), "r");
    if (!fStdout) {
        fclose(fStdin);
        fStdin = nullptr;
        CloseHandle(stdout_read);
        TerminateProcess(pi.hProcess, 1);
        WaitForSingleObject(pi.hProcess, INFINITE);
        CloseHandle(pi.hProcess);
        return -1;
    }

    hHandle = pi.hProcess;
    return 0;
}

#else
#endif

server_process::server_process(server_process && o) noexcept
    : hHandle(o.hHandle), fStdin(o.fStdin), fStdout(o.fStdout)
{
    o.hHandle = nullptr;
    o.fStdin  = nullptr;
    o.fStdout = nullptr;
}

server_process & server_process::operator=(server_process && o) noexcept {
    if (this != &o) {
        terminate();
        join();
        hHandle  = o.hHandle;
        fStdin  = o.fStdin;
        fStdout = o.fStdout;
        o.hHandle  = nullptr;
        o.fStdin  = nullptr;
        o.fStdout = nullptr;
    }
    return *this;
}

server_process::~server_process() {
    terminate();
    join();
}
