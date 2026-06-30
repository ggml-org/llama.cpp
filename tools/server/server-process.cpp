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

#else

#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>
#include <cerrno>
#include <csignal>
#include <string>
#include <vector>

extern char ** environ;

bool server_process::is_alive() const {
    if (pid <= 0 || joined) return false;

    int wstatus = 0;
    if (const auto r = waitpid(pid, &wstatus, WNOHANG); r == 0) {
        return true; // still running
    }
    return false;
}

int server_process::join() {
    if (pid > 0 && !joined) {
        int wstatus = 0;
        if (waitpid(pid, &wstatus, 0) > 0) {
            status = WIFEXITED(wstatus) ? WEXITSTATUS(wstatus) : -1;
            joined = true;
            pid    = 0;
        }
    }

    close_stdin();
    close_stdout();

    return joined ? status : -1;
}

void server_process::terminate() const {
    if (pid > 0) {
        kill(pid, SIGKILL);
    }
}

int server_process::run(const std::vector<std::string> & args, const std::vector<std::string> & env) {
    if (pid > 0) return -1; // already running

    int stdin_pipe[2]  = {-1, -1};
    int stdout_pipe[2] = {-1, -1};

    if (pipe(stdin_pipe) != 0) return -1;
    if (pipe(stdout_pipe) != 0) {
        close(stdin_pipe[0]);
        close(stdin_pipe[1]);
        return -1;
    }

    std::vector<const char *> argv;
    argv.reserve(args.size() + 1);
    for (const auto & a : args) argv.push_back(a.c_str());
    argv.push_back(nullptr);

    std::vector<const char *> envp;
    if (!env.empty()) {
        envp.reserve(env.size() + 1);
        for (const auto & e : env) envp.push_back(e.c_str());
        envp.push_back(nullptr);
    }

    const auto child = fork();
    if (child < 0) {
        close(stdin_pipe[0]);
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        return -1;
    }

    if (child == 0) {
        // child
        dup2(stdin_pipe[0],  STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stdout_pipe[1], STDERR_FILENO);

        close(stdin_pipe[0]);
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);

        execve(args.empty() ? nullptr : args[0].c_str(),
               const_cast<char * const *>(argv.data()),
               envp.empty() ? environ : const_cast<char * const *>(envp.data()));
        _exit(127);
    }

    // parent - close child-side ends
    close(stdin_pipe[0]);
    close(stdout_pipe[1]);

    fStdin = fdopen(stdin_pipe[1], "w");
    if (!fStdin) {
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);
        kill(child, SIGKILL);
        waitpid(child, nullptr, 0);
        return -1;
    }

    fStdout = fdopen(stdout_pipe[0], "r");
    if (!fStdout) {
        fclose(fStdin);
        fStdin = nullptr;
        close(stdout_pipe[0]);
        kill(child, SIGKILL);
        waitpid(child, nullptr, 0);
        return -1;
    }

    pid    = child;
    status = 0;
    joined = false;
    return 0;
}

server_process::server_process(server_process && o) noexcept
    : pid(o.pid), fStdin(o.fStdin), fStdout(o.fStdout)
{
    o.pid = -1;
    o.fStdin  = nullptr;
    o.fStdout = nullptr;
}

server_process & server_process::operator=(server_process && o) noexcept {
    if (this != &o) {
        terminate();
        join();
        pid  = o.pid;
        fStdin  = o.fStdin;
        fStdout = o.fStdout;
        o.pid  = -1;
        o.fStdin  = nullptr;
        o.fStdout = nullptr;
    }
    return *this;
}

#endif


