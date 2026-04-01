#include "../tool-registry.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <sstream>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>
#endif

static const size_t MAX_OUTPUT_LENGTH = 100000;
static const size_t MAX_COLLECT_LENGTH = MAX_OUTPUT_LENGTH * 2;  // Collect 2x, then tail-truncate
static const int MAX_OUTPUT_LINES = 2000;

// Truncate output to max lines (keeps the TAIL — errors are at the end)
static std::string truncate_lines(const std::string & text, int max_lines) {
    std::vector<std::string> lines;
    std::istringstream stream(text);
    std::string line;

    while (std::getline(stream, line)) {
        lines.push_back(line);
    }

    if ((int)lines.size() <= max_lines) {
        return text;
    }

    int omitted = (int)lines.size() - max_lines;
    std::ostringstream result;
    result << "[First " << omitted << " lines omitted]\n";
    for (int i = omitted; i < (int)lines.size(); i++) {
        result << lines[i] << "\n";
    }
    return result.str();
}

static tool_result bash_execute(const json & args, const tool_context & ctx) {
    std::string command = args.value("command", "");
    int timeout_ms = args.value("timeout", ctx.timeout_ms);

    if (command.empty()) {
        return {false, "", "command parameter is required"};
    }

    std::string output;
    size_t total_bytes_seen = 0;  // Track total output for truncation annotation
    int exit_code = 0;
    bool timed_out = false;
    bool was_interrupted = false;

#ifdef _WIN32
    // Windows implementation
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(SECURITY_ATTRIBUTES);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = NULL;

    HANDLE hReadPipe, hWritePipe;
    if (!CreatePipe(&hReadPipe, &hWritePipe, &sa, 0)) {
        return {false, "", "Failed to create pipe"};
    }

    SetHandleInformation(hReadPipe, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si = {sizeof(STARTUPINFOA)};
    si.hStdOutput = hWritePipe;
    si.hStdError = hWritePipe;
    si.dwFlags |= STARTF_USESTDHANDLES;

    PROCESS_INFORMATION pi;
    std::string cmd_line = "cmd /c " + command;

    if (!CreateProcessA(NULL, (LPSTR)cmd_line.c_str(), NULL, NULL, TRUE,
                        CREATE_NO_WINDOW, NULL, ctx.working_dir.c_str(), &si, &pi)) {
        CloseHandle(hReadPipe);
        CloseHandle(hWritePipe);
        return {false, "", "Failed to create process"};
    }

    CloseHandle(hWritePipe);

    // Read output with timeout
    auto start = std::chrono::steady_clock::now();
    char buffer[4096];
    DWORD bytesRead;

    while (true) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();

        if (elapsed > timeout_ms) {
            TerminateProcess(pi.hProcess, 1);
            timed_out = true;
            break;
        }

        if (ctx.is_interrupted && ctx.is_interrupted->load()) {
            TerminateProcess(pi.hProcess, 1);
            was_interrupted = true;
            break;
        }

        DWORD available = 0;
        PeekNamedPipe(hReadPipe, NULL, 0, NULL, &available, NULL);
        if (available == 0) {
            DWORD wait_result = WaitForSingleObject(pi.hProcess, 100);
            if (wait_result == WAIT_OBJECT_0) break;
            continue;
        }

        if (ReadFile(hReadPipe, buffer, sizeof(buffer) - 1, &bytesRead, NULL) && bytesRead > 0) {
            buffer[bytesRead] = '\0';
            total_bytes_seen += bytesRead;
            output.append(buffer, bytesRead);
            // Rolling tail: keep only the last MAX_COLLECT_LENGTH bytes
            if (output.size() > MAX_COLLECT_LENGTH) {
                output.erase(0, output.size() - MAX_OUTPUT_LENGTH);
            }
        }
    }

    DWORD exitCodeDword;
    GetExitCodeProcess(pi.hProcess, &exitCodeDword);
    exit_code = (int)exitCodeDword;

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(hReadPipe);

#else
    // Unix implementation
    int pipe_fd[2];
    if (pipe(pipe_fd) == -1) {
        return {false, "", "Failed to create pipe"};
    }

    pid_t pid = fork();
    if (pid == -1) {
        close(pipe_fd[0]);
        close(pipe_fd[1]);
        return {false, "", "Failed to fork process"};
    }

    if (pid == 0) {
        // Child process
        close(pipe_fd[0]);
        dup2(pipe_fd[1], STDOUT_FILENO);
        dup2(pipe_fd[1], STDERR_FILENO);
        close(pipe_fd[1]);

        if (chdir(ctx.working_dir.c_str()) != 0) {
            _exit(127);
        }

        execl("/bin/sh", "sh", "-c", command.c_str(), nullptr);
        _exit(127);
    }

    // Parent process
    close(pipe_fd[1]);

    // Set non-blocking read
    int flags = fcntl(pipe_fd[0], F_GETFL, 0);
    fcntl(pipe_fd[0], F_SETFL, flags | O_NONBLOCK);

    auto start = std::chrono::steady_clock::now();
    char buffer[4096];

    while (true) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();

        if (elapsed > timeout_ms) {
            kill(pid, SIGKILL);
            timed_out = true;
            break;
        }

        if (ctx.is_interrupted && ctx.is_interrupted->load()) {
            kill(pid, SIGKILL);
            was_interrupted = true;
            break;
        }

        ssize_t n = read(pipe_fd[0], buffer, sizeof(buffer) - 1);
        if (n > 0) {
            buffer[n] = '\0';
            total_bytes_seen += n;
            output.append(buffer, n);
            // Rolling tail: keep only the last MAX_COLLECT_LENGTH bytes
            if (output.size() > MAX_COLLECT_LENGTH) {
                output.erase(0, output.size() - MAX_OUTPUT_LENGTH);
            }
        } else if (n == 0) {
            // EOF
            break;
        } else {
            // EAGAIN - no data available, check if process ended
            int status;
            pid_t result = waitpid(pid, &status, WNOHANG);
            if (result == pid) {
                // Process ended, read remaining data
                while ((n = read(pipe_fd[0], buffer, sizeof(buffer) - 1)) > 0) {
                    buffer[n] = '\0';
                    total_bytes_seen += n;
                    output.append(buffer, n);
                    if (output.size() > MAX_COLLECT_LENGTH) {
                        output.erase(0, output.size() - MAX_OUTPUT_LENGTH);
                    }
                }
                exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
                break;
            }
            // Sleep briefly to avoid busy-waiting
            usleep(10000);  // 10ms
        }
    }

    close(pipe_fd[0]);

    // Wait for child if not already done
    int status;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status)) {
        exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        exit_code = 128 + WTERMSIG(status);  // Standard shell convention
    }
#endif

    // Tail-truncate raw output to MAX_OUTPUT_LENGTH, snapping to a line boundary
    bool output_was_truncated = (total_bytes_seen > MAX_OUTPUT_LENGTH);
    if (output.size() > MAX_OUTPUT_LENGTH) {
        output = output.substr(output.size() - MAX_OUTPUT_LENGTH);
        // Snap to first newline to avoid a partial opening line
        size_t nl = output.find('\n');
        if (nl != std::string::npos && nl < 200) {
            output = output.substr(nl + 1);
        }
    }

    // Then truncate to max lines (also keeps tail)
    std::string truncated_output = truncate_lines(output, MAX_OUTPUT_LINES);

    std::ostringstream result_output;
    result_output << truncated_output;

    if (output_was_truncated) {
        result_output << "\n[Output truncated to last " << MAX_OUTPUT_LENGTH << " characters]";
    }

    if (was_interrupted) {
        result_output << "\n[Interrupted]";
    }

    if (timed_out) {
        result_output << "\n[Timed out after " << timeout_ms << "ms]";
    }

    if (exit_code != 0) {
        result_output << "\n[Exit code: " << exit_code << "]";
    }

    return {exit_code == 0 && !timed_out && !was_interrupted, result_output.str(), ""};
}

static tool_def bash_tool = {
    "bash",
    "Execute a bash/shell command. Use for running programs, git operations, build commands, etc. The command runs in the project working directory.",
    R"json({
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute"
            },
            "timeout": {
                "type": "integer",
                "description": "Optional timeout in milliseconds (default 120000)"
            }
        },
        "required": ["command"]
    })json",
    bash_execute
};

REGISTER_TOOL(bash, bash_tool);
