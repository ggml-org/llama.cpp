#include "app-update.h"

#include "download.h"
#include "build-info.h"

#include <string>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/wait.h>
#endif

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

// the release channel this binary updates from
// the packager bakes these per variant, an empty channel disables self update
#ifndef LLAMA_APP_REPO
#define LLAMA_APP_REPO ""
#endif

#ifndef LLAMA_APP_ID_BINARY
#define LLAMA_APP_ID_BINARY ""
#endif

namespace fs = std::filesystem;

// reads a whole file into a string
static std::string read_text(const std::string & path) {
    std::string out;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return out;
    }
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        out.append(buf, n);
    }
    fclose(f);
    return out;
}

// turns a "bNNNN" tag into its integer, returns 0 when malformed
static int parse_build(const std::string & tag) {
    size_t i = 0;
    while (i < tag.size() && (tag[i] == 'b' || isspace((unsigned char) tag[i]))) {
        i++;
    }
    int v = 0;
    while (i < tag.size() && isdigit((unsigned char) tag[i])) {
        v = v * 10 + (tag[i] - '0');
        i++;
    }
    return v;
}

// true when the http status denotes a delivered file
static bool ok(int status) {
    return status >= 200 && status < 300;
}

// absolute path of the running executable
static std::string self_path() {
#if defined(_WIN32)
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(NULL, buf, sizeof(buf));
    if (n == 0) {
        return std::string();
    }
    return std::string(buf, n);
#elif defined(__APPLE__)
    char buf[PATH_MAX];
    uint32_t sz = sizeof(buf);
    if (_NSGetExecutablePath(buf, &sz) != 0) {
        return std::string();
    }
    char real[PATH_MAX];
    if (realpath(buf, real)) {
        return std::string(real);
    }
    return std::string(buf);
#else
    char buf[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf));
    if (n <= 0) {
        return std::string();
    }
    return std::string(buf, (size_t) n);
#endif
}

// pipes src through the unzstd helper into dst, returns true on success
static bool run_unzstd(const std::string & unzstd, const std::string & src, const std::string & dst) {
#if defined(_WIN32)
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = NULL;
    HANDLE in = CreateFileA(src.c_str(), GENERIC_READ, FILE_SHARE_READ, &sa, OPEN_EXISTING, 0, NULL);
    if (in == INVALID_HANDLE_VALUE) {
        return false;
    }
    HANDLE out = CreateFileA(dst.c_str(), GENERIC_WRITE, 0, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (out == INVALID_HANDLE_VALUE) {
        CloseHandle(in);
        return false;
    }
    STARTUPINFOA si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = in;
    si.hStdOutput = out;
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION pi;
    ZeroMemory(&pi, sizeof(pi));
    std::string cmd = unzstd;
    BOOL started = CreateProcessA(unzstd.c_str(), &cmd[0], NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi);
    CloseHandle(in);
    CloseHandle(out);
    if (!started) {
        return false;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return code == 0;
#else
    int fd_in = open(src.c_str(), O_RDONLY);
    if (fd_in < 0) {
        return false;
    }
    int fd_out = open(dst.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0755);
    if (fd_out < 0) {
        close(fd_in);
        return false;
    }
    pid_t pid = fork();
    if (pid == 0) {
        dup2(fd_in, 0);
        dup2(fd_out, 1);
        close(fd_in);
        close(fd_out);
        execl(unzstd.c_str(), unzstd.c_str(), (char *) NULL);
        _exit(127);
    }
    close(fd_in);
    close(fd_out);
    if (pid < 0) {
        return false;
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
#endif
}

// replaces the running executable with staged, the current image stays valid until exit
static bool swap_in_place(const std::string & exe, const std::string & staged) {
#if defined(_WIN32)
    // a running exe cannot be overwritten in place, but it can be renamed aside
    const std::string old = exe + ".old";
    MoveFileExA(exe.c_str(), old.c_str(), MOVEFILE_REPLACE_EXISTING);
    return MoveFileExA(staged.c_str(), exe.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
#else
    return rename(staged.c_str(), exe.c_str()) == 0;
#endif
}

// resolves the channel base and the variant id, false when this build has no channel
static bool channel(std::string & base, std::string & id, std::string & arch, std::string & os) {
    base = LLAMA_APP_REPO;
    id = LLAMA_APP_ID_BINARY;
    if (base.empty() || id.empty()) {
        return false;
    }
    size_t a = id.find('/');
    size_t b = id.find('/', a + 1);
    if (a == std::string::npos || b == std::string::npos) {
        return false;
    }
    arch = id.substr(0, a);
    os = id.substr(a + 1, b - a - 1);
    return true;
}

void llama_app_startup(void) {
#if defined(_WIN32)
    const std::string old = self_path() + ".old";
    DeleteFileA(old.c_str());
#endif
}

void llama_app_update_notice(void) {
    std::string base, id, arch, os;
    if (!channel(base, id, arch, os)) {
        return;
    }
    const fs::path tmp = fs::temp_directory_path() / ("llama-notice-" + std::to_string((long) time(NULL)));
    std::error_code ec;
    fs::create_directories(tmp, ec);
    if (ec) {
        return;
    }
    const std::string latest = (tmp / "latest").string();
    common_download_opts opts;
    if (ok(common_download_file_single(base + "/latest", latest, opts, true))) {
        const int remote = parse_build(read_text(latest));
        if (remote > llama_build_number()) {
            printf("a newer build is available (b%d), run: llama update\n", remote);
        }
    }
    fs::remove_all(tmp, ec);
}

int llama_app_update(bool force) {
    std::string base, id, arch, os;
    if (!channel(base, id, arch, os)) {
        printf("update: this build has no release channel configured\n");
        return 0;
    }

    // scratch dir for the download artifacts
    const fs::path tmp = fs::temp_directory_path() / ("llama-update-" + std::to_string((long) time(NULL)));
    std::error_code ec;
    fs::create_directories(tmp, ec);
    if (ec) {
        printf("update: cannot create scratch dir\n");
        return 1;
    }

    common_download_opts opts;
    std::string staged;
    auto work = [&]() -> int {
        // ask the channel for its latest build
        const std::string latest_path = (tmp / "latest").string();
        if (!ok(common_download_file_single(base + "/latest", latest_path, opts, true))) {
            printf("update: cannot reach the channel at %s\n", base.c_str());
            return 1;
        }
        const int remote = parse_build(read_text(latest_path));
        const int current = llama_build_number();
        printf("update: local b%d, channel b%d\n", current, remote);
        if (remote <= 0) {
            printf("update: channel returned no usable version\n");
            return 1;
        }
        if (!force && remote <= current) {
            printf("update: already up to date\n");
            return 0;
        }

        // pull the binary for this exact variant
        const std::string zst = (tmp / "llama-app.zst").string();
        const std::string url = base + "/b" + std::to_string(remote) + "/" + id + "/llama-app.zst";
        printf("update: downloading %s\n", url.c_str());
        if (!ok(common_download_file_single(url, zst, opts, true))) {
            printf("update: download failed\n");
            return 1;
        }

        // pull the matching unzstd helper, the same way the installer does
        const std::string helper = (os == "windows") ? "unzstd.exe" : "unzstd";
        const std::string unz = (tmp / helper).string();
        const std::string unz_url = base + "/b" + std::to_string(remote) + "/" + arch + "/" + os + "/" + helper;
        if (!ok(common_download_file_single(unz_url, unz, opts, true))) {
            printf("update: cannot fetch the unzstd helper\n");
            return 1;
        }
        fs::permissions(unz, fs::perms::owner_all, ec);

        // decompress next to the running binary
        const std::string exe = self_path();
        if (exe.empty()) {
            printf("update: cannot locate the running binary\n");
            return 1;
        }
        staged = exe + ".new";
        if (!run_unzstd(unz, zst, staged)) {
            printf("update: decompress failed\n");
            return 1;
        }

        // swap in place
        if (!swap_in_place(exe, staged)) {
            printf("update: cannot replace %s\n", exe.c_str());
            return 1;
        }
        printf("update: installed b%d, restart llama to run it\n", remote);
        return 0;
    };

    const int rc = work();

    // leave nothing behind
    fs::remove_all(tmp, ec);
    if (rc != 0 && !staged.empty()) {
        fs::remove(staged, ec);
    }
    return rc;
}

int llama_update(int argc, char ** argv) {
    bool force = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--force") == 0) {
            force = true;
        }
    }
    return llama_app_update(force);
}
