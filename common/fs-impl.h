#pragma once

#include <stdexcept>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(__linux__)
#include <sys/types.h>
#include <pwd.h>
#endif

#ifndef DIRECTORY_SEPARATOR
#ifdef _WIN32
#define DIRECTORY_SEPARATOR '\\'
#else
#define DIRECTORY_SEPARATOR '/'
#endif
#endif

#ifdef _WIN32
inline std::wstring utf8_to_wstring(const std::string & str) {
    if (str.empty()) {
        return std::wstring();
    }

    int size = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0);

    if (size <= 0) {
        return std::wstring();
    }

    std::wstring wstr(size, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstr[0], size);

    return wstr;
}
#endif

#ifndef COMMON_EXPORT
#define COMMON_EXPORT inline
#endif

COMMON_EXPORT bool fs_create_directory_with_parents(const std::string & path) {
#ifdef _WIN32
    std::wstring wpath = utf8_to_wstring(path);

    // if the path already exists, check whether it's a directory
    const DWORD attributes = GetFileAttributesW(wpath.c_str());
    if ((attributes != INVALID_FILE_ATTRIBUTES) && (attributes & FILE_ATTRIBUTE_DIRECTORY)) {
        return true;
    }

    size_t pos_slash = 0;

    // process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('\\', pos_slash)) != std::string::npos) {
        const std::wstring subpath = wpath.substr(0, pos_slash);

        pos_slash += 1;

        // skip the drive letter, in some systems it can return an access denied error
        if (subpath.length() == 2 && subpath[1] == ':') {
            continue;
        }

        const bool success = CreateDirectoryW(subpath.c_str(), NULL);

        if (!success) {
            const DWORD error = GetLastError();

            // if the path already exists, ensure that it's a directory
            if (error == ERROR_ALREADY_EXISTS) {
                const DWORD attributes = GetFileAttributesW(subpath.c_str());
                if (attributes == INVALID_FILE_ATTRIBUTES || !(attributes & FILE_ATTRIBUTE_DIRECTORY)) {
                    return false;
                }
            } else {
                return false;
            }
        }
    }

    return true;
#else
    // if the path already exists, check whether it's a directory
    struct stat info;
    if (stat(path.c_str(), &info) == 0) {
        return S_ISDIR(info.st_mode);
    }

    size_t pos_slash = 1; // skip leading slashes for directory creation

    // process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('/', pos_slash)) != std::string::npos) {
        const std::string subpath = path.substr(0, pos_slash);
        struct stat info;

        // if the path already exists, ensure that it's a directory
        if (stat(subpath.c_str(), &info) == 0) {
            if (!S_ISDIR(info.st_mode)) {
                return false;
            }
        } else {
            // create parent directories
            const int ret = mkdir(subpath.c_str(), 0755);
            if (ret != 0) {
                return false;
            }
        }

        pos_slash += 1;
    }

    return true;
#endif // _WIN32
}

COMMON_EXPORT std::string fs_get_cache_directory() {
    std::string cache_directory = "";
    auto ensure_trailing_slash = [](std::string p) {
        // Make sure to add trailing slash
        if (p.back() != DIRECTORY_SEPARATOR) {
            p += DIRECTORY_SEPARATOR;
        }
        return p;
    };
    if (getenv("LLAMA_CACHE")) {
        cache_directory = std::getenv("LLAMA_CACHE");
    } else {
#if defined(__linux__) || defined(__FreeBSD__) || defined(_AIX) || \
    defined(__OpenBSD__) || defined(__NetBSD__)
        if (std::getenv("XDG_CACHE_HOME")) {
            cache_directory = std::getenv("XDG_CACHE_HOME");
        } else if (std::getenv("HOME")) {
            cache_directory = std::getenv("HOME") + std::string("/.cache/");
        } else {
#if defined(__linux__)
            /* no $HOME is defined, fallback to getpwuid */
            struct passwd *pw = getpwuid(getuid());
            if ((!pw) || (!pw->pw_dir)) {
                throw std::runtime_error("Failed to find $HOME directory");
            }

            cache_directory = std::string(pw->pw_dir) + std::string("/.cache/");
#else /* defined(__linux__) */
            throw std::runtime_error("Failed to find $HOME directory");
#endif /* defined(__linux__) */
        }
#elif defined(__APPLE__)
        cache_directory = std::getenv("HOME") + std::string("/Library/Caches/");
#elif defined(_WIN32)
        cache_directory = std::getenv("LOCALAPPDATA");
#elif defined(__EMSCRIPTEN__)
        throw std::runtime_error("fs_get_cache_directory not implemented on this platform");
#else
#  error Unknown architecture
#endif
        cache_directory = ensure_trailing_slash(cache_directory);
        cache_directory += "llama.cpp";
    }
    return ensure_trailing_slash(cache_directory);
}

#undef COMMON_EXPORT
