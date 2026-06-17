#!/usr/bin/env python3

import urllib.request
import os
import sys
import subprocess

HTTPLIB_VERSION = "refs/tags/v0.47.0"

vendor = {
    "https://github.com/nlohmann/json/releases/latest/download/json.hpp":     "vendor/nlohmann/json.hpp",
    "https://github.com/nlohmann/json/releases/latest/download/json_fwd.hpp": "vendor/nlohmann/json_fwd.hpp",

    "https://raw.githubusercontent.com/nothings/stb/refs/heads/master/stb_image.h": "vendor/stb/stb_image.h",

    # not using latest tag to avoid this issue: https://github.com/ggml-org/llama.cpp/pull/17179#discussion_r2515877926
    # "https://github.com/mackron/miniaudio/raw/refs/tags/0.11.24/miniaudio.h": "vendor/miniaudio/miniaudio.h",
    "https://github.com/mackron/miniaudio/raw/9634bedb5b5a2ca38c1ee7108a9358a4e233f14d/miniaudio.h": "vendor/miniaudio/miniaudio.h",

    f"https://raw.githubusercontent.com/yhirose/cpp-httplib/{HTTPLIB_VERSION}/httplib.h": "httplib.h",
    f"https://raw.githubusercontent.com/yhirose/cpp-httplib/{HTTPLIB_VERSION}/split.py":  "split.py",
    f"https://raw.githubusercontent.com/yhirose/cpp-httplib/{HTTPLIB_VERSION}/LICENSE":   "vendor/cpp-httplib/LICENSE",

    "https://raw.githubusercontent.com/sheredom/subprocess.h/b49c56e9fe214488493021017bf3954b91c7c1f5/subprocess.h": "vendor/sheredom/subprocess.h",
}


def _apply_wifsignaled_patch(path: str) -> None:
    """Apply the WIFSIGNALED encoding patch — without it subprocess_join
    and subprocess_alive collapse all signal deaths (SIGABRT, SIGTERM,
    SIGKILL) to EXIT_FAILURE(1), making them indistinguishable from
    normal errors.  Store the negated signal number so callers can
    report the actual cause of death."""
    with open(path) as f:
        content = f.read()

    # Replace the error-only else in subprocess_join
    old = """    if (WIFEXITED(status)) {
      process->return_status = WEXITSTATUS(status);
    } else {
      process->return_status = EXIT_FAILURE;
    }"""
    new = """    if (WIFEXITED(status)) {
      process->return_status = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        // Store negated signal number so callers can distinguish signal death
        // (e.g. -6 for SIGABRT from OOM, -15 for SIGTERM from force-kill) from
        // normal error exit (positive exit code) and clean exit (exit code 0).
        process->return_status = -WTERMSIG(status);
    } else {
      process->return_status = EXIT_FAILURE;
    }"""
    content = content.replace(old, new)

    # Same fix in subprocess_alive
    old = """    if (WIFEXITED(status)) {
        process->return_status = WEXITSTATUS(status);
      } else {
        process->return_status = EXIT_FAILURE;
      }"""
    new = """    if (WIFEXITED(status)) {
        process->return_status = WEXITSTATUS(status);
      } else if (WIFSIGNALED(status)) {
        // Store negated signal number so callers can distinguish signal death
        // (e.g. -6 for SIGABRT from OOM, -15 for SIGTERM from force-kill) from
        // normal error exit (positive exit code) and clean exit (exit code 0).
        process->return_status = -WTERMSIG(status);
      } else {
        process->return_status = EXIT_FAILURE;
      }"""
    content = content.replace(old, new)

    with open(path, "w") as f:
        f.write(content)


for url, filename in vendor.items():
    print(f"downloading {url} to {filename}") # noqa: NP100
    urllib.request.urlretrieve(url, filename)

_apply_wifsignaled_patch("vendor/sheredom/subprocess.h")

print("Splitting httplib.h...") # noqa: NP100
try:
    subprocess.check_call([
        sys.executable, "split.py",
        "--extension", "cpp",
        "--out", "vendor/cpp-httplib"
    ])
except Exception as e:
    print(f"Error: {e}") # noqa: NP100
    sys.exit(1)
finally:
    os.remove("split.py")
    os.remove("httplib.h")
