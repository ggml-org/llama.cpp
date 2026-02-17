#!/usr/bin/env python3

import urllib.request
import os
import sys
import subprocess

HTTPLIB_VERSION = "d4180e923f846b44a3d30acd938438d6e64fc9f6"

vendor = {
    "https://github.com/nlohmann/json/releases/latest/download/json.hpp":     "vendor/nlohmann/json.hpp",
    "https://github.com/nlohmann/json/releases/latest/download/json_fwd.hpp": "vendor/nlohmann/json_fwd.hpp",

    "https://raw.githubusercontent.com/nothings/stb/refs/heads/master/stb_image.h": "vendor/stb/stb_image.h",

    # not using latest tag to avoid this issue: https://github.com/ggml-org/llama.cpp/pull/17179#discussion_r2515877926
    # "https://github.com/mackron/miniaudio/raw/refs/tags/0.11.23/miniaudio.h": "vendor/miniaudio/miniaudio.h",
    "https://github.com/mackron/miniaudio/raw/669ed3e844524fcd883231b13095baee9f6de304/miniaudio.h": "vendor/miniaudio/miniaudio.h",

    f"https://raw.githubusercontent.com/yhirose/cpp-httplib/{HTTPLIB_VERSION}/httplib.h": "httplib.h",
    f"https://raw.githubusercontent.com/yhirose/cpp-httplib/{HTTPLIB_VERSION}/split.py":  "split.py",
    f"https://raw.githubusercontent.com/yhirose/cpp-httplib/{HTTPLIB_VERSION}/LICENSE":   "vendor/cpp-httplib/LICENSE",

    "https://raw.githubusercontent.com/sheredom/subprocess.h/b49c56e9fe214488493021017bf3954b91c7c1f5/subprocess.h": "vendor/sheredom/subprocess.h",
}

for url, filename in vendor.items():
    print(f"downloading {url} to {filename}") # noqa: NP100
    urllib.request.urlretrieve(url, filename)

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
