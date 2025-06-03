#!/usr/bin/env python3

import urllib.request

vendor = {
    "https://github.com/nlohmann/json/releases/latest/download/json.hpp":     "vendor/nlohmann/json.hpp",
    "https://github.com/nlohmann/json/releases/latest/download/json_fwd.hpp": "vendor/nlohmann/json_fwd.hpp",

    # sync manually
    # "https://raw.githubusercontent.com/ochafik/minja/refs/heads/main/include/minja/minja.hpp":         "vendor/minja/minja.hpp",
    # "https://raw.githubusercontent.com/ochafik/minja/refs/heads/main/include/minja/chat-template.hpp": "vendor/minja/chat-template.hpp",

    "https://raw.githubusercontent.com/nothings/stb/refs/heads/master/stb_image.h": "vendor/stb/stb_image.h",

    "https://github.com/mackron/miniaudio/raw/refs/tags/0.11.22/miniaudio.h": "vendor/miniaudio/miniaudio.h",

    "https://raw.githubusercontent.com/yhirose/cpp-httplib/refs/tags/v0.20.1/httplib.h": "vendor/cpp-httplib/httplib.h",
}

for url, filename in vendor.items():
    print(f"downloading {url} to {filename}") # noqa: NP100
    urllib.request.urlretrieve(url, filename)
