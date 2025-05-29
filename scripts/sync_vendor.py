#!/usr/bin/env python3

import urllib.request

vendor = {
    "https://github.com/nlohmann/json/releases/latest/download/json.hpp": "common/nlohmann/json.hpp",

    # sync manually
    # "https://raw.githubusercontent.com/ochafik/minja/refs/heads/main/include/minja/minja.hpp":         "common/minja/minja.hpp",
    # "https://raw.githubusercontent.com/ochafik/minja/refs/heads/main/include/minja/chat-template.hpp": "common/minja/chat-template.hpp",

    "https://raw.githubusercontent.com/nothings/stb/refs/heads/master/stb_image.h": "tools/mtmd/vendor/stb_image.h",

    "https://github.com/mackron/miniaudio/raw/refs/tags/0.11.22/miniaudio.h": "tools/mtmd/vendor/miniaudio.h",

    "https://raw.githubusercontent.com/yhirose/cpp-httplib/refs/tags/v0.20.1/httplib.h": "tools/server/httplib.h",
}

for url, filename in vendor.items():
    print(f"downloading {url} to {filename}") # noqa: NP100
    urllib.request.urlretrieve(url, filename)
