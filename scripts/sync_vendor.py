#!/usr/bin/env python3

vendor = {
    "https://github.com/nlohmann/json/releases/latest/download/json.hpp": "common/nlohmann/json.hpp",

    # sync manually
    #"https://raw.githubusercontent.com/ochafik/minja/refs/heads/main/include/minja/minja.hpp":         "common/minja/minja.hpp",
    #"https://raw.githubusercontent.com/ochafik/minja/refs/heads/main/include/minja/chat-template.hpp": "common/minja/chat-template.hpp",

    "https://raw.githubusercontent.com/nothings/stb/refs/heads/master/stb_image.h": "tools/mtmd/vendor/stb_image.h",

    "https://raw.githubusercontent.com/mackron/miniaudio/refs/heads/master/miniaudio.h": "tools/mtmd/vendor/miniaudio.h",

    "https://raw.githubusercontent.com/yhirose/cpp-httplib/refs/heads/master/httplib.h": "tools/server/httplib.h",
}

import urllib.request

for url, filename in vendor.items():
    print(f"downloading {url} to {filename}")
    urllib.request.urlretrieve(url, filename)

