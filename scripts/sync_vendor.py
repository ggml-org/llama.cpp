#!/usr/bin/env python3

import urllib.request

vendor = {
    "https://github.com/nlohmann/json/releases/latest/download/json.hpp":     "vendor/nlohmann/json.hpp",
    "https://github.com/nlohmann/json/releases/latest/download/json_fwd.hpp": "vendor/nlohmann/json_fwd.hpp",

    "https://raw.githubusercontent.com/nothings/stb/refs/heads/master/stb_image.h": "vendor/stb/stb_image.h",

    # not using latest tag to avoid this issue: https://github.com/ggml-org/llama.cpp/pull/17179#discussion_r2515877926
    # "https://github.com/mackron/miniaudio/raw/refs/tags/0.11.23/miniaudio.h": "vendor/miniaudio/miniaudio.h",
    "https://github.com/mackron/miniaudio/raw/669ed3e844524fcd883231b13095baee9f6de304/miniaudio.h": "vendor/miniaudio/miniaudio.h",

    "https://raw.githubusercontent.com/yhirose/cpp-httplib/refs/tags/v0.30.2/httplib.h": "vendor/cpp-httplib/httplib.h",
    "https://raw.githubusercontent.com/yhirose/cpp-httplib/refs/tags/v0.30.2/LICENSE":   "vendor/cpp-httplib/LICENSE",

    "https://raw.githubusercontent.com/sheredom/subprocess.h/b49c56e9fe214488493021017bf3954b91c7c1f5/subprocess.h": "vendor/sheredom/subprocess.h",
}

for url, filename in vendor.items():
    print(f"downloading {url} to {filename}") # noqa: NP100
    urllib.request.urlretrieve(url, filename)

    # split cpp/h files for httplib
    # see: https://github.com/yhirose/cpp-httplib/blob/master/split.py
    if 'httplib.h' in filename:
        border = '// ----------------------------------------------------------------------------'
        with open(filename, 'r') as f:
            content = f.read()
        header, implementation, footer = content.split(border, 2)
        fname_cpp = filename.replace('.h', '.cpp')
        with open(filename, 'w') as fh:
            fh.write(header)
            fh.write(footer)
        with open(fname_cpp, 'w') as fc:
            fc.write('#include "httplib.h"\n')
            fc.write('namespace httplib {\n')
            fc.write(implementation.replace('\ninline ', '\n'))
            fc.write('} // namespace httplib\n')
