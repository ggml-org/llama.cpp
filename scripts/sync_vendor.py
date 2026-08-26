#!/usr/bin/env python3

import urllib.request
import os
import sys
import subprocess

HTTPLIB_VERSION = "refs/tags/v0.54.1"

# used by examples/gguf-hash, these repos have no release tag, so we pin a commit
XXHASH_COMMIT      = "9f465f1ea932d6ad9a26cd77496311ffa544cd68"
SHA1_COMMIT        = "e1e2536fcf6a8f9703be8c85d58724b408552287"
SHA256_COMMIT      = "5e637272c13f200872d55ff579f7e2ab6c3f252f"
ROTATE_BITS_COMMIT = "27e784942f67db44abf2115c6638e735b579acd1"

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

    "https://raw.githubusercontent.com/sheredom/subprocess.h/0dccaa9aa176dd6d7ef8afeca3c18d6e80a32795/subprocess.h": "vendor/sheredom/subprocess.h",

    f"https://raw.githubusercontent.com/Cyan4973/xxHash/{XXHASH_COMMIT}/xxhash.c":      "vendor/hash/xxhash/xxhash.c",
    f"https://raw.githubusercontent.com/Cyan4973/xxHash/{XXHASH_COMMIT}/xxhash.h":      "vendor/hash/xxhash/xxhash.h",
    f"https://raw.githubusercontent.com/Cyan4973/xxHash/{XXHASH_COMMIT}/LICENSE":       "vendor/hash/xxhash/LICENSE",

    # clibs/sha1 ships no license file, the source header says public domain
    f"https://raw.githubusercontent.com/clibs/sha1/{SHA1_COMMIT}/sha1.c": "vendor/hash/sha1/sha1.c",
    f"https://raw.githubusercontent.com/clibs/sha1/{SHA1_COMMIT}/sha1.h": "vendor/hash/sha1/sha1.h",

    f"https://raw.githubusercontent.com/jb55/sha256.c/{SHA256_COMMIT}/sha256.c": "vendor/hash/sha256/sha256.c",
    f"https://raw.githubusercontent.com/jb55/sha256.c/{SHA256_COMMIT}/sha256.h": "vendor/hash/sha256/sha256.h",
    f"https://raw.githubusercontent.com/jb55/sha256.c/{SHA256_COMMIT}/LICENSE":  "vendor/hash/sha256/LICENSE",

    f"https://raw.githubusercontent.com/jb55/rotate-bits.h/{ROTATE_BITS_COMMIT}/rotate-bits.h": "vendor/hash/rotate-bits/rotate-bits.h",
    f"https://raw.githubusercontent.com/jb55/rotate-bits.h/{ROTATE_BITS_COMMIT}/LICENSE.md":   "vendor/hash/rotate-bits/LICENSE.md",
}

# local changes kept on top of the upstream sources
patches = {
    "vendor/hash/xxhash/xxhash.h": [(
        '#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L) /* >= C11 */\n',
        '/* Windows SDK under 10.0.22000 is missing stdalign.h so we add a check\n'
        '   before allowing the windows compiler to use the C11 form.\n'
        '   Reference: https://github.com/Cyan4973/xxHash/issues/955 */\n'
        '#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L) \\\n'
        '    && (defined(_MSC_VER) && (_MSC_VER >= 1000) || !defined(_MSC_VER)) /* >= C11 */\n'
    )],

    # sha1 exports a bare "SHA1" symbol, which clashes with the boringssl one at link time.
    # we compile it as C++ (see vendor/hash/CMakeLists.txt) and put it in a namespace.
    "vendor/hash/sha1/sha1.h": [
        (
            '#if defined(__cplusplus)\n'
            'extern "C" {\n'
            '#endif\n',

            'namespace vendor_hash {\n'
        ),
        (
            '#if defined(__cplusplus)\n'
            '}\n'
            '#endif\n',

            '} // namespace vendor_hash\n'
        ),
    ],

    "vendor/hash/sha1/sha1.c": [
        (
            '#include "sha1.h"\n',

            '#include "sha1.h"\n'
            '\n'
            'namespace vendor_hash {\n'
        ),
        (
            '    SHA1Final((unsigned char *)hash_out, &ctx);\n'
            '}\n',

            '    SHA1Final((unsigned char *)hash_out, &ctx);\n'
            '}\n'
            '\n'
            '} // namespace vendor_hash\n'
        ),
    ],

    # silence a maybe-uninitialized warning
    "vendor/hash/sha256/sha256.c": [(
        "  uint32_t W[16];\n",
        "  uint32_t W[16] = {0};\n"
    )],
}

for url, filename in vendor.items():
    print(f"downloading {url} to {filename}") # noqa: NP100
    urllib.request.urlretrieve(url, filename)

for filename, replacements in patches.items():
    print(f"patching {filename}") # noqa: NP100
    with open(filename, "r", encoding="utf-8", newline="") as f:
        content = f.read()
    for old, new in replacements:
        if content.count(old) != 1:
            print(f"Error: cannot apply patch on {filename}, upstream code has changed") # noqa: NP100
            sys.exit(1)
        content = content.replace(old, new)
    with open(filename, "w", encoding="utf-8", newline="") as f:
        f.write(content)

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
