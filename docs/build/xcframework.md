# XCFramework

> [!WARNING]
> This fork does not support building the XCFramework and publishes none. `build-xcframework.sh` is upstream's Metal-based script: it copies `ggml/include/ggml-metal.h` and links `libggml-metal.a`, and the Metal backend is not in this tree, so the script cannot complete (under `set -e` it aborts at that header copy, if not earlier). The release linked below is an upstream ggml-org build; it contains Metal but none of this fork's changes. On macOS, this fork's CMake still offers the CPU backend and the BLAS backend (Accelerate, on by default); see [build.md](build.md).

The XCFramework is a precompiled version of the library for iOS, visionOS, tvOS,
and macOS. It can be used in Swift projects without the need to compile the
library from source. For example:

```swift
// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MyLlamaPackage",
    targets: [
        .executableTarget(
            name: "MyLlamaPackage",
            dependencies: [
                "LlamaFramework"
            ]),
        .binaryTarget(
            name: "LlamaFramework",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b5046/llama-b5046-xcframework.zip",
            checksum: "c19be78b5f00d8d29a25da41042cb7afa094cbf6280a225abe614b03b20029ab"
        )
    ]
)
```

The above example is using an intermediate build `b5046` of the library. This can be modified
to use a different version by changing the URL and checksum.
