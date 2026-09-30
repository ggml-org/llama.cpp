## Build profiling
This page is a working document for analyzing the current build and try to
identify ways to improve the build time.

### Requirements
The profiling script requires clang to be used as the compiler tool chain and
also requires that ClangBuildAnalyzer is installed.

Mac:
```console
brew install clang-build-analyzer
```

Linux:
```console
git clone https://github.com/aras-p/ClangBuildAnalyzer.git
cd ClangBuildAnalyzer
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
sudo cp build/ClangBuildAnalyzer /usr/local/bin/
```

Windows: install LLVM/clang and Ninja (e.g. via the
[LLVM releases page](https://github.com/llvm/llvm-project/releases) and
`winget install Ninja-build.Ninja`), then build ClangBuildAnalyzer the same
way as on Linux:
```console
git clone https://github.com/aras-p/ClangBuildAnalyzer.git
cd ClangBuildAnalyzer
cmake -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```
Then add `ClangBuildAnalyzer\build` to `PATH`.

### Usage
Mac/Linux:
```console
$ ./scripts/build-profile.sh
```

Windows:
```console
> .\scripts\build-profile.ps1
```

Both accept `--full`/`-Full` (include Server, Tools, and Tests) and a jobs
override (`-jN` / `-Jobs N`).

Note: on Windows, `cmake` defaults to the Visual Studio generator, which
ignores `CMAKE_C_COMPILER`/`CMAKE_CXX_COMPILER` and silently falls back to
MSVC. `build-profile.ps1` passes `-G Ninja` so clang is actually used, this
is required on ARM64.

### Linux (Ubuntu 24.04)

Environment:
- Clang:     18.1.3 (Ubuntu clang version 18.1.3 (1ubuntu1))
- libstdc++: GCC 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1)
- Target:    x86_64-pc-linux-gnu

```console
+------------------------+-----+------------+------------+------------+
| Build                  | TUs | Frontend   | Backend    | Total      |
+------------------------+-----+------------+------------+------------+
| Minimal, master        | 249 |   468.2 s  |   270.3 s  |   738.5 s  |
| Minimal, with PCH      | 253 |   177.1 s  |   265.8 s  |   442.9 s  |
| Full,    master        | 396 |   811.0 s  |   692.2 s  | 1,503.2 s  |
| Full,    with PCH      | 405 |   380.0 s  |   664.7 s  | 1,044.7 s  |
| Full,    with PCH + UB | 264 |   357.7 s  |   635.7 s  |   993.4 s  |
+------------------------+-----+------------+------------+------------+

PCH  = precompiled header.
Full = includes building Server, Tools, and Tests.
UB   = unity build for models
```
Note that the number of translation units (TUs) increases when using precompiled
headers — each PCH target adds one extra TU for the precompilation step itself.

### Mac (Apple M3)

Environment:
- Clang:  Apple clang version 17.0.0 (clang-1700.3.19.1)
- libc++: ships with Apple clang 17.0.0 (Xcode toolchain)
- Target: arm64-apple-macosx15.6

```console
+------------------------+-----+------------+------------+------------+
| Build                  | TUs | Frontend   | Backend    | Total      |
+------------------------+-----+------------+------------+------------+
| Minimal, master        | 256 |   154.5 s  |    94.8 s  |   249.3 s  |
| Minimal, with PCH      | 261 |    65.9 s  |    90.0 s  |   155.9 s  |
| Full,    master        | 407 |   265.7 s  |   209.7 s  |   475.4 s  |
| Full,    with PCH      | 414 |   154.6 s  |   197.5 s  |   352.1 s  |
| Full,    with PCH + UB | 274 |   143.0 s  |   192.2 s  |   335.2 s  |
+------------------------+-----+------------+------------+------------+

PCH = precompiled header.
Full = includes building Server, Tools, and Tests.
UB   = unity build for models
```

### Windows (ARM64)

Environment:
- Clang:  clang version 22.1.8 (LLVM, `C:\Program Files\LLVM`)
- STL:    MSVC STL (Visual Studio 2022 Build Tools 14.44.35207)
- Target: aarch64-pc-windows-msvc

```console
+------------------------+-----+------------+------------+------------+
| Build                  | TUs | Frontend   | Backend    | Total      |
+------------------------+-----+------------+------------+------------+
| Minimal, master        | 249 |   159.4 s  |    82.2 s  |   241.6 s  |
| Full,    master        | 373 |   337.2 s  |   167.4 s  |   504.6 s  |
| Minimal, with PCH + UB | 113 |    62.3 s  |    82.4 s  |   144.7 s  |
| Full,    with PCH + UB | 240 |   233.0 s  |   185.1 s  |   418.1 s  |
+------------------------+-----+------------+------------+------------+

PCH = precompiled header.
Full = includes building Server, Tools, and Tests.
UB   = unity build for models
```
