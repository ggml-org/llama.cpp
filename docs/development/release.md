# Release process

llama.cpp uses [semantic versioning](https://semver.org) (`MAJOR.MINOR.PATCH`).

## Version bump guidelines

| Change type | Version component |
|---|---|
| Breaking change to the public C API (`include/llama.h`)         | `MAJOR` |
| Backward-compatible features, model support, or API addition    | `MINOR` |
| Bug fix with no API change                                      | `PATCH` |

The version is set in the three variables at the top of the root `CMakeLists.txt`:

```cmake
set(LLAMA_VERSION_MAJOR 0)
set(LLAMA_VERSION_MINOR 5)
set(LLAMA_VERSION_PATCH 0)
```

_A version bump should be included in the PR that introduces the change, or in a
dedicated bump commit merged before the release is cut._

_TODO: add PR labels (`semver: patch`, `semver: minor`, `semver: major`) to help
identify which PRs require a version bump before cutting a release._

## Making a release

This fork has no release automation. Upstream cuts releases with its manual
`make-release` GitHub workflow, but `.github/` is not part of this fork's tree,
and no `vX.Y.Z` release tag has been cut here.

## Building a release

By default, `LLAMA_BUILD_IS_DEV=ON` which appends a `-dev` suffix to `LLAMA_VERSION`,
marking the build as a nightly/development build. Distributors building from a
release tag must pass `-DLLAMA_BUILD_IS_DEV=OFF` to produce a clean version string
(e.g. `0.1.0` instead of `0.1.0-dev`).

## How releases reach users
This fork publishes no GitHub releases, pre-built binaries, or packages. Users
clone the repository and build from source (see [build.md](../build/build.md)). The
pre-built packages listed in [install.md](../build/install.md) are upstream builds.
