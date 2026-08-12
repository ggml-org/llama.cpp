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
set(LLAMA_VERSION_MINOR 1)
set(LLAMA_VERSION_PATCH 0)
```

_A version bump should be included in the PR that introduces the change, or in a
dedicated bump commit merged before the release is cut._

_TODO: add PR labels (`semver: patch`, `semver: minor`, `semver: major`) to help
identify which PRs require a version bump before cutting a release._

## Making a release

Releases are created by running the [make-release](.github/workflows/make-release.yml)
which is a manual workflow.

The workflow creates an annotated git tag (e.g. `v0.1.0`) and pushes it to the
remote. No GitHub Release object is created, the tag is the release artifact.

## Building a release

By default, `LLAMA_BUILD_IS_DEV=ON` which appends a `-dev` suffix to `LLAMA_VERSION`,
marking the build as a nightly/development build. Distributors building from a
release tag must pass `-DLLAMA_BUILD_IS_DEV=OFF` to produce a clean version string
(e.g. `0.1.0` instead of `0.1.0-dev`).

## How releases reach users
Currently releases are not published to github releases, only nightly/development
builds are available there. The way users can access releases are using the following
channels:

- **llama-install.sh**  — downloads pre-built binaries built from the release tag.
- **Package managers**  — consume the git tag directly.
- **Build from source** — users clone the repo and check out the tag.
