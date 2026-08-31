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
TODO:
```

### Usage
```console
$ ./scripts/build-profile-baseline.sh
```


### Baseline
The initial run of this produces the following report:
* [baseline-report.txt](./profiling-reports/baseline-report.txt)
