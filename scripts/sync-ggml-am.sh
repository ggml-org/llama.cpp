#!/bin/bash
#
# Synchronize ggml changes to llama.cpp
#
# Usage:
#
#   $ cd /path/to/llama.cpp
#   $ ./scripts/sync-ggml-am.sh -skip hash0,hash1,hash2... -C 3
#

set -e

sd=$(dirname $0)
cd $sd/../

SRC_LLAMA=$(pwd)
SRC_GGML=$(cd ../ggml; pwd)

if [ ! -d $SRC_GGML ]; then
    echo "ggml not found at $SRC_GGML"
    exit 1
fi

lc=$(cat $SRC_LLAMA/scripts/sync-ggml.last)
echo "Syncing ggml changes since commit $lc"

to_skip=""

# context for git patches in number of lines
ctx="8"

while [ "$1" != "" ]; do
    case $1 in
        -skip )
            shift
            to_skip=$1
            ;;
        -C )
            shift
            ctx=$1
            ;;
    esac
    shift
done

cd $SRC_GGML

git log --oneline $lc..HEAD
git log --oneline $lc..HEAD --reverse | grep -v "(llama/[0-9]*)" | cut -d' ' -f1 > $SRC_LLAMA/ggml-commits

if [ ! -s $SRC_LLAMA/ggml-commits ]; then
    rm -v $SRC_LLAMA/ggml-commits
    echo "No new commits"
    exit 0
fi

if [ -f $SRC_LLAMA/ggml-src.patch ]; then
    rm -v $SRC_LLAMA/ggml-src.patch
fi

while read c; do
    if [ -n "$to_skip" ]; then
        if [[ $to_skip == *"$c"* ]]; then
            echo "Skipping $c"
            continue
        fi
    fi

    git format-patch -U${ctx} -k $c~1..$c --stdout -- \
        CMakeLists.txt \
        src/CMakeLists.txt \
        cmake/FindSIMD.cmake \
        src/ggml*.h \
        src/ggml*.c \
        src/ggml*.cpp \
        src/ggml*.m \
        src/ggml*.metal \
        src/ggml*.cu \
        src/ggml-cann/* \
        src/ggml-cuda/* \
        src/ggml-sycl/* \
        src/vulkan-shaders/* \
        include/ggml*.h \
        tests/test-opt.cpp \
        tests/test-grad0.cpp \
        tests/test-quantize-fns.cpp \
        tests/test-quantize-perf.cpp \
        tests/test-backend-ops.cpp \
        LICENSE \
        scripts/gen-authors.sh \
        >> $SRC_LLAMA/ggml-src.patch
done < $SRC_LLAMA/ggml-commits

rm -v $SRC_LLAMA/ggml-commits

# delete files if empty
if [ ! -s $SRC_LLAMA/ggml-src.patch ]; then
    rm -v $SRC_LLAMA/ggml-src.patch
fi

cd $SRC_LLAMA

if [ -f $SRC_LLAMA/ggml-src.patch ]; then
    # replace PR numbers
    #
    # Subject: some text (#1234)
    # Subject: some text (ggml/1234)
    cat ggml-src.patch | sed -e 's/^Subject: \(.*\) (#\([0-9]*\))/Subject: \1 (ggml\/\2)/' > ggml-src.patch.tmp
    mv ggml-src.patch.tmp ggml-src.patch

    cat ggml-src.patch | sed -e 's/^\(.*\) (#\([0-9]*\))$/\1 (ggml\/\2)/' > ggml-src.patch.tmp
    mv ggml-src.patch.tmp ggml-src.patch

    # replace filenames:
    #
    # CMakelists.txt          -> ggml/CMakeLists.txt
    # src/CMakeLists.txt      -> ggml/src/CMakeLists.txt
    # cmake/FindSIMD.cmake    -> ggml/cmake/FindSIMD.cmake
    #
    # src/ggml.c              -> ggml/src/ggml.c
    # src/ggml-aarch64.c      -> ggml/src/ggml-aarch64.c
    # src/ggml-aarch64.h      -> ggml/src/ggml-aarch64.h
    # src/ggml-alloc.c        -> ggml/src/ggml-alloc.c
    # src/ggml-backend-impl.h -> ggml/src/ggml-backend-impl.h
    # src/ggml-backend.cpp    -> ggml/src/ggml-backend.cpp
    # src/ggml-cann/*         -> ggml/src/ggml-cann/
    # src/ggml-cann.cpp       -> ggml/src/ggml-cann.cpp
    # src/ggml-common.h       -> ggml/src/ggml-common.h
    # src/ggml-cuda/*         -> ggml/src/ggml-cuda/
    # src/ggml-cuda.cu        -> ggml/src/ggml-cuda.cu
    # src/ggml-impl.h         -> ggml/src/ggml-impl.h
    # src/ggml-kompute.cpp    -> ggml/src/ggml-kompute.cpp
    # src/ggml-metal.m        -> ggml/src/ggml-metal.m
    # src/ggml-quants.c       -> ggml/src/ggml-quants.c
    # src/ggml-quants.h       -> ggml/src/ggml-quants.h
    # src/ggml-rpc.cpp        -> ggml/src/ggml-rpc.cpp
    # src/ggml-sycl/*         -> ggml/src/ggml-sycl/
    # src/ggml-sycl.cpp       -> ggml/src/ggml-sycl.cpp
    # src/ggml-vulkan.cpp     -> ggml/src/ggml-vulkan.cpp
    # src/vulkan-shaders/*    -> ggml/src/vulkan-shaders/
    #
    # include/ggml.h         -> ggml/include/ggml.h
    # include/ggml-alloc.h   -> ggml/include/ggml-alloc.h
    # include/ggml-backend.h -> ggml/include/ggml-backend.h
    # include/ggml-blas.h    -> ggml/include/ggml-blas.h
    # include/ggml-cann.h    -> ggml/include/ggml-cann.h
    # include/ggml-cuda.h    -> ggml/include/ggml-cuda.h
    # include/ggml-kompute.h -> ggml/include/ggml-kompute.h
    # include/ggml-metal.h   -> ggml/include/ggml-metal.h
    # include/ggml-rpc.h     -> ggml/include/ggml-rpc.h
    # include/ggml-sycl.h    -> ggml/include/ggml-sycl.h
    # include/ggml-vulkan.h  -> ggml/include/ggml-vulkan.h
    #
    # tests/test-opt.cpp           -> tests/test-opt.cpp
    # tests/test-grad0.cpp         -> tests/test-grad0.cpp
    # tests/test-quantize-fns.cpp  -> tests/test-quantize-fns.cpp
    # tests/test-quantize-perf.cpp -> tests/test-quantize-perf.cpp
    # tests/test-backend-ops.cpp   -> tests/test-backend-ops.cpp
    #
    # LICENSE                -> LICENSE
    # scripts/gen-authors.sh -> scripts/gen-authors.sh

    cat ggml-src.patch | sed -E \
        -e 's/([[:space:]]|[ab]\/)CMakeLists.txt/\1ggml\/CMakeLists.txt/g' \
        -e 's/([[:space:]]|[ab]\/)src\/CMakeLists.txt/\1ggml\/src\/CMakeLists.txt/g' \
        -e 's/([[:space:]]|[ab]\/)cmake\/FindSIMD.cmake/\1ggml\/cmake\/FindSIMD.cmake/g' \
        -e 's/([[:space:]]|[ab]\/)src\/ggml\.c/\1ggml\/src\/ggml.c/g' \
        -e 's/([[:space:]]|[ab]\/)src\/ggml-aarch64\.c/\1ggml\/src\/ggml-aarch64.c/g' \
        -e 's/([[:space:]]|[ab]\/)src\/ggml-aarch64\.h/\1ggml\/src\/ggml-aarch64.h/g' \
        -e 's/([[:space:]]|[ab]\/)src\/ggml-alloc\.c/\1ggml\/src\/ggml-alloc.c/g' \
        -e 's/([[:space:]]|[ab]\/)src\/ggml-backend-impl\.h/\1ggml\/src\/ggml-backend-impl.h/g' \
        -e 's/([[:space:]]|[ab]\/)src\/ggml-backend\.cpp/\1ggml\/src\/ggml-backend.cpp/g' \
        -e 's/([[:space:]]|[ab]\/)src\/ggml-cann\//\1ggml\/src\/ggml-cann\//g' \
        -e 's/([[:space:]]|[ab]\/)src\/ggml-cann\.cpp/\1ggml\/src\/ggml-cann.cpp/g' \
        -e 's/([[:space:]]|[ab]\/)src\/ggml-common\.h/\1ggml\/src\/ggml-common.h/g' \
        -e 's/([[:space:]]|[ab]\/)src\/ggml-cuda\//\1ggml\/src\/ggml-cuda\//g' \
        -e 's/([[:space:]]|[ab]\/)src\/ggml-cuda\.cu/\1ggml\/src\/ggml-cuda.cu/g' \
        -e 's/([[:space:]]|[ab]\/)src\/ggml-impl\.h/\1ggml\/src\/ggml-impl.h/g' \
        -e 's/([[:space:]]|[ab]\/)src\/ggml-kompute\.cpp/\1ggml\/src\/ggml-kompute.cpp/g' \
        -e 's/([[:space:]]|[ab]\/)src\/ggml-metal\.m/\1ggml\/src\/ggml-metal.m/g' \
        -e 's/([[:space:]]|[ab]\/)src\/ggml-quants\.c/\1ggml\/src\/ggml-quants.c/g' \
        -e 's/([[:space:]]|[ab]\/)src\/ggml-quants\.h/\1ggml\/src\/ggml-quants.h/g' \
        -e 's/([[:space:]]|[ab]\/)src\/ggml-rpc\.cpp/\1ggml\/src\/ggml-rpc.cpp/g' \
        -e 's/([[:space:]]|[ab]\/)src\/ggml-sycl\//\1ggml\/src\/ggml-sycl\//g' \
        -e 's/([[:space:]]|[ab]\/)src\/ggml-sycl\.cpp/\1ggml\/src\/ggml-sycl.cpp/g' \
        -e 's/([[:space:]]|[ab]\/)src\/ggml-vulkan\.cpp/\1ggml\/src\/ggml-vulkan.cpp/g' \
        -e 's/([[:space:]]|[ab]\/)src\/vulkan-shaders\//\1ggml\/src\/vulkan-shaders\//g' \
        -e 's/([[:space:]]|[ab]\/)include\/ggml\.h/\1ggml\/include\/ggml.h/g' \
        -e 's/([[:space:]]|[ab]\/)include\/ggml-alloc\.h/\1ggml\/include\/ggml-alloc.h/g' \
        -e 's/([[:space:]]|[ab]\/)include\/ggml-backend\.h/\1ggml\/include\/ggml-backend.h/g' \
        -e 's/([[:space:]]|[ab]\/)include\/ggml-blas\.h/\1ggml\/include\/ggml-blas.h/g' \
        -e 's/([[:space:]]|[ab]\/)include\/ggml-cann\.h/\1ggml\/include\/ggml-cann.h/g' \
        -e 's/([[:space:]]|[ab]\/)include\/ggml-cuda\.h/\1ggml\/include\/ggml-cuda.h/g' \
        -e 's/([[:space:]]|[ab]\/)include\/ggml-kompute\.h/\1ggml\/include\/ggml-kompute.h/g' \
        -e 's/([[:space:]]|[ab]\/)include\/ggml-metal\.h/\1ggml\/include\/ggml-metal.h/g' \
        -e 's/([[:space:]]|[ab]\/)include\/ggml-rpc\.h/\1ggml\/include\/ggml-rpc.h/g' \
        -e 's/([[:space:]]|[ab]\/)include\/ggml-sycl\.h/\1ggml\/include\/ggml-sycl.h/g' \
        -e 's/([[:space:]]|[ab]\/)include\/ggml-vulkan\.h/\1ggml\/include\/ggml-vulkan.h/g' \
        -e 's/([[:space:]]|[ab]\/)examples\/common\.h/\1examples\/common.h/g' \
        -e 's/([[:space:]]|[ab]\/)examples\/common\.cpp/\1examples\/common.cpp/g' \
        -e 's/([[:space:]]|[ab]\/)examples\/common-ggml\.h/\1examples\/common-ggml.h/g' \
        -e 's/([[:space:]]|[ab]\/)examples\/common-ggml\.cpp/\1examples\/common-ggml.cpp/g' \
        -e 's/([[:space:]]|[ab]\/)LICENSE/\1LICENSE/g' \
        -e 's/([[:space:]]|[ab]\/)scripts\/gen-authors\.sh/\1scripts\/gen-authors.sh/g' \
        > ggml-src.patch.tmp
    mv ggml-src.patch.tmp ggml-src.patch

    git am -C${ctx} ggml-src.patch

    rm -v $SRC_LLAMA/ggml-src.patch
fi

# update last commit
cd $SRC_GGML
git log -1 --format=%H > $SRC_LLAMA/scripts/sync-ggml.last

echo "Done"

exit 0
