#!/bin/sh
# Check for backwards-incompatible API and ABI changes between two builds
#
# Backwards-incompatible API changes, such as removing a value from an enum,
# are checked by abi-compliance-checker. Such changes can break compilation of
# existing programs.
#
# Backwards-incompatible ABI changes, such as the removal of a public function,
# are checked by libigail-tools. Such changes could break run-time dynamic
# linking of existing binaries. (We don't use a-c-c for ABI checks because it
# needs a debug build, whereas abigail does not.)
#
# Commands:
#   --generate <build-dir>: Creates API/ABI dumps in <build-dir>
#                           <build-dir> is expected to be a CMake build result
#   --check <dir1> <dir2>:  Compares dumps in <dir1> and <dir2>
#                           Comparison exit codes
#                              0: all good
#                              1: backwards-incompatible changes found
#
# Options:
#   --include-path <dir>:   a-c-c calls gcc on headers; use this option to add
#                           directories to gcc's search path
#
#
# This script would typically be used before cutting a release:
#
#   1. Generate API/ABI dump for the old version
#
#      $ check-apiabi-compat.sh --generate <build-dir-old> libfoo [ libbar ...]
#
#   2. <update source>
#
#   3. Generate API/ABI dump for the new version
#
#      $ check-apiabi-compat.sh --generate <build-dir-new> libfoo [ libbar ...]
#
#   4. Compare the two dumps
#
#      $ check-apiabi-compat.sh --check <old-build-dir> <new-build-dir>
#
#      If the check exits 0, all is fine. Otherwise, backwards-incompatible
#      changes were found, and the librar(ies) need a SOVER bump.
set -eu

# Preconditions
if ! command -v abi-compliance-checker >/dev/null 2>&1; then
    echo "abi-compliance-checker is not installed." >&2
    exit 1
elif ! command -v abidw >/dev/null 2>&1; then
    echo "abigail-tools are not installed." >&2
    exit 1
fi

# Some generic functions
usage() {
    echo "Usage: $0 [ --include-path <dir> ] --generate <build-dir> libXXX [ libYYY ... ]" >&2
    echo "       $0 --check <old-build-dir> <new-build-dir>" >&2
}

get_cmake_project_name() {
    sed -nr 's/^project\("(.*)".*$/\1/p' CMakeLists.txt
}

get_cmake_version() {
    major="$(sed -nr 's/^set\([A-Z]+_VERSION_MAJOR ([0-9]+)\)$/\1/p' CMakeLists.txt)"
    minor="$(sed -nr 's/^set\([A-Z]+_VERSION_MINOR ([0-9]+)\)$/\1/p' CMakeLists.txt)"
    patch="$(sed -nr 's/^set\([A-Z]+_VERSION_PATCH ([0-9]+)\)$/\1/p' CMakeLists.txt)"
    echo "$major.$minor.$patch"
}

# Option parsing and validation
DO_GEN=0
DO_CHECK=0
BUILD_DIR=
BUILD_DIR_NEW=
INCLUDE_PATHS=
while [ "$#" -gt 0 ]; do
    case "$1" in
    --generate=*)
        DO_GEN=1
        BUILD_DIR="${1#*=}"
        shift
        ;;
    --generate)
        DO_GEN=1
        if [ -z "${2:-}" ]; then
            usage
            exit 1
        fi
        BUILD_DIR="$2"
        shift 2
        ;;
    --check)
        DO_CHECK=1
        if [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
            usage
            exit 1
        elif ! [ -d "$2" ]; then
            echo "$2 is not a directory." >&2
            exit 1
        elif ! [ -d "$3" ]; then
            echo "$3 is not a directory." >&2
            exit 1
        fi
        BUILD_DIR="$2"
        BUILD_DIR_NEW="$3"
        shift 3
        ;;
    --include-path=*)
        INCLUDE_PATHS="$INCLUDE_PATHS ${1#*=}"
        shift
        ;;
    --include-path)
        if [ -z "${2:-}" ]; then
            usage
            exit 1
        fi
        INCLUDE_PATHS="$INCLUDE_PATHS $2"
        shift 2
        ;;
    -h | --help)
        usage
        exit 1
        ;;
    -?*)
        usage
        exit 1
        ;;
    *)
        break
        ;;
    esac
done
if [ $((DO_GEN + DO_CHECK)) -gt 1 ]; then
    echo "Can only use one --command." >&2
    exit
fi
PROJECT_NAME="$(get_cmake_project_name)"
PROJECT_VERSION="$(get_cmake_version)"
LIB_NAMES=""
while [ "$#" -gt 0 ]; do
    if [ "${1#lib}" = "$1" ]; then
        echo "Library to check must start with libXXX." >&2
        exit 1
    fi
    LIB_NAMES="$LIB_NAMES $1"
    shift
done

dump_current_api() {
    echo "Dumping API..."

    DESCRIPTOR="$BUILD_DIR/apiabi/acc-descriptor.xml"
    mkdir -p "$BUILD_DIR/apiabi"
    cat >"$DESCRIPTOR" <<EOF
<version>$PROJECT_VERSION</version>
<headers>include</headers>
<add_include_paths>$INCLUDE_PATHS</add_include_paths>
EOF

    # This addresses a bug between a-c-c and universal-ctags, manifested when
    # a name is use both for a tag and a function name
    mkdir -p "$BUILD_DIR/apiabi/.ctags.d"
    echo "--fields=-t" >"$BUILD_DIR/apiabi/.ctags.d/acc.ctags"

    # Change HOME so that .ctags.d gets picked up by universal-ctags, if used
    HOME="$BUILD_DIR/apiabi" abi-compliance-checker \
        -headers-only \
        -lib "$PROJECT_NAME" \
        -dump "$DESCRIPTOR" \
        -log-path "$BUILD_DIR/apiabi/acc.log" \
        -dump-path "$BUILD_DIR/apiabi/api.dump"
    # acc generates this file with an ancient timestamp, which confuses gzip
    touch "$BUILD_DIR/apiabi/api.dump"
}

dump_current_abi() {
    echo "Dumping ABIs ..."
    mkdir -p "$BUILD_DIR/apiabi"
    # The suppressions are needed to avoid including all the internal C++
    # symbols, and system types
    cat >"$BUILD_DIR/apiabi/abidw.suppress" <<EOF
[suppress_function]

label = suppress internal C++ mangled functions
symbol_name_regexp = ^_Z
drop = yes

[suppress_variable]
label = suppress internal C++ mangled variables
symbol_name_regexp = ^_Z
drop = yes

[suppress_type]
label = Suppress types outside of our own source
source_location_not_regexp = ^include/
drop = yes
EOF

    # In abidw 2.5, handling of undefined stuff was changed a bit
    abidw_version="$(abidw --version | sed -r 's/^abidw: ([0-9]+\.[0-9]+).*$/\1/')"
    abidw_major="${abidw_version%.*}"
    abidw_minor="${abidw_version#*.}"
    if [ "$abidw_major" -gt 2 ] || [ "$abidw_minor" -gt 4 ]; then
        abidw_undefined_syms_options="--no-load-undefined-interfaces"
    else
        abidw_undefined_syms_options="--drop-undefined-syms"
    fi

    for lib_name in $LIB_NAMES; do
        # Depending on where add_library resides, the libraries can end up in
        # build/src or build/bin
        lib_path="$BUILD_DIR/src/$lib_name.so"
        if ! [ -f "$lib_path" ]; then
            lib_path="$BUILD_DIR/bin/$lib_name.so"
            if ! [ -f "$lib_path" ]; then
                echo "Cannot find library $lib_name.so" >&2
                exit 1
            fi
        fi
        abidw \
            --headers-dir include \
            "$abidw_undefined_syms_options" \
            --suppressions "$BUILD_DIR/apiabi/abidw.suppress" \
            --out-file "${BUILD_DIR}/apiabi/$lib_name.abi.xml" \
            "$lib_path"
    done
}

# Run the actual commands
if [ "$DO_GEN" -eq 1 ]; then
    dump_current_api
    dump_current_abi
    exit 0
elif [ "$DO_CHECK" -eq 1 ]; then
    # From here on, we don't want to exit on first error
    set +e

    abi-compliance-checker \
        -strict \
        -source \
        -library "$PROJECT_NAME" \
        -old "$BUILD_DIR/apiabi/api.dump" \
        -new "$BUILD_DIR_NEW/apiabi/api.dump" \
        -src-report-path "$BUILD_DIR_NEW/apiabi/api_compat_report.html"
    API_RESULT=$?

    ABI_RESULT=0
    for xml_file in "$BUILD_DIR/apiabi/"lib*.abi.xml; do
        xml_file_new="$BUILD_DIR_NEW/apiabi/$(basename "$xml_file")"

        if ! [ -f "$xml_file_new" ]; then
            echo "Cannot compare, missing file: $xml_file_new" >&2
            exit 1
        fi

        abidiff "$xml_file" "$xml_file_new"
        res=$?
        [ "$((res & 8))" -ne 0 ] && ABI_RESULT=1
    done

    if [ "$API_RESULT" -gt 0 ]; then
        echo "ERROR: API changed with possible backwards-compatibility problems." >&2
    fi
    if [ "$ABI_RESULT" -gt 0 ]; then
        echo "ERROR: ABI changed with possible backwards-compatibility problems." >&2
    fi
    if [ "$((API_RESULT + ABI_RESULT))" -gt 0 ]; then
        exit 1
    fi
fi
