#!/usr/bin/env bash
#
# Options
IOS_MIN_OS_VERSION=16.4
MACOS_MIN_OS_VERSION=13.3
VISIONOS_MIN_OS_VERSION=1.0
TVOS_MIN_OS_VERSION=16.4

BUILD_SHARED_LIBS=OFF
LLAMA_BUILD_EXAMPLES=OFF
LLAMA_BUILD_TOOLS=OFF
LLAMA_BUILD_TESTS=OFF
LLAMA_BUILD_SERVER=OFF
GGML_METAL=ON
GGML_METAL_EMBED_LIBRARY=ON
GGML_BLAS_DEFAULT=ON
GGML_METAL_USE_BF16=ON
GGML_OPENMP=OFF

COMMON_C_FLAGS="-Wno-macro-redefined -Wno-shorten-64-to-32 -Wno-unused-command-line-argument -g"
COMMON_CXX_FLAGS="-Wno-macro-redefined -Wno-shorten-64-to-32 -Wno-unused-command-line-argument -g"

# Common options for all builds
COMMON_CMAKE_ARGS=(
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY=""
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO
    -DCMAKE_XCODE_ATTRIBUTE_DEBUG_INFORMATION_FORMAT="dwarf-with-dsym"
    -DCMAKE_XCODE_ATTRIBUTE_GCC_GENERATE_DEBUGGING_SYMBOLS=YES
    -DCMAKE_XCODE_ATTRIBUTE_COPY_PHASE_STRIP=NO
    -DCMAKE_XCODE_ATTRIBUTE_STRIP_INSTALLED_PRODUCT=NO
    -DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM=ggml
    -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
    -DLLAMA_BUILD_EXAMPLES=${LLAMA_BUILD_EXAMPLES}
    -DLLAMA_BUILD_TOOLS=${LLAMA_BUILD_TOOLS}
    -DLLAMA_BUILD_TESTS=${LLAMA_BUILD_TESTS}
    -DLLAMA_BUILD_SERVER=${LLAMA_BUILD_SERVER}
    -DGGML_METAL_EMBED_LIBRARY=${GGML_METAL_EMBED_LIBRARY}
    -DGGML_BLAS_DEFAULT=${GGML_BLAS_DEFAULT}
    -DGGML_METAL=${GGML_METAL}
    -DGGML_METAL_USE_BF16=${GGML_METAL_USE_BF16}
    -DGGML_NATIVE=OFF
    -DGGML_OPENMP=${GGML_OPENMP}
)

XCODE_VERSION=$(xcodebuild -version 2>/dev/null | head -n1 | awk '{ print $2 }')
MAJOR_VERSION=$(echo $XCODE_VERSION | cut -d. -f1)
MINOR_VERSION=$(echo $XCODE_VERSION | cut -d. -f2)
echo "Detected Xcode version: $XCODE_VERSION"

check_required_tool() {
    local tool=$1
    local install_message=$2

    if ! command -v $tool &> /dev/null; then
        echo "Error: $tool is required but not found."
        echo "$install_message"
        exit 1
    fi
}
echo "Checking for required tools..."
check_required_tool "cmake" "Please install CMake 3.28.0 or later (brew install cmake)"
check_required_tool "xcodebuild" "Please install Xcode and Xcode Command Line Tools (xcode-select --install)"
check_required_tool "libtool" "Please install libtool which should be available with Xcode Command Line Tools (CLT). Make sure Xcode CLT is installed (xcode-select --install)"
check_required_tool "dsymutil" "Please install Xcode and Xcode Command Line Tools (xcode-select --install)"

set -e

## Clean up previous builds
rm -rf build-apple
rm -rf build-ios-sim
rm -rf build-ios-device
rm -rf build-macos
rm -rf build-visionos
rm -rf build-visionos-sim
rm -rf build-tvos-sim
rm -rf build-tvos-device

# Setup the xcframework build directory structure
setup_framework_structure() {
    local build_dir=$1
    local min_os_version=$2
    local platform=$3  # "ios", "macos", "visionos", or "tvos"
    local framework_name="llama"

    echo "Creating ${platform}-style framework structure for ${build_dir}"

    if [[ "$platform" == "macos" ]]; then
        # macOS versioned structure uses versioned directories
        mkdir -p ${build_dir}/framework/${framework_name}.framework/Versions/A/Headers
        mkdir -p ${build_dir}/framework/${framework_name}.framework/Versions/A/Modules
        mkdir -p ${build_dir}/framework/${framework_name}.framework/Versions/A/Resources

        # Create symbolic links
        ln -sf A ${build_dir}/framework/${framework_name}.framework/Versions/Current
        ln -sf Versions/Current/Headers ${build_dir}/framework/${framework_name}.framework/Headers
        ln -sf Versions/Current/Modules ${build_dir}/framework/${framework_name}.framework/Modules
        ln -sf Versions/Current/Resources ${build_dir}/framework/${framework_name}.framework/Resources
        ln -sf Versions/Current/${framework_name} ${build_dir}/framework/${framework_name}.framework/${framework_name}

        # Set header and module paths
        local header_path=${build_dir}/framework/${framework_name}.framework/Versions/A/Headers/
        local module_path=${build_dir}/framework/${framework_name}.framework/Versions/A/Modules/
    else
        # iOS/VisionOS/tvOS use a flat structure
        mkdir -p ${build_dir}/framework/${framework_name}.framework/Headers
        mkdir -p ${build_dir}/framework/${framework_name}.framework/Modules

        # Remove any existing structure to ensure clean build
        rm -rf ${build_dir}/framework/${framework_name}.framework/Versions

        # Set header and module paths
        local header_path=${build_dir}/framework/${framework_name}.framework/Headers/
        local module_path=${build_dir}/framework/${framework_name}.framework/Modules/
    fi

    # Copy all required headers (common for all platforms)
    cp include/llama.h             ${header_path}
    cp ggml/include/ggml.h         ${header_path}
    cp ggml/include/ggml-opt.h     ${header_path}
    cp ggml/include/ggml-alloc.h   ${header_path}
    cp ggml/include/ggml-backend.h ${header_path}
    cp ggml/include/ggml-metal.h   ${header_path}
    cp ggml/include/ggml-cpu.h     ${header_path}
    cp ggml/include/ggml-blas.h    ${header_path}
    cp ggml/include/gguf.h         ${header_path}

    # Create module map (common for all platforms)
    cat > ${module_path}module.modulemap << EOF
framework module llama {
    header "llama.h"
    header "ggml.h"
    header "ggml-alloc.h"
    header "ggml-backend.h"
    header "ggml-metal.h"
    header "ggml-cpu.h"
    header "ggml-blas.h"
    header "gguf.h"

    link "c++"
    link framework "Accelerate"
    link framework "Metal"
    link framework "Foundation"

    export *
}
EOF

    # Platform-specific settings for Info.plist
    local platform_name=""
    local sdk_name=""
    local supported_platform=""

    case "$platform" in
        "ios")
            platform_name="iphoneos"
            sdk_name="iphoneos${min_os_version}"
            supported_platform="iPhoneOS"
            local plist_path="${build_dir}/framework/${framework_name}.framework/Info.plist"
            local device_family='    <key>UIDeviceFamily</key>
    <array>
        <integer>1</integer>
        <integer>2</integer>
    </array>'
            ;;
        "macos")
            platform_name="macosx"
            sdk_name="macosx${min_os_version}"
            supported_platform="MacOSX"
            local plist_path="${build_dir}/framework/${framework_name}.framework/Versions/A/Resources/Info.plist"
            local device_family=""
            ;;
        "visionos")
            platform_name="xros"
            sdk_name="xros${min_os_version}"
            supported_platform="XRPlatform"
            local plist_path="${build_dir}/framework/${framework_name}.framework/Info.plist"
            local device_family=""
            ;;
        "tvos")
            platform_name="appletvos"
            sdk_name="appletvos${min_os_version}"
            supported_platform="AppleTVOS"
            local plist_path="${build_dir}/framework/${framework_name}.framework/Info.plist"
            local device_family='    <key>UIDeviceFamily</key>
    <array>
        <integer>3</integer>
    </array>'
            ;;
    esac

    # Create Info.plist
    cat > ${plist_path} << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>llama</string>
    <key>CFBundleIdentifier</key>
    <string>org.ggml.llama</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>llama</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>MinimumOSVersion</key>
    <string>${min_os_version}</string>
    <key>CFBundleSupportedPlatforms</key>
    <array>
        <string>${supported_platform}</string>
    </array>${device_family}
    <key>DTPlatformName</key>
    <string>${platform_name}</string>
    <key>DTSDKName</key>
    <string>${sdk_name}</string>
</dict>
</plist>
EOF
}

# Create dynamic libraries from static libraries.
combine_static_libraries() {
    local build_dir="$1"
    local release_dir="$2"
    local platform="$3"  # "ios", "macos", "visionos", or "tvos"
    local is_simulator="$4"
    local base_dir="$(pwd)"
    local framework_name="llama"

    # Determine output path based on platform
    local output_lib=""
    if [[ "$platform" == "macos" ]]; then
        # macOS uses versioned structure
        output_lib="${build_dir}/framework/${framework_name}.framework/Versions/A/${framework_name}"
    else
        # iOS, visionOS, and tvOS use a directory flat structure
        output_lib="${build_dir}/framework/${framework_name}.framework/${framework_name}"
    fi

    local libs=(
        "${base_dir}/${build_dir}/src/${release_dir}/libllama.a"
        "${base_dir}/${build_dir}/ggml/src/${release_dir}/libggml.a"
        "${base_dir}/${build_dir}/ggml/src/${release_dir}/libggml-base.a"
        "${base_dir}/${build_dir}/ggml/src/${release_dir}/libggml-cpu.a"
        "${base_dir}/${build_dir}/ggml/src/ggml-metal/${release_dir}/libggml-metal.a"
        "${base_dir}/${build_dir}/ggml/src/ggml-blas/${release_dir}/libggml-blas.a"
    )

    # Create temporary directory for processing
    local temp_dir="${base_dir}/${build_dir}/temp"
    mkdir -p "${temp_dir}"

    # Since we have multiple architectures libtool will find object files that do not
    # match the target architecture. We suppress these warnings.
    libtool -static -o "${temp_dir}/combined.a" "${libs[@]}" 2> /dev/null

    # Determine SDK, architectures, and install_name based on platform and simulator flag.
    local sdk=""
    local archs=""
    local min_version_flag=""
    local install_name=""

    case "$platform" in
        "ios")
            if [[ "$is_simulator" == "true" ]]; then
                sdk="iphonesimulator"
                archs="arm64 x86_64"
                min_version_flag="-mios-simulator-version-min=${IOS_MIN_OS_VERSION}"
            else
                sdk="iphoneos"
                archs="arm64"
                min_version_flag="-mios-version-min=${IOS_MIN_OS_VERSION}"
            fi
            install_name="@rpath/llama.framework/llama"
            ;;
        "macos")
            sdk="macosx"
            archs="arm64 x86_64"
            min_version_flag="-mmacosx-version-min=${MACOS_MIN_OS_VERSION}"
            install_name="@rpath/llama.framework/Versions/Current/llama"
            ;;
        "visionos")
            if [[ "$is_simulator" == "true" ]]; then
                sdk="xrsimulator"
                archs="arm64 x86_64"
                min_version_flag="-mtargetos=xros${VISIONOS_MIN_OS_VERSION}-simulator"
            else
                sdk="xros"
                archs="arm64"
                min_version_flag="-mtargetos=xros${VISIONOS_MIN_OS_VERSION}"
            fi
            # Use flat structure for visionOS, same as iOS
            install_name="@rpath/llama.framework/llama"
            ;;
        "tvos")
            if [[ "$is_simulator" == "true" ]]; then
                sdk="appletvsimulator"
                archs="arm64 x86_64"
                min_version_flag="-mtvos-simulator-version-min=${TVOS_MIN_OS_VERSION}"
            else
                sdk="appletvos"
                archs="arm64"
                min_version_flag="-mtvos-version-min=${TVOS_MIN_OS_VERSION}"
            fi
            install_name="@rpath/llama.framework/llama"
            ;;
    esac

    # Build architecture flags
    local arch_flags=""
    for arch in $archs; do
        arch_flags+=" -arch $arch"
    done

    # Create dynamic library
    echo "Creating dynamic library for ${platform}."
    xcrun -sdk $sdk clang++ -dynamiclib \
        -isysroot $(xcrun --sdk $sdk --show-sdk-path) \
        $arch_flags \
        $min_version_flag \
        -Wl,-force_load,"${temp_dir}/combined.a" \
        -framework Foundation -framework Metal -framework Accelerate \
        -install_name "$install_name" \
        -o "${base_dir}/${output_lib}"

    # Platform-specific post-processing for device builds
    if [[ "$is_simulator" == "false" ]]; then
        if command -v xcrun vtool &>/dev/null; then
            case "$platform" in
                "ios")
                    echo "Marking binary as a framework binary for iOS..."
                    xcrun vtool -set-build-version ios ${IOS_MIN_OS_VERSION} ${IOS_MIN_OS_VERSION} -replace \
                        -output "${base_dir}/${output_lib}" "${base_dir}/${output_lib}"
                    ;;
                "visionos")
                    echo "Marking binary as a framework binary for visionOS..."
                    if [[ "$MAJOR_VERSION" -gt 16 ]] || [[ "$MAJOR_VERSION" -eq 16 && "$MINOR_VERSION" -gt 2 ]]; then
                        echo "Xcode version greater than 16.2, using visionOS."
                        VISION_OS_BUILD_VERSION="visionos"
                    else
                        echo "Xcode version less than or equal to 16.2, using xros."
                        VISION_OS_BUILD_VERSION="xros"
                    fi
                    xcrun vtool -set-build-version ${VISION_OS_BUILD_VERSION} ${VISIONOS_MIN_OS_VERSION} ${VISIONOS_MIN_OS_VERSION} -replace \
                        -output "${base_dir}/${output_lib}" "${base_dir}/${output_lib}"
                    ;;
                "tvos")
                    echo "Marking binary as a framework binary for tvOS..."
                    xcrun vtool -set-build-version tvos ${TVOS_MIN_OS_VERSION} ${TVOS_MIN_OS_VERSION} -replace \
                        -output "${base_dir}/${output_lib}" "${base_dir}/${output_lib}"
                    ;;
            esac
        else
            echo "Warning: vtool not found. Binary may not pass App Store validation."
        fi
    fi

    echo "Creating properly formatted dSYM..."
    # Create a separate directory for dSYMs for all platforms
    mkdir -p "${base_dir}/${build_dir}/dSYMs"

    # iOS and visionOS style dSYM (flat structure)
    if [[ "$platform" == "ios" || "$platform" == "visionos" || "$platform" == "tvos" ]]; then
        # Generate dSYM in the dSYMs directory
        xcrun dsymutil "${base_dir}/${output_lib}" -o "${base_dir}/${build_dir}/dSYMs/llama.dSYM"

        # Create a copy of the binary that will be stripped
        cp "${base_dir}/${output_lib}" "${temp_dir}/binary_to_strip"

        # Strip debug symbols from the copy
        xcrun strip -S "${temp_dir}/binary_to_strip" -o "${temp_dir}/stripped_lib"

        # Replace the original with the stripped version
        mv "${temp_dir}/stripped_lib" "${base_dir}/${output_lib}"
    else
        # macOS style dSYM
        # First strip debug info to a separate file
        xcrun strip -S "${base_dir}/${output_lib}" -o "${temp_dir}/stripped_lib"

        # Generate dSYM in the dSYMs directory
        xcrun dsymutil "${base_dir}/${output_lib}" -o "${base_dir}/${build_dir}/dSYMs/llama.dSYM"

        # Replace original binary with stripped version
        mv "${temp_dir}/stripped_lib" "${base_dir}/${output_lib}"
    fi

    # Remove any automatically generated dSYM files in the framework structure as they will
    # otherwise case Invalid Bundle Structure validation errors.
    if [ -d "${base_dir}/${output_lib}.dSYM" ]; then
        echo "Removing generated dSYM file in framework structure: ${base_dir}/${output_lib}.dSYM"
        rm -rf "${base_dir}/${output_lib}.dSYM"
    fi

    # Clean up
    rm -rf "${temp_dir}"
}

echo "Building for iOS simulator..."
cmake -B build-ios-sim -G Xcode \
    "${COMMON_CMAKE_ARGS[@]}" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${IOS_MIN_OS_VERSION} \
    -DIOS=ON \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_SYSROOT=iphonesimulator \
    -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
    -DCMAKE_XCODE_ATTRIBUTE_SUPPORTED_PLATFORMS=iphonesimulator \
    -DCMAKE_C_FLAGS="${COMMON_C_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${COMMON_CXX_FLAGS}" \
    -DLLAMA_CURL=OFF \
    -S .
cmake --build build-ios-sim --config Release -- -quiet

echo "Building for iOS devices..."
cmake -B build-ios-device -G Xcode \
    "${COMMON_CMAKE_ARGS[@]}" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${IOS_MIN_OS_VERSION} \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_SYSROOT=iphoneos \
    -DCMAKE_OSX_ARCHITECTURES="arm64" \
    -DCMAKE_XCODE_ATTRIBUTE_SUPPORTED_PLATFORMS=iphoneos \
    -DCMAKE_C_FLAGS="${COMMON_C_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${COMMON_CXX_FLAGS}" \
    -DLLAMA_CURL=OFF \
    -S .
cmake --build build-ios-device --config Release -- -quiet

echo "Building for macOS..."
cmake -B build-macos -G Xcode \
    "${COMMON_CMAKE_ARGS[@]}" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOS_MIN_OS_VERSION} \
    -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
    -DCMAKE_C_FLAGS="${COMMON_C_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${COMMON_CXX_FLAGS}" \
    -DLLAMA_CURL=OFF \
    -S .
cmake --build build-macos --config Release -- -quiet

echo "Building for visionOS..."
cmake -B build-visionos -G Xcode \
    "${COMMON_CMAKE_ARGS[@]}" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${VISIONOS_MIN_OS_VERSION} \
    -DCMAKE_OSX_ARCHITECTURES="arm64" \
    -DCMAKE_SYSTEM_NAME=visionOS \
    -DCMAKE_OSX_SYSROOT=xros \
    -DCMAKE_XCODE_ATTRIBUTE_SUPPORTED_PLATFORMS=xros \
    -DCMAKE_C_FLAGS="-D_XOPEN_SOURCE=700 ${COMMON_C_FLAGS}" \
    -DCMAKE_CXX_FLAGS="-D_XOPEN_SOURCE=700 ${COMMON_CXX_FLAGS}" \
    -DLLAMA_CURL=OFF \
    -DLLAMA_HTTPLIB=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -S .
cmake --build build-visionos --config Release -- -quiet

echo "Building for visionOS simulator..."
cmake -B build-visionos-sim -G Xcode \
    "${COMMON_CMAKE_ARGS[@]}" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${VISIONOS_MIN_OS_VERSION} \
    -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
    -DCMAKE_SYSTEM_NAME=visionOS \
    -DCMAKE_OSX_SYSROOT=xrsimulator \
    -DCMAKE_XCODE_ATTRIBUTE_SUPPORTED_PLATFORMS=xrsimulator \
    -DCMAKE_C_FLAGS="-D_XOPEN_SOURCE=700 ${COMMON_C_FLAGS}" \
    -DCMAKE_CXX_FLAGS="-D_XOPEN_SOURCE=700 ${COMMON_CXX_FLAGS}" \
    -DLLAMA_CURL=OFF \
    -DLLAMA_HTTPLIB=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -S .
cmake --build build-visionos-sim --config Release -- -quiet

# Add tvOS builds (might need the same u_int definitions as watchOS and visionOS)
echo "Building for tvOS simulator..."
cmake -B build-tvos-sim -G Xcode \
    "${COMMON_CMAKE_ARGS[@]}" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${TVOS_MIN_OS_VERSION} \
    -DCMAKE_SYSTEM_NAME=tvOS \
    -DCMAKE_OSX_SYSROOT=appletvsimulator \
    -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
    -DGGML_METAL=ON \
    -DCMAKE_XCODE_ATTRIBUTE_SUPPORTED_PLATFORMS=appletvsimulator \
    -DCMAKE_C_FLAGS="${COMMON_C_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${COMMON_CXX_FLAGS}" \
    -DLLAMA_CURL=OFF \
    -S .
cmake --build build-tvos-sim --config Release -- -quiet

echo "Building for tvOS devices..."
cmake -B build-tvos-device -G Xcode \
    "${COMMON_CMAKE_ARGS[@]}" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${TVOS_MIN_OS_VERSION} \
    -DCMAKE_SYSTEM_NAME=tvOS \
    -DCMAKE_OSX_SYSROOT=appletvos \
    -DCMAKE_OSX_ARCHITECTURES="arm64" \
    -DGGML_METAL=ON \
    -DCMAKE_XCODE_ATTRIBUTE_SUPPORTED_PLATFORMS=appletvos \
    -DCMAKE_C_FLAGS="${COMMON_C_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${COMMON_CXX_FLAGS}" \
    -DLLAMA_CURL=OFF \
    -S .
cmake --build build-tvos-device --config Release -- -quiet

# Setup frameworks and copy binaries and headers
echo "Setting up framework structures..."
setup_framework_structure "build-ios-sim" ${IOS_MIN_OS_VERSION} "ios"
setup_framework_structure "build-ios-device" ${IOS_MIN_OS_VERSION} "ios"
setup_framework_structure "build-macos" ${MACOS_MIN_OS_VERSION} "macos"
setup_framework_structure "build-visionos" ${VISIONOS_MIN_OS_VERSION} "visionos"
setup_framework_structure "build-visionos-sim" ${VISIONOS_MIN_OS_VERSION} "visionos"
setup_framework_structure "build-tvos-sim" ${TVOS_MIN_OS_VERSION} "tvos"
setup_framework_structure "build-tvos-device" ${TVOS_MIN_OS_VERSION} "tvos"

# Create dynamic libraries from static libraries
echo "Creating dynamic libraries from static libraries..."
combine_static_libraries "build-ios-sim" "Release-iphonesimulator" "ios" "true"
combine_static_libraries "build-ios-device" "Release-iphoneos" "ios" "false"
combine_static_libraries "build-macos" "Release" "macos" "false"
combine_static_libraries "build-visionos" "Release-xros" "visionos" "false"
combine_static_libraries "build-visionos-sim" "Release-xrsimulator" "visionos" "true"
combine_static_libraries "build-tvos-sim" "Release-appletvsimulator" "tvos" "true"
combine_static_libraries "build-tvos-device" "Release-appletvos" "tvos" "false"

# Create XCFramework with correct debug symbols paths
echo "Creating XCFramework..."
xcodebuild -create-xcframework \
    -framework $(pwd)/build-ios-sim/framework/llama.framework \
    -debug-symbols $(pwd)/build-ios-sim/dSYMs/llama.dSYM \
    -framework $(pwd)/build-ios-device/framework/llama.framework \
    -debug-symbols $(pwd)/build-ios-device/dSYMs/llama.dSYM \
    -framework $(pwd)/build-macos/framework/llama.framework \
    -debug-symbols $(pwd)/build-macos/dSYMS/llama.dSYM \
    -framework $(pwd)/build-visionos/framework/llama.framework \
    -debug-symbols $(pwd)/build-visionos/dSYMs/llama.dSYM \
    -framework $(pwd)/build-visionos-sim/framework/llama.framework \
    -debug-symbols $(pwd)/build-visionos-sim/dSYMs/llama.dSYM \
    -framework $(pwd)/build-tvos-device/framework/llama.framework \
    -debug-symbols $(pwd)/build-tvos-device/dSYMs/llama.dSYM \
    -framework $(pwd)/build-tvos-sim/framework/llama.framework \
    -debug-symbols $(pwd)/build-tvos-sim/dSYMs/llama.dSYM \
    -output $(pwd)/build-apple/llama.xcframework
