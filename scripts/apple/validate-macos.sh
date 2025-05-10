#!/bin/bash
# validate-macos.sh - Validate macOS Application with embedded llama.xcframework using SwiftUI

# Authentication options (optional) (can be set via environment variables)
# To use: export APPLE_ID=your.email@example.com
#         export APPLE_PASSWORD=your-app-specific-password
#         ./validate-macos.sh
APPLE_ID=${APPLE_ID:-""}
APPLE_PASSWORD=${APPLE_PASSWORD:-""}

# Ensure the script exits on error
set -e

# Function to print usage instructions
print_usage() {
  echo "Usage: ./validate-macos.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --help                 Show this help message"
  echo "  --apple-id EMAIL       Apple ID email for validation"
  echo "  --apple-password PWD   App-specific password for Apple ID"
  echo ""
  echo "Environment variables:"
  echo "  APPLE_ID               Apple ID email for validation"
  echo "  APPLE_PASSWORD         App-specific password for Apple ID"
  echo ""
  echo "Notes:"
  echo "  - Command line options take precedence over environment variables"
  echo "  - Authentication is optional. If not provided, alternative validation will be performed"
  echo "  - For APPLE_PASSWORD, use an app-specific password generated at https://appleid.apple.com/account/manage"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --help)
      print_usage
      exit 0
      ;;
    --apple-id)
      APPLE_ID="$2"
      shift 2
      ;;
    --apple-password)
      APPLE_PASSWORD="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      print_usage
      exit 1
      ;;
  esac
done

# Function to clean up in case of error
cleanup() {
  # Don't clean up temp files on error to help with debugging
  echo "===== macOS Validation Process Failed ====="
  exit 1
}

# Set up trap to call cleanup function on error
trap cleanup ERR

set -e  # Exit on any error

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
BUILD_DIR="${ROOT_DIR}/validation-builds/ios"

# Configuration
APP_NAME="MacOSLlamaTest"
BUNDLE_ID="org.ggml.MacOSLlamaTest"
XCFRAMEWORK_PATH="${ROOT_DIR}/build-apple/llama.xcframework"
TEMP_DIR="${BUILD_DIR}/temp"
ARCHIVE_PATH="${BUILD_DIR}/${APP_NAME}.xcarchive"
APP_PATH="${BUILD_DIR}/${APP_NAME}.app"
ZIP_PATH="${BUILD_DIR}/${APP_NAME}.zip"
VALIDATION_DIR="${BUILD_DIR}/validation"

# Create necessary directories
mkdir -p "${BUILD_DIR}"
mkdir -p "${TEMP_DIR}"
mkdir -p "${VALIDATION_DIR}"

echo "===== macOS Validation Process Started ====="

# 1. Create a simple test app project
echo "Creating test macOS app project..."
mkdir -p "${TEMP_DIR}/${APP_NAME}/${APP_NAME}"
cat > "${TEMP_DIR}/${APP_NAME}/${APP_NAME}/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>${APP_NAME}</string>
    <key>CFBundleIdentifier</key>
    <string>${BUNDLE_ID}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSHumanReadableCopyright</key>
    <string>Copyright © 2025 GGML. All rights reserved.</string>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
</dict>
</plist>
EOF

# Create SwiftUI app files
mkdir -p "${TEMP_DIR}/${APP_NAME}/${APP_NAME}/Sources"

# Create App.swift
cat > "${TEMP_DIR}/${APP_NAME}/${APP_NAME}/Sources/App.swift" << EOF
import SwiftUI
import llama

@main
struct LlamaTestApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
EOF

# Create ContentView.swift with macOS specific elements
cat > "${TEMP_DIR}/${APP_NAME}/${APP_NAME}/Sources/ContentView.swift" << EOF
import SwiftUI
import llama

struct ContentView: View {
    // Test that we can initialize a llama context params struct
    let params = llama_context_default_params()

    var body: some View {
        VStack(spacing: 20) {
            Text("Llama Framework Test on macOS")
                .font(.largeTitle)
                .padding()

            Text("llama_context_default_params() created successfully")
                .font(.headline)
                .multilineTextAlignment(.center)
                .padding()

            // Display some param values to confirm the framework is working
            Text("n_ctx: \(params.n_ctx)")
                .font(.body)

            Text("n_batch: \(params.n_batch)")
                .font(.body)

            Spacer()
        }
        .padding()
        .frame(width: 600, height: 400)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
EOF

# Create project.pbxproj, fixing the framework search paths issues
mkdir -p "${TEMP_DIR}/${APP_NAME}/${APP_NAME}.xcodeproj"
cat > "${TEMP_DIR}/${APP_NAME}/${APP_NAME}.xcodeproj/project.pbxproj" << 'EOF'
// !$*UTF8*$!
{
    archiveVersion = 1;
    classes = {
    };
    objectVersion = 54;
    objects = {

/* Begin PBXBuildFile section */
        11111111111111111111111 /* App.swift in Sources */ = {isa = PBXBuildFile; fileRef = 22222222222222222222222; };
        33333333333333333333333 /* ContentView.swift in Sources */ = {isa = PBXBuildFile; fileRef = 44444444444444444444444; };
        55555555555555555555555 /* llama.xcframework in Frameworks */ = {isa = PBXBuildFile; fileRef = 66666666666666666666666; };
        77777777777777777777777 /* llama.xcframework in Embed Frameworks */ = {isa = PBXBuildFile; fileRef = 66666666666666666666666; };
/* End PBXBuildFile section */

/* Begin PBXCopyFilesBuildPhase section */
        88888888888888888888888 /* Embed Frameworks */ = {
            isa = PBXCopyFilesBuildPhase;
            buildActionMask = 2147483647;
            dstPath = "";
            dstSubfolderSpec = 10;
            files = (
                77777777777777777777777 /* llama.xcframework in Embed Frameworks */,
            );
            name = "Embed Frameworks";
            runOnlyForDeploymentPostprocessing = 0;
        };
/* End PBXCopyFilesBuildPhase section */

/* Begin PBXFileReference section */
EOF

# Continue with the project.pbxproj file, using the APP_NAME variable appropriately
cat >> "${TEMP_DIR}/${APP_NAME}/${APP_NAME}.xcodeproj/project.pbxproj" << EOF
        99999999999999999999999 /* ${APP_NAME}.app */ = {isa = PBXFileReference; explicitFileType = wrapper.application; includeInIndex = 0; path = "${APP_NAME}.app"; sourceTree = BUILT_PRODUCTS_DIR; };
        22222222222222222222222 /* App.swift */ = {isa = PBXFileReference; lastKnownFileType = sourcecode.swift; path = App.swift; sourceTree = "<group>"; };
        44444444444444444444444 /* ContentView.swift */ = {isa = PBXFileReference; lastKnownFileType = sourcecode.swift; path = ContentView.swift; sourceTree = "<group>"; };
        AAAAAAAAAAAAAAAAAAAAAAA /* Info.plist */ = {isa = PBXFileReference; lastKnownFileType = text.plist.xml; path = Info.plist; sourceTree = "<group>"; };
        66666666666666666666666 /* llama.xcframework */ = {isa = PBXFileReference; lastKnownFileType = wrapper.xcframework; path = llama.xcframework; sourceTree = "<group>"; };
/* End PBXFileReference section */
EOF

# Add the rest of the project file with fixed framework search paths
cat >> "${TEMP_DIR}/${APP_NAME}/${APP_NAME}.xcodeproj/project.pbxproj" << 'EOF'
/* Begin PBXFrameworksBuildPhase section */
        BBBBBBBBBBBBBBBBBBBBBBBB /* Frameworks */ = {
            isa = PBXFrameworksBuildPhase;
            buildActionMask = 2147483647;
            files = (
                55555555555555555555555 /* llama.xcframework in Frameworks */,
            );
            runOnlyForDeploymentPostprocessing = 0;
        };
/* End PBXFrameworksBuildPhase section */

/* Begin PBXGroup section */
EOF

# Continue with the project.pbxproj file, using the APP_NAME variable appropriately
cat >> "${TEMP_DIR}/${APP_NAME}/${APP_NAME}.xcodeproj/project.pbxproj" << EOF
        CCCCCCCCCCCCCCCCCCCCCCCC /* Products */ = {
            isa = PBXGroup;
            children = (
                99999999999999999999999 /* ${APP_NAME}.app */,
            );
            name = Products;
            sourceTree = "<group>";
        };
EOF

cat >> "${TEMP_DIR}/${APP_NAME}/${APP_NAME}.xcodeproj/project.pbxproj" << 'EOF'
        DDDDDDDDDDDDDDDDDDDDDDDD /* Frameworks */ = {
            isa = PBXGroup;
            children = (
                66666666666666666666666 /* llama.xcframework */,
            );
            name = Frameworks;
            sourceTree = "<group>";
        };
        EEEEEEEEEEEEEEEEEEEEEEEE = {
            isa = PBXGroup;
            children = (
                FFFFFFFFFFFFFFFFFFFFFFFF /* MacOSLlamaTest */,
                CCCCCCCCCCCCCCCCCCCCCCCC /* Products */,
                DDDDDDDDDDDDDDDDDDDDDDDD /* Frameworks */,
            );
            sourceTree = "<group>";
        };
        FFFFFFFFFFFFFFFFFFFFFFFF /* MacOSLlamaTest */ = {
            isa = PBXGroup;
            children = (
                1111111111111111111111AA /* Sources */,
                AAAAAAAAAAAAAAAAAAAAAAA /* Info.plist */,
            );
            path = "MacOSLlamaTest";
            sourceTree = "<group>";
        };
        1111111111111111111111AA /* Sources */ = {
            isa = PBXGroup;
            children = (
                22222222222222222222222 /* App.swift */,
                44444444444444444444444 /* ContentView.swift */,
            );
            path = Sources;
            sourceTree = "<group>";
        };
/* End PBXGroup section */
EOF

# Continue with the project.pbxproj file, using the APP_NAME variable appropriately
cat >> "${TEMP_DIR}/${APP_NAME}/${APP_NAME}.xcodeproj/project.pbxproj" << EOF
/* Begin PBXNativeTarget section */
        3333333333333333333333AA /* ${APP_NAME} */ = {
            isa = PBXNativeTarget;
            buildConfigurationList = 4444444444444444444444AA /* Build configuration list for PBXNativeTarget "${APP_NAME}" */;
            buildPhases = (
                5555555555555555555555AA /* Sources */,
                BBBBBBBBBBBBBBBBBBBBBBBB /* Frameworks */,
                6666666666666666666666AA /* Resources */,
                88888888888888888888888 /* Embed Frameworks */,
            );
            buildRules = (
            );
            dependencies = (
            );
            name = "${APP_NAME}";
            productName = "${APP_NAME}";
            productReference = 99999999999999999999999 /* ${APP_NAME}.app */;
            productType = "com.apple.product-type.application";
        };
/* End PBXNativeTarget section */

/* Begin PBXProject section */
        7777777777777777777777AA /* Project object */ = {
            isa = PBXProject;
            attributes = {
                LastSwiftUpdateCheck = 1240;
                LastUpgradeCheck = 1240;
                TargetAttributes = {
                    3333333333333333333333AA = {
                        CreatedOnToolsVersion = 12.4;
                    };
                };
            };
            buildConfigurationList = 8888888888888888888888AA /* Build configuration list for PBXProject "${APP_NAME}" */;
            compatibilityVersion = "Xcode 12.0";
            developmentRegion = en;
            hasScannedForEncodings = 0;
            knownRegions = (
                en,
                Base,
            );
            mainGroup = EEEEEEEEEEEEEEEEEEEEEEEE;
            productRefGroup = CCCCCCCCCCCCCCCCCCCCCCCC /* Products */;
            projectDirPath = "";
            projectRoot = "";
            targets = (
                3333333333333333333333AA /* ${APP_NAME} */,
            );
        };
/* End PBXProject section */
EOF

# Add the rest of the file with correct FRAMEWORK_SEARCH_PATHS and macOS settings
cat >> "${TEMP_DIR}/${APP_NAME}/${APP_NAME}.xcodeproj/project.pbxproj" << 'EOF'
/* Begin PBXResourcesBuildPhase section */
        6666666666666666666666AA /* Resources */ = {
            isa = PBXResourcesBuildPhase;
            buildActionMask = 2147483647;
            files = (
            );
            runOnlyForDeploymentPostprocessing = 0;
        };
/* End PBXResourcesBuildPhase section */

/* Begin PBXSourcesBuildPhase section */
        5555555555555555555555AA /* Sources */ = {
            isa = PBXSourcesBuildPhase;
            buildActionMask = 2147483647;
            files = (
                33333333333333333333333 /* ContentView.swift in Sources */,
                11111111111111111111111 /* App.swift in Sources */,
            );
            runOnlyForDeploymentPostprocessing = 0;
        };
/* End PBXSourcesBuildPhase section */

/* Begin XCBuildConfiguration section */
        9999999999999999999999AA /* Debug */ = {
            isa = XCBuildConfiguration;
            buildSettings = {
                ALWAYS_SEARCH_USER_PATHS = NO;
                CLANG_ANALYZER_NONNULL = YES;
                CLANG_ANALYZER_NUMBER_OBJECT_CONVERSION = YES_AGGRESSIVE;
                CLANG_CXX_LANGUAGE_STANDARD = "gnu++14";
                CLANG_CXX_LIBRARY = "libc++";
                CLANG_ENABLE_MODULES = YES;
                CLANG_ENABLE_OBJC_ARC = YES;
                CLANG_ENABLE_OBJC_WEAK = YES;
                CLANG_WARN_BLOCK_CAPTURE_AUTORELEASING = YES;
                CLANG_WARN_BOOL_CONVERSION = YES;
                CLANG_WARN_COMMA = YES;
                CLANG_WARN_CONSTANT_CONVERSION = YES;
                CLANG_WARN_DEPRECATED_OBJC_IMPLEMENTATIONS = YES;
                CLANG_WARN_DIRECT_OBJC_ISA_USAGE = YES_ERROR;
                CLANG_WARN_DOCUMENTATION_COMMENTS = YES;
                CLANG_WARN_EMPTY_BODY = YES;
                CLANG_WARN_ENUM_CONVERSION = YES;
                CLANG_WARN_INFINITE_RECURSION = YES;
                CLANG_WARN_INT_CONVERSION = YES;
                CLANG_WARN_NON_LITERAL_NULL_CONVERSION = YES;
                CLANG_WARN_OBJC_IMPLICIT_RETAIN_SELF = YES;
                CLANG_WARN_OBJC_LITERAL_CONVERSION = YES;
                CLANG_WARN_OBJC_ROOT_CLASS = YES_ERROR;
                CLANG_WARN_QUOTED_INCLUDE_IN_FRAMEWORK_HEADER = YES;
                CLANG_WARN_RANGE_LOOP_ANALYSIS = YES;
                CLANG_WARN_STRICT_PROTOTYPES = YES;
                CLANG_WARN_SUSPICIOUS_MOVE = YES;
                CLANG_WARN_UNGUARDED_AVAILABILITY = YES_AGGRESSIVE;
                CLANG_WARN_UNREACHABLE_CODE = YES;
                CLANG_WARN__DUPLICATE_METHOD_MATCH = YES;
                COPY_PHASE_STRIP = NO;
                DEBUG_INFORMATION_FORMAT = dwarf;
                ENABLE_STRICT_OBJC_MSGSEND = YES;
                ENABLE_TESTABILITY = YES;
                GCC_C_LANGUAGE_STANDARD = gnu11;
                GCC_DYNAMIC_NO_PIC = NO;
                GCC_NO_COMMON_BLOCKS = YES;
                GCC_OPTIMIZATION_LEVEL = 0;
                GCC_PREPROCESSOR_DEFINITIONS = (
                    "DEBUG=1",
                    "$(inherited)",
                );
                GCC_WARN_64_TO_32_BIT_CONVERSION = YES;
                GCC_WARN_ABOUT_RETURN_TYPE = YES_ERROR;
                GCC_WARN_UNDECLARED_SELECTOR = YES;
                GCC_WARN_UNINITIALIZED_AUTOS = YES_AGGRESSIVE;
                GCC_WARN_UNUSED_FUNCTION = YES;
                GCC_WARN_UNUSED_VARIABLE = YES;
                MACOSX_DEPLOYMENT_TARGET = 12.0;
                MTL_ENABLE_DEBUG_INFO = INCLUDE_SOURCE;
                MTL_FAST_MATH = YES;
                ONLY_ACTIVE_ARCH = YES;
                SDKROOT = macosx;
                SWIFT_ACTIVE_COMPILATION_CONDITIONS = DEBUG;
                SWIFT_OPTIMIZATION_LEVEL = "-Onone";
            };
            name = Debug;
        };
        AAAAAAAAAAAAAAAAAAAAABBB /* Release */ = {
            isa = XCBuildConfiguration;
            buildSettings = {
                ALWAYS_SEARCH_USER_PATHS = NO;
                CLANG_ANALYZER_NONNULL = YES;
                CLANG_ANALYZER_NUMBER_OBJECT_CONVERSION = YES_AGGRESSIVE;
                CLANG_CXX_LANGUAGE_STANDARD = "gnu++14";
                CLANG_CXX_LIBRARY = "libc++";
                CLANG_ENABLE_MODULES = YES;
                CLANG_ENABLE_OBJC_ARC = YES;
                CLANG_ENABLE_OBJC_WEAK = YES;
                CLANG_WARN_BLOCK_CAPTURE_AUTORELEASING = YES;
                CLANG_WARN_BOOL_CONVERSION = YES;
                CLANG_WARN_COMMA = YES;
                CLANG_WARN_CONSTANT_CONVERSION = YES;
                CLANG_WARN_DEPRECATED_OBJC_IMPLEMENTATIONS = YES;
                CLANG_WARN_DIRECT_OBJC_ISA_USAGE = YES_ERROR;
                CLANG_WARN_DOCUMENTATION_COMMENTS = YES;
                CLANG_WARN_EMPTY_BODY = YES;
                CLANG_WARN_ENUM_CONVERSION = YES;
                CLANG_WARN_INFINITE_RECURSION = YES;
                CLANG_WARN_INT_CONVERSION = YES;
                CLANG_WARN_NON_LITERAL_NULL_CONVERSION = YES;
                CLANG_WARN_OBJC_IMPLICIT_RETAIN_SELF = YES;
                CLANG_WARN_OBJC_LITERAL_CONVERSION = YES;
                CLANG_WARN_OBJC_ROOT_CLASS = YES_ERROR;
                CLANG_WARN_QUOTED_INCLUDE_IN_FRAMEWORK_HEADER = YES;
                CLANG_WARN_RANGE_LOOP_ANALYSIS = YES;
                CLANG_WARN_STRICT_PROTOTYPES = YES;
                CLANG_WARN_SUSPICIOUS_MOVE = YES;
                CLANG_WARN_UNGUARDED_AVAILABILITY = YES_AGGRESSIVE;
                CLANG_WARN_UNREACHABLE_CODE = YES;
                CLANG_WARN__DUPLICATE_METHOD_MATCH = YES;
                COPY_PHASE_STRIP = NO;
                DEBUG_INFORMATION_FORMAT = "dwarf-with-dsym";
                ENABLE_NS_ASSERTIONS = NO;
                ENABLE_STRICT_OBJC_MSGSEND = YES;
                GCC_C_LANGUAGE_STANDARD = gnu11;
                GCC_NO_COMMON_BLOCKS = YES;
                GCC_WARN_64_TO_32_BIT_CONVERSION = YES;
                GCC_WARN_ABOUT_RETURN_TYPE = YES_ERROR;
                GCC_WARN_UNDECLARED_SELECTOR = YES;
                GCC_WARN_UNINITIALIZED_AUTOS = YES_AGGRESSIVE;
                GCC_WARN_UNUSED_FUNCTION = YES;
                GCC_WARN_UNUSED_VARIABLE = YES;
                MACOSX_DEPLOYMENT_TARGET = 12.0;
                MTL_ENABLE_DEBUG_INFO = NO;
                MTL_FAST_MATH = YES;
                SDKROOT = macosx;
                SWIFT_COMPILATION_MODE = wholemodule;
                SWIFT_OPTIMIZATION_LEVEL = "-O";
            };
            name = Release;
        };
        BBBBBBBBBBBBBBBBBBBBBBCCC /* Debug */ = {
            isa = XCBuildConfiguration;
            buildSettings = {
                ASSETCATALOG_COMPILER_APPICON_NAME = AppIcon;
                ASSETCATALOG_COMPILER_GLOBAL_ACCENT_COLOR_NAME = AccentColor;
                CODE_SIGN_STYLE = Manual;
                COMBINE_HIDPI_IMAGES = YES;
                DEVELOPMENT_TEAM = "";
                ENABLE_HARDENED_RUNTIME = YES;
                ENABLE_PREVIEWS = YES;
                FRAMEWORK_SEARCH_PATHS = "$(PROJECT_DIR)";
                INFOPLIST_FILE = "MacOSLlamaTest/Info.plist";
                LD_RUNPATH_SEARCH_PATHS = (
                    "$(inherited)",
                    "@executable_path/../Frameworks",
                );
                PRODUCT_BUNDLE_IDENTIFIER = "org.ggml.MacOSLlamaTest";
                PRODUCT_NAME = "$(TARGET_NAME)";
                PROVISIONING_PROFILE_SPECIFIER = "";
                SWIFT_VERSION = 5.0;
            };
            name = Debug;
        };
        CCCCCCCCCCCCCCCCCCCCCCDDD /* Release */ = {
            isa = XCBuildConfiguration;
            buildSettings = {
                ASSETCATALOG_COMPILER_APPICON_NAME = AppIcon;
                ASSETCATALOG_COMPILER_GLOBAL_ACCENT_COLOR_NAME = AccentColor;
                CODE_SIGN_STYLE = Manual;
                COMBINE_HIDPI_IMAGES = YES;
                DEVELOPMENT_TEAM = "";
                ENABLE_HARDENED_RUNTIME = YES;
                ENABLE_PREVIEWS = YES;
                FRAMEWORK_SEARCH_PATHS = (
                    "$(inherited)",
                    "$(PROJECT_DIR)",
                );
                INFOPLIST_FILE = "MacOSLlamaTest/Info.plist";
                LD_RUNPATH_SEARCH_PATHS = (
                    "$(inherited)",
                    "@executable_path/../Frameworks",
                );
                PRODUCT_BUNDLE_IDENTIFIER = "org.ggml.MacOSLlamaTest";
                PRODUCT_NAME = "$(TARGET_NAME)";
                PROVISIONING_PROFILE_SPECIFIER = "";
                SWIFT_VERSION = 5.0;
            };
            name = Release;
        };
/* End XCBuildConfiguration section */
EOF

# Finish the project.pbxproj file
cat >> "${TEMP_DIR}/${APP_NAME}/${APP_NAME}.xcodeproj/project.pbxproj" << EOF
/* Begin XCConfigurationList section */
        8888888888888888888888AA /* Build configuration list for PBXProject "${APP_NAME}" */ = {
            isa = XCConfigurationList;
            buildConfigurations = (
                9999999999999999999999AA /* Debug */,
                AAAAAAAAAAAAAAAAAAAAABBB /* Release */,
            );
            defaultConfigurationIsVisible = 0;
            defaultConfigurationName = Release;
        };
        4444444444444444444444AA /* Build configuration list for PBXNativeTarget "${APP_NAME}" */ = {
            isa = XCConfigurationList;
            buildConfigurations = (
                BBBBBBBBBBBBBBBBBBBBBBCCC /* Debug */,
                CCCCCCCCCCCCCCCCCCCCCCDDD /* Release */,
            );
            defaultConfigurationIsVisible = 0;
            defaultConfigurationName = Release;
        };
/* End XCConfigurationList section */
    };
    rootObject = 7777777777777777777777AA /* Project object */;
}
EOF

# 2. Copy XCFramework to test project
echo "Copying XCFramework to test project..."
cp -R "${XCFRAMEWORK_PATH}" "${TEMP_DIR}/${APP_NAME}/"

# 3. Build and archive the app
echo "Building and archiving test app..."
cd "${TEMP_DIR}/${APP_NAME}"

# Create a simple xcscheme file to avoid xcodebuild scheme issues
mkdir -p "${APP_NAME}.xcodeproj/xcshareddata/xcschemes"
cat > "${APP_NAME}.xcodeproj/xcshareddata/xcschemes/${APP_NAME}.xcscheme" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<Scheme
   LastUpgradeVersion = "1240"
   version = "1.3">
   <BuildAction
      parallelizeBuildables = "YES"
      buildImplicitDependencies = "YES">
      <BuildActionEntries>
         <BuildActionEntry
            buildForTesting = "YES"
            buildForRunning = "YES"
            buildForProfiling = "YES"
            buildForArchiving = "YES"
            buildForAnalyzing = "YES">
            <BuildableReference
               BuildableIdentifier = "primary"
               BlueprintIdentifier = "3333333333333333333333AA"
               BuildableName = "${APP_NAME}.app"
               BlueprintName = "${APP_NAME}"
               ReferencedContainer = "container:${APP_NAME}.xcodeproj">
            </BuildableReference>
         </BuildActionEntry>
      </BuildActionEntries>
   </BuildAction>
   <TestAction
      buildConfiguration = "Debug"
      selectedDebuggerIdentifier = "Xcode.DebuggerFoundation.Debugger.LLDB"
      selectedLauncherIdentifier = "Xcode.DebuggerFoundation.Launcher.LLDB"
      shouldUseLaunchSchemeArgsEnv = "YES">
      <Testables>
      </Testables>
   </TestAction>
   <LaunchAction
      buildConfiguration = "Debug"
      selectedDebuggerIdentifier = "Xcode.DebuggerFoundation.Debugger.LLDB"
      selectedLauncherIdentifier = "Xcode.DebuggerFoundation.Launcher.LLDB"
      launchStyle = "0"
      useCustomWorkingDirectory = "NO"
      ignoresPersistentStateOnLaunch = "NO"
      debugDocumentVersioning = "YES"
      debugServiceExtension = "internal"
      allowLocationSimulation = "YES">
      <BuildableProductRunnable
         runnableDebuggingMode = "0">
         <BuildableReference
            BuildableIdentifier = "primary"
            BlueprintIdentifier = "3333333333333333333333AA"
            BuildableName = "${APP_NAME}.app"
            BlueprintName = "${APP_NAME}"
            ReferencedContainer = "container:${APP_NAME}.xcodeproj">
         </BuildableReference>
      </BuildableProductRunnable>
   </LaunchAction>
   <ProfileAction
      buildConfiguration = "Release"
      shouldUseLaunchSchemeArgsEnv = "YES"
      savedToolIdentifier = ""
      useCustomWorkingDirectory = "NO"
      debugDocumentVersioning = "YES">
      <BuildableProductRunnable
         runnableDebuggingMode = "0">
         <BuildableReference
            BuildableIdentifier = "primary"
            BlueprintIdentifier = "3333333333333333333333AA"
            BuildableName = "${APP_NAME}.app"
            BlueprintName = "${APP_NAME}"
            ReferencedContainer = "container:${APP_NAME}.xcodeproj">
         </BuildableReference>
      </BuildableProductRunnable>
   </ProfileAction>
   <AnalyzeAction
      buildConfiguration = "Debug">
   </AnalyzeAction>
   <ArchiveAction
      buildConfiguration = "Release"
      revealArchiveInOrganizer = "YES">
   </ArchiveAction>
</Scheme>
EOF

# Now use xcodebuild with an explicitly defined product name for macOS
xcodebuild -project "${APP_NAME}.xcodeproj" -scheme "${APP_NAME}" -sdk macosx -configuration Release archive -archivePath "${ARCHIVE_PATH}" CODE_SIGN_IDENTITY="-" CODE_SIGNING_REQUIRED=NO CODE_SIGNING_ALLOWED=NO PRODUCT_NAME="${APP_NAME}" SWIFT_OPTIMIZATION_LEVEL="-Onone" -quiet

# 4. Create a package for distribution
echo "Creating distributable package from archive..."
cp -R "${ARCHIVE_PATH}/Products/Applications/${APP_NAME}.app" "${APP_PATH}"

# Check and log app structure
echo "App structure:"
ls -la "${APP_PATH}"
echo "Frameworks:"
ls -la "${APP_PATH}/Contents/Frameworks/" 2>/dev/null || echo "No Frameworks directory found"

# Create a zip file for potential distribution
cd "${BUILD_DIR}"
zip -r "${ZIP_PATH}" "${APP_NAME}.app"

# Check embedded provisioning profile
echo "Checking provisioning profile (if any)..."
PROVISIONING_PROFILE=$(find "${APP_PATH}/Contents" -name "embedded.provisionprofile" 2>/dev/null)
if [ -n "$PROVISIONING_PROFILE" ]; then
    echo "Found embedded provisioning profile:"
    security cms -D -i "$PROVISIONING_PROFILE" || echo "Unable to decode provisioning profile"
else
    echo "No embedded provisioning profile found (expected for ad-hoc builds)"
fi

# 5. Validate the app
echo "Validating macOS app..."
VALIDATION_OUTPUT="${VALIDATION_DIR}/validation_output.txt"

# Check if authentication credentials are provided
AUTH_ARGS=""
if [ -n "$APPLE_ID" ] && [ -n "$APPLE_PASSWORD" ]; then
    echo "Using Apple ID authentication for validation..."
    AUTH_ARGS="--username \"$APPLE_ID\" --password \"$APPLE_PASSWORD\""
else
    echo "No authentication credentials provided. Will perform basic validation."
    echo "To use your personal developer account, you can run the script with:"
    echo "  APPLE_ID='your.email@example.com' APPLE_PASSWORD='your-app-specific-password' ./validate-macos.sh"
    echo "Note: You need to create an app-specific password at https://appleid.apple.com/account/manage"
fi

# For macOS we need to use notarytool or alternative checks because altool doesn't support macOS apps in the same way
echo "Note: For macOS, formal notarization process would require Apple Developer credentials."
echo "Performing alternative validation checks..."

# Final validation result
FINAL_VALIDATION_RESULT=0

# Check if app was created successfully
if [ -d "${APP_PATH}" ] && [ -s "${APP_PATH}/Contents/MacOS/${APP_NAME}" ]; then
    echo "✅ App package created successfully"
else
    echo "❌ App package not created or binary missing"
    FINAL_VALIDATION_RESULT=1
fi

# Check if app binary exists and is executable
if [ -f "${APP_PATH}/Contents/MacOS/${APP_NAME}" ] && [ -x "${APP_PATH}/Contents/MacOS/${APP_NAME}" ]; then
    echo "✅ App binary exists and is executable"
else
    echo "❌ App binary missing or not executable"
    FINAL_VALIDATION_RESULT=1
fi

# Check if framework was properly embedded
if [ -d "${APP_PATH}/Contents/Frameworks/llama.framework" ]; then
    echo "✅ llama.framework properly embedded"
else
    echo "❌ llama.framework not properly embedded"
    FINAL_VALIDATION_RESULT=1
fi

# Check if framework binary exists
if [ -f "${APP_PATH}/Contents/Frameworks/llama.framework/Versions/A/llama" ]; then
    echo "✅ Framework binary exists"

    # Further validate framework by checking architecture
    ARCHS=$(lipo -info "${APP_PATH}/Contents/Frameworks/llama.framework/Versions/A/llama" 2>/dev/null | grep -o "arm64\\|x86_64" | tr '\n' ' ')
    if [ -n "$ARCHS" ]; then
        echo "✅ Framework architecture(s): $ARCHS"
    else
        echo "⚠️ Could not determine framework architecture"
    fi
else
    echo "❌ Framework binary missing"
    FINAL_VALIDATION_RESULT=1
fi

# Check code signing
echo ""
echo "==== CODE SIGNING INFO ===="
codesign -vv -d "${APP_PATH}" 2>&1 || echo "Code signing verification not available (expected for ad-hoc builds)"

if [ $FINAL_VALIDATION_RESULT -eq 0 ]; then
    if [ -n "$AUTH_ARGS" ]; then
        echo ""
        echo "To notarize this app with Apple (requires Apple Developer account):"
        echo "xcrun notarytool submit \"${ZIP_PATH}\" --apple-id \"your-apple-id\" --password \"your-app-specific-password\" --team-id \"your-team-id\" --wait"
        echo ""
    fi
    echo "✅ Validation PASSED: macOS app built successfully with embedded framework"
else
    echo "❌ Validation FAILED: Issues found with the app or framework"
fi

# Don't clean up on error to allow inspection
if [ $FINAL_VALIDATION_RESULT -ne 0 ]; then
    echo ""
    echo "Temporary files kept for inspection at: ${TEMP_DIR}"
    echo "===== macOS Validation Process Failed ====="
    exit 1
fi

# Clean up temporary files but keep build artifacts
if [ $FINAL_VALIDATION_RESULT -eq 0 ]; then
    echo "Cleaning up temporary files..."
    #rm -rf "${TEMP_DIR}"
fi

echo "===== macOS Validation Process Completed ====="
echo "App package available at: ${APP_PATH}"
echo "Zipped app available at: ${ZIP_PATH}"
exit $FINAL_VALIDATION_RESULT
