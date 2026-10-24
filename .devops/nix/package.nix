{
  lib,
  glibc,
  config,
  stdenv,
  stdenvNoCC,
  runCommand,
  cmake,
  ninja,
  pkg-config,
  git,
  mpi,
  blas,
  cudaPackages,
  autoAddDriverRunpath,
  darwin,
  rocmPackages,
  vulkan-headers,
  vulkan-loader,
  openssl,
  shaderc,
  spirv-headers,
  nodejs,
  importNpmLock,
  syclToolkit ? null,
  level-zero,
  ocl-icd,
  intel-graphics-compiler,
  intel-compute-runtime,
  makeBinaryWrapper,
  useBlas ?
    builtins.all (x: !x) [
      useCuda
      useMetalKit
      useRocm
      useVulkan
      useSycl
    ]
    && blas.meta.available,
  useCuda ? config.cudaSupport,
  useMetalKit ? stdenv.hostPlatform.isAarch64 && stdenv.hostPlatform.isDarwin,
  # Increases the runtime closure size by ~700M
  useMpi ? false,
  useRocm ? config.rocmSupport,
  rocmGpuTargets ? builtins.concatStringsSep ";" rocmPackages.clr.gpuTargets,
  useVulkan ? false,
  useRpc ? false,
  useSycl ? false,
  # Rebuild, not a runtime switch. cmake default is off.
  syclF16 ? false,
  # null keeps JIT. AOT makes icpx run ocloc, which DPC++ does not ship.
  syclDeviceArch ? null,
  syclMaxParallelLinkJobs ? null,
  llamaVersion ? "0.0.0", # Arbitrary version, substituted by the flake

  # CUDA needs its backend stdenv or libstdc++ mismatches downstream.
  # SYCL needs icpx which is provided by the oneAPI stdenv
  effectiveStdenv ?
    if useCuda && useSycl then
      throw "llama-cpp: useCuda and useSycl are mutually exclusive"
    else if useCuda then
      cudaPackages.backendStdenv
    else if useSycl then
      if syclToolkit == null then
        throw "llama-cpp: useSycl requires syclToolkit, consider setting it to intel-oneapi-toolkit"
      else
        syclToolkit.stdenv
    else
      stdenv,
  enableStatic ? effectiveStdenv.hostPlatform.isStatic,
  precompileMetalShaders ? false,
  useWebUi ? true,
}:

let
  inherit (lib)
    cmakeBool
    cmakeFeature
    optionalAttrs
    optionals
    strings
    ;

  stdenv = throw "Use effectiveStdenv instead";

  suffices =
    lib.optionals useBlas [ "BLAS" ]
    ++ lib.optionals useCuda [ "CUDA" ]
    ++ lib.optionals useMetalKit [ "MetalKit" ]
    ++ lib.optionals useMpi [ "MPI" ]
    ++ lib.optionals useRocm [ "ROCm" ]
    ++ lib.optionals useVulkan [ "Vulkan" ]
    ++ lib.optionals useSycl [ "SYCL" ];

  pnameSuffix =
    strings.optionalString (suffices != [ ])
      "-${strings.concatMapStringsSep "-" strings.toLower suffices}";
  descriptionSuffix = strings.optionalString (
    suffices != [ ]
  ) ", accelerated with ${strings.concatStringsSep ", " suffices}";

  xcrunHost = runCommand "xcrunHost" { } ''
    mkdir -p $out/bin
    ln -s /usr/bin/xcrun $out/bin
  '';

  # apple_sdk is supposed to choose sane defaults, no need to handle isAarch64
  # separately
  darwinBuildInputs =
    with darwin.apple_sdk.frameworks;
    [
      Accelerate
      CoreVideo
      CoreGraphics
    ]
    ++ optionals useMetalKit [ MetalKit ];

  cudaBuildInputs = with cudaPackages; [
    cuda_cudart
    cccl # <nv/target>
    libcublas
  ];

  rocmBuildInputs = with rocmPackages; [
    clr
    hipblas
    rocblas
  ];

  vulkanBuildInputs = [
    vulkan-headers
    vulkan-loader
    shaderc
    spirv-headers
  ];

  syclAot = useSycl && syclDeviceArch != null && syclDeviceArch != "";

  # source ocloc binary from intel-compute-runtime
  oclocAot = runCommand "ocloc-aot" { nativeBuildInputs = [ makeBinaryWrapper ]; } ''
    mkdir -p "$out/bin"
    shopt -s nullglob
    exact=(${lib.getBin intel-compute-runtime}/bin/ocloc)
    vers=(${lib.getBin intel-compute-runtime}/bin/ocloc-*)
    if [ ''${#exact[@]} -eq 1 ]; then
      src=''${exact[0]}
    elif [ ''${#vers[@]} -eq 1 ]; then
      src=''${vers[0]}
    else
      echo "expected one ocloc binary from intel-compute-runtime" >&2
      exit 1
    fi
    makeWrapper "$src" "$out/bin/ocloc" \
      --prefix LD_LIBRARY_PATH : ${lib.makeLibraryPath [ intel-graphics-compiler ]}
  '';
in

effectiveStdenv.mkDerivation (finalAttrs: {
  pname = "llama-cpp${pnameSuffix}";
  version = llamaVersion;

  # Note: none of the files discarded here are visible in the sandbox or
  # affect the output hash. This also means they can be modified without
  # triggering a rebuild.
  src = lib.cleanSourceWith {
    filter =
      name: type:
      let
        noneOf = builtins.all (x: !x);
        baseName = baseNameOf name;
      in
      noneOf [
        (lib.hasSuffix ".nix" name) # Ignore *.nix files when computing outPaths
        (lib.hasSuffix ".md" name) # Ignore *.md changes whe computing outPaths
        (lib.hasPrefix "." baseName) # Skip hidden files and directories
        (baseName == "flake.lock")
      ];
    src = lib.cleanSource ../../.;
  };

  # Builds the webui locally, taking care not to require updating any sha256 hash.
  webui = stdenvNoCC.mkDerivation {
    pname = "webui";
    version = llamaVersion;
    src = lib.cleanSource ../../tools/ui;

    nativeBuildInputs = [
      nodejs
      importNpmLock.linkNodeModulesHook
    ];

    # no sha256 required when using buildNodeModules
    npmDeps = importNpmLock.buildNodeModules {
      npmRoot = ../../tools/ui;
      inherit nodejs;
    };

    installPhase = ''
      LLAMA_UI_OUT_DIR=$out npm run build --offline
    '';
  };

  postPatch = lib.optionalString useWebUi ''
    cp -r ${finalAttrs.webui} tools/ui/dist
    chmod -R u+w tools/ui/dist
  '';

  # With PR#6015 https://github.com/ggml-org/llama.cpp/pull/6015,
  # `default.metallib` may be compiled with Metal compiler from XCode
  # and we need to escape sandbox on MacOS to access Metal compiler.
  # `xcrun` is used find the path of the Metal compiler, which is varible
  # and not on $PATH
  # see https://github.com/ggml-org/llama.cpp/pull/6118 for discussion
  __noChroot = effectiveStdenv.hostPlatform.isDarwin && useMetalKit && precompileMetalShaders;

  # Do not fortify under SYCL as it breaks compiled memcpy invocations
  # _FORTIFY_SOURCE rewrites memcpy -> __memcpy_chk
  hardeningDisable = optionals useSycl [
    "fortify"
    "fortify3"
  ];

  nativeBuildInputs =
    [
      cmake
      ninja
      pkg-config
      git
    ]
    ++ optionals useCuda [
      cudaPackages.cuda_nvcc

      autoAddDriverRunpath
    ]
    ++ optionals (effectiveStdenv.hostPlatform.isGnu && enableStatic) [ glibc.static ]
    ++ optionals (effectiveStdenv.hostPlatform.isDarwin && useMetalKit && precompileMetalShaders) [ xcrunHost ]
    ++ optionals useSycl [
      autoAddDriverRunpath
      makeBinaryWrapper
    ]
    ++ optionals syclAot [ oclocAot ];

  buildInputs =
    optionals effectiveStdenv.hostPlatform.isDarwin darwinBuildInputs
    ++ optionals useCuda cudaBuildInputs
    ++ optionals useMpi [ mpi ]
    ++ optionals useRocm rocmBuildInputs
    ++ optionals useBlas [ blas ]
    ++ optionals useVulkan vulkanBuildInputs
    ++ optionals useSycl [
      level-zero
      ocl-icd
      intel-graphics-compiler
    ]
    ++ [ openssl ];

  cmakeFlags = [
    (cmakeBool "LLAMA_BUILD_SERVER" true)
    (cmakeBool "LLAMA_BUILD_WEBUI" useWebUi)
    (cmakeBool "BUILD_SHARED_LIBS" (!enableStatic))
    (cmakeBool "CMAKE_SKIP_BUILD_RPATH" true)
    (cmakeBool "GGML_NATIVE" false)
    (cmakeBool "GGML_BLAS" useBlas)
    (cmakeBool "GGML_CUDA" useCuda)
    (cmakeBool "GGML_HIP" useRocm)
    (cmakeBool "GGML_METAL" useMetalKit)
    (cmakeBool "GGML_VULKAN" useVulkan)
    (cmakeBool "GGML_STATIC" enableStatic)
    (cmakeBool "GGML_RPC" useRpc)
  ]
  ++ optionals useCuda [
    (
      with cudaPackages.flags;
      cmakeFeature "CMAKE_CUDA_ARCHITECTURES" (
        builtins.concatStringsSep ";" (map dropDot cudaCapabilities)
      )
    )
  ]
  ++ optionals useRocm [
    (cmakeFeature "CMAKE_HIP_COMPILER" "${rocmPackages.llvm.clang}/bin/clang")
    (cmakeFeature "CMAKE_HIP_ARCHITECTURES" rocmGpuTargets)
  ]
  ++ optionals useMetalKit [
    (lib.cmakeFeature "CMAKE_C_FLAGS" "-D__ARM_FEATURE_DOTPROD=1")
    (cmakeBool "GGML_METAL_EMBED_LIBRARY" (!precompileMetalShaders))
  ]
  ++ optionals useSycl [
    (cmakeBool "GGML_SYCL" true)
    (cmakeBool "GGML_SYCL_F16" syclF16)
  ]
  ++ optionals syclAot [
    (cmakeFeature "GGML_SYCL_DEVICE_ARCH" syclDeviceArch)
  ]
  ++ optionals (syclAot && syclMaxParallelLinkJobs != null) [
    (cmakeFeature "GGML_SYCL_MAX_PARALLEL_LINK_JOBS" (toString syclMaxParallelLinkJobs))
  ];

  # Environment variables needed for ROCm
  env = optionalAttrs useRocm {
    ROCM_PATH = "${rocmPackages.clr}";
    HIP_DEVICE_LIB_PATH = "${rocmPackages.rocm-device-libs}/amdgcn/bitcode";
  };

  # TODO(SomeoneSerge): It's better to add proper install targets at the CMake level,
  # if they haven't been added yet.
  postInstall = ''
    mkdir -p $out/include
    cp $src/include/llama.h $out/include/
  '';

  # Fix SYCL binary calling missing important libraries
  # Also use Level Zero + sysman INSTEAD of OpenCL
  # due to OpenCL reporting free memory as total memory,
  # breaking `-fit` functionality
  postFixup = lib.optionalString useSycl ''
    driverLibs=${
      lib.makeLibraryPath [
        intel-graphics-compiler
        ocl-icd
      ]
    }
    for b in "$out"/bin/*; do
      [ -f "$b" ] && [ -x "$b" ] || continue
      patchelf --print-interpreter "$b" >/dev/null 2>&1 || continue
      patchelf --add-needed libOpenCL.so.1 --add-needed libigc.so.2 \
        --add-rpath "$driverLibs" "$b"
      wrapProgram "$b" \
        --set-default ONEAPI_DEVICE_SELECTOR level_zero:gpu \
        --set-default ZES_ENABLE_SYSMAN 1
    done
  '';

  meta = {
    # Configurations we don't want even the CI to evaluate. Results in the
    # "unsupported platform" messages. This is mostly a no-op, because
    # cudaPackages would've refused to evaluate anyway.
    badPlatforms =
      optionals useCuda lib.platforms.darwin
      ++ optionals useSycl (lib.subtractLists [ "x86_64-linux" ] lib.platforms.all);

    # Configurations that are known to result in build failures. Can be
    # overridden by importing Nixpkgs with `allowBroken = true`.
    broken = (useMetalKit && !effectiveStdenv.hostPlatform.isDarwin);

    description = "Inference of LLaMA model in pure C/C++${descriptionSuffix}";
    homepage = "https://github.com/ggml-org/llama.cpp/";
    license = lib.licenses.mit;

    # Accommodates `nix run` and `lib.getExe`
    mainProgram = "llama-cli";

    # These people might respond, on the best effort basis, if you ping them
    # in case of Nix-specific regressions or for reviewing Nix-specific PRs.
    # Consider adding yourself to this list if you want to ensure this flake
    # stays maintained and you're willing to invest your time. Do not add
    # other people without their consent. Consider removing people after
    # they've been unreachable for long periods of time.

    # Note that lib.maintainers is defined in Nixpkgs, but you may just add
    # an attrset following the same format as in
    # https://github.com/NixOS/nixpkgs/blob/f36a80e54da29775c78d7eff0e628c2b4e34d1d7/maintainers/maintainer-list.nix
    maintainers = with lib.maintainers; [
      philiptaron
      SomeoneSerge
    ];

    # Extend `badPlatforms` instead
    platforms = lib.platforms.all;
  };
})
