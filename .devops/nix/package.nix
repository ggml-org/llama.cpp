{
  lib,
  glibc,
  config,
  stdenv,
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
  curl,
  shaderc,
  useBlas ?
    builtins.all (x: !x) [
      useCuda
      useMetalKit
      useRocm
      useVulkan
    ]
    && blas.meta.available,
  useCuda ? config.cudaSupport,
  useMetalKit ? stdenv.isAarch64 && stdenv.isDarwin,
  # Increases the runtime closure size by ~700M
  useMpi ? false,
  useRocm ? config.rocmSupport,
  enableCurl ? true,
  useVulkan ? false,
  llamaVersion ? "0.0.0", # Arbitrary version, substituted by the flake

  # It's necessary to consistently use backendStdenv when building with CUDA support,
  # otherwise we get libstdc++ errors downstream.
  effectiveStdenv ? if useCuda then cudaPackages.backendStdenv else stdenv,
  enableStatic ? effectiveStdenv.hostPlatform.isStatic,
  precompileMetalShaders ? false,
}:

let
  inherit (lib)
    cmakeBool
    cmakeFeature
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
    ++ lib.optionals useVulkan [ "Vulkan" ];

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
    cuda_cccl # <nv/target>
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
  ];
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

  postPatch = ''
    substituteInPlace ./ggml/src/ggml-metal.m \
      --replace '[bundle pathForResource:@"ggml-metal" ofType:@"metal"];' "@\"$out/bin/ggml-metal.metal\";"
    substituteInPlace ./ggml/src/ggml-metal.m \
      --replace '[bundle pathForResource:@"default" ofType:@"metallib"];' "@\"$out/bin/default.metallib\";"
  '';

  # With PR#6015 https://github.com/ggerganov/llama.cpp/pull/6015,
  # `default.metallib` may be compiled with Metal compiler from XCode
  # and we need to escape sandbox on MacOS to access Metal compiler.
  # `xcrun` is used find the path of the Metal compiler, which is varible
  # and not on $PATH
  # see https://github.com/ggerganov/llama.cpp/pull/6118 for discussion
  __noChroot = effectiveStdenv.isDarwin && useMetalKit && precompileMetalShaders;

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
    ++ optionals (effectiveStdenv.isDarwin && useMetalKit && precompileMetalShaders) [ xcrunHost ];

  buildInputs =
    optionals effectiveStdenv.isDarwin darwinBuildInputs
    ++ optionals useCuda cudaBuildInputs
    ++ optionals useMpi [ mpi ]
    ++ optionals useRocm rocmBuildInputs
    ++ optionals useBlas [ blas ]
    ++ optionals useVulkan vulkanBuildInputs
    ++ optionals enableCurl [ curl ];

  cmakeFlags =
    [
      (cmakeBool "LLAMA_BUILD_SERVER" true)
      (cmakeBool "BUILD_SHARED_LIBS" (!enableStatic))
      (cmakeBool "CMAKE_SKIP_BUILD_RPATH" true)
      (cmakeBool "LLAMA_CURL" enableCurl)
      (cmakeBool "GGML_NATIVE" false)
      (cmakeBool "GGML_BLAS" useBlas)
      (cmakeBool "GGML_CUDA" useCuda)
      (cmakeBool "GGML_HIPBLAS" useRocm)
      (cmakeBool "GGML_METAL" useMetalKit)
      (cmakeBool "GGML_VULKAN" useVulkan)
      (cmakeBool "GGML_STATIC" enableStatic)
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
      (cmakeFeature "CMAKE_HIP_ARCHITECTURES" (builtins.concatStringsSep ";" rocmPackages.clr.gpuTargets))
    ]
    ++ optionals useMetalKit [
      (lib.cmakeFeature "CMAKE_C_FLAGS" "-D__ARM_FEATURE_DOTPROD=1")
      (cmakeBool "GGML_METAL_EMBED_LIBRARY" (!precompileMetalShaders))
    ];

  # Environment variables needed for ROCm
  env = optionals useRocm {
    ROCM_PATH = "${rocmPackages.clr}";
    HIP_DEVICE_LIB_PATH = "${rocmPackages.rocm-device-libs}/amdgcn/bitcode";
  };

  # TODO(SomeoneSerge): It's better to add proper install targets at the CMake level,
  # if they haven't been added yet.
  postInstall = ''
    mkdir -p $out/include
    cp $src/include/llama.h $out/include/
  '';

  meta = {
    # Configurations we don't want even the CI to evaluate. Results in the
    # "unsupported platform" messages. This is mostly a no-op, because
    # cudaPackages would've refused to evaluate anyway.
    badPlatforms = optionals useCuda lib.platforms.darwin;

    # Configurations that are known to result in build failures. Can be
    # overridden by importing Nixpkgs with `allowBroken = true`.
    broken = (useMetalKit && !effectiveStdenv.isDarwin);

    description = "Inference of LLaMA model in pure C/C++${descriptionSuffix}";
    homepage = "https://github.com/ggerganov/llama.cpp/";
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
