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
  openssl,
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
  rocmGpuTargets ? builtins.concatStringsSep ";" rocmPackages.clr.gpuTargets,
  useVulkan ? false,
  useRpc ? false,
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
    optionalAttrs
    optionals
    strings
    ;

  # Safety: shadow stdenv to force use of effectiveStdenv.
  # callPackage auto-injects stdenv, but CUDA builds require backendStdenv
  # for consistent libstdc++. This throw catches accidental uses at eval time.
  stdenv = throw "Use effectiveStdenv instead";

  # Import backend definitions
  backendDefs = import ./backends.nix {
    inherit lib cmakeBool cmakeFeature;
  };

  xcrunHost = runCommand "xcrunHost" { } ''
    mkdir -p $out/bin
    ln -s /usr/bin/xcrun $out/bin
  '';

  # Build the list of active backends by checking each use* flag
  activeBackends =
    optionals useBlas [
      (backendDefs.blas { inherit blas; })
    ]
    ++ optionals useCuda [
      (backendDefs.cuda { inherit cudaPackages autoAddDriverRunpath; })
    ]
    ++ optionals useRocm [
      (backendDefs.rocm { inherit rocmPackages rocmGpuTargets; })
    ]
    ++ optionals useMetalKit [
      (backendDefs.metalkit { inherit darwin precompileMetalShaders xcrunHost; })
    ]
    ++ optionals useVulkan [
      (backendDefs.vulkan { inherit vulkan-headers vulkan-loader shaderc; })
    ]
    ++ optionals useMpi [
      (backendDefs.mpi { inherit mpi; })
    ];

  # Merge all active backend configs into a single attrset
  emptyBackend = {
    suffixes = [ ];
    buildInputs = [ ];
    nativeBuildInputs = [ ];
    cmakeFlags = [ ];
    env = { };
  };

  mergedBackends = lib.foldl' (acc: backend: {
    suffixes = acc.suffixes ++ [ backend.suffix ];
    buildInputs = acc.buildInputs ++ backend.buildInputs;
    nativeBuildInputs = acc.nativeBuildInputs ++ backend.nativeBuildInputs;
    cmakeFlags = acc.cmakeFlags ++ backend.cmakeFlags;
    env = acc.env // backend.env;
  }) emptyBackend activeBackends;

  pnameSuffix =
    strings.optionalString (mergedBackends.suffixes != [ ])
      "-${strings.concatMapStringsSep "-" strings.toLower mergedBackends.suffixes}";
  descriptionSuffix = strings.optionalString (
    mergedBackends.suffixes != [ ]
  ) ", accelerated with ${strings.concatStringsSep ", " mergedBackends.suffixes}";

  # apple_sdk is supposed to choose sane defaults, no need to handle isAarch64
  # separately
  darwinBuildInputs = with darwin.apple_sdk.frameworks; [
    Accelerate
    CoreVideo
    CoreGraphics
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

  postPatch = "";

  # Last-resort sandbox escape, only used for Darwin Metal precompilation.
  # When precompileMetalShaders is true, the build needs `xcrun` to locate
  # the Metal compiler, which lives outside the Nix sandbox at a variable
  # system path. This is safe because the Metal shader compilation is
  # deterministic — the escape only grants read access to Apple's toolchain.
  # See https://github.com/ggml-org/llama.cpp/pull/6118 for discussion.
  __noChroot = effectiveStdenv.isDarwin && useMetalKit && precompileMetalShaders;

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    git
  ]
  ++ mergedBackends.nativeBuildInputs
  ++ optionals (effectiveStdenv.hostPlatform.isGnu && enableStatic) [ glibc.static ];

  buildInputs = [
    curl
    openssl
  ] # For HTTPS model downloads (cpp-httplib + OpenSSL)
  ++ optionals effectiveStdenv.isDarwin darwinBuildInputs
  ++ mergedBackends.buildInputs;

  cmakeFlags = [
    (cmakeBool "LLAMA_BUILD_SERVER" true)
    (cmakeBool "BUILD_SHARED_LIBS" (!enableStatic))
    (cmakeBool "CMAKE_SKIP_BUILD_RPATH" true)
    (cmakeBool "GGML_NATIVE" true) # Enable CPU-native optimizations (AVX2, etc)
    (cmakeBool "GGML_BLAS" useBlas)
    (cmakeBool "GGML_CUDA" useCuda)
    (cmakeBool "GGML_HIP" useRocm)
    (cmakeBool "GGML_METAL" useMetalKit)
    (cmakeBool "GGML_VULKAN" useVulkan)
    (cmakeBool "GGML_STATIC" enableStatic)
    (cmakeBool "GGML_RPC" useRpc)
  ]
  ++ mergedBackends.cmakeFlags;

  env = mergedBackends.env;

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
