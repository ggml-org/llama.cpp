# Hardware backend definitions for llama.cpp.
#
# Each backend is a function that takes its specific dependencies and returns:
#   { suffix, buildInputs, nativeBuildInputs, cmakeFlags, env }
#
# This keeps package.nix focused on orchestration while making it trivial
# to add new backends: just add an entry here.
{
  lib,
  cmakeBool,
  cmakeFeature,
}:

{
  blas =
    { blas }:
    {
      suffix = "BLAS";
      buildInputs = [ blas ];
      nativeBuildInputs = [ ];
      cmakeFlags = [ ];
      env = { };
    };

  cuda =
    {
      cudaPackages,
      autoAddDriverRunpath,
    }:
    {
      suffix = "CUDA";
      buildInputs = with cudaPackages; [
        cuda_cudart
        cuda_cccl # <nv/target>
        libcublas
      ];
      nativeBuildInputs = [
        cudaPackages.cuda_nvcc
        autoAddDriverRunpath
      ];
      cmakeFlags = [
        (
          with cudaPackages.flags;
          cmakeFeature "CMAKE_CUDA_ARCHITECTURES" (
            builtins.concatStringsSep ";" (map dropDot cudaCapabilities)
          )
        )
      ];
      env = { };
    };

  rocm =
    {
      rocmPackages,
      rocmGpuTargets,
    }:
    let
      isGfx906 = lib.hasInfix "gfx906" rocmGpuTargets;
    in
    {
      suffix = "ROCm";
      buildInputs = with rocmPackages; [
        clr
        hipblas
        rocblas
      ];
      nativeBuildInputs = [ ];
      cmakeFlags = [
        (cmakeFeature "CMAKE_HIP_COMPILER" "${rocmPackages.llvm.clang}/bin/clang")
        (cmakeFeature "CMAKE_HIP_ARCHITECTURES" rocmGpuTargets)
        # General ROCm performance flags
        (cmakeBool "GGML_HIP_GRAPHS" true)
        (cmakeBool "GGML_HIP_EXPORT_METRICS" true)
        (cmakeBool "GGML_CUDA_FA" true)
        (cmakeBool "GGML_CUDA_FA_ALL_QUANTS" true)
      ]
      # MI50/MI60 (gfx906) workarounds — only when targeting gfx906
      ++ lib.optionals isGfx906 [
        (cmakeBool "GGML_HIP_NO_VMM" true) # Required for MI50 - disable Virtual Memory Management
        (cmakeBool "GGML_CUDA_NO_PEER_COPY" true) # Disable peer-to-peer GPU copies (safer for MI50)
      ];
      env = {
        ROCM_PATH = "${rocmPackages.clr}";
        HIP_DEVICE_LIB_PATH = "${rocmPackages.rocm-device-libs}/amdgcn/bitcode";
      };
    };

  metalkit =
    {
      darwin,
      precompileMetalShaders,
      xcrunHost,
    }:
    {
      suffix = "MetalKit";
      buildInputs = [ darwin.apple_sdk.frameworks.MetalKit ];
      nativeBuildInputs = lib.optionals precompileMetalShaders [ xcrunHost ];
      cmakeFlags = [
        (cmakeFeature "CMAKE_C_FLAGS" "-D__ARM_FEATURE_DOTPROD=1")
        (cmakeBool "GGML_METAL_EMBED_LIBRARY" (!precompileMetalShaders))
      ];
      env = { };
    };

  vulkan =
    {
      vulkan-headers,
      vulkan-loader,
      shaderc,
    }:
    {
      suffix = "Vulkan";
      buildInputs = [
        vulkan-headers
        vulkan-loader
        shaderc
      ];
      nativeBuildInputs = [ ];
      cmakeFlags = [ ];
      env = { };
    };

  mpi =
    { mpi }:
    {
      suffix = "MPI";
      buildInputs = [ mpi ];
      nativeBuildInputs = [ ];
      cmakeFlags = [ ];
      env = { };
    };
}
