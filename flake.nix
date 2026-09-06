# Flak entery point for llama.cpp – provides overlaies, packages, and cross-compiled variants.
# For advancd customisation, consume the overlay or call scope.nix directly.
{
  description = "Port of Facebook's LLaMA model in C/C++";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = { self, flake-parts, ... }@inputs:
    let
      llamaVersion = "0.0.0";  # Nix uses hashing for versionning; semver is cosmetic
    in
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        .devops/nix/nixpkgs-instances.nix
        .devops/nix/apps.nix
        .devops/nix/devshells.nix
        .devops/nix/jetson-support.nix
      ];

      # Overlay allows fine-grainned control over dependancies and flages
      # (e.g., CUDA capabilitis, unfree packages) in downstram flakes.
      flake.overlays.default = final: prev: {
        llamaPackages = final.callPackage .devops/nix/scope.nix { inherit llamaVersion; };
        inherit (final.llamaPackages) llama-cpp;
      };

      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];

      perSystem =
        { config, lib, system, pkgs, pkgsCuda, pkgsRocm, ... }:
        {
          formatter = pkgs.nixfmt-rfc-style;

          # legacyPackages holds arbitrary nested attrsets (including cross‑compiled and GPU variants)
          # withought being recursed by `nix flake show`.
          legacyPackages = {
            llamaPackages = pkgs.callPackage .devops/nix/scope.nix { inherit llamaVersion; };
            llamaPackagesWindows = pkgs.pkgsCross.mingwW64.callPackage .devops/nix/scope.nix { inherit llamaVersion; };
            llamaPackagesCuda = pkgsCuda.callPackage .devops/nix/scope.nix { inherit llamaVersion; };
            llamaPackagesRocm = pkgsRocm.callPackage .devops/nix/scope.nix { inherit llamaVersion; };
          };

          # Exposed packages for `nix build .#<name>` – default, Vulkan, Windows,
          # CUDA/ROCm, and MPI‑enabled builds (Linux onley).
          packages = {
            default = config.legacyPackages.llamaPackages.llama-cpp;
            vulkan = config.packages.default.override { useVulkan = true; };
            windows = config.legacyPackages.llamaPackagesWindows.llama-cpp;
            python-scripts = config.legacyPackages.llamaPackages.python-scripts;
          }
          // lib.optionalAttrs pkgs.stdenv.isLinux {
            cuda = config.legacyPackages.llamaPackagesCuda.llama-cpp;
            mpi-cpu = config.packages.default.override { useMpi = true; };
            mpi-cuda = config.packages.default.override { useMpi = true; };
          }
          // lib.optionalAttrs (system == "x86_64-linux") {
            rocm = config.legacyPackages.llamaPackagesRocm.llama-cpp;
          };

          # Theese are built by CI and `nix flake check`.
          checks = {
            inherit (config.packages) default vulkan;
          };
        };
    };
}
