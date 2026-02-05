# The flake interface to llama.cpp's Nix expressions. The flake is used as a
# more discoverable entry-point, as well as a way to pin the dependencies and
# expose default outputs, including the outputs built by the CI.

# For more serious applications involving some kind of customization  you may
# want to consider consuming the overlay, or instantiating `llamaPackages`
# directly:
#
# ```nix
# pkgs.callPackage ${llama-cpp-root}/.devops/nix/scope.nix { }`
# ```

# Cf. https://jade.fyi/blog/flakes-arent-real/ for a more detailed exposition
# of the relation between Nix and the Nix Flakes.
{
  description = "Port of Facebook's LLaMA model in C/C++";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  # There's an optional binary cache available. The details are below, but they're commented out.
  #
  # Why? The terrible experience of being prompted to accept them on every single Nix command run.
  # Plus, there are warnings shown about not being a trusted user on a default Nix install
  # if you *do* say yes to the prompts.
  #
  # This experience makes having `nixConfig` in a flake a persistent UX problem.
  #
  # To make use of the binary cache, please add the relevant settings to your `nix.conf`.
  # It's located at `/etc/nix/nix.conf` on non-NixOS systems. On NixOS, adjust the `nix.settings`
  # option in your NixOS configuration to add `extra-substituters` and `extra-trusted-public-keys`,
  # as shown below.
  #
  # ```
  # nixConfig = {
  #   extra-substituters = [
  #     # A development cache for nixpkgs imported with `config.cudaSupport = true`.
  #     # Populated by https://hercules-ci.com/github/SomeoneSerge/nixpkgs-cuda-ci.
  #     # This lets one skip building e.g. the CUDA-enabled openmpi.
  #     # TODO: Replace once nix-community obtains an official one.
  #     "https://cuda-maintainers.cachix.org"
  #   ];
  #
  #   # Verify these are the same keys as published on
  #   # - https://app.cachix.org/cache/cuda-maintainers
  #   extra-trusted-public-keys = [
  #     "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
  #   ];
  # };
  # ```

  # For inspection, use `nix flake show github:ggml-org/llama.cpp` or the nix repl:
  #
  # ```bash
  # ❯ nix repl
  # nix-repl> :lf github:ggml-org/llama.cpp
  # Added 13 variables.
  # nix-repl> outputs.apps.x86_64-linux.quantize
  # { program = "/nix/store/00000000000000000000000000000000-llama.cpp/bin/llama-quantize"; type = "app"; }
  # ```
  outputs =
    { self, flake-parts, ... }@inputs:
    let
      # Use the git revision for dev builds. This triggers rebuilds only when
      # the git tree actually changes, which is acceptable for a development
      # fork. Falls back to "0.0.0" for builds outside a git repo.
      llamaVersion = self.dirtyShortRev or self.shortRev or "0.0.0";
    in
    flake-parts.lib.mkFlake { inherit inputs; }

      {

        imports = [
          .devops/nix/nixpkgs-instances.nix
          .devops/nix/apps.nix
          .devops/nix/devshells.nix
          .devops/nix/jetson-support.nix
          .devops/nix/nixos-module.nix
        ];

        # An overlay can be used to have a more granular control over llama-cpp's
        # dependencies and configuration, than that offered by the `.override`
        # mechanism. Cf. https://nixos.org/manual/nixpkgs/stable/#chap-overlays.
        #
        # E.g. in a flake:
        # ```
        # { nixpkgs, llama-cpp, ... }:
        # let pkgs = import nixpkgs {
        #     overlays = [ (llama-cpp.overlays.default) ];
        #     system = "aarch64-linux";
        #     config.allowUnfree = true;
        #     config.cudaSupport = true;
        #     config.cudaCapabilities = [ "7.2" ];
        #     config.cudaEnableForwardCompat = false;
        # }; in {
        #     packages.aarch64-linux.llamaJetsonXavier = pkgs.llamaPackages.llama-cpp;
        # }
        # ```
        #
        # Cf. https://nixos.org/manual/nix/unstable/command-ref/new-cli/nix3-flake.html?highlight=flake#flake-format
        flake.overlays.default = (
          final: prev: {
            llamaPackages = final.callPackage .devops/nix/scope.nix { inherit llamaVersion; };
            inherit (final.llamaPackages) llama-cpp;
          }
        );

        systems = [
          "aarch64-darwin"
          "aarch64-linux"
          "x86_64-darwin" # x86_64-darwin isn't tested (and likely isn't relevant)
          "x86_64-linux"
        ];

        perSystem =
          {
            config,
            lib,
            system,
            pkgs,
            pkgsCuda,
            pkgsRocm,
            ...
          }:
          {
            # For standardised reproducible formatting with `nix fmt`
            formatter = pkgs.nixfmt-rfc-style;

            # Unlike `.#packages`, legacyPackages may contain values of
            # arbitrary types (including nested attrsets) and may even throw
            # exceptions. This attribute isn't recursed into by `nix flake
            # show` either.
            #
            # You can add arbitrary scripts to `.devops/nix/scope.nix` and
            # access them as `nix build .#llamaPackages.${scriptName}` using
            # the same path you would with an overlay.
            legacyPackages = {
              llamaPackages = pkgs.callPackage .devops/nix/scope.nix { inherit llamaVersion; };
              llamaPackagesWindows = pkgs.pkgsCross.mingwW64.callPackage .devops/nix/scope.nix {
                inherit llamaVersion;
              };
              llamaPackagesCuda = pkgsCuda.callPackage .devops/nix/scope.nix { inherit llamaVersion; };
              llamaPackagesRocm = pkgsRocm.callPackage .devops/nix/scope.nix { inherit llamaVersion; };
            };

            # We don't use the overlay here so as to avoid making too many instances of nixpkgs,
            # cf. https://zimbatm.com/notes/1000-instances-of-nixpkgs
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
            // lib.optionalAttrs (system == "x86_64-linux") (
              let
                rocmBase = config.legacyPackages.llamaPackagesRocm.llama-cpp;
                rocmFor = gfx: rocmBase.override { rocmGpuTargets = gfx; };
                dockerFor = llama: config.legacyPackages.llamaPackagesRocm.docker.override { llama-cpp = llama; };
              in
              {
                rocm = rocmBase; # All supported GPU architectures

                # Card-specific ROCm builds — targets a single GPU for faster compilation
                rocm-gfx906 = rocmFor "gfx906"; # MI50, MI60
                rocm-gfx908 = rocmFor "gfx908"; # MI100
                rocm-gfx90a = rocmFor "gfx90a"; # MI210, MI250, MI250X
                rocm-gfx942 = rocmFor "gfx942"; # MI300A, MI300X, MI325X
                rocm-gfx1030 = rocmFor "gfx1030"; # Radeon PRO W6800, Radeon PRO V620
                rocm-gfx1100 = rocmFor "gfx1100"; # Radeon RX 7900 XTX, Radeon RX 7900 XT
                rocm-gfx1101 = rocmFor "gfx1101"; # Radeon RX 7800 XT, Radeon RX 7700 XT, Radeon PRO W7700
                rocm-gfx1200 = rocmFor "gfx1200"; # Radeon RX 9060 XT
                rocm-gfx1201 = rocmFor "gfx1201"; # Radeon RX 9070 XT, Radeon RX 9070

                # Docker images — GPU-specific containers (reuses cached llama-cpp derivation)
                docker-rocm = dockerFor rocmBase;
                docker-rocm-gfx906 = dockerFor (rocmFor "gfx906");
                docker-rocm-gfx908 = dockerFor (rocmFor "gfx908");
                docker-rocm-gfx90a = dockerFor (rocmFor "gfx90a");
                docker-rocm-gfx942 = dockerFor (rocmFor "gfx942");
                docker-rocm-gfx1030 = dockerFor (rocmFor "gfx1030");
                docker-rocm-gfx1100 = dockerFor (rocmFor "gfx1100");
                docker-rocm-gfx1101 = dockerFor (rocmFor "gfx1101");
                docker-rocm-gfx1200 = dockerFor (rocmFor "gfx1200");
                docker-rocm-gfx1201 = dockerFor (rocmFor "gfx1201");
              }
            );

            # Packages exposed in `.#checks` will be built by the CI and by
            # `nix flake check`.
            #
            # We could test all outputs e.g. as `checks = confg.packages`.
            #
            # TODO: Build more once https://github.com/ggml-org/llama.cpp/issues/6346 has been addressed
            checks = {
              inherit (config.packages) default vulkan;

              nix-formatting =
                pkgs.runCommandLocal "check-nix-formatting"
                  {
                    nativeBuildInputs = [ pkgs.nixfmt-rfc-style ];
                    src = lib.fileset.toSource {
                      root = ./.;
                      fileset = lib.fileset.fileFilter (f: f.hasExt "nix") ./.;
                    };
                  }
                  ''
                    nixfmt --check "$src"
                    touch $out
                  '';
            };
          };
      };
}
