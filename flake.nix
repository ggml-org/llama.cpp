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
  #     # Populated by the CI in ggml-org/llama.cpp
  #     "https://llama-cpp.cachix.org"
  #
  #     # A development cache for nixpkgs imported with `config.cudaSupport = true`.
  #     # Populated by https://hercules-ci.com/github/SomeoneSerge/nixpkgs-cuda-ci.
  #     # This lets one skip building e.g. the CUDA-enabled openmpi.
  #     # TODO: Replace once nix-community obtains an official one.
  #     "https://cuda-maintainers.cachix.org"
  #   ];
  #
  #   # Verify these are the same keys as published on
  #   # - https://app.cachix.org/cache/llama-cpp
  #   # - https://app.cachix.org/cache/cuda-maintainers
  #   extra-trusted-public-keys = [
  #     "llama-cpp.cachix.org-1:H75X+w83wUKTIPSO1KWy9ADUrzThyGs8P5tmAbkWhQc="
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
      # We could include the git revisions in the package names but those would
      # needlessly trigger rebuilds:
      # llamaVersion = self.dirtyShortRev or self.shortRev;

      # Nix already uses cryptographic hashes for versioning, so we'll just fix
      # the fake semver for now:
      llamaVersion = "0.0.0";
    in
    flake-parts.lib.mkFlake { inherit inputs; }

      {

        imports = [
          .devops/nix/nixpkgs-instances.nix
          .devops/nix/apps.nix
          .devops/nix/devshells.nix
          .devops/nix/jetson-support.nix
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
            packages =
              {
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

            # Packages exposed in `.#checks` will be built by the CI and by
            # `nix flake check`.
            #
            # We could test all outputs e.g. as `checks = confg.packages`.
            #
            # TODO: Build more once https://github.com/ggml-org/llama.cpp/issues/6346 has been addressed
            checks = {
              inherit (config.packages) default vulkan;
            };
          };
      };
}
