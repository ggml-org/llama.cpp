{ inputs, ... }:
{
  perSystem =
    {
      config,
      system,
      lib,
      pkgsCuda,
      ...
    }:
    {
      legacyPackages =
        let
          caps.llamaPackagesXavier = "7.2";
          caps.llamaPackagesOrin = "8.7";
          caps.llamaPackagesTX2 = "6.2";
          caps.llamaPackagesNano = "5.3";

          pkgsFor =
            cap:
            import inputs.nixpkgs {
              inherit system;
              config = {
                cudaSupport = true;
                cudaCapabilities = [ cap ];
                cudaEnableForwardCompat = false;
                inherit (pkgsCuda.config) allowUnfreePredicate;
              };
            };
        in
        builtins.mapAttrs (name: cap: (pkgsFor cap).callPackage ./scope.nix { }) caps;

      packages = lib.optionalAttrs (system == "aarch64-linux") {
        jetson-xavier = config.legacyPackages.llamaPackagesXavier.llama-cpp;
        jetson-orin = config.legacyPackages.llamaPackagesOrin.llama-cpp;
        jetson-nano = config.legacyPackages.llamaPackagesNano.llama-cpp;
      };
    };
}
