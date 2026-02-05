{ inputs, ... }:
{
  # The _module.args definitions are passed on to modules as arguments. E.g.
  # the module `{ pkgs ... }: { /* config */ }` implicitly uses
  # `_module.args.pkgs` (defined in this case by flake-parts).
  perSystem =
    { lib, system, ... }:
    {
      _module.args = {
        # Why separate nixpkgs instances?
        #
        # CUDA and ROCm require config-level settings (cudaSupport, rocmSupport)
        # that propagate through the *entire* dependency tree. For example,
        # openmpi, ucc, and ucx all need to be built with CUDA/ROCm support
        # when the top-level package uses them. Overlays cannot achieve this —
        # they modify individual packages, not the config that flows into every
        # callPackage invocation. This is a fundamental nixpkgs constraint.
        #
        # Cf. https://zimbatm.com/notes/1000-instances-of-nixpkgs
        #
        # Note that you can use these expressions without Nix
        # (`pkgs.callPackage ./devops/nix/scope.nix { }` is the entry point).

        pkgsCuda = import inputs.nixpkgs {
          inherit system;
          # Ensure dependencies use CUDA consistently (e.g. that openmpi, ucc,
          # and ucx are built with CUDA support)
          config.cudaSupport = true;
          config.allowUnfreePredicate =
            p:
            builtins.all (
              license:
              license.free
              || builtins.elem license.shortName [
                "CUDA EULA"
                "cuDNN EULA"
              ]
            ) (p.meta.licenses or (lib.toList p.meta.license));
        };
        # Ensure dependencies use ROCm consistently
        pkgsRocm = import inputs.nixpkgs {
          inherit system;
          config.rocmSupport = true;
          config.allowUnfreePredicate =
            p:
            builtins.all (
              license:
              license.free
              || builtins.elem license.shortName [
                "AMD ROCm License"
              ]
            ) (p.meta.licenses or (lib.toList p.meta.license));
        };
      };
    };
}
