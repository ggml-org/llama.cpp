{ inputs, ... }:
{
  # The _module.args definitions are passed on to modules as arguments. E.g.
  # the module `{ pkgs ... }: { /* config */ }` implicitly uses
  # `_module.args.pkgs` (defined in this case by flake-parts).
  perSystem =
    { lib, system, ... }:
    {
      _module.args = {
        # Note: bringing up https://zimbatm.com/notes/1000-instances-of-nixpkgs
        # again, the below creates several nixpkgs instances which the
        # flake-centric CLI will be forced to evaluate e.g. on `nix flake show`.
        #
        # This is currently "slow" and "expensive", on a certain scale.
        # This also isn't "right" in that this hinders dependency injection at
        # the level of flake inputs. This might get removed in the foreseeable
        # future.
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
        };
      };
    };
}
