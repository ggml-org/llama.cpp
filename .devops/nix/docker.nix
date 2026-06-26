{
  lib,
  dockerTools,
  buildEnv,
  llama-cpp,
  interactive ? true,
  coreutils,
}:

# A tar that can be fed into `docker load`:
#
# $ nix build .#llamaPackages.docker
# $ docker load < result

# For details and variations cf.
# - https://nixos.org/manual/nixpkgs/unstable/#ssec-pkgs-dockerTools-buildLayeredImage
# - https://discourse.nixos.org/t/a-faster-dockertools-buildimage-prototype/16922
# - https://nixery.dev/

# Approximate (compressed) sizes, at the time of writing, are:
#
# .#llamaPackages.docker: 125M;
# .#llamaPackagesCuda.docker: 537M;
# .#legacyPackages.aarch64-linux.llamaPackagesXavier.docker: 415M.

dockerTools.buildLayeredImage {
  name = llama-cpp.pname;
  tag = "latest";

  contents =
    [ llama-cpp ]
    ++ lib.optionals interactive [
      coreutils
      dockerTools.binSh
      dockerTools.caCertificates
    ];
}
