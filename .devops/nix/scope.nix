{
  lib,
  newScope,
  python3,
  llamaVersion ? "0.0.0",
}:

let
  pythonPackages = python3.pkgs;
in

# We're using `makeScope` instead of just writing out an attrset
# because it allows users to apply overlays later using `overrideScope'`.
# Cf. https://noogle.dev/f/lib/makeScope

lib.makeScope newScope (self: {
  inherit llamaVersion;
  gguf-py = self.callPackage ./package-gguf-py.nix {
    inherit (pythonPackages)
      numpy
      tqdm
      sentencepiece
      pyyaml
      pytestCheckHook
      requests
      buildPythonPackage
      poetry-core
      ;
  };
  python-scripts = self.callPackage ./python-scripts.nix { inherit (pythonPackages) buildPythonPackage poetry-core; };
  llama-cpp = self.callPackage ./package.nix { };
  docker = self.callPackage ./docker.nix { };
  docker-min = self.callPackage ./docker.nix { interactive = false; };
  sif = self.callPackage ./sif.nix { };
})
