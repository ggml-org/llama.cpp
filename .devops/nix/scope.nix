{
  lib,
  newScope,
  python3,
  llamaVersion ? "0.0.0",
}:

let
  pythonPackages = python3.pkgs;
  buildPythonPackage = pythonPackages.buildPythonPackage;
  numpy = pythonPackages.numpy;
  tqdm = pythonPackages.tqdm;
  sentencepiece = pythonPackages.sentencepiece;
  pyyaml = pythonPackages.pyyaml;
  poetry-core = pythonPackages.poetry-core;
  pytestCheckHook = pythonPackages.pytestCheckHook;
in

# We're using `makeScope` instead of just writing out an attrset
# because it allows users to apply overlays later using `overrideScope'`.
# Cf. https://noogle.dev/f/lib/makeScope

lib.makeScope newScope (self: {
  inherit llamaVersion;
  gguf-py = self.callPackage ./package-gguf-py.nix {
    inherit
      buildPythonPackage
      numpy
      tqdm
      sentencepiece
      poetry-core
      pyyaml
      pytestCheckHook
      ;
  };
  python-scripts = self.callPackage ./python-scripts.nix { inherit buildPythonPackage poetry-core; };
  llama-cpp = self.callPackage ./package.nix { };
  docker = self.callPackage ./docker.nix { };
  docker-min = self.callPackage ./docker.nix { interactive = false; };
  sif = self.callPackage ./sif.nix { };
})
