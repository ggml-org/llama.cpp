{
  lib,
  singularity-tools,
  llama-cpp,
  bashInteractive,
  interactive ? false,
}:

let
  optionalInt = cond: x: if cond then x else 0;
in
singularity-tools.buildImage rec {
  inherit (llama-cpp) name;
  contents = [ llama-cpp ] ++ lib.optionals interactive [ bashInteractive ];

  # These are excessive (but safe) for most variants. Building singularity
  # images requires superuser privileges, so we build them inside a VM in a
  # writable image of pre-determined size.
  #
  # ROCm is currently affected by https://github.com/NixOS/nixpkgs/issues/276846
  #
  # Expected image sizes:
  # - cpu/blas: 150M,
  # - cuda, all gencodes: 560M,
  diskSize = 4096 + optionalInt llama-cpp.useRocm 16384;
  memSize = diskSize;
}
