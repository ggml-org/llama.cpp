# Install pre-built version of llama.cpp

## Homebrew

On Mac and Linux, the homebrew package manager can be used via

```sh
brew install llama.cpp
```
The formula is automatically updated with new `llama.cpp` releases. More info: https://github.com/ggerganov/llama.cpp/discussions/7668

## Nix

On Mac and Linux, the Nix package manager can be used via

```sh
nix profile install nixpkgs#llama-cpp
```
For flake enabled installs.

Or

```sh
nix-env --file '<nixpkgs>' --install --attr llama-cpp
```

For non-flake enabled installs.

This expression is automatically updated within the [nixpkgs repo](https://github.com/NixOS/nixpkgs/blob/nixos-24.05/pkgs/by-name/ll/llama-cpp/package.nix#L164).

## Flox

On Mac and Linux, Flox can be used to install llama.cpp within a Flox environment via

```sh
flox install llama-cpp
```

Flox follows the nixpkgs build of llama.cpp.
