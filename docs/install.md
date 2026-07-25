# Install pre-built version of llama.cpp

| Install via | Windows | Mac  | Linux |
|-------------|---------|------|-------|
| conda-forge | ✅      | ✅   | ✅   |
| Winget      | ✅      |      |      |
| Homebrew    |         | ✅   | ✅   |
| MacPorts    |         | ✅   |      |
| Nix         |         | ✅   | ✅   |

## conda-forge (Windows, Mac and Linux)

conda-forge provides builds for:
 - CUDA (Windows and Linux)
 - Vulkan (Windows and Linux)
 - Apple Metal (macOS)

```sh
conda install -c conda-forge llama-cpp
```

```sh
mamba install -c conda-forge llama-cpp
```

```sh
# Project-local installation
pixi add llama-cpp

# Global installation
pixi global install llama-cpp
```

This distribution is managed on [`conda-forge/llama-cpp-feedstock`](https://github.com/conda-forge/llama.cpp-feedstock/).

Shall you have any problems, please open an issue on [its issue tracker](https://github.com/conda-forge/llama.cpp-feedstock/issues).

## Winget (Windows)

```sh
winget install llama.cpp
```

The package is automatically updated with new `llama.cpp` releases. More info: https://github.com/ggml-org/llama.cpp/issues/8188

## Homebrew (Mac and Linux)

```sh
brew install llama.cpp
```

The formula is automatically updated with new `llama.cpp` releases. More info: https://github.com/ggml-org/llama.cpp/discussions/7668

## MacPorts (Mac)

```sh
sudo port install llama.cpp
```

See also: https://ports.macports.org/port/llama.cpp/details/

## Nix (Mac and Linux)

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
