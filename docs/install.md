# Install pre-built version of llama.cpp

| Install via | Windows | Mac | Linux |
|-------------|---------|-----|-------|
| Winget      | ✅      |      |      |
| Homebrew    |         | ✅   | ✅   |
| MacPorts    |         | ✅   |      |
| Nix         |         | ✅   | ✅   |

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
