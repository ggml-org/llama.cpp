{ inputs, ... }:

{
  perSystem =
    {
      config,
      lib,
      system,
      ...
    }:
    {
      devShells =
        let
          pkgs = import inputs.nixpkgs { inherit system; };
          stdenv = pkgs.stdenv;
          scripts = config.packages.python-scripts;
          backendCmakeFlags =
            name:
            if name == "cuda" then
              "-DGGML_CUDA=ON"
            else if name == "rocm" then
              "-DGGML_HIP=ON"
            else if name == "vulkan" then
              "-DGGML_VULKAN=ON"
            else
              "";
          mkShellHook =
            name:
            let
              cmakeFlags = backendCmakeFlags name;
              cmakeFlagsSuffix = lib.optionalString (cmakeFlags != "") " ${cmakeFlags}";
            in
            ''
              echo "Entering ${name} devShell"
              echo
              echo "Local build:"
              echo "  cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON${cmakeFlagsSuffix}"
              echo "  cmake --build build"
              echo
              echo "LSP:"
              echo "  ln -sf build/compile_commands.json compile_commands.json"
            '';
          mkDevShell =
            name: package:
            pkgs.mkShell {
              name = name;
              inputsFrom = [ package ];
              packages = [
                pkgs.ccls
                pkgs.clang-tools
                pkgs.cmake
                pkgs.ninja
                pkgs.pkg-config
              ];
              shellHook = mkShellHook name;
            };
        in
        lib.pipe (config.packages) [
          (lib.concatMapAttrs (
            name: package: {
              ${name} = mkDevShell name package;
              "${name}-extra" =
                if (name == "python-scripts") then
                  null
                else
                  pkgs.mkShell {
                    name = "${name}-extra";
                    inputsFrom = [
                      package
                      scripts
                    ];
                    # Extra packages that *may* be used by some scripts
                    packages = [
                      pkgs.ccls
                      pkgs.clang-tools
                      pkgs.cmake
                      pkgs.ninja
                      pkgs.pkg-config
                      pkgs.python3Packages.tiktoken
                    ];
                    shellHook =
                      mkShellHook name
                      + ''
                        addToSearchPath "LD_LIBRARY_PATH" "${lib.getLib stdenv.cc.cc}/lib"
                      '';
                  };
            }
          ))
          (lib.filterAttrs (name: value: value != null))
        ];
    };
}
