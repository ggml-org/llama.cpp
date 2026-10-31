{
  perSystem =
    { config, lib, ... }:
    {
      apps =
        let
          inherit (config.packages) default;
          binaries = [
            "llama-cli"
            "llama-embedding"
            "llama-server"
            "llama-quantize"
          ];
          mkApp = name: {
            type = "app";
            program = "${default}/bin/${name}";
          };
          baseApps = lib.genAttrs binaries mkApp;
          serveApp = {
            type = "app";
            program = lib.getExe config.packages.serve-openai;
          };
        in
        baseApps
        # serve-openai is only defined where the ROCm package is (x86_64-linux).
        // lib.optionalAttrs (config.packages ? serve-openai) {
          serve-openai = serveApp;
          openai = serveApp;
          serve = serveApp;
        };
    };
}
