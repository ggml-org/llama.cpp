# NixOS module for llama-server, exposed as a flake-parts module.
#
# Usage in a NixOS configuration:
#
#   services.llama-server = {
#     enable = true;
#     model = "/path/to/model.gguf";
#     gpuLayers = 99;
#   };
{ lib, ... }:

{
  flake.nixosModules.default =
    {
      config,
      lib,
      pkgs,
      ...
    }:
    let
      cfg = config.services.llama-server;
    in
    {
      options.services.llama-server = {
        enable = lib.mkEnableOption "llama.cpp inference server";

        package = lib.mkOption {
          type = lib.types.package;
          default = pkgs.llama-cpp;
          defaultText = lib.literalExpression "pkgs.llama-cpp";
          description = "The llama.cpp package to use.";
        };

        model = lib.mkOption {
          type = lib.types.path;
          description = "Path to the GGUF model file.";
        };

        host = lib.mkOption {
          type = lib.types.str;
          default = "127.0.0.1";
          description = "Address to listen on.";
        };

        port = lib.mkOption {
          type = lib.types.port;
          default = 8000;
          description = "Port to listen on.";
        };

        gpuLayers = lib.mkOption {
          type = lib.types.int;
          default = 99;
          description = "Number of layers to offload to GPU (-ngl).";
        };

        contextSize = lib.mkOption {
          type = lib.types.nullOr lib.types.int;
          default = null;
          description = "Context size (-c). Uses model default when null.";
        };

        parallel = lib.mkOption {
          type = lib.types.nullOr lib.types.int;
          default = null;
          description = "Number of parallel sequences to decode (-np).";
        };

        apiKey = lib.mkOption {
          type = lib.types.nullOr lib.types.str;
          default = null;
          description = "API key for authentication. Mutually exclusive with apiKeyFile.";
        };

        apiKeyFile = lib.mkOption {
          type = lib.types.nullOr lib.types.path;
          default = null;
          description = "Path to a file containing the API key. Mutually exclusive with apiKey.";
        };

        enableMetrics = lib.mkOption {
          type = lib.types.bool;
          default = false;
          description = "Enable Prometheus-compatible metrics endpoint.";
        };

        extraArgs = lib.mkOption {
          type = lib.types.listOf lib.types.str;
          default = [ ];
          description = "Extra command-line arguments appended to llama-server.";
        };

        environment = lib.mkOption {
          type = lib.types.attrsOf lib.types.str;
          default = { };
          description = "Environment variables passed to the service.";
        };
      };

      config = lib.mkIf cfg.enable {
        assertions = [
          {
            assertion = !(cfg.apiKey != null && cfg.apiKeyFile != null);
            message = "services.llama-server: apiKey and apiKeyFile are mutually exclusive.";
          }
        ];

        systemd.services.llama-server = {
          description = "llama.cpp inference server";
          after = [ "network.target" ];
          wantedBy = [ "multi-user.target" ];

          environment = cfg.environment;

          serviceConfig = {
            ExecStart = lib.concatStringsSep " " (
              [
                "${lib.getExe' cfg.package "llama-server"}"
                "--host"
                cfg.host
                "--port"
                (toString cfg.port)
                "-m"
                cfg.model
                "-ngl"
                (toString cfg.gpuLayers)
              ]
              ++ lib.optionals (cfg.contextSize != null) [
                "-c"
                (toString cfg.contextSize)
              ]
              ++ lib.optionals (cfg.parallel != null) [
                "-np"
                (toString cfg.parallel)
              ]
              ++ lib.optionals (cfg.apiKey != null) [
                "--api-key"
                cfg.apiKey
              ]
              ++ lib.optionals (cfg.apiKeyFile != null) [
                "--api-key-file"
                cfg.apiKeyFile
              ]
              ++ lib.optionals cfg.enableMetrics [ "--metrics" ]
              ++ cfg.extraArgs
            );

            DynamicUser = true;
            ProtectSystem = "strict";
            ProtectHome = true;
            PrivateTmp = true;
            NoNewPrivileges = true;
            DeviceAllow = [
              "/dev/kfd rw"
              "/dev/dri rw"
            ];
            SupplementaryGroups = [
              "video"
              "render"
            ];
            ReadOnlyPaths = [ cfg.model ];
          };
        };
      };
    };
}
