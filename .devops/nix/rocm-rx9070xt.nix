# Qualified ROCm configuration for the Radeon RX 9070 XT (gfx1201, RDNA 4): the Nix package, a
# runtime wrapper that exposes only that GPU, a matching CMake configure helper for local builds,
# the OpenAI-compatible server launcher, and the development shell.
#
#   nix build .#rocm-rx9070xt          # qualified llama.cpp build
#   nix run .#serve-openai -- -m model.gguf
#   nix develop .#rocm-rx9070xt        # then: llama-rocm-configure build-rocm && cmake --build build-rocm
{
  perSystem =
    {
      config,
      lib,
      pkgs,
      pkgsRocm,
      system,
      ...
    }:
    let
      supported = system == "x86_64-linux";
      rocm = pkgsRocm.rocmPackages;

      # One source of truth for the qualified configuration, shared by the package and local builds.
      gpuTarget = "gfx1201";
      # KFD topology encoding of gfx1201 (major * 10000 + minor * 100 + stepping).
      gfxTargetVersion = "120001";
      # Mixed Q8_0 K / Q4_0 V needs each side's same-type kernel as well; without them streaming
      # silently falls back to the F16 conversion path (README "Required FlashAttention kernels").
      fattnQuants = "q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16,q8_0-q4_0";
      # Ryzen 9 8945HX (Zen 4) instruction sets. Every one must be listed: Nix sets SOURCE_DATE_EPOCH, which
      # makes ggml default GGML_NATIVE off *and* enable no x86 extensions, and the cc-wrapper strips
      # -march=native anyway. Zen 4 has AVX-512 VNNI but not the VEX-encoded AVX-VNNI.
      cpuFeatures = [
        "GGML_SSE42"
        "GGML_AVX"
        "GGML_AVX2"
        "GGML_BMI2"
        "GGML_FMA"
        "GGML_F16C"
        "GGML_AVX512"
        "GGML_AVX512_VBMI"
        "GGML_AVX512_VNNI"
        "GGML_AVX512_BF16"
      ];

      llama-cpp = config.legacyPackages.llamaPackagesRocm.llama-cpp.override {
        rocmGpuTargets = gpuTarget;
        inherit fattnQuants cpuFeatures;
      };

      # The Ryzen iGPU (gfx1036) is a second KFD agent with no kernels in this build. Expose only the
      # gfx1201 device so ROCm0 is always the RX 9070 XT. An explicit ROCR_VISIBLE_DEVICES wins.
      rocm-env = pkgs.writeShellApplication {
        name = "llama-rocm-env";
        runtimeInputs = [
          pkgs.gawk
          pkgs.bc
        ];
        text = ''
          target="''${LLAMA_ROCM_GFX_TARGET_VERSION:-${gfxTargetVersion}}"

          # ROCr matches "GPU-" plus the 16-digit upper-case hex KFD unique_id.
          find_devices() {
            local topology=/sys/class/kfd/kfd/topology/nodes node=0 props version id hex list=""
            while [ -r "$topology/$node/properties" ]; do
              props="$topology/$node/properties"
              version=$(awk '$1 == "gfx_target_version" { print $2 }' "$props")
              id=$(awk '$1 == "unique_id" { print $2 }' "$props")
              if [ "$version" = "$target" ] && [ -n "$id" ] && [ "$id" != 0 ]; then
                hex=$(echo "obase=16; $id" | bc)
                while [ "''${#hex}" -lt 16 ]; do hex="0$hex"; done
                list="''${list:+$list,}GPU-$hex"
              fi
              node=$((node + 1))
            done
            printf '%s' "$list"
          }

          # An empty filter hides every GPU; fail instead of silently running without one.
          if [ -n "''${ROCR_VISIBLE_DEVICES+x}" ]; then
            devices="$ROCR_VISIBLE_DEVICES"
            if [ -z "$devices" ]; then
              echo "llama-rocm-env: ROCR_VISIBLE_DEVICES is set but empty, which hides every GPU" >&2
              exit 1
            fi
          else
            devices=$(find_devices)
            if [ -z "$devices" ]; then
              echo "llama-rocm-env: no GPU with gfx_target_version $target in KFD topology; set ROCR_VISIBLE_DEVICES" >&2
              exit 1
            fi
          fi

          if [ "''${1:-}" = "--print" ]; then
            printf '%s\n' "$devices"
            exit 0
          fi
          if [ "$#" -eq 0 ]; then
            echo "usage: llama-rocm-env --print | llama-rocm-env <command> [args...]" >&2
            exit 2
          fi
          export ROCR_VISIBLE_DEVICES="$devices"
          exec "$@"
        '';
      };

      # Configure a local CMake build with the same options as the package, plus tests.
      rocm-configure = pkgs.writeShellApplication {
        name = "llama-rocm-configure";
        text = ''
          if [ ! -f include/llama.h ] || [ ! -f CMakeLists.txt ]; then
            echo "llama-rocm-configure: run from the llama.cpp source root" >&2
            exit 1
          fi
          build_dir="''${1:-build-rocm}"
          [ "$#" -gt 0 ] && shift

          # CMake deletes the whole cache (and with it -D options such as GGML_HIP) when an existing build
          # directory sees a different compiler path, even a symlink to the same compiler. Pass the HIP
          # compiler only to fresh directories, and refuse a real mismatch instead of silently resetting.
          hip_compiler=${rocm.llvm.clang}/bin/clang
          compiler_args=()
          cache="$build_dir/CMakeCache.txt"
          cached=""
          [ -f "$cache" ] && cached=$(sed -n 's/^CMAKE_HIP_COMPILER:[A-Z]*=//p' "$cache" | head -n 1)
          if [ -z "$cached" ]; then
            compiler_args=(-DCMAKE_HIP_COMPILER="$hip_compiler")
          elif [ "$(readlink -f "$cached")" != "$(readlink -f "$hip_compiler")" ]; then
            echo "llama-rocm-configure: $build_dir uses HIP compiler $cached, not $hip_compiler." >&2
            echo "Reconfiguring would make CMake reset the cache. Use a new build directory, or remove" >&2
            echo "$cache and $build_dir/CMakeFiles first." >&2
            exit 1
          fi

          # No ccache: after a ggml-cuda header change it served objects compiled against the old header for
          # most HIP translation units, linking two layouts of ggml_backend_cuda_context into libggml-hip.
          exec cmake -S . -B "$build_dir" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DGGML_HIP=ON \
            -DGGML_CCACHE=OFF \
            "''${compiler_args[@]}" \
            -DGPU_TARGETS=${gpuTarget} \
            -DAMDGPU_TARGETS=${gpuTarget} \
            -DCMAKE_HIP_ARCHITECTURES=${gpuTarget} \
            -DGGML_CUDA_FA_QUANTS=${fattnQuants} \
            -DGGML_NATIVE=OFF \
            ${lib.concatMapStringsSep " " (feature: "-D${feature}=ON") cpuFeatures} \
            -DLLAMA_BUILD_TESTS=ON \
            "$@"
        '';
      };

      # Qualified server defaults (report §5, §9.6); every value can be overridden by env or by
      # appending the flag, since later llama.cpp arguments win. defaultServer is a shell expression.
      mkServe =
        { defaultServer, missingHint }:
        let
          script = pkgs.writeShellApplication {
            name = "llama-serve-openai";
            runtimeInputs = [ rocm-env ];
            text = ''
              has_model=0
              [ -n "''${LLAMA_ARG_MODEL:-}" ] && has_model=1
              for arg in "$@"; do
                case "$arg" in
                  -m | --model | -hf | --hf-repo | -mu | --model-url) has_model=1 ;;
                esac
              done
              if [ "$has_model" -eq 0 ]; then
                echo "usage: llama-serve-openai -m <model.gguf> [llama-server args...]  (or set LLAMA_ARG_MODEL)" >&2
                exit 2
              fi

              server="''${LLAMA_SERVER_BIN:-${defaultServer}}"
              if [ ! -x "$server" ]; then
                echo "llama-serve-openai: $server is not executable; ${missingHint}" >&2
                exit 1
              fi

              ROCR_VISIBLE_DEVICES=$(llama-rocm-env --print)
              export ROCR_VISIBLE_DEVICES

              # Two HW queues instead of four: the second queue's preemption save area (~140 MiB) appears during
              # the first prompt batch and made long prefills at full context run out of memory. No measured cost.
              export GPU_MAX_HW_QUEUES="''${GPU_MAX_HW_QUEUES:-2}"

              host="''${LLAMA_ARG_HOST:-127.0.0.1}"
              port="''${LLAMA_ARG_PORT:-8080}"
              ctx_size="''${LLAMA_ARG_CTX_SIZE:-262144}"
              resident_mib="''${LLAMA_ARG_KV_RESIDENT_MIB:-450}"
              resident_mib_decode="''${LLAMA_ARG_KV_RESIDENT_MIB_DECODE:-2100}"
              # The model's embedded Qwen template defaults to reasoning effort xhigh, which can reason through the
              # whole token budget without answering. Requests can still ask for more via reasoning_effort.
              reasoning_effort="''${LLAMA_REASONING_EFFORT:-medium}"
              alias_name="''${LLAMA_ARG_ALIAS:-qwen3.8-27b}"
              # MTP drafts 3 tokens every step; ngram-mod drafts instead when the output repeats the context (code
              # being copied or edited): up to 16 tokens after a 24-token match. On copies and edits that decoded 2-2.6x
              # faster than MTP alone at 8k and 1.8x at 71k; on reasoning it rarely drafts (about -2% at 71k). n_max 64
              # was faster at 8k but slower at depth. The match length stays at the default 24: in agent sessions its
              # false drafts (~80-100 ms each: checkpoint, 17-token verify, restore and replay) cost under 0.5% of decode
              # with 24 and 32 alike, and the longest coding session gained 4.4% with 24. A later
              # --spec-ngram-mod-n-match overrides it.
              # Its decode buffers take ~40 MiB more VRAM; measured without a desktop session (peak 16,140 of 16,304
              # MiB at 71k). LLAMA_SPEC_TYPE=draft-mtp turns n-gram drafts off.
              # (Not LLAMA_ARG_SPEC_TYPE: the server reads that too, and --spec-type adds to the list.)
              spec_type="''${LLAMA_SPEC_TYPE:-draft-mtp,ngram-mod}"
              spec_note=""
              case "$spec_type" in *ngram-mod*) spec_note=" (ngram-mod n_max 16, n_min 12, n_match 24)" ;; esac

              {
                echo "=========================================================="
                echo " llama.cpp OpenAI-compatible server (ROCm, ${gpuTarget})"
                echo " Binary:      $server"
                echo " Model:       ''${LLAMA_ARG_MODEL:-(from arguments)}"
                echo " Endpoint:    http://$host:$port/v1"
                echo " Context:     $ctx_size tokens"
                echo " KV resident: $resident_mib MiB (prefill), $resident_mib_decode MiB (decode)"
                echo " Reasoning:   effort $reasoning_effort by default (per request: reasoning_effort)"
                echo " Speculative: $spec_type$spec_note"
                echo " Devices:     ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES"
                echo " Tuning env:  GPU_MAX_HW_QUEUES=''${GPU_MAX_HW_QUEUES:-default}" \
                  "ROC_ACTIVE_WAIT_TIMEOUT=''${ROC_ACTIVE_WAIT_TIMEOUT:-default}" \
                  "GGML_CUDA_DISABLE_GRAPHS=''${GGML_CUDA_DISABLE_GRAPHS:-unset}"
                echo "=========================================================="
              } >&2

              exec "$server" \
                --host "$host" \
                --port "$port" \
                -c "$ctx_size" \
                -np 1 \
                -dev ROCm0 \
                -fa on \
                -fit off \
                -ctk q8_0 \
                -ctv q4_0 \
                -ctkd q8_0 \
                -ctvd q4_0 \
                --kv-stream \
                --kv-resident-mib "$resident_mib" \
                --kv-resident-mib-decode "$resident_mib_decode" \
                --spec-type "$spec_type" \
                --spec-ngram-mod-n-max 16 \
                --spec-ngram-mod-n-min 12 \
                --spec-ngram-mod-n-match 24 \
                --chat-template-kwargs "{\"reasoning_effort\":\"$reasoning_effort\"}" \
                -b 2048 \
                -ub 512 \
                --alias "$alias_name" \
                "$@"
            '';
          };
        in
        pkgs.runCommand "llama-serve-openai" { meta.mainProgram = "llama-serve-openai"; } ''
          mkdir -p $out/bin
          ln -s ${lib.getExe script} $out/bin/llama-serve-openai
          ln -s ${lib.getExe script} $out/bin/serve-openai
        '';

      # `nix run .#serve-openai` runs the Nix-built qualified server.
      serve-openai = mkServe {
        defaultServer = "${llama-cpp}/bin/llama-server";
        missingHint = "set LLAMA_SERVER_BIN";
      };
      # The dev shell serves the local build instead, so entering it never builds the package.
      serve-openai-local = mkServe {
        defaultServer = "$PWD/build-rocm/bin/llama-server";
        missingHint = "run llama-rocm-configure && cmake --build build-rocm, or set LLAMA_SERVER_BIN";
      };
    in
    {
      packages = lib.optionalAttrs supported {
        rocm-rx9070xt = llama-cpp;
        llama-rocm-env = rocm-env;
        inherit serve-openai;
      };

      devShells = lib.optionalAttrs supported {
        rocm-rx9070xt = pkgs.mkShell {
          name = "rocm-rx9070xt";
          inputsFrom = [ llama-cpp ];
          packages = [
            rocm.rocminfo
            rocm.rocm-smi
            rocm.amdsmi
            rocm.rocprofiler-sdk
            rocm.rocm-bandwidth-test
            (pkgs.python3.withPackages (p: [
              p.numpy
              p.matplotlib
            ]))
            rocm-env
            rocm-configure
            serve-openai-local
          ];
          # mkShell does not inherit the package's env attribute through inputsFrom.
          env = {
            ROCM_PATH = "${rocm.clr}";
            HIP_DEVICE_LIB_PATH = "${rocm.rocm-device-libs}/amdgcn/bitcode";
          };
          shellHook = ''
            if [ -z "''${ROCR_VISIBLE_DEVICES+x}" ] && devices=$(llama-rocm-env --print 2>/dev/null); then
              export ROCR_VISIBLE_DEVICES="$devices"
            fi
            echo "Entering rocm-rx9070xt devShell (ROCR_VISIBLE_DEVICES=''${ROCR_VISIBLE_DEVICES:-unset})"
            echo "  llama-rocm-configure [build-dir]   configure a local build with the qualified flags"
            echo "  serve-openai -m <model.gguf>       launch ./build-rocm/bin/llama-server with the qualified flags"
          '';
        };
      };
    };
}
