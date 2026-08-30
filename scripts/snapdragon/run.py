#!/usr/bin/env python3
#
# Run llama.cpp tools on Snapdragon devices (natively, via ADB, or SSH).
#

import sys
import os
import argparse
import subprocess
import platform
import shlex
import logging

logger = logging.getLogger("run")


def parse_target(target_str):
    if not target_str:
        return None, None
    if target_str.startswith("adb") or target_str.startswith("android"):
        parts = target_str.split(":", 1)
        serial = parts[1] if len(parts) > 1 else None
        return "android", serial
    elif target_str.startswith("lnx") or target_str.startswith("linux") or target_str.startswith("ubuntu"):
        parts = target_str.split(":", 1)
        host = parts[1] if len(parts) > 1 else None
        return "linux", host
    elif target_str in ("wos", "windows"):
        return "windows", None
    else:
        return None, None


def shlex_join(args_list):
    if hasattr(shlex, 'join'):
        return shlex.join(args_list)
    import pipes
    return " ".join(pipes.quote(x) for x in args_list)


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    # Split arguments at '--'
    if '--' in sys.argv:
        idx = sys.argv.index('--')
        run_args = sys.argv[1:idx]
        cmd_args = sys.argv[idx + 1:]
    else:
        run_args = sys.argv[1:]
        cmd_args = []

    parser = argparse.ArgumentParser(
        description="Unified runner for llama.cpp tools on Snapdragon (natively, via ADB, or via SSH)."
    )
    parser.add_argument("--target", help="Execution target (e.g. android[:serial]/adb[:serial], linux:[user@]host/lnx:[user@]host/ubuntu:[user@]host, windows/wos) (default: local run)")
    parser.add_argument("--target-dir", help="Target directory on the device (default: /data/local/tmp/llama.cpp for Android, ~/llama.cpp for Linux)")
    parser.add_argument("--install-dir", help="Install directory name (defaults to pkg-TARGET or pkg-TARGET-dbg prefix based on target)")
    parser.add_argument("--debug", action="store_true", help="Use debug build (defaults to pkg-TARGET-dbg folder)")
    parser.add_argument("--devices", "--device", "-d", help="Select execution devices (split into NPU and OpenCL GPUs automatically, default: HTP0)")
    parser.add_argument("--verbose", help="Verbose level (enables both Hexagon and OpenCL kernel cache debugging)")
    parser.add_argument("--profile", help="Profiling flag (enables Hexagon profiling and OpenCL autotuning)")
    parser.add_argument("--sched-debug", action="store_true", help="Enable GGML/llama.cpp scheduler debug output (GGML_SCHED_DEBUG=2)")
    parser.add_argument("--mtmd-device", help="Specify the backend device ID for Multi-Threaded Multi-Device setup (MTMD_BACKEND_DEVICE)")

    # Hexagon specific parameters
    parser.add_argument("--hex-verbose", help="Enable verbose logging (GGML_HEXAGON_VERBOSE)")
    parser.add_argument("--hex-profile", help="Enable NPU/Hexagon profiling and performance metrics print (GGML_HEXAGON_PROFILE)")
    parser.add_argument("--hex-nhvx", help="Number of HVX units to use (GGML_HEXAGON_NHVX)")
    parser.add_argument("--hex-nhmx", help="Number of HMX units to use. 0 disables HMX power-up (GGML_HEXAGON_NHMX)")
    parser.add_argument("--hex-hostbuf", help="Enable host buffers (GGML_HEXAGON_HOSTBUF)")
    parser.add_argument("--hex-opbatch", help="Maximum number of operations to batch into a single HTP execution (GGML_HEXAGON_OPBATCH)")
    parser.add_argument("--hex-opqueue", help="Size of the asynchronous NPU operation queue (GGML_HEXAGON_OPQUEUE)")
    parser.add_argument("--hex-oppoll", default="1", help="Enable (1) or Disable (0) polling for NPU opbatch completion (GGML_HEXAGON_OPPOLL) (default: 1)")
    parser.add_argument("--hex-opfilter", help="Regex pattern to filter/select which operators are offloaded to NPU (GGML_HEXAGON_OPFILTER)")
    parser.add_argument("--hex-opfusion", help="NPU graph node fusion optimization level (0: disabled, 1: enabled) (GGML_HEXAGON_OPFUSION)")
    parser.add_argument("--hex-vmem", help="Maximum NPU VMEM size limit in MB to allocate (GGML_HEXAGON_VMEM)")
    parser.add_argument("--hex-mbuf", help="Maximum host buffer size limit in MB to allocate (GGML_HEXAGON_MBUF)")
    parser.add_argument("--hex-mm-select", help="Select MUL_MAT and MUL_MAT_ID kernel (GGML_HEXAGON_MM_SELECT) 3:HMX,2:HVX-tiled,1:HVX-flat,0:disable")
    parser.add_argument("--hex-fa-select", help="Select Flash Attention kernel (GGML_HEXAGON_FA_SELECT) 2:HMX,1:HVX,0:disable")
    parser.add_argument("--hex-ar-select", help="Select All-Reduce kernel (GGML_HEXAGON_AR_SELECT) 1:enable,0:disable")
    parser.add_argument("--hex-etm", help="Enable Embedded Trace Macrocell hardware tracing / trace logging (GGML_HEXAGON_ETM)")
    parser.add_argument("--hex-arch", help="Target Hexagon NPU architecture version override (v73, v75, v79, v81, etc.) (GGML_HEXAGON_ARCH)")
    parser.add_argument("--hex-optrace", help="Trace buffer size in number of records (GGML_HEXAGON_OPTRACE)")

    # OpenCL specific parameters
    parser.add_argument("--cl-platform", help="Select OpenCL platform name/regex (e.g. Qualified Qualcomm OpenCL platform) (GGML_OPENCL_PLATFORM)")
    parser.add_argument("--cl-device", help="Select OpenCL device name/regex (e.g. Adreno GPU) (GGML_OPENCL_DEVICE)")
    parser.add_argument("--cl-opfilter", help="Regex pattern to filter/select which operators are offloaded to OpenCL (GGML_OPENCL_OPFILTER)")
    parser.add_argument("--cl-disable-fusion", action="store_true", help="Disable OpenCL kernel fusion optimizations (GGML_OPENCL_DISABLE_FUSION)")
    parser.add_argument("--cl-cache-dir", help="Directory path to store compiled OpenCL program binaries (GGML_OPENCL_KERNEL_CACHE_DIR)")
    parser.add_argument("--cl-cache-debug", help="Enable verbose debugging logs for the kernel caching system (GGML_OPENCL_KERNEL_CACHE_DEBUG)")
    parser.add_argument("--cl-fa-tune", action="store_true", help="Enable automatic Flash Attention kernel autotuning (GGML_OPENCL_FA_TUNE)")
    parser.add_argument("--cl-adreno-xmem", action="store_true", help="Enforce matmul using texture/image (xmem) memory paths on Adreno GPUs (GGML_OPENCL_ADRENO_XMEM_GEMM)")
    parser.add_argument("--cl-adreno-large-buffer", action="store_true", help="Allow allocating larger buffer sizes on Adreno GPUs (GGML_OPENCL_ADRENO_USE_LARGE_BUFFER)")

    args = parser.parse_args(run_args)

    if not cmd_args:
        parser.print_help()
        logger.error("\nError: No command specified after '--'")
        sys.exit(1)

    target_type = None
    target_val = None
    target_prefix = None
    if args.target:
        target_type, target_val = parse_target(args.target)
        if not target_type:
            logger.error(f"Error: Invalid target format '{args.target}'. Must be android[:serial]/adb[:serial], linux:[user@]host/lnx:[user@]host/ubuntu:[user@]host, or windows/wos.")
            sys.exit(1)
        target_prefix = args.target.split(":", 1)[0]

    # Resolve install directory
    install_dir = args.install_dir
    if not install_dir:
        if target_prefix:
            suffix = "-dbg" if args.debug else ""
            install_dir = f"pkg-{target_prefix}{suffix}"
        else:
            # Smart branch folder detection for local run if default is not set
            prefixes = ("wos", "windows", "lnx", "linux", "ubuntu", "adb", "android")
            suffixes = ("-dbg", "") if args.debug else ("", "-dbg")
            found = False
            for suffix in suffixes:
                for prefix in prefixes:
                    test_path = f"./pkg-{prefix}{suffix}/llama.cpp"
                    if os.path.exists(test_path):
                        install_dir = f"pkg-{prefix}{suffix}"
                        found = True
                        break
                if found:
                    break
            if not install_dir:
                install_dir = "pkg-android"  # Fallback default

    # Host side package path
    package_path = os.path.join(install_dir, "llama.cpp")

    # Environment variables to map
    env_vars = {}

    def set_env(env_name, opt_val):
        if opt_val is not None:
            env_vars[env_name] = str(opt_val)
        elif env_name in os.environ:
            env_vars[env_name] = os.environ[env_name]

    # Resolve and filter devices (HTP vs OpenCL)
    devices_val = args.devices if args.devices is not None else "HTP0"
    if devices_val.isdigit():
        hex_devices = devices_val
        cl_device = ""
    else:
        parts = [p.strip() for p in devices_val.split(",")]
        # Any device containing "htp" is Hexagon, rest is OpenCL
        hex_parts = [p for p in parts if "htp" in p.lower()]
        cl_parts = [p for p in parts if "htp" not in p.lower()]
        hex_devices = ",".join(hex_parts)
        cl_device = ",".join(cl_parts)

    # Set Hexagon devices
    if hex_devices:
        env_vars["GGML_HEXAGON_DEVICES"] = hex_devices
    elif "GGML_HEXAGON_DEVICES" in os.environ:
        env_vars["GGML_HEXAGON_DEVICES"] = os.environ["GGML_HEXAGON_DEVICES"]

    # Set OpenCL device (unless overridden by --cl-device)
    final_cl_device = args.cl_device if args.cl_device is not None else cl_device
    if final_cl_device:
        env_vars["GGML_OPENCL_DEVICE"] = final_cl_device
    elif "GGML_OPENCL_DEVICE" in os.environ:
        env_vars["GGML_OPENCL_DEVICE"] = os.environ["GGML_OPENCL_DEVICE"]

    # Map shared & backend-specific parameters with correct overrides

    # Verbose logging mapping
    hex_verbose_val = args.hex_verbose if args.hex_verbose is not None else args.verbose
    set_env("GGML_HEXAGON_VERBOSE", hex_verbose_val)

    cl_cache_debug_val = args.cl_cache_debug if args.cl_cache_debug is not None else args.verbose
    set_env("GGML_OPENCL_KERNEL_CACHE_DEBUG", cl_cache_debug_val)

    # Profiling mapping
    hex_profile_val = args.hex_profile if args.hex_profile is not None else args.profile
    set_env("GGML_HEXAGON_PROFILE", hex_profile_val)

    if args.cl_fa_tune or args.profile is not None:
        env_vars["GGML_OPENCL_FA_TUNE"] = "1"
    elif "GGML_OPENCL_FA_TUNE" in os.environ:
        env_vars["GGML_OPENCL_FA_TUNE"] = os.environ["GGML_OPENCL_FA_TUNE"]

    # Other Hexagon environment variables
    set_env("GGML_HEXAGON_NHVX", args.hex_nhvx)
    set_env("GGML_HEXAGON_NHMX", args.hex_nhmx)
    set_env("GGML_HEXAGON_HOSTBUF", args.hex_hostbuf)
    set_env("GGML_HEXAGON_OPBATCH", args.hex_opbatch)
    set_env("GGML_HEXAGON_OPQUEUE", args.hex_opqueue)
    set_env("GGML_HEXAGON_OPPOLL", args.hex_oppoll)
    set_env("GGML_HEXAGON_OPFILTER", args.hex_opfilter)
    set_env("GGML_HEXAGON_OPFUSION", args.hex_opfusion)
    set_env("GGML_HEXAGON_VMEM", args.hex_vmem)
    set_env("GGML_HEXAGON_MBUF", args.hex_mbuf)
    set_env("GGML_HEXAGON_MM_SELECT", args.hex_mm_select)
    set_env("GGML_HEXAGON_FA_SELECT", args.hex_fa_select)
    set_env("GGML_HEXAGON_AR_SELECT", args.hex_ar_select)
    set_env("GGML_HEXAGON_ETM", args.hex_etm)
    set_env("GGML_HEXAGON_ARCH", args.hex_arch)
    set_env("GGML_HEXAGON_OPTRACE", args.hex_optrace)
    set_env("MTMD_BACKEND_DEVICE", args.mtmd_device)

    # OpenCL environment variables
    set_env("GGML_OPENCL_PLATFORM", args.cl_platform)
    set_env("GGML_OPENCL_OPFILTER", args.cl_opfilter)
    set_env("GGML_OPENCL_KERNEL_CACHE_DIR", args.cl_cache_dir)

    if args.cl_disable_fusion:
        env_vars["GGML_OPENCL_DISABLE_FUSION"] = "1"
    elif "GGML_OPENCL_DISABLE_FUSION" in os.environ:
        env_vars["GGML_OPENCL_DISABLE_FUSION"] = os.environ["GGML_OPENCL_DISABLE_FUSION"]

    if args.cl_adreno_xmem:
        env_vars["GGML_OPENCL_ADRENO_XMEM_GEMM"] = "1"
    elif "GGML_OPENCL_ADRENO_XMEM_GEMM" in os.environ:
        env_vars["GGML_OPENCL_ADRENO_XMEM_GEMM"] = os.environ["GGML_OPENCL_ADRENO_XMEM_GEMM"]

    if args.cl_adreno_large_buffer:
        env_vars["GGML_OPENCL_ADRENO_USE_LARGE_BUFFER"] = "1"
    elif "GGML_OPENCL_ADRENO_USE_LARGE_BUFFER" in os.environ:
        env_vars["GGML_OPENCL_ADRENO_USE_LARGE_BUFFER"] = os.environ["GGML_OPENCL_ADRENO_USE_LARGE_BUFFER"]

    if args.sched_debug:
        env_vars["GGML_SCHED_DEBUG"] = "2"

    # Resolve executable path
    executable = cmd_args[0]
    known_binaries = ["llama-cli", "llama-bench", "llama-completion", "llama-mtmd-cli", "test-backend-ops"]
    if executable in known_binaries:
        if target_type in ("android", "linux"):
            resolved_exec = f"./bin/{executable}"
        else:
            if platform.system() == "Windows":
                resolved_exec = os.path.normpath(os.path.join(package_path, "bin", f"{executable}.exe"))
            else:
                resolved_exec = os.path.normpath(os.path.join(package_path, "bin", executable))
        cmd_args[0] = resolved_exec

    # Infer device string to pass to the tool
    basename = os.path.basename(executable)
    if basename.endswith(".exe"):
        basename = basename[:-4]

    device_val = None
    if basename == "test-backend-ops":
        for i in range(len(cmd_args)):
            if cmd_args[i] in ("-p", "--params") and i + 1 < len(cmd_args):
                val = cmd_args[i + 1]
                new_val = ""
                for j, char in enumerate(val):
                    if char in ('[', ']'):
                        if j > 0 and val[j - 1] == '\\':
                            new_val += char
                        else:
                            new_val += '\\' + char
                    else:
                        new_val += char
                cmd_args[i + 1] = new_val

        has_b = any(arg == "-b" for arg in cmd_args)
        if not has_b:
            if args.devices:
                if args.devices.isdigit():
                    n = int(args.devices)
                    device_val = ",".join(f"HTP{i}" for i in range(n))
                else:
                    device_val = args.devices
            elif "D" in os.environ:
                device_val = os.environ["D"]
            elif "DEVICE" in os.environ:
                device_val = os.environ["DEVICE"]
            else:
                device_val = "HTP0"
            if device_val:
                cmd_args += ["-b", device_val]
    else:
        has_device = any(arg.startswith("--device") for arg in cmd_args)
        if not has_device:
            if args.devices:
                if args.devices.isdigit():
                    n = int(args.devices)
                    device_val = ",".join(f"HTP{i}" for i in range(n))
                else:
                    device_val = args.devices
            elif "D" in os.environ:
                device_val = os.environ["D"]
            elif "DEVICE" in os.environ:
                device_val = os.environ["DEVICE"]
            else:
                device_val = "HTP0"
            if device_val:
                cmd_args += ["--device", device_val]

    # Automatically add -v to known llama tools if sched-debug, verbose, or profile are set
    verbose_trigger = (
        args.sched_debug
        or args.verbose is not None
        or args.profile is not None
        or args.hex_verbose is not None
        or args.hex_profile is not None
        or args.hex_optrace is not None
    )
    if verbose_trigger and basename in ("llama-cli", "llama-completion", "llama-bench", "llama-server", "llama-mtmd-cli"):
        if "-v" not in cmd_args and "--verbose" not in cmd_args:
            cmd_args.append("-v")

    # Inject defaults for llama-cli, llama-completion, and llama-server if not overridden by the user
    if basename in ("llama-cli", "llama-completion", "llama-server"):
        if "-ngl" not in cmd_args and "--n-gpu-layers" not in cmd_args:
            cmd_args += ["-ngl", "99"]
        if "--ubatch-size" not in cmd_args and "-ub" not in cmd_args:
            cmd_args += ["--ubatch-size", "1024"]
        if "-fa" not in cmd_args and "--flash-attn" not in cmd_args:
            cmd_args += ["-fa", "on"]

    if basename in ("llama-cli", "llama-completion", "llama-server", "llama-bench"):
        if "-t" not in cmd_args and "--threads" not in cmd_args:
            cmd_args += ["-t", "6"]

    # Resolve target directory on device
    target_dir = args.target_dir
    if not target_dir:
        target_dir = "/data/local/tmp/llama.cpp" if target_type == "android" else "~/llama.cpp"

    if target_type == "android":
        # Run via ADB
        adb_base = ["adb"]
        if target_val: # serial
            adb_base += ["-s", target_val]

        env_parts = [
            "LD_LIBRARY_PATH=./lib",
            "ADSP_LIBRARY_PATH=./lib"
        ]
        for k, v in env_vars.items():
            env_parts.append(f"{k}={v}")
        env_str = " ".join(env_parts)

        cmd_str = shlex_join(cmd_args)
        adb_shell_cmd = f"cd {target_dir} && ulimit -c unlimited && {env_str} {cmd_str}"
        full_cmd = adb_base + ["shell", adb_shell_cmd]

        logger.info(f"+ {' '.join(full_cmd)}")
        res = subprocess.run(full_cmd)
        sys.exit(res.returncode)

    elif target_type == "linux":
        ssh_host = target_val
        if not ssh_host:
            logger.error("Error: SSH host not specified in target (e.g. use linux:user@host, lnx:user@host, or ubuntu:user@host). Cannot execute.")
            sys.exit(1)

        # Linux remote run via SSH
        env_parts = [
            "LD_LIBRARY_PATH=./lib",
            "ADSP_LIBRARY_PATH=./lib"
        ]
        for k, v in env_vars.items():
            env_parts.append(f"{k}={v}")
        env_str = " ".join(env_parts)

        cmd_str = shlex_join(cmd_args)
        ssh_shell_cmd = f"cd {target_dir} && ulimit -c unlimited && {env_str} {cmd_str}"
        full_cmd = ["ssh", ssh_host, ssh_shell_cmd]

        logger.info(f"+ {' '.join(full_cmd)}")
        res = subprocess.run(full_cmd)
        sys.exit(res.returncode)

    elif target_type == "windows":
        logger.info("Windows target execution is currently a stub.")
        sys.exit(0)

    else:
        # Run locally
        local_env = os.environ.copy()
        lib_dir = os.path.normpath(os.path.join(package_path, "lib"))
        local_env["ADSP_LIBRARY_PATH"] = lib_dir
        if platform.system() == "Windows":
            local_env["PATH"] = lib_dir + os.path.pathsep + local_env.get("PATH", "")
        else:
            local_env["LD_LIBRARY_PATH"] = lib_dir + os.path.pathsep + local_env.get("LD_LIBRARY_PATH", "")

        for k, v in env_vars.items():
            local_env[k] = v

        logger.info(f"+ {shlex_join(cmd_args)}")
        res = subprocess.run(cmd_args, env=local_env)
        sys.exit(res.returncode)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
        sys.exit(130)
