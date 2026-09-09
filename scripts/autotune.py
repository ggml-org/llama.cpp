#!/usr/bin/env python3
import os
import sys
import subprocess
import json
import itertools
import time
import argparse
import multiprocessing

def parse_args():
    parser = argparse.ArgumentParser(description="llamar.cpp GIGA Auto-tuning and Parameter Sweep Utility")
    parser.add_argument("--models-dir", type=str, default="./models", help="Directory containing GGUF models or direct path to a GGUF file")
    parser.add_argument("--threads", type=str, default="", help="Comma-separated CPU threads to test (default: auto-detect physical cores)")
    parser.add_argument("--ngl", type=str, default="16,24,32", help="Comma-separated GPU layers to test")
    parser.add_argument("--fa", type=str, default="on,off", help="Comma-separated Flash Attention options (on, off)")
    parser.add_argument("--recurrent-t", type=int, default=2, help="Recurrence passes T (RECURRENT_T) for benchmarks (1 = vanilla)")
    parser.add_argument("--recurrent-layer", type=int, default=16, help="Recurrent core layer index (RECURRENT_LAYER)")
    parser.add_argument("--builds", type=str, default="standard,no-vnni,native-o3", help="Comma-separated build configs to test")
    parser.add_argument("--output", type=str, default="benchmark_report.md", help="Path to generate markdown benchmark report")
    return parser.parse_args()

def run_cmd(cmd, env=None, timeout=900):
    try:
        res = subprocess.run(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return res.returncode, res.stdout, res.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"

def get_cpu_cores():
    try:
        return multiprocessing.cpu_count()
    except Exception:
        return 8

def find_gguf_models(path):
    if os.path.isfile(path) and path.endswith(".gguf"):
        return {os.path.basename(path): path}
    
    models = {}
    if os.path.isdir(path):
        for f in os.listdir(path):
            if f.endswith(".gguf") and "vocab" not in f.lower():
                models[f] = os.path.join(path, f)
    return models

def build_targets(src_dir, builds_to_test):
    print("=== COMPILING BUILD TARGETS ===")
    build_dir = os.path.join(src_dir, "build")
    run_cmd(f"rm -rf {build_dir} && mkdir -p {build_dir}")
    
    configs = {
        "standard": {
            "cmake": "cmake .. -DGGML_CUDA=ON -DGGML_AVX_VNNI=ON",
            "suffix": "-standard"
        },
        "no-vnni": {
            "cmake": "cmake .. -DGGML_CUDA=ON -DGGML_AVX_VNNI=OFF",
            "suffix": "-no-vnni"
        },
        "native-o3": {
            "cmake": "cmake .. -DGGML_CUDA=ON -DGGML_AVX_VNNI=ON -DCMAKE_CXX_FLAGS=\"-march=native -O3\"",
            "suffix": "-native-o3"
        }
    }
    
    compiled = {}
    for name in builds_to_test:
        if name not in configs:
            continue
        cfg = configs[name]
        print(f"Building: {name}...")
        
        # cmake
        ret, out, err = run_cmd(f"cd {build_dir} && {cfg['cmake']}")
        if ret != 0:
            print(f"CMake failed for config {name}: {err}")
            continue
            
        # make
        ret, out, err = run_cmd(f"cd {build_dir} && make -j{get_cpu_cores()} llama-bench llama-cli llama-server")
        if ret != 0:
            print(f"Make failed for config {name}: {err}")
            continue
            
        # Rename binaries
        bin_dir = os.path.join(build_dir, "bin")
        for target in ["llama-bench", "llama-cli", "llama-server"]:
            t_path = os.path.join(bin_dir, target)
            dest_path = f"{t_path}{cfg['suffix']}"
            if os.path.exists(t_path):
                run_cmd(f"cp {t_path} {dest_path}")
        compiled[name] = cfg
        print(f"Build {name} completed successfully!")
        
    return compiled

def parse_bench_output(stdout):
    pp16_ts = 0.0
    tg32_ts = 0.0
    for line in stdout.splitlines():
        if "pp16" in line:
            parts = line.split("|")
            if len(parts) >= 11:
                try:
                    pp16_ts = float(parts[10].split("±")[0].strip())
                except ValueError:
                    pass
        elif "tg32" in line:
            parts = line.split("|")
            if len(parts) >= 11:
                try:
                    tg32_ts = float(parts[10].split("±")[0].strip())
                except ValueError:
                    pass
    return pp16_ts, tg32_ts

def sweep(src_dir, models, compiled_builds, args_threads, args_ngl, args_fa, recurrent_t, recurrent_layer):
    print("=== PARAMETER SWEEP ===")
    bin_dir = os.path.join(src_dir, "build", "bin")
    
    # Process threads list
    if args_threads:
        threads = [int(t) for t in args_threads.split(",")]
    else:
        cores = get_cpu_cores()
        threads = sorted(list(set([cores // 2, cores, int(cores * 0.75)])))
        threads = [t for t in threads if t > 0]
        
    ngls = [int(n) for n in args_ngl.split(",")]
    fa_options = [f.strip() for f in args_fa.split(",")]
    
    results = []
    
    for model_name, model_path in models.items():
        print(f"\nModel: {model_name}")
        for build_name, cfg in compiled_builds.items():
            bench_bin = os.path.join(bin_dir, f"llama-bench{cfg['suffix']}")
            if not os.path.exists(bench_bin):
                continue
                
            for ngl, t, fa in itertools.product(ngls, threads, fa_options):
                # Clean processes and check memory usage
                run_cmd("pkill -9 -f llama-cli; pkill -9 -f llama-server; pkill -9 -f llama-bench; sleep 1")
                
                env = os.environ.copy()
                env["RECURRENT_T"] = str(recurrent_t)
                if recurrent_layer >= 0:
                    env["RECURRENT_LAYER"] = str(recurrent_layer)
                
                cmd = f"{bench_bin} -m {model_path} -p 16 -n 32 -r 1 --no-warmup -ngl {ngl} -ncmoe 36 -fa {fa} -t {t}"
                
                print(f"  Test: Build={build_name}, NGL={ngl}, Threads={t}, FA={fa} -> ", end="", flush=True)
                ret, out, err = run_cmd(cmd, env=env, timeout=120)
                
                if ret != 0:
                    print("FAILED / OOM")
                    results.append({
                        "model": model_name,
                        "build": build_name,
                        "ngl": ngl,
                        "threads": t,
                        "fa": fa,
                        "status": "failed",
                        "pp16": 0.0,
                        "tg32": 0.0
                    })
                else:
                    pp16, tg32 = parse_bench_output(out)
                    print(f"Prompt: {pp16:.2f} t/s, Gen: {tg32:.2f} t/s")
                    results.append({
                        "model": model_name,
                        "build": build_name,
                        "ngl": ngl,
                        "threads": t,
                        "fa": fa,
                        "status": "success",
                        "pp16": pp16,
                        "tg32": tg32
                    })
    return results

def write_report(results, report_path, compiled_builds, src_dir):
    print(f"\n=== GENERATING REPORT: {report_path} ===")
    
    # Group results by model
    grouped = {}
    for r in results:
        grouped.setdefault(r["model"], []).append(r)
        
    with open(report_path, "w") as f:
        f.write("# llamar.cpp GIGA Auto-tuning Report\n\n")
        f.write("Evaluation of dynamic parameter sweeps for optimized inference-time recurrence.\n\n")
        
        for model, res_list in grouped.items():
            f.write(f"## Model: {model}\n\n")
            f.write("| Build Config | NGL | Threads | Flash Attention | Status | PP16 (Prompt) | TG32 (Gen) |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
            
            # Sort success runs by generation speed descending
            res_list.sort(key=lambda x: x["tg32"], reverse=True)
            
            for r in res_list:
                status_str = "✅ Success" if r["status"] == "success" else "❌ OOM/Failed"
                f.write(f"| {r['build']} | {r['ngl']} | {r['threads']} | {r['fa']} | {r['status']} | {r['pp16']:.2f} t/s | {r['tg32']:.2f} t/s |\n")
            f.write("\n")
            
            best = next((x for x in res_list if x["status"] == "success"), None)
            if best:
                f.write(f"> **🏆 Best Parameters:** Build `{best['build']}` with NGL `{best['ngl']}`, Threads `{best['threads']}`, FA `{best['fa']}` yielding **{best['tg32']:.2f} t/s**.\n\n")
                
                # Copy champion binaries as default binaries
                bin_dir = os.path.join(src_dir, "build", "bin")
                suffix = compiled_builds[best['build']]['suffix']
                for target in ["llama-bench", "llama-cli", "llama-server"]:
                    run_cmd(f"cp {os.path.join(bin_dir, target + suffix)} {os.path.join(bin_dir, target)}")
                print(f"Copied best binaries from {best['build']} config to default targets!")

def main():
    args = parse_args()
    
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    models = find_gguf_models(args.models_dir)
    if not models:
        print(f"Error: No GGUF models found in path: {args.models_dir}")
        sys.exit(1)
        
    builds_to_test = [b.strip() for b in args.builds.split(",")]
    
    compiled_builds = build_targets(src_dir, builds_to_test)
    if not compiled_builds:
        print("Error: All build compilation failed.")
        sys.exit(1)
        
    results = sweep(src_dir, models, compiled_builds, args.threads, args.ngl, args.fa, args.recurrent_t, args.recurrent_layer)
    write_report(results, args.output, compiled_builds, src_dir)

if __name__ == "__main__":
    main()
