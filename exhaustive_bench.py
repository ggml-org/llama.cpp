import subprocess
import json
import os

model = "/mnt/p5/models/Qwen3.5-35B-A3B-abliterated-Q4_K_M.gguf"
bench_bin = "/home/cune/llama.cpp/build/bin/llama-bench"

ngls = [26, 28, 30]
ncmoes = [32, 36, 40]
threads = [10, 12, 14]
ubatches = [256, 512]
polls = [0, 50, 100]

print("| NGL | NCMOE | Threads | Ubatch | Poll | Prompt (t/s) | Gen (t/s) | Status |")
print("|---:|---:|---:|---:|---:|---:|---:|---|")

best_gen = 0.0
best_config = ""

for ngl in ngls:
    for ncmoe in ncmoes:
        for t in threads:
            for ub in ubatches:
                for poll in [50]: # test key poll values
                    cmd = [
                        bench_bin,
                        "-m", model,
                        "-p", "16",
                        "-n", "32",
                        "-r", "1",
                        "--no-warmup",
                        "-ngl", str(ngl),
                        "-ncmoe", str(ncmoe),
                        "-t", str(t),
                        "-ub", str(ub),
                        "--poll", str(poll),
                        "-fa", "1",
                        "-o", "json"
                    ]
                    env = os.environ.copy()
                    env["RECURRENT_D"] = "12"
                    
                    try:
                        res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
                        if res.returncode == 0:
                            data = json.loads(res.stdout)
                            pp_ts = 0.0
                            tg_ts = 0.0
                            for item in data:
                                if item.get("n_prompt", 0) > 0:
                                    pp_ts = item.get("avg_ts", 0.0)
                                if item.get("n_gen", 0) > 0:
                                    tg_ts = item.get("avg_ts", 0.0)
                            
                            print(f"| {ngl} | {ncmoe} | {t} | {ub} | {poll} | {pp_ts:.2f} | {tg_ts:.2f} | OK |", flush=True)
                            if tg_ts > best_gen:
                                best_gen = tg_ts
                                best_config = f"-ngl {ngl} --n-cpu-moe {ncmoe} -t {t} -ub {ub} --poll {poll} -fa 1"
                        else:
                            print(f"| {ngl} | {ncmoe} | {t} | {ub} | {poll} | - | - | OOM / Fail |", flush=True)
                    except Exception as e:
                        print(f"| {ngl} | {ncmoe} | {t} | {ub} | {poll} | - | - | Timeout / Fail |", flush=True)

print(f"\nBEST CONFIGURATION: {best_config} with {best_gen:.2f} t/s")
