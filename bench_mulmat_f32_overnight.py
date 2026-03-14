#!/usr/bin/env python3
"""Overnight MUL_MAT F32×F32 benchmark (~11 hrs).

Injects C++ loops into make_test_cases_perf(), single build, single run.

Phases:
  A) Fine square sweep M=K=N step 16, 16..8192
  B) 2D M×K grids at N∈{1,16,64,128,256,1024,4096}, step 128
  C) Fine N sweeps at 10 key (M,K) pairs, step 16
  D) Random fill on known slices
"""

import subprocess, re, os, shutil, sys, time, json

SRCFILE = "tests/test-backend-ops.cpp"
CSVFILE = "build/mul_mat_f32_perf.csv"
TIMEFILE = "build/bench_overnight_time.json"
MARKER_BEGIN = "// BENCH_OVERNIGHT_BEGIN"
MARKER_END = "// BENCH_OVERNIGHT_END"

def count_expected_points():
    """Count total test cases the C++ loops will generate."""
    seen = set()
    # Phase A: square step 16
    for s in range(16, 8192+1, 16):
        seen.add((s, s, s))
    # Phase B: 2D grids
    m_vals = list(range(16, 8192+1, 128)) + [8192]
    k_vals = list(range(16, 8192+1, 128)) + [8192]
    for n in [1, 16, 64, 128, 256, 1024, 4096]:
        for m in m_vals:
            for k in k_vals:
                seen.add((m, k, n))
    # Phase C: N sweeps
    for m, k in [(128,128),(256,256),(512,512),(1024,1024),(2048,2048),
                 (4096,4096),(8192,8192),(4096,1024),(4096,14336),(8192,2048)]:
        for n in range(1, 4096+1, 16):
            seen.add((m, k, n))
    # +1 for the existing F32 perf test case in source
    return len(seen) + 1

INJECTED_CODE = r"""
    // Phase A: fine square sweep M=K=N, step 16
    for (int s = 16; s <= 8192; s += 16) {
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, s, s, s, {1,1}, {1,1}));
    }

    // Phase B: 2D M×K grids at key N slices, step 128
    for (int n : {1, 16, 64, 128, 256, 1024, 4096}) {
        for (int m = 16; m <= 8192; m += 128) {
            for (int k = 16; k <= 8192; k += 128) {
                test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, m, n, k, {1,1}, {1,1}));
            }
        }
        // include endpoints 8192 if not hit by step
        for (int m = 16; m <= 8192; m += 128) {
            test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, m, n, 8192, {1,1}, {1,1}));
        }
        for (int k = 16; k <= 8192; k += 128) {
            test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 8192, n, k, {1,1}, {1,1}));
        }
    }

    // Phase C: fine N sweeps at key (M,K) pairs, step 16
    for (auto [m, k] : std::vector<std::pair<int,int>>{
        {128,128}, {256,256}, {512,512}, {1024,1024}, {2048,2048},
        {4096,4096}, {8192,8192}, {4096,1024}, {4096,14336}, {8192,2048}
    }) {
        for (int n = 1; n <= 4096; n += 16) {
            test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, m, n, k, {1,1}, {1,1}));
        }
    }
"""


def save_timing(t_start, phase="running"):
    with open(TIMEFILE, 'w') as f:
        json.dump({"start": t_start, "phase": phase,
                   "elapsed_h": (time.time() - t_start) / 3600}, f)


def main():
    t_start = time.time()
    save_timing(t_start, "init")

    # Patch source
    with open(SRCFILE) as f:
        src = f.read()
    shutil.copy2(SRCFILE, SRCFILE + ".bak")

    anchor = ('static std::vector<std::unique_ptr<test_case>> make_test_cases_perf() {\n'
              '    std::vector<std::unique_ptr<test_case>> test_cases;\n')
    idx = src.find(anchor)
    if idx == -1:
        print("ERROR: Could not find insertion point")
        shutil.move(SRCFILE + ".bak", SRCFILE)
        return
    idx += len(anchor)

    block = f"\n{MARKER_BEGIN}\n{INJECTED_CODE}\n{MARKER_END}\n"
    patched = src[:idx] + block + src[idx:]

    # Patch n_runs to have a minimum of 5 for more stable measurements
    old_nruns = ('n_runs = (int)std::min<int64_t>(ggml_graph_size(gf) - ggml_graph_n_nodes(gf), '
                 'target_flops / op_flops(out)) + 1;')
    new_nruns = ('n_runs = std::max(5, (int)std::min<int64_t>(ggml_graph_size(gf) - ggml_graph_n_nodes(gf), '
                 'target_flops / op_flops(out)) + 1);')
    if old_nruns in patched:
        patched = patched.replace(old_nruns, new_nruns)
        print("Patched n_runs minimum to 5")
    else:
        print("ERROR: Could not find n_runs line to patch")
        shutil.move(SRCFILE + ".bak", SRCFILE)
        return

    with open(SRCFILE, 'w') as f:
        f.write(patched)

    # Build
    print("Building...")
    sys.stdout.flush()
    save_timing(t_start, "building")
    r = subprocess.run(["cmake", "--build", "build", "-t", "test-backend-ops", f"-j{os.cpu_count()}"],
                       capture_output=True, text=True)
    print("Restoring source...")
    shutil.move(SRCFILE + ".bak", SRCFILE)
    if r.returncode != 0:
        print(f"Build failed:\n{r.stderr[-500:]}")
        return
    print("Build OK.")

    # Run
    print("Running benchmarks...")
    sys.stdout.flush()
    save_timing(t_start, "running")

    results = {}
    count = 0
    from collections import deque
    recent_times = deque(maxlen=200)  # rolling window for ETA
    t_last = time.time()
    est_total = count_expected_points()
    print(f"Expected ~{est_total} data points")

    proc = subprocess.Popen(
        ["./build/bin/test-backend-ops", "perf", "-o", "MUL_MAT", "-p", "type_a=f32,type_b=f32"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    for line in proc.stdout:
        if 'MUL_MAT(type_a=f32,type_b=f32' not in line or 'not supported' in line:
            continue
        m_m = re.search(r'm=(\d+)', line)
        n_m = re.search(r'n=(\d+)', line)
        k_m = re.search(r'k=(\d+)', line)
        clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
        flops = re.findall(r'([\d.]+)\s+([GTMK])FLOPS', clean)
        # Also extract runs and us/run: "  1234 runs -  567.89 us/run"
        runs_m = re.search(r'(\d+)\s+runs\s+-\s+([\d.]+)\s+us/run', clean)
        if m_m and n_m and k_m and flops:
            val, unit = flops[0]
            val = float(val)
            if unit == 'T': val *= 1000
            elif unit == 'M': val /= 1000
            elif unit == 'K': val /= 1e6
            n_runs = int(runs_m.group(1)) if runs_m else 0
            us_per_run = float(runs_m.group(2)) if runs_m else 0
            key = (int(m_m.group(1)), int(k_m.group(1)), int(n_m.group(1)))
            results[key] = (val, n_runs, us_per_run)
            count += 1
            now = time.time()
            recent_times.append(now - t_last)
            t_last = now
            elapsed_h = (now - t_start) / 3600
            # ETA from rolling median of recent per-point times
            sorted_times = sorted(recent_times)
            median_s = sorted_times[len(sorted_times) // 2]
            remaining = max(0, est_total - count)
            eta_h = remaining * median_s / 3600
            rate = 1.0 / median_s if median_s > 0 else 0
            if count % 100 == 0:
                save_timing(t_start, f"running ({count} pts, ETA {eta_h:.1f}h)")
            print(f"  [{count} | {elapsed_h:.2f}h | ETA {eta_h:.1f}h | {rate:.1f}pt/s] "
                  f"{key[0]:>5}x{key[1]:>5}x{key[2]:>5} = {val:.2f} GFLOPS "
                  f"({n_runs} runs, {us_per_run:.1f} us/run)")
            sys.stdout.flush()
    proc.wait()

    # Write CSV
    with open(CSVFILE, 'w') as f:
        f.write("M,K,N,GFLOPS,RUNS,US_PER_RUN\n")
        for key in sorted(results.keys()):
            m, k, n = key
            g, runs, us = results[key]
            gfmt = f"{g:.4f}" if g < 1 else f"{g:.2f}"
            f.write(f"{m},{k},{n},{gfmt},{runs},{us:.2f}\n")

    total_h = (time.time() - t_start) / 3600
    save_timing(t_start, "done")
    print(f"\nDone. {len(results)} total data points in {CSVFILE}")
    print(f"Total wall time: {total_h:.2f} hours ({total_h*60:.0f} minutes)")


if __name__ == "__main__":
    main()
