import subprocess
import pandas as pd
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

def _run_single(bin_file, runner, settings):
    cmd = [runner, str(bin_file)]
    if settings:
        cmd.append(settings)
    try:
        output = subprocess.check_output(cmd, text=True)
        parts = output.strip().split()
        filename = parts[0]
        prob_name = filename.removesuffix(".bin")
        return {
            "name": Path(prob_name).name,
            "exit_code": int(parts[1]),
            "iters": int(parts[2]),
            "ir_iters": int(parts[3]),
            "setup_time": float(parts[4]),
            "solve_time": float(parts[5]),
        }
    except subprocess.CalledProcessError as e:
        print(f"Error running {bin_file}: {e}")
        prob_name = str(bin_file).removesuffix(".bin")
        return {
            "name": Path(prob_name).name,
            "exit_code": -1,
            "iters": -1,
            "ir_iters": -1,
            "setup_time": None,
            "solve_time": None,
        }

def run_benchmarks(bin_dir, runner="./build/benchmark_runner", output_csv="benchmark_results.csv", settings=None, max_workers=None):
    bin_dir = Path(bin_dir)
    bin_files = sorted(bin_dir.rglob("*.bin"))
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_single, f, runner, settings): f for f in bin_files}
        for future in as_completed(futures):
            print(futures[future])
            results.append(future.result())

    results.sort(key=lambda r: r["name"])
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
