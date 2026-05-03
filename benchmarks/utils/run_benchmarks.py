import subprocess
import pandas as pd
from pathlib import Path
import sys

def run_benchmarks(bin_dir, runner="./build/benchmark_runner", output_csv="benchmark_results.csv", settings=None):
    bin_dir = Path(bin_dir)
    results = []
    # Loop over all .bin files in the directory tree
    for bin_file in sorted(bin_dir.rglob("*.bin")):
        print(bin_file)
        # Call the benchmark_runner
        try:
            if settings:
                output = subprocess.check_output([runner, str(bin_file), settings], text=True)
            else:
                output = subprocess.check_output([runner, str(bin_file)], text=True)
            parts = output.strip().split()
            filename = parts[0]
            exit_code = int(parts[1])
            iters = int(parts[2])
            # Older benchmark_runner builds omit ir_iters (5 fields vs 6)
            if len(parts) == 6:
                ir_iters = int(parts[3])
                setup_time = float(parts[4])
                solve_time = float(parts[5])
            else:
                ir_iters = 0
                setup_time = float(parts[3])
                solve_time = float(parts[4])

            prob_name = filename.removesuffix(".bin")

            results.append({
                "name": Path(prob_name).name,
                "exit_code": exit_code,
                "iters": iters,
                "ir_iters": ir_iters,
                "setup_time": setup_time,
                "solve_time": solve_time
            })
        except subprocess.CalledProcessError as e:
            print(f"Error running {bin_file}: {e}")
            results.append({
                "name": prob_name,
                "exit_code": -1,
                "iters": -1,
                "ir_iters": -1,
                "setup_time": None,
                "solve_time": None
            })
    # Convert to pandas dataframe
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

# if __name__ == "__main__":
#     # If a command-line argument is given, use it as the CSV filename
#     output_name = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"
#     run_benchmarks(bin_dir="./benchmarks/data", output_csv=f"{output_name}.csv")
