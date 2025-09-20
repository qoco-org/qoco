import subprocess
import pandas as pd
from pathlib import Path
import sys

def run_benchmarks(bin_dir, runner="./build/benchmark_runner", output_csv="benchmark_results.csv"):
    bin_dir = Path(bin_dir)
    results = []

    # Loop over all .bin files in the directory
    num = 0
    for bin_file in sorted(bin_dir.glob("*.bin")):
        if num > 3:
            break
        print(bin_file)
        # Call the benchmark_runner
        try:
            output = subprocess.check_output([runner, str(bin_file)], text=True)
            # Example output: ./benchmarks/data/TAME.bin 1 0.000026 0.000021
            parts = output.strip().split()
            filename = parts[0]
            exit_code = int(parts[1])
            setup_time = float(parts[2])
            solve_time = float(parts[3])

            results.append({
                "name": Path(filename).name,
                "exit_code": exit_code,
                "setup_time": setup_time,
                "solve_time": solve_time
            })
        except subprocess.CalledProcessError as e:
            print(f"Error running {bin_file}: {e}")
            results.append({
                "name": bin_file.name,
                "exit_code": -1,
                "setup_time": None,
                "solve_time": None
            })
    num += 1
    # Convert to pandas dataframe
    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)

if __name__ == "__main__":
    # If a command-line argument is given, use it as the CSV filename
    output_name = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"
    run_benchmarks("./benchmarks/data", output_csv=f"{output_name}.csv")
