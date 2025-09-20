import subprocess
import pandas as pd
from pathlib import Path

def run_benchmarks(bin_dir, runner="./build/benchmark_runner"):
    bin_dir = Path(bin_dir)
    results = []

    # Loop over all .bin files in the directory
    for bin_file in sorted(bin_dir.glob("*.bin")):
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

    # Convert to pandas dataframe
    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    df = run_benchmarks("./benchmarks/data")
    print(df)
    # Optionally save to CSV
    df.to_csv("benchmark_results.csv", index=False)
