import yaml
import sys
import subprocess
from utils.run_benchmarks import run_benchmarks
import os

def checkout_branch(branch_name, is_diff=False):
    """
    Checkout a branch in CI.
    - For baseline branches (e.g. main), fetch from upstream.
    - For diff branch (the PR), just use HEAD.
    """
    try:
        # If in CI and if we need to check out pr branch, 
        # checkout branch_name which should contain the HEAD commit of pr branch.
        if os.environ.get("BRANCH_NAME") and is_diff:
            subprocess.run(["git", "checkout", branch_name], check=True)
            print(f"Checked out branch {branch_name}")
        else:
            # Fetch from upstream (your canonical repo)
            subprocess.run(["git", "fetch", "origin", branch_name], check=True)
            subprocess.run(["git", "checkout", "-B", branch_name, f"origin/{branch_name}"], check=True)
            print(f"Checked out branch {branch_name} from upstream")
    except subprocess.CalledProcessError as e:
        print(f"Failed to checkout branch {branch_name}: {e}")
        raise

def build_solver():
    build_dir = "build"
    cmake_cmd = [
    "cmake",
    "-B", build_dir,
    "-DCMAKE_CXX_COMPILER=clang++",
    "-DCMAKE_C_COMPILER=clang",
    "-DQOCO_BUILD_TYPE:STR=Release",
    "-DBUILD_QOCO_BENCHMARK_RUNNER:BOOL=True"
    ]

    # Run the command
    subprocess.run(cmake_cmd, check=True)
    subprocess.run(["cmake", "--build", build_dir], check=True)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def format_settings(yaml_data):
    if not yaml_data:  # YAML is empty
        return None

    settings = yaml_data.get("solver_settings")  # returns None if missing
    if not settings:  # None or empty dict
        return None

    return " ".join(f"{k}={v}" for k, v in settings.items())

if __name__ == "__main__":
    baseline_config = load_yaml(sys.argv[1])
    baseline_config_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    
    diff_config = load_yaml(sys.argv[2])
    diff_config_name = os.path.splitext(os.path.basename(sys.argv[2]))[0]

    # Safely get the branch names
    baseline_branch = baseline_config.get("qoco", {}).get("branch")
    if not baseline_branch:
        raise ValueError("Baseline YAML is missing qoco.branch")

    # If BRANCH_NAME is set (by CI), the diff_branch = BRANCH_NAME
    diff_branch = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() if os.environ.get("BRANCH_NAME") else diff_config.get("qoco", {}).get("branch")
    if not diff_branch:
        raise ValueError("Diff YAML is missing qoco.branch and BRANCH_NAME is not set")

    # Make results directory
    temp_results_dir = "/tmp/results/"
    subprocess.run(["mkdir", "-p", temp_results_dir], check=True)

    # Checkout and build the diff branch
    checkout_branch(diff_branch, is_diff=True)
    build_solver()

    # Run diff solver
    diff_settings = format_settings(diff_config)
    diff_results = temp_results_dir+f"{diff_config_name}.csv"
    run_benchmarks(bin_dir="./benchmarks/data", settings=diff_settings, output_csv=diff_results)

    # Checkout and build the baseline branch
    checkout_branch(baseline_branch, is_diff=False)
    build_solver()

    # Run baseline solver
    baseline_settings = format_settings(baseline_config)
    baseline_results = temp_results_dir+f"{baseline_config_name}.csv"
    run_benchmarks(bin_dir="./benchmarks/data", settings=baseline_settings, output_csv=baseline_results)

    # Generate performance profiles
    subprocess.run(["python", "benchmarks/utils/compute_performance_profiles.py", baseline_results, diff_results], check=True)

    # Copy results directory here
    subprocess.run(["cp", "-r", temp_results_dir, "."], check=True)

    # Generate regression report
    subprocess.run(["python", "benchmarks/utils/regression_report.py", baseline_results, diff_results], check=True)
    subprocess.run(["cp", "/tmp/pr_comment.txt", "./results"], check=True)
