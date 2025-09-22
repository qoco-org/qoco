import os
import argparse
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings

def compute_absolute_profile(csv_files):
    """
    Compute performance profile from a list of solver CSV files.

    Parameters
    ----------
    csv_files : list[str]
        List of CSV file paths. Each corresponds to one solver.
    """

    t = {}
    status = {}

    all_runtimes = []

    # First pass: collect raw runtimes (without assigning tmax)
    raw_runtimes = []
    data_frames = {}

    for path in csv_files:
        solver = os.path.splitext(os.path.basename(path))[0]  # infer solver name
        df = pd.read_csv(path)

        run_time = df["setup_time"].values + df["solve_time"].values
        data_frames[solver] = (df, run_time)
        raw_runtimes.extend(run_time)

    # Automatically set tmax
    observed_max = max(raw_runtimes)
    tmax = 5 * observed_max
    print(f"Automatically setting tmax = {tmax:.4f}")

    # Second pass: assign runtimes with failures set to tmax
    for solver, (df, run_time) in data_frames.items():
        exit_code = df["exit_code"].values
        for idx in range(len(run_time)):
            if exit_code[idx] != 1:  # non-success
                run_time[idx] = tmax

        t[solver] = run_time
        status[solver] = exit_code
        all_runtimes.extend(run_time)

    # determine xrange automatically
    min_time = min(all_runtimes)
    xmin = np.log10(min_time) - 0.2
    xmax = np.log10(tmax) - 0.2

    # compute curve for all solvers
    n_tau = 1000
    tau_vec = np.logspace(xmin, xmax, n_tau)
    rho = {"tau": tau_vec}

    n_prob = len(next(iter(t.values())))  # number of problems
    for solver in t.keys():
        rho[solver] = np.zeros(n_tau)
        for tau_idx in range(n_tau):
            rho[solver][tau_idx] = np.sum(t[solver] <= tau_vec[tau_idx]) / n_prob

    # Store final dataframe in same directory as first CSV
    out_dir = os.path.dirname(csv_files[0]) if csv_files else "."
    df_performance_profiles = pd.DataFrame(rho)
    performance_profiles_file = os.path.join(out_dir, "absolute_profile.csv")
    df_performance_profiles.to_csv(performance_profiles_file, index=False)

    print(f"Performance profile saved to {performance_profiles_file}")
    return df_performance_profiles

def compute_relative_profile(csv_files):
    """
    Compute relative performance profile from a list of solver CSV files.

    Parameters
    ----------
    csv_files : list[str]
        List of CSV file paths. Each corresponds to one solver.
    """

    t = {}
    status = {}

    all_runtimes = []

    # First pass: load runtimes (without assigning tmax yet)
    raw_runtimes = []
    data_frames = {}

    for path in csv_files:
        solver = os.path.splitext(os.path.basename(path))[0]  # infer solver name
        df = pd.read_csv(path)

        run_time = df["setup_time"].values + df["solve_time"].values
        data_frames[solver] = (df, run_time)
        raw_runtimes.extend(run_time)

    # Automatically set tmax
    observed_max = max(raw_runtimes)
    tmax = 5 * observed_max
    print(f"Automatically setting tmax = {tmax:.4f}")

    # Second pass: assign runtimes with failures set to tmax
    for solver, (df, run_time) in data_frames.items():
        exit_code = df["exit_code"].values
        for idx in range(len(run_time)):
            if exit_code[idx] != 1:  # non-success
                run_time[idx] = tmax

        t[solver] = run_time
        status[solver] = exit_code
        all_runtimes.extend(run_time)

    solvers = list(t.keys())
    n_prob = len(next(iter(t.values())))

    # Dictionary of relative times
    r = {s: np.zeros(n_prob) for s in solvers}

    # Compute relative times
    all_relative_times = []
    for p in range(n_prob):
        min_time = np.min([t[s][p] for s in solvers])
        for s in solvers:
            r[s][p] = t[s][p] / min_time
            all_relative_times.append(r[s][p])

    # determine xrange automatically
    xmin = 0
    xmax = np.log10(tmax) - 0.2

    # compute curve for all solvers
    n_tau = 1000
    tau_vec = np.logspace(xmin, xmax, n_tau)
    rho = {"tau": tau_vec}

    for s in solvers:
        rho[s] = np.zeros(n_tau)
        for tau_idx in range(n_tau):
            rho[s][tau_idx] = np.sum(r[s] <= tau_vec[tau_idx]) / n_prob

    # Store final dataframe in same directory as first CSV
    out_dir = os.path.dirname(csv_files[0]) if csv_files else "."
    df_performance_profiles = pd.DataFrame(rho)
    performance_profiles_file = os.path.join(out_dir, "relative_profile.csv")
    df_performance_profiles.to_csv(performance_profiles_file, index=False)

    print(f"Relative performance profile saved to {performance_profiles_file}")
    return df_performance_profiles

def plot_profiles(abs_csv, rel_csv, out_file="profiles.png"):
    """
    Plot absolute and relative performance profiles side by side.

    Parameters
    ----------
    abs_csv : str
        Path to absolute_profile.csv
    rel_csv : str
        Path to relative_profile.csv
    out_file : str
        Path to save output figure
    """
    df_abs = pd.read_csv(abs_csv)
    df_rel = pd.read_csv(rel_csv)

    # solvers are all columns except "tau"
    solvers_abs = [c for c in df_abs.columns if c != "tau"]
    solvers_rel = [c for c in df_rel.columns if c != "tau"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Absolute profile
    palette = sns.color_palette("Set2", len(solvers_abs))
    for i, s in enumerate(solvers_abs):
        axes[0].plot(df_abs["tau"], df_abs[s], label=s, color=palette[i])
    axes[0].set_xscale("log")
    axes[0].set_ylim([0, 1.05])
    axes[0].set_xlabel("Runtime (s)")
    axes[0].set_ylabel("Proportion of problems solved")
    axes[0].set_title("Absolute Performance Profile")
    axes[0].legend()

    # Relative profile
    palette = sns.color_palette("Set2", len(solvers_rel))
    for i, s in enumerate(solvers_rel):
        axes[1].plot(df_rel["tau"], df_rel[s], label=s, color=palette[i])
    axes[1].set_xscale("log")
    axes[1].set_ylim([0, 1.05])
    axes[1].set_xlabel("Performance ratio")
    axes[1].set_title("Relative Performance Profile")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Plot saved to {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Compute absolute performance profiles from solver CSVs.")
    parser.add_argument("csvs", nargs="+", help="Paths to solver CSV files")
    args = parser.parse_args()

    valid_csvs = []
    for path in args.csvs:
        if os.path.exists(path):
            valid_csvs.append(path)
        else:
            warnings.warn(f"CSV not found: {path} — skipping.")

    if not valid_csvs:
        warnings.warn("No valid CSVs found. Exiting.")
        return

    compute_absolute_profile(valid_csvs)
    compute_relative_profile(valid_csvs)
    out_dir = os.path.dirname(valid_csvs[0]) if valid_csvs else "."
    abs_csv = os.path.join(out_dir, "absolute_profile.csv")
    rel_csv = os.path.join(out_dir, "relative_profile.csv")
    plot_profiles(abs_csv, rel_csv, out_file=os.path.join(out_dir, "performance_profiles.png"))

if __name__ == "__main__":
    main()