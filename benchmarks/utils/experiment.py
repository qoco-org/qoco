"""Parallel benchmark experiment runner.

Used by ROBUSTNESS_NOTES.md experiments. Lets us:

* Run any subset of {misc, mpc, mm, cutest} in parallel.
* Skip large problems for fast experimentation (--max-bytes).
* Pass solver settings as key=value strings.
* Print a per-set status summary.

Usage:
    python benchmarks/utils/experiment.py mpc misc                       # quick sets
    python benchmarks/utils/experiment.py cutest --max-bytes 1000000     # small only
    python benchmarks/utils/experiment.py all --settings 'kkt_static_reg_A=1e-8'
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "benchmarks" / "data"
RUNNER_DEFAULT = str(ROOT / "build" / "benchmark_runner")
RESULTS_DIR = ROOT / "results"

# Status code -> name (matches include/enums.h).
STATUS_NAMES = {
    -2: "TIMEOUT",
    -1: "RUN_ERROR",
    0: "UNSOLVED",
    1: "SOLVED",
    2: "SOLVED_INACCURATE",
    3: "NUMERICAL_ERROR",
    4: "MAX_ITER",
}

ALL_SETS = ["mpc", "misc", "mm", "cutest"]


def run_one(bin_path, runner, settings_args, timeout):
    cmd = [runner, str(bin_path)] + list(settings_args)
    name = bin_path.stem
    try:
        out = subprocess.check_output(cmd, text=True, timeout=timeout)
        parts = out.strip().split()
        # benchmark_runner prints: filename exit iters ir_iters setup solve
        return {
            "name": name,
            "exit": int(parts[1]),
            "iters": int(parts[2]),
            "ir_iters": int(parts[3]),
            "setup": float(parts[4]),
            "solve": float(parts[5]),
        }
    except subprocess.TimeoutExpired:
        return {"name": name, "exit": -2, "iters": -1, "ir_iters": -1, "setup": 0.0, "solve": 0.0}
    except subprocess.CalledProcessError:
        return {"name": name, "exit": -1, "iters": -1, "ir_iters": -1, "setup": 0.0, "solve": 0.0}


def collect_problems(set_name, max_bytes=None):
    sub = DATA_DIR / set_name
    files = sorted(sub.rglob("*.bin"))
    if max_bytes is not None:
        files = [f for f in files if f.stat().st_size <= max_bytes]
    return files


def run_set(set_name, runner, settings_args, max_bytes=None,
            workers=None, timeout=None, verbose=True):
    files = collect_problems(set_name, max_bytes=max_bytes)
    if not files:
        return [], 0.0
    started = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, f, runner, settings_args, timeout): f for f in files}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            if verbose:
                f = futures[fut]
                tag = STATUS_NAMES.get(res["exit"], f"?{res['exit']}")
                print(f"  [{set_name}] {f.stem:<24} {tag:<20} iters={res['iters']:<3} "
                      f"ir={res['ir_iters']:<3} t={res['solve']:.3f}s", flush=True)
    results.sort(key=lambda r: r["name"])
    return results, time.time() - started


def summarize(results):
    counts = {}
    for r in results:
        counts[r["exit"]] = counts.get(r["exit"], 0) + 1
    total = len(results)
    return total, counts


def fmt_summary(total, counts):
    parts = []
    for code in sorted(counts.keys()):
        name = STATUS_NAMES.get(code, f"code_{code}")
        parts.append(f"{name}={counts[code]}")
    solved = counts.get(1, 0)
    return f"{solved}/{total} solved | " + " ".join(parts)


def write_csv(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("name,exit_code,iters,ir_iters,setup_time,solve_time\n")
        for r in results:
            f.write(f"{r['name']},{r['exit']},{r['iters']},{r['ir_iters']},"
                    f"{r['setup']:.6f},{r['solve']:.6f}\n")


def list_failures(results):
    return [r for r in results if r["exit"] != 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sets", nargs="*", default=ALL_SETS,
                        help="space-separated set names; 'all' expands to mpc misc mm cutest")
    parser.add_argument("--runner", default=RUNNER_DEFAULT,
                        help="path to benchmark_runner binary")
    parser.add_argument("--settings", default="",
                        help="space-separated key=value solver settings to forward")
    parser.add_argument("--max-bytes", type=int, default=None,
                        help="skip .bin files larger than this size (bytes)")
    parser.add_argument("--workers", type=int, default=None,
                        help="ThreadPool workers (default: os.cpu_count())")
    parser.add_argument("--timeout", type=float, default=None,
                        help="per-problem timeout in seconds")
    parser.add_argument("--out-prefix", default=None,
                        help="if set, write CSV results to results/<prefix>_<set>.csv")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-problem progress; only print summary")
    parser.add_argument("--show-failures", action="store_true",
                        help="list non-solved problems at the end")
    args = parser.parse_args()

    sets = []
    for s in args.sets:
        if s == "all":
            sets.extend(ALL_SETS)
        else:
            sets.append(s)
    sets = [s for s in dict.fromkeys(sets) if s in ALL_SETS]
    if not sets:
        print("no valid sets selected", file=sys.stderr)
        sys.exit(2)

    settings_args = args.settings.split() if args.settings else []
    runner = os.path.abspath(args.runner)
    if not os.access(runner, os.X_OK):
        print(f"runner not executable: {runner}", file=sys.stderr)
        sys.exit(2)

    grand_total = 0
    grand_counts = {}
    grand_failures = []
    for s in sets:
        print(f"\n=== {s} ===")
        results, elapsed = run_set(s, runner, settings_args,
                                   max_bytes=args.max_bytes,
                                   workers=args.workers,
                                   timeout=args.timeout,
                                   verbose=not args.quiet)
        total, counts = summarize(results)
        print(f"  {s}: {fmt_summary(total, counts)}  [wall={elapsed:.1f}s]")
        grand_total += total
        for k, v in counts.items():
            grand_counts[k] = grand_counts.get(k, 0) + v
        if args.show_failures:
            for r in list_failures(results):
                print(f"    FAIL [{s}] {r['name']:<24} {STATUS_NAMES.get(r['exit'])}")
            grand_failures.extend((s, r) for r in list_failures(results))
        if args.out_prefix:
            write_csv(results, RESULTS_DIR / f"{args.out_prefix}_{s}.csv")

    print("\n=== overall ===")
    print(f"  {fmt_summary(grand_total, grand_counts)}")


if __name__ == "__main__":
    main()
