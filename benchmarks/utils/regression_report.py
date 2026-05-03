import pandas as pd
import sys
import os

baseline_csv = sys.argv[1]
diff_csv = sys.argv[2]
actions_run_url = os.environ.get("ACTIONS_RUN_URL")

baseline_df = pd.read_csv(baseline_csv)
diff_df = pd.read_csv(diff_csv)

msg_lines = []
msg_lines.append(f"### [Download Benchmark Artifacts]({actions_run_url})\n")
msg_lines.append("### Benchmark Summary\n")

# Per-dataset solved counts
msg_lines.append("#### Problems Solved\n")
msg_lines.append("| Dataset | Baseline | Diff |")
msg_lines.append("|---------|----------|------|")
datasets = sorted(baseline_df["dataset"].unique())
for ds in datasets:
    b = baseline_df[baseline_df["dataset"] == ds]
    d = diff_df[diff_df["dataset"] == ds]
    b_solved = (b["exit_code"] == 1).sum()
    d_solved = (d["exit_code"] == 1).sum()
    msg_lines.append(f"| {ds} | {b_solved} / {len(b)} | {d_solved} / {len(d)} |")
msg_lines.append("")

baseline_df = baseline_df.set_index("name")
diff_df = diff_df.set_index("name")

# Find problems solved only by one branch
diff_only = set(diff_df[diff_df["exit_code"] == 1].index) - set(baseline_df[baseline_df["exit_code"] == 1].index)
baseline_only = set(baseline_df[baseline_df["exit_code"] == 1].index) - set(diff_df[diff_df["exit_code"] == 1].index)

if diff_only or baseline_only:
    msg_lines.append("#### Differences in Solved Problems")
    if diff_only:
        msg_lines.append(f"- Diff branch additionally solved: {', '.join(sorted(diff_only))}")
    if baseline_only:
        msg_lines.append(f"- Baseline additionally solved: {', '.join(sorted(baseline_only))}")
    msg_lines.append("")

# Iteration regressions/improvements where both solved
both_solved = (baseline_df["exit_code"] == 1) & (diff_df["exit_code"] == 1)

iter_regressions = []
iter_improvements = []
for name in baseline_df[both_solved].index:
    b_iters = baseline_df.loc[name, "iters"]
    d_iters = diff_df.loc[name, "iters"]
    if d_iters > b_iters:
        iter_regressions.append((name, d_iters, b_iters))
    elif d_iters < b_iters:
        iter_improvements.append((name, d_iters, b_iters))

if iter_regressions:
    msg_lines.append("#### Iteration Regressions (diff took more iterations)")
    for name, d, b in iter_regressions:
        msg_lines.append(f"- {name}: diff={d}, baseline={b}")
    msg_lines.append("")

if iter_improvements:
    msg_lines.append("#### Iteration Improvements (diff took fewer iterations)")
    for name, d, b in iter_improvements:
        msg_lines.append(f"- {name}: diff={d}, baseline={b}")
    msg_lines.append("")

with open("/tmp/regression-report.txt", "w") as f:
    f.write("\n".join(msg_lines))
