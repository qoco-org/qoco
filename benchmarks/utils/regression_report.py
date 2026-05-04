import pandas as pd
import sys
import os

baseline_csv = sys.argv[1]
diff_csv = sys.argv[2]
actions_run_url = os.environ.get("ACTIONS_RUN_URL")

baseline_df = pd.read_csv(baseline_csv)
diff_df = pd.read_csv(diff_csv)

required_columns = {"name", "exit_code", "iters", "ir_iters"}
for csv_path, df in ((baseline_csv, baseline_df), (diff_csv, diff_df)):
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing required columns: {', '.join(missing_columns)}"
        )

# The report job merges per-dataset artifacts and adds this column. The matrix
# benchmark job also invokes this script on a single raw CSV pair before upload.
for df in (baseline_df, diff_df):
    if "dataset" not in df.columns:
        df["dataset"] = "all"

msg_lines = []
msg_lines.append(f"### [Download Benchmark Artifacts]({actions_run_url})\n")
msg_lines.append("### Benchmark Summary\n")

# Per-dataset solved counts, total iters, total IR iters
msg_lines.append("#### Problems Solved\n")
msg_lines.append("| Dataset | Main Solved | Diff Solved | Main Iters | Diff Iters | Main IR Iters | Diff IR Iters |")
msg_lines.append("|---------|-------------|-------------|------------|------------|---------------|---------------|")
datasets = sorted(baseline_df["dataset"].unique())
for ds in datasets:
    b = baseline_df[baseline_df["dataset"] == ds]
    d = diff_df[diff_df["dataset"] == ds]
    b_solved = (b["exit_code"] == 1).sum()
    d_solved = (d["exit_code"] == 1).sum()
    both = set(b.loc[b["exit_code"] == 1, "name"]) & set(d.loc[d["exit_code"] == 1, "name"])
    b_iters = b.loc[b["name"].isin(both), "iters"].sum()
    d_iters = d.loc[d["name"].isin(both), "iters"].sum()
    b_ir_iters = b.loc[b["name"].isin(both), "ir_iters"].sum()
    d_ir_iters = d.loc[d["name"].isin(both), "ir_iters"].sum()

    bs = f"🟢 {b_solved}" if b_solved > d_solved else str(b_solved)
    ds_ = f"🟢 {d_solved}" if d_solved > b_solved else str(d_solved)
    bi = f"🟢 {b_iters}" if b_iters < d_iters else str(b_iters)
    di = f"🟢 {d_iters}" if d_iters < b_iters else str(d_iters)
    bir = f"🟢 {b_ir_iters}" if b_ir_iters < d_ir_iters else str(b_ir_iters)
    dir_ = f"🟢 {d_ir_iters}" if d_ir_iters < b_ir_iters else str(d_ir_iters)

    msg_lines.append(f"| {ds} | {bs} / {len(b)} | {ds_} / {len(d)} | {bi} | {di} | {bir} | {dir_} |")
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
        msg_lines.append(f"- Main additionally solved: {', '.join(sorted(baseline_only))}")
    msg_lines.append("")

# Per-problem iteration differences where both solved
both_solved = (baseline_df["exit_code"] == 1) & (diff_df["exit_code"] == 1)

iter_regressions = []
iter_improvements = []
ir_iter_regressions = []
ir_iter_improvements = []

for name in baseline_df[both_solved].index:
    b_iters = baseline_df.loc[name, "iters"]
    d_iters = diff_df.loc[name, "iters"]
    if d_iters > b_iters:
        iter_regressions.append((name, d_iters, b_iters))
    elif d_iters < b_iters:
        iter_improvements.append((name, d_iters, b_iters))

    b_ir = baseline_df.loc[name, "ir_iters"]
    d_ir = diff_df.loc[name, "ir_iters"]
    if d_ir > b_ir:
        ir_iter_regressions.append((name, d_ir, b_ir))
    elif d_ir < b_ir:
        ir_iter_improvements.append((name, d_ir, b_ir))

if iter_regressions:
    msg_lines.append("#### Iteration Regressions (diff took more iterations)")
    for name, d, b in iter_regressions:
        msg_lines.append(f"- {name}: diff={d}, main={b}")
    msg_lines.append("")

if iter_improvements:
    msg_lines.append("#### Iteration Improvements (diff took fewer iterations)")
    for name, d, b in iter_improvements:
        msg_lines.append(f"- {name}: diff={d}, main={b}")
    msg_lines.append("")

if ir_iter_regressions:
    msg_lines.append("#### IR Iteration Regressions (diff used more IR iterations)")
    for name, d, b in ir_iter_regressions:
        msg_lines.append(f"- {name}: diff={d}, main={b}")
    msg_lines.append("")

if ir_iter_improvements:
    msg_lines.append("#### IR Iteration Improvements (diff used fewer IR iterations)")
    for name, d, b in ir_iter_improvements:
        msg_lines.append(f"- {name}: diff={d}, main={b}")
    msg_lines.append("")

with open("/tmp/regression-report.txt", "w") as f:
    f.write("\n".join(msg_lines))
