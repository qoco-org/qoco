import pandas as pd
import sys
import os

# ==== CONFIGURATION ====
RUNTIME_THRESHOLD = 0.05  # 5% relative difference threshold
# =======================

baseline_csv = sys.argv[1]  # e.g., /tmp/results/main_results.csv
diff_csv = sys.argv[2]      # e.g., /tmp/results/branch_results.csv
pr_number = os.environ.get("PR_NUMBER")
artifact_url = os.environ.get("ARTIFACT_URL")

baseline_df = pd.read_csv(baseline_csv).set_index("name")
diff_df = pd.read_csv(diff_csv).set_index("name")

# Count solved problems (exit_code == 1)
baseline_solved = (baseline_df["exit_code"] == 1).sum()
diff_solved = (diff_df["exit_code"] == 1).sum()

msg_lines = []
msg_lines.append(f"### [Download benchmark artifacts]({artifact_url})\n")
msg_lines.append("### Benchmark Summary\n")
msg_lines.append(f"- Baseline solved: **{baseline_solved}** problems")
msg_lines.append(f"- Diff branch solved: **{diff_solved}** problems\n")

# Find problems solved only by one branch
diff_only = set(diff_df[diff_df["exit_code"] == 1].index) - set(baseline_df[baseline_df["exit_code"] == 1].index)
baseline_only = set(baseline_df[baseline_df["exit_code"] == 1].index) - set(diff_df[diff_df["exit_code"] == 1].index)

if diff_only or baseline_only:
    msg_lines.append("#### Differences in solved problems")
    if diff_only:
        msg_lines.append(f"- Diff branch solved additional problems: {', '.join(sorted(diff_only))}")
    if baseline_only:
        msg_lines.append(f"- Baseline solved additional problems: {', '.join(sorted(baseline_only))}")
    msg_lines.append("")

# Compare iterations where both solved
both_solved = (baseline_df["exit_code"] == 1) & (diff_df["exit_code"] == 1)

iter_regressions = []
iter_improvements = []
for name in baseline_df[both_solved].index:
    b_iters = baseline_df.loc[name, "iters"]
    d_iters = diff_df.loc[name, "iters"]
    if d_iters > b_iters:
        iter_regressions.append((name, d_iters, b_iters, d_iters - b_iters))
    elif d_iters < b_iters:
        iter_improvements.append((name, d_iters, b_iters, d_iters - b_iters))

if iter_regressions:
    msg_lines.append("#### Iteration regressions (diff took more iterations)")
    for name, d, b, diff in iter_regressions:
        msg_lines.append(f"- {name}: diff={d}, baseline={b}, Δ={diff:+}")
    msg_lines.append("")

if iter_improvements:
    msg_lines.append("#### Iteration improvements (diff took fewer iterations)")
    for name, d, b, diff in iter_improvements:
        msg_lines.append(f"- {name}: diff={d}, baseline={b}, Δ={diff:+}")
    msg_lines.append("")

# Runtime differences (parameterized threshold)
runtime_regressions = []
runtime_improvements = []
for name in baseline_df[both_solved].index:
    b_time = baseline_df.loc[name, "solve_time"]
    d_time = diff_df.loc[name, "solve_time"]

    if b_time > 0:
        rel_diff = (d_time - b_time) / b_time
        if rel_diff > RUNTIME_THRESHOLD:
            runtime_regressions.append((name, d_time, b_time, rel_diff))
        elif rel_diff < -RUNTIME_THRESHOLD:
            runtime_improvements.append((name, d_time, b_time, rel_diff))

if runtime_regressions:
    msg_lines.append(f"#### Runtime regressions (> {RUNTIME_THRESHOLD*100:.1f}%)")
    for name, d, b, rel in runtime_regressions:
        pct = rel * 100
        msg_lines.append(f"- {name}: diff={d:.4f}s, baseline={b:.4f}s, Δ={pct:+.1f}%")
    msg_lines.append("")

if runtime_improvements:
    msg_lines.append(f"#### Runtime improvements (> {RUNTIME_THRESHOLD*100:.1f}%)")
    for name, d, b, rel in runtime_improvements:
        pct = rel * 100
        msg_lines.append(f"- {name}: diff={d:.4f}s, baseline={b:.4f}s, Δ={pct:+.1f}%")
    msg_lines.append("")

# Final message
msg = "\n".join(msg_lines)

# Write to file
comment_file = "/tmp/pr_comment.txt"
with open(comment_file, "w") as f:
    f.write(msg)

# Post comment
os.system(f"gh pr comment {pr_number} --body-file {comment_file}")
