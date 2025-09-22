import pandas as pd
import sys
import os

branch_csv = sys.argv[1]  # e.g., /tmp/results/branch_results.csv
main_csv = sys.argv[2]    # e.g., /tmp/results/main_results.csv
pr_number = os.environ.get("PR_NUMBER")
artifact_url = os.environ.get("ARTIFACT_URL")

branch_df = pd.read_csv(branch_csv)
main_df = pd.read_csv(main_csv)

# Ensure consistent sorting/merging by problem name
branch_df = branch_df.set_index("name")
main_df = main_df.set_index("name")

# Count solved problems (exit_code == 1)
branch_solved = (branch_df["exit_code"] == 1).sum()
main_solved = (main_df["exit_code"] == 1).sum()

msg_lines = []
msg_lines.append(f"### 📊 [Download benchmark artifacts]({artifact_url})\n")
msg_lines.append("### Benchmark Summary\n")
msg_lines.append(f"- PR branch solved: **{branch_solved}** problems")
msg_lines.append(f"- Main branch solved: **{main_solved}** problems\n")

# Find problems solved only by one branch
branch_only = set(branch_df[branch_df["exit_code"] == 1].index) - set(main_df[main_df["exit_code"] == 1].index)
main_only = set(main_df[main_df["exit_code"] == 1].index) - set(branch_df[branch_df["exit_code"] == 1].index)

if branch_only or main_only:
    msg_lines.append("#### Differences in solved problems")
    if branch_only:
        msg_lines.append(f"- PR branch solved additional problems: {', '.join(sorted(branch_only))}")
    if main_only:
        msg_lines.append(f"- Main branch solved additional problems: {', '.join(sorted(main_only))}")
    msg_lines.append("")

# Compare iterations where both solved the problem
both_solved = (branch_df["exit_code"] == 1) & (main_df["exit_code"] == 1)
iter_diffs = []
for name in branch_df[both_solved].index:
    b_iters = branch_df.loc[name, "iters"]
    m_iters = main_df.loc[name, "iters"]
    if b_iters != m_iters:
        iter_diffs.append((name, b_iters, m_iters, b_iters - m_iters))

if iter_diffs:
    msg_lines.append("#### Iteration count differences (both solved)")
    for name, b, m, diff in iter_diffs:
        msg_lines.append(f"- {name}: branch={b}, main={m}")
    msg_lines.append("")

# Final message
msg = "\n".join(msg_lines)

# Write to file
comment_file = "/tmp/pr_comment.txt"
with open(comment_file, "w") as f:
    f.write(msg)

# Post comment
os.system(f"gh pr comment {pr_number} --body-file {comment_file}")
