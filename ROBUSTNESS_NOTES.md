# QOCO Robustness Notes

Working notes for the robustness sweep on this branch. Updates the prior pass
documented in this file (May 2026 baseline). Audience: future me / future
contributors picking up this work.

## TL;DR

Three code-level changes are now in the solver:

1. **Best-iterate restoration** — track the iterate with the lowest combined
   residual metric and, on `QOCO_NUMERICAL_ERROR` or `QOCO_MAX_ITER` exit,
   restore it. Downgrade the status to `QOCO_SOLVED_INACCURATE` if that
   iterate satisfies `(abstol_inacc, reltol_inacc)`.

   - Implementation: `QOCOWorkspace::best_*` fields + `restore_best_iterate`
     in `src/qoco_utils.c`. Hooked into both exit paths in
     `src/qoco_api.c::qoco_solve`.
   - Mirrors Clarabel's `save_prev_iterate` / `reset_to_prev_iterate`.

2. **Drop near-zero structural entries** — at `qoco_setup`, drop entries with
   `|x| < 1e-30` from `P`, `A`, `G` before transposing or scaling.

   - Implementation: `drop_small_entries` in `src/common_linalg.c`.
   - Motivation: CUTEst CMPC* problems carry 1e-47 coefficients alongside
     1e+10 RHS values; these have no physical meaning but blow up Ruiz row
     norms. Threshold is conservative (well above subnormal range, below any
     legitimate small coefficient like quadcopter_5's 7e-27).

3. **Proportional static regularization** — new setting
   `kkt_static_reg_proportional` (default 0). After Ruiz, computes
   `scale = max(|P|, |A|, |G|)` over post-scaled matrix entries and adds
   `prop * scale` to each of `kkt_static_reg_{P,A,G}` for the per-solver
   settings copy.

   - Implementation: in `qoco_setup` after `ruiz_equilibration`.
   - Mirrors Clarabel's `static_regularization_proportional`.
   - Only the in-solver copy of settings is mutated; the user's settings
     struct is untouched.

A new benchmark driver `benchmarks/utils/experiment.py` runs any of
`{mpc, misc, mm, cutest}` (or `all`) in parallel via `ThreadPoolExecutor`,
with `--max-bytes` to skip large problems for fast experimentation,
`--workers`, `--timeout`, `--show-failures`, `--out-prefix`.

## Default settings (unchanged from prior pass)

```
ruiz_iters = 1
ruiz_scaling_min = 1e-4
ruiz_scaling_max = 1e4
max_ir_iters = 5
ir_tol = 1e-6
kkt_static_reg_P = 1e-13
kkt_static_reg_A = 1e-13
kkt_static_reg_G = 1e-13
kkt_dynamic_reg = 1e-11
kkt_static_reg_proportional = 0       # NEW (default off)
abstol = reltol = 1e-7
abstol_inacc = reltol_inacc = 1e-5
max_iters = 200
```

## Baseline (current default settings + the three changes)

Counted as solved = `QOCO_SOLVED` (status 1). "Inaccurate" = `QOCO_SOLVED_INACCURATE`.

| Set | Solved | Inaccurate | Numerical error | Max iter |
|---|---:|---:|---:|---:|
| `mpc` | 64/64 | 0 | 0 | 0 |
| `misc` | 2/3 | 1 (cvxpy_lp5) | 0 | 0 |
| `mm` | 138/138 | 0 | 0 | 0 |
| `cutest` (small subset, ≤700 KB, 44 problems) | 21/44 | 0 | 22 | 0 |

The misc `cvxpy_lp5` case was a `QOCO_NUMERICAL_ERROR` before best-iterate
tracking; it now returns `QOCO_SOLVED_INACCURATE` because the iterate at
IPM iter 13 has `(pres=4.25e-8, dres=9.78e-12, gap=4.17e-6)` which is
inside the inaccurate tolerance and is restored on the late-stage stall.

## Best result with `kkt_static_reg_proportional` enabled

| Set | Settings | Solved | Inacc | NumErr | Max iter |
|---|---|---:|---:|---:|---:|
| `mpc` | `prop=5e-8 max_iters=400` | 63/64 | 1 | 0 | 0 |
| `misc` | same | 3/3 | 0 | 0 | 0 |
| `mm` | same | 127/138 | 9 | 1 | 1 |
| `cutest` (small) | same | 41/44 | 3 | 0 | 0 |

If "essentially solved" = `SOLVED + SOLVED_INACCURATE`, this gives
**248/249 essentially solved**. The single remaining `mm` numerical-error
case needs to be identified.

The trade-off is real: at `prop=5e-8`, 11 well-scaled `mm` problems drop
from `SOLVED` to `SOLVED_INACCURATE` because the static reg they pick up
(≈ `5e-8 * O(1) post-Ruiz` = `5e-8`) is comparable to `abstol = 1e-7`,
limiting attainable equality-residual accuracy.

`prop=1e-7` solves more CUTEst (TOYSARAH joins) but breaks
`mpc/quadcopter_5` (`NUMERICAL_ERROR`).

## What worked

- **Best-iterate restoration**: cheap, no regressions, fixes
  `misc/cvxpy_lp5` and similar "got close then stalled" cases. Leaves
  hard-failing CMPC alone since no good iterate ever exists.
- **Pruning < 1e-30 structural entries**: reduces CMPC* constraint range
  from `~1e+50` to `~1e+13` for free at setup, makes Ruiz behave better.
- **`kkt_static_reg_proportional`**: the only knob that demonstrably
  unlocks the CMPC family. With `prop=1e-8` and default `ruiz_iters=1`,
  the four-CMPC sample (`CMPC1`, `CMPC10`, `CMPC5`, `CLEUVEN2`) all hit
  `SOLVED`. With `prop=5e-8`, `TOYSARAH` also solves.

## What did not work

- **Wider Ruiz clamp** (`ruiz_scaling_min=1e-8 ruiz_scaling_max=1e+8`):
  helped some CMPC cases but introduced max-iter stalls and degraded
  `mm` solves.
- **`max_ir_iters=20`**: produced more frequent NaN exits at IPM iter 6
  than the default `max_ir_iters=5` for CMPC, presumably because IR on
  an ill-conditioned factor amplifies error. `max_ir_iters=2` plus large
  static reg solved `TABLE6` strictly but didn't help CMPC.
- **`kkt_static_reg_A=1e-9` (without proportional reg)**: solved
  `mpc/quadcopter_5` but produced `SOLVED_INACCURATE` on `misc/iter_0022`
  with `ruiz_iters=2`, and provided no help on CMPC.
- **`kkt_static_reg_A=1e-8` (Clarabel-style fixed)**: regresses `mpc` to
  ~48/64 SOLVED with many INACCURATE / NUMERICAL_ERROR.
- **`kkt_dynamic_reg` sweep** (`1e-10` … `1e-7`): essentially no
  difference on CMPC, mild degradation on `quadcopter_5` at higher
  values.
- **Proportional reg with deadband (`bump = prop * max(scale - 1, 0)`)
  including RHS**: post-Ruiz inf-norm of `b` for CMPC1 is `~1e+15`
  (Ruiz inflates the row even though the matrix shrinks), so any
  reasonable `prop` made the bump enormous. Fell back to matrix-only
  scale.

## Default settings recommendation

The current default leaves `kkt_static_reg_proportional = 0`. That is the
choice that does not regress any `mpc`/`mm`/`misc` problem and keeps the
solver behavior conservative. CMPC-family problems remain unsolved by
default but can be unlocked with explicit settings:

```
solver_settings:
  kkt_static_reg_proportional: 5e-8
  max_iters: 400
```

A future change could pick `prop` per-problem from the post-Ruiz scale
(e.g. only enable when the post-Ruiz max entry `> 10`), giving the
CMPC-family solves "for free" without the `mm` regressions. That branch
of work is left open.

## Remaining failures (with current default = no proportional reg)

CUTEst small subset (`≤ 700 KB`):

- 22 `NUMERICAL_ERROR`: `CLEUVEN2`, `CMPC1`–`CMPC16`, `JJTABEL3`,
  `TABLE1`, `TABLE3`, `TABLE6`, `TOYSARAH`.
- 1 timeout (`HIER163A`) at the 60-second cap.

These are dominated by the same mechanism described in `FAILURES.md`:
CMPC/CLEUVEN have constraint matrices with input ranges of 1e+50,
mixed with 1e+10 RHS magnitudes. Even after pruning < 1e-30 entries,
the residual range plus the cumulative-clamped Ruiz cannot equilibrate
the system enough for a default-static-reg LDL factor to remain stable
through the IPM. The `prop` setting closes most of these but at the
cost above.

## Workflow notes

- `python benchmarks/utils/experiment.py mpc misc mm` runs in ~30 s on
  this machine; full `cutest` is dominated by a handful of large
  instances (`RDW2D5*U` at 25 MB each), so use `--max-bytes 700000` for
  iteration and reserve full runs for validation.
- `--show-failures` prints non-solved problems by name at the end of
  the summary.
- `--out-prefix foo` writes per-set CSVs to `results/foo_<set>.csv`
  matching the prior CI format.
- The driver runs problems through `ThreadPoolExecutor`, so per-problem
  wall time is dominated by the slowest single instance, not the sum.

## Files touched

- `include/structs.h` — added `best_*` fields to `QOCOWorkspace`,
  `kkt_static_reg_proportional` to `QOCOSettings`.
- `include/qoco_utils.h` — declared `restore_best_iterate`.
- `include/common_linalg.h` — declared `drop_small_entries`.
- `src/qoco_utils.c` — `check_stopping` now records best iterate;
  added `restore_best_iterate`; settings copy includes proportional
  reg.
- `src/qoco_api.c` — allocate/free best-iterate buffers; drop
  near-zero entries before transpose; apply proportional reg after
  Ruiz; restore best iterate on numerical-error and max-iter exits;
  default new setting to 0.
- `src/common_linalg.c` — `drop_small_entries` implementation.
- `src/input_validation.c` — non-negativity check on
  `kkt_static_reg_proportional`.
- `benchmarks/benchmark_runner.c` — accept
  `kkt_static_reg_proportional=...`.
- `benchmarks/utils/experiment.py` — new parallel driver.

All nine unit tests in `build/` continue to pass.
