# Remaining Benchmark Failures

This note summarizes the remaining CUTEst failures after testing the following
fallback policy:

1. `ruiz_iters=0 kkt_static_reg_A=1e-8`
2. if failed and data range >= `1e10`, retry `ruiz_iters=1 kkt_static_reg_A=1e-8`
3. if failed and data range >= `1e10`, retry `ruiz_iters=1 kkt_static_reg_A=1e-13`

The policy solved all `misc`, `mm`, and `mpc` problems in the local test run,
but left 12 CUTEst problems unsolved.

## Summary

| Problem | Best observed status | Main issue |
|---|---:|---|
| `CLEUVEN2` | `QOCO_NUMERICAL_ERROR` | Pathological matrix/RHS scaling; structural coefficients near `1e-47` mixed with `1e11` RHS values. |
| `CMPC1` | `QOCO_NUMERICAL_ERROR` | Same CMPC/CLEUVEN scaling pathology; Ruiz with extra iterations reaches NaNs. |
| `CMPC10` | `QOCO_NUMERICAL_ERROR` / `QOCO_MAX_ITER` | Same CMPC scaling pathology; `kkt_static_reg_A=1e-8` lowers residuals but stalls at max iterations. |
| `CMPC12` | `QOCO_NUMERICAL_ERROR` | Same CMPC scaling pathology; large residuals remain after all tested settings. |
| `CMPC2` | `QOCO_NUMERICAL_ERROR` | Same CMPC scaling pathology; more Ruiz does not consistently help. |
| `CMPC3` | `QOCO_NUMERICAL_ERROR` / `QOCO_MAX_ITER` | Same CMPC scaling pathology; `kkt_static_reg_A=1e-8` stalls at max iterations. |
| `CMPC4` | `QOCO_NUMERICAL_ERROR` / `QOCO_MAX_ITER` | Same CMPC scaling pathology; `kkt_static_reg_A=1e-8` stalls at max iterations. |
| `CMPC5` | `QOCO_NUMERICAL_ERROR` | Same CMPC scaling pathology; residuals remain far above solved tolerances. |
| `CMPC6` | `QOCO_NUMERICAL_ERROR` | Same CMPC scaling pathology; very large RHS range remains difficult. |
| `CMPC9` | `QOCO_NUMERICAL_ERROR` / `QOCO_MAX_ITER` | Same CMPC scaling pathology; `kkt_static_reg_A=1e-8` stalls at max iterations. |
| `TABLE6` | `QOCO_NUMERICAL_ERROR`; `QOCO_SOLVED_INACCURATE` with `ruiz_iters=2 kkt_static_reg_A=1e-13` | Constraint matrix is well scaled, but RHS/objective magnitudes are badly mismatched; strict solve stalls around gap/primal feasibility. |
| `TOYSARAH` | `QOCO_NUMERICAL_ERROR` | Constraint matrix is well scaled, but objective/RHS scaling causes gap-dominated late-stage failure. |

## Evidence

Status codes here follow `include/enums.h`: `1 = QOCO_SOLVED`,
`2 = QOCO_SOLVED_INACCURATE`, `3 = QOCO_NUMERICAL_ERROR`,
`4 = QOCO_MAX_ITER`.

### CMPC/CLEUVEN family

These failures are dominated by extreme nonzero ranges in the constraint matrix.
The smallest retained constraint coefficient is `1.24795e-47`, while the largest
constraint coefficient is `1.92e3` to `2.64e3`. That creates constraint ranges
around `1e50`. Several RHS vectors also mix very small values with entries near
`5.5e10` or `5.0125e11`.

| Problem | Overall range | Constraint range | RHS range |
|---|---:|---:|---:|
| `CLEUVEN2` | `4.0e58` | `1.5e50` | `5.2e29` |
| `CMPC1` | `4.0e58` | `1.5e50` | `1.1e16` |
| `CMPC10` | `4.4e57` | `2.1e50` | `2.6e19` |
| `CMPC12` | `4.4e57` | `2.1e50` | `2.6e19` |
| `CMPC2` | `4.4e57` | `2.1e50` | `5.4e19` |
| `CMPC3` | `4.4e57` | `2.1e50` | `5.4e19` |
| `CMPC4` | `4.4e57` | `2.1e50` | `5.4e19` |
| `CMPC5` | `4.4e57` | `2.1e50` | `5.4e19` |
| `CMPC6` | `4.4e57` | `2.1e50` | `5.7e28` |
| `CMPC9` | `4.4e57` | `2.1e50` | `2.6e19` |

Representative final residuals:

| Problem/settings | Status | Iters | IR | Pres | Dres | Gap |
|---|---:|---:|---:|---:|---:|---:|
| `CMPC10`, `ruiz=0 eA=1e-8` | 3 | 14 | 91 | `9.33e2` | `3.58e7` | `7.50e18` |
| `CMPC10`, `ruiz=1 eA=1e-8` | 4 | 200 | 854 | `3.17e1` | `9.98e-1` | `6.29e11` |
| `CMPC10`, `ruiz=1 eA=1e-13` | 3 | 12 | 65 | `1.78e5` | `8.99e8` | `5.10e20` |
| `CMPC3`, `ruiz=1 eA=1e-8` | 4 | 200 | 1113 | `2.83e1` | `2.35e0` | `1.07e10` |
| `CMPC1`, `ruiz=2 eA=1e-13` | 3 | 6 | 60 | `NaN` | `NaN` | `NaN` |

Root cause: these are not merely "moderately ill-scaled" problems. They contain
near-zero coefficients that are still represented as structural nonzeros, plus
large RHS entries. One Ruiz pass with cumulative clamp `[1e-4, 1e4]` cannot
equilibrate a `1e50` matrix range. More aggressive scaling can make the KKT
path numerically unstable and produce NaNs. Increasing `kkt_static_reg_A` to
`1e-8` sometimes lowers residuals, but several cases then stall at max
iterations rather than reaching solved tolerances.

Likely fixes for this class:

- Drop/prune structural coefficients below a numerical threshold before setup.
- Add stronger or repeated scaling only after pruning near-zero coefficients.
- Make regularization scale-aware instead of relying on fixed `1e-8`/`1e-13`.
- Add fallback result selection, but do not expect the current fixed-parameter
  settings to solve all of these robustly.

### TABLE6

`TABLE6` does not have a badly scaled constraint matrix:

| Quantity | Range |
|---|---:|
| objective data | `2.0e9` |
| constraints | `1.0` |
| RHS | `2.5e12` |
| overall | `9.8e20` |

Final residuals under tested settings:

| Settings | Status | Iters | IR | Pres | Dres | Gap |
|---|---:|---:|---:|---:|---:|---:|
| `ruiz=0 eA=1e-8` | 3 | 29 | 133 | `3.66e-4` | `6.61e-8` | `1.93e13` |
| `ruiz=1 eA=1e-8` | 3 | 29 | 124 | `3.66e-4` | `9.81e-9` | `5.46e12` |
| `ruiz=1 eA=1e-13` | 3 | 32 | 127 | `5.99e4` | `8.30e-1` | `2.11e11` |
| `ruiz=2 eA=1e-13` | 2 | 51 | 199 | `1.12e6` | `2.74e-2` | `6.03e-3` |

Root cause: this case is dominated by objective/RHS magnitude mismatch, not by
constraint matrix scaling. The solver can drive some residual components down,
but strict original-unit feasibility/duality is not reached before the late
stage becomes numerically fragile. `ruiz_iters=2 kkt_static_reg_A=1e-13`
returns only `QOCO_SOLVED_INACCURATE`, which is not counted as solved in the
benchmark summary.

Likely fixes:

- Add best-iterate restoration so numerical-error exits return the least-bad
  iterate.
- Improve gap/late-stage stopping and regularization for large RHS/objective
  magnitude problems.
- Consider objective/RHS-specific scaling separate from row/column Ruiz.

### TOYSARAH

`TOYSARAH` also has a well-scaled constraint matrix but a large objective/RHS
scale mismatch:

| Quantity | Range |
|---|---:|
| objective data | `2.2e11` |
| constraints | `1.0` |
| RHS | `1.2e7` |
| overall | `1.2e22` |

Final residuals under tested settings:

| Settings | Status | Iters | IR | Pres | Dres | Gap |
|---|---:|---:|---:|---:|---:|---:|
| `ruiz=0 eA=1e-8` | 3 | 21 | 179 | `3.05e-5` | `3.01e-5` | `8.09e15` |
| `ruiz=1 eA=1e-8` | 3 | 20 | 168 | `2.66e-3` | `7.33e-2` | `3.39e16` |
| `ruiz=1 eA=1e-13` | 3 | 21 | 168 | `2.19e-2` | `5.71e-3` | `1.16e16` |
| `ruiz=2 eA=1e-13` | 3 | 22 | 175 | `9.94e-5` | `3.24e-4` | `2.28e15` |

Root cause: the constraints are easy to scale, but the objective/gap remains
the limiting component. Ruiz does not address this enough, and changing
`kkt_static_reg_A` mostly trades primal/dual residuals without reaching strict
solved tolerances.

Likely fixes:

- Improve objective/gap scaling and stopping for large-cost LP/QP instances.
- Add best-iterate tracking to avoid returning a late-stage degraded point.
- Treat this separately from the CMPC/CLEUVEN near-zero-coefficient pathology.

