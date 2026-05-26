# Batch cuDSS Benchmark Findings

This note summarizes the batch GPU backend timing checks performed while adding
cuDSS batch support.

## Instrumentation

The CUDA/cuDSS backend was instrumented with opt-in counters around only the
cuDSS factorization and solve `cudssExecute` calls. The timers synchronize the
device immediately before and after each measured cuDSS call.

Measured phases:

- Serial solver: `CUDSS_PHASE_FACTORIZATION` and `CUDSS_PHASE_SOLVE` calls from
  the normal per-solver cuDSS backend.
- Batch solver: `CUDSS_PHASE_FACTORIZATION` and `CUDSS_PHASE_SOLVE` calls from
  the cuDSS batch matrices created with `cudssMatrixCreateBatchCsr` and
  `cudssMatrixCreateBatchDn`.

Setup and cuDSS analysis are excluded from the cuDSS factor/solve totals.

## Small SOCP Benchmark

Problem: `cvxpy_qoco.socp_0`

- `n = 3`
- `m = 3`
- `p = 2`
- one SOC of size `3`
- 100 variants with small deterministic perturbations to `b` and `h`
- CUDA backend
- `ruiz_iters = 0`
- `max_ir_iters = 0`

Results:

```text
serial setup:        0.008239 s
serial update:       0.001327 s
serial solve:        3.508448 s
serial update+solve: 3.509775 s
serial solved:       100 / 100

serial cuDSS factor:       0.019527 s  (594 calls)
serial cuDSS solve:        0.038367 s  (1088 calls)
serial cuDSS factor+solve: 0.057893 s
```

```text
batch setup:         0.757654 s
batch update:        0.001081 s
batch solve:         3.519075 s
batch update+solve:  3.520156 s
batch solved:        100 / 100

batch cuDSS factor:        0.000474 s  (6 calls)
batch cuDSS solve:         0.000840 s  (11 calls)
batch cuDSS factor+solve:  0.001314 s
```

Speedups:

```text
solve speedup serial/batch:             0.997x
update+solve speedup serial/batch:      0.997x
including setup speedup serial/batch:   0.822x

cuDSS solve speedup serial/batch:       45.677x
cuDSS factor+solve speedup:             44.060x
```

Interpretation: for this tiny SOCP, the batched cuDSS calls are much faster, but
the full solver runtime is dominated by per-item QOCO work outside cuDSS.

## PDG Benchmark

Problem: PDG test problem with 100 variants of the initial condition entries in
`b`.

- `n = 2698`
- `nsoc = 598`
- initial condition offset in `b`: `1794`
- base initial condition: `100 50 50 -9 5 -9`
- relative perturbation: `1e-3`
- batch width: `100`
- CUDA backend

Results:

```text
serial setup:        0.031077 s
serial update:       0.009657 s
serial solve:        9.073872 s
serial update+solve: 9.083528 s
serial solved:       100 / 100

serial cuDSS factor:       0.325587 s  (1245 calls)
serial cuDSS solve:        0.536786 s  (3690 calls)
serial cuDSS factor+solve: 0.862373 s
```

```text
batch setup:         4.563072 s
batch update:        0.010039 s
batch solve:         8.693753 s
batch update+solve:  8.703792 s
batch solved:        100 / 100

batch cuDSS factor:        0.146206 s  (13 calls)
batch cuDSS solve:         0.182242 s  (38 calls)
batch cuDSS factor+solve:  0.328448 s
```

Speedups:

```text
solve speedup serial/batch:             1.0437x
update+solve speedup serial/batch:      1.0436x
including setup speedup serial/batch:   0.6870x

cuDSS solve speedup serial/batch:       2.9455x
cuDSS factor+solve speedup:             2.6256x
```

Interpretation: for PDG, batched cuDSS is clearly faster internally, but the
end-to-end solve only improves by about 4.4% excluding setup. Non-cuDSS QOCO
work still dominates total runtime.

## Why Batch Setup Is Expensive

The current `qoco_batch_setup` implementation allocates and fully initializes
one complete `QOCOSolver` per batch item:

```text
for each item:
  allocate QOCOSolver
  call qoco_setup(...)
```

For the PDG benchmark with 100 items, batch setup does roughly:

```text
100x normal solver setup
+ batch CSR pointer arrays and dense wrappers
+ batch cuDSS analysis
```

Each per-item `qoco_setup` builds/scales problem data, constructs KKT structures,
creates individual cuDSS handles/matrices/data, and runs per-solver cuDSS
analysis. The serial benchmark only sets up one solver and then updates `b` for
each instance, so its setup time is much lower.

## Main Takeaways

- The cuDSS batch API is working and reducing total cuDSS factor/solve time.
- End-to-end speedup is currently limited by QOCO-side per-item work around the
  batched linear solves.
- Batch setup is expensive because the batch API currently owns many fully
  initialized solver instances rather than sharing immutable problem data and
  symbolic structure.
- Improving setup and end-to-end solve time likely requires a more shared batch
  representation: one common problem/KKT structure, per-item vector/workspace
  state, and per-item numeric KKT values only where needed.
