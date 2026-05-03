# Benchmarks

## Binary file format

Each `.bin` file in `data/` encodes a QP of the form:

```
min  0.5 x'Px + c'x
s.t. Ax  = b
     Gx  ∈ K
```

where `K = R+^l × SOC(q[0]) × ... × SOC(q[nsoc-1])`.

All matrices are stored in CSC format. The layout is little-endian and has no padding:

| Field | Type | Size |
|---|---|---|
| `n, m, p, l, nsoc, Pnnz, Annz, Gnnz` | `int32` | 8 × 4 bytes |
| `c` | `float64` | `n` × 8 bytes |
| `b` | `float64` | `p` × 8 bytes |
| `h` | `float64` | `m` × 8 bytes |
| `q` | `int32` | `nsoc` × 4 bytes |
| `P.data`, `P.indices`, `P.indptr` | `float64`, `int32`, `int32` | `Pnnz×8`, `Pnnz×4`, `(n+1)×4` bytes |
| `A.data`, `A.indices`, `A.indptr` | `float64`, `int32`, `int32` | `Annz×8`, `Annz×4`, `(p+1)×4` bytes |
| `G.data`, `G.indices`, `G.indptr` | `float64`, `int32`, `int32` | `Gnnz×8`, `Gnnz×4`, `(n+1)×4` bytes |

`P` stores the upper triangle only. `m = l + sum(q)`.

## Generating a `.bin` file from a CVXPY problem

```python
import cvxpy as cp
from dump import dump

x = cp.Variable(3)
prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [x >= 0])

dump(prob)                        # writes problem.bin in the current directory
dump(prob, "data/my_problem.bin") # custom path; parent directories are created automatically
```

`dump.py` canonicalizes the problem through the QOCO solver interface (`cp.QOCO`) before serializing, so the resulting matrices match exactly what QOCO receives at solve time.
