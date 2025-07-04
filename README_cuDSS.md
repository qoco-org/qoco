# cuDSS Integration for QOCO

This branch adds GPU-accelerated linear system solving to QOCO using NVIDIA's cuDSS (Direct Sparse Solver) library.

## Overview

The cuDSS integration provides an alternative backend for solving the KKT linear systems that arise in the interior-point method. When enabled, cuDSS can provide significant speedup for large-scale problems by leveraging GPU acceleration.

## Building with cuDSS

### Prerequisites

1. **CUDA Toolkit** (version 11.0 or later)
2. **cuDSS Library** - Download from NVIDIA's website
3. **CMake** (version 3.10 or later)

### Build Options

#### Using QDLDL (CPU, default)

```bash
mkdir build && cd build
cmake .. -DUSE_CUDSS=OFF
make -j4
```

#### Using cuDSS (GPU-accelerated)

```bash
export CU_DSS_ROOT=/path/to/cudss/installation
mkdir build && cd build
cmake .. -DUSE_CUDSS=ON
make -j4
```

## Usage

The solver automatically uses the appropriate backend based on the build configuration:

- **QDLDL**: CPU-based LDL factorization (default)
- **cuDSS**: GPU-accelerated direct sparse solver (when `USE_CUDSS=ON`)

No changes to the API are required - the same `qoco_solve()` function will use the appropriate solver backend.

## Implementation Details

### Conditional Compilation

The code uses conditional compilation to switch between backends:

```c
void kkt_solve(QOCOSolver* solver, QOCOFloat* b, QOCOInt iters)
{
#ifdef QOCO_USE_CUDSS
  // Use cuDSS for GPU-accelerated linear system solve
  kkt_solve_cudss(solver, b, iters);
#else
  // Use QDLDL for CPU-based linear system solve
  // ... existing QDLDL implementation ...
#endif
}
```

### Memory Management

When cuDSS is enabled, the solver:

1. Allocates GPU memory for the CSC matrix data
2. Transfers matrix data to GPU during initialization
3. Transfers right-hand side vectors to GPU for each solve
4. Retrieves solutions back to CPU memory

### Resource Cleanup

cuDSS resources are automatically cleaned up when `qoco_cleanup()` is called.

## Performance Considerations

- **Small problems**: QDLDL may be faster due to lower overhead
- **Large problems**: cuDSS can provide significant speedup
- **Memory transfer**: Consider the cost of CPU-GPU data transfer
- **Matrix structure**: cuDSS performance depends on the sparsity pattern

## Troubleshooting

### Build Issues

1. **CUDA not found**: Install CUDA toolkit
2. **cuDSS not found**: Set `CU_DSS_ROOT` environment variable
3. **CMake version**: Ensure CMake 3.10+ is installed

### Runtime Issues

1. **GPU memory**: Ensure sufficient GPU memory for the problem size
2. **CUDA context**: Ensure CUDA is properly initialized
3. **Matrix format**: Verify the CSC matrix format is correct

## Future Work

- [ ] Implement actual cuDSS API calls in `kkt_solve_cudss()`
- [ ] Add runtime backend selection
- [ ] Optimize memory transfer patterns
- [ ] Add performance benchmarking
- [ ] Support for multiple GPU configurations
