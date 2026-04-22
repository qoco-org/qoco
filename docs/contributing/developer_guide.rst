Developer Guide
===============

.. contents:: Table of Contents
   :local:
   :depth: 2

Repository Layout
-----------------

.. code-block:: text

   qoco/
   ├── src/                  # Core solver (backend-agnostic)
   │   ├── qoco_api.c        # Public API: setup, solve, cleanup
   │   ├── kkt.c             # KKT matrix construction and RHS assembly
   │   ├── cone.c / cone.cu  # Cone operations (CPU or GPU, selected at build time)
   │   ├── equilibration.c   # Ruiz scaling
   │   ├── common_linalg.c   # Linalg helpers that don't depend on backend
   │   └── qoco_utils.c      # Printing, stopping criteria, solution copy
   ├── include/              # Public and internal headers
   │   ├── structs.h         # All struct definitions including LinSysBackend
   │   └── qoco_linalg.h     # Backend-agnostic linalg interface (types + ops)
   ├── algebra/
   │   ├── CMakeLists.txt    # Selects backend, sets compile definitions
   │   ├── builtin/          # CPU backend (QDLDL + AMD)
   │   └── cuda/             # GPU backend (cuDSS + cuSPARSE + cuBLAS)
   ├── tests/
   │   ├── unit_tests/       # Component-level tests
   │   ├── simple_tests/     # Small end-to-end problems
   │   ├── ocp/              # Optimal control problem tests
   │   └── portfolio/        # Portfolio optimization tests
   ├── devtools/             # Local developer scripts
   ├── benchmarks/           # Benchmark runner and configs
   └── .github/workflows/    # CI definitions

Backend Architecture
--------------------

The solver core in ``src/`` is completely backend-agnostic. It interacts with the
linear algebra layer only through two abstractions:

- **Opaque types** — ``QOCOMatrix``, ``QOCOVectorf``, ``QOCOVectori`` are forward-declared
  in ``include/qoco_linalg.h`` and defined differently by each backend.
- **Function pointer table** — ``LinSysBackend`` in ``include/structs.h`` holds pointers
  to the backend's setup, factor, solve, and cleanup functions. The solver calls these
  through ``solver->linsys->...``.

Backend Interface
~~~~~~~~~~~~~~~~~

``LinSysBackend`` is defined in ``include/structs.h``:

.. code-block:: c

   typedef struct {
     const char* (*linsys_name)();
     LinSysData* (*linsys_setup)(QOCOProblemData*, QOCOSettings*, QOCOInt Wnnz);
     void (*linsys_set_nt_identity)(LinSysData*, QOCOInt m);
     void (*linsys_update_nt)(LinSysData*, QOCOVectorf* WtW_vec,
                              QOCOFloat kkt_static_reg_G, QOCOInt m);
     void (*linsys_update_data)(LinSysData*, QOCOProblemData*);
     void (*linsys_factor)(LinSysData*, QOCOInt n, QOCOFloat kkt_dynamic_reg);
     void (*linsys_solve)(LinSysData*, QOCOWorkspace*, QOCOVectorf* b,
                          QOCOVectorf* x, QOCOFloat ir_tol,
                          QOCOInt max_ir_iters);
     void (*linsys_cleanup)(LinSysData*);
   } LinSysBackend;

Each backend exports a ``LinSysBackend backend`` global that is linked into the
final binary. The solver calls ``linsys_setup`` at startup and thereafter calls
``linsys_factor`` / ``linsys_solve`` each iteration to solve the KKT system.

**Backend selection** happens at configure time via the CMake variable
``QOCO_ALGEBRA_BACKEND`` (default: ``builtin``). ``algebra/CMakeLists.txt`` validates
the choice, adds the corresponding directory to the include path, defines either
``QOCO_ALGEBRA_BACKEND_BUILTIN`` or ``QOCO_ALGEBRA_BACKEND_CUDA``, and calls
``add_subdirectory`` on the backend folder. The root ``CMakeLists.txt`` then picks
``src/cone.c`` (builtin) or ``src/cone.cu`` (CUDA) accordingly.

CPU (Builtin) Backend
~~~~~~~~~~~~~~~~~~~~~

**Location:** ``algebra/builtin/``

.. list-table::
   :header-rows: 1

   * - File
     - Purpose
   * - ``builtin_types.h``
     - Concrete struct definitions for ``QOCOMatrix``, ``QOCOVectorf``, ``QOCOVectori``
   * - ``builtin_linalg.c``
     - All linalg operations: SpMv, norms, element-wise ops, etc.
   * - ``qdldl_backend.c``
     - ``LinSysBackend`` implementation: setup, factor, solve, cleanup

**Type layout** (``builtin_types.h``):

.. code-block:: c

   struct QOCOVectorf_ { QOCOFloat* data; QOCOInt len; };
   struct QOCOVectori_ { QOCOInt*   data; QOCOInt len; };
   struct QOCOMatrix_  { QOCOCscMatrix* csc; };

Everything lives on the CPU. ``get_data_vectorf(v)`` returns ``v->data`` directly.

**Linear system** (``qdldl_backend.c``):

``linsys_setup`` builds the KKT matrix from P, A, G using ``construct_kkt``
(``src/kkt.c``), computes an AMD reordering for fill reduction, and permutes the
matrix to ``PKPt``. Index mappings (``PregtoKKT``, ``AttoKKT``, ``GttoKKT``, ``nt2kkt``,
``ntdiag2kkt``) are stored so that subsequent NT scaling updates can write directly
into the correct entries of ``PKPt`` without rebuilding it from scratch.

``linsys_factor`` calls ``QDLDL_factor`` on the permuted KKT matrix.

``linsys_solve`` calls ``QDLDL_solve`` then runs adaptive iterative refinement:
it repeats up to ``max_ir_iters`` times, stopping early when the KKT residual
:math:`\|Kx - b\|_\infty` falls below ``ir_tol``. A best-solution checkpoint
is maintained in permuted space; if a refinement step worsens the residual the
best solution is restored and refinement stops immediately. The number of
refinement iterations taken is accumulated in ``work->ir_iters`` and printed in
the IR column of the iteration log.

Iterative Refinement Stopping Criteria
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each call to ``linsys_solve`` runs the following loop after the initial
triangular solve:

1. Compute the residual :math:`r_0 = \|Kx_0 - b\|_\infty` and save
   :math:`x_0` as the best solution seen so far.
2. For :math:`i = 0, 1, \ldots, \texttt{max\_ir\_iters} - 1`:

   a. **Tolerance check** — if :math:`\|Kx_i - b\|_\infty < \texttt{ir\_tol}`,
      stop.
   b. **Correction step** — solve :math:`K \, dx = r_i` using the cached
      factorization, then update :math:`x_{i+1} = x_i + dx`.
   c. **Monotonicity check** — compute :math:`\|Kx_{i+1} - b\|_\infty`.
      If the residual did not improve (:math:`\ge` best seen so far), restore
      the best solution and stop. Otherwise update the best solution and
      continue.

The loop therefore stops as soon as **any** of the following holds:

- :math:`\|Kx - b\|_\infty < \texttt{ir\_tol}` (converged)
- a refinement step fails to reduce the residual (stagnation / divergence)
- ``max_ir_iters`` steps have been taken

The best-solution checkpoint ensures that a diverging step can never make
the returned solution worse than the initial solve.

**Dependencies:** QDLDL (``lib/qdldl/``), AMD (``lib/amd/``), both built as part of
the CMake project.

GPU (CUDA) Backend
~~~~~~~~~~~~~~~~~~

**Location:** ``algebra/cuda/``

.. list-table::
   :header-rows: 1

   * - File
     - Purpose
   * - ``cuda_types.h``
     - Concrete struct definitions — each type holds both host and device pointers
   * - ``cuda_linalg.cu``
     - CUDA kernels for SpMv, norms, element-wise ops, etc.
   * - ``cudss_backend.cu``
     - ``LinSysBackend`` implementation using cuDSS

**Type layout** (``cuda_types.h``):

.. code-block:: c

   struct QOCOVectorf_ { QOCOFloat* host; QOCOFloat* device; QOCOInt len; };
   struct QOCOVectori_ { QOCOInt*   host; QOCOInt*   device; QOCOInt len; };
   struct QOCOMatrix_  {
     QOCOCscMatrix* csc_host;     // CSC on host
     CusparseMatrix* csr_device;  // CSR on device (data)
     CusparseMatrix* csr_meta;    // CSR on device (metadata/structure)
   };

**CPU mode flag:** A thread-local ``cpu_mode`` flag controls which pointer
``get_data_vectorf()`` returns. Core solver code that runs on the CPU calls
``set_cpu_mode(1)`` before accessing data, ensuring it gets the host pointer.
GPU kernel launches use ``set_cpu_mode(0)``.

**Dynamic library loading:** CUDA libraries are loaded at runtime with ``dlopen()``
in ``cudss_setup()`` rather than linked at build time. This allows the binary to
run on systems without a GPU (returning a graceful error) and avoids mandatory
CUDA toolkit installation for users of the CPU backend. The libraries loaded are:

- ``libcudss.so`` — NVIDIA cuDSS sparse direct solver
- ``libcusparse.so`` — Sparse matrix operations
- ``libcublas.so`` — Dense linear algebra

**Matrix format:** The core solver uses CSC throughout. The CUDA backend converts
to CSR for cuDSS (which requires CSR) during setup and stores the result on the
device. Problem matrices A and G are stored in both formats.

**Linear system:** ``linsys_setup`` constructs the KKT matrix on the CPU via the
shared ``construct_kkt`` function, converts it to CSR, uploads to device, and
initialises a cuDSS solver handle. ``linsys_factor`` and ``linsys_solve`` call into
cuDSS. The solve result is left on device; ``sync_vector_to_host`` is called
explicitly when the CPU needs to read the result.

Cone Implementation
~~~~~~~~~~~~~~~~~~~

Cone operations (products, divisions, NT scaling, linesearch) are in
``src/cone.c`` for the builtin backend and ``src/cone.cu`` for the CUDA backend.
The file is selected at build time — only one is ever compiled. The CUDA version
implements the same logic as CUDA kernels dispatched via the same function
signatures.

Closed-Form SOC Step Length
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The linesearch for the second-order cone (``soc_step_length`` in ``src/cone.c``)
computes the maximum step length :math:`\alpha \ge 0` such that
:math:`x + \alpha \, dx` remains in the second-order cone

.. math::

   \mathcal{Q}^n = \{ (x_0, x_1) \in \mathbb{R} \times \mathbb{R}^{n-1} :
   x_0 \ge \|x_1\| \}

rather than performing a bisection search.

**Derivation.** The membership condition for :math:`x + \alpha \, dx` is

.. math::

   (x_0 + \alpha \, dx_0)^2 \ge \|x_1 + \alpha \, dx_1\|^2.

Expanding and collecting by powers of :math:`\alpha`:

.. math::

   \underbrace{(dx_0^2 - \|dx_1\|^2)}_{a} \, \alpha^2
   + \underbrace{2(x_0 \, dx_0 - x_1^\top dx_1)}_{b} \, \alpha
   + \underbrace{(x_0^2 - \|x_1\|^2)}_{c} \ge 0.

Because :math:`x` is already in the cone, :math:`c = \det(x) = x_0^2 - \|x_1\|^2 \ge 0`,
so :math:`\alpha = 0` is always feasible. The maximum feasible :math:`\alpha` is
therefore the smallest positive real root of the quadratic :math:`a \alpha^2 + b \alpha + c = 0`.

**Case analysis.** Before solving the quadratic the code handles four degenerate cases:

1. **Scalar safeguard.** If :math:`dx_0 < 0`, the first component could go negative.
   An independent upper bound :math:`-x_0 / dx_0` is applied first.

2. **No positive root** (:math:`a > 0` and :math:`b > 0`, or discriminant :math:`d = b^2 - 4ac < 0`).
   The parabola either opens upward with a positive vertex shift or has no real roots.
   Either way the quadratic stays non-negative for all :math:`\alpha \ge 0`, so the
   current bound is returned unchanged.

3. **Linear case** (:math:`|a| < 10^{-14}`). The leading term vanishes; the constraint
   is linear in :math:`\alpha`. With :math:`c \ge 0` and the sign structure this
   imposes no additional restriction, so the bound is returned unchanged.

4. **Boundary case** (:math:`c = 0`, i.e. :math:`x` is on the cone boundary). If
   :math:`a \ge 0` there is no positive root; otherwise :math:`\alpha = 0` is the
   only feasible point.

**Numerically stable root computation.** When none of the degenerate cases applies,
the citardauq formula is used to avoid catastrophic cancellation. Let
:math:`\sqrt{d} = \sqrt{b^2 - 4ac}`. Define

.. math::

   t = \begin{cases}
       -b - \sqrt{d} & \text{if } b \ge 0 \\
       -b + \sqrt{d} & \text{if } b < 0
   \end{cases}

Then the two roots are computed as

.. math::

   r_1 = \frac{2c}{t}, \qquad r_2 = \frac{t}{2a}.

This form ensures that both :math:`r_1` and :math:`r_2` are computed by dividing
two numbers of the same sign, avoiding the large relative error that arises when
subtracting nearly equal quantities. Negative roots are discarded (replaced by
:math:`+\infty`), and the smaller of :math:`r_1`, :math:`r_2` is taken as the
step-length restriction for this cone.

Static Regularization
~~~~~~~~~~~~~~~~~~~~~

The KKT system solved at each IPM iteration is a symmetric indefinite linear system
of the form

.. math::

   \begin{bmatrix}
     P & A^\top & G^\top \\
     A &   0    &   0    \\
     G &   0    & -W^\top W
   \end{bmatrix}
   \begin{bmatrix} \Delta x \\ \Delta y \\ \Delta z \end{bmatrix}
   = \begin{bmatrix} r_x \\ r_y \\ r_z \end{bmatrix}

To keep the system nonsingular and to give each diagonal block a well-defined sign
for the factorization, a small positive constant is added or subtracted on each
block's diagonal before every factorization. The three parameters are kept separate
because the blocks have different signs and may require different magnitudes:

.. list-table::
   :header-rows: 1

   * - Setting
     - Block
     - Applied as
     - Rationale
   * - ``kkt_static_reg_P``
     - (1,1) — :math:`P`
     - :math:`P \leftarrow P + \varepsilon_P I`
     - Ensures the (1,1) block is positive definite even when :math:`P` is
       only positive semidefinite.
   * - ``kkt_static_reg_A``
     - (2,2) — equality constraints
     - diagonal :math:`\leftarrow -\varepsilon_A`
     - Gives the zero (2,2) block a definite (negative) sign, preventing
       near-zero pivots on problems with redundant equality constraints.
   * - ``kkt_static_reg_G``
     - (3,3) — NT scaling :math:`W^\top W`
     - diagonal :math:`\leftarrow -\varepsilon_G` added to :math:`-W^\top W`
     - Guards against near-zero pivots when the NT scaling matrix is
       ill-conditioned near the cone boundary.

**Implementation.** ``kkt_static_reg_P`` is applied once during setup by ``regularize_P``
(or ``construct_identity`` when :math:`P = 0`) in ``src/qoco_api.c``, and is
corrected for in ``compute_objective`` and ``compute_kkt_residual`` so that
reported objectives and residuals reflect the original unregularized problem.
``kkt_static_reg_A`` is baked into the KKT matrix structure by ``construct_kkt``
(``src/kkt.c``) at setup time and does not change across iterations.
``kkt_static_reg_G`` is re-applied every iteration by ``linsys_update_nt`` after the
NT scaling block is refreshed.

Adaptive Dynamic Regularization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the IPM step length drops below :math:`10^{-8}` (a stall), the solver
does not immediately declare failure. Instead it multiplies ``kkt_dynamic_reg``
by 10 and retries the current iteration. If ``kkt_dynamic_reg`` exceeds
:math:`10^{-6}` the solver falls back to the usual inaccurate / numerical-error
exit. This lets the solver recover from near-singular KKT systems on
ill-conditioned problems without requiring the user to tune
``kkt_dynamic_reg`` manually.

Building
--------

Prerequisites
~~~~~~~~~~~~~

- CMake ≥ 3.18
- C compiler: clang or gcc (Linux/macOS), MSVC (Windows)
- Python 3.11+ with ``cvxpy`` (for test data generation)
- CUDA toolkit ≥ 13.0 (GPU backend only)

CPU backend (default)
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cmake -B build -DQOCO_BUILD_TYPE=Release -DENABLE_TESTING=True
   cmake --build build

GPU backend
~~~~~~~~~~~

.. code-block:: bash

   cmake -B build \
     -DQOCO_ALGEBRA_BACKEND=cuda \
     -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
     -DQOCO_BUILD_TYPE=Release \
     -DENABLE_TESTING=True
   cmake --build build -j$(nproc)

CMake options
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``QOCO_ALGEBRA_BACKEND``
     - ``builtin``
     - Backend: ``builtin`` or ``cuda``
   * - ``QOCO_BUILD_TYPE``
     - ``Release``
     - ``Debug`` (adds ``-g``, ASAN/UBSAN on Unix) or ``Release`` (``-O3``)
   * - ``QOCO_SINGLE_PRECISION``
     - ``OFF``
     - Use ``float`` instead of ``double``
   * - ``ENABLE_TESTING``
     - ``OFF``
     - Build and register test suite
   * - ``BUILD_QOCO_DEMO``
     - ``OFF``
     - Build ``examples/qoco_demo``
   * - ``BUILD_QOCO_BENCHMARK_RUNNER``
     - ``OFF``
     - Build benchmark runner

Unit Tests
----------

Tests use Google Test and are run with ``ctest``. All test executables link against
``qocostatic``.

Test categories
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Directory
     - Executable(s)
     - What it covers
   * - ``tests/unit_tests/``
     - ``linalg_test``, ``cone_test``, ``input_validation_test``
     - Individual components
   * - ``tests/simple_tests/``
     - ``missing_constraints_test``
     - End-to-end with missing constraint types (LP-only, SOC-only)
   * - ``tests/ocp/``
     - ``lcvx_test``, ``lcvx_bad_scaling_test``, ``pdg_test``
     - Optimal control problems
   * - ``tests/portfolio/``
     - ``markowitz_test``
     - Portfolio optimization (Markowitz)

Unit test details
~~~~~~~~~~~~~~~~~

**linalg_test** — covers the linalg layer:

- CSC matrix creation and copying
- Array copy / negate / scale
- Dot products, sparse matrix-vector products

**cone_test** — covers ``src/cone.c``:

- Cone products and divisions for LP and SOC cones
- Mixed LP + SOC problems

**input_validation_test** — covers ``src/input_validation.c``:

- Rejects invalid settings (tolerances, iteration counts, etc.)

Integration tests
~~~~~~~~~~~~~~~~~

The OCP and portfolio tests load pre-generated problem data from header files
(e.g. ``lcvx_data.h``, ``markowitz_data.h``) and call the full solve pipeline.
They assert that the optimal objective matches a reference value within 0.01%
relative error.

Problem data is generated by the Python scripts in each test directory
(``generate_problem_data.py``), which use ``cvxpy`` to solve the reference problem.
The generated ``.h`` files are committed to the repository, so ``cvxpy`` is only
needed if you regenerate them.

Running tests locally
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   ctest --test-dir build --verbose

   # Run a specific test
   ctest --test-dir build -R lcvx_test --verbose

   # Run with output on failure and retry
   ctest --test-dir build --rerun-failed --output-on-failure

CI Workflows
------------

All workflows are in ``.github/workflows/``.

unit_tests.yml — primary test suite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Triggers on every push and pull request.

Runs the full test matrix in parallel (``fail-fast: false``):

.. list-table::
   :header-rows: 1

   * - OS
     - Compiler
     - Build types
   * - ubuntu-latest
     - clang
     - Debug, Release
   * - macos-latest
     - clang
     - Debug, Release
   * - windows-latest
     - MSVC
     - Debug, Release

The Debug build enables ``-fsanitize=address,undefined`` on Linux and macOS, so
memory errors and undefined behaviour are caught automatically.

**To reproduce a CI failure locally (e.g. ubuntu clang Debug):**

.. code-block:: bash

   cmake -B build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
         -DQOCO_BUILD_TYPE=Debug -DENABLE_TESTING=True -S .
   cmake --build build
   ctest --test-dir build --verbose --rerun-failed --output-on-failure

clang_tidy.yml — static analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Triggers on every push and pull request.

Builds with ``CMAKE_EXPORT_COMPILE_COMMANDS=ON`` then runs ``clang-tidy`` on all
``src/*.c`` files (excluding OS-specific timers). Config is in ``.clang-tidy``.

Enabled check families: ``bugprone-*``, ``clang-analyzer-*``, ``misc-unused-parameters``.
Disabled: ``bugprone-easily-swappable-parameters``,
``clang-analyzer-security.insecureAPI.DeprecatedOrUnsafeBufferHandling``.
All warnings are treated as errors.

**To reproduce locally:**

.. code-block:: bash

   devtools/run_clang_tidy.sh

clang_format.yml — formatting enforcement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Triggers on every push and pull request.

Runs ``clang-format --dry-run --Werror`` on all ``.c`` and ``.h`` files under
``src/``, ``include/``, and ``algebra/builtin/``. Fails if any file is not
formatted according to ``.clang-format``.

**To reproduce locally:**

.. code-block:: bash

   devtools/run_clang_format.sh --check   # check only (same as CI)
   devtools/run_clang_format.sh           # fix in place

benchmark_regression.yml — performance regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Triggers only on pull requests targeting ``main``.

Builds both ``main`` and the PR branch, runs the benchmark suite against both,
and posts a comparison report as a PR comment. Uses configs in
``benchmarks/configs/main.yml`` and ``benchmarks/configs/branch.yml``.

docs.yml — documentation deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Triggers manually (``workflow_dispatch`` only).

Builds the Sphinx docs from ``docs/`` and deploys to the ``gh-pages`` branch.
Also deploys to the root of ``gh-pages`` if the version being built is the
latest released version.

Developer Tools
---------------

All scripts in ``devtools/`` are intended to be run from the repository root.

run_tests.sh
~~~~~~~~~~~~

Builds in Release mode with testing, demo, and benchmark runner enabled, then
runs the full test suite.

.. code-block:: bash

   devtools/run_tests.sh

run_tests_gpu.sh
~~~~~~~~~~~~~~~~

Same as above but configures the CUDA backend with CUDA 13.0.

.. code-block:: bash

   devtools/run_tests_gpu.sh

run_clang_tidy.sh
~~~~~~~~~~~~~~~~~

Generates a temporary build directory with compile commands, runs ``clang-tidy``
on all ``src/*.c`` files, then removes the build directory.

.. code-block:: bash

   devtools/run_clang_tidy.sh

run_clang_format.sh
~~~~~~~~~~~~~~~~~~~

Checks or fixes formatting for ``src/``, ``include/``, and ``algebra/builtin/``.

.. code-block:: bash

   devtools/run_clang_format.sh           # fix in place
   devtools/run_clang_format.sh --check   # report violations and exit non-zero

profile.sh
~~~~~~~~~~

Profiles a benchmark run on CPU using Valgrind's callgrind tool and opens the
result in KCachegrind.

.. code-block:: bash

   devtools/profile.sh path/to/benchmark/data

profile_gpu.sh
~~~~~~~~~~~~~~

Profiles a benchmark run on GPU using NVIDIA Nsight Systems.

.. code-block:: bash

   devtools/profile_gpu.sh path/to/benchmark/data
