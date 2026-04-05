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
                              QOCOFloat kkt_static_reg, QOCOInt m);
     void (*linsys_update_data)(LinSysData*, QOCOProblemData*);
     void (*linsys_factor)(LinSysData*, QOCOInt n, QOCOFloat kkt_dynamic_reg);
     void (*linsys_solve)(LinSysData*, QOCOWorkspace*, QOCOVectorf* b,
                          QOCOVectorf* x, QOCOInt iter_ref_iters);
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

``linsys_solve`` calls ``QDLDL_solve`` then runs up to ``iter_ref_iters`` steps of
iterative refinement to improve accuracy.

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
