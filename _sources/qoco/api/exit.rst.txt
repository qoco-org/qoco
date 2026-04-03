.. _exit_flags:

Exit flags
-----------

QOCO's exit flags are defined in the :code:`include/enums.h` file.

+------------------------------+-----------------------------------+-------+
| Status                       | Status Code                       | Value |
+==============================+===================================+=======+
| Solver hasn't been called    | :code:`QOCO_UNSOLVED`             |   0   |
+------------------------------+-----------------------------------+-------+
| Solved to desired accuracy   | :code:`QOCO_SOLVED`               |   1   |
+------------------------------+-----------------------------------+-------+
| Solved to low accuracy       | :code:`QOCO_SOLVED_INACCURATE`    |   2   |
+------------------------------+-----------------------------------+-------+
| Numerical error or infeasible| :code:`QOCO_NUMERICAL_ERROR`      |   3   |
+------------------------------+-----------------------------------+-------+
| Iteration limit reached      | :code:`QOCO_MAX_ITER`             |   4   |
+------------------------------+-----------------------------------+-------+

Status descriptions
~~~~~~~~~~~~~~~~~~~

**QOCO_SOLVED**
   The problem was solved to the tolerances specified by :code:`abstol` and :code:`reltol`. The solution in :code:`results` is reliable.

**QOCO_SOLVED_INACCURATE**
   The solver could not reach the primary tolerances (:code:`abstol`, :code:`reltol`) but the solution satisfies the looser tolerances :code:`abstol_inacc` and :code:`reltol_inacc`. The returned solution is an approximation and may be usable depending on the application.

**QOCO_NUMERICAL_ERROR**
   The solver encountered a numerical failure (e.g. a singular KKT system) the problem is primal or dual infeasible. Check that the problem data is well-scaled. If the problem is feasible, try increasing :code:`kkt_static_reg` or :code:`kkt_dynamic_reg` to improve numerical stability.

**QOCO_MAX_ITER**
   The solver reached the iteration limit set by :code:`max_iters` without converging. Try increasing :code:`max_iters`, enabling Ruiz equilibration with :code:`ruiz_iters`, or reformulating the problem to improve conditioning.

**QOCO_UNSOLVED**
   :code:`solve()` has not been called yet.

Setup error codes
~~~~~~~~~~~~~~~~~

:code:`qoco_setup()` and :code:`qoco_update_settings()` return one of the following integer error codes:

+------------------------------------------+-------+-----------------------------------------------------------+
| Error Code                               | Value | Meaning                                                   |
+==========================================+=======+===========================================================+
| :code:`QOCO_NO_ERROR`                    |   0   | Setup completed successfully.                             |
+------------------------------------------+-------+-----------------------------------------------------------+
| :code:`QOCO_DATA_VALIDATION_ERROR`       |   1   | Invalid problem data (e.g. wrong dimensions, null matrix).|
+------------------------------------------+-------+-----------------------------------------------------------+
| :code:`QOCO_SETTINGS_VALIDATION_ERROR`   |   2   | Invalid settings (e.g. non-positive tolerance).           |
+------------------------------------------+-------+-----------------------------------------------------------+
| :code:`QOCO_SETUP_ERROR`                 |   3   | Internal setup failure.                                   |
+------------------------------------------+-------+-----------------------------------------------------------+
| :code:`QOCO_AMD_ERROR`                   |   4   | Failure during AMD fill-reducing ordering.                |
+------------------------------------------+-------+-----------------------------------------------------------+
| :code:`QOCO_MALLOC_ERROR`                |   5   | Memory allocation failure.                                |
+------------------------------------------+-------+-----------------------------------------------------------+
