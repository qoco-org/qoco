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