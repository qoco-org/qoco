API for Custom Solver
---------------------


Main solver API
^^^^^^^^^^^^^^^
The main solver functions are in the table below.

+-----------------------------------------------------------------+------------------------------------------------+
|                          Function                               |            Description                         |
+=================================================================+================================================+
| :code:`void set_default_settings(Workspace* work)`              |      Sets the default settings                 |
+-----------------------------------------------------------------+------------------------------------------------+
| :code:`void load_data(Workspace* work)`                         |      Loads the default problem data            |
+-----------------------------------------------------------------+------------------------------------------------+
| :code:`void qoco_custom_solve(Workspace* work)`                 |      Solves the problem with qoco_custom       |
+-----------------------------------------------------------------+------------------------------------------------+


Settings
^^^^^^^^
The settings are defined in the :code:`workspace.h` file of :code:`qoco_custom` and are identical to the QOCO settings which are described in :ref:`settings <settings>`. To change the settings, directly modify the settings struct as shown in the code snippet in :ref:`calling <calling>`.

Solution
^^^^^^^^
The solution struct is within the workspace struct. To see how to access members fo the solution struct refer to the code snippet in :ref:`calling <calling>` (for example :code:`work.sol.obj` returns the objective value). The members of the solution struct are given below.

+-----------------------+------------------------------------------------+
| Member                | Description                                    |
+=======================+================================================+
| :code:`x`             | Primal solution                                |
+-----------------------+------------------------------------------------+
| :code:`s`             | Slack variable for conic constraints           |
+-----------------------+------------------------------------------------+
| :code:`y`             | Dual solution for equality constraints         |
+-----------------------+------------------------------------------------+
| :code:`z`             | Dual solution for conic constraints            |
+-----------------------+------------------------------------------------+
| :code:`iters`         | Number of iterations                           |
+-----------------------+------------------------------------------------+
| :code:`obj`           | Objective value                                |
+-----------------------+------------------------------------------------+
| :code:`pres`          | Primal residual                                |
+-----------------------+------------------------------------------------+
| :code:`dres`          | Dual residual                                  |
+-----------------------+------------------------------------------------+
| :code:`gap`           | Duality gap                                    |
+-----------------------+------------------------------------------------+
| :code:`status`        | Solve status see :ref:`exitflags <exit_flags>` |
+-----------------------+------------------------------------------------+

Status Flags
^^^^^^^^^^^^
The solution struct contains the solve status flag. The various flags are given in the table below.

+------------------------------+--------------------------------------+-------+
| Status                       | Status Code                          | Value |
+==============================+======================================+=======+
| Solver hasn't been called    | :code:`QOCO_CUSTOM_UNSOLVED`         |   0   |
+------------------------------+--------------------------------------+-------+
| Solved to desired accuracy   | :code:`QOCO_CUSTOM_SOLVED`           |   1   |
+------------------------------+--------------------------------------+-------+
| Solved to low accuracy       | :code:`QOCO_CUSTOM_SOLVED_INACCURATE`|   2   |
+------------------------------+--------------------------------------+-------+
| Numerical error or infeasible| :code:`QOCO_CUSTOM_NUMERICAL_ERROR`  |   3   |
+------------------------------+--------------------------------------+-------+
| Iteration limit reached      | :code:`QOCO_CUSTOM_MAX_ITER`         |   4   |
+------------------------------+--------------------------------------+-------+

Changing problem instance
^^^^^^^^^^^^^^^^^^^^^^^^^
Since the custom solvers are generated for a given problem family, users can solve any problem with identical sparsity patterns. This means that users can change
the vector data :code:`c`, :code:`b`, and :code:`h`, as well as the nonzero values in :code:`P`, :code:`A`, and :code:`G`. To change the values, the user should directly
modify the data within the :code:`Workspace` struct as shown in :ref:`calling <calling>`.
