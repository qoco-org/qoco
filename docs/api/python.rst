.. _python_interface:

Python
=======

.. _python_main_API:

Import
------
The QOCO can be imported as follows

.. code:: python

    import qoco

Setup
-----

QOCO is initialized by creating a QOCO object as follows

.. code:: python

    solver = qoco.QOCO()

The problem is specified in the setup phase by running

.. code:: python

    solver.setup(n, m, p, P, c, A, b, G, h, l, nsoc, q, **settings)

The arguments :code:`c`, :code:`b` and :code:`q` are numpy arrays, and :code:`P`, :code:`A`, and :code:`G` are scipy sparse matrices in CSC format. The matrix :code:`P` can be either complete or just the upper triangular part, but QOCO will only use the upper triangular part. 

If you do not have equality constraints, pass in :code:`None` for :code:`A` and :code:`b`. Pass in :code:`None` for :code:`G` and :code:`h` if you do not have inequality or second-order cone constraints, and pass in :code:`None` for :code:`q` and :code:`0` for :code:`nsoc` if you do not have second-order cone constraints.

The keyword arguments :code:`**settings` specify the solver settings. The allowed parameters are defined in :ref:`settings <settings>`.

Solve
-----

To solve the problem, run

.. code:: python

   results = solver.solve()


The :code:`results` object contains the following


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
| :code:`status`        | Solve status see :ref:`exitflags <exit_flags>` |
+-----------------------+------------------------------------------------+
| :code:`obj`           | Objective value                                |
+-----------------------+------------------------------------------------+
| :code:`iters`         | Number of iterations                           |
+-----------------------+------------------------------------------------+
| :code:`setup_time_sec`| Setup time in seconds                          |
+-----------------------+------------------------------------------------+
| :code:`solve_time_sec`| Solve time in seconds                          |
+-----------------------+------------------------------------------------+
| :code:`pres`          | Primal residual                                |
+-----------------------+------------------------------------------------+
| :code:`dres`          | Dual residual                                  |
+-----------------------+------------------------------------------------+
| :code:`gap`           | Duality gap                                    |
+-----------------------+------------------------------------------------+