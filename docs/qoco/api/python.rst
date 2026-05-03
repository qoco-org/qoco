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

GPU Backend
~~~~~~~~~~~

To use the GPU-accelerated CUDA backend, instantiate QOCO with the :code:`algebra="cuda"` parameter:

.. code:: python

    solver = qoco.QOCO(algebra="cuda")

This requires that you have installed QOCO with CUDA support (see :ref:`python_installation`) and have CUDA 13 and `cuDSS <https://developer.nvidia.com/cudss>`_ installed on your system. cuDSS is NVIDIA's GPU-accelerated sparse direct solver library, available from the NVIDIA developer website.

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
| :code:`ir_iters`      | Total iterative refinement iterations          |
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

Update Settings
---------------

Solver settings can be updated after :code:`setup()` without re-initializing the solver:

.. code:: python

    solver.update_settings(verbose=1, max_iters=100)

Any subset of the settings defined in :ref:`settings <settings>` can be passed as keyword arguments.

Update Vector Data
------------------

The vectors :code:`c`, :code:`b`, and :code:`h` can be updated after :code:`setup()` without re-initializing the solver. This is more efficient than calling :code:`setup()` again when only the data vectors change (e.g. in a receding-horizon MPC loop).

.. code:: python

    solver.update_vector_data(c=c_new, b=b_new, h=h_new)

Any subset of :code:`c`, :code:`b`, :code:`h` can be passed. Arguments that are :code:`None` (the default) are left unchanged.

Update Matrix Data
------------------

The nonzero values of :code:`P`, :code:`A`, and :code:`G` can be updated after :code:`setup()` without re-initializing the solver. This is more efficient than calling :code:`setup()` again when only the matrix values change.

.. warning::

    The new matrices must have the **same sparsity structure** as those passed to :code:`setup()`. Only the nonzero values are updated; the sparsity pattern cannot change.

Each argument should be a 1-D numpy array of the nonzero values in CSC order, with length equal to the number of nonzeros in the corresponding original matrix:

.. code:: python

    solver.update_matrix_data(P=P_new_vals, A=A_new_vals, G=G_new_vals)

Any subset of :code:`P`, :code:`A`, :code:`G` can be passed. Arguments that are :code:`None` (the default) are left unchanged.

Checking Available Backends
---------------------------

To check which algebra backends are available on your system:

.. code:: python

    qoco.algebras_available()   # returns e.g. ['builtin'] or ['builtin', 'cuda']

To check whether a specific backend is available:

.. code:: python

    qoco.algebra_available('cuda')    # returns True or False
    qoco.algebra_available('builtin') # returns True or False
